"""JAX port of Satellite.simulate_fast.

Pure-functional, jittable, vmappable. Designed as the rollout backend for the
T1 active-sensing env.

Key API:
    SatParams                — NamedTuple capturing static satellite/RW config
    rk4_step(state, tau, p, h)  — one fixed-step RK4 substep, pure function
    simulate_jax(s0, taus, p, dt, substeps)  — full trajectory rollout
    simulate_jax_jit         — jitted version (dt and substeps are static)

The NumPy oracle (Satellite.simulate_fast) is the reference. Differences:
  - JAX uses zero-order hold: tau_array[i] is held constant over the interval
    [t_i, t_{i+1}). The NumPy oracle's searchsorted control evaluates at
    multiple times within the interval and is mostly equivalent.
  - JAX defaults to float32 (CPU GPU friendly); the NumPy version is float64.
    Tests allow ~1e-3 relative tolerance for this reason.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax


class SatParams(NamedTuple):
    """Static satellite/RW configuration passed to the dynamics."""
    I_sat: jnp.ndarray       # (3, 3) inertia tensor
    I_inv: jnp.ndarray       # (3, 3) precomputed inverse
    I_rw: float              # scalar wheel inertia
    rw_axes: jnp.ndarray     # (3, 3) RW axes (rows)
    rw_speed_max: float
    rw_torque_max: float


def _apply_rw_limits(rw_speeds: jnp.ndarray, rw_acc_unlim: jnp.ndarray,
                     tau_cmd: jnp.ndarray, p: SatParams):
    """Vectorized version of Satellite._apply_rw_limits — no Python branches."""
    # Torque saturation
    tau_sat = jnp.abs(tau_cmd) > p.rw_torque_max
    tau_actual = jnp.where(tau_sat,
                           jnp.sign(tau_cmd) * p.rw_torque_max,
                           tau_cmd)
    rw_acc = tau_actual / p.I_rw

    # Speed saturation: only allow deceleration when at speed limit
    at_speed_limit = jnp.abs(rw_speeds) >= p.rw_speed_max
    same_sign = jnp.sign(rw_speeds) == jnp.sign(rw_acc)
    block_accel = at_speed_limit & same_sign
    rw_acc = jnp.where(block_accel, 0.0, rw_acc)
    tau_actual = jnp.where(block_accel, 0.0, tau_actual)

    return rw_acc, tau_actual


def _deriv(state: jnp.ndarray, tau_cmd: jnp.ndarray, p: SatParams,
           tau_ext: jnp.ndarray | None = None):
    """xdot = f(x, u). state = [omega(3), rw_speed(3)]. tau_ext is an
    external body-frame torque (gravity gradient, residual aero, etc.)."""
    omega = state[0:3]
    rw_speeds = state[3:6]
    rw_acc_unlim = tau_cmd / p.I_rw
    rw_acc, _ = _apply_rw_limits(rw_speeds, rw_acc_unlim, tau_cmd, p)
    h_rw_total = p.rw_axes.T @ (p.I_rw * rw_speeds)
    h_rw_dot = p.rw_axes.T @ (p.I_rw * rw_acc)
    h_total = p.I_sat @ omega + h_rw_total
    ext = jnp.zeros(3) if tau_ext is None else tau_ext
    domega = p.I_inv @ (ext - jnp.cross(omega, h_total) - h_rw_dot)
    return jnp.concatenate([domega, rw_acc])


def rk4_step(state: jnp.ndarray, tau_cmd: jnp.ndarray, p: SatParams, h: float,
             tau_ext: jnp.ndarray | None = None):
    k1 = _deriv(state, tau_cmd, p, tau_ext)
    k2 = _deriv(state + (h * 0.5) * k1, tau_cmd, p, tau_ext)
    k3 = _deriv(state + (h * 0.5) * k2, tau_cmd, p, tau_ext)
    k4 = _deriv(state + h * k3, tau_cmd, p, tau_ext)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _step_dt(state: jnp.ndarray, tau_cmd: jnp.ndarray,
             p: SatParams, h: float, substeps: int,
             tau_ext: jnp.ndarray | None = None):
    def substep(s, _):
        return rk4_step(s, tau_cmd, p, h, tau_ext), None
    new_state, _ = lax.scan(substep, state, None, length=substeps)
    return new_state


def simulate_jax(state0: jnp.ndarray, tau_array: jnp.ndarray, p: SatParams,
                 dt: float, substeps: int):
    """Full trajectory rollout.

    Args:
        state0:    (6,) initial state [omega, rw_speed]
        tau_array: (T, 3) zero-order-hold torque commands
        p:         SatParams
        dt:        output time step (seconds)
        substeps:  internal RK4 substeps per dt

    Returns:
        states: (T, 6) — states at the same time grid as tau_array.
                states[0] == state0 (the initial state, no integration done).
                states[i+1] = state after applying tau_array[i] for one dt.
    """
    h = dt / substeps

    def outer(s, tau_cmd):
        ns = _step_dt(s, tau_cmd, p, h, substeps)
        return ns, ns

    # Match the NumPy oracle's ZOH convention: tau_array[i+1] is held during
    # the i-th transition (state[i] -> state[i+1]). tau_array[0] is at t=0 only
    # and not used to drive any integration step.
    _, traj = lax.scan(outer, state0, tau_array[1:])
    return jnp.concatenate([state0[None, :], traj], axis=0)


simulate_jax_jit = jax.jit(simulate_jax, static_argnums=(3, 4))
