"""Dual MPC planner: gradient through the diff-sim, FIM log-det objective.

Used as a baseline against the RL policy. The planner takes the current
dynamics state + accumulated FIM, picks a piecewise-constant torque
sequence over a horizon, and returns it after a fixed number of Adam steps.

Pure JAX; designed to be jit-compiled and vmap-compatible over batches.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import optax

from sim.dynamics_jax import SatParams, _step_dt
from utils.observability import regression_rows_full


def _slogdet_psd_6(F, eps):
    L = jnp.linalg.cholesky(F + eps * jnp.eye(6))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def _saturation_penalty(sat, tau, rw_speed):
    tau_excess = jnp.maximum(jnp.abs(tau) - sat.rw_torque_max, 0.0)
    speed_excess = jnp.maximum(jnp.abs(rw_speed) - 0.95 * sat.rw_speed_max, 0.0)
    return (tau_excess.sum() / (sat.rw_torque_max + 1e-12)
            + speed_excess.sum() / (sat.rw_speed_max + 1e-12))


def _rollout_loss(z, init_state, F0, sat, horizon, dt, substeps,
                  tau_max, sat_penalty, fim_eps):
    """Loss = -logdet(F_final) + sat_penalty * total_sat_pen.

    z has shape (horizon, 3); tau = tau_max * tanh(z) (box constraint).
    """
    tau_seq = tau_max * jnp.tanh(z)

    def body(carry, tau):
        sat_state, F, last_omega = carry
        new_state = _step_dt(sat_state, tau, sat, dt / substeps, substeps)
        new_omega = new_state[0:3]
        domega = (new_omega - last_omega) / dt
        mid_omega = 0.5 * (new_omega + last_omega)
        R = regression_rows_full(mid_omega, domega)
        F = F + R.T @ R
        rw_speed = new_state[3:6]
        sp = _saturation_penalty(sat, tau, rw_speed)
        return (new_state, F, new_omega), sp

    last_omega_init = init_state[0:3]
    (final_state, F_final, _), sat_pens = jax.lax.scan(
        body, (init_state, F0, last_omega_init), tau_seq
    )
    ld = _slogdet_psd_6(F_final, fim_eps)
    return -ld + sat_penalty * sat_pens.sum()


def plan(init_state, F0, sat: SatParams, *, horizon: int,
         n_opt_steps: int, tau_max: float, lr: float,
         dt: float, substeps: int,
         sat_penalty: float = 0.1, fim_eps: float = 1e-6,
         warm_start: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return the optimized tau sequence of shape (horizon, 3).

    Cold start uses a small deterministic seed in z rather than zeros: from
    rest with zero torque the FIM gradient w.r.t. z is identically zero
    (loss is flat to leading order in z near 0), so pure-zero init never
    moves. A tiny non-zero pattern breaks the symmetry while remaining
    well within the tanh linear regime.
    """
    if warm_start is not None:
        z = warm_start
    else:
        # Small alternating ±1e-2 seed: deterministic, jit-safe, breaks the
        # three-axis symmetry that makes the gradient vanish at z=0.
        steps = jnp.arange(horizon)
        seed = 1e-2 * jnp.stack([
            ((steps % 2) * 2 - 1).astype(jnp.float32),
            ((steps % 3) - 1).astype(jnp.float32),
            (1 - 2 * ((steps // 2) % 2)).astype(jnp.float32),
        ], axis=-1)
        z = seed

    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = opt.init(z)

    @jax.jit
    def grad_step(z, opt_state):
        loss, g = jax.value_and_grad(_rollout_loss)(
            z, init_state, F0, sat, horizon, dt, substeps,
            tau_max, sat_penalty, fim_eps,
        )
        updates, opt_state = opt.update(g, opt_state)
        z = optax.apply_updates(z, updates)
        return z, opt_state, loss

    for _ in range(n_opt_steps):
        z, opt_state, _ = grad_step(z, opt_state)
    return tau_max * jnp.tanh(z)
