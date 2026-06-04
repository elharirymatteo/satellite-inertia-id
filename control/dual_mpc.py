"""Dual MPC planner: gradient through the diff-sim, FIM log-det objective.

Used as a baseline against the RL policy. The planner takes the current
dynamics state + accumulated FIM, picks a piecewise-constant torque
sequence over a horizon, and returns it after a fixed number of Adam steps.

Pure JAX; designed to be jit-compiled and vmap-compatible over batches.
"""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
import optax

from sim.dynamics_jax import SatParams, _step_dt
from utils.observability import regression_rows_full

# Shared helpers live in rl/t1_env.py — the env uses the exact same FIM
# log-det and saturation penalty, so the MPC plans against the same scoring
# the policy is rewarded by.
from rl.t1_env import _slogdet_psd, _saturation_penalty


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
    ld = _slogdet_psd(F_final, fim_eps)
    return -ld + sat_penalty * sat_pens.sum()


def _make_optimizer(lr: float):
    """Adam + global-norm clip + NaN scrubbing."""
    return optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )


@functools.partial(jax.jit, static_argnames=("horizon", "substeps"))
def _adam_step(z, opt_state, init_state, F0, sat,
               tau_max, sat_penalty, fim_eps, lr,
               horizon, dt, substeps):
    """One Adam step on z toward minimizing _rollout_loss.

    Module-level + jitted so its XLA trace is cached across the many replan
    calls within an episode — and across episodes — rather than re-tracing
    every time `plan()` is invoked (which was the root cause of the LLVM JIT
    memory growth that motivated the `jax.clear_caches()` band-aid).

    `horizon`/`dt`/`substeps` are kwargs that travel as static args (they
    affect the scan length and substep count); `sat`, `init_state`, `F0`,
    and the scalar weights flow as runtime values, so changing them across
    replans within an episode does NOT trigger retraces.
    """
    opt = _make_optimizer(lr)
    loss, g = jax.value_and_grad(_rollout_loss)(
        z, init_state, F0, sat, horizon, dt, substeps,
        tau_max, sat_penalty, fim_eps,
    )
    updates, opt_state = opt.update(g, opt_state)
    z = optax.apply_updates(z, updates)
    return z, opt_state, loss


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

    opt = _make_optimizer(lr)
    opt_state = opt.init(z)

    for _ in range(n_opt_steps):
        z, opt_state, _ = _adam_step(
            z, opt_state, init_state, F0, sat,
            tau_max, sat_penalty, fim_eps, lr,
            horizon=horizon, dt=dt, substeps=substeps,
        )
    return tau_max * jnp.tanh(z)
