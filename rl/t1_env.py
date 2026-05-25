"""T1 environment: active-sensing inertia identification.

The agent commands reaction-wheel torque each dt. The environment integrates
the rigid-body dynamics (via the JAX simulator), accumulates the Fisher
Information Matrix (FIM) over the full 6-parameter symmetric inertia tensor,
and rewards incremental D-optimal information gain.

Pure-functional: env_state passes through; works under jax.jit / jax.vmap.
"""
from __future__ import annotations
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from sim.dynamics_jax import SatParams, _step_dt
from utils.observability import regression_rows_full


class T1EnvConfig(NamedTuple):
    sat: SatParams
    dt: float
    substeps: int
    horizon: int           # max steps before done
    tau_max: float         # action will be clipped to [-tau_max, tau_max]
    fim_eps: float         # log-det regularizer for FIM
    sat_penalty: float     # weight on the saturation penalty term
    init_omega_scale: float  # std of initial omega noise


class T1EnvState(NamedTuple):
    sat_state: jnp.ndarray    # (6,) [omega, rw_speed]
    F: jnp.ndarray            # (6, 6) accumulated Fisher information over full I tensor
    step: jnp.int32           # current step count
    last_omega: jnp.ndarray   # (3,) for finite-diff of omega_dot
    sat: SatParams            # per-episode sat config (for domain randomization)


def _fim_row_contribution(omega: jnp.ndarray, domega: jnp.ndarray) -> jnp.ndarray:
    """Return R^T R where R is the (3,6) regression block at this step.

    Used to accumulate the 6x6 oracle FIM in the T1 env over the full inertia
    tensor parameterization theta = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
    """
    R = regression_rows_full(omega, domega)  # (3, 6)
    return R.T @ R                            # (6, 6)


def _slogdet_psd(F: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Log-det of a PSD matrix via Cholesky (stable forward + backward).

    With eps regularization, F + eps*I is positive-definite, so Cholesky never
    fails and its backward is well-conditioned (unlike slogdet, whose gradient
    can blow up near singular matrices and produce NaN under autodiff).
    """
    L = jnp.linalg.cholesky(F + eps * jnp.eye(F.shape[-1]))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def _saturation_penalty(sat: SatParams, tau_cmd: jnp.ndarray,
                        rw_speed: jnp.ndarray) -> jnp.ndarray:
    """Smooth penalty proxy for hitting torque or RW-speed limits."""
    tau_excess = jnp.maximum(jnp.abs(tau_cmd) - sat.rw_torque_max, 0.0)
    speed_excess = jnp.maximum(jnp.abs(rw_speed) - 0.95 * sat.rw_speed_max, 0.0)
    return (tau_excess.sum() / (sat.rw_torque_max + 1e-12)
            + speed_excess.sum() / (sat.rw_speed_max + 1e-12))


class T1Env:
    """A thin namespace of reset/step functions for one env configuration.

    Built via make_env(cfg). Holds the config; reset/step are pure functions
    that take (env_state, action) and return next env_state plus signals.
    """

    def __init__(self, cfg: T1EnvConfig):
        self.cfg = cfg
        # Observation: omega(3) + rw_speed(3) + flattened F upper-triangle(21)
        #              + step/horizon(1) = 28-dim (F is 6x6 over full inertia tensor)
        self.obs_shape = (28,)

    def reset(self, key, sat: SatParams | None = None) -> Tuple[T1EnvState, jnp.ndarray]:
        """Initialize an episode. `sat` overrides cfg.sat (for domain randomization)."""
        sat = sat if sat is not None else self.cfg.sat
        omega0 = self.cfg.init_omega_scale * jax.random.normal(key, (3,))
        rw0 = jnp.zeros(3)
        sat_state = jnp.concatenate([omega0, rw0])
        F = jnp.zeros((6, 6))
        env_state = T1EnvState(sat_state=sat_state, F=F,
                               step=jnp.int32(0), last_omega=omega0, sat=sat)
        return env_state, self._obs(env_state)

    def step(self, env_state: T1EnvState, action: jnp.ndarray):
        cfg = self.cfg
        sat = env_state.sat  # per-episode sat (may differ from cfg.sat under DR)
        tau_cmd = jnp.clip(action, -cfg.tau_max, cfg.tau_max)

        # Advance dynamics one dt
        h = cfg.dt / cfg.substeps
        new_sat_state = _step_dt(env_state.sat_state, tau_cmd, sat,
                                 h, cfg.substeps)
        new_omega = new_sat_state[0:3]
        new_rw = new_sat_state[3:6]

        # Finite-difference omega_dot (centered at current step's midpoint)
        domega = (new_omega - env_state.last_omega) / cfg.dt
        mid_omega = 0.5 * (new_omega + env_state.last_omega)

        # FIM accumulation
        delta_F = _fim_row_contribution(mid_omega, domega)
        new_F = env_state.F + delta_F

        # Reward = info gain - saturation penalty
        ld_before = _slogdet_psd(env_state.F, cfg.fim_eps)
        ld_after = _slogdet_psd(new_F, cfg.fim_eps)
        info_gain = ld_after - ld_before
        sat_pen = _saturation_penalty(sat, tau_cmd, new_rw)
        reward = info_gain - cfg.sat_penalty * sat_pen

        next_step = env_state.step + 1
        done = next_step >= cfg.horizon

        new_state = T1EnvState(
            sat_state=new_sat_state, F=new_F,
            step=next_step, last_omega=new_omega, sat=sat,
        )
        obs = self._obs(new_state)
        info = {"info_gain": info_gain, "sat_pen": sat_pen,
                "logdet_F": ld_after}
        return new_state, obs, reward, done, info

    def _obs(self, env_state: T1EnvState) -> jnp.ndarray:
        F = env_state.F
        # Upper-triangle entries (including diagonal) of the 6x6 FIM: 21 values.
        iu, ju = jnp.triu_indices(6)
        F_flat = F[iu, ju]
        progress = env_state.step.astype(jnp.float32) / self.cfg.horizon
        return jnp.concatenate([
            env_state.sat_state[0:3],          # omega (3)
            env_state.sat_state[3:6] / self.cfg.sat.rw_speed_max,  # normalized rw (3)
            jnp.log1p(F_flat),                 # log-scaled FIM entries (21)
            jnp.array([progress]),             # progress (1)
        ])


def make_env(cfg: T1EnvConfig) -> T1Env:
    return T1Env(cfg)


def sample_sat(key, I_range: Tuple[float, float],
               I_rw: float, rw_axes: jnp.ndarray,
               rw_speed_max: float, rw_torque_max: float,
               log_uniform: bool = False) -> SatParams:
    """Sample a SatParams with diagonal inertia uniform in I_range per axis.

    Off-diagonals are zero; this matches the LS estimator's diagonal model and
    the FIM regression assumption.

    When log_uniform=True, samples in log-space — better when I_range spans
    multiple orders of magnitude (e.g., 0.1–20).
    """
    if log_uniform:
        log_lo, log_hi = jnp.log(I_range[0]), jnp.log(I_range[1])
        log_diag = jax.random.uniform(key, (3,), minval=log_lo, maxval=log_hi)
        diag = jnp.exp(log_diag)
    else:
        diag = jax.random.uniform(key, (3,), minval=I_range[0], maxval=I_range[1])
    I_sat = jnp.diag(diag)
    I_inv = jnp.diag(1.0 / diag)
    return SatParams(
        I_sat=I_sat, I_inv=I_inv, I_rw=I_rw, rw_axes=rw_axes,
        rw_speed_max=rw_speed_max, rw_torque_max=rw_torque_max,
    )
