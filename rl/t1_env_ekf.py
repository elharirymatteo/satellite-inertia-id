"""T1 environment with EKF-in-the-loop reward.

Differences from rl/t1_env.py::T1Env (oracle FIM reward):
  - Observations are *noisy* measurements of (ω, rw_speed), not ground truth.
  - The env carries an EKF state. Each step the EKF predicts then updates on
    the noisy measurement.
  - Reward = log-det reduction of the *inertia 3x3 block of the EKF posterior
    covariance*. This is the actual estimation uncertainty the agent reduces,
    not a regression on the true ω̇.
  - For comparability we also return the oracle log-det F in `info` so plots
    can compare both reward signals on the same rollout.

Pure-functional; works under jax.jit / jax.vmap. env_state carries an RNG key
for sensor-noise draws.
"""
from __future__ import annotations
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from sim.dynamics_jax import SatParams, _step_dt
from rl.ekf_jax import EKFParams, EKFState, step as ekf_step
from rl.t1_env import _fim_row_contribution, _slogdet_psd, _saturation_penalty


class T1EnvEKFConfig(NamedTuple):
    sat: SatParams
    dt: float
    substeps: int
    horizon: int
    tau_max: float
    sat_penalty: float
    init_omega_scale: float
    # Sensor noise (Gaussian, per axis)
    sigma_omega: float
    sigma_rw: float
    # EKF priors
    I0_scale: float        # initial inertia mean = I0_scale * true I
    sigma_I0_rel: float    # relative std on initial inertia (e.g. 0.3 = 30%)
    Qc_omega: float        # diag of continuous-time process noise on omega
    Qc_I_rel: float        # relative random-walk variance on inertia
    Qc_rw: float           # diag of continuous-time process noise on rw_speed
    # Reward mode: "info_gain" (D-optimal info gain on posterior cov) or
    # "neg_rel_err" (negative squared relative I-estimation error vs ground truth)
    reward_mode: str = "info_gain"


class T1EnvEKFState(NamedTuple):
    sat_state: jnp.ndarray    # (6,) [ω, rw_speed]   — TRUE dynamics
    ekf: EKFState              # EKF posterior over [ω, I, rw_speed]
    F_oracle: jnp.ndarray      # (6, 6) oracle FIM over full I tensor (for comparison only)
    step: jnp.int32
    last_omega_true: jnp.ndarray  # (3,) for the oracle FIM finite-diff
    sat: SatParams
    key: jax.Array


class T1EnvEKF:

    def __init__(self, cfg: T1EnvEKFConfig):
        self.cfg = cfg
        # Observation: ω_noisy(3) + rw_noisy(3)/rw_max + EKF I-estimate(3, normalized by I0)
        #              + log of inertia-block cov diagonal(3) + progress(1) = 13-dim
        self.obs_shape = (13,)

    def _build_ekf_params(self, sat: SatParams) -> EKFParams:
        return EKFParams(
            dt=self.cfg.dt,
            I_rw=jnp.full((3,), sat.I_rw),
            Qc=jnp.concatenate([
                jnp.full((3,), self.cfg.Qc_omega),
                self.cfg.Qc_I_rel * jnp.diag(sat.I_sat) ** 2,
                jnp.full((3,), self.cfg.Qc_rw),
            ]),
            R=jnp.concatenate([
                jnp.full((3,), self.cfg.sigma_omega ** 2),
                jnp.full((3,), self.cfg.sigma_rw ** 2),
            ]),
        )

    def reset(self, key, sat: SatParams | None = None):
        sat = sat if sat is not None else self.cfg.sat
        k_omega, k_I, k_used = jax.random.split(key, 3)
        omega0 = self.cfg.init_omega_scale * jax.random.normal(k_omega, (3,))
        rw0 = jnp.zeros(3)
        sat_state = jnp.concatenate([omega0, rw0])

        true_I = jnp.diag(sat.I_sat)
        I_init = self.cfg.I0_scale * true_I  # biased initial estimate
        sigma_I = self.cfg.sigma_I0_rel * true_I
        x0 = jnp.concatenate([omega0, I_init, rw0])
        P0 = jnp.diag(jnp.concatenate([
            jnp.full((3,), 1e-4),
            sigma_I ** 2,
            jnp.full((3,), 1e-2),
        ]))
        ekf_state = EKFState(x=x0, P=P0)

        env_state = T1EnvEKFState(
            sat_state=sat_state, ekf=ekf_state, F_oracle=jnp.zeros((6, 6)),
            step=jnp.int32(0), last_omega_true=omega0, sat=sat, key=k_used,
        )
        return env_state, self._obs(env_state)

    def step(self, env_state: T1EnvEKFState, action: jnp.ndarray):
        cfg = self.cfg
        sat = env_state.sat
        tau_cmd = jnp.clip(action, -cfg.tau_max, cfg.tau_max)

        # Advance true dynamics
        h = cfg.dt / cfg.substeps
        new_sat_state = _step_dt(env_state.sat_state, tau_cmd, sat, h, cfg.substeps)
        new_omega_true = new_sat_state[0:3]
        new_rw_true = new_sat_state[3:6]

        # Noisy measurement
        k_obs, k_next = jax.random.split(env_state.key)
        noise_omega = cfg.sigma_omega * jax.random.normal(k_obs, (3,))
        k_obs2, k_next2 = jax.random.split(k_next)
        noise_rw = cfg.sigma_rw * jax.random.normal(k_obs2, (3,))
        z = jnp.concatenate([new_omega_true + noise_omega,
                             new_rw_true + noise_rw])

        # EKF step: needs u (rw acceleration command) — approximate via tau_cmd / I_rw
        # since the true RW limits may have changed the actual acceleration; the
        # EKF doesn't know that, so it uses the commanded value (realistic).
        u_ekf = tau_cmd / sat.I_rw
        ekf_params = self._build_ekf_params(sat)
        new_ekf, info_gain = ekf_step(env_state.ekf, u_ekf, z, ekf_params)

        # Oracle FIM (for comparison only, not used in reward)
        domega_true = (new_omega_true - env_state.last_omega_true) / cfg.dt
        mid_omega = 0.5 * (new_omega_true + env_state.last_omega_true)
        delta_F = _fim_row_contribution(mid_omega, domega_true)
        new_F_oracle = env_state.F_oracle + delta_F
        ld_oracle = _slogdet_psd(new_F_oracle, 1e-6)

        # Reward signals
        sat_pen = _saturation_penalty(sat, tau_cmd, new_rw_true)
        I_true = jnp.diag(sat.I_sat)
        I_est = new_ekf.x[3:6]
        # Negative squared relative I-estimation error (uses ground truth — OK in sim).
        rel_sq_err = jnp.sum(((I_est - I_true) / I_true) ** 2)
        if cfg.reward_mode == "neg_rel_err":
            reward_unpenalized = -rel_sq_err
        else:  # "info_gain"
            reward_unpenalized = info_gain
        reward = reward_unpenalized - cfg.sat_penalty * sat_pen

        next_step = env_state.step + 1
        done = next_step >= cfg.horizon

        new_state = T1EnvEKFState(
            sat_state=new_sat_state, ekf=new_ekf, F_oracle=new_F_oracle,
            step=next_step, last_omega_true=new_omega_true,
            sat=sat, key=k_next2,
        )
        obs = self._obs(new_state)
        info = {
            "info_gain": info_gain,
            "sat_pen": sat_pen,
            "logdet_F_oracle": ld_oracle,
            "I_est": new_ekf.x[3:6],
            "I_true": jnp.diag(sat.I_sat),
            "rel_sq_err": rel_sq_err,
        }
        return new_state, obs, reward, done, info

    def _obs(self, env_state: T1EnvEKFState) -> jnp.ndarray:
        ekf = env_state.ekf
        # Observed (noisy) state from EKF — use the EKF's filtered estimate of ω.
        # (The agent never sees true ω directly under this env.)
        omega_est = ekf.x[0:3]
        rw_est = ekf.x[6:9]
        I_est = ekf.x[3:6]
        # Diagonal of the I-block covariance, log-scaled (always positive).
        P_I_diag = jnp.diag(ekf.P[3:6, 3:6])
        # Normalize: I_est by self.cfg.sat.I_sat diagonal (fixed reference) so
        # the policy sees a roughly O(1) signal across very different sats.
        ref_I = jnp.diag(self.cfg.sat.I_sat)
        progress = env_state.step.astype(jnp.float32) / self.cfg.horizon
        return jnp.concatenate([
            omega_est,
            rw_est / self.cfg.sat.rw_speed_max,
            I_est / ref_I,
            jnp.log(jnp.maximum(P_I_diag, 1e-30)),
            jnp.array([progress]),
        ])


def make_env_ekf(cfg: T1EnvEKFConfig) -> T1EnvEKF:
    return T1EnvEKF(cfg)
