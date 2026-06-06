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
from rl.ekf_jax import EKFParams, EKFState, step as ekf_step, inertia_from_state
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
    # Body-frame external torque disturbance scale (Nm). Sampled per episode
    # as N(0, disturbance_scale^2 * I_3). The EKF augments tau_ext into its
    # state to estimate it. Default 0.0 reproduces the noiseless setup.
    disturbance_scale: float = 0.0
    # Disturbance mode: "constant" (default, episode-fixed bias matching what
    # the augmented EKF expects) or "sinusoidal" (time-varying amplitude
    # * cos(omega*t + phase), with omega ~ U(omega_lo, omega_hi) and
    # phase ~ U(0, 2π) per episode). Sinusoidal is the F4 sim-to-sim test:
    # constant-trained policies / constant-prior EKF face a structure they
    # never saw during training.
    disturbance_mode: str = "constant"
    disturbance_omega_lo: float = 0.05
    disturbance_omega_hi: float = 0.5


class T1EnvEKFState(NamedTuple):
    sat_state: jnp.ndarray    # (6,) [ω, rw_speed]   — TRUE dynamics
    ekf: EKFState              # EKF posterior over [ω, I, rw_speed]
    F_oracle: jnp.ndarray      # (6, 6) oracle FIM over full I tensor (for comparison only)
    step: jnp.int32
    last_omega_true: jnp.ndarray  # (3,) for the oracle FIM finite-diff
    sat: SatParams
    key: jax.Array
    tau_ext: jnp.ndarray       # (3,) constant amplitude / bias
    dist_omega: jnp.ndarray    # (3,) per-axis angular freq (0 for constant mode)
    dist_phase: jnp.ndarray    # (3,) per-axis phase


class T1EnvEKF:

    def __init__(self, cfg: T1EnvEKFConfig):
        self.cfg = cfg
        # Observation: ω_est(3) + rw_est(3)/rw_max + EKF I-estimate(6, normalized)
        #              + log diag(P_I) (6) + tau_ext_est(3) + log diag(P_tau)(3)
        #              + progress(1) = 25-dim
        self.obs_shape = (25,)

    def _build_ekf_params(self, sat: SatParams) -> EKFParams:
        diag_I = jnp.diag(sat.I_sat)
        # Process noise for tau_ext: very small (bias is near-constant); use
        # 1% of the disturbance scale as the random-walk std per dt.
        dist_scale = jnp.maximum(self.cfg.disturbance_scale, 1e-9)
        return EKFParams(
            dt=self.cfg.dt,
            I_rw=jnp.full((3,), sat.I_rw),
            Qc=jnp.concatenate([
                jnp.full((3,), self.cfg.Qc_omega),                          # omega
                self.cfg.Qc_I_rel * diag_I ** 2,                            # I_diag
                jnp.full((3,), self.cfg.Qc_I_rel * (diag_I.mean()) ** 2),   # I_off
                jnp.full((3,), self.cfg.Qc_rw),                             # rw
                jnp.full((3,), (0.01 * dist_scale) ** 2),                   # tau_ext
            ]),
            R=jnp.concatenate([
                jnp.full((3,), self.cfg.sigma_omega ** 2),
                jnp.full((3,), self.cfg.sigma_rw ** 2),
            ]),
        )

    def reset(self, key, sat: SatParams | None = None):
        sat = sat if sat is not None else self.cfg.sat
        k_omega, k_dist, k_om, k_ph, k_used = jax.random.split(key, 5)
        omega0 = self.cfg.init_omega_scale * jax.random.normal(k_omega, (3,))
        rw0 = jnp.zeros(3)
        sat_state = jnp.concatenate([omega0, rw0])
        tau_ext = self.cfg.disturbance_scale * jax.random.normal(k_dist, (3,))
        if self.cfg.disturbance_mode == "sinusoidal":
            dist_omega = jax.random.uniform(
                k_om, (3,),
                minval=self.cfg.disturbance_omega_lo,
                maxval=self.cfg.disturbance_omega_hi,
            )
            dist_phase = jax.random.uniform(k_ph, (3,), minval=0.0,
                                            maxval=2.0 * jnp.pi)
        else:
            dist_omega = jnp.zeros(3)
            dist_phase = jnp.zeros(3)

        diag_true = jnp.diag(sat.I_sat)
        I_diag_init = self.cfg.I0_scale * diag_true  # biased initial diagonal
        sigma_I_diag = self.cfg.sigma_I0_rel * diag_true
        # Off-diag prior 6x tighter than diag: real spacecraft have small
        # off-diagonals; loose prior here lets the Kalman update over-correct.
        sigma_I_off = jnp.full((3,), 0.05 * diag_true.mean())
        # tau_ext prior: mean 0 (matches reset sampling), variance matches
        # the env's actual sampling distribution.
        sigma_tau_ext = jnp.full((3,), jnp.maximum(self.cfg.disturbance_scale, 1e-9))
        x0 = jnp.concatenate([omega0, I_diag_init, jnp.zeros(3), rw0,
                              jnp.zeros(3)])
        P0 = jnp.diag(jnp.concatenate([
            jnp.full((3,), 1e-4),
            sigma_I_diag ** 2,
            sigma_I_off ** 2,
            jnp.full((3,), 1e-2),
            sigma_tau_ext ** 2,
        ]))
        ekf_state = EKFState(x=x0, P=P0)

        env_state = T1EnvEKFState(
            sat_state=sat_state, ekf=ekf_state, F_oracle=jnp.zeros((6, 6)),
            step=jnp.int32(0), last_omega_true=omega0, sat=sat, key=k_used,
            tau_ext=tau_ext, dist_omega=dist_omega, dist_phase=dist_phase,
        )
        return env_state, self._obs(env_state)

    def step(self, env_state: T1EnvEKFState, action: jnp.ndarray):
        cfg = self.cfg
        sat = env_state.sat
        tau_cmd = jnp.clip(action, -cfg.tau_max, cfg.tau_max)

        # Disturbance value at this step: constant or sinusoidal depending on
        # cfg.disturbance_mode. For "constant" mode dist_omega=0 so the cosine
        # term is just 1, falling through to env_state.tau_ext * 1.
        t = env_state.step.astype(jnp.float32) * cfg.dt
        tau_ext_now = env_state.tau_ext * jnp.cos(
            env_state.dist_omega * t + env_state.dist_phase
        )
        h = cfg.dt / cfg.substeps
        new_sat_state = _step_dt(env_state.sat_state, tau_cmd, sat, h, cfg.substeps,
                                 tau_ext=tau_ext_now)
        new_omega_true = new_sat_state[0:3]
        new_rw_true = new_sat_state[3:6]

        # Noisy measurement
        k_obs, k_next = jax.random.split(env_state.key)
        noise_omega = cfg.sigma_omega * jax.random.normal(k_obs, (3,))
        k_obs2, k_next2 = jax.random.split(k_next)
        noise_rw = cfg.sigma_rw * jax.random.normal(k_obs2, (3,))
        z = jnp.concatenate([new_omega_true + noise_omega,
                             new_rw_true + noise_rw])

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
        I_true_mat = sat.I_sat
        I_est_mat = inertia_from_state(new_ekf.x)
        # Squared Frobenius relative error over the full tensor.
        rel_sq_err = (jnp.linalg.norm(I_est_mat - I_true_mat) ** 2
                      / jnp.linalg.norm(I_true_mat) ** 2)
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
            sat=sat, key=k_next2, tau_ext=env_state.tau_ext,
            dist_omega=env_state.dist_omega, dist_phase=env_state.dist_phase,
        )
        obs = self._obs(new_state)
        info = {
            "info_gain": info_gain,
            "sat_pen": sat_pen,
            "logdet_F_oracle": ld_oracle,
            "I_est": jnp.concatenate([new_ekf.x[3:6], new_ekf.x[6:9]]),
            "I_true": I_true_mat,
            "rel_sq_err": rel_sq_err,
        }
        return new_state, obs, reward, done, info

    def _obs(self, env_state: T1EnvEKFState) -> jnp.ndarray:
        ekf = env_state.ekf
        omega_est = ekf.x[0:3]
        I_diag = ekf.x[3:6]
        I_off = ekf.x[6:9]
        rw_est = ekf.x[9:12]
        tau_ext_est = ekf.x[12:15]
        I_est = jnp.concatenate([I_diag, I_off])
        P_I_diag = jnp.diag(ekf.P[3:9, 3:9])
        P_tau_diag = jnp.diag(ekf.P[12:15, 12:15])
        ref_I3 = jnp.diag(self.cfg.sat.I_sat)
        ref_off = jnp.array([self.cfg.sat.I_sat[0, 1],
                             self.cfg.sat.I_sat[0, 2],
                             self.cfg.sat.I_sat[1, 2]])
        ref_norm = jnp.concatenate([
            ref_I3,
            jnp.maximum(jnp.abs(ref_off), 0.01 * ref_I3.mean()),
        ])
        tau_ref = jnp.maximum(self.cfg.disturbance_scale, self.cfg.tau_max * 1e-3)
        progress = env_state.step.astype(jnp.float32) / self.cfg.horizon
        return jnp.concatenate([
            omega_est,
            rw_est / self.cfg.sat.rw_speed_max,
            I_est / ref_norm,
            jnp.log(jnp.maximum(P_I_diag, 1e-30)),
            tau_ext_est / tau_ref,
            jnp.log(jnp.maximum(P_tau_diag, 1e-30)),
            jnp.array([progress]),
        ])


def make_env_ekf(cfg: T1EnvEKFConfig) -> T1EnvEKF:
    return T1EnvEKF(cfg)
