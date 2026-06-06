"""Smoke test that the new envs accept actions and produce sensible
observations of the new shape (T1Env: 28-dim with 6x6 FIM upper triangle;
T1EnvEKF: 19-dim with 6-vector I_est and log diag(P_I))."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

jax = pytest.importorskip("jax")
jnp = jax.numpy

from rl.t1_env import T1EnvConfig, make_env, sample_sat  # noqa: E402
from rl.t1_env_ekf import T1EnvEKFConfig, make_env_ekf  # noqa: E402


def _toy_sat():
    return sample_sat(
        jax.random.PRNGKey(0), (0.1, 20.0),
        I_rw=1e-4, rw_axes=jnp.eye(3),
        rw_speed_max=600.0, rw_torque_max=0.05,
        log_uniform=True, max_tilt_angle=np.pi / 8,
    )


def test_t1env_obs_shape_28():
    sat = _toy_sat()
    cfg = T1EnvConfig(
        sat=sat, dt=0.1, substeps=10, horizon=20,
        tau_max=0.01, fim_eps=1e-6, sat_penalty=0.1,
        init_omega_scale=1e-3,
    )
    env = make_env(cfg)
    assert env.obs_shape == (28,)
    state, obs = env.reset(jax.random.PRNGKey(1))
    assert obs.shape == (28,)
    state, obs, r, done, info = env.step(state, jnp.array([0.001, 0.0, 0.0]))
    assert obs.shape == (28,)
    assert jnp.isfinite(r)


def test_t1envekf_obs_shape_25():
    sat = _toy_sat()
    cfg = T1EnvEKFConfig(
        sat=sat, dt=0.1, substeps=10, horizon=20, tau_max=0.01,
        sat_penalty=0.1, init_omega_scale=1e-3,
        sigma_omega=1e-4, sigma_rw=1e-3,
        I0_scale=0.85, sigma_I0_rel=0.30,
        Qc_omega=1e-9, Qc_I_rel=1e-7, Qc_rw=1e-9,
        reward_mode="info_gain",
        disturbance_scale=0.001,
    )
    env = make_env_ekf(cfg)
    assert env.obs_shape == (25,)
    state, obs = env.reset(jax.random.PRNGKey(2))
    assert obs.shape == (25,)
    assert state.ekf.x.shape == (15,)
    assert state.ekf.P.shape == (15, 15)
    state, obs, r, done, info = env.step(state, jnp.array([0.001, 0.0, 0.0]))
    assert obs.shape == (25,)
    assert jnp.isfinite(r)
