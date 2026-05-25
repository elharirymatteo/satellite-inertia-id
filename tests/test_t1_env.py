"""Tests for the T1 active-sensing inertia-ID RL environment.

Env contract:
  reset(key) -> (env_state, obs)
  step(env_state, action) -> (env_state, obs, reward, done, info)

The env is pure-functional (no mutation), JAX jit/vmap compatible.
Reward is the increment in log-det of the accumulated Fisher Information Matrix
over diagonal inertia parameters, minus a saturation penalty.
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

jax = pytest.importorskip("jax")
jnp = jax.numpy

from sim.dynamics_jax import SatParams  # noqa: E402
from rl.t1_env import T1EnvConfig, make_env, sample_sat  # noqa: E402


def _build_cfg() -> T1EnvConfig:
    with open(ROOT / "config_sat1.yaml") as f:
        cfg = yaml.safe_load(f)
    I_sat = np.diag(cfg["satellite"]["inertia_tensor"])
    sat = SatParams(
        I_sat=jnp.asarray(I_sat),
        I_inv=jnp.asarray(np.linalg.inv(I_sat)),
        I_rw=float(cfg["reaction_wheels"]["inertia"]),
        rw_axes=jnp.asarray(cfg["reaction_wheels"]["alignment_matrix"], dtype=jnp.float32),
        rw_speed_max=float(cfg["reaction_wheels"]["max_speed"]),
        rw_torque_max=float(cfg["reaction_wheels"]["max_torque"]),
    )
    return T1EnvConfig(
        sat=sat, dt=float(cfg["sim"]["dt"]), substeps=10,
        horizon=100, tau_max=float(cfg["reaction_wheels"]["max_torque"]),
        fim_eps=1e-6, sat_penalty=0.1, init_omega_scale=1e-3,
    )


def test_env_reset_returns_correct_shapes():
    env = make_env(_build_cfg())
    key = jax.random.PRNGKey(0)
    env_state, obs = env.reset(key)
    assert obs.shape == env.obs_shape
    # First-state log_det should equal log_det(eps * I3) = 3 * log(eps)
    assert env_state.step == 0
    np.testing.assert_array_equal(np.asarray(env_state.F), np.zeros((6, 6)))


def test_env_step_advances_state_and_terminates_at_horizon():
    cfg = _build_cfg()._replace(horizon=5)
    env = make_env(cfg)
    key = jax.random.PRNGKey(1)
    env_state, _ = env.reset(key)
    done = False
    n = 0
    while not done and n < 10:
        env_state, obs, reward, done, info = env.step(env_state, jnp.zeros(3))
        n += 1
    assert done
    assert n == cfg.horizon, f"Episode ended after {n} steps, expected {cfg.horizon}"


def test_env_random_excitation_accumulates_more_fim_than_zero_action():
    """Random actions should give a higher final log-det FIM than constant zero."""
    env = make_env(_build_cfg())
    key = jax.random.PRNGKey(7)
    tau_max = _build_cfg().tau_max

    # zero action
    env_state, _ = env.reset(key)
    for _ in range(50):
        env_state, *_ = env.step(env_state, jnp.zeros(3))
    F_zero = env_state.F

    # random action
    env_state, _ = env.reset(key)
    k = jax.random.PRNGKey(99)
    for _ in range(50):
        k, sub = jax.random.split(k)
        a = jax.random.uniform(sub, (3,), minval=-tau_max, maxval=tau_max)
        env_state, *_ = env.step(env_state, a)
    F_rand = env_state.F

    sign_z, logdet_zero = jnp.linalg.slogdet(F_zero + 1e-6 * jnp.eye(6))
    sign_r, logdet_rand = jnp.linalg.slogdet(F_rand + 1e-6 * jnp.eye(6))
    assert float(logdet_rand) > float(logdet_zero) + 5.0, (
        f"Random excitation should accumulate much more FIM "
        f"(zero={float(logdet_zero):.2f}, rand={float(logdet_rand):.2f})"
    )


def test_env_reward_is_logdet_increment():
    """Reward at step k must equal slogdet(F_new+eps*I) - slogdet(F_old+eps*I) - sat_pen."""
    env = make_env(_build_cfg())
    env_state, _ = env.reset(jax.random.PRNGKey(2))
    # warm up FIM with random actions to escape singular regime
    k = jax.random.PRNGKey(3)
    for _ in range(20):
        k, sub = jax.random.split(k)
        a = jax.random.uniform(sub, (3,), minval=-0.005, maxval=0.005)
        env_state, *_ = env.step(env_state, a)

    F_before = env_state.F
    eps = _build_cfg().fim_eps
    a = jnp.array([0.003, -0.002, 0.001])
    env_state_after, _, reward, _, info = env.step(env_state, a)
    F_after = env_state_after.F

    sign_b, ld_before = jnp.linalg.slogdet(F_before + eps * jnp.eye(6))
    sign_a, ld_after = jnp.linalg.slogdet(F_after + eps * jnp.eye(6))
    info_gain_expected = float(ld_after - ld_before)
    info_gain_observed = float(info["info_gain"])
    assert abs(info_gain_observed - info_gain_expected) < 1e-4, (
        f"info_gain mismatch: env={info_gain_observed} expected={info_gain_expected}"
    )


def test_env_supports_per_episode_sat_override():
    """reset(key, sat) must use the override and propagate it into env_state."""
    base_cfg = _build_cfg()
    env = make_env(base_cfg)

    # Build a "different" sat — 10x larger inertia
    big_I = jnp.diag(jnp.array([2.6, 3.0, 1.6]))
    custom_sat = base_cfg.sat._replace(I_sat=big_I,
                                       I_inv=jnp.linalg.inv(big_I))

    env_state, _ = env.reset(jax.random.PRNGKey(0), sat=custom_sat)
    np.testing.assert_allclose(np.asarray(env_state.sat.I_sat), np.asarray(big_I))

    # Step under the larger sat must produce slower omega rise than default
    action = jnp.array([0.005, 0.005, 0.005])
    s_big, *_ = env.step(env_state, action)
    s_def, _ = env.reset(jax.random.PRNGKey(0))  # uses base_cfg.sat (small inertia)
    s_def, *_ = env.step(s_def, action)
    omega_big = float(jnp.linalg.norm(s_big.sat_state[:3]))
    omega_def = float(jnp.linalg.norm(s_def.sat_state[:3]))
    assert omega_big < omega_def, (
        f"Larger inertia should give smaller omega: big={omega_big}, def={omega_def}"
    )


def test_sample_sat_returns_valid_satparams():
    cfg = _build_cfg()
    sat = sample_sat(jax.random.PRNGKey(0), (0.1, 1.0),
                     I_rw=cfg.sat.I_rw, rw_axes=cfg.sat.rw_axes,
                     rw_speed_max=cfg.sat.rw_speed_max,
                     rw_torque_max=cfg.sat.rw_torque_max)
    diag = jnp.diag(sat.I_sat)
    assert jnp.all(diag >= 0.1) and jnp.all(diag <= 1.0)
    # I_inv must invert I_sat (diagonal case)
    np.testing.assert_allclose(np.asarray(sat.I_sat @ sat.I_inv),
                               np.eye(3), atol=1e-6)


def test_env_full_episode_runs_under_jit_and_vmap():
    cfg = _build_cfg()._replace(horizon=20)
    env = make_env(cfg)

    def rollout(key):
        env_state, _ = env.reset(key)
        def body(carry, _):
            env_state, k = carry
            k, sub = jax.random.split(k)
            a = jax.random.uniform(sub, (3,), minval=-cfg.tau_max, maxval=cfg.tau_max)
            env_state, obs, r, d, _info = env.step(env_state, a)
            return (env_state, k), r
        (final_state, _), rewards = jax.lax.scan(
            body, (env_state, key), None, length=cfg.horizon
        )
        return rewards.sum(), final_state.F

    rollout_jit = jax.jit(rollout)
    R, F = rollout_jit(jax.random.PRNGKey(0))
    assert jnp.isfinite(R)
    assert jnp.all(jnp.isfinite(F))

    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    Rs, Fs = jax.jit(jax.vmap(rollout))(keys)
    assert Rs.shape == (8,)
    assert Fs.shape == (8, 6, 6)
    assert jnp.all(jnp.isfinite(Rs))
