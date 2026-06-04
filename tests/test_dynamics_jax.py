"""Tests for the JAX port of simulate_fast.

Must agree with the NumPy oracle (sim.dynamics.Satellite.simulate_fast) up to
floating-point tolerance, work under jit, and vmap correctly over a batch axis.
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from sim.dynamics import Satellite
from control.torque_generators import generate_torque_profile

jax = pytest.importorskip("jax")
jnp = jax.numpy

from sim.dynamics_jax import SatParams, simulate_jax, simulate_jax_jit  # noqa: E402


def _load_cfg(name):
    with open(ROOT / name) as f:
        return yaml.safe_load(f)


def _build_params(cfg) -> SatParams:
    I_sat = np.diag(cfg["satellite"]["inertia_tensor"])
    return SatParams(
        I_sat=jnp.asarray(I_sat),
        I_inv=jnp.asarray(np.linalg.inv(I_sat)),
        I_rw=float(cfg["reaction_wheels"]["inertia"]),
        rw_axes=jnp.asarray(cfg["reaction_wheels"]["alignment_matrix"], dtype=jnp.float32),
        rw_speed_max=float(cfg["reaction_wheels"]["max_speed"]),
        rw_torque_max=float(cfg["reaction_wheels"]["max_torque"]),
    )


def _torque_array(profile, dt, horizon, amplitude):
    t = np.arange(0, horizon + dt, dt)
    params = {
        "sine":       dict(frequency=0.01, amplitude=amplitude),
        "chirp":      dict(f0=0.005, f1=0.05, amplitude=amplitude),
        "prbs":       dict(amplitude=amplitude, switch_time=20),
    }[profile]
    return generate_torque_profile(profile, t, **params), t


@pytest.mark.parametrize("profile,amp", [
    ("sine", 0.0005),    # smooth, non-saturating
    ("sine", 0.005),     # saturating
    ("chirp", 0.005),
    ("prbs", 0.005),
])
def test_simulate_jax_matches_numpy_simulate_fast(profile, amp):
    """JAX trajectory must equal the NumPy oracle within fp tolerance."""
    cfg = _load_cfg("config_sat1.yaml")
    dt = cfg["sim"]["dt"]
    horizon = 60
    torque, _ = _torque_array(profile, dt, horizon, amp)

    sat = Satellite(
        cfg["satellite"]["inertia_tensor"], cfg["reaction_wheels"]["inertia"],
        cfg["reaction_wheels"]["alignment_matrix"],
        rw_speed_max=cfg["reaction_wheels"]["max_speed"],
        rw_torque_max=cfg["reaction_wheels"]["max_torque"],
    )

    # Numpy oracle: zero-order-hold control aligned with the dt grid
    t_grid = np.arange(0, horizon + dt, dt)
    def control_fn(t):
        return torque[int(np.clip(np.searchsorted(t_grid, t, side="left"),
                                  0, len(t_grid) - 1))]
    t_np, x_np = sat.simulate_fast(np.zeros(3), np.zeros(3), control_fn,
                                   (0, horizon), dt)

    # JAX: takes (T, 3) torque array, returns (T+1, 6) states (includes initial)
    params = _build_params(cfg)
    state0 = jnp.zeros(6)
    x_jax = simulate_jax(state0, jnp.asarray(torque), params, dt, substeps=10)

    # x_jax has one more time point than torque (includes t=0 init)
    # Compare on the same length as numpy output
    assert x_jax.shape[0] == len(t_np)
    # Loose tolerance: JAX uses float32 by default, NumPy uses float64
    np.testing.assert_allclose(np.asarray(x_jax[:, :3]), x_np[:, :3],
                               atol=1e-3, rtol=1e-3)


def test_simulate_jax_jit_compiles_and_runs():
    """JIT-compiled version must run and match non-jitted output."""
    cfg = _load_cfg("config_sat1.yaml")
    dt = cfg["sim"]["dt"]
    horizon = 30
    torque, _ = _torque_array("sine", dt, horizon, 0.0005)
    params = _build_params(cfg)
    state0 = jnp.zeros(6)

    x_eager = simulate_jax(state0, jnp.asarray(torque), params, dt, substeps=10)
    x_jit   = simulate_jax_jit(state0, jnp.asarray(torque), params, dt, substeps=10)

    np.testing.assert_allclose(np.asarray(x_jit), np.asarray(x_eager), atol=1e-6)


def test_simulate_jax_vmap_over_batch():
    """vmap over batched initial states + torques must produce (B, T+1, 6)."""
    cfg = _load_cfg("config_sat1.yaml")
    dt = cfg["sim"]["dt"]
    horizon = 20
    torque, _ = _torque_array("sine", dt, horizon, 0.0005)
    params = _build_params(cfg)

    B = 4
    rng = np.random.default_rng(0)
    state0_batch = jnp.asarray(rng.standard_normal((B, 6)) * 0.01)
    torque_batch = jnp.broadcast_to(jnp.asarray(torque), (B,) + torque.shape)

    sim_batched = jax.vmap(
        simulate_jax, in_axes=(0, 0, None, None, None)
    )
    x = sim_batched(state0_batch, torque_batch, params, dt, 10)
    assert x.shape == (B, len(torque), 6)
    # The first state of each batch must equal the input init
    np.testing.assert_allclose(np.asarray(x[:, 0]), np.asarray(state0_batch),
                               atol=1e-6)


def test_simulate_jax_runs_without_nan_at_saturation():
    """Even with saturating dynamics, output must be finite."""
    cfg = _load_cfg("config_sat1.yaml")
    dt = cfg["sim"]["dt"]
    horizon = 80
    torque, _ = _torque_array("sine", dt, horizon, 0.01)  # forces saturation
    params = _build_params(cfg)

    x = simulate_jax(jnp.zeros(6), jnp.asarray(torque), params, dt, substeps=10)
    assert jnp.all(jnp.isfinite(x))
    # RW speeds should be clipped (or limited via apply_rw_limits) to ≤ max + small margin
    rw_max = cfg["reaction_wheels"]["max_speed"]
    assert float(jnp.max(jnp.abs(x[:, 3:6]))) <= rw_max * 1.01
