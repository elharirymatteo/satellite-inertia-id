"""Verify the full-tensor regression-row function matches the analytic Jacobian
of Euler's equation in the 6 inertia parameters."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # noqa: E402  needed for 1e-10 tolerance
import jax.numpy as jnp
import pytest

from utils.observability import regression_rows_full


def _euler_tau(I_params, omega, omega_dot):
    """tau = I @ omega_dot + omega x (I @ omega) with I built from the 6-vector."""
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I_params
    I = jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])
    return I @ omega_dot + jnp.cross(omega, I @ omega)


@pytest.mark.parametrize("seed", range(8))
def test_regression_rows_match_numerical_jacobian(seed):
    rng = np.random.default_rng(seed)
    omega = jnp.asarray(rng.normal(0, 0.5, size=3))
    omega_dot = jnp.asarray(rng.normal(0, 0.5, size=3))

    # Analytic Jacobian of tau w.r.t. the 6-vector at any I (linear -> constant)
    I_params0 = jnp.zeros(6)
    J = jax.jacobian(_euler_tau, argnums=0)(I_params0, omega, omega_dot)  # (3, 6)

    R = regression_rows_full(omega, omega_dot)
    np.testing.assert_allclose(np.asarray(R), np.asarray(J), atol=1e-10)


def test_ls_recovers_known_full_tensor_noise_free():
    """With perfect (omega, omega_dot, tau) data over a varied trajectory, stacked
    LS must recover the true 6-vector exactly (up to float precision)."""
    rng = np.random.default_rng(42)
    I_true = jnp.array([0.30, 0.40, 0.50, 0.02, -0.015, 0.01])
    N = 100
    omegas = jnp.asarray(rng.normal(0, 0.4, size=(N, 3)))
    omega_dots = jnp.asarray(rng.normal(0, 0.4, size=(N, 3)))
    taus = jnp.stack([
        regression_rows_full(o, od) @ I_true
        for o, od in zip(omegas, omega_dots)
    ])
    R_blocks = jnp.stack([regression_rows_full(o, od)
                          for o, od in zip(omegas, omega_dots)])
    R = R_blocks.reshape(3 * N, 6)
    y = taus.reshape(3 * N)
    theta_hat = jnp.linalg.solve(R.T @ R, R.T @ y)
    np.testing.assert_allclose(np.asarray(theta_hat), np.asarray(I_true), atol=1e-10)
