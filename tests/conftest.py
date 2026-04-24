import numpy as np
import pytest


@pytest.fixture
def synthetic_diagonal_data():
    """Noise-free synthetic dynamics data for a known diagonal inertia.

    Uses Euler's equation directly (no RW) so LS should recover I_true exactly.
    """
    rng = np.random.default_rng(42)
    I_true = np.array([0.30, 0.50, 0.80])
    N = 400
    omega = rng.standard_normal((N, 3)) * 0.3
    tau = rng.standard_normal((N, 3)) * 0.05

    domega = np.zeros((N, 3))
    I_mat = np.diag(I_true)
    for k in range(N):
        domega[k] = np.linalg.solve(
            I_mat, tau[k] - np.cross(omega[k], I_mat @ omega[k])
        )

    return {
        'omega': omega,
        'domega': domega,
        'torque': tau,
        'I_true': I_true,
    }


@pytest.fixture
def noisy_diagonal_data(synthetic_diagonal_data):
    """Same as synthetic_diagonal_data but with 1% Gaussian noise added."""
    rng = np.random.default_rng(7)
    data = dict(synthetic_diagonal_data)
    noise_scale = 0.01 * np.std(data['omega'])
    data['omega'] = data['omega'] + rng.standard_normal(data['omega'].shape) * noise_scale
    acc_noise = 0.01 * np.std(data['domega'])
    data['domega'] = data['domega'] + rng.standard_normal(data['domega'].shape) * acc_noise
    return data
