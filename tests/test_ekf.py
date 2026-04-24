import numpy as np
import pytest
from estimation.ekf import EKFInertiaRW, EKFConfig


@pytest.fixture
def ekf_instance():
    """EKF initialized with 10% biased inertia guess for sat1-like parameters."""
    I_true = np.array([0.26, 0.26, 0.16])
    I_init = I_true * 0.9

    x0 = np.zeros(9)
    x0[3:6] = I_init

    cfg = EKFConfig(
        dt=1.0,
        I_rw_diag=np.array([1e-4, 1e-4, 1e-4]),
        Qc_diag=np.array([1e-9]*3 + [1e-5]*3 + [1e-9]*3),
        R_diag=np.array([1e-12]*3 + [1e-12]*3),
        x0=x0,
        P0=np.diag([1e-4]*3 + [1.0]*3 + [1e-2]*3),
    )
    return EKFInertiaRW(cfg), I_true


def _simulate_gt(I_true, N=300, dt=1.0, seed=42):
    """Simulate ground-truth trajectory under sinusoidal torque, no RW."""
    from utils.math_utils import skew
    rng = np.random.default_rng(seed)
    t = np.arange(N) * dt
    tau_rw = 0.005 * np.column_stack([
        np.sin(2 * np.pi * 0.01 * t),
        np.sin(2 * np.pi * 0.03 * t + 0.5),
        np.sin(2 * np.pi * 0.07 * t + 1.0),
    ])

    I_rw_scalar = 1e-4
    I_mat = np.diag(I_true)
    omega = np.zeros(3)
    rw_speeds = np.zeros(3)
    omega_hist = np.zeros((N, 3))
    rw_hist = np.zeros((N, 3))

    for k in range(N):
        rw_acc = tau_rw[k] / I_rw_scalar
        h_rw = I_rw_scalar * rw_speeds
        h_rw_dot = I_rw_scalar * rw_acc
        h_total = I_mat @ omega + h_rw
        domega = np.linalg.solve(I_mat, -skew(omega) @ h_total - h_rw_dot)
        omega = omega + dt * domega
        rw_speeds = rw_speeds + dt * rw_acc
        omega_hist[k] = omega
        rw_hist[k] = rw_speeds

    omega_meas = omega_hist + rng.standard_normal(omega_hist.shape) * 1e-4
    rw_meas = rw_hist + rng.standard_normal(rw_hist.shape) * 1e-4

    return tau_rw, omega_meas, rw_meas


class TestEKFStep:
    def test_step_returns_correct_shapes(self, ekf_instance):
        ekf, _ = ekf_instance
        u = np.zeros(3)
        z = np.zeros(6)
        x, P = ekf.step(u, z)
        assert x.shape == (9,)
        assert P.shape == (9, 9)

    def test_inertia_stays_positive(self, ekf_instance):
        """EKF must enforce positive inertia at all times."""
        ekf, I_true = ekf_instance
        tau_rw, omega_meas, rw_meas = _simulate_gt(I_true)
        for k in range(len(omega_meas)):
            z_k = np.concatenate([omega_meas[k], rw_meas[k]])
            u_k = tau_rw[k] / 1e-4
            x, _ = ekf.step(u_k, z_k)
            assert np.all(x[3:6] > 0), f"Inertia became non-positive at step {k}: {x[3:6]}"

    def test_covariance_is_symmetric_positive_semidefinite(self, ekf_instance):
        ekf, I_true = ekf_instance
        tau_rw, omega_meas, rw_meas = _simulate_gt(I_true)
        for k in range(0, len(omega_meas), 30):
            z_k = np.concatenate([omega_meas[k], rw_meas[k]])
            u_k = tau_rw[k] / 1e-4
            _, P = ekf.step(u_k, z_k)
            np.testing.assert_allclose(P, P.T, atol=1e-10,
                                        err_msg=f"Covariance not symmetric at step {k}")
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-10), (
                f"Covariance has negative eigenvalue at step {k}: min={eigvals.min():.3e}"
            )


class TestEKFConvergence:
    def test_inertia_converges_within_5pct(self, ekf_instance):
        """After 300 steps, EKF inertia estimate should be within 5% of truth."""
        ekf, I_true = ekf_instance
        tau_rw, omega_meas, rw_meas = _simulate_gt(I_true)
        N = len(omega_meas)
        for k in range(N):
            z_k = np.concatenate([omega_meas[k], rw_meas[k]])
            u_k = tau_rw[k] / 1e-4
            x, _ = ekf.step(u_k, z_k)

        I_est = x[3:6]
        rel_err = np.abs(I_est - I_true) / I_true
        assert np.all(rel_err < 0.05), (
            f"EKF did not converge within 5%: rel_err={rel_err}, "
            f"I_est={I_est}, I_true={I_true}"
        )
