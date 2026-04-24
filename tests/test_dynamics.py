import numpy as np
import pytest
from sim.dynamics import Satellite


@pytest.fixture
def simple_satellite():
    I_sat = [0.26, 0.26, 0.16]
    I_rw = 0.0001
    rw_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return Satellite(I_sat, I_rw, rw_axes, rw_speed_max=460, rw_torque_max=0.01)


def test_tau_ext_exists_after_simulate(simple_satellite):
    """Base Satellite must have tau_ext after simulate() so EKF can access sat.tau_ext[k]."""
    sat = simple_satellite
    t_span = (0, 10)
    dt = 1.0
    omega0 = np.zeros(3)
    rw_speed0 = np.zeros(3)
    t_sim, states = sat.simulate(omega0, rw_speed0, lambda t: np.zeros(3), t_span, dt)

    assert hasattr(sat, 'tau_ext'), "sat.tau_ext must exist after simulate()"
    assert sat.tau_ext.shape == (len(t_sim), 3), (
        f"Expected shape ({len(t_sim)}, 3), got {sat.tau_ext.shape}"
    )
    np.testing.assert_allclose(sat.tau_ext, 0.0)


def test_angular_accel_consistent_with_torque_smoothing(simple_satellite):
    """compute_angular_accelerations must not mutate smoothing state."""
    I_sat = [0.26, 0.26, 0.16]
    I_rw = 0.0001
    rw_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sat = Satellite(I_sat, I_rw, rw_axes, rw_speed_max=460, rw_torque_max=0.01,
                    torque_smoothing=True, torque_rate_limit=0.005)

    t_span = (0, 30)
    dt = 1.0
    omega0 = np.zeros(3)
    rw_speed0 = np.zeros(3)

    def constant_torque(t):
        return np.array([0.005, 0.003, 0.004])

    t_sim, states = sat.simulate(omega0, rw_speed0, constant_torque, t_span, dt)

    prev_tau_after_sim = sat.prev_tau.copy()
    prev_time_after_sim = sat.prev_time

    # Call again — must not change smoothing state
    sat.compute_angular_accelerations(t_sim, states, constant_torque)

    np.testing.assert_allclose(sat.prev_tau, prev_tau_after_sim, atol=1e-15,
                                err_msg="compute_angular_accelerations must not mutate prev_tau")
    assert sat.prev_time == prev_time_after_sim, (
        "compute_angular_accelerations must not mutate prev_time")


def test_angular_accel_uses_stored_history(simple_satellite):
    """angular_accelerations array must be finite and have correct shape."""
    sat = simple_satellite
    t_span = (0, 20)
    dt = 1.0
    t_sim, states = sat.simulate(
        np.zeros(3), np.zeros(3), lambda t: np.array([0.005, 0.003, 0.004]),
        t_span, dt
    )
    accs = sat.angular_accelerations
    assert accs.shape == (len(t_sim), 3), f"Expected ({len(t_sim)}, 3), got {accs.shape}"
    assert np.all(np.isfinite(accs)), "angular_accelerations must be finite"
