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
