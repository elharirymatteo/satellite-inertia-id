"""Verify the MPC planner increases log-det(F) over its horizon and
respects the torque box constraint."""
import jax
import jax.numpy as jnp
import numpy as np

from sim.dynamics_jax import SatParams
from rl.t1_env import sample_sat
from control.dual_mpc import plan


def _toy_sat():
    return sample_sat(jax.random.PRNGKey(0), (0.1, 20.0),
                      I_rw=1e-4, rw_axes=jnp.eye(3),
                      rw_speed_max=600.0, rw_torque_max=0.05,
                      log_uniform=True, max_tilt_angle=np.pi / 8)


def _rollout_logdet(sat, sat_state, F0, tau_seq, dt, substeps):
    """Roll out dynamics under tau_seq, return final logdet(F + eps I)."""
    from sim.dynamics_jax import _step_dt
    from utils.observability import regression_rows_full
    F = F0
    last_omega = sat_state[0:3]
    for k in range(tau_seq.shape[0]):
        sat_state = _step_dt(sat_state, tau_seq[k], sat, dt / substeps, substeps)
        new_omega = sat_state[0:3]
        domega = (new_omega - last_omega) / dt
        mid_omega = 0.5 * (new_omega + last_omega)
        R = regression_rows_full(mid_omega, domega)
        F = F + R.T @ R
        last_omega = new_omega
    L = jnp.linalg.cholesky(F + 1e-6 * jnp.eye(6))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L))), sat_state, F


def test_plan_improves_logdet_over_zero_actions():
    sat = _toy_sat()
    init_state = jnp.zeros(6)
    F0 = jnp.zeros((6, 6))
    tau_max = 0.01
    horizon = 10
    dt, substeps = 0.1, 10

    zero_seq = jnp.zeros((horizon, 3))
    ld_zero, _, _ = _rollout_logdet(sat, init_state, F0, zero_seq, dt, substeps)

    tau_plan = plan(init_state, F0, sat, horizon=horizon,
                    n_opt_steps=30, tau_max=tau_max,
                    lr=0.01 * tau_max, dt=dt, substeps=substeps,
                    sat_penalty=0.0)
    ld_plan, _, _ = _rollout_logdet(sat, init_state, F0, tau_plan, dt, substeps)

    assert ld_plan > ld_zero, f"plan logdet {ld_plan} did not improve over zero {ld_zero}"


def test_plan_respects_tau_max():
    sat = _toy_sat()
    init_state = jnp.zeros(6)
    F0 = jnp.zeros((6, 6))
    tau_max = 0.005
    tau_plan = plan(init_state, F0, sat, horizon=10,
                    n_opt_steps=30, tau_max=tau_max,
                    lr=0.01 * tau_max, dt=0.1, substeps=10,
                    sat_penalty=0.0)
    assert jnp.max(jnp.abs(tau_plan)) <= tau_max + 1e-6
