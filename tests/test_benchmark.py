"""
End-to-end benchmark: LS and EKF estimation accuracy across satellite configs and profiles.

Run with:   pytest tests/test_benchmark.py -v -s
Skip with:  pytest -m "not slow"

All assertions use a 10% relative error threshold — deliberately generous, since this
benchmark documents actual performance rather than enforcing tight tolerances.
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

from sim.dynamics import Satellite
from sim.sensors import SensorSuite, get_high_accuracy_sensor_config
from estimation.ekf import EKFInertiaRW, EKFConfig
from estimation.ls_estimator import LeastSquaresEstimator, DiagonalInertiaModel
from control.torque_generators import generate_torque_profile
from utils.signal_processing import compute_smooth_derivative

ROOT = Path(__file__).parent.parent


def _load_config(config_file: str) -> dict:
    with open(ROOT / config_file) as f:
        return yaml.safe_load(f)


def _run_pipeline(config_file: str, profile: str, torque_params: dict,
                  horizon: int = 150, seed: int = 42,
                  use_noisy_sensors: bool = True,
                  use_ekf: bool = True, use_ls: bool = True) -> dict:
    """
    Minimal estimation pipeline (no visualization) for benchmarking.

    Returns dict with keys:
        ls_estimate   (3,) or None
        ekf_estimate  (3,) or None
        true_I        (3,)
        ls_rel_err    float or None
        ekf_rel_err   float or None
    """
    np.random.seed(seed)
    cfg = _load_config(config_file)

    I_sat = cfg["satellite"]["inertia_tensor"]
    I_rw = cfg["reaction_wheels"]["inertia"]
    rw_axes = cfg["reaction_wheels"]["alignment_matrix"]
    dt = cfg["sim"]["dt"]
    omega0 = cfg["sim"]["initial_omega"]
    rw_speed0 = cfg["sim"]["initial_rw_speed"]
    rw_max_speed = cfg["reaction_wheels"]["max_speed"]
    rw_max_torque = cfg["reaction_wheels"]["max_torque"]
    I_true = np.array(I_sat)

    # Generate torque profile
    t = np.arange(0, horizon + dt, dt)
    torque_data = generate_torque_profile(profile, t, **torque_params)

    def control_fn(time_val):
        idx = int(np.clip(np.searchsorted(t, time_val, side='left'), 0, len(t) - 1))
        return torque_data[idx]

    # Simulate
    sat = Satellite(I_sat, I_rw, rw_axes, rw_speed_max=rw_max_speed,
                    rw_torque_max=rw_max_torque)
    t_sim, states = sat.simulate(omega0, rw_speed0, control_fn, (0, horizon), dt)

    # Noisy observations
    if use_noisy_sensors:
        np.random.seed(seed + 1)
        sensor = SensorSuite(**get_high_accuracy_sensor_config())
        measured_states = np.array([
            np.concatenate([
                sensor.get_all_measurements({'omega': s[:3]}, t_sim[i])['omega'],
                s[3:6]
            ])
            for i, s in enumerate(states)
        ])
    else:
        measured_states = states.copy()

    omega_data = measured_states[:, :3]
    rw_speeds_data = measured_states[:, 3:6]
    actual_torques = sat.get_tau_actual_at_times(t_sim)

    domega_data = np.zeros_like(omega_data)
    for axis in range(3):
        domega_data[:, axis] = compute_smooth_derivative(
            t_sim, omega_data[:, axis], method='central_diff'
        )

    dynamics_data = {
        'omega': omega_data,
        'domega': domega_data,
        'torque': actual_torques,
        'rw_speeds': rw_speeds_data,
        'time': t_sim,
    }

    result = {'true_I': I_true, 'ls_estimate': None, 'ekf_estimate': None,
              'ls_rel_err': None, 'ekf_rel_err': None}

    # LS estimation
    if use_ls:
        ls = LeastSquaresEstimator(DiagonalInertiaModel(), satellite_model=sat,
                                   robust_loss='huber')
        ls_result = ls.estimate(dynamics_data, verbose=False)
        if ls_result['success']:
            result['ls_estimate'] = ls_result['inertia_tensor'].diagonal()
            result['ls_rel_err'] = float(
                np.linalg.norm(result['ls_estimate'] - I_true) / np.linalg.norm(I_true)
            )

    # EKF estimation
    if use_ekf:
        I_init = I_true * 0.85
        x0 = np.zeros(9)
        x0[3:6] = I_init
        sigma_omega = 1e-6 if not use_noisy_sensors else 1e-4
        # Scale P0 and Qc with I_init so tuning is valid across satellites of different sizes.
        # P0: 30% initial uncertainty covers the 15% initialization bias at ~0.5σ.
        # Qc: slow random walk at 1e-7 relative variance per second.
        sigma_I = 0.30 * I_init
        cfg_ekf = EKFConfig(
            dt=dt,
            I_rw_diag=np.array([I_rw, I_rw, I_rw]),
            Qc_diag=np.array([1e-9]*3 + list(1e-7 * I_init**2) + [1e-9]*3),
            R_diag=np.array([sigma_omega**2]*3 + [sigma_omega**2]*3),
            x0=x0,
            P0=np.diag([1e-4]*3 + list(sigma_I**2) + [1e-2]*3),
        )
        ekf = EKFInertiaRW(cfg_ekf)
        rw_accels = sat.get_rw_acc_at_times(t_sim)
        for k in range(len(t_sim)):
            z_k = measured_states[k]
            u_k = rw_accels[k]
            x_k, _ = ekf.step(u_k, z_k, tau_ext=sat.tau_ext[k])

        result['ekf_estimate'] = x_k[3:6].copy()
        result['ekf_rel_err'] = float(
            np.linalg.norm(result['ekf_estimate'] - I_true) / np.linalg.norm(I_true)
        )

    return result


# ---------------------------------------------------------------------------
# Parametric benchmark tests
# ---------------------------------------------------------------------------

SAT_CONFIGS = [
    ("config_sat1.yaml", "sat1"),
    ("config_sat2.yaml", "sat2"),
    ("config_sat3.yaml", "sat3"),
]

PROFILE_PARAMS = {
    'sat1': {
        'sine':       {'frequency': 0.01,  'amplitude': 0.01},
        'chirp':      {'f0': 0.005, 'f1': 0.05, 'amplitude': 0.01},
        'prbs':       {'amplitude': 0.01,  'switch_time': 20},
        'multi step': {'amplitude': 0.012, 'step_duration': 40.0},
    },
    'sat2': {
        'sine':       {'frequency': 0.01,  'amplitude': 0.05},
        'chirp':      {'f0': 0.005, 'f1': 0.05, 'amplitude': 0.05},
        'prbs':       {'amplitude': 0.05,  'switch_time': 20},
        'multi step': {'amplitude': 0.06,  'step_duration': 40.0},
    },
    'sat3': {
        'sine':       {'frequency': 0.01,  'amplitude': 0.10},
        'chirp':      {'f0': 0.005, 'f1': 0.05, 'amplitude': 0.10},
        'prbs':       {'amplitude': 0.10,  'switch_time': 20},
        'multi step': {'amplitude': 0.12,  'step_duration': 40.0},
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("config_file,sat_key", SAT_CONFIGS)
@pytest.mark.parametrize("profile", ['sine', 'chirp', 'prbs', 'multi step'])
def test_ls_accuracy_within_10pct(config_file, sat_key, profile, capsys):
    """LS estimation must achieve <10% relative error on any satellite and profile."""
    params = PROFILE_PARAMS[sat_key][profile]
    result = _run_pipeline(config_file, profile, params, horizon=150, use_ekf=False)

    with capsys.disabled():
        print(f"\n[LS] {sat_key} / {profile}: rel_err={result['ls_rel_err']:.4f}  "
              f"estimate={np.round(result['ls_estimate'], 4)}  "
              f"true={result['true_I']}")

    assert result['ls_estimate'] is not None, "LS estimation returned None (failed)"
    assert result['ls_rel_err'] < 0.10, (
        f"LS relative error {result['ls_rel_err']:.4f} > 10% for {sat_key}/{profile}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("config_file,sat_key", SAT_CONFIGS)
@pytest.mark.parametrize("profile", ['sine', 'chirp', 'prbs', 'multi step'])
def test_ekf_accuracy_within_10pct(config_file, sat_key, profile, capsys):
    """EKF estimation must achieve <10% relative error on any satellite and profile."""
    params = PROFILE_PARAMS[sat_key][profile]
    result = _run_pipeline(config_file, profile, params, horizon=150, use_ls=False)

    with capsys.disabled():
        print(f"\n[EKF] {sat_key} / {profile}: rel_err={result['ekf_rel_err']:.4f}  "
              f"estimate={np.round(result['ekf_estimate'], 4)}  "
              f"true={result['true_I']}")

    assert result['ekf_estimate'] is not None
    assert result['ekf_rel_err'] < 0.10, (
        f"EKF relative error {result['ekf_rel_err']:.4f} > 10% for {sat_key}/{profile}"
    )


@pytest.mark.slow
def test_observability_score_correlates_with_estimation_error():
    """The observability score should negatively correlate with estimation error.

    This tests the paper's core claim: higher observability → lower estimation error.
    We run all profiles on sat1 and verify Pearson correlation < 0.0.
    """
    from utils.observability import score_profiles_canonical

    cfg = _load_config("config_sat1.yaml")
    I_ref = tuple(cfg["satellite"]["inertia_tensor"])
    t = np.linspace(0, 150, 150)

    profiles = ['sine', 'chirp', 'prbs', 'multi step']

    obs_scores = []
    est_errors = []

    for profile in profiles:
        params = PROFILE_PARAMS['sat1'][profile]
        torques = generate_torque_profile(profile, t, **params)
        obs_score = score_profiles_canonical({'p': torques}, dt=1.0, I_ref=I_ref)['p']['score']
        obs_scores.append(obs_score)

        result = _run_pipeline("config_sat1.yaml", profile, params,
                               horizon=150, use_ekf=False)
        if result['ls_rel_err'] is not None:
            est_errors.append(result['ls_rel_err'])
        else:
            est_errors.append(1.0)

    obs_scores = np.array(obs_scores)
    est_errors = np.array(est_errors)

    corr = float(np.corrcoef(obs_scores, est_errors)[0, 1])
    print(f"\nObservability vs. estimation error correlation: {corr:.3f}")
    print(f"Profiles: {profiles}")
    print(f"Obs scores: {np.round(obs_scores, 4)}")
    print(f"Est errors: {np.round(est_errors, 4)}")

    assert corr < 0.0, (
        f"Expected negative correlation (higher obs → lower error), got corr={corr:.3f}. "
        f"Observability metric may not be predictive."
    )
