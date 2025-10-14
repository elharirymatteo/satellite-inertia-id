#!/usr/bin/env python3
"""
Enhanced simulation script with multiple estimation methods and external torques
"""
import os
import csv
import yaml
import numpy as np
from tqdm import tqdm
from sim.dynamics import Satellite, SatelliteWithExternalTorques
from sim.actuators import ReactionWheelArray, get_standard_rw_config
from sim.sensors import SensorSuite, get_high_accuracy_sensor_config
from estimation.ls_estimator import LeastSquaresEstimator, DiagonalInertiaModel, FullInertiaModel
from scripts.visualize import create_comprehensive_report
from control.torque_generators import generate_torque_profile
from utils.signal_processing import compute_smooth_derivative
from scipy.signal import savgol_filter
from estimation.ekf import EKFInertiaRW, EKFConfig

# === Baseline from tuned sat1 ===
baseline = {
    'sine': 0.01,
    'chirp': 0.01,
    'prbs': 0.01,
    'one step': 0.015,
    'multi step': 0.012,
}

# === Scaling factors ===
scaling = {
    'sat1': 1.0,
    'sat2': 5.0,
    'sat3': 9.85,
}

# === Adjusted amplitude dictionaries ===
def get_profile_params_for_sat(sat_key):
    s = scaling[sat_key]
    return {
        'sine': {'frequency': 0.01, 'amplitude': baseline['sine'] * s},
        'chirp': {'f0': 0.005, 'f1': 0.05, 'amplitude': baseline['chirp'] * s},
        'prbs': {'amplitude': baseline['prbs'] * s, 'switch_time': 20},
        'one step': {'amplitude': baseline['one step'] * s, 'step_duration': 40.0},
        'multi step': {'amplitude': baseline['multi step'] * s, 'step_duration': 40.0},
        'sawtooth': {'amplitude': 0.02 * s, 'frequency': 0.01},  
        'multi sine': {
            'frequencies': [[0.002, 0.005], [0.003, 0.007], [0.004, 0.008]],
            'amplitudes': [
                [0.01 * s, 0.007 * s],
                [0.009 * s, 0.006 * s],
                [0.008 * s, 0.005 * s]
            ],
            'phases': [[0, 0], [np.pi/4, 0], [-np.pi/3, 0]],
        },

        'sine-3axis': {
            'freqs': [0.01, 0.01, 0.01],
            'amps': [0.03 * s, 0.015 * s, 0.025 * s],
            'phases': [0.0, 0.0, 0.0],
        }
    }

def load_config(config_file="config.yaml"):
    """Load configuration from file"""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
    
def run_enhanced_simulation(config_file="config_sat1.yaml", ls_model=DiagonalInertiaModel(),
                            use_external_torques=False, use_ekf=False, use_ls=True,
                          use_realistic_actuators=False, use_noisy_sensors=True,
                          torque_profile="multi_sine", torque_params=None,
                          seed=42, verbose=True, horizon=False, dynamic_I_func_name=None):
    """
    Run enhanced simulation with all features
    
    Args:
        config_file: Configuration file path
        use_external_torques: Whether to include gravity gradient and SRP
        use_realistic_actuators: Whether to use realistic RW dynamics
        use_noisy_sensors: Whether to add sensor noise
        torque_profile: Type of control torque pattern ('multi_sine', 'prbs', 'chirp', etc.)
        torque_params: Parameters for the selected torque profile
    """
    seed = seed  # For reproducibility
    np.random.seed(seed)

    if verbose:
        print("🚀 Enhanced Inertia Identification Simulation")
        print("=" * 50)

    ########################################################################
    #                1) Load configuration and parameters                  #
    ########################################################################
    # Load configuration
    cfg = load_config(config_file)
    # Parameters
    I_sat = cfg["satellite"]["inertia_tensor"]
    I_rw = cfg["reaction_wheels"]["inertia"]
    rw_axes = cfg["reaction_wheels"]["alignment_matrix"]
    dt = cfg["sim"]["dt"]
    omega0 = cfg["sim"]["initial_omega"]
    rw_speed0 = cfg["sim"]["initial_rw_speed"]
    rw_max_speed = cfg["reaction_wheels"]["max_speed"]
    rw_max_torque = cfg["reaction_wheels"]["max_torque"]
    t_max = horizon if horizon else int(cfg["sim"]["t_max"])
    if dynamic_I_func_name is not None:
        Inertia = np.array(I_sat, dtype=float)
        Inertia = np.diag(Inertia) if Inertia.ndim == 1 else Inertia
        use_dynamic_I = True
        if dynamic_I_func_name == "step_change": # if t<t_max/2 then I_sat else I_sat * 1.1
            dynamic_I_func = lambda t: Inertia if t < t_max / 2 else Inertia * 1.1
        elif dynamic_I_func_name == "slow_drift":
            # 𝐼 ( 𝑡 ) = 𝐼 0 + 𝐼 ˙ 𝑡 I(t)=I 0 ​ + I ˙ t or a smooth ramp (e.g., tanh or spline).
            dynamic_I_func = lambda t: Inertia * (1 + 0.05 * t / t_max)  # Linear drift
        elif dynamic_I_func_name == "periodic": #sloshing proxy
            # I(t)=I 0 ​ +Asin(2πft) (per-axis amplitudes and phases). Magnitude: peak ±2–5% (keep small to avoid unphysical extremes).
            dynamic_I_func = lambda t: Inertia * (1 + 0.03 * np.sin(2 * np.pi * 0.02 * t))
    else:
        use_dynamic_I = False
        dynamic_I_func = None

    if verbose:
        print(f"True inertia: {I_sat}")
        print(f"Simulation time: {t_max} s")
        print(f"External torques: {use_external_torques}")
        print(f"Realistic actuators: {use_realistic_actuators}")
        print(f"Noisy sensors: {use_noisy_sensors}")
        print(f"Torque profile: {torque_profile}")
        print()

    # Generate time vector FIRST (this was the problem!)
    t = np.arange(0, t_max + dt, dt)

    # Generate ideal torque profile (now t is defined)
    if verbose:
        print(f"Generating {torque_profile} torque profile...")
    ideal_torque_data = generate_torque_profile(torque_profile, t, **torque_params)
    
    # Apply realistic actuator dynamics if requested
    if use_realistic_actuators:
        if verbose:
            print("Applying realistic actuator dynamics...")
        # Convert satellite RW config to ReactionWheelArray format
        rw_config = {
            'wheels': [],
            'noise_std': 0.05 * rw_max_torque  # Array-level noise setting
        }
        # Create individual wheel configurations from satellite config
        for i, axis in enumerate(rw_axes):
            wheel_config = {
                'inertia': I_rw,
                'max_torque': rw_max_torque,
                'max_speed': rw_max_speed,
                'axis': axis,
                'damping': 1e-5,
                'friction': 1e-6
            }
            rw_config['wheels'].append(wheel_config)
        
        rw_array = ReactionWheelArray(rw_config)
        
        # Precompute realistic torques
        realistic_torque_data = np.zeros_like(ideal_torque_data)
        wheel_speeds_estimate = np.zeros(3)  # Or use a reasonable model
        
        for i, (time_val, ideal_torque) in enumerate(zip(t, ideal_torque_data)):
            # Apply actuator dynamics (could include speed evolution model)
            realistic_torque_data[i] = rw_array.apply_actuator_dynamics(
                ideal_torque, time_val, wheel_speeds_estimate
            )
            
            # Simple wheel speed evolution (if needed for more realism)
            # wheel_speeds_estimate += (ideal_torque / I_rw) * dt
        
        torque_data_full = realistic_torque_data
        if verbose:
            print("Realistic actuator effects applied")
    else:
        torque_data_full = ideal_torque_data
    
    # Control function now just interpolates precomputed data
    def enhanced_control_input(time_val):
        idx = np.argmin(np.abs(t - time_val))
        return torque_data_full[idx]

    ########################################################################
    #               2) Satellite init & trajectory simulation              #
    ########################################################################
    # Initialize satellite
    if use_external_torques:
        orbital_params = {
            'orbital_rate': 0.001,  # rad/s (about 1000s orbit period)
            'srp_coefficient': 1e-6,  # N (small satellite)
            'cp_offset': [0.01, 0.01, 0.01]  # m (1cm offset from center of mass)
        }
        sat = SatelliteWithExternalTorques(I_sat, I_rw, rw_axes, rw_max_speed, rw_max_torque, orbital_params)
        if verbose:
            print("Using satellite model with external disturbances")
            print(f"  Orbital rate: {orbital_params['orbital_rate']} rad/s")
            print(f"  SRP coefficient: {orbital_params['srp_coefficient']} N")
            print(f"  CP offset: {orbital_params['cp_offset']} m")
    else:
        sat = Satellite(I_sat, I_rw, rw_axes, rw_max_speed, rw_max_torque, use_dynamic_inertia=use_dynamic_I, dynamic_inertia_func=dynamic_I_func)
        if verbose:
            print("Using satellite model without external disturbances")

    print("Running simulation...")
    if use_external_torques:
        t_sim, states = sat.simulate_with_external_torques(
            omega0, rw_speed0, enhanced_control_input, (0, t_max), dt
        )

    else:
        t_sim, states = sat.simulate(omega0, rw_speed0, enhanced_control_input, (0, t_max), dt)

    actual_torques = sat.get_tau_actual_at_times(t_sim)
    
    # Add sensor noise if requested
    if use_noisy_sensors:
        np.random.seed(seed + 1)  # Different seed for sensors
        if verbose:
            print("Using realistic sensor models with noise")
        sensor_suite = SensorSuite(**get_high_accuracy_sensor_config())
        noisy_states = []
        for i, state in enumerate(states):
            measurements = sensor_suite.get_all_measurements({'omega': state[:3]}, t_sim[i])
            noisy_state = state.copy()
            noisy_state[:3] = measurements['omega']
            noisy_states.append(noisy_state)
        states_measured = np.array(noisy_states)
    else:
        states_measured = states
    
    ########################################################################
    #                3) Data preparation for estimation                    #
    ########################################################################
    # Prepare data including RW speeds
    omega_data = states_measured[:, :3]
    rw_speeds_data = states_measured[:, 3:6]  # Extract RW speeds from simulation
    rw_accels = np.diff(rw_speeds_data, axis=0) / dt
    rw_accels = np.vstack([rw_accels, rw_accels[-1]]) # pad the last value to match time steps

    # Create smooth angular velocity
    # omega_smooth = savgol_filter(omega_data, window_length=15, polyorder=3, axis=0)

    # Then compute the smooth derivatives
    domega_data = np.zeros_like(omega_data)
    for axis in range(3):
        domega_data[:, axis] = compute_smooth_derivative(t_sim, omega_data[:, axis], method='central_diff')

    if verbose:
        domega_true = sat.angular_accelerations
        print("  Angular acceleration data prepared.")
        print(f"  True angular acceleration error: {np.linalg.norm(domega_true - domega_data):.4f}")

    dynamics_data = {
        'omega': omega_data,
        'domega': domega_data,
        'torque': actual_torques,
        'rw_speeds': rw_speeds_data,  # Add RW speeds to data
        'time': t_sim  # Add time data for external torque computation
    }
    
    # Print info about the ls model being used
    if verbose:
        print(f"Optimizing with {ls_model.__class__.__name__}...")
        print(f"Parameter bounds: {ls_model.get_param_bounds()}")
    
    optimization_results = {}
    estimators_results = {}

    ########################################################################
    #                          4) LS Estimation                            #
    ########################################################################

    estimation_data = {}
    estimated_states = {}
    if use_ls:
        if verbose:
            print("Running Least Squares estimation...")

        ls_estimator = LeastSquaresEstimator(ls_model, satellite_model=sat, robust_loss='huber')
        # if not use_dynamic_I:
        # Create LS estimator with satellite model
        result = ls_estimator.estimate(dynamics_data, verbose=False)
        if verbose:
            validation_diag = ls_estimator.validate_estimate(result, dynamics_data)
            print(f"Diagonal model: {validation_diag['message']}")
        # else:
        #     W_fraction = 0.4   # e.g. 40% of trajectory
        #     W_secs = t_max * W_fraction
        #     W_steps = int(W_secs / dt)
        #     step_interval = 20  # how often to recompute LS

        #     N = len(t_sim)
        #     I_ls_piecewise = np.zeros((N, 3))  # time series
        #     for k in range(N):
        #         if k < W_steps or k % step_interval != 0:
        #             I_ls_piecewise[k] = I_ls_piecewise[k - 1] if k > 0 else np.array([np.nan]*3)
        #             continue

        #         # Select windowed data
        #         window_data = {
        #             key: val[k - W_steps:k]
        #             for key, val in dynamics_data.items()
        #             if val is not None and len(val) >= k
        #         }

        #         result = ls_estimator.estimate(window_data, verbose=False)

        if result['success']:
            optimization_results['LS'] = result['inertia_tensor'].diagonal()
            estimators_results['LS'] = optimization_results['LS']
        else:
            optimization_results['LS'] = np.array([10.0, 10.0, 10.0])  # Fallback to fixed values
        # Add LS data to estimation_data as final estimate only
        estimation_data['LS'] = optimization_results['LS']
    ########################################################################
    #                          5) EKF Estimation                            #
    ########################################################################
    # get satellite name removing "config" and "yaml"
    sat_name = config_file.split('_')[1].replace("config_", "").replace(".yaml", "")

    if use_ekf:
        # Initialize EKF with the satellite model
        if use_realistic_actuators:
            sigma_omega_meas = 1e-3
            sigma_rw_meas    = 1e-0
            Qc_diag = np.array([1e-6]*3 + [1e-10]*3 + [1e-6]*3, dtype=float)
        else:
            sigma_omega_meas = 1e-6
            sigma_rw_meas    = 1e-6
            Qc_diag = np.array([1e-9]*3 + [1e-5]*3 + [1e-9]*3, dtype=float)

        if sat_name == "sat1":
            I_noise = 0.8
        else:
            I_noise = 0.85

        I0 = I_sat * np.array([1.0, 1.0, 1.0])  * I_noise # Initial guess for inertia
        x0 = np.zeros(9, dtype=float)
        x0[3:6] = I0
        P0 = np.diag([1e-4]*3 + [1.0]*3 + [1e-2]*3)  # fairly uninformative on inertia
        R_diag = np.array([sigma_omega_meas**2]*3 + [sigma_rw_meas**2]*3, dtype=float)

        add_ekf_initial_noise = False  # Whether to add noise to initial state
        if add_ekf_initial_noise:
            rng_ekf = np.random.default_rng(seed + 2)  # Independent RNG for EKF
            x0[3:6] += rng_ekf.normal(0, 0.01, size=3)  # Perturb inertia guess by small amount
            P0 += np.diag(rng_ekf.uniform(0, 0.001, size=9))  # Slight variation in prior belief

        I_rw_diag = np.array([I_rw, I_rw, I_rw], dtype=float)  # or per-wheel values if different
        ekf = EKFInertiaRW(EKFConfig(dt=dt, I_rw_diag=I_rw_diag, Qc_diag=Qc_diag, R_diag=R_diag, x0=x0, P0=P0))
        rw_accels_applied = sat.get_rw_acc_at_times(t_sim)       # (N,3), applied u_k

        N = len(t_sim)
        Xhat = np.zeros((N, 9), dtype=float)  # State = [omega(0:3), inertia(3:6), rw_speeds(6:9)]
        Phat = np.zeros((N, 9, 9), dtype=float) # Covariance matrix of the state
        for k in range(N):
            z_k = states_measured[k]
            u_k = rw_accels_applied[k]
            tau_ext_k = sat.tau_ext[k]              # <-- use the simulator's external torque at this step
            xk, Pk = ekf.step(u_k, z_k, tau_ext=tau_ext_k)

            Xhat[k] = xk
            Phat[k] = Pk
        all_I_ekf = Xhat[:, 3:6]
        I_cov_ekf = Phat[:, 3:6, 3:6]
        final_I_ekf = all_I_ekf[-1]
        
        # Add EKF data to both structures
        estimation_data['EKF'] = all_I_ekf  # Time series of inertia estimates
        estimated_states['EKF'] = {'states': Xhat, 'covariances': Phat}  # Full state estimation
        estimators_results['EKF'] = final_I_ekf

    # Build time-series of the true inertia (vector of principal moments) 
    if use_dynamic_I and (dynamic_I_func is not None):
        true_inertia_series = np.vstack([np.diag(dynamic_I_func(tt)).astype(float) for tt in t_sim])
    else:
        Ivec = np.array(I_sat, dtype=float)
        if Ivec.ndim == 2:  # matrix -> diagonal
            Ivec = np.diag(Ivec)
        true_inertia_series = np.tile(Ivec, (len(t_sim), 1))


    # Configuration info for report
    config_info = {
        'External Torques': use_external_torques,
        'Realistic Actuators': use_realistic_actuators, 
        'Noisy Sensors': use_noisy_sensors,
        'Torque Profile': torque_profile,
        'Simulation Time': t_max,
        'Time Step': dt
    }

    # Generate comprehensive report
    create_comprehensive_report(
        t=t,
        states=states,
        estimators_data=estimation_data if use_ekf or use_ls else None,
        estimated_states=estimated_states if use_ekf else None,
        true_inertia=np.array(I_sat),
        actual_torques=actual_torques,
        control_func=enhanced_control_input,
        I_rw=I_rw,
        rw_axes=rw_axes,
        config_info=config_info,
        ext_torques=None,
        save_plots=False,
        show_plots=True,
        verbose=False,
        torque_profile=torque_profile,
        satellite=sat_name,
        dynamic_I_func=dynamic_I_func,
        dynamic_I_func_name=dynamic_I_func_name
    )
    
    return {
        'estimators': estimators_results,
        'true_inertia': np.array(I_sat),
        'estimation_data': estimation_data,
        'simulation_data': {
            'time': t,
            'states': states_measured,
            'control': enhanced_control_input,
            'true_inertia_series': true_inertia_series, 
        }
    }


DYNAMIC_INERTIA_FUNCTIONS = {
    0: None,
    1: "step_change",  # if t<t_max/2 then I_sat else I_sat * 1.1
    2: "slow_drift",  # I(t)=I_0 + I˙ t or a smooth ramp (e.g., tanh or spline).
    3: "periodic",  # I(t) = I_0 + A * sin(ωt + φ)
}

def main_seeds():
    """Run the simulation with different seeds and configurations
    """
    # Configuration
    save_csv_results = True
    satellite = 3
    num_exp = 1 # if save_csv_results else 1
    DYN_FUNC_IDX = 3  # Select dynamic inertia function index (0-3)
    dyn_func = DYNAMIC_INERTIA_FUNCTIONS[DYN_FUNC_IDX]

    sim_config = {
        'use_external_torques': True, 
        'use_realistic_actuators': True,
        'use_noisy_sensors': True
    }

    test_profiles = [
        # ("one step", "One step torque profile"),
        # ("multi step", "Multi step torque profile"),
        # ("sawtooth", "Sawtooth torque profile"),
        # ("sine", "Sine torque profile"),
        # ("multi sine", "Multi-sine torque profile (Full Enhancement)"),
        # ("prbs", "PRBS torque profile (Full Enhancement)"),
        # ("chirp", "Chirp torque profile (Traditional Methods)"),
        ("sine-3axis", "Sine 3-axis torque profile (Full Enhancement)"),
    ]

    for s in range(1, 4):
        print(f"🚀 Testing satellite # {s}...")
        if s == 1:
            satellite = 1
        elif s == 2:
            satellite = 2
        elif s == 3:
            satellite = 3
        else:
            raise ValueError("Invalid satellite number. Must be 1, 2, or 3.")
        satellite = 3

        for i in range(1, 4):  # Loop over dynamic inertia functions

            dyn_func = DYNAMIC_INERTIA_FUNCTIONS[i]

            config_file = f"config_sat{satellite}.yaml"
            config_name = os.path.splitext(os.path.basename(config_file))[0]
            torque_params = get_profile_params_for_sat(f"sat{satellite}")

            # Setup CSV writer if needed
            writer = None
            csv_file = None
            if save_csv_results:
                csv_filename = (f"results_{config_name}.csv" if DYN_FUNC_IDX == 0
                                else f"results_{config_name}_dyn_{dyn_func}.csv")
                csv_file = open(csv_filename, mode='w', newline='')
                writer = csv.writer(csv_file)
                writer.writerow(["satellite_config", "seed", "profile", "ls_error", "ekf_error"])
                print("\n" + "="*80)
                print(f"RUNNING PROFILE COMPARISON ON {num_exp} SEEDS — {config_name}")
                print(f"Using dynamic inertia function: {dyn_func}")
                print("="*80)

            try:
                # Main simulation loop
                for seed in tqdm(range(num_exp), desc="Random Seeds"):
                    for i, (profile, description) in enumerate(test_profiles, 1):
                        print(f"\n🔧 Seed {seed} - TEST {i}: {description}")

                        result = run_enhanced_simulation(
                            config_file=config_file,
                            torque_profile=profile,
                            seed=seed,
                            use_ekf=True,
                            use_ls=True,
                            torque_params=torque_params[profile],
                            **sim_config,
                            verbose=False,
                            horizon=False,
                            dynamic_I_func_name=dyn_func, 
                        )

                        # Process results
                        true_inertia = result['true_inertia']
                        estimators = result['estimators']
                        print(f"  Profile: {profile}")
                        print("+++"* 20)
                        # --- Common prep ---
                        ls_error = None
                        rel_error_vector = None
                        ekf_error = None
                        ekf_relative_error = None

                        if dyn_func is not None:
                            # We assume EKF ran; reuse the same window + ground-truth series for both methods
                            time_arr   = result['simulation_data']['time']
                            I_true_ts  = result['simulation_data'].get(
                                'true_inertia_series',
                                np.tile(true_inertia, (len(time_arr), 1))
                            )

                            # Window size (last K steps = 20% of total duration)
                            if len(time_arr) > 1:
                                dt = float(np.mean(np.diff(time_arr)))
                                T_total = float(time_arr[-1] - time_arr[0])
                            else:
                                dt = 1.0
                                T_total = len(time_arr) * dt
                            W_fraction = 0.2
                            W_steps = max(1, int((T_total * W_fraction) / dt))

                            I_true_win = I_true_ts[-W_steps:]                          # (K,3)
                            denom_t = np.clip(np.linalg.norm(I_true_win, axis=1), 1e-12, None)  # (K,)

                            # --- EKF metrics over window ---
                            if 'EKF' in estimators and 'EKF' in result['estimation_data']:
                                I_ekf_all = result['estimation_data']['EKF']           # (N,3)
                                I_ekf_win = I_ekf_all[-W_steps:]                       # (K,3)

                                per_step_rel = np.linalg.norm(I_ekf_win - I_true_win, axis=1) / denom_t
                                ekf_error = float(np.mean(per_step_rel))

                                comp_rel = np.abs(I_ekf_win - I_true_win) / np.clip(I_true_win, 1e-12, None)
                                ekf_relative_error = np.mean(comp_rel, axis=0)         # (3,)

                                print(f"EKF (dyn) — window K={W_steps}, err={ekf_error:.6f}, rel={ekf_relative_error}")

                            # --- LS metrics over the same window (constant estimate vs I_true(t)) ---
                            if 'LS' in estimators:
                                ls_estimate = estimators['LS']                         # (3,)
                                I_ls_win = np.tile(ls_estimate, (W_steps, 1))          # (K,3)

                                per_step_rel_ls = np.linalg.norm(I_ls_win - I_true_win, axis=1) / denom_t
                                ls_error = float(np.mean(per_step_rel_ls))

                                comp_rel_ls = np.abs(I_ls_win - I_true_win) / np.clip(I_true_win, 1e-12, None)
                                rel_error_vector = np.mean(comp_rel_ls, axis=0)        # (3,)

                                print(f"LS  (dyn) — window K={W_steps}, err={ls_error:.6f}, rel={rel_error_vector}")

                        else:
                            # --- Static scoring (unchanged semantics) ---
                            if 'EKF' in estimators:
                                ekf_error = np.linalg.norm(estimators['EKF'] - true_inertia) / np.linalg.norm(true_inertia)
                                ekf_relative_error = np.abs(estimators['EKF'] - true_inertia) / np.clip(true_inertia, 1e-12, None)
                                print(f"EKF (static) — err={ekf_error:.6f}, rel={ekf_relative_error}")

                            if 'LS' in estimators:
                                ls_estimate = estimators['LS']
                                ls_error = np.linalg.norm(ls_estimate - true_inertia) / np.linalg.norm(true_inertia)
                                rel_error_vector = np.abs(ls_estimate - true_inertia) / np.clip(true_inertia, 1e-12, None)
                                print(f"LS  (static) — err={ls_error:.6f}, rel={rel_error_vector}")

                        print("+++"* 20)

                        # Save to CSV if enabled
                        if writer:
                            writer.writerow([config_name, seed, profile, f"{ls_error:.6f}", f"{ekf_error:.6f}"])

            finally:
                # Ensure file is properly closed
                if csv_file:
                    csv_file.close()
                    print(f"\n✅ Done. Results saved to: {csv_filename}")
            
        print("Simulation completed.")


def main_times():
    """Run the simulation with different time horizons and configurations
    """
    horizons = range(10, 610, 10)  # From 10 to 600 seconds in steps of 10
    save_csv_results = True
    # Configuration
    satellite = 2
    results = {}
    
    print("🚀 Testing satellite # {}...".format(satellite))

    sim_config = {
        'use_external_torques': True,
        'use_realistic_actuators': True,
        'use_noisy_sensors': True
    }

    test_profiles = [
        ("one step", "One step torque profile"),
        ("multi step", "Multi step torque profile"),
        ("sawtooth", "Sawtooth torque profile"),
        ("sine", "Sine torque profile"),
        ("multi sine", "Multi-sine torque profile (Full Enhancement)"),
        ("prbs", "PRBS torque profile (Full Enhancement)"),
        ("chirp", "Chirp torque profile (Traditional Methods)"),
        ("sine-3axis", "Sine 3-axis torque profile (Full Enhancement)"),
    ]

    config_file = f"config_sat{satellite}.yaml"
    torque_params = get_profile_params_for_sat(f"sat{satellite}")
    config_name = f"results_horizons_sat{satellite}"
    # Setup CSV writer if needed
    writer = None
    csv_file = None
    if save_csv_results:
        csv_filename = f"{config_name}.csv"
        csv_file = open(csv_filename, mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["satellite_config", "duration", "seed", "profile", "ls_error", "ekf_error"])
        print("\n" + "="*80)
    seed = 42  # Fixed seed for reproducibility
    try:
        # Main simulation loop
        for duration in tqdm(horizons):
            for i, (profile, description) in enumerate(test_profiles, 1):
                print(f"\n🔧 Horizon {duration} - TEST {i}: {description}")
                result = run_enhanced_simulation(
                    config_file=config_file,
                    torque_profile=profile,
                    use_ekf=True,
                    use_ls=True,
                    seed=seed,
                    torque_params=torque_params[profile],
                    **sim_config,
                    verbose=False,
                    horizon=duration
                )

                # Process results
                true_inertia = result['true_inertia']
                estimators = result['estimators']
                print(f"  Profile: {profile}")
                print("+++"* 20)
                if 'EKF' in estimators:
                    ekf_error = np.linalg.norm(estimators['EKF'] - true_inertia) / np.linalg.norm(true_inertia)
                    ekf_relative_error = np.abs(estimators['EKF'] - true_inertia) / true_inertia
                    print(f"EKF Error: {ekf_error:.6f}, EKF Relative Error: {ekf_relative_error}")
                if 'LS' in estimators:
                    ls_estimate = estimators['LS']
                    ls_error = np.linalg.norm(ls_estimate - true_inertia)/ np.linalg.norm(true_inertia)
                    rel_error_vector = np.abs(ls_estimate - true_inertia) / true_inertia
                    print(f"  LS Error: {ls_error:.6f}, LS Relative Error: {rel_error_vector}")
                print("+++"* 20)

                # Save to CSV if enabled
                if writer:
                    writer.writerow([config_name, duration, seed, profile, f"{ls_error:.6f}", f"{ekf_error:.6f}"])



    finally:
        # Ensure file is properly closed
        if csv_file:
            csv_file.close()
            print(f"\n✅ Done. Results saved to: {csv_filename}")
    
    print("Simulation completed.")



if __name__ == "__main__":
   main_seeds()
#    main_times()