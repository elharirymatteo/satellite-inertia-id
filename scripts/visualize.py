import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Union
import pandas as pd

def plot_simulation_data(t, states, I_sat, I_rw, rw_axes, control_func, actual_torques, title="Simulation Results", save=True, torque_profile="test", satellite="default"):
    """Plot simulation results including angular velocity, RW speeds, control input, and kinetic energy"""

    plt.style.use('ggplot')

    omega = states[:, :3]       # Satellite angular velocity
    rw_speeds = states[:, 3:6]  # RW speeds

    # Ensure I_sat is a diagonal matrix
    if isinstance(I_sat, (list, tuple)) or (isinstance(I_sat, np.ndarray) and I_sat.ndim == 1):
        I_sat_matrix = np.diag(I_sat)
    else:
        I_sat_matrix = I_sat

    # Ensure rw_axes is numpy array
    rw_axes = np.array(rw_axes)

    # Angular momentum and kinetic energy
    h_sat = I_sat_matrix @ omega.T  # Satellite angular momentum (3 x N)
    
    # RW angular momentum calculation
    h_rw_total = np.zeros((3, len(t)))
    for i in range(len(t)):
        for j in range(3):  # For each RW
            # RW j contributes momentum along its axis direction
            h_rw_axis = rw_axes[j] * (I_rw * rw_speeds[i, j])
            h_rw_total[:, i] += h_rw_axis
    
    h_total = h_sat + h_rw_total  # Total angular momentum
    E_sat = 0.5 * np.sum(omega * (I_sat_matrix @ omega.T).T, axis=1)  # Kinetic energy of satellite
    E_rw = 0.5 * I_rw * np.sum(rw_speeds**2, axis=1)  # Kinetic energy of RWs
    E_total = E_sat + E_rw
    
    # Evaluate control torques over time
    torques = np.array([control_func(ti) for ti in t])
    plt.style.use('seaborn-darkgrid')
    
    # Set font sizes for paper visibility
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold'
    })

    plt.figure(figsize=(14, 10))

    plt.subplot(5, 1, 1)
    plt.plot(t, omega)
    plt.title("Satellite Angular Velocity")
    plt.ylabel("ω [rad/s]")
    plt.legend(["ωx", "ωy", "ωz"])
    plt.grid(color='white', linewidth=0.5)


    plt.subplot(5, 1, 2)
    plt.plot(t, rw_speeds)
    plt.title("Reaction Wheel Speeds")
    plt.ylabel("RW speed [rad/s]")
    plt.legend(["RWx", "RWy", "RWz"])
    plt.grid(color='white', linewidth=0.5)

    plt.subplot(5, 1, 3)
    plt.plot(t, torques)
    plt.title("Commanded Torque Input")
    plt.ylabel("Torque [Nm]")
    plt.legend(["τx", "τy", "τz"])
    plt.grid(color='white', linewidth=0.5)

    plt.subplot(5, 1, 4)
    plt.plot(t, actual_torques)
    plt.title("Actual Torque Input")
    plt.ylabel("Torque [Nm]")
    plt.legend(["τx", "τy", "τz"])
    plt.grid(color='white', linewidth=0.5)

    plt.subplot(5, 1, 5)
    plt.plot(t, E_total, label="Total")
    plt.plot(t, E_sat, '--', label="Satellite")
    plt.plot(t, E_rw, '--', label="RW")
    plt.title("Kinetic Energy Over Time")
    plt.ylabel("Energy [J]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(color='white', linewidth=0.5)

    plt.tight_layout()
    
    if save:
        import os
        save_path = f"logs/{satellite}"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{torque_profile}_states.png', dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}/{torque_profile}_states.png")
        plt.close()  # Close figure to free memory
    else:
        plt.show()

def plot_estimation_results(t, estimators_data: Dict, true_inertia, dynamic_I_func, dynamic_I_func_name, title="Inertia Estimation Results", save=True, torque_profile="default", satellite="default"):
    """
    Plot estimation results from multiple methods
    
    Args:
        t: Time vector
        estimators_data: Dict with keys as method names and values as estimation time series or final estimates
        true_inertia: True inertia values [Ixx, Iyy, Izz]
        dynamic_I_func: Function for dynamic inertia (if applicable)
        title: Plot title
        save: Whether to save the plot
        torque_profile: Torque profile name for filename
        satellite: Satellite name for folder structure
    """
    plt.style.use('seaborn-darkgrid')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    true_inertia = np.array(true_inertia)
    if dynamic_I_func is not None:
        true_inertia = np.array([dynamic_I_func(ti) for ti in t])
        true_inertia = np.array([np.diag(true_inertia[i]) for i in range(true_inertia.shape[0])])

    # Separate time series and final estimates
    time_series_methods = {}
    final_only_methods = {}
    
    for method, estimates in estimators_data.items():
        if isinstance(estimates, np.ndarray):
            if len(estimates.shape) > 1 and estimates.shape[0] == len(t):
                # Time series data
                time_series_methods[method] = estimates
            elif len(estimates.shape) == 1 and len(estimates) == 3:
                # Final estimate only
                final_only_methods[method] = estimates
            else:
                print(f"Warning: Unexpected shape for {method}: {estimates.shape}")
        else:
            print(f"Warning: {method} data is not a numpy array")
    
    if dynamic_I_func is not None and 'LS' in final_only_methods:
        ls_vec = final_only_methods['LS']                  # (3,)
        if isinstance(ls_vec, np.ndarray) and ls_vec.shape == (3,):
            ls_ts = np.tile(ls_vec, (len(t), 1))           # (T,3)
            time_series_methods['LS (const)'] = ls_ts      # will be plotted like EKF
            
    # Plot 1: Inertia estimates over time (only time series data)
    for method, estimates in time_series_methods.items():
        axes[0, 0].plot(t, estimates[:, 0], label=f'{method} Ixx', alpha=0.8)
        axes[0, 1].plot(t, estimates[:, 1], label=f'{method} Iyy', alpha=0.8)
            
    # Add true values depending on the shape of true_inertia
    if true_inertia.ndim == 1:
        # If true_inertia is a single value for each inertia component
        axes[0, 0].axhline(true_inertia[0], color='k', linestyle='--', label='True Ixx', alpha=0.7)
        axes[0, 1].axhline(true_inertia[1], color='k', linestyle='--', label='True Iyy', alpha=0.7)
    else:
        # If true_inertia is a time series
        axes[0, 0].plot(t, true_inertia[:, 0], color='k', linestyle='--', label='True Ixx', alpha=0.7)
        axes[0, 1].plot(t, true_inertia[:, 1], color='k', linestyle='--', label='True Iyy', alpha=0.7)

    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Ixx [kg*m^2]')
    axes[0, 0].set_title('Ixx Estimation')
    axes[0, 0].legend()
    axes[0, 0].grid(color='white', linewidth=0.5)

    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Iyy [kg*m^2]')
    axes[0, 1].set_title('Iyy Estimation')
    axes[0, 1].legend()
    axes[0, 1].grid(color='white', linewidth=0.5)
    
    # Plot 2: Izz estimates over time
    for method, estimates in time_series_methods.items():
        axes[1, 0].plot(t, estimates[:, 2], label=f'{method} Izz', alpha=0.8)

    if true_inertia.ndim == 1:
        axes[1, 0].axhline(true_inertia[2], color='k', linestyle='--', label='True Izz', alpha=0.7)
    else:
        axes[1, 0].plot(t, true_inertia[:, 2], color='k', linestyle='--', label='True Izz', alpha=0.7)

    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Izz [kg*m^2]')
    axes[1, 0].set_title('Izz Estimation')
    axes[1, 0].legend()
    axes[1, 0].grid(color='white', linewidth=0.5)

    # Plot 3: Final estimation errors (both time series final values and final-only estimates)
    methods = []
    final_estimates = []
    
    # Get final estimates from time series
    for method, estimates in time_series_methods.items():
        final_estimates.append(estimates[-1])  # Final estimate
        methods.append(method)
        
    # Add final-only estimates
    for method, estimates in final_only_methods.items():
        if method == 'LS' and dynamic_I_func is not None:
        # Skip LS here if we've already plotted it as LS (const)
            continue
        final_estimates.append(estimates)  # Single estimate
        methods.append(method)
    
    if final_estimates:
        final_estimates = np.array(final_estimates)
        
        # For error calculation, use the final true inertia value if it's a time series
        if true_inertia.ndim > 1:
            # Use the final value from the time series
            true_inertia_final = true_inertia[-1]
        else:
            # Use the constant value
            true_inertia_final = true_inertia
            
        errors = np.abs(final_estimates - true_inertia_final)
        
        x_pos = np.arange(len(methods))
        width = 0.25
        
        axes[1, 1].bar(x_pos - width, errors[:, 0], width, label='Ixx Error', alpha=0.8)
        axes[1, 1].bar(x_pos, errors[:, 1], width, label='Iyy Error', alpha=0.8)
        axes[1, 1].bar(x_pos + width, errors[:, 2], width, label='Izz Error', alpha=0.8)
        
        axes[1, 1].set_xlabel('Estimation Method')
        axes[1, 1].set_ylabel('Absolute Error [kg*m^2]')
        axes[1, 1].set_title('Final Estimation Errors')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(color='white', linewidth=0.5)

    plt.tight_layout()
    
    if save:
        import os
        save_path = f"logs/{satellite}" if dynamic_I_func_name is None else f"logs/dynamic_{dynamic_I_func_name}/{satellite}"
        os.makedirs(save_path, exist_ok=True)
        method_name = "_".join(methods) if methods else "estimation"
        plt.savefig(f'{save_path}/{torque_profile}_{method_name}_estimation_results.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}/{torque_profile}_{method_name}_estimation_results.png")
        plt.close()  # Close figure to free memory
    else:
        plt.show()

def plot_estimated_filter_states(t, states: Dict, true_states: Optional[np.ndarray] = None, dynamic_I_func=None, dynamic_I_func_name=None,
                                 title="Estimated Filter States", save=True, torque_profile="default", satellite="default"):
    """
    Plot estimated filter states (omega, inertia, rw_speed) over time with 3σ uncertainty bands.

    Args:
        t: Time vector
        states: Dict of estimators with 'states' and optionally 'covariances'
        true_states: Ground truth states (optional, shape: [T, 6] or [T, 9])
        title: Overall plot title
        save: Whether to save the figure to file
    """
    plt.style.use('seaborn-darkgrid')

    num_estimators = len(states)
    fig, axes = plt.subplots(num_estimators, 3, figsize=(18, 6 * num_estimators))
    fig.suptitle(title, fontsize=16, y=0.95)
    plt.subplots_adjust(top=0.93)

    if num_estimators == 1:
        axes = np.expand_dims(axes, axis=0)

    state_labels = ['Angular Velocity [rad/s]', 'Inertia [kg·m²]', 'RW Speeds [rad/s]']
    state_indices = [(0, 3), (3, 6), (6, 9)]
    axis_labels = [['ωx', 'ωy', 'ωz'], ['Ixx', 'Iyy', 'Izz'], ['RWx', 'RWy', 'RWz']]

    for i, (estimator, data) in enumerate(states.items()):
        Xhat = data['states']
        Phat = data.get('covariances', None)
        uncertainty = np.zeros_like(Xhat) if Phat is not None else None

        if Phat is not None:
            for k in range(len(t)):
                uncertainty[k] = np.sqrt(np.diag(Phat[k]))

        for j, (start_idx, end_idx) in enumerate(state_indices):
            ax = axes[i, j]
            state_data = Xhat[:, start_idx:end_idx]

            for axis_idx in range(3):
                label = f'{estimator} {axis_labels[j][axis_idx]}'
                ax.plot(t, state_data[:, axis_idx], label=label, alpha=0.8)

                if uncertainty is not None:
                    sigma = 3 * uncertainty[:, start_idx + axis_idx]
                    ax.fill_between(t,
                                    state_data[:, axis_idx] - sigma,
                                    state_data[:, axis_idx] + sigma,
                                    alpha=0.2)

            if true_states is not None and true_states.shape[1] >= end_idx:
                if j == 0 or j == 1:
                    pass  # Inertia is constant, skip plotting
                else:
                    if true_states.shape[1] == 6:
                        true_data = true_states[:, 0:3] if j == 0 else true_states[:, 3:6]
                    else:
                        true_data = true_states[:, start_idx:end_idx]

                    for axis_idx in range(3):
                        ax.plot(t, true_data[:, axis_idx], color='k', linestyle='--', alpha=0.7)

            ax.set_title(f"{estimator} - {state_labels[j]}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(state_labels[j])
            ax.grid(True)

            # Combine and deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        import os
        save_path = f"logs/{satellite}" if dynamic_I_func_name is None else f"logs/dynamic_{dynamic_I_func_name}/{satellite}"
        os.makedirs(save_path, exist_ok=True)
        fname = f'{save_path}/{torque_profile}_{title.replace(" ", "_").lower()}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    else:
        plt.show()


def plot_estimation_convergence(t, estimators_data: Dict, true_inertia, dynamic_I_func, dynamic_I_func_name, title="Estimation Convergence", save=True, torque_profile="default", satellite="default"):
    """
    Plot convergence of estimation methods
    
    Args:
        t: Time vector
        estimators_data: Dict with estimation time series
        true_inertia: True inertia values
        title: Plot title
        save: Whether to save the plot
    """
    plt.style.use('seaborn-darkgrid')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    
    true_inertia = np.array(true_inertia)
    if dynamic_I_func is not None:
        true_inertia = np.array([dynamic_I_func(ti) for ti in t])
        true_inertia = np.array([np.diag(true_inertia[i]) for i in range(true_inertia.shape[0])])

    # Plot 1: Relative error over time (only for time series data)
    time_series_found = False
    for method, estimates in estimators_data.items():
        if isinstance(estimates, np.ndarray) and len(estimates.shape) > 1 and estimates.shape[0] == len(t):
            rel_errors = np.abs(estimates - true_inertia) / true_inertia
            mean_rel_error = np.mean(rel_errors, axis=1)
            axes[0].semilogy(t, mean_rel_error, label=f'{method}', alpha=0.8)
            time_series_found = True
    
    if time_series_found:
        axes[0].set_xlabel('Time [s]')
        axes[0].set_ylabel('Mean Relative Error')
        axes[0].set_title('Estimation Error Convergence')
        axes[0].legend()
        axes[0].grid(color='white', linewidth=0.5)
    else:
        axes[0].text(0.5, 0.5, 'No time series data available\nfor convergence analysis', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Estimation Error Convergence')

    # Plot 2: Parameter uncertainty (if available)
    found_uncertainty = False
    for method, covariances in estimators_data.items():
        if "covariances" in method.lower() and isinstance(covariances, np.ndarray) and covariances.shape[1:] == (3, 3):
            # Extract standard deviation (sqrt of diagonal) or trace as uncertainty
            stds = np.sqrt([np.diag(cov) for cov in covariances])  # shape: [T, 3]
            mean_uncertainty = np.mean(stds, axis=1)
            axes[1].plot(t, mean_uncertainty, label=method.replace('_cov', ''), alpha=0.8)
            found_uncertainty = True

    if found_uncertainty:
        axes[1].set_title('Inertia Estimation Uncertainty (σ)')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No covariance data found',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Inertia Estimation Uncertainty')

    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Mean σ of Inertia')
    axes[1].grid(color='white', linewidth=0.5)


    plt.tight_layout()
    
    if save:
        import os
        save_path = f"logs/{satellite}" if dynamic_I_func_name is None else f"logs/dynamic_{dynamic_I_func_name}/{satellite}"
        os.makedirs(save_path, exist_ok=True)
        method_names = [k for k, v in estimators_data.items() 
                       if isinstance(v, np.ndarray) and len(v.shape) > 1]
        method_name = "_".join(method_names) if method_names else "estimation"
        plt.savefig(f'{save_path}/{torque_profile}_{method_name}_estimation_convergence.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}/{torque_profile}_{method_name}_estimation_convergence.png")
    else:
        plt.show()

def plot_comparison_study(results: Dict, title="Method Comparison Study", save=True, torque_profile="default", satellite="default"):
    """
    Plot comparison of different simulation configurations
    
    Args:
        results: Dict with configuration names as keys and result dicts as values
        title: Plot title
    """
    if not results:
        print("No results to plot")
        return
    plt.style.use('seaborn-darkgrid')
       
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    configs = list(results.keys())
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Extract data for plotting
    methods = []
    config_errors = {config: [] for config in valid_results.keys()}
    
    # Get all unique estimation methods
    for config, result in valid_results.items():
        if 'estimators' in result:
            methods.extend(result['estimators'].keys())
    methods = list(set(methods))
    
    # Calculate errors for each configuration and method
    for config, result in valid_results.items():
        true_inertia = result['true_inertia']
        estimators = result.get('estimators', {})
        
        for method in methods:
            if method in estimators:
                estimate = estimators[method]
                error = np.linalg.norm(estimate - true_inertia)
                config_errors[config].append(error)
            else:
                config_errors[config].append(np.nan)
    
    # Plot 1: Error comparison across configurations
    x_pos = np.arange(len(methods))
    width = 0.8 / len(valid_results)
    plt.style.use('seaborn-darkgrid')

    for i, (config, errors) in enumerate(config_errors.items()):
        axes[0, 0].bar(x_pos + i*width, errors, width, label=config, alpha=0.8)
    
    axes[0, 0].set_xlabel('Estimation Method')
    axes[0, 0].set_ylabel('L2 Norm Error [kg*m^2]')
    axes[0, 0].set_title('Error Comparison Across Configurations')
    axes[0, 0].set_xticks(x_pos + width * (len(valid_results) - 1) / 2)
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(color='white', linewidth=0.5)

    # Additional plots can be added here for detailed analysis
    
    plt.tight_layout()
    
    if save:
        import os
        save_path = f"logs/{satellite}"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{torque_profile}_method_comparison_study.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}/{torque_profile}_method_comparison_study.png")
    else:
        plt.show()

def plot_external_torques(t, ext_torques, control_func, title="External Torques Over Time", save=True, torque_profile="default", satellite="default"):
    """
    Plot external torques over time
    
    Args:
        t: Time vector
        ext_torques: External torques (shape: [T, 3])
        control_func: Control function to compare against
        title: Plot title
        save: Whether to save the plot
    """

    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    # save same colors for each axis
    colors = ['r', 'g', 'b']

    for i in range(3):
        ax.plot(t, ext_torques[:, i], label=f'τ{i}', color=colors[i], alpha=0.8)
    
    # Evaluate control torques over time
    control_torques = np.array([control_func(ti) for ti in t])
    for i in range(3):
        ax.plot(t, control_torques[:, i], label=f'Control τ{i}', linestyle='--', color=colors[i], alpha=0.5)    
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Torque [Nm]')
    ax.legend()
    ax.grid(color='white', linewidth=0.5)

    plt.tight_layout()
    
    if save:
        import os
        save_path = f"logs/{satellite}"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{torque_profile}_external_torques.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}/{torque_profile}_external_torques.png")
    
    plt.show()

def create_comprehensive_report(t, states, estimators_data, estimated_states, true_inertia, actual_torques,
                              control_func, I_rw, rw_axes, ext_torques=None, config_info=None, save_plots=True, 
                              show_plots=True, verbose=True, torque_profile=None, satellite='default', dynamic_I_func=None,
                              dynamic_I_func_name=None):
    """
    Create a comprehensive visualization report
    
    Args:
        t: Time vector
        states: Simulation states
        estimators_data: Dict of estimation results
        true_inertia: True inertia values
        control_func: Control function
        I_rw: RW inertia
        rw_axes: RW axes
        ext_torques: External torques (if disturbances are used)
        config_info: Additional configuration information
        save_plots: Whether to save plots to logs/ folder
    """    
    save_trajectory = False
    # 1. Simulation data plots
    if save_plots or show_plots:
        plot_simulation_data(t, states, true_inertia, I_rw, rw_axes, control_func, actual_torques,
                        title="Simulation Data", save=save_plots, torque_profile=torque_profile, satellite=satellite)
    if save_trajectory:
        # save in a csv the states, the torqies, the rw speeds, actual torques
        energy = 0.5 * np.sum(states[:, :3] * (I_rw @ states[:, :3].T).T, axis=1) + \
                 0.5 * I_rw * np.sum(states[:, 3:6]**2, axis=1)
        trajectory_data = {
            'time': t,
            'omega': states[:, :3].tolist(),  # Satellite angular velocity
            'rw_speeds': states[:, 3:6].tolist(),  # RW speeds
            'torques': np.array([control_func(ti) for ti in t]).tolist(),  # Control torques
            'actual_torques': actual_torques.tolist(),  # Actual torques
            'energy': energy.tolist()  # Energy of the system
        }
        if pd is not None:
            import os
            save_path = f"logs/{satellite}" if dynamic_I_func_name is None else f"logs/dynamic_{dynamic_I_func_name}/{satellite}"
            os.makedirs(save_path, exist_ok=True)
            df_trajectory = pd.DataFrame(trajectory_data)
            df_trajectory.to_csv(f'{save_path}/{torque_profile}_simulation_trajectory.csv', index=False)
            print(f"Saved simulation trajectory to {save_path}/{torque_profile}_simulation_trajectory.csv")
        else:
            print("Pandas not available, skipping trajectory CSV export")

    # 2. Estimation results
    if estimators_data and (show_plots or save_plots):
        plot_estimation_results(t, estimators_data, true_inertia, dynamic_I_func, dynamic_I_func_name,
                              title="Inertia Estimation Results", save=save_plots, 
                              torque_profile=torque_profile, satellite=satellite)

        plot_estimation_convergence(t, estimators_data, true_inertia, dynamic_I_func,dynamic_I_func_name,
                                   title="Estimation Convergence Analysis", save=save_plots,
                                   torque_profile=torque_profile, satellite=satellite)
    if estimated_states is not None and show_plots:
        # Extract EKF data properly for filter state plotting
        if 'EKF' in estimated_states:
            ekf_data = estimated_states['EKF']
            # Create 6D true states [omega, rw_speeds] for comparison with EKF
            true_states_6d = np.hstack([states[:, :3], states[:, 6:9]])
            plot_estimated_filter_states(t, {'EKF': ekf_data}, true_states_6d, dynamic_I_func=dynamic_I_func,dynamic_I_func_name=dynamic_I_func_name,
                                        title="Estimated Filter States", save=save_plots,
                                        torque_profile=torque_profile, satellite=satellite)
    if ext_torques is not None and show_plots:
        plot_external_torques(t, ext_torques, control_func,
                              title="External Torques Over Time", save=save_plots,
                              torque_profile=torque_profile, satellite=satellite)  
    
    if config_info and verbose:
        print(f"\nSimulation Configuration:")
        for key, value in config_info.items():
            print(f"  {key}: {value}")

# Backward compatibility
def plot_results(t, states, I_sat, I_rw, rw_axes, control_func):
    """Backward compatibility wrapper"""
    plot_simulation_data(t, states, I_sat, I_rw, rw_axes, control_func)
