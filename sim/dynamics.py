import numpy as np
from scipy.integrate import solve_ivp
import warnings
from utils.math_utils import skew

class Satellite:
    def __init__(self, I_sat, I_rw, rw_axes, rw_speed_max=None, rw_torque_max=None, 
                 torque_smoothing=False, torque_rate_limit=None, verbose=False,
                  use_dynamic_inertia=False, dynamic_inertia_func=None):
        self.I_sat = np.array(I_sat)
        if self.I_sat.ndim == 1:
            self.I_sat = np.diag(self.I_sat)
        elif self.I_sat.shape != (3, 3):
            raise ValueError("Inertia tensor must be either a 3-element vector or a 3x3 matrix.")
        self.I_rw = I_rw                      # Scalar RW inertia
        self.rw_axes = np.array(rw_axes)      # Each row is an RW axis vector (3x3)
        # RW limits
        self.rw_speed_max = rw_speed_max if rw_speed_max is not None else np.inf
        self.rw_torque_max = rw_torque_max if rw_torque_max is not None else np.inf
        # Torque smoothing options
        self.torque_smoothing = torque_smoothing
        self.torque_rate_limit = torque_rate_limit
        self.prev_tau = np.zeros(3)
        self.prev_time = 0.0
        # Tracking variables
        self.saturation_events = []
        self.clamping_events = []
        self.angular_accelerations = []
        # NEW: Store actual torques during simulation
        self.tau_actual_history = []
        self.tau_commanded_history = []
        self.time_history = []
        self.rw_acc_history = []           # NEW: applied RW accelerations (after limits)
        # New dynamic inertia settings
        self.use_dynamic_inertia = use_dynamic_inertia
        self.dynamic_inertia_func = dynamic_inertia_func
        # Keep a copy of current inertia (updated per step if dynamic)
        self.verbose = verbose

    def _get_current_inertia(self, t):
        """Return current inertia diagonal vector (3,)."""
        if self.use_dynamic_inertia and self.dynamic_inertia_func is not None:
            return np.array(self.dynamic_inertia_func(t), dtype=float)
        else:
            return self.I_sat

    def _apply_rw_limits(self, rw_speeds, rw_acc, tau_commanded):
        """
        Apply reaction wheel speed and torque limits
        
        Returns:
            rw_acc_limited: Limited angular acceleration
            tau_actual: Actual torque applied (may differ from commanded)
            saturation_flags: Boolean array indicating which RWs are saturated
        """
        tau_actual = tau_commanded.copy()
        rw_acc_limited = rw_acc.copy()
        saturation_flags = np.zeros(3, dtype=bool)
        
        for i in range(3):
            # Check torque saturation first
            if abs(tau_commanded[i]) > self.rw_torque_max:
                tau_actual[i] = np.sign(tau_commanded[i]) * self.rw_torque_max
                rw_acc_limited[i] = tau_actual[i] / self.I_rw
                saturation_flags[i] = True
            
            # Check speed saturation
            if abs(rw_speeds[i]) >= self.rw_speed_max:
                # If at speed limit, only allow deceleration
                if np.sign(rw_speeds[i]) == np.sign(rw_acc_limited[i]):
                    # Trying to accelerate further - saturate
                    rw_acc_limited[i] = 0.0
                    tau_actual[i] = 0.0
                    saturation_flags[i] = True
        
        return rw_acc_limited, tau_actual, saturation_flags

    def _smooth_torque_command(self, tau_commanded, t):
        """Apply torque rate limiting for smoother control"""
        if not self.torque_smoothing or self.torque_rate_limit is None:
            return tau_commanded
        
        dt = t - self.prev_time
        if dt <= 0:
            return tau_commanded
        
        # Apply rate limiting
        max_change = self.torque_rate_limit * dt
        tau_diff = tau_commanded - self.prev_tau
        tau_diff_limited = np.clip(tau_diff, -max_change, max_change)
        tau_smoothed = self.prev_tau + tau_diff_limited
        
        # Update history
        self.prev_tau = tau_smoothed.copy()
        self.prev_time = t
        
        return tau_smoothed

    def dynamics(self, t, state, control_func):
        omega = state[0:3]                    # Angular velocity of satellite
        rw_speeds = state[3:6]                # Speeds of reaction wheels

        # NEW: get inertia (possibly time-varying)
        I_diag = self._get_current_inertia(t)
        self.I_sat = I_diag  # store for logging if needed

        # Compute torque command from controller
        tau_commanded = control_func(t)       # 3x1 vector (commanded torques)
        
        # Apply torque smoothing if enabled
        tau_commanded = self._smooth_torque_command(tau_commanded, t)

        # Angular acceleration of reaction wheels (before limits)
        rw_acc_unlimited = tau_commanded / self.I_rw
        
        # Apply RW limits
        rw_acc, tau_actual, sat_flags = self._apply_rw_limits(
            rw_speeds, rw_acc_unlimited, tau_commanded
        )
        
        # NEW: Store actual torques and time
        self.tau_actual_history.append(tau_actual.copy())
        self.tau_commanded_history.append(tau_commanded.copy())
        self.time_history.append(t)
        self.rw_acc_history.append(rw_acc.copy())

        
        if self.verbose:
            # Store saturation events for analysis
            if np.any(sat_flags):
                self.saturation_events.append({
                    'time': t,
                    'rw_speeds': rw_speeds.copy(),
                    'tau_commanded': tau_commanded.copy(),
                    'tau_actual': tau_actual.copy(),
                    'saturated_wheels': sat_flags.copy()
                })

        # Total angular momentum of RWs projected onto satellite body
        h_rw_total = np.zeros(3)
        for i in range(3):
            h_rw_total += self.rw_axes[i] * (self.I_rw * rw_speeds[i])

        # Rate of change of RW angular momentum (using actual torques)
        h_rw_dot = np.zeros(3)
        for i in range(3):
            h_rw_dot += self.rw_axes[i] * (self.I_rw * rw_acc[i])

        # Total angular momentum = satellite + RW
        h_total = self.I_sat @ omega + h_rw_total

        # Euler's rigid body equation with RW coupling
        omega_cross = skew(omega)
        domega = np.linalg.inv(self.I_sat) @ (-omega_cross @ h_total - h_rw_dot)

        return np.concatenate([domega, rw_acc])

    def _clamp_rw_speeds(self, states):
        """
        Post-integration clamping of RW speeds to ensure physical bounds
        
        Args:
            states: Array of states (N, 6) where states[:, 3:6] are RW speeds
            
        Returns:
            states_clamped: States with RW speeds clamped
            clamping_applied: Boolean indicating if any clamping was needed
        """
        states_clamped = states.copy()
        original_rw_speeds = states[:, 3:6]
        
        # Apply clamping
        clamped_rw_speeds = np.clip(original_rw_speeds, -self.rw_speed_max, self.rw_speed_max)
        states_clamped[:, 3:6] = clamped_rw_speeds
        
        # Check if clamping was applied
        clamping_mask = np.abs(original_rw_speeds) > self.rw_speed_max
        clamping_applied = np.any(clamping_mask)
        
        if clamping_applied:
            # Log clamping events
            clamp_indices = np.where(clamping_mask)
            if self.verbose:
                for i, j in zip(clamp_indices[0], clamp_indices[1]):
                    self.clamping_events.append({
                        'time_index': i,
                        'wheel': j,
                        'original_speed': original_rw_speeds[i, j],
                        'clamped_speed': clamped_rw_speeds[i, j]
                    })
            
            # Issue warning
            max_violation = np.max(np.abs(original_rw_speeds) - self.rw_speed_max)
            if self.verbose:
                warnings.warn(
                    f"RW speed limits exceeded during integration. Maximum violation: {max_violation:.6f} rad/s. "
                    f"Applied post-integration clamping. Consider reducing time step or improving controller.",
                    UserWarning
                )
        
        return states_clamped, clamping_applied

    def compute_angular_accelerations(self, t_eval, states, control_func):
        """Compute angular accelerations at specific time points after simulation"""
        self.angular_accelerations = []
        
        for i, t_val in enumerate(t_eval):
            state = states[i]
            omega = state[:3]
            rw_speeds = state[3:6]
            
            # Compute commanded torque
            tau_commanded = control_func(t_val)
            tau_commanded = self._smooth_torque_command(tau_commanded, t_val)
            rw_acc_unlimited = tau_commanded / self.I_rw
            
            # Apply limits
            rw_acc, tau_actual, _ = self._apply_rw_limits(
                rw_speeds, rw_acc_unlimited, tau_commanded
            )
            
            # RW momentum calculations
            h_rw_total = np.zeros(3)
            for j in range(3):
                h_rw_total += self.rw_axes[j] * (self.I_rw * rw_speeds[j])
            
            h_rw_dot = np.zeros(3)
            for j in range(3):
                h_rw_dot += self.rw_axes[j] * (self.I_rw * rw_acc[j])
            
            # Total angular momentum
            h_total = self.I_sat @ omega + h_rw_total
            
            # Compute angular acceleration
            omega_cross = skew(omega)
            domega = np.linalg.inv(self.I_sat) @ (-omega_cross @ h_total - h_rw_dot)
            
            self.angular_accelerations.append(domega)
        
        self.angular_accelerations = np.array(self.angular_accelerations)

    def simulate(self, omega0, rw_speed0, control_func, t_span, dt):
        # Clear previous events and histories
        self.saturation_events = []
        self.clamping_events = []
        self.tau_actual_history = []
        self.tau_commanded_history = []
        self.time_history = []
        
        # Reset torque smoothing state
        self.prev_tau = np.zeros(3)
        self.prev_time = t_span[0]
        
        state0 = np.concatenate([omega0, rw_speed0])
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        sol = solve_ivp(
            self.dynamics, 
            t_span, 
            state0, 
            args=(control_func,), 
            t_eval=t_eval,
            method='RK45',
            max_step=dt,
            first_step=dt,
            rtol=1e-8,
            atol=1e-10
        )
        
        # Apply post-integration clamping
        states_clamped, clamping_applied = self._clamp_rw_speeds(sol.y.T)
        
        # Convert histories to numpy arrays
        self.tau_actual_history = np.array(self.tau_actual_history)
        self.tau_commanded_history = np.array(self.tau_commanded_history)
        self.time_history = np.array(self.time_history)
        self.rw_acc_history = np.array(self.rw_acc_history)

        # Zero external torques — subclass SatelliteWithExternalTorques overrides this
        self.tau_ext = np.zeros((len(sol.t), 3))

        # Compute angular accelerations at evaluation points
        self.compute_angular_accelerations(sol.t, states_clamped, control_func)

        return sol.t, states_clamped  # shape (T, 6)

    def get_rw_acc_at_times(self, t_eval):
        """
        Get applied RW accelerations interpolated at specific time points.
        Returns (len(t_eval), 3) array.
        """
        if len(self.rw_acc_history) == 0:
            return np.zeros((len(t_eval), 3))

        # Robust conversion in case simulate() hasn't cast to np.array yet
        time_hist = np.asarray(self.time_history, dtype=float)
        acc_hist  = np.asarray(self.rw_acc_history, dtype=float)
        if acc_hist.ndim == 1:  # list of length-N 1D vectors -> stack
            acc_hist = np.stack(self.rw_acc_history, axis=0).astype(float)

        # Fast path: same grid -> no interpolation
        t_eval = np.asarray(t_eval, dtype=float)
        if len(t_eval) == len(time_hist) and np.allclose(t_eval, time_hist):
            return acc_hist

        # Otherwise, interpolate per axis
        rw_acc_interp = np.zeros((len(t_eval), 3), dtype=float)
        for i in range(3):
            rw_acc_interp[:, i] = np.interp(t_eval, time_hist, acc_hist[:, i])
        return rw_acc_interp


    def get_tau_actual_at_times(self, t_eval):
        """
        Get actual torques interpolated at specific time points
        
        Args:
            t_eval: Array of time points where torques are needed
            
        Returns:
            tau_actual_interp: Interpolated actual torques at t_eval
        """
        if len(self.tau_actual_history) == 0:
            return np.zeros((len(t_eval), 3))
        
        # Interpolate actual torques to match t_eval
        tau_actual_interp = np.zeros((len(t_eval), 3))
        for i in range(3):
            tau_actual_interp[:, i] = np.interp(
                t_eval, 
                self.time_history, 
                self.tau_actual_history[:, i]
            )
        
        return tau_actual_interp

class SatelliteWithExternalTorques(Satellite):
    """Extended satellite class that includes external torques"""
    def __init__(self, I_sat, I_rw, rw_axes, rw_speed_max, rw_torque_max, external_torque_func=None, torque_smoothing=False,
        torque_rate_limit=None, orbital_params=None, use_dynamic_inertia=False, dynamic_inertia_func=None, verbose=False):
        
        self.external_torque_func = external_torque_func
        super().__init__(I_sat=I_sat, I_rw=I_rw, rw_axes=rw_axes, rw_speed_max=rw_speed_max, rw_torque_max=rw_torque_max,
                                    torque_smoothing=torque_smoothing,torque_rate_limit=torque_rate_limit, verbose=verbose,
                                    use_dynamic_inertia=use_dynamic_inertia, dynamic_inertia_func=dynamic_inertia_func)
        
        self.orbital_params = orbital_params or {}
        self.tau_ext = []  # Store external torques for visualization/debugging
        # Set default orbital parameters if not provided
        self.orbital_rate = self.orbital_params.get('orbital_rate', 0.001)  # rad/s
        self.srp_coefficient = self.orbital_params.get('srp_coefficient', 1e-6)  # N
        self.cp_offset = np.array(self.orbital_params.get('cp_offset', [0.01, 0.01, 0.01]))  # m

    def gravity_gradient_torque(self, attitude_quat, orbital_position):
        """Compute gravity gradient torque based on attitude and orbital position"""
        n = self.orbital_params.get('orbital_rate', 0.001)  # rad/s
        r_hat = np.array([0, 0, 1])  # Simplified nadir direction
        I_r = self.I_sat @ r_hat
        tau_gg = 3 * n**2 * np.cross(r_hat, I_r)
        return tau_gg
    
    def solar_radiation_pressure_torque(self, sun_vector, cp_offset):
        """Compute SRP torque based on sun direction and center of pressure offset"""
        if sun_vector is None:
            sun_vector = np.array([1, 0, 0])  # Simplified sun direction

        srp_coeff = self.orbital_params.get('srp_coefficient', 1e-6)  # N
        cp_offset = np.array(cp_offset) if cp_offset is not None else np.zeros(3)
        
        F_srp = srp_coeff * sun_vector
        tau_srp = np.cross(cp_offset, F_srp)
        return tau_srp
        
    def dynamics_with_external_torques(self, t, state, control_func, attitude_quat=None, 
                                 orbital_position=None, sun_vector=None, cp_offset=None):
        """Extended dynamics including external torques"""
        omega = state[0:3]
        rw_speeds = state[3:6]

        tau_commanded = control_func(t)
        tau_commanded = self._smooth_torque_command(tau_commanded, t)
        
        # Apply RW limits
        rw_acc_unlimited = tau_commanded / self.I_rw
        rw_acc, tau_actual, sat_flags = self._apply_rw_limits(
            rw_speeds, rw_acc_unlimited, tau_commanded
        )
        
        # NEW: Store actual torques and time (same as base class)
        self.tau_actual_history.append(tau_actual.copy())
        self.tau_commanded_history.append(tau_commanded.copy())
        self.time_history.append(t)
        self.rw_acc_history.append(rw_acc.copy())

        if self.verbose:
            # Store saturation events
            if np.any(sat_flags):
                self.saturation_events.append({
                    'time': t,
                    'rw_speeds': rw_speeds.copy(),
                    'tau_commanded': tau_commanded.copy(),
                    'tau_actual': tau_actual.copy(),
                    'saturated_wheels': sat_flags.copy()
                })

        if sun_vector is None:
            sun_vector = np.array([1, 0, 0])

        # RW momentum calculations
        h_rw_total = np.zeros(3)
        for i in range(3):
            h_rw_total += self.rw_axes[i] * (self.I_rw * rw_speeds[i])

        h_rw_dot = np.zeros(3)
        for i in range(3):
            h_rw_dot += self.rw_axes[i] * (self.I_rw * rw_acc[i])

        h_total = self.I_sat @ omega + h_rw_total

        # External torques
        tau_ext = np.zeros(3)
        
        if attitude_quat is not None and orbital_position is not None:
            tau_ext += self.gravity_gradient_torque(attitude_quat, orbital_position)
            
        if sun_vector is not None and cp_offset is not None:
            tau_ext += self.solar_radiation_pressure_torque(sun_vector, cp_offset)

        # Modified Euler equation with external torques
        omega_cross = skew(omega)
        domega = np.linalg.inv(self.I_sat) @ (tau_ext - omega_cross @ h_total - h_rw_dot)

        return np.concatenate([domega, rw_acc])

    def _default_attitude_func(self, t):
        """Default attitude evolution (identity for simplicity)"""
        return np.eye(3)  # No rotation from reference frame
    
    def _default_orbital_func(self, t):
        """Default orbital position (circular orbit)"""
        R_orbit = 6378e3 + 400e3  # LEO altitude (m)
        return R_orbit * np.array([
            np.cos(self.orbital_rate * t), 
            np.sin(self.orbital_rate * t), 
            0
        ])
    
    def _default_sun_func(self, t):
        """Default sun direction (simplified - fixed direction)"""
        return np.array([1, 0, 0])  # Sun direction in inertial frame
        
    def simulate_with_external_torques(self, omega0, rw_speed0, control_func, t_span, dt,
                                  attitude_func=None, orbital_func=None, sun_func=None, cp_offset=None):
        """Simulate with external torques using default functions if not provided"""
        # Clear previous events and histories
        self.saturation_events = []
        self.clamping_events = []
        self.tau_actual_history = []
        self.tau_commanded_history = []
        self.time_history = []
        self.prev_tau = np.zeros(3)
        self.prev_time = t_span[0]
        
        state0 = np.concatenate([omega0, rw_speed0])
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

        # Use provided functions or defaults
        attitude_func = attitude_func or self._default_attitude_func
        orbital_func = orbital_func or self._default_orbital_func
        sun_func = sun_func or self._default_sun_func
        cp_offset = cp_offset or self.cp_offset

        # Clear previous external torques
        self.tau_ext = []

        # Create wrapper for dynamics with external torques
        def dynamics_wrapper(t, state):
            attitude = attitude_func(t)
            orbital_pos = orbital_func(t)
            sun_vec = sun_func(t)
            
            return self.dynamics_with_external_torques(
                t, state, control_func, attitude, orbital_pos, sun_vec, cp_offset
            )
        
        sol = solve_ivp(
            dynamics_wrapper,
            t_span,
            state0,
            t_eval=t_eval,
            method='RK45',
            max_step=dt,
            first_step=dt,
            rtol=1e-8,
            atol=1e-10
        )

        # Apply post-integration clamping
        states_clamped, clamping_applied = self._clamp_rw_speeds(sol.y.T)

        # Convert histories to numpy arrays
        self.tau_actual_history = np.array(self.tau_actual_history)
        self.tau_commanded_history = np.array(self.tau_commanded_history)
        self.time_history = np.array(self.time_history)

        # Compute external torques at output points after integration
        self.tau_ext = []
        for i, t_val in enumerate(t_eval):
            attitude = attitude_func(t_val)
            orbital_pos = orbital_func(t_val)
            sun_vec = sun_func(t_val)
            
            # Compute external torques at this time point
            tau_ext = np.zeros(3)
            if attitude is not None and orbital_pos is not None:
                tau_ext += self.gravity_gradient_torque(attitude, orbital_pos)
            if sun_vec is not None and cp_offset is not None:
                tau_ext += self.solar_radiation_pressure_torque(sun_vec, cp_offset)
            
            self.tau_ext.append(tau_ext)
        
        self.tau_ext = np.array(self.tau_ext)  # Now has same length as sol.t
        
        self.compute_angular_accelerations(sol.t, states_clamped, control_func)
        return sol.t, states_clamped
    
    def compute_sample_external_torques(self, t=0.0):
        """Compute sample external torques for debugging"""
        attitude = self._default_attitude_func(t)
        orbital_pos = self._default_orbital_func(t)
        sun_vec = self._default_sun_func(t)
        
        tau_gg = self.gravity_gradient_torque(attitude, orbital_pos)
        tau_srp = self.solar_radiation_pressure_torque(sun_vec, self.cp_offset)
        
        return {
            'gravity_gradient': tau_gg,
            'solar_radiation_pressure': tau_srp,
            'total': tau_gg + tau_srp
        }



if __name__ == "__main__":
    import yaml
    from control.torque_generators import generate_torque_profile

    def load_config(config_file="config.yaml"):
        """Load configuration from file"""
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    cfg = load_config()

        # Parameters
    I_sat = cfg["satellite"]["inertia_tensor"]
    I_rw = cfg["reaction_wheels"]["inertia"]
    rw_axes = cfg["reaction_wheels"]["alignment_matrix"]
    dt = cfg["sim"]["dt"]
    t_max = cfg["sim"]["t_max"]
    omega0 = cfg["sim"]["initial_omega"]
    rw_speed0 = cfg["sim"]["initial_rw_speed"]
    rw_max_speed = cfg["reaction_wheels"]["max_speed"]
    rw_max_torque = cfg["reaction_wheels"]["max_torque"]
    
    # Create satellite model
    sat = SatelliteWithExternalTorques(I_sat, I_rw, rw_axes, rw_max_speed, rw_max_torque, torque_smoothing=True)
    print(f"True inertia: {I_sat}")
    print(f"Simulation time: {t_max} s")
    print(f"Initial angular velocity: {omega0}")
    print(f"Initial RW speeds: {rw_speed0}")
    print(f"RW max speed: {rw_max_speed}")
    print(f"RW max torque: {rw_max_torque}")
    print("Using satellite model with reaction wheels and external torques")

    # Run simulation
    t_span = (0, t_max)
    t = np.linspace(0, 400, 400)  # 400 seconds at 1 Hz sampling rate
    # Define a simple control function (e.g., zero torque)
    control_sequence = generate_torque_profile("sine", t)
    def enhanced_control_input(time_val):
        idx = np.argmin(np.abs(t - time_val))
        return control_sequence[idx]
    
    t_eval, states = sat.simulate_with_external_torques(
        omega0, rw_speed0, enhanced_control_input, t_span, dt
    )
    # Plot the commanded vs actual torques, then RW speeds, and angular velocities
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_eval, states[:, :3], label=['ω_x', 'ω_y', 'ω_z'])
    plt.title('Angular Velocity of Satellite')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(t_eval, states[:, 3:6], label=['RW Speed 1', 'RW Speed 2', 'RW Speed 3'])
    plt.title('Reaction Wheel Speeds')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (rad/s)')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 3)
    # Plot commanded torques (add one padding to align with t_eval)
    plt.plot(t_eval, np.pad(control_sequence, (1, 0), 'edge'), label='Commanded Torque', linestyle='--')
    # Plot actual torques from the dynamics model
    plt.plot(t_eval, sat.tau_ext, label='Actual Torque', linestyle='-')
    plt.title('Torque Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()