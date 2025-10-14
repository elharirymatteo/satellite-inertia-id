import numpy as np

class ReactionWheel:
    """Model for a single reaction wheel with realistic dynamics"""
    
    def __init__(self, inertia, max_torque, max_speed, axis, damping=1e-5, friction=1e-6, noise_std=0.001):
        """
        Initialize reaction wheel parameters
        
        Args:
            inertia: Wheel inertia [kg⋅m²]
            max_torque: Maximum torque output [N⋅m]
            max_speed: Maximum wheel speed [rad/s]
            axis: Wheel axis direction in body frame [3x1]
            damping: Viscous damping coefficient
            friction: Coulomb friction coefficient
        """
        self.inertia = inertia
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.axis = np.array(axis) / np.linalg.norm(axis)  # Normalize
        self.damping = damping
        self.friction = friction
        self.noise_std = noise_std
        
    def torque_model(self, commanded_torque, wheel_speed):
        """
        Apply realistic torque limits and dynamics
        
        Args:
            commanded_torque: Desired torque [N⋅m]
            wheel_speed: Current wheel speed [rad/s]
            
        Returns:
            actual_torque: Actual torque after limits [N⋅m]
        """
        # Torque saturation
        actual_torque = np.clip(commanded_torque, -self.max_torque, self.max_torque)
        
        # Speed-dependent torque reduction (simplified motor model)
        if abs(wheel_speed) > 0.8 * self.max_speed:
            speed_factor = max(0, 1 - (abs(wheel_speed) - 0.8 * self.max_speed) / (0.2 * self.max_speed))
            actual_torque *= speed_factor
            
        return actual_torque
    
    def friction_torque(self, wheel_speed):
        """
        Compute friction torque opposing wheel motion
        
        Args:
            wheel_speed: Current wheel speed [rad/s]
            
        Returns:
            friction_torque: Torque due to friction [N⋅m]
        """
        # Viscous damping + Coulomb friction
        viscous = -self.damping * wheel_speed
        coulomb = -self.friction * np.sign(wheel_speed) if abs(wheel_speed) > 1e-3 else 0
        
        return viscous + coulomb

class ReactionWheelArray:
    """Array of reaction wheels for 3-axis control"""
    
    def __init__(self, config):
        """
        Initialize array of reaction wheels
        
        Args:
            config: Configuration dictionary or list for the reaction wheel array
        """
        # Handle both dictionary and list configurations
        if isinstance(config, dict):
            self.config = config
            wheel_configs = config.get('wheels', [])
        elif isinstance(config, list):
            # Convert list to dictionary format
            self.config = {'wheels': config}
            wheel_configs = config
        else:
            raise ValueError("Config must be dictionary or list")
            
        self.wheels = []
        for wheel_config in wheel_configs:
            self.wheels.append(ReactionWheel(**wheel_config))
        
        self.num_wheels = len(self.wheels)
        
        # Array-level parameters (with defaults)
        self.time_constant = self.config.get('time_constant', 0.1)  # s
        self.noise_std = self.config.get('noise_std', 0.001)  # N⋅m
        self.previous_output = np.zeros(3)
        self.previous_time = 0
        
        # Build axis transformation matrix (wheels to body axes)
        self.wheel_axes = self.get_wheel_axes()  # Shape: (num_wheels, 3)
        
        # For 3-wheel orthogonal configuration, we can use pseudo-inverse
        # For overdetermined systems (4+ wheels), this handles redundancy
        if self.num_wheels >= 3:
            self.torque_allocation_matrix = np.linalg.pinv(self.wheel_axes.T)
        else:
            raise ValueError("Need at least 3 wheels for 3-axis control")
            
    def compute_torques(self, commanded_torques, wheel_speeds):
        """
        Compute actual torques from all wheels
        
        Args:
            commanded_torques: Desired torques for each wheel [N x 1]
            wheel_speeds: Current wheel speeds [N x 1]
            
        Returns:
            actual_torques: Actual torques after limits [N x 1]
            friction_torques: Friction torques [N x 1]
        """
        actual_torques = np.zeros(self.num_wheels)
        friction_torques = np.zeros(self.num_wheels)
        
        for i, wheel in enumerate(self.wheels):
            actual_torques[i] = wheel.torque_model(commanded_torques[i], wheel_speeds[i])
            friction_torques[i] = wheel.friction_torque(wheel_speeds[i])
            
        return actual_torques, friction_torques
    
    def get_wheel_axes(self):
        """Get the axis directions for all wheels"""
        axes = np.zeros((self.num_wheels, 3))
        for i, wheel in enumerate(self.wheels):
            axes[i] = wheel.axis
        return axes
    
    def apply_actuator_dynamics(self, commanded_torque_3axis, current_time, wheel_speeds=None):
        """
        Apply realistic actuator dynamics to commanded 3-axis torque
        
        Args:
            commanded_torque_3axis: Desired 3-axis torque [3x1]
            current_time: Current simulation time
            wheel_speeds: Current wheel speeds [N x 1], if None uses zeros
            
        Returns:
            actual_torque_3axis: Actual 3-axis torque after realistic dynamics
        """
        dt = current_time - self.previous_time
        
        # If wheel speeds not provided, assume zero (or use previous state)
        if wheel_speeds is None:
            wheel_speeds = np.zeros(self.num_wheels)
        
        # Step 1: Allocate 3-axis torque command to individual wheels
        # commanded_torque_3axis = wheel_axes @ wheel_torques
        # Solve: wheel_torques = pinv(wheel_axes) @ commanded_torque_3axis
        commanded_wheel_torques = self.torque_allocation_matrix @ commanded_torque_3axis
        
        # Step 2: Apply individual wheel dynamics
        actual_wheel_torques, friction_torques = self.compute_torques(
            commanded_wheel_torques, wheel_speeds
        )
        
        # Step 3: Transform back to 3-axis body torques
        actual_torque_3axis = self.wheel_axes.T @ actual_wheel_torques
        friction_torque_3axis = self.wheel_axes.T @ friction_torques
        
        # Step 4: Apply system-level dynamics (time constant, filtering)
        if dt > 0:
            alpha = dt / (self.time_constant + dt)
            filtered_torque = (1 - alpha) * self.previous_output + alpha * actual_torque_3axis
        else:
            filtered_torque = actual_torque_3axis
        
        # Step 5: Add system noise
        noise = np.random.normal(0, self.noise_std, 3)
        final_torque = filtered_torque + friction_torque_3axis + noise
        
        # Update state
        self.previous_output = filtered_torque
        self.previous_time = current_time
        
        return final_torque
    
    def get_wheel_states(self, wheel_speeds):
        """Get diagnostic information about wheel states"""
        states = {}
        for i, wheel in enumerate(self.wheels):
            states[f'wheel_{i}'] = {
                'speed': wheel_speeds[i],
                'speed_fraction': abs(wheel_speeds[i]) / wheel.max_speed,
                'friction_torque': wheel.friction_torque(wheel_speeds[i])
            }
        return states

# Predefined configurations
def get_standard_rw_config():
    """Standard 3-wheel configuration aligned with body axes"""
    return {
        'wheels': [
            {
                'inertia': 0.005,
                'max_torque': 0.01,
                'max_speed': 628,  # 6000 RPM converted to rad/s
                'axis': [1, 0, 0],
                'damping': 1e-5,
                'friction': 1e-6
            },
            {
                'inertia': 0.005,
                'max_torque': 0.01,
                'max_speed': 628,
                'axis': [0, 1, 0],
                'damping': 1e-5,
                'friction': 1e-6
            },
            {
                'inertia': 0.005,
                'max_torque': 0.01,
                'max_speed': 628,
                'axis': [0, 0, 1],
                'damping': 1e-5,
                'friction': 1e-6
            }
        ],
        'time_constant': 0.1,
        'noise_std': 0.001
    }

def get_redundant_rw_config():
    """4-wheel pyramid configuration for redundancy"""
    # Pyramid configuration with wheels at 45° angles
    sqrt2 = np.sqrt(2)
    
    return {
        'wheels': [
            {
                'inertia': 0.005,
                'max_torque': 0.01,
                'max_speed': 628,
                'axis': [sqrt2/2, sqrt2/2, 0],
                'damping': 1e-5,
                'friction': 1e-6
            },
            {
                'inertia': 0.005,
                'max_torque': 0.01,
                'max_speed': 628,
                'axis': [-sqrt2/2, sqrt2/2, 0],
                'damping': 1e-5,
                'friction': 1e-6
            },
            {
                'inertia': 0.005,
                'max_torque': 0.01,
                'max_speed': 628,
                'axis': [0, -sqrt2/2, sqrt2/2],
                'damping': 1e-5,
                'friction': 1e-6
            },
            {
                'inertia': 0.005,
                'max_torque': 0.01,
                'max_speed': 628,
                'axis': [0, sqrt2/2, sqrt2/2],
                'damping': 1e-5,
                'friction': 1e-6
            }
        ],
        'time_constant': 0.1,
        'noise_std': 0.001
    }
