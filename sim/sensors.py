import numpy as np

class IMU:
    """Inertial Measurement Unit model with realistic noise and biases"""
    
    def __init__(self, gyro_noise_std=1e-4, gyro_bias_std=1e-5, accel_noise_std=1e-3, 
                 accel_bias_std=1e-4, dt=0.01):
        """
        Initialize IMU parameters
        
        Args:
            gyro_noise_std: Gyroscope white noise std dev [rad/s]
            gyro_bias_std: Gyroscope bias random walk std dev [rad/s]
            accel_noise_std: Accelerometer white noise std dev [m/s²]
            accel_bias_std: Accelerometer bias random walk std dev [m/s²]
            dt: Time step for bias random walk [s]
        """
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_std = gyro_bias_std
        self.accel_noise_std = accel_noise_std
        self.accel_bias_std = accel_bias_std
        self.dt = dt
        
        # Initialize biases
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
    def measure_angular_velocity(self, true_omega):
        """
        Simulate gyroscope measurement
        
        Args:
            true_omega: True angular velocity [rad/s]
            
        Returns:
            measured_omega: Measured angular velocity with noise and bias
        """
        # Update bias (random walk)
        self.gyro_bias += np.random.normal(0, self.gyro_bias_std * np.sqrt(self.dt), 3)
        
        # Add noise and bias
        noise = np.random.normal(0, self.gyro_noise_std, 3)
        measured_omega = true_omega + self.gyro_bias + noise
        
        return measured_omega
    
    def measure_acceleration(self, true_accel):
        """
        Simulate accelerometer measurement
        
        Args:
            true_accel: True linear acceleration [m/s²]
            
        Returns:
            measured_accel: Measured acceleration with noise and bias
        """
        # Update bias (random walk)
        self.accel_bias += np.random.normal(0, self.accel_bias_std * np.sqrt(self.dt), 3)
        
        # Add noise and bias
        noise = np.random.normal(0, self.accel_noise_std, 3)
        measured_accel = true_accel + self.accel_bias + noise
        
        return measured_accel

class StarTracker:
    """Star tracker model for attitude determination"""
    
    def __init__(self, attitude_noise_std=1e-6, update_rate=1.0, fov_deg=20):
        """
        Initialize star tracker parameters
        
        Args:
            attitude_noise_std: Attitude determination noise std dev [rad]
            update_rate: Update rate [Hz]
            fov_deg: Field of view [degrees]
        """
        self.attitude_noise_std = attitude_noise_std
        self.update_rate = update_rate
        self.fov_deg = fov_deg
        self.last_update_time = 0
        
    def measure_attitude(self, true_attitude_quat, time, sun_vector=None):
        """
        Simulate star tracker attitude measurement
        
        Args:
            true_attitude_quat: True attitude quaternion [4x1] (w,x,y,z)
            time: Current time [s]
            sun_vector: Sun direction vector (if provided, may cause blinding)
            
        Returns:
            measured_quat: Measured attitude quaternion with noise
            measurement_valid: Boolean indicating if measurement is valid
        """
        # Check update rate
        if time - self.last_update_time < 1.0 / self.update_rate:
            return None, False
            
        self.last_update_time = time
        
        # Check for sun blinding (simplified)
        measurement_valid = True
        if sun_vector is not None:
            # If sun is within FOV, measurement may be invalid
            sun_angle = np.arccos(np.abs(sun_vector[2]))  # Assuming boresight is +Z
            if np.degrees(sun_angle) < self.fov_deg / 2:
                measurement_valid = False
                return None, False
        
        # Add noise to attitude (small angle approximation)
        noise_angles = np.random.normal(0, self.attitude_noise_std, 3)
        
        # Convert noise to quaternion (small angle)
        noise_quat = np.array([
            1,
            noise_angles[0] / 2,
            noise_angles[1] / 2,
            noise_angles[2] / 2
        ])
        noise_quat = noise_quat / np.linalg.norm(noise_quat)
        
        # Quaternion multiplication (noise_quat * true_quat)
        measured_quat = quaternion_multiply(noise_quat, true_attitude_quat)
        
        return measured_quat, measurement_valid

class Magnetometer:
    """Magnetometer model for magnetic field measurement"""
    
    def __init__(self, noise_std=1e-9, bias_std=1e-10, scale_factor_error=1e-3):
        """
        Initialize magnetometer parameters
        
        Args:
            noise_std: Measurement noise std dev [T]
            bias_std: Bias uncertainty [T]
            scale_factor_error: Scale factor error (fraction)
        """
        self.noise_std = noise_std
        self.bias = np.random.normal(0, bias_std, 3)
        self.scale_factors = 1 + np.random.normal(0, scale_factor_error, 3)
        
    def measure_magnetic_field(self, true_field):
        """
        Simulate magnetometer measurement
        
        Args:
            true_field: True magnetic field vector [T]
            
        Returns:
            measured_field: Measured field with noise, bias, and scale errors
        """
        noise = np.random.normal(0, self.noise_std, 3)
        measured_field = self.scale_factors * true_field + self.bias + noise
        
        return measured_field

class SensorSuite:
    """Combined sensor suite for spacecraft"""
    
    def __init__(self, imu_config=None, star_tracker_config=None, mag_config=None):
        """
        Initialize complete sensor suite
        
        Args:
            imu_config: Dictionary of IMU parameters
            star_tracker_config: Dictionary of star tracker parameters
            mag_config: Dictionary of magnetometer parameters
        """
        self.imu = IMU(**(imu_config or {}))
        self.star_tracker = StarTracker(**(star_tracker_config or {}))
        self.magnetometer = Magnetometer(**(mag_config or {}))
        
    def get_all_measurements(self, true_state, time, external_vectors=None):
        """
        Get measurements from all sensors
        
        Args:
            true_state: Dictionary with 'omega', 'accel', 'attitude_quat'
            time: Current time [s]
            external_vectors: Dictionary with 'sun_vector', 'mag_field'
            
        Returns:
            measurements: Dictionary with all sensor outputs
        """
        external_vectors = external_vectors or {}
        
        measurements = {
            'omega': self.imu.measure_angular_velocity(true_state['omega']),
            'time': time
        }
        
        if 'accel' in true_state:
            measurements['accel'] = self.imu.measure_acceleration(true_state['accel'])
            
        if 'attitude_quat' in true_state:
            quat, valid = self.star_tracker.measure_attitude(
                true_state['attitude_quat'], time, 
                external_vectors.get('sun_vector')
            )
            measurements['attitude_quat'] = quat
            measurements['attitude_valid'] = valid
            
        if 'mag_field' in external_vectors:
            measurements['mag_field'] = self.magnetometer.measure_magnetic_field(
                external_vectors['mag_field']
            )
            
        return measurements

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q1 * q2
    Quaternions are in [w, x, y, z] format
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def get_high_accuracy_sensor_config():
    """Configuration for high-accuracy sensors"""
    return {
        'imu_config': {
            'gyro_noise_std': 1e-6,
            'gyro_bias_std': 1e-7,
            'accel_noise_std': 1e-5,
            'accel_bias_std': 1e-6
        },
        'star_tracker_config': {
            'attitude_noise_std': 1e-6,
            'update_rate': 10.0,
            'fov_deg': 20
        },
        'mag_config': {
            'noise_std': 1e-10,
            'bias_std': 1e-11,
            'scale_factor_error': 1e-4
        }
    }

def get_low_cost_sensor_config():
    """Configuration for low-cost sensors"""
    return {
        'imu_config': {
            'gyro_noise_std': 1e-4,
            'gyro_bias_std': 1e-5,
            'accel_noise_std': 1e-3,
            'accel_bias_std': 1e-4
        },
        'star_tracker_config': {
            'attitude_noise_std': 1e-4,
            'update_rate': 1.0,
            'fov_deg': 10
        },
        'mag_config': {
            'noise_std': 1e-8,
            'bias_std': 1e-9,
            'scale_factor_error': 1e-2
        }
    }


if __name__ == "__main__":
    # Test IMU sensor functionality and plot real omega vs measured omega with error model
    imu = IMU()
    # generate a continuous function over time for true omega
    t = np.linspace(0, 10, 100)
    true_omega = np.sin(t)  # Example: sinusoidal angular velocity
    measured_omega = np.array([imu.measure_angular_velocity(np.array([w, 0, 0])) for w in true_omega]) 
    measured_omega = np.array(measured_omega).squeeze()  # Remove extra dimension
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(t, true_omega, label='True Omega', color='blue')
    plt.plot(t, measured_omega[:, 0], label='Measured Omega', color='orange')
    plt.fill_between(t, measured_omega[:, 0] - imu.gyro_noise_std, 
                     measured_omega[:, 0] + imu.gyro_noise_std,
                     color='orange', alpha=0.2, label='Measurement Noise Range')
    plt.title('True vs Measured Angular Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid()
    plt.show()
