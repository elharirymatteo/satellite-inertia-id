import numpy as np
from scipy.linalg import cholesky

class UKFInertiaEstimator:
    """Enhanced UKF with RW states for inertia identification"""

    def __init__(self, dt, W_init, I_init, rw_init=None, Q_diag=1e-8, R_diag=1e-5, 
                 alpha=1e-3, beta=2.0, kappa=0, P0=None):
        self.dt = dt
        self.n = 9  # State dimension [ωx, ωy, ωz, Ixx, Iyy, Izz, rw1, rw2, rw3]
        
        # Initialize RW speeds
        if rw_init is None:
            rw_init = np.zeros(3)
        
        # State: [angular velocity, inertia, rw_speeds]
        self.x = np.concatenate([W_init, I_init, rw_init])
        
        if P0 is None:
            self.P = np.diag([1e-3, 1e-3, 1e-3,  # omega
                             1e-3, 1e-3, 1e-3,  # inertia
                             1e-1, 1e-1, 1e-1])  # rw speeds
        else:
            self.P = P0
        
        # Noise covariances
        self.Q = np.diag([Q_diag]*3 + [1e-10]*3 + [1e-4]*3)  # Process noise (ω + I + rw)
        self.R = np.eye(3) * R_diag                          # Measurement noise (ω)
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Derived parameters
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        
        # Weights
        self.Wm = np.zeros(2*self.n + 1)  # Mean weights
        self.Wc = np.zeros(2*self.n + 1)  # Covariance weights
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)
        
        for i in range(1, 2*self.n + 1):
            self.Wm[i] = 1 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.n + self.lambda_))
    
    def generate_sigma_points(self, x, P):
        """Generate sigma points for UKF"""
        n = len(x)
        sigma_points = np.zeros((2*n + 1, n))
        
        # Compute square root of (n + λ)P
        try:
            sqrt = cholesky((n + self.lambda_) * P, lower=True)
        except np.linalg.LinAlgError:
            sqrt = np.sqrt((n + self.lambda_)) * np.eye(n) * 1e-6
        
        # Central point
        sigma_points[0] = x
        
        # Positive and negative perturbations
        for i in range(n):
            sigma_points[i + 1] = x + sqrt[:, i]
            sigma_points[i + 1 + n] = x - sqrt[:, i]
            
        return sigma_points
    
    def dynamics_model(self, x, tau, satellite_model=None):
        """
        Enhanced dynamics model with full RW coupling
        """
        ω = x[0:3]
        I = x[3:6]
        rw_speeds = x[6:9]
        I_mat = np.diag(I)
        
        if satellite_model is not None:
            # Use full satellite dynamics
            domega, drw_speeds = self._compute_full_dynamics(
                I_mat, ω, rw_speeds, tau, satellite_model
            )
        else:
            # Fallback to simplified dynamics
            omega_cross_I_omega = np.cross(ω, I * ω)
            domega = np.linalg.solve(I_mat, tau - omega_cross_I_omega)
            drw_speeds = tau / 0.0001
        
        # Use Gauss-Markov process for inertia decay (variability trick)
        # dI = -0.001 * I  # Example decay rate, can be adjusted
        dI = np.zeros(3)  # No inertia decay
        return np.concatenate([domega, dI, drw_speeds])

    def _compute_full_dynamics(self, I_mat, omega, rw_speeds, tau_rw, satellite_model):
        """Same as EKF implementation"""
        I_rw = satellite_model.I_rw
        rw_axes = satellite_model.rw_axes
        
        rw_acc = tau_rw / I_rw
        
        h_rw_total = np.zeros(3)
        for j in range(3):
            h_rw_total += rw_axes[j] * (I_rw * rw_speeds[j])
        
        h_rw_dot = np.zeros(3)
        for j in range(3):
            h_rw_dot += rw_axes[j] * (I_rw * rw_acc[j])
        
        h_total = I_mat @ omega + h_rw_total
        
        omega_cross = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        try:
            domega = np.linalg.solve(I_mat, -omega_cross @ h_total - h_rw_dot)
        except np.linalg.LinAlgError:
            domega = np.zeros(3)
        
        return domega, rw_acc
    
    def measurement_model(self, x):
        """Measurement model: only observe omega"""
        return x[0:3]  # Only angular velocity is measured
    
    def predict(self, tau, satellite_model=None):
        """UKF prediction step"""
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2*self.n + 1):
            dx = self.dynamics_model(sigma_points[i], tau, satellite_model)
            sigma_points_pred[i] = sigma_points[i] + self.dt * dx
            # Ensure positive inertia
            sigma_points_pred[i, 3:6] = np.maximum(sigma_points_pred[i, 3:6], 1e-6)
        
        # Compute predicted mean and covariance
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        
        P_pred = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            dx = sigma_points_pred[i] - self.x
            P_pred += self.Wc[i] * np.outer(dx, dx)
        
        self.P = P_pred + self.Q
    
    def update(self, omega_meas):
        """UKF update step"""
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate through measurement model
        sigma_meas = np.zeros((2*self.n + 1, 3))
        for i in range(2*self.n + 1):
            sigma_meas[i] = self.measurement_model(sigma_points[i])
        
        # Predicted measurement
        z_pred = np.sum(self.Wm[:, np.newaxis] * sigma_meas, axis=0)
        
        # Innovation covariance
        S = np.zeros((3, 3))
        for i in range(2*self.n + 1):
            dz = sigma_meas[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)
        S += self.R
        
        # Cross covariance
        Pxz = np.zeros((self.n, 3))
        for i in range(2*self.n + 1):
            dx = sigma_points[i] - self.x
            dz = sigma_meas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)
        
        # Kalman gain and update
        K = Pxz @ np.linalg.inv(S)
        innovation = omega_meas - z_pred
        
        self.x += K @ innovation
        self.P -= K @ S @ K.T
        
        # Ensure positive inertia
        self.x[3:6] = np.maximum(self.x[3:6], 1e-6)
    
    def step(self, tau, omega_meas, satellite_model=None):
        """Full UKF step"""
        self.predict(tau, satellite_model)
        self.update(omega_meas)
        return self.x.copy()
