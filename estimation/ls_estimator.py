#!/usr/bin/env python3
"""
Least Squares Estimator for Inertia Identification

This module implements various least squares approaches for spacecraft inertia estimation,
including basic least squares, robust methods, and regularized approaches.
"""

import numpy as np
from scipy.optimize import least_squares
import warnings
from utils.math_utils import skew


class InertiaModel:
    """Base class for inertia parameterization models"""
    
    def __init__(self, param_bounds=None):
        self.param_bounds = param_bounds
        
    def params_to_inertia(self, params):
        """Convert parameter vector to inertia tensor"""
        raise NotImplementedError
        
    def inertia_to_params(self, inertia_tensor):
        """Convert inertia tensor to parameter vector"""
        raise NotImplementedError
        
    def get_param_bounds(self):
        """Get parameter bounds for optimization"""
        return self.param_bounds


class DiagonalInertiaModel(InertiaModel):
    """Diagonal inertia tensor model: I = diag(Ixx, Iyy, Izz)"""
    
    def __init__(self, param_bounds=None):
        if param_bounds is None:
            param_bounds = [(0.01, 20.0)] * 3  # Default bounds for Ixx, Iyy, Izz
        super().__init__(param_bounds)
        
    def params_to_inertia(self, params):
        """Convert [Ixx, Iyy, Izz] to diagonal matrix"""
        return np.diag(params)
        
    def inertia_to_params(self, inertia_tensor):
        """Extract diagonal elements"""
        return np.diag(inertia_tensor)


class FullInertiaModel(InertiaModel):
    """Full 3x3 symmetric inertia tensor model"""
    
    def __init__(self, param_bounds=None):
        if param_bounds is None:
            # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
            param_bounds = [(0.01, 10.0)] * 3 + [(-5.0, 5.0)] * 3
        super().__init__(param_bounds)
        
    def params_to_inertia(self, params):
        """Convert 6-element vector to symmetric 3x3 matrix"""
        I = np.zeros((3, 3))
        I[0, 0] = params[0]  # Ixx
        I[1, 1] = params[1]  # Iyy
        I[2, 2] = params[2]  # Izz
        I[0, 1] = I[1, 0] = params[3]  # Ixy
        I[0, 2] = I[2, 0] = params[4]  # Ixz
        I[1, 2] = I[2, 1] = params[5]  # Iyz
        return I
        
    def inertia_to_params(self, inertia_tensor):
        """Extract 6-element vector from symmetric matrix"""
        return np.array([
            inertia_tensor[0, 0],  # Ixx
            inertia_tensor[1, 1],  # Iyy
            inertia_tensor[2, 2],  # Izz
            inertia_tensor[0, 1],  # Ixy
            inertia_tensor[0, 2],  # Ixz
            inertia_tensor[1, 2]   # Iyz
        ])


class LeastSquaresEstimator:
    """
    Least squares estimator for spacecraft inertia identification
    
    This class implements nonlinear least squares estimation based on the full
    satellite dynamics model including reaction wheel coupling.
    """
    
    def __init__(self, model=None, satellite_model=None, regularization_weight=0.0, robust_loss=None):
        """
        Initialize the least squares estimator
        
        Args:
            model: InertiaModel instance (defaults to DiagonalInertiaModel)
            satellite_model: Satellite instance with RW dynamics (optional)
            regularization_weight: L2 regularization weight
            robust_loss: Robust loss function ('soft_l1', 'huber', 'cauchy', 'arctan')
        """
        self.model = model if model is not None else DiagonalInertiaModel()
        self.satellite_model = satellite_model
        self.regularization_weight = regularization_weight
        self.robust_loss = robust_loss
        self.estimation_history = []
        
    def residual_function(self, params, dynamics_data):
        """
        Enhanced residual function using actual torques and full dynamics
        """
        I_tensor = self.model.params_to_inertia(params)
        omega = dynamics_data['omega']
        domega_measured = dynamics_data['domega']
        tau_actual = dynamics_data.get('tau_actual', dynamics_data['torque'])  # Use actual torques
        rw_speeds = dynamics_data.get('rw_speeds', None)
        time_data = dynamics_data.get('time', None)  # Get time data if available
        dt = dynamics_data.get('dt', 1.0)  # Default dt=1.0s
        
        residuals = []
        
        for i in range(len(omega)):
            # Get time value for external torque computation
            time_val = time_data[i] if time_data is not None else float(i) * dt  # Default dt=1.0s

            if self.satellite_model is not None and rw_speeds is not None:
                # Use full satellite dynamics with actual torques and time index
                domega_predicted = self._compute_domega_with_satellite_model(
                    I_tensor, omega[i], rw_speeds[i], tau_actual[i], time_val, time_index=i
                )
            else:
                # Fallback to simplified dynamics
                domega_predicted = self._compute_domega_simple(
                    I_tensor, omega[i], tau_actual[i]
                )
            
            residual = domega_measured[i] - domega_predicted
            residuals.append(residual)
        
        residuals = np.array(residuals).flatten()
        
        # Add regularization
        if self.regularization_weight > 0:
            reg_term = self.regularization_weight * np.linalg.norm(params)
            residuals = np.append(residuals, reg_term)
        
        return residuals
    
    def _compute_domega_with_satellite_model(self, I_tensor, omega, rw_speeds, tau_rw, time_val=0.0, time_index=None):
        """
        Compute angular acceleration using full satellite model (matches simulator)
        
        This method detects if external torques are supported and uses pre-computed
        external torques from the satellite model for improved accuracy and efficiency.
        """
        # Check if satellite model supports external torques
        from sim.dynamics import SatelliteWithExternalTorques
        has_external_torques = isinstance(self.satellite_model, SatelliteWithExternalTorques)
        
        if has_external_torques and hasattr(self.satellite_model, 'tau_ext') and len(self.satellite_model.tau_ext) > 0:
            # Use pre-computed external torques from satellite simulation
            if time_index is not None and time_index < len(self.satellite_model.tau_ext):
                tau_ext = self.satellite_model.tau_ext[time_index]
            else:
                # Fallback: use the last available external torque
                tau_ext = self.satellite_model.tau_ext[-1] if len(self.satellite_model.tau_ext) > 0 else np.zeros(3)
            
            # Compute angular acceleration using basic RW dynamics + external torques
            domega = self._compute_domega_with_external_torques(I_tensor, omega, rw_speeds, tau_rw, tau_ext)

        else:
            # Use basic RW dynamics for regular Satellite model or when no external torques available
            domega = self._compute_domega_basic_rw(I_tensor, omega, rw_speeds, tau_rw)
        
        return domega
    
    def _compute_domega_with_external_torques(self, I_tensor, omega, rw_speeds, tau_rw, tau_ext):
        """
        Compute angular acceleration using RW dynamics + pre-computed external torques
        
        Args:
            I_tensor: 3x3 inertia tensor
            omega: Angular velocity (3,)
            rw_speeds: Reaction wheel speeds (3,)
            tau_rw: RW torques (3,)
            tau_ext: Pre-computed external torques (3,)
            
        Returns:
            domega: Angular acceleration (3,)
        """
        I_rw = self.satellite_model.I_rw
        rw_axes = self.satellite_model.rw_axes
        
        # RW angular acceleration
        rw_acc = tau_rw / I_rw
        
        # Total angular momentum of RWs
        h_rw_total = np.zeros(3)
        for j in range(3):
            h_rw_total += rw_axes[j] * (I_rw * rw_speeds[j])
        
        # Rate of change of RW angular momentum
        h_rw_dot = np.zeros(3)
        for j in range(3):
            h_rw_dot += rw_axes[j] * (I_rw * rw_acc[j])
        
        # Total angular momentum
        h_total = I_tensor @ omega + h_rw_total
        
        # Modified Euler's equation with external torques (matches SatelliteWithExternalTorques)
        omega_cross = skew(omega)
        try:
            domega = np.linalg.solve(I_tensor, tau_ext - omega_cross @ h_total - h_rw_dot)
        except np.linalg.LinAlgError:
            domega = np.zeros(3)
        
        return domega
    
    def _compute_domega_basic_rw(self, I_tensor, omega, rw_speeds, tau_rw):
        """
        Compute angular acceleration using basic RW dynamics (no external torques)
        """
        I_rw = self.satellite_model.I_rw
        rw_axes = self.satellite_model.rw_axes
        
        # RW angular acceleration
        rw_acc = tau_rw / I_rw
        
        # Total angular momentum of RWs
        h_rw_total = np.zeros(3)
        for j in range(3):
            h_rw_total += rw_axes[j] * (I_rw * rw_speeds[j])
        
        # Rate of change of RW angular momentum
        h_rw_dot = np.zeros(3)
        for j in range(3):
            h_rw_dot += rw_axes[j] * (I_rw * rw_acc[j])
        
        # Total angular momentum
        h_total = I_tensor @ omega + h_rw_total
        
        # Euler's equation with RW coupling (no external torques)
        omega_cross = skew(omega)
        try:
            domega = np.linalg.solve(I_tensor, -omega_cross @ h_total - h_rw_dot)
        except np.linalg.LinAlgError:
            domega = np.zeros(3)
        
        return domega
    
    def _compute_domega_simple(self, I_tensor, omega, tau):
        """
        Compute angular acceleration using simplified Euler's equation
        
        Args:
            I_tensor: 3x3 inertia tensor
            omega: Angular velocity (3,)
            tau: Applied torques (3,)
            
        Returns:
            domega: Predicted angular acceleration (3,)
        """
        # Euler's equation: I * ω̇ = τ - ω × (I * ω)
        I_omega = I_tensor @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)
        
        try:
            domega = np.linalg.solve(I_tensor, tau - omega_cross_I_omega)
        except np.linalg.LinAlgError:
            # Handle singular matrices
            domega = np.zeros(3)
        
        return domega
    
    def set_satellite_model(self, satellite_model):
        """Set the satellite model to use for dynamics computation"""
        self.satellite_model = satellite_model
    
    def estimate(self, dynamics_data, initial_guess=None, verbose=False):
        """
        Perform least squares estimation
        
        Args:
            dynamics_data: Dictionary with 'omega', 'domega', 'torque'
            initial_guess: Initial parameter guess (auto-generated if None)
            verbose: Whether to print optimization details
            
        Returns:
            Dictionary with estimation results
        """
        
        # Generate initial guess if not provided
        if initial_guess is None:
            initial_guess = self._generate_initial_guess(dynamics_data)
        
        # Get parameter bounds
        bounds = self.model.get_param_bounds()
        
        # Ensure initial guess is within bounds
        if bounds:
            lower_bounds = [b[0] for b in bounds]
            upper_bounds = [b[1] for b in bounds]
            initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)
            bounds_tuple = (lower_bounds, upper_bounds)
        else:
            bounds_tuple = (-np.inf, np.inf)
        
        try:
            # Run nonlinear least squares optimization
            result = least_squares(
                self.residual_function,
                initial_guess,
                args=(dynamics_data,),
                bounds=bounds_tuple,
                loss=self.robust_loss if self.robust_loss is not None else 'linear',
                max_nfev=1000,
                ftol=1e-9,
                xtol=1e-9
            )
            
            # Convert parameters to inertia tensor
            estimated_inertia = self.model.params_to_inertia(result.x)
            
            # Compute final cost and validation metrics
            final_residuals = self.residual_function(result.x, dynamics_data)
            rms_error = np.sqrt(np.mean(final_residuals**2))
            
            estimation_result = {
                'success': result.success,
                'parameters': result.x,
                'inertia_tensor': estimated_inertia,
                'cost': result.cost,
                'rms_error': rms_error,
                'residuals': final_residuals,
                'message': result.message,
                'n_function_evaluations': result.nfev,
                'optimality': result.optimality,
                'initial_guess': initial_guess
            }
            
            if verbose:
                print(f"LS Estimation {'SUCCESS' if result.success else 'FAILED'}")
                print(f"  Parameters: {result.x}")
                print(f"  RMS Error: {rms_error:.6f}")
                print(f"  Function Evaluations: {result.nfev}")
                if not result.success:
                    print(f"  Message: {result.message}")
            
            # Store in history
            self.estimation_history.append(estimation_result)
            
            return estimation_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'parameters': initial_guess,
                'inertia_tensor': self.model.params_to_inertia(initial_guess),
                'cost': np.inf,
                'rms_error': np.inf,
                'residuals': None,
                'message': f"Optimization failed: {str(e)}",
                'n_function_evaluations': 0,
                'optimality': np.inf,
                'initial_guess': initial_guess
            }
            
            if verbose:
                print(f"LS Estimation FAILED: {str(e)}")
            
            self.estimation_history.append(error_result)
            return error_result
    
    def _generate_initial_guess(self, dynamics_data):
        """Generate initial parameter guess based on data characteristics"""
        
        omega = dynamics_data['omega']
        torque = dynamics_data['torque']
        
        # Simple heuristic based on energy scaling
        avg_omega_magnitude = np.mean(np.linalg.norm(omega, axis=1))
        avg_torque_magnitude = np.mean(np.linalg.norm(torque, axis=1))
        
        # Rough estimate assuming spherical inertia
        if avg_omega_magnitude > 1e-6:
            I_estimate = avg_torque_magnitude / (avg_omega_magnitude**2 + 1e-6)
            I_estimate = np.clip(I_estimate, 0.01, 10.0)  # Keep within reasonable bounds
        else:
            I_estimate = 1.0  # Default value
        
        if isinstance(self.model, DiagonalInertiaModel):
            # For diagonal model, use slightly different values for each axis
            return np.array([I_estimate * 0.8, I_estimate * 1.0, I_estimate * 1.2])
        elif isinstance(self.model, FullInertiaModel):
            # For full model, start with diagonal + small off-diagonal terms
            return np.array([
                I_estimate * 0.8,   # Ixx
                I_estimate * 1.0,   # Iyy
                I_estimate * 1.2,   # Izz
                0.01,               # Ixy
                0.01,               # Ixz
                0.01                # Iyz
            ])
        else:
            # Generic case
            n_params = len(self.model.get_param_bounds())
            return np.ones(n_params) * I_estimate
    
    def validate_estimate(self, estimation_result, dynamics_data, tolerance=0.1):
        """
        Validate estimation result against dynamics data
        
        Args:
            estimation_result: Result from estimate() method
            dynamics_data: Original dynamics data
            tolerance: Error tolerance for validation
            
        Returns:
            Validation metrics dictionary
        """
        
        if not estimation_result['success']:
            return {
                'is_valid': False,
                'mean_error': np.inf,
                'max_error': np.inf,
                'message': 'Estimation failed'
            }
        
        inertia_tensor = estimation_result['inertia_tensor']
        omega = dynamics_data['omega']
        domega = dynamics_data['domega']
        torque = dynamics_data['torque']
        
        errors = []
        
        for i in range(len(omega)):
            w = omega[i]
            dw = domega[i]
            tau = torque[i]
            
            # Predict angular acceleration using estimated inertia
            I_omega = inertia_tensor @ w
            omega_cross_I_omega = np.cross(w, I_omega)
            
            try:
                dw_pred = np.linalg.solve(inertia_tensor, tau - omega_cross_I_omega)
                error = np.linalg.norm(dw - dw_pred)
                errors.append(error)
            except np.linalg.LinAlgError:
                errors.append(np.inf)
        
        errors = np.array(errors)
        mean_error = np.mean(errors[np.isfinite(errors)])
        max_error = np.max(errors[np.isfinite(errors)]) if np.any(np.isfinite(errors)) else np.inf
        
        is_valid = mean_error < tolerance and np.isfinite(mean_error)
        
        return {
            'is_valid': is_valid,
            'mean_error': mean_error,
            'max_error': max_error,
            'error_history': errors,
            'message': 'Valid' if is_valid else f'Mean error {mean_error:.4f} > tolerance {tolerance}'
        }
    
    def get_estimation_history(self):
        """Return the history of all estimations performed"""
        return self.estimation_history
    
    def reset_history(self):
        """Clear the estimation history"""
        self.estimation_history = []

def batch_estimation(estimator, dynamics_data_list, **kwargs):
    """
    Perform estimation on multiple datasets
    
    Args:
        estimator: LeastSquaresEstimator instance
        dynamics_data_list: List of dynamics data dictionaries
        **kwargs: Additional arguments for estimate()
        
    Returns:
        List of estimation results
    """
    
    results = []
    
    for i, dynamics_data in enumerate(dynamics_data_list):
        print(f"Processing dataset {i+1}/{len(dynamics_data_list)}")
        result = estimator.estimate(dynamics_data, **kwargs)
        results.append(result)
    
    return results

# Example usage and testing
if __name__ == "__main__":
    
    # Test with synthetic data
    print("Testing Least Squares Estimator...")
    
    # Generate synthetic dynamics data
    np.random.seed(42)
    horizon = 400
    dt = 1.0

    estimator_diag = LeastSquaresEstimator(model=DiagonalInertiaModel(),  robust_loss='huber')
    # True inertia
    I_true = np.diag([0.5, 0.8, 1.2])
    
    # Generate random angular velocities and torques
    omega = np.random.randn(horizon, 3) * 0.5
    torque = np.random.randn(horizon, 3) * 0.1

    # Compute true angular acceleration using Euler's equation
    domega = np.zeros_like(omega)
    for i in range(horizon):
        w = omega[i]
        tau = torque[i]
        I_omega = I_true @ w
        omega_cross_I_omega = np.cross(w, I_omega)
        domega[i] = np.linalg.solve(I_true, tau - omega_cross_I_omega)
    
    # Add noise to measurements
    omega_noisy = omega + np.random.randn(*omega.shape) * 0.01
    domega_noisy = domega + np.random.randn(*domega.shape) * 0.01
    
    dynamics_data = {
        'omega': omega_noisy,
        'domega': domega_noisy,
        'torque': torque
    }
    
    # Test diagonal inertia estimation
    print("\nTesting Diagonal Inertia Model:")
    
    
    result_diag = estimator_diag.estimate(dynamics_data, verbose=True)
    
    print(f"True diagonal inertia: {np.diag(I_true)}")
    print(f"Estimated diagonal inertia: {result_diag['parameters']}")
    print(f"Error: {np.abs(result_diag['parameters'] - np.diag(I_true))}")
    
    # Test full inertia estimation
    print("\nTesting Full Inertia Model:")
    estimator_full = LeastSquaresEstimator(model=FullInertiaModel(), robust_loss='huber')
    result_full = estimator_full.estimate(dynamics_data, verbose=True)

    # Validate results
    validation_diag = estimator_diag.validate_estimate(result_diag, dynamics_data)
    validation_full = estimator_full.validate_estimate(result_full, dynamics_data)
    
    print(f"\nValidation Results:")
    print(f"Diagonal model: {validation_diag['message']}")
    print(f"Full model: {validation_full['message']}")
