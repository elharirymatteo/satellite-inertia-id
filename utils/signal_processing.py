#!/usr/bin/env python3
"""
Signal processing utilities for robust derivative computation

This module provides various methods to compute smooth derivatives from noisy data,
which is critical for accurate angular acceleration estimation in satellite dynamics.
"""

import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.ndimage import gaussian_filter1d
import warnings


def compute_smooth_derivative(t, y, method='savgol', **kwargs):
    """
    Compute smooth derivatives using various methods
    
    Args:
        t: Time vector [N,]
        y: Signal array [N, 3] for 3-axis data or [N,] for single axis
        method: 'savgol', 'spline', 'gaussian', 'central_diff', 'adaptive'
        **kwargs: Method-specific parameters
        
    Returns:
        dydt: Smooth derivative with same shape as y
    """
    
    if method not in DERIVATIVE_METHODS:
        available = ', '.join(DERIVATIVE_METHODS.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")
    
    # Handle both single axis and multi-axis data
    single_axis = len(y.shape) == 1
    if single_axis:
        y = y.reshape(-1, 1)
    
    n_samples, n_axes = y.shape
    dydt = np.zeros_like(y)
    
    # Apply smoothing method to each axis
    for axis in range(n_axes):
        dydt[:, axis] = DERIVATIVE_METHODS[method](t, y[:, axis], **kwargs)
    
    return dydt.flatten() if single_axis else dydt


def savgol_derivative(t, y, window_length=None, polyorder=3, deriv=1, **kwargs):
    """
    Savitzky-Golay filter for smooth derivatives
    
    Args:
        t: Time vector
        y: Signal data
        window_length: Filter window length (auto-computed if None)
        polyorder: Polynomial order for fitting
        deriv: Derivative order (1 for first derivative)
        
    Returns:
        dydt: Smooth derivative
    """
    n = len(y)
    
    # Auto-compute window length if not provided
    if window_length is None:
        # Use ~5% of data length, but ensure it's odd and reasonable
        window_length = max(5, min(n // 10, 51))
        if window_length % 2 == 0:
            window_length += 1
    
    # Ensure window length is valid
    window_length = min(window_length, n)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < polyorder + 1:
        polyorder = max(1, window_length - 1)
    
    # Check if we have uniform time spacing
    dt = np.diff(t)
    is_uniform = np.allclose(dt, dt[0], rtol=1e-6)
    
    if is_uniform:
        # Use standard Savitzky-Golay for uniform spacing
        dydt = signal.savgol_filter(y, window_length, polyorder, deriv=deriv, delta=dt[0])
    else:
        # For non-uniform spacing, interpolate to uniform grid first
        warnings.warn("Non-uniform time spacing detected. Interpolating to uniform grid.")
        t_uniform = np.linspace(t[0], t[-1], len(t))
        y_interp = np.interp(t_uniform, t, y)
        dt_uniform = t_uniform[1] - t_uniform[0]
        dydt_uniform = signal.savgol_filter(y_interp, window_length, polyorder, 
                                          deriv=deriv, delta=dt_uniform)
        # Interpolate back to original time grid
        dydt = np.interp(t, t_uniform, dydt_uniform)
    
    return dydt


def spline_derivative(t, y, smoothing=None, k=3, **kwargs):
    """
    Cubic spline interpolation with derivative computation
    
    Args:
        t: Time vector
        y: Signal data
        smoothing: Smoothing factor (None for interpolating spline)
        k: Spline degree (3 for cubic)
        
    Returns:
        dydt: Smooth derivative
    """
    
    # Auto-compute smoothing factor if not provided
    if smoothing is None:
        # Estimate noise level and set smoothing accordingly
        noise_est = np.std(np.diff(y, n=2))  # Second difference as noise estimate
        smoothing = len(y) * noise_est**2 if noise_est > 1e-10 else 0
    
    try:
        if smoothing > 0:
            # Smoothing spline
            spline = UnivariateSpline(t, y, s=smoothing, k=k)
        else:
            # Interpolating spline
            spline = CubicSpline(t, y)
        
        # Compute derivative
        dydt = spline.derivative()(t)
        
    except Exception as e:
        warnings.warn(f"Spline fitting failed: {e}. Falling back to Savgol.")
        dydt = savgol_derivative(t, y, **kwargs)
    
    return dydt


def gaussian_derivative(t, y, sigma=None, order=1, **kwargs):
    """
    Gaussian filter followed by finite difference
    
    Args:
        t: Time vector
        y: Signal data
        sigma: Gaussian filter standard deviation (auto-computed if None)
        order: Derivative order
        
    Returns:
        dydt: Smooth derivative
    """
    
    # Auto-compute sigma if not provided
    if sigma is None:
        # Base sigma on sampling rate and signal characteristics
        dt_mean = np.mean(np.diff(t))
        sigma = 2.0 / dt_mean  # Smooth over ~2 time steps
    
    # Apply Gaussian smoothing
    y_smooth = gaussian_filter1d(y, sigma)
    
    # Compute derivative using central differences
    dydt = central_diff_derivative(t, y_smooth, **kwargs)
    
    return dydt


def central_diff_derivative(t, y, **kwargs):
    """
    Central difference with edge handling
    
    Args:
        t: Time vector
        y: Signal data
        
    Returns:
        dydt: Derivative using central differences
    """
    
    n = len(y)
    dydt = np.zeros_like(y)
    
    # Central differences for interior points
    for i in range(1, n-1):
        dt_back = t[i] - t[i-1]
        dt_forward = t[i+1] - t[i]
        
        if np.abs(dt_back - dt_forward) < 1e-10:
            # Uniform spacing
            dydt[i] = (y[i+1] - y[i-1]) / (2 * dt_forward)
        else:
            # Non-uniform spacing - weighted central difference
            dydt[i] = (y[i+1] * dt_back**2 - y[i-1] * dt_forward**2 + 
                      y[i] * (dt_forward**2 - dt_back**2)) / (dt_back * dt_forward * (dt_back + dt_forward))
    
    # Forward difference for first point
    dydt[0] = (y[1] - y[0]) / (t[1] - t[0])
    
    # Backward difference for last point
    dydt[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    
    return dydt


def adaptive_derivative(t, y, noise_threshold=None, **kwargs):
    """
    Adaptive method that chooses the best approach based on signal characteristics
    
    Args:
        t: Time vector
        y: Signal data
        noise_threshold: Threshold for noise detection
        
    Returns:
        dydt: Smooth derivative using best method
    """
    
    # Estimate signal-to-noise ratio
    if noise_threshold is None:
        noise_est = np.std(np.diff(y, n=2))  # Second difference as noise estimate
        signal_est = np.std(y)
        snr = signal_est / (noise_est + 1e-12)
        noise_threshold = 10.0  # SNR threshold
    else:
        snr = noise_threshold + 1  # Force method selection
    
    # Choose method based on signal characteristics
    if snr > noise_threshold:
        # Low noise - use spline
        method = 'spline'
        params = {'smoothing': 0}  # Interpolating spline
    else:
        # High noise - use Savitzky-Golay
        method = 'savgol'
        params = {'window_length': None, 'polyorder': 3}
    
    # Update params with any user-provided kwargs
    params.update(kwargs)
    
    return DERIVATIVE_METHODS[method](t, y, **params)


def validate_derivative_quality(t, y, dydt, method_name="Unknown"):
    """
    Validate the quality of computed derivatives
    
    Args:
        t: Time vector
        y: Original signal
        dydt: Computed derivative
        method_name: Name of method used
        
    Returns:
        dict: Quality metrics
    """
    
    # Compute reference derivative using simple differences
    dydt_ref = np.gradient(y, t)
    
    # Quality metrics
    rmse = np.sqrt(np.mean((dydt - dydt_ref)**2))
    max_diff = np.max(np.abs(dydt - dydt_ref))
    correlation = np.corrcoef(dydt, dydt_ref)[0, 1]
    
    # Smoothness metric (second derivative)
    d2ydt2 = np.gradient(dydt, t)
    smoothness = np.std(d2ydt2)
    
    metrics = {
        'method': method_name,
        'rmse_vs_gradient': rmse,
        'max_difference': max_diff,
        'correlation': correlation,
        'smoothness_metric': smoothness,
        'is_smooth': smoothness < np.std(np.gradient(dydt_ref, t))
    }
    
    return metrics


# Registry of available methods
DERIVATIVE_METHODS = {
    'savgol': savgol_derivative,
    'spline': spline_derivative,
    'gaussian': gaussian_derivative,
    'central_diff': central_diff_derivative,
    'adaptive': adaptive_derivative,
}


def compare_derivative_methods(t, y, methods=None, plot=False):
    """
    Compare different derivative computation methods
    
    Args:
        t: Time vector
        y: Signal data
        methods: List of methods to compare (None for all)
        plot: Whether to create comparison plots
        
    Returns:
        dict: Results from each method with quality metrics
    """
    
    if methods is None:
        methods = list(DERIVATIVE_METHODS.keys())
    
    results = {}
    
    for method in methods:
        try:
            dydt = compute_smooth_derivative(t, y, method=method)
            metrics = validate_derivative_quality(t, y, dydt, method)
            results[method] = {
                'derivative': dydt,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Method '{method}' failed: {e}")
            continue
    
    if plot and len(results) > 0:
        _plot_derivative_comparison(t, y, results)
    
    return results


def _plot_derivative_comparison(t, y, results):
    """Internal function to plot derivative comparison"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original signal
        axes[0].plot(t, y, 'k-', linewidth=2, label='Original Signal')
        axes[0].set_ylabel('Signal')
        axes[0].set_title('Signal and Derivative Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot derivatives
        for method, data in results.items():
            dydt = data['derivative']
            metrics = data['metrics']
            label = f"{method} (corr={metrics['correlation']:.3f})"
            axes[1].plot(t, dydt, label=label, alpha=0.8)
        
        # Add reference gradient
        dydt_ref = np.gradient(y, t)
        axes[1].plot(t, dydt_ref, 'k--', alpha=0.5, label='np.gradient (reference)')
        
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Derivative')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logs/derivative_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: logs/derivative_comparison.png")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


# Example usage and testing
if __name__ == "__main__":
    # Test with noisy sinusoidal signal
    t = np.linspace(0, 10, 1000)
    y_clean = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    noise = 0.1 * np.random.randn(len(t))
    y_noisy = y_clean + noise
    
    print("Testing derivative computation methods...")
    
    # Compare methods
    results = compare_derivative_methods(t, y_noisy, plot=True)
    
    # Print quality metrics
    print("\nQuality Metrics:")
    print("-" * 60)
    for method, data in results.items():
        metrics = data['metrics']
        print(f"{method:12s}: RMSE={metrics['rmse_vs_gradient']:.4f}, "
              f"Corr={metrics['correlation']:.3f}, "
              f"Smooth={'Yes' if metrics['is_smooth'] else 'No'}")
