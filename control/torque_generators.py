#!/usr/bin/env python3
"""
Torque generation patterns for spacecraft inertia identification

This module provides various torque excitation patterns optimized for system identification.
Each pattern is designed to provide rich frequency content for accurate parameter estimation.
"""

import numpy as np
from typing import Union, Dict, List, Optional, Callable
from scipy import signal
import warnings
from utils.observability import compute_observability_metric, score_profiles_canonical


def get_safe_amplitude(rw_max_torque: float, safety_factor: float = 0.7) -> float:
    """
    Calculate safe amplitude for torque profiles based on reaction wheel limits
    
    Args:
        rw_max_torque: Maximum torque capability of reaction wheels (Nm)
        safety_factor: Safety factor to avoid saturation (0.0-1.0)
        
    Returns:
        Safe amplitude for torque profiles
    """
    return rw_max_torque * safety_factor

def generate_torque_profile(profile: str, t: np.ndarray, axes: List[str] = ['x', 'y', 'z'], 
                          **kwargs) -> np.ndarray:
    """
    Generate torque profiles for spacecraft inertia identification
    
    Args:
        profile: Type of torque pattern ('one step', 'multi step', 'sawtooth', 'sine',
                 'multi_sine', 'chirp', 'prbs', 'sine-3axis')
        t: Time vector [N,] in seconds
        axes: List of axes to generate torques for ['x', 'y', 'z']
        **kwargs: Pattern-specific parameters
        
    Returns:
        torques: Torque array [N, 3] for X, Y, Z axes
        
    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> torques = generate_torque_profile('sine', t, frequency=0.5, amplitude=0.01)
        >>> torques = generate_torque_profile('chirp', t, f0=0.1, f1=1.0, amplitude=0.005)
    """
    
    if profile not in TORQUE_GENERATORS:
        available = ', '.join(TORQUE_GENERATORS.keys())
        raise ValueError(f"Unknown profile '{profile}'. Available: {available}")
    
    generator = TORQUE_GENERATORS[profile]
    torques = generator(t, axes, **kwargs)
    
    # Ensure output shape is [N, 3]
    if torques.shape != (len(t), 3):
        raise ValueError(f"Generator output shape {torques.shape} != expected {(len(t), 3)}")
    
    return torques

def sine_torque(t: np.ndarray, axes: List[str], 
                frequency: Union[float, List[float]] = 0.01,  
                amplitude: Union[float, List[float]] = 0.005,
                phase: Union[float, List[float]] = 0.0,
                **kwargs) -> np.ndarray:
    """
    Generate sinusoidal torque profiles
    
    Args:
        t: Time vector
        axes: Axes to generate for
        frequency: Frequency(ies) in Hz [scalar or list of 3]
        amplitude: Amplitude(s) in Nm [scalar or list of 3]
        phase: Phase offset(s) in radians [scalar or list of 3]
        
    Returns:
        Torque array [N, 3]
    """
    n_axes = 3
    torques = np.zeros((len(t), n_axes))
    
    # Convert scalar inputs to lists
    freq = _ensure_list(frequency, n_axes)
    amp = _ensure_list(amplitude, n_axes)
    ph = _ensure_list(phase, n_axes)
    
    for i in range(n_axes):
        torques[:, i] = amp[i] * np.sin(2 * np.pi * freq[i] * t + ph[i])
    
    return torques

def chirp_torque(t: np.ndarray, axes: List[str],
                 f0: Union[float, List[float]] = 0.005,
                 f1: Union[float, List[float]] = 0.05,
                 amplitude: Union[float, List[float]] = 0.005,
                 method: str = 'linear',
                 **kwargs) -> np.ndarray:

    """
    Generate chirp (frequency sweep) torque profiles
    
    Args:
        t: Time vector
        axes: Axes to generate for
        f0: Starting frequency(ies) in Hz
        f1: Ending frequency(ies) in Hz
        amplitude: Amplitude(s) in Nm
        method: 'linear', 'quadratic', 'logarithmic', or 'hyperbolic'
        
    Returns:
        Torque array [N, 3]
    """
    n_axes = 3
    torques = np.zeros((len(t), n_axes))
    
    # Convert scalar inputs to lists
    f0_list = _ensure_list(f0, n_axes)
    f1_list = _ensure_list(f1, n_axes)
    amp_list = _ensure_list(amplitude, n_axes)
    
    for i in range(n_axes):
        chirp_signal = signal.chirp(t, f0_list[i], t[-1], f1_list[i], method=method)
        torques[:, i] = amp_list[i] * chirp_signal
    
    return torques


def sine_3axis_profile(t: np.ndarray, axes: List[str],
                      rw_max_torque: float = 0.01,
                      freqs: Optional[List[float]] = None,
                      amps: Optional[List[float]] = None,
                      phases: Optional[List[float]] = None,
                      safety_factor: float = 0.7,
                      **kwargs) -> np.ndarray:
    """
    Generate sinusoidal torque profiles with separate parameters for each axis
    
    Args:
        t: Time vector
        axes: Axes to generate for
        rw_max_torque: Maximum reaction wheel torque capability
        freqs: Frequencies for [x, y, z] axes in Hz
        amps: Amplitudes for [x, y, z] axes in Nm (if None, uses safe amplitude)
        phases: Phase offsets for [x, y, z] axes in radians
        safety_factor: Safety factor for amplitude scaling
        
    Returns:
        Torque array [N, 3]
    """    
    # Default parameters
    if freqs is None:
        freqs = [0.01, 0.03, 0.07]
    if amps is None:
        amps = [get_safe_amplitude(rw_max_torque, safety_factor)] * 3
    if phases is None:
        phases = [0.0, 0.0, 0.0]
    
    n_axes = 3
    torques = np.zeros((len(t), n_axes))
    
    for i in range(n_axes):
        torques[:, i] = amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])
    
    return torques

def prbs_torque(t: np.ndarray, axes: List[str],
                amplitude: Union[float, List[float]] = 0.005,
                switch_time: Union[float, List[float]] = 5.0,
                seed: int = 1,
                **kwargs) -> np.ndarray:
    """Generate Pseudo-Random Binary Sequence (PRBS) torque profiles using LFSR.

    Uses a maximal-length Linear Feedback Shift Register so the sequence has
    period 2^n - 1 and near-ideal flat autocorrelation.

    Args:
        t: Time vector
        axes: Axes to generate for
        amplitude: Amplitude(s) in Nm [scalar or list of 3]
        switch_time: Clock period between sequence bits in seconds
        seed: LFSR initial state (non-zero; each axis uses seed+axis_index)

    Returns:
        Torque array [N, 3]
    """
    n_axes = 3
    torques = np.zeros((len(t), n_axes))

    amp_list = _ensure_list(amplitude, n_axes)
    switch_list = _ensure_list(switch_time, n_axes)

    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    for i in range(n_axes):
        clock_samples = max(1, round(switch_list[i] / dt))
        n_switches = int(np.ceil(len(t) / clock_samples)) + 2

        n_bits = max(4, min(12, int(np.ceil(np.log2(n_switches + 1)))))
        seq = _lfsr_sequence(n_bits=n_bits, n_samples=n_switches, seed=seed + i)

        for j, val in enumerate(seq):
            start = j * clock_samples
            end = min((j + 1) * clock_samples, len(t))
            if start >= len(t):
                break
            torques[start:end, i] = amp_list[i] * val

    return torques

def one_step_torque(t: np.ndarray, axes: List[str],
                    amplitude: Union[float, List[float]] = 0.005, 
                    step_duration: Union[float, List[float]] = 40.0, 
                    **kwargs) -> np.ndarray:
    """    Generate a single step torque profile
    """
    n_axes = len(axes)
    torques = np.zeros((len(t), n_axes))

    # Convert step_duration to a list
    step_duration = t[-1] / 3

    for i in range(n_axes):
        start_time = 0
        end_time = step_duration
        mask = (t >= start_time) & (t < end_time)
        torques[mask, i] = amplitude

    return torques

# construct a torque profile with 6 sequential steps +1/-1/0/+1/-1/0 where each step lasts a fraction of the total time t_max
def step_torque(t: np.ndarray, axes: List[str],
                amplitude: Union[float, List[float]] = 0.005, 
                step_duration: Union[float, List[float]] = 40.0,
                **kwargs) -> np.ndarray:

    n_axes = len(axes)
    torques = np.zeros((len(t), n_axes))

    # Convert step_duration to a list
    step_duration = t[-1] / 6

    for i in range(n_axes):
        for j in range(6):
            start_time = j * step_duration
            end_time = start_time + step_duration
            mask = (t >= start_time) & (t < end_time)
            if j % 3 == 0:
                torques[mask, i] = amplitude
            elif j % 3 == 1:
                torques[mask, i] = -amplitude
            else:
                torques[mask, i] = 0

    return torques

def random_torque(t: np.ndarray, axes: List[str],
                  amplitude: Union[float, List[float]] = 0.003,
                  noise_type: str = 'white',  
                  cutoff_freq: Optional[float] = None,
                  seed: int = 42,
                  **kwargs) -> np.ndarray:
    """
    Generate random torque profiles
    
    Args:
        t: Time vector
        axes: Axes to generate for
        amplitude: Amplitude(s) in Nm (RMS)
        noise_type: 'white', 'colored', or 'filtered'
        cutoff_freq: Cutoff frequency for filtered noise (Hz)
        seed: Random seed for reproducibility
        
    Returns:
        Torque array [N, 3]
    """
    n_axes = 3
    torques = np.zeros((len(t), n_axes))
    
    amp_list = _ensure_list(amplitude, n_axes)
    
    np.random.seed(seed)
    
    for i in range(n_axes):
        if noise_type == 'white':
            noise = np.random.normal(0, amp_list[i], len(t))
        elif noise_type == 'colored':
            # Generate colored noise using AR process
            noise = _generate_colored_noise(len(t), amp_list[i])
        elif noise_type == 'filtered':
            # Generate white noise and filter it
            white_noise = np.random.normal(0, amp_list[i], len(t))
            if cutoff_freq is not None:
                dt = t[1] - t[0]
                nyquist = 0.5 / dt
                normalized_cutoff = cutoff_freq / nyquist
                b, a = signal.butter(4, normalized_cutoff, btype='low')
                noise = signal.filtfilt(b, a, white_noise)
            else:
                noise = white_noise
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        
        torques[:, i] = noise
    
    return torques

def zero_torque(t: np.ndarray, axes: List[str], **kwargs) -> np.ndarray:
    """
    Generate zero torque (no control input)
    
    Useful for baseline testing or free-motion analysis
    
    Args:
        t: Time vector
        axes: Axes to generate for
        
    Returns:
        Zero torque array [N, 3]
    """
    return np.zeros((len(t), 3))

def constant_torque(t: np.ndarray, axes: List[str],
                    amplitude: Union[float, List[float]] = 0.005,
                    **kwargs) -> np.ndarray:
    """
    Generate constant torque profiles

    Args:
        t: Time vector
        axes: Axes to generate for
        amplitude: Amplitude(s) in Nm

    Returns:
        Torque array [N, 3]
    """
    n_axes = 3
    torques = np.zeros((len(t), n_axes))

    amp_list = _ensure_list(amplitude, n_axes)

    for i in range(n_axes):
        torques[:, i] = amp_list[i]

    return torques

def custom_torque(t: np.ndarray, axes: List[str], 
                  torque_func: Callable[[np.ndarray], np.ndarray],
                  **kwargs) -> np.ndarray:
    """
    Generate custom torque profiles using a user-provided function
    
    Args:
        t: Time vector
        axes: Axes to generate for
        torque_func: Function that takes time array and returns [N, 3] torques
        
    Returns:
        Torque array [N, 3]
    """
    torques = torque_func(t)
    
    if torques.shape != (len(t), 3):
        raise ValueError(f"Custom function output shape {torques.shape} != expected {(len(t), 3)}")
    
    return torques

def multi_sine_torque(t: np.ndarray, axes: List[str],
                     frequencies: List[List[float]] = [[0.002, 0.005], [0.003, 0.007], [0.004, 0.008]],
                     amplitudes: List[List[float]] = [[0.003, 0.0015], [0.0035, 0.002], [0.0025, 0.0015]], 
                     phases: List[List[float]] = [[0, 0], [np.pi/4, 0], [-np.pi/3, 0]], 
                     **kwargs) -> np.ndarray:
    """
    Generate multi-frequency sinusoidal torque profiles (sum of sines)
    
    This creates rich frequency content by combining multiple sinusoids per axis
    
    Args:
        t: Time vector
        axes: Axes to generate for
        frequencies: List of frequency lists for each axis [[x_freqs], [y_freqs], [z_freqs]]
        amplitudes: List of amplitude lists for each axis
        phases: List of phase lists for each axis
        
    Returns:
        Torque array [N, 3]
    """
    n_axes = 3
    torques = np.zeros((len(t), n_axes))
    
    # Ensure we have data for all 3 axes
    if len(frequencies) != n_axes or len(amplitudes) != n_axes or len(phases) != n_axes:
        raise ValueError("frequencies, amplitudes, and phases must have 3 elements (one per axis)")
    
    for axis in range(n_axes):
        axis_freqs = frequencies[axis]
        axis_amps = amplitudes[axis]
        axis_phases = phases[axis]
        
        if len(axis_freqs) != len(axis_amps) or len(axis_freqs) != len(axis_phases):
            raise ValueError(f"Axis {axis}: frequencies, amplitudes, and phases must have same length")
        
        # Sum multiple sinusoids for this axis
        for freq, amp, phase in zip(axis_freqs, axis_amps, axis_phases):
            torques[:, axis] += amp * np.sin(2 * np.pi * freq * t + phase)
    
    return torques

def sawtooth_torque(t: np.ndarray, axes: List[str],
                    frequency: Union[float, List[float]] = 0.01,
                    amplitude: Union[float, List[float]] = 0.005,
                    phase: Union[float, List[float]] = 0.0,
                    **kwargs) -> np.ndarray:
    """Generate sawtooth torque profiles"""
    n_axes = 3
    torques = np.zeros((len(t), n_axes))

    freq_list = _ensure_list(frequency, n_axes)
    amp_list = _ensure_list(amplitude, n_axes)
    phase_list = _ensure_list(phase, n_axes)

    for i in range(n_axes):
        torques[:, i] = amp_list[i] * (2 * (t * freq_list[i] - np.floor(0.5 + t * freq_list[i])) + phase_list[i])

    return torques


# Helper functions
def _ensure_list(value: Union[float, List[float]], length: int) -> List[float]:
    """Convert scalar to list or validate list length"""
    if isinstance(value, (int, float)):
        return [float(value)] * length
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == length:
            return list(value)
        elif len(value) == 1:
            return list(value) * length
        else:
            raise ValueError(f"Length mismatch: expected {length}, got {len(value)}")
    else:
        raise TypeError(f"Expected scalar or list, got {type(value)}")


def _generate_colored_noise(n_samples: int, amplitude: float, alpha: float = 1.0) -> np.ndarray:
    """Generate colored noise using AR(1) process"""
    white_noise = np.random.normal(0, 1, n_samples)
    colored_noise = np.zeros_like(white_noise)

    for i in range(1, n_samples):
        colored_noise[i] = alpha * colored_noise[i-1] + white_noise[i]

    # Scale to desired amplitude
    return amplitude * colored_noise / np.std(colored_noise)


def _lfsr_sequence(n_bits: int, n_samples: int, seed: int = 1) -> np.ndarray:
    """Maximal-length LFSR sequence of ±1 values.

    Uses well-known primitive polynomials to guarantee period = 2^n_bits - 1.
    Sequences from different seeds differ by their starting register state.

    Args:
        n_bits: LFSR width (4–12). Clamped to nearest available.
        n_samples: Number of output samples (can exceed one full period).
        seed: Initial register state (non-zero; 0 is forced to 1).

    Returns:
        Array of ±1.0 values, shape (n_samples,).
    """
    _PRIMS = {
        4:  0b10011,
        5:  0b100101,
        6:  0b1000011,
        7:  0b10000011,
        8:  0b100011101,
        10: 0b10000001001,
        12: 0b100000101001,
    }
    valid = sorted(_PRIMS.keys())
    n_bits = min(valid, key=lambda x: abs(x - n_bits))
    poly = _PRIMS[n_bits]
    mask = (1 << n_bits) - 1

    state = int(seed) & mask
    if state == 0:
        state = 1

    output = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        output[k] = 1.0 if (state & 1) else -1.0
        feedback = bin(state & poly).count('1') % 2
        state = ((state >> 1) | (feedback << (n_bits - 1))) & mask

    return output


# Registry of available torque generators
TORQUE_GENERATORS = {
    # 'random': random_torque,
    'one step': one_step_torque,
    'multi step': step_torque,
    'sawtooth': sawtooth_torque,
    'sine': sine_torque,
    'multi sine': multi_sine_torque,
    'chirp': chirp_torque,
    'prbs': prbs_torque,
    'sine-3axis': sine_3axis_profile,
}


def list_available_profiles() -> List[str]:
    """List all available torque profiles"""
    return list(TORQUE_GENERATORS.keys())


def get_profile_info(profile: str) -> str:
    """Get information about a specific torque profile"""
    info = {
        'sine': 'Single frequency sinusoidal excitation - good for specific frequency analysis',
        'chirp': 'Frequency sweep - excellent for broadband excitation and frequency response',
        'prbs': 'Pseudo-random binary sequence - optimal for system identification',
        'step': 'Sequential step changes - useful for step response analysis',
        'one_step': 'Single step change - useful for quick response analysis',
        'random': 'Random noise - provides broad spectrum excitation',
        'simple_multi_sine': 'Simple multi-frequency sinusoidal excitation - rich harmonic content',
        'multi_sine': 'Multiple frequency sinusoids - rich harmonic content for identification',
        'sine-3axis': 'Multi-axis sinusoidal excitation - rich frequency content across axes',
    }
    
    return info.get(profile, f"No information available for profile '{profile}'")

# function to generate a tight figure with all profiles
def generate_comprehensive_plots(t: np.ndarray, torques: Dict[str, np.ndarray]) -> None:
    """
    Generate comprehensive plots for all torque profiles

    Args:
        t: Time vector
        torques: Dictionary of torque arrays for each profile
    """
    import matplotlib.pyplot as plt

    # Configure enhanced text formatting
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 13
    })

    # Apply seaborn-darkgrid style
    plt.style.use('ggplot')

    n_profiles = len(torques)
    fig, axs = plt.subplots(n_profiles, 1, figsize=(8, 6), sharex=True)

    for ax, (profile, torque) in zip(axs, torques.items()):
        ax.plot(t, torque, label=profile)
        ax.set_title(f"{profile.capitalize()}", fontsize=14, weight='bold')  
        ax.grid(color='white', linewidth=0.6)

        ax.locator_params(axis='y', nbins=4)
        ax.tick_params(axis='x', labelsize=12)  # enhanced tick labels
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # axs[3].set_ylabel("Torque (Nm)", fontsize=8)
    axs[-1].set_xlabel("Time (s)", fontsize=14, weight='bold')

    plt.subplots_adjust(hspace=0.3)  # adjust spacing between rows

    plt.tight_layout()
    

    plt.rcParams.update(plt.rcParamsDefault)
    
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')

    # pre-set the signals with  fixed parameters for testing
    t = np.linspace(0, 600, 600)  
    print("Testing individual torque generators...")
    params = {
        'sine': {'frequency': 0.01, 'amplitude': 0.01},

        'chirp': {'f0': 0.005, 'f1': 0.05, 'amplitude': 0.01},  
        'prbs': {'amplitude': 0.01, 'switch_time': 20},         
        'one step': {'amplitude': 0.015, 'step_duration': 40.0},
        'multi step': {'amplitude': 0.012, 'step_duration': 40.0},

        'multi sine': {
            'frequencies': [[0.002, 0.005], [0.003, 0.007], [0.004, 0.008]],
            'amplitudes': [[0.01, 0.007], [0.009, 0.006], [0.008, 0.005]], 
            'phases': [[0, 0], [np.pi/4, 0], [-np.pi/3, 0]]
        },

        'sine-3axis': {
            'freqs': [0.01, 0.01, 0.01],
            'amps': [0.03, 0.015, 0.025], 
            'phases': [0.0, 0.0, 0.0]
        }
    }

    all_profiles = list_available_profiles()
    torques = {}

    for profile in all_profiles:
        try:
            torques[profile] = generate_torque_profile(profile, t, axes=['x', 'y', 'z'])
        except Exception as e:
            print(f"Error generating {profile} profile: {e}")

    generate_comprehensive_plots(t, torques)

    scores = score_profiles_canonical(torques, dt=1.0, I_ref=(1.0,1.0,2.0), normalize_energy=True)

