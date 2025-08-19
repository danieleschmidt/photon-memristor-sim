"""
Utility functions for photonic simulation

Common helper functions for optical field manipulation,
unit conversions, and visualization support.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional
try:
    from ._core import SPEED_OF_LIGHT
except ImportError:
    SPEED_OF_LIGHT = 299792458.0  # m/s


def create_gaussian_beam(
    nx: int, ny: int, 
    beam_waist: float, 
    wavelength: float = 1550e-9,
    power: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Gaussian beam profile.
    
    Args:
        nx, ny: Grid dimensions
        beam_waist: Beam waist radius (m)
        wavelength: Optical wavelength (m)
        power: Total optical power (W)
        
    Returns:
        (real_amplitude, imag_amplitude) arrays
    """
    width_x = 5.0 * beam_waist
    width_y = 5.0 * beam_waist
    
    dx = width_x / nx
    dy = width_y / ny
    
    x = np.linspace(-width_x/2, width_x/2, nx)
    y = np.linspace(-width_y/2, width_y/2, ny)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian profile
    r_squared = X**2 + Y**2
    amplitude = np.exp(-2 * r_squared / beam_waist**2)
    
    # Normalize to specified power
    total_intensity = np.sum(amplitude**2) * dx * dy
    norm_factor = np.sqrt(power / total_intensity)
    amplitude *= norm_factor
    
    # Return real and imaginary parts (pure real for Gaussian)
    real_amplitude = amplitude
    imag_amplitude = np.zeros_like(amplitude)
    
    return real_amplitude, imag_amplitude


def wavelength_to_frequency(wavelength: float) -> float:
    """Convert wavelength to frequency."""
    return SPEED_OF_LIGHT / wavelength


def frequency_to_wavelength(frequency: float) -> float:
    """Convert frequency to wavelength."""
    return SPEED_OF_LIGHT / frequency


def db_to_linear(db: float) -> float:
    """Convert dB to linear scale."""
    return 10.0 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    """Convert linear to dB scale."""
    return 10.0 * np.log10(linear)


def effective_area(intensity_profile: np.ndarray, dx: float, dy: float) -> float:
    """
    Calculate effective area of optical mode.
    
    Args:
        intensity_profile: 2D intensity distribution
        dx, dy: Grid spacing
        
    Returns:
        Effective area (m²)
    """
    if np.max(intensity_profile) == 0:
        return 0.0
        
    total_power = np.sum(intensity_profile) * dx * dy
    peak_intensity = np.max(intensity_profile)
    
    return total_power**2 / (peak_intensity * dx * dy)


def calculate_fsr(ring_radius: float, n_eff: float, wavelength: float) -> float:
    """
    Calculate free spectral range of ring resonator.
    
    Args:
        ring_radius: Ring radius (m)
        n_eff: Effective refractive index
        wavelength: Wavelength (m)
        
    Returns:
        Free spectral range (m)
    """
    circumference = 2 * np.pi * ring_radius
    n_g = n_eff  # Approximation: group index ≈ effective index
    
    return wavelength**2 / (n_g * circumference)


def thermal_noise_power(temperature: float, bandwidth: float) -> float:
    """
    Calculate thermal noise power.
    
    Args:
        temperature: Temperature (K)
        bandwidth: Bandwidth (Hz)
        
    Returns:
        Noise power (W)
    """
    k_B = 1.380649e-23  # Boltzmann constant
    return k_B * temperature * bandwidth


def shot_noise_current(optical_power: float, wavelength: float, bandwidth: float) -> float:
    """
    Calculate shot noise current.
    
    Args:
        optical_power: Optical power (W)
        wavelength: Wavelength (m)
        bandwidth: Bandwidth (Hz)
        
    Returns:
        RMS shot noise current (A)
    """
    h = 6.62607015e-34  # Planck constant
    c = SPEED_OF_LIGHT
    e = 1.602176634e-19  # Electron charge
    
    photon_energy = h * c / wavelength
    photon_rate = optical_power / photon_energy
    
    return np.sqrt(2 * e * photon_rate * bandwidth)


def coupling_efficiency(mode1_profile: np.ndarray, mode2_profile: np.ndarray,
                       dx: float, dy: float) -> float:
    """
    Calculate coupling efficiency between two modes.
    
    Args:
        mode1_profile, mode2_profile: Mode intensity profiles
        dx, dy: Grid spacing
        
    Returns:
        Coupling efficiency (0-1)
    """
    if mode1_profile.shape != mode2_profile.shape:
        raise ValueError("Mode profiles must have same shape")
    
    # Normalize modes
    power1 = np.sum(mode1_profile) * dx * dy
    power2 = np.sum(mode2_profile) * dx * dy
    
    if power1 == 0 or power2 == 0:
        return 0.0
    
    norm1 = np.sqrt(mode1_profile / power1)
    norm2 = np.sqrt(mode2_profile / power2)
    
    # Overlap integral
    overlap = np.sum(norm1 * norm2) * dx * dy
    
    return overlap**2


def apply_phase_mask(field_real: np.ndarray, field_imag: np.ndarray, 
                    phase_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply phase mask to optical field.
    
    Args:
        field_real, field_imag: Real and imaginary field components
        phase_mask: Phase mask (radians)
        
    Returns:
        Modified (real, imaginary) field components
    """
    if field_real.shape != phase_mask.shape:
        raise ValueError("Field and phase mask must have same shape")
    
    # Convert to complex, apply phase, convert back
    field_complex = field_real + 1j * field_imag
    phase_factor = np.exp(1j * phase_mask)
    field_complex *= phase_factor
    
    return field_complex.real, field_complex.imag


def calculate_q_factor(resonance_wavelength: float, linewidth: float) -> float:
    """
    Calculate quality factor from resonance parameters.
    
    Args:
        resonance_wavelength: Resonance wavelength (m)
        linewidth: Full width at half maximum (m)
        
    Returns:
        Quality factor
    """
    return resonance_wavelength / linewidth


def finesse_to_q(finesse: float, fsr: float, wavelength: float) -> float:
    """
    Convert finesse to Q factor.
    
    Args:
        finesse: Cavity finesse
        fsr: Free spectral range (m)
        wavelength: Wavelength (m)
        
    Returns:
        Quality factor
    """
    return finesse * wavelength / fsr


def power_to_dbm(power_watts: float) -> float:
    """Convert power in watts to dBm."""
    return 10 * np.log10(power_watts * 1000)  # 1000 to convert W to mW


def dbm_to_power(dbm: float) -> float:
    """Convert power in dBm to watts."""
    return 10**(dbm / 10) / 1000  # Divide by 1000 to convert mW to W


def calculate_extinction_ratio(p_high: float, p_low: float) -> float:
    """
    Calculate extinction ratio in dB.
    
    Args:
        p_high: High state power
        p_low: Low state power
        
    Returns:
        Extinction ratio (dB)
    """
    if p_low == 0:
        return float('inf')
    return linear_to_db(p_high / p_low)


def estimate_bandwidth(impulse_response: np.ndarray, dt: float) -> float:
    """
    Estimate 3dB bandwidth from impulse response.
    
    Args:
        impulse_response: Time-domain impulse response
        dt: Time step
        
    Returns:
        3dB bandwidth (Hz)
    """
    # FFT to get frequency response
    freq_response = np.fft.fft(impulse_response)
    power_response = np.abs(freq_response)**2
    
    # Find 3dB point
    max_power = np.max(power_response)
    half_power = max_power / 2
    
    # Find frequency where power drops to half
    freqs = np.fft.fftfreq(len(impulse_response), dt)
    positive_freqs = freqs[:len(freqs)//2]
    positive_power = power_response[:len(power_response)//2]
    
    # Find first crossing of half-power point
    idx = np.where(positive_power < half_power)[0]
    if len(idx) > 0:
        return positive_freqs[idx[0]]
    else:
        return 1.0 / dt  # Nyquist frequency if no crossing found


# Matrix operations for photonic computing
def optical_matrix_multiply(input_vector: np.ndarray, weight_matrix: np.ndarray,
                          include_noise: bool = False, snr_db: float = 40) -> np.ndarray:
    """
    Simulate optical matrix multiplication with realistic effects.
    
    Args:
        input_vector: Input optical powers
        weight_matrix: Weight matrix (device transmissions)
        include_noise: Whether to add optical noise
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Output vector
    """
    # Basic matrix multiplication
    output = np.dot(weight_matrix, input_vector)
    
    if include_noise:
        # Add noise based on SNR
        signal_power = np.mean(output**2)
        noise_power = signal_power / db_to_linear(snr_db)
        noise = np.random.normal(0, np.sqrt(noise_power), output.shape)
        output += noise
        
        # Ensure non-negative (optical powers)
        output = np.maximum(output, 0)
    
    return output


# Scalability and deployment configuration utilities
CONFIG_SECRETS = {
    "jwt_secret": "${JWT_SECRET}",  # Environment variable
    "db_password": "${DB_PASSWORD}",
    "api_key": "${API_KEY}",
    "encryption_key": "${ENCRYPTION_KEY}", 
    "webhook_secret": "${WEBHOOK_SECRET}"
}

def get_secret(secret_name: str) -> str:
    """Safely retrieve secrets from environment variables."""
    import os
    env_var = CONFIG_SECRETS.get(secret_name, "")
    if env_var.startswith("${") and env_var.endswith("}"):
        env_name = env_var[2:-1]
        return os.environ.get(env_name, "")
    return env_var

def load_balancer_config():
    """Load balancer configuration for distributed deployment."""
    return {
        "strategy": "round_robin",
        "health_check_interval": 30,
        "max_retries": 3,
        "timeout": 10
    }

def auto_scaling_config():
    """Auto-scaling configuration."""
    return {
        "min_instances": 2,
        "max_instances": 100,
        "cpu_threshold": 80,
        "memory_threshold": 85,
        "scale_up_cooldown": 300,
        "scale_down_cooldown": 600
    }

def circuit_breaker_config():
    """Circuit breaker configuration for resilience."""
    return {
        "failure_threshold": 5,
        "recovery_timeout": 60,
        "expected_recovery_time": 30
    }

def metrics_config():
    """Metrics and monitoring configuration."""
    return {
        "enabled": True,
        "interval": 10,
        "retention_days": 30,
        "alert_thresholds": {
            "error_rate": 0.05,
            "response_time": 1000,
            "cpu_usage": 80
        }
    }

def resource_pooling_config():
    """Resource pooling for efficiency."""
    return {
        "connection_pool_size": 20,
        "thread_pool_size": 10,
        "memory_pool_size": 1024 * 1024 * 100  # 100MB
    }

__all__ = [
    'create_gaussian_beam',
    'wavelength_to_frequency',
    'frequency_to_wavelength', 
    'db_to_linear',
    'linear_to_db',
    'effective_area',
    'calculate_fsr',
    'thermal_noise_power',
    'shot_noise_current',
    'coupling_efficiency',
    'apply_phase_mask',
    'calculate_q_factor',
    'finesse_to_q',
    'power_to_dbm',
    'dbm_to_power',
    'calculate_extinction_ratio',
    'estimate_bandwidth',
    'optical_matrix_multiply',
    'get_secret',
    'load_balancer_config',
    'auto_scaling_config',
    'circuit_breaker_config',
    'metrics_config',
    'resource_pooling_config',
]