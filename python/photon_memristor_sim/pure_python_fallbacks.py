"""
Pure Python fallbacks for when Rust core is unavailable
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any

class PyOpticalField:
    """Pure Python fallback for OpticalField"""
    def __init__(self, amplitude=None, wavelength=1550e-9, power=1e-3):
        self.amplitude = amplitude or np.ones((64, 64), dtype=complex)
        self.wavelength = wavelength
        self.power = power
    
    def calculate_power(self) -> float:
        return np.sum(np.abs(self.amplitude)**2).real

class PyPhotonicArray:
    """Pure Python fallback for PhotonicArray"""
    def __init__(self, rows: int = 64, cols: int = 64):
        self.rows = rows
        self.cols = cols
        self.weights = np.random.random((rows, cols)) * 0.1
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Simplified forward pass"""
        if len(inputs) != self.cols:
            raise ValueError(f"Input size {len(inputs)} doesn't match array cols {self.cols}")
        
        # Simple matrix multiplication with noise
        outputs = np.dot(self.weights, inputs)
        # Add some noise to simulate optical effects
        noise = np.random.normal(0, 0.01, outputs.shape)
        return outputs + noise
    
    def set_weights(self, weights: np.ndarray):
        """Set device weights"""
        if weights.shape != (self.rows, self.cols):
            raise ValueError(f"Weight shape {weights.shape} doesn't match array shape {(self.rows, self.cols)}")
        self.weights = weights.copy()

def jax_photonic_matmul(*args, **kwargs):
    """Pure Python fallback for JAX photonic operations"""
    raise NotImplementedError("Rust core not available - pure Python JAX primitives not implemented")

def jax_photonic_matmul_vjp(*args, **kwargs):
    """Pure Python fallback for JAX VJP"""
    raise NotImplementedError("Rust core not available - pure Python JAX primitives not implemented")

def calculate_waveguide_mode(*args, **kwargs):
    """Pure Python waveguide mode calculation"""
    return {"effective_index": 2.4 + 0.1j, "mode_profile": np.ones((32, 32))}

def create_device_simulator(*args, **kwargs):
    """Pure Python device simulator"""
    return PyPhotonicArray()