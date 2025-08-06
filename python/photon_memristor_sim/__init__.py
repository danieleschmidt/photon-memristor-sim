"""
Photon-Memristor-Sim: High-performance neuromorphic photonic simulation

A Rust/WASM simulator for neuromorphic photonic-memristor arrays with 
JAX integration for differentiable device-algorithm co-design.
"""

__version__ = "0.1.0"

# Import core Rust module
try:
    from ._core import *
except ImportError as e:
    raise ImportError(
        "Failed to import Rust core module. "
        "Make sure the package was built with 'maturin develop'. "
        f"Original error: {e}"
    )

# High-level Python APIs
from .neural_networks import PhotonicNeuralNetwork, PhotonicLayer
from .jax_interface import photonic_matmul, photonic_conv2d, create_photonic_primitive
from .devices import PCMDevice, OxideMemristor, MicroringResonator
from .training import HardwareAwareOptimizer, CoDesignOptimizer
from .visualization import PhotonicCircuitVisualizer, FieldVisualizer
from .quantum_planning import (
    QuantumTaskPlanner, 
    PhotonicTaskPlannerFactory,
    TaskAssignment,
    QuantumPlanningReport,
    benchmark_quantum_vs_classical
)

# Utility functions
from .utils import (
    create_gaussian_beam,
    wavelength_to_frequency,
    db_to_linear,
    linear_to_db,
    effective_area,
)

__all__ = [
    # Core classes from Rust
    "PyOpticalField",
    "PyPhotonicArray", 
    "jax_photonic_matmul",
    "jax_photonic_matmul_vjp",
    "calculate_waveguide_mode",
    "create_device_simulator",
    
    # High-level Python APIs
    "PhotonicNeuralNetwork",
    "PhotonicLayer",
    "photonic_matmul",
    "photonic_conv2d",
    "create_photonic_primitive",
    
    # Device models
    "PCMDevice",
    "OxideMemristor", 
    "MicroringResonator",
    
    # Training and optimization
    "HardwareAwareOptimizer",
    "CoDesignOptimizer",
    
    # Visualization
    "PhotonicCircuitVisualizer",
    "FieldVisualizer",
    
    # Quantum-Inspired Planning
    "QuantumTaskPlanner",
    "PhotonicTaskPlannerFactory", 
    "TaskAssignment",
    "QuantumPlanningReport",
    "benchmark_quantum_vs_classical",
    
    # Utilities
    "create_gaussian_beam",
    "wavelength_to_frequency",
    "db_to_linear",
    "linear_to_db",
    "effective_area",
]

# Version and metadata
VERSION = __version__
SPEED_OF_LIGHT = 299792458.0  # m/s