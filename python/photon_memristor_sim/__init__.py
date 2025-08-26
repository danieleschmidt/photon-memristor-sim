"""
Photon-Memristor-Sim: High-performance neuromorphic photonic simulation

A Rust/WASM simulator for neuromorphic photonic-memristor arrays with 
JAX integration for differentiable device-algorithm co-design.
"""

__version__ = "0.1.0"

# Import core Rust module with fallbacks
try:
    from ._core import *
    _RUST_CORE_AVAILABLE = True
    print("🦀 Rust core module loaded successfully!")
except ImportError as e:
    print(f"⚠️  Rust core module unavailable: {e}")
    print("🐍 Running in Pure Python mode with fallbacks")
    _RUST_CORE_AVAILABLE = False
    # Import fallbacks
    from .pure_python_fallbacks import (
        PyOpticalField, PyPhotonicArray, 
        jax_photonic_matmul, jax_photonic_matmul_vjp,
        calculate_waveguide_mode, create_device_simulator
    )

# High-level Python APIs
from .advanced_memristor_interface import AdvancedMemristorDevice, MemristorArray, MemristorConfig, create_standard_memristor_configs
from .neural_networks import PhotonicNeuralNetwork, PhotonicLayer
from .jax_interface import photonic_matmul, photonic_conv2d, create_photonic_primitive
from .devices import PCMDevice, MolecularMemristor, OxideMemristor, MicroringResonator
from .training import HardwareAwareOptimizer, CoDesignOptimizer
from .visualization import PhotonicCircuitVisualizer, FieldVisualizer
from .quantum_planning import (
    QuantumTaskPlanner, 
    PhotonicTaskPlannerFactory,
    TaskAssignment,
    QuantumPlanningReport,
    benchmark_quantum_vs_classical
)

# 2025 Breakthrough Enhancement Modules
from .quantum_hybrid import (
    QuantumPhotonicProcessor,
    QuantumState,
    PhotonicQuantumConfig,
    create_quantum_photonic_processor,
    quantum_accelerated_gradient_descent
)
from .gpu_accelerated import (
    CUDAOptimizedFDTD,
    ParallelPhotonicArray,
    RealTimeVisualizer,
    create_gpu_photonic_simulator,
    create_parallel_photonic_array
)
from .edge_computing import (
    EdgeAI,
    EdgeNode,
    EdgeComputingConfig,
    create_edge_ai_system
)
from .ai_optimization import (
    NeuralArchitectureSearch,
    BioInspiredOptimization,
    AdaptivePerformanceOptimizer,
    create_neural_architecture_search,
    create_bio_inspired_optimizer,
    create_adaptive_optimizer
)

# Utility functions
from .utils import (
    create_gaussian_beam,
    wavelength_to_frequency,
    db_to_linear,
    linear_to_db,
    effective_area,
    get_secret,
    load_balancer_config,
    auto_scaling_config,
    circuit_breaker_config,
    metrics_config,
    resource_pooling_config,
)

# Resilience and production utilities
from .resilience import (
    ResilientSystem,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryPolicy,
    HealthCheck,
    BulkheadIsolation,
    CacheManager,
    MetricsCollector,
    get_resilient_system,
    with_circuit_breaker,
    with_retry,
    with_metrics,
)

# Performance optimization utilities  
from .performance_optimizer import (
    OptimizedPhotonic,
    PerformanceProfiler,
    IntelligentCache,
    BatchProcessor,
    AdaptiveScheduler,
    MemoryOptimizer,
    JAXOptimizer,
    get_optimizer,
    optimized_computation,
    profile_performance,
)

__all__ = [
    # Core classes from Rust
    "PyOpticalField",
    "PyPhotonicArray",
    "PhotonicArray",  # Convenience constructor
    "jax_photonic_matmul",
    "jax_photonic_matmul_vjp",
    "calculate_waveguide_mode",
    "create_device_simulator",
    
    # Advanced Memristor Models  
    "AdvancedMemristorDevice",
    "MemristorArray",
    "MemristorConfig", 
    "create_standard_memristor_configs",
    
    # High-level Python APIs
    "PhotonicNeuralNetwork",
    "PhotonicLayer",
    "photonic_matmul",
    "photonic_conv2d",
    "create_photonic_primitive",
    
    # Device models
    "PCMDevice",
    "MolecularMemristor",
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
    
    # 2025 Breakthrough Enhancement Classes
    # Quantum-Photonic Hybrid
    "QuantumPhotonicProcessor",
    "QuantumState",
    "PhotonicQuantumConfig",
    "create_quantum_photonic_processor",
    "quantum_accelerated_gradient_descent",
    
    # GPU-Accelerated Computing
    "CUDAOptimizedFDTD",
    "ParallelPhotonicArray",
    "RealTimeVisualizer",
    "create_gpu_photonic_simulator",
    "create_parallel_photonic_array",
    
    # Edge Computing
    "EdgeAI",
    "EdgeNode",
    "EdgeComputingConfig",
    "create_edge_ai_system",
    
    # AI-Driven Optimization
    "NeuralArchitectureSearch",
    "BioInspiredOptimization",
    "AdaptivePerformanceOptimizer",
    "create_neural_architecture_search",
    "create_bio_inspired_optimizer",
    "create_adaptive_optimizer",
    
    # Utilities
    "create_gaussian_beam",
    "wavelength_to_frequency",
    "db_to_linear",
    "linear_to_db",
    "effective_area",
]

# Convenience constructors
def PhotonicArray(rows, cols, topology="crossbar"):
    """Create a photonic array with simplified interface"""
    if _RUST_CORE_AVAILABLE:
        return PyPhotonicArray(topology, rows, cols)
    else:
        return PyPhotonicArray(rows, cols)

# Version and metadata
VERSION = __version__
SPEED_OF_LIGHT = 299792458.0  # m/s