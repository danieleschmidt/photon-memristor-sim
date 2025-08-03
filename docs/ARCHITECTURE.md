# Architecture Overview

## System Design

Photon-Memristor-Sim employs a multi-layered architecture optimized for performance, accuracy, and flexibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Python High-Level APIs  │  JavaScript/WASM Frontend      │
│  - JAX Integration        │  - Web Playground              │
│  - Neural Networks        │  - Real-time Visualization     │
│  - Optimization           │  - Interactive Demos           │
├─────────────────────────────────────────────────────────────┤
│                   Rust Core Engine                         │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │   Physics Core  │   Device Models │  Optimization   │   │
│  │  - Waveguides   │  - PCM Materials│  - Gradients    │   │
│  │  - Propagation  │  - Memristors   │  - Constraints  │   │
│  │  - Coupling     │  - Resonators   │  - Co-design    │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. Physics Core (`src/core/`)

**Waveguide Modeling** (`waveguide.rs`)
- Modal analysis for single and multi-mode guides
- Effective index calculation with material dispersion
- Loss mechanisms (absorption, scattering, bending)
- Nonlinear effects (Kerr, thermal)

**Light Propagation** (`propagation.rs`)
- Beam Propagation Method (BPM) solver
- Finite-Difference Time-Domain (FDTD) for accuracy
- Transfer matrix method for linear devices
- Split-step Fourier for nonlinear propagation

**Optical Coupling** (`coupling.rs`)
- Evanescent coupling between waveguides
- Vertical coupling (grating couplers, edge coupling)
- Mode matching and overlap integrals
- Coupling strength optimization

### 2. Device Models (`src/devices/`)

**Phase Change Materials** (`pcm.rs`)
```rust
pub struct PCMDevice {
    material: PCMMaterial,        // GST, GSST, etc.
    geometry: DeviceGeometry,     // Dimensions and shape
    thermal_model: ThermalSolver, // Heat equation solver
    optical_model: OpticalSolver, // Complex permittivity
}

impl PCMDevice {
    pub fn phase_transition(&mut self, temperature: f64) -> f64 {
        // Implement crystallization kinetics
        // Returns new crystallinity fraction
    }
    
    pub fn optical_constants(&self, wavelength: f64) -> Complex<f64> {
        // Return n + ik based on crystallinity
    }
}
```

**Metal Oxide Memristors** (`oxide.rs`)
```rust
pub struct OxideMemristor {
    oxide_stack: Vec<Layer>,      // HfO2, TaOx, etc.
    electrodes: (Material, Material), // Top/bottom contacts
    ion_dynamics: IonModel,       // Vacancy migration
    electronic_transport: TransportModel, // I-V characteristics
}
```

**Microring Resonators** (`ring_resonator.rs`)
```rust
pub struct MicroringResonator {
    radius: f64,
    waveguide: WaveguideGeometry,
    coupling_gap: f64,
    quality_factor: f64,
    thermal_tuning: ThermalTuner,
}

impl MicroringResonator {
    pub fn transfer_function(&self, wavelengths: &[f64]) -> Vec<Complex<f64>> {
        // Calculate transmission/reflection spectrum
    }
}
```

### 3. Numerical Methods (`src/simulation/`)

**FDTD Solver** (`fdtd.rs`)
- 3D Yee grid implementation
- Perfectly Matched Layer (PML) boundaries
- Dispersive material handling (Debye, Drude, Lorentz)
- Parallel execution with Rayon

**Beam Propagation Method** (`bem.rs`)
- Fast Fourier Transform based
- Adaptive step size control
- Cylindrical coordinate support
- Paraxial and wide-angle variants

**Monte Carlo Methods** (`monte_carlo.rs`)
- Statistical process modeling
- Manufacturing variation analysis
- Thermal noise simulation
- Device-to-device variation

### 4. Optimization Engine (`src/optimization/`)

**Gradient Computation** (`gradient.rs`)
```rust
pub trait Differentiable {
    type Input;
    type Output;
    
    fn forward(&self, input: &Self::Input) -> Self::Output;
    fn backward(&self, grad_output: &Self::Output) -> Self::Input;
}

// Automatic differentiation for device parameters
impl Differentiable for PhotonicDevice {
    fn backward(&self, grad_output: &OpticalField) -> DeviceGradients {
        // Compute gradients w.r.t. device parameters
    }
}
```

**Physical Constraints** (`constraints.rs`)
- Power consumption limits
- Thermal constraints  
- Manufacturing tolerances
- Signal integrity requirements

**Co-design Optimization** (`co_design.rs`)
- Multi-objective optimization (NSGA-II, MOEA/D)
- Simultaneous device and algorithm optimization
- Pareto frontier exploration
- Robustness analysis

## Data Flow Architecture

### Forward Simulation Path
```
Input Optical Signal → Waveguide Model → Device Interactions → 
Nonlinear Effects → Noise Addition → Output Detection
```

### Gradient Computation Path
```
Loss Function → Adjoint Simulation → Device Sensitivities → 
Parameter Gradients → Optimization Update
```

### Python Integration
```python
# Rust core wrapped in Python
import photon_memristor_sim._core as core

# High-level Python APIs
class PhotonicNeuralNetwork:
    def __init__(self, layers):
        self._core = core.PhotonicArray(layers)
        
    def forward(self, inputs):
        # Call into Rust, return NumPy array
        return self._core.simulate(inputs.ctypes.data_as(POINTER(c_double)))
```

### JAX Integration
```python
from jax import custom_vjp
import jax.numpy as jnp

@custom_vjp
def photonic_matmul(inputs, weights):
    # Forward pass through Rust
    return core.matrix_multiply(inputs, weights)

def photonic_matmul_fwd(inputs, weights):
    outputs = photonic_matmul(inputs, weights)
    return outputs, (inputs, weights)

def photonic_matmul_bwd(res, grad_outputs):
    inputs, weights = res
    # Backward pass through Rust adjoint solver
    grad_inputs = core.backward_inputs(grad_outputs, weights)
    grad_weights = core.backward_weights(grad_outputs, inputs)
    return grad_inputs, grad_weights

photonic_matmul.defvjp(photonic_matmul_fwd, photonic_matmul_bwd)
```

## Memory Management

### Rust Core
- Zero-copy data sharing where possible
- Memory pools for frequently allocated objects
- RAII for automatic resource cleanup
- Custom allocators for SIMD-aligned data

### Python Bindings
- PyO3 for safe Rust-Python interface
- NumPy C API for efficient array handling
- Reference counting to prevent memory leaks
- Automatic GIL handling for thread safety

### WASM Frontend
- Linear memory model with manual management
- Shared array buffers for large datasets
- Streaming for real-time visualization
- Worker threads for background computation

## Performance Optimizations

### Computational
- SIMD vectorization (AVX2, NEON)
- Multi-threading with work stealing
- GPU acceleration via compute shaders
- Just-in-time compilation for hot paths

### Memory
- Cache-friendly data layouts
- Memory prefetching for predictable access
- Copy-on-write for large read-only data
- Compression for serialized states

### I/O
- Memory-mapped files for large datasets
- Asynchronous I/O for concurrent operations
- Binary serialization formats (bincode)
- Streaming protocols for real-time data

## Extensibility

### Device Plugin Architecture
```rust
pub trait PhotonicDevice: Send + Sync {
    fn simulate(&self, input: &OpticalField) -> OpticalField;
    fn gradient(&self, grad_output: &OpticalField) -> DeviceGradients;
    fn parameters(&self) -> &[f64];
    fn update_parameters(&mut self, params: &[f64]);
}

// Register new device types
register_device_type::<CustomDevice>("custom_device");
```

### Solver Registration
```rust
pub trait NumericalSolver {
    type State;
    type Parameters;
    
    fn solve(&self, initial: Self::State, params: Self::Parameters) -> Self::State;
}

// Plugin new solvers
register_solver::<CustomFDTD>("custom_fdtd");
```

## Error Handling

### Rust Core
- Comprehensive `Result<T, E>` usage
- Custom error types with context
- Graceful fallbacks for numerical issues
- Detailed error propagation chains

### Python Interface
- Python exception mapping from Rust errors
- Input validation with helpful messages
- Automatic error recovery where possible
- Detailed stack traces for debugging

## Testing Strategy

### Unit Tests
- Property-based testing with Proptest
- Numerical accuracy verification
- Performance regression detection
- Cross-platform compatibility

### Integration Tests
- End-to-end simulation workflows
- Python-Rust interface validation
- WASM functionality verification
- Multi-threading safety checks

### Benchmarks
- Criterion.rs for Rust benchmarking
- pytest-benchmark for Python
- Continuous performance monitoring
- Comparison with reference implementations

## Implementation Status

### Current Implementation (v0.1.0)
- [x] Core Rust simulation engine structure
- [x] Basic device model traits and interfaces
- [x] Python bindings with maturin integration
- [x] JAX custom_vjp integration
- [x] Fundamental physics implementations
- [x] Test infrastructure and examples

### Device Models Implemented
- [x] PCM (Phase Change Material) device with GST
- [x] Basic metal oxide memristor (HfO2)
- [x] Waveguide propagation with modal analysis
- [x] Optical coupling mechanisms

### Simulation Methods Active
- [x] Transfer matrix method for linear devices
- [x] Basic thermal coupling
- [x] Gradient computation for optimization
- [x] Multi-physics device interactions