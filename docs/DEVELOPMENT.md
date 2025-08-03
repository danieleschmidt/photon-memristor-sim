# Development Guide

## Prerequisites

- **Rust** (latest stable) - Install via [rustup.rs](https://rustup.rs/)
- **Python** 3.8+ with pip
- **Node.js** 16+ (for WASM tooling)
- **Git** with SSH keys configured

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/photon-memristor-sim
cd photon-memristor-sim

# Install development dependencies
pip install -e ".[dev]"
cargo install wasm-pack
cargo install maturin

# Build Rust core
maturin develop --release

# Run tests
cargo test
pytest python/tests/

# Build WASM package
wasm-pack build --target web
```

## Architecture Overview

```
src/                    # Rust core simulation engine
├── lib.rs             # Main library entry point
├── core/              # Core physics simulation modules
├── devices/           # Device-specific implementations
├── simulation/        # Numerical methods
└── optimization/      # Gradient computation

python/                # Python bindings and high-level APIs
├── photon_memristor_sim/
│   ├── __init__.py   # Main Python module
│   ├── jax_interface.py    # JAX integration
│   ├── neural_networks.py  # NN architectures
│   └── visualization.py    # 3D rendering

wasm/                  # WebAssembly frontend
└── pkg/              # Generated WASM package
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes to Rust core
# Edit src/*.rs files

# Rebuild Python module
maturin develop

# Test your changes
cargo test
pytest python/tests/test_your_feature.py

# Update documentation if needed
```

### 2. Testing Strategy

```bash
# Unit tests (Rust)
cargo test --lib

# Integration tests (Rust)
cargo test --test integration

# Python tests
pytest python/tests/ -v

# Benchmarks
cargo bench

# WASM tests
wasm-pack test --node
```

### 3. Performance Profiling

```bash
# Profile Rust code
cargo install flamegraph
cargo flamegraph --bin your_binary

# Profile Python bindings
python -m cProfile -o profile.prof your_script.py
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

# Benchmark against baseline
cargo bench --bench simulation_benchmark
```

## Code Style

### Rust
- Follow `rustfmt` defaults
- Use `clippy` for linting
- Document public APIs with `///` comments
- Prefer `Result<T, E>` over panics

### Python  
- Follow Black formatting (88 character line limit)
- Use type hints for all public functions
- Follow Google docstring format
- Import sorting with isort

### Commit Messages
```
type(scope): brief description

Longer explanation if needed.

Closes #123
```

Types: feat, fix, docs, style, refactor, test, chore

## Building for Different Targets

### Development Build
```bash
maturin develop  # Debug build with Python bindings
```

### Release Build
```bash
maturin build --release  # Optimized wheel
```

### WASM Build
```bash
wasm-pack build --target web --release
```

### Cross-compilation
```bash
# Linux ARM64
cargo build --target aarch64-unknown-linux-gnu

# Windows
cargo build --target x86_64-pc-windows-gnu
```

## Debugging

### Rust Core
```bash
# Enable debug logs
RUST_LOG=debug cargo test

# GDB debugging
rust-gdb target/debug/your_binary

# Valgrind memory checking
cargo install cargo-valgrind
cargo valgrind test
```

### Python Bindings
```bash
# Debug Python extension
python -X dev your_script.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py
```

## Documentation

### API Documentation
```bash
# Generate Rust docs
cargo doc --open

# Generate Python docs
sphinx-build -b html docs/ docs/_build/html/
```

### Examples
- Add working examples to `examples/`
- Include performance comparisons
- Provide Jupyter notebooks for complex workflows

## Performance Guidelines

### Rust Core
- Use `#[inline]` for hot path functions
- Prefer stack allocation over heap when possible
- Use SIMD when applicable (`nalgebra` provides this)
- Profile before optimizing

### Python Interface
- Minimize Python <-> Rust boundary crossings
- Batch operations when possible
- Use NumPy arrays for large data transfers
- Consider JAX `jit` compilation for pure Python code

## Troubleshooting

### Common Issues

**Build fails with linker errors**
```bash
# Install required system dependencies
sudo apt-get install build-essential  # Ubuntu/Debian
# or
brew install gcc  # macOS
```

**Python import fails**
```bash
# Ensure maturin develop was run
maturin develop --release

# Check Python path
python -c "import sys; print(sys.path)"
```

**WASM build fails**
```bash
# Ensure wasm-pack is installed
cargo install wasm-pack

# Check wasm target
rustup target add wasm32-unknown-unknown
```

### Getting Help

- Check existing [GitHub issues](https://github.com/yourusername/photon-memristor-sim/issues)
- Join our [Discord community](https://discord.gg/photonic-sim)
- Read the [documentation](https://photon-memristor-sim.readthedocs.io)

## Advanced Development Topics

### Multi-threading and Parallelization
```rust
use rayon::prelude::*;

// Parallel device simulation
devices.par_iter_mut()
    .for_each(|device| device.simulate(&input_field));

// Parallel waveguide array processing
waveguides.par_chunks_mut(chunk_size)
    .enumerate()
    .for_each(|(chunk_idx, chunk)| {
        process_waveguide_chunk(chunk, chunk_idx);
    });
```

### SIMD Optimization
```rust
use std::simd::f64x8;

fn vectorized_field_addition(field1: &[f64], field2: &[f64]) -> Vec<f64> {
    let chunks1 = field1.chunks_exact(8);
    let chunks2 = field2.chunks_exact(8);
    
    chunks1.zip(chunks2)
        .map(|(c1, c2)| {
            let v1 = f64x8::from_slice(c1);
            let v2 = f64x8::from_slice(c2);
            (v1 + v2).to_array()
        })
        .flatten()
        .collect()
}
```

### Custom JAX Primitives
```python
from jax import custom_vjp
import jax.numpy as jnp
from photon_memristor_sim._core import photonic_forward, photonic_backward

@custom_vjp
def photonic_layer(inputs, weights, device_params):
    return photonic_forward(inputs, weights, device_params)

def photonic_layer_fwd(inputs, weights, device_params):
    outputs = photonic_layer(inputs, weights, device_params)
    return outputs, (inputs, weights, device_params)

def photonic_layer_bwd(res, grad_outputs):
    inputs, weights, device_params = res
    grad_inputs, grad_weights = photonic_backward(
        grad_outputs, inputs, weights, device_params
    )
    return grad_inputs, grad_weights, None

photonic_layer.defvjp(photonic_layer_fwd, photonic_layer_bwd)
```

### GPU Acceleration (Experimental)
```rust
#[cfg(feature = "gpu")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "gpu")]
impl PhotonicArray {
    fn simulate_gpu(&self, inputs: &[OpticalField]) -> Result<Vec<OpticalField>> {
        let device = CudaDevice::new(0)?;
        // Transfer data to GPU
        // Run CUDA kernels
        // Transfer results back
    }
}
```

### WebAssembly Optimization
```rust
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmPhotonicSimulator {
    core: PhotonicArray,
}

#[wasm_bindgen]
impl WasmPhotonicSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize) -> Self {
        // Initialize with optimal settings for WASM
        Self {
            core: PhotonicArray::new_optimized_for_wasm(rows, cols),
        }
    }
    
    #[wasm_bindgen]
    pub fn simulate_streaming(&mut self, input_ptr: *const f64, output_ptr: *mut f64) {
        // Zero-copy simulation for real-time applications
    }
}
```

## Physics Implementation Guidelines

### Device Model Template
```rust
use crate::core::{OpticalField, DeviceGeometry, PhotonicError};
use crate::devices::PhotonicDevice;

pub struct CustomDevice {
    geometry: DeviceGeometry,
    material_params: MaterialParameters,
    state: DeviceState,
}

impl PhotonicDevice for CustomDevice {
    fn simulate(&self, input: &OpticalField) -> Result<OpticalField, PhotonicError> {
        // 1. Validate input parameters
        self.validate_input(input)?;
        
        // 2. Apply physics model
        let processed_field = self.apply_physics_model(input)?;
        
        // 3. Add noise and variations
        let noisy_field = self.add_realistic_noise(processed_field)?;
        
        // 4. Return result
        Ok(noisy_field)
    }
    
    fn gradient(&self, grad_output: &OpticalField) -> Result<DeviceGradients, PhotonicError> {
        // Implement adjoint method for gradient computation
        self.compute_adjoint_gradients(grad_output)
    }
}
```

### Numerical Solver Interface
```rust
pub trait NumericalSolver: Send + Sync {
    type State;
    type Parameters;
    type Error;
    
    fn solve_step(&mut self, 
                  state: &Self::State, 
                  dt: f64, 
                  params: &Self::Parameters) -> Result<Self::State, Self::Error>;
    
    fn adaptive_step_size(&self, error_estimate: f64) -> f64;
    fn convergence_check(&self, state: &Self::State, prev_state: &Self::State) -> bool;
}
```

## Testing Best Practices

### Property-Based Testing
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_energy_conservation(
        input_power in 0.001..0.1,
        wavelength in 1200e-9..1700e-9,
        device_config in arbitrary_device_config()
    ) {
        let device = create_device(device_config);
        let input_field = OpticalField::new(input_power, wavelength);
        let output_field = device.simulate(&input_field).unwrap();
        
        // Energy should be conserved (within losses)
        let energy_ratio = output_field.total_energy() / input_field.total_energy();
        prop_assert!(energy_ratio <= 1.0);
        prop_assert!(energy_ratio >= 0.1); // Reasonable loss bound
    }
}
```

### Benchmark-Driven Development
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_array_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_simulation");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("photonic_array", size),
            size,
            |b, &size| {
                let array = PhotonicArray::new(size, size);
                let inputs = create_test_inputs(size);
                b.iter(|| array.forward(black_box(&inputs)))
            },
        );
    }
    group.finish();
}
```

### Experimental Validation Framework
```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExperimentalData:
    measurement: np.ndarray
    uncertainty: np.ndarray
    conditions: Dict[str, Any]
    metadata: Dict[str, str]

class ValidationFramework:
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self.experimental_data: Dict[str, ExperimentalData] = {}
    
    def register_experiment(self, name: str, data: ExperimentalData):
        self.experimental_data[name] = data
    
    def validate_simulation(self, name: str, simulation_result: np.ndarray) -> bool:
        exp_data = self.experimental_data[name]
        
        # Statistical comparison with uncertainties
        chi_squared = np.sum(
            ((simulation_result - exp_data.measurement) ** 2) / 
            (exp_data.uncertainty ** 2)
        )
        
        degrees_of_freedom = len(simulation_result) - 1
        normalized_chi_squared = chi_squared / degrees_of_freedom
        
        return normalized_chi_squared < (1 + self.tolerance)
```

## Continuous Integration Setup

### GitHub Actions Workflow
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run tests
        run: cargo test --all-features
      - name: Run clippy
        run: cargo clippy -- -D warnings

  test-python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install maturin pytest
          maturin develop
      - name: Run Python tests
        run: pytest python/tests/

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: cargo bench
      - name: Upload benchmark results
        uses: benchmark-action/github-action-benchmark@v1
```

## Release Process

### Automated Release Pipeline
1. **Version Management**: Use `cargo-release` for version bumping
2. **Changelog Generation**: Automated from commit messages
3. **Multi-platform Builds**: Cross-compilation for Linux, macOS, Windows
4. **PyPI Publishing**: Automated wheel uploads
5. **Documentation**: Auto-generated and deployed
6. **Docker Images**: Multi-arch container builds

### Semantic Versioning
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, performance improvements

```bash
# Release workflow
cargo release patch  # or minor/major
git push --tags
# CI will handle the rest
```