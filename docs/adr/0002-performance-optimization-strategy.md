# ADR-0002: Performance Optimization Strategy for Photonic Simulation

## Status

Accepted

## Context

Photon-Memristor-Sim aims to achieve 100x speedup over existing Python-based photonic simulators while maintaining accuracy. The core challenges are:

1. **Computational Complexity**: Photonic device simulation involves solving Maxwell's equations with complex boundary conditions
2. **Memory Requirements**: Large photonic arrays require significant memory for field storage and device states  
3. **Python Interface Overhead**: Frequent Rust-Python boundary crossings can create bottlenecks
4. **Gradient Computation**: Automatic differentiation for optimization requires careful memory management

## Decision

We adopt a multi-layered performance optimization strategy:

### 1. Rust Core Optimizations
- **SIMD Vectorization**: Use `std::simd` for optical field operations
- **Memory Layout**: Struct-of-Arrays (SoA) for device parameters
- **Parallel Processing**: Rayon for device-level parallelism
- **Zero-Copy Operations**: Minimize allocations in hot paths

### 2. Python Interface Design
- **Batch Operations**: Process multiple inputs simultaneously
- **Memory Mapping**: Share memory between Rust and Python via PyO3
- **JAX Integration**: Custom primitives for gradient computation
- **Async Support**: Non-blocking operations for large simulations

### 3. Algorithm Selection
- **Adaptive Methods**: Choose solver based on problem characteristics
- **Hierarchical Modeling**: Multiple fidelity levels for different use cases
- **Caching**: Memoize expensive computations (mode profiles, coupling matrices)
- **Approximations**: Fast analytical models where appropriate

### 4. Memory Management
- **Object Pools**: Reuse OpticalField and Device objects
- **Streaming**: Process large datasets without full memory loading
- **Compression**: Compressed storage for simulation states
- **Memory Mapping**: Use mmap for large read-only datasets

## Implementation Strategy

### Phase 1: Core Optimizations (Current)
```rust
// SIMD-optimized field operations
use std::simd::f64x8;

impl OpticalField {
    pub fn add_simd(&self, other: &OpticalField) -> OpticalField {
        let self_chunks = self.amplitude.chunks_exact(8);
        let other_chunks = other.amplitude.chunks_exact(8);
        
        let result: Vec<f64> = self_chunks
            .zip(other_chunks)
            .flat_map(|(a, b)| {
                let va = f64x8::from_slice(a);
                let vb = f64x8::from_slice(b);
                (va + vb).to_array()
            })
            .collect();
            
        OpticalField::from_amplitude(result)
    }
}
```

### Phase 2: Parallel Device Simulation
```rust
use rayon::prelude::*;

impl PhotonicArray {
    pub fn forward_parallel(&self, inputs: &[OpticalField]) -> Vec<OpticalField> {
        self.devices
            .par_iter()
            .zip(inputs.par_iter())
            .map(|(device, input)| device.simulate(input))
            .collect()
    }
}
```

### Phase 3: Adaptive Solver Selection
```rust
pub enum SolverType {
    FastAnalytical,    // For simple devices
    TransferMatrix,    // For linear systems
    BeamPropagation,   // For guided wave problems
    FDTD,             // For complex structures
}

impl PhotonicDevice {
    fn choose_solver(&self, accuracy_requirement: f64) -> SolverType {
        match (self.complexity(), accuracy_requirement) {
            (Low, _) => SolverType::FastAnalytical,
            (Medium, req) if req < 0.01 => SolverType::TransferMatrix,
            (Medium, _) => SolverType::BeamPropagation,
            (High, _) => SolverType::FDTD,
        }
    }
}
```

## Performance Targets

### Simulation Speed
- **Small Arrays** (10x10): < 1ms per forward pass
- **Medium Arrays** (100x100): < 100ms per forward pass  
- **Large Arrays** (1000x1000): < 10s per forward pass
- **Gradient Computation**: < 2x forward pass time

### Memory Usage
- **Field Storage**: < 1GB for 1000x1000 arrays
- **Device Parameters**: Compressed storage < 100MB
- **Python Interface**: Zero-copy where possible
- **Streaming**: Process arrays larger than available RAM

### Scalability
- **Multi-threading**: Linear scaling to 16 cores
- **Memory Bandwidth**: 80% of theoretical peak
- **Cache Efficiency**: > 95% L1 cache hit rate for hot loops
- **GPU Acceleration**: 10x speedup for large problems (future)

## Validation Strategy

### Benchmarking Framework
```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_simulation_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("photonic_simulation");
    
    for &size in &[10, 50, 100, 500, 1000] {
        group.bench_function(
            format!("array_{}x{}", size, size),
            |b| {
                let array = PhotonicArray::new(size, size);
                let inputs = create_random_inputs(size);
                b.iter(|| array.forward(&inputs))
            }
        );
    }
}
```

### Memory Profiling
- **Valgrind**: Memory leak detection and usage analysis
- **Heaptrack**: Memory allocation profiling
- **Custom Metrics**: Track peak memory usage during simulation

### Accuracy Validation
- **Numerical Precision**: Compare against high-precision reference
- **Physical Conservation**: Energy, momentum conservation checks
- **Experimental Validation**: Compare with measured device data

## Risks and Mitigations

### Risk: SIMD Compatibility
**Mitigation**: Fallback to scalar operations on unsupported platforms

### Risk: Memory Fragmentation
**Mitigation**: Custom allocators and object pools

### Risk: Python GIL Contention
**Mitigation**: Release GIL during compute-intensive operations

### Risk: Gradient Accuracy Loss
**Mitigation**: Extensive testing against analytical gradients

## Success Metrics

### Primary Metrics
- **Speed**: 100x faster than baseline Python implementation
- **Accuracy**: < 1% error vs reference implementation
- **Memory**: Process 10x larger problems than baseline
- **Scalability**: Linear scaling to available cores

### Secondary Metrics
- **Development Velocity**: New device models implementable in < 1 week
- **User Experience**: < 10 second setup time for new users
- **Community Adoption**: Used by 10+ research groups within 6 months

## References

- [High-Performance Computing for Photonic Simulation](https://example.com/hpc-photonics)
- [SIMD Optimization in Rust](https://doc.rust-lang.org/std/simd/)
- [JAX Custom Primitives Guide](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
- [Rayon Parallel Iterator Documentation](https://docs.rs/rayon/)

## Changelog

- **2025-01-XX**: Initial version - Performance strategy definition
- **Future**: Updates based on implementation experience and benchmarking results