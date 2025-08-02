# ADR-0001: Rust Core with Python Bindings Architecture

## Status
**Status:** Accepted  
**Date:** 2025-01-02  
**Deciders:** Daniel Schmidt, Core Development Team  
**Technical Story:** Choose the optimal language and architecture for high-performance photonic simulation with scientific Python ecosystem integration

## Context and Problem Statement

Photon-Memristor-Sim requires extremely high computational performance for real-time photonic device simulation while maintaining accessibility to the scientific Python ecosystem (NumPy, JAX, SciPy). The simulator must:

- Process large-scale electromagnetic field calculations (millions of data points)
- Support real-time parameter optimization with automatic differentiation
- Integrate seamlessly with existing Python-based neural network frameworks
- Provide WASM compilation for browser-based simulation
- Maintain numerical stability for sensitive optical calculations

Current Python-based photonic simulators suffer from performance bottlenecks that make large-scale optimization impractical. Pure C++ solutions lack the ecosystem integration needed for modern ML workflows.

## Decision Drivers

- **Performance:** 100x speedup target over pure Python implementations
- **Ecosystem Integration:** Seamless JAX/NumPy interoperability for ML workflows
- **Memory Safety:** Eliminate segfaults and memory leaks in long-running simulations
- **Cross-Platform:** Support Linux, macOS, Windows, and WebAssembly
- **Developer Productivity:** Modern tooling and package management
- **Scientific Accuracy:** IEEE 754 compliance and numerical stability
- **Community Adoption:** Accessibility to photonics researchers familiar with Python

## Considered Options

### Option 1: Pure Python with NumPy/JAX
- **Description:** Implement all simulation logic in Python using NumPy for numerical computation
- **Pros:** 
  - Maximum ecosystem compatibility
  - Rapid development and prototyping
  - Familiar to scientific Python community
  - Direct JAX integration
- **Cons:** 
  - Significant performance limitations
  - GIL bottlenecks for multi-threading
  - Memory overhead for large arrays
  - No path to WebAssembly deployment
- **Implementation Effort:** Low

### Option 2: C++ Core with Python Bindings
- **Description:** Implement simulation engine in C++ with pybind11 bindings
- **Pros:** 
  - Maximum performance potential
  - Mature numerical libraries (Eigen, BLAS)
  - Direct memory control
  - Established in scientific computing
- **Cons:** 
  - Memory safety risks (segfaults, leaks)
  - Complex build system and dependency management
  - Slower development iteration
  - Manual memory management overhead
  - Limited WebAssembly support
- **Implementation Effort:** High

### Option 3: Rust Core with Python Bindings
- **Description:** Implement core simulation engine in Rust with PyO3/maturin bindings
- **Pros:** 
  - Memory safety without garbage collection
  - Performance comparable to C++
  - Excellent WebAssembly support
  - Modern tooling (Cargo, formatting, testing)
  - Zero-cost abstractions
  - Growing scientific computing ecosystem
- **Cons:** 
  - Steeper learning curve for team
  - Smaller ecosystem than C++
  - Less mature numerical libraries
  - Potential Python interop overhead
- **Implementation Effort:** Medium-High

### Option 4: Julia with Python Interop
- **Description:** Implement core in Julia with PyCall for Python integration
- **Pros:** 
  - Designed for scientific computing
  - Good performance without manual optimization
  - Native Python interoperability
  - Multiple dispatch paradigm
- **Cons:** 
  - Limited WebAssembly support
  - Smaller community and ecosystem
  - JIT compilation overhead
  - Less mature packaging for Python distribution
- **Implementation Effort:** Medium

## Decision Outcome

**Chosen Option:** Rust Core with Python Bindings (Option 3)

**Rationale:** 

Rust provides the optimal balance of performance, safety, and ecosystem compatibility for our requirements:

1. **Performance:** Rust's zero-cost abstractions and LLVM backend deliver C++-level performance while maintaining memory safety
2. **WebAssembly:** First-class WASM support enables browser-based simulation without porting
3. **Memory Safety:** Eliminates entire classes of bugs that plague long-running scientific simulations
4. **Python Integration:** PyO3/maturin provides seamless Python bindings with minimal overhead
5. **JAX Compatibility:** Custom VJP implementation allows transparent automatic differentiation
6. **Developer Experience:** Cargo provides superior dependency management and tooling
7. **Future-Proof:** Growing adoption in scientific computing (Polars, PyTorch 2.0 components)

The initial learning curve is offset by long-term maintainability benefits and the elimination of memory-related debugging overhead.

## Consequences

### Positive Consequences
- **Performance:** Expected 50-100x speedup over pure Python for numerical kernels
- **Reliability:** Memory safety eliminates segmentation faults and data races
- **Deployment:** Single binary distribution with minimal system dependencies
- **WebAssembly:** Browser-based simulation enables widespread educational access
- **Maintainability:** Strong type system catches errors at compile time
- **Concurrency:** Safe parallel processing without data races

### Negative Consequences
- **Learning Curve:** Team needs to gain Rust expertise, especially for advanced features
- **Ecosystem Maturity:** Some specialized numerical libraries may need custom implementation
- **Compilation Time:** Rust compilation slower than Python development iteration
- **Python Interop:** Potential serialization overhead for large data transfers

### Neutral Consequences
- **Development Timeline:** Medium-term investment for long-term productivity gains
- **Binary Size:** Larger binary than dynamic linking, but acceptable for scientific software
- **Platform Support:** Excellent coverage but requires per-platform compilation

## Implementation Plan

1. **Phase 1 (Weeks 1-2):** Core Rust infrastructure setup
   - Project structure with Cargo.toml configuration
   - PyO3 bindings foundation with basic data types
   - CI/CD pipeline for multi-platform builds
   - Basic numerical operations benchmarking

2. **Phase 2 (Weeks 3-6):** Core simulation engine
   - Waveguide propagation algorithms in Rust
   - Transfer matrix method implementation
   - Memory-efficient array operations
   - Python API design and implementation

3. **Phase 3 (Weeks 7-10):** JAX integration and optimization
   - Custom VJP implementation for automatic differentiation
   - Performance optimization and profiling
   - Comprehensive test suite across Python/Rust boundary
   - Documentation and examples

**Success Criteria:**
- Rust simulation kernels achieve >50x speedup vs. NumPy baseline
- Python API maintains ergonomic scientific workflow
- JAX gradients match finite-difference reference within 1e-6 tolerance
- WebAssembly build compiles and runs in browser
- Zero memory leaks detected in 24-hour stress tests

## Compliance

- [x] Security review completed - Rust's memory safety eliminates major vulnerability classes
- [x] Performance impact assessed - Expected significant improvement over alternatives
- [x] Documentation updated - Architecture documentation reflects Rust core design
- [x] Team training planned - Rust learning path established for core developers
- [x] Migration plan created - Incremental adoption starting with performance-critical components
- [x] Rollback plan defined - Python reference implementation maintained during transition

## Follow-up Actions

- [x] Set up Rust development environment and CI
- [x] Create initial project structure with maturin
- [x] Implement basic PyO3 bindings for core data types
- [ ] Benchmark performance against Python reference implementation
- [ ] Design Python API for ease of use and JAX compatibility
- [ ] Establish WebAssembly build pipeline

---

## Notes

### References
- [PyO3 User Guide](https://pyo3.rs/v0.20.0/)
- [maturin Documentation](https://maturin.rs/)
- [JAX Custom Derivatives](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)

### Related ADRs
- ADR-0002: JAX Integration Strategy (planned)
- ADR-0003: WebAssembly Build Pipeline (planned)

### Revision History
- **v1.0 (2025-01-02):** Initial decision and rationale