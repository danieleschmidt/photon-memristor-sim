# Contributing to Photon-Memristor-Sim

Thank you for your interest in contributing! This project aims to advance neuromorphic photonic computing through high-performance simulation tools.

## Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Make** your changes
5. **Test** thoroughly: `cargo test && pytest`
6. **Commit** with clear messages
7. **Push** to your fork: `git push origin feature/amazing-feature`
8. **Submit** a Pull Request

## Development Setup

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed setup instructions.

### Prerequisites
- Rust (latest stable)
- Python 3.8+
- Node.js 16+ (for WASM)

### Quick Setup
```bash
git clone https://github.com/yourusername/photon-memristor-sim
cd photon-memristor-sim
pip install -e ".[dev]"
maturin develop --release
```

## Contribution Areas

### 🔬 Physics & Modeling
- New device models (memristors, modulators, detectors)
- Advanced material models (nonlinear, quantum effects)
- Improved numerical solvers (FDTD, BPM, FEM)
- Experimental validation and calibration

### ⚡ Performance & Optimization
- SIMD vectorization and GPU acceleration
- Memory optimization and cache efficiency  
- Parallel algorithms and distributed computing
- Benchmark improvements and profiling

### 🐍 Python Ecosystem
- JAX integration enhancements
- New neural network architectures
- Visualization and plotting improvements
- Jupyter notebook examples

### 🌐 WebAssembly & Frontend
- Browser-based simulation tools
- Interactive demos and tutorials
- Real-time visualization improvements
- Mobile device optimization

### 📚 Documentation & Education
- API documentation improvements
- Tutorial creation and examples
- Educational content for photonics
- Scientific paper reproductions

## Code Standards

### Rust
- **Formatting**: Use `cargo fmt` (rustfmt defaults)
- **Linting**: Pass `cargo clippy` without warnings
- **Documentation**: Document all public APIs with `///`
- **Testing**: Include unit tests for new functionality
- **Error Handling**: Use `Result<T, E>`, avoid panics in library code

### Python
- **Formatting**: Use Black with 88-character line limit
- **Imports**: Sort with isort (profile = "black")
- **Type Hints**: Include for all public functions
- **Documentation**: Use Google-style docstrings
- **Testing**: pytest with >80% coverage

### Commit Messages
```
type(scope): brief description

Longer explanation if needed, wrapped to 72 characters.
Explain the problem this solves and how.

Fixes #123
```

**Types**: feat, fix, docs, style, refactor, test, chore, perf

## Pull Request Process

### Before Submitting
- [ ] Code passes all tests: `cargo test && pytest`
- [ ] Code passes linting: `cargo clippy && black . && mypy`
- [ ] Documentation is updated if needed
- [ ] New functionality includes tests
- [ ] Performance benchmarks included for optimization PRs

### PR Description Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Benchmarks show improvement (if performance-related)

## Performance Impact
(For performance-related changes)
- Benchmark results before/after
- Memory usage comparison
- Scaling behavior analysis

## Breaking Changes
(If applicable)
- List any breaking changes
- Migration guide for users
```

### Review Process
1. **Automated Checks**: CI must pass (build, test, lint)
2. **Code Review**: At least one maintainer approval required
3. **Testing**: Complex changes require additional testing
4. **Documentation**: Updates must include relevant docs
5. **Performance**: Significant changes require benchmark validation

## Testing Guidelines

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_waveguide_effective_index() {
        let waveguide = Waveguide::new(450e-9, 220e-9);
        let n_eff = waveguide.effective_index(1550e-9);
        assert_relative_eq!(n_eff, 2.4, epsilon = 1e-3);
    }
}
```

### Integration Tests
```python
def test_neural_network_training():
    """Test end-to-end neural network training."""
    pnn = PhotonicNeuralNetwork([784, 256, 10])
    
    # Generate synthetic data
    X_train = np.random.randn(100, 784)
    y_train = np.random.randint(0, 10, 100)
    
    # Train for a few epochs
    initial_loss = pnn.evaluate(X_train, y_train)
    pnn.train(X_train, y_train, epochs=5)
    final_loss = pnn.evaluate(X_train, y_train)
    
    assert final_loss < initial_loss
```

### Property-Based Testing
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn waveguide_reciprocity(
        width in 200e-9..800e-9,
        height in 100e-9..400e-9,
        wavelength in 1200e-9..1700e-9
    ) {
        let wg = Waveguide::new(width, height);
        let forward = wg.propagate_forward(wavelength);
        let backward = wg.propagate_backward(wavelength);
        prop_assert_relative_eq!(forward.abs(), backward.abs(), epsilon = 1e-10);
    }
}
```

## Performance Guidelines

### Optimization Principles
1. **Measure First**: Profile before optimizing
2. **Algorithm Choice**: O(n) improvements beat micro-optimizations
3. **Memory Patterns**: Cache-friendly access patterns
4. **Parallelization**: Use Rayon for CPU, consider GPU for large problems
5. **SIMD**: Leverage nalgebra's SIMD support

### Benchmarking
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_simulation(c: &mut Criterion) {
    let device = create_test_device();
    let input = create_test_input();
    
    c.bench_function("photonic_simulation", |b| {
        b.iter(|| device.simulate(black_box(&input)))
    });
}

criterion_group!(benches, benchmark_simulation);
criterion_main!(benches);
```

## Issue Reporting

### Bug Reports
Use the bug report template with:
- **Environment**: OS, Rust version, Python version
- **Reproduction**: Minimal example that reproduces the issue
- **Expected vs Actual**: Clear description of the problem
- **Context**: What you were trying to accomplish

### Feature Requests
Use the feature request template with:
- **Problem**: What problem does this solve?
- **Solution**: Describe your proposed solution
- **Alternatives**: What alternatives have you considered?
- **Additional Context**: Links to papers, implementations, etc.

## Scientific Contributions

### Experimental Validation
- Include measurement data and setup details
- Compare simulation vs experiment with error analysis
- Document calibration procedures
- Provide uncertainty estimates

### New Physics Models
- Cite relevant literature and theoretical foundation
- Include validation against known analytical solutions
- Provide parameter fitting procedures
- Document model limitations and accuracy

### Algorithm Improvements
- Include complexity analysis (time/space)
- Compare against existing implementations
- Provide convergence analysis
- Include stability and accuracy studies

## Community Guidelines

### Code of Conduct
This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). 

### Communication
- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: General questions, ideas, showcase
- **Discord**: Real-time chat and community support
- **Email**: Security issues and private matters

### Recognition
Contributors are recognized through:
- **README**: Contributor list with areas of contribution
- **Releases**: Acknowledgment in release notes
- **Papers**: Co-authorship for significant scientific contributions
- **Presentations**: Speaking opportunities at conferences

## Getting Help

### Resources
- **Documentation**: https://photon-memristor-sim.readthedocs.io
- **Examples**: Check the `examples/` directory
- **Tests**: Reference tests for usage patterns
- **Discord**: Join our community chat

### Mentorship
New contributors can request mentorship for:
- Getting started with the codebase
- Physics and photonics concepts
- Rust and Python best practices
- Scientific software development

### Office Hours
Maintainers hold weekly office hours for:
- Architecture discussions
- Code review sessions
- Scientific consultation
- Career advice

---

We appreciate all contributions, from typo fixes to major algorithmic improvements. Every contribution makes this project better for the entire photonics community! 🚀