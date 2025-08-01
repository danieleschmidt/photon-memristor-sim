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

## Release Process

1. Update version in `Cargo.toml` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite: `cargo test && pytest`
4. Build and test wheel: `maturin build --release`
5. Create release PR
6. Tag release: `git tag v0.x.0`
7. CI will handle PyPI and crates.io publishing