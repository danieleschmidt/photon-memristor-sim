# Photon-Memristor-Sim Justfile
# Run `just --list` to see all available commands

# Default recipe
default:
    @just --list

# Build commands
build:
    cargo build --release

build-dev:
    cargo build

build-python:
    maturin develop --release

build-wasm:
    wasm-pack build --target web --release

build-all: build build-python build-wasm
    @echo "All builds complete!"

# Testing commands
test:
    cargo test --all-features

test-python:
    pytest python/tests/ -v

test-all: test test-python
    @echo "All tests passed!"

# Code quality
fmt:
    cargo fmt --all
    black python/
    isort python/

lint:
    cargo clippy --all-targets --all-features -- -D warnings
    flake8 python/
    mypy python/

check: fmt lint
    @echo "Code quality checks passed!"

# Documentation
docs:
    cargo doc --open --all-features

docs-python:
    cd python && python -m sphinx.cmd.build -b html docs docs/_build

# Development
dev: build-python
    @echo "Development build ready!"

clean:
    cargo clean
    rm -rf python/build/
    rm -rf python/dist/
    rm -rf python/*.egg-info/
    rm -rf target/
    find . -name "__pycache__" -type d -exec rm -rf {} +
    find . -name "*.pyc" -delete

# Benchmarks
bench:
    cargo bench

bench-python:
    pytest python/tests/ --benchmark-only

# Examples
example-basic:
    python examples/basic_usage.py

example-notebook:
    jupyter notebook examples/

# Release
release-patch:
    cargo release patch --execute

release-minor:
    cargo release minor --execute

release-major:
    cargo release major --execute

# Setup
setup:
    rustup update
    rustup component add clippy rustfmt rust-src
    rustup target add wasm32-unknown-unknown
    cargo install maturin wasm-pack
    pip install -e ".[dev]"

# Git hooks
install-hooks:
    pre-commit install
    pre-commit install --hook-type commit-msg

# Performance
profile:
    cargo build --release
    perf record -g ./target/release/photon_memristor_sim
    perf report

flamegraph:
    cargo flamegraph --root -- bench

# Docker
docker-build:
    docker build -t photon-memristor-sim .

docker-run:
    docker run -it --rm -v $(pwd):/workspace photon-memristor-sim

# CI/CD helpers
ci-test: test test-python lint

ci-build: build-all

# Utilities
lines:
    @echo "Rust code:"
    @find src -name "*.rs" | xargs wc -l | tail -1
    @echo "Python code:"
    @find python -name "*.py" | xargs wc -l | tail -1

deps:
    @echo "Rust dependencies:"
    @cargo tree --depth 1
    @echo ""
    @echo "Python dependencies:"
    @pip list

# Help
help:
    @echo "Photon-Memristor-Sim Development Commands"
    @echo "========================================"
    @echo ""
    @echo "Building:"
    @echo "  build        Build Rust in release mode"
    @echo "  build-dev    Build Rust in debug mode" 
    @echo "  build-python Build Python bindings"
    @echo "  build-wasm   Build WebAssembly module"
    @echo "  build-all    Build everything"
    @echo ""
    @echo "Testing:"
    @echo "  test         Run Rust tests"
    @echo "  test-python  Run Python tests"
    @echo "  test-all     Run all tests"
    @echo ""
    @echo "Code Quality:"
    @echo "  fmt          Format all code"
    @echo "  lint         Lint all code"
    @echo "  check        Format and lint"
    @echo ""
    @echo "Development:"
    @echo "  dev          Quick development build"
    @echo "  clean        Clean all build artifacts"
    @echo "  setup        Install development dependencies"
    @echo ""
    @echo "Documentation:"
    @echo "  docs         Generate and open Rust docs"
    @echo "  docs-python  Generate Python docs"
    @echo ""
    @echo "Performance:"
    @echo "  bench        Run Rust benchmarks"
    @echo "  profile      Profile with perf"
    @echo "  flamegraph   Generate flamegraph"
    @echo ""
    @echo "Examples:"
    @echo "  example-basic    Run basic usage example"
    @echo "  example-notebook Start Jupyter with examples"