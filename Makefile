.PHONY: help build test lint fmt clean install dev benchmark docs docker

help:
	@echo "Photon-Memristor-Sim Development Commands"
	@echo "========================================"
	@echo "build     - Build Rust core and Python bindings"
	@echo "test      - Run all tests (Rust + Python)"
	@echo "lint      - Run linting checks"
	@echo "fmt       - Format code (Rust + Python)"
	@echo "clean     - Clean build artifacts"
	@echo "install   - Install development dependencies"
	@echo "dev       - Quick development build"
	@echo "benchmark - Run performance benchmarks"
	@echo "docs      - Generate documentation"
	@echo "docker    - Build Docker development image"

install:
	@echo "Installing Rust dependencies..."
	cargo build --workspace
	@echo "Installing Python dependencies..."
	pip install -e .[dev,viz,docs]
	@echo "Installing pre-commit hooks..."
	pre-commit install

build:
	@echo "Building Rust core..."
	cargo build --release
	@echo "Building Python bindings..."
	maturin develop --release

dev:
	@echo "Quick development build..."
	maturin develop

test:
	@echo "Running Rust tests..."
	cargo test --workspace
	@echo "Running Python tests..."
	python -m pytest python/tests/ -v
	@echo "Running integration tests..."
	python -m pytest tests/ -v

lint:
	@echo "Linting Rust code..."
	cargo clippy --workspace -- -D warnings
	@echo "Linting Python code..."
	black --check python/
	isort --check-only python/
	mypy python/photon_memristor_sim/

fmt:
	@echo "Formatting Rust code..."
	cargo fmt --all
	@echo "Formatting Python code..."
	black python/
	isort python/

benchmark:
	@echo "Running Rust benchmarks..."
	cargo bench
	@echo "Running Python benchmarks..."
	python -m pytest benchmarks/ --benchmark-only

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf python/photon_memristor_sim/*.so
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +

docs:
	@echo "Generating Rust documentation..."
	cargo doc --no-deps --workspace
	@echo "Generating Python documentation..."
	cd docs && make html

docker:
	@echo "Building Docker development image..."
	docker build -t photon-memristor-sim:dev .

wasm:
	@echo "Building WASM package..."
	wasm-pack build --target web --out-dir pkg --release

release:
	@echo "Creating release build..."
	cargo build --release
	maturin build --release
	@echo "Release artifacts in target/wheels/"

.DEFAULT_GOAL := help