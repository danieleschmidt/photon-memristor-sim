#!/bin/bash
set -e

echo "🚀 Setting up Photon-Memristor-Sim development environment..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies for photonic simulation
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libfftw3-dev \
    graphviz \
    graphviz-dev \
    libgraphviz-dev \
    libsuitesparse-dev \
    libeigen3-dev \
    valgrind \
    gdb \
    lldb

# Install Rust toolchain components
echo "📦 Installing Rust toolchain..."
rustup update stable
rustup component add clippy rustfmt rust-src
rustup target add wasm32-unknown-unknown

# Install additional Rust tools
cargo install --locked \
    maturin \
    wasm-pack \
    cargo-edit \
    cargo-watch \
    cargo-expand \
    cargo-criterion \
    cargo-flamegraph \
    cargo-valgrind \
    cargo-audit \
    cargo-outdated \
    cargo-release \
    just

# Install Python dependencies
echo "🐍 Setting up Python environment..."
python -m pip install --upgrade pip setuptools wheel

# Install core Python dependencies
pip install \
    numpy \
    scipy \
    matplotlib \
    jax[cpu] \
    jaxlib \
    flax \
    optax \
    pandas \
    scikit-learn \
    jupyter \
    ipykernel \
    plotly \
    seaborn \
    tqdm \
    h5py \
    zarr \
    xarray

# Install development dependencies
pip install \
    pytest \
    pytest-cov \
    pytest-benchmark \
    pytest-xdist \
    black \
    pylint \
    mypy \
    isort \
    pre-commit \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    nbsphinx \
    memory-profiler \
    line-profiler

# Install additional scientific dependencies
pip install \
    networkx \
    sympy \
    numba \
    cupy-cuda11x \
    tensorflow \
    torch \
    torchvision \
    transformers

# Install Node.js dependencies for WASM frontend
echo "📦 Setting up Node.js environment..."
npm install -g \
    typescript \
    webpack \
    webpack-cli \
    webpack-dev-server \
    @types/node \
    prettier \
    eslint

# Create Python virtual environment for project
echo "🔧 Creating project virtual environment..."
python -m venv venv
source venv/bin/activate

# Install project in development mode
pip install -e ".[dev]"

# Build Rust core with maturin
echo "⚙️ Building Rust core..."
maturin develop --release

# Initialize pre-commit hooks
echo "🔨 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Setup Jupyter kernel
python -m ipykernel install --user --name=photon-memristor-sim

# Create necessary directories
mkdir -p \
    data \
    experiments \
    notebooks \
    results \
    logs \
    cache \
    .pytest_cache \
    .mypy_cache

# Set up git configuration for development
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Create example environment file
cat > .env.example << 'EOF'
# Photon-Memristor-Sim Environment Configuration

# Logging
RUST_LOG=info
PYTHON_LOG_LEVEL=INFO

# Performance
RAYON_NUM_THREADS=4
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4

# CUDA (if available)
CUDA_VISIBLE_DEVICES=0

# JAX configuration
JAX_ENABLE_X64=True
JAX_PLATFORMS=cpu

# Development
DEVELOPMENT=True
DEBUG=True

# Paths
DATA_DIR=./data
RESULTS_DIR=./results
CACHE_DIR=./cache

# Simulation defaults
DEFAULT_WAVELENGTH=1550e-9
DEFAULT_POWER_BUDGET=100e-3
DEFAULT_TEMPERATURE=300.0

# Testing
PYTEST_WORKERS=auto
COVERAGE_THRESHOLD=80
EOF

# Create VS Code workspace settings
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "rust-analyzer.cargo.features": ["all"],
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.procMacro.enable": true,
    "rust-analyzer.cargo.buildScripts.enable": true,
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.analysis.typeCheckingMode": "strict",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.associations": {
        "*.toml": "toml",
        "*.md": "markdown"
    },
    "files.exclude": {
        "**/target": true,
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true,
        "**/.mypy_cache": true,
        "**/venv": true
    },
    "search.exclude": {
        "**/target": true,
        "**/venv": true,
        "**/node_modules": true
    }
}
EOF

# Create launch configuration for debugging
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/python"
            }
        },
        {
            "name": "Python: Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Rust: Debug",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/photon_memristor_sim",
            "args": [],
            "cwd": "${workspaceFolder}",
            "stopOnEntry": false,
            "environment": []
        },
        {
            "name": "Rust: Test",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/deps/photon_memristor_sim",
            "args": [],
            "cwd": "${workspaceFolder}",
            "stopOnEntry": false
        }
    ]
}
EOF

# Create tasks configuration
cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cargo: build",
            "type": "cargo",
            "command": "build",
            "group": "build",
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "cargo: test",
            "type": "cargo", 
            "command": "test",
            "group": "test",
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "cargo: clippy",
            "type": "cargo",
            "command": "clippy",
            "args": ["--all-targets", "--all-features"],
            "group": "build",
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "maturin: develop",
            "type": "shell",
            "command": "maturin",
            "args": ["develop", "--release"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "python: test",
            "type": "shell",
            "command": "python",
            "args": ["-m", "pytest", "tests/", "-v"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "full: test",
            "dependsOrder": "sequence",
            "dependsOn": ["cargo: test", "python: test"],
            "group": "test"
        }
    ]
}
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  cargo build          # Build Rust core"
echo "  cargo test           # Run Rust tests"
echo "  maturin develop      # Build Python bindings"
echo "  pytest tests/        # Run Python tests"
echo "  cargo clippy         # Lint Rust code"
echo "  black python/        # Format Python code"
echo "  cargo bench          # Run benchmarks"
echo ""
echo "📚 Documentation:"
echo "  cargo doc --open     # Open Rust documentation"
echo "  jupyter notebook     # Start Jupyter for examples"
echo ""
echo "🔧 Development tools installed and configured!"