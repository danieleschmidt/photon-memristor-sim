#!/bin/bash
set -e

echo "🌟 Starting Photon-Memristor-Sim development session..."

# Activate Python virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Python virtual environment activated"
fi

# Start background services if needed
echo "🔄 Checking development services..."

# Update Rust analyzer index
echo "📚 Updating Rust analyzer index..."
rust-analyzer --version > /dev/null 2>&1 && echo "✅ Rust analyzer ready"

# Check if we need to rebuild Python bindings
if [ -f "Cargo.toml" ] && [ ! -f "python/photon_memristor_sim/_core.*.so" ]; then
    echo "🔧 Building Python bindings..."
    maturin develop --release
fi

# Display environment information
echo ""
echo "🔍 Environment Information:"
echo "  Python: $(python --version)"
echo "  Rust: $(rustc --version)"
echo "  Node.js: $(node --version)"
echo "  Git: $(git --version | head -n1)"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)"
else
    echo "  GPU: Not available (CPU-only mode)"
fi

# Show project status
echo ""
echo "📊 Project Status:"
if [ -d ".git" ]; then
    echo "  Branch: $(git branch --show-current)"
    echo "  Commits: $(git rev-list --count HEAD)"
    
    # Show any uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "  Status: Uncommitted changes present"
    else
        echo "  Status: Clean working directory"
    fi
fi

# Display useful commands
echo ""
echo "🚀 Useful Commands:"
echo "  make help            # Show available make targets"
echo "  just --list          # Show available just recipes"
echo "  cargo --list         # Show cargo subcommands"
echo ""

# Check for any setup issues
echo "🔍 Health Check:"

# Check Rust toolchain
if cargo --version > /dev/null 2>&1; then
    echo "  ✅ Rust toolchain"
else
    echo "  ❌ Rust toolchain issue"
fi

# Check Python environment
if python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')" > /dev/null 2>&1; then
    echo "  ✅ Python environment"
else
    echo "  ❌ Python environment issue"
fi

# Check key dependencies
if python -c "import jax, numpy, scipy" > /dev/null 2>&1; then
    echo "  ✅ Core Python dependencies"
else
    echo "  ❌ Missing Python dependencies"
fi

# Start development servers if requested
if [ "$START_SERVERS" = "true" ]; then
    echo "🌐 Starting development servers..."
    
    # Start Jupyter server in background
    if command -v jupyter &> /dev/null; then
        nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root > /dev/null 2>&1 &
        echo "  📊 Jupyter server started on port 8888"
    fi
fi

echo ""
echo "🎉 Development environment ready!"
echo "   Happy coding! 🚀"