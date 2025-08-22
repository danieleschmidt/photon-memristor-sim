# Multi-stage production Dockerfile for Photon-Memristor-Sim
FROM rust:1.70 as rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Rust source and build
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build optimized Rust binary
RUN cargo build --release

# Python runtime stage
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r photonicuser && useradd -r -g photonicuser photonicuser

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY pyproject.toml ./
COPY python/ ./python/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy Rust binary from builder stage
COPY --from=rust-builder /app/target/release/libphoton_memristor_sim.so /app/

# Copy remaining application files
COPY . .

# Set ownership to non-root user
RUN chown -R photonicuser:photonicuser /app

# Switch to non-root user
USER photonicuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Production command
CMD ["python", "-m", "photon_memristor_sim.server", "--host", "0.0.0.0", "--port", "8080"]
