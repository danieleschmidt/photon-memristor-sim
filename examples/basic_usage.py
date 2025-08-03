#!/usr/bin/env python3
"""
Basic usage examples for Photon-Memristor-Sim

This script demonstrates core functionality and serves as a tutorial
for getting started with photonic neural network simulation.
"""

import numpy as np
import jax.numpy as jnp
from jax import random, grad, jit
import matplotlib.pyplot as plt

# Import photon-memristor-sim
try:
    import photon_memristor_sim as pms
    from photon_memristor_sim.neural_networks import PhotonicNeuralNetwork
    from photon_memristor_sim.jax_interface import photonic_matmul
except ImportError as e:
    print(f"Error importing photon_memristor_sim: {e}")
    print("Make sure the package is installed with 'maturin develop'")
    exit(1)


def example_1_basic_simulation():
    """Example 1: Basic photonic array simulation."""
    print("=" * 60)
    print("Example 1: Basic Photonic Array Simulation")
    print("=" * 60)
    
    # Create photonic crossbar array
    array = pms.PyPhotonicArray("crossbar", rows=4, cols=4)
    print(f"Created {array.dimensions()} photonic crossbar array")
    
    # Create optical fields
    amplitude_real = np.random.randn(8, 8)
    amplitude_imag = np.zeros((8, 8))
    
    inputs = []
    for i in range(4):
        field = pms.PyOpticalField(
            amplitude_real, amplitude_imag,
            wavelength=1550e-9,  # 1550nm telecom wavelength
            power=1e-3  # 1mW
        )
        inputs.append(field)
    
    print(f"Created {len(inputs)} input optical fields")
    print(f"Input wavelength: {inputs[0].wavelength*1e9:.1f} nm")
    print(f"Input power: {inputs[0].power*1e3:.1f} mW")
    
    # Simulate forward propagation
    try:
        outputs = array.forward(inputs)
        print(f"Forward propagation successful: {len(outputs)} outputs")
        
        total_output_power = sum(output.power for output in outputs)
        print(f"Total output power: {total_output_power*1e3:.2f} mW")
        
    except Exception as e:
        print(f"Forward propagation failed: {e}")
    
    # Get array metrics
    metrics = array.metrics()
    print("\nArray Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def example_2_waveguide_analysis():
    """Example 2: Waveguide mode analysis."""
    print("\n" + "=" * 60)
    print("Example 2: Waveguide Mode Analysis")
    print("=" * 60)
    
    # Standard silicon photonic waveguide parameters
    width = 450e-9      # 450 nm
    height = 220e-9     # 220 nm
    core_index = 3.47   # Silicon at 1550nm
    cladding_index = 1.44  # SiO2
    wavelength = 1550e-9   # C-band
    
    print("Analyzing silicon photonic waveguide:")
    print(f"  Width: {width*1e9:.0f} nm")
    print(f"  Height: {height*1e9:.0f} nm") 
    print(f"  Core index: {core_index}")
    print(f"  Cladding index: {cladding_index}")
    print(f"  Wavelength: {wavelength*1e9:.0f} nm")
    
    # Calculate effective index and mode profile
    n_eff, intensity_profile = pms.calculate_waveguide_mode(
        width, height, core_index, cladding_index, wavelength
    )
    
    print(f"\nResults:")
    print(f"  Effective index: {n_eff:.4f}")
    print(f"  Mode profile shape: {intensity_profile.shape}")
    print(f"  Peak intensity: {np.max(intensity_profile):.3f}")
    
    # Optional: Plot mode profile if matplotlib available
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(intensity_profile, extent=[-1.5, 1.5, -1.5, 1.5], 
                  cmap='hot', origin='lower')
        plt.colorbar(label='Normalized Intensity')
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        plt.title('Fundamental Mode Profile')
        plt.savefig('mode_profile.png', dpi=150, bbox_inches='tight')
        print("  Mode profile saved as 'mode_profile.png'")
    except:
        print("  (Matplotlib not available for plotting)")


def example_3_jax_integration():
    """Example 3: JAX integration and automatic differentiation."""
    print("\n" + "=" * 60)
    print("Example 3: JAX Integration and Auto-Differentiation")
    print("=" * 60)
    
    # Create test data
    key = random.PRNGKey(42)
    inputs = jnp.array([1.0, 0.5, 0.8, 0.3])
    weights = random.normal(key, (3, 4)) * 0.5
    
    print("Test data:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Weight shape: {weights.shape}")
    print(f"  Input values: {inputs}")
    
    # Forward pass through photonic matmul
    outputs = photonic_matmul(inputs, weights, wavelength=1550e-9)
    print(f"\nPhotonic matmul output: {outputs}")
    print(f"Output shape: {outputs.shape}")
    
    # Define loss function
    def loss_fn(w):
        out = photonic_matmul(inputs, w, wavelength=1550e-9)
        target = jnp.array([1.0, 0.5, 0.2])
        return jnp.mean((out - target) ** 2)
    
    # Compute gradients
    grad_fn = grad(loss_fn)
    gradients = grad_fn(weights)
    
    print(f"\nLoss: {loss_fn(weights):.6f}")
    print(f"Gradient shape: {gradients.shape}")
    print(f"Gradient norm: {jnp.linalg.norm(gradients):.6f}")
    
    # Test JIT compilation
    jit_loss_fn = jit(loss_fn)
    jit_loss = jit_loss_fn(weights)
    print(f"JIT compiled loss: {jit_loss:.6f}")


def example_4_neural_network():
    """Example 4: Photonic neural network training."""
    print("\n" + "=" * 60)
    print("Example 4: Photonic Neural Network")
    print("=" * 60)
    
    # Create photonic neural network
    layers = [784, 128, 64, 10]  # MNIST-like architecture
    pnn = PhotonicNeuralNetwork(layers, activation="photonic_relu")
    
    print("Created photonic neural network:")
    print(f"  Architecture: {' → '.join(map(str, layers))}")
    print(f"  Activation: {pnn.activation}")
    print(f"  Number of layers: {len(pnn.layers)}")
    
    # Initialize parameters
    key = random.PRNGKey(42)
    params = pnn.init_params(key, (1, 784))
    
    print(f"\nInitialized {len(params)} parameter groups")
    
    # Create dummy training data
    batch_size = 32
    data_key, noise_key = random.split(key)
    inputs = random.normal(data_key, (batch_size, 784))
    targets = random.normal(noise_key, (batch_size, 10))
    
    print(f"Created training data:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    
    # Forward pass
    outputs = pnn(inputs, params, training=False)
    print(f"Forward pass output shape: {outputs.shape}")
    
    # Compute loss
    loss = pnn.loss_fn(params, inputs, targets)
    print(f"Loss: {loss:.6f}")
    
    # Hardware metrics
    print(f"\nHardware metrics:")
    print(f"  Total devices: {pnn.device_count()}")
    print(f"  Power consumption: {pnn.total_power()*1e3:.2f} mW")
    
    # Benchmark performance
    try:
        benchmark = pnn.benchmark_performance(batch_size=100)
        print(f"\nBenchmark results:")
        for metric, value in benchmark.items():
            if isinstance(value, float):
                if 'time' in metric:
                    print(f"  {metric}: {value*1000:.2f} ms")
                elif 'efficiency' in metric:
                    print(f"  {metric}: {value:.2e} FLOPS/W")
                else:
                    print(f"  {metric}: {value:.2e}")
            else:
                print(f"  {metric}: {value}")
    except Exception as e:
        print(f"Benchmarking failed: {e}")


def example_5_device_simulation():
    """Example 5: Individual device simulation."""
    print("\n" + "=" * 60)
    print("Example 5: Device Simulation")
    print("=" * 60)
    
    # Test different device types
    device_types = ["pcm", "oxide", "ring"]
    
    for device_type in device_types:
        try:
            result = pms.create_device_simulator(device_type)
            print(f"✓ {device_type.upper()}: {result}")
        except Exception as e:
            print(f"✗ {device_type.upper()}: {e}")
    
    # Test invalid device type
    try:
        pms.create_device_simulator("invalid")
    except ValueError as e:
        print(f"✓ Error handling working: {e}")


def example_6_optimization_demo():
    """Example 6: Simple optimization demonstration."""
    print("\n" + "=" * 60)
    print("Example 6: Optimization Demonstration")
    print("=" * 60)
    
    # Simple optimization problem: minimize photonic network output
    def objective(weights):
        inputs = jnp.array([1.0, 0.5])
        outputs = photonic_matmul(inputs, weights)
        return jnp.sum(outputs ** 2)  # Minimize total power
    
    # Initial weights
    key = random.PRNGKey(123)
    initial_weights = random.normal(key, (2, 2)) * 0.5
    
    print("Optimization problem: minimize photonic network output power")
    print(f"Initial weights:\n{initial_weights}")
    print(f"Initial objective: {objective(initial_weights):.6f}")
    
    # Simple gradient descent
    learning_rate = 0.1
    weights = initial_weights
    
    print("\nGradient descent optimization:")
    for i in range(10):
        grad_fn = grad(objective)
        gradients = grad_fn(weights)
        weights = weights - learning_rate * gradients
        
        if i % 2 == 0:
            obj_val = objective(weights)
            print(f"  Step {i:2d}: objective = {obj_val:.6f}")
    
    print(f"\nFinal weights:\n{weights}")
    print(f"Final objective: {objective(weights):.6f}")
    print(f"Improvement: {objective(initial_weights) - objective(weights):.6f}")


def main():
    """Run all examples."""
    print("Photon-Memristor-Sim Examples")
    print("============================")
    print(f"Version: {pms.VERSION}")
    print(f"Speed of light: {pms.SPEED_OF_LIGHT} m/s")
    
    try:
        example_1_basic_simulation()
        example_2_waveguide_analysis()
        example_3_jax_integration()
        example_4_neural_network()
        example_5_device_simulation()
        example_6_optimization_demo()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()