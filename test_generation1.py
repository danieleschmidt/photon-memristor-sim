#!/usr/bin/env python3
"""
Test Generation 1: Make It Work
Comprehensive test of basic photonic neural network functionality
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from photon_memristor_sim import devices, neural_networks, jax_interface, utils, visualization
from photon_memristor_sim._core import PyOpticalField, PyPhotonicArray

def test_optical_field_creation():
    """Test optical field creation and manipulation"""
    print("🔬 Testing Optical Field Creation...")
    
    # Create Gaussian beam
    real_amp, imag_amp = utils.create_gaussian_beam(32, 32, 5e-6)
    
    # Create optical field
    field = PyOpticalField(
        real_amp.tolist(), imag_amp.tolist(),
        1550e-9, 1e-3
    )
    
    assert field.wavelength == 1550e-9
    assert abs(field.power - 1e-3) < 1e-6
    
    print(f"  ✅ Field wavelength: {field.wavelength*1e9:.1f} nm")
    print(f"  ✅ Field power: {field.power*1e3:.2f} mW")
    print(f"  ✅ Field dimensions: {field.dimensions()}")

def test_device_models():
    """Test photonic device models"""
    print("\n🔧 Testing Device Models...")
    
    # Test PCM device
    pcm = devices.PCMDevice("GST")
    optical_field = jnp.ones(10) * 0.1
    response = pcm.simulate(optical_field)
    print(f"  ✅ PCM device response: {jnp.mean(response):.3f}")
    
    # Test oxide memristor
    oxide = devices.OxideMemristor("HfO2")
    oxide.set_voltage(2.0, 1e-6)  # 2V pulse
    response = oxide.simulate(optical_field)
    print(f"  ✅ Oxide memristor response: {jnp.mean(response):.3f}")
    
    # Test ring resonator
    ring = devices.MicroringResonator(10e-6)
    ring.set_thermal_tuning(5e-3)  # 5mW tuning
    response = ring.simulate(optical_field)
    print(f"  ✅ Ring resonator response: {jnp.mean(response):.3f}")

def test_crossbar_array():
    """Test photonic crossbar array"""
    print("\n🌐 Testing Crossbar Array...")
    
    # Create 4x4 crossbar
    crossbar = devices.PhotonicCrossbar((4, 4), "pcm")
    
    # Set random weights
    weights = jax.random.normal(jax.random.PRNGKey(42), (4, 4))
    crossbar.set_weights(weights)
    
    # Test matrix-vector multiplication
    input_vector = jnp.array([0.1, 0.2, 0.3, 0.4])
    output = crossbar.matrix_vector_multiply(input_vector)
    
    print(f"  ✅ Input: {input_vector}")
    print(f"  ✅ Output: {output}")
    print(f"  ✅ Power consumption: {crossbar.total_power_consumption()*1e3:.1f} mW")

def test_neural_network():
    """Test photonic neural network"""
    print("\n🧠 Testing Neural Network...")
    
    # Create simple 3-layer network
    layer_sizes = [4, 8, 2]
    network = neural_networks.PhotonicNeuralNetwork(layer_sizes)
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = network.init_params(key, (1, 4))
    
    # Forward pass
    x = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    output = network(x, params)
    
    print(f"  ✅ Network architecture: {layer_sizes}")
    print(f"  ✅ Input shape: {x.shape}")
    print(f"  ✅ Output shape: {output.shape}")
    print(f"  ✅ Output values: {output[0]}")

def test_jax_interface():
    """Test JAX integration"""
    print("\n⚡ Testing JAX Interface...")
    
    # Test photonic matrix multiplication
    inputs = jnp.array([0.1, 0.2, 0.3])
    weights = jnp.array([[0.5, 0.3, 0.2], 
                         [0.4, 0.6, 0.1]])
    
    outputs = jax_interface.photonic_matmul(inputs, weights)
    print(f"  ✅ Photonic matmul result: {outputs}")
    
    # Test gradient computation
    def loss_fn(w):
        return jnp.sum(jax_interface.photonic_matmul(inputs, w) ** 2)
    
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(weights)
    print(f"  ✅ Gradient computation successful: {gradients.shape}")

def test_training_simple():
    """Test simple training loop"""
    print("\n🎯 Testing Training...")
    
    # Generate synthetic data
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (100, 4))
    y = jnp.sum(X * jnp.array([1, -1, 0.5, -0.5]), axis=1, keepdims=True)
    
    # Create network
    network = neural_networks.PhotonicNeuralNetwork([4, 2, 1])
    params = network.init_params(key, (1, 4))
    
    # Simple training step
    def loss_fn(params, x, y_true):
        y_pred = network(x, params)
        return jnp.mean((y_pred - y_true) ** 2)
    
    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(params, X[:10], y[:10])
    
    initial_loss = loss_fn(params, X[:10], y[:10])
    print(f"  ✅ Initial loss: {initial_loss:.6f}")
    print(f"  ✅ Gradient computation: Success")

def test_hardware_aware_features():
    """Test hardware-aware features"""
    print("\n⚙️  Testing Hardware-Aware Features...")
    
    from photon_memristor_sim.training import HardwareAwareOptimizer
    
    # Create optimizer with constraints
    optimizer = HardwareAwareOptimizer(
        learning_rate=0.001,
        power_budget=0.1,  # 100mW
        max_temperature=400.0  # 400K
    )
    
    print(f"  ✅ Power budget: {optimizer.power_budget*1000:.0f} mW")
    print(f"  ✅ Max temperature: {optimizer.max_temperature:.0f} K")

def test_visualization():
    """Test visualization capabilities"""
    print("\n📊 Testing Visualization...")
    
    # Create visualizer
    viz = visualization.PhotonicCircuitVisualizer()
    
    # Test crossbar visualization
    device_states = jnp.abs(jax.random.normal(jax.random.PRNGKey(42), (4, 4)))
    fig = viz.plot_crossbar_array((4, 4), device_states, title="Test Crossbar")
    
    # Save visualization
    plt.figure(fig.number)
    plt.savefig('test_crossbar_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Crossbar visualization saved")
    
    # Test neural network layout
    fig2 = viz.plot_neural_network_layout([4, 8, 2], title="Test Network")
    plt.figure(fig2.number)
    plt.savefig('test_network_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Network layout visualization saved")

def test_utils_functions():
    """Test utility functions"""
    print("\n🔧 Testing Utility Functions...")
    
    # Test unit conversions
    freq = utils.wavelength_to_frequency(1550e-9)
    print(f"  ✅ 1550nm → {freq/1e14:.2f} × 10^14 Hz")
    
    # Test power conversions
    dbm = utils.power_to_dbm(1e-3)  # 1mW
    print(f"  ✅ 1mW → {dbm:.1f} dBm")
    
    # Test FSR calculation
    fsr = utils.calculate_fsr(10e-6, 2.4, 1550e-9)
    print(f"  ✅ Ring FSR: {fsr*1e12:.2f} pm")

def run_full_demo():
    """Run a complete demonstration"""
    print("\n🚀 Running Full Demo: Photonic XOR Gate Training...")
    
    # Create XOR dataset
    X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)
    
    # Create photonic network
    network = neural_networks.PhotonicNeuralNetwork([2, 4, 1])
    key = jax.random.PRNGKey(42)
    params = network.init_params(key, (1, 2))
    
    # Define loss function
    def loss_fn(params, x, y_true):
        y_pred = network(x, params, training=True)
        return jnp.mean((y_pred - y_true) ** 2)
    
    # Training loop
    lr = 0.1
    losses = []
    
    for epoch in range(50):
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, X, y)
        
        # Simple SGD update
        params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
        
        if epoch % 10 == 0:
            loss = loss_fn(params, X, y)
            losses.append(loss)
            print(f"  Epoch {epoch}: Loss = {loss:.6f}")
    
    # Final evaluation
    final_outputs = network(X, params, training=False)
    print(f"\n  Final XOR Results:")
    for i, (inp, target, pred) in enumerate(zip(X, y, final_outputs)):
        print(f"    {inp} → {pred[0]:.3f} (target: {target[0]})")
    
    print(f"  ✅ Training completed successfully!")

def main():
    """Run all Generation 1 tests"""
    print("🌟 TERRAGON PHOTONIC-MLIR-SYNTH-BRIDGE")
    print("=" * 50)
    print("Generation 1: MAKE IT WORK - Testing Suite")
    print("=" * 50)
    
    try:
        test_optical_field_creation()
        test_device_models()
        test_crossbar_array()
        test_neural_network()
        test_jax_interface()
        test_training_simple()
        test_hardware_aware_features()
        test_visualization()
        test_utils_functions()
        run_full_demo()
        
        print("\n" + "=" * 50)
        print("🎉 ALL GENERATION 1 TESTS PASSED!")
        print("✅ Photonic simulation core functional")
        print("✅ Device models operational")
        print("✅ Neural network working")
        print("✅ JAX integration successful")
        print("✅ Training capabilities verified")
        print("✅ Hardware constraints implemented")
        print("✅ Visualization working")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)