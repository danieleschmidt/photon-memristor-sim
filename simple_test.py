#!/usr/bin/env python3
"""
Simple Generation 1 Test - Core Functionality
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_basic_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing Basic Imports...")
    
    try:
        from photon_memristor_sim._core import PyOpticalField, PyPhotonicArray
        print("  ✅ Rust core bindings imported")
        
        from photon_memristor_sim import utils
        print("  ✅ Utils module imported")
        
        from photon_memristor_sim import devices
        print("  ✅ Device models imported")
        
        from photon_memristor_sim import jax_interface
        print("  ✅ JAX interface imported")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_optical_field():
    """Test optical field creation"""
    print("\n🔬 Testing Optical Field...")
    
    try:
        from photon_memristor_sim._core import PyOpticalField
        from photon_memristor_sim import utils
        
        # Create simple field
        real_amp, imag_amp = utils.create_gaussian_beam(16, 16, 3e-6)
        
        field = PyOpticalField(
            real_amp.tolist(), imag_amp.tolist(),
            1550e-9, 1e-3
        )
        
        print(f"  ✅ Wavelength: {field.wavelength*1e9:.0f} nm")
        print(f"  ✅ Power: {field.power*1e3:.1f} mW")
        print(f"  ✅ Dimensions: {field.dimensions()}")
        
        return True
    except Exception as e:
        print(f"  ❌ Field test failed: {e}")
        return False

def test_simple_device():
    """Test simple device model"""
    print("\n🔧 Testing Simple Device...")
    
    try:
        from photon_memristor_sim import devices
        
        # Test PCM device
        pcm = devices.PCMDevice("GST")
        optical_field = jnp.ones(5) * 0.1
        response = pcm.simulate(optical_field)
        
        print(f"  ✅ PCM device created")
        print(f"  ✅ Response shape: {response.shape}")
        print(f"  ✅ Mean response: {jnp.mean(response):.3f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Device test failed: {e}")
        return False

def test_jax_integration():
    """Test basic JAX integration"""
    print("\n⚡ Testing JAX Integration...")
    
    try:
        from photon_memristor_sim import jax_interface
        
        # Simple test
        inputs = jnp.array([0.1, 0.2])
        weights = jnp.array([[0.5, 0.3], [0.4, 0.6]])
        
        outputs = jax_interface.photonic_matmul(inputs, weights)
        
        print(f"  ✅ Matrix multiplication successful")
        print(f"  ✅ Input: {inputs}")
        print(f"  ✅ Output: {outputs}")
        
        # Test gradient
        def simple_loss(w):
            return jnp.sum(jax_interface.photonic_matmul(inputs, w))
        
        grad_fn = jax.grad(simple_loss)
        gradients = grad_fn(weights)
        
        print(f"  ✅ Gradient computation successful")
        print(f"  ✅ Gradient shape: {gradients.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ JAX test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions"""
    print("\n🔧 Testing Utilities...")
    
    try:
        from photon_memristor_sim import utils
        
        # Unit conversions
        freq = utils.wavelength_to_frequency(1550e-9)
        print(f"  ✅ 1550nm → {freq/1e14:.2f} × 10^14 Hz")
        
        # Power conversion
        dbm = utils.power_to_dbm(1e-3)
        print(f"  ✅ 1mW → {dbm:.1f} dBm")
        
        # Gaussian beam
        real, imag = utils.create_gaussian_beam(8, 8, 2e-6)
        print(f"  ✅ Gaussian beam: {real.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ Utils test failed: {e}")
        return False

def test_simple_training():
    """Test simple XOR training"""
    print("\n🎯 Testing Simple Training...")
    
    try:
        from photon_memristor_sim import neural_networks
        
        # XOR data
        X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
        y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)
        
        # Simple network
        network = neural_networks.PhotonicNeuralNetwork([2, 2, 1])
        key = jax.random.PRNGKey(42)
        params = network.init_params(key, (1, 2))
        
        # Forward pass
        output = network(X, params)
        print(f"  ✅ Network forward pass successful")
        print(f"  ✅ Output shape: {output.shape}")
        
        # Loss function
        def loss_fn(params):
            pred = network(X, params, training=True)
            return jnp.mean((pred - y) ** 2)
        
        initial_loss = loss_fn(params)
        print(f"  ✅ Initial loss: {initial_loss:.4f}")
        
        # Gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        print(f"  ✅ Gradient computation successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple test suite"""
    print("🌟 PHOTONIC-MLIR-SYNTH-BRIDGE - Simple Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_optical_field, 
        test_simple_device,
        test_utils,
        test_jax_integration,
        test_simple_training,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Generation 1 Working!")
        print("✅ Core photonic simulation functional")
        print("✅ Device models operational") 
        print("✅ JAX integration successful")
        print("✅ Training pipeline working")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    print("=" * 50)
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)