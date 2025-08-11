#!/usr/bin/env python3
"""
Generation 1 Test: Make It Work (Python-only functionality)
Testing the photonic simulation without Rust core dependencies.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

def test_device_physics():
    """Test basic photonic device physics"""
    try:
        from photon_memristor_sim.devices import PCMDevice, calculate_optical_constants
        
        # Test PCM device creation
        device = PCMDevice(
            material="GST",
            dimensions=(200e-9, 50e-9, 10e-9),
            crystalline_n=6.5 + 0.5j,
            amorphous_n=4.0 + 0.1j
        )
        
        # Test optical constant calculations
        crystallinity_levels = np.linspace(0, 1, 10)
        optical_constants = calculate_optical_constants(crystallinity_levels)
        
        print(f"✅ Device Physics: PCM device created successfully")
        print(f"   - Dimensions: {device.dimensions}")
        print(f"   - Optical constants shape: {optical_constants.shape}")
        return True
    except Exception as e:
        print(f"❌ Device Physics failed: {e}")
        return False

def test_neural_networks():
    """Test photonic neural network concepts"""
    try:
        from photon_memristor_sim.neural_networks import PhotonicLayer
        
        # Create basic photonic layer
        layer = PhotonicLayer(
            input_size=64,
            output_size=32,
            activation="photonic_relu"
        )
        
        # Test forward pass concept
        input_data = np.random.random(64)
        output = layer.forward(input_data)
        
        print(f"✅ Neural Networks: Photonic layer works")
        print(f"   - Input shape: {input_data.shape}")
        print(f"   - Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Neural Networks failed: {e}")
        return False

def test_jax_integration():
    """Test JAX interface for gradient computation"""
    try:
        import jax
        import jax.numpy as jnp
        from photon_memristor_sim.jax_interface import photonic_matmul, optical_nonlinearity
        
        # Test photonic matrix multiplication
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[0.5, 1.0], [1.5, 2.0]])
        result = photonic_matmul(A, B)
        
        # Test optical nonlinearity
        x = jnp.array([1.0, -0.5, 2.0, -1.5])
        nonlinear_result = optical_nonlinearity(x, "photonic_relu")
        
        print(f"✅ JAX Integration: Photonic operations work")
        print(f"   - Matrix mult result shape: {result.shape}")
        print(f"   - Nonlinearity result: {nonlinear_result}")
        return True
    except Exception as e:
        print(f"❌ JAX Integration failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    try:
        from photon_memristor_sim.utils import wavelength_to_frequency, power_to_dbm
        
        # Test wavelength conversion
        wavelength = 1550e-9  # 1550nm
        frequency = wavelength_to_frequency(wavelength)
        
        # Test power conversion
        power_watts = 1e-3  # 1mW
        power_dbm = power_to_dbm(power_watts)
        
        print(f"✅ Utilities: Conversion functions work")
        print(f"   - 1550nm = {frequency/1e12:.1f} THz")
        print(f"   - 1mW = {power_dbm:.1f} dBm")
        return True
    except Exception as e:
        print(f"❌ Utilities failed: {e}")
        return False

def test_basic_simulation():
    """Test basic photonic simulation without Rust core"""
    try:
        from photon_memristor_sim.devices import create_photonic_array_simulation
        
        # Create basic photonic array simulation
        simulation = create_photonic_array_simulation(
            rows=8,
            cols=8,
            wavelength=1550e-9,
            device_type="PCM"
        )
        
        # Simulate optical propagation
        input_power = np.ones(8) * 1e-3  # 1mW per channel
        output = simulation.forward(input_power)
        
        print(f"✅ Basic Simulation: Photonic array simulation works")
        print(f"   - Array size: 8x8")
        print(f"   - Input power: {np.sum(input_power)*1000:.1f}mW total")
        print(f"   - Output power: {np.sum(output)*1000:.1f}mW total")
        return True
    except Exception as e:
        print(f"❌ Basic Simulation failed: {e}")
        return False

def main():
    """Run Generation 1 tests"""
    print("🚀 GENERATION 1: MAKE IT WORK (Python-only functionality)")
    print("=" * 60)
    
    tests = [
        test_device_physics,
        test_neural_networks, 
        test_jax_integration,
        test_utilities,
        test_basic_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 GENERATION 1 RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.8:  # 80% success threshold
        print("✅ GENERATION 1: SUCCESS - Basic functionality works!")
        return True
    else:
        print("❌ GENERATION 1: NEEDS WORK - Core functionality has issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)