#!/usr/bin/env python3
"""
Generation 1 Test - Basic functionality verification
Tests the core photonic simulation capabilities
"""

import sys
import traceback
import numpy as np

def test_basic_import():
    """Test basic module import"""
    try:
        print("Testing basic imports...")
        # Test core Python functionality without Rust bindings
        import photon_memristor_sim.utils as utils
        import photon_memristor_sim.devices as devices  
        import photon_memristor_sim.neural_networks as nn
        print("✓ Python modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_wavelength_utilities():
    """Test wavelength conversion utilities"""
    try:
        print("Testing wavelength utilities...")
        from photon_memristor_sim.utils import wavelength_to_frequency, db_to_linear, linear_to_db
        
        # Test wavelength conversion
        freq = wavelength_to_frequency(1550e-9)  # 1550nm
        expected_freq = 299792458.0 / 1550e-9
        assert abs(freq - expected_freq) < 1e6, f"Frequency mismatch: {freq} vs {expected_freq}"
        
        # Test dB conversions
        power_linear = 0.001  # 1mW
        power_db = linear_to_db(power_linear)
        power_back = db_to_linear(power_db)
        assert abs(power_back - power_linear) < 1e-6, "dB conversion failed"
        
        print(f"✓ Wavelength utilities working (f={freq/1e14:.2f}THz, P={power_db:.1f}dBm)")
        return True
    except Exception as e:
        print(f"✗ Wavelength utilities failed: {e}")
        return False

def test_device_models():
    """Test device model creation"""
    try:
        print("Testing device model creation...")
        from photon_memristor_sim.devices import PCMDevice, OxideMemristor
        
        # Test PCM device
        pcm = PCMDevice(
            material="GST",
            dimensions=(200e-9, 50e-9, 10e-9),
            crystallinity=0.5
        )
        
        # Test basic properties
        assert pcm.get_crystallinity() == 0.5, "PCM crystallinity not set correctly"
        assert pcm.get_transmission(1550e-9) > 0, "PCM transmission should be positive"
        
        # Test oxide device  
        oxide = OxideMemristor(
            oxide="HfO2",
            thickness=5e-9,
            conductance=1e-6
        )
        
        assert oxide.get_conductance() == 1e-6, "Oxide conductance not set correctly"
        assert oxide.get_resistance() == 1e6, "Oxide resistance calculation failed"
        
        print("✓ Device models created and basic properties verified")
        return True
    except Exception as e:
        print(f"✗ Device model test failed: {e}")
        return False

def test_neural_network():
    """Test photonic neural network functionality"""
    try:
        print("Testing photonic neural network...")
        from photon_memristor_sim.neural_networks import PhotonicLayer, PhotonicNeuralNetwork
        
        # Create simple network
        pnn = PhotonicNeuralNetwork(
            layers=[784, 128, 10],
            wavelength=1550e-9,
            device_type="PCM"
        )
        
        # Test network properties
        assert len(pnn.layers) == 3, "Network should have 3 layers"
        assert pnn.layers[0].input_size == 784, "Input layer size mismatch"
        assert pnn.layers[-1].output_size == 10, "Output layer size mismatch"
        
        # Test forward pass with dummy data
        dummy_input = np.random.rand(784) * 0.001  # 1mW per channel
        output = pnn.forward(dummy_input, include_device_physics=False)
        
        assert output.shape == (10,), f"Output shape mismatch: {output.shape}"
        assert np.all(output >= 0), "Output powers should be non-negative"
        
        print(f"✓ Neural network forward pass successful (input: {dummy_input.shape}, output: {output.shape})")
        return True
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_quantum_planning():
    """Test quantum-inspired task planning"""
    try:
        print("Testing quantum-inspired planning...")
        from photon_memristor_sim.quantum_planning import QuantumTaskPlanner, PhotonicTaskPlannerFactory
        
        # Create task planner
        planner = PhotonicTaskPlannerFactory.create_planner(
            planner_type="quantum_annealing",
            num_qubits=16
        )
        
        # Define simple optimization problem
        tasks = [
            {"id": "photonic_matmul", "complexity": 5, "optical_power": 10e-3},
            {"id": "memristor_update", "complexity": 3, "optical_power": 5e-3},
            {"id": "thermal_analysis", "complexity": 7, "optical_power": 15e-3},
        ]
        
        # Plan task execution
        plan = planner.plan_tasks(
            tasks=tasks,
            constraints={"max_power": 25e-3, "max_parallel": 2}
        )
        
        assert len(plan.task_assignments) <= len(tasks), "Too many task assignments"
        assert plan.estimated_execution_time > 0, "Execution time should be positive"
        
        print(f"✓ Quantum planning successful ({len(plan.task_assignments)} assignments, {plan.estimated_execution_time:.2f}s estimated)")
        return True
    except Exception as e:
        print(f"✗ Quantum planning test failed: {e}")
        return False

def test_jax_integration():
    """Test JAX integration for differentiable operations"""
    try:
        print("Testing JAX integration...")
        from photon_memristor_sim.jax_interface import photonic_matmul, create_photonic_primitive
        import jax.numpy as jnp
        
        # Test matrix multiplication
        inputs = jnp.array([0.001, 0.002, 0.001])  # 1mW, 2mW, 1mW
        weights = jnp.array([
            [0.8, 0.7, 0.9],
            [0.6, 0.8, 0.5]
        ])
        
        # Forward pass
        outputs = photonic_matmul(inputs, weights, wavelength=1550e-9)
        
        assert outputs.shape == (2,), f"Output shape mismatch: {outputs.shape}"
        assert jnp.all(outputs > 0), "Outputs should be positive"
        
        # Test gradient computation
        from jax import grad
        def loss_fn(w):
            return jnp.sum(photonic_matmul(inputs, w, wavelength=1550e-9) ** 2)
        
        grad_fn = grad(loss_fn)
        gradients = grad_fn(weights)
        
        assert gradients.shape == weights.shape, "Gradient shape mismatch"
        assert jnp.any(gradients != 0), "Gradients should be non-zero"
        
        print(f"✓ JAX integration working (outputs: {outputs}, grad_norm: {jnp.linalg.norm(gradients):.4f})")
        return True
    except Exception as e:
        print(f"✗ JAX integration test failed: {e}")
        return False

def run_generation1_tests():
    """Run all Generation 1 tests"""
    print("=" * 60)
    print("TERRAGON SDLC - GENERATION 1 VERIFICATION")
    print("Testing basic photonic simulation functionality...")
    print("=" * 60)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Wavelength Utilities", test_wavelength_utilities), 
        ("Device Models", test_device_models),
        ("Neural Networks", test_neural_network),
        ("Quantum Planning", test_quantum_planning),
        ("JAX Integration", test_jax_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print("GENERATION 1 TEST RESULTS")
    print("=" * 60)
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"TOTAL:  {passed + failed}")
    
    if failed == 0:
        print("\n🚀 GENERATION 1 COMPLETE - ALL TESTS PASSED!")
        print("Ready to proceed to Generation 2 (Robust)")
        return True
    else:
        print(f"\n⚠️  GENERATION 1 INCOMPLETE - {failed} tests failed")
        print("Fix issues before proceeding to Generation 2")
        return False

if __name__ == "__main__":
    success = run_generation1_tests()
    sys.exit(0 if success else 1)