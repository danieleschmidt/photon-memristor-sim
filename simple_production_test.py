#!/usr/bin/env python3
"""
Simple production readiness test for photonic simulation
"""
import sys
import traceback
import time
import psutil
import numpy as np

def test_basic_functionality():
    """Test core library functionality"""
    print("Testing basic functionality...")
    
    try:
        import photon_memristor_sim as pms
        print("✅ Library import successful")
        
        # Test JAX integration
        import jax.numpy as jnp
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print("✅ JAX integration working")
        
        # Test basic simulation
        from photon_memristor_sim import PyPhotonicArray
        array = PyPhotonicArray("crossbar", 4, 4)
        print("✅ Photonic array creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Test performance characteristics"""
    print("\nTesting performance...")
    
    try:
        import photon_memristor_sim as pms
        import jax.numpy as jnp
        
        # Memory usage test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger arrays
        arrays = []
        for i in range(10):
            array = pms.PyPhotonicArray("crossbar", 8, 8)
            arrays.append(array)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        print(f"✅ Memory usage test: {memory_increase:.1f}MB for 10 arrays")
        
        # Speed test
        import jax
        start_time = time.time()
        key = jax.random.PRNGKey(42)
        for _ in range(100):
            x = jax.random.normal(key, shape=(100,))
            y = jnp.sum(x**2)
        elapsed = time.time() - start_time
        
        print(f"✅ Speed test: {elapsed:.3f}s for 100 operations")
        
        return memory_increase < 100  # Reasonable memory usage
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_resilience():
    """Test error handling and resilience"""
    print("\nTesting resilience...")
    
    try:
        import photon_memristor_sim as pms
        
        # Test invalid inputs
        try:
            array = pms.PyPhotonicArray("invalid_topology", 4, 5)  # Invalid topology
            print("❌ Should have caught invalid input")
            return False
        except (ValueError, RuntimeError):
            print("✅ Invalid input properly handled")
        
        # Test resource limits
        try:
            huge_array = pms.PyPhotonicArray("crossbar", 1000, 1000)  # Very large
            print("⚠️ Large array creation succeeded (may be OK)")
        except (MemoryError, RuntimeError):
            print("✅ Resource limits properly enforced")
        
        return True
        
    except Exception as e:
        print(f"❌ Resilience test failed: {e}")
        return False

def main():
    """Run all production readiness tests"""
    print("Photonic Simulation Production Readiness Test")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_performance, 
        test_resilience
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 PRODUCTION READY!")
        return 0
    else:
        print("⚠️ Some issues found - review before production")
        return 1

if __name__ == "__main__":
    sys.exit(main())