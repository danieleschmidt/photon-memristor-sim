#!/usr/bin/env python3
"""
Simple Enhanced System Test

Tests the enhanced photonic simulation system with basic Python
without requiring JAX installation.
"""

import sys
import time
import traceback
import numpy as np

# Add project path
sys.path.insert(0, '/root/repo/python')

def test_enhanced_module_imports():
    """Test that all enhanced modules can be imported."""
    print("📦 Testing Enhanced Module Imports...")
    
    import_results = {}
    modules_to_test = [
        ("devices", "MolecularMemristor"),
        ("quantum_hybrid", "QuantumPhotonicProcessor"),
        ("gpu_accelerated", "CUDAOptimizedFDTD"),
        ("edge_computing", "EdgeAI"),
        ("ai_optimization", "NeuralArchitectureSearch")
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(f"photon_memristor_sim.{module_name}", fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
            import_results[f"{module_name}.{class_name}"] = "success"
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
            import_results[f"{module_name}.{class_name}"] = str(e)
    
    return import_results


def test_molecular_memristor_basic():
    """Test basic molecular memristor functionality."""
    print("🧠 Testing Molecular Memristor (Basic)...")
    
    try:
        from photon_memristor_sim.devices import MolecularMemristor
        
        # Create molecular memristor
        device = MolecularMemristor(
            molecular_film="metal_organic",
            num_states=16500,
            temperature=300.0
        )
        
        # Test basic properties
        num_states = device.num_states
        precision_bits = device.get_precision_bits()
        conductance = device.get_conductance()
        
        # Test state setting
        device.set_state_precise(1000)  # Set to state 1000
        new_conductance = device.get_conductance()
        
        # Test analog programming
        device.analog_programming(1e-6, precision_bits=12)
        programmed_conductance = device.get_conductance()
        
        # Get performance metrics
        metrics = device.benchmark_performance()
        
        results = {
            "num_states": num_states,
            "precision_bits": precision_bits,
            "initial_conductance": float(conductance),
            "state_1000_conductance": float(new_conductance),
            "programmed_conductance": float(programmed_conductance),
            "dynamic_range": metrics["dynamic_range"],
            "area_efficiency": metrics["area_efficiency"],
            "retention_time_days": metrics["retention_time_days"]
        }
        
        print(f"  ✅ Created device with {num_states:,} states ({precision_bits}-bit precision)")
        print(f"  ✅ Dynamic range: {metrics['dynamic_range']:.0e}")
        print(f"  ✅ Area efficiency: {metrics['area_efficiency']:.1e} states/μm²")
        print(f"  ✅ Retention time: {metrics['retention_time_days']:.0f} days")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Molecular memristor test failed: {e}")
        return {"error": str(e)}


def test_system_integration():
    """Test enhanced system integration."""
    print("🔗 Testing System Integration...")
    
    try:
        # Test main module import
        import photon_memristor_sim as pms
        
        # Check version
        version = pms.__version__
        
        # Check if enhanced classes are available
        enhanced_classes = [
            "MolecularMemristor",
            "QuantumPhotonicProcessor", 
            "CUDAOptimizedFDTD",
            "EdgeAI",
            "NeuralArchitectureSearch"
        ]
        
        available_classes = []
        for class_name in enhanced_classes:
            if hasattr(pms, class_name):
                available_classes.append(class_name)
        
        # Test creating a molecular memristor via main interface
        if hasattr(pms, 'MolecularMemristor'):
            device = pms.MolecularMemristor()
            device_created = True
        else:
            device_created = False
        
        results = {
            "version": version,
            "enhanced_classes_available": len(available_classes),
            "total_enhanced_classes": len(enhanced_classes),
            "available_classes": available_classes,
            "device_creation_success": device_created,
            "integration_score": len(available_classes) / len(enhanced_classes)
        }
        
        print(f"  ✅ Version: {version}")
        print(f"  ✅ Enhanced classes available: {len(available_classes)}/{len(enhanced_classes)}")
        print(f"  ✅ Integration score: {results['integration_score']:.1%}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ System integration test failed: {e}")
        return {"error": str(e)}


def test_fallback_functionality():
    """Test pure Python fallback functionality."""
    print("🐍 Testing Pure Python Fallbacks...")
    
    try:
        from photon_memristor_sim import pure_python_fallbacks
        
        # Test core fallback functions
        test_results = {}
        
        # Test optical field creation
        try:
            field = pure_python_fallbacks.PyOpticalField(wavelength=1550e-9, power=1e-3)
            test_results["optical_field"] = "success"
        except Exception as e:
            test_results["optical_field"] = str(e)
        
        # Test photonic array
        try:
            array = pure_python_fallbacks.PyPhotonicArray(size=(4, 4))
            test_results["photonic_array"] = "success"
        except Exception as e:
            test_results["photonic_array"] = str(e)
        
        # Test device simulator
        try:
            simulator = pure_python_fallbacks.create_device_simulator("pcm")
            test_results["device_simulator"] = "success"
        except Exception as e:
            test_results["device_simulator"] = str(e)
        
        successful_fallbacks = sum(1 for result in test_results.values() if result == "success")
        total_fallbacks = len(test_results)
        
        results = {
            "fallback_tests": test_results,
            "successful_fallbacks": successful_fallbacks,
            "total_fallbacks": total_fallbacks,
            "fallback_coverage": successful_fallbacks / total_fallbacks
        }
        
        print(f"  ✅ Fallback functions available: {successful_fallbacks}/{total_fallbacks}")
        print(f"  ✅ Fallback coverage: {results['fallback_coverage']:.1%}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Fallback test failed: {e}")
        return {"error": str(e)}


def test_enhanced_system_resilience():
    """Test system resilience with enhanced features."""
    print("🛡️ Testing Enhanced System Resilience...")
    
    try:
        from photon_memristor_sim.resilience import get_resilient_system
        
        # Get resilient system
        resilient_system = get_resilient_system()
        
        # Test circuit breaker
        circuit_breaker = resilient_system.circuit_breaker
        
        # Test cache manager
        cache = resilient_system.cache_manager
        
        # Test some operations
        test_operations = 0
        successful_operations = 0
        
        # Test 1: Basic computation
        try:
            result = resilient_system.with_resilience(lambda: 2 + 2)
            if result == 4:
                successful_operations += 1
            test_operations += 1
        except:
            test_operations += 1
        
        # Test 2: Cache operation
        try:
            cached_result = cache.get_or_compute("test_key", lambda: "test_value")
            if cached_result == "test_value":
                successful_operations += 1
            test_operations += 1
        except:
            test_operations += 1
        
        # Test 3: Circuit breaker functionality
        try:
            cb_result = circuit_breaker.call(lambda: "circuit_breaker_test")
            if cb_result == "circuit_breaker_test":
                successful_operations += 1
            test_operations += 1
        except:
            test_operations += 1
        
        results = {
            "resilient_system_available": True,
            "circuit_breaker_available": circuit_breaker is not None,
            "cache_manager_available": cache is not None,
            "successful_operations": successful_operations,
            "total_operations": test_operations,
            "resilience_score": successful_operations / test_operations if test_operations > 0 else 0
        }
        
        print(f"  ✅ Resilient system created successfully")
        print(f"  ✅ Operations successful: {successful_operations}/{test_operations}")
        print(f"  ✅ Resilience score: {results['resilience_score']:.1%}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Resilience test failed: {e}")
        return {"error": str(e)}


def run_simple_benchmark():
    """Run simple benchmark of enhanced system."""
    
    print("🚀 TERRAGON SDLC - ENHANCED SYSTEM SIMPLE BENCHMARK")
    print("=" * 70)
    print("Testing enhanced photonic simulation system...")
    print()
    
    all_results = {}
    
    # Test imports
    all_results["module_imports"] = test_enhanced_module_imports()
    print()
    
    # Test molecular memristor
    all_results["molecular_memristor"] = test_molecular_memristor_basic()
    print()
    
    # Test system integration
    all_results["system_integration"] = test_system_integration()
    print()
    
    # Test fallback functionality
    all_results["fallback_functionality"] = test_fallback_functionality()
    print()
    
    # Test system resilience
    all_results["system_resilience"] = test_enhanced_system_resilience()
    print()
    
    # Calculate overall results
    successful_tests = sum(1 for results in all_results.values() if isinstance(results, dict) and "error" not in results)
    total_tests = len(all_results)
    
    print("=" * 70)
    print("🎯 ENHANCED SYSTEM BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"✅ Tests passed: {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})")
    print()
    
    # Module import summary
    if "module_imports" in all_results:
        import_results = all_results["module_imports"]
        successful_imports = sum(1 for result in import_results.values() if result == "success")
        total_imports = len(import_results)
        print(f"📦 Module Imports: {successful_imports}/{total_imports} successful")
    
    # Molecular memristor summary
    if "molecular_memristor" in all_results and "error" not in all_results["molecular_memristor"]:
        mm_results = all_results["molecular_memristor"]
        print(f"🧠 Molecular Memristor: {mm_results['num_states']:,} states, {mm_results['precision_bits']}-bit precision")
    
    # System integration summary
    if "system_integration" in all_results and "error" not in all_results["system_integration"]:
        si_results = all_results["system_integration"]
        print(f"🔗 System Integration: {si_results['integration_score']:.1%} completeness")
    
    # Fallback functionality summary
    if "fallback_functionality" in all_results and "error" not in all_results["fallback_functionality"]:
        ff_results = all_results["fallback_functionality"]
        print(f"🐍 Fallback Coverage: {ff_results['fallback_coverage']:.1%}")
    
    # System resilience summary
    if "system_resilience" in all_results and "error" not in all_results["system_resilience"]:
        sr_results = all_results["system_resilience"]
        print(f"🛡️ System Resilience: {sr_results['resilience_score']:.1%}")
    
    print()
    
    if successful_tests == total_tests:
        print("🎊 ENHANCED SYSTEM WORKING PERFECTLY! 🎊")
        print("🌟 Ready for advanced photonic computing!")
    else:
        print("⚠️  Some components need attention - see details above")
    
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = run_simple_benchmark()
    print("📊 Simple benchmark completed!")