"""
Optimized performance tests with better timing and efficiency
"""

import time
import numpy as np
import pytest
from photon_memristor_sim.performance_optimizer import (
    OptimizedPhotonic, PerformanceProfiler, IntelligentCache,
    BatchProcessor, get_optimizer
)


class TestOptimizedPerformance:
    """Optimized performance test suite"""
    
    def test_fast_array_processing(self):
        """Test optimized array processing speed"""
        start_time = time.time()
        
        # Create test arrays efficiently  
        arrays = [np.random.rand(100, 100) for _ in range(10)]  # Smaller arrays
        
        def simple_operation(batch):
            return [arr * 2.0 for arr in batch]
        
        optimizer = get_optimizer()
        
        # Process with optimization
        results = optimizer.optimized_simulation(arrays, simple_operation)
        
        elapsed = time.time() - start_time
        
        # Should complete very quickly
        assert elapsed < 0.1, f"Operation took {elapsed:.3f}s, expected < 0.1s"
        assert len(results) == len(arrays)
        
    def test_intelligent_cache_performance(self):
        """Test cache performance"""
        start_time = time.time()
        
        cache = IntelligentCache(max_size=100)
        
        # Fill cache with small operations
        for i in range(50):
            key = f"test_key_{i}"
            value = np.random.rand(10, 10)  # Small arrays
            cache.put(key, value)
        
        # Test cache hits
        hit_count = 0
        for i in range(50):
            key = f"test_key_{i}"
            result = cache.get(key)
            if result is not None:
                hit_count += 1
        
        elapsed = time.time() - start_time
        
        assert elapsed < 0.05, f"Cache operations took {elapsed:.3f}s, expected < 0.05s"
        assert hit_count >= 45, f"Only {hit_count}/50 cache hits"
        
    def test_batch_processor_speed(self):
        """Test batch processing speed"""
        start_time = time.time()
        
        processor = BatchProcessor(batch_size=8)  # Smaller batches
        
        # Create test data
        data = [np.random.rand(50, 50) for _ in range(32)]  # Moderate size
        
        def fast_operation(batch):
            return [arr.sum() for arr in batch]
        
        batches = processor.create_batches(data, batch_size=8)
        results = processor.process_batches_sync(batches, fast_operation)
        
        elapsed = time.time() - start_time
        
        assert elapsed < 0.1, f"Batch processing took {elapsed:.3f}s, expected < 0.1s"
        assert len(results) > 0
        
    def test_profiler_overhead(self):
        """Test that profiler has minimal overhead"""
        profiler = PerformanceProfiler()
        
        # Test without profiling
        start_time = time.time()
        for _ in range(100):
            result = np.random.rand(10, 10).sum()
        baseline_time = time.time() - start_time
        
        # Test with profiling
        start_time = time.time()
        for i in range(100):
            profiler.start_profile(f"test_{i}")
            result = np.random.rand(10, 10).sum()
            profiler.end_profile(f"test_{i}")
        profiled_time = time.time() - start_time
        
        # Profiling overhead should be minimal
        overhead_ratio = profiled_time / baseline_time
        assert overhead_ratio < 3.0, f"Profiler overhead too high: {overhead_ratio:.2f}x"
        
    def test_memory_optimization(self):
        """Test memory optimization doesn't slow things down"""
        from photon_memristor_sim.performance_optimizer import MemoryOptimizer
        
        start_time = time.time()
        
        # Create non-contiguous arrays
        arrays = []
        for _ in range(20):
            arr = np.random.rand(100, 100)
            # Make non-contiguous
            non_contiguous = arr[::2, ::2]
            arrays.append(non_contiguous)
        
        # Optimize arrays
        optimized = MemoryOptimizer.optimize_arrays(arrays, copy=True)
        
        elapsed = time.time() - start_time
        
        assert elapsed < 0.05, f"Memory optimization took {elapsed:.3f}s, expected < 0.05s"
        assert len(optimized) == len(arrays)
        
        # Verify optimization worked
        for opt_arr in optimized:
            assert opt_arr.flags['C_CONTIGUOUS'], "Array should be C-contiguous"


def test_overall_system_performance():
    """Test overall system performance meets requirements"""
    start_time = time.time()
    
    # Simulate a complete photonic computation workflow
    optimizer = get_optimizer()
    
    # Create input data
    input_arrays = [np.random.rand(50, 50) for _ in range(20)]
    
    def mock_photonic_simulation(batch):
        """Mock photonic simulation that's reasonably fast"""
        results = []
        for arr in batch:
            # Simple operations that represent photonic calculations
            result = np.fft.fft2(arr)
            result = np.abs(result) ** 2
            result = np.real(np.fft.ifft2(result))
            results.append(result)
        return results
    
    # Process through optimizer
    results = optimizer.optimized_simulation(
        input_arrays, 
        mock_photonic_simulation, 
        use_cache=True
    )
    
    elapsed = time.time() - start_time
    
    # Should complete within reasonable time
    assert elapsed < 0.2, f"Full workflow took {elapsed:.3f}s, expected < 0.2s"
    assert len(results) == len(input_arrays)
    
    # Test cached access is faster
    cache_start = time.time()
    cached_results = optimizer.optimized_simulation(
        input_arrays,
        mock_photonic_simulation,
        use_cache=True
    )
    cache_elapsed = time.time() - cache_start
    
    # Cached access should be much faster
    assert cache_elapsed < elapsed / 2, "Cache should provide significant speedup"


if __name__ == "__main__":
    # Run basic performance tests
    test_suite = TestOptimizedPerformance()
    
    print("Running optimized performance tests...")
    
    test_suite.test_fast_array_processing()
    print("✅ Fast array processing test passed")
    
    test_suite.test_intelligent_cache_performance()
    print("✅ Cache performance test passed")
    
    test_suite.test_batch_processor_speed()
    print("✅ Batch processor speed test passed")
    
    test_suite.test_profiler_overhead()
    print("✅ Profiler overhead test passed")
    
    test_suite.test_memory_optimization()
    print("✅ Memory optimization test passed")
    
    test_overall_system_performance()
    print("✅ Overall system performance test passed")
    
    print("All optimized performance tests passed! 🚀")