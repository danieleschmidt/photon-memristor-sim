#!/usr/bin/env python3
"""
Scalable Photonic Computation - Generation 3 Features

This example demonstrates high-performance optimization, caching, 
and parallel processing capabilities for large-scale photonic 
neural network simulations.
"""

import numpy as np
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Tuple
import multiprocessing as mp
import psutil
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from python.photon_memristor_sim.quantum_planning import (
        QuantumTaskPlanner,
        PhotonicTaskPlannerFactory,
    )
except ImportError:
    print("Warning: Running in standalone mode - some features may not be available")

@dataclass
class PerformanceMetrics:
    """Performance metrics for scalability analysis."""
    execution_time: float
    throughput: float  # operations per second
    memory_usage: float  # MB
    cpu_utilization: float  # percentage
    cache_hit_rate: float  # percentage
    parallel_efficiency: float  # percentage

class InMemoryCache:
    """High-performance in-memory cache implementation."""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.access_count = {}
        self.access_time = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, operation: str, params: Tuple) -> str:
        """Create cache key from operation and parameters."""
        return f"{operation}:{hash(params)}"
    
    def get(self, operation: str, params: Tuple) -> Any:
        """Get cached result if available."""
        key = self._make_key(operation, params)
        
        if key in self.cache:
            self.hits += 1
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.access_time[key] = time.time()
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, operation: str, params: Tuple, result: Any) -> None:
        """Store result in cache."""
        key = self._make_key(operation, params)
        
        # Evict least recently used items if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = result
        self.access_count[key] = 1
        self.access_time[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used cache entries."""
        if not self.access_time:
            return
        
        # Find 10% of entries to evict
        evict_count = max(1, len(self.cache) // 10)
        
        # Sort by access time (oldest first)
        sorted_by_time = sorted(self.access_time.items(), key=lambda x: x[1])
        
        for key, _ in sorted_by_time[:evict_count]:
            self.cache.pop(key, None)
            self.access_count.pop(key, None)
            self.access_time.pop(key, None)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate(),
            'size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_count.clear()
        self.access_time.clear()
        self.hits = 0
        self.misses = 0

class ParallelPhotonicSimulator:
    """High-performance parallel photonic simulation engine."""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.cache = InMemoryCache(max_size=50000)
        self.performance_history = []
        
        print(f"🚀 Initialized parallel simulator with {self.num_workers} workers")
    
    def simulate_optical_propagation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate optical field propagation with caching."""
        # Create cache key from parameters
        cache_params = (
            params['wavelength'],
            params['power'],
            params['num_devices'],
            tuple(params.get('device_config', []))
        )
        
        # Check cache first
        cached_result = self.cache.get('optical_propagation', cache_params)
        if cached_result is not None:
            return cached_result
        
        # Simulate computation time based on problem size
        computation_time = params['num_devices'] * 0.001 + np.random.exponential(0.01)
        time.sleep(computation_time)
        
        # Generate realistic simulation results
        num_devices = params['num_devices']
        field_intensity = np.random.random(num_devices) * params['power']
        
        # Add some physics-based correlations
        for i in range(1, num_devices):
            field_intensity[i] += field_intensity[i-1] * 0.1  # Coupling effect
        
        result = {
            'field_intensity': field_intensity,
            'total_power': np.sum(field_intensity),
            'max_intensity': np.max(field_intensity),
            'phase_distribution': np.random.random(num_devices) * 2 * np.pi,
            'computation_time': computation_time,
            'wavelength': params['wavelength'],
            'coupling_efficiency': 0.85 + np.random.random() * 0.1
        }
        
        # Cache the result
        self.cache.put('optical_propagation', cache_params, result)
        
        return result
    
    def parallel_device_simulation(self, device_configs: List[Dict]) -> List[Dict]:
        """Simulate multiple photonic devices in parallel."""
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"🔄 Simulating {len(device_configs)} devices in parallel...")
        
        # Use thread pool for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.simulate_optical_propagation, device_configs))
        
        end_time = time.time()
        final_memory = self._get_memory_usage()
        
        # Calculate performance metrics
        execution_time = end_time - start_time
        throughput = len(device_configs) / execution_time
        memory_delta = final_memory - initial_memory
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=memory_delta,
            cpu_utilization=self._get_cpu_usage(),
            cache_hit_rate=self.cache.hit_rate(),
            parallel_efficiency=self._calculate_parallel_efficiency(len(device_configs), execution_time)
        )
        
        self.performance_history.append(metrics)
        
        print(f"✅ Completed simulation:")
        print(f"   - Time: {execution_time:.2f}s")
        print(f"   - Throughput: {throughput:.1f} devices/sec")
        print(f"   - Memory usage: {memory_delta:.1f}MB")
        print(f"   - Cache hit rate: {metrics.cache_hit_rate:.1f}%")
        print(f"   - Parallel efficiency: {metrics.parallel_efficiency:.1f}%")
        
        return results
    
    def process_batch_simulation(self, batch_configs: List[List[Dict]]) -> List[List[Dict]]:
        """Process multiple batches using process-based parallelism."""
        start_time = time.time()
        
        print(f"🔄 Processing {len(batch_configs)} batches with process parallelism...")
        
        # Use process pool for CPU-intensive operations
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            batch_results = list(executor.map(self._process_single_batch, batch_configs))
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        total_devices = sum(len(batch) for batch in batch_configs)
        throughput = total_devices / execution_time
        
        print(f"✅ Batch processing completed:")
        print(f"   - Total time: {execution_time:.2f}s")
        print(f"   - Total devices: {total_devices}")
        print(f"   - Throughput: {throughput:.1f} devices/sec")
        
        return batch_results
    
    def _process_single_batch(self, device_configs: List[Dict]) -> List[Dict]:
        """Process a single batch of device configurations."""
        # Create a new cache for this process
        local_cache = InMemoryCache(max_size=1000)
        
        results = []
        for config in device_configs:
            # Simple simulation without actual threading (for subprocess)
            result = self._simple_device_simulation(config, local_cache)
            results.append(result)
        
        return results
    
    def _simple_device_simulation(self, params: Dict, cache: InMemoryCache) -> Dict:
        """Simplified device simulation for subprocess execution."""
        cache_params = (
            params['wavelength'],
            params['power'],
            params['num_devices']
        )
        
        cached_result = cache.get('simple_simulation', cache_params)
        if cached_result is not None:
            return cached_result
        
        # Simulate some computation
        time.sleep(params['num_devices'] * 0.0005)
        
        result = {
            'intensity': np.random.random() * params['power'],
            'phase': np.random.random() * 2 * np.pi,
            'wavelength': params['wavelength'],
            'efficiency': 0.8 + np.random.random() * 0.15
        }
        
        cache.put('simple_simulation', cache_params, result)
        return result
    
    def adaptive_load_balancing(self, device_configs: List[Dict]) -> List[Dict]:
        """Implement adaptive load balancing based on device complexity."""
        start_time = time.time()
        
        print(f"🎯 Adaptive load balancing for {len(device_configs)} devices...")
        
        # Classify devices by complexity
        simple_devices = []
        complex_devices = []
        
        for config in device_configs:
            complexity_score = config['num_devices'] * config.get('complexity_factor', 1.0)
            if complexity_score < 100:
                simple_devices.append(config)
            else:
                complex_devices.append(config)
        
        print(f"   - Simple devices: {len(simple_devices)}")
        print(f"   - Complex devices: {len(complex_devices)}")
        
        # Process simple devices with high parallelism
        simple_results = []
        if simple_devices:
            with ThreadPoolExecutor(max_workers=self.num_workers * 2) as executor:
                simple_results = list(executor.map(self.simulate_optical_propagation, simple_devices))
        
        # Process complex devices with lower parallelism to avoid resource contention
        complex_results = []
        if complex_devices:
            with ThreadPoolExecutor(max_workers=max(1, self.num_workers // 2)) as executor:
                complex_results = list(executor.map(self.simulate_optical_propagation, complex_devices))
        
        # Combine results maintaining original order
        all_results = []
        simple_idx = complex_idx = 0
        
        for config in device_configs:
            complexity_score = config['num_devices'] * config.get('complexity_factor', 1.0)
            if complexity_score < 100:
                all_results.append(simple_results[simple_idx])
                simple_idx += 1
            else:
                all_results.append(complex_results[complex_idx])
                complex_idx += 1
        
        execution_time = time.time() - start_time
        throughput = len(device_configs) / execution_time
        
        print(f"✅ Adaptive processing completed:")
        print(f"   - Time: {execution_time:.2f}s")
        print(f"   - Throughput: {throughput:.1f} devices/sec")
        
        return all_results
    
    def memory_efficient_streaming(self, device_generator) -> List[Dict]:
        """Process devices using memory-efficient streaming."""
        print("📊 Starting memory-efficient streaming simulation...")
        
        results = []
        batch_size = 100  # Process in chunks to manage memory
        current_batch = []
        
        for device_config in device_generator:
            current_batch.append(device_config)
            
            if len(current_batch) >= batch_size:
                # Process current batch
                batch_results = self.parallel_device_simulation(current_batch)
                results.extend(batch_results)
                current_batch.clear()
                
                # Optional: Force garbage collection
                import gc
                gc.collect()
        
        # Process remaining devices
        if current_batch:
            batch_results = self.parallel_device_simulation(current_batch)
            results.extend(batch_results)
        
        print(f"📊 Streaming completed: processed {len(results)} devices")
        return results
    
    def benchmark_scaling(self, device_counts: List[int]) -> Dict[str, List[float]]:
        """Benchmark performance scaling with different device counts."""
        print("📈 Running scaling benchmark...")
        
        execution_times = []
        throughputs = []
        memory_usages = []
        cache_hit_rates = []
        
        for count in device_counts:
            print(f"   Testing with {count} devices...")
            
            # Generate test configurations
            configs = self._generate_test_configs(count)
            
            # Clear cache for fair comparison
            self.cache.clear()
            
            # Run simulation
            start_time = time.time()
            initial_memory = self._get_memory_usage()
            
            results = self.parallel_device_simulation(configs)
            
            end_time = time.time()
            final_memory = self._get_memory_usage()
            
            # Record metrics
            execution_time = end_time - start_time
            throughput = count / execution_time
            memory_usage = final_memory - initial_memory
            
            execution_times.append(execution_time)
            throughputs.append(throughput)
            memory_usages.append(memory_usage)
            cache_hit_rates.append(self.cache.hit_rate())
        
        return {
            'device_counts': device_counts,
            'execution_times': execution_times,
            'throughputs': throughputs,
            'memory_usages': memory_usages,
            'cache_hit_rates': cache_hit_rates
        }
    
    def _generate_test_configs(self, count: int) -> List[Dict]:
        """Generate test device configurations."""
        configs = []
        
        for i in range(count):
            config = {
                'wavelength': 1550e-9 + np.random.random() * 100e-9,
                'power': np.random.random() * 10e-3,  # 0-10mW
                'num_devices': np.random.randint(1, 20),
                'complexity_factor': 0.5 + np.random.random() * 1.5,
                'device_config': [np.random.random() for _ in range(5)]
            }
            configs.append(config)
        
        return configs
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def _calculate_parallel_efficiency(self, num_tasks: int, execution_time: float) -> float:
        """Calculate parallel efficiency as percentage of ideal speedup."""
        # Estimate sequential time (rough approximation)
        estimated_sequential_time = num_tasks * 0.01  # Assume 10ms per task sequentially
        
        # Ideal parallel time with perfect scaling
        ideal_parallel_time = estimated_sequential_time / self.num_workers
        
        # Parallel efficiency
        efficiency = (ideal_parallel_time / execution_time) * 100
        return min(100.0, max(0.0, efficiency))  # Clamp to 0-100%
    
    def visualize_performance(self, benchmark_data: Dict) -> None:
        """Create performance visualization charts."""
        print("📊 Creating performance visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        device_counts = benchmark_data['device_counts']
        
        # Execution time vs device count
        ax1.plot(device_counts, benchmark_data['execution_times'], 'b-o', linewidth=2)
        ax1.set_xlabel('Number of Devices')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time Scaling')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Throughput vs device count
        ax2.plot(device_counts, benchmark_data['throughputs'], 'g-o', linewidth=2)
        ax2.set_xlabel('Number of Devices')
        ax2.set_ylabel('Throughput (devices/sec)')
        ax2.set_title('Throughput Scaling')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Memory usage vs device count
        ax3.plot(device_counts, benchmark_data['memory_usages'], 'r-o', linewidth=2)
        ax3.set_xlabel('Number of Devices')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage Scaling')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Cache hit rate vs device count
        ax4.plot(device_counts, benchmark_data['cache_hit_rates'], 'm-o', linewidth=2)
        ax4.set_xlabel('Number of Devices')
        ax4.set_ylabel('Cache Hit Rate (%)')
        ax4.set_title('Cache Performance')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('photonic_scaling_performance.png', dpi=300, bbox_inches='tight')
        print("📊 Performance charts saved to photonic_scaling_performance.png")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.performance_history[-10:]  # Last 10 measurements
        
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        avg_parallel_efficiency = np.mean([m.parallel_efficiency for m in recent_metrics])
        
        cache_stats = self.cache.stats()
        
        return {
            'average_throughput': avg_throughput,
            'average_memory_usage': avg_memory,
            'average_cache_hit_rate': avg_cache_hit_rate,
            'average_parallel_efficiency': avg_parallel_efficiency,
            'total_simulations': len(self.performance_history),
            'cache_statistics': cache_stats,
            'worker_count': self.num_workers
        }

def device_config_generator(count: int):
    """Generator for memory-efficient device configuration creation."""
    for i in range(count):
        yield {
            'wavelength': 1550e-9 + (i % 100) * 1e-9,  # Vary wavelength
            'power': (1 + i % 10) * 1e-3,  # 1-10mW
            'num_devices': 5 + (i % 15),  # 5-20 devices
            'complexity_factor': 0.5 + (i % 10) * 0.15,
            'device_config': [0.1 * (j + i % 5) for j in range(3)]
        }

def run_comprehensive_scaling_test():
    """Run comprehensive scaling and performance tests."""
    print("🚀 GENERATION 3: SCALABLE PHOTONIC COMPUTATION")
    print("High-Performance Optimization, Caching & Parallel Processing")
    print("=" * 70)
    
    # Initialize parallel simulator
    simulator = ParallelPhotonicSimulator()
    
    # Test 1: Basic parallel processing
    print("\n🧪 Test 1: Basic Parallel Processing")
    print("-" * 40)
    
    device_configs = simulator._generate_test_configs(500)
    results = simulator.parallel_device_simulation(device_configs)
    
    print(f"✅ Processed {len(results)} devices successfully")
    
    # Test 2: Process-based batch simulation
    print("\n🧪 Test 2: Process-Based Batch Simulation")
    print("-" * 40)
    
    # Create batches for process-based parallelism
    batch_size = 50
    batches = [device_configs[i:i+batch_size] for i in range(0, min(200, len(device_configs)), batch_size)]
    
    batch_results = simulator.process_batch_simulation(batches)
    total_processed = sum(len(batch) for batch in batch_results)
    
    print(f"✅ Processed {total_processed} devices in {len(batch_results)} batches")
    
    # Test 3: Adaptive load balancing
    print("\n🧪 Test 3: Adaptive Load Balancing")
    print("-" * 40)
    
    # Create mixed complexity configurations
    mixed_configs = []
    for i in range(100):
        complexity = 1.0 if i % 3 == 0 else 0.3  # Mix of simple and complex
        config = {
            'wavelength': 1550e-9,
            'power': 5e-3,
            'num_devices': int(20 * complexity) + 5,
            'complexity_factor': complexity,
            'device_config': [0.1, 0.2, 0.3]
        }
        mixed_configs.append(config)
    
    adaptive_results = simulator.adaptive_load_balancing(mixed_configs)
    print(f"✅ Adaptive processing completed for {len(adaptive_results)} devices")
    
    # Test 4: Memory-efficient streaming
    print("\n🧪 Test 4: Memory-Efficient Streaming")
    print("-" * 40)
    
    device_generator = device_config_generator(300)
    streaming_results = simulator.memory_efficient_streaming(device_generator)
    
    print(f"✅ Streaming processed {len(streaming_results)} devices")
    
    # Test 5: Scaling benchmark
    print("\n🧪 Test 5: Performance Scaling Benchmark")
    print("-" * 40)
    
    device_counts = [50, 100, 200, 500, 1000, 2000]
    benchmark_data = simulator.benchmark_scaling(device_counts)
    
    print("📊 Scaling Benchmark Results:")
    for i, count in enumerate(device_counts):
        print(f"   {count:4d} devices: "
              f"{benchmark_data['execution_times'][i]:6.2f}s, "
              f"{benchmark_data['throughputs'][i]:8.1f} dev/sec, "
              f"{benchmark_data['memory_usages'][i]:6.1f}MB, "
              f"{benchmark_data['cache_hit_rates'][i]:5.1f}% cache")
    
    # Test 6: Cache performance analysis
    print("\n🧪 Test 6: Cache Performance Analysis")
    print("-" * 40)
    
    cache_stats = simulator.cache.stats()
    print(f"Cache Statistics:")
    print(f"   - Total hits: {cache_stats['hits']}")
    print(f"   - Total misses: {cache_stats['misses']}")
    print(f"   - Hit rate: {cache_stats['hit_rate']:.1f}%")
    print(f"   - Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    
    # Performance summary
    print("\n📊 Overall Performance Summary")
    print("-" * 40)
    
    summary = simulator.get_performance_summary()
    print(f"Average Throughput: {summary['average_throughput']:.1f} devices/sec")
    print(f"Average Memory Usage: {summary['average_memory_usage']:.1f}MB")
    print(f"Average Cache Hit Rate: {summary['average_cache_hit_rate']:.1f}%")
    print(f"Average Parallel Efficiency: {summary['average_parallel_efficiency']:.1f}%")
    print(f"Total Simulations: {summary['total_simulations']}")
    print(f"Worker Threads: {summary['worker_count']}")
    
    # Create visualizations
    try:
        simulator.visualize_performance(benchmark_data)
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")
    
    print(f"\n🎯 Generation 3 Features Demonstrated:")
    print("✅ Multi-threaded parallel processing with ThreadPoolExecutor")
    print("✅ Multi-process parallelism with ProcessPoolExecutor")
    print("✅ High-performance in-memory caching with LRU eviction")
    print("✅ Adaptive load balancing based on task complexity")
    print("✅ Memory-efficient streaming for large datasets")
    print("✅ Performance scaling benchmarks and analysis")
    print("✅ Real-time memory and CPU usage monitoring")
    print("✅ Cache hit rate optimization and statistics")
    print("✅ Parallel efficiency measurements")
    print("✅ Performance visualization and reporting")
    
    return simulator

def main():
    """Main demonstration function."""
    try:
        simulator = run_comprehensive_scaling_test()
        
        print("\n🎉 Generation 3 demonstration completed successfully!")
        print(f"Peak performance: {max([m.throughput for m in simulator.performance_history]):.1f} devices/sec")
        
        return 0
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)