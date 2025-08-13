#!/usr/bin/env python3
"""
Generation 3 Scalable Demo: MAKE IT SCALE
Performance optimization, caching, concurrency, and auto-scaling.
"""

import numpy as np
import time
import logging
import asyncio
import threading
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache, wraps
import gc
import psutil
import weakref

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class CacheStrategy(Enum):
    """Cache replacement strategies"""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"

@dataclass
class PerformanceProfile:
    """Performance profiling data"""
    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    memory_usage: List[float] = field(default_factory=list)
    cache_stats: Dict[str, int] = field(default_factory=dict)
    throughput_history: List[float] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)

@dataclass
class ScalingMetrics:
    """Auto-scaling metrics"""
    cpu_usage_percent: float
    memory_usage_percent: float
    throughput_ops_per_sec: float
    latency_p95_ms: float
    queue_depth: int
    active_threads: int

class AdaptiveCache:
    """High-performance adaptive cache with multiple strategies"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.data = {}
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.insertion_order = deque()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive parameters
        self.lru_weight = 0.5
        self.lfu_weight = 0.5
        
    def _hash_key(self, key: Any) -> str:
        """Create hash key from arbitrary input"""
        if isinstance(key, (str, int, float)):
            return str(key)
        elif isinstance(key, np.ndarray):
            return hashlib.md5(key.tobytes()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        hash_key = self._hash_key(key)
        
        if hash_key in self.data:
            self.hits += 1
            self.access_counts[hash_key] += 1
            self.access_times[hash_key] = time.time()
            return self.data[hash_key]
        else:
            self.misses += 1
            return None
    
    def put(self, key: Any, value: Any):
        """Put value in cache"""
        hash_key = self._hash_key(key)
        
        if len(self.data) >= self.max_size and hash_key not in self.data:
            self._evict()
        
        self.data[hash_key] = value
        self.access_counts[hash_key] += 1
        self.access_times[hash_key] = time.time()
        
        if hash_key not in self.insertion_order:
            self.insertion_order.append(hash_key)
    
    def _evict(self):
        """Evict item based on strategy"""
        if not self.data:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        else:  # ADAPTIVE
            # Adaptive strategy combining LRU and LFU
            current_time = time.time()
            scores = {}
            
            for key in self.data.keys():
                recency_score = 1.0 / (current_time - self.access_times.get(key, 0) + 1)
                frequency_score = self.access_counts.get(key, 0)
                
                scores[key] = (
                    self.lru_weight * recency_score + 
                    self.lfu_weight * frequency_score
                )
            
            oldest_key = min(scores.keys(), key=lambda k: scores[k])
        
        # Remove from all structures
        del self.data[oldest_key]
        del self.access_counts[oldest_key]
        del self.access_times[oldest_key]
        if oldest_key in self.insertion_order:
            self.insertion_order.remove(oldest_key)
        
        self.evictions += 1
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "size": len(self.data),
            "max_size": self.max_size
        }

class ParallelExecutor:
    """High-performance parallel execution engine"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
    def map_threaded(self, func: Callable, items: List[Any]) -> List[Any]:
        """Parallel map using threads (good for I/O bound tasks)"""
        futures = [self.thread_pool.submit(func, item) for item in items]
        return [future.result() for future in as_completed(futures)]
    
    def map_processes(self, func: Callable, items: List[Any]) -> List[Any]:
        """Parallel map using processes (good for CPU bound tasks)"""
        # Note: func must be pickleable for multiprocessing
        futures = [self.process_pool.submit(func, item) for item in items]
        return [future.result() for future in as_completed(futures)]
    
    def batch_process(self, func: Callable, items: List[Any], batch_size: int = 100) -> List[Any]:
        """Process items in batches for better memory efficiency"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = self.map_threaded(func, batch)
            results.extend(batch_results)
            
            # Yield control and allow garbage collection
            time.sleep(0.001)
            gc.collect()
        
        return results
    
    def shutdown(self):
        """Shutdown executor pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class MemoryOptimizer:
    """Memory optimization and management"""
    
    def __init__(self):
        self.memory_threshold = 80.0  # 80% memory usage threshold
        self.cleanup_callbacks = []
        
    def register_cleanup(self, callback: Callable):
        """Register cleanup callback for memory pressure"""
        self.cleanup_callbacks.append(callback)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def optimize_memory(self, force: bool = False):
        """Optimize memory usage"""
        current_usage = self.get_memory_usage()
        
        if current_usage > self.memory_threshold or force:
            logging.info(f"Memory usage {current_usage:.1f}% - triggering cleanup")
            
            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logging.error(f"Cleanup callback failed: {e}")
            
            # Force garbage collection
            gc.collect()
            
            new_usage = self.get_memory_usage()
            logging.info(f"Memory optimized: {current_usage:.1f}% -> {new_usage:.1f}%")

class AutoScaler:
    """Automatic scaling based on performance metrics"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 16):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.metrics_history = deque(maxlen=10)
        
    def update_metrics(self, metrics: ScalingMetrics):
        """Update scaling metrics"""
        self.metrics_history.append(metrics)
    
    def should_scale_up(self) -> bool:
        """Determine if we should scale up"""
        if len(self.metrics_history) < 3:
            return False
        
        recent_metrics = list(self.metrics_history)[-3:]
        
        # Scale up if CPU > 80% and latency > 100ms consistently
        high_cpu = all(m.cpu_usage_percent > 80 for m in recent_metrics)
        high_latency = all(m.latency_p95_ms > 100 for m in recent_metrics)
        
        return high_cpu and high_latency and self.current_workers < self.max_workers
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down"""
        if len(self.metrics_history) < 5:
            return False
        
        recent_metrics = list(self.metrics_history)[-5:]
        
        # Scale down if CPU < 30% and latency < 50ms consistently
        low_cpu = all(m.cpu_usage_percent < 30 for m in recent_metrics)
        low_latency = all(m.latency_p95_ms < 50 for m in recent_metrics)
        
        return low_cpu and low_latency and self.current_workers > self.min_workers
    
    def auto_scale(self) -> Optional[int]:
        """Perform automatic scaling decision"""
        if self.should_scale_up():
            self.current_workers = min(self.current_workers + 1, self.max_workers)
            return self.current_workers
        elif self.should_scale_down():
            self.current_workers = max(self.current_workers - 1, self.min_workers)
            return self.current_workers
        
        return None

class OptimizedPhotonicDevice:
    """High-performance optimized photonic device"""
    
    def __init__(self, rows: int = 32, cols: int = 32, device_id: str = "opt_device"):
        self.device_id = device_id
        self.rows = rows
        self.cols = cols
        self.wavelength = 1550e-9
        
        # Optimized data structures
        self.transmission_matrix = np.random.uniform(0.1, 0.9, (rows, cols)).astype(np.float32)
        
        # Performance optimization components
        self.cache = AdaptiveCache(max_size=1000, strategy=CacheStrategy.ADAPTIVE)
        self.executor = ParallelExecutor()
        self.memory_optimizer = MemoryOptimizer()
        self.performance_profile = PerformanceProfile()
        
        # Register memory cleanup
        self.memory_optimizer.register_cleanup(self._cleanup_cache)
        
        self.logger = logging.getLogger(f"OptDevice.{device_id}")
        
    def _cleanup_cache(self):
        """Cache cleanup for memory optimization"""
        if hasattr(self.cache, 'data'):
            original_size = len(self.cache.data)
            # Clear 50% of cache entries
            items_to_remove = original_size // 2
            
            keys_to_remove = list(self.cache.data.keys())[:items_to_remove]
            for key in keys_to_remove:
                if key in self.cache.data:
                    del self.cache.data[key]
                if key in self.cache.access_counts:
                    del self.cache.access_counts[key]
                if key in self.cache.access_times:
                    del self.cache.access_times[key]
            
            self.logger.info(f"Cache cleanup: {original_size} -> {len(self.cache.data)} entries")
    
    @lru_cache(maxsize=128)
    def _compute_thermal_factor(self, temperature: float) -> float:
        """Cached thermal factor computation"""
        return 1.0 - 0.001 * (temperature - 25.0)
    
    def forward_propagation_optimized(self, input_power: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """Optimized forward propagation with caching and vectorization"""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cached_result = self.cache.get(input_power)
            if cached_result is not None:
                return cached_result
        
        # Validate input (optimized checks)
        if input_power.shape[0] != self.rows:
            raise ValueError(f"Input shape mismatch: {input_power.shape[0]} != {self.rows}")
        
        # Vectorized computation (much faster than loops)
        thermal_factor = self._compute_thermal_factor(25.0)  # Cached
        
        # Use efficient numpy operations
        transmission_scaled = self.transmission_matrix * thermal_factor
        output = np.dot(input_power, transmission_scaled)
        
        # Add noise efficiently
        noise = np.random.normal(0, 1e-6, output.shape).astype(np.float32)
        output += noise
        
        # Ensure physical constraints
        output = np.maximum(output, 0)
        
        # Cache result
        if use_cache:
            self.cache.put(input_power, output.copy())
        
        # Track performance
        execution_time = (time.time() - start_time) * 1000
        self.performance_profile.operation_times.setdefault('forward_propagation', []).append(execution_time)
        
        return output
    
    def batch_forward_propagation(self, input_batch: List[np.ndarray], parallel: bool = True) -> List[np.ndarray]:
        """Batch processing with optional parallelization"""
        if not parallel or len(input_batch) < 4:
            # Sequential processing for small batches
            return [self.forward_propagation_optimized(inp) for inp in input_batch]
        else:
            # Parallel processing for larger batches
            return self.executor.map_threaded(self.forward_propagation_optimized, input_batch)
    
    def update_weights_vectorized(self, new_weights: np.ndarray):
        """Vectorized weight updates"""
        if new_weights.shape != self.transmission_matrix.shape:
            raise ValueError("Weight shape mismatch")
        
        # Vectorized clipping and assignment
        self.transmission_matrix = np.clip(new_weights, 0.1, 1.0).astype(np.float32)
    
    def benchmark_performance(self, num_operations: int = 1000, batch_size: int = 10) -> Dict[str, float]:
        """Comprehensive performance benchmark"""
        self.logger.info(f"Starting performance benchmark ({num_operations} operations)")
        
        # Generate test data
        test_inputs = [
            np.random.uniform(0.1e-3, 2e-3, self.rows).astype(np.float32)
            for _ in range(num_operations)
        ]
        
        # Benchmark sequential processing
        start_time = time.time()
        for inp in test_inputs[:100]:  # Sample
            self.forward_propagation_optimized(inp, use_cache=False)
        sequential_time = time.time() - start_time
        
        # Benchmark with caching
        start_time = time.time()
        for inp in test_inputs[:100]:
            self.forward_propagation_optimized(inp, use_cache=True)
        cached_time = time.time() - start_time
        
        # Benchmark batch processing
        start_time = time.time()
        for i in range(0, min(200, len(test_inputs)), batch_size):
            batch = test_inputs[i:i+batch_size]
            self.batch_forward_propagation(batch, parallel=True)
        batch_time = time.time() - start_time
        
        # Calculate metrics
        sequential_ops_per_sec = 100 / sequential_time
        cached_ops_per_sec = 100 / cached_time
        batch_ops_per_sec = min(200, len(test_inputs)) / batch_time
        
        cache_stats = self.cache.stats()
        
        results = {
            "sequential_ops_per_sec": sequential_ops_per_sec,
            "cached_ops_per_sec": cached_ops_per_sec,
            "batch_ops_per_sec": batch_ops_per_sec,
            "cache_hit_rate": cache_stats["hit_rate"],
            "speedup_cache": cached_ops_per_sec / sequential_ops_per_sec,
            "speedup_batch": batch_ops_per_sec / sequential_ops_per_sec,
            "memory_usage_mb": self.memory_optimizer.get_memory_usage()
        }
        
        self.logger.info(f"Benchmark completed: {results}")
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown()
        self.cache = None
        gc.collect()

class ScalablePhotonicCluster:
    """Scalable cluster of photonic devices"""
    
    def __init__(self, initial_devices: int = 4):
        self.devices = []
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(min_workers=2, max_workers=16)
        self.memory_optimizer = MemoryOptimizer()
        
        # Initialize devices
        for i in range(initial_devices):
            device = OptimizedPhotonicDevice(rows=16, cols=16, device_id=f"cluster_device_{i}")
            self.devices.append(device)
            self.load_balancer.add_device(device)
        
        self.logger = logging.getLogger("PhotonicCluster")
        self.logger.info(f"Initialized cluster with {initial_devices} devices")
    
    def process_request_batch(self, requests: List[np.ndarray]) -> List[np.ndarray]:
        """Process batch of requests with load balancing"""
        start_time = time.time()
        
        # Distribute requests across devices
        distributed_requests = self.load_balancer.distribute_requests(requests)
        
        # Process in parallel across devices
        results = []
        for device, device_requests in distributed_requests.items():
            if device_requests:
                device_results = device.batch_forward_propagation(device_requests, parallel=True)
                results.extend(device_results)
        
        # Update scaling metrics
        execution_time = (time.time() - start_time) * 1000
        metrics = ScalingMetrics(
            cpu_usage_percent=psutil.cpu_percent(),
            memory_usage_percent=self.memory_optimizer.get_memory_usage(),
            throughput_ops_per_sec=len(requests) / (execution_time / 1000),
            latency_p95_ms=execution_time * 0.95,  # Approximation
            queue_depth=0,  # Simplified
            active_threads=len(self.devices)
        )
        
        self.auto_scaler.update_metrics(metrics)
        
        # Check for auto-scaling
        new_worker_count = self.auto_scaler.auto_scale()
        if new_worker_count:
            self._adjust_device_count(new_worker_count)
        
        return results
    
    def _adjust_device_count(self, target_count: int):
        """Adjust number of devices based on auto-scaling decision"""
        current_count = len(self.devices)
        
        if target_count > current_count:
            # Add devices
            for i in range(target_count - current_count):
                device_id = f"cluster_device_{current_count + i}"
                device = OptimizedPhotonicDevice(rows=16, cols=16, device_id=device_id)
                self.devices.append(device)
                self.load_balancer.add_device(device)
            
            self.logger.info(f"Scaled up: {current_count} -> {target_count} devices")
            
        elif target_count < current_count:
            # Remove devices
            devices_to_remove = self.devices[target_count:]
            self.devices = self.devices[:target_count]
            
            for device in devices_to_remove:
                self.load_balancer.remove_device(device)
                device.cleanup()
            
            self.logger.info(f"Scaled down: {current_count} -> {target_count} devices")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        total_cache_hits = sum(device.cache.hits for device in self.devices)
        total_cache_misses = sum(device.cache.misses for device in self.devices)
        total_requests = total_cache_hits + total_cache_misses
        
        return {
            "device_count": len(self.devices),
            "total_requests": total_requests,
            "overall_cache_hit_rate": total_cache_hits / total_requests if total_requests > 0 else 0,
            "memory_usage_percent": self.memory_optimizer.get_memory_usage(),
            "cpu_usage_percent": psutil.cpu_percent(),
            "auto_scaler_workers": self.auto_scaler.current_workers
        }

class LoadBalancer:
    """Simple round-robin load balancer"""
    
    def __init__(self):
        self.devices = []
        self.current_index = 0
    
    def add_device(self, device):
        """Add device to load balancer"""
        self.devices.append(device)
    
    def remove_device(self, device):
        """Remove device from load balancer"""
        if device in self.devices:
            self.devices.remove(device)
    
    def distribute_requests(self, requests: List[Any]) -> Dict[Any, List[Any]]:
        """Distribute requests across devices using round-robin"""
        if not self.devices:
            return {}
        
        distribution = defaultdict(list)
        
        for i, request in enumerate(requests):
            device_index = (self.current_index + i) % len(self.devices)
            device = self.devices[device_index]
            distribution[device].append(request)
        
        self.current_index = (self.current_index + len(requests)) % len(self.devices)
        
        return distribution

# Demo functions

def demo_optimized_device():
    """Demonstrate optimized device performance"""
    print("\n=== Optimized Device Demo ===")
    
    device = OptimizedPhotonicDevice(rows=32, cols=32, device_id="perf_001")
    
    # Benchmark performance
    benchmark_results = device.benchmark_performance(num_operations=500, batch_size=20)
    
    print(f"Performance Benchmark Results:")
    print(f"  Sequential: {benchmark_results['sequential_ops_per_sec']:,.0f} ops/sec")
    print(f"  With Cache: {benchmark_results['cached_ops_per_sec']:,.0f} ops/sec")
    print(f"  Batch Mode: {benchmark_results['batch_ops_per_sec']:,.0f} ops/sec")
    print(f"  Cache Hit Rate: {benchmark_results['cache_hit_rate']:.1%}")
    print(f"  Cache Speedup: {benchmark_results['speedup_cache']:.1f}x")
    print(f"  Batch Speedup: {benchmark_results['speedup_batch']:.1f}x")
    print(f"  Memory Usage: {benchmark_results['memory_usage_mb']:.1f}%")
    
    device.cleanup()
    return device

def demo_adaptive_cache():
    """Demonstrate adaptive caching"""
    print("\n=== Adaptive Cache Demo ===")
    
    cache = AdaptiveCache(max_size=100, strategy=CacheStrategy.ADAPTIVE)
    
    # Generate test data with access patterns
    test_data = []
    for i in range(200):
        # Create realistic access patterns
        if i < 50:
            # Frequent early items
            key = f"frequent_{i % 10}"
        elif i < 150:
            # Mix of frequent and infrequent
            key = f"mixed_{i}"
        else:
            # Recent items
            key = f"recent_{i}"
        
        test_data.append((key, f"value_{i}"))
    
    # Simulate cache access patterns
    for key, value in test_data:
        cached_value = cache.get(key)
        if cached_value is None:
            cache.put(key, value)
        
        # Simulate repeated access to some items
        if "frequent" in key:
            for _ in range(3):
                cache.get(key)
    
    stats = cache.stats()
    print(f"Adaptive Cache Performance:")
    print(f"  Cache Size: {stats['size']}/{stats['max_size']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
    print(f"  Total Hits: {stats['hits']}")
    print(f"  Total Misses: {stats['misses']}")
    print(f"  Evictions: {stats['evictions']}")
    
    return cache

def demo_parallel_processing():
    """Demonstrate parallel processing capabilities"""
    print("\n=== Parallel Processing Demo ===")
    
    executor = ParallelExecutor()
    
    def compute_heavy_task(size):
        """Simulate CPU-intensive task"""
        matrix = np.random.random((size, size))
        return np.sum(np.linalg.eigvals(matrix))
    
    # Test different execution modes
    sizes = [50, 60, 70, 80, 90] * 4  # 20 tasks
    
    # Sequential execution
    start_time = time.time()
    sequential_results = [compute_heavy_task(size) for size in sizes[:5]]  # Sample
    sequential_time = time.time() - start_time
    
    # Threaded execution
    start_time = time.time()
    threaded_results = executor.map_threaded(compute_heavy_task, sizes[:5])
    threaded_time = time.time() - start_time
    
    # Batch processing
    start_time = time.time()
    batch_results = executor.batch_process(compute_heavy_task, sizes, batch_size=4)
    batch_time = time.time() - start_time
    
    print(f"Parallel Processing Results:")
    print(f"  Sequential (5 tasks): {sequential_time:.2f}s")
    print(f"  Threaded (5 tasks): {threaded_time:.2f}s")
    print(f"  Batch ({len(sizes)} tasks): {batch_time:.2f}s")
    print(f"  Threading Speedup: {sequential_time/threaded_time:.1f}x")
    print(f"  Tasks per second: {len(sizes)/batch_time:.1f}")
    
    executor.shutdown()
    return executor

def demo_auto_scaling():
    """Demonstrate auto-scaling capabilities"""
    print("\n=== Auto-Scaling Demo ===")
    
    cluster = ScalablePhotonicCluster(initial_devices=2)
    
    # Simulate increasing load
    for load_phase in range(5):
        print(f"\n--- Load Phase {load_phase + 1} ---")
        
        # Generate increasing load
        num_requests = (load_phase + 1) * 50
        requests = [
            np.random.uniform(0.1e-3, 2e-3, 16).astype(np.float32)
            for _ in range(num_requests)
        ]
        
        # Process requests
        start_time = time.time()
        results = cluster.process_request_batch(requests)
        processing_time = time.time() - start_time
        
        # Get cluster stats
        stats = cluster.get_cluster_stats()
        
        print(f"  Processed {num_requests} requests in {processing_time:.2f}s")
        print(f"  Throughput: {num_requests/processing_time:.0f} req/sec")
        print(f"  Active Devices: {stats['device_count']}")
        print(f"  Cache Hit Rate: {stats['overall_cache_hit_rate']:.1%}")
        print(f"  Memory Usage: {stats['memory_usage_percent']:.1f}%")
        print(f"  CPU Usage: {stats['cpu_usage_percent']:.1f}%")
        
        # Simulate load-based scaling triggers
        time.sleep(0.5)
    
    final_stats = cluster.get_cluster_stats()
    print(f"\nFinal Cluster State:")
    print(f"  Total Devices: {final_stats['device_count']}")
    print(f"  Total Requests Processed: {final_stats['total_requests']}")
    print(f"  Overall Cache Hit Rate: {final_stats['overall_cache_hit_rate']:.1%}")
    
    return cluster

def demo_memory_optimization():
    """Demonstrate memory optimization"""
    print("\n=== Memory Optimization Demo ===")
    
    optimizer = MemoryOptimizer()
    
    # Create memory-intensive objects
    large_arrays = []
    cleanup_called = False
    
    def cleanup_callback():
        nonlocal cleanup_called, large_arrays
        cleanup_called = True
        large_arrays.clear()
        print("  🧹 Memory cleanup callback executed")
    
    optimizer.register_cleanup(cleanup_callback)
    
    print(f"Initial memory usage: {optimizer.get_memory_usage():.1f}%")
    
    # Simulate memory pressure
    print("Creating memory pressure...")
    for i in range(5):
        # Create large arrays to increase memory usage
        large_array = np.random.random((1000, 1000))
        large_arrays.append(large_array)
        print(f"  Step {i+1}: {optimizer.get_memory_usage():.1f}% memory usage")
    
    # Force optimization
    print("\nForcing memory optimization...")
    optimizer.optimize_memory(force=True)
    
    print(f"Final memory usage: {optimizer.get_memory_usage():.1f}%")
    print(f"Cleanup callback called: {cleanup_called}")
    
    return optimizer

def main():
    """Run all Generation 3 demos"""
    print("Photon-Memristor-Sim: Generation 3 Scalable Demo")
    print("=" * 60)
    
    try:
        # Optimized device performance
        opt_device = demo_optimized_device()
        
        # Adaptive caching
        cache = demo_adaptive_cache()
        
        # Parallel processing
        executor = demo_parallel_processing()
        
        # Auto-scaling
        cluster = demo_auto_scaling()
        
        # Memory optimization
        memory_opt = demo_memory_optimization()
        
        print("\n" + "=" * 60)
        print("✅ Generation 3 Demo Completed Successfully!")
        print("Advanced scaling and optimization features implemented.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()