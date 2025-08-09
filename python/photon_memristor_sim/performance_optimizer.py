"""
Advanced performance optimization and scaling utilities
"""

import time
import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
from functools import lru_cache, partial
from dataclasses import dataclass
import gc
import psutil
import os


@dataclass
class PerformanceProfile:
    """Performance profiling results"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0


class AdaptiveScheduler:
    """Adaptive task scheduler that optimizes based on system resources"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.cpu_count = multiprocessing.cpu_count()
        self.max_workers = max_workers or min(32, (self.cpu_count or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        self.current_load = 0.0
        self.load_history = []
        
    def get_system_load(self) -> float:
        """Get current system CPU usage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 50.0  # Default fallback
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 50.0  # Default fallback
    
    def should_use_parallel(self, task_size: int, complexity: int = 1) -> bool:
        """Determine if task should be parallelized"""
        load = self.get_system_load()
        memory = self.get_memory_usage()
        
        # Don't parallelize if system is under heavy load
        if load > 80.0 or memory > 85.0:
            return False
            
        # Parallelize large tasks
        if task_size > 1000 and complexity > 2:
            return True
            
        return task_size > 100 and load < 60.0
    
    def execute_optimized(self, func: Callable, data_chunks: List[Any], 
                         use_processes: bool = False) -> List[Any]:
        """Execute function optimally across data chunks"""
        if not data_chunks:
            return []
            
        task_size = len(data_chunks)
        
        # Small tasks - execute sequentially
        if not self.should_use_parallel(task_size):
            return [func(chunk) for chunk in data_chunks]
        
        # Large tasks - execute in parallel
        executor = self.process_pool if use_processes else self.thread_pool
        
        try:
            futures = [executor.submit(func, chunk) for chunk in data_chunks]
            results = [future.result() for future in futures]
            return results
        except Exception as e:
            # Fallback to sequential execution
            return [func(chunk) for chunk in data_chunks]


class IntelligentCache:
    """Intelligent caching system with prediction and preloading"""
    
    def __init__(self, max_size: int = 1000, prediction_enabled: bool = True):
        self.max_size = max_size
        self.cache = {}
        self.access_counts = {}
        self.access_patterns = {}
        self.prediction_enabled = prediction_enabled
        self.lock = threading.Lock()
        
    @lru_cache(maxsize=128)
    def _compute_key_similarity(self, key1: str, key2: str) -> float:
        """Compute similarity between cache keys for prediction"""
        # Simple Jaccard similarity
        set1 = set(key1.split('_'))
        set2 = set(key2.split('_'))
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access tracking"""
        with self.lock:
            if key in self.cache:
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                # Track access patterns for prediction
                if self.prediction_enabled:
                    current_time = time.time()
                    if key not in self.access_patterns:
                        self.access_patterns[key] = []
                    self.access_patterns[key].append(current_time)
                    
                    # Keep only recent accesses
                    cutoff = current_time - 3600  # 1 hour
                    self.access_patterns[key] = [
                        t for t in self.access_patterns[key] if t > cutoff
                    ]
                
                return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Store value in cache with intelligent eviction"""
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._intelligent_evict()
            
            self.cache[key] = value
            self.access_counts[key] = 1
    
    def _intelligent_evict(self) -> None:
        """Evict items intelligently based on access patterns"""
        if not self.cache:
            return
            
        # Score items based on access frequency and recency
        current_time = time.time()
        scores = {}
        
        for key in self.cache.keys():
            frequency = self.access_counts.get(key, 1)
            
            # Recency score
            if key in self.access_patterns and self.access_patterns[key]:
                last_access = max(self.access_patterns[key])
                recency = max(0.1, 3600 - (current_time - last_access))  # Favor recent
            else:
                recency = 0.1
            
            # Combined score (higher is better)
            scores[key] = frequency * recency
        
        # Evict lowest scoring item
        if scores:
            worst_key = min(scores.keys(), key=lambda k: scores[k])
            del self.cache[worst_key]
            self.access_counts.pop(worst_key, None)
            self.access_patterns.pop(worst_key, None)
    
    def predict_next_access(self, current_key: str, limit: int = 5) -> List[str]:
        """Predict next likely cache accesses"""
        if not self.prediction_enabled or not self.access_patterns:
            return []
        
        similarities = {}
        for key in self.cache.keys():
            if key != current_key:
                similarity = self._compute_key_similarity(current_key, key)
                if similarity > 0.1:  # Threshold for relevance
                    access_frequency = len(self.access_patterns.get(key, []))
                    similarities[key] = similarity * access_frequency
        
        # Return top predictions
        sorted_keys = sorted(similarities.keys(), 
                           key=lambda k: similarities[k], reverse=True)
        return sorted_keys[:limit]


class BatchProcessor:
    """Optimized batch processing for large datasets"""
    
    def __init__(self, batch_size: int = 32, prefetch_batches: int = 2):
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.scheduler = AdaptiveScheduler()
        
    def create_batches(self, data: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
        """Create optimally sized batches"""
        batch_size = batch_size or self.batch_size
        
        # Adaptive batch sizing based on system resources
        memory_usage = self.scheduler.get_memory_usage()
        if memory_usage > 80:
            batch_size = max(1, batch_size // 2)  # Reduce batch size
        elif memory_usage < 40:
            batch_size = min(len(data), batch_size * 2)  # Increase batch size
        
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    async def process_batches_async(self, batches: List[List[Any]], 
                                  process_func: Callable) -> List[Any]:
        """Process batches asynchronously with prefetching"""
        results = []
        
        # Create semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(self.prefetch_batches)
        
        async def process_batch(batch):
            async with semaphore:
                # Run in thread pool for CPU-bound tasks
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, process_func, batch)
        
        # Process all batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def process_batches_sync(self, batches: List[List[Any]], 
                           process_func: Callable) -> List[Any]:
        """Process batches synchronously with optimization"""
        return self.scheduler.execute_optimized(process_func, batches)


class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def optimize_arrays(arrays: List[np.ndarray], copy: bool = False) -> List[np.ndarray]:
        """Optimize memory layout of numpy arrays"""
        optimized = []
        
        for arr in arrays:
            if copy or not arr.flags['C_CONTIGUOUS']:
                # Ensure C-contiguous for better cache performance
                optimized_arr = np.ascontiguousarray(arr)
            else:
                optimized_arr = arr
                
            optimized.append(optimized_arr)
        
        return optimized
    
    @staticmethod
    def clear_jax_cache():
        """Clear JAX compilation cache to free memory"""
        try:
            jax.clear_caches()
        except:
            pass
    
    @staticmethod
    def force_garbage_collect():
        """Force garbage collection"""
        gc.collect()
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage information"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident memory
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_mb': 1024}


class JAXOptimizer:
    """JAX-specific performance optimizations"""
    
    @staticmethod
    def optimize_for_device(arrays: List[jnp.ndarray], device: str = 'cpu') -> List[jnp.ndarray]:
        """Optimize arrays for specific device"""
        try:
            device_arrays = []
            for arr in arrays:
                device_arr = jax.device_put(arr, device)
                device_arrays.append(device_arr)
            return device_arrays
        except:
            return arrays  # Return original if device placement fails
    
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def optimized_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled matrix multiplication"""
        return jnp.dot(a, b)
    
    @staticmethod
    def create_optimized_function(func: Callable, static_args: tuple = ()) -> Callable:
        """Create JIT-optimized version of function"""
        return jax.jit(func, static_argnums=static_args)
    
    @staticmethod
    def batch_operations(operations: List[Callable], inputs: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Batch multiple operations for better performance"""
        if not operations or not inputs:
            return []
        
        # Use vmap for vectorized operations when possible
        try:
            if len(operations) == 1 and len(inputs) > 1:
                # Single operation, multiple inputs
                vectorized_op = jax.vmap(operations[0])
                stacked_inputs = jnp.stack(inputs)
                return [vectorized_op(stacked_inputs)]
        except:
            pass
        
        # Fallback to sequential execution
        results = []
        for op, inp in zip(operations, inputs):
            results.append(op(inp))
        return results


class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
        
    def start_profile(self, name: str):
        """Start profiling a named section"""
        self.active_profiles[name] = {
            'start_time': time.time(),
            'start_memory': MemoryOptimizer.get_memory_info()['rss_mb'],
            'start_cpu': psutil.cpu_percent()
        }
    
    def end_profile(self, name: str) -> PerformanceProfile:
        """End profiling and return results"""
        if name not in self.active_profiles:
            raise ValueError(f"No active profile named '{name}'")
        
        start_data = self.active_profiles.pop(name)
        end_time = time.time()
        end_memory = MemoryOptimizer.get_memory_info()['rss_mb']
        end_cpu = psutil.cpu_percent()
        
        profile = PerformanceProfile(
            execution_time=end_time - start_data['start_time'],
            memory_usage=end_memory - start_data['start_memory'],
            cpu_usage=(end_cpu + start_data['start_cpu']) / 2  # Average
        )
        
        self.profiles[name] = profile
        return profile
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary"""
        summary = {}
        for name, profile in self.profiles.items():
            summary[name] = {
                'execution_time': profile.execution_time,
                'memory_usage': profile.memory_usage,
                'cpu_usage': profile.cpu_usage
            }
        return summary


class OptimizedPhotonic:
    """Optimized photonic computation utilities"""
    
    def __init__(self):
        self.cache = IntelligentCache(max_size=500)
        self.batch_processor = BatchProcessor(batch_size=64)
        self.scheduler = AdaptiveScheduler()
        self.profiler = PerformanceProfiler()
    
    def optimized_simulation(self, arrays: List[np.ndarray], 
                           simulation_func: Callable,
                           use_cache: bool = True) -> List[np.ndarray]:
        """Run optimized photonic simulation"""
        self.profiler.start_profile('optimized_simulation')
        
        try:
            # Check cache first
            if use_cache:
                cache_key = self._generate_cache_key(arrays)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Optimize arrays for better memory access
            optimized_arrays = MemoryOptimizer.optimize_arrays(arrays)
            
            # Create optimal batches
            batches = self.batch_processor.create_batches(optimized_arrays)
            
            # Process batches
            results = self.batch_processor.process_batches_sync(
                batches, simulation_func
            )
            
            # Flatten results
            flattened_results = [item for sublist in results for item in sublist]
            
            # Cache results
            if use_cache:
                self.cache.put(cache_key, flattened_results)
            
            return flattened_results
            
        finally:
            self.profiler.end_profile('optimized_simulation')
    
    def _generate_cache_key(self, arrays: List[np.ndarray]) -> str:
        """Generate cache key for arrays"""
        key_parts = []
        for i, arr in enumerate(arrays[:3]):  # Limit to first 3 arrays
            shape_str = 'x'.join(map(str, arr.shape))
            dtype_str = str(arr.dtype)
            # Use hash of a small sample for large arrays
            if arr.size > 1000:
                sample = arr.flat[:100]
                hash_val = hash(tuple(sample))
            else:
                hash_val = hash(tuple(arr.flat))
            
            key_parts.append(f"{i}_{shape_str}_{dtype_str}_{hash_val}")
        
        return "_".join(key_parts)
    
    def benchmark_performance(self, test_func: Callable, 
                            test_data: Any, iterations: int = 10) -> Dict[str, float]:
        """Benchmark function performance"""
        times = []
        memory_deltas = []
        
        for i in range(iterations):
            # Clear caches before each run
            if i == 0:  # Warmup run
                MemoryOptimizer.clear_jax_cache()
                MemoryOptimizer.force_garbage_collect()
            
            start_memory = MemoryOptimizer.get_memory_info()['rss_mb']
            start_time = time.time()
            
            result = test_func(test_data)
            
            end_time = time.time()
            end_memory = MemoryOptimizer.get_memory_info()['rss_mb']
            
            if i > 0:  # Skip warmup
                times.append(end_time - start_time)
                memory_deltas.append(end_memory - start_memory)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times), 
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory_delta': np.mean(memory_deltas),
            'throughput': len(test_data) / np.mean(times) if hasattr(test_data, '__len__') else 1.0 / np.mean(times)
        }


# Global optimizer instance
_global_optimizer = None


def get_optimizer() -> OptimizedPhotonic:
    """Get global optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = OptimizedPhotonic()
    return _global_optimizer


# Convenience decorators
def optimized_computation(use_cache: bool = True):
    """Decorator for optimized computation"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            
            # Convert args to list for optimization
            if args and isinstance(args[0], (list, tuple)) and all(isinstance(x, np.ndarray) for x in args[0]):
                arrays = args[0]
                other_args = args[1:]
                
                return optimizer.optimized_simulation(
                    arrays, 
                    lambda batch: func(batch, *other_args, **kwargs),
                    use_cache=use_cache
                )
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_performance(name: str):
    """Decorator to profile function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            optimizer.profiler.start_profile(name)
            try:
                return func(*args, **kwargs)
            finally:
                optimizer.profiler.end_profile(name)
        return wrapper
    return decorator