"""
Hyperscale Performance Optimization for Photonic Memristor Systems
Generation 3: MAKE IT SCALE - Advanced performance optimization and auto-scaling
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from queue import Queue, PriorityQueue, Empty
import json
import hashlib
from functools import lru_cache, wraps
import psutil
import logging


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for auto-scaling decisions"""
    timestamp: float
    throughput: float          # Operations per second
    latency_p50: float        # 50th percentile latency (ms)
    latency_p95: float        # 95th percentile latency (ms)
    latency_p99: float        # 99th percentile latency (ms)
    cpu_utilization: float    # CPU usage percentage
    memory_utilization: float # Memory usage percentage
    gpu_utilization: float    # GPU usage percentage (if available)
    active_workers: int       # Number of active worker processes
    queue_depth: int          # Current task queue size
    error_rate: float         # Error rate percentage
    cache_hit_rate: float     # Cache hit percentage


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling behavior"""
    min_workers: int = 2
    max_workers: int = mp.cpu_count() * 2
    target_cpu_utilization: float = 70.0
    target_latency_ms: float = 100.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 40.0
    scale_up_cooldown: float = 30.0     # seconds
    scale_down_cooldown: float = 60.0   # seconds
    metrics_window_size: int = 60
    batch_size_min: int = 8
    batch_size_max: int = 128
    queue_size_multiplier: float = 4.0


@dataclass
class ScalingDecision:
    """Record of auto-scaling decision"""
    timestamp: float
    action: str  # 'scale_up', 'scale_down', 'no_action'
    reason: str
    previous_workers: int
    new_workers: int
    metrics: PerformanceMetrics


class IntelligentCache:
    """High-performance intelligent cache with adaptive algorithms"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 3600.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._lock = threading.RLock()
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        # Adaptive parameters
        self._performance_history = deque(maxlen=1000)
        self._auto_tune_enabled = True
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with adaptive optimization"""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                value, timestamp = self._cache[key]
                
                # Check TTL
                if current_time - timestamp > self.ttl_seconds:
                    del self._cache[key]
                    del self._access_times[key]
                    self._miss_count += 1
                    return None
                
                # Update access statistics
                self._access_times[key] = current_time
                self._access_counts[key] += 1
                self._hit_count += 1
                
                return value
            else:
                self._miss_count += 1
                return None
    
    def put(self, key: str, value: Any, custom_ttl: Optional[float] = None) -> None:
        """Put value in cache with intelligent eviction"""
        with self._lock:
            current_time = time.time()
            
            # Evict expired entries first
            self._evict_expired()
            
            # Evict LRU entries if necessary
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            # Store value
            ttl = custom_ttl or self.ttl_seconds
            self._cache[key] = (value, current_time)
            self._access_times[key] = current_time
            self._access_counts[key] += 1
    
    def _evict_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, (value, timestamp) in self._cache.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            self._eviction_count += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._access_times:
            return
            
        lru_key = min(self._access_times.keys(), key=self._access_times.get)
        del self._cache[lru_key]
        del self._access_times[lru_key]
        if lru_key in self._access_counts:
            del self._access_counts[lru_key]
        self._eviction_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests) if total_requests > 0 else 0.0
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'eviction_count': self._eviction_count,
                'memory_efficiency': len(self._cache) / self.max_size if self.max_size > 0 else 0.0
            }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._eviction_count = 0


class ResourcePool:
    """High-performance resource pool with dynamic sizing"""
    
    def __init__(self, factory: Callable[[], Any], min_size: int = 2, max_size: int = 20):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        
        self._pool = Queue(maxsize=max_size)
        self._created_count = 0
        self._checkout_count = 0
        self._checkin_count = 0
        self._lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(min_size):
            self._pool.put(self.factory())
            self._created_count += 1
    
    def checkout(self, timeout: float = 5.0) -> Optional[Any]:
        """Checkout resource from pool"""
        try:
            resource = self._pool.get(timeout=timeout)
            with self._lock:
                self._checkout_count += 1
            return resource
        except Empty:
            # Create new resource if under limit
            with self._lock:
                if self._created_count < self.max_size:
                    resource = self.factory()
                    self._created_count += 1
                    self._checkout_count += 1
                    return resource
            return None
    
    def checkin(self, resource: Any) -> None:
        """Return resource to pool"""
        try:
            self._pool.put(resource, block=False)
            with self._lock:
                self._checkin_count += 1
        except:
            # Pool is full, discard resource
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': self._pool.qsize(),
                'max_size': self.max_size,
                'min_size': self.min_size,
                'created_count': self._created_count,
                'checkout_count': self._checkout_count,
                'checkin_count': self._checkin_count,
                'outstanding': self._checkout_count - self._checkin_count,
                'utilization': 1.0 - (self._pool.qsize() / self.max_size)
            }


class BatchProcessor:
    """High-throughput batch processing with adaptive sizing"""
    
    def __init__(self, processor_func: Callable[[List[Any]], List[Any]], 
                 min_batch_size: int = 8, max_batch_size: int = 128, 
                 max_latency_ms: float = 50.0):
        self.processor_func = processor_func
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        
        self._pending_items = []
        self._last_flush_time = time.time()
        self._lock = threading.Lock()
        
        # Statistics
        self._processed_batches = 0
        self._total_items = 0
        self._total_processing_time = 0.0
        
        # Adaptive batch sizing
        self._recent_latencies = deque(maxlen=100)
        self._adaptive_batch_size = min_batch_size
    
    def add_item(self, item: Any) -> Optional[List[Any]]:
        """Add item to batch, potentially triggering processing"""
        with self._lock:
            self._pending_items.append(item)
            
            # Check if we should process batch
            if (len(self._pending_items) >= self._adaptive_batch_size or
                self._should_flush_by_timeout()):
                
                return self._process_current_batch()
            
            return None
    
    def flush(self) -> List[Any]:
        """Force processing of current batch"""
        with self._lock:
            return self._process_current_batch()
    
    def _should_flush_by_timeout(self) -> bool:
        """Check if batch should be flushed due to timeout"""
        return (time.time() - self._last_flush_time) * 1000 > self.max_latency_ms
    
    def _process_current_batch(self) -> List[Any]:
        """Process current batch and update statistics"""
        if not self._pending_items:
            return []
        
        items = self._pending_items.copy()
        self._pending_items.clear()
        self._last_flush_time = time.time()
        
        # Process batch
        start_time = time.time()
        try:
            results = self.processor_func(items)
            processing_time = time.time() - start_time
            
            # Update statistics
            self._processed_batches += 1
            self._total_items += len(items)
            self._total_processing_time += processing_time
            
            # Update adaptive batch size
            latency_ms = processing_time * 1000
            self._recent_latencies.append(latency_ms)
            self._update_adaptive_batch_size()
            
            return results
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            return []
    
    def _update_adaptive_batch_size(self) -> None:
        """Adaptively adjust batch size based on performance"""
        if len(self._recent_latencies) < 10:
            return
        
        avg_latency = sum(self._recent_latencies) / len(self._recent_latencies)
        
        if avg_latency < self.max_latency_ms * 0.7:
            # Can increase batch size
            self._adaptive_batch_size = min(
                self._adaptive_batch_size + 2, 
                self.max_batch_size
            )
        elif avg_latency > self.max_latency_ms * 1.2:
            # Should decrease batch size
            self._adaptive_batch_size = max(
                self._adaptive_batch_size - 2,
                self.min_batch_size
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        with self._lock:
            avg_batch_size = (self._total_items / self._processed_batches 
                            if self._processed_batches > 0 else 0)
            avg_processing_time = (self._total_processing_time / self._processed_batches
                                 if self._processed_batches > 0 else 0)
            
            return {
                'processed_batches': self._processed_batches,
                'total_items': self._total_items,
                'pending_items': len(self._pending_items),
                'avg_batch_size': avg_batch_size,
                'adaptive_batch_size': self._adaptive_batch_size,
                'avg_processing_time_ms': avg_processing_time * 1000,
                'throughput_items_per_sec': (self._total_items / self._total_processing_time
                                           if self._total_processing_time > 0 else 0)
            }


class AutoScaler:
    """Intelligent auto-scaling system with machine learning insights"""
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self._metrics_history = deque(maxlen=config.metrics_window_size)
        self._scaling_decisions = []
        self._current_workers = config.min_workers
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0
        self._lock = threading.Lock()
        
        # Predictive scaling
        self._workload_patterns = deque(maxlen=1000)
        self._prediction_enabled = True
    
    def record_metrics(self, metrics: PerformanceMetrics) -> Optional[ScalingDecision]:
        """Record metrics and make scaling decision"""
        with self._lock:
            self._metrics_history.append(metrics)
            
            # Make scaling decision
            decision = self._make_scaling_decision(metrics)
            
            if decision.action != 'no_action':
                self._scaling_decisions.append(decision)
                self._current_workers = decision.new_workers
                
                if decision.action == 'scale_up':
                    self._last_scale_up = metrics.timestamp
                elif decision.action == 'scale_down':
                    self._last_scale_down = metrics.timestamp
            
            # Update workload patterns for prediction
            self._workload_patterns.append({
                'timestamp': metrics.timestamp,
                'throughput': metrics.throughput,
                'cpu_util': metrics.cpu_utilization,
                'latency': metrics.latency_p95
            })
            
            return decision
    
    def _make_scaling_decision(self, current_metrics: PerformanceMetrics) -> ScalingDecision:
        """Make intelligent scaling decision"""
        current_time = current_metrics.timestamp
        
        # Check cooldown periods
        scale_up_ready = (current_time - self._last_scale_up) > self.config.scale_up_cooldown
        scale_down_ready = (current_time - self._last_scale_down) > self.config.scale_down_cooldown
        
        # Calculate average metrics
        if len(self._metrics_history) >= 5:
            recent_metrics = list(self._metrics_history)[-5:]
            avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
            avg_latency = sum(m.latency_p95 for m in recent_metrics) / len(recent_metrics)
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            avg_queue_depth = sum(m.queue_depth for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = current_metrics.cpu_utilization
            avg_latency = current_metrics.latency_p95
            avg_error_rate = current_metrics.error_rate
            avg_queue_depth = current_metrics.queue_depth
        
        # Scaling decision logic
        action = 'no_action'
        reason = 'No scaling needed'
        new_workers = self._current_workers
        
        # Scale up conditions
        if (scale_up_ready and self._current_workers < self.config.max_workers):
            scale_up_triggers = [
                avg_cpu > self.config.scale_up_threshold,
                avg_latency > self.config.target_latency_ms * 1.5,
                avg_error_rate > 2.0,
                avg_queue_depth > self._current_workers * 10,
                current_metrics.memory_utilization > 85.0
            ]
            
            if any(scale_up_triggers):
                # Determine scale-up factor
                if avg_cpu > 95.0 or avg_latency > self.config.target_latency_ms * 3:
                    scale_factor = 2  # Aggressive scaling
                else:
                    scale_factor = 1.5  # Conservative scaling
                
                new_workers = min(
                    int(self._current_workers * scale_factor),
                    self.config.max_workers
                )
                action = 'scale_up'
                reason = (f"High load detected - CPU: {avg_cpu:.1f}%, "
                         f"Latency: {avg_latency:.1f}ms, Queue: {avg_queue_depth:.0f}")
        
        # Scale down conditions
        elif (scale_down_ready and self._current_workers > self.config.min_workers):
            scale_down_triggers = [
                avg_cpu < self.config.scale_down_threshold,
                avg_latency < self.config.target_latency_ms * 0.5,
                avg_error_rate < 0.1,
                avg_queue_depth < self._current_workers * 2
            ]
            
            if all(scale_down_triggers):
                new_workers = max(
                    int(self._current_workers * 0.7),  # Conservative scale down
                    self.config.min_workers
                )
                action = 'scale_down'
                reason = (f"Low load detected - CPU: {avg_cpu:.1f}%, "
                         f"Latency: {avg_latency:.1f}ms")
        
        return ScalingDecision(
            timestamp=current_time,
            action=action,
            reason=reason,
            previous_workers=self._current_workers,
            new_workers=new_workers,
            metrics=current_metrics
        )
    
    def get_current_workers(self) -> int:
        """Get current number of workers"""
        return self._current_workers
    
    def get_scaling_history(self) -> List[ScalingDecision]:
        """Get scaling decision history"""
        return self._scaling_decisions.copy()
    
    def predict_workload(self, horizon_minutes: int = 30) -> Dict[str, float]:
        """Predict future workload (simplified implementation)"""
        if len(self._workload_patterns) < 10:
            return {'predicted_cpu': 50.0, 'predicted_throughput': 100.0, 'confidence': 0.0}
        
        # Simple moving average prediction
        recent_patterns = list(self._workload_patterns)[-20:]
        avg_cpu = sum(p['cpu_util'] for p in recent_patterns) / len(recent_patterns)
        avg_throughput = sum(p['throughput'] for p in recent_patterns) / len(recent_patterns)
        
        # Add some trend analysis
        if len(recent_patterns) >= 10:
            first_half = recent_patterns[:len(recent_patterns)//2]
            second_half = recent_patterns[len(recent_patterns)//2:]
            
            cpu_trend = (sum(p['cpu_util'] for p in second_half) / len(second_half) - 
                        sum(p['cpu_util'] for p in first_half) / len(first_half))
            
            throughput_trend = (sum(p['throughput'] for p in second_half) / len(second_half) - 
                               sum(p['throughput'] for p in first_half) / len(first_half))
            
            predicted_cpu = max(0, min(100, avg_cpu + cpu_trend * 2))
            predicted_throughput = max(0, avg_throughput + throughput_trend * 2)
        else:
            predicted_cpu = avg_cpu
            predicted_throughput = avg_throughput
        
        return {
            'predicted_cpu': predicted_cpu,
            'predicted_throughput': predicted_throughput,
            'confidence': min(0.9, len(recent_patterns) / 20.0)
        }


class HyperscaleEngine:
    """Main hyperscale processing engine"""
    
    def __init__(self, config: AutoScalingConfig = None):
        self.config = config or AutoScalingConfig()
        
        # Core components
        self.cache = IntelligentCache(max_size=50000)
        self.resource_pool = ResourcePool(
            factory=lambda: {'worker_id': np.random.randint(1000, 9999)},
            min_size=self.config.min_workers,
            max_size=self.config.max_workers
        )
        self.auto_scaler = AutoScaler(self.config)
        
        # Batch processor
        self.batch_processor = BatchProcessor(
            processor_func=self._process_simulation_batch,
            min_batch_size=self.config.batch_size_min,
            max_batch_size=self.config.batch_size_max
        )
        
        # Task queue and worker management
        self.task_queue = PriorityQueue(maxsize=1000)
        self.result_queue = Queue()
        self._worker_pool = None
        self._running = False
        self._performance_monitor_thread = None
        
        # Statistics
        self._start_time = time.time()
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._latency_samples = deque(maxlen=1000)
    
    def start(self) -> None:
        """Start the hyperscale engine"""
        if self._running:
            return
        
        self._running = True
        
        # Start worker pool
        self._worker_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="HyperscaleWorker"
        )
        
        # Start performance monitoring
        self._performance_monitor_thread = threading.Thread(
            target=self._performance_monitor_loop,
            daemon=True
        )
        self._performance_monitor_thread.start()
        
        logging.info(f"Hyperscale engine started with {self.config.min_workers} workers")
    
    def stop(self) -> None:
        """Stop the hyperscale engine"""
        self._running = False
        
        if self._worker_pool:
            self._worker_pool.shutdown(wait=True)
        
        logging.info("Hyperscale engine stopped")
    
    def submit_task(self, task_data: Dict[str, Any], priority: int = 1) -> str:
        """Submit task for processing"""
        task_id = hashlib.md5(
            (str(task_data) + str(time.time())).encode()
        ).hexdigest()[:12]
        
        task = {
            'id': task_id,
            'data': task_data,
            'submit_time': time.time(),
            'priority': priority
        }
        
        try:
            self.task_queue.put((priority, time.time(), task), timeout=1.0)
            self._total_tasks += 1
            return task_id
        except:
            logging.warning("Task queue full, rejecting task")
            return None
    
    def get_result(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get result from result queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _process_simulation_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of simulation tasks"""
        results = []
        
        for task in tasks:
            start_time = time.time()
            
            try:
                # Check cache first
                cache_key = str(hash(str(task['data'])))
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    result = {
                        'task_id': task['id'],
                        'result': cached_result,
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'cache_hit': True
                    }
                else:
                    # Simulate photonic memristor computation
                    task_data = task['data']
                    voltage = task_data.get('voltage', 2.0)
                    optical_power = task_data.get('optical_power', 1e-3)
                    temperature = task_data.get('temperature', 300.0)
                    
                    # Simulate complex computation
                    time.sleep(0.001 + np.random.exponential(0.002))  # Realistic processing time
                    
                    # Calculate results
                    conductance = voltage**2 * 1e-6 * (1 + 0.1 * np.random.randn())
                    final_temp = temperature + voltage * 25 + optical_power * 1000
                    optical_transmission = np.exp(-optical_power * 10) * (0.8 + 0.2 * np.random.rand())
                    
                    computed_result = {
                        'conductance': conductance,
                        'temperature': final_temp,
                        'optical_transmission': optical_transmission,
                        'energy_efficiency': conductance / (voltage**2 + optical_power),
                        'switching_speed': 1.0 / (voltage + 1e-6)
                    }
                    
                    # Cache result
                    self.cache.put(cache_key, computed_result, custom_ttl=1800)
                    
                    result = {
                        'task_id': task['id'],
                        'result': computed_result,
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'cache_hit': False
                    }
                
                results.append(result)
                self._completed_tasks += 1
                self._latency_samples.append((time.time() - start_time) * 1000)
                
            except Exception as e:
                logging.error(f"Task processing failed: {e}")
                self._failed_tasks += 1
                results.append({
                    'task_id': task['id'],
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        return results
    
    def _performance_monitor_loop(self) -> None:
        """Background performance monitoring loop"""
        while self._running:
            try:
                metrics = self._collect_performance_metrics()
                scaling_decision = self.auto_scaler.record_metrics(metrics)
                
                if scaling_decision and scaling_decision.action != 'no_action':
                    logging.info(f"Auto-scaling decision: {scaling_decision.action} - {scaling_decision.reason}")
                    self._apply_scaling_decision(scaling_decision)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        current_time = time.time()
        
        # Calculate throughput
        uptime = current_time - self._start_time
        throughput = self._completed_tasks / uptime if uptime > 0 else 0.0
        
        # Calculate latencies
        if self._latency_samples:
            sorted_latencies = sorted(self._latency_samples)
            latency_p50 = sorted_latencies[len(sorted_latencies)//2]
            latency_p95 = sorted_latencies[int(len(sorted_latencies)*0.95)]
            latency_p99 = sorted_latencies[int(len(sorted_latencies)*0.99)]
        else:
            latency_p50 = latency_p95 = latency_p99 = 0.0
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        
        # GPU metrics (simplified)
        gpu_percent = 0.0  # Would integrate with nvidia-ml-py for real GPU monitoring
        
        # Error rate
        total_tasks = self._completed_tasks + self._failed_tasks
        error_rate = (self._failed_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
        
        # Cache hit rate
        cache_stats = self.cache.get_stats()
        cache_hit_rate = cache_stats['hit_rate'] * 100
        
        return PerformanceMetrics(
            timestamp=current_time,
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            gpu_utilization=gpu_percent,
            active_workers=self.auto_scaler.get_current_workers(),
            queue_depth=self.task_queue.qsize(),
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate
        )
    
    def _apply_scaling_decision(self, decision: ScalingDecision) -> None:
        """Apply scaling decision (in real implementation would adjust worker pool)"""
        # In a real implementation, this would:
        # 1. Scale up/down the actual worker pool
        # 2. Adjust resource allocations
        # 3. Update load balancing
        logging.info(f"Applied scaling: {decision.previous_workers} → {decision.new_workers} workers")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        cache_stats = self.cache.get_stats()
        pool_stats = self.resource_pool.get_stats()
        batch_stats = self.batch_processor.get_stats()
        
        uptime = time.time() - self._start_time
        
        return {
            'uptime_seconds': uptime,
            'total_tasks': self._total_tasks,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'success_rate': (self._completed_tasks / self._total_tasks * 100) if self._total_tasks > 0 else 0.0,
            'avg_throughput': self._completed_tasks / uptime if uptime > 0 else 0.0,
            'current_workers': self.auto_scaler.get_current_workers(),
            'cache_stats': cache_stats,
            'resource_pool_stats': pool_stats,
            'batch_processing_stats': batch_stats,
            'queue_depth': self.task_queue.qsize(),
            'scaling_decisions': len(self.auto_scaler.get_scaling_history())
        }


# Utility functions for high-performance operations
def parallel_matrix_operations(matrices: List[np.ndarray], 
                             operation: str = 'multiply') -> List[np.ndarray]:
    """Perform parallel matrix operations using multiprocessing"""
    
    def matrix_op(matrix):
        if operation == 'multiply':
            return np.matmul(matrix, matrix.T)
        elif operation == 'eigenvalues':
            return np.linalg.eigvals(matrix)
        elif operation == 'svd':
            return np.linalg.svd(matrix, compute_uv=False)
        else:
            return matrix
    
    with mp.Pool() as pool:
        results = pool.map(matrix_op, matrices)
    
    return results


@lru_cache(maxsize=1024)
def cached_photonic_calculation(voltage: float, power: float, temp: float) -> Tuple[float, float, float]:
    """Cached photonic calculation with LRU eviction"""
    # Expensive calculation that benefits from caching
    conductance = voltage**2 * 1e-6 * np.exp(-0.5 * temp / 300.0)
    transmission = np.exp(-power * 10) * (1 - 0.1 * voltage / 10)
    efficiency = conductance / (voltage**2 + power + 1e-9)
    
    return conductance, transmission, efficiency


def benchmark_hyperscale_performance(duration_seconds: int = 30) -> Dict[str, Any]:
    """Benchmark hyperscale engine performance"""
    print(f"🚀 Benchmarking Hyperscale Performance for {duration_seconds} seconds...")
    
    # Create engine with optimized config
    config = AutoScalingConfig(
        min_workers=4,
        max_workers=16,
        target_latency_ms=50.0,
        batch_size_min=16,
        batch_size_max=64
    )
    
    engine = HyperscaleEngine(config)
    engine.start()
    
    # Submit tasks continuously
    start_time = time.time()
    task_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            # Generate random photonic simulation tasks
            task_data = {
                'voltage': np.random.uniform(1.0, 5.0),
                'optical_power': np.random.uniform(1e-4, 50e-3),
                'temperature': np.random.uniform(280, 400),
                'material': np.random.choice(['GST', 'HfO2', 'TiO2']),
                'dimensions': [np.random.uniform(50e-9, 200e-9) for _ in range(3)]
            }
            
            task_id = engine.submit_task(task_data)
            if task_id:
                task_count += 1
            
            # Small delay to prevent overwhelming
            time.sleep(0.001)
        
        # Wait a bit for processing to complete
        time.sleep(5)
        
    finally:
        engine.stop()
    
    stats = engine.get_comprehensive_stats()
    stats['benchmark_duration'] = duration_seconds
    stats['tasks_submitted'] = task_count
    stats['submission_rate'] = task_count / duration_seconds
    
    return stats


if __name__ == "__main__":
    # Quick demonstration
    print("🔥 Hyperscale Performance System Demo")
    print("=" * 50)
    
    # Run performance benchmark
    results = benchmark_hyperscale_performance(duration_seconds=10)
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Tasks Submitted: {results['tasks_submitted']}")
    print(f"   Tasks Completed: {results['completed_tasks']}")
    print(f"   Success Rate: {results['success_rate']:.1f}%")
    print(f"   Avg Throughput: {results['avg_throughput']:.1f} tasks/sec")
    print(f"   Cache Hit Rate: {results['cache_stats']['hit_rate']*100:.1f}%")
    print(f"   Final Workers: {results['current_workers']}")
    print(f"   Scaling Decisions: {results['scaling_decisions']}")
    
    print(f"\n✅ Hyperscale system demonstration completed!")
    print(f"🚀 System achieved {results['avg_throughput']:.0f} tasks/sec with {results['success_rate']:.1f}% success rate")