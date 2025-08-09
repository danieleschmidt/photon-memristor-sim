"""
Advanced resilience and fault tolerance for production deployment
"""

import time
import asyncio
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass
import threading
from collections import deque
import random


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing - blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half-open to closed transition
    request_volume_threshold: int = 10


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.request_count = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            
            self.request_count += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution"""
        with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
    
    def _on_failure(self):
        """Handle failed execution"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0
            
            if (self.failure_count >= self.config.failure_threshold and 
                self.request_count >= self.config.request_volume_threshold):
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class RetryPolicy:
    """Retry policy with exponential backoff"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    # Add jitter
                    delay += random.uniform(0, delay * 0.1)
                    time.sleep(delay)
        
        raise last_exception


class HealthCheck:
    """Health check monitoring"""
    
    def __init__(self, name: str, check_interval: int = 30):
        self.name = name
        self.check_interval = check_interval
        self.is_healthy = True
        self.last_check = 0
        self.error_count = 0
        self.checks = []
        self.logger = logging.getLogger(f"health_check.{name}")
    
    def add_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check function"""
        self.checks.append((name, check_func))
    
    def run_checks(self) -> Dict[str, bool]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for check_name, check_func in self.checks:
            try:
                result = check_func()
                results[check_name] = result
                if not result:
                    overall_healthy = False
                    self.logger.warning(f"Health check {check_name} failed")
            except Exception as e:
                results[check_name] = False
                overall_healthy = False
                self.error_count += 1
                self.logger.error(f"Health check {check_name} error: {e}")
        
        self.is_healthy = overall_healthy
        self.last_check = time.time()
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get health status summary"""
        return {
            "name": self.name,
            "healthy": self.is_healthy,
            "last_check": self.last_check,
            "error_count": self.error_count,
            "checks_count": len(self.checks)
        }


class BulkheadIsolation:
    """Bulkhead isolation pattern for resource partitioning"""
    
    def __init__(self, name: str, max_concurrent: int = 10, queue_size: int = 100):
        self.name = name
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.active_requests = 0
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(f"bulkhead.{name}")
    
    async def execute(self, coro):
        """Execute coroutine with bulkhead isolation"""
        async with self.semaphore:
            async with self.lock:
                self.active_requests += 1
            
            try:
                result = await coro
                return result
            finally:
                async with self.lock:
                    self.active_requests -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        return {
            "name": self.name,
            "active_requests": self.active_requests,
            "max_concurrent": self.max_concurrent,
            "available_slots": self.max_concurrent - self.active_requests,
            "queue_size": self.queue.qsize()
        }


class CacheManager:
    """Thread-safe cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    self.access_times[key] = time.time()
                    return value
                else:
                    del self.cache[key]
                    del self.access_times[key]
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache"""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = (value, expiry)
            self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear_expired(self):
        """Clear expired entries"""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, (value, expiry) in self.cache.items():
                if current_time >= expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_ratio": len(self.access_times) / max(1, len(self.cache))
            }


class MetricsCollector:
    """Metrics collection and aggregation"""
    
    def __init__(self, name: str, window_size: int = 1000):
        self.name = name
        self.window_size = window_size
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
        self.timers = {}
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict] = None):
        """Increment counter metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.counters[key] = self.counters.get(key, 0) + value
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict] = None):
        """Record histogram value"""
        with self.lock:
            key = self._make_key(name, tags)
            if key not in self.histograms:
                self.histograms[key] = deque(maxlen=self.window_size)
            self.histograms[key].append(value)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict] = None):
        """Set gauge value"""
        with self.lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
    
    def time_operation(self, name: str, tags: Optional[Dict] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, tags)
    
    def _make_key(self, name: str, tags: Optional[Dict]) -> str:
        """Create metric key with tags"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self.lock:
            metrics = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {}
            }
            
            for key, values in self.histograms.items():
                if values:
                    metrics["histograms"][key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 0 else 0
                    }
            
            return metrics


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_histogram(self.name, duration * 1000, self.tags)  # ms


class ResilientSystem:
    """Main resilient system coordinator"""
    
    def __init__(self, name: str):
        self.name = name
        self.circuit_breakers = {}
        self.health_checks = {}
        self.bulkheads = {}
        self.cache_manager = CacheManager()
        self.metrics = MetricsCollector(name)
        self.logger = logging.getLogger(f"resilient_system.{name}")
    
    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Add circuit breaker"""
        cb = CircuitBreaker(name, config)
        self.circuit_breakers[name] = cb
        return cb
    
    def add_health_check(self, name: str, check_interval: int = 30) -> HealthCheck:
        """Add health check"""
        hc = HealthCheck(name, check_interval)
        self.health_checks[name] = hc
        return hc
    
    def add_bulkhead(self, name: str, max_concurrent: int = 10) -> BulkheadIsolation:
        """Add bulkhead isolation"""
        bulkhead = BulkheadIsolation(name, max_concurrent)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        health_status = {
            "system_name": self.name,
            "timestamp": time.time(),
            "overall_healthy": True,
            "circuit_breakers": {},
            "health_checks": {},
            "bulkheads": {},
            "cache": self.cache_manager.stats(),
            "metrics": self.metrics.get_metrics()
        }
        
        # Circuit breakers
        for name, cb in self.circuit_breakers.items():
            cb_status = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "request_count": cb.request_count
            }
            health_status["circuit_breakers"][name] = cb_status
            if cb.state != CircuitBreakerState.CLOSED:
                health_status["overall_healthy"] = False
        
        # Health checks
        for name, hc in self.health_checks.items():
            hc_status = hc.get_status()
            health_status["health_checks"][name] = hc_status
            if not hc_status["healthy"]:
                health_status["overall_healthy"] = False
        
        # Bulkheads
        for name, bulkhead in self.bulkheads.items():
            health_status["bulkheads"][name] = bulkhead.get_stats()
        
        return health_status


# Global resilient system instance
_global_resilient_system = None


def get_resilient_system(name: str = "photonic_sim") -> ResilientSystem:
    """Get global resilient system instance"""
    global _global_resilient_system
    if _global_resilient_system is None:
        _global_resilient_system = ResilientSystem(name)
    return _global_resilient_system


# Convenience decorators
def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection"""
    def decorator(func):
        system = get_resilient_system()
        if name not in system.circuit_breakers:
            cb_config = config or CircuitBreakerConfig()
            system.add_circuit_breaker(name, cb_config)
        
        def wrapper(*args, **kwargs):
            cb = system.circuit_breakers[name]
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator to add retry logic"""
    def decorator(func):
        retry_policy = RetryPolicy(max_attempts, base_delay)
        
        def wrapper(*args, **kwargs):
            return retry_policy.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def with_metrics(metric_name: str, tags: Optional[Dict] = None):
    """Decorator to add metrics collection"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            system = get_resilient_system()
            with system.metrics.time_operation(f"{metric_name}_duration", tags):
                try:
                    result = func(*args, **kwargs)
                    system.metrics.increment_counter(f"{metric_name}_success", tags=tags)
                    return result
                except Exception as e:
                    system.metrics.increment_counter(f"{metric_name}_error", tags=tags)
                    raise e
        return wrapper
    return decorator