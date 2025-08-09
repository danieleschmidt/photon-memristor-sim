"""
Comprehensive resilience and fault tolerance tests
"""

import time
import pytest
import threading
from unittest.mock import Mock, patch
from photon_memristor_sim.resilience import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
    RetryPolicy, HealthCheck, CacheManager, MetricsCollector,
    get_resilient_system, with_circuit_breaker, with_retry, with_metrics
)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation"""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
        cb = CircuitBreaker("test", config)
        
        # Normal function
        def success_func():
            return "success"
        
        # Should work normally
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker("test", config)
        
        def failing_func():
            raise ValueError("Test error")
        
        # First failures should pass through
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        # After threshold, circuit should open
        assert cb.state == CircuitBreakerState.OPEN
        
        # Further calls should be blocked
        with pytest.raises(Exception):  # Circuit breaker exception
            cb.call(failing_func)


class TestRetryPolicy:
    """Test retry policy functionality"""
    
    def test_retry_success_after_failures(self):
        """Test retry succeeds after initial failures"""
        call_count = 0
        
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        retry_policy = RetryPolicy(max_attempts=5, base_delay=0.01)
        result = retry_policy.execute(flaky_func)
        
        assert result == "success"
        assert call_count == 3
        
    def test_retry_exhausted(self):
        """Test retry policy when max attempts reached"""
        def always_fail():
            raise ValueError("Always fails")
        
        retry_policy = RetryPolicy(max_attempts=2, base_delay=0.01)
        
        with pytest.raises(ValueError):
            retry_policy.execute(always_fail)


class TestHealthCheck:
    """Test health check functionality"""
    
    def test_health_check_passing(self):
        """Test passing health checks"""
        health_check = HealthCheck("test_service")
        
        def check_database():
            return True
        
        def check_api():
            return True
        
        health_check.add_check("database", check_database)
        health_check.add_check("api", check_api)
        
        results = health_check.run_checks()
        
        assert results["database"] is True
        assert results["api"] is True
        assert health_check.is_healthy is True
        
    def test_health_check_failing(self):
        """Test failing health checks"""
        health_check = HealthCheck("test_service")
        
        def check_failing():
            return False
        
        def check_error():
            raise Exception("Check failed")
        
        health_check.add_check("failing", check_failing)
        health_check.add_check("error", check_error)
        
        results = health_check.run_checks()
        
        assert results["failing"] is False
        assert results["error"] is False
        assert health_check.is_healthy is False


class TestCacheManager:
    """Test cache management functionality"""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = CacheManager(max_size=10, default_ttl=1)
        
        # Store and retrieve
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Non-existent key
        assert cache.get("nonexistent") is None
        
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        cache = CacheManager(max_size=10, default_ttl=0.1)  # 100ms TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None
        
    def test_cache_lru_eviction(self):
        """Test LRU eviction"""
        cache = CacheManager(max_size=3, default_ttl=10)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should still exist
        assert cache.get("key4") == "value4"  # Should exist


class TestMetricsCollector:
    """Test metrics collection"""
    
    def test_counter_metrics(self):
        """Test counter metrics"""
        metrics = MetricsCollector("test")
        
        metrics.increment_counter("requests", 1, {"endpoint": "/api"})
        metrics.increment_counter("requests", 2, {"endpoint": "/api"})
        metrics.increment_counter("requests", 1, {"endpoint": "/health"})
        
        collected = metrics.get_metrics()
        
        assert collected["counters"]["requests[endpoint=/api]"] == 3
        assert collected["counters"]["requests[endpoint=/health]"] == 1
        
    def test_histogram_metrics(self):
        """Test histogram metrics"""
        metrics = MetricsCollector("test")
        
        values = [10, 20, 30, 40, 50]
        for value in values:
            metrics.record_histogram("response_time", value)
        
        collected = metrics.get_metrics()
        histogram = collected["histograms"]["response_time"]
        
        assert histogram["count"] == 5
        assert histogram["min"] == 10
        assert histogram["max"] == 50
        assert histogram["avg"] == 30
        
    def test_gauge_metrics(self):
        """Test gauge metrics"""
        metrics = MetricsCollector("test")
        
        metrics.set_gauge("cpu_usage", 75.5)
        metrics.set_gauge("memory_usage", 60.2)
        
        collected = metrics.get_metrics()
        
        assert collected["gauges"]["cpu_usage"] == 75.5
        assert collected["gauges"]["memory_usage"] == 60.2


class TestDecorators:
    """Test resilience decorators"""
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator"""
        
        @with_circuit_breaker("test_func")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Check that circuit breaker was created
        system = get_resilient_system()
        assert "test_func" in system.circuit_breakers
        
    def test_retry_decorator(self):
        """Test retry decorator"""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Flaky error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 2
        
    def test_metrics_decorator(self):
        """Test metrics decorator"""
        
        @with_metrics("test_operation")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "done"
        
        result = test_function()
        assert result == "done"
        
        # Check metrics were recorded
        system = get_resilient_system()
        metrics = system.metrics.get_metrics()
        
        assert "test_operation_success" in metrics["counters"]
        assert "test_operation_duration" in metrics["histograms"]


class TestResilientSystem:
    """Test overall resilient system"""
    
    def test_system_health_monitoring(self):
        """Test system health monitoring"""
        system = get_resilient_system()
        
        # Add some components
        cb_config = CircuitBreakerConfig(failure_threshold=3)
        cb = system.add_circuit_breaker("test_service", cb_config)
        
        hc = system.add_health_check("test_health")
        hc.add_check("dummy", lambda: True)
        hc.run_checks()
        
        # Get health status
        health = system.get_system_health()
        
        assert health["system_name"] == "photonic_sim"
        assert "circuit_breakers" in health
        assert "health_checks" in health
        assert "cache" in health
        assert "metrics" in health
        
        # Initially should be healthy
        assert health["overall_healthy"] is True
        
    def test_concurrent_access(self):
        """Test thread safety of resilient components"""
        cache = CacheManager(max_size=100)
        
        def worker(thread_id):
            for i in range(50):
                key = f"thread_{thread_id}_key_{i}"
                value = f"value_{i}"
                cache.put(key, value)
                retrieved = cache.get(key)
                assert retrieved == value or retrieved is None  # May be evicted
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        stats = cache.stats()
        assert stats["size"] <= 100  # Respects max size


def test_integration_scenario():
    """Test integrated resilience scenario"""
    system = get_resilient_system()
    
    # Set up resilient service
    @with_circuit_breaker("integration_test", CircuitBreakerConfig(failure_threshold=2))
    @with_retry(max_attempts=3, base_delay=0.01)
    @with_metrics("integration_operation")
    def resilient_service(should_fail=False):
        if should_fail:
            raise ConnectionError("Service unavailable")
        return "service_response"
    
    # Test normal operation
    result = resilient_service(should_fail=False)
    assert result == "service_response"
    
    # Test with failures
    with pytest.raises(ConnectionError):
        resilient_service(should_fail=True)
    
    # Check system health
    health = system.get_system_health()
    assert "integration_test" in health["circuit_breakers"]
    
    # Check metrics
    metrics = system.metrics.get_metrics()
    assert "integration_operation_success" in metrics["counters"]
    assert "integration_operation_error" in metrics["counters"]


if __name__ == "__main__":
    # Run basic resilience tests
    print("Running resilience tests...")
    
    # Test circuit breaker
    test_cb = TestCircuitBreaker()
    test_cb.test_circuit_breaker_normal_operation()
    test_cb.test_circuit_breaker_failure_handling()
    print("✅ Circuit breaker tests passed")
    
    # Test retry policy
    test_retry = TestRetryPolicy()
    test_retry.test_retry_success_after_failures()
    test_retry.test_retry_exhausted()
    print("✅ Retry policy tests passed")
    
    # Test health checks
    test_hc = TestHealthCheck()
    test_hc.test_health_check_passing()
    test_hc.test_health_check_failing()
    print("✅ Health check tests passed")
    
    # Test cache
    test_cache = TestCacheManager()
    test_cache.test_cache_basic_operations()
    test_cache.test_cache_ttl_expiration()
    test_cache.test_cache_lru_eviction()
    print("✅ Cache manager tests passed")
    
    # Test metrics
    test_metrics = TestMetricsCollector()
    test_metrics.test_counter_metrics()
    test_metrics.test_histogram_metrics()
    test_metrics.test_gauge_metrics()
    print("✅ Metrics collector tests passed")
    
    # Test decorators
    test_decorators = TestDecorators()
    test_decorators.test_circuit_breaker_decorator()
    test_decorators.test_retry_decorator()
    test_decorators.test_metrics_decorator()
    print("✅ Decorator tests passed")
    
    # Test system
    test_system = TestResilientSystem()
    test_system.test_system_health_monitoring()
    test_system.test_concurrent_access()
    print("✅ Resilient system tests passed")
    
    # Integration test
    test_integration_scenario()
    print("✅ Integration tests passed")
    
    print("All resilience tests passed! 🛡️")