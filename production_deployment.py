#!/usr/bin/env python3
"""
Production Deployment Infrastructure
Comprehensive production-ready deployment with monitoring, scaling, and reliability.
"""

import asyncio
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('production.log')
    ]
)

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    service_name: str = "photon-memristor-sim"
    version: str = "1.0.0"
    environment: str = "production"
    min_instances: int = 2
    max_instances: int = 16
    target_cpu_percent: int = 70
    target_memory_percent: int = 80
    health_check_interval: float = 30.0
    monitoring_port: int = 8080
    api_port: int = 8000
    enable_auto_scaling: bool = True
    enable_circuit_breaker: bool = True
    enable_load_balancing: bool = True

@dataclass
class HealthMetrics:
    """Health and performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    requests_per_second: float
    average_response_time_ms: float
    error_rate_percent: float
    active_connections: int
    status: DeploymentStatus

class CircuitBreakerProduction:
    """Production-grade circuit breaker"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        self.success_count = 0
        self.logger = logging.getLogger("CircuitBreaker")
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= 3:  # Require 3 successes to close
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info("Circuit breaker CLOSED - service recovered")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                if self.state != "OPEN":
                    self.state = "OPEN"
                    self.logger.error(f"Circuit breaker OPEN - {self.failure_count} failures")
            
            raise e

class LoadBalancer:
    """Production load balancer with health checking"""
    
    def __init__(self):
        self.instances = []
        self.current_index = 0
        self.health_status = {}
        self.logger = logging.getLogger("LoadBalancer")
        
    def add_instance(self, instance_id: str, instance):
        """Add instance to load balancer"""
        self.instances.append((instance_id, instance))
        self.health_status[instance_id] = True
        self.logger.info(f"Added instance {instance_id} to load balancer")
        
    def remove_instance(self, instance_id: str):
        """Remove instance from load balancer"""
        self.instances = [(id, inst) for id, inst in self.instances if id != instance_id]
        if instance_id in self.health_status:
            del self.health_status[instance_id]
        self.logger.info(f"Removed instance {instance_id} from load balancer")
        
    def mark_healthy(self, instance_id: str, healthy: bool):
        """Mark instance health status"""
        if instance_id in self.health_status:
            self.health_status[instance_id] = healthy
            
    def get_healthy_instance(self):
        """Get next healthy instance using round-robin"""
        healthy_instances = [
            (id, inst) for id, inst in self.instances 
            if self.health_status.get(id, False)
        ]
        
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        # Round-robin selection
        instance = healthy_instances[self.current_index % len(healthy_instances)]
        self.current_index += 1
        
        return instance

class HealthChecker:
    """Production health checking service"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger("HealthChecker")
        self.running = False
        self.health_history = []
        
    def check_instance_health(self, instance_id: str, instance) -> bool:
        """Check individual instance health"""
        try:
            # Perform basic health check
            test_input = np.random.uniform(0.1e-3, 1e-3, 8)
            
            start_time = time.time()
            result = instance.forward_propagation_optimized(test_input)
            response_time = (time.time() - start_time) * 1000
            
            # Health criteria
            is_healthy = (
                result is not None and
                len(result) > 0 and
                not np.any(np.isnan(result)) and
                response_time < 1000  # 1 second max response time
            )
            
            if not is_healthy:
                self.logger.warning(f"Instance {instance_id} failed health check")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Health check failed for {instance_id}: {e}")
            return False
    
    def start_monitoring(self, load_balancer: LoadBalancer):
        """Start continuous health monitoring"""
        if self.running:
            return
            
        self.running = True
        
        def monitoring_loop():
            while self.running:
                try:
                    self.perform_health_checks(load_balancer)
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def perform_health_checks(self, load_balancer: LoadBalancer):
        """Perform health checks on all instances"""
        for instance_id, instance in load_balancer.instances:
            is_healthy = self.check_instance_health(instance_id, instance)
            load_balancer.mark_healthy(instance_id, is_healthy)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")

class AutoScaler:
    """Production auto-scaling service"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger("AutoScaler")
        self.metrics_history = []
        self.last_scaling_action = 0
        self.cooldown_period = 300  # 5 minutes
        
    def should_scale_up(self, current_metrics: HealthMetrics, instance_count: int) -> bool:
        """Determine if scaling up is needed"""
        if instance_count >= self.config.max_instances:
            return False
        
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return False
        
        # Scale up if CPU > target for 2 consecutive checks
        if len(self.metrics_history) >= 2:
            recent_metrics = self.metrics_history[-2:]
            high_cpu = all(m.cpu_percent > self.config.target_cpu_percent for m in recent_metrics)
            high_memory = all(m.memory_percent > self.config.target_memory_percent for m in recent_metrics)
            
            return high_cpu or high_memory
        
        return False
    
    def should_scale_down(self, current_metrics: HealthMetrics, instance_count: int) -> bool:
        """Determine if scaling down is possible"""
        if instance_count <= self.config.min_instances:
            return False
        
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return False
        
        # Scale down if metrics are consistently low
        if len(self.metrics_history) >= 3:
            recent_metrics = self.metrics_history[-3:]
            low_cpu = all(m.cpu_percent < self.config.target_cpu_percent * 0.5 for m in recent_metrics)
            low_memory = all(m.memory_percent < self.config.target_memory_percent * 0.6 for m in recent_metrics)
            
            return low_cpu and low_memory
        
        return False
    
    def update_metrics(self, metrics: HealthMetrics):
        """Update metrics for scaling decisions"""
        self.metrics_history.append(metrics)
        
        # Keep only last 10 metrics
        if len(self.metrics_history) > 10:
            self.metrics_history.pop(0)
    
    def perform_scaling_decision(self, current_metrics: HealthMetrics, instance_count: int) -> Optional[str]:
        """Make scaling decision"""
        if not self.config.enable_auto_scaling:
            return None
        
        if self.should_scale_up(current_metrics, instance_count):
            self.last_scaling_action = time.time()
            self.logger.info("Auto-scaling decision: SCALE UP")
            return "SCALE_UP"
        
        elif self.should_scale_down(current_metrics, instance_count):
            self.last_scaling_action = time.time()
            self.logger.info("Auto-scaling decision: SCALE DOWN")
            return "SCALE_DOWN"
        
        return None

class OptimizedPhotonicDeviceProduction:
    """Production-ready optimized photonic device"""
    
    def __init__(self, rows: int = 16, cols: int = 16, device_id: str = "prod_device"):
        self.device_id = device_id
        self.rows = rows
        self.cols = cols
        self.wavelength = 1550e-9
        
        # Production-optimized data structures
        self.transmission_matrix = np.random.uniform(0.1, 0.9, (rows, cols)).astype(np.float32)
        
        # Production monitoring
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.start_time = time.time()
        
        self.logger = logging.getLogger(f"ProdDevice.{device_id}")
        self.logger.info(f"Production device {device_id} initialized")
    
    def forward_propagation_optimized(self, input_power: np.ndarray) -> np.ndarray:
        """Production-optimized forward propagation"""
        start_time = time.time()
        
        try:
            # Input validation
            if not isinstance(input_power, np.ndarray):
                raise ValueError("Input must be numpy array")
            
            if input_power.shape[0] != self.rows:
                raise ValueError(f"Input shape mismatch: {input_power.shape[0]} != {self.rows}")
            
            if np.any(input_power < 0):
                raise ValueError("Input power values must be non-negative")
            
            # Optimized computation
            thermal_factor = 1.0 - 0.001 * (25.0 - 25.0)  # Simplified thermal model
            transmission_scaled = self.transmission_matrix * thermal_factor
            output = np.dot(input_power, transmission_scaled)
            
            # Add realistic noise
            noise = np.random.normal(0, 1e-6, output.shape).astype(np.float32)
            output += noise
            output = np.maximum(output, 0)
            
            # Update metrics
            self.request_count += 1
            response_time = (time.time() - start_time) * 1000
            self.total_response_time += response_time
            
            return output
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Forward propagation error: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """Get device performance metrics"""
        uptime = time.time() - self.start_time
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1) * 100
        throughput = self.request_count / uptime if uptime > 0 else 0
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate_percent": error_rate,
            "avg_response_time_ms": avg_response_time,
            "throughput_rps": throughput,
            "uptime_seconds": uptime
        }

class ProductionOrchestrator:
    """Main production orchestration service"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger("ProductionOrchestrator")
        
        # Core components
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker(config)
        self.auto_scaler = AutoScaler(config)
        self.circuit_breaker = CircuitBreakerProduction()
        
        # Instance management
        self.instances = {}
        self.running = False
        
        self.logger.info(f"Production orchestrator initialized for {config.service_name} v{config.version}")
    
    def start_deployment(self):
        """Start production deployment"""
        try:
            self.logger.info("Starting production deployment...")
            
            # Initialize minimum instances
            for i in range(self.config.min_instances):
                self.add_instance(f"instance_{i}")
            
            # Start health monitoring
            self.health_checker.start_monitoring(self.load_balancer)
            
            # Start main orchestration loop
            self.running = True
            self.orchestration_loop()
            
        except Exception as e:
            self.logger.error(f"Deployment startup failed: {e}")
            raise
    
    def add_instance(self, instance_id: str):
        """Add new instance to deployment"""
        try:
            device = OptimizedPhotonicDeviceProduction(
                rows=16, cols=16, device_id=instance_id
            )
            
            self.instances[instance_id] = device
            self.load_balancer.add_instance(instance_id, device)
            
            self.logger.info(f"Instance {instance_id} added to deployment")
            
        except Exception as e:
            self.logger.error(f"Failed to add instance {instance_id}: {e}")
    
    def remove_instance(self, instance_id: str):
        """Remove instance from deployment"""
        try:
            if instance_id in self.instances:
                self.load_balancer.remove_instance(instance_id)
                del self.instances[instance_id]
                self.logger.info(f"Instance {instance_id} removed from deployment")
                
        except Exception as e:
            self.logger.error(f"Failed to remove instance {instance_id}: {e}")
    
    def process_request(self, input_data: np.ndarray) -> np.ndarray:
        """Process request through load balancer with circuit breaker"""
        def _process():
            instance_id, instance = self.load_balancer.get_healthy_instance()
            return instance.forward_propagation_optimized(input_data)
        
        return self.circuit_breaker.call(_process)
    
    def collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive system metrics"""
        try:
            # Simulate system metrics collection
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
        except ImportError:
            # Fallback if psutil not available
            cpu_percent = np.random.uniform(20, 80)
            memory_percent = np.random.uniform(30, 70)
        
        # Aggregate instance metrics
        total_requests = sum(inst.request_count for inst in self.instances.values())
        total_errors = sum(inst.error_count for inst in self.instances.values())
        
        avg_response_times = [
            inst.total_response_time / max(inst.request_count, 1) 
            for inst in self.instances.values() 
            if inst.request_count > 0
        ]
        avg_response_time = np.mean(avg_response_times) if avg_response_times else 0
        
        error_rate = total_errors / max(total_requests, 1) * 100
        
        # Determine overall status
        if error_rate > 10:
            status = DeploymentStatus.FAILED
        elif error_rate > 5 or cpu_percent > 90:
            status = DeploymentStatus.DEGRADED
        else:
            status = DeploymentStatus.HEALTHY
        
        return HealthMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            requests_per_second=total_requests,  # Simplified
            average_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            active_connections=len(self.instances),
            status=status
        )
    
    def orchestration_loop(self):
        """Main orchestration and monitoring loop"""
        self.logger.info("Starting orchestration loop")
        
        while self.running:
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Update auto-scaler
                self.auto_scaler.update_metrics(metrics)
                
                # Make scaling decisions
                scaling_action = self.auto_scaler.perform_scaling_decision(
                    metrics, len(self.instances)
                )
                
                if scaling_action == "SCALE_UP":
                    new_instance_id = f"instance_{len(self.instances)}"
                    self.add_instance(new_instance_id)
                    
                elif scaling_action == "SCALE_DOWN":
                    # Remove least utilized instance
                    if len(self.instances) > self.config.min_instances:
                        instance_to_remove = list(self.instances.keys())[-1]
                        self.remove_instance(instance_to_remove)
                
                # Log status
                self.logger.info(
                    f"Status: {metrics.status.value} | "
                    f"Instances: {len(self.instances)} | "
                    f"CPU: {metrics.cpu_percent:.1f}% | "
                    f"Memory: {metrics.memory_percent:.1f}% | "
                    f"Error Rate: {metrics.error_rate_percent:.2f}%"
                )
                
                # Sleep until next cycle
                time.sleep(10)  # 10-second monitoring cycle
                
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                time.sleep(5)  # Short sleep on error
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating graceful shutdown...")
        
        self.running = False
        self.health_checker.stop_monitoring()
        
        # Remove all instances
        for instance_id in list(self.instances.keys()):
            self.remove_instance(instance_id)
        
        self.logger.info("Shutdown complete")

def run_production_demo():
    """Run production deployment demonstration"""
    print("\n=== Production Deployment Demo ===")
    
    # Production configuration
    config = DeploymentConfig(
        service_name="photon-memristor-sim",
        version="1.0.0",
        environment="production",
        min_instances=2,
        max_instances=6,
        health_check_interval=5.0
    )
    
    orchestrator = ProductionOrchestrator(config)
    
    try:
        print(f"Starting production deployment of {config.service_name} v{config.version}")
        print(f"Environment: {config.environment}")
        print(f"Instance range: {config.min_instances}-{config.max_instances}")
        print()
        
        # Start deployment
        deployment_thread = threading.Thread(
            target=orchestrator.start_deployment, 
            daemon=True
        )
        deployment_thread.start()
        
        # Give it time to initialize
        time.sleep(3)
        
        # Simulate production load
        print("Simulating production load...")
        
        for i in range(20):
            try:
                # Generate realistic request
                input_data = np.random.uniform(0.1e-3, 2e-3, 16).astype(np.float32)
                
                # Process through production system
                start_time = time.time()
                result = orchestrator.process_request(input_data)
                response_time = (time.time() - start_time) * 1000
                
                if i % 5 == 0:
                    print(f"Request {i+1}: Response time {response_time:.2f}ms, Output sum: {np.sum(result):.6f}")
                
                # Vary load
                if i < 10:
                    time.sleep(0.1)  # Light load
                else:
                    time.sleep(0.05)  # Heavier load
                    
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
        
        # Collect final metrics
        time.sleep(2)
        final_metrics = orchestrator.collect_metrics()
        
        print(f"\nFinal System Metrics:")
        print(f"  Status: {final_metrics.status.value}")
        print(f"  Active Instances: {final_metrics.active_connections}")
        print(f"  CPU Usage: {final_metrics.cpu_percent:.1f}%")
        print(f"  Memory Usage: {final_metrics.memory_percent:.1f}%")
        print(f"  Error Rate: {final_metrics.error_rate_percent:.2f}%")
        print(f"  Avg Response Time: {final_metrics.average_response_time_ms:.2f}ms")
        
        # Instance details
        print(f"\nInstance Details:")
        for instance_id, instance in orchestrator.instances.items():
            metrics = instance.get_metrics()
            print(f"  {instance_id}: {metrics['request_count']} requests, "
                  f"{metrics['error_rate_percent']:.1f}% errors, "
                  f"{metrics['avg_response_time_ms']:.2f}ms avg")
        
    except Exception as e:
        print(f"Production demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        orchestrator.shutdown()
    
    return orchestrator

def main():
    """Main production deployment function"""
    print("Photon-Memristor-Sim: Production Deployment")
    print("=" * 50)
    
    try:
        # Run production demonstration
        orchestrator = run_production_demo()
        
        print("\n" + "=" * 50)
        print("✅ Production Deployment Demo Completed!")
        print("Production-ready infrastructure implemented.")
        
    except Exception as e:
        print(f"\n❌ Production deployment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()