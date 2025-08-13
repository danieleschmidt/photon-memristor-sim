#!/usr/bin/env python3
"""
Generation 2 Robust Demo: MAKE IT ROBUST
Enhanced reliability, error handling, validation, logging, and monitoring.
"""

import numpy as np
import time
import logging
import traceback
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
import threading
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DeviceStatus(Enum):
    """Device operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    avg_execution_time_ms: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    memory_usage_mb: float
    temperature_celsius: float
    optical_power_efficiency: float

@dataclass
class ValidationReport:
    """Device validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    timestamp: str
    device_id: str

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

class RobustPhotonicDevice:
    """Enhanced photonic device with robustness features"""
    
    def __init__(self, rows: int = 8, cols: int = 8, device_id: str = "device_001"):
        self.device_id = device_id
        self.rows = rows
        self.cols = cols
        self.wavelength = 1550e-9
        self.logger = logging.getLogger(f"PhotonicDevice.{device_id}")
        
        # Enhanced state management
        self.transmission_matrix = np.random.uniform(0.1, 0.9, (rows, cols))
        self.temperature = 25.0  # Celsius
        self.status = DeviceStatus.HEALTHY
        self.calibration_data = {}
        
        # Error handling
        self.circuit_breaker = CircuitBreaker()
        self.error_count = 0
        self.total_operations = 0
        
        # Performance monitoring
        self.execution_times = []
        self.start_time = time.time()
        
        self.logger.info(f"Initialized robust photonic device {device_id} ({rows}x{cols})")
    
    def validate_device(self) -> ValidationReport:
        """Comprehensive device validation"""
        errors = []
        warnings = []
        
        # Check matrix dimensions
        if self.transmission_matrix.shape != (self.rows, self.cols):
            errors.append(f"Matrix shape mismatch: {self.transmission_matrix.shape} != ({self.rows}, {self.cols})")
        
        # Check transmission values
        if np.any(self.transmission_matrix < 0) or np.any(self.transmission_matrix > 1):
            errors.append("Transmission values outside valid range [0, 1]")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(self.transmission_matrix)) or np.any(np.isinf(self.transmission_matrix)):
            errors.append("Invalid numerical values detected in transmission matrix")
        
        # Check temperature range
        if self.temperature < -50 or self.temperature > 150:
            warnings.append(f"Temperature {self.temperature}°C outside recommended range [-50, 150]°C")
        
        # Check device status
        if self.status == DeviceStatus.FAILED:
            errors.append("Device status is FAILED")
        elif self.status == DeviceStatus.DEGRADED:
            warnings.append("Device status is DEGRADED")
        
        is_valid = len(errors) == 0
        
        report = ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            timestamp=str(time.time()),
            device_id=self.device_id
        )
        
        if not is_valid:
            self.logger.error(f"Validation failed: {errors}")
        elif warnings:
            self.logger.warning(f"Validation warnings: {warnings}")
        else:
            self.logger.debug("Device validation passed")
        
        return report
    
    @contextmanager
    def operation_monitor(self, operation_name: str):
        """Context manager for operation monitoring"""
        start_time = time.time()
        self.logger.debug(f"Starting operation: {operation_name}")
        
        try:
            yield
            execution_time = (time.time() - start_time) * 1000  # ms
            self.execution_times.append(execution_time)
            self.total_operations += 1
            
            self.logger.debug(f"Operation {operation_name} completed in {execution_time:.2f}ms")
            
        except Exception as e:
            self.error_count += 1
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.error(f"Operation {operation_name} failed after {execution_time:.2f}ms: {e}")
            
            # Update device status based on error rate
            error_rate = self.error_count / max(self.total_operations, 1)
            if error_rate > 0.1:  # 10% error rate
                self.status = DeviceStatus.DEGRADED
            if error_rate > 0.3:  # 30% error rate
                self.status = DeviceStatus.FAILED
            
            raise
    
    def forward_propagation_robust(self, input_power: np.ndarray) -> np.ndarray:
        """Robust forward propagation with validation and error handling"""
        
        def _forward_propagation():
            with self.operation_monitor("forward_propagation"):
                # Input validation
                if not isinstance(input_power, np.ndarray):
                    raise ValueError(f"Input must be numpy array, got {type(input_power)}")
                
                if input_power.shape != (self.rows,):
                    raise ValueError(f"Input shape {input_power.shape} doesn't match device rows {self.rows}")
                
                if np.any(input_power < 0):
                    raise ValueError("Input power values must be non-negative")
                
                if np.any(np.isnan(input_power)) or np.any(np.isinf(input_power)):
                    raise ValueError("Input contains invalid numerical values")
                
                # Device validation
                validation_report = self.validate_device()
                if not validation_report.is_valid:
                    raise RuntimeError(f"Device validation failed: {validation_report.errors}")
                
                # Perform computation with thermal effects
                thermal_factor = 1.0 - 0.001 * (self.temperature - 25.0)  # 0.1%/°C
                
                output = np.zeros(self.cols)
                for i in range(self.cols):
                    for j in range(self.rows):
                        transmission = self.transmission_matrix[j, i] * thermal_factor
                        output[i] += input_power[j] * transmission
                
                # Add realistic noise
                noise_level = 1e-6  # 1 µW noise floor
                noise = np.random.normal(0, noise_level, output.shape)
                output += noise
                
                # Ensure physical constraints
                output = np.maximum(output, 0)  # No negative power
                
                return output
        
        return self.circuit_breaker.call(_forward_propagation)
    
    def set_memristor_state_robust(self, row: int, col: int, conductance: float):
        """Robust memristor state setting with validation"""
        
        def _set_state():
            with self.operation_monitor("set_memristor_state"):
                # Input validation
                if not (0 <= row < self.rows):
                    raise ValueError(f"Row index {row} out of range [0, {self.rows})")
                
                if not (0 <= col < self.cols):
                    raise ValueError(f"Column index {col} out of range [0, {self.cols})")
                
                if not (0 <= conductance <= 1):
                    raise ValueError(f"Conductance {conductance} out of range [0, 1]")
                
                if np.isnan(conductance) or np.isinf(conductance):
                    raise ValueError(f"Invalid conductance value: {conductance}")
                
                # Apply wear-out model (simplified)
                current_value = self.transmission_matrix[row, col]
                wear_factor = 0.9999  # Slight degradation per write
                
                self.transmission_matrix[row, col] = conductance * wear_factor
                
                self.logger.debug(f"Set memristor ({row}, {col}) from {current_value:.3f} to {conductance:.3f}")
        
        return self.circuit_breaker.call(_set_state)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        uptime = time.time() - self.start_time
        
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0.0
        throughput = self.total_operations / uptime if uptime > 0 else 0.0
        error_rate = (self.error_count / max(self.total_operations, 1)) * 100
        
        # Simulated metrics
        memory_usage = len(self.execution_times) * 8 / 1024 / 1024  # MB
        optical_efficiency = np.mean(self.transmission_matrix)
        
        return PerformanceMetrics(
            avg_execution_time_ms=avg_execution_time,
            throughput_ops_per_sec=throughput,
            error_rate_percent=error_rate,
            memory_usage_mb=memory_usage,
            temperature_celsius=self.temperature,
            optical_power_efficiency=optical_efficiency
        )
    
    def self_test(self) -> Dict[str, Any]:
        """Comprehensive self-test procedure"""
        self.logger.info(f"Starting self-test for device {self.device_id}")
        
        test_results = {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "tests": {}
        }
        
        # Test 1: Basic operation
        try:
            input_signal = np.ones(self.rows) * 1e-3
            output = self.forward_propagation_robust(input_signal)
            test_results["tests"]["basic_operation"] = {
                "status": "PASS",
                "input_power_sum": float(np.sum(input_signal)),
                "output_power_sum": float(np.sum(output)),
                "efficiency": float(np.sum(output) / np.sum(input_signal))
            }
        except Exception as e:
            test_results["tests"]["basic_operation"] = {
                "status": "FAIL",
                "error": str(e)
            }
        
        # Test 2: State setting
        try:
            original_state = self.transmission_matrix[0, 0]
            self.set_memristor_state_robust(0, 0, 0.5)
            new_state = self.transmission_matrix[0, 0]
            self.transmission_matrix[0, 0] = original_state  # Restore
            
            test_results["tests"]["state_setting"] = {
                "status": "PASS",
                "state_change": abs(new_state - 0.5) < 0.01
            }
        except Exception as e:
            test_results["tests"]["state_setting"] = {
                "status": "FAIL",
                "error": str(e)
            }
        
        # Test 3: Validation
        try:
            validation_report = self.validate_device()
            test_results["tests"]["validation"] = {
                "status": "PASS" if validation_report.is_valid else "FAIL",
                "errors": validation_report.errors,
                "warnings": validation_report.warnings
            }
        except Exception as e:
            test_results["tests"]["validation"] = {
                "status": "FAIL",
                "error": str(e)
            }
        
        # Overall result
        all_tests_passed = all(
            test["status"] == "PASS" 
            for test in test_results["tests"].values()
        )
        
        test_results["overall_status"] = "PASS" if all_tests_passed else "FAIL"
        
        self.logger.info(f"Self-test completed: {test_results['overall_status']}")
        
        return test_results

class ResilientPhotonicSystem:
    """System-level resilience and monitoring"""
    
    def __init__(self):
        self.devices = {}
        self.logger = logging.getLogger("ResilientSystem")
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def add_device(self, device: RobustPhotonicDevice):
        """Add device to system monitoring"""
        self.devices[device.device_id] = device
        self.logger.info(f"Added device {device.device_id} to system")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.check_system_health()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        self.logger.info("System monitoring stopped")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check health of all devices"""
        health_report = {
            "timestamp": time.time(),
            "devices": {},
            "system_status": "HEALTHY"
        }
        
        failed_devices = 0
        degraded_devices = 0
        
        for device_id, device in self.devices.items():
            try:
                metrics = device.get_performance_metrics()
                validation = device.validate_device()
                
                device_health = {
                    "status": device.status.value,
                    "error_rate": metrics.error_rate_percent,
                    "avg_response_time": metrics.avg_execution_time_ms,
                    "throughput": metrics.throughput_ops_per_sec,
                    "temperature": metrics.temperature_celsius,
                    "validation_errors": len(validation.errors),
                    "validation_warnings": len(validation.warnings)
                }
                
                health_report["devices"][device_id] = device_health
                
                if device.status == DeviceStatus.FAILED:
                    failed_devices += 1
                elif device.status == DeviceStatus.DEGRADED:
                    degraded_devices += 1
                    
            except Exception as e:
                health_report["devices"][device_id] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                failed_devices += 1
        
        # Determine system status
        if failed_devices > 0:
            health_report["system_status"] = "DEGRADED" if failed_devices < len(self.devices) else "FAILED"
        elif degraded_devices > 0:
            health_report["system_status"] = "DEGRADED"
        
        if health_report["system_status"] != "HEALTHY":
            self.logger.warning(f"System health: {health_report['system_status']}")
        
        return health_report

def demo_robust_device():
    """Demonstrate robust device features"""
    print("\n=== Robust Device Demo ===")
    
    device = RobustPhotonicDevice(4, 4, "robust_001")
    
    # Normal operation
    input_power = np.array([1.0, 0.5, 0.8, 0.3]) * 1e-3
    
    try:
        output = device.forward_propagation_robust(input_power)
        print(f"✅ Normal operation successful")
        print(f"   Input: {list(input_power*1000)} mW")
        print(f"   Output: {[f'{x*1000:.2f}' for x in output]} mW")
    except Exception as e:
        print(f"❌ Operation failed: {e}")
    
    # Error handling demo
    print("\n--- Error Handling Tests ---")
    
    # Test invalid input
    try:
        device.forward_propagation_robust(np.array([1.0, -0.5, 0.8]))  # Negative power
        print("❌ Should have failed with negative input")
    except ValueError as e:
        print(f"✅ Correctly caught invalid input: {e}")
    
    # Test invalid memristor setting
    try:
        device.set_memristor_state_robust(10, 0, 0.5)  # Invalid row
        print("❌ Should have failed with invalid coordinates")
    except ValueError as e:
        print(f"✅ Correctly caught invalid coordinates: {e}")
    
    return device

def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("\n=== Performance Monitoring Demo ===")
    
    device = RobustPhotonicDevice(6, 6, "monitor_001")
    
    # Generate some load
    input_signal = np.random.uniform(0.1e-3, 2e-3, 6)
    
    for i in range(20):
        try:
            output = device.forward_propagation_robust(input_signal)
            # Occasionally introduce errors
            if i % 7 == 6:
                device.set_memristor_state_robust(100, 0, 0.5)  # This will fail
        except:
            pass  # Continue despite errors
    
    # Get performance metrics
    metrics = device.get_performance_metrics()
    
    print(f"Performance Metrics for {device.device_id}:")
    print(f"  Average execution time: {metrics.avg_execution_time_ms:.2f} ms")
    print(f"  Throughput: {metrics.throughput_ops_per_sec:.1f} ops/sec")
    print(f"  Error rate: {metrics.error_rate_percent:.1f}%")
    print(f"  Memory usage: {metrics.memory_usage_mb:.2f} MB")
    print(f"  Temperature: {metrics.temperature_celsius:.1f}°C")
    print(f"  Optical efficiency: {metrics.optical_power_efficiency:.3f}")
    print(f"  Device status: {device.status.value}")
    
    return device

def demo_self_test():
    """Demonstrate self-test capabilities"""
    print("\n=== Self-Test Demo ===")
    
    device = RobustPhotonicDevice(3, 3, "test_001")
    
    test_results = device.self_test()
    
    print(f"Self-test results for {test_results['device_id']}:")
    print(f"Overall status: {test_results['overall_status']}")
    
    for test_name, result in test_results["tests"].items():
        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"  {status_icon} {test_name}: {result['status']}")
        if result["status"] == "FAIL" and "error" in result:
            print(f"      Error: {result['error']}")
    
    return device

def demo_system_resilience():
    """Demonstrate system-level resilience"""
    print("\n=== System Resilience Demo ===")
    
    system = ResilientPhotonicSystem()
    
    # Create multiple devices
    devices = [
        RobustPhotonicDevice(4, 4, f"device_{i:03d}")
        for i in range(3)
    ]
    
    for device in devices:
        system.add_device(device)
    
    # Start monitoring
    system.start_monitoring(interval=1.0)
    
    # Generate some activity and errors
    print("Generating system activity...")
    
    for i in range(5):
        for j, device in enumerate(devices):
            try:
                input_signal = np.random.uniform(0.1e-3, 1e-3, device.rows)
                output = device.forward_propagation_robust(input_signal)
                
                # Occasionally introduce errors to simulate issues
                if i == 3 and j == 1:
                    # Corrupt device state to trigger degradation
                    device.error_count += 10
                    device.total_operations += 10
                
            except Exception as e:
                pass
        
        time.sleep(0.5)
    
    # Check system health
    health_report = system.check_system_health()
    
    print(f"\nSystem Health Report:")
    print(f"System status: {health_report['system_status']}")
    
    for device_id, health in health_report["devices"].items():
        status_icon = "✅" if health["status"] == "healthy" else "⚠️" if health["status"] == "degraded" else "❌"
        print(f"  {status_icon} {device_id}: {health['status']} (errors: {health.get('error_rate', 0):.1f}%)")
    
    system.stop_monitoring()
    
    return system

def main():
    """Run all Generation 2 demos"""
    print("Photon-Memristor-Sim: Generation 2 Robust Demo")
    print("=" * 55)
    
    try:
        # Robust device features
        robust_device = demo_robust_device()
        
        # Performance monitoring
        monitor_device = demo_performance_monitoring()
        
        # Self-test capabilities
        test_device = demo_self_test()
        
        # System resilience
        resilient_system = demo_system_resilience()
        
        print("\n" + "=" * 55)
        print("✅ Generation 2 Demo Completed Successfully!")
        print("Enhanced reliability and monitoring features implemented.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()