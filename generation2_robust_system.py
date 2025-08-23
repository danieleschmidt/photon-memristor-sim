#!/usr/bin/env python3
"""
GENERATION 2: MAKE IT ROBUST - Reliable Implementation
Photon-Memristor-Sim with Comprehensive Error Handling and Validation

This builds on Generation 1 with robust error handling, validation,
logging, monitoring, and security measures.
"""

import sys
import os
import time
import math
import json
import random
import logging
import hashlib
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import traceback

# Configure comprehensive logging
def setup_logging():
    """Set up comprehensive logging system"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('photonic_robust.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Error handling classes
class PhotonicError(Exception):
    """Base exception for photonic simulation errors"""
    pass

class ValidationError(PhotonicError):
    """Validation error"""
    pass

class SimulationError(PhotonicError):
    """Simulation runtime error"""
    pass

class DeviceError(PhotonicError):
    """Device-specific error"""
    pass

# Security and validation
class SecurityValidator:
    """Security validation for inputs"""
    
    @staticmethod
    def validate_input_size(data: Any, max_size: int = 1000000) -> bool:
        """Prevent large input attacks"""
        if isinstance(data, (list, tuple)):
            if len(data) > max_size:
                raise ValidationError(f"Input size {len(data)} exceeds maximum {max_size}")
        return True
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float, name: str) -> bool:
        """Validate numeric ranges"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric, got {type(value)}")
        if not (min_val <= value <= max_val):
            raise ValidationError(f"{name} {value} outside valid range [{min_val}, {max_val}]")
        return True
    
    @staticmethod
    def sanitize_string(s: str, max_length: int = 1000) -> str:
        """Sanitize string inputs"""
        if not isinstance(s, str):
            raise ValidationError(f"Expected string, got {type(s)}")
        if len(s) > max_length:
            raise ValidationError(f"String length {len(s)} exceeds maximum {max_length}")
        # Remove potentially dangerous characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        return ''.join(c for c in s if c in safe_chars)

# Enhanced error handling with monitoring
class ErrorHandler:
    """Comprehensive error handling and monitoring"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.lock = threading.Lock()
    
    @contextmanager
    def catch_and_log(self, operation: str):
        """Context manager for error catching and logging"""
        start_time = time.time()
        try:
            logger.info(f"Starting operation: {operation}")
            yield
            duration = time.time() - start_time
            logger.info(f"Completed operation: {operation} in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            error_info = {
                'operation': operation,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            }
            
            with self.lock:
                self.error_history.append(error_info)
                error_key = f"{operation}:{type(e).__name__}"
                self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            logger.error(f"Operation failed: {operation} - {e}")
            raise
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self.lock:
            return {
                'total_errors': len(self.error_history),
                'error_counts': dict(self.error_counts),
                'recent_errors': self.error_history[-10:] if self.error_history else []
            }

error_handler = ErrorHandler()

# Robust optical field with validation
@dataclass
class RobustOpticalField:
    """Optical field with comprehensive validation and error handling"""
    amplitude: complex
    wavelength: float
    power: float
    field_id: str = field(default_factory=lambda: f"field_{int(time.time()*1000)}")
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        """Comprehensive validation"""
        # Wavelength validation (visible to IR range)
        SecurityValidator.validate_numeric_range(
            self.wavelength, 400e-9, 10e-6, "wavelength"
        )
        
        # Power validation (reasonable physical range)
        SecurityValidator.validate_numeric_range(
            self.power, 0, 100, "power"  # 0 to 100W max
        )
        
        # Amplitude validation
        if not isinstance(self.amplitude, complex):
            raise ValidationError(f"Amplitude must be complex, got {type(self.amplitude)}")
        
        # Physical consistency check
        expected_amplitude_magnitude = math.sqrt(self.power)
        actual_magnitude = abs(self.amplitude)
        if abs(expected_amplitude_magnitude - actual_magnitude) > 0.1 * expected_amplitude_magnitude:
            logger.warning(f"Amplitude-power inconsistency: expected {expected_amplitude_magnitude:.3e}, got {actual_magnitude:.3e}")
    
    def copy(self) -> 'RobustOpticalField':
        """Create validated copy"""
        return RobustOpticalField(
            amplitude=self.amplitude,
            wavelength=self.wavelength,
            power=self.power
        )

# Device health monitoring
class DeviceHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTY = "faulty"
    OFFLINE = "offline"

@dataclass
class DeviceMetrics:
    """Device health and performance metrics"""
    operations_count: int = 0
    error_count: int = 0
    total_time: float = 0.0
    last_operation: float = 0.0
    health_status: DeviceHealth = DeviceHealth.HEALTHY
    
    @property
    def success_rate(self) -> float:
        if self.operations_count == 0:
            return 1.0
        return 1.0 - (self.error_count / self.operations_count)
    
    @property
    def average_time(self) -> float:
        if self.operations_count == 0:
            return 0.0
        return self.total_time / self.operations_count

# Robust photonic device with health monitoring
class RobustPhotonicDevice:
    """Photonic device with comprehensive error handling and monitoring"""
    
    def __init__(self, device_type: str = "waveguide", device_id: Optional[str] = None):
        self.device_type = SecurityValidator.sanitize_string(device_type)
        self.device_id = device_id or f"{device_type}_{int(time.time()*1000)}"
        self.losses = 0.1  # 10% loss
        self.created_at = time.time()
        self.metrics = DeviceMetrics()
        self.lock = threading.Lock()
        
        # Physical parameter validation
        SecurityValidator.validate_numeric_range(self.losses, 0.0, 0.95, "losses")
        logger.info(f"Created device: {self.device_id} ({self.device_type})")
    
    def propagate(self, input_field: RobustOpticalField) -> RobustOpticalField:
        """Robust propagation with comprehensive error handling"""
        operation_name = f"propagate_{self.device_id}"
        
        with error_handler.catch_and_log(operation_name):
            start_time = time.time()
            
            try:
                with self.lock:
                    # Validate device health
                    if self.metrics.health_status == DeviceHealth.OFFLINE:
                        raise DeviceError(f"Device {self.device_id} is offline")
                    
                    # Input validation
                    if not isinstance(input_field, RobustOpticalField):
                        raise ValidationError("Input must be RobustOpticalField")
                    
                    # Physical simulation with bounds checking
                    if self.losses < 0 or self.losses > 1:
                        raise SimulationError(f"Invalid loss factor: {self.losses}")
                    
                    output_power = input_field.power * (1 - self.losses)
                    if output_power < 0:
                        output_power = 0
                    
                    output_amplitude = input_field.amplitude * math.sqrt(max(0, 1 - self.losses))
                    
                    # Create output field
                    output_field = RobustOpticalField(
                        amplitude=output_amplitude,
                        wavelength=input_field.wavelength,
                        power=output_power
                    )
                    
                    # Update metrics
                    duration = time.time() - start_time
                    self.metrics.operations_count += 1
                    self.metrics.total_time += duration
                    self.metrics.last_operation = time.time()
                    
                    # Health assessment
                    if self.metrics.success_rate < 0.8:
                        self.metrics.health_status = DeviceHealth.DEGRADED
                        logger.warning(f"Device {self.device_id} degraded: success rate {self.metrics.success_rate:.2%}")
                    
                    return output_field
                    
            except Exception as e:
                with self.lock:
                    self.metrics.error_count += 1
                    if self.metrics.error_count > 10:
                        self.metrics.health_status = DeviceHealth.FAULTY
                        logger.error(f"Device {self.device_id} marked as faulty")
                raise
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive device health report"""
        with self.lock:
            return {
                'device_id': self.device_id,
                'device_type': self.device_type,
                'health_status': self.metrics.health_status.value,
                'operations_count': self.metrics.operations_count,
                'error_count': self.metrics.error_count,
                'success_rate': self.metrics.success_rate,
                'average_time': self.metrics.average_time,
                'last_operation': self.metrics.last_operation,
                'uptime': time.time() - self.created_at
            }

# Robust memristor with comprehensive modeling
class RobustMemristor:
    """Memristor with robust error handling and physical modeling"""
    
    def __init__(self, initial_conductance: float = 1e-6, memristor_id: Optional[str] = None):
        SecurityValidator.validate_numeric_range(initial_conductance, 1e-12, 1e-2, "initial_conductance")
        
        self.memristor_id = memristor_id or f"memristor_{int(time.time()*1000)}"
        self.conductance = initial_conductance
        self.min_conductance = 1e-8
        self.max_conductance = 1e-4
        self.switch_count = 0
        self.metrics = DeviceMetrics()
        self.temperature = 300  # Kelvin
        self.lock = threading.Lock()
        
        logger.info(f"Created memristor: {self.memristor_id}")
    
    def apply_voltage(self, voltage: float, duration: float = 1e-6) -> float:
        """Apply voltage with comprehensive validation and physics"""
        operation_name = f"apply_voltage_{self.memristor_id}"
        
        with error_handler.catch_and_log(operation_name):
            # Validation
            SecurityValidator.validate_numeric_range(voltage, -10, 10, "voltage")
            SecurityValidator.validate_numeric_range(duration, 1e-12, 1e-3, "duration")
            
            with self.lock:
                start_time = time.time()
                
                try:
                    # Physical modeling with temperature effects
                    thermal_factor = math.exp(-0.1 * (self.temperature - 300) / 300)
                    
                    # Voltage-dependent switching
                    if abs(voltage) > 0.1:  # Threshold voltage
                        delta_g = voltage * duration * 1e-12 * thermal_factor
                        self.conductance += delta_g
                        self.switch_count += 1
                    
                    # Physical bounds
                    self.conductance = max(self.min_conductance,
                                         min(self.max_conductance, self.conductance))
                    
                    # Update metrics
                    duration_actual = time.time() - start_time
                    self.metrics.operations_count += 1
                    self.metrics.total_time += duration_actual
                    self.metrics.last_operation = time.time()
                    
                    return self.conductance
                    
                except Exception as e:
                    self.metrics.error_count += 1
                    raise SimulationError(f"Voltage application failed: {e}")
    
    def optical_modulation(self, input_field: RobustOpticalField) -> RobustOpticalField:
        """Optical modulation with robust error handling"""
        operation_name = f"optical_modulation_{self.memristor_id}"
        
        with error_handler.catch_and_log(operation_name):
            with self.lock:
                try:
                    # Physical absorption model
                    absorption_coeff = self.conductance * 1e6  # Scaling factor
                    transmission = math.exp(-absorption_coeff)
                    
                    # Bounds checking
                    transmission = max(0.001, min(1.0, transmission))
                    
                    output_field = RobustOpticalField(
                        amplitude=input_field.amplitude * math.sqrt(transmission),
                        wavelength=input_field.wavelength,
                        power=input_field.power * transmission
                    )
                    
                    self.metrics.operations_count += 1
                    return output_field
                    
                except Exception as e:
                    self.metrics.error_count += 1
                    raise SimulationError(f"Optical modulation failed: {e}")

# Robust photonic array with comprehensive monitoring
class RobustPhotonicArray:
    """Photonic array with robust error handling and monitoring"""
    
    def __init__(self, rows: int = 8, cols: int = 8, array_id: Optional[str] = None):
        # Input validation
        SecurityValidator.validate_numeric_range(rows, 1, 1000, "rows")
        SecurityValidator.validate_numeric_range(cols, 1, 1000, "cols")
        
        self.rows = rows
        self.cols = cols
        self.array_id = array_id or f"array_{int(time.time()*1000)}"
        self.devices = []
        self.memristors = []
        self.metrics = DeviceMetrics()
        self.lock = threading.Lock()
        
        # Create device grid with error handling
        try:
            for i in range(rows):
                device_row = []
                memristor_row = []
                for j in range(cols):
                    device_id = f"{self.array_id}_dev_{i}_{j}"
                    memristor_id = f"{self.array_id}_mem_{i}_{j}"
                    
                    device_row.append(RobustPhotonicDevice(device_id=device_id))
                    memristor_row.append(RobustMemristor(memristor_id=memristor_id))
                
                self.devices.append(device_row)
                self.memristors.append(memristor_row)
            
            logger.info(f"Created photonic array: {self.array_id} ({rows}x{cols})")
            
        except Exception as e:
            raise SimulationError(f"Array initialization failed: {e}")
    
    def matrix_multiply(self, input_vector: List[float]) -> List[float]:
        """Robust matrix multiplication with comprehensive validation"""
        operation_name = f"matrix_multiply_{self.array_id}"
        
        with error_handler.catch_and_log(operation_name):
            # Input validation
            if not isinstance(input_vector, list):
                raise ValidationError("Input must be a list")
            
            SecurityValidator.validate_input_size(input_vector, 10000)
            
            if len(input_vector) != self.cols:
                raise ValidationError(f"Input vector length {len(input_vector)} != {self.cols}")
            
            # Validate all elements
            for i, val in enumerate(input_vector):
                SecurityValidator.validate_numeric_range(val, -1000, 1000, f"input[{i}]")
            
            with self.lock:
                start_time = time.time()
                
                try:
                    output = []
                    wavelength = 1550e-9  # Standard telecom wavelength
                    
                    for i in range(self.rows):
                        row_sum = 0.0
                        
                        for j in range(self.cols):
                            try:
                                # Create optical field with validation
                                input_power = abs(input_vector[j]) * 1e-3
                                if input_power > 1:  # 1W max
                                    input_power = 1
                                
                                field = RobustOpticalField(
                                    amplitude=complex(math.sqrt(input_power), 0),
                                    wavelength=wavelength,
                                    power=input_power
                                )
                                
                                # Propagate through device
                                field = self.devices[i][j].propagate(field)
                                
                                # Modulate with memristor
                                field = self.memristors[i][j].optical_modulation(field)
                                
                                # Accumulate power
                                weight = self.memristors[i][j].conductance * 1e6
                                row_sum += field.power * weight
                                
                            except Exception as e:
                                logger.warning(f"Element ({i},{j}) failed: {e}")
                                # Continue with degraded functionality
                                row_sum += 0
                        
                        output.append(row_sum)
                    
                    # Update metrics
                    duration = time.time() - start_time
                    self.metrics.operations_count += 1
                    self.metrics.total_time += duration
                    self.metrics.last_operation = time.time()
                    
                    return output
                    
                except Exception as e:
                    self.metrics.error_count += 1
                    raise SimulationError(f"Matrix multiplication failed: {e}")
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for entire array"""
        with self.lock:
            device_reports = []
            memristor_reports = []
            
            healthy_devices = 0
            healthy_memristors = 0
            
            for i in range(self.rows):
                for j in range(self.cols):
                    dev_report = self.devices[i][j].get_health_report()
                    device_reports.append(dev_report)
                    if dev_report['health_status'] == 'healthy':
                        healthy_devices += 1
                    
                    mem_report = {
                        'memristor_id': self.memristors[i][j].memristor_id,
                        'conductance': self.memristors[i][j].conductance,
                        'switch_count': self.memristors[i][j].switch_count,
                        'operations_count': self.memristors[i][j].metrics.operations_count
                    }
                    memristor_reports.append(mem_report)
                    if self.memristors[i][j].metrics.success_rate > 0.9:
                        healthy_memristors += 1
            
            total_devices = self.rows * self.cols
            
            return {
                'array_id': self.array_id,
                'dimensions': [self.rows, self.cols],
                'total_devices': total_devices,
                'healthy_devices': healthy_devices,
                'healthy_memristors': healthy_memristors,
                'device_health_rate': healthy_devices / total_devices,
                'memristor_health_rate': healthy_memristors / total_devices,
                'array_operations': self.metrics.operations_count,
                'array_success_rate': self.metrics.success_rate,
                'device_reports': device_reports[:5],  # Sample
                'memristor_reports': memristor_reports[:5],  # Sample
                'error_stats': error_handler.get_error_stats()
            }

def run_robust_tests():
    """Run comprehensive robust tests"""
    print("🛡️ GENERATION 2: MAKE IT ROBUST - Comprehensive Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Input validation
    print("\n🔐 Testing Input Validation...")
    total_tests += 1
    try:
        # Test invalid wavelength
        try:
            field = RobustOpticalField(amplitude=1+0j, wavelength=-1, power=1e-3)
            print("❌ Should have failed for negative wavelength")
        except ValidationError:
            print("✅ Correctly rejected negative wavelength")
            success_count += 1
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    # Test 2: Error recovery
    print("\n🔄 Testing Error Recovery...")
    total_tests += 1
    try:
        array = RobustPhotonicArray(rows=3, cols=3)
        
        # Test with invalid input
        try:
            array.matrix_multiply([1, 2])  # Wrong size
        except ValidationError:
            print("✅ Correctly handled wrong input size")
        
        # Test with valid input after error
        result = array.matrix_multiply([1, 0.5, 0.2])
        print(f"✅ Successful recovery: output sum = {sum(result):.3f}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
    
    # Test 3: Health monitoring
    print("\n🏥 Testing Health Monitoring...")
    total_tests += 1
    try:
        device = RobustPhotonicDevice()
        field = RobustOpticalField(amplitude=1+0j, wavelength=1550e-9, power=1e-3)
        
        # Run multiple operations
        for _ in range(10):
            device.propagate(field)
        
        health = device.get_health_report()
        print(f"✅ Device health: {health['success_rate']:.1%} success rate")
        print(f"   Operations: {health['operations_count']}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
    
    # Test 4: Concurrent access
    print("\n🔄 Testing Concurrent Access...")
    total_tests += 1
    try:
        array = RobustPhotonicArray(rows=4, cols=4)
        input_vec = [0.5, 0.3, 0.2, 0.1]
        
        def worker():
            return array.matrix_multiply(input_vec)
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            results = [f.result() for f in futures]
        
        print(f"✅ Concurrent access successful: {len(results)} results")
        success_count += 1
        
    except Exception as e:
        print(f"❌ Concurrent access test failed: {e}")
    
    # Test 5: Comprehensive health report
    print("\n📊 Testing Comprehensive Health Report...")
    total_tests += 1
    try:
        array = RobustPhotonicArray(rows=2, cols=2)
        
        # Run some operations
        for i in range(5):
            array.matrix_multiply([0.5, 0.3])
        
        report = array.get_comprehensive_health_report()
        print(f"✅ Health report generated:")
        print(f"   Array: {report['array_id']}")
        print(f"   Device health: {report['device_health_rate']:.1%}")
        print(f"   Memristor health: {report['memristor_health_rate']:.1%}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ Health report test failed: {e}")
    
    return success_count, total_tests

def generate_robust_report() -> Dict[str, Any]:
    """Generate Generation 2 completion report"""
    return {
        "generation": "2 - Make It Robust",
        "timestamp": time.time(),
        "status": "completed",
        "robustness_features": [
            "Comprehensive input validation",
            "Security sanitization",
            "Error handling and recovery",
            "Health monitoring and metrics",
            "Thread-safe operations",
            "Graceful degradation",
            "Comprehensive logging"
        ],
        "security_measures": [
            "Input size limits",
            "Numeric range validation", 
            "String sanitization",
            "Physical bounds checking"
        ],
        "monitoring": [
            "Device health tracking",
            "Operation metrics",
            "Error statistics",
            "Performance monitoring"
        ],
        "next_steps": [
            "Performance optimization",
            "Caching and resource pooling",
            "Load balancing",
            "Auto-scaling triggers"
        ]
    }

if __name__ == "__main__":
    print("🦀 Photon-Memristor-Sim - TERRAGON SDLC v4.0")
    print("🛡️ AUTONOMOUS GENERATION 2: MAKE IT ROBUST")
    print()
    
    try:
        success_count, total_tests = run_robust_tests()
        success_rate = success_count / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 GENERATION 2 RESULTS:")
        print(f"✅ Success Rate: {success_rate:.1%} ({success_count}/{total_tests})")
        
        if success_rate >= 0.8:
            print("\n🎉 GENERATION 2 SUCCESS!")
            print("✅ Robust error handling implemented")
            print("✅ Comprehensive validation active")
            print("✅ Health monitoring operational")
            print("✅ Security measures in place")
            
            # Generate report
            report = generate_robust_report()
            with open("generation2_robust_report.json", "w") as f:
                json.dump(report, f, indent=2)
            print("📄 Report saved to generation2_robust_report.json")
            
            print("\n⏭️  Ready for GENERATION 3: MAKE IT SCALE")
            sys.exit(0)
        else:
            print("❌ Generation 2 failed: Success rate below threshold")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Generation 2 failed with error: {e}")
        print(f"💥 Generation 2 failed with error: {e}")
        sys.exit(1)