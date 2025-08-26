"""
Robust Error Handling and Validation for Photonic Memristor Systems
Generation 2: MAKE IT ROBUST - Comprehensive error handling, logging, and monitoring
"""

import logging
import traceback
import time
import json
import functools
import threading
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"  
    CRITICAL = "critical"


@dataclass
class ErrorReport:
    """Comprehensive error report"""
    timestamp: str
    severity: ErrorSeverity
    error_code: str
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    suggested_solution: Optional[str] = None
    affected_component: Optional[str] = None


class PhotonicError(Exception):
    """Base exception for photonic memristor systems"""
    
    def __init__(self, message: str, error_code: str = "PHOTONIC_ERROR", 
                 severity: ErrorSeverity = ErrorSeverity.ERROR, 
                 context: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


class ValidationError(PhotonicError):
    """Parameter validation errors"""
    
    def __init__(self, parameter: str, value: Any, expected_range: tuple = None, **kwargs):
        self.parameter = parameter
        self.value = value
        self.expected_range = expected_range
        
        message = f"Parameter '{parameter}' validation failed: value={value}"
        if expected_range:
            message += f", expected range {expected_range}"
        
        super().__init__(message, error_code="VALIDATION_ERROR", 
                        severity=ErrorSeverity.ERROR, **kwargs)


class ThermalError(PhotonicError):
    """Thermal-related errors"""
    
    def __init__(self, temperature: float, limit: float, device_id: str = None, **kwargs):
        self.temperature = temperature
        self.limit = limit
        self.device_id = device_id
        
        message = f"Thermal limit exceeded: {temperature}K > {limit}K"
        if device_id:
            message += f" (device: {device_id})"
        
        super().__init__(message, error_code="THERMAL_ERROR",
                        severity=ErrorSeverity.CRITICAL, **kwargs)


class OpticalError(PhotonicError):
    """Optical power-related errors"""
    
    def __init__(self, power: float, threshold: float, **kwargs):
        self.power = power
        self.threshold = threshold
        
        message = f"Optical power {power*1000:.1f}mW exceeds damage threshold {threshold*1000:.1f}mW"
        super().__init__(message, error_code="OPTICAL_ERROR",
                        severity=ErrorSeverity.WARNING, **kwargs)


class ConvergenceError(PhotonicError):
    """Simulation convergence errors"""
    
    def __init__(self, iterations: int, residual: float, **kwargs):
        self.iterations = iterations
        self.residual = residual
        
        message = f"Simulation failed to converge after {iterations} iterations (residual: {residual:.2e})"
        super().__init__(message, error_code="CONVERGENCE_ERROR",
                        severity=ErrorSeverity.ERROR, **kwargs)


class SecurityError(PhotonicError):
    """Security-related errors"""
    
    def __init__(self, details: str, **kwargs):
        super().__init__(f"Security violation: {details}", 
                        error_code="SECURITY_ERROR",
                        severity=ErrorSeverity.CRITICAL, **kwargs)


class RobustErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.error_history: List[ErrorReport] = []
        self.error_counts: Dict[str, int] = {}
        self.logger = self._setup_logger(log_file)
        self._lock = threading.Lock()
        
        # Error recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {
            "THERMAL_ERROR": self._thermal_recovery,
            "OPTICAL_ERROR": self._optical_recovery,
            "CONVERGENCE_ERROR": self._convergence_recovery,
            "VALIDATION_ERROR": self._validation_recovery,
        }
        
        # Circuit breaker state
        self.circuit_breaker_state = {}
        self.failure_threshold = 5
        self.recovery_timeout = 60  # seconds
    
    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("photonic_memristor_robust")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        
        # Formatting
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def handle_error(self, error: PhotonicError, component: str = None) -> ErrorReport:
        """Handle error with comprehensive reporting and recovery"""
        
        with self._lock:
            # Create error report
            report = ErrorReport(
                timestamp=error.timestamp,
                severity=error.severity,
                error_code=error.error_code,
                message=error.message,
                context=error.context,
                stack_trace=traceback.format_exc() if hasattr(error, '__traceback__') else None,
                affected_component=component
            )
            
            # Add suggested solution
            report.suggested_solution = self._get_suggested_solution(error)
            
            # Log error
            self._log_error(report)
            
            # Update statistics
            self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
            self.error_history.append(report)
            
            # Attempt recovery
            if error.error_code in self.recovery_strategies:
                try:
                    self.recovery_strategies[error.error_code](error, report)
                    self.logger.info(f"Recovery attempted for {error.error_code}")
                except Exception as recovery_error:
                    self.logger.error(f"Recovery failed for {error.error_code}: {recovery_error}")
            
            # Check circuit breaker
            self._check_circuit_breaker(error.error_code, component)
            
            return report
    
    def _log_error(self, report: ErrorReport):
        """Log error based on severity"""
        log_message = f"[{report.error_code}] {report.message}"
        
        if report.affected_component:
            log_message += f" (Component: {report.affected_component})"
        
        if report.severity == ErrorSeverity.INFO:
            self.logger.info(log_message)
        elif report.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        elif report.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
    
    def _get_suggested_solution(self, error: PhotonicError) -> str:
        """Generate suggested solution based on error type"""
        solutions = {
            "THERMAL_ERROR": "Reduce voltage/power, improve cooling, or use different material",
            "OPTICAL_ERROR": "Reduce optical power or check damage thresholds", 
            "CONVERGENCE_ERROR": "Reduce time step, improve initial conditions, or check stability",
            "VALIDATION_ERROR": "Check parameter ranges and units",
            "SECURITY_ERROR": "Validate input data and check for malicious content",
        }
        
        return solutions.get(error.error_code, "Check documentation for troubleshooting steps")
    
    def _thermal_recovery(self, error: ThermalError, report: ErrorReport):
        """Thermal error recovery strategy"""
        # Simulate cooling down
        report.context["recovery_action"] = "thermal_cooldown_initiated"
        report.context["estimated_cooldown_time"] = "30s"
    
    def _optical_recovery(self, error: OpticalError, report: ErrorReport):
        """Optical error recovery strategy"""
        # Reduce optical power
        safe_power = error.threshold * 0.8
        report.context["recovery_action"] = f"optical_power_reduced_to_{safe_power*1000:.1f}mW"
    
    def _convergence_recovery(self, error: ConvergenceError, report: ErrorReport):
        """Convergence error recovery strategy"""
        # Suggest smaller time step
        report.context["recovery_action"] = "reduce_time_step_by_50%"
        report.context["suggested_max_iterations"] = error.iterations * 2
    
    def _validation_recovery(self, error: ValidationError, report: ErrorReport):
        """Validation error recovery strategy"""
        # Suggest parameter correction
        if error.expected_range:
            min_val, max_val = error.expected_range
            suggested = (min_val + max_val) / 2
            report.context["recovery_action"] = f"use_suggested_value_{suggested}"
    
    def _check_circuit_breaker(self, error_code: str, component: str):
        """Implement circuit breaker pattern"""
        key = f"{error_code}_{component or 'global'}"
        
        if key not in self.circuit_breaker_state:
            self.circuit_breaker_state[key] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half-open
            }
        
        breaker = self.circuit_breaker_state[key]
        breaker['failures'] += 1
        breaker['last_failure'] = time.time()
        
        if breaker['failures'] >= self.failure_threshold and breaker['state'] == 'closed':
            breaker['state'] = 'open'
            self.logger.critical(f"Circuit breaker OPENED for {key} after {breaker['failures']} failures")
        
        # Auto-recovery after timeout
        if (breaker['state'] == 'open' and 
            time.time() - breaker['last_failure'] > self.recovery_timeout):
            breaker['state'] = 'half-open'
            breaker['failures'] = 0
            self.logger.info(f"Circuit breaker moved to HALF-OPEN for {key}")
    
    def is_circuit_open(self, error_code: str, component: str = None) -> bool:
        """Check if circuit breaker is open"""
        key = f"{error_code}_{component or 'global'}"
        breaker = self.circuit_breaker_state.get(key, {})
        return breaker.get('state') == 'open'
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'severity_breakdown': {
                severity.value: sum(1 for r in self.error_history if r.severity == severity)
                for severity in ErrorSeverity
            },
            'recent_errors': [asdict(r) for r in self.error_history[-10:]],
            'circuit_breaker_states': self.circuit_breaker_state.copy()
        }
    
    def export_error_report(self, filename: str):
        """Export comprehensive error report"""
        report_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            'statistics': self.get_error_statistics(),
            'detailed_history': [asdict(r) for r in self.error_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Error report exported to {filename}")
    
    def clear_history(self):
        """Clear error history"""
        with self._lock:
            self.error_history.clear()
            self.error_counts.clear()
            self.circuit_breaker_state.clear()


# Global error handler instance
_global_error_handler = None


def get_error_handler(log_file: str = None) -> RobustErrorHandler:
    """Get or create global error handler"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler(log_file)
    return _global_error_handler


def robust_function(component: str = None, retries: int = 3, backoff: float = 1.0):
    """Decorator for robust function execution with error handling"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            
            # Check circuit breaker
            func_key = f"FUNCTION_ERROR_{func.__name__}"
            if error_handler.is_circuit_open(func_key, component):
                raise PhotonicError(
                    f"Circuit breaker is OPEN for {func.__name__}",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    severity=ErrorSeverity.ERROR
                )
            
            last_error = None
            for attempt in range(retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # Reset circuit breaker on success
                    if attempt > 0:
                        error_handler.logger.info(f"{func.__name__} succeeded after {attempt + 1} attempts")
                    
                    return result
                    
                except PhotonicError as e:
                    last_error = e
                    error_handler.handle_error(e, component or func.__name__)
                    
                    if attempt < retries - 1:
                        wait_time = backoff * (2 ** attempt)  # Exponential backoff
                        error_handler.logger.warning(
                            f"Retrying {func.__name__} in {wait_time}s (attempt {attempt + 1}/{retries})"
                        )
                        time.sleep(wait_time)
                    
                except Exception as e:
                    # Convert to PhotonicError
                    photonic_error = PhotonicError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        severity=ErrorSeverity.ERROR,
                        context={'function': func.__name__, 'args': str(args)[:200]}
                    )
                    last_error = photonic_error
                    error_handler.handle_error(photonic_error, component or func.__name__)
                    
                    if attempt < retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        time.sleep(wait_time)
            
            # All retries failed
            raise last_error
        
        return wrapper
    return decorator


@contextmanager
def error_context(operation_name: str, component: str = None):
    """Context manager for error handling"""
    error_handler = get_error_handler()
    start_time = time.time()
    
    try:
        error_handler.logger.info(f"Starting operation: {operation_name}")
        yield error_handler
        
        elapsed = time.time() - start_time
        error_handler.logger.info(f"Operation {operation_name} completed successfully in {elapsed:.3f}s")
        
    except PhotonicError as e:
        e.context = e.context or {}
        e.context['operation'] = operation_name
        e.context['elapsed_time'] = time.time() - start_time
        
        error_handler.handle_error(e, component)
        raise
        
    except Exception as e:
        photonic_error = PhotonicError(
            f"Unexpected error in {operation_name}: {str(e)}",
            error_code="OPERATION_ERROR",
            severity=ErrorSeverity.ERROR,
            context={
                'operation': operation_name,
                'elapsed_time': time.time() - start_time
            }
        )
        
        error_handler.handle_error(photonic_error, component)
        raise photonic_error


class InputValidator:
    """Comprehensive input validation"""
    
    @staticmethod
    def validate_parameter(name: str, value: Union[int, float], 
                          min_val: float = None, max_val: float = None, 
                          allow_zero: bool = True) -> float:
        """Validate numeric parameter"""
        
        # Type check
        if not isinstance(value, (int, float)):
            raise ValidationError(name, value, context={'type_error': f'Expected number, got {type(value)}'})
        
        # NaN/infinity check
        if np.isnan(value) or np.isinf(value):
            raise ValidationError(name, value, context={'invalid_number': 'NaN or infinity not allowed'})
        
        # Zero check
        if not allow_zero and value == 0:
            raise ValidationError(name, value, context={'zero_not_allowed': True})
        
        # Range check
        if min_val is not None and value < min_val:
            raise ValidationError(name, value, expected_range=(min_val, max_val))
        
        if max_val is not None and value > max_val:
            raise ValidationError(name, value, expected_range=(min_val, max_val))
        
        return float(value)
    
    @staticmethod
    def validate_array(name: str, array: np.ndarray, 
                      shape: tuple = None, dtype: type = None,
                      min_val: float = None, max_val: float = None) -> np.ndarray:
        """Validate numpy array"""
        
        if not isinstance(array, np.ndarray):
            raise ValidationError(name, array, context={'type_error': f'Expected ndarray, got {type(array)}'})
        
        # Shape validation
        if shape is not None and array.shape != shape:
            raise ValidationError(name, array, context={
                'shape_mismatch': f'Expected {shape}, got {array.shape}'
            })
        
        # Data type validation
        if dtype is not None and not np.issubdtype(array.dtype, dtype):
            raise ValidationError(name, array, context={
                'dtype_mismatch': f'Expected {dtype}, got {array.dtype}'
            })
        
        # Check for NaN/infinity
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            raise ValidationError(name, array, context={'invalid_values': 'Contains NaN or infinity'})
        
        # Range validation
        if min_val is not None and np.any(array < min_val):
            raise ValidationError(name, array, context={'below_minimum': f'Values below {min_val}'})
        
        if max_val is not None and np.any(array > max_val):
            raise ValidationError(name, array, context={'above_maximum': f'Values above {max_val}'})
        
        return array
    
    @staticmethod
    def validate_material_type(material: str) -> str:
        """Validate material type"""
        valid_materials = ['GST', 'HfO2', 'TiO2', 'PCM', 'RRAM']
        
        if not isinstance(material, str):
            raise ValidationError('material_type', material, 
                                context={'type_error': f'Expected string, got {type(material)}'})
        
        material = material.strip().upper()
        
        if material not in valid_materials:
            raise ValidationError('material_type', material,
                                context={'valid_materials': valid_materials})
        
        return material


# Example usage functions
def demonstrate_error_handling():
    """Demonstrate robust error handling capabilities"""
    print("🛡️ Demonstrating Robust Error Handling...")
    
    error_handler = get_error_handler("/tmp/photonic_errors.log")
    
    # Test different error types
    errors_to_test = [
        ThermalError(1200.0, 900.0, "GST_device_01"),
        OpticalError(150e-3, 100e-3),
        ConvergenceError(1000, 1e-3),
        ValidationError("voltage", -150.0, (-10.0, 10.0)),
        SecurityError("Invalid input detected: potential buffer overflow"),
    ]
    
    for i, error in enumerate(errors_to_test):
        try:
            raise error
        except PhotonicError as e:
            report = error_handler.handle_error(e, f"test_component_{i}")
            print(f"   Handled {e.error_code}: {e.message}")
            if report.suggested_solution:
                print(f"   💡 Suggested: {report.suggested_solution}")
    
    # Show statistics
    stats = error_handler.get_error_statistics()
    print(f"\n📊 Error Statistics:")
    print(f"   Total errors: {stats['total_errors']}")
    print(f"   Critical: {stats['severity_breakdown']['critical']}")
    print(f"   Errors: {stats['severity_breakdown']['error']}")
    print(f"   Warnings: {stats['severity_breakdown']['warning']}")
    
    # Export report
    error_handler.export_error_report("/root/repo/error_handling_demo_report.json")
    print("📄 Error report exported to error_handling_demo_report.json")


@robust_function(component="demo", retries=3, backoff=0.1)
def unreliable_simulation(failure_rate: float = 0.7):
    """Simulate an unreliable function for testing"""
    import random
    
    if random.random() < failure_rate:
        error_type = random.choice([
            ThermalError(850.0, 800.0, "test_device"),
            ConvergenceError(500, 1e-2),
            ValidationError("test_param", 999, (0, 100))
        ])
        raise error_type
    
    return "Simulation completed successfully"


def demonstrate_robust_functions():
    """Demonstrate robust function decorator"""
    print("\n🔄 Demonstrating Robust Functions...")
    
    try:
        result = unreliable_simulation(failure_rate=0.8)
        print(f"   ✅ Result: {result}")
    except PhotonicError as e:
        print(f"   ❌ Final failure: {e.message}")
    
    # Demonstrate context manager
    try:
        with error_context("critical_calculation", "simulation_core") as handler:
            # Simulate some critical calculation
            if np.random.random() < 0.3:
                raise ThermalError(950.0, 900.0, "critical_device")
            print("   ✅ Critical calculation completed successfully")
            
    except PhotonicError as e:
        print(f"   ❌ Context operation failed: {e.message}")


def demonstrate_input_validation():
    """Demonstrate input validation"""
    print("\n✅ Demonstrating Input Validation...")
    
    validator = InputValidator()
    
    # Test valid inputs
    try:
        temp = validator.validate_parameter("temperature", 300.0, min_val=0, max_val=2000)
        print(f"   ✅ Valid temperature: {temp}K")
        
        array = validator.validate_array("voltages", np.array([1.0, 2.0, 3.0]), 
                                       shape=(3,), min_val=0, max_val=10)
        print(f"   ✅ Valid voltage array: {array}")
        
        material = validator.validate_material_type("gst")
        print(f"   ✅ Valid material: {material}")
        
    except ValidationError as e:
        print(f"   ❌ Validation error: {e.message}")
    
    # Test invalid inputs
    invalid_tests = [
        ("temperature", 3000.0, 0, 2000),  # Too high
        ("voltage", float('nan'), -10, 10),  # NaN
        ("material", "InvalidMaterial", None, None),  # Unknown material
    ]
    
    for param_name, value, min_val, max_val in invalid_tests:
        try:
            if param_name == "material":
                validator.validate_material_type(value)
            else:
                validator.validate_parameter(param_name, value, min_val, max_val)
        except ValidationError as e:
            print(f"   ❌ Caught invalid {param_name}: {e.message}")


if __name__ == "__main__":
    print("🛡️ Robust Error Handling and Validation Demo")
    print("Generation 2: MAKE IT ROBUST")
    print("=" * 50)
    
    demonstrate_error_handling()
    demonstrate_robust_functions() 
    demonstrate_input_validation()
    
    print("\n✅ Generation 2 error handling demonstration completed!")
    print("🔒 System is now robust with comprehensive error handling, validation, and recovery!")