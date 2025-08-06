#!/usr/bin/env python3
"""
Robust Photonic Simulation with Generation 2 Features

This example demonstrates the enhanced error handling, validation, 
logging, and monitoring capabilities of the photonic simulation system.
"""

import numpy as np
import time
import sys
import os
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from python.photon_memristor_sim.quantum_planning import (
        QuantumTaskPlanner,
        PhotonicTaskPlannerFactory,
    )
except ImportError:
    print("Warning: Running in standalone mode - some features may not be available")

class PhotonicSimulationMonitor:
    """Enhanced photonic simulation with monitoring and error handling."""
    
    def __init__(self):
        self.metrics = {}
        self.errors = []
        self.warnings = []
        self.performance_data = []
        self.validation_results = []
        
    def validate_input_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate simulation input parameters."""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Wavelength validation
        if 'wavelength' in params:
            wavelength = params['wavelength']
            if wavelength < 200e-9 or wavelength > 20e-6:
                validation_report['errors'].append(
                    f"Wavelength {wavelength*1e9:.1f}nm outside valid range (200nm - 20μm)"
                )
                validation_report['valid'] = False
            elif wavelength < 400e-9 or wavelength > 2000e-9:
                validation_report['warnings'].append(
                    f"Wavelength {wavelength*1e9:.1f}nm outside typical photonic range"
                )
        
        # Power validation
        if 'power' in params:
            power = params['power']
            if power < 0:
                validation_report['errors'].append("Optical power cannot be negative")
                validation_report['valid'] = False
            elif power > 1.0:  # 1W
                validation_report['warnings'].append(
                    f"Power {power:.3f}W is very high - check for damage"
                )
        
        # Device count validation
        if 'num_devices' in params:
            num_devices = params['num_devices']
            if num_devices <= 0:
                validation_report['errors'].append("Number of devices must be positive")
                validation_report['valid'] = False
            elif num_devices > 1000:
                validation_report['warnings'].append(
                    f"Large number of devices ({num_devices}) may impact performance"
                )
        
        # Temperature validation
        if 'temperature' in params:
            temperature = params['temperature']
            if temperature < 0:
                validation_report['errors'].append("Temperature cannot be negative (Kelvin)")
                validation_report['valid'] = False
            elif temperature > 400:  # 400K = 127°C
                validation_report['warnings'].append(
                    f"Temperature {temperature:.1f}K exceeds typical operating range"
                )
        
        self.validation_results.append(validation_report)
        return validation_report
    
    def log_operation(self, level: str, operation: str, details: Dict[str, Any] = None):
        """Log operation with timestamp and metadata."""
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'operation': operation,
            'details': details or {}
        }
        
        if level == 'ERROR':
            self.errors.append(log_entry)
        elif level == 'WARNING':
            self.warnings.append(log_entry)
        
        # Print log (in real implementation, would use proper logging)
        timestamp_str = time.strftime('%H:%M:%S', time.localtime(log_entry['timestamp']))
        print(f"[{timestamp_str}] {level:7} | {operation}")
        if details:
            for key, value in details.items():
                print(f"                     {key}: {value}")
    
    def record_performance(self, operation: str, duration_ms: float, **kwargs):
        """Record performance metrics for operations."""
        perf_entry = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': time.time(),
            **kwargs
        }
        self.performance_data.append(perf_entry)
        
        # Record in metrics
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_duration': 0,
                'min_duration': float('inf'),
                'max_duration': 0,
                'durations': []
            }
        
        metric = self.metrics[operation]
        metric['count'] += 1
        metric['total_duration'] += duration_ms
        metric['min_duration'] = min(metric['min_duration'], duration_ms)
        metric['max_duration'] = max(metric['max_duration'], duration_ms)
        metric['durations'].append(duration_ms)
        
        # Log performance if unusually slow
        if duration_ms > 1000:  # > 1 second
            self.log_operation('WARNING', f'{operation} performance', {
                'duration_ms': duration_ms,
                'threshold_ms': 1000
            })
    
    def execute_with_monitoring(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute operation with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            self.log_operation('INFO', f'Starting {operation_name}')
            
            # Execute operation
            result = operation_func(*args, **kwargs)
            
            # Record successful completion
            duration_ms = (time.time() - start_time) * 1000
            self.record_performance(operation_name, duration_ms, status='success')
            
            self.log_operation('INFO', f'Completed {operation_name}', {
                'duration_ms': f'{duration_ms:.2f}',
                'status': 'success'
            })
            
            return result
            
        except Exception as e:
            # Record error
            duration_ms = (time.time() - start_time) * 1000
            self.record_performance(operation_name, duration_ms, status='error')
            
            self.log_operation('ERROR', f'Failed {operation_name}', {
                'duration_ms': f'{duration_ms:.2f}',
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            # Re-raise exception
            raise
    
    def execute_with_retry(self, operation_name: str, operation_func, 
                          max_retries: int = 3, delay_seconds: float = 0.1, 
                          *args, **kwargs):
        """Execute operation with retry logic and exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.log_operation('INFO', f'Retrying {operation_name}', {
                        'attempt': attempt + 1,
                        'max_attempts': max_retries + 1
                    })
                
                return self.execute_with_monitoring(
                    f'{operation_name}_attempt_{attempt + 1}', 
                    operation_func, *args, **kwargs
                )
                
            except Exception as e:
                if attempt < max_retries:
                    sleep_time = delay_seconds * (2 ** attempt)  # Exponential backoff
                    self.log_operation('WARNING', f'{operation_name} attempt failed', {
                        'attempt': attempt + 1,
                        'error': str(e),
                        'retry_delay_s': sleep_time
                    })
                    time.sleep(sleep_time)
                else:
                    self.log_operation('ERROR', f'{operation_name} all attempts failed', {
                        'total_attempts': max_retries + 1,
                        'final_error': str(e)
                    })
                    raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        now = time.time()
        recent_window = 300  # 5 minutes
        
        recent_errors = [e for e in self.errors if now - e['timestamp'] < recent_window]
        recent_warnings = [w for w in self.warnings if now - w['timestamp'] < recent_window]
        
        # Calculate performance metrics
        perf_summary = {}
        for op_name, metrics in self.metrics.items():
            if metrics['count'] > 0:
                perf_summary[op_name] = {
                    'count': metrics['count'],
                    'avg_duration_ms': metrics['total_duration'] / metrics['count'],
                    'min_duration_ms': metrics['min_duration'],
                    'max_duration_ms': metrics['max_duration']
                }
        
        # Determine health status
        if len(recent_errors) > 10:
            health = 'CRITICAL'
        elif len(recent_errors) > 5 or len(recent_warnings) > 20:
            health = 'WARNING'
        elif len(recent_errors) > 0 or len(recent_warnings) > 5:
            health = 'DEGRADED'
        else:
            health = 'HEALTHY'
        
        return {
            'status': health,
            'recent_errors': len(recent_errors),
            'recent_warnings': len(recent_warnings),
            'total_operations': sum(m['count'] for m in self.metrics.values()),
            'performance_summary': perf_summary,
            'validation_failures': sum(1 for v in self.validation_results if not v['valid'])
        }
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report."""
        health = self.get_health_status()
        
        report = f"""
Photonic Simulation Monitoring Report
=====================================

Health Status: {health['status']}
Recent Errors: {health['recent_errors']}
Recent Warnings: {health['recent_warnings']}
Total Operations: {health['total_operations']}
Validation Failures: {health['validation_failures']}

Performance Summary:
"""
        
        for op_name, metrics in health['performance_summary'].items():
            report += f"""
  {op_name}:
    Count: {metrics['count']}
    Avg Duration: {metrics['avg_duration_ms']:.2f}ms
    Min Duration: {metrics['min_duration_ms']:.2f}ms
    Max Duration: {metrics['max_duration_ms']:.2f}ms
"""
        
        if self.errors:
            report += f"\n\nRecent Errors ({len(self.errors)} total):\n"
            for error in self.errors[-5:]:  # Show last 5 errors
                timestamp = time.strftime('%H:%M:%S', time.localtime(error['timestamp']))
                report += f"  [{timestamp}] {error['operation']}: {error['details'].get('error', 'Unknown')}\n"
        
        return report

def simulate_photonic_neural_network_with_errors(monitor: PhotonicSimulationMonitor, 
                                                num_devices: int, 
                                                add_artificial_errors: bool = True):
    """Simulate a photonic neural network with potential errors for testing."""
    
    # Validate parameters
    params = {
        'num_devices': num_devices,
        'wavelength': 1550e-9,
        'power': 1e-3,
        'temperature': 300.0
    }
    
    validation = monitor.validate_input_parameters(params)
    if not validation['valid']:
        raise ValueError(f"Validation failed: {validation['errors']}")
    
    if validation['warnings']:
        for warning in validation['warnings']:
            monitor.log_operation('WARNING', 'Parameter validation', {'warning': warning})
    
    # Create quantum task planner (might fail if not available)
    def create_planner():
        if add_artificial_errors and np.random.random() < 0.2:  # 20% chance of error
            raise RuntimeError("Simulated planner creation failure")
        
        try:
            planner = PhotonicTaskPlannerFactory.create_photonic_planner(num_devices)
            monitor.log_operation('INFO', 'Quantum planner created', {
                'num_devices': num_devices,
                'planner_type': 'photonic_optimized'
            })
            return planner
        except Exception as e:
            monitor.log_operation('WARNING', 'Quantum planner unavailable', {
                'fallback': 'classical_simulation',
                'error': str(e)
            })
            return None  # Fallback to classical simulation
    
    # Create planner with retry
    planner = monitor.execute_with_retry('create_quantum_planner', create_planner)
    
    # Simulate device initialization
    def initialize_devices():
        if add_artificial_errors and np.random.random() < 0.1:  # 10% chance of error
            raise RuntimeError("Device initialization failed")
        
        # Simulate some initialization time
        time.sleep(0.05 * num_devices / 100)  # Scale with device count
        
        return {f'device_{i}': {'status': 'initialized'} for i in range(num_devices)}
    
    devices = monitor.execute_with_retry('initialize_devices', initialize_devices)
    
    # Simulate optical field propagation
    def propagate_optical_field():
        if add_artificial_errors and np.random.random() < 0.05:  # 5% chance of error
            raise RuntimeError("Optical field propagation failed")
        
        # Simulate computation time based on device count
        computation_time = 0.01 * num_devices / 10
        time.sleep(computation_time)
        
        # Simulate some results
        field_intensity = np.random.random(num_devices) * params['power']
        return {
            'field_intensity': field_intensity,
            'total_power': np.sum(field_intensity),
            'max_intensity': np.max(field_intensity)
        }
    
    field_result = monitor.execute_with_monitoring('optical_propagation', propagate_optical_field)
    
    # Simulate quantum optimization (if planner available)
    optimization_result = None
    if planner:
        def run_quantum_optimization():
            if add_artificial_errors and np.random.random() < 0.15:  # 15% chance of error
                raise RuntimeError("Quantum optimization convergence failed")
            
            # Simulate annealing time
            time.sleep(0.1 + 0.01 * num_devices / 50)
            
            # Simulate measurement
            assignment = planner.measure()
            return {
                'task_assignment': assignment,
                'quantum_fidelity': planner.fidelity(),
                'optimization_successful': True
            }
        
        optimization_result = monitor.execute_with_retry('quantum_optimization', run_quantum_optimization)
    
    # Return simulation results
    results = {
        'devices': devices,
        'optical_field': field_result,
        'quantum_optimization': optimization_result,
        'parameters': params,
        'validation': validation
    }
    
    monitor.log_operation('INFO', 'Simulation completed successfully', {
        'num_devices': num_devices,
        'total_power': field_result['total_power'],
        'quantum_used': optimization_result is not None
    })
    
    return results

def run_monitoring_stress_test():
    """Run comprehensive monitoring stress test."""
    print("🧪 Starting Photonic Simulation Monitoring Stress Test")
    print("=" * 60)
    
    monitor = PhotonicSimulationMonitor()
    
    # Test various scenarios
    test_scenarios = [
        {'num_devices': 8, 'errors': False, 'description': 'Small system, no errors'},
        {'num_devices': 16, 'errors': True, 'description': 'Medium system, with errors'},
        {'num_devices': 32, 'errors': True, 'description': 'Large system, with errors'},
        {'num_devices': 4, 'errors': False, 'description': 'Small system, optimistic'},
        {'num_devices': 64, 'errors': True, 'description': 'Very large system, stress test'},
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n📊 Scenario {i+1}: {scenario['description']}")
        print("-" * 40)
        
        try:
            results = simulate_photonic_neural_network_with_errors(
                monitor, 
                scenario['num_devices'],
                add_artificial_errors=scenario['errors']
            )
            print(f"✅ Scenario completed - {scenario['num_devices']} devices simulated")
            
        except Exception as e:
            print(f"❌ Scenario failed: {e}")
    
    # Generate and display comprehensive report
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE MONITORING REPORT")
    print("=" * 60)
    
    report = monitor.generate_monitoring_report()
    print(report)
    
    # Display health status
    health = monitor.get_health_status()
    print(f"\n🏥 FINAL HEALTH STATUS: {health['status']}")
    
    if health['status'] == 'HEALTHY':
        print("✅ All systems operating normally")
    elif health['status'] == 'DEGRADED':
        print("⚠️  System performance is degraded but functional")
    elif health['status'] == 'WARNING':
        print("🔶 System has warnings that require attention")
    else:
        print("🚨 System is in critical state - immediate attention required")
    
    return monitor

def demonstrate_circuit_breaker_pattern():
    """Demonstrate circuit breaker pattern for fault tolerance."""
    print("\n⚡ Circuit Breaker Pattern Demonstration")
    print("=" * 50)
    
    monitor = PhotonicSimulationMonitor()
    
    # Simulate a flaky operation
    failure_count = 0
    
    def flaky_operation():
        nonlocal failure_count
        failure_count += 1
        
        if failure_count <= 5:  # First 5 calls fail
            raise RuntimeError(f"Simulated failure #{failure_count}")
        else:
            return "Operation successful"
    
    # Circuit breaker state
    consecutive_failures = 0
    circuit_open = False
    circuit_open_time = None
    
    # Simulate circuit breaker logic
    for attempt in range(10):
        try:
            if circuit_open:
                # Check if we should try again (after timeout)
                if time.time() - circuit_open_time > 1.0:  # 1 second timeout
                    circuit_open = False
                    consecutive_failures = 0
                    monitor.log_operation('INFO', 'Circuit breaker half-open', {
                        'attempt': attempt + 1
                    })
                else:
                    monitor.log_operation('WARNING', 'Circuit breaker blocking call', {
                        'attempt': attempt + 1,
                        'time_remaining': 1.0 - (time.time() - circuit_open_time)
                    })
                    continue
            
            # Attempt operation
            result = flaky_operation()
            consecutive_failures = 0
            circuit_open = False
            
            monitor.log_operation('INFO', 'Operation succeeded', {
                'attempt': attempt + 1,
                'result': result
            })
            
        except Exception as e:
            consecutive_failures += 1
            
            if consecutive_failures >= 3:  # Open circuit after 3 failures
                circuit_open = True
                circuit_open_time = time.time()
                monitor.log_operation('ERROR', 'Circuit breaker opened', {
                    'attempt': attempt + 1,
                    'consecutive_failures': consecutive_failures
                })
            else:
                monitor.log_operation('ERROR', 'Operation failed', {
                    'attempt': attempt + 1,
                    'error': str(e),
                    'consecutive_failures': consecutive_failures
                })
        
        time.sleep(0.2)  # Small delay between attempts
    
    print("Circuit breaker demonstration completed")

def main():
    """Main demonstration function."""
    print("🛡️  GENERATION 2: ROBUST PHOTONIC SIMULATION")
    print("Enhanced Error Handling, Validation, Logging & Monitoring")
    print("=" * 70)
    
    try:
        # Run comprehensive monitoring stress test
        monitor = run_monitoring_stress_test()
        
        # Demonstrate circuit breaker pattern
        demonstrate_circuit_breaker_pattern()
        
        print("\n🎯 Generation 2 Features Demonstrated:")
        print("✅ Input parameter validation with custom constraints")
        print("✅ Comprehensive error handling with context")
        print("✅ Structured logging with metadata and timestamps")
        print("✅ Real-time performance monitoring and metrics")
        print("✅ Retry logic with exponential backoff")
        print("✅ Health status monitoring and reporting")
        print("✅ Circuit breaker pattern for fault tolerance")
        print("✅ Graceful degradation and fallback mechanisms")
        
        print(f"\n📊 Final Statistics:")
        health = monitor.get_health_status()
        print(f"Total Operations: {health['total_operations']}")
        print(f"System Status: {health['status']}")
        print(f"Error Rate: {health['recent_errors']}/{health['total_operations']} recent")
        
        print("\n🎉 Generation 2 demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)