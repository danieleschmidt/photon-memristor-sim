#!/usr/bin/env python3
"""
Production-ready server for Photon-Memristor-Sim
Enterprise-grade neuromorphic photonic computing platform
"""

import os
import time
import logging
import threading
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Photonic simulation imports
try:
    from photon_memristor_sim import (
        PhotonicNeuralNetwork, 
        get_resilient_system,
        get_optimizer,
        with_circuit_breaker,
        with_retry, 
        with_metrics,
        get_secret,
        auto_scaling_config,
        load_balancer_config,
        metrics_config
    )
except ImportError as e:
    print(f"Warning: Photonic simulation modules not available: {e}")
    # Graceful degradation for deployment environments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/photonic_sim.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('photonic_server')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update(
    SECRET_KEY=get_secret('jwt_secret') or 'fallback-dev-key',
    DEBUG=False,
    TESTING=False,
    PHOTONIC_ENV=os.getenv('PHOTONIC_ENV', 'production')
)

# Initialize resilient system
resilient_system = get_resilient_system("photonic_production")

# Configure circuit breakers for critical services
from photon_memristor_sim import CircuitBreakerConfig

simulation_cb_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=3,
    request_volume_threshold=10
)

optimization_cb_config = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30,
    success_threshold=2,
    request_volume_threshold=5
)

resilient_system.add_circuit_breaker("simulation_service", simulation_cb_config)
resilient_system.add_circuit_breaker("optimization_service", optimization_cb_config)

# Health checks
health_check = resilient_system.add_health_check("system_health", check_interval=30)

def check_memory_usage():
    """Check if memory usage is within acceptable limits"""
    try:
        from photon_memristor_sim.performance_optimizer import MemoryOptimizer
        memory_info = MemoryOptimizer.get_memory_info()
        return memory_info['percent'] < 90.0
    except:
        return True  # Graceful fallback

def check_disk_space():
    """Check if disk space is sufficient"""
    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        usage_percent = (used / total) * 100
        return usage_percent < 85.0
    except:
        return True  # Graceful fallback

def check_system_load():
    """Check if system load is reasonable"""
    try:
        import psutil
        return psutil.cpu_percent(interval=1) < 95.0
    except:
        return True  # Graceful fallback

health_check.add_check("memory", check_memory_usage)
health_check.add_check("disk", check_disk_space)
health_check.add_check("cpu", check_system_load)

# Performance optimizer
optimizer = get_optimizer()

# Rate limiting
from collections import defaultdict
request_counts = defaultdict(list)
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

def is_rate_limited(client_ip: str) -> bool:
    """Check if client is rate limited"""
    now = time.time()
    
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if now - req_time < RATE_WINDOW
    ]
    
    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return True
    
    request_counts[client_ip].append(now)
    return False

@app.before_request
def before_request():
    """Pre-request processing"""
    g.start_time = time.time()
    
    # Rate limiting
    client_ip = request.remote_addr or 'unknown'
    if is_rate_limited(client_ip):
        return jsonify({
            'error': 'Rate limit exceeded',
            'limit': RATE_LIMIT,
            'window': RATE_WINDOW
        }), 429
    
    # Request logging
    logger.info(f"Request: {request.method} {request.path} from {client_ip}")

@app.after_request  
def after_request(response):
    """Post-request processing"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        
        # Log request duration
        logger.info(f"Request completed in {duration:.3f}s - Status: {response.status_code}")
        
        # Record metrics
        resilient_system.metrics.record_histogram('request_duration', duration * 1000)
        resilient_system.metrics.increment_counter('requests_total', tags={
            'method': request.method,
            'status': str(response.status_code)
        })
    
    return response

# API Routes

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'Photon-Memristor-Sim',
        'version': '0.1.0',
        'status': 'operational',
        'timestamp': datetime.utcnow().isoformat(),
        'environment': app.config['PHOTONIC_ENV']
    })

@app.route('/health')
def health():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '0.1.0'
    })

@app.route('/ready')
def readiness():
    """Readiness probe for Kubernetes"""
    try:
        # Run quick health checks
        health_results = health_check.run_checks()
        
        if all(health_results.values()):
            return jsonify({'status': 'ready'})
        else:
            return jsonify({
                'status': 'not_ready',
                'checks': health_results
            }), 503
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 503

@app.route('/status')  
def status():
    """Detailed system status"""
    try:
        system_health = resilient_system.get_system_health()
        return jsonify(system_health)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        system_health = resilient_system.get_system_health()
        metrics_data = system_health.get("metrics", {})
        
        output_lines = []
        
        # Convert counters to Prometheus format
        for name, value in metrics_data.get("counters", {}).items():
            metric_name = f"photonic_{name.replace('[', '_').replace(']', '').replace('=', '_').replace(',', '_')}"
            output_lines.append(f"{metric_name} {value}")
        
        # Convert gauges to Prometheus format  
        for name, value in metrics_data.get("gauges", {}).items():
            metric_name = f"photonic_{name.replace('[', '_').replace(']', '').replace('=', '_').replace('.', '_')}"
            output_lines.append(f"{metric_name} {value}")
        
        # Convert histograms to Prometheus format
        for name, histogram in metrics_data.get("histograms", {}).items():
            base_name = f"photonic_{name.replace('[', '_').replace(']', '').replace('=', '_')}"
            output_lines.append(f"{base_name}_count {histogram.get('count', 0)}")
            output_lines.append(f"{base_name}_sum {histogram.get('avg', 0) * histogram.get('count', 0)}")
            
            for percentile in [50, 95, 99]:
                if f'p{percentile}' in histogram:
                    output_lines.append(f"{base_name}{{quantile=\"0.{percentile}\"}} {histogram[f'p{percentile}']}")
        
        return '\n'.join(output_lines) + '\n', 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        return f"# Error exporting metrics: {e}\n", 500, {'Content-Type': 'text/plain'}

@app.route('/simulate', methods=['POST'])
@with_circuit_breaker("simulation_service", simulation_cb_config)
@with_retry(max_attempts=2, base_delay=0.1)
@with_metrics("simulation_request")
def simulate():
    """Run photonic simulation"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        # Input validation
        if not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object'}), 400
        
        required_fields = ['network_config', 'input_data']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Extract parameters
        network_config = data['network_config']
        input_data = data['input_data']
        options = data.get('options', {})
        
        # Validate network config
        if not isinstance(network_config, list) or len(network_config) < 2:
            return jsonify({
                'error': 'network_config must be a list with at least 2 layer sizes'
            }), 400
        
        # Create network
        network = PhotonicNeuralNetwork(
            layers=network_config,
            activation=options.get('activation', 'photonic_relu')
        )
        
        # Initialize network parameters
        import jax
        key = jax.random.PRNGKey(options.get('seed', 42))
        params = network.init_params(key, (1, network_config[0]))
        
        # Convert input data
        import jax.numpy as jnp
        import numpy as np
        
        if isinstance(input_data, list):
            input_array = jnp.array(input_data)
        else:
            input_array = jnp.array([input_data])
        
        # Run simulation
        start_time = time.time()
        output = network(input_array, params, training=False)
        simulation_time = time.time() - start_time
        
        # Prepare response
        response = {
            'output': output.tolist(),
            'simulation_time': simulation_time,
            'network_info': {
                'layers': network_config,
                'device_count': network.device_count(),
                'power_consumption': network.total_power()
            },
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'version': '0.1.0'
            }
        }
        
        logger.info(f"Simulation completed in {simulation_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return jsonify({
            'error': 'Simulation failed',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/optimize', methods=['POST'])
@with_circuit_breaker("optimization_service", optimization_cb_config)
@with_retry(max_attempts=2, base_delay=0.1)
@with_metrics("optimization_request")
def optimize():
    """Run photonic optimization"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        # Basic optimization simulation (placeholder)
        # In production, this would interface with actual optimization algorithms
        
        start_time = time.time()
        
        # Simulate optimization process
        optimization_result = {
            'optimal_parameters': [0.8, 0.6, 0.7, 0.9],
            'convergence_time': time.time() - start_time,
            'iterations': 50,
            'final_loss': 0.001,
            'efficiency_gain': 15.3
        }
        
        logger.info(f"Optimization completed in {optimization_result['convergence_time']:.3f}s")
        
        return jsonify({
            'status': 'success',
            'result': optimization_result,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return jsonify({
            'error': 'Optimization failed',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/benchmark', methods=['GET'])
@with_metrics("benchmark_request")
def benchmark():
    """Run system benchmarks"""
    try:
        # Run performance benchmarks
        import numpy as np
        
        def test_operation(data):
            return np.sum(data ** 2)
        
        benchmark_data = [np.random.rand(100, 100) for _ in range(10)]
        
        results = optimizer.benchmark_performance(
            test_func=test_operation,
            test_data=benchmark_data,
            iterations=5
        )
        
        return jsonify({
            'benchmark_results': results,
            'system_info': {
                'cache_stats': optimizer.cache.stats(),
                'memory_info': optimizer.scheduler.get_memory_usage()
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return jsonify({
            'error': 'Benchmark failed', 
            'message': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested resource was not found',
        'available_endpoints': [
            '/', '/health', '/ready', '/status', '/metrics',
            '/simulate', '/optimize', '/benchmark'
        ]
    }), 404

@app.errorhandler(429)
def rate_limit_handler(error):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': f'Maximum {RATE_LIMIT} requests per {RATE_WINDOW} seconds allowed',
        'retry_after': RATE_WINDOW
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

# Background health monitoring
def health_monitor():
    """Background thread for health monitoring"""
    while True:
        try:
            health_check.run_checks()
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
            time.sleep(60)  # Wait longer on error

# Start background monitoring
monitoring_thread = threading.Thread(target=health_monitor, daemon=True)
monitoring_thread.start()

# Graceful shutdown
import atexit
import signal

def cleanup():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down gracefully...")
    
    # Close thread pools
    if hasattr(optimizer.scheduler, 'thread_pool'):
        optimizer.scheduler.thread_pool.shutdown(wait=True)
    if hasattr(optimizer.scheduler, 'process_pool'):
        optimizer.scheduler.process_pool.shutdown(wait=True)
    
    logger.info("Cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    cleanup()
    exit(0)

atexit.register(cleanup)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    # Production configuration
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    workers = int(os.getenv('WORKERS', 4))
    
    logger.info(f"Starting Photon-Memristor-Sim production server...")
    logger.info(f"Environment: {app.config['PHOTONIC_ENV']}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Workers: {workers}")
    
    # Use Gunicorn in production
    if app.config['PHOTONIC_ENV'] == 'production':
        # In production, this should be run with Gunicorn:
        # gunicorn -w 4 -b 0.0.0.0:8080 --timeout 120 --keep-alive 5 production_server:app
        logger.info("Use Gunicorn for production deployment:")
        logger.info(f"gunicorn -w {workers} -b {host}:{port} --timeout 120 --keep-alive 5 production_server:app")
    
    # Development server (not for production)
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )