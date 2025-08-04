"""
Training utilities and optimizers for photonic neural networks

Provides hardware-aware optimization algorithms and training utilities
specifically designed for photonic computing systems.
"""

import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax
from typing import Dict, Tuple, List, Optional, Callable, Any
import numpy as np
from functools import partial

from .neural_networks import PhotonicNeuralNetwork


class HardwareAwareOptimizer:
    """
    Optimizer that respects photonic hardware constraints.
    
    Includes power budget constraints, thermal management, 
    device switching limits, and fabrication tolerances.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 power_budget: float = 0.1,  # Watts
                 max_temperature: float = 400.0,  # Kelvin
                 device_constraints: Optional[Dict[str, Any]] = None):
        """
        Initialize hardware-aware optimizer.
        
        Args:
            learning_rate: Base learning rate
            power_budget: Maximum power consumption in Watts
            max_temperature: Maximum device temperature in Kelvin
            device_constraints: Additional device-specific constraints
        """
        self.learning_rate = learning_rate
        self.power_budget = power_budget
        self.max_temperature = max_temperature
        self.device_constraints = device_constraints or {}
        
        # Base optimizer (Adam)
        self.base_optimizer = optax.adam(learning_rate)
        
        # Constraint penalty weights
        self.power_penalty_weight = 0.01
        self.thermal_penalty_weight = 0.005
        self.switching_penalty_weight = 0.001
        
    def init(self, params: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """Initialize optimizer state."""
        base_state = self.base_optimizer.init(params)
        
        # Additional state for hardware constraints
        hardware_state = {
            'power_history': [],
            'temperature_history': [],
            'switching_count': jnp.zeros_like(_flatten_params(params)),
            'constraint_violations': 0
        }
        
        return {
            'base_state': base_state,
            'hardware_state': hardware_state
        }
    
    def update(self, gradients: Dict[str, jnp.ndarray], 
               opt_state: Dict[str, Any], 
               params: Optional[Dict[str, jnp.ndarray]] = None) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
        """Update parameters with hardware constraints."""
        
        # Apply hardware constraints to gradients
        constrained_grads = self._apply_hardware_constraints(gradients, params, opt_state)
        
        # Base optimizer update
        updates, new_base_state = self.base_optimizer.update(
            constrained_grads, opt_state['base_state'], params
        )
        
        # Update hardware state
        new_hardware_state = self._update_hardware_state(
            opt_state['hardware_state'], params, updates
        )
        
        new_opt_state = {
            'base_state': new_base_state,
            'hardware_state': new_hardware_state
        }
        
        return updates, new_opt_state
    
    def _apply_hardware_constraints(self, gradients: Dict[str, jnp.ndarray], 
                                   params: Optional[Dict[str, jnp.ndarray]], 
                                   opt_state: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Apply hardware constraints to gradients."""
        constrained_grads = {}
        
        for layer_name, layer_grads in gradients.items():
            constrained_layer_grads = {}
            
            for param_name, grad in layer_grads.items():
                constrained_grad = grad
                
                if param_name == "device_states" and params is not None:
                    # Device state constraints [0, 1]
                    current_states = params[layer_name][param_name]
                    
                    # Zero gradients at boundaries
                    at_lower = current_states <= 0.0
                    at_upper = current_states >= 1.0
                    
                    constrained_grad = jnp.where(at_lower & (grad < 0), 0, grad)
                    constrained_grad = jnp.where(at_upper & (grad > 0), 0, constrained_grad)
                    
                    # Limit switching rate
                    max_change = 0.1  # Maximum state change per update
                    constrained_grad = jnp.clip(constrained_grad, -max_change, max_change)
                
                elif param_name == "weights":
                    # Power constraint on weights
                    weight_norm = jnp.linalg.norm(grad)
                    if weight_norm > self.power_budget:
                        constrained_grad = grad * self.power_budget / weight_norm
                
                elif param_name == "wavelength":
                    # Wavelength stability constraint
                    max_wavelength_change = 1e-12  # 1pm per update
                    constrained_grad = jnp.clip(grad, -max_wavelength_change, max_wavelength_change)
                
                constrained_layer_grads[param_name] = constrained_grad
            
            constrained_grads[layer_name] = constrained_layer_grads
        
        return constrained_grads
    
    def _update_hardware_state(self, hardware_state: Dict[str, Any], 
                              params: Optional[Dict[str, jnp.ndarray]], 
                              updates: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """Update hardware monitoring state."""
        new_state = hardware_state.copy()
        
        if params is not None:
            # Calculate power consumption
            total_power = self._calculate_power_consumption(params)
            new_state['power_history'].append(float(total_power))
            
            # Calculate temperature (simplified model)
            avg_temp = 300.0 + total_power * 100  # K, simplified thermal model
            new_state['temperature_history'].append(float(avg_temp))
            
            # Count switching events
            flat_updates = _flatten_params(updates)
            switching_mask = jnp.abs(flat_updates) > 0.01
            new_state['switching_count'] += switching_mask.astype(jnp.float32)
            
            # Check constraint violations
            violations = 0
            if total_power > self.power_budget:
                violations += 1
            if avg_temp > self.max_temperature:
                violations += 1
            
            new_state['constraint_violations'] = violations
        
        return new_state
    
    def _calculate_power_consumption(self, params: Dict[str, jnp.ndarray]) -> float:
        """Estimate total power consumption."""
        total_power = 0.0
        
        for layer_params in params.values():
            if "device_states" in layer_params:
                # Power proportional to active devices
                active_devices = jnp.sum(layer_params["device_states"] > 0.1)
                total_power += active_devices * 1e-3  # 1mW per active device
            
            if "wavelength" in layer_params:
                # Laser power
                total_power += 10e-3  # 10mW per laser
        
        return float(total_power)
    
    def get_constraints_report(self, opt_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get hardware constraints status report."""
        hw_state = opt_state['hardware_state']
        
        return {
            'current_power': hw_state['power_history'][-1] if hw_state['power_history'] else 0,
            'power_budget': self.power_budget,
            'power_utilization': (hw_state['power_history'][-1] / self.power_budget) if hw_state['power_history'] else 0,
            'current_temperature': hw_state['temperature_history'][-1] if hw_state['temperature_history'] else 300,
            'max_temperature': self.max_temperature,
            'total_switching_events': float(jnp.sum(hw_state['switching_count'])),
            'constraint_violations': hw_state['constraint_violations']
        }


class CoDesignOptimizer:
    """
    Co-design optimizer for joint device-algorithm optimization.
    
    Simultaneously optimizes neural network weights and photonic device parameters
    for optimal performance under fabrication and operational constraints.
    """
    
    def __init__(self, 
                 network: PhotonicNeuralNetwork,
                 device_param_ranges: Dict[str, Tuple[float, float]],
                 objectives: Dict[str, float] = None):
        """
        Initialize co-design optimizer.
        
        Args:
            network: Photonic neural network to optimize
            device_param_ranges: Bounds for device parameters
            objectives: Multi-objective weights (accuracy, power, area, latency)
        """
        self.network = network
        self.device_param_ranges = device_param_ranges
        self.objectives = objectives or {
            'accuracy': 1.0,
            'power': 0.1, 
            'area': 0.01,
            'latency': 0.001
        }
        
        # Separate optimizers for network and device parameters
        self.network_optimizer = optax.adam(0.001)
        self.device_optimizer = optax.adam(0.0001)  # Slower for devices
        
    def codesign_loss(self, network_params: Dict[str, jnp.ndarray], 
                     device_params: Dict[str, jnp.ndarray],
                     inputs: jnp.ndarray, 
                     targets: jnp.ndarray) -> jnp.ndarray:
        """Multi-objective co-design loss function."""
        
        # Update network with device parameters
        combined_params = self._merge_params(network_params, device_params)
        
        # Compute network outputs
        outputs = self.network(inputs, combined_params, training=True)
        
        # Primary loss (accuracy)
        accuracy_loss = jnp.mean((outputs - targets) ** 2)
        
        # Hardware metrics
        power_consumption = self._estimate_power(device_params)
        chip_area = self._estimate_area(device_params)
        latency = self._estimate_latency(device_params)
        
        # Multi-objective loss
        total_loss = (
            self.objectives['accuracy'] * accuracy_loss +
            self.objectives['power'] * power_consumption / 0.1 +  # Normalize to 100mW
            self.objectives['area'] * chip_area / 1e-6 +  # Normalize to 1mm²
            self.objectives['latency'] * latency / 1e-9  # Normalize to 1ns
        )
        
        return total_loss
    
    def optimize(self, 
                train_data: Tuple[jnp.ndarray, jnp.ndarray],
                val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
                num_iterations: int = 1000,
                alternating_frequency: int = 10) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Perform co-design optimization with alternating updates.
        
        Args:
            train_data: Training data (inputs, targets)
            val_data: Validation data
            num_iterations: Number of optimization iterations
            alternating_frequency: How often to alternate between network and device optimization
            
        Returns:
            Optimized network and device parameters
        """
        X_train, y_train = train_data
        
        # Initialize parameters
        key = random.PRNGKey(42)
        network_params = self.network.init_params(key, (1, self.network.layer_sizes[0]))
        device_params = self._init_device_params(key)
        
        # Initialize optimizer states
        net_opt_state = self.network_optimizer.init(network_params)
        dev_opt_state = self.device_optimizer.init(device_params)
        
        # Training history
        history = {'loss': [], 'accuracy': [], 'power': [], 'area': []}
        
        for iteration in range(num_iterations):
            # Alternate between network and device optimization
            optimize_network = (iteration % alternating_frequency) < (alternating_frequency // 2)
            
            if optimize_network:
                # Optimize network parameters (device parameters frozen)
                def net_loss_fn(net_params):
                    return self.codesign_loss(net_params, device_params, X_train, y_train)
                
                loss, net_grads = value_and_grad(net_loss_fn)(network_params)
                net_updates, net_opt_state = self.network_optimizer.update(net_grads, net_opt_state, network_params)
                network_params = optax.apply_updates(network_params, net_updates)
                
            else:
                # Optimize device parameters (network parameters frozen)
                def dev_loss_fn(dev_params):
                    return self.codesign_loss(network_params, dev_params, X_train, y_train)
                
                loss, dev_grads = value_and_grad(dev_loss_fn)(device_params)
                dev_updates, dev_opt_state = self.device_optimizer.update(dev_grads, dev_opt_state, device_params)
                device_params = optax.apply_updates(device_params, dev_updates)
                
                # Project device parameters to feasible region
                device_params = self._project_device_params(device_params)
            
            # Record history
            if iteration % 50 == 0:
                combined_params = self._merge_params(network_params, device_params)
                outputs = self.network(X_train, combined_params, training=False)
                accuracy = jnp.mean((outputs - y_train) ** 2)  # MSE for now
                power = self._estimate_power(device_params)
                area = self._estimate_area(device_params)
                
                history['loss'].append(float(loss))
                history['accuracy'].append(float(accuracy))
                history['power'].append(float(power))
                history['area'].append(float(area))
                
                print(f"Iter {iteration}: Loss={loss:.4f}, Acc={accuracy:.4f}, "
                      f"Power={power*1000:.1f}mW, Area={area*1e6:.2f}mm²")
        
        return network_params, device_params
    
    def _init_device_params(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize device parameters within bounds."""
        device_params = {}
        
        for param_name, (low, high) in self.device_param_ranges.items():
            # Initialize uniformly within bounds
            param_key, key = random.split(key)
            
            if param_name == "waveguide_width":
                shape = (len(self.network.layers),)
            elif param_name == "device_area":
                total_devices = sum(l1 * l2 for l1, l2 in zip(self.network.layer_sizes[:-1], self.network.layer_sizes[1:]))
                shape = (total_devices,)
            else:
                shape = ()
            
            if shape:
                device_params[param_name] = random.uniform(param_key, shape, minval=low, maxval=high)
            else:
                device_params[param_name] = random.uniform(param_key, (), minval=low, maxval=high)
        
        return device_params
    
    def _merge_params(self, network_params: Dict[str, jnp.ndarray], 
                     device_params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Merge network and device parameters."""
        combined = network_params.copy()
        
        # Add device parameters to each layer
        for layer_name in combined.keys():
            combined[layer_name] = combined[layer_name].copy()
            for param_name, param_value in device_params.items():
                combined[layer_name][f"device_{param_name}"] = param_value
                
        return combined
    
    def _project_device_params(self, device_params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Project device parameters to feasible region."""
        projected = {}
        
        for param_name, param_value in device_params.items():
            if param_name in self.device_param_ranges:
                low, high = self.device_param_ranges[param_name]
                projected[param_name] = jnp.clip(param_value, low, high)
            else:
                projected[param_name] = param_value
                
        return projected
    
    def _estimate_power(self, device_params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Estimate power consumption from device parameters."""
        base_power = 10e-3  # 10mW base laser power
        
        # Add tuning power
        tuning_power = 0.0
        if "tuning_voltage" in device_params:
            tuning_power += jnp.sum(device_params["tuning_voltage"] ** 2) * 1e-3
            
        return base_power + tuning_power
    
    def _estimate_area(self, device_params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Estimate chip area from device parameters.""" 
        base_area = 1e-6  # 1mm² base area
        
        # Scale with device sizes
        device_area = 0.0
        if "device_area" in device_params:
            device_area = jnp.sum(device_params["device_area"])
            
        return base_area + device_area
    
    def _estimate_latency(self, device_params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Estimate computation latency."""
        base_latency = 1e-9  # 1ns base latency
        
        # Add switching latency
        switching_latency = 0.0
        if "switching_time" in device_params:
            switching_latency = jnp.max(device_params["switching_time"])
            
        return base_latency + switching_latency


class QuantumNoiseSimulator:
    """
    Simulator for quantum noise effects in photonic neural networks.
    
    Models shot noise, thermal noise, and quantum correlations
    that affect low-power photonic computing systems.
    """
    
    def __init__(self, temperature: float = 300.0, include_correlations: bool = False):
        """
        Initialize quantum noise simulator.
        
        Args:
            temperature: Operating temperature in Kelvin
            include_correlations: Whether to include quantum correlations
        """
        self.temperature = temperature
        self.include_correlations = include_correlations
        self.k_B = 1.38e-23  # Boltzmann constant
        self.h = 6.626e-34   # Planck constant
        
    def add_shot_noise(self, optical_powers: jnp.ndarray, 
                      measurement_time: float = 1e-6,
                      key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Add shot noise to optical power measurements."""
        if key is None:
            key = random.PRNGKey(0)
            
        # Shot noise is Poisson distributed
        # For large photon numbers, approximate as Gaussian
        wavelength = 1550e-9
        photon_energy = self.h * 3e8 / wavelength
        
        photon_numbers = optical_powers * measurement_time / photon_energy
        shot_noise_std = jnp.sqrt(photon_numbers)
        
        noise = random.normal(key, optical_powers.shape) * shot_noise_std * photon_energy / measurement_time
        
        return optical_powers + noise
    
    def add_thermal_noise(self, optical_powers: jnp.ndarray,
                         bandwidth: float = 1e9,
                         key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Add thermal noise to optical signals."""
        if key is None:
            key = random.PRNGKey(1)
            
        # Thermal noise power
        thermal_power = self.k_B * self.temperature * bandwidth
        
        noise = random.normal(key, optical_powers.shape) * jnp.sqrt(thermal_power)
        
        return optical_powers + noise
    
    def add_phase_noise(self, optical_fields: jnp.ndarray,
                       linewidth: float = 1e6,
                       key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Add laser phase noise to optical fields."""
        if key is None:
            key = random.PRNGKey(2)
            
        # Phase noise from laser linewidth
        phase_variance = 2 * jnp.pi * linewidth * 1e-6  # Assume 1μs coherence time
        phase_noise = random.normal(key, optical_fields.shape) * jnp.sqrt(phase_variance)
        
        # Apply phase noise
        noisy_fields = optical_fields * jnp.exp(1j * phase_noise)
        
        return noisy_fields


# Utility functions
def _flatten_params(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Flatten nested parameter dictionary to single array."""
    flat_params = []
    for layer_params in params.values():
        if isinstance(layer_params, dict):
            for param in layer_params.values():
                flat_params.append(param.flatten())
        else:
            flat_params.append(layer_params.flatten())
    
    return jnp.concatenate(flat_params) if flat_params else jnp.array([])


def create_hardware_constraints(max_power: float = 0.1,
                               max_temperature: float = 400.0,
                               device_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create standard hardware constraint configuration."""
    constraints = {
        'power_budget': max_power,
        'max_temperature': max_temperature,
        'switching_energy': 1e-15,  # 1fJ per switch
        'retention_time': 3600,     # 1 hour state retention
        'endurance_cycles': 1e6,    # 1M write cycles
    }
    
    if device_limits:
        constraints.update(device_limits)
        
    return constraints


__all__ = [
    "HardwareAwareOptimizer",
    "CoDesignOptimizer", 
    "QuantumNoiseSimulator",
    "create_hardware_constraints",
]