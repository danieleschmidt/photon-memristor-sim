# ADR-0003: Neural Network Architecture Integration

## Status

Accepted

## Context

Photon-Memristor-Sim must seamlessly integrate with modern neural network frameworks while providing hardware-specific optimizations for photonic computing. Key requirements:

1. **JAX Integration**: First-class support for automatic differentiation
2. **Hardware Constraints**: Model physical limitations of photonic devices
3. **Co-optimization**: Simultaneous optimization of device parameters and network weights
4. **Multiple Architectures**: Support for MLPs, CNNs, RNNs, and emerging architectures

## Decision

Implement a layered neural network integration approach:

### 1. Core Photonic Layers (Rust)
Low-level photonic operations with automatic differentiation support:
- `PhotonicLinear`: Matrix multiplication via optical interference
- `PhotonicConvolution`: Spatial convolution using waveguide arrays
- `PhotonicActivation`: Nonlinear optical effects (Kerr, saturation)
- `PhotonicMemory`: Memristive state storage and retrieval

### 2. Python Neural Network API
High-level neural network construction using JAX/Flax patterns:
- `PhotonicMLP`: Multi-layer perceptron with optical layers
- `PhotonicCNN`: Convolutional networks with photonic kernels
- `PhotonicRNN`: Recurrent networks with optical memory
- `PhotonicTransformer`: Attention mechanisms in photonic domain

### 3. Hardware-Aware Optimization
Co-design optimization considering device physics:
- `HardwareConstraints`: Power, thermal, fabrication limits
- `CoOptimizer`: Joint device-algorithm optimization
- `NoiseAwareTraining`: Training with realistic device noise
- `RobustnessAnalysis`: Sensitivity to manufacturing variations

## Architecture Design

### Core Photonic Layers

```rust
// Rust core implementation
pub struct PhotonicLinear {
    pub array: PhotonicArray,
    pub device_params: DeviceParameters,
    pub constraints: HardwareConstraints,
}

impl PhotonicLinear {
    pub fn forward(&self, inputs: &Tensor, weights: &Tensor) -> Result<Tensor> {
        // 1. Encode electrical signals to optical
        let optical_inputs = self.encode_to_optical(inputs)?;
        
        // 2. Configure memristor weights
        self.array.configure_weights(weights)?;
        
        // 3. Optical matrix multiplication
        let optical_outputs = self.array.matrix_multiply(&optical_inputs)?;
        
        // 4. Add device impairments
        let noisy_outputs = self.add_device_noise(&optical_outputs)?;
        
        // 5. Convert back to electrical domain
        let electrical_outputs = self.decode_to_electrical(&noisy_outputs)?;
        
        Ok(electrical_outputs)
    }
    
    pub fn backward(&self, grad_output: &Tensor) -> Result<(Tensor, Tensor)> {
        // Implement adjoint method for gradient computation
        let grad_inputs = self.compute_input_gradients(grad_output)?;
        let grad_weights = self.compute_weight_gradients(grad_output)?;
        Ok((grad_inputs, grad_weights))
    }
}
```

### JAX Integration Layer

```python
from jax import custom_vjp
import jax.numpy as jnp
from photon_memristor_sim._core import PhotonicLinear as CorePhotonicLinear

@custom_vjp
def photonic_linear(inputs, weights, device_params):
    """JAX-compatible photonic linear layer."""
    core_layer = CorePhotonicLinear(device_params)
    return core_layer.forward(inputs, weights)

def photonic_linear_fwd(inputs, weights, device_params):
    outputs = photonic_linear(inputs, weights, device_params)
    return outputs, (inputs, weights, device_params)

def photonic_linear_bwd(res, grad_outputs):
    inputs, weights, device_params = res
    core_layer = CorePhotonicLinear(device_params)
    grad_inputs, grad_weights = core_layer.backward(grad_outputs)
    return grad_inputs, grad_weights, None

photonic_linear.defvjp(photonic_linear_fwd, photonic_linear_bwd)
```

### High-Level Neural Network API

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class PhotonicMLP(nn.Module):
    """Multi-layer perceptron using photonic layers."""
    
    features: Sequence[int]
    device_config: dict
    activation: str = 'photonic_relu'
    use_batch_norm: bool = False
    
    def setup(self):
        self.layers = [
            PhotonicLinear(
                features=feat,
                device_config=self.device_config,
                name=f'layer_{i}'
            )
            for i, feat in enumerate(self.features)
        ]
        
        if self.use_batch_norm:
            self.batch_norms = [
                nn.BatchNorm() for _ in range(len(self.features) - 1)
            ]
    
    def __call__(self, x, training=False):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x, use_running_average=not training)
            
            # Photonic activation functions
            if self.activation == 'photonic_relu':
                x = photonic_relu(x, self.device_config)
            elif self.activation == 'optical_sigmoid':
                x = optical_sigmoid(x, self.device_config)
        
        # Final layer (no activation)
        x = self.layers[-1](x)
        return x
```

### Hardware-Aware Training

```python
class HardwareAwareOptimizer:
    """Optimizer that considers hardware constraints."""
    
    def __init__(self, base_optimizer, constraints):
        self.base_optimizer = base_optimizer
        self.constraints = constraints
    
    def apply_gradients(self, grads, params):
        # 1. Apply hardware constraints to gradients
        constrained_grads = self.apply_constraints(grads, params)
        
        # 2. Project onto feasible parameter space
        projected_grads = self.project_gradients(constrained_grads, params)
        
        # 3. Apply base optimizer update
        new_params = self.base_optimizer.apply_gradients(projected_grads, params)
        
        # 4. Ensure parameters remain feasible
        feasible_params = self.enforce_feasibility(new_params)
        
        return feasible_params
    
    def apply_constraints(self, grads, params):
        """Apply power, thermal, and fabrication constraints."""
        constrained_grads = {}
        
        for key, grad in grads.items():
            if 'weight' in key:
                # Power constraint: limit total optical power
                power_penalty = self.compute_power_penalty(params[key])
                grad = grad + self.constraints.power_lambda * power_penalty
                
                # Thermal constraint: prevent hot spots
                thermal_penalty = self.compute_thermal_penalty(params[key])
                grad = grad + self.constraints.thermal_lambda * thermal_penalty
                
                # Fabrication constraint: realistic device parameters
                fab_penalty = self.compute_fabrication_penalty(params[key])
                grad = grad + self.constraints.fab_lambda * fab_penalty
            
            constrained_grads[key] = grad
        
        return constrained_grads
```

### Co-Design Optimization

```python
class PhotonicCoDesigner:
    """Joint optimization of neural network and device parameters."""
    
    def __init__(self, network, device_params, objectives):
        self.network = network
        self.device_params = device_params
        self.objectives = objectives
    
    def optimize(self, data, num_iterations=1000):
        """Multi-objective co-design optimization."""
        
        # Initialize parameters
        network_params = self.network.init(data[0])
        device_params = self.device_params.copy()
        
        # Pareto optimization loop
        pareto_front = []
        
        for iteration in range(num_iterations):
            # Evaluate objectives
            accuracy = self.evaluate_accuracy(network_params, device_params, data)
            power = self.evaluate_power(network_params, device_params)
            area = self.evaluate_area(device_params)
            latency = self.evaluate_latency(network_params, device_params)
            
            objectives = {
                'accuracy': accuracy,
                'power': power, 
                'area': area,
                'latency': latency
            }
            
            # Multi-objective gradient computation
            grads = self.compute_multi_objective_gradients(
                network_params, device_params, objectives
            )
            
            # Update parameters
            network_params = self.update_network_params(network_params, grads['network'])
            device_params = self.update_device_params(device_params, grads['device'])
            
            # Track Pareto solutions
            if self.is_pareto_optimal(objectives, pareto_front):
                pareto_front.append((objectives, network_params.copy(), device_params.copy()))
        
        return pareto_front
```

## Implementation Strategy

### Phase 1: Core Layer Implementation (Weeks 1-2)
- Implement `PhotonicLinear` in Rust with gradient support
- Create JAX custom primitive bindings
- Basic unit tests and benchmarks

### Phase 2: Neural Network API (Weeks 3-4)
- Implement `PhotonicMLP`, `PhotonicCNN` classes
- Integration with Flax/Optax ecosystem
- Training loop examples and tutorials

### Phase 3: Hardware Constraints (Weeks 5-6)
- Implement constraint handling in optimizer
- Power, thermal, fabrication models
- Robust training with device variations

### Phase 4: Co-Design Optimization (Weeks 7-8)
- Multi-objective optimization framework
- Pareto front exploration algorithms
- Validation against analytical models

## Validation Strategy

### Unit Tests
```python
def test_photonic_linear_gradient():
    """Test gradient computation accuracy."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (10, 784))
    
    # Compare numerical and analytical gradients
    def loss_fn(weights):
        y = photonic_linear(x, weights, device_config)
        return jnp.sum(y ** 2)
    
    analytical_grad = jax.grad(loss_fn)(weights)
    numerical_grad = compute_numerical_gradient(loss_fn, weights)
    
    assert jnp.allclose(analytical_grad, numerical_grad, rtol=1e-5)
```

### Integration Tests
```python
def test_photonic_mlp_training():
    """Test end-to-end neural network training."""
    # Create synthetic classification dataset
    X_train, y_train = make_classification_dataset(1000, 784, 10)
    
    # Initialize photonic MLP
    model = PhotonicMLP([784, 256, 128, 10])
    
    # Train for several epochs
    params = train_photonic_model(model, X_train, y_train, epochs=10)
    
    # Evaluate performance
    accuracy = evaluate_model(model, params, X_test, y_test)
    assert accuracy > 0.8  # Reasonable performance threshold
```

### Hardware Validation
```python
def test_hardware_constraints():
    """Test that hardware constraints are enforced."""
    model = PhotonicMLP([100, 50, 10])
    constraints = HardwareConstraints(max_power=100e-3, max_temperature=350)
    
    # Train with constraints
    params = train_with_constraints(model, data, constraints)
    
    # Verify constraints are satisfied
    power = compute_total_power(params)
    temperature = compute_max_temperature(params)
    
    assert power <= constraints.max_power
    assert temperature <= constraints.max_temperature
```

## Performance Targets

### Training Speed
- **Small Networks** (< 1K parameters): Real-time training (< 1s/epoch)
- **Medium Networks** (1K-100K parameters): Interactive training (< 10s/epoch)
- **Large Networks** (> 100K parameters): Batch training (< 300s/epoch)

### Memory Efficiency
- **Parameter Storage**: Compressed representations for device states
- **Gradient Computation**: Checkpointing for large networks
- **Batch Processing**: Efficient batching for optical matrix operations

### Hardware Accuracy
- **Device Physics**: < 5% error vs experimental measurements
- **Noise Modeling**: Realistic thermal and shot noise
- **Manufacturing Variations**: Monte Carlo analysis with measured statistics

## Risks and Mitigations

### Risk: JAX Integration Complexity
**Mitigation**: Extensive testing of custom primitives, fallback implementations

### Risk: Hardware Constraint Optimization Difficulty
**Mitigation**: Hierarchical optimization, constraint relaxation methods

### Risk: Gradient Computation Accuracy
**Mitigation**: Analytical validation, higher-precision reference implementations

### Risk: Training Instability with Device Noise
**Mitigation**: Noise-aware training algorithms, robust optimization methods

## Success Metrics

### Functionality
- All major neural network architectures supported (MLP, CNN, RNN)
- Full gradient computation with < 1% error vs numerical gradients
- Hardware constraints enforced during training

### Performance
- Training speed competitive with GPU-based implementations
- Memory usage scales linearly with network size
- Convergence rates comparable to ideal (noiseless) training

### Usability
- API familiar to JAX/Flax users
- Comprehensive documentation and examples
- Integration with popular ML libraries (Optax, Weights & Biases)

## References

- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/autodiff_cookbook.html)
- [Flax Neural Network Library](https://flax.readthedocs.io/)
- [Photonic Neural Networks: A Survey](https://example.com/photonic-nn-survey)
- [Hardware-Aware Neural Architecture Search](https://example.com/hardware-aware-nas)

## Changelog

- **2025-01-XX**: Initial version - Neural network integration strategy
- **Future**: Updates based on implementation experience and user feedback