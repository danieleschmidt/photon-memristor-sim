"""
Basic tests for photon-memristor-sim

These tests verify core functionality and serve as integration tests
for the Rust-Python interface.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

try:
    import photon_memristor_sim as pms
    from photon_memristor_sim import jax_interface as jax_pms
    from photon_memristor_sim.neural_networks import PhotonicNeuralNetwork
except ImportError:
    pytest.skip("photon_memristor_sim not available", allow_module_level=True)


class TestCore:
    """Test core Rust functionality through Python interface."""
    
    def test_library_imports(self):
        """Test that core library imports successfully."""
        assert hasattr(pms, 'VERSION')
        assert hasattr(pms, 'SPEED_OF_LIGHT')
        assert pms.SPEED_OF_LIGHT == 299792458.0
        
    def test_optical_field_creation(self):
        """Test OpticalField creation and basic operations."""
        # Create simple optical field
        amplitude_real = np.ones((10, 10))
        amplitude_imag = np.zeros((10, 10))
        
        field = pms.PyOpticalField(
            amplitude_real, amplitude_imag, 
            wavelength=1550e-9, power=1e-3
        )
        
        assert field.wavelength == 1550e-9
        assert field.power == 1e-3
        assert field.dimensions() == (10, 10)
        
        # Test power calculation
        calculated_power = field.calculate_power()
        assert calculated_power > 0
        
    def test_photonic_array_creation(self):
        """Test PhotonicArray creation and basic operations."""
        array = pms.PyPhotonicArray("crossbar", 4, 4)
        
        assert array.dimensions() == (4, 4)
        assert array.total_power() == 0.0
        
        # Test metrics
        metrics = array.metrics()
        assert "total_devices" in metrics
        assert "total_power" in metrics
        
    def test_waveguide_mode_calculation(self):
        """Test waveguide mode calculation."""
        n_eff, intensity_profile = pms.calculate_waveguide_mode(
            width=450e-9,
            height=220e-9, 
            core_index=3.47,
            cladding_index=1.44,
            wavelength=1550e-9
        )
        
        # Effective index should be between core and cladding
        assert 1.44 < n_eff < 3.47
        
        # Intensity profile should be 32x32
        assert intensity_profile.shape == (32, 32)
        assert np.all(intensity_profile >= 0)
        

class TestJAXInterface:
    """Test JAX integration and differentiable operations."""
    
    def test_photonic_matmul(self):
        """Test photonic matrix multiplication."""
        inputs = jnp.array([1.0, 0.5, 0.8])
        weights = jnp.array([[0.5, 0.3, 0.2], 
                            [0.1, 0.7, 0.2],
                            [0.3, 0.3, 0.4]])
        
        outputs = jax_pms.photonic_matmul(inputs, weights)
        
        assert outputs.shape == (3,)
        assert jnp.all(outputs >= 0)  # Optical powers are non-negative
        
    def test_photonic_matmul_gradient(self):
        """Test automatic differentiation through photonic operations."""
        def loss_fn(weights):
            inputs = jnp.array([1.0, 0.5])
            outputs = jax_pms.photonic_matmul(inputs, weights)
            return jnp.sum(outputs ** 2)
            
        weights = jnp.array([[0.5, 0.3], [0.2, 0.7]])
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(weights)
        
        assert gradients.shape == weights.shape
        assert not jnp.allclose(gradients, 0)  # Should have non-zero gradients
        
    def test_photonic_conv2d(self):
        """Test photonic 2D convolution."""
        inputs = jnp.ones((8, 8, 3))  # 8x8 image with 3 channels
        kernel = jnp.ones((3, 3, 3, 2))  # 3x3 kernel, 3 input channels, 2 output channels
        
        outputs = jax_pms.photonic_conv2d(inputs, kernel, stride=1, padding="VALID")
        
        expected_shape = (6, 6, 2)  # (8-3+1, 8-3+1, 2)
        assert outputs.shape == expected_shape


class TestNeuralNetworks:
    """Test high-level neural network interfaces."""
    
    def test_photonic_neural_network_creation(self):
        """Test PhotonicNeuralNetwork creation."""
        layers = [784, 256, 128, 10]  # MNIST-like architecture
        pnn = PhotonicNeuralNetwork(layers)
        
        assert pnn.layer_sizes == layers
        assert len(pnn.layers) == 3  # 3 layer transitions
        
    def test_photonic_neural_network_forward(self):
        """Test forward pass through photonic neural network."""
        layers = [10, 5, 2]
        pnn = PhotonicNeuralNetwork(layers)
        
        # Initialize parameters
        key = random.PRNGKey(42)
        params = pnn.init_params(key, (1, 10))
        
        # Forward pass
        inputs = jnp.ones((1, 10))
        outputs = pnn(inputs, params, training=False)
        
        assert outputs.shape == (1, 2)
        
    def test_photonic_neural_network_training_step(self):
        """Test training step with loss computation."""
        layers = [5, 3, 2]
        pnn = PhotonicNeuralNetwork(layers)
        
        # Initialize
        key = random.PRNGKey(42)
        params = pnn.init_params(key, (1, 5))
        
        # Create dummy data
        inputs = random.normal(key, (10, 5))
        targets = random.normal(key, (10, 2))
        
        # Compute loss
        loss = pnn.loss_fn(params, inputs, targets)
        
        assert loss.shape == ()
        assert loss > 0
        
    def test_power_consumption_calculation(self):
        """Test power consumption estimation."""
        layers = [10, 5, 2]
        pnn = PhotonicNeuralNetwork(layers)
        
        key = random.PRNGKey(42)
        params = pnn.init_params(key, (1, 10))
        
        power = pnn.power_consumption(params)
        assert power >= 0
        
        total_power = pnn.total_power()
        assert total_power >= 0
        
    def test_device_count(self):
        """Test device counting."""
        layers = [10, 5, 2]
        pnn = PhotonicNeuralNetwork(layers)
        
        key = random.PRNGKey(42)
        params = pnn.init_params(key, (1, 10))
        
        device_count = pnn.device_count()
        expected_devices = 10*5 + 5*2  # Two weight matrices
        assert device_count == expected_devices


class TestDeviceSimulation:
    """Test device-specific simulation functions."""
    
    def test_device_simulator_creation(self):
        """Test device simulator creation."""
        pcm_sim = pms.create_device_simulator("pcm")
        assert "PCM" in pcm_sim
        
        oxide_sim = pms.create_device_simulator("oxide")
        assert "Oxide" in oxide_sim
        
        ring_sim = pms.create_device_simulator("ring")
        assert "Ring" in ring_sim
        
    def test_invalid_device_type(self):
        """Test error handling for invalid device types."""
        with pytest.raises(ValueError):
            pms.create_device_simulator("invalid_device")


class TestPerformance:
    """Performance and benchmarking tests."""
    
    def test_large_array_simulation(self):
        """Test simulation with large arrays."""
        array = pms.PyPhotonicArray("crossbar", 64, 64)
        
        # Should handle large arrays without errors
        assert array.dimensions() == (64, 64)
        
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        layers = [100, 50, 10]
        pnn = PhotonicNeuralNetwork(layers)
        
        key = random.PRNGKey(42)
        params = pnn.init_params(key, (1, 100))
        
        # Large batch
        batch_size = 1000
        inputs = random.normal(key, (batch_size, 100))
        
        outputs = pnn(inputs, params, training=False)
        assert outputs.shape == (batch_size, 10)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        inputs = jnp.array([1.0, 2.0])
        weights = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Wrong dimensions
        
        with pytest.raises((ValueError, RuntimeError)):
            jax_pms.photonic_matmul(inputs, weights)
            
    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError):
            pms.calculate_waveguide_mode(
                width=-1e-6,  # Negative width should be invalid
                height=220e-9,
                core_index=3.47,
                cladding_index=1.44,
                wavelength=1550e-9
            )


if __name__ == "__main__":
    pytest.main([__file__])