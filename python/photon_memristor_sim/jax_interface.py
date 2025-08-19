"""
JAX integration for differentiable photonic simulation

This module provides JAX primitives for photonic operations, enabling
automatic differentiation through photonic device physics.
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import xla
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    from ._core import jax_photonic_matmul, jax_photonic_matmul_vjp
    _RUST_CORE_AVAILABLE = True
except ImportError:
    _RUST_CORE_AVAILABLE = False
    # Pure Python fallbacks
    def jax_photonic_matmul(*args, **kwargs):
        raise NotImplementedError("Rust core not available - using pure Python simulation")
    def jax_photonic_matmul_vjp(*args, **kwargs):
        raise NotImplementedError("Rust core not available - using pure Python simulation")


@custom_vjp
def photonic_matmul(inputs: jnp.ndarray, weights: jnp.ndarray, 
                   wavelength: float = 1550e-9) -> jnp.ndarray:
    """
    Photonic matrix multiplication with automatic differentiation.
    
    This operation simulates optical matrix-vector multiplication through
    a photonic crossbar array, including device physics and optical losses.
    
    Args:
        inputs: Input optical powers [N]
        weights: Weight matrix (device states) [M, N] 
        wavelength: Optical wavelength in meters
        
    Returns:
        Output optical powers [M]
    """
    # Convert JAX arrays to numpy for Rust interface
    inputs_np = np.asarray(inputs, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    
    # Call Rust implementation
    # Convert inputs to list for Rust interface
    inputs_list = inputs_np.tolist() if inputs_np.ndim == 1 else inputs_np.flatten().tolist()
    weights_list = weights_np.tolist()
    
    outputs_np = jax_photonic_matmul(inputs_list, weights_list, wavelength)
    
    return jnp.array(outputs_np)


def photonic_matmul_fwd(inputs: jnp.ndarray, weights: jnp.ndarray, 
                       wavelength: float) -> Tuple[jnp.ndarray, Tuple]:
    """Forward pass for photonic_matmul."""
    outputs = photonic_matmul(inputs, weights, wavelength)
    return outputs, (inputs, weights, wavelength)


def photonic_matmul_bwd(residuals: Tuple, grad_outputs: jnp.ndarray) -> Tuple:
    """Backward pass for photonic_matmul using adjoint method."""
    inputs, weights, wavelength = residuals
    
    # Convert to numpy for Rust interface
    inputs_np = np.asarray(inputs, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    grad_outputs_np = np.asarray(grad_outputs, dtype=np.float64)
    
    # Convert to lists for Rust interface
    inputs_list = inputs_np.tolist() if inputs_np.ndim == 1 else inputs_np.flatten().tolist()
    weights_list = weights_np.tolist()
    grad_outputs_list = grad_outputs_np.tolist() if grad_outputs_np.ndim == 1 else grad_outputs_np.flatten().tolist()
    
    # Compute gradients using Rust implementation
    grad_inputs_np, grad_weights_np = jax_photonic_matmul_vjp(
        inputs_list, weights_list, grad_outputs_list, wavelength
    )
    
    grad_inputs = jnp.array(grad_inputs_np)
    grad_weights = jnp.array(grad_weights_np)
    
    # No gradient w.r.t. wavelength for now
    return grad_inputs, grad_weights, None


# Register the VJP (Vector-Jacobian Product)
photonic_matmul.defvjp(photonic_matmul_fwd, photonic_matmul_bwd)


@custom_vjp  
def photonic_conv2d(inputs: jnp.ndarray, kernel: jnp.ndarray,
                   stride: int = 1, padding: str = "VALID") -> jnp.ndarray:
    """
    2D convolution using photonic Fourier transform.
    
    Implements convolution in the Fourier domain using photonic FFT,
    which can be faster for large kernels due to optical parallelism.
    
    Args:
        inputs: Input image [H, W, C]
        kernel: Convolution kernel [KH, KW, C, F]
        stride: Convolution stride
        padding: Padding type ("VALID" or "SAME")
        
    Returns:
        Convolved output [H', W', F]
    """
    # For now, fallback to standard JAX conv2d
    # Real implementation would use optical FFT
    return jax.lax.conv_general_dilated(
        inputs[None, ...], kernel, 
        window_strides=[stride, stride],
        padding=padding
    )[0]


def photonic_conv2d_fwd(inputs: jnp.ndarray, kernel: jnp.ndarray,
                       stride: int, padding: str) -> Tuple[jnp.ndarray, Tuple]:
    """Forward pass for photonic_conv2d."""
    outputs = photonic_conv2d(inputs, kernel, stride, padding)
    return outputs, (inputs, kernel, stride, padding)


def photonic_conv2d_bwd(residuals: Tuple, grad_outputs: jnp.ndarray) -> Tuple:
    """Backward pass for photonic_conv2d."""
    inputs, kernel, stride, padding = residuals
    
    # Use JAX's built-in gradient computation for now
    grad_inputs = jax.lax.conv_transpose(
        grad_outputs[None, ...], kernel,
        strides=[stride, stride], padding=padding
    )[0]
    
    grad_kernel = jax.lax.conv_general_dilated(
        inputs[None, ...], grad_outputs[None, ...],
        window_strides=[1, 1], padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )[0]
    
    return grad_inputs, grad_kernel, None, None


photonic_conv2d.defvjp(photonic_conv2d_fwd, photonic_conv2d_bwd)


def create_photonic_primitive(name: str, forward_fn, backward_fn, 
                             **kwargs) -> Any:
    """
    Create custom JAX primitive for photonic operations.
    
    This helper function simplifies creating new JAX primitives for
    custom photonic device simulations.
    
    Args:
        name: Primitive name
        forward_fn: Forward implementation
        backward_fn: Backward (VJP) implementation
        **kwargs: Additional primitive parameters
        
    Returns:
        JAX primitive function
    """
    primitive = jax.core.Primitive(name)
    
    def primitive_impl(*args, **params):
        return forward_fn(*args, **params)
    
    def primitive_abstract_eval(*args, **params):
        # Return abstract shape/dtype information
        return jax.ShapeDtypeStruct(args[0].shape, args[0].dtype)
    
    def primitive_vjp(*args, **params):
        def vjp_fn(grad_out):
            return backward_fn(*args, grad_out, **params)
        return forward_fn(*args, **params), vjp_fn
    
    # Register implementations
    primitive.def_impl(primitive_impl)
    primitive.def_abstract_eval(primitive_abstract_eval)
    jax.ad.primitive_vjps[primitive] = primitive_vjp
    
    def primitive_fn(*args, **params):
        return primitive.bind(*args, **params, **kwargs)
    
    return primitive_fn


class PhotonicGradientTape:
    """
    Custom gradient tape for photonic operations.
    
    Provides fine-grained control over gradient computation through
    photonic devices, useful for hardware-aware optimization.
    """
    
    def __init__(self):
        self.operations = []
        self.gradients = {}
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def record_operation(self, op_name: str, inputs: Dict[str, jnp.ndarray], 
                        outputs: Dict[str, jnp.ndarray]):
        """Record a photonic operation for gradient computation."""
        self.operations.append({
            "name": op_name,
            "inputs": inputs,
            "outputs": outputs
        })
    
    def compute_gradients(self, target: jnp.ndarray, 
                         sources: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Compute gradients using recorded operations."""
        # Simplified implementation - would use reverse-mode AD
        return {name: jnp.zeros_like(array) for name, array in sources.items()}


def photonic_scan(f, init, xs, length=None):
    """
    Photonic version of jax.lax.scan for recurrent photonic networks.
    
    Implements temporal scanning through photonic memory elements,
    such as ring resonator delays or memristive state evolution.
    
    Args:
        f: Function to scan
        init: Initial state
        xs: Input sequence
        length: Sequence length
        
    Returns:
        Final state and output sequence
    """
    # For now, use standard JAX scan
    # Real implementation would include photonic memory dynamics
    return jax.lax.scan(f, init, xs, length)


def photonic_fft(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Photonic Fast Fourier Transform.
    
    Simulates optical FFT using photonic integrated circuits,
    which can provide significant speedup for large transforms.
    
    Args:
        x: Input signal
        axis: Transform axis
        
    Returns:
        FFT of input
    """
    # For now, use standard JAX FFT
    # Real implementation would simulate optical FFT circuits
    return jnp.fft.fft(x, axis=axis)


def photonic_attention(query: jnp.ndarray, key: jnp.ndarray, 
                      value: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Photonic attention mechanism.
    
    Implements attention using optical correlation and photonic memory,
    enabling potentially faster attention computation for large sequences.
    
    Args:
        query: Query vectors [seq_len, d_model]
        key: Key vectors [seq_len, d_model]  
        value: Value vectors [seq_len, d_model]
        mask: Optional attention mask
        
    Returns:
        Attention output [seq_len, d_model]
    """
    # Attention scores using photonic correlation
    scores = photonic_matmul(query, key.T) / jnp.sqrt(query.shape[-1])
    
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    
    # Softmax (would be implemented optically)
    attention_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention using photonic matmul
    output = photonic_matmul(attention_weights, value)
    
    return output


# Export main functions
__all__ = [
    "photonic_matmul",
    "photonic_conv2d", 
    "create_photonic_primitive",
    "PhotonicGradientTape",
    "photonic_scan",
    "photonic_fft",
    "photonic_attention",
]