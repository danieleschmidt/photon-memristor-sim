"""
Device models for photonic-memristor simulation

High-level Python interfaces for photonic device modeling,
including phase change materials, metal oxide memristors, and ring resonators.
"""

import jax.numpy as jnp
from jax import random
from typing import Dict, Tuple, Optional, Any
import numpy as np

try:
    from ._core import create_device_simulator
except ImportError:
    from .pure_python_fallbacks import create_device_simulator


class PhotonicDevice:
    """Base class for all photonic devices."""
    
    def __init__(self, device_type: str, **kwargs):
        self.device_type = device_type
        self.parameters = kwargs
        self._state = 0.0
        
    def simulate(self, optical_field: jnp.ndarray, wavelength: float = 1550e-9) -> jnp.ndarray:
        """Simulate device response to optical input."""
        raise NotImplementedError
        
    def update_state(self, new_state: float):
        """Update device state."""
        self._state = jnp.clip(new_state, 0.0, 1.0)
    
    def get_state(self) -> float:
        """Get current device state."""
        return float(self._state)


class PCMDevice(PhotonicDevice):
    """
    Phase Change Material device model.
    
    Models optical switching using materials like GST (Ge2Sb2Te5) that can
    switch between amorphous and crystalline states, changing optical properties.
    """
    
    def __init__(self, material: str = "GST", dimensions: Tuple[float, float, float] = (200e-9, 50e-9, 10e-9), crystallinity: float = 0.0):
        """
        Initialize PCM device.
        
        Args:
            material: PCM material type ("GST", "GSST")
            dimensions: Device dimensions (length, width, height) in meters
            crystallinity: Initial crystallinity level (0=amorphous, 1=crystalline)
        """
        super().__init__("pcm", material=material, dimensions=dimensions)
        self.material = material
        self.dimensions = dimensions
        self.crystallinity = jnp.clip(crystallinity, 0.0, 1.0)
        self.temperature = 300.0  # Kelvin
        
        # Material properties
        if material == "GST":
            self.amorphous_n = 4.0 + 0.1j
            self.crystalline_n = 6.5 + 0.5j
            self.melting_point = 888.0  # K
            self.crystallization_temp = 423.0  # K
        elif material == "GSST":
            self.amorphous_n = 3.8 + 0.08j
            self.crystalline_n = 6.2 + 0.4j
            self.melting_point = 850.0  # K
            self.crystallization_temp = 400.0  # K
        else:
            raise ValueError(f"Unknown PCM material: {material}")
    
    def simulate(self, optical_field: jnp.ndarray, wavelength: float = 1550e-9) -> jnp.ndarray:
        """Simulate PCM device response."""
        # Calculate effective refractive index based on crystallinity
        n_eff = (1 - self.crystallinity) * self.amorphous_n + self.crystallinity * self.crystalline_n
        
        # Calculate transmission (simplified model)
        absorption = jnp.abs(n_eff.imag)
        thickness = self.dimensions[2]  # height
        transmission = jnp.exp(-4 * jnp.pi * absorption * thickness / wavelength)
        
        # Apply Fresnel reflection losses
        fresnel_loss = 0.96  # Typical AR coating
        
        return optical_field * transmission * fresnel_loss
    
    def switch_crystallinity(self, target_crystallinity: float, pulse_power: float, pulse_duration: float):
        """Switch crystallinity using optical pulse."""
        # Calculate temperature rise
        volume = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        energy = pulse_power * pulse_duration
        
        # Simplified thermal model
        density = 6150.0  # kg/m³ for GST
        specific_heat = 200.0  # J/kg/K
        mass = density * volume
        
        delta_T = energy / (mass * specific_heat)
        peak_temperature = self.temperature + delta_T
        
        # Update crystallinity based on temperature
        if peak_temperature > self.melting_point:
            # Melt and quench to amorphous
            self.crystallinity = 0.0
        elif peak_temperature > self.crystallization_temp:
            # Crystallize
            self.crystallinity = target_crystallinity
            
        self.update_state(self.crystallinity)
    
    def get_optical_constants(self, wavelength: float) -> complex:
        """Get wavelength-dependent optical constants."""
        # Linear interpolation between amorphous and crystalline
        return (1 - self.crystallinity) * self.amorphous_n + self.crystallinity * self.crystalline_n
    
    def get_crystallinity(self) -> float:
        """Get current crystallinity level."""
        return self.crystallinity
    
    def get_transmission(self, wavelength: float) -> float:
        """Get transmission coefficient at given wavelength."""
        n_eff = self.get_optical_constants(wavelength)
        absorption = jnp.abs(n_eff.imag)
        thickness = self.dimensions[2]
        return jnp.exp(-4 * jnp.pi * absorption * thickness / wavelength)


class OxideMemristor(PhotonicDevice):
    """
    Metal oxide memristor device model.
    
    Models resistive switching in metal oxides like HfO2, TaOx, or TiO2,
    which can modulate optical transmission through the plasma dispersion effect.
    """
    
    def __init__(self, oxide: str = "HfO2", thickness: float = 5e-9, area: float = 100e-18, conductance: float = 1e-6):
        """
        Initialize oxide memristor.
        
        Args:
            oxide: Oxide material ("HfO2", "TaOx", "TiO2")
            thickness: Oxide thickness in meters
            area: Device area in m²
            conductance: Initial conductance in Siemens
        """
        super().__init__("oxide", oxide_type=oxide, thickness=thickness, area=area)
        
        self.oxide_type = oxide
        self.thickness = thickness
        self.area = area
        self.conductance = conductance
        
        # Material properties
        self.material_props = {
            "HfO2": {"bandgap": 5.8, "dielectric": 25, "mobility": 1e-14},
            "TaOx": {"bandgap": 4.2, "dielectric": 22, "mobility": 2e-14}, 
            "TiO2": {"bandgap": 3.2, "dielectric": 80, "mobility": 5e-14}
        }
        
        if oxide_type not in self.material_props:
            raise ValueError(f"Unknown oxide type: {oxide_type}")
    
    def simulate(self, optical_field: jnp.ndarray, wavelength: float = 1550e-9) -> jnp.ndarray:
        """Simulate oxide memristor optical response."""
        # Conductance affects free carrier concentration
        carrier_density = self.conductance * self.thickness / (1.6e-19 * self.material_props[self.oxide_type]["mobility"])
        
        # Simplified plasma dispersion effect
        omega = 2 * jnp.pi * 3e8 / wavelength
        
        # Simplified model: conductance affects transmission
        modulation_strength = jnp.clip(self.conductance * 1e6, 0, 1)  # Normalize
        
        # Apply modulation (simplified - real part only)
        transmission = 0.5 + 0.5 * modulation_strength
        
        return optical_field * transmission
    
    def set_voltage(self, voltage: float, duration: float):
        """Apply voltage to change conductance state."""
        # Simplified conductance switching model
        threshold_voltage = 1.0  # V
        
        if jnp.abs(voltage) > threshold_voltage:
            if voltage > 0:
                # SET process
                self.conductance = jnp.minimum(self.conductance * 10, 1e-3)
            else:
                # RESET process  
                self.conductance = jnp.maximum(self.conductance / 10, 1e-9)
                
        self.update_state(jnp.log10(self.conductance + 1e-9) / 6 + 1)  # Normalize to [0,1]
    
    def get_resistance(self) -> float:
        """Get current resistance."""
        return 1.0 / max(self.conductance, 1e-12)
    
    def get_conductance(self) -> float:
        """Get current conductance."""
        return float(self.conductance)


class MicroringResonator(PhotonicDevice):
    """
    Microring resonator device model.
    
    Models optical filtering and switching using ring resonators,
    which can be tuned thermally or electro-optically.
    """
    
    def __init__(self, radius: float = 10e-6, coupling_gap: float = 200e-9, quality_factor: float = 10000):
        """
        Initialize microring resonator.
        
        Args:
            radius: Ring radius in meters
            coupling_gap: Gap between ring and bus waveguide in meters  
            quality_factor: Quality factor of the resonator
        """
        super().__init__("ring", radius=radius, coupling_gap=coupling_gap, quality_factor=quality_factor)
        
        self.radius = radius
        self.coupling_gap = coupling_gap
        self.quality_factor = quality_factor
        self.effective_index = 2.4  # Silicon photonics
        self.thermo_optic_coeff = 1.8e-4  # /K for silicon
        self.tuning_power = 0.0  # Thermal tuning power
        
    def simulate(self, optical_field: jnp.ndarray, wavelength: float = 1550e-9) -> jnp.ndarray:
        """Simulate ring resonator transmission."""
        # Calculate resonance wavelengths
        circumference = 2 * jnp.pi * self.radius
        
        # Include thermal tuning
        delta_n = self.thermo_optic_coeff * self.tuning_power / 1e-3  # Assume 1mW changes n by thermo_optic_coeff
        n_eff = self.effective_index + delta_n
        
        # Free spectral range
        fsr = wavelength**2 / (n_eff * circumference)
        
        # Round trip phase
        round_trip_phase = 2 * jnp.pi * n_eff * circumference / wavelength
        
        # Coupling coefficients (simplified)
        kappa = 0.1  # Power coupling coefficient
        
        # Ring transmission (all-pass filter approximation)
        finesse = jnp.pi * jnp.sqrt(self.quality_factor)
        detuning = round_trip_phase - 2 * jnp.pi * jnp.round(round_trip_phase / (2 * jnp.pi))
        
        transmission = (1 - kappa) / (1 + finesse**2 * jnp.sin(detuning/2)**2)
        
        return optical_field * jnp.sqrt(transmission)
    
    def set_thermal_tuning(self, power: float):
        """Set thermal tuning power."""
        self.tuning_power = jnp.clip(power, 0, 50e-3)  # Max 50mW
        self.update_state(self.tuning_power / 50e-3)
    
    def get_fsr(self, wavelength: float = 1550e-9) -> float:
        """Get free spectral range at given wavelength."""
        circumference = 2 * jnp.pi * self.radius
        return wavelength**2 / (self.effective_index * circumference)
    
    def get_resonance_wavelengths(self, center_wavelength: float = 1550e-9, num_modes: int = 10) -> jnp.ndarray:
        """Get resonance wavelengths around center wavelength."""
        fsr = self.get_fsr(center_wavelength)
        modes = jnp.arange(-num_modes//2, num_modes//2 + 1)
        return center_wavelength + modes * fsr


class PhotonicCrossbar:
    """
    Photonic crossbar array for neural network computation.
    
    Implements matrix-vector multiplication using arrays of photonic devices.
    """
    
    def __init__(self, size: Tuple[int, int], cell_type: str = "pcm"):
        """
        Initialize photonic crossbar.
        
        Args:
            size: Array size (rows, cols)
            cell_type: Type of devices in crossbar ("pcm", "oxide", "ring")
        """
        self.size = size
        self.cell_type = cell_type
        
        # Initialize device array
        self.devices = []
        for i in range(size[0]):
            row = []
            for j in range(size[1]):
                if cell_type == "pcm":
                    device = PCMDevice()
                elif cell_type == "oxide":
                    device = OxideMemristor()
                elif cell_type == "ring":
                    device = MicroringResonator()
                else:
                    raise ValueError(f"Unknown cell type: {cell_type}")
                row.append(device)
            self.devices.append(row)
    
    def set_weights(self, weight_matrix: jnp.ndarray):
        """Set device states according to weight matrix."""
        if weight_matrix.shape != self.size:
            raise ValueError(f"Weight matrix shape {weight_matrix.shape} doesn't match crossbar size {self.size}")
            
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Normalize weights to device state range [0, 1]
                normalized_weight = (weight_matrix[i, j] + 1) / 2  # Assume weights in [-1, 1]
                self.devices[i][j].update_state(normalized_weight)
    
    def matrix_vector_multiply(self, input_vector: jnp.ndarray, wavelength: float = 1550e-9) -> jnp.ndarray:
        """Perform optical matrix-vector multiplication."""
        if len(input_vector) != self.size[1]:
            raise ValueError(f"Input vector length {len(input_vector)} doesn't match crossbar columns {self.size[1]}")
        
        output_vector = jnp.zeros(self.size[0])
        
        for i in range(self.size[0]):
            row_sum = 0.0
            for j in range(self.size[1]):
                # Simulate device response to input optical power
                optical_field = jnp.array([input_vector[j]])
                response = self.devices[i][j].simulate(optical_field, wavelength)
                row_sum += jnp.sum(jnp.abs(response)**2)  # Power detection
                
            output_vector = output_vector.at[i].set(row_sum)
            
        return output_vector
    
    def get_weight_matrix(self) -> jnp.ndarray:
        """Get current weight matrix from device states."""
        weights = jnp.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                state = self.devices[i][j].get_state()
                weights = weights.at[i, j].set(2 * state - 1)  # Convert [0,1] to [-1,1]
        return weights
    
    def total_power_consumption(self) -> float:
        """Calculate total power consumption."""
        total_power = 0.0
        for row in self.devices:
            for device in row:
                if hasattr(device, 'tuning_power'):
                    total_power += device.tuning_power
                else:
                    total_power += 1e-3  # Assume 1mW per device
        return total_power


# Convenience functions
def create_pcm_array(size: Tuple[int, int], material: str = "GST") -> PhotonicCrossbar:
    """Create crossbar array with PCM devices."""
    return PhotonicCrossbar(size, "pcm")


def create_oxide_array(size: Tuple[int, int], oxide_type: str = "HfO2") -> PhotonicCrossbar:
    """Create crossbar array with oxide memristors.""" 
    return PhotonicCrossbar(size, "oxide")


def create_ring_array(size: Tuple[int, int]) -> PhotonicCrossbar:
    """Create crossbar array with ring resonators."""
    return PhotonicCrossbar(size, "ring")


__all__ = [
    "PhotonicDevice",
    "PCMDevice", 
    "OxideMemristor",
    "MicroringResonator",
    "PhotonicCrossbar",
    "create_pcm_array",
    "create_oxide_array", 
    "create_ring_array",
]