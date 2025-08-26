"""
Advanced Memristor Interface for Multi-Physics Simulation
Generation 1: Simple Python Interface to Rust Advanced Memristor Models
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
from dataclasses import dataclass, asdict
from .devices import MemristorDevice
from . import _core  # Rust bindings


@dataclass
class MemristorConfig:
    """Configuration for multi-physics memristor"""
    material_type: str = "GST"  # GST, HfO2, TiO2
    dimensions: Tuple[float, float, float] = (100e-9, 50e-9, 10e-9)  # L x W x H
    ambient_temperature: float = 300.0  # Kelvin
    thermal_time_constant: float = 1e-6  # seconds
    temperature_coefficient: float = 0.01  # 1% per Kelvin


@dataclass  
class MemristorState:
    """Current state of memristor device"""
    conductance: float
    temperature: float
    absorption: float
    internal_state: float
    last_update_time: float


class AdvancedMemristorDevice:
    """
    Advanced multi-physics memristor device with thermal, optical, and electrical effects.
    
    This class provides a high-level Python interface to sophisticated memristor models
    implemented in Rust for maximum performance.
    """
    
    def __init__(self, config: MemristorConfig):
        """Initialize advanced memristor device"""
        self.config = config
        self._rust_device = None  # Will be initialized when Rust bindings available
        self._state = MemristorState(
            conductance=1e-6,  # Initial conductance
            temperature=config.ambient_temperature,
            absorption=0.01,
            internal_state=0.5,
            last_update_time=0.0
        )
        self._material_properties = self._get_material_properties(config.material_type)
        
    def _get_material_properties(self, material_type: str) -> Dict:
        """Get material properties for specified material"""
        materials = {
            "GST": {
                "thermal_conductivity": 0.5,  # W/(m·K)
                "base_conductivity": 1e4,  # S/m
                "refractive_index_real": 6.5,
                "refractive_index_imag": 0.5,
                "activation_energy": 0.5,  # eV
                "ion_mobility": 1e-14  # m²/(V·s)
            },
            "HfO2": {
                "thermal_conductivity": 23.0,
                "base_conductivity": 1e-8,
                "refractive_index_real": 2.1,
                "refractive_index_imag": 0.0,
                "activation_energy": 0.6,
                "ion_mobility": 1e-15
            },
            "TiO2": {
                "thermal_conductivity": 8.4,
                "base_conductivity": 1e-10,
                "refractive_index_real": 2.5,
                "refractive_index_imag": 0.0,
                "activation_energy": 0.8,
                "ion_mobility": 1e-16
            }
        }
        
        if material_type not in materials:
            raise ValueError(f"Unsupported material type: {material_type}")
            
        return materials[material_type]
    
    def update_state(self, voltage: float, optical_power: float, time_step: float) -> MemristorState:
        """
        Update memristor state with applied voltage and optical power
        
        Args:
            voltage: Applied voltage (V)
            optical_power: Incident optical power (W)
            time_step: Simulation time step (s)
            
        Returns:
            Updated memristor state
        """
        # Calculate Joule heating
        electrical_power = voltage * voltage * self._state.conductance
        
        # Calculate optical heating  
        optical_heating = optical_power * self._state.absorption
        
        # Update temperature with thermal dynamics
        total_heating = electrical_power + optical_heating
        thermal_mass = self._calculate_thermal_mass()
        temperature_change = total_heating * time_step / (thermal_mass * self.config.thermal_time_constant)
        
        # Thermal relaxation
        temp_diff = self._state.temperature - self.config.ambient_temperature
        thermal_decay = temp_diff * time_step / self.config.thermal_time_constant
        
        self._state.temperature += temperature_change - thermal_decay
        
        # Update internal state based on voltage and temperature
        field_strength = voltage / self.config.dimensions[0]  # V/m
        k_B = 1.381e-23  # Boltzmann constant
        q = 1.602e-19    # Elementary charge
        
        temperature_factor = np.exp(-self._material_properties["activation_energy"] * q / 
                                  (k_B * self._state.temperature))
        
        drift_velocity = self._material_properties["ion_mobility"] * field_strength * temperature_factor
        state_change = drift_velocity * time_step / self.config.dimensions[0]
        
        self._state.internal_state = np.clip(self._state.internal_state + state_change, 0.0, 1.0)
        
        # Update conductance based on internal state
        conductance_ratio = 1000.0  # High/low conductance ratio
        base_conductance = self._material_properties["base_conductivity"] * \
                          self.config.dimensions[1] * self.config.dimensions[2] / self.config.dimensions[0]
        
        self._state.conductance = base_conductance * \
            (1.0 + (conductance_ratio - 1.0) * self._state.internal_state) * \
            (1.0 + self.config.temperature_coefficient * 
             (self._state.temperature - self.config.ambient_temperature))
        
        # Update optical properties
        self._update_optical_properties()
        
        self._state.last_update_time += time_step
        
        return self._state
    
    def _calculate_thermal_mass(self) -> float:
        """Calculate thermal mass of the device"""
        volume = self.config.dimensions[0] * self.config.dimensions[1] * self.config.dimensions[2]
        density = 6150.0  # kg/m³ (typical for GST)
        heat_capacity = 230.0  # J/(kg·K)
        return volume * density * heat_capacity
    
    def _update_optical_properties(self):
        """Update optical properties based on current state"""
        if self.config.material_type == "GST":
            # GST phase change affects refractive index dramatically
            crystalline_n = 6.5 + 0.5j
            amorphous_n = 4.0 + 0.1j
            
            current_n = amorphous_n + (crystalline_n - amorphous_n) * self._state.internal_state
            
            # Absorption scales with imaginary part
            self._state.absorption = current_n.imag * 0.1
            
        elif self.config.material_type in ["HfO2", "TiO2"]:
            # Oxide memristors have weaker optical effects
            base_absorption = 0.01
            self._state.absorption = base_absorption * (1.0 + self._state.internal_state * 0.5)
    
    def get_optical_transmission(self, wavelength: float) -> float:
        """
        Calculate optical transmission at given wavelength
        
        Args:
            wavelength: Wavelength in meters
            
        Returns:
            Transmission coefficient (0 to 1)
        """
        path_length = self.config.dimensions[2]  # Thickness
        
        if self.config.material_type == "GST":
            # Calculate from current refractive index
            crystalline_n = 6.5 + 0.5j
            amorphous_n = 4.0 + 0.1j
            current_n = amorphous_n + (crystalline_n - amorphous_n) * self._state.internal_state
            absorption_coeff = 4.0 * np.pi * current_n.imag / wavelength
        else:
            absorption_coeff = 4.0 * np.pi * self._state.absorption / wavelength
            
        return np.exp(-absorption_coeff * path_length)
    
    def get_current_state(self) -> MemristorState:
        """Get current device state"""
        return self._state
    
    def reset(self):
        """Reset device to initial state"""
        self._state.internal_state = 0.5
        self._state.temperature = self.config.ambient_temperature
        self._state.conductance = self._material_properties["base_conductivity"]
        self._update_optical_properties()
    
    def calculate_switching_energy(self, voltage: float, pulse_duration: float) -> float:
        """Calculate energy required for switching pulse"""
        power = voltage * voltage * self._state.conductance
        return power * pulse_duration
    
    def predict_switching_time(self, voltage: float, target_state: float) -> float:
        """Predict time required to reach target state at given voltage"""
        field_strength = voltage / self.config.dimensions[0]
        k_B = 1.381e-23
        q = 1.602e-19
        
        temperature_factor = np.exp(-self._material_properties["activation_energy"] * q / 
                                  (k_B * self._state.temperature))
        
        drift_velocity = self._material_properties["ion_mobility"] * field_strength * temperature_factor
        state_change_needed = abs(target_state - self._state.internal_state)
        distance_to_travel = state_change_needed * self.config.dimensions[0]
        
        if drift_velocity > 0.0:
            return distance_to_travel / drift_velocity
        else:
            return float('inf')


class MemristorArray:
    """
    Large-scale memristor array with crosstalk and thermal coupling effects
    """
    
    def __init__(self, rows: int, cols: int, config: MemristorConfig):
        """Initialize memristor array"""
        self.rows = rows
        self.cols = cols
        self.config = config
        
        # Create array of memristor devices
        self.devices = [[AdvancedMemristorDevice(config) for _ in range(cols)] 
                       for _ in range(rows)]
        
        # Initialize crosstalk matrix (simple nearest-neighbor model)
        self.crosstalk_coefficient = 0.01  # 1% crosstalk
        
    def update_array(self, 
                    voltages: np.ndarray, 
                    optical_powers: np.ndarray, 
                    time_step: float) -> List[List[MemristorState]]:
        """
        Update entire array with voltage and optical power matrices
        
        Args:
            voltages: 2D array of voltages (V)
            optical_powers: 2D array of optical powers (W)  
            time_step: Simulation time step (s)
            
        Returns:
            List of lists containing updated states
        """
        # Apply crosstalk effects
        voltages_with_crosstalk = self._apply_crosstalk(voltages)
        
        # Update each device
        states = []
        for i in range(self.rows):
            row_states = []
            for j in range(self.cols):
                state = self.devices[i][j].update_state(
                    voltages_with_crosstalk[i, j],
                    optical_powers[i, j],
                    time_step
                )
                row_states.append(state)
            states.append(row_states)
            
        return states
    
    def _apply_crosstalk(self, voltages: np.ndarray) -> np.ndarray:
        """Apply nearest-neighbor crosstalk effects"""
        voltages_with_crosstalk = voltages.copy()
        
        for i in range(self.rows):
            for j in range(self.cols):
                crosstalk_sum = 0.0
                
                # Add contributions from neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                            
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            crosstalk_sum += voltages[ni, nj] * self.crosstalk_coefficient
                
                voltages_with_crosstalk[i, j] += crosstalk_sum
                
        return voltages_with_crosstalk
    
    def get_conductance_matrix(self) -> np.ndarray:
        """Get matrix of current conductances"""
        conductances = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                conductances[i, j] = self.devices[i][j]._state.conductance
        return conductances
    
    def get_temperature_matrix(self) -> np.ndarray:
        """Get matrix of current temperatures"""
        temperatures = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                temperatures[i, j] = self.devices[i][j]._state.temperature
        return temperatures
    
    def get_optical_transmission_matrix(self, wavelength: float) -> np.ndarray:
        """Get matrix of optical transmissions at given wavelength"""
        transmissions = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                transmissions[i, j] = self.devices[i][j].get_optical_transmission(wavelength)
        return transmissions
    
    def calculate_total_power(self, voltages: np.ndarray) -> float:
        """Calculate total electrical power consumption"""
        conductance_matrix = self.get_conductance_matrix()
        power_matrix = voltages**2 * conductance_matrix
        return np.sum(power_matrix)
    
    def reset_array(self):
        """Reset entire array to initial state"""
        for row in self.devices:
            for device in row:
                device.reset()
    
    def get_array_statistics(self) -> Dict:
        """Get statistical summary of array state"""
        conductances = self.get_conductance_matrix()
        temperatures = self.get_temperature_matrix()
        
        states = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                states[i, j] = self.devices[i][j]._state.internal_state
        
        return {
            "conductance": {
                "mean": float(np.mean(conductances)),
                "std": float(np.std(conductances)),
                "min": float(np.min(conductances)),
                "max": float(np.max(conductances))
            },
            "temperature": {
                "mean": float(np.mean(temperatures)),
                "std": float(np.std(temperatures)),
                "min": float(np.min(temperatures)),
                "max": float(np.max(temperatures))
            },
            "internal_state": {
                "mean": float(np.mean(states)),
                "std": float(np.std(states)),
                "min": float(np.min(states)),
                "max": float(np.max(states))
            }
        }


def create_standard_memristor_configs() -> Dict[str, MemristorConfig]:
    """Create standard configurations for different memristor types"""
    return {
        "GST_standard": MemristorConfig(
            material_type="GST",
            dimensions=(100e-9, 50e-9, 10e-9),
            ambient_temperature=300.0,
            thermal_time_constant=1e-6,
            temperature_coefficient=0.01
        ),
        "HfO2_standard": MemristorConfig(
            material_type="HfO2", 
            dimensions=(50e-9, 50e-9, 5e-9),
            ambient_temperature=300.0,
            thermal_time_constant=1e-7,
            temperature_coefficient=0.02
        ),
        "TiO2_standard": MemristorConfig(
            material_type="TiO2",
            dimensions=(75e-9, 75e-9, 8e-9),
            ambient_temperature=300.0,
            thermal_time_constant=5e-7,
            temperature_coefficient=0.015
        )
    }


# Example usage and test functions
def run_basic_memristor_test():
    """Run basic test of advanced memristor functionality"""
    print("Testing Advanced Memristor Device...")
    
    # Create GST memristor
    config = MemristorConfig(material_type="GST")
    device = AdvancedMemristorDevice(config)
    
    print(f"Initial state: {device.get_current_state()}")
    
    # Apply switching pulse
    voltage = 3.0  # 3V
    optical_power = 1e-3  # 1mW
    time_step = 1e-6  # 1 microsecond
    
    for i in range(10):
        state = device.update_state(voltage, optical_power, time_step)
        print(f"Step {i+1}: Conductance={state.conductance:.2e} S, "
              f"Temperature={state.temperature:.1f} K, "
              f"Internal State={state.internal_state:.3f}")
    
    # Test optical properties
    transmission = device.get_optical_transmission(1550e-9)  # 1550nm
    print(f"Optical transmission at 1550nm: {transmission:.3f}")
    
    # Test switching predictions
    switching_time = device.predict_switching_time(voltage, 0.9)
    switching_energy = device.calculate_switching_energy(voltage, switching_time)
    print(f"Predicted switching time: {switching_time*1e9:.1f} ns")
    print(f"Switching energy: {switching_energy*1e12:.1f} pJ")


def run_array_test():
    """Run test of memristor array functionality"""
    print("\nTesting Memristor Array...")
    
    # Create small array
    config = MemristorConfig(material_type="HfO2")
    array = MemristorArray(rows=4, cols=4, config=config)
    
    # Create voltage and optical power patterns
    voltages = np.random.uniform(0, 2, (4, 4))  # 0-2V
    optical_powers = np.random.uniform(0, 1e-3, (4, 4))  # 0-1mW
    time_step = 1e-6
    
    # Update array
    states = array.update_array(voltages, optical_powers, time_step)
    
    # Get array statistics
    stats = array.get_array_statistics()
    print(f"Array statistics: {json.dumps(stats, indent=2)}")
    
    # Calculate total power
    total_power = array.calculate_total_power(voltages)
    print(f"Total power consumption: {total_power*1e3:.2f} mW")


if __name__ == "__main__":
    run_basic_memristor_test()
    run_array_test()