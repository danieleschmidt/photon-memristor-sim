//! Advanced Memristor Models for Neuromorphic Photonics
//! Generation 1: Simple Implementation of Multi-Physics Memristor Dynamics

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Advanced memristor model incorporating thermal, optical, and electrical effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPhysicsMemristor {
    /// Physical dimensions [length, width, height] in meters
    pub dimensions: [f64; 3],
    /// Current state variables
    pub state: MemristorState,
    /// Material properties
    pub material: MaterialProperties,
    /// Temperature-dependent parameters
    pub thermal_model: ThermalModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemristorState {
    /// Conductance in Siemens
    pub conductance: f64,
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Optical absorption coefficient
    pub absorption: f64,
    /// Internal state variable (0 to 1)
    pub internal_state: f64,
    /// Time since last update
    pub last_update_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    /// Material type identifier
    pub material_type: String,
    /// Thermal conductivity W/(m·K)
    pub thermal_conductivity: f64,
    /// Electrical conductivity S/m
    pub base_conductivity: f64,
    /// Optical refractive index real part
    pub refractive_index_real: f64,
    /// Optical refractive index imaginary part  
    pub refractive_index_imag: f64,
    /// Activation energy for switching (eV)
    pub activation_energy: f64,
    /// Ion mobility m²/(V·s)
    pub ion_mobility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalModel {
    /// Ambient temperature in Kelvin
    pub ambient_temperature: f64,
    /// Thermal time constant in seconds
    pub thermal_time_constant: f64,
    /// Temperature coefficient of conductance
    pub temperature_coefficient: f64,
}

impl MultiPhysicsMemristor {
    /// Create a new multi-physics memristor with specified material
    pub fn new(material_type: &str, dimensions: [f64; 3]) -> Result<Self, Box<dyn std::error::Error>> {
        let material = Self::create_material_properties(material_type)?;
        let state = MemristorState {
            conductance: material.base_conductivity * dimensions[1] * dimensions[2] / dimensions[0],
            temperature: 300.0, // Room temperature
            absorption: 0.01,
            internal_state: 0.5,
            last_update_time: 0.0,
        };
        
        let thermal_model = ThermalModel {
            ambient_temperature: 300.0,
            thermal_time_constant: 1e-6, // 1 microsecond
            temperature_coefficient: 0.01, // 1% per Kelvin
        };

        Ok(MultiPhysicsMemristor {
            dimensions,
            state,
            material,
            thermal_model,
        })
    }

    /// Create material properties based on type
    fn create_material_properties(material_type: &str) -> Result<MaterialProperties, Box<dyn std::error::Error>> {
        match material_type {
            "GST" => Ok(MaterialProperties {
                material_type: "GST".to_string(),
                thermal_conductivity: 0.5,
                base_conductivity: 1e4,
                refractive_index_real: 6.5,
                refractive_index_imag: 0.5,
                activation_energy: 0.5,
                ion_mobility: 1e-14,
            }),
            "HfO2" => Ok(MaterialProperties {
                material_type: "HfO2".to_string(),
                thermal_conductivity: 23.0,
                base_conductivity: 1e-8,
                refractive_index_real: 2.1,
                refractive_index_imag: 0.0,
                activation_energy: 0.6,
                ion_mobility: 1e-15,
            }),
            "TiO2" => Ok(MaterialProperties {
                material_type: "TiO2".to_string(),
                thermal_conductivity: 8.4,
                base_conductivity: 1e-10,
                refractive_index_real: 2.5,
                refractive_index_imag: 0.0,
                activation_energy: 0.8,
                ion_mobility: 1e-16,
            }),
            _ => Err(format!("Unsupported material type: {}", material_type).into()),
        }
    }

    /// Update memristor state with applied voltage and optical power
    pub fn update_state(
        &mut self,
        voltage: f64,
        optical_power: f64,
        time_step: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Calculate Joule heating
        let electrical_power = voltage * voltage * self.state.conductance;
        
        // Calculate optical heating
        let optical_heating = optical_power * self.state.absorption;
        
        // Update temperature with thermal dynamics
        let total_heating = electrical_power + optical_heating;
        let temperature_change = total_heating * time_step / 
            (self.get_thermal_mass() * self.thermal_model.thermal_time_constant);
        
        // Thermal relaxation
        let temp_diff = self.state.temperature - self.thermal_model.ambient_temperature;
        let thermal_decay = temp_diff * time_step / self.thermal_model.thermal_time_constant;
        
        self.state.temperature += temperature_change - thermal_decay;
        
        // Update internal state based on voltage and temperature
        let field_strength = voltage / self.dimensions[0]; // V/m
        let temperature_factor = (-self.material.activation_energy * 1.602e-19 / 
                                 (1.381e-23 * self.state.temperature)).exp();
        
        let drift_velocity = self.material.ion_mobility * field_strength * temperature_factor;
        let state_change = drift_velocity * time_step / self.dimensions[0];
        
        self.state.internal_state = (self.state.internal_state + state_change).clamp(0.0, 1.0);
        
        // Update conductance based on internal state
        let conductance_ratio = 1000.0; // High/low conductance ratio
        self.state.conductance = self.material.base_conductivity * 
            (1.0 + (conductance_ratio - 1.0) * self.state.internal_state) *
            (1.0 + self.thermal_model.temperature_coefficient * 
             (self.state.temperature - self.thermal_model.ambient_temperature));
        
        // Update optical properties
        self.update_optical_properties();
        
        self.state.last_update_time += time_step;
        
        Ok(())
    }

    /// Calculate thermal mass of the device
    fn get_thermal_mass(&self) -> f64 {
        let volume = self.dimensions[0] * self.dimensions[1] * self.dimensions[2];
        let density = 6150.0; // kg/m³ (typical for GST)
        let heat_capacity = 230.0; // J/(kg·K)
        volume * density * heat_capacity
    }

    /// Update optical properties based on current state
    fn update_optical_properties(&mut self) {
        match self.material.material_type.as_str() {
            "GST" => {
                // GST phase change affects refractive index dramatically
                let crystalline_n = Complex64::new(6.5, 0.5);
                let amorphous_n = Complex64::new(4.0, 0.1);
                
                let current_n = amorphous_n + (crystalline_n - amorphous_n) * self.state.internal_state;
                
                // Absorption scales with imaginary part
                self.state.absorption = current_n.im * 0.1;
            },
            "HfO2" | "TiO2" => {
                // Oxide memristors have weaker optical effects
                let base_absorption = 0.01;
                self.state.absorption = base_absorption * (1.0 + self.state.internal_state * 0.5);
            },
            _ => {}
        }
    }

    /// Get current optical transmission
    pub fn get_optical_transmission(&self, wavelength: f64) -> f64 {
        let path_length = self.dimensions[2]; // Thickness
        let absorption_coeff = 4.0 * std::f64::consts::PI * 
                             self.material.refractive_index_imag / wavelength;
        (-absorption_coeff * path_length).exp()
    }

    /// Get current electrical conductance
    pub fn get_conductance(&self) -> f64 {
        self.state.conductance
    }

    /// Get current temperature
    pub fn get_temperature(&self) -> f64 {
        self.state.temperature
    }

    /// Get current internal state (0 = low resistance, 1 = high resistance)
    pub fn get_internal_state(&self) -> f64 {
        self.state.internal_state
    }

    /// Reset device to initial state
    pub fn reset(&mut self) {
        self.state.internal_state = 0.5;
        self.state.temperature = self.thermal_model.ambient_temperature;
        self.state.conductance = self.material.base_conductivity;
        self.update_optical_properties();
    }

    /// Calculate switching energy for a given voltage pulse
    pub fn calculate_switching_energy(&self, voltage: f64, pulse_duration: f64) -> f64 {
        let power = voltage * voltage * self.state.conductance;
        power * pulse_duration
    }

    /// Predict switching time for given voltage
    pub fn predict_switching_time(&self, voltage: f64, target_state: f64) -> f64 {
        let field_strength = voltage / self.dimensions[0];
        let temperature_factor = (-self.material.activation_energy * 1.602e-19 / 
                                 (1.381e-23 * self.state.temperature)).exp();
        
        let drift_velocity = self.material.ion_mobility * field_strength * temperature_factor;
        let state_change_needed = (target_state - self.state.internal_state).abs();
        let distance_to_travel = state_change_needed * self.dimensions[0];
        
        if drift_velocity > 0.0 {
            distance_to_travel / drift_velocity
        } else {
            f64::INFINITY
        }
    }
}

/// Array of multi-physics memristors for large-scale simulation
#[derive(Debug, Clone)]
pub struct MemristorArray {
    pub memristors: Vec<Vec<MultiPhysicsMemristor>>,
    pub rows: usize,
    pub cols: usize,
    pub crosstalk_matrix: Vec<Vec<f64>>,
}

impl MemristorArray {
    /// Create a new memristor array
    pub fn new(rows: usize, cols: usize, material_type: &str, 
               dimensions: [f64; 3]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut memristors = Vec::with_capacity(rows);
        
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                row.push(MultiPhysicsMemristor::new(material_type, dimensions)?);
            }
            memristors.push(row);
        }
        
        // Initialize simple crosstalk matrix (can be made more sophisticated)
        let total_devices = rows * cols;
        let crosstalk_matrix = vec![vec![0.01; total_devices]; total_devices]; // 1% crosstalk
        
        Ok(MemristorArray {
            memristors,
            rows,
            cols,
            crosstalk_matrix,
        })
    }

    /// Update entire array with voltage matrix and optical power matrix
    pub fn update_array(
        &mut self,
        voltages: &[Vec<f64>],
        optical_powers: &[Vec<f64>],
        time_step: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Calculate crosstalk effects
        let mut voltage_with_crosstalk = voltages.to_vec();
        self.apply_crosstalk(&mut voltage_with_crosstalk);
        
        // Update each memristor
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.memristors[i][j].update_state(
                    voltage_with_crosstalk[i][j],
                    optical_powers[i][j],
                    time_step,
                )?;
            }
        }
        
        Ok(())
    }

    /// Apply crosstalk effects to voltage matrix
    fn apply_crosstalk(&self, voltages: &mut [Vec<f64>]) {
        // Simple nearest-neighbor crosstalk model
        let original_voltages = voltages.to_vec();
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut crosstalk_sum = 0.0;
                let crosstalk_coefficient = 0.01; // 1%
                
                // Add contributions from neighbors
                for di in -1i32..=1i32 {
                    for dj in -1i32..=1i32 {
                        if di == 0 && dj == 0 { continue; }
                        
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;
                        
                        if ni >= 0 && ni < self.rows as i32 && 
                           nj >= 0 && nj < self.cols as i32 {
                            crosstalk_sum += original_voltages[ni as usize][nj as usize] * 
                                           crosstalk_coefficient;
                        }
                    }
                }
                
                voltages[i][j] += crosstalk_sum;
            }
        }
    }

    /// Get conductance matrix
    pub fn get_conductance_matrix(&self) -> Vec<Vec<f64>> {
        self.memristors.iter().map(|row| 
            row.iter().map(|m| m.get_conductance()).collect()
        ).collect()
    }

    /// Get temperature matrix  
    pub fn get_temperature_matrix(&self) -> Vec<Vec<f64>> {
        self.memristors.iter().map(|row|
            row.iter().map(|m| m.get_temperature()).collect()
        ).collect()
    }

    /// Get optical transmission matrix
    pub fn get_optical_transmission_matrix(&self, wavelength: f64) -> Vec<Vec<f64>> {
        self.memristors.iter().map(|row|
            row.iter().map(|m| m.get_optical_transmission(wavelength)).collect()
        ).collect()
    }

    /// Calculate total power consumption
    pub fn calculate_total_power(&self, voltages: &[Vec<f64>]) -> f64 {
        let mut total_power = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let conductance = self.memristors[i][j].get_conductance();
                total_power += voltages[i][j] * voltages[i][j] * conductance;
            }
        }
        total_power
    }

    /// Reset entire array
    pub fn reset_array(&mut self) {
        for row in &mut self.memristors {
            for memristor in row {
                memristor.reset();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_memristor_creation() {
        let memristor = MultiPhysicsMemristor::new("GST", [100e-9, 50e-9, 10e-9]).unwrap();
        assert_eq!(memristor.material.material_type, "GST");
        assert_relative_eq!(memristor.state.temperature, 300.0, epsilon = 1e-6);
    }

    #[test]
    fn test_memristor_switching() {
        let mut memristor = MultiPhysicsMemristor::new("HfO2", [50e-9, 50e-9, 5e-9]).unwrap();
        let initial_state = memristor.get_internal_state();
        
        // Apply switching pulse
        memristor.update_state(3.0, 0.0, 1e-6).unwrap();
        
        assert_ne!(memristor.get_internal_state(), initial_state);
        assert!(memristor.get_temperature() > 300.0); // Should heat up
    }

    #[test]
    fn test_array_creation() {
        let array = MemristorArray::new(4, 4, "GST", [100e-9, 50e-9, 10e-9]).unwrap();
        assert_eq!(array.rows, 4);
        assert_eq!(array.cols, 4);
        assert_eq!(array.memristors.len(), 4);
        assert_eq!(array.memristors[0].len(), 4);
    }

    #[test]
    fn test_optical_transmission() {
        let memristor = MultiPhysicsMemristor::new("GST", [100e-9, 50e-9, 10e-9]).unwrap();
        let transmission = memristor.get_optical_transmission(1550e-9);
        assert!(transmission > 0.0 && transmission <= 1.0);
    }
}