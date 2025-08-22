//! Common types and data structures for photonic simulation

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64 as NumComplex64;


/// Complex number type for optical calculations
pub type Complex64 = NumComplex64;

/// 2D complex matrix for optical field representations
pub type ComplexMatrix = DMatrix<Complex64>;

/// 1D complex vector for optical field profiles
pub type ComplexVector = DVector<Complex64>;

/// Optical field structure containing amplitude and phase information
#[derive(Debug, Clone)]
pub struct OpticalField {
    /// Complex amplitude distribution
    pub amplitude: ComplexMatrix,
    /// Wavelength in meters
    pub wavelength: f64,
    /// Power in watts
    pub power: f64,
    /// Spatial coordinates (x, y) in meters
    pub coordinates: (DVector<f64>, DVector<f64>),
}

impl OpticalField {
    /// Create a new optical field
    pub fn new(
        amplitude: ComplexMatrix,
        wavelength: f64,
        power: f64,
        x_coords: DVector<f64>,
        y_coords: DVector<f64>,
    ) -> Self {
        Self {
            amplitude,
            wavelength,
            power,
            coordinates: (x_coords, y_coords),
        }
    }
    
    /// Calculate total power by integrating intensity
    pub fn calculate_power(&self) -> f64 {
        let intensity: f64 = self.amplitude
            .iter()
            .map(|c| c.norm_sqr())
            .sum();
        
        let dx = if self.coordinates.0.len() > 1 {
            self.coordinates.0[1] - self.coordinates.0[0]
        } else { 1.0 };
        let dy = if self.coordinates.1.len() > 1 {
            self.coordinates.1[1] - self.coordinates.1[0]
        } else { 1.0 };
        
        intensity * dx * dy
    }
    
    /// Normalize field to specified power
    pub fn normalize_to_power(&mut self, target_power: f64) {
        let current_power = self.calculate_power();
        if current_power > 0.0 {
            let scale_factor = (target_power / current_power).sqrt();
            self.amplitude *= Complex64::new(scale_factor, 0.0);
            self.power = target_power;
        }
    }
    
    /// Get field dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.amplitude.nrows(), self.amplitude.ncols())
    }
}

/// Waveguide geometry parameters
#[derive(Debug, Clone)]
pub struct WaveguideGeometry {
    /// Core width in meters
    pub width: f64,
    /// Core height in meters
    pub height: f64,
    /// Core refractive index
    pub core_index: Complex64,
    /// Cladding refractive index
    pub cladding_index: Complex64,
    /// Substrate refractive index
    pub substrate_index: Complex64,
    /// Sidewall angle in radians
    pub sidewall_angle: f64,
}

impl WaveguideGeometry {
    /// Create a standard silicon waveguide at 1550nm
    pub fn silicon_photonic_standard() -> Self {
        Self {
            width: 450e-9,     // 450nm
            height: 220e-9,    // 220nm
            core_index: Complex64::new(3.47, 0.0),    // Silicon
            cladding_index: Complex64::new(1.44, 0.0), // SiO2
            substrate_index: Complex64::new(1.44, 0.0), // SiO2
            sidewall_angle: 80.0_f64.to_radians(),
        }
    }
    
    /// Calculate effective area for single mode
    pub fn effective_area(&self) -> f64 {
        // Simplified calculation - actual implementation would use mode solver
        std::f64::consts::PI * self.width * self.height / 4.0
    }
}

/// Device geometry for photonic components
#[derive(Debug, Clone)]
pub struct DeviceGeometry {
    /// Length in meters
    pub length: f64,
    /// Width in meters
    pub width: f64,
    /// Height in meters
    pub height: f64,
    /// Position (x, y, z) in meters
    pub position: (f64, f64, f64),
    /// Rotation angles (rx, ry, rz) in radians
    pub rotation: (f64, f64, f64),
}

impl DeviceGeometry {
    /// Create new device geometry
    pub fn new(length: f64, width: f64, height: f64) -> Self {
        Self {
            length,
            width,
            height,
            position: (0.0, 0.0, 0.0),
            rotation: (0.0, 0.0, 0.0),
        }
    }
    
    /// Calculate volume
    pub fn volume(&self) -> f64 {
        self.length * self.width * self.height
    }
    
    /// Set position
    pub fn at_position(mut self, x: f64, y: f64, z: f64) -> Self {
        self.position = (x, y, z);
        self
    }
}

/// Material properties for optical devices
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    /// Refractive index (real part)
    pub refractive_index: f64,
    /// Extinction coefficient (imaginary part)
    pub extinction_coefficient: f64,
    /// Thermal conductivity (W/m/K)
    pub thermal_conductivity: f64,
    /// Specific heat (J/kg/K)
    pub specific_heat: f64,
    /// Density (kg/m³)
    pub density: f64,
    /// Thermo-optic coefficient (1/K)
    pub thermo_optic_coefficient: f64,
}

impl MaterialProperties {
    /// Silicon properties at 1550nm, 300K
    pub fn silicon() -> Self {
        Self {
            refractive_index: 3.47,
            extinction_coefficient: 0.0,
            thermal_conductivity: 130.0,
            specific_heat: 700.0,
            density: 2330.0,
            thermo_optic_coefficient: 1.8e-4,
        }
    }
    
    /// SiO2 properties at 1550nm, 300K
    pub fn silica() -> Self {
        Self {
            refractive_index: 1.44,
            extinction_coefficient: 0.0,
            thermal_conductivity: 1.4,
            specific_heat: 730.0,
            density: 2200.0,
            thermo_optic_coefficient: 1.0e-5,
        }
    }
    
    /// Get complex refractive index
    pub fn complex_index(&self) -> Complex64 {
        Complex64::new(self.refractive_index, self.extinction_coefficient)
    }
}

/// Simulation parameters
#[derive(Debug, Clone)]
pub struct SimulationParams {
    /// Wavelength range (start, end, num_points)
    pub wavelength_range: (f64, f64, usize),
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Include thermal effects
    pub include_thermal: bool,
    /// Include nonlinear effects
    pub include_nonlinear: bool,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            wavelength_range: (1520e-9, 1580e-9, 61), // C-band
            temperature: 300.0, // Room temperature
            include_thermal: true,
            include_nonlinear: false,
            tolerance: 1e-9,
            max_iterations: 1000,
        }
    }
}