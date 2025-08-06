//! Input validation and constraint checking for photonic simulation parameters

use crate::core::{Result, PhotonicError, OpticalField, WaveguideGeometry};
use nalgebra::DVector;
use std::collections::HashMap;

/// Physical constants and limits for validation
pub struct PhysicalLimits {
    /// Speed of light in vacuum (m/s)
    pub speed_of_light: f64,
    /// Planck constant (J⋅s)
    pub planck_constant: f64,
    /// Minimum wavelength for photonic simulation (m)
    pub min_wavelength: f64,
    /// Maximum wavelength for photonic simulation (m)
    pub max_wavelength: f64,
    /// Maximum optical power (W)
    pub max_optical_power: f64,
    /// Maximum temperature for devices (K)
    pub max_device_temperature: f64,
}

impl Default for PhysicalLimits {
    fn default() -> Self {
        Self {
            speed_of_light: 299792458.0,
            planck_constant: 6.62607015e-34,
            min_wavelength: 200e-9,    // 200nm (UV)
            max_wavelength: 20e-6,     // 20μm (mid-IR)
            max_optical_power: 10.0,   // 10W
            max_device_temperature: 1000.0, // 1000K
        }
    }
}

/// Comprehensive validation system for photonic parameters
pub struct PhotonicValidator {
    limits: PhysicalLimits,
    custom_constraints: HashMap<String, (f64, f64)>,
}

impl PhotonicValidator {
    /// Create new validator with default physical limits
    pub fn new() -> Self {
        Self {
            limits: PhysicalLimits::default(),
            custom_constraints: HashMap::new(),
        }
    }
    
    /// Create validator with custom physical limits
    pub fn with_limits(limits: PhysicalLimits) -> Self {
        Self {
            limits,
            custom_constraints: HashMap::new(),
        }
    }
    
    /// Add custom constraint for parameter validation
    pub fn add_constraint(&mut self, name: String, min_val: f64, max_val: f64) -> Result<()> {
        if min_val >= max_val {
            return Err(PhotonicError::invalid_parameter(
                "constraint_bounds",
                format!("min={}, max={}", min_val, max_val),
                "min < max"
            ));
        }
        
        self.custom_constraints.insert(name, (min_val, max_val));
        Ok(())
    }
    
    /// Validate optical field parameters
    pub fn validate_optical_field(&self, field: &OpticalField) -> Result<ValidationReport> {
        let mut report = ValidationReport::new("optical_field");
        
        // Wavelength validation
        if field.wavelength < self.limits.min_wavelength {
            report.add_error(format!(
                "Wavelength {:.2e}m below minimum {:.2e}m", 
                field.wavelength, self.limits.min_wavelength
            ));
        }
        
        if field.wavelength > self.limits.max_wavelength {
            report.add_error(format!(
                "Wavelength {:.2e}m above maximum {:.2e}m",
                field.wavelength, self.limits.max_wavelength
            ));
        }
        
        // Power validation
        if field.power < 0.0 {
            report.add_error("Optical power cannot be negative".to_string());
        }
        
        if field.power > self.limits.max_optical_power {
            report.add_warning(format!(
                "Power {:.3}W exceeds typical maximum {:.3}W",
                field.power, self.limits.max_optical_power
            ));
        }
        
        // Field amplitude validation
        if field.amplitude.nrows() == 0 || field.amplitude.ncols() == 0 {
            report.add_error("Field amplitude matrix cannot be empty".to_string());
        }
        
        // Check for NaN or infinite values
        let mut has_invalid_amplitude = false;
        for element in field.amplitude.iter() {
            if !element.re.is_finite() || !element.im.is_finite() {
                has_invalid_amplitude = true;
                break;
            }
        }
        
        if has_invalid_amplitude {
            report.add_error("Field amplitude contains NaN or infinite values".to_string());
        }
        
        // Spatial coordinate validation
        if field.coordinates.0.len() != field.amplitude.ncols() {
            report.add_error(format!(
                "X coordinate count {} doesn't match amplitude columns {}",
                field.coordinates.0.len(), field.amplitude.ncols()
            ));
        }
        
        if field.coordinates.1.len() != field.amplitude.nrows() {
            report.add_error(format!(
                "Y coordinate count {} doesn't match amplitude rows {}",
                field.coordinates.1.len(), field.amplitude.nrows()
            ));
        }
        
        // Check coordinate ordering
        if !self.is_monotonic(&field.coordinates.0) {
            report.add_warning("X coordinates are not monotonic".to_string());
        }
        
        if !self.is_monotonic(&field.coordinates.1) {
            report.add_warning("Y coordinates are not monotonic".to_string());
        }
        
        Ok(report)
    }
    
    /// Validate waveguide geometry parameters
    pub fn validate_waveguide(&self, geometry: &WaveguideGeometry) -> Result<ValidationReport> {
        let mut report = ValidationReport::new("waveguide_geometry");
        
        // Core dimensions
        if geometry.width <= 0.0 {
            report.add_error("Core width must be positive".to_string());
        }
        
        if geometry.height <= 0.0 {
            report.add_error("Core height must be positive".to_string());
        }
        
        // Typical size validation (warnings for unusual values)
        if geometry.width < 100e-9 {
            report.add_warning(format!(
                "Core width {:.1}nm is very small (< 100nm)",
                geometry.width * 1e9
            ));
        }
        
        if geometry.width > 100e-6 {
            report.add_warning(format!(
                "Core width {:.1}μm is very large (> 100μm)",
                geometry.width * 1e6
            ));
        }
        
        // Refractive index validation
        if geometry.core_index.re < 1.0 {
            report.add_error("Core refractive index real part must be ≥ 1.0".to_string());
        }
        
        if geometry.cladding_index.re < 1.0 {
            report.add_error("Cladding refractive index real part must be ≥ 1.0".to_string());
        }
        
        if geometry.core_index.re <= geometry.cladding_index.re {
            report.add_error("Core index must be higher than cladding index for guidance".to_string());
        }
        
        // Loss validation (imaginary part)
        if geometry.core_index.im > 0.1 {
            report.add_warning(format!(
                "Core material loss {:.3} is very high",
                geometry.core_index.im
            ));
        }
        
        if geometry.cladding_index.im > 0.1 {
            report.add_warning(format!(
                "Cladding material loss {:.3} is very high",
                geometry.cladding_index.im
            ));
        }
        
        Ok(report)
    }
    
    /// Validate device parameters against bounds
    pub fn validate_parameters(&self, name: &str, params: &DVector<f64>, 
                              bounds: &[(f64, f64)]) -> Result<ValidationReport> {
        let mut report = ValidationReport::new(&format!("{}_parameters", name));
        
        if params.len() != bounds.len() {
            report.add_error(format!(
                "Parameter count {} doesn't match bounds count {}",
                params.len(), bounds.len()
            ));
            return Ok(report);
        }
        
        for (i, (&param, &(min_val, max_val))) in params.iter().zip(bounds.iter()).enumerate() {
            // Check for valid floating-point value
            if !param.is_finite() {
                report.add_error(format!(
                    "Parameter {} has invalid value: {}",
                    i, param
                ));
                continue;
            }
            
            // Check bounds
            if param < min_val {
                report.add_error(format!(
                    "Parameter {} value {:.6e} below minimum {:.6e}",
                    i, param, min_val
                ));
            }
            
            if param > max_val {
                report.add_error(format!(
                    "Parameter {} value {:.6e} above maximum {:.6e}",
                    i, param, max_val
                ));
            }
            
            // Check if parameter is near bounds (warning)
            let range = max_val - min_val;
            if param - min_val < 0.05 * range {
                report.add_warning(format!(
                    "Parameter {} is close to lower bound",
                    i
                ));
            }
            
            if max_val - param < 0.05 * range {
                report.add_warning(format!(
                    "Parameter {} is close to upper bound",
                    i
                ));
            }
        }
        
        Ok(report)
    }
    
    /// Validate custom constraint
    pub fn validate_custom(&self, name: &str, value: f64) -> Result<ValidationReport> {
        let mut report = ValidationReport::new(&format!("custom_{}", name));
        
        if let Some(&(min_val, max_val)) = self.custom_constraints.get(name) {
            if !value.is_finite() {
                report.add_error(format!("Value {} is not finite", value));
            } else if value < min_val {
                report.add_error(format!(
                    "Value {:.6e} below minimum {:.6e}",
                    value, min_val
                ));
            } else if value > max_val {
                report.add_error(format!(
                    "Value {:.6e} above maximum {:.6e}",
                    value, max_val
                ));
            }
        } else {
            report.add_warning(format!("No constraint defined for '{}'", name));
        }
        
        Ok(report)
    }
    
    /// Check if vector is monotonically increasing
    fn is_monotonic(&self, vec: &DVector<f64>) -> bool {
        for i in 1..vec.len() {
            if vec[i] <= vec[i-1] {
                return false;
            }
        }
        true
    }
}

/// Validation report containing errors and warnings
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub context: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub is_valid: bool,
}

impl ValidationReport {
    /// Create new validation report
    pub fn new(context: &str) -> Self {
        Self {
            context: context.to_string(),
            errors: Vec::new(),
            warnings: Vec::new(),
            is_valid: true,
        }
    }
    
    /// Add error to report
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }
    
    /// Add warning to report
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }
    
    /// Get formatted summary
    pub fn summary(&self) -> String {
        let mut summary = format!("Validation Report: {}\n", self.context);
        
        if self.is_valid {
            summary.push_str("✅ PASSED");
        } else {
            summary.push_str("❌ FAILED");
        }
        
        if !self.errors.is_empty() {
            summary.push_str(&format!("\n\nErrors ({}):", self.errors.len()));
            for (i, error) in self.errors.iter().enumerate() {
                summary.push_str(&format!("\n  {}. {}", i + 1, error));
            }
        }
        
        if !self.warnings.is_empty() {
            summary.push_str(&format!("\n\nWarnings ({}):", self.warnings.len()));
            for (i, warning) in self.warnings.iter().enumerate() {
                summary.push_str(&format!("\n  {}. {}", i + 1, warning));
            }
        }
        
        summary
    }
    
    /// Merge another report into this one
    pub fn merge(&mut self, other: ValidationReport) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        if !other.is_valid {
            self.is_valid = false;
        }
    }
}

/// Batch validator for multiple objects
pub struct BatchValidator {
    validator: PhotonicValidator,
    reports: Vec<ValidationReport>,
}

impl BatchValidator {
    /// Create new batch validator
    pub fn new() -> Self {
        Self {
            validator: PhotonicValidator::new(),
            reports: Vec::new(),
        }
    }
    
    /// Validate multiple optical fields
    pub fn validate_fields(&mut self, fields: &[OpticalField]) -> Result<()> {
        for (i, field) in fields.iter().enumerate() {
            let mut report = self.validator.validate_optical_field(field)?;
            report.context = format!("field_{}", i);
            self.reports.push(report);
        }
        Ok(())
    }
    
    /// Get combined validation report
    pub fn combined_report(&self) -> ValidationReport {
        let mut combined = ValidationReport::new("batch_validation");
        
        for report in &self.reports {
            combined.merge(report.clone());
        }
        
        combined
    }
    
    /// Get number of failed validations
    pub fn failure_count(&self) -> usize {
        self.reports.iter().filter(|r| !r.is_valid()).count()
    }
}

/// Validation middleware for function calls
pub trait Validatable {
    /// Validate object before processing
    fn validate(&self) -> Result<ValidationReport>;
}

impl Validatable for OpticalField {
    fn validate(&self) -> Result<ValidationReport> {
        let validator = PhotonicValidator::new();
        validator.validate_optical_field(self)
    }
}

impl Validatable for WaveguideGeometry {
    fn validate(&self) -> Result<ValidationReport> {
        let validator = PhotonicValidator::new();
        validator.validate_waveguide(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use crate::core::Complex64;
    
    fn create_test_field() -> OpticalField {
        let amplitude = DMatrix::from_element(10, 10, Complex64::new(1.0, 0.0));
        let x_coords = DVector::from_iterator(10, (0..10).map(|i| i as f64 * 1e-6));
        let y_coords = DVector::from_iterator(10, (0..10).map(|i| i as f64 * 1e-6));
        
        OpticalField::new(amplitude, 1550e-9, 1e-3, x_coords, y_coords)
    }
    
    fn create_test_waveguide() -> WaveguideGeometry {
        WaveguideGeometry {
            core_width: 450e-9,
            core_height: 220e-9,
            core_index: Complex64::new(3.48, 0.0),
            cladding_index: Complex64::new(1.44, 0.0),
        }
    }
    
    #[test]
    fn test_validator_creation() {
        let validator = PhotonicValidator::new();
        assert_eq!(validator.limits.speed_of_light, 299792458.0);
    }
    
    #[test]
    fn test_valid_optical_field() {
        let validator = PhotonicValidator::new();
        let field = create_test_field();
        
        let report = validator.validate_optical_field(&field).unwrap();
        assert!(report.is_valid());
        assert!(report.errors.is_empty());
    }
    
    #[test]
    fn test_invalid_wavelength() {
        let validator = PhotonicValidator::new();
        let amplitude = DMatrix::from_element(5, 5, Complex64::new(1.0, 0.0));
        let x_coords = DVector::linspace(0.0, 5e-6, 5);
        let y_coords = DVector::linspace(0.0, 5e-6, 5);
        
        // Invalid wavelength (too small)
        let field = OpticalField::new(amplitude, 50e-9, 1e-3, x_coords, y_coords);
        let report = validator.validate_optical_field(&field).unwrap();
        
        assert!(!report.is_valid());
        assert!(!report.errors.is_empty());
    }
    
    #[test]
    fn test_valid_waveguide() {
        let validator = PhotonicValidator::new();
        let waveguide = create_test_waveguide();
        
        let report = validator.validate_waveguide(&waveguide).unwrap();
        assert!(report.is_valid());
    }
    
    #[test]
    fn test_invalid_waveguide_index() {
        let validator = PhotonicValidator::new();
        let mut waveguide = create_test_waveguide();
        
        // Core index lower than cladding
        waveguide.core_index = Complex64::new(1.0, 0.0);
        
        let report = validator.validate_waveguide(&waveguide).unwrap();
        assert!(!report.is_valid());
    }
    
    #[test]
    fn test_parameter_validation() {
        let validator = PhotonicValidator::new();
        let params = DVector::from_vec(vec![0.5, 1.5, 2.5]);
        let bounds = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
        
        let report = validator.validate_parameters("test", &params, &bounds).unwrap();
        assert!(report.is_valid());
    }
    
    #[test]
    fn test_parameter_bounds_violation() {
        let validator = PhotonicValidator::new();
        let params = DVector::from_vec(vec![1.5, 0.5]); // Second param below bounds
        let bounds = vec![(0.0, 1.0), (1.0, 2.0)];
        
        let report = validator.validate_parameters("test", &params, &bounds).unwrap();
        assert!(!report.is_valid());
        assert!(report.errors.len() == 1);
    }
    
    #[test]
    fn test_custom_constraints() {
        let mut validator = PhotonicValidator::new();
        validator.add_constraint("power".to_string(), 0.0, 10.0).unwrap();
        
        let report = validator.validate_custom("power", 5.0).unwrap();
        assert!(report.is_valid());
        
        let report = validator.validate_custom("power", 15.0).unwrap();
        assert!(!report.is_valid());
    }
    
    #[test]
    fn test_batch_validation() {
        let mut batch_validator = BatchValidator::new();
        let fields = vec![create_test_field(), create_test_field()];
        
        batch_validator.validate_fields(&fields).unwrap();
        let combined = batch_validator.combined_report();
        
        assert!(combined.is_valid());
        assert_eq!(batch_validator.failure_count(), 0);
    }
    
    #[test]
    fn test_validation_trait() {
        let field = create_test_field();
        let report = field.validate().unwrap();
        assert!(report.is_valid());
        
        let waveguide = create_test_waveguide();
        let report = waveguide.validate().unwrap();
        assert!(report.is_valid());
    }
}