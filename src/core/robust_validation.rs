//! Robust Validation and Error Handling for Photonic Memristor Systems
//! Generation 2: MAKE IT ROBUST - Comprehensive validation, error handling, and monitoring

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Comprehensive validation errors for photonic memristor systems
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    #[error("Physical parameter out of range: {parameter} = {value}, expected range [{min}, {max}]")]
    ParameterOutOfRange {
        parameter: String,
        value: f64,
        min: f64,
        max: f64,
    },
    
    #[error("Invalid material configuration: {reason}")]
    InvalidMaterial { reason: String },
    
    #[error("Thermal instability detected: temperature {temperature}K exceeds safe limit {limit}K")]
    ThermalInstability { temperature: f64, limit: f64 },
    
    #[error("Optical power exceeds damage threshold: {power}W > {threshold}W")]
    OpticalDamage { power: f64, threshold: f64 },
    
    #[error("Electrical stress violation: field {field}V/m > {threshold}V/m")]
    ElectricalStress { field: f64, threshold: f64 },
    
    #[error("Simulation convergence failed after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
    
    #[error("Array dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    
    #[error("Time step too large: dt={dt}s, stability limit={limit}s")]
    TimeStepTooLarge { dt: f64, limit: f64 },
    
    #[error("Manufacturing tolerance exceeded: variation {variation}% > {tolerance}%")]
    ManufacturingTolerance { variation: f64, tolerance: f64 },
    
    #[error("Security violation: {details}")]
    SecurityViolation { details: String },
}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Validation result with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub severity: Severity,
    pub message: String,
    pub parameter: Option<String>,
    pub suggested_value: Option<f64>,
    pub timestamp: String,
}

impl ValidationResult {
    pub fn new(is_valid: bool, severity: Severity, message: String) -> Self {
        Self {
            is_valid,
            severity,
            message,
            parameter: None,
            suggested_value: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
    
    pub fn with_parameter(mut self, parameter: String, suggested_value: Option<f64>) -> Self {
        self.parameter = Some(parameter);
        self.suggested_value = suggested_value;
        self
    }
}

/// Comprehensive validator for photonic memristor systems
#[derive(Debug, Clone)]
pub struct RobustValidator {
    /// Physical parameter limits
    pub physical_limits: HashMap<String, (f64, f64)>, // (min, max)
    /// Material-specific constraints
    pub material_constraints: HashMap<String, MaterialConstraints>,
    /// Safety thresholds
    pub safety_thresholds: SafetyThresholds,
    /// Validation history
    pub validation_history: Vec<ValidationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialConstraints {
    pub max_temperature: f64,
    pub max_field_strength: f64,
    pub max_optical_power: f64,
    pub thermal_time_constant_range: (f64, f64),
    pub conductivity_range: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyThresholds {
    pub thermal_runaway_temp: f64,
    pub optical_damage_threshold: f64,
    pub electrical_breakdown_field: f64,
    pub max_simulation_time: f64,
    pub max_array_size: usize,
}

impl Default for SafetyThresholds {
    fn default() -> Self {
        Self {
            thermal_runaway_temp: 1000.0, // K
            optical_damage_threshold: 100e-3, // W
            electrical_breakdown_field: 1e8, // V/m
            max_simulation_time: 3600.0, // s
            max_array_size: 10000,
        }
    }
}

impl RobustValidator {
    /// Create new validator with default constraints
    pub fn new() -> Self {
        let mut physical_limits = HashMap::new();
        
        // Temperature limits (K)
        physical_limits.insert("temperature".to_string(), (4.0, 2000.0));
        
        // Voltage limits (V)
        physical_limits.insert("voltage".to_string(), (-100.0, 100.0));
        
        // Optical power limits (W)
        physical_limits.insert("optical_power".to_string(), (0.0, 1.0));
        
        // Time step limits (s)
        physical_limits.insert("time_step".to_string(), (1e-12, 1e-3));
        
        // Dimension limits (m)
        physical_limits.insert("dimension".to_string(), (1e-12, 1e-3));
        
        // Conductance limits (S)
        physical_limits.insert("conductance".to_string(), (1e-15, 1e-3));
        
        // Internal state limits (dimensionless)
        physical_limits.insert("internal_state".to_string(), (0.0, 1.0));
        
        // Material constraints
        let mut material_constraints = HashMap::new();
        
        material_constraints.insert("GST".to_string(), MaterialConstraints {
            max_temperature: 900.0,
            max_field_strength: 1e7,
            max_optical_power: 50e-3,
            thermal_time_constant_range: (1e-9, 1e-3),
            conductivity_range: (1e-6, 1e4),
        });
        
        material_constraints.insert("HfO2".to_string(), MaterialConstraints {
            max_temperature: 1200.0,
            max_field_strength: 5e7,
            max_optical_power: 100e-3,
            thermal_time_constant_range: (1e-8, 1e-4),
            conductivity_range: (1e-12, 1e-3),
        });
        
        material_constraints.insert("TiO2".to_string(), MaterialConstraints {
            max_temperature: 1000.0,
            max_field_strength: 2e7,
            max_optical_power: 75e-3,
            thermal_time_constant_range: (1e-7, 1e-4),
            conductivity_range: (1e-15, 1e-6),
        });
        
        Self {
            physical_limits,
            material_constraints,
            safety_thresholds: SafetyThresholds::default(),
            validation_history: Vec::new(),
        }
    }
    
    /// Validate a physical parameter
    pub fn validate_parameter(&mut self, name: &str, value: f64) -> ValidationResult {
        if let Some(&(min, max)) = self.physical_limits.get(name) {
            if value < min || value > max {
                let result = ValidationResult::new(
                    false,
                    Severity::Error,
                    format!("Parameter {} = {} is outside valid range [{}, {}]", name, value, min, max),
                ).with_parameter(name.to_string(), Some((min + max) / 2.0));
                
                self.validation_history.push(result.clone());
                return result;
            }
        }
        
        let result = ValidationResult::new(
            true,
            Severity::Info,
            format!("Parameter {} = {} is valid", name, value),
        );
        
        self.validation_history.push(result.clone());
        result
    }
    
    /// Validate material-specific constraints
    pub fn validate_material(&mut self, material_type: &str, temperature: f64, field: f64, power: f64) -> Vec<ValidationResult> {
        let mut results = Vec::new();
        
        if let Some(constraints) = self.material_constraints.get(material_type) {
            // Temperature check
            if temperature > constraints.max_temperature {
                results.push(ValidationResult::new(
                    false,
                    Severity::Critical,
                    format!("Temperature {}K exceeds {} limit {}K", 
                           temperature, material_type, constraints.max_temperature),
                ).with_parameter("temperature".to_string(), Some(constraints.max_temperature * 0.8)));
            }
            
            // Electric field check
            if field > constraints.max_field_strength {
                results.push(ValidationResult::new(
                    false,
                    Severity::Error,
                    format!("Electric field {:.2e}V/m exceeds {} limit {:.2e}V/m", 
                           field, material_type, constraints.max_field_strength),
                ).with_parameter("field".to_string(), Some(constraints.max_field_strength * 0.8)));
            }
            
            // Optical power check
            if power > constraints.max_optical_power {
                results.push(ValidationResult::new(
                    false,
                    Severity::Warning,
                    format!("Optical power {}W exceeds {} safe limit {}W", 
                           power, material_type, constraints.max_optical_power),
                ).with_parameter("optical_power".to_string(), Some(constraints.max_optical_power * 0.8)));
            }
            
            if results.is_empty() {
                results.push(ValidationResult::new(
                    true,
                    Severity::Info,
                    format!("{} material constraints satisfied", material_type),
                ));
            }
        } else {
            results.push(ValidationResult::new(
                false,
                Severity::Warning,
                format!("Unknown material type: {}", material_type),
            ));
        }
        
        // Store in history
        for result in &results {
            self.validation_history.push(result.clone());
        }
        
        results
    }
    
    /// Validate array dimensions and configuration
    pub fn validate_array_config(&mut self, rows: usize, cols: usize, time_step: f64) -> Vec<ValidationResult> {
        let mut results = Vec::new();
        
        // Array size check
        let total_devices = rows * cols;
        if total_devices > self.safety_thresholds.max_array_size {
            results.push(ValidationResult::new(
                false,
                Severity::Error,
                format!("Array size {}x{}={} exceeds maximum {}", 
                       rows, cols, total_devices, self.safety_thresholds.max_array_size),
            ));
        }
        
        // Time step stability check
        let stability_limit = self.calculate_stability_limit(rows, cols);
        if time_step > stability_limit {
            results.push(ValidationResult::new(
                false,
                Severity::Critical,
                format!("Time step {}s exceeds stability limit {}s", time_step, stability_limit),
            ).with_parameter("time_step".to_string(), Some(stability_limit * 0.5)));
        }
        
        if results.is_empty() {
            results.push(ValidationResult::new(
                true,
                Severity::Info,
                format!("Array configuration {}x{} is valid", rows, cols),
            ));
        }
        
        // Store in history
        for result in &results {
            self.validation_history.push(result.clone());
        }
        
        results
    }
    
    /// Calculate numerical stability limit for time step
    fn calculate_stability_limit(&self, rows: usize, cols: usize) -> f64 {
        // Courant-Friedrichs-Lewy (CFL) condition for array simulation
        let min_dimension = 1e-9; // Minimum feature size (nm)
        let max_velocity = 3e8; // Speed of light (m/s)
        let safety_factor = 0.5;
        
        // Scale with array size for crosstalk effects
        let coupling_factor = 1.0 / (1.0 + (rows * cols) as f64 / 100.0).sqrt();
        
        min_dimension / max_velocity * safety_factor * coupling_factor
    }
    
    /// Validate simulation convergence
    pub fn validate_convergence(&mut self, residual: f64, iteration: usize, max_iterations: usize) -> ValidationResult {
        let tolerance = 1e-6;
        
        let result = if residual < tolerance {
            ValidationResult::new(
                true,
                Severity::Info,
                format!("Simulation converged after {} iterations (residual: {:.2e})", iteration, residual),
            )
        } else if iteration >= max_iterations {
            ValidationResult::new(
                false,
                Severity::Error,
                format!("Simulation failed to converge after {} iterations (residual: {:.2e})", iteration, residual),
            )
        } else {
            ValidationResult::new(
                true,
                Severity::Info,
                format!("Simulation progressing (iteration {}, residual: {:.2e})", iteration, residual),
            )
        };
        
        self.validation_history.push(result.clone());
        result
    }
    
    /// Validate input sanitization and security
    pub fn validate_security(&mut self, input_data: &[f64], source: &str) -> Vec<ValidationResult> {
        let mut results = Vec::new();
        
        // Check for NaN or infinite values
        for (i, &value) in input_data.iter().enumerate() {
            if value.is_nan() {
                results.push(ValidationResult::new(
                    false,
                    Severity::Critical,
                    format!("NaN value detected at index {} from {}", i, source),
                ));
            } else if value.is_infinite() {
                results.push(ValidationResult::new(
                    false,
                    Severity::Critical,
                    format!("Infinite value detected at index {} from {}", i, source),
                ));
            }
        }
        
        // Check for suspiciously large values (potential attack)
        let max_reasonable = 1e6;
        for (i, &value) in input_data.iter().enumerate() {
            if value.abs() > max_reasonable {
                results.push(ValidationResult::new(
                    false,
                    Severity::Warning,
                    format!("Unusually large value {} at index {} from {}", value, i, source),
                ));
            }
        }
        
        // Check for potential buffer overflow
        if input_data.len() > 1_000_000 {
            results.push(ValidationResult::new(
                false,
                Severity::Error,
                format!("Input data size {} exceeds safety limit from {}", input_data.len(), source),
            ));
        }
        
        if results.is_empty() {
            results.push(ValidationResult::new(
                true,
                Severity::Info,
                format!("Security validation passed for {} values from {}", input_data.len(), source),
            ));
        }
        
        // Store in history
        for result in &results {
            self.validation_history.push(result.clone());
        }
        
        results
    }
    
    /// Get validation summary
    pub fn get_validation_summary(&self) -> ValidationSummary {
        let total = self.validation_history.len();
        let critical = self.validation_history.iter().filter(|r| r.severity == Severity::Critical).count();
        let errors = self.validation_history.iter().filter(|r| r.severity == Severity::Error).count();
        let warnings = self.validation_history.iter().filter(|r| r.severity == Severity::Warning).count();
        let info = self.validation_history.iter().filter(|r| r.severity == Severity::Info).count();
        
        let overall_status = if critical > 0 {
            ValidationStatus::Critical
        } else if errors > 0 {
            ValidationStatus::HasErrors
        } else if warnings > 0 {
            ValidationStatus::HasWarnings
        } else {
            ValidationStatus::Healthy
        };
        
        ValidationSummary {
            total_validations: total,
            critical_issues: critical,
            errors,
            warnings,
            info_messages: info,
            overall_status,
            last_validation: self.validation_history.last().map(|r| r.timestamp.clone()),
        }
    }
    
    /// Clear validation history
    pub fn clear_history(&mut self) {
        self.validation_history.clear();
    }
    
    /// Export validation report
    pub fn export_report(&self) -> Result<String, Box<dyn std::error::Error>> {
        let summary = self.get_validation_summary();
        let report = ValidationReport {
            summary,
            detailed_history: self.validation_history.clone(),
        };
        
        Ok(serde_json::to_string_pretty(&report)?)
    }
}

/// Validation status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Healthy,
    HasWarnings,
    HasErrors,
    Critical,
}

impl fmt::Display for ValidationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationStatus::Healthy => write!(f, "✅ Healthy"),
            ValidationStatus::HasWarnings => write!(f, "⚠️ Warnings"),
            ValidationStatus::HasErrors => write!(f, "❌ Errors"),
            ValidationStatus::Critical => write!(f, "🚨 Critical"),
        }
    }
}

/// Validation summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_validations: usize,
    pub critical_issues: usize,
    pub errors: usize,
    pub warnings: usize,
    pub info_messages: usize,
    pub overall_status: ValidationStatus,
    pub last_validation: Option<String>,
}

/// Complete validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub summary: ValidationSummary,
    pub detailed_history: Vec<ValidationResult>,
}

/// Input sanitizer for secure parameter handling
#[derive(Debug, Clone)]
pub struct InputSanitizer {
    max_string_length: usize,
    max_array_size: usize,
    allowed_characters: regex::Regex,
}

impl Default for InputSanitizer {
    fn default() -> Self {
        Self {
            max_string_length: 1000,
            max_array_size: 1_000_000,
            allowed_characters: regex::Regex::new(r"^[a-zA-Z0-9._\-\s]+$").unwrap(),
        }
    }
}

impl InputSanitizer {
    /// Sanitize string input
    pub fn sanitize_string(&self, input: &str) -> Result<String, ValidationError> {
        if input.len() > self.max_string_length {
            return Err(ValidationError::SecurityViolation {
                details: format!("String length {} exceeds maximum {}", input.len(), self.max_string_length),
            });
        }
        
        if !self.allowed_characters.is_match(input) {
            return Err(ValidationError::SecurityViolation {
                details: "String contains invalid characters".to_string(),
            });
        }
        
        Ok(input.trim().to_string())
    }
    
    /// Sanitize numeric array
    pub fn sanitize_array(&self, input: &[f64]) -> Result<Vec<f64>, ValidationError> {
        if input.len() > self.max_array_size {
            return Err(ValidationError::SecurityViolation {
                details: format!("Array size {} exceeds maximum {}", input.len(), self.max_array_size),
            });
        }
        
        let mut sanitized = Vec::with_capacity(input.len());
        for (i, &value) in input.iter().enumerate() {
            if value.is_nan() || value.is_infinite() {
                return Err(ValidationError::SecurityViolation {
                    details: format!("Invalid numeric value at index {}", i),
                });
            }
            sanitized.push(value);
        }
        
        Ok(sanitized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parameter_validation() {
        let mut validator = RobustValidator::new();
        
        // Valid temperature
        let result = validator.validate_parameter("temperature", 300.0);
        assert!(result.is_valid);
        
        // Invalid temperature
        let result = validator.validate_parameter("temperature", 3000.0);
        assert!(!result.is_valid);
        assert_eq!(result.severity, Severity::Error);
    }

    #[test]
    fn test_material_validation() {
        let mut validator = RobustValidator::new();
        
        // Valid GST parameters
        let results = validator.validate_material("GST", 300.0, 1e6, 10e-3);
        assert!(results.iter().all(|r| r.is_valid));
        
        // Invalid temperature for GST
        let results = validator.validate_material("GST", 1200.0, 1e6, 10e-3);
        assert!(results.iter().any(|r| !r.is_valid && r.severity == Severity::Critical));
    }

    #[test]
    fn test_array_validation() {
        let mut validator = RobustValidator::new();
        
        // Valid small array
        let results = validator.validate_array_config(10, 10, 1e-6);
        assert!(results.iter().all(|r| r.is_valid));
        
        // Invalid large array
        let results = validator.validate_array_config(1000, 1000, 1e-6);
        assert!(results.iter().any(|r| !r.is_valid));
    }

    #[test]
    fn test_security_validation() {
        let mut validator = RobustValidator::new();
        
        // Valid data
        let data = vec![1.0, 2.0, 3.0];
        let results = validator.validate_security(&data, "test");
        assert!(results.iter().all(|r| r.is_valid));
        
        // Invalid data with NaN
        let data = vec![1.0, f64::NAN, 3.0];
        let results = validator.validate_security(&data, "test");
        assert!(results.iter().any(|r| !r.is_valid && r.severity == Severity::Critical));
    }

    #[test]
    fn test_input_sanitizer() {
        let sanitizer = InputSanitizer::default();
        
        // Valid string
        assert!(sanitizer.sanitize_string("test_123").is_ok());
        
        // Invalid string with special characters
        assert!(sanitizer.sanitize_string("test<script>").is_err());
        
        // Valid array
        let data = vec![1.0, 2.0, 3.0];
        assert!(sanitizer.sanitize_array(&data).is_ok());
        
        // Invalid array with NaN
        let data = vec![1.0, f64::NAN, 3.0];
        assert!(sanitizer.sanitize_array(&data).is_err());
    }
}