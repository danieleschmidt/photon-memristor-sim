//! Error types for the photonic simulation library

use thiserror::Error;

/// Result type alias for photonic operations
pub type Result<T> = std::result::Result<T, PhotonicError>;

/// Main error type for photonic simulation operations
#[derive(Error, Debug)]
pub enum PhotonicError {
    #[error("Simulation error: {message}")]
    Simulation { message: String },
    
    #[error("Device configuration error: {message}")]
    DeviceConfig { message: String },
    
    #[error("Numerical convergence failed: {details}")]
    ConvergenceFailure { details: String },
    
    #[error("Invalid parameter: {param} = {value}, expected {constraint}")]
    InvalidParameter {
        param: String,
        value: String,
        constraint: String,
    },
    
    #[error("Physics violation: {description}")]
    PhysicsViolation { description: String },
    
    #[error("Optimization error: {reason}")]
    Optimization { reason: String },
    
    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocation { size: usize },
    
    #[error("I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },
    
    #[error("Serialization error: {message}")]
    Serialization { message: String },
    
    #[error("WASM interface error: {details}")]
    WasmInterface { details: String },
}

impl PhotonicError {
    /// Create a simulation error
    pub fn simulation(message: impl Into<String>) -> Self {
        Self::Simulation {
            message: message.into(),
        }
    }
    
    /// Create a device configuration error
    pub fn device_config(message: impl Into<String>) -> Self {
        Self::DeviceConfig {
            message: message.into(),
        }
    }
    
    /// Create a convergence failure error
    pub fn convergence_failure(details: impl Into<String>) -> Self {
        Self::ConvergenceFailure {
            details: details.into(),
        }
    }
    
    /// Create an invalid parameter error
    pub fn invalid_parameter(
        param: impl Into<String>,
        value: impl std::fmt::Display,
        constraint: impl Into<String>,
    ) -> Self {
        Self::InvalidParameter {
            param: param.into(),
            value: value.to_string(),
            constraint: constraint.into(),
        }
    }
}