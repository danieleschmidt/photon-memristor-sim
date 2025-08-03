//! Optimization algorithms and gradient computation

use crate::core::{Result, PhotonicError};
use nalgebra::DVector;

/// Gradient computation trait
pub trait GradientComputer {
    /// Compute gradients using automatic differentiation
    fn compute_gradients(&self, parameters: &DVector<f64>) -> Result<DVector<f64>>;
}

/// Simple gradient descent optimizer
pub struct GradientDescent {
    learning_rate: f64,
    momentum: f64,
    velocity: Option<DVector<f64>>,
}

impl GradientDescent {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: None,
        }
    }
    
    pub fn step(&mut self, parameters: &mut DVector<f64>, gradients: &DVector<f64>) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(PhotonicError::invalid_parameter(
                "gradient_size", gradients.len(), 
                format!("equal to parameter size {}", parameters.len())
            ));
        }
        
        match &mut self.velocity {
            None => {
                // Initialize velocity
                self.velocity = Some(DVector::zeros(parameters.len()));
            }
            Some(velocity) => {
                // Update with momentum
                *velocity = velocity.clone() * self.momentum - gradients * self.learning_rate;
                *parameters += &*velocity;
                return Ok(());
            }
        }
        
        // First step without momentum
        *parameters -= gradients * self.learning_rate;
        Ok(())
    }
}