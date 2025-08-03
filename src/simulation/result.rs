//! Simulation results and metrics


use nalgebra::DVector;
use std::collections::HashMap;

/// Results from photonic simulation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Simulation metrics
    pub metrics: SimulationMetrics,
    /// Output optical fields
    pub outputs: Vec<crate::core::OpticalField>,
    /// Device states after simulation
    pub device_states: HashMap<String, DVector<f64>>,
    /// Timing information
    pub timing: TimingInfo,
    /// Convergence information
    pub convergence: ConvergenceInfo,
}

/// Comprehensive simulation metrics
#[derive(Debug, Clone)]
pub struct SimulationMetrics {
    /// Total power consumption (W)
    pub power_consumption: f64,
    /// Energy efficiency (TOPS/W)
    pub energy_efficiency: f64,
    /// Compute throughput (TOPS)
    pub throughput: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// Average temperature (K)
    pub temperature: f64,
    /// Insertion loss (dB)
    pub insertion_loss: f64,
    /// Signal-to-noise ratio (dB)
    pub signal_to_noise_ratio: f64,
}

/// Timing information
#[derive(Debug, Clone)]
pub struct TimingInfo {
    /// Total simulation time (seconds)
    pub total_time: f64,
    /// Forward propagation time (seconds)
    pub forward_time: f64,
    /// Backward propagation time (seconds)
    pub backward_time: f64,
    /// Device update time (seconds)
    pub device_update_time: f64,
    /// Memory allocation time (seconds)
    pub memory_time: f64,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether simulation converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual
    pub final_residual: f64,
    /// Convergence rate
    pub convergence_rate: f64,
}

impl SimulationResult {
    /// Create new simulation result
    pub fn new() -> Self {
        Self {
            metrics: SimulationMetrics::default(),
            outputs: Vec::new(),
            device_states: HashMap::new(),
            timing: TimingInfo::default(),
            convergence: ConvergenceInfo::default(),
        }
    }
    
    /// Calculate efficiency metrics
    pub fn calculate_efficiency(&mut self) {
        if self.metrics.power_consumption > 0.0 {
            self.metrics.energy_efficiency = self.metrics.throughput / self.metrics.power_consumption;
        }
    }
    
    /// Add timing measurement
    pub fn add_timing(&mut self, component: &str, duration: f64) {
        match component {
            "forward" => self.timing.forward_time += duration,
            "backward" => self.timing.backward_time += duration,
            "device_update" => self.timing.device_update_time += duration,
            "memory" => self.timing.memory_time += duration,
            _ => {}
        }
        self.timing.total_time = self.timing.forward_time 
            + self.timing.backward_time 
            + self.timing.device_update_time 
            + self.timing.memory_time;
    }
    
    /// Check if results are valid
    pub fn validate(&self) -> bool {
        self.metrics.power_consumption >= 0.0 
            && self.metrics.throughput >= 0.0
            && self.timing.total_time >= 0.0
    }
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self {
            power_consumption: 0.0,
            energy_efficiency: 0.0,
            throughput: 0.0,
            memory_usage: 0.0,
            temperature: 300.0, // Room temperature
            insertion_loss: 0.0,
            signal_to_noise_ratio: 60.0, // Good SNR
        }
    }
}

impl Default for TimingInfo {
    fn default() -> Self {
        Self {
            total_time: 0.0,
            forward_time: 0.0,
            backward_time: 0.0,
            device_update_time: 0.0,
            memory_time: 0.0,
        }
    }
}

impl Default for ConvergenceInfo {
    fn default() -> Self {
        Self {
            converged: true,
            iterations: 0,
            final_residual: 0.0,
            convergence_rate: 1.0,
        }
    }
}