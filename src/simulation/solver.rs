//! Simulation solver configuration and management

use crate::core::{Result, PhotonicError};

use std::collections::HashMap;

/// Solver configuration for photonic simulation
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Solver type
    pub solver_type: SolverType,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Time step for dynamic simulation
    pub time_step: f64,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Number of threads
    pub num_threads: Option<usize>,
    /// Additional solver-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Available solver types
#[derive(Debug, Clone)]
pub enum SolverType {
    /// Beam Propagation Method
    BeamPropagation,
    /// Finite Difference Time Domain
    FDTD,
    /// Transfer Matrix Method
    TransferMatrix,
    /// Monte Carlo simulation
    MonteCarlo,
    /// Eigenmode solver
    Eigenmode,
}

/// Main simulation solver
pub struct SimulationSolver {
    config: SolverConfig,
    state: SolverState,
}

/// Internal solver state
#[derive(Debug)]
struct SolverState {
    initialized: bool,
    current_iteration: usize,
    residual: f64,
    converged: bool,
}

impl SimulationSolver {
    /// Create new solver with configuration
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            state: SolverState {
                initialized: false,
                current_iteration: 0,
                residual: f64::INFINITY,
                converged: false,
            },
        }
    }
    
    /// Initialize solver
    pub fn initialize(&mut self) -> Result<()> {
        // Validate configuration
        self.validate_config()?;
        
        // Setup threading
        if let Some(num_threads) = self.config.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .map_err(|e| PhotonicError::simulation(format!("Failed to setup threads: {}", e)))?;
        }
        
        self.state.initialized = true;
        Ok(())
    }
    
    /// Validate solver configuration
    fn validate_config(&self) -> Result<()> {
        if self.config.tolerance <= 0.0 {
            return Err(PhotonicError::invalid_parameter(
                "tolerance", self.config.tolerance, "> 0"
            ));
        }
        
        if self.config.max_iterations == 0 {
            return Err(PhotonicError::invalid_parameter(
                "max_iterations", self.config.max_iterations, "> 0"
            ));
        }
        
        if self.config.time_step <= 0.0 {
            return Err(PhotonicError::invalid_parameter(
                "time_step", self.config.time_step, "> 0"
            ));
        }
        
        Ok(())
    }
    
    /// Check convergence
    pub fn check_convergence(&mut self, residual: f64) -> bool {
        self.state.residual = residual;
        self.state.converged = residual < self.config.tolerance;
        self.state.converged
    }
    
    /// Get current iteration
    pub fn current_iteration(&self) -> usize {
        self.state.current_iteration
    }
    
    /// Increment iteration counter
    pub fn increment_iteration(&mut self) {
        self.state.current_iteration += 1;
    }
    
    /// Reset solver state
    pub fn reset(&mut self) {
        self.state.current_iteration = 0;
        self.state.residual = f64::INFINITY;
        self.state.converged = false;
    }
    
    /// Get solver configuration
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }
    
    /// Check if solver has converged
    pub fn has_converged(&self) -> bool {
        self.state.converged
    }
    
    /// Check if maximum iterations reached
    pub fn max_iterations_reached(&self) -> bool {
        self.state.current_iteration >= self.config.max_iterations
    }
    
    /// Get current residual
    pub fn residual(&self) -> f64 {
        self.state.residual
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            solver_type: SolverType::BeamPropagation,
            tolerance: 1e-9,
            max_iterations: 1000,
            time_step: 1e-12,
            use_gpu: false,
            num_threads: None,
            parameters: HashMap::new(),
        }
    }
}

impl SolverConfig {
    /// Create BPM solver configuration
    pub fn beam_propagation() -> Self {
        let mut config = Self::default();
        config.solver_type = SolverType::BeamPropagation;
        config.parameters.insert("step_size".to_string(), 1e-6);
        config
    }
    
    /// Create FDTD solver configuration
    pub fn fdtd() -> Self {
        let mut config = Self::default();
        config.solver_type = SolverType::FDTD;
        config.time_step = 1e-15; // Femtosecond time steps
        config.parameters.insert("grid_spacing".to_string(), 10e-9);
        config
    }
    
    /// Create Transfer Matrix configuration
    pub fn transfer_matrix() -> Self {
        let mut config = Self::default();
        config.solver_type = SolverType::TransferMatrix;
        config.max_iterations = 1; // Single pass for linear devices
        config
    }
    
    /// Set GPU acceleration
    pub fn with_gpu(mut self) -> Self {
        self.use_gpu = true;
        self
    }
    
    /// Set number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }
    
    /// Set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solver_config_creation() {
        let config = SolverConfig::default();
        assert_eq!(config.tolerance, 1e-9);
        assert_eq!(config.max_iterations, 1000);
    }
    
    #[test]
    fn test_solver_initialization() {
        let config = SolverConfig::default();
        let mut solver = SimulationSolver::new(config);
        
        assert!(solver.initialize().is_ok());
        assert_eq!(solver.current_iteration(), 0);
        assert!(!solver.has_converged());
    }
    
    #[test]
    fn test_convergence_check() {
        let config = SolverConfig::default();
        let mut solver = SimulationSolver::new(config);
        solver.initialize().unwrap();
        
        // Should not converge with large residual
        assert!(!solver.check_convergence(1e-3));
        
        // Should converge with small residual
        assert!(solver.check_convergence(1e-12));
    }
    
    #[test]
    fn test_invalid_config() {
        let mut config = SolverConfig::default();
        config.tolerance = -1.0; // Invalid
        
        let mut solver = SimulationSolver::new(config);
        assert!(solver.initialize().is_err());
    }
}