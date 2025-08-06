//! Quantum-inspired task planning and optimization algorithms

use crate::core::{Result, PhotonicError, Complex64};
use nalgebra::{DVector, DMatrix};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use rand::{self, Rng};

/// Quantum-inspired superposition state for task planning
#[derive(Debug, Clone)]
pub struct QuantumSuperposition {
    /// Amplitude coefficients for each basis state
    pub amplitudes: DVector<Complex64>,
    /// Corresponding task assignments
    pub task_assignments: Vec<TaskAssignment>,
    /// Coherence time (how long superposition lasts)
    pub coherence_time: f64,
}

/// Task assignment in the quantum-inspired planner
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    /// Task identifier
    pub task_id: usize,
    /// Resource allocation
    pub resources: Vec<f64>,
    /// Priority level
    pub priority: f64,
    /// Execution time estimate
    pub execution_time: f64,
    /// Dependencies on other tasks
    pub dependencies: Vec<usize>,
}

impl std::hash::Hash for TaskAssignment {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.task_id.hash(state);
        // Hash floats by converting to bits for deterministic behavior
        self.priority.to_bits().hash(state);
        self.execution_time.to_bits().hash(state);
        self.dependencies.hash(state);
        // For resources vector, hash each float as bits
        for &resource in &self.resources {
            resource.to_bits().hash(state);
        }
    }
}

impl PartialEq for TaskAssignment {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id &&
        self.resources.len() == other.resources.len() &&
        self.resources.iter().zip(other.resources.iter()).all(|(a, b)| (a - b).abs() < 1e-10) &&
        (self.priority - other.priority).abs() < 1e-10 &&
        (self.execution_time - other.execution_time).abs() < 1e-10 &&
        self.dependencies == other.dependencies
    }
}

impl Eq for TaskAssignment {}

/// Quantum-inspired task planner using photonic principles
pub struct QuantumTaskPlanner {
    /// Number of qubits (log2 of problem space)
    pub num_qubits: usize,
    /// Current superposition state
    pub state: QuantumSuperposition,
    /// Measurement history
    pub measurement_history: Vec<TaskAssignment>,
    /// Entanglement matrix between tasks
    pub entanglement_matrix: DMatrix<Complex64>,
    /// Phase accumulation rate
    pub phase_rate: f64,
}

impl QuantumTaskPlanner {
    /// Create new quantum-inspired task planner
    pub fn new(num_tasks: usize) -> Result<Self> {
        let num_qubits = (num_tasks as f64).log2().ceil() as usize;
        let dim = 1 << num_qubits; // 2^num_qubits
        
        if dim > 1024 {
            return Err(PhotonicError::invalid_parameter(
                "num_tasks", num_tasks, "≤ 1024 (10 qubits max)"
            ));
        }
        
        // Initialize uniform superposition (Hadamard gate equivalent)
        let amplitudes = DVector::from_element(dim, 
            Complex64::new(1.0 / (dim as f64).sqrt(), 0.0));
        
        // Create task assignments for each computational basis state
        let task_assignments = (0..dim)
            .map(|i| Self::generate_task_assignment(i, num_tasks))
            .collect();
        
        let state = QuantumSuperposition {
            amplitudes,
            task_assignments,
            coherence_time: 1.0, // Initial coherence
        };
        
        // Initialize entanglement matrix (identity for independent tasks)
        let entanglement_matrix = DMatrix::identity(dim, dim);
        
        Ok(Self {
            num_qubits,
            state,
            measurement_history: Vec::new(),
            entanglement_matrix,
            phase_rate: 1.0,
        })
    }
    
    /// Generate task assignment for a given basis state
    fn generate_task_assignment(state_index: usize, num_tasks: usize) -> TaskAssignment {
        // Interpret binary representation as resource allocation
        let mut resources = Vec::with_capacity(num_tasks);
        let mut priority = 0.0;
        
        for task_id in 0..num_tasks {
            let bit = (state_index >> task_id) & 1;
            let resource_allocation = if bit == 1 { 0.8 } else { 0.2 };
            resources.push(resource_allocation);
            priority += resource_allocation;
        }
        
        TaskAssignment {
            task_id: state_index % num_tasks,
            resources,
            priority: priority / num_tasks as f64,
            execution_time: 1.0 + priority * 2.0, // Priority affects execution time
            dependencies: vec![], // Simplified for now
        }
    }
    
    /// Apply quantum interference to enhance optimal solutions
    pub fn apply_interference(&mut self, target_pattern: &TaskAssignment) -> Result<()> {
        // Calculate amplitude modifications based on similarity to target
        for (i, assignment) in self.state.task_assignments.iter().enumerate() {
            let similarity = self.calculate_assignment_similarity(assignment, target_pattern);
            
            // Constructive interference for similar assignments
            let phase_shift = Complex64::new(0.0, similarity * std::f64::consts::PI);
            let interference_factor = 1.0 + similarity;
            
            self.state.amplitudes[i] *= Complex64::new(interference_factor, 0.0) * phase_shift.exp();
        }
        
        // Renormalize state
        self.normalize_state()?;
        Ok(())
    }
    
    /// Calculate similarity between task assignments
    fn calculate_assignment_similarity(&self, a: &TaskAssignment, b: &TaskAssignment) -> f64 {
        if a.resources.len() != b.resources.len() {
            return 0.0;
        }
        
        let dot_product: f64 = a.resources.iter()
            .zip(b.resources.iter())
            .map(|(x, y)| x * y)
            .sum();
        
        let norm_a: f64 = a.resources.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.resources.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// Apply quantum evolution (Schrödinger equation equivalent)
    pub fn evolve(&mut self, time_step: f64) -> Result<()> {
        // Hamiltonian evolution: |ψ(t+dt)⟩ = e^(-iHdt)|ψ(t)⟩
        let hamiltonion = self.construct_hamiltonian()?;
        
        // Simple matrix exponentiation approximation
        let evolution_operator = self.matrix_exponential(&hamiltonion, -time_step)?;
        
        // Apply evolution
        self.state.amplitudes = &evolution_operator * &self.state.amplitudes;
        
        // Update coherence (decoherence model)
        self.state.coherence_time *= (-time_step / 10.0).exp(); // T2 = 10 time units
        
        Ok(())
    }
    
    /// Construct problem Hamiltonian for task scheduling
    fn construct_hamiltonian(&self) -> Result<DMatrix<Complex64>> {
        let dim = self.state.amplitudes.len();
        let mut hamiltonian = DMatrix::zeros(dim, dim);
        
        // Diagonal terms: task execution costs
        for (i, assignment) in self.state.task_assignments.iter().enumerate() {
            let cost = assignment.execution_time * assignment.priority;
            hamiltonian[(i, i)] = Complex64::new(cost, 0.0);
        }
        
        // Off-diagonal terms: task interactions/dependencies
        for i in 0..dim {
            for j in i+1..dim {
                let interaction = self.calculate_task_interaction(
                    &self.state.task_assignments[i],
                    &self.state.task_assignments[j]
                );
                
                hamiltonian[(i, j)] = Complex64::new(interaction, 0.0);
                hamiltonian[(j, i)] = Complex64::new(interaction, 0.0); // Hermitian
            }
        }
        
        Ok(hamiltonian)
    }
    
    /// Calculate interaction between two task assignments
    fn calculate_task_interaction(&self, task1: &TaskAssignment, task2: &TaskAssignment) -> f64 {
        // Resource contention: negative interaction if both use same resources heavily
        let resource_overlap: f64 = task1.resources.iter()
            .zip(task2.resources.iter())
            .map(|(r1, r2)| (r1 * r2).min(0.5)) // Cap overlap effect
            .sum();
        
        -resource_overlap * 0.1 // Small negative interaction for contention
    }
    
    /// Matrix exponential approximation (first order)
    fn matrix_exponential(&self, matrix: &DMatrix<Complex64>, scale: f64) -> Result<DMatrix<Complex64>> {
        let dim = matrix.nrows();
        let mut result = DMatrix::identity(dim, dim);
        
        // First-order approximation: exp(A) ≈ I + A
        for i in 0..dim {
            for j in 0..dim {
                result[(i, j)] += matrix[(i, j)] * scale;
            }
        }
        
        Ok(result)
    }
    
    /// Perform quantum measurement to collapse to classical solution
    pub fn measure(&mut self) -> Result<TaskAssignment> {
        // Calculate measurement probabilities
        let probabilities: Vec<f64> = self.state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        // Verify normalization
        let total_prob: f64 = probabilities.iter().sum();
        if (total_prob - 1.0).abs() > 1e-6 {
            return Err(PhotonicError::simulation(format!(
                "State not normalized: total probability = {}", total_prob
            )));
        }
        
        // Random measurement according to Born rule
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen();
        let mut cumulative_prob = 0.0;
        
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                let measured_assignment = self.state.task_assignments[i].clone();
                self.measurement_history.push(measured_assignment.clone());
                
                // Collapse state to measured eigenstate
                self.collapse_to_eigenstate(i)?;
                
                return Ok(measured_assignment);
            }
        }
        
        // Fallback (shouldn't reach here with proper normalization)
        let last_assignment = self.state.task_assignments.last().unwrap().clone();
        self.measurement_history.push(last_assignment.clone());
        Ok(last_assignment)
    }
    
    /// Collapse quantum state after measurement
    fn collapse_to_eigenstate(&mut self, measured_index: usize) -> Result<()> {
        // Set all amplitudes to zero except the measured state
        for i in 0..self.state.amplitudes.len() {
            if i == measured_index {
                self.state.amplitudes[i] = Complex64::new(1.0, 0.0);
            } else {
                self.state.amplitudes[i] = Complex64::new(0.0, 0.0);
            }
        }
        
        // Reset coherence (classical state has infinite coherence)
        self.state.coherence_time = f64::INFINITY;
        
        Ok(())
    }
    
    /// Apply quantum annealing for optimization
    pub fn quantum_anneal(&mut self, num_steps: usize, initial_temp: f64) -> Result<TaskAssignment> {
        let mut temperature = initial_temp;
        let cooling_rate = (initial_temp / 0.01_f64).powf(1.0 / num_steps as f64);
        
        for step in 0..num_steps {
            // Evolve system
            let time_step = 0.1 / (1.0 + temperature);
            self.evolve(time_step)?;
            
            // Apply thermal noise (decoherence)
            self.apply_thermal_noise(temperature)?;
            
            // Cool down
            temperature /= cooling_rate;
            
            if step % (num_steps / 10) == 0 {
                let energy = self.calculate_energy()?;
                println!("Annealing step {}: T={:.3}, E={:.3}", step, temperature, energy);
            }
        }
        
        // Final measurement
        self.measure()
    }
    
    /// Apply thermal noise for annealing
    fn apply_thermal_noise(&mut self, temperature: f64) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for amplitude in self.state.amplitudes.iter_mut() {
            // Add thermal phase noise
            let phase_noise = rng.gen_range(-temperature..temperature) * 0.01;
            let noise = Complex64::new(0.0, phase_noise).exp();
            *amplitude *= noise;
        }
        
        self.normalize_state()?;
        Ok(())
    }
    
    /// Calculate system energy expectation value
    fn calculate_energy(&self) -> Result<f64> {
        let hamiltonian = self.construct_hamiltonian()?;
        
        // ⟨ψ|H|ψ⟩
        let h_psi = &hamiltonian * &self.state.amplitudes;
        let energy = self.state.amplitudes.iter()
            .zip(h_psi.iter())
            .map(|(psi_conj, h_psi_i)| (psi_conj.conj() * h_psi_i).re)
            .sum();
        
        Ok(energy)
    }
    
    /// Normalize quantum state
    fn normalize_state(&mut self) -> Result<()> {
        let norm_squared: f64 = self.state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        if norm_squared == 0.0 {
            return Err(PhotonicError::simulation("Cannot normalize zero state"));
        }
        
        let norm = norm_squared.sqrt();
        for amplitude in self.state.amplitudes.iter_mut() {
            *amplitude /= norm;
        }
        
        Ok(())
    }
    
    /// Get current state fidelity (how "quantum" the state is)
    pub fn fidelity(&self) -> f64 {
        // Measure of quantum coherence
        let coherence_factor = (-self.state.coherence_time.recip()).exp().max(0.0);
        
        // Participation ratio (inverse of effective dimension)
        let participation_ratio = 1.0 / self.state.amplitudes.iter()
            .map(|amp| amp.norm_sqr().powi(2))
            .sum::<f64>();
        
        coherence_factor * (participation_ratio / self.state.amplitudes.len() as f64)
    }
    
    /// Apply quantum error correction
    pub fn error_correction(&mut self) -> Result<()> {
        // Simple amplitude damping correction
        let correction_threshold = 0.01;
        
        for amplitude in self.state.amplitudes.iter_mut() {
            if amplitude.norm() < correction_threshold {
                *amplitude = Complex64::new(0.0, 0.0);
            }
        }
        
        // Check if we still have a valid state
        let total_norm: f64 = self.state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if total_norm < 0.1 {
            // State too corrupted, reinitialize
            let dim = self.state.amplitudes.len();
            self.state.amplitudes = DVector::from_element(dim, 
                Complex64::new(1.0 / (dim as f64).sqrt(), 0.0));
            self.state.coherence_time = 1.0;
        } else {
            self.normalize_state()?;
        }
        
        Ok(())
    }
    
    /// Generate optimization report
    pub fn generate_report(&self) -> QuantumPlanningReport {
        let optimal_assignment = self.measurement_history.last().cloned()
            .unwrap_or_else(|| TaskAssignment {
                task_id: 0,
                resources: vec![],
                priority: 0.0,
                execution_time: 0.0,
                dependencies: vec![],
            });
        
        let fidelity = self.fidelity();
        let coherence = self.state.coherence_time;
        
        let total_measurements = self.measurement_history.len();
        let convergence_rate = if total_measurements > 1 {
            let recent_assignments = &self.measurement_history[total_measurements.saturating_sub(10)..];
            let unique_assignments = recent_assignments.iter().collect::<std::collections::HashSet<_>>().len();
            1.0 - (unique_assignments as f64 / recent_assignments.len() as f64)
        } else {
            0.0
        };
        
        QuantumPlanningReport {
            optimal_assignment,
            quantum_fidelity: fidelity,
            coherence_time: coherence,
            total_measurements,
            convergence_rate,
            entanglement_entropy: self.calculate_entanglement_entropy(),
        }
    }
    
    /// Calculate von Neumann entropy as measure of entanglement
    fn calculate_entanglement_entropy(&self) -> f64 {
        let probabilities: Vec<f64> = self.state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .filter(|&p| p > 1e-10) // Avoid log(0)
            .collect();
        
        -probabilities.iter()
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}

/// Report generated by quantum task planner
#[derive(Debug, Clone)]
pub struct QuantumPlanningReport {
    /// Best task assignment found
    pub optimal_assignment: TaskAssignment,
    /// Quantum fidelity of final state
    pub quantum_fidelity: f64,
    /// Remaining coherence time
    pub coherence_time: f64,
    /// Total number of measurements performed
    pub total_measurements: usize,
    /// Convergence rate (how often same solution is found)
    pub convergence_rate: f64,
    /// Entanglement entropy
    pub entanglement_entropy: f64,
}

/// Factory for creating quantum-inspired planners
pub struct QuantumPlannerFactory;

impl QuantumPlannerFactory {
    /// Create planner optimized for photonic neural networks
    pub fn create_photonic_planner(num_devices: usize) -> Result<QuantumTaskPlanner> {
        let mut planner = QuantumTaskPlanner::new(num_devices)?;
        
        // Optimize for photonic constraints
        planner.phase_rate = 2.0; // Faster evolution for optical frequencies
        
        // Set up entanglement for device coupling
        let dim = planner.state.amplitudes.len();
        let mut entanglement = DMatrix::identity(dim, dim);
        
        // Add nearest-neighbor entanglement (for coupled photonic devices)
        for i in 0..dim-1 {
            entanglement[(i, i+1)] = Complex64::new(0.1, 0.0);
            entanglement[(i+1, i)] = Complex64::new(0.1, 0.0);
        }
        
        planner.entanglement_matrix = entanglement;
        
        Ok(planner)
    }
    
    /// Create planner for large-scale optimization
    pub fn create_scalable_planner(num_tasks: usize) -> Result<QuantumTaskPlanner> {
        if num_tasks > 1024 {
            return Err(PhotonicError::invalid_parameter(
                "num_tasks", num_tasks, "≤ 1024 for scalable planner"
            ));
        }
        
        let mut planner = QuantumTaskPlanner::new(num_tasks)?;
        
        // Optimize for large problems
        planner.phase_rate = 0.5; // Slower, more stable evolution
        
        Ok(planner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_planner_creation() {
        let planner = QuantumTaskPlanner::new(8).unwrap();
        assert_eq!(planner.num_qubits, 3); // log2(8) = 3
        assert_eq!(planner.state.amplitudes.len(), 8);
    }
    
    #[test]
    fn test_state_normalization() {
        let mut planner = QuantumTaskPlanner::new(4).unwrap();
        
        // Modify state
        planner.state.amplitudes[0] = Complex64::new(2.0, 0.0);
        
        planner.normalize_state().unwrap();
        
        let norm_squared: f64 = planner.state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_measurement() {
        let mut planner = QuantumTaskPlanner::new(4).unwrap();
        let assignment = planner.measure().unwrap();
        
        assert!(assignment.resources.len() > 0);
        assert_eq!(planner.measurement_history.len(), 1);
    }
    
    #[test]
    fn test_quantum_annealing() {
        let mut planner = QuantumTaskPlanner::new(4).unwrap();
        let result = planner.quantum_anneal(50, 1.0).unwrap();
        
        assert!(result.priority >= 0.0);
        assert!(result.execution_time > 0.0);
    }
    
    #[test]
    fn test_factory_creation() {
        let planner = QuantumPlannerFactory::create_photonic_planner(16).unwrap();
        assert_eq!(planner.phase_rate, 2.0);
        assert_eq!(planner.num_qubits, 4); // log2(16) = 4
    }
}