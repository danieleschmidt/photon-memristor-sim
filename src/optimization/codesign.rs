//! Co-design optimization for simultaneous device and algorithm optimization

use crate::core::{Result, PhotonicError, OpticalField};
use crate::devices::traits::PhotonicDevice;
use crate::simulation::PhotonicArray;
use nalgebra::{DVector, DMatrix};
use std::collections::HashMap;
use rand::{self, Rng};

/// Multi-objective optimization objectives
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum Objective {
    /// Minimize prediction error
    Accuracy,
    /// Minimize power consumption
    Power,
    /// Minimize chip area
    Area,
    /// Minimize inference latency
    Latency,
    /// Maximize robustness to variations
    Robustness,
    /// Minimize fabrication cost
    Cost,
}

/// Constraints for co-design optimization
#[derive(Debug, Clone)]
pub struct CoDesignConstraints {
    /// Maximum power budget (W)
    pub max_power: f64,
    /// Maximum temperature (K)
    pub max_temperature: f64,
    /// Maximum chip area (m²)
    pub max_area: f64,
    /// Device parameter bounds
    pub device_bounds: HashMap<String, (f64, f64)>,
    /// Fabrication tolerances
    pub fabrication_tolerances: HashMap<String, f64>,
}

impl Default for CoDesignConstraints {
    fn default() -> Self {
        let mut device_bounds = HashMap::new();
        device_bounds.insert("crystallinity".to_string(), (0.0, 1.0));
        device_bounds.insert("wavelength".to_string(), (1500e-9, 1600e-9));
        device_bounds.insert("power".to_string(), (0.0, 10e-3));
        
        let mut tolerances = HashMap::new();
        tolerances.insert("width".to_string(), 5e-9);    // 5nm width variation
        tolerances.insert("height".to_string(), 2e-9);   // 2nm height variation
        tolerances.insert("index".to_string(), 0.01);    // 1% index variation
        
        Self {
            max_power: 100e-3,        // 100mW
            max_temperature: 350.0,    // 350K (77°C)
            max_area: 1e-6,           // 1 mm²
            device_bounds,
            fabrication_tolerances: tolerances,
        }
    }
}

/// Pareto-optimal solution in multi-objective optimization
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// Device parameters
    pub device_params: DVector<f64>,
    /// Neural network weights
    pub network_weights: DVector<f64>,
    /// Objective function values
    pub objectives: HashMap<Objective, f64>,
    /// Constraint violations
    pub constraint_violations: f64,
}

/// Co-design optimizer using multi-objective evolutionary algorithm
pub struct CoDesignOptimizer {
    /// Population size for evolutionary algorithm
    pub population_size: usize,
    /// Number of generations
    pub num_generations: usize,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability  
    pub mutation_prob: f64,
    /// Constraints
    pub constraints: CoDesignConstraints,
    /// Current Pareto front
    pub pareto_front: Vec<ParetoSolution>,
}

impl CoDesignOptimizer {
    /// Create new co-design optimizer
    pub fn new(constraints: CoDesignConstraints) -> Self {
        Self {
            population_size: 100,
            num_generations: 500,
            crossover_prob: 0.8,
            mutation_prob: 0.1,
            constraints,
            pareto_front: Vec::new(),
        }
    }
    
    /// Run multi-objective co-design optimization
    pub fn optimize<F>(&mut self, 
                      objectives: Vec<Objective>,
                      evaluation_fn: F) -> Result<Vec<ParetoSolution>>
    where F: Fn(&DVector<f64>, &DVector<f64>) -> Result<HashMap<Objective, f64>>
    {
        // Initialize population
        let mut population = self.initialize_population()?;
        
        // Evolution loop
        for generation in 0..self.num_generations {
            // Evaluate population
            let mut evaluated_pop = Vec::new();
            for individual in &population {
                let objectives_vals = evaluation_fn(&individual.device_params, &individual.network_weights)?;
                let violations = self.evaluate_constraints(&individual.device_params);
                
                evaluated_pop.push(ParetoSolution {
                    device_params: individual.device_params.clone(),
                    network_weights: individual.network_weights.clone(),
                    objectives: objectives_vals,
                    constraint_violations: violations,
                });
            }
            
            // Non-dominated sorting
            let fronts = self.non_dominated_sort(&evaluated_pop);
            
            // Update Pareto front
            self.pareto_front = fronts[0].clone();
            
            // Selection for next generation
            population = self.environmental_selection(evaluated_pop, &objectives)?;
            
            // Apply genetic operators
            population = self.apply_genetic_operators(population)?;
            
            // Progress reporting
            if generation % 50 == 0 {
                println!("Generation {}: Pareto front size = {}", generation, self.pareto_front.len());
            }
        }
        
        Ok(self.pareto_front.clone())
    }
    
    /// Initialize random population
    fn initialize_population(&self) -> Result<Vec<ParetoSolution>> {
        let mut population = Vec::with_capacity(self.population_size);
        
        for _ in 0..self.population_size {
            // Random device parameters within bounds
            let device_params = self.random_device_params()?;
            
            // Random network weights (small initialization)
            let network_weights = self.random_network_weights()?;
            
            population.push(ParetoSolution {
                device_params,
                network_weights,
                objectives: HashMap::new(),
                constraint_violations: 0.0,
            });
        }
        
        Ok(population)
    }
    
    /// Generate random device parameters within constraints
    fn random_device_params(&self) -> Result<DVector<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // For demo, assume 10 device parameters
        let mut params = Vec::with_capacity(10);
        
        for i in 0..10 {
            let param_name = format!("param_{}", i);
            let bounds = self.constraints.device_bounds.get(&param_name)
                .unwrap_or(&(0.0, 1.0));
            
            let value = rng.gen_range(bounds.0..=bounds.1);
            params.push(value);
        }
        
        Ok(DVector::from_vec(params))
    }
    
    /// Generate random network weights
    fn random_network_weights(&self) -> Result<DVector<f64>> {
        // use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        
        // let normal = Normal::new(0.0, 0.1).map_err(|_| 
        //     PhotonicError::simulation("Failed to create normal distribution"))?;
        
        // For demo, assume 1000 network weights
        let weights: Vec<f64> = (0..1000)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        
        Ok(DVector::from_vec(weights))
    }
    
    /// Evaluate constraint violations
    fn evaluate_constraints(&self, device_params: &DVector<f64>) -> f64 {
        let mut violations = 0.0;
        
        // Power constraint
        let power = self.estimate_power(device_params);
        if power > self.constraints.max_power {
            violations += (power - self.constraints.max_power) / self.constraints.max_power;
        }
        
        // Temperature constraint
        let temperature = self.estimate_temperature(device_params);
        if temperature > self.constraints.max_temperature {
            violations += (temperature - self.constraints.max_temperature) / self.constraints.max_temperature;
        }
        
        // Area constraint
        let area = self.estimate_area(device_params);
        if area > self.constraints.max_area {
            violations += (area - self.constraints.max_area) / self.constraints.max_area;
        }
        
        violations
    }
    
    /// Estimate power consumption from device parameters
    fn estimate_power(&self, device_params: &DVector<f64>) -> f64 {
        // Simplified power model
        device_params.iter().map(|&p| p * p * 1e-3).sum()  // Quadratic in parameters
    }
    
    /// Estimate temperature from device parameters
    fn estimate_temperature(&self, device_params: &DVector<f64>) -> f64 {
        let power = self.estimate_power(device_params);
        300.0 + power * 100.0  // Simple thermal model: T = T_ambient + R_th * P
    }
    
    /// Estimate chip area from device parameters
    fn estimate_area(&self, device_params: &DVector<f64>) -> f64 {
        // Simplified area model
        let num_devices = device_params.len() as f64;
        num_devices * 10e-6 * 10e-6  // 10μm x 10μm per device
    }
    
    /// Non-dominated sorting for multi-objective optimization
    fn non_dominated_sort(&self, population: &[ParetoSolution]) -> Vec<Vec<ParetoSolution>> {
        let mut fronts = Vec::new();
        let mut current_front = Vec::new();
        let mut remaining: Vec<_> = population.iter().collect();
        
        while !remaining.is_empty() {
            current_front.clear();
            let mut next_remaining = Vec::new();
            
            for &solution in &remaining {
                let mut is_dominated = false;
                
                for &other in &remaining {
                    if self.dominates(other, solution) {
                        is_dominated = true;
                        break;
                    }
                }
                
                if !is_dominated {
                    current_front.push(solution.clone());
                } else {
                    next_remaining.push(solution);
                }
            }
            
            fronts.push(current_front.clone());
            remaining = next_remaining;
        }
        
        fronts
    }
    
    /// Check if solution a dominates solution b
    fn dominates(&self, a: &ParetoSolution, b: &ParetoSolution) -> bool {
        // a dominates b if a is at least as good in all objectives and better in at least one
        let mut at_least_as_good = true;
        let mut better_in_one = false;
        
        for (objective, &value_a) in &a.objectives {
            if let Some(&value_b) = b.objectives.get(objective) {
                match objective {
                    Objective::Accuracy => {
                        // Higher is better for accuracy
                        if value_a < value_b {
                            at_least_as_good = false;
                            break;
                        }
                        if value_a > value_b {
                            better_in_one = true;
                        }
                    },
                    _ => {
                        // Lower is better for other objectives
                        if value_a > value_b {
                            at_least_as_good = false;
                            break;
                        }
                        if value_a < value_b {
                            better_in_one = true;
                        }
                    }
                }
            }
        }
        
        // Also consider constraint violations
        if a.constraint_violations > b.constraint_violations {
            at_least_as_good = false;
        } else if a.constraint_violations < b.constraint_violations {
            better_in_one = true;
        }
        
        at_least_as_good && better_in_one
    }
    
    /// Environmental selection for next generation
    fn environmental_selection(&self, 
                             population: Vec<ParetoSolution>,
                             _objectives: &[Objective]) -> Result<Vec<ParetoSolution>> {
        // Simple strategy: keep best half of population
        let mut selected = population;
        selected.sort_by(|a, b| {
            // Sort by constraint violations first, then by first objective
            if a.constraint_violations != b.constraint_violations {
                a.constraint_violations.partial_cmp(&b.constraint_violations).unwrap()
            } else {
                // Sort by first objective (assuming it exists)
                let obj_a = a.objectives.values().next().unwrap_or(&f64::MAX);
                let obj_b = b.objectives.values().next().unwrap_or(&f64::MAX);
                obj_a.partial_cmp(obj_b).unwrap()
            }
        });
        
        selected.truncate(self.population_size);
        Ok(selected)
    }
    
    /// Apply genetic operators (crossover and mutation)
    fn apply_genetic_operators(&self, population: Vec<ParetoSolution>) -> Result<Vec<ParetoSolution>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut new_population = Vec::with_capacity(self.population_size);
        
        while new_population.len() < self.population_size {
            // Selection (tournament selection)
            let parent1 = self.tournament_selection(&population)?;
            let parent2 = self.tournament_selection(&population)?;
            
            // Crossover
            let mut offspring = if rng.gen::<f64>() < self.crossover_prob {
                self.crossover(&parent1, &parent2)?
            } else {
                parent1.clone()
            };
            
            // Mutation
            if rng.gen::<f64>() < self.mutation_prob {
                self.mutate(&mut offspring)?;
            }
            
            new_population.push(offspring);
        }
        
        Ok(new_population)
    }
    
    /// Tournament selection
    fn tournament_selection<'a>(&self, population: &'a [ParetoSolution]) -> Result<&'a ParetoSolution> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let tournament_size = 3.min(population.len());
        let mut best = &population[rng.gen_range(0..population.len())];
        
        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if self.dominates(candidate, best) {
                best = candidate;
            }
        }
        
        Ok(best)
    }
    
    /// Crossover operation
    fn crossover(&self, parent1: &ParetoSolution, parent2: &ParetoSolution) -> Result<ParetoSolution> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Uniform crossover for device parameters
        let mut device_params = parent1.device_params.clone();
        for i in 0..device_params.len() {
            if rng.gen::<f64>() < 0.5 {
                device_params[i] = parent2.device_params[i];
            }
        }
        
        // Uniform crossover for network weights
        let mut network_weights = parent1.network_weights.clone();
        for i in 0..network_weights.len() {
            if rng.gen::<f64>() < 0.5 {
                network_weights[i] = parent2.network_weights[i];
            }
        }
        
        Ok(ParetoSolution {
            device_params,
            network_weights,
            objectives: HashMap::new(),
            constraint_violations: 0.0,
        })
    }
    
    /// Mutation operation
    fn mutate(&self, solution: &mut ParetoSolution) -> Result<()> {
        use rand::Rng;
        // use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        
        // let normal = Normal::new(0.0, 0.1).map_err(|_| 
        //     PhotonicError::simulation("Failed to create normal distribution"))?;
        
        // Mutate device parameters
        for i in 0..solution.device_params.len() {
            if rng.gen::<f64>() < 0.1 {  // 10% chance per parameter
                let mutation = rng.gen_range(-0.1..0.1);
                solution.device_params[i] += mutation;
                
                // Clip to bounds
                let param_name = format!("param_{}", i);
                if let Some(&(min_val, max_val)) = self.constraints.device_bounds.get(&param_name) {
                    solution.device_params[i] = solution.device_params[i].max(min_val).min(max_val);
                }
            }
        }
        
        // Mutate network weights
        for weight in solution.network_weights.iter_mut() {
            if rng.gen::<f64>() < 0.01 {  // 1% chance per weight
                let mutation = rng.gen_range(-0.1..0.1);
                *weight += mutation;
            }
        }
        
        Ok(())
    }
}

/// Robustness analysis for manufacturing variations
pub struct RobustnessAnalyzer {
    pub num_samples: usize,
    pub variation_model: VariationModel,
}

#[derive(Debug, Clone)]
pub struct VariationModel {
    pub parameter_variations: HashMap<String, f64>,  // Standard deviations
    pub correlation_matrix: Option<DMatrix<f64>>,    // Parameter correlations
}

impl RobustnessAnalyzer {
    /// Create new robustness analyzer
    pub fn new(num_samples: usize) -> Self {
        let mut parameter_variations = HashMap::new();
        parameter_variations.insert("width".to_string(), 5e-9);      // 5nm std
        parameter_variations.insert("height".to_string(), 2e-9);     // 2nm std
        parameter_variations.insert("index".to_string(), 0.01);      // 1% std
        parameter_variations.insert("crystallinity".to_string(), 0.05); // 5% std
        
        Self {
            num_samples,
            variation_model: VariationModel {
                parameter_variations,
                correlation_matrix: None,
            },
        }
    }
    
    /// Monte Carlo analysis of design robustness
    pub fn monte_carlo_analysis<F>(&self, 
                                  nominal_params: &DVector<f64>,
                                  evaluation_fn: F) -> Result<RobustnessMetrics>
    where F: Fn(&DVector<f64>) -> f64
    {
        // use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        
        let mut performance_samples = Vec::with_capacity(self.num_samples);
        
        for _ in 0..self.num_samples {
            // Generate parameter variation
            let mut varied_params = nominal_params.clone();
            
            for (i, param) in varied_params.iter_mut().enumerate() {
                let param_name = format!("param_{}", i);
                if let Some(&std_dev) = self.variation_model.parameter_variations.get(&param_name) {
                    // let normal = Normal::new(0.0, std_dev).unwrap();
                    let variation = rng.gen_range(-0.1..0.1);
                    *param += variation;
                }
            }
            
            // Evaluate performance with variations
            let performance = evaluation_fn(&varied_params);
            performance_samples.push(performance);
        }
        
        // Calculate statistics
        let mean = performance_samples.iter().sum::<f64>() / performance_samples.len() as f64;
        let variance = performance_samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / performance_samples.len() as f64;
        let std_dev = variance.sqrt();
        
        // Calculate yield (percentage within spec)
        let nominal_performance = evaluation_fn(nominal_params);
        let spec_limit = nominal_performance * 0.9;  // 10% degradation limit
        let within_spec = performance_samples.iter()
            .filter(|&&x| x >= spec_limit)
            .count();
        let yield_rate = within_spec as f64 / self.num_samples as f64;
        
        Ok(RobustnessMetrics {
            mean_performance: mean,
            std_dev: std_dev,
            yield_rate,
            worst_case: performance_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            best_case: performance_samples.iter().cloned().fold(f64::INFINITY, f64::min),
        })
    }
}

/// Robustness analysis results
#[derive(Debug, Clone)]
pub struct RobustnessMetrics {
    pub mean_performance: f64,
    pub std_dev: f64,
    pub yield_rate: f64,
    pub worst_case: f64,
    pub best_case: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_codesign_optimizer_creation() {
        let constraints = CoDesignConstraints::default();
        let optimizer = CoDesignOptimizer::new(constraints);
        
        assert_eq!(optimizer.population_size, 100);
        assert_eq!(optimizer.num_generations, 500);
    }
    
    #[test]
    fn test_robustness_analyzer() {
        let analyzer = RobustnessAnalyzer::new(1000);
        assert_eq!(analyzer.num_samples, 1000);
    }
}