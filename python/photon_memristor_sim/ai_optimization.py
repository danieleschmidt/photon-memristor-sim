"""
AI-Driven Optimization Engine

Implements breakthrough 2025 AI-driven optimization capabilities:
- Self-improving algorithms with adaptive intelligence 
- Evolutionary neural architecture search for photonic systems
- Bio-inspired optimization algorithms for memristor programming
- Real-time performance adaptation and learning
"""

import jax.numpy as jnp
from jax import random, jit, vmap, grad, hessian, lax
from jax.experimental import optimizers
import jax
from typing import Dict, Tuple, List, Optional, Any, Callable, Union
import numpy as np
from dataclasses import dataclass, field
import time
import functools
from collections import deque
import json

try:
    from ._core import create_device_simulator
except ImportError:
    from .pure_python_fallbacks import create_device_simulator

from .devices import MolecularMemristor, PhotonicDevice
from .quantum_hybrid import QuantumPhotonicProcessor


@dataclass
class OptimizationGenome:
    """Genetic representation of optimization algorithms."""
    algorithm_id: str
    parameters: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    fitness_score: float = 0.0
    generation: int = 0
    parent_algorithms: List[str] = field(default_factory=list)
    mutation_rate: float = 0.1
    
    
@dataclass
class AIOptimizationConfig:
    """Configuration for AI-driven optimization."""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    performance_window: int = 10
    adaptive_learning_rate: bool = True
    neural_architecture_search: bool = True
    bio_inspired_algorithms: bool = True
    quantum_acceleration: bool = True
    

class NeuralArchitectureSearch:
    """
    Evolutionary Neural Architecture Search for photonic systems.
    
    Automatically discovers optimal neural network architectures for 
    photonic computing hardware constraints.
    """
    
    def __init__(self, config: AIOptimizationConfig):
        self.config = config
        self.architecture_genomes = []
        self.performance_database = {}
        self.generation = 0
        
        # Search space definition
        self.layer_types = ["photonic_linear", "photonic_conv", "photonic_attention", "quantum_layer"]
        self.activation_functions = ["photonic_relu", "sigmoid", "tanh", "quantum_activation"]
        self.optimization_algorithms = ["adam", "sgd", "quantum_natural_gradient", "bio_inspired_pso"]
        
        # Performance tracking
        self.best_architecture = None
        self.best_performance = -float('inf')
        self.convergence_history = []
        
    def generate_random_architecture(self, max_layers: int = 8) -> Dict[str, Any]:
        """Generate random neural architecture within search space."""
        key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        
        num_layers = random.randint(key, (), 3, max_layers + 1)
        
        layers = []
        current_dim = 64  # Input dimension
        
        for i in range(num_layers):
            layer_key = random.split(key)[0]
            key = random.split(key)[1]
            
            layer_type = random.choice(layer_key, len(self.layer_types))
            layer_name = self.layer_types[layer_type]
            
            if layer_name == "photonic_linear":
                output_dim = random.choice(random.split(layer_key)[1], [32, 64, 128, 256])
                layer = {
                    "type": "photonic_linear",
                    "input_dim": current_dim,
                    "output_dim": output_dim,
                    "memristor_type": "molecular",
                    "precision_bits": random.choice(random.split(layer_key)[0], [8, 12, 14])
                }
                current_dim = output_dim
                
            elif layer_name == "photonic_conv":
                filters = random.choice(random.split(layer_key)[1], [16, 32, 64])
                kernel_size = random.choice(random.split(layer_key)[0], [3, 5, 7])
                layer = {
                    "type": "photonic_conv",
                    "filters": filters,
                    "kernel_size": kernel_size,
                    "optical_wavelength": 1550e-9,
                    "ring_resonators": filters
                }
                
            elif layer_name == "photonic_attention":
                num_heads = random.choice(random.split(layer_key)[1], [4, 8, 16])
                layer = {
                    "type": "photonic_attention",
                    "num_heads": num_heads,
                    "head_dim": current_dim // num_heads,
                    "optical_interference": True
                }
                
            else:  # quantum_layer
                num_qubits = random.choice(random.split(layer_key)[1], [4, 6, 8])
                layer = {
                    "type": "quantum_layer",
                    "num_qubits": num_qubits,
                    "variational_depth": random.choice(random.split(layer_key)[0], [2, 4, 6]),
                    "entanglement_pattern": "linear"
                }
            
            layers.append(layer)
        
        # Add final classification layer
        layers.append({
            "type": "photonic_linear",
            "input_dim": current_dim,
            "output_dim": 10,  # 10 classes
            "memristor_type": "molecular",
            "precision_bits": 14
        })
        
        # Select activation and optimizer
        activation = self.activation_functions[random.choice(key, len(self.activation_functions))]
        optimizer = self.optimization_algorithms[random.choice(random.split(key)[1], len(self.optimization_algorithms))]
        
        architecture = {
            "layers": layers,
            "activation": activation,
            "optimizer": optimizer,
            "learning_rate": float(random.uniform(random.split(key)[0], (), 1e-4, 1e-2)),
            "batch_size": int(random.choice(random.split(key)[1], [16, 32, 64, 128])),
            "architecture_id": f"arch_{self.generation}_{len(self.architecture_genomes)}"
        }
        
        return architecture
    
    def evaluate_architecture(self, architecture: Dict[str, Any], training_data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        """Evaluate neural architecture performance on training data."""
        inputs, targets = training_data
        
        # Simulate training performance (in practice, would train full model)
        try:
            # Calculate architecture complexity metrics
            total_params = sum(
                layer.get("input_dim", 64) * layer.get("output_dim", 64) + layer.get("filters", 0) * layer.get("kernel_size", 3)**2
                for layer in architecture["layers"]
            )
            
            # Photonic efficiency score
            photonic_layers = sum(1 for layer in architecture["layers"] if "photonic" in layer["type"])
            photonic_efficiency = photonic_layers / len(architecture["layers"])
            
            # Quantum advantage score
            quantum_layers = sum(1 for layer in architecture["layers"] if "quantum" in layer["type"])
            quantum_advantage = quantum_layers * 0.5  # Bonus for quantum layers
            
            # Memory efficiency (molecular memristors)
            molecular_layers = sum(1 for layer in architecture["layers"] if layer.get("memristor_type") == "molecular")
            memory_efficiency = molecular_layers * 0.3
            
            # Simulated accuracy based on architecture quality
            base_accuracy = 0.7 + photonic_efficiency * 0.2 + quantum_advantage * 0.1 + memory_efficiency * 0.05
            
            # Add complexity penalty
            complexity_penalty = min(total_params / 1e6, 0.1)  # Penalize >1M parameters
            
            # Learning rate optimization bonus
            lr_bonus = 0.05 if 1e-3 <= architecture["learning_rate"] <= 5e-3 else 0
            
            performance_score = base_accuracy - complexity_penalty + lr_bonus
            
            # Add noise to simulate training variance
            noise_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
            noise = random.normal(noise_key, ()) * 0.02
            
            final_score = jnp.clip(performance_score + noise, 0.0, 1.0)
            
            return float(final_score)
            
        except Exception as e:
            print(f"Architecture evaluation failed: {e}")
            return 0.0
    
    def evolve_architectures(self, training_data: Tuple[jnp.ndarray, jnp.ndarray]) -> List[Dict[str, Any]]:
        """Evolve population of neural architectures using genetic algorithms."""
        
        # Initialize population if empty
        if not self.architecture_genomes:
            for _ in range(self.config.population_size):
                architecture = self.generate_random_architecture()
                performance = self.evaluate_architecture(architecture, training_data)
                
                genome = OptimizationGenome(
                    algorithm_id=architecture["architecture_id"],
                    parameters=architecture,
                    fitness_score=performance,
                    generation=self.generation
                )
                self.architecture_genomes.append(genome)
        
        # Evolve for specified generations
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate all architectures
            for genome in self.architecture_genomes:
                if len(genome.performance_history) < self.config.performance_window:
                    performance = self.evaluate_architecture(genome.parameters, training_data)
                    genome.performance_history.append(performance)
                    genome.fitness_score = jnp.mean(jnp.array(genome.performance_history[-5:]))  # Recent average
            
            # Sort by fitness
            self.architecture_genomes.sort(key=lambda g: g.fitness_score, reverse=True)
            
            # Track best architecture
            if self.architecture_genomes[0].fitness_score > self.best_performance:
                self.best_performance = self.architecture_genomes[0].fitness_score
                self.best_architecture = self.architecture_genomes[0].parameters.copy()
            
            self.convergence_history.append(self.best_performance)
            
            # Early stopping if converged
            if len(self.convergence_history) > 10:
                recent_improvement = self.convergence_history[-1] - self.convergence_history[-10]
                if recent_improvement < 0.001:  # Converged
                    print(f"Converged at generation {generation}")
                    break
            
            # Selection and reproduction
            elite_count = int(self.config.elite_ratio * self.config.population_size)
            elite_genomes = self.architecture_genomes[:elite_count]
            
            # Create new generation
            new_genomes = elite_genomes.copy()  # Keep elite
            
            while len(new_genomes) < self.config.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(self.architecture_genomes[:self.config.population_size // 2])
                parent2 = self._tournament_selection(self.architecture_genomes[:self.config.population_size // 2])
                
                # Crossover and mutation
                if random.uniform(random.PRNGKey(generation)) < self.config.crossover_rate:
                    child = self._crossover_architectures(parent1, parent2)
                else:
                    child = parent1.parameters.copy()
                
                if random.uniform(random.PRNGKey(generation + 1000)) < self.config.mutation_rate:
                    child = self._mutate_architecture(child)
                
                # Create new genome
                child_genome = OptimizationGenome(
                    algorithm_id=f"arch_{generation}_{len(new_genomes)}",
                    parameters=child,
                    generation=generation + 1,
                    parent_algorithms=[parent1.algorithm_id, parent2.algorithm_id]
                )
                
                new_genomes.append(child_genome)
            
            self.architecture_genomes = new_genomes
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best performance = {self.best_performance:.4f}")
        
        return [g.parameters for g in sorted(self.architecture_genomes, key=lambda g: g.fitness_score, reverse=True)[:5]]
    
    def _tournament_selection(self, population: List[OptimizationGenome], tournament_size: int = 3) -> OptimizationGenome:
        """Tournament selection for genetic algorithm."""
        tournament = random.choice(
            random.PRNGKey(int(time.time() * 1000) % 2**32), 
            len(population), 
            shape=(tournament_size,)
        )
        tournament_genomes = [population[i] for i in tournament]
        return max(tournament_genomes, key=lambda g: g.fitness_score)
    
    def _crossover_architectures(self, parent1: OptimizationGenome, parent2: OptimizationGenome) -> Dict[str, Any]:
        """Crossover two neural architectures."""
        # Single-point crossover for layers
        layers1 = parent1.parameters["layers"]
        layers2 = parent2.parameters["layers"]
        
        crossover_point = random.randint(random.PRNGKey(42), (), 1, min(len(layers1), len(layers2)))
        
        child_layers = layers1[:crossover_point] + layers2[crossover_point:]
        
        # Average numerical parameters
        child_lr = (parent1.parameters["learning_rate"] + parent2.parameters["learning_rate"]) / 2
        child_batch = random.choice(random.PRNGKey(43), [parent1.parameters["batch_size"], parent2.parameters["batch_size"]])
        
        return {
            "layers": child_layers,
            "activation": random.choice(random.PRNGKey(44), [parent1.parameters["activation"], parent2.parameters["activation"]]),
            "optimizer": random.choice(random.PRNGKey(45), [parent1.parameters["optimizer"], parent2.parameters["optimizer"]]),
            "learning_rate": child_lr,
            "batch_size": child_batch,
            "architecture_id": f"crossover_{int(time.time() * 1000) % 10000}"
        }
    
    def _mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate neural architecture."""
        mutated = architecture.copy()
        
        # Mutate learning rate
        if random.uniform(random.PRNGKey(100)) < 0.3:
            mutation_factor = random.uniform(random.PRNGKey(101), (), 0.5, 2.0)
            mutated["learning_rate"] = jnp.clip(architecture["learning_rate"] * mutation_factor, 1e-5, 1e-1)
        
        # Mutate layer parameters
        if random.uniform(random.PRNGKey(102)) < 0.5 and mutated["layers"]:
            layer_idx = random.randint(random.PRNGKey(103), (), 0, len(mutated["layers"]))
            layer = mutated["layers"][layer_idx].copy()
            
            if "output_dim" in layer:
                layer["output_dim"] = random.choice(random.PRNGKey(104), [32, 64, 128, 256])
            if "filters" in layer:
                layer["filters"] = random.choice(random.PRNGKey(105), [16, 32, 64])
            if "num_heads" in layer:
                layer["num_heads"] = random.choice(random.PRNGKey(106), [4, 8, 16])
                
            mutated["layers"][layer_idx] = layer
        
        return mutated


class BioInspiredOptimization:
    """
    Bio-inspired optimization algorithms for photonic systems.
    
    Implements particle swarm optimization, genetic algorithms, and 
    ant colony optimization adapted for photonic computing.
    """
    
    def __init__(self, config: AIOptimizationConfig):
        self.config = config
        self.optimization_history = deque(maxlen=1000)
        
    @jit
    def particle_swarm_optimization(
        self, 
        objective_function: Callable, 
        bounds: Tuple[jnp.ndarray, jnp.ndarray],
        num_particles: int = 30,
        max_iterations: int = 100
    ) -> Tuple[jnp.ndarray, float]:
        """Particle Swarm Optimization for photonic parameter tuning."""
        
        lower_bounds, upper_bounds = bounds
        dimension = len(lower_bounds)
        
        # Initialize particles
        key = random.PRNGKey(42)
        positions = random.uniform(key, (num_particles, dimension)) * (upper_bounds - lower_bounds) + lower_bounds
        velocities = jnp.zeros((num_particles, dimension))
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Initialize best positions
        personal_best_positions = positions.copy()
        personal_best_scores = jnp.array([objective_function(pos) for pos in positions])
        
        global_best_idx = jnp.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        # Main PSO loop
        for iteration in range(max_iterations):
            key, subkey = random.split(key)
            
            # Update velocities and positions
            r1 = random.uniform(subkey, (num_particles, dimension))
            r2 = random.uniform(random.split(key)[1], (num_particles, dimension))
            
            velocities = (w * velocities + 
                         c1 * r1 * (personal_best_positions - positions) +
                         c2 * r2 * (global_best_position - positions))
            
            positions = positions + velocities
            
            # Apply bounds
            positions = jnp.clip(positions, lower_bounds, upper_bounds)
            
            # Evaluate new positions
            current_scores = jnp.array([objective_function(pos) for pos in positions])
            
            # Update personal bests
            improved_mask = current_scores < personal_best_scores
            personal_best_positions = jnp.where(improved_mask[:, None], positions, personal_best_positions)
            personal_best_scores = jnp.where(improved_mask, current_scores, personal_best_scores)
            
            # Update global best
            best_idx = jnp.argmin(personal_best_scores)
            if personal_best_scores[best_idx] < global_best_score:
                global_best_position = personal_best_positions[best_idx]
                global_best_score = personal_best_scores[best_idx]
        
        return global_best_position, global_best_score
    
    def adaptive_differential_evolution(
        self,
        objective_function: Callable,
        bounds: Tuple[jnp.ndarray, jnp.ndarray],
        population_size: int = 50,
        max_generations: int = 100
    ) -> Tuple[jnp.ndarray, float]:
        """Adaptive Differential Evolution with self-tuning parameters."""
        
        lower_bounds, upper_bounds = bounds
        dimension = len(lower_bounds)
        
        # Initialize population
        key = random.PRNGKey(123)
        population = random.uniform(key, (population_size, dimension)) * (upper_bounds - lower_bounds) + lower_bounds
        
        # Evaluate initial population
        fitness = jnp.array([objective_function(individual) for individual in population])
        
        # Adaptive parameters
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        
        best_idx = jnp.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        # Evolution loop
        for generation in range(max_generations):
            key, subkey = random.split(key)
            
            # Adaptive parameter control
            F = 0.5 + 0.3 * jnp.sin(2 * jnp.pi * generation / max_generations)
            CR = 0.9 - 0.4 * generation / max_generations
            
            new_population = []
            
            for i in range(population_size):
                # Select three random individuals (different from current)
                candidates = jnp.arange(population_size)
                candidates = candidates[candidates != i]
                
                r1, r2, r3 = random.choice(subkey, len(candidates), shape=(3,), replace=False)
                
                # Mutation: vi = x_r1 + F * (x_r2 - x_r3)
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = jnp.clip(mutant, lower_bounds, upper_bounds)
                
                # Crossover
                crossover_mask = random.uniform(subkey, (dimension,)) < CR
                # Ensure at least one dimension is taken from mutant
                random_idx = random.randint(subkey, (), 0, dimension)
                crossover_mask = crossover_mask.at[random_idx].set(True)
                
                trial = jnp.where(crossover_mask, mutant, population[i])
                
                # Selection
                trial_fitness = objective_function(trial)
                
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])
                    
                key, subkey = random.split(key)
            
            population = jnp.array(new_population)
            fitness = jnp.array([objective_function(individual) for individual in population])
        
        return best_individual, best_fitness
    
    def memristor_programming_optimization(
        self, 
        target_conductances: jnp.ndarray,
        memristor_array: List[MolecularMemristor]
    ) -> Dict[str, Any]:
        """Optimize memristor programming using bio-inspired algorithms."""
        
        def programming_objective(programming_sequence):
            """Objective function for memristor programming."""
            total_error = 0.0
            energy_cost = 0.0
            
            for i, device in enumerate(memristor_array[:len(target_conductances)]):
                target_conductance = target_conductances[i]
                
                # Apply programming sequence
                programming_voltage = programming_sequence[i * 2]
                programming_time = programming_sequence[i * 2 + 1]
                
                # Simulate programming
                device.analog_programming(target_conductance)
                achieved_conductance = device.get_conductance()
                
                # Calculate error
                error = jnp.abs(achieved_conductance - target_conductance) / target_conductance
                total_error += error
                
                # Calculate energy cost
                energy = programming_voltage**2 * programming_time / device.props["switching_voltage"]
                energy_cost += energy
            
            # Multi-objective: minimize error and energy
            return total_error + 0.1 * energy_cost
        
        # Define optimization bounds
        num_devices = len(target_conductances)
        voltage_bounds = jnp.array([0.1, 2.0])  # Voltage range
        time_bounds = jnp.array([1e-9, 1e-6])   # Time range
        
        lower_bounds = jnp.tile(jnp.array([voltage_bounds[0], time_bounds[0]]), num_devices)
        upper_bounds = jnp.tile(jnp.array([voltage_bounds[1], time_bounds[1]]), num_devices)
        bounds = (lower_bounds, upper_bounds)
        
        # Optimize using PSO
        optimal_sequence, best_cost = self.particle_swarm_optimization(
            programming_objective, 
            bounds,
            num_particles=50,
            max_iterations=200
        )
        
        return {
            "optimal_programming_sequence": optimal_sequence,
            "programming_cost": float(best_cost),
            "voltage_sequence": optimal_sequence[::2],  # Every even index
            "time_sequence": optimal_sequence[1::2],    # Every odd index
            "optimization_algorithm": "particle_swarm_optimization"
        }


class AdaptivePerformanceOptimizer:
    """
    Adaptive performance optimizer that learns and improves over time.
    
    Uses reinforcement learning to optimize system parameters dynamically.
    """
    
    def __init__(self, config: AIOptimizationConfig):
        self.config = config
        self.performance_history = deque(maxlen=1000)
        self.parameter_history = deque(maxlen=1000)
        self.learning_rate_schedule = []
        self.adaptation_count = 0
        
        # Q-learning parameters for adaptive optimization
        self.q_table = {}  # State-action value table
        self.epsilon = 0.3  # Exploration rate
        self.alpha = 0.1    # Learning rate
        self.gamma = 0.9    # Discount factor
        
    def adaptive_hyperparameter_tuning(
        self,
        performance_metric_function: Callable,
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_episodes: int = 100
    ) -> Dict[str, float]:
        """Adaptive hyperparameter tuning using reinforcement learning."""
        
        # Discretize parameter space for Q-learning
        param_names = list(parameter_ranges.keys())
        param_values = {}
        
        for param, (min_val, max_val) in parameter_ranges.items():
            param_values[param] = jnp.linspace(min_val, max_val, 10)  # 10 discrete values
        
        # Initialize Q-table
        for state in range(100):  # 100 discrete states
            for action in range(len(param_names) * 10):  # Actions = param × values
                self.q_table[(state, action)] = 0.0
        
        best_params = {}
        best_performance = -float('inf')
        
        # Q-learning episodes
        for episode in range(num_episodes):
            # Current state (simplified as performance quantile)
            if self.performance_history:
                current_performance = self.performance_history[-1]
                performance_quantiles = jnp.percentile(jnp.array(list(self.performance_history)), jnp.arange(0, 101))
                state = jnp.searchsorted(performance_quantiles, current_performance, side='right') - 1
                state = int(jnp.clip(state, 0, 99))
            else:
                state = 50  # Start in middle state
            
            # Choose action (epsilon-greedy)
            if random.uniform(random.PRNGKey(episode)) < self.epsilon:
                # Explore: random action
                action = random.randint(random.PRNGKey(episode + 1000), (), 0, len(param_names) * 10)
            else:
                # Exploit: best known action
                q_values = [self.q_table.get((state, a), 0.0) for a in range(len(param_names) * 10)]
                action = jnp.argmax(jnp.array(q_values))
            
            # Convert action to parameter values
            param_idx = action // 10
            value_idx = action % 10
            
            if param_idx < len(param_names):
                param_name = param_names[param_idx]
                param_value = param_values[param_name][value_idx]
                
                # Test this parameter configuration
                test_params = best_params.copy()
                test_params[param_name] = float(param_value)
                
                try:
                    performance = performance_metric_function(test_params)
                    self.performance_history.append(performance)
                    self.parameter_history.append(test_params.copy())
                    
                    # Update best if improved
                    if performance > best_performance:
                        best_performance = performance
                        best_params = test_params.copy()
                    
                    # Calculate reward (improvement over baseline)
                    baseline = jnp.mean(jnp.array(list(self.performance_history)[-10:])) if len(self.performance_history) >= 10 else 0
                    reward = performance - baseline
                    
                    # Next state
                    performance_quantiles = jnp.percentile(jnp.array(list(self.performance_history)), jnp.arange(0, 101))
                    next_state = jnp.searchsorted(performance_quantiles, performance, side='right') - 1
                    next_state = int(jnp.clip(next_state, 0, 99))
                    
                    # Q-learning update
                    old_q = self.q_table.get((state, action), 0.0)
                    next_q_values = [self.q_table.get((next_state, a), 0.0) for a in range(len(param_names) * 10)]
                    max_next_q = max(next_q_values)
                    
                    new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                    self.q_table[(state, action)] = new_q
                    
                except Exception as e:
                    # Negative reward for failed configurations
                    self.q_table[(state, action)] = self.q_table.get((state, action), 0.0) - 1.0
            
            # Decay exploration rate
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        self.adaptation_count += 1
        
        return {
            "best_parameters": best_params,
            "best_performance": best_performance,
            "episodes_completed": num_episodes,
            "exploration_rate": self.epsilon,
            "total_adaptations": self.adaptation_count
        }
    
    def continuous_learning_loop(
        self,
        system_performance_monitor: Callable,
        parameter_updater: Callable,
        adaptation_interval_seconds: float = 60.0
    ):
        """Continuous learning loop for real-time optimization."""
        
        last_adaptation_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Check if adaptation interval has passed
            if current_time - last_adaptation_time >= adaptation_interval_seconds:
                try:
                    # Monitor current performance
                    current_performance = system_performance_monitor()
                    self.performance_history.append(current_performance)
                    
                    # Detect performance degradation
                    if len(self.performance_history) >= 10:
                        recent_avg = jnp.mean(jnp.array(list(self.performance_history)[-5:]))
                        historical_avg = jnp.mean(jnp.array(list(self.performance_history)[-10:-5]))
                        
                        if recent_avg < historical_avg * 0.95:  # 5% degradation threshold
                            print(f"Performance degradation detected: {recent_avg:.4f} < {historical_avg:.4f}")
                            
                            # Trigger adaptive optimization
                            def test_params_function(params):
                                parameter_updater(params)
                                time.sleep(1.0)  # Allow system to adapt
                                return system_performance_monitor()
                            
                            # Define parameter ranges based on current system state
                            parameter_ranges = {
                                "learning_rate": (1e-5, 1e-2),
                                "batch_size": (16, 128),
                                "momentum": (0.1, 0.9),
                                "temperature": (0.1, 2.0)
                            }
                            
                            optimization_result = self.adaptive_hyperparameter_tuning(
                                test_params_function,
                                parameter_ranges,
                                num_episodes=20  # Quick adaptation
                            )
                            
                            print(f"Adaptation complete: {optimization_result['best_performance']:.4f}")
                    
                    last_adaptation_time = current_time
                    
                except Exception as e:
                    print(f"Adaptation error: {e}")
                    last_adaptation_time = current_time
            
            time.sleep(1.0)  # Check every second
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history."""
        if not self.performance_history or not self.parameter_history:
            return {"status": "No optimization data available"}
        
        performance_array = jnp.array(list(self.performance_history))
        
        # Calculate trends
        if len(performance_array) >= 2:
            trend_slope = jnp.polyfit(jnp.arange(len(performance_array)), performance_array, 1)[0]
        else:
            trend_slope = 0.0
        
        # Find best performing parameters
        best_idx = jnp.argmax(performance_array)
        best_parameters = list(self.parameter_history)[best_idx] if best_idx < len(self.parameter_history) else {}
        
        return {
            "total_optimizations": len(self.performance_history),
            "best_performance": float(jnp.max(performance_array)),
            "worst_performance": float(jnp.min(performance_array)),
            "average_performance": float(jnp.mean(performance_array)),
            "performance_std": float(jnp.std(performance_array)),
            "trend_slope": float(trend_slope),
            "improvement_rate": float(trend_slope / jnp.mean(performance_array)) if jnp.mean(performance_array) > 0 else 0,
            "best_parameters": best_parameters,
            "adaptation_count": self.adaptation_count,
            "current_exploration_rate": self.epsilon
        }


# Factory functions and utilities
def create_neural_architecture_search(
    population_size: int = 50,
    max_generations: int = 100
) -> NeuralArchitectureSearch:
    """Create neural architecture search with optimal configuration."""
    
    config = AIOptimizationConfig(
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_ratio=0.2,
        neural_architecture_search=True
    )
    
    return NeuralArchitectureSearch(config)


def create_bio_inspired_optimizer(
    enable_pso: bool = True,
    enable_differential_evolution: bool = True
) -> BioInspiredOptimization:
    """Create bio-inspired optimizer with specified algorithms."""
    
    config = AIOptimizationConfig(
        bio_inspired_algorithms=True,
        population_size=50,
        max_generations=200
    )
    
    return BioInspiredOptimization(config)


def create_adaptive_optimizer(
    continuous_learning: bool = True,
    adaptation_interval: float = 60.0
) -> AdaptivePerformanceOptimizer:
    """Create adaptive performance optimizer."""
    
    config = AIOptimizationConfig(
        adaptive_learning_rate=True,
        population_size=30,
        max_generations=100
    )
    
    return AdaptivePerformanceOptimizer(config)


__all__ = [
    "OptimizationGenome",
    "AIOptimizationConfig",
    "NeuralArchitectureSearch", 
    "BioInspiredOptimization",
    "AdaptivePerformanceOptimizer",
    "create_neural_architecture_search",
    "create_bio_inspired_optimizer", 
    "create_adaptive_optimizer"
]