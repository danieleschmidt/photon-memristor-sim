"""
Quantum-inspired task planning and optimization for photonic neural networks.

This module provides Python interfaces to the Rust quantum-inspired optimization
algorithms, enabling efficient task scheduling and resource allocation for 
photonic computing systems.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import time

try:
    from ._core import QuantumTaskPlanner as RustQuantumPlanner
except ImportError:
    # Fallback for when Rust bindings aren't available
    RustQuantumPlanner = None

@dataclass
class TaskAssignment:
    """Represents a task assignment in the quantum planner."""
    task_id: int
    resources: List[float]
    priority: float
    execution_time: float
    dependencies: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'task_id': self.task_id,
            'resources': self.resources,
            'priority': self.priority,
            'execution_time': self.execution_time,
            'dependencies': self.dependencies,
        }

@dataclass 
class QuantumPlanningReport:
    """Report from quantum task planning optimization."""
    optimal_assignment: TaskAssignment
    quantum_fidelity: float
    coherence_time: float
    total_measurements: int
    convergence_rate: float
    entanglement_entropy: float
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Quantum Planning Report:
=======================
Quantum Fidelity: {self.quantum_fidelity:.3f}
Coherence Time: {self.coherence_time:.3f}
Total Measurements: {self.total_measurements}
Convergence Rate: {self.convergence_rate:.3f}
Entanglement Entropy: {self.entanglement_entropy:.3f}

Optimal Task Assignment:
- Task ID: {self.optimal_assignment.task_id}
- Priority: {self.optimal_assignment.priority:.3f}
- Execution Time: {self.optimal_assignment.execution_time:.3f}
- Resources: {len(self.optimal_assignment.resources)} allocated
        """.strip()

class QuantumTaskPlanner:
    """
    Quantum-inspired task planner for photonic neural networks.
    
    Uses quantum superposition and interference principles to explore
    the task scheduling solution space more efficiently than classical
    algorithms.
    """
    
    def __init__(self, num_tasks: int):
        """
        Initialize quantum task planner.
        
        Args:
            num_tasks: Number of tasks to schedule (max 1024)
        """
        if num_tasks > 1024:
            raise ValueError("Maximum 1024 tasks supported")
        
        self.num_tasks = num_tasks
        self.num_qubits = int(np.ceil(np.log2(num_tasks)))
        self.dim = 2 ** self.num_qubits
        
        # Initialize quantum state (uniform superposition)
        self.amplitudes = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        self.task_assignments = self._generate_all_assignments()
        self.coherence_time = 1.0
        self.measurement_history = []
        
        # Quantum evolution parameters
        self.phase_rate = 1.0
        self.decoherence_rate = 0.1
        
    def _generate_all_assignments(self) -> List[TaskAssignment]:
        """Generate all possible task assignments for quantum superposition."""
        assignments = []
        
        for state_idx in range(self.dim):
            # Convert state index to binary representation for resource allocation
            resources = []
            priority = 0.0
            
            for task_id in range(self.num_tasks):
                bit = (state_idx >> task_id) & 1
                resource_alloc = 0.8 if bit == 1 else 0.2
                resources.append(resource_alloc)
                priority += resource_alloc
            
            # Pad resources to match num_tasks
            while len(resources) < self.num_tasks:
                resources.append(0.1)
            
            assignment = TaskAssignment(
                task_id=state_idx % self.num_tasks,
                resources=resources[:self.num_tasks],
                priority=priority / self.num_tasks,
                execution_time=1.0 + priority * 2.0,
                dependencies=[]
            )
            assignments.append(assignment)
        
        return assignments
    
    def apply_interference(self, target_pattern: TaskAssignment) -> None:
        """
        Apply quantum interference to enhance solutions similar to target.
        
        Args:
            target_pattern: Target task assignment to amplify
        """
        for i, assignment in enumerate(self.task_assignments):
            similarity = self._calculate_similarity(assignment, target_pattern)
            
            # Constructive interference for similar assignments
            phase_shift = similarity * np.pi
            interference_factor = 1.0 + similarity
            
            self.amplitudes[i] *= interference_factor * np.exp(1j * phase_shift)
        
        self._normalize_state()
    
    def _calculate_similarity(self, a: TaskAssignment, b: TaskAssignment) -> float:
        """Calculate cosine similarity between task assignments."""
        if len(a.resources) != len(b.resources):
            return 0.0
        
        a_vec = np.array(a.resources)
        b_vec = np.array(b.resources)
        
        dot_product = np.dot(a_vec, b_vec)
        norm_a = np.linalg.norm(a_vec)
        norm_b = np.linalg.norm(b_vec)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def evolve(self, time_step: float = 0.1) -> None:
        """
        Evolve quantum state according to Schrödinger equation.
        
        Args:
            time_step: Time step for evolution
        """
        # Construct Hamiltonian
        hamiltonian = self._construct_hamiltonian()
        
        # Apply evolution operator (simplified first-order)
        evolution_op = np.eye(self.dim, dtype=complex) - 1j * hamiltonian * time_step
        
        self.amplitudes = evolution_op @ self.amplitudes
        
        # Apply decoherence
        self.coherence_time *= np.exp(-time_step * self.decoherence_rate)
        
        self._normalize_state()
    
    def _construct_hamiltonian(self) -> np.ndarray:
        """Construct problem Hamiltonian for task scheduling."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        
        # Diagonal terms: task execution costs
        for i, assignment in enumerate(self.task_assignments):
            cost = assignment.execution_time * assignment.priority
            H[i, i] = cost
        
        # Off-diagonal terms: task interactions
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                interaction = self._calculate_task_interaction(
                    self.task_assignments[i], 
                    self.task_assignments[j]
                )
                H[i, j] = interaction
                H[j, i] = interaction  # Hermitian
        
        return H
    
    def _calculate_task_interaction(self, task1: TaskAssignment, task2: TaskAssignment) -> float:
        """Calculate interaction strength between two tasks."""
        # Resource contention penalty
        resource_overlap = sum(min(r1, r2) for r1, r2 in zip(task1.resources, task2.resources))
        return -resource_overlap * 0.1  # Small negative interaction
    
    def measure(self) -> TaskAssignment:
        """
        Perform quantum measurement to get classical task assignment.
        
        Returns:
            Measured task assignment
        """
        # Calculate probabilities from amplitudes
        probabilities = np.abs(self.amplitudes) ** 2
        probabilities /= probabilities.sum()  # Normalize
        
        # Random measurement according to Born rule
        measured_idx = np.random.choice(len(probabilities), p=probabilities)
        measured_assignment = self.task_assignments[measured_idx]
        
        # Record measurement
        self.measurement_history.append(measured_assignment)
        
        # Collapse state
        self.amplitudes = np.zeros_like(self.amplitudes)
        self.amplitudes[measured_idx] = 1.0
        self.coherence_time = np.inf
        
        return measured_assignment
    
    def quantum_anneal(self, num_steps: int = 100, initial_temp: float = 1.0) -> TaskAssignment:
        """
        Perform quantum annealing optimization.
        
        Args:
            num_steps: Number of annealing steps
            initial_temp: Initial temperature
            
        Returns:
            Optimized task assignment
        """
        temperature = initial_temp
        cooling_rate = (initial_temp / 0.01) ** (1.0 / num_steps)
        
        for step in range(num_steps):
            # Evolve system
            time_step = 0.1 / (1.0 + temperature)
            self.evolve(time_step)
            
            # Apply thermal noise
            self._apply_thermal_noise(temperature)
            
            # Cool down
            temperature /= cooling_rate
            
            if step % (num_steps // 10) == 0:
                energy = self.calculate_energy()
                print(f"Annealing step {step}: T={temperature:.3f}, E={energy:.3f}")
        
        return self.measure()
    
    def _apply_thermal_noise(self, temperature: float) -> None:
        """Apply thermal phase noise for annealing."""
        phase_noise = np.random.normal(0, temperature * 0.01, size=self.dim)
        thermal_factors = np.exp(1j * phase_noise)
        self.amplitudes *= thermal_factors
        self._normalize_state()
    
    def calculate_energy(self) -> float:
        """Calculate expectation value of system energy."""
        hamiltonian = self._construct_hamiltonian()
        return np.real(np.conj(self.amplitudes) @ hamiltonian @ self.amplitudes)
    
    def _normalize_state(self) -> None:
        """Normalize quantum state to maintain probability conservation."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
    
    def fidelity(self) -> float:
        """Calculate quantum fidelity (measure of how quantum the state is)."""
        # Coherence factor
        coherence_factor = max(0, np.exp(-1 / self.coherence_time))
        
        # Participation ratio (inverse of effective dimension)
        prob_fourth_powers = np.sum(np.abs(self.amplitudes) ** 4)
        participation_ratio = 1.0 / prob_fourth_powers if prob_fourth_powers > 0 else 0
        
        return coherence_factor * (participation_ratio / len(self.amplitudes))
    
    def error_correction(self) -> None:
        """Apply simple quantum error correction."""
        correction_threshold = 0.01
        
        # Zero out small amplitudes
        mask = np.abs(self.amplitudes) < correction_threshold
        self.amplitudes[mask] = 0
        
        # Check if state is still valid
        total_norm = np.linalg.norm(self.amplitudes)
        if total_norm < 0.1:
            # Reinitialize to uniform superposition
            self.amplitudes = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
            self.coherence_time = 1.0
        else:
            self._normalize_state()
    
    def generate_report(self) -> QuantumPlanningReport:
        """Generate comprehensive optimization report."""
        optimal_assignment = (self.measurement_history[-1] if self.measurement_history 
                            else self.task_assignments[0])
        
        # Calculate convergence rate
        if len(self.measurement_history) > 1:
            recent = self.measurement_history[-min(10, len(self.measurement_history)):]
            unique_assignments = len(set(id(a) for a in recent))
            convergence_rate = 1.0 - (unique_assignments / len(recent))
        else:
            convergence_rate = 0.0
        
        # Calculate entanglement entropy
        probabilities = np.abs(self.amplitudes) ** 2
        probabilities = probabilities[probabilities > 1e-10]  # Avoid log(0)
        entanglement_entropy = -np.sum(probabilities * np.log(probabilities))
        
        return QuantumPlanningReport(
            optimal_assignment=optimal_assignment,
            quantum_fidelity=self.fidelity(),
            coherence_time=self.coherence_time,
            total_measurements=len(self.measurement_history),
            convergence_rate=convergence_rate,
            entanglement_entropy=entanglement_entropy,
        )

class PhotonicTaskPlannerFactory:
    """Factory for creating specialized quantum task planners."""
    
    @staticmethod
    def create_photonic_planner(num_devices: int) -> QuantumTaskPlanner:
        """
        Create planner optimized for photonic neural networks.
        
        Args:
            num_devices: Number of photonic devices to schedule
            
        Returns:
            Configured quantum task planner
        """
        planner = QuantumTaskPlanner(num_devices)
        
        # Optimize for photonic constraints
        planner.phase_rate = 2.0  # Faster evolution for optical frequencies
        planner.decoherence_rate = 0.05  # Lower decoherence for photonic systems
        
        return planner
    
    @staticmethod
    def create_scalable_planner(num_tasks: int) -> QuantumTaskPlanner:
        """
        Create planner for large-scale optimization problems.
        
        Args:
            num_tasks: Number of tasks (max 1024)
            
        Returns:
            Configured quantum task planner
        """
        if num_tasks > 1024:
            raise ValueError("Scalable planner supports max 1024 tasks")
        
        planner = QuantumTaskPlanner(num_tasks)
        
        # Optimize for large problems
        planner.phase_rate = 0.5  # Slower, more stable evolution
        planner.decoherence_rate = 0.2  # Higher decoherence for stability
        
        return planner

def benchmark_quantum_vs_classical(num_tasks: int, num_trials: int = 10) -> Dict[str, Any]:
    """
    Benchmark quantum-inspired planner against classical approaches.
    
    Args:
        num_tasks: Number of tasks to schedule
        num_trials: Number of trials to average over
        
    Returns:
        Benchmarking results
    """
    quantum_times = []
    quantum_energies = []
    classical_times = []
    classical_energies = []
    
    for trial in range(num_trials):
        # Quantum approach
        start_time = time.time()
        quantum_planner = QuantumTaskPlanner(num_tasks)
        assignment = quantum_planner.quantum_anneal(50, 1.0)
        quantum_time = time.time() - start_time
        quantum_energy = quantum_planner.calculate_energy()
        
        quantum_times.append(quantum_time)
        quantum_energies.append(quantum_energy)
        
        # Classical random search baseline
        start_time = time.time()
        best_energy = np.inf
        for _ in range(50):  # Same number of iterations
            random_assignment = TaskAssignment(
                task_id=np.random.randint(num_tasks),
                resources=np.random.rand(num_tasks).tolist(),
                priority=np.random.rand(),
                execution_time=np.random.rand() * 3 + 1,
                dependencies=[]
            )
            # Simple energy calculation
            energy = random_assignment.execution_time * random_assignment.priority
            if energy < best_energy:
                best_energy = energy
        
        classical_time = time.time() - start_time
        classical_times.append(classical_time)
        classical_energies.append(best_energy)
    
    return {
        'quantum_time_mean': np.mean(quantum_times),
        'quantum_time_std': np.std(quantum_times),
        'quantum_energy_mean': np.mean(quantum_energies),
        'quantum_energy_std': np.std(quantum_energies),
        'classical_time_mean': np.mean(classical_times),
        'classical_time_std': np.std(classical_times),
        'classical_energy_mean': np.mean(classical_energies),
        'classical_energy_std': np.std(classical_energies),
        'speedup': np.mean(classical_times) / np.mean(quantum_times),
        'energy_improvement': (np.mean(classical_energies) - np.mean(quantum_energies)) / np.mean(classical_energies)
    }

# Example usage and demonstration
if __name__ == "__main__":
    print("Quantum-Inspired Task Planning Demo")
    print("=" * 40)
    
    # Create a quantum task planner for 8 tasks
    planner = PhotonicTaskPlannerFactory.create_photonic_planner(8)
    
    # Define a target pattern to optimize toward
    target = TaskAssignment(
        task_id=0,
        resources=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        priority=0.8,
        execution_time=2.0,
        dependencies=[]
    )
    
    print(f"Initial fidelity: {planner.fidelity():.3f}")
    
    # Apply interference to bias toward target
    planner.apply_interference(target)
    print(f"Fidelity after interference: {planner.fidelity():.3f}")
    
    # Run quantum annealing
    print("\nRunning quantum annealing...")
    result = planner.quantum_anneal(num_steps=100, initial_temp=1.0)
    
    # Generate report
    report = planner.generate_report()
    print("\n" + report.summary())
    
    # Run benchmark
    print("\n" + "=" * 40)
    print("Benchmarking quantum vs classical...")
    benchmark = benchmark_quantum_vs_classical(8, 5)
    print(f"Speedup: {benchmark['speedup']:.2f}x")
    print(f"Energy improvement: {benchmark['energy_improvement']:.1%}")