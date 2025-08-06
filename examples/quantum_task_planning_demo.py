#!/usr/bin/env python3
"""
Quantum-Inspired Task Planning Demo

This example demonstrates how to use the quantum-inspired task planner
for optimizing resource allocation in photonic neural networks.

The quantum approach uses superposition and interference principles to
explore the solution space more efficiently than classical algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import photon_memristor_sim as pms
    from photon_memristor_sim.quantum_planning import (
        QuantumTaskPlanner,
        PhotonicTaskPlannerFactory, 
        TaskAssignment,
        benchmark_quantum_vs_classical
    )
except ImportError:
    print("Warning: photon_memristor_sim not available. Running in demo mode.")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
    from photon_memristor_sim.quantum_planning import (
        QuantumTaskPlanner,
        PhotonicTaskPlannerFactory,
        TaskAssignment,
        benchmark_quantum_vs_classical
    )

def create_photonic_neural_network_tasks() -> List[Dict]:
    """Create a realistic set of tasks for a photonic neural network."""
    tasks = [
        {
            "name": "Input Optical Coupling",
            "resource_requirements": [0.8, 0.2, 0.1, 0.3],  # [power, bandwidth, memory, compute]
            "execution_time": 1.5,
            "priority": 0.9,
            "dependencies": []
        },
        {
            "name": "Weight Matrix Multiplication",
            "resource_requirements": [0.6, 0.9, 0.7, 0.8],
            "execution_time": 3.2,
            "priority": 0.95,
            "dependencies": [0]
        },
        {
            "name": "Photonic Nonlinear Activation", 
            "resource_requirements": [0.9, 0.4, 0.2, 0.6],
            "execution_time": 2.1,
            "priority": 0.8,
            "dependencies": [1]
        },
        {
            "name": "Optical Routing",
            "resource_requirements": [0.3, 0.8, 0.1, 0.4],
            "execution_time": 1.8,
            "priority": 0.7,
            "dependencies": [2]
        },
        {
            "name": "Memristor State Update",
            "resource_requirements": [0.7, 0.3, 0.9, 0.5],
            "execution_time": 2.5,
            "priority": 0.85,
            "dependencies": [1]
        },
        {
            "name": "Output Detection",
            "resource_requirements": [0.4, 0.6, 0.3, 0.7],
            "execution_time": 1.2,
            "priority": 0.9,
            "dependencies": [3, 4]
        },
        {
            "name": "Error Feedback", 
            "resource_requirements": [0.2, 0.5, 0.6, 0.8],
            "execution_time": 1.9,
            "priority": 0.6,
            "dependencies": [5]
        },
        {
            "name": "Thermal Management",
            "resource_requirements": [0.5, 0.1, 0.4, 0.3],
            "execution_time": 0.8,
            "priority": 0.7,
            "dependencies": []
        }
    ]
    return tasks

def visualize_quantum_evolution(planner: QuantumTaskPlanner, num_steps: int = 20):
    """Visualize the evolution of quantum amplitudes over time."""
    print("\n🌊 Quantum State Evolution Visualization")
    print("=" * 50)
    
    amplitude_history = []
    energy_history = []
    fidelity_history = []
    
    # Record initial state
    amplitude_history.append(np.abs(planner.amplitudes)**2)
    energy_history.append(planner.calculate_energy())
    fidelity_history.append(planner.fidelity())
    
    # Evolve the system
    for step in range(num_steps):
        planner.evolve(time_step=0.1)
        amplitude_history.append(np.abs(planner.amplitudes)**2)
        energy_history.append(planner.calculate_energy())
        fidelity_history.append(planner.fidelity())
        
        if step % 5 == 0:
            print(f"Step {step:2d}: Energy={energy_history[-1]:.3f}, Fidelity={fidelity_history[-1]:.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Amplitude evolution heatmap
    amplitude_matrix = np.array(amplitude_history).T
    im1 = axes[0, 0].imshow(amplitude_matrix, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Amplitude Evolution')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Quantum State Index')
    plt.colorbar(im1, ax=axes[0, 0], label='Probability')
    
    # Energy evolution
    axes[0, 1].plot(energy_history, 'b-', linewidth=2)
    axes[0, 1].set_title('System Energy Evolution')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quantum fidelity
    axes[1, 0].plot(fidelity_history, 'r-', linewidth=2)
    axes[1, 0].set_title('Quantum Fidelity')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Fidelity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Final state distribution
    final_probs = amplitude_history[-1]
    axes[1, 1].bar(range(len(final_probs)), final_probs, alpha=0.7)
    axes[1, 1].set_title('Final State Distribution')
    axes[1, 1].set_xlabel('Quantum State Index')
    axes[1, 1].set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('quantum_evolution.png', dpi=300, bbox_inches='tight')
    print("📊 Saved visualization to quantum_evolution.png")
    
    return amplitude_history, energy_history, fidelity_history

def demonstrate_quantum_interference():
    """Demonstrate quantum interference effects in task planning."""
    print("\n🌀 Quantum Interference Demonstration")
    print("=" * 50)
    
    # Create planner
    planner = QuantumTaskPlanner(8)
    
    # Define an optimal target pattern
    optimal_target = TaskAssignment(
        task_id=0,
        resources=[0.9, 0.8, 0.3, 0.6, 0.4, 0.2, 0.1, 0.5],  # High priority tasks get more resources
        priority=0.85,
        execution_time=1.8,
        dependencies=[]
    )
    
    print(f"Initial state fidelity: {planner.fidelity():.4f}")
    print(f"Initial energy: {planner.calculate_energy():.4f}")
    
    # Apply quantum interference
    print("\nApplying quantum interference to enhance optimal solutions...")
    planner.apply_interference(optimal_target)
    
    print(f"Post-interference fidelity: {planner.fidelity():.4f}")
    print(f"Post-interference energy: {planner.calculate_energy():.4f}")
    
    # Measure the state multiple times to see the bias
    measurements = []
    for i in range(10):
        # Reset to superposition (simulate fresh measurement)
        fresh_planner = QuantumTaskPlanner(8)
        fresh_planner.apply_interference(optimal_target)
        measurement = fresh_planner.measure()
        measurements.append(measurement)
        
        similarity = fresh_planner._calculate_similarity(measurement, optimal_target)
        print(f"Measurement {i+1}: Similarity to target = {similarity:.3f}")
    
    # Calculate average similarity
    avg_similarity = np.mean([
        planner._calculate_similarity(m, optimal_target) for m in measurements
    ])
    print(f"\nAverage similarity to target: {avg_similarity:.3f}")
    print("✨ Higher similarity indicates successful quantum enhancement!")

def run_optimization_comparison():
    """Compare quantum annealing with classical optimization."""
    print("\n⚡ Optimization Algorithm Comparison") 
    print("=" * 50)
    
    num_tasks = 8
    num_trials = 5
    
    print(f"Running {num_trials} trials with {num_tasks} tasks...")
    
    # Run comprehensive benchmark
    results = benchmark_quantum_vs_classical(num_tasks, num_trials)
    
    print("\n📊 Results Summary:")
    print(f"Quantum Approach:")
    print(f"  - Average Time: {results['quantum_time_mean']:.3f}±{results['quantum_time_std']:.3f}s")
    print(f"  - Average Energy: {results['quantum_energy_mean']:.3f}±{results['quantum_energy_std']:.3f}")
    
    print(f"\nClassical Approach:")
    print(f"  - Average Time: {results['classical_time_mean']:.3f}±{results['classical_time_std']:.3f}s")
    print(f"  - Average Energy: {results['classical_energy_mean']:.3f}±{results['classical_energy_std']:.3f}")
    
    print(f"\n🏆 Performance Metrics:")
    print(f"  - Speedup: {results['speedup']:.2f}x")
    print(f"  - Energy Improvement: {results['energy_improvement']:.1%}")
    
    if results['speedup'] > 1.0:
        print("✅ Quantum approach is faster!")
    else:
        print("⚠️  Classical approach is faster (may need parameter tuning)")
    
    if results['energy_improvement'] > 0:
        print("✅ Quantum approach finds better solutions!")
    else:
        print("⚠️  Classical approach finds better solutions")

def demonstrate_quantum_annealing():
    """Demonstrate quantum annealing optimization process."""
    print("\n🧊 Quantum Annealing Demonstration")
    print("=" * 50)
    
    # Create photonic-optimized planner
    planner = PhotonicTaskPlannerFactory.create_photonic_planner(8)
    
    print("Starting quantum annealing optimization...")
    print("(This simulates finding optimal resource allocation)")
    
    start_time = time.time()
    optimal_solution = planner.quantum_anneal(
        num_steps=100, 
        initial_temp=2.0
    )
    end_time = time.time()
    
    print(f"\n⏱️  Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Generate comprehensive report
    report = planner.generate_report()
    print("\n" + report.summary())
    
    # Analyze the solution quality
    print(f"\n🔍 Solution Analysis:")
    print(f"Task ID: {optimal_solution.task_id}")
    print(f"Resource allocation: {[f'{r:.2f}' for r in optimal_solution.resources[:5]]}")
    print(f"Priority: {optimal_solution.priority:.3f}")
    print(f"Execution time: {optimal_solution.execution_time:.2f}s")
    
    # Calculate resource utilization efficiency
    total_resources = sum(optimal_solution.resources)
    efficiency = optimal_solution.priority / optimal_solution.execution_time
    print(f"Resource utilization: {total_resources:.2f}")
    print(f"Efficiency ratio: {efficiency:.3f}")
    
    return optimal_solution, report

def demonstrate_error_correction():
    """Demonstrate quantum error correction capabilities."""
    print("\n🛠️  Quantum Error Correction Demo")
    print("=" * 50)
    
    planner = QuantumTaskPlanner(8)
    
    print(f"Initial fidelity: {planner.fidelity():.4f}")
    
    # Introduce artificial errors
    print("Introducing amplitude errors...")
    noise_level = 0.1
    planner.amplitudes += np.random.normal(0, noise_level, len(planner.amplitudes))
    planner.coherence_time *= 0.5  # Reduce coherence
    
    print(f"After noise: fidelity = {planner.fidelity():.4f}")
    
    # Apply error correction
    print("Applying quantum error correction...")
    planner.error_correction()
    
    print(f"After correction: fidelity = {planner.fidelity():.4f}")
    print("✨ Error correction helps maintain quantum advantage!")

def main():
    """Main demonstration function."""
    print("🚀 Quantum-Inspired Task Planning for Photonic Neural Networks")
    print("=" * 70)
    print("This demo showcases quantum superposition and interference principles")
    print("applied to optimize resource allocation in photonic computing systems.\n")
    
    try:
        # Run all demonstrations
        demonstrate_quantum_interference()
        
        demonstrate_quantum_annealing() 
        
        run_optimization_comparison()
        
        # Create quantum evolution visualization
        print("\n🎨 Creating quantum evolution visualization...")
        demo_planner = QuantumTaskPlanner(8)
        visualize_quantum_evolution(demo_planner, num_steps=30)
        
        demonstrate_error_correction()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey takeaways:")
        print("• Quantum superposition explores multiple solutions simultaneously")
        print("• Interference amplifies good solutions and suppresses poor ones")  
        print("• Annealing provides a systematic path to optimal solutions")
        print("• Error correction maintains quantum coherence for better performance")
        print("• The approach shows promise for photonic neural network optimization")
        
    except Exception as e:
        print(f"❌ Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)