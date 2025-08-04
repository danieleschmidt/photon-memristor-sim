"""
Visualization tools for photonic neural networks and devices

Provides 2D/3D visualization of photonic circuits, optical field distributions,
device states, and neural network performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some interactive features will be disabled.")


class PhotonicCircuitVisualizer:
    """
    Visualizer for photonic integrated circuits and neural networks.
    
    Creates 2D layout plots showing waveguides, devices, and optical routing.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize circuit visualizer."""
        self.figsize = figsize
        self.device_colors = {
            'pcm': '#FF6B6B',
            'oxide': '#4ECDC4', 
            'ring': '#45B7D1',
            'waveguide': '#96CEB4',
            'laser': '#FFEAA7',
            'detector': '#DDA0DD'
        }
        
    def plot_crossbar_array(self, size: Tuple[int, int], 
                           device_states: Optional[jnp.ndarray] = None,
                           device_type: str = 'pcm',
                           title: str = "Photonic Crossbar Array") -> plt.Figure:
        """
        Plot photonic crossbar array layout.
        
        Args:
            size: Array dimensions (rows, cols)
            device_states: Device states for color coding
            device_type: Type of devices in array
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        rows, cols = size
        device_size = 0.8
        spacing = 1.0
        
        # Plot devices
        for i in range(rows):
            for j in range(cols):
                x = j * spacing
                y = (rows - 1 - i) * spacing  # Flip y for matrix convention
                
                # Color based on device state
                if device_states is not None:
                    state = device_states[i, j]
                    color = plt.cm.viridis(state)
                else:
                    color = self.device_colors.get(device_type, 'gray')
                
                # Draw device
                device = patches.Circle((x, y), device_size/2, 
                                      facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(device)
                
                # Add state value if available
                if device_states is not None:
                    ax.text(x, y, f'{state:.2f}', ha='center', va='center', 
                           fontsize=8, fontweight='bold')
        
        # Plot input/output waveguides
        waveguide_width = 0.1
        
        # Input waveguides (left side)
        for i in range(rows):
            y = (rows - 1 - i) * spacing
            waveguide = patches.Rectangle((-spacing/2, y - waveguide_width/2), 
                                        spacing/2 - device_size/2, waveguide_width,
                                        facecolor=self.device_colors['waveguide'], 
                                        edgecolor='darkgreen')
            ax.add_patch(waveguide)
            
            # Input label
            ax.text(-spacing/2 - 0.1, y, f'In{i}', ha='right', va='center', fontweight='bold')
        
        # Output waveguides (right side) 
        for j in range(cols):
            x = j * spacing
            waveguide = patches.Rectangle((x - waveguide_width/2, -spacing/2), 
                                        waveguide_width, spacing/2 - device_size/2,
                                        facecolor=self.device_colors['waveguide'],
                                        edgecolor='darkgreen')
            ax.add_patch(waveguide)
            
            # Output label
            ax.text(x, -spacing/2 - 0.1, f'Out{j}', ha='center', va='top', fontweight='bold')
        
        # Styling
        ax.set_xlim(-spacing, cols * spacing)
        ax.set_ylim(-spacing, rows * spacing)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        
        # Add colorbar if states provided
        if device_states is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                     norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Device State', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_neural_network_layout(self, layer_sizes: List[int],
                                  device_states: Optional[Dict[str, jnp.ndarray]] = None,
                                  title: str = "Photonic Neural Network") -> plt.Figure:
        """
        Plot neural network as interconnected photonic layers.
        
        Args:
            layer_sizes: Number of neurons in each layer
            device_states: Device states for each layer
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        num_layers = len(layer_sizes)
        max_neurons = max(layer_sizes)
        
        layer_spacing = 3.0
        neuron_spacing = 1.0
        neuron_size = 0.3
        
        # Plot neurons
        for layer_idx, num_neurons in enumerate(layer_sizes):
            x = layer_idx * layer_spacing
            
            # Center neurons vertically
            start_y = (max_neurons - num_neurons) * neuron_spacing / 2
            
            for neuron_idx in range(num_neurons):
                y = start_y + neuron_idx * neuron_spacing
                
                # Color based on layer type
                if layer_idx == 0:
                    color = self.device_colors['laser']  # Input layer
                elif layer_idx == num_layers - 1:
                    color = self.device_colors['detector']  # Output layer
                else:
                    color = self.device_colors['pcm']  # Hidden layers
                
                neuron = patches.Circle((x, y), neuron_size, 
                                      facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(neuron)
        
        # Plot connections (weight matrices)
        for layer_idx in range(num_layers - 1):
            x1 = layer_idx * layer_spacing + neuron_size
            x2 = (layer_idx + 1) * layer_spacing - neuron_size
            
            num_neurons1 = layer_sizes[layer_idx]
            num_neurons2 = layer_sizes[layer_idx + 1]
            
            start_y1 = (max_neurons - num_neurons1) * neuron_spacing / 2
            start_y2 = (max_neurons - num_neurons2) * neuron_spacing / 2
            
            # Draw connection lines
            for i in range(num_neurons1):
                y1 = start_y1 + i * neuron_spacing
                for j in range(num_neurons2):
                    y2 = start_y2 + j * neuron_spacing
                    
                    # Line weight based on device state
                    if device_states and f'layer_{layer_idx}' in device_states:
                        if 'weights' in device_states[f'layer_{layer_idx}']:
                            weight = device_states[f'layer_{layer_idx}']['weights'][j, i]
                            alpha = min(abs(weight), 1.0)
                            color = 'red' if weight < 0 else 'blue'
                        else:
                            alpha = 0.3
                            color = 'gray'
                    else:
                        alpha = 0.3
                        color = 'gray'
                    
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1)
        
        # Layer labels
        for layer_idx, num_neurons in enumerate(layer_sizes):
            x = layer_idx * layer_spacing
            y = max_neurons * neuron_spacing / 2 + 0.5
            
            if layer_idx == 0:
                label = f"Input\n({num_neurons})"
            elif layer_idx == num_layers - 1:
                label = f"Output\n({num_neurons})"
            else:
                label = f"Hidden {layer_idx}\n({num_neurons})"
                
            ax.text(x, y, label, ha='center', va='bottom', fontweight='bold')
        
        # Styling
        ax.set_xlim(-0.5, (num_layers - 1) * layer_spacing + 0.5)
        ax.set_ylim(-0.5, max_neurons * neuron_spacing)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_ring_resonator_array(self, num_rings: int, 
                                 ring_states: Optional[jnp.ndarray] = None,
                                 title: str = "Ring Resonator Array") -> plt.Figure:
        """Plot array of ring resonators."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ring_radius = 0.4
        spacing = 2.0
        
        for i in range(num_rings):
            x = i * spacing
            y = 0
            
            # Ring color based on state
            if ring_states is not None:
                state = ring_states[i]
                color = plt.cm.plasma(state)
            else:
                color = self.device_colors['ring']
            
            # Draw ring
            ring = patches.Circle((x, y), ring_radius, 
                                fill=False, edgecolor=color, linewidth=3)
            ax.add_patch(ring)
            
            # Draw bus waveguide
            waveguide = patches.Rectangle((x - ring_radius - 0.2, y - 0.05), 
                                        2 * ring_radius + 0.4, 0.1,
                                        facecolor=self.device_colors['waveguide'],
                                        edgecolor='darkgreen')
            ax.add_patch(waveguide)
            
            # Ring label
            ax.text(x, y - ring_radius - 0.3, f'Ring {i}', 
                   ha='center', va='top', fontweight='bold')
        
        ax.set_xlim(-1, num_rings * spacing)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig


class FieldVisualizer:
    """
    Visualizer for optical field distributions and propagation.
    
    Creates 2D/3D plots of optical intensity, phase, and field evolution.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """Initialize field visualizer."""
        self.figsize = figsize
        
    def plot_field_intensity(self, field_amplitude: jnp.ndarray,
                           x_coords: Optional[jnp.ndarray] = None,
                           y_coords: Optional[jnp.ndarray] = None,
                           title: str = "Optical Field Intensity") -> plt.Figure:
        """
        Plot 2D optical field intensity distribution.
        
        Args:
            field_amplitude: Complex field amplitude array
            x_coords: X coordinate array
            y_coords: Y coordinate array  
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        intensity = jnp.abs(field_amplitude) ** 2
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create coordinate grids if not provided
        if x_coords is None:
            x_coords = jnp.arange(intensity.shape[1]) * 1e-6  # Default 1μm spacing
        if y_coords is None:
            y_coords = jnp.arange(intensity.shape[0]) * 1e-6
            
        X, Y = jnp.meshgrid(x_coords, y_coords)
        
        # Plot intensity
        im = ax.contourf(X * 1e6, Y * 1e6, intensity, levels=50, cmap='hot')
        
        # Styling
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity (a.u.)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_field_phase(self, field_amplitude: jnp.ndarray,
                        x_coords: Optional[jnp.ndarray] = None,
                        y_coords: Optional[jnp.ndarray] = None,
                        title: str = "Optical Field Phase") -> plt.Figure:
        """Plot 2D optical field phase distribution."""
        phase = jnp.angle(field_amplitude)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if x_coords is None:
            x_coords = jnp.arange(phase.shape[1]) * 1e-6
        if y_coords is None:
            y_coords = jnp.arange(phase.shape[0]) * 1e-6
            
        X, Y = jnp.meshgrid(x_coords, y_coords)
        
        # Plot phase
        im = ax.contourf(X * 1e6, Y * 1e6, phase, levels=50, cmap='hsv')
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Phase (rad)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_field_3d(self, field_amplitude: jnp.ndarray,
                     x_coords: Optional[jnp.ndarray] = None,
                     y_coords: Optional[jnp.ndarray] = None,
                     plot_type: str = 'intensity',
                     title: str = "3D Optical Field") -> plt.Figure:
        """Plot 3D surface of optical field."""
        if plot_type == 'intensity':
            data = jnp.abs(field_amplitude) ** 2
            colormap = 'hot'
        elif plot_type == 'phase':
            data = jnp.angle(field_amplitude)
            colormap = 'hsv'
        else:
            data = jnp.real(field_amplitude)
            colormap = 'RdBu'
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if x_coords is None:
            x_coords = jnp.arange(data.shape[1]) * 1e-6
        if y_coords is None:
            y_coords = jnp.arange(data.shape[0]) * 1e-6
            
        X, Y = jnp.meshgrid(x_coords, y_coords)
        
        # 3D surface plot
        surf = ax.plot_surface(X * 1e6, Y * 1e6, data, 
                              cmap=colormap, alpha=0.8)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel(plot_type.capitalize())
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(surf, ax=ax, shrink=0.5)
        
        return fig
    
    def plot_propagation(self, field_sequence: List[jnp.ndarray],
                        distances: List[float],
                        title: str = "Field Propagation") -> plt.Figure:
        """Plot field evolution during propagation."""
        num_steps = len(field_sequence)
        
        fig, axes = plt.subplots(2, min(num_steps, 4), figsize=(16, 8))
        if num_steps == 1:
            axes = axes.reshape(2, 1)
        
        step_indices = jnp.linspace(0, num_steps-1, min(num_steps, 4), dtype=int)
        
        for i, step_idx in enumerate(step_indices):
            field = field_sequence[step_idx]
            distance = distances[step_idx]
            
            # Intensity plot
            intensity = jnp.abs(field) ** 2
            im1 = axes[0, i].imshow(intensity, cmap='hot', aspect='equal')
            axes[0, i].set_title(f'Intensity at z={distance*1e3:.1f}mm')
            axes[0, i].set_xlabel('X')
            axes[0, i].set_ylabel('Y')
            
            # Phase plot
            phase = jnp.angle(field)
            im2 = axes[1, i].imshow(phase, cmap='hsv', aspect='equal')
            axes[1, i].set_title(f'Phase at z={distance*1e3:.1f}mm')
            axes[1, i].set_xlabel('X')
            axes[1, i].set_ylabel('Y')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


class PerformanceVisualizer:
    """
    Visualizer for neural network training and performance metrics.
    
    Creates plots for training curves, hardware metrics, and benchmarks.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize performance visualizer.""" 
        self.figsize = figsize
        
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = "Training History") -> plt.Figure:
        """Plot training curves and hardware metrics."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Loss and accuracy
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        if 'train_accuracy' in history:
            axes[0, 1].plot(history['train_accuracy'], label='Train Acc', linewidth=2)
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hardware metrics
        if 'power_consumption' in history:
            axes[1, 0].plot(history['power_consumption'], 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Power (W)')
            axes[1, 0].set_title('Power Consumption')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'device_efficiency' in history:
            axes[1, 1].plot(history['device_efficiency'], 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Efficiency (Acc/W)')
            axes[1, 1].set_title('Device Efficiency')
            axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_benchmark_comparison(self, benchmarks: Dict[str, Dict[str, float]],
                                 title: str = "Performance Comparison") -> plt.Figure:
        """Plot benchmark comparison between different implementations."""
        metrics = list(next(iter(benchmarks.values())).keys())
        systems = list(benchmarks.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot first 4 metrics
            values = [benchmarks[system][metric] for system in systems]
            
            bars = axes[i].bar(systems, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(systems)])
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel(self._get_metric_unit(metric))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2e}' if value < 0.01 else f'{value:.2f}',
                           ha='center', va='bottom')
            
            # Rotate x labels if needed
            if len(max(systems, key=len)) > 8:
                axes[i].tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _get_metric_unit(self, metric: str) -> str:
        """Get appropriate unit for metric."""
        units = {
            'throughput': 'samples/s',
            'power_consumption': 'W',
            'energy_efficiency': 'TOPS/W',
            'accuracy': '%',
            'latency': 's',
            'area': 'mm²'
        }
        return units.get(metric, '')
    
    def plot_device_state_evolution(self, state_history: Dict[str, List[jnp.ndarray]],
                                   title: str = "Device State Evolution") -> plt.Figure:
        """Plot evolution of device states during training."""
        num_layers = len(state_history)
        
        fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 6))
        if num_layers == 1:
            axes = [axes]
        
        for i, (layer_name, states) in enumerate(state_history.items()):
            # Show state distribution at different time points
            time_points = [0, len(states)//4, len(states)//2, 3*len(states)//4, -1]
            
            for j, t in enumerate(time_points):
                if t < len(states):
                    state_flat = states[t].flatten()
                    axes[i].hist(state_flat, bins=20, alpha=0.7, 
                               label=f'Epoch {t}', density=True)
            
            axes[i].set_xlabel('Device State')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{layer_name} States')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


# Interactive visualization functions (require plotly)
if PLOTLY_AVAILABLE:
    def plot_interactive_crossbar(size: Tuple[int, int], 
                                device_states: Optional[jnp.ndarray] = None,
                                title: str = "Interactive Crossbar Array"):
        """Create interactive plotly visualization of crossbar array."""
        rows, cols = size
        
        # Create hover text
        hover_text = []
        for i in range(rows):
            hover_row = []
            for j in range(cols):
                if device_states is not None:
                    state = device_states[i, j]
                    hover_row.append(f'Device ({i},{j})<br>State: {state:.3f}')
                else:
                    hover_row.append(f'Device ({i},{j})')
            hover_text.append(hover_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=device_states if device_states is not None else jnp.ones((rows, cols)),
            colorscale='Viridis',
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            colorbar=dict(title="Device State")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Column",
            yaxis_title="Row",
            width=800,
            height=600
        )
        
        return fig


def create_photonic_colormap(name: str = 'photonic') -> LinearSegmentedColormap:
    """Create custom colormap for photonic visualizations."""
    colors = ['#000428', '#004e92', '#009ffd', '#00d2ff', '#ffffff']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n_bins)
    return cmap


# Register custom colormap
plt.cm.register_cmap(name='photonic', cmap=create_photonic_colormap())


__all__ = [
    "PhotonicCircuitVisualizer",
    "FieldVisualizer", 
    "PerformanceVisualizer",
    "create_photonic_colormap",
]

if PLOTLY_AVAILABLE:
    __all__.append("plot_interactive_crossbar")