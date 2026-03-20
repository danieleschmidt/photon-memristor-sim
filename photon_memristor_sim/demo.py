"""
demo.py — Photon-Memristor Simulator Demo

Demonstrates a small two-layer network forward pass using:
- CrossbarArray for analog weight storage and VMM
- PhotonicLink for optical interconnects between layers
- PCMWeightUpdate for programming target weights
"""

import numpy as np

from .crossbar import CrossbarArray
from .photonic import PhotonicLink
from .pcm import PCMWeightUpdate
from .memristor import MemristorDevice


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def run_demo(verbose: bool = True) -> dict:
    """
    Run a small 2-layer network demo.

    Architecture:
        Input (4) → [Crossbar 4x8] → [PhotonicLink] → [Crossbar 8x3] → Output (3)

    Returns
    -------
    dict with 'layer1_out', 'layer2_out', 'probabilities'
    """
    np.random.seed(42)

    if verbose:
        print("=" * 60)
        print("Photon-Memristor Simulator — Network Forward Pass Demo")
        print("=" * 60)

    # --- Layer 1: 4 inputs → 8 neurons ---
    W1_target = np.random.uniform(0.0, 1.0, (4, 8))
    layer1 = CrossbarArray(rows=4, cols=8, Ron=1e3, Roff=1e5)

    # Program weights using PCM
    pcm = PCMWeightUpdate(pulse_amplitude=1.0, pulse_width=100e-9, tolerance=0.01)
    G_on = 1.0 / 1e3
    G_off = 1.0 / 1e5
    G1_target = G_off + W1_target * (G_on - G_off)
    layer1.set_conductances(G1_target)

    if verbose:
        print(f"\nLayer 1: CrossbarArray 4x8")
        print(f"  {layer1}")

    # --- Photonic link between layers ---
    link = PhotonicLink(insertion_loss_db=2.5, coupling_efficiency=0.85)

    if verbose:
        print(f"\nPhotonic Interconnect:")
        print(f"  {link}")
        print(f"  Total efficiency: {link.total_efficiency:.4f}")

    # --- Layer 2: 8 neurons → 3 outputs ---
    W2_target = np.random.uniform(0.0, 1.0, (8, 3))
    layer2 = CrossbarArray(rows=8, cols=3, Ron=1e3, Roff=1e5)
    G2_target = G_off + W2_target * (G_on - G_off)
    layer2.set_conductances(G2_target)

    if verbose:
        print(f"\nLayer 2: CrossbarArray 8x3")
        print(f"  {layer2}")

    # --- Forward pass ---
    x_input = np.array([0.8, 0.3, 0.6, 0.1])  # 4-dim input
    if verbose:
        print(f"\nInput: {x_input}")

    # Layer 1 VMM
    layer1_out = layer1.vmm(x_input)

    # Apply ReLU activation
    layer1_out = np.maximum(layer1_out, 0)

    # Simulate optical transmission: scale by photonic link efficiency
    # (each neuron's signal passes through one photonic link)
    optical_power_in = layer1_out / layer1_out.max() if layer1_out.max() > 0 else layer1_out
    optical_power_out = link.transmit(optical_power_in)

    if verbose:
        print(f"\nLayer 1 output (after ReLU): {layer1_out.round(4)}")
        print(f"After photonic link (normalized): {optical_power_out.round(4)}")

    # Layer 2 VMM
    layer2_out = layer2.vmm(optical_power_out)

    # Softmax for classification
    probs = softmax(layer2_out)

    if verbose:
        print(f"\nLayer 2 output (logits): {layer2_out.round(4)}")
        print(f"Output probabilities: {probs.round(4)}")
        print(f"Predicted class: {np.argmax(probs)}")
        print("\n" + "=" * 60)
        print("Demo complete.")
        print("=" * 60)

    return {
        "layer1_out": layer1_out,
        "layer2_out": layer2_out,
        "probabilities": probs,
        "predicted_class": int(np.argmax(probs)),
    }


def run_pcm_programming_demo(verbose: bool = True) -> dict:
    """
    Demonstrate PCM iterative weight programming convergence.

    Returns convergence statistics.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PCM Weight Programming Demo")
        print("=" * 60)

    device = MemristorDevice(Ron=1e3, Roff=1e5, w_init=0.0)
    pcm = PCMWeightUpdate(
        pulse_amplitude=0.5,
        pulse_width=50e-9,
        tolerance=0.02,
        max_iterations=200,
    )

    G_on = 1.0 / device.Ron
    G_off = 1.0 / device.Roff
    targets = [0.25, 0.5, 0.75]  # fractions of G range
    results = []

    for frac in targets:
        device.reset(w=0.0)
        target_g = G_off + frac * (G_on - G_off)
        achieved_g, n_pulses, converged = pcm.program_weight(device, target_g)
        error_pct = abs(achieved_g - target_g) / (G_on - G_off) * 100

        result = {
            "target_fraction": frac,
            "target_G": target_g,
            "achieved_G": achieved_g,
            "error_pct": error_pct,
            "pulses": n_pulses,
            "converged": converged,
        }
        results.append(result)

        if verbose:
            status = "✓" if converged else "✗"
            print(
                f"  Target {frac:.2f}: G={achieved_g:.4e} S  "
                f"(err={error_pct:.2f}%, {n_pulses} pulses) {status}"
            )

    return {"pcm_results": results}


if __name__ == "__main__":
    run_demo(verbose=True)
    run_pcm_programming_demo(verbose=True)
