# photon-memristor-sim

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-required-orange)](https://numpy.org)

A physics-based simulation library for **photonic-memristive computing systems** — enabling research into neuromorphic hardware that combines resistive switching memory with optical interconnects.

## Overview

This library models the core building blocks of photonic-memristive neural networks:

- **Memristive devices** — resistive switching elements with analog state (Ron/Roff model)
- **Photonic links** — optical waveguide interconnects with insertion loss and coupling efficiency
- **Crossbar arrays** — NxM grids of memristors performing analog vector-matrix multiply (VMM)
- **PCM weight programming** — iterative SET/RESET pulse scheme for phase-change material devices

## Installation

```bash
pip install numpy scipy
git clone https://github.com/danieleschmidt/photon-memristor-sim
cd photon-memristor-sim
```

## Quick Start

```python
import numpy as np
from photon_memristor_sim import MemristorDevice, CrossbarArray, PhotonicLink, PCMWeightUpdate

# Single memristor device
dev = MemristorDevice(Ron=1e3, Roff=1e5, w_init=0.5)
print(f"Conductance: {dev.conductance:.4e} S")
dev.update(voltage=1.0, dt=1e-6)  # apply 1V for 1µs

# Crossbar array (4x4) for analog VMM
cb = CrossbarArray(rows=4, cols=4)
x = np.array([1.0, 0.5, 0.8, 0.3])
output = cb.vmm(x)  # vector-matrix multiply

# Photonic interconnect
link = PhotonicLink(insertion_loss_db=3.0, coupling_efficiency=0.85)
p_out = link.transmit(power_in=1e-3)  # 1 mW input

# PCM weight programming
pcm = PCMWeightUpdate(pulse_amplitude=1.0, pulse_width=100e-9)
target_G = 5e-4  # Siemens
achieved, pulses, converged = pcm.program_weight(dev, target_G)
```

## Components

### `MemristorDevice`

Linear drift model with bounded state variable `w ∈ [0, 1]`:

```
G(w) = G_on × w + G_off × (1 - w)
dw/dt = μ × V
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Ron` | 1 kΩ | ON-state resistance |
| `Roff` | 100 kΩ | OFF-state resistance |
| `w_init` | 0.5 | Initial state variable |
| `mobility` | 1×10⁻¹⁰ | Ion drift mobility |

### `PhotonicLink`

Models a waveguide interconnect:

```
P_out = P_in × η_coupling × 10^(-IL_dB / 10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `insertion_loss_db` | 3.0 dB | Propagation + connector loss |
| `coupling_efficiency` | 0.8 | Source/detector coupling |
| `wavelength_nm` | 1550 nm | Operating wavelength |

### `CrossbarArray`

NxM grid of `MemristorDevice` instances. Core operation:

```
output = input_vector @ G_matrix
```

Supports:
- Single and batched VMM
- Direct conductance programming via `set_conductances()`
- Voltage-driven update via `update_all()`

### `PCMWeightUpdate`

Iterative verify-after-write pulse programming:

- **SET pulse** (positive voltage) → crystallization → higher conductance
- **RESET pulse** (negative voltage) → amorphization → lower conductance
- Converges when `|G_achieved - G_target| ≤ tolerance × (G_on - G_off)`

## Demo

```bash
python -m photon_memristor_sim.demo
```

Runs a 2-layer network (4→8→3) with crossbar VMM, photonic interconnect, and softmax output.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

15 tests covering all components: initialization, conductance models, state updates, VMM correctness, photonic loss calculations, PCM convergence, and end-to-end integration.

## Architecture

```
Input Vector
     │
     ▼
┌──────────────┐
│ CrossbarArray │  ← memristor weights (NxM)
│   (VMM)      │
└──────┬───────┘
       │ electrical currents
       ▼
┌──────────────┐
│ PhotonicLink  │  ← optical transmission
│ (loss model) │
└──────┬───────┘
       │ optical signals
       ▼
┌──────────────┐
│ CrossbarArray │  ← second layer
│   (VMM)      │
└──────┬───────┘
       │
       ▼
  Output (logits)
```

## References

- Strukov et al., "The missing memristor found", *Nature* 453 (2008)
- Cheng et al., "In-memory computing with resistive switching devices", *Nature Electronics* (2020)
- Feldmann et al., "Integrated 256 cell photonic phase-change memory", *Nature* 569 (2019)

## License

MIT
