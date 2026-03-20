# photon-memristor-sim

Simulation of photonic memristor devices and crossbar arrays for neuromorphic in-memory computing.

## Components

- **MemristorDevice** — Ron/Roff threshold memristor with I-V curve simulation
- **PCMWeightUpdater** — Phase-change memory weight programming (potentiate/depress)
- **PhotonicLink** — Optical signal encoding/decoding via memristor state
- **CrossbarArray** — N×M photonic memristor crossbar for vector-matrix multiply
- **PhotonicMemristorNetwork** — 2-layer network demo

## Usage

```python
from photon_memristor.crossbar import PhotonicMemristorNetwork
net = PhotonicMemristorNetwork(input_dim=4, hidden_dim=8, output_dim=2)
print(net.demo())
```

## Install & Test

```bash
pip install -r requirements.txt
pytest tests/ -v
```
