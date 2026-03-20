"""
photon_memristor_sim — Photonic Memristor Simulator

A physics-based simulation library for photonic-memristive computing systems.
Supports analog vector-matrix multiplication via resistive crossbars and
photonic interconnects for neuromorphic computing research.
"""

from .memristor import MemristorDevice
from .photonic import PhotonicLink
from .crossbar import CrossbarArray
from .pcm import PCMWeightUpdate
from .demo import run_demo, run_pcm_programming_demo

__version__ = "0.1.0"
__author__ = "danieleschmidt"

__all__ = [
    "MemristorDevice",
    "PhotonicLink",
    "CrossbarArray",
    "PCMWeightUpdate",
    "run_demo",
    "run_pcm_programming_demo",
]
