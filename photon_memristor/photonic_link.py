"""Photonic interconnect for memristor-based neuromorphic computing."""
import numpy as np


class PhotonicLink:
    """Models optical signal transmission through a photonic memristor link."""

    def __init__(self, wavelength_nm: float = 1550.0, loss_db: float = 1.0,
                 modulation_depth: float = 0.9):
        self.wavelength_nm = wavelength_nm
        self.loss_db = loss_db
        self.modulation_depth = modulation_depth

    @property
    def transmission(self) -> float:
        return 10 ** (-self.loss_db / 10)

    def encode(self, weight: float, signal: float) -> float:
        """Encode weight as optical intensity modulation."""
        modulated = signal * (1 - self.modulation_depth + self.modulation_depth * weight)
        return float(modulated * self.transmission)

    def decode(self, optical_signal: float, signal_ref: float) -> float:
        """Decode weight from received optical signal."""
        received = optical_signal / (self.transmission + 1e-12)
        weight = (received / (signal_ref + 1e-12) - (1 - self.modulation_depth)) / (self.modulation_depth + 1e-12)
        return float(np.clip(weight, 0.0, 1.0))

    def snr_db(self, signal_power: float = 1e-3, noise_floor: float = 1e-7) -> float:
        received = signal_power * self.transmission
        return 10 * np.log10(received / noise_floor)
