"""Memristor device model with Ron/Roff switching."""
import numpy as np


class MemristorDevice:
    """Threshold-based memristor with Ron/Roff resistance states."""

    def __init__(self, Ron: float = 100.0, Roff: float = 16000.0,
                 threshold_set: float = 1.0, threshold_reset: float = -0.5):
        self.Ron = Ron
        self.Roff = Roff
        self.threshold_set = threshold_set
        self.threshold_reset = threshold_reset
        self._w: float = 0.0  # internal state [0,1], 0=Roff, 1=Ron
        self._history = []

    @property
    def resistance(self) -> float:
        return self.Roff - self._w * (self.Roff - self.Ron)

    @property
    def weight(self) -> float:
        return self._w

    @property
    def conductance(self) -> float:
        return 1.0 / self.resistance

    def apply_voltage(self, V: float) -> float:
        """Apply voltage, update state, return current (A)."""
        if V > self.threshold_set:
            dw = min(0.1 * (V - self.threshold_set), 1.0 - self._w)
            self._w = min(1.0, self._w + dw)
        elif V < self.threshold_reset:
            dw = min(0.1 * abs(V - self.threshold_reset), self._w)
            self._w = max(0.0, self._w - dw)
        I = V / self.resistance
        self._history.append((V, I, self._w))
        return I

    def set_weight(self, w: float) -> None:
        """Directly set internal state w in [0,1]."""
        self._w = float(np.clip(w, 0.0, 1.0))

    def reset(self) -> None:
        self._w = 0.0
        self._history.clear()

    def iv_curve(self, V_max: float = 2.0, n_points: int = 100):
        """Sweep voltage and return (voltages, currents)."""
        self.reset()
        V_sweep = np.concatenate([
            np.linspace(0, V_max, n_points // 2),
            np.linspace(V_max, -V_max, n_points // 2)
        ])
        currents = [self.apply_voltage(v) for v in V_sweep]
        return V_sweep, np.array(currents)


class PCMWeightUpdater:
    """Phase-change memory (PCM) weight update rule for synaptic plasticity."""

    def __init__(self, n_levels: int = 16, Gmin: float = 1e-6, Gmax: float = 1e-4):
        self.n_levels = n_levels
        self.Gmin = Gmin
        self.Gmax = Gmax
        self._conductance: float = Gmin

    @property
    def weight(self) -> float:
        return (self._conductance - self.Gmin) / (self.Gmax - self.Gmin)

    def potentiate(self, n_pulses: int = 1) -> float:
        """Apply n SET pulses (increase conductance)."""
        step = (self.Gmax - self.Gmin) / self.n_levels
        self._conductance = min(self.Gmax, self._conductance + n_pulses * step)
        return self.weight

    def depress(self, n_pulses: int = 1) -> float:
        """Apply n RESET pulses (decrease conductance)."""
        step = (self.Gmax - self.Gmin) / self.n_levels
        self._conductance = max(self.Gmin, self._conductance - n_pulses * step)
        return self.weight

    def set_weight(self, target: float) -> float:
        """Program to target weight [0,1] via iterative write-verify."""
        target = float(np.clip(target, 0.0, 1.0))
        self._conductance = self.Gmin + target * (self.Gmax - self.Gmin)
        return self.weight
