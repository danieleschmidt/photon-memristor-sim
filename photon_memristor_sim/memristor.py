"""
MemristorDevice: Resistive switching device model.

Models a memristor using a linear drift model with a normalized state variable w ∈ [0, 1].
Conductance interpolates between Ron (fully ON) and Roff (fully OFF) states.
"""

import numpy as np


class MemristorDevice:
    """
    Linear drift memristor model.

    Parameters
    ----------
    Ron : float
        ON-state resistance in Ohms (default 1000 Ω = 1 kΩ)
    Roff : float
        OFF-state resistance in Ohms (default 100000 Ω = 100 kΩ)
    w_init : float
        Initial state variable in [0, 1] (default 0.5)
    mobility : float
        Ion mobility coefficient controlling drift speed (default 1e-10)
    """

    def __init__(
        self,
        Ron: float = 1e3,
        Roff: float = 1e5,
        w_init: float = 0.5,
        mobility: float = 1e-10,
    ):
        if Ron <= 0:
            raise ValueError("Ron must be positive")
        if Roff <= 0:
            raise ValueError("Roff must be positive")
        if Ron >= Roff:
            raise ValueError("Ron must be less than Roff")
        if not (0.0 <= w_init <= 1.0):
            raise ValueError("w_init must be in [0, 1]")

        self.Ron = Ron
        self.Roff = Roff
        self.w = w_init
        self.mobility = mobility

    @property
    def conductance(self) -> float:
        """
        Device conductance in Siemens.

        G = w/Ron + (1-w)/Roff  (parallel conductance model)
        Or equivalently: G = G_on * w + G_off * (1 - w)
        """
        G_on = 1.0 / self.Ron
        G_off = 1.0 / self.Roff
        return G_on * self.w + G_off * (1.0 - self.w)

    @property
    def resistance(self) -> float:
        """Device resistance in Ohms."""
        return 1.0 / self.conductance

    def update(self, voltage: float, dt: float) -> None:
        """
        Update state variable based on applied voltage and time step.

        Uses linear drift model: dw/dt = mobility * voltage
        State is clamped to [0, 1] (hard boundary).

        Parameters
        ----------
        voltage : float
            Applied voltage in Volts
        dt : float
            Time step in seconds
        """
        dw = self.mobility * voltage * dt
        self.w = float(np.clip(self.w + dw, 0.0, 1.0))

    def reset(self, w: float = 0.5) -> None:
        """Reset state variable to given value (default 0.5)."""
        if not (0.0 <= w <= 1.0):
            raise ValueError("w must be in [0, 1]")
        self.w = w

    def __repr__(self) -> str:
        return (
            f"MemristorDevice(Ron={self.Ron:.1e}, Roff={self.Roff:.1e}, "
            f"w={self.w:.4f}, G={self.conductance:.4e} S)"
        )
