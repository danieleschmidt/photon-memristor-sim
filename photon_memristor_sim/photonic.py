"""
PhotonicLink: Optical power transmission model.

Models a photonic interconnect with insertion loss and coupling efficiency.
"""

import numpy as np


class PhotonicLink:
    """
    Photonic waveguide link model.

    Models optical power transmission through a waveguide with:
    - Insertion loss (waveguide propagation + connector losses)
    - Coupling efficiency (source-to-waveguide and waveguide-to-detector)

    Parameters
    ----------
    insertion_loss_db : float
        Total insertion loss in dB (must be >= 0). Default 3.0 dB.
    coupling_efficiency : float
        Combined input/output coupling efficiency in [0, 1]. Default 0.8.
    wavelength_nm : float
        Operating wavelength in nm (informational). Default 1550 nm.
    """

    def __init__(
        self,
        insertion_loss_db: float = 3.0,
        coupling_efficiency: float = 0.8,
        wavelength_nm: float = 1550.0,
    ):
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must be >= 0")
        if not (0.0 < coupling_efficiency <= 1.0):
            raise ValueError("coupling_efficiency must be in (0, 1]")
        if wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")

        self.insertion_loss_db = insertion_loss_db
        self.coupling_efficiency = coupling_efficiency
        self.wavelength_nm = wavelength_nm

    @property
    def loss_linear(self) -> float:
        """Insertion loss as a linear power ratio (< 1)."""
        return 10.0 ** (-self.insertion_loss_db / 10.0)

    @property
    def total_efficiency(self) -> float:
        """Combined transmission efficiency (coupling * insertion loss)."""
        return self.coupling_efficiency * self.loss_linear

    def transmit(self, power_in: float) -> float:
        """
        Compute output optical power after transmission.

        P_out = P_in * coupling_efficiency * 10^(-insertion_loss_db / 10)

        Parameters
        ----------
        power_in : float
            Input optical power in Watts (must be >= 0)

        Returns
        -------
        float
            Output optical power in Watts
        """
        power_in = np.asarray(power_in)
        if np.any(power_in < 0):
            raise ValueError("power_in must be >= 0")
        return power_in * self.total_efficiency

    def power_loss_db(self, power_in: float) -> float:
        """
        Compute total power loss in dB for a given input power.

        Parameters
        ----------
        power_in : float
            Input optical power in Watts (must be > 0)

        Returns
        -------
        float
            Power loss in dB (positive means loss)
        """
        if power_in <= 0:
            raise ValueError("power_in must be > 0 for dB calculation")
        power_out = self.transmit(power_in)
        return -10.0 * np.log10(power_out / power_in)

    def __repr__(self) -> str:
        return (
            f"PhotonicLink(insertion_loss={self.insertion_loss_db:.1f} dB, "
            f"coupling_eff={self.coupling_efficiency:.3f}, "
            f"λ={self.wavelength_nm:.0f} nm, "
            f"total_eff={self.total_efficiency:.4f})"
        )
