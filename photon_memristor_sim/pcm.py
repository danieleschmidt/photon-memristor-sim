"""
PCMWeightUpdate: Phase-change material (PCM) weight programming.

Implements iterative SET/RESET pulse programming to reach a target conductance.
PCM devices switch between amorphous (high-R) and crystalline (low-R) phases.
"""

import numpy as np
from typing import Optional, Tuple

from .memristor import MemristorDevice


class PCMWeightUpdate:
    """
    Iterative pulse programming for PCM-based memristive devices.

    Uses a verify-after-write (VAW) scheme:
    1. Apply SET pulse → increases conductance (crystallization)
    2. Apply RESET pulse → decreases conductance (amorphization)
    3. Read back and compare to target
    4. Repeat until within tolerance or max pulses reached

    Parameters
    ----------
    pulse_amplitude : float
        Voltage amplitude of programming pulses in Volts (default 1.0 V)
    pulse_width : float
        Duration of each pulse in seconds (default 100e-9 = 100 ns)
    tolerance : float
        Convergence tolerance as fraction of G_on - G_off (default 0.02 = 2%)
    max_iterations : int
        Maximum number of programming iterations (default 100)
    """

    def __init__(
        self,
        pulse_amplitude: float = 1.0,
        pulse_width: float = 100e-9,
        tolerance: float = 0.02,
        max_iterations: int = 100,
    ):
        if pulse_amplitude <= 0:
            raise ValueError("pulse_amplitude must be positive")
        if pulse_width <= 0:
            raise ValueError("pulse_width must be positive")
        if not (0.0 < tolerance < 1.0):
            raise ValueError("tolerance must be in (0, 1)")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self.pulse_amplitude = pulse_amplitude
        self.pulse_width = pulse_width
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def program_weight(
        self,
        device: MemristorDevice,
        target_conductance: float,
        pulses: Optional[int] = None,
    ) -> Tuple[float, int, bool]:
        """
        Program a device to a target conductance using iterative pulsing.

        Parameters
        ----------
        device : MemristorDevice
            Device to program
        target_conductance : float
            Target conductance in Siemens
        pulses : int, optional
            Override max iterations (uses self.max_iterations if None)

        Returns
        -------
        achieved_conductance : float
            Final conductance after programming
        iterations_used : int
            Number of pulses applied
        converged : bool
            True if target was reached within tolerance
        """
        G_on = 1.0 / device.Ron
        G_off = 1.0 / device.Roff
        g_range = G_on - G_off

        # Clamp target to valid device range
        target_conductance = float(np.clip(target_conductance, G_off, G_on))

        max_iter = pulses if pulses is not None else self.max_iterations
        abs_tolerance = self.tolerance * g_range

        for iteration in range(max_iter):
            current_g = device.conductance
            error = target_conductance - current_g

            if abs(error) <= abs_tolerance:
                return current_g, iteration, True

            # Apply SET pulse (positive voltage → increase w → increase G)
            # or RESET pulse (negative voltage → decrease w → decrease G)
            if error > 0:
                # Need higher conductance: SET pulse (crystallize)
                device.update(+self.pulse_amplitude, self.pulse_width)
            else:
                # Need lower conductance: RESET pulse (amorphize)
                device.update(-self.pulse_amplitude, self.pulse_width)

        # Final check after max iterations
        final_g = device.conductance
        converged = abs(target_conductance - final_g) <= abs_tolerance
        return final_g, max_iter, converged

    def program_array(
        self,
        devices,
        target_conductances: np.ndarray,
        pulses: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Program a 2D array of devices to target conductances.

        Parameters
        ----------
        devices : np.ndarray of MemristorDevice, shape (M, N)
        target_conductances : np.ndarray, shape (M, N)
        pulses : int, optional

        Returns
        -------
        achieved : np.ndarray, shape (M, N) — final conductances
        iterations : np.ndarray, shape (M, N) — pulses used per device
        converged : np.ndarray, shape (M, N) — convergence flags
        """
        devices = np.asarray(devices)
        target_conductances = np.asarray(target_conductances, dtype=float)

        if devices.shape != target_conductances.shape:
            raise ValueError("devices and target_conductances must have same shape")

        shape = devices.shape
        achieved = np.zeros(shape, dtype=float)
        iterations = np.zeros(shape, dtype=int)
        converged = np.zeros(shape, dtype=bool)

        for idx in np.ndindex(shape):
            g, n, c = self.program_weight(devices[idx], target_conductances[idx], pulses)
            achieved[idx] = g
            iterations[idx] = n
            converged[idx] = c

        return achieved, iterations, converged

    def __repr__(self) -> str:
        return (
            f"PCMWeightUpdate(V={self.pulse_amplitude:.2f} V, "
            f"τ={self.pulse_width*1e9:.0f} ns, "
            f"tol={self.tolerance*100:.1f}%, "
            f"max_iter={self.max_iterations})"
        )
