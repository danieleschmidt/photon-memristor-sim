"""
CrossbarArray: NxM resistive crossbar for analog vector-matrix multiply (VMM).

Each cell is a MemristorDevice. Rows are inputs, columns are outputs.
VMM computes: output[j] = sum_i( input[i] * G[i,j] )
"""

import numpy as np
from typing import Optional, Tuple

from .memristor import MemristorDevice


class CrossbarArray:
    """
    NxM crossbar array of memristive devices.

    Performs analog in-memory vector-matrix multiplication (VMM).
    The crossbar conductance matrix G[i,j] is used directly in VMM.

    Parameters
    ----------
    rows : int
        Number of row lines (input dimension)
    cols : int
        Number of column lines (output dimension)
    Ron : float
        Default ON resistance for all devices (Ω)
    Roff : float
        Default OFF resistance for all devices (Ω)
    w_init : float or np.ndarray
        Initial state for all devices. If scalar, all devices get same w.
        If array of shape (rows, cols), each device gets its value.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        Ron: float = 1e3,
        Roff: float = 1e5,
        w_init=0.5,
    ):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")

        self.rows = rows
        self.cols = cols

        # Build 2D array of MemristorDevices
        if np.isscalar(w_init):
            w_array = np.full((rows, cols), float(w_init))
        else:
            w_array = np.asarray(w_init, dtype=float)
            if w_array.shape != (rows, cols):
                raise ValueError(
                    f"w_init shape {w_array.shape} must match ({rows}, {cols})"
                )

        self.devices = np.empty((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                self.devices[i, j] = MemristorDevice(
                    Ron=Ron, Roff=Roff, w_init=float(w_array[i, j])
                )

    @property
    def conductance_matrix(self) -> np.ndarray:
        """
        Return current conductance matrix G of shape (rows, cols) in Siemens.
        """
        G = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                G[i, j] = self.devices[i, j].conductance
        return G

    def vmm(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Analog vector-matrix multiply: output = input @ G

        Parameters
        ----------
        input_vector : np.ndarray
            Input voltages/signals of shape (rows,) or (batch, rows)

        Returns
        -------
        np.ndarray
            Output currents of shape (cols,) or (batch, cols)
        """
        x = np.asarray(input_vector, dtype=float)
        G = self.conductance_matrix

        if x.ndim == 1:
            if x.shape[0] != self.rows:
                raise ValueError(
                    f"input_vector length {x.shape[0]} must match rows={self.rows}"
                )
            return x @ G  # shape: (cols,)
        elif x.ndim == 2:
            if x.shape[1] != self.rows:
                raise ValueError(
                    f"input batch dim {x.shape[1]} must match rows={self.rows}"
                )
            return x @ G  # shape: (batch, cols)
        else:
            raise ValueError("input_vector must be 1D or 2D array")

    def set_conductances(self, G_target: np.ndarray) -> None:
        """
        Set device states to approximate target conductance matrix.

        Parameters
        ----------
        G_target : np.ndarray
            Target conductance matrix of shape (rows, cols) in Siemens
        """
        G_target = np.asarray(G_target, dtype=float)
        if G_target.shape != (self.rows, self.cols):
            raise ValueError(
                f"G_target shape {G_target.shape} must be ({self.rows}, {self.cols})"
            )

        for i in range(self.rows):
            for j in range(self.cols):
                dev = self.devices[i, j]
                G_on = 1.0 / dev.Ron
                G_off = 1.0 / dev.Roff
                # Invert: G = G_on * w + G_off * (1 - w)  =>  w = (G - G_off) / (G_on - G_off)
                g = float(np.clip(G_target[i, j], G_off, G_on))
                w = (g - G_off) / (G_on - G_off)
                dev.w = float(np.clip(w, 0.0, 1.0))

    def update_all(self, voltage_matrix: np.ndarray, dt: float) -> None:
        """
        Apply voltage update to all devices simultaneously.

        Parameters
        ----------
        voltage_matrix : np.ndarray
            Applied voltages of shape (rows, cols)
        dt : float
            Time step in seconds
        """
        V = np.asarray(voltage_matrix, dtype=float)
        if V.shape != (self.rows, self.cols):
            raise ValueError(
                f"voltage_matrix shape {V.shape} must be ({self.rows}, {self.cols})"
            )
        for i in range(self.rows):
            for j in range(self.cols):
                self.devices[i, j].update(V[i, j], dt)

    def __repr__(self) -> str:
        G = self.conductance_matrix
        return (
            f"CrossbarArray({self.rows}x{self.cols}, "
            f"G_mean={G.mean():.3e} S, G_range=[{G.min():.3e}, {G.max():.3e}] S)"
        )
