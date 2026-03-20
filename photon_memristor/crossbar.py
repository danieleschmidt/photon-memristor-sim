"""Photonic memristor crossbar array for in-memory computing."""
import numpy as np
from photon_memristor.memristor import MemristorDevice, PCMWeightUpdater
from photon_memristor.photonic_link import PhotonicLink


class CrossbarArray:
    """N×M photonic memristor crossbar for vector-matrix multiply."""

    def __init__(self, rows: int, cols: int, use_pcm: bool = False):
        self.rows = rows
        self.cols = cols
        self.use_pcm = use_pcm
        if use_pcm:
            self._cells = [[PCMWeightUpdater() for _ in range(cols)] for _ in range(rows)]
        else:
            self._cells = [[MemristorDevice() for _ in range(cols)] for _ in range(rows)]
        self._link = PhotonicLink()

    def set_weights(self, W: np.ndarray) -> None:
        """Program crossbar weights from matrix W (values in [0,1])."""
        assert W.shape == (self.rows, self.cols), f"Expected ({self.rows},{self.cols}), got {W.shape}"
        W_norm = np.clip(W, 0.0, 1.0)
        for i in range(self.rows):
            for j in range(self.cols):
                self._cells[i][j].set_weight(float(W_norm[i, j]))

    def get_weights(self) -> np.ndarray:
        return np.array([[self._cells[i][j].weight for j in range(self.cols)]
                         for i in range(self.rows)])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vector-matrix multiply: y = W^T @ x using photonic links."""
        assert x.shape == (self.rows,)
        W = self.get_weights()
        # Photonic encoding: each element encoded via link
        y = np.zeros(self.cols)
        for j in range(self.cols):
            for i in range(self.rows):
                y[j] += self._link.encode(W[i, j], float(x[i]))
        return y

    def update_weights_hebbian(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> None:
        """Hebbian weight update: ΔW = lr * x ⊗ y."""
        W = self.get_weights()
        dW = lr * np.outer(x, y)
        W_new = np.clip(W + dW, 0.0, 1.0)
        self.set_weights(W_new)


class PhotonicMemristorNetwork:
    """Small 2-layer network using photonic memristor crossbars."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 8, output_dim: int = 2):
        self.layer1 = CrossbarArray(input_dim, hidden_dim, use_pcm=True)
        self.layer2 = CrossbarArray(hidden_dim, output_dim, use_pcm=True)
        # Initialize with random weights
        rng = np.random.default_rng(42)
        self.layer1.set_weights(rng.uniform(0, 1, (input_dim, hidden_dim)))
        self.layer2.set_weights(rng.uniform(0, 1, (hidden_dim, output_dim)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.layer1.forward(x))
        return self.layer2.forward(h)

    def demo(self) -> str:
        x = np.random.default_rng(0).uniform(0, 1, 4)
        y = self.forward(x)
        return f"PhotonicMemristorNetwork: input={x.round(3)}, output={y.round(4)}"
