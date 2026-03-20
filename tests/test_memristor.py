"""
Tests for photon_memristor_sim — 15 tests covering all components.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from photon_memristor_sim.memristor import MemristorDevice
from photon_memristor_sim.photonic import PhotonicLink
from photon_memristor_sim.crossbar import CrossbarArray
from photon_memristor_sim.pcm import PCMWeightUpdate


# ─────────────────────────────────────────────────────────────────────────────
# MemristorDevice tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemristorDevice:
    """Tests for MemristorDevice."""

    def test_init_defaults(self):
        """Device initializes with correct default parameters."""
        dev = MemristorDevice()
        assert dev.Ron == 1e3
        assert dev.Roff == 1e5
        assert dev.w == 0.5

    def test_conductance_at_w0(self):
        """At w=0 device is fully OFF: G = 1/Roff."""
        dev = MemristorDevice(Ron=1e3, Roff=1e5, w_init=0.0)
        expected = 1.0 / 1e5
        assert abs(dev.conductance - expected) < 1e-15

    def test_conductance_at_w1(self):
        """At w=1 device is fully ON: G = 1/Ron."""
        dev = MemristorDevice(Ron=1e3, Roff=1e5, w_init=1.0)
        expected = 1.0 / 1e3
        assert abs(dev.conductance - expected) < 1e-15

    def test_conductance_at_midpoint(self):
        """At w=0.5 conductance is midpoint between G_on and G_off."""
        dev = MemristorDevice(Ron=1e3, Roff=1e5, w_init=0.5)
        G_on = 1.0 / 1e3
        G_off = 1.0 / 1e5
        expected = 0.5 * G_on + 0.5 * G_off
        assert abs(dev.conductance - expected) < 1e-15

    def test_resistance_inverse_of_conductance(self):
        """Resistance is reciprocal of conductance."""
        dev = MemristorDevice(w_init=0.3)
        assert abs(dev.resistance - 1.0 / dev.conductance) < 1e-10

    def test_update_positive_voltage_increases_w(self):
        """Positive voltage increases state variable w."""
        dev = MemristorDevice(w_init=0.3, mobility=1e-10)
        initial_w = dev.w
        dev.update(voltage=1.0, dt=1e7)  # large dt to see clear change
        assert dev.w > initial_w

    def test_update_negative_voltage_decreases_w(self):
        """Negative voltage decreases state variable w."""
        dev = MemristorDevice(w_init=0.7, mobility=1e-10)
        initial_w = dev.w
        dev.update(voltage=-1.0, dt=1e7)
        assert dev.w < initial_w

    def test_update_clamps_w_to_0_1(self):
        """State variable never exceeds [0, 1] boundary."""
        dev = MemristorDevice(w_init=0.99, mobility=1e-10)
        dev.update(voltage=100.0, dt=1e12)  # huge push
        assert dev.w <= 1.0

        dev.reset(0.01)
        dev.update(voltage=-100.0, dt=1e12)
        assert dev.w >= 0.0

    def test_invalid_ron_roff(self):
        """ValueError raised when Ron >= Roff."""
        with pytest.raises(ValueError):
            MemristorDevice(Ron=1e5, Roff=1e3)  # Ron > Roff

    def test_invalid_w_init(self):
        """ValueError raised for w_init outside [0, 1]."""
        with pytest.raises(ValueError):
            MemristorDevice(w_init=1.5)
        with pytest.raises(ValueError):
            MemristorDevice(w_init=-0.1)

    def test_reset(self):
        """reset() correctly sets state variable."""
        dev = MemristorDevice(w_init=0.2)
        dev.reset(0.9)
        assert dev.w == pytest.approx(0.9)


# ─────────────────────────────────────────────────────────────────────────────
# PhotonicLink tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhotonicLink:
    """Tests for PhotonicLink."""

    def test_transmit_with_no_loss(self):
        """Zero insertion loss + perfect coupling → P_out == P_in."""
        link = PhotonicLink(insertion_loss_db=0.0, coupling_efficiency=1.0)
        assert link.transmit(1.0) == pytest.approx(1.0, rel=1e-9)

    def test_transmit_3db_loss(self):
        """3 dB insertion loss → ~50% power reduction (coupling=1.0)."""
        link = PhotonicLink(insertion_loss_db=3.0, coupling_efficiency=1.0)
        p_out = link.transmit(1.0)
        assert abs(p_out - 0.5) < 0.005  # 3 dB ≈ 0.5012, within 1%

    def test_coupling_efficiency_applied(self):
        """Coupling efficiency scales output correctly."""
        link = PhotonicLink(insertion_loss_db=0.0, coupling_efficiency=0.8)
        assert link.transmit(1.0) == pytest.approx(0.8)

    def test_power_loss_db(self):
        """power_loss_db matches insertion_loss_db when coupling=1."""
        link = PhotonicLink(insertion_loss_db=5.0, coupling_efficiency=1.0)
        loss = link.power_loss_db(1.0)
        assert loss == pytest.approx(5.0, rel=1e-6)

    def test_invalid_negative_power(self):
        """ValueError raised for negative input power."""
        link = PhotonicLink()
        with pytest.raises(ValueError):
            link.transmit(-1.0)

    def test_invalid_coupling(self):
        """ValueError raised for coupling outside (0, 1]."""
        with pytest.raises(ValueError):
            PhotonicLink(coupling_efficiency=0.0)
        with pytest.raises(ValueError):
            PhotonicLink(coupling_efficiency=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# CrossbarArray tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossbarArray:
    """Tests for CrossbarArray."""

    def test_vmm_shape(self):
        """VMM output has correct shape."""
        cb = CrossbarArray(rows=4, cols=3)
        x = np.ones(4)
        out = cb.vmm(x)
        assert out.shape == (3,)

    def test_vmm_batch(self):
        """VMM works with batched inputs."""
        cb = CrossbarArray(rows=4, cols=3)
        x = np.ones((10, 4))
        out = cb.vmm(x)
        assert out.shape == (10, 3)

    def test_vmm_correctness(self):
        """VMM computes x @ G correctly."""
        # Set all devices to w=1 (fully ON), G = 1/Ron
        cb = CrossbarArray(rows=2, cols=2, Ron=1e3, Roff=1e5, w_init=1.0)
        x = np.array([1.0, 2.0])
        out = cb.vmm(x)
        G_on = 1.0 / 1e3
        expected = np.array([3.0 * G_on, 3.0 * G_on])  # (1+2) * G_on
        np.testing.assert_allclose(out, expected, rtol=1e-9)

    def test_set_conductances(self):
        """set_conductances correctly programs device states."""
        cb = CrossbarArray(rows=2, cols=2)
        G_on = 1.0 / 1e3
        G_off = 1.0 / 1e5
        G_target = np.array([[G_on, G_off], [G_off, G_on]])
        cb.set_conductances(G_target)
        G_actual = cb.conductance_matrix
        np.testing.assert_allclose(G_actual, G_target, rtol=1e-9)

    def test_invalid_dimensions(self):
        """ValueError raised for mismatched input size."""
        cb = CrossbarArray(rows=4, cols=3)
        with pytest.raises(ValueError):
            cb.vmm(np.ones(3))  # wrong input size


# ─────────────────────────────────────────────────────────────────────────────
# PCMWeightUpdate tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPCMWeightUpdate:
    """Tests for PCMWeightUpdate."""

    def test_program_weight_converges_to_midpoint(self):
        """Programming converges to midpoint conductance."""
        dev = MemristorDevice(Ron=1e3, Roff=1e5, w_init=0.0, mobility=1e-3)
        pcm = PCMWeightUpdate(
            pulse_amplitude=1.0,
            pulse_width=1.0,
            tolerance=0.02,
            max_iterations=500,
        )
        G_on = 1.0 / 1e3
        G_off = 1.0 / 1e5
        target_g = G_off + 0.5 * (G_on - G_off)
        achieved, n_pulses, converged = pcm.program_weight(dev, target_g)
        assert converged, f"PCM did not converge in {n_pulses} pulses"
        assert abs(achieved - target_g) <= 0.02 * (G_on - G_off)

    def test_program_weight_returns_correct_types(self):
        """program_weight returns (float, int, bool)."""
        dev = MemristorDevice(Ron=1e3, Roff=1e5, w_init=0.5, mobility=1e-3)
        pcm = PCMWeightUpdate()
        G_on = 1.0 / 1e3
        result = pcm.program_weight(dev, G_on)
        assert isinstance(result[0], float)
        assert isinstance(result[1], int)
        assert isinstance(result[2], bool)

    def test_program_clamps_to_device_range(self):
        """Target above G_on is clamped to G_on."""
        dev = MemristorDevice(Ron=1e3, Roff=1e5, w_init=0.5, mobility=1e-3)
        pcm = PCMWeightUpdate(
            pulse_amplitude=1.0, pulse_width=1.0, tolerance=0.02, max_iterations=500
        )
        G_on = 1.0 / 1e3
        achieved, _, converged = pcm.program_weight(dev, G_on * 10.0)  # way above range
        # Should have converged to G_on (clamped)
        assert achieved <= G_on * 1.05

    def test_invalid_pcm_params(self):
        """ValueError raised for invalid PCM parameters."""
        with pytest.raises(ValueError):
            PCMWeightUpdate(pulse_amplitude=-1.0)
        with pytest.raises(ValueError):
            PCMWeightUpdate(tolerance=0.0)
        with pytest.raises(ValueError):
            PCMWeightUpdate(max_iterations=0)


# ─────────────────────────────────────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_forward_pass(self):
        """Two-layer crossbar network produces valid output."""
        np.random.seed(0)
        layer1 = CrossbarArray(rows=4, cols=8)
        layer2 = CrossbarArray(rows=8, cols=3)
        link = PhotonicLink(insertion_loss_db=2.0, coupling_efficiency=0.9)

        x = np.random.rand(4)
        h = layer1.vmm(x)
        h = np.maximum(h, 0)  # ReLU
        if h.max() > 0:
            h = h / h.max()
        h = link.transmit(h)
        out = layer2.vmm(h)

        assert out.shape == (3,)
        assert np.all(np.isfinite(out))

    def test_demo_runs(self):
        """Demo runs without errors and returns expected keys."""
        from photon_memristor_sim.demo import run_demo
        result = run_demo(verbose=False)
        assert "layer1_out" in result
        assert "probabilities" in result
        probs = result["probabilities"]
        assert abs(probs.sum() - 1.0) < 1e-9  # probabilities sum to 1
