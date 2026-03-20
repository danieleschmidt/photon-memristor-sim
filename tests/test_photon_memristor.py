"""Tests for photon-memristor-sim."""
import numpy as np
import pytest
from photon_memristor.memristor import MemristorDevice, PCMWeightUpdater
from photon_memristor.photonic_link import PhotonicLink
from photon_memristor.crossbar import CrossbarArray, PhotonicMemristorNetwork


class TestMemristorDevice:
    def test_initial_resistance_is_roff(self):
        m = MemristorDevice(Ron=100, Roff=16000)
        assert m.resistance == 16000

    def test_apply_voltage_set(self):
        m = MemristorDevice()
        for _ in range(10):
            m.apply_voltage(2.0)
        assert m.resistance < m.Roff

    def test_apply_voltage_reset(self):
        m = MemristorDevice()
        m.set_weight(1.0)
        assert m.resistance < m.Roff
        for _ in range(10):
            m.apply_voltage(-2.0)
        assert m.resistance > m.Ron

    def test_conductance_inverse_resistance(self):
        m = MemristorDevice()
        m.set_weight(0.5)
        assert abs(m.conductance - 1.0 / m.resistance) < 1e-12

    def test_set_weight_clips(self):
        m = MemristorDevice()
        m.set_weight(2.0)
        assert m._w == 1.0
        m.set_weight(-1.0)
        assert m._w == 0.0

    def test_iv_curve_shape(self):
        m = MemristorDevice()
        V, I = m.iv_curve(n_points=50)
        assert len(V) == 50
        assert len(I) == 50

    def test_reset_clears_history(self):
        m = MemristorDevice()
        m.apply_voltage(1.5)
        m.reset()
        assert len(m._history) == 0
        assert m._w == 0.0


class TestPCMWeightUpdater:
    def test_initial_weight_zero(self):
        pcm = PCMWeightUpdater()
        assert pcm.weight == 0.0

    def test_potentiate_increases_weight(self):
        pcm = PCMWeightUpdater()
        w0 = pcm.weight
        pcm.potentiate(2)
        assert pcm.weight > w0

    def test_depress_decreases_weight(self):
        pcm = PCMWeightUpdater()
        pcm.set_weight(0.5)
        w0 = pcm.weight
        pcm.depress(2)
        assert pcm.weight < w0

    def test_set_weight_accurate(self):
        pcm = PCMWeightUpdater()
        pcm.set_weight(0.75)
        assert abs(pcm.weight - 0.75) < 1e-9


class TestPhotonicLink:
    def test_transmission_positive(self):
        link = PhotonicLink(loss_db=1.0)
        assert 0 < link.transmission < 1

    def test_encode_decode_roundtrip(self):
        link = PhotonicLink(loss_db=0.0)
        weight = 0.6
        signal = 1.0
        encoded = link.encode(weight, signal)
        recovered = link.decode(encoded, signal)
        assert abs(recovered - weight) < 1e-6

    def test_snr_positive(self):
        link = PhotonicLink()
        snr = link.snr_db()
        assert snr > 0


class TestCrossbarArray:
    def test_set_get_weights(self):
        cb = CrossbarArray(4, 4)
        W = np.random.default_rng(0).uniform(0, 1, (4, 4))
        cb.set_weights(W)
        W_out = cb.get_weights()
        np.testing.assert_allclose(W_out, W, atol=1e-9)

    def test_forward_shape(self):
        cb = CrossbarArray(4, 8)
        W = np.ones((4, 8)) * 0.5
        cb.set_weights(W)
        x = np.ones(4)
        y = cb.forward(x)
        assert y.shape == (8,)

    def test_hebbian_update(self):
        cb = CrossbarArray(4, 4)
        W0 = np.full((4, 4), 0.5)
        cb.set_weights(W0)
        x = np.ones(4)
        y = np.ones(4)
        cb.update_weights_hebbian(x, y, lr=0.05)
        W1 = cb.get_weights()
        assert np.all(W1 >= W0)


class TestPhotonicMemristorNetwork:
    def test_forward_shape(self):
        net = PhotonicMemristorNetwork(input_dim=4, hidden_dim=8, output_dim=2)
        x = np.random.default_rng(5).uniform(0, 1, 4)
        y = net.forward(x)
        assert y.shape == (2,)

    def test_demo_runs(self):
        net = PhotonicMemristorNetwork()
        msg = net.demo()
        assert "PhotonicMemristorNetwork" in msg
