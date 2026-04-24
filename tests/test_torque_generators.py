import numpy as np
import pytest
from control.torque_generators import generate_torque_profile, _lfsr_sequence


class TestLFSRSequence:
    def test_sequence_is_binary(self):
        seq = _lfsr_sequence(n_bits=4, n_samples=100, seed=1)
        assert set(np.unique(seq)).issubset({-1.0, 1.0})

    def test_period_is_2n_minus_1(self):
        """An n-bit LFSR has period 2^n - 1."""
        for n_bits in (4, 5, 6, 8, 10, 12):
            period = 2 ** n_bits - 1
            seq = _lfsr_sequence(n_bits=n_bits, n_samples=2 * period, seed=1)
            np.testing.assert_array_equal(seq[:period], seq[period:2 * period],
                                          err_msg=f"Period wrong for n_bits={n_bits}")

    def test_different_seeds_give_different_sequences(self):
        s1 = _lfsr_sequence(n_bits=6, n_samples=63, seed=1)
        s2 = _lfsr_sequence(n_bits=6, n_samples=63, seed=3)
        assert not np.array_equal(s1, s2)

    def test_autocorrelation_near_ideal(self):
        """For PRBS of length L=2^n-1, off-peak autocorr ≈ -1/L."""
        n_bits = 10
        L = 2 ** n_bits - 1
        seq = _lfsr_sequence(n_bits=n_bits, n_samples=L, seed=1)
        seq_norm = seq / np.std(seq)
        expected_ac = -1.0 / L
        for lag in range(1, 11):
            ac = float(np.mean(seq_norm[:-lag] * seq_norm[lag:]))
            assert abs(ac - expected_ac) < 0.05, (
                f"Autocorr at lag {lag}: {ac:.4f}, expected ~{expected_ac:.4f}"
            )


class TestPRBSTorque:
    def test_output_is_binary(self):
        t = np.linspace(0, 300, 300)
        torques = generate_torque_profile('prbs', t, amplitude=0.01, switch_time=20)
        amplitude = 0.01
        for val in torques.flatten():
            assert abs(abs(val) - amplitude) < 1e-10 or abs(val) < 1e-10, (
                f"PRBS value {val} is not ±{amplitude} or 0"
            )

    def test_output_shape(self):
        t = np.linspace(0, 100, 100)
        torques = generate_torque_profile('prbs', t, amplitude=0.005)
        assert torques.shape == (100, 3)

    def test_different_seeds_differ(self):
        t = np.linspace(0, 300, 300)
        t1 = generate_torque_profile('prbs', t, amplitude=0.01, seed=1)
        t2 = generate_torque_profile('prbs', t, amplitude=0.01, seed=7)
        assert not np.array_equal(t1, t2)
