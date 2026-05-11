import numpy as np
import pytest
from utils.observability import compute_observability_metric, score_profiles_canonical, compute_fim_from_data


@pytest.fixture
def chirp_torques():
    """A chirp signal that should have high observability."""
    from scipy import signal
    t = np.linspace(0, 300, 300)
    torques = np.zeros((300, 3))
    for i, (f0, f1) in enumerate([(0.005, 0.05), (0.007, 0.04), (0.003, 0.06)]):
        torques[:, i] = 0.01 * signal.chirp(t, f0, t[-1], f1)
    return torques


@pytest.fixture
def zero_torques():
    return np.zeros((300, 3))


def test_observability_returns_required_keys(chirp_torques):
    result = compute_observability_metric(chirp_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))
    for key in ('min_sv', 'max_sv', 'condition_number', 'log_det', 'score'):
        assert key in result


def test_zero_torques_have_zero_observability(zero_torques):
    result = compute_observability_metric(zero_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))
    assert result['min_sv'] < 1e-10
    assert result['score'] < 1e-10


def test_chirp_beats_zero(chirp_torques, zero_torques):
    score_chirp = compute_observability_metric(
        chirp_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))['score']
    score_zero = compute_observability_metric(
        zero_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0))['score']
    assert score_chirp > score_zero


def test_score_profiles_canonical_ranks(chirp_torques, zero_torques):
    profiles = {'chirp': chirp_torques, 'zero': zero_torques}
    scores = score_profiles_canonical(profiles, dt=1.0, I_ref=(1.0, 1.0, 2.0))
    assert scores['chirp']['rank'] < scores['zero']['rank']  # lower rank = better


def test_energy_normalization_does_not_crash(chirp_torques):
    result = compute_observability_metric(
        chirp_torques, dt=1.0, I_ref=(1.0, 1.0, 2.0), normalize_energy=True)
    assert np.isfinite(result['score'])


# --- compute_fim_from_data ---

def test_compute_fim_from_data_returns_required_keys():
    np.random.seed(0)
    omega = np.random.randn(100, 3) * 0.1
    domega = np.random.randn(100, 3) * 0.01
    result = compute_fim_from_data(omega, domega)
    for key in ('W', 'F', 'log_det_F', 'min_sv', 'condition_number'):
        assert key in result


def test_compute_fim_w_shape():
    N = 80
    np.random.seed(1)
    omega = np.random.randn(N, 3) * 0.1
    domega = np.random.randn(N, 3) * 0.01
    result = compute_fim_from_data(omega, domega)
    assert result['W'].shape == (3 * N, 3)


def test_compute_fim_f_equals_wt_w():
    np.random.seed(2)
    omega = np.random.randn(100, 3) * 0.1
    domega = np.random.randn(100, 3) * 0.01
    result = compute_fim_from_data(omega, domega)
    np.testing.assert_allclose(result['F'], result['W'].T @ result['W'])


def test_compute_fim_full_rank_for_rich_data():
    np.random.seed(3)
    omega = np.random.randn(200, 3) * 0.3
    domega = np.random.randn(200, 3) * 0.05
    result = compute_fim_from_data(omega, domega)
    assert result['min_sv'] > 1e-6


def test_compute_fim_zero_data_is_rank_deficient():
    omega = np.zeros((100, 3))
    domega = np.zeros((100, 3))
    result = compute_fim_from_data(omega, domega)
    assert result['min_sv'] < 1e-10


def test_compute_fim_log_det_increases_with_data_richness():
    """More and richer data should give higher log-det(F)."""
    np.random.seed(4)
    omega_poor = np.random.randn(50, 3) * 0.01
    domega_poor = np.random.randn(50, 3) * 0.001
    omega_rich = np.random.randn(200, 3) * 0.3
    domega_rich = np.random.randn(200, 3) * 0.05
    result_poor = compute_fim_from_data(omega_poor, domega_poor)
    result_rich = compute_fim_from_data(omega_rich, domega_rich)
    assert result_rich['log_det_F'] > result_poor['log_det_F']
