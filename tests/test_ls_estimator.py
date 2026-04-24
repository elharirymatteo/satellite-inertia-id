import numpy as np
import pytest
from estimation.ls_estimator import LeastSquaresEstimator, DiagonalInertiaModel, FullInertiaModel


class TestLSNoiseFree:
    def test_diagonal_recovery_noise_free(self, synthetic_diagonal_data):
        """LS must recover true diagonal inertia to <0.1% on noise-free data."""
        data = synthetic_diagonal_data
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(data, verbose=False)

        assert result['success'], f"Estimation failed: {result['message']}"
        rel_err = np.abs(result['parameters'] - data['I_true']) / data['I_true']
        assert np.all(rel_err < 1e-3), (
            f"Relative errors {rel_err} exceed 0.1% on noise-free data"
        )

    def test_estimation_result_keys(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(synthetic_diagonal_data)
        for key in ('success', 'parameters', 'inertia_tensor', 'cost', 'rms_error',
                    'residuals', 'message', 'n_function_evaluations'):
            assert key in result, f"Missing key: {key}"

    def test_inertia_tensor_is_diagonal(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(synthetic_diagonal_data)
        I = result['inertia_tensor']
        off_diag = I - np.diag(np.diag(I))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)


class TestLSWithNoise:
    def test_diagonal_recovery_with_noise(self, noisy_diagonal_data):
        """LS must recover true inertia to <5% under 1% measurement noise."""
        data = noisy_diagonal_data
        estimator = LeastSquaresEstimator(DiagonalInertiaModel(), robust_loss='huber')
        result = estimator.estimate(data, verbose=False)

        assert result['success']
        rel_err = np.abs(result['parameters'] - data['I_true']) / data['I_true']
        assert np.all(rel_err < 0.05), (
            f"Relative errors {rel_err} exceed 5% under 1% noise"
        )

    def test_robust_loss_vs_plain_ls(self, noisy_diagonal_data):
        """Huber loss should produce comparable or better accuracy than plain LS."""
        data = noisy_diagonal_data
        I_true = data['I_true']

        est_plain = LeastSquaresEstimator(DiagonalInertiaModel(), robust_loss=None)
        est_huber = LeastSquaresEstimator(DiagonalInertiaModel(), robust_loss='huber')

        r_plain = est_plain.estimate(data)
        r_huber = est_huber.estimate(data)

        err_plain = np.linalg.norm(r_plain['parameters'] - I_true) / np.linalg.norm(I_true)
        err_huber = np.linalg.norm(r_huber['parameters'] - I_true) / np.linalg.norm(I_true)

        assert err_huber <= err_plain * 2.0, (
            f"Huber error {err_huber:.4f} is much worse than plain LS {err_plain:.4f}"
        )


class TestLSValidation:
    def test_validate_passes_on_good_estimate(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        result = estimator.estimate(synthetic_diagonal_data)
        validation = estimator.validate_estimate(result, synthetic_diagonal_data)
        assert validation['is_valid'], f"Validation failed: {validation['message']}"

    def test_history_grows_with_each_estimate(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        assert len(estimator.get_estimation_history()) == 0
        estimator.estimate(synthetic_diagonal_data)
        assert len(estimator.get_estimation_history()) == 1
        estimator.estimate(synthetic_diagonal_data)
        assert len(estimator.get_estimation_history()) == 2

    def test_reset_history(self, synthetic_diagonal_data):
        estimator = LeastSquaresEstimator(DiagonalInertiaModel())
        estimator.estimate(synthetic_diagonal_data)
        estimator.reset_history()
        assert len(estimator.get_estimation_history()) == 0
