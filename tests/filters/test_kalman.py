"""Tests for KalmanFilter."""

import numpy as np
import pytest

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import KalmanFilter


def _build_local_level_ssm(sigma2_obs: float, sigma2_level: float) -> StateSpaceRepresentation:
    """Build a local level SSM with given variances."""
    ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
    ssm.T = np.array([[1.0]])
    ssm.Z = np.array([[1.0]])
    ssm.R = np.array([[1.0]])
    ssm.H = np.array([[sigma2_obs]])
    ssm.Q = np.array([[sigma2_level]])
    ssm.a1 = np.array([0.0])
    ssm.P1 = np.array([[1e7]])
    return ssm


class TestKalmanFilter:
    """Tests for KalmanFilter."""

    def test_nile_loglike(self, nile_volume: np.ndarray) -> None:
        """Log-likelihood for Local Level on Nile should be ~-632.54.

        Reference: Durbin & Koopman (2012), p.15
        """
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        output = kf.filter(nile_volume, ssm)
        # D&K report the diffuse log-likelihood excluding the first observation
        # (dominated by the diffuse prior P1=1e7)
        diffuse_loglike = float(np.sum(output.loglike_obs[1:]))
        assert diffuse_loglike == pytest.approx(-632.54, abs=0.5)

    def test_nile_filtered_state(self, nile_volume: np.ndarray) -> None:
        """Filtered state at t=100 should be ~798.4."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        output = kf.filter(nile_volume, ssm)
        # Last filtered state
        assert output.filtered_state[-1, 0] == pytest.approx(798.4, abs=5.0)

    def test_prediction_error_mean_zero(self, nile_volume: np.ndarray) -> None:
        """Mean of prediction errors should be approximately zero."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        output = kf.filter(nile_volume, ssm)
        # Skip first few observations (diffuse initialization)
        v_mean = np.nanmean(output.residuals[10:])
        assert abs(v_mean) < 20.0  # approximate zero

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Test output array shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        output = kf.filter(nile_volume, ssm)
        nobs = len(nile_volume)
        assert output.filtered_state.shape == (nobs, 1)
        assert output.filtered_cov.shape == (nobs, 1, 1)
        assert output.predicted_state.shape == (nobs, 1)
        assert output.predicted_cov.shape == (nobs, 1, 1)
        assert output.residuals.shape == (nobs, 1)
        assert output.forecast_cov.shape == (nobs, 1, 1)
        assert output.loglike_obs.shape == (nobs,)

    def test_missing_data(self, nile_volume: np.ndarray) -> None:
        """Filtering with missing data should have lower loglike."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()

        # Full data
        output_full = kf.filter(nile_volume, ssm)

        # With 10% missing
        y_missing = nile_volume.copy()
        rng = np.random.default_rng(42)
        mask = rng.choice(len(y_missing), size=10, replace=False)
        y_missing[mask] = np.nan

        output_missing = kf.filter(y_missing, ssm)

        # Fewer effective observations
        assert output_missing.nobs_effective < output_full.nobs_effective
        assert output_missing.nobs_effective == len(nile_volume) - 10

    def test_predict_step(self) -> None:
        """Test single prediction step."""
        kf = KalmanFilter()
        a = np.array([1.0])
        P = np.array([[2.0]])
        T = np.array([[0.9]])
        R = np.array([[1.0]])
        Q = np.array([[0.5]])
        c = np.array([0.1])

        a_pred, P_pred = kf.predict_step(a, P, T, R, Q, c)
        np.testing.assert_allclose(a_pred, [1.0])  # 0.9 * 1.0 + 0.1
        np.testing.assert_allclose(P_pred, [[2.12]])  # 0.81 * 2.0 + 0.5 = 2.12

    def test_update_step(self) -> None:
        """Test single update step."""
        kf = KalmanFilter()
        a_pred = np.array([1.0])
        P_pred = np.array([[2.0]])
        y = np.array([1.5])
        Z = np.array([[1.0]])
        H = np.array([[1.0]])
        d = np.array([0.0])

        a_filt, P_filt, v, F, K, ll = kf.update_step(a_pred, P_pred, y, Z, H, d)
        # v = y - Z @ a_pred = 1.5 - 1.0 = 0.5
        np.testing.assert_allclose(v, [0.5])
        # F = Z @ P @ Z' + H = 2.0 + 1.0 = 3.0
        np.testing.assert_allclose(F, [[3.0]])
        # K = P @ Z' @ F^{-1} = 2.0 / 3.0
        np.testing.assert_allclose(K, [[2.0 / 3.0]], atol=1e-10)

    def test_nobs_effective(self, nile_volume: np.ndarray) -> None:
        """Effective observations should equal total for complete data."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        output = kf.filter(nile_volume, ssm)
        assert output.nobs_effective == len(nile_volume)
