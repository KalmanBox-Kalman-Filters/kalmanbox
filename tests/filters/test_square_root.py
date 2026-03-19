"""Tests for SquareRootKalmanFilter."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.filters.square_root import SquareRootKalmanFilter, _qr_factor


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


def _build_multivariate_ssm() -> StateSpaceRepresentation:
    """Build a 2-state, 2-obs SSM for broader testing."""
    ssm = StateSpaceRepresentation(k_states=2, k_endog=2)
    ssm.T = np.array([[0.9, 0.1], [0.0, 0.95]])
    ssm.Z = np.array([[1.0, 0.0], [0.5, 1.0]])
    ssm.R = np.eye(2)
    ssm.H = np.diag([1.0, 2.0])
    ssm.Q = np.diag([0.5, 0.3])
    ssm.a1 = np.zeros(2)
    ssm.P1 = np.eye(2) * 10.0
    return ssm


class TestQRFactor:
    """Tests for the _qr_factor helper."""

    def test_identity_factor(self) -> None:
        """QR factor of identity columns should be identity."""
        M = np.eye(3)
        S = _qr_factor(M)
        assert_allclose(S @ S.T, np.eye(3), atol=1e-14)

    def test_reconstructs_product(self) -> None:
        """S @ S.T should equal M.T @ M."""
        rng = np.random.default_rng(42)
        M = rng.normal(size=(5, 3))
        S = _qr_factor(M)
        assert_allclose(S @ S.T, M.T @ M, atol=1e-12)


class TestSquareRootKalmanFilter:
    """Tests for SquareRootKalmanFilter."""

    def test_equivalent_to_kf_local_level(self, nile_volume: np.ndarray) -> None:
        """SRKF must produce IDENTICAL results to KF on local level (tol=1e-10)."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        srkf = SquareRootKalmanFilter()
        sr_out = srkf.filter(nile_volume, ssm)

        assert_allclose(
            sr_out.filtered_state,
            kf_out.filtered_state,
            atol=1e-10,
            err_msg="Filtered states differ",
        )
        assert_allclose(
            sr_out.filtered_cov,
            kf_out.filtered_cov,
            atol=1e-10,
            err_msg="Filtered covariances differ",
        )
        assert_allclose(
            sr_out.predicted_state,
            kf_out.predicted_state,
            atol=1e-10,
            err_msg="Predicted states differ",
        )
        assert_allclose(
            sr_out.predicted_cov,
            kf_out.predicted_cov,
            atol=1e-10,
            err_msg="Predicted covariances differ",
        )
        assert sr_out.loglike == pytest.approx(kf_out.loglike, abs=1e-10)

    def test_equivalent_to_kf_multivariate(self) -> None:
        """SRKF must match KF on multivariate model."""
        ssm = _build_multivariate_ssm()
        rng = np.random.default_rng(123)
        y = rng.normal(0, 2, (80, 2))

        kf = KalmanFilter()
        kf_out = kf.filter(y, ssm)

        srkf = SquareRootKalmanFilter()
        sr_out = srkf.filter(y, ssm)

        assert_allclose(sr_out.filtered_state, kf_out.filtered_state, atol=1e-10)
        assert_allclose(sr_out.filtered_cov, kf_out.filtered_cov, atol=1e-10)
        assert sr_out.loglike == pytest.approx(kf_out.loglike, abs=1e-10)

    def test_covariance_positive_definite(self, nile_volume: np.ndarray) -> None:
        """All filtered covariances must be positive-definite."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        srkf = SquareRootKalmanFilter()
        output = srkf.filter(nile_volume, ssm)

        for t in range(len(nile_volume)):
            eigvals = np.linalg.eigvalsh(output.filtered_cov[t])
            assert np.all(eigvals > 0), f"P not PD at t={t}"

    def test_loglike_nile(self, nile_volume: np.ndarray) -> None:
        """Log-likelihood should match KF exactly for Nile with given params."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        srkf = SquareRootKalmanFilter()
        output = srkf.filter(nile_volume, ssm)
        assert output.loglike == pytest.approx(kf_out.loglike, abs=1e-10)

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Output arrays must have correct shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        srkf = SquareRootKalmanFilter()
        output = srkf.filter(nile_volume, ssm)
        nobs = len(nile_volume)
        assert output.filtered_state.shape == (nobs, 1)
        assert output.filtered_cov.shape == (nobs, 1, 1)
        assert output.predicted_state.shape == (nobs, 1)
        assert output.loglike_obs.shape == (nobs,)

    def test_missing_data(self, nile_volume: np.ndarray) -> None:
        """SRKF should handle missing data identically to KF."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        y_missing = nile_volume.copy()
        y_missing[10:15] = np.nan

        kf = KalmanFilter()
        kf_out = kf.filter(y_missing, ssm)

        srkf = SquareRootKalmanFilter()
        sr_out = srkf.filter(y_missing, ssm)

        assert sr_out.nobs_effective == kf_out.nobs_effective
        assert_allclose(sr_out.filtered_state, kf_out.filtered_state, atol=1e-10)

    def test_nobs_effective(self, nile_volume: np.ndarray) -> None:
        """Effective observations should equal total for complete data."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        srkf = SquareRootKalmanFilter()
        output = srkf.filter(nile_volume, ssm)
        assert output.nobs_effective == len(nile_volume)

    def test_forecast_error_variance(self, nile_volume: np.ndarray) -> None:
        """Forecast error variance should match KF exactly."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        srkf = SquareRootKalmanFilter()
        sr_out = srkf.filter(nile_volume, ssm)

        assert_allclose(sr_out.forecast_cov, kf_out.forecast_cov, atol=1e-10)
