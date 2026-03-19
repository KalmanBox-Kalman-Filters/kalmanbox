"""Tests for UnscentedKalmanFilter."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.filters.ukf import (
    LinearUKFModel,
    UKFModel,
    UnscentedKalmanFilter,
)


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
    """Build a 2-state, 2-obs SSM."""
    ssm = StateSpaceRepresentation(k_states=2, k_endog=2)
    ssm.T = np.array([[0.9, 0.1], [0.0, 0.95]])
    ssm.Z = np.array([[1.0, 0.0], [0.5, 1.0]])
    ssm.R = np.eye(2)
    ssm.H = np.diag([1.0, 2.0])
    ssm.Q = np.diag([0.5, 0.3])
    ssm.a1 = np.zeros(2)
    ssm.P1 = np.eye(2) * 10.0
    return ssm


class SimpleNonlinearModel:
    """Simple nonlinear model for testing (no Jacobians needed)."""

    def __init__(self) -> None:
        self.k_states = 1
        self.k_endog = 1
        self.R = np.array([[1.0]])
        self.Q = np.array([[10.0]])
        self.H = np.array([[1.0]])
        self.a1 = np.array([0.0])
        self.P1 = np.array([[5.0]])

    def transition(self, alpha: np.ndarray, t: int) -> np.ndarray:
        x = alpha[0]
        return np.array([0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)])

    def observation(self, alpha: np.ndarray, t: int) -> np.ndarray:
        x = alpha[0]
        return np.array([x**2 / 20.0])


class TestSigmaPoints:
    """Tests for sigma point generation and weights."""

    def test_num_sigma_points(self) -> None:
        """Should generate 2*n+1 sigma points."""
        ukf = UnscentedKalmanFilter()
        a = np.zeros(3)
        P = np.eye(3)
        X = ukf._generate_sigma_points(a, P, 3)
        assert X.shape == (7, 3)

    def test_sigma_points_mean(self) -> None:
        """Weighted mean of sigma points should equal the input mean."""
        ukf = UnscentedKalmanFilter(alpha=1.0, beta=0.0, kappa=0.0)
        n = 2
        a = np.array([3.0, -1.0])
        P = np.array([[4.0, 0.5], [0.5, 2.0]])

        X = ukf._generate_sigma_points(a, P, n)
        W_m, _ = ukf._compute_weights(n)

        mean = np.zeros(n)
        for j in range(2 * n + 1):
            mean += W_m[j] * X[j]
        assert_allclose(mean, a, atol=1e-12)

    def test_sigma_points_covariance(self) -> None:
        """Weighted covariance of sigma points should equal input P."""
        ukf = UnscentedKalmanFilter(alpha=1.0, beta=0.0, kappa=0.0)
        n = 2
        a = np.array([3.0, -1.0])
        P = np.array([[4.0, 0.5], [0.5, 2.0]])

        X = ukf._generate_sigma_points(a, P, n)
        W_m, W_c = ukf._compute_weights(n)

        mean = np.zeros(n)
        for j in range(2 * n + 1):
            mean += W_m[j] * X[j]

        cov = np.zeros((n, n))
        for j in range(2 * n + 1):
            diff = X[j] - mean
            cov += W_c[j] * np.outer(diff, diff)

        assert_allclose(cov, P, atol=1e-10)

    def test_weights_sum_to_one(self) -> None:
        """Mean weights should sum to 1."""
        ukf = UnscentedKalmanFilter()
        W_m, W_c = ukf._compute_weights(5)
        assert W_m.sum() == pytest.approx(1.0, abs=1e-10)


class TestUnscentedKalmanFilter:
    """Tests for UnscentedKalmanFilter."""

    def test_ukf_on_linear_equals_kf(self, nile_volume: np.ndarray) -> None:
        """UKF with LinearUKFModel must match KF on linear model."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        model = LinearUKFModel(ssm)
        ukf = UnscentedKalmanFilter(alpha=1.0, beta=0.0, kappa=0.0)
        ukf_out = ukf.filter(nile_volume, model)

        assert_allclose(
            ukf_out.filtered_state,
            kf_out.filtered_state,
            atol=1e-8,
            err_msg="UKF filtered states differ from KF on linear model",
        )
        assert_allclose(
            ukf_out.filtered_cov,
            kf_out.filtered_cov,
            atol=1e-8,
            err_msg="UKF filtered covariances differ from KF on linear model",
        )
        assert ukf_out.loglike == pytest.approx(kf_out.loglike, abs=1e-6)

    def test_ukf_on_linear_multivariate(self) -> None:
        """UKF must match KF on multivariate linear model."""
        ssm = _build_multivariate_ssm()
        rng = np.random.default_rng(42)
        y = rng.normal(0, 2, (80, 2))

        kf = KalmanFilter()
        kf_out = kf.filter(y, ssm)

        model = LinearUKFModel(ssm)
        ukf = UnscentedKalmanFilter(alpha=1.0, beta=0.0, kappa=0.0)
        ukf_out = ukf.filter(y, model)

        assert_allclose(ukf_out.filtered_state, kf_out.filtered_state, atol=1e-8)
        assert ukf_out.loglike == pytest.approx(kf_out.loglike, abs=1e-6)

    def test_ukf_nonlinear_runs(self) -> None:
        """UKF should run without error on a nonlinear model."""
        model = SimpleNonlinearModel()
        rng = np.random.default_rng(42)

        nobs = 50
        x = np.zeros(nobs)
        y = np.zeros(nobs)
        x[0] = rng.normal(0, np.sqrt(5.0))
        y[0] = x[0] ** 2 / 20.0 + rng.normal(0, 1.0)
        for t_idx in range(1, nobs):
            x[t_idx] = (
                0.5 * x[t_idx - 1]
                + 25.0 * x[t_idx - 1] / (1.0 + x[t_idx - 1] ** 2)
                + 8.0 * np.cos(1.2 * t_idx)
                + rng.normal(0, np.sqrt(10.0))
            )
            y[t_idx] = x[t_idx] ** 2 / 20.0 + rng.normal(0, 1.0)

        ukf = UnscentedKalmanFilter()
        output = ukf.filter(y, model)

        assert output.filtered_state.shape == (nobs, 1)
        assert output.nobs_effective == nobs
        assert np.isfinite(output.loglike)

    def test_ukf_output_shapes(self) -> None:
        """Output arrays must have correct shapes."""
        model = SimpleNonlinearModel()
        rng = np.random.default_rng(42)
        y = rng.normal(0, 5, 30)

        ukf = UnscentedKalmanFilter()
        output = ukf.filter(y, model)
        assert output.filtered_state.shape == (30, 1)
        assert output.filtered_cov.shape == (30, 1, 1)
        assert output.predicted_state.shape == (30, 1)
        assert output.loglike_obs.shape == (30,)

    def test_ukf_missing_data(self, nile_volume: np.ndarray) -> None:
        """UKF should handle missing data."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        y_missing = nile_volume.copy()
        y_missing[10:15] = np.nan

        model = LinearUKFModel(ssm)
        ukf = UnscentedKalmanFilter(alpha=1.0, beta=0.0, kappa=0.0)
        output = ukf.filter(y_missing, model)

        assert output.nobs_effective == len(nile_volume) - 5

    def test_linear_ukf_model_protocol(self) -> None:
        """LinearUKFModel should satisfy UKFModel protocol."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        model = LinearUKFModel(ssm)
        assert isinstance(model, UKFModel)

    def test_tuning_parameters(self) -> None:
        """Different tuning parameters should produce different results on nonlinear."""
        model = SimpleNonlinearModel()
        rng = np.random.default_rng(42)
        y = rng.normal(0, 5, 30)

        ukf1 = UnscentedKalmanFilter(alpha=0.001, beta=2.0, kappa=0.0)
        out1 = ukf1.filter(y, model)

        ukf2 = UnscentedKalmanFilter(alpha=1.0, beta=2.0, kappa=0.0)
        out2 = ukf2.filter(y, model)

        # Results should differ for nonlinear model
        assert not np.allclose(out1.filtered_state, out2.filtered_state)
