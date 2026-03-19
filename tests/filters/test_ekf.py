"""Tests for ExtendedKalmanFilter."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.ekf import (
    EKFModel,
    ExtendedKalmanFilter,
    LinearEKFModel,
)
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


class SimpleNonlinearModel:
    """A simple nonlinear model for testing.

    State: x_{t+1} = 0.5 * x_t + 25 * x_t / (1 + x_t^2) + 8 * cos(1.2 * t) + eta_t
    Obs:   y_t = x_t^2 / 20 + eps_t

    This is the classic benchmark from Julier & Uhlmann (1997).
    """

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

    def transition_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        x = alpha[0]
        df_dx = 0.5 + 25.0 * (1.0 - x**2) / (1.0 + x**2) ** 2
        return np.array([[df_dx]])

    def observation(self, alpha: np.ndarray, t: int) -> np.ndarray:
        x = alpha[0]
        return np.array([x**2 / 20.0])

    def observation_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        x = alpha[0]
        return np.array([[x / 10.0]])


class TestLinearEKFModel:
    """Tests for LinearEKFModel adapter."""

    def test_is_ekf_model(self) -> None:
        """LinearEKFModel should satisfy EKFModel protocol."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        model = LinearEKFModel(ssm)
        assert isinstance(model, EKFModel)

    def test_transition_equals_T_alpha_plus_c(self) -> None:
        """Linear transition should be T @ alpha + c."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        model = LinearEKFModel(ssm)
        alpha = np.array([100.0])
        result = model.transition(alpha, 0)
        expected = ssm.T @ alpha + ssm.c
        assert_allclose(result, expected)

    def test_transition_jacobian_equals_T(self) -> None:
        """Transition Jacobian should be constant T."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        model = LinearEKFModel(ssm)
        alpha = np.array([100.0])
        J = model.transition_jacobian(alpha, 0)
        assert_allclose(J, ssm.T)

    def test_observation_jacobian_equals_Z(self) -> None:
        """Observation Jacobian should be constant Z."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        model = LinearEKFModel(ssm)
        alpha = np.array([100.0])
        J = model.observation_jacobian(alpha, 0)
        assert_allclose(J, ssm.Z)


class TestExtendedKalmanFilter:
    """Tests for ExtendedKalmanFilter."""

    def test_ekf_on_linear_equals_kf(self, nile_volume: np.ndarray) -> None:
        """EKF with LinearEKFModel must produce IDENTICAL results to KF."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        model = LinearEKFModel(ssm)
        ekf = ExtendedKalmanFilter()
        ekf_out = ekf.filter(nile_volume, model)

        assert_allclose(
            ekf_out.filtered_state,
            kf_out.filtered_state,
            atol=1e-10,
            err_msg="EKF filtered states differ from KF on linear model",
        )
        assert_allclose(
            ekf_out.filtered_cov,
            kf_out.filtered_cov,
            atol=1e-10,
            err_msg="EKF filtered covariances differ from KF on linear model",
        )
        assert ekf_out.loglike == pytest.approx(kf_out.loglike, abs=1e-10)

    def test_ekf_nonlinear_runs(self) -> None:
        """EKF should run without error on a nonlinear model."""
        model = SimpleNonlinearModel()
        rng = np.random.default_rng(42)

        # Generate synthetic data from the model
        nobs = 50
        x = np.zeros(nobs)
        y = np.zeros(nobs)
        x[0] = rng.normal(0, np.sqrt(5.0))
        y[0] = x[0] ** 2 / 20.0 + rng.normal(0, 1.0)
        for t in range(1, nobs):
            x[t] = (
                0.5 * x[t - 1]
                + 25.0 * x[t - 1] / (1.0 + x[t - 1] ** 2)
                + 8.0 * np.cos(1.2 * t)
                + rng.normal(0, np.sqrt(10.0))
            )
            y[t] = x[t] ** 2 / 20.0 + rng.normal(0, 1.0)

        ekf = ExtendedKalmanFilter()
        output = ekf.filter(y, model)

        assert output.filtered_state.shape == (nobs, 1)
        assert output.nobs_effective == nobs
        assert np.isfinite(output.loglike)

    def test_ekf_nonlinear_tracks_state(self) -> None:
        """EKF on nonlinear model should track the true state reasonably.

        Uses a model with nonlinear transition but linear observation,
        where EKF linearization is adequate.
        """
        model = SimpleNonlinearModel()
        # Use linear observation (x directly) so EKF doesn't face sign ambiguity
        model.H = np.array([[2.0]])
        rng = np.random.default_rng(42)

        nobs = 100
        x_true = np.zeros(nobs)
        y = np.zeros(nobs)
        x_true[0] = rng.normal(0, np.sqrt(5.0))
        y[0] = x_true[0] + rng.normal(0, np.sqrt(2.0))
        for t in range(1, nobs):
            x_true[t] = (
                0.5 * x_true[t - 1]
                + 25.0 * x_true[t - 1] / (1.0 + x_true[t - 1] ** 2)
                + 8.0 * np.cos(1.2 * t)
                + rng.normal(0, np.sqrt(10.0))
            )
            y[t] = x_true[t] + rng.normal(0, np.sqrt(2.0))

        # Override observation to be linear: h(x) = x
        class LinearObsModel(SimpleNonlinearModel):
            def __init__(self) -> None:
                super().__init__()
                self.H = np.array([[2.0]])

            def observation(self, alpha: np.ndarray, t: int) -> np.ndarray:
                return alpha.copy()

            def observation_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
                return np.array([[1.0]])

        lin_obs_model = LinearObsModel()
        ekf = ExtendedKalmanFilter()
        output = ekf.filter(y, lin_obs_model)

        # The RMSE should be less than the state std
        rmse = np.sqrt(np.mean((output.filtered_state[:, 0] - x_true) ** 2))
        state_std = np.std(x_true)
        assert rmse < state_std, f"RMSE {rmse:.2f} should be < state_std {state_std:.2f}"

    def test_ekf_loglike_nile(self, nile_volume: np.ndarray) -> None:
        """EKF on linear model should match Nile reference loglike."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        model = LinearEKFModel(ssm)
        ekf = ExtendedKalmanFilter()
        output = ekf.filter(nile_volume, model)
        # Reference: KF on this exact model produces -641.59
        assert output.loglike == pytest.approx(-641.59, abs=0.5)

    def test_ekf_output_shapes(self) -> None:
        """Output arrays must have correct shapes for nonlinear model."""
        model = SimpleNonlinearModel()
        rng = np.random.default_rng(42)
        y = rng.normal(0, 5, 30)

        ekf = ExtendedKalmanFilter()
        output = ekf.filter(y, model)
        assert output.filtered_state.shape == (30, 1)
        assert output.filtered_cov.shape == (30, 1, 1)
        assert output.predicted_state.shape == (30, 1)
        assert output.loglike_obs.shape == (30,)

    def test_ekf_missing_data(self, nile_volume: np.ndarray) -> None:
        """EKF should handle missing data identically to KF on linear model."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        y_missing = nile_volume.copy()
        y_missing[10:15] = np.nan

        kf = KalmanFilter()
        kf_out = kf.filter(y_missing, ssm)

        model = LinearEKFModel(ssm)
        ekf = ExtendedKalmanFilter()
        ekf_out = ekf.filter(y_missing, model)

        assert ekf_out.nobs_effective == kf_out.nobs_effective
        assert_allclose(ekf_out.filtered_state, kf_out.filtered_state, atol=1e-10)

    def test_simple_nonlinear_model_protocol(self) -> None:
        """SimpleNonlinearModel should satisfy EKFModel protocol."""
        model = SimpleNonlinearModel()
        assert isinstance(model, EKFModel)
