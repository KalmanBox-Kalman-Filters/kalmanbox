"""Tests for EnsembleKalmanFilter."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.enkf import (
    EnKFModel,
    EnsembleKalmanFilter,
    LinearEnKFModel,
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


def _build_high_dim_ssm(k_states: int) -> StateSpaceRepresentation:
    """Build a high-dimensional SSM for scalability testing."""
    ssm = StateSpaceRepresentation(k_states=k_states, k_endog=1)
    ssm.T = 0.95 * np.eye(k_states)
    ssm.Z = np.ones((1, k_states)) / k_states
    ssm.R = np.eye(k_states)
    ssm.H = np.array([[1.0]])
    ssm.Q = 0.1 * np.eye(k_states)
    ssm.a1 = np.zeros(k_states)
    ssm.P1 = np.eye(k_states)
    return ssm


class TestEnsembleKalmanFilter:
    """Tests for EnsembleKalmanFilter."""

    def test_converges_to_kf_large_ensemble(self, nile_volume: np.ndarray) -> None:
        """With large ensemble, EnKF should approximate KF closely."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        model = LinearEnKFModel(ssm)
        enkf = EnsembleKalmanFilter(n_ensemble=5000, random_state=42)
        enkf_out = enkf.filter(nile_volume, model)

        # With 5000 members, should be within ~1% of exact
        assert_allclose(
            enkf_out.filtered_state[20:],
            kf_out.filtered_state[20:],
            rtol=0.05,
            err_msg="EnKF with large ensemble should approximate KF",
        )

    def test_runs_on_local_level(self, nile_volume: np.ndarray) -> None:
        """EnKF should run without error on local level model."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        model = LinearEnKFModel(ssm)
        enkf = EnsembleKalmanFilter(n_ensemble=100, random_state=42)
        output = enkf.filter(nile_volume, model)

        assert output.filtered_state.shape == (len(nile_volume), 1)
        assert output.nobs_effective == len(nile_volume)
        assert np.isfinite(output.loglike)

    def test_output_shapes(self) -> None:
        """Output arrays must have correct shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        model = LinearEnKFModel(ssm)
        rng = np.random.default_rng(42)
        y = rng.normal(1000, 200, 50)

        enkf = EnsembleKalmanFilter(n_ensemble=50, random_state=42)
        output = enkf.filter(y, model)
        assert output.filtered_state.shape == (50, 1)
        assert output.filtered_cov.shape == (50, 1, 1)
        assert output.predicted_state.shape == (50, 1)
        assert output.loglike_obs.shape == (50,)

    def test_reproducibility(self, nile_volume: np.ndarray) -> None:
        """Same random_state should produce identical results."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        model = LinearEnKFModel(ssm)

        enkf1 = EnsembleKalmanFilter(n_ensemble=100, random_state=42)
        out1 = enkf1.filter(nile_volume, model)

        enkf2 = EnsembleKalmanFilter(n_ensemble=100, random_state=42)
        out2 = enkf2.filter(nile_volume, model)

        assert_allclose(out1.filtered_state, out2.filtered_state)
        assert out1.loglike == pytest.approx(out2.loglike)

    def test_different_seeds_differ(self, nile_volume: np.ndarray) -> None:
        """Different random seeds should produce different results."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        model = LinearEnKFModel(ssm)

        enkf1 = EnsembleKalmanFilter(n_ensemble=50, random_state=42)
        out1 = enkf1.filter(nile_volume, model)

        enkf2 = EnsembleKalmanFilter(n_ensemble=50, random_state=99)
        out2 = enkf2.filter(nile_volume, model)

        assert not np.allclose(out1.filtered_state, out2.filtered_state)

    def test_high_dimensional(self) -> None:
        """EnKF should handle high-dimensional (k_states=200) models."""
        ssm = _build_high_dim_ssm(k_states=200)
        model = LinearEnKFModel(ssm)
        rng = np.random.default_rng(42)
        y = rng.normal(0, 2, 30)

        enkf = EnsembleKalmanFilter(n_ensemble=50, random_state=42)
        output = enkf.filter(y, model)

        assert output.filtered_state.shape == (30, 200)
        assert output.nobs_effective == 30
        assert np.isfinite(output.loglike)

    def test_inflation(self, nile_volume: np.ndarray) -> None:
        """Inflation factor > 1 should increase filtered covariance."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        model = LinearEnKFModel(ssm)

        enkf_no_infl = EnsembleKalmanFilter(n_ensemble=100, inflation=1.0, random_state=42)
        out_no = enkf_no_infl.filter(nile_volume, model)

        enkf_infl = EnsembleKalmanFilter(n_ensemble=100, inflation=1.1, random_state=42)
        out_infl = enkf_infl.filter(nile_volume, model)

        # Inflated ensemble should have larger predicted covariance
        assert np.mean(out_infl.predicted_cov[5:]) > np.mean(out_no.predicted_cov[5:])

    def test_missing_data(self, nile_volume: np.ndarray) -> None:
        """EnKF should handle missing data."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        model = LinearEnKFModel(ssm)
        y_missing = nile_volume.copy()
        y_missing[10:15] = np.nan

        enkf = EnsembleKalmanFilter(n_ensemble=100, random_state=42)
        output = enkf.filter(y_missing, model)

        assert output.nobs_effective == len(nile_volume) - 5
        # Filtered state at missing positions should equal predicted
        for t_idx in range(10, 15):
            assert_allclose(output.filtered_state[t_idx], output.predicted_state[t_idx])

    def test_linear_enkf_model_protocol(self) -> None:
        """LinearEnKFModel should satisfy EnKFModel protocol."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        model = LinearEnKFModel(ssm)
        assert isinstance(model, EnKFModel)
