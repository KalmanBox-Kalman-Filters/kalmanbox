"""Tests for DiffuseInitialization."""

import numpy as np
import pytest

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.estimation.diffuse import DiffuseInitialization
from kalmanbox.filters.kalman import KalmanFilter


def _build_local_level_ssm(
    sigma2_obs: float, sigma2_level: float, P1_value: float = 1e7
) -> StateSpaceRepresentation:
    """Build local level SSM."""
    ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
    ssm.T = np.array([[1.0]])
    ssm.Z = np.array([[1.0]])
    ssm.R = np.array([[1.0]])
    ssm.H = np.array([[sigma2_obs]])
    ssm.Q = np.array([[sigma2_level]])
    ssm.a1 = np.array([0.0])
    ssm.P1 = np.array([[P1_value]])
    return ssm


def _build_local_linear_trend_ssm(
    sigma2_obs: float, sigma2_level: float, sigma2_trend: float
) -> StateSpaceRepresentation:
    """Build local linear trend SSM."""
    ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
    ssm.T = np.array([[1.0, 1.0], [0.0, 1.0]])
    ssm.Z = np.array([[1.0, 0.0]])
    ssm.R = np.eye(2)
    ssm.H = np.array([[sigma2_obs]])
    ssm.Q = np.diag([sigma2_level, sigma2_trend])
    ssm.a1 = np.zeros(2)
    ssm.P1 = np.eye(2) * 1e7
    return ssm


class TestDiffuseInitialization:
    """Tests for DiffuseInitialization."""

    def test_decompose_initial(self) -> None:
        """P1 should decompose into P_star (finite) + P_inf (diffuse)."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        diffuse = DiffuseInitialization()
        P_star, P_inf = diffuse.decompose_initial(ssm)

        # P_inf should have 1 on diagonal for diffuse state
        assert P_inf[0, 0] == 1.0
        # P_star should have 0 for the diffuse part
        assert P_star[0, 0] == 0.0

    def test_diffuse_loglike_matches_approximate(self, nile_volume: np.ndarray) -> None:
        """Diffuse loglike should be close to approximate (large P1) loglike."""
        ssm = _build_local_level_ssm(15099.0, 1469.0, P1_value=1e7)

        # Approximate filter
        kf = KalmanFilter()
        approx_output = kf.filter(nile_volume, ssm)

        # Diffuse filter
        diffuse = DiffuseInitialization()
        diffuse_output = diffuse.filter(nile_volume, ssm)

        # Diffuse loglike excludes first d observations' data contribution,
        # so the difference is ~9 for local level (dominated by log|F| of
        # the first obs with inflated variance in the approximate filter).
        assert diffuse_output.loglike == pytest.approx(approx_output.loglike, abs=10.0)

    def test_diffuse_local_level_params(self, nile_volume: np.ndarray) -> None:
        """Local Level with diffuse init should match approximate results."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        diffuse = DiffuseInitialization()
        output = diffuse.filter(nile_volume, ssm)

        # Loglike should be close to -632.54
        assert output.loglike == pytest.approx(-632.54, abs=2.0)

    def test_diffuse_period_local_level(self, nile_volume: np.ndarray) -> None:
        """Local Level should have d=1 diffuse period."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        diffuse = DiffuseInitialization()
        output = diffuse.filter(nile_volume, ssm)
        assert output.diffuse_periods == 1

    def test_diffuse_period_llt(self, nile_volume: np.ndarray) -> None:
        """Local Linear Trend should have d=2 diffuse periods."""
        ssm = _build_local_linear_trend_ssm(15099.0, 1469.0, 100.0)
        diffuse = DiffuseInitialization()
        output = diffuse.filter(nile_volume, ssm)
        assert output.diffuse_periods == 2

    def test_p_inf_converges_to_zero(self, nile_volume: np.ndarray) -> None:
        """P_inf should converge to zero after diffuse period."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        diffuse = DiffuseInitialization()
        output = diffuse.filter(nile_volume, ssm)
        assert output.p_inf is not None
        assert np.max(np.abs(output.p_inf)) < 1e-8

    def test_explicit_diffuse_states(self, nile_volume: np.ndarray) -> None:
        """User can explicitly specify which states are diffuse."""
        ssm = _build_local_linear_trend_ssm(15099.0, 1469.0, 100.0)
        mask = np.array([True, True])  # Both states diffuse
        diffuse = DiffuseInitialization(diffuse_states=mask)
        output = diffuse.filter(nile_volume, ssm)
        assert output.diffuse_periods >= 1

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Output shapes should match standard filter output."""
        ssm = _build_local_level_ssm(15099.0, 1469.0)
        diffuse = DiffuseInitialization()
        output = diffuse.filter(nile_volume, ssm)
        nobs = len(nile_volume)
        assert output.filtered_state.shape == (nobs, 1)
        assert output.filtered_cov.shape == (nobs, 1, 1)
        assert output.residuals.shape == (nobs, 1)

    def test_no_diffuse_states(self) -> None:
        """With no diffuse states, should behave like normal filter."""
        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[0.9]])
        ssm.Z = np.array([[1.0]])
        ssm.R = np.array([[1.0]])
        ssm.H = np.array([[1.0]])
        ssm.Q = np.array([[0.5]])
        ssm.a1 = np.array([0.0])
        ssm.P1 = np.array([[1.0]])  # Small P1, not diffuse

        rng = np.random.default_rng(42)
        y = rng.normal(0, 2, 50)

        diffuse = DiffuseInitialization(threshold=1e5)
        output = diffuse.filter(y, ssm)
        assert output.diffuse_periods == 0
