"""Tests for state-space model simulation."""

import numpy as np
import pytest

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.models.local_level import LocalLevel
from kalmanbox.simulation.simulate import simulate_from_model, simulate_missing, simulate_ssm


def _build_local_level_ssm(
    sigma2_obs: float = 15099.0, sigma2_level: float = 1469.0
) -> StateSpaceRepresentation:
    """Build a local level SSM with given variances."""
    ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
    ssm.T = np.array([[1.0]])
    ssm.Z = np.array([[1.0]])
    ssm.R = np.array([[1.0]])
    ssm.H = np.array([[sigma2_obs]])
    ssm.Q = np.array([[sigma2_level]])
    ssm.a1 = np.array([1000.0])
    ssm.P1 = np.array([[1e4]])
    return ssm


class TestSimulateSSM:
    """Tests for simulate_ssm."""

    def test_output_shapes(self) -> None:
        """Simulated output should have correct shapes."""
        ssm = _build_local_level_ssm()
        y, states = simulate_ssm(ssm, n_periods=100, seed=42)
        assert y.shape == (100, 1)
        assert states.shape == (100, 1)

    def test_reproducible_with_seed(self) -> None:
        """Same seed should produce same output."""
        ssm = _build_local_level_ssm()
        y1, s1 = simulate_ssm(ssm, n_periods=50, seed=123)
        y2, s2 = simulate_ssm(ssm, n_periods=50, seed=123)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seed_different_output(self) -> None:
        """Different seeds should produce different output."""
        ssm = _build_local_level_ssm()
        y1, _ = simulate_ssm(ssm, n_periods=50, seed=1)
        y2, _ = simulate_ssm(ssm, n_periods=50, seed=2)
        assert not np.allclose(y1, y2)

    def test_simulate_recover(self) -> None:
        """Filter should recover simulated states (correlation > 0.9).

        This is a key integration test: simulate data, then filter,
        and check that filtered states are correlated with true states.
        """
        ssm = _build_local_level_ssm(sigma2_obs=100.0, sigma2_level=10.0)
        y, true_states = simulate_ssm(ssm, n_periods=200, seed=42)

        kf = KalmanFilter()
        output = kf.filter(y, ssm)

        # Correlation between filtered and true states
        filtered = output.filtered_state[:, 0]
        true = true_states[:, 0]

        # Skip initial diffuse period
        corr = np.corrcoef(filtered[10:], true[10:])[0, 1]
        assert corr > 0.9, f"Correlation {corr:.3f} too low"

    def test_observation_noise(self) -> None:
        """Simulated y should differ from states by observation noise."""
        ssm = _build_local_level_ssm(sigma2_obs=100.0, sigma2_level=0.01)
        y, states = simulate_ssm(ssm, n_periods=500, seed=42)

        # With near-zero state noise, y - states ~ N(0, H)
        diff = y[:, 0] - states[:, 0]
        # Variance should be approximately sigma2_obs
        assert np.var(diff) == pytest.approx(100.0, rel=0.3)


class TestSimulateFromModel:
    """Tests for simulate_from_model."""

    def test_simulate_from_fitted(self, nile_volume: np.ndarray) -> None:
        """Should simulate from a fitted model."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        y_sim, states_sim = simulate_from_model(
            model, n_periods=100, params=results.params, seed=42
        )
        assert y_sim.shape == (100, 1)
        assert states_sim.shape == (100, 1)

    def test_simulate_with_start_params(self, nile_volume: np.ndarray) -> None:
        """Should use start_params when params not provided."""
        model = LocalLevel(nile_volume)
        y_sim, states_sim = simulate_from_model(model, n_periods=50, seed=42)
        assert y_sim.shape[0] == 50


class TestSimulateMissing:
    """Tests for simulate_missing."""

    def test_missing_rate(self) -> None:
        """Correct fraction of observations should be missing."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        y_miss = simulate_missing(y, missing_rate=0.2, seed=42)
        n_missing = np.sum(np.isnan(y_miss))
        assert n_missing == 20

    def test_zero_missing_rate(self) -> None:
        """Zero missing rate should not introduce NaN."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        y_miss = simulate_missing(y, missing_rate=0.0, seed=42)
        assert not np.any(np.isnan(y_miss))

    def test_invalid_missing_rate(self) -> None:
        """Invalid missing rate should raise ValueError."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        with pytest.raises(ValueError, match="missing_rate"):
            simulate_missing(y, missing_rate=1.0)
        with pytest.raises(ValueError, match="missing_rate"):
            simulate_missing(y, missing_rate=-0.1)

    def test_2d_data(self) -> None:
        """Should handle 2D data."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal((100, 2))
        y_miss = simulate_missing(y, missing_rate=0.1, seed=42)
        assert y_miss.shape == (100, 2)
        n_missing = np.sum(np.any(np.isnan(y_miss), axis=1))
        assert n_missing == 10

    def test_preserves_non_missing(self) -> None:
        """Non-missing values should be unchanged."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        y_miss = simulate_missing(y, missing_rate=0.2, seed=42)
        non_missing = ~np.isnan(y_miss)
        np.testing.assert_array_equal(y_miss[non_missing], y[non_missing])
