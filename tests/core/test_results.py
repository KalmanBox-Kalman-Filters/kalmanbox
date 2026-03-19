"""Tests for StateSpaceResults."""

import numpy as np
import pytest

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.core.results import StateSpaceResults
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.rts import RTSSmoother


def _make_results(nile_volume: np.ndarray) -> StateSpaceResults:
    """Create a StateSpaceResults for testing."""
    ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
    ssm.T = np.array([[1.0]])
    ssm.Z = np.array([[1.0]])
    ssm.R = np.array([[1.0]])
    ssm.H = np.array([[15099.0]])
    ssm.Q = np.array([[1469.0]])
    ssm.a1 = np.array([0.0])
    ssm.P1 = np.array([[1e7]])

    kf = KalmanFilter()
    smoother = RTSSmoother()
    filter_output = kf.filter(nile_volume, ssm)
    smoother_output = smoother.smooth(filter_output, ssm)

    return StateSpaceResults(
        params=np.array([15099.0, 1469.0]),
        param_names=["sigma2_obs", "sigma2_level"],
        se=np.array([3520.0, 1120.0]),
        loglike=filter_output.loglike,
        nobs=len(nile_volume),
        filter_output=filter_output,
        smoother_output=smoother_output,
        ssm=ssm,
    )


class TestStateSpaceResults:
    """Tests for StateSpaceResults."""

    def test_aic_bic(self, nile_volume: np.ndarray) -> None:
        """Test AIC = -2*loglike + 2*k, BIC = -2*loglike + k*log(n)."""
        results = _make_results(nile_volume)
        k = 2
        n = 100
        expected_aic = -2 * results.loglike + 2 * k
        expected_bic = -2 * results.loglike + k * np.log(n)
        assert results.aic == pytest.approx(expected_aic, rel=1e-10)
        assert results.bic == pytest.approx(expected_bic, rel=1e-10)

    def test_tvalues(self, nile_volume: np.ndarray) -> None:
        """Test t-values are params / se."""
        results = _make_results(nile_volume)
        expected = results.params / results.se
        np.testing.assert_allclose(results.tvalues, expected)

    def test_pvalues_range(self, nile_volume: np.ndarray) -> None:
        """Test p-values are between 0 and 1."""
        results = _make_results(nile_volume)
        assert all(0 <= p <= 1 for p in results.pvalues)

    def test_summary_string(self, nile_volume: np.ndarray) -> None:
        """Test summary produces a formatted string without error."""
        results = _make_results(nile_volume)
        s = results.summary()
        assert isinstance(s, str)
        assert "Log-Likelihood" in s
        assert "AIC" in s
        assert "BIC" in s
        assert "sigma2_obs" in s
        assert "sigma2_level" in s

    def test_to_dataframe(self, nile_volume: np.ndarray) -> None:
        """Test conversion to DataFrame."""
        results = _make_results(nile_volume)
        df = results.to_dataframe()
        assert list(df.columns) == ["estimate", "std_error", "t_value", "p_value"]
        assert list(df.index) == ["sigma2_obs", "sigma2_level"]

    def test_forecast(self, nile_volume: np.ndarray) -> None:
        """Test forecast returns dict with correct keys and shapes."""
        results = _make_results(nile_volume)
        fc = results.forecast(steps=10)
        assert "mean" in fc
        assert "lower" in fc
        assert "upper" in fc
        assert fc["mean"].shape == (10, 1)
        assert fc["lower"].shape == (10, 1)
        assert fc["upper"].shape == (10, 1)

    def test_forecast_intervals_widen(self, nile_volume: np.ndarray) -> None:
        """Forecast intervals should widen with horizon (random walk)."""
        results = _make_results(nile_volume)
        fc = results.forecast(steps=20)
        widths = fc["upper"][:, 0] - fc["lower"][:, 0]
        # Each width should be >= previous (for random walk)
        for i in range(1, len(widths)):
            assert widths[i] >= widths[i - 1] - 1e-10

    def test_forecast_mean_flat(self, nile_volume: np.ndarray) -> None:
        """For local level (random walk), forecast mean should be constant."""
        results = _make_results(nile_volume)
        fc = results.forecast(steps=10)
        means = fc["mean"][:, 0]
        np.testing.assert_allclose(means, means[0], atol=1e-10)

    def test_save_load(self, nile_volume: np.ndarray, tmp_path) -> None:
        """Test save/load roundtrip."""
        results = _make_results(nile_volume)
        path = tmp_path / "results.pkl"
        results.save(path)
        loaded = StateSpaceResults.load(path)
        assert loaded.loglike == pytest.approx(results.loglike)
        np.testing.assert_allclose(loaded.params, results.params)

    def test_filtered_smoothed_properties(self, nile_volume: np.ndarray) -> None:
        """Test property accessors."""
        results = _make_results(nile_volume)
        assert results.filtered_state.shape == (100, 1)
        assert results.smoothed_state is not None
        assert results.smoothed_state.shape == (100, 1)
        assert results.residuals.shape == (100, 1)
        assert results.fitted_values.shape == (100, 1)
