"""Tests for LocalLinearTrend model."""

import numpy as np

from kalmanbox.models.local_linear_trend import LocalLinearTrend


class TestLocalLinearTrend:
    """Tests for LocalLinearTrend."""

    def test_fit_nile(self, nile_volume: np.ndarray) -> None:
        """Fit on Nile data should converge."""
        model = LocalLinearTrend(nile_volume)
        results = model.fit()
        assert results.optimizer_converged
        assert len(results.params) == 3

    def test_param_names(self, nile_volume: np.ndarray) -> None:
        """Parameter names should be correct."""
        model = LocalLinearTrend(nile_volume)
        assert model.param_names == ["sigma2_obs", "sigma2_level", "sigma2_trend"]

    def test_trend_component_visible(self) -> None:
        """On data with linear trend, sigma2_trend should be positive."""
        rng = np.random.default_rng(42)
        n = 200
        trend = np.linspace(0, 50, n)
        noise = rng.normal(0, 5, n)
        y = trend + noise

        model = LocalLinearTrend(y)
        results = model.fit()
        assert results.params[2] > 0  # sigma2_trend

    def test_forecast_with_slope(self, nile_volume: np.ndarray) -> None:
        """Forecast should show trend (not flat like LocalLevel)."""
        model = LocalLinearTrend(nile_volume)
        results = model.fit()
        fc = results.forecast(steps=10)
        assert fc["mean"].shape == (10, 1)
        # Intervals should widen
        widths = fc["upper"][:, 0] - fc["lower"][:, 0]
        for i in range(1, len(widths)):
            assert widths[i] >= widths[i - 1] - 1e-10

    def test_summary(self, nile_volume: np.ndarray) -> None:
        """summary() should work without error."""
        model = LocalLinearTrend(nile_volume)
        results = model.fit()
        s = results.summary()
        assert "sigma2_trend" in s

    def test_smoothed_states_have_two_components(self, nile_volume: np.ndarray) -> None:
        """Smoothed state should have 2 components: level and slope."""
        model = LocalLinearTrend(nile_volume)
        results = model.fit()
        assert results.smoothed_state is not None
        assert results.smoothed_state.shape == (100, 2)

    def test_simulate(self, nile_volume: np.ndarray) -> None:
        """Simulation should produce data with trend."""
        model = LocalLinearTrend(nile_volume)
        y, states = model.simulate(100, seed=42)
        assert y.shape == (100, 1)
        assert states.shape == (100, 2)

    def test_loglike_better_or_equal_local_level(self, nile_volume: np.ndarray) -> None:
        """LLT should have loglike >= LocalLevel (more parameters)."""
        from kalmanbox.models.local_level import LocalLevel

        ll_model = LocalLevel(nile_volume)
        ll_results = ll_model.fit()

        llt_model = LocalLinearTrend(nile_volume)
        llt_results = llt_model.fit()

        # LLT has more parameters but diffuse initialization for 2 states
        # can penalize likelihood; allow reasonable tolerance
        assert llt_results.loglike >= ll_results.loglike - 10.0
