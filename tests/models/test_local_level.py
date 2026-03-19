"""Tests for LocalLevel model."""

import numpy as np
import pytest

from kalmanbox.models.local_level import LocalLevel


class TestLocalLevel:
    """Tests for LocalLevel model."""

    def test_fit_nile(self, nile_volume: np.ndarray) -> None:
        """Full fit on Nile data should match reference values."""
        model = LocalLevel(nile_volume)
        results = model.fit()

        # Parameters close to Durbin & Koopman
        assert results.params[0] == pytest.approx(15099.0, rel=0.05)
        assert results.params[1] == pytest.approx(1469.0, rel=0.05)
        assert results.loglike == pytest.approx(-632.54, abs=1.0)

    def test_forecast(self, nile_volume: np.ndarray) -> None:
        """Forecast should be flat (random walk) with widening intervals."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        fc = results.forecast(steps=10)

        # Mean should be constant (random walk forecast)
        means = fc["mean"][:, 0]
        np.testing.assert_allclose(means, means[0], atol=1e-10)

        # Intervals should widen
        widths = fc["upper"][:, 0] - fc["lower"][:, 0]
        for i in range(1, len(widths)):
            assert widths[i] >= widths[i - 1] - 1e-10

    def test_summary(self, nile_volume: np.ndarray) -> None:
        """summary() should produce a formatted string without error."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        s = results.summary()
        assert isinstance(s, str)
        assert "sigma2_obs" in s
        assert "sigma2_level" in s
        assert "Log-Likelihood" in s

    def test_start_params(self, nile_volume: np.ndarray) -> None:
        """Start params should be positive and reasonable."""
        model = LocalLevel(nile_volume)
        sp = model.start_params
        assert len(sp) == 2
        assert all(p > 0 for p in sp)

    def test_param_names(self, nile_volume: np.ndarray) -> None:
        """Parameter names should be correct."""
        model = LocalLevel(nile_volume)
        assert model.param_names == ["sigma2_obs", "sigma2_level"]

    def test_simulate(self, nile_volume: np.ndarray) -> None:
        """Simulation should produce reasonable output."""
        model = LocalLevel(nile_volume)
        params = np.array([15099.0, 1469.0])
        y, states = model.simulate(100, params, seed=42)
        assert y.shape == (100, 1)
        assert states.shape == (100, 1)
        # Simulated data should have reasonable variance
        assert np.var(y) > 0

    def test_filter_only(self, nile_volume: np.ndarray) -> None:
        """Filter-only should work without smoother."""
        model = LocalLevel(nile_volume)
        params = np.array([15099.0, 1469.0])
        results = model.filter(params)
        assert results.smoother_output is None
        assert results.smoothed_state is None

    def test_smooth_only(self, nile_volume: np.ndarray) -> None:
        """Smooth should include smoother output."""
        model = LocalLevel(nile_volume)
        params = np.array([15099.0, 1469.0])
        results = model.smooth(params)
        assert results.smoother_output is not None
        assert results.smoothed_state is not None
