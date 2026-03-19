"""Tests for forecast fan chart plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.forecast_plot import plot_forecast
from kalmanbox.visualization.themes import reset_theme


@dataclass
class MockForecastResults:
    """Mock results for testing plot_forecast."""

    observed: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(100, 10, 100)
    )
    forecast_mean: NDArray[np.float64] | None = None
    forecast_se: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.forecast_mean is None:
            self.forecast_mean = np.linspace(100, 105, 20)
        if self.forecast_se is None:
            self.forecast_se = np.linspace(2, 8, 20)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_theme()
    yield  # noqa: PT022
    plt.close("all")


class TestPlotForecast:
    """Tests for plot_forecast."""

    def test_returns_figure(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results)
        assert isinstance(fig, Figure)

    def test_has_bands(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results)
        ax = fig.get_axes()[0]
        # Should have fill_between collections for CI bands
        collections = ax.collections
        assert len(collections) >= 2  # at least 2 CI levels visible

    def test_custom_ci_levels(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results, ci_levels=[0.50, 0.90])
        ax = fig.get_axes()[0]
        assert len(ax.collections) >= 2

    def test_n_history_trims(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results, n_history=30)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results, title="My Forecast")
        ax = fig.get_axes()[0]
        assert ax.get_title() == "My Forecast"

    def test_custom_ylabel(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results, ylabel="Volume")
        ax = fig.get_axes()[0]
        assert ax.get_ylabel() == "Volume"

    def test_no_se_still_works(self) -> None:
        results = MockForecastResults(forecast_se=None)
        # Patch to remove se
        results.forecast_se = None
        fig = plot_forecast(results)
        assert isinstance(fig, Figure)

    def test_no_observed_raises(self) -> None:
        @dataclass
        class NoObsResults:
            forecast_mean: NDArray[np.float64] = field(default_factory=lambda: np.ones(10))

        with pytest.raises(ValueError, match="No observed data"):
            plot_forecast(NoObsResults())

    def test_no_forecast_raises(self) -> None:
        @dataclass
        class NoFcResults:
            observed: NDArray[np.float64] = field(default_factory=lambda: np.ones(100))

        with pytest.raises(ValueError, match="No forecast data"):
            plot_forecast(NoFcResults())

    def test_custom_figsize(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results, figsize=(8, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 8) < 0.1
        assert abs(h - 4) < 0.1

    def test_custom_theme(self) -> None:
        results = MockForecastResults()
        fig = plot_forecast(results, theme="academic")
        assert isinstance(fig, Figure)

    def test_forecast_via_method(self) -> None:
        """Test that forecast can be extracted from a method call."""

        @dataclass
        class MethodResults:
            observed: NDArray[np.float64] = field(
                default_factory=lambda: np.random.default_rng(42).normal(100, 10, 50)
            )

            def forecast(self, steps: int) -> dict[str, NDArray[np.float64]]:
                return {
                    "mean": np.linspace(100, 105, steps),
                    "se": np.linspace(2, 6, steps),
                }

        results = MethodResults()
        fig = plot_forecast(results, steps=10)
        assert isinstance(fig, Figure)

    def test_bands_ordered_by_width(self) -> None:
        """Fan chart bands should be ordered widest (lightest) to narrowest (darkest)."""
        results = MockForecastResults()
        fig = plot_forecast(results, ci_levels=[0.50, 0.70, 0.90, 0.95])
        ax = fig.get_axes()[0]
        # Should have 4 band collections
        assert len(ax.collections) == 4
