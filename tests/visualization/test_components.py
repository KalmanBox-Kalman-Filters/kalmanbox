"""Tests for component decomposition plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.components import plot_components
from kalmanbox.visualization.themes import reset_theme


@dataclass
class MockResults:
    """Mock results object for testing plot_components."""

    observed: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(100, 10, 100)
    )
    fitted_values: NDArray[np.float64] | None = None
    trend: NDArray[np.float64] | None = None
    trend_std: NDArray[np.float64] | None = None
    slope: NDArray[np.float64] | None = None
    seasonal: NDArray[np.float64] | None = None
    seasonal_std: NDArray[np.float64] | None = None
    cycle: NDArray[np.float64] | None = None
    irregular: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(42)
        n = len(self.observed)
        if self.fitted_values is None:
            self.fitted_values = self.observed + rng.normal(0, 2, n)
        if self.trend is None:
            self.trend = np.linspace(90, 110, n) + rng.normal(0, 1, n)
        if self.trend_std is None:
            self.trend_std = np.full(n, 2.0)
        if self.irregular is None:
            self.irregular = rng.normal(0, 1, n)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_theme()
    yield  # noqa: PT022
    plt.close("all")


class TestPlotComponents:
    """Tests for plot_components."""

    def test_returns_figure(self) -> None:
        results = MockResults()
        fig = plot_components(results)
        assert isinstance(fig, Figure)

    def test_generates_multiple_panels(self) -> None:
        results = MockResults()
        fig = plot_components(results)
        axes = fig.get_axes()
        assert len(axes) >= 3  # observed, trend, irregular at minimum

    def test_which_filters_components(self) -> None:
        results = MockResults()
        fig = plot_components(results, which=["observed", "trend"])
        axes = fig.get_axes()
        assert len(axes) == 2

    def test_single_component(self) -> None:
        results = MockResults()
        fig = plot_components(results, which=["observed"])
        axes = fig.get_axes()
        assert len(axes) == 1

    def test_with_seasonal(self) -> None:
        results = MockResults(
            seasonal=np.sin(np.linspace(0, 4 * np.pi, 100)) * 5,
            seasonal_std=np.full(100, 1.0),
        )
        fig = plot_components(results, which=["seasonal"])
        assert isinstance(fig, Figure)

    def test_custom_theme(self) -> None:
        results = MockResults()
        fig = plot_components(results, theme="academic")
        assert isinstance(fig, Figure)

    def test_custom_figsize(self) -> None:
        results = MockResults()
        fig = plot_components(results, figsize=(8, 4), which=["observed"])
        w, h = fig.get_size_inches()
        assert abs(w - 8) < 0.1
        assert abs(h - 4) < 0.1

    def test_custom_title(self) -> None:
        results = MockResults()
        fig = plot_components(results, title="My Decomposition", which=["observed"])
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "My Decomposition"

    def test_custom_ci(self) -> None:
        results = MockResults()
        fig = plot_components(results, ci=0.99, which=["trend"])
        assert isinstance(fig, Figure)

    def test_no_components_raises(self) -> None:
        @dataclass
        class EmptyResults:
            pass

        with pytest.raises(ValueError, match="No components found"):
            plot_components(EmptyResults())

    def test_nonexistent_component_ignored(self) -> None:
        results = MockResults()
        fig = plot_components(results, which=["observed", "nonexistent"])
        axes = fig.get_axes()
        assert len(axes) == 1  # only observed found
