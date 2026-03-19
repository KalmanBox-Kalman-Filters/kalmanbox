"""Tests for factor model plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.factor_plot import (
    plot_factors,
    plot_loadings,
    plot_variance_decomposition,
)
from kalmanbox.visualization.themes import reset_theme


@dataclass
class MockFactorResults:
    """Mock results for factor model plots."""

    factors: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 1, (100, 3))
    )
    loadings: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).uniform(-1, 1, (5, 3))
    )
    idiosyncratic_var: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).uniform(0.1, 0.5, 5)
    )
    variance_decomposition: NDArray[np.float64] | None = None


@pytest.fixture(autouse=True)
def _reset():
    reset_theme()
    yield
    plt.close("all")


class TestPlotFactors:
    """Tests for plot_factors."""

    def test_returns_figure(self) -> None:
        results = MockFactorResults()
        fig = plot_factors(results)
        assert isinstance(fig, Figure)

    def test_plots_all_factors(self) -> None:
        results = MockFactorResults()
        fig = plot_factors(results)
        assert len(fig.get_axes()) == 3

    def test_select_factors(self) -> None:
        results = MockFactorResults()
        fig = plot_factors(results, factors=[0, 2])
        assert len(fig.get_axes()) == 2

    def test_single_factor(self) -> None:
        results = MockFactorResults()
        fig = plot_factors(results, factors=[0])
        assert len(fig.get_axes()) == 1

    def test_custom_names(self) -> None:
        results = MockFactorResults()
        fig = plot_factors(results, factor_names=["Activity", "Inflation", "Finance"])
        assert isinstance(fig, Figure)

    def test_no_factors_raises(self) -> None:
        @dataclass
        class EmptyResults:
            pass

        with pytest.raises(ValueError, match="No factor data"):
            plot_factors(EmptyResults())


class TestPlotLoadings:
    """Tests for plot_loadings."""

    def test_returns_figure(self) -> None:
        results = MockFactorResults()
        fig = plot_loadings(results)
        assert isinstance(fig, Figure)

    def test_has_colorbar(self) -> None:
        results = MockFactorResults()
        fig = plot_loadings(results)
        # Figure should have an extra axis for colorbar
        assert len(fig.get_axes()) >= 2

    def test_custom_names(self) -> None:
        results = MockFactorResults()
        fig = plot_loadings(
            results,
            series_names=["GDP", "CPI", "IP", "Rate", "M2"],
            factor_names=["F1", "F2", "F3"],
        )
        assert isinstance(fig, Figure)

    def test_no_annotate(self) -> None:
        results = MockFactorResults()
        fig = plot_loadings(results, annotate=False)
        assert isinstance(fig, Figure)

    def test_no_loadings_raises(self) -> None:
        @dataclass
        class EmptyResults:
            pass

        with pytest.raises(ValueError, match="No loadings"):
            plot_loadings(EmptyResults())


class TestPlotVarianceDecomposition:
    """Tests for plot_variance_decomposition."""

    def test_returns_figure_from_attribute(self) -> None:
        var_decomp = np.array(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.5, 0.2, 0.1, 0.2],
                [0.3, 0.4, 0.1, 0.2],
            ]
        )

        @dataclass
        class VDResults:
            variance_decomposition: NDArray[np.float64] = field(default_factory=lambda: var_decomp)

        results = VDResults()
        fig = plot_variance_decomposition(results)
        assert isinstance(fig, Figure)

    def test_computed_from_loadings(self) -> None:
        results = MockFactorResults()
        fig = plot_variance_decomposition(results)
        assert isinstance(fig, Figure)

    def test_no_data_raises(self) -> None:
        @dataclass
        class EmptyResults:
            pass

        with pytest.raises(ValueError, match="No variance decomposition"):
            plot_variance_decomposition(EmptyResults())

    def test_custom_names(self) -> None:
        results = MockFactorResults()
        fig = plot_variance_decomposition(
            results,
            series_names=["GDP", "CPI", "IP", "Rate", "M2"],
            factor_names=["Activity", "Inflation", "Finance", "Idio"],
        )
        assert isinstance(fig, Figure)
