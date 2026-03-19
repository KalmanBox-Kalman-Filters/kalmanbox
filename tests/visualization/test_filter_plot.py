"""Tests for filter internal diagnostics plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.filter_plot import (
    plot_covariance_convergence,
    plot_prediction_errors,
)
from kalmanbox.visualization.themes import reset_theme


@dataclass
class MockFilterResults:
    """Mock results for filter plots."""

    prediction_errors: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 5, 100)
    )
    prediction_error_var: NDArray[np.float64] = field(default_factory=lambda: np.full(100, 25.0))
    filtered_state_cov: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.filtered_state_cov is None:
            n = 100
            # Simulate convergence: start high, decrease
            diag_vals = 100.0 * np.exp(-0.05 * np.arange(n)) + 5.0
            self.filtered_state_cov = np.zeros((n, 2, 2))
            for t in range(n):
                self.filtered_state_cov[t] = np.diag([diag_vals[t], diag_vals[t] * 0.5])


@pytest.fixture(autouse=True)
def _reset():
    reset_theme()
    yield
    plt.close("all")


class TestPlotPredictionErrors:
    """Tests for plot_prediction_errors."""

    def test_returns_figure(self) -> None:
        results = MockFilterResults()
        fig = plot_prediction_errors(results)
        assert isinstance(fig, Figure)

    def test_with_bands(self) -> None:
        results = MockFilterResults()
        fig = plot_prediction_errors(results, ci=0.95)
        ax = fig.get_axes()[0]
        # Should have fill_between (collection)
        assert len(ax.collections) >= 1

    def test_without_var(self) -> None:
        """Test without prediction error variance (no bands)."""

        @dataclass
        class NoVarResults:
            prediction_errors: NDArray[np.float64] = field(
                default_factory=lambda: np.random.default_rng(42).normal(0, 1, 50)
            )

        fig = plot_prediction_errors(NoVarResults())
        assert isinstance(fig, Figure)

    def test_no_errors_raises(self) -> None:
        @dataclass
        class EmptyResults:
            pass

        with pytest.raises(ValueError, match="No prediction errors"):
            plot_prediction_errors(EmptyResults())

    def test_custom_title(self) -> None:
        results = MockFilterResults()
        fig = plot_prediction_errors(results, title="Innovations")
        ax = fig.get_axes()[0]
        assert ax.get_title() == "Innovations"


class TestPlotCovarianceConvergence:
    """Tests for plot_covariance_convergence."""

    def test_returns_figure(self) -> None:
        results = MockFilterResults()
        fig = plot_covariance_convergence(results)
        assert isinstance(fig, Figure)

    def test_plots_all_states(self) -> None:
        results = MockFilterResults()
        fig = plot_covariance_convergence(results)
        ax = fig.get_axes()[0]
        # Should have 2 lines (2 states)
        assert len(ax.get_lines()) == 2

    def test_select_states(self) -> None:
        results = MockFilterResults()
        fig = plot_covariance_convergence(results, states=[0])
        ax = fig.get_axes()[0]
        assert len(ax.get_lines()) == 1

    def test_log_scale(self) -> None:
        results = MockFilterResults()
        fig = plot_covariance_convergence(results, log_scale=True)
        ax = fig.get_axes()[0]
        assert ax.get_yscale() == "log"

    def test_custom_state_names(self) -> None:
        results = MockFilterResults()
        fig = plot_covariance_convergence(results, state_names=["Level", "Slope"])
        assert isinstance(fig, Figure)

    def test_no_cov_raises(self) -> None:
        @dataclass
        class EmptyResults:
            pass

        with pytest.raises(ValueError, match="No filtered state covariance"):
            plot_covariance_convergence(EmptyResults())

    def test_2d_cov(self) -> None:
        """Test with 2D diagonal-only covariance."""

        @dataclass
        class DiagCovResults:
            filtered_state_cov: NDArray[np.float64] = field(
                default_factory=lambda: np.random.default_rng(42).uniform(1, 10, (100, 2))
            )

        fig = plot_covariance_convergence(DiagCovResults())
        assert isinstance(fig, Figure)
