"""Tests for diagnostic plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.diagnostics_plot import plot_diagnostics
from kalmanbox.visualization.themes import reset_theme


@dataclass
class MockDiagResults:
    """Mock results for testing diagnostics."""

    standardized_residuals: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 1, 100)
    )


@pytest.fixture(autouse=True)
def _reset():
    reset_theme()
    yield
    plt.close("all")


class TestPlotDiagnostics:
    """Tests for plot_diagnostics."""

    def test_returns_figure(self) -> None:
        results = MockDiagResults()
        fig = plot_diagnostics(results)
        assert isinstance(fig, Figure)

    def test_has_4_panels(self) -> None:
        results = MockDiagResults()
        fig = plot_diagnostics(results)
        axes = fig.get_axes()
        assert len(axes) == 4

    def test_custom_lags(self) -> None:
        results = MockDiagResults()
        fig = plot_diagnostics(results, lags=10)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        results = MockDiagResults()
        fig = plot_diagnostics(results, title="My Diagnostics")
        assert fig._suptitle.get_text() == "My Diagnostics"

    def test_custom_theme(self) -> None:
        results = MockDiagResults()
        fig = plot_diagnostics(results, theme="presentation")
        assert isinstance(fig, Figure)

    def test_with_nan_residuals(self) -> None:
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, 100)
        resid[0] = np.nan
        resid[5] = np.nan
        results = MockDiagResults(standardized_residuals=resid)
        fig = plot_diagnostics(results)
        assert isinstance(fig, Figure)

    def test_no_residuals_raises(self) -> None:
        @dataclass
        class EmptyResults:
            pass

        with pytest.raises(ValueError, match="No residuals"):
            plot_diagnostics(EmptyResults())

    def test_resid_attribute_fallback(self) -> None:
        """Test that 'resid' attribute is also detected."""

        @dataclass
        class ResidResults:
            resid: NDArray[np.float64] = field(
                default_factory=lambda: np.random.default_rng(42).normal(0, 1, 50)
            )

        fig = plot_diagnostics(ResidResults())
        assert isinstance(fig, Figure)

    def test_custom_figsize(self) -> None:
        results = MockDiagResults()
        fig = plot_diagnostics(results, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1
        assert abs(h - 8) < 0.1
