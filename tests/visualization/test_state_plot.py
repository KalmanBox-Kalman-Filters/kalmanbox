"""Tests for state comparison plots."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.state_plot import plot_states
from kalmanbox.visualization.themes import reset_theme


@dataclass
class MockStateResults:
    """Mock results for testing plot_states."""

    observed: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(100, 10, 100)
    )
    filtered_state: NDArray[np.float64] | None = None
    smoothed_state: NDArray[np.float64] | None = None
    filtered_state_cov: NDArray[np.float64] | None = None
    smoothed_state_cov: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(42)
        n = len(self.observed)
        if self.filtered_state is None:
            self.filtered_state = self.observed.reshape(-1, 1) + rng.normal(0, 3, (n, 1))
        if self.smoothed_state is None:
            self.smoothed_state = self.observed.reshape(-1, 1) + rng.normal(0, 1, (n, 1))
        if self.filtered_state_cov is None:
            self.filtered_state_cov = np.full((n, 1, 1), 9.0)
        if self.smoothed_state_cov is None:
            self.smoothed_state_cov = np.full((n, 1, 1), 4.0)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_theme()
    yield  # noqa: PT022
    plt.close("all")


class TestPlotStates:
    """Tests for plot_states."""

    def test_returns_figure(self) -> None:
        results = MockStateResults()
        fig = plot_states(results)
        assert isinstance(fig, Figure)

    def test_shows_filtered_and_smoothed(self) -> None:
        results = MockStateResults()
        fig = plot_states(results)
        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        # At least: observed, filtered, smoothed
        assert len(lines) >= 2

    def test_only_filtered(self) -> None:
        results = MockStateResults()
        fig = plot_states(results, show_smoothed=False)
        assert isinstance(fig, Figure)

    def test_only_smoothed(self) -> None:
        results = MockStateResults()
        fig = plot_states(results, show_filtered=False)
        assert isinstance(fig, Figure)

    def test_multistate(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        results = MockStateResults(
            filtered_state=rng.normal(0, 1, (n, 3)),
            smoothed_state=rng.normal(0, 1, (n, 3)),
            filtered_state_cov=np.stack([np.eye(3) * 2] * n),
            smoothed_state_cov=np.stack([np.eye(3)] * n),
        )
        fig = plot_states(results)
        assert len(fig.get_axes()) == 3

    def test_select_states(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        results = MockStateResults(
            filtered_state=rng.normal(0, 1, (n, 3)),
            smoothed_state=rng.normal(0, 1, (n, 3)),
            filtered_state_cov=np.stack([np.eye(3) * 2] * n),
            smoothed_state_cov=np.stack([np.eye(3)] * n),
        )
        fig = plot_states(results, states=[0, 2])
        assert len(fig.get_axes()) == 2

    def test_custom_state_names(self) -> None:
        results = MockStateResults()
        fig = plot_states(results, state_names=["Level"])
        assert isinstance(fig, Figure)

    def test_no_observed(self) -> None:
        results = MockStateResults()
        fig = plot_states(results, show_observed=False)
        assert isinstance(fig, Figure)

    def test_no_states_raises(self) -> None:
        @dataclass
        class EmptyResults:
            observed: NDArray[np.float64] = field(default_factory=lambda: np.zeros(10))

        with pytest.raises(ValueError, match="No state estimates"):
            plot_states(EmptyResults())

    def test_custom_ci(self) -> None:
        results = MockStateResults()
        fig = plot_states(results, ci=0.99)
        assert isinstance(fig, Figure)

    def test_custom_theme(self) -> None:
        results = MockStateResults()
        fig = plot_states(results, theme="presentation")
        assert isinstance(fig, Figure)
