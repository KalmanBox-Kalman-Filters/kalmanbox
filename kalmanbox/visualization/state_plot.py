"""State comparison plots for Kalman filter results.

Plots filtered vs smoothed state estimates with confidence interval bands.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.themes import get_theme


def plot_states(
    results: Any,
    states: Sequence[int] | None = None,
    show_filtered: bool = True,
    show_smoothed: bool = True,
    show_observed: bool = True,
    ci: float = 0.95,
    theme: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    state_names: Sequence[str] | None = None,
) -> Figure:
    """Plot filtered and/or smoothed state estimates with CI bands.

    Parameters
    ----------
    results : StateSpaceResults or similar
        Fitted model results. Expected attributes:
        - `filtered_state`: (T, n_states) array of filtered states
        - `smoothed_state`: (T, n_states) array of smoothed states
        - `filtered_state_cov`: (T, n_states, n_states) filtered covariance
        - `smoothed_state_cov`: (T, n_states, n_states) smoothed covariance
        - `observed` or `endog`: (T,) observed data
    states : list of int or None
        Which state indices to plot. Default None plots all.
    show_filtered : bool
        Whether to show filtered states. Default True.
    show_smoothed : bool
        Whether to show smoothed states. Default True.
    show_observed : bool
        Whether to show observed data. Default True.
    ci : float
        Confidence interval level. Default 0.95.
    theme : str or None
        Theme name. If None, uses current active theme.
    figsize : tuple or None
        Figure size (width, height). Default auto-calculated.
    title : str or None
        Figure title. Default 'State Estimates'.
    state_names : list of str or None
        Custom names for each state. Default 'State 0', 'State 1', etc.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    theme_config = get_theme(theme)
    colors = theme_config.colors

    # Extract data
    filtered = _get_state_array(results, "filtered")
    smoothed = _get_state_array(results, "smoothed")
    filtered_cov = _get_cov_array(results, "filtered")
    smoothed_cov = _get_cov_array(results, "smoothed")
    observed = _get_observed(results)

    if filtered is None and smoothed is None:
        msg = "No state estimates found in results."
        raise ValueError(msg)

    # Determine state array shape
    ref_state = filtered if filtered is not None else smoothed
    assert ref_state is not None

    if ref_state.ndim == 1:
        ref_state = ref_state.reshape(-1, 1)
        if filtered is not None:
            filtered = filtered.reshape(-1, 1)
        if smoothed is not None:
            smoothed = smoothed.reshape(-1, 1)

    n_time, n_states = ref_state.shape
    t = np.arange(n_time)

    # Get time index
    time_index = getattr(results, "time_index", None)
    if time_index is not None:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            t = np.asarray(time_index)

    # Select states to plot
    if states is None:
        states = list(range(n_states))

    n_panels = len(states)
    if figsize is None:
        figsize = (12, 3.0 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    from scipy import stats

    z_score = float(stats.norm.ppf((1 + ci) / 2))

    for panel_idx, state_idx in enumerate(states):
        ax = axes[panel_idx]

        # Observed data (only on first panel if univariate)
        if show_observed and observed is not None and panel_idx == 0:
            ax.plot(
                t,
                observed,
                color=colors.text,
                linewidth=0.5,
                alpha=0.4,
                label="Observed",
                zorder=1,
            )

        # Filtered state
        if show_filtered and filtered is not None:
            filt = filtered[:, state_idx]
            ax.plot(
                t,
                filt,
                color=colors.secondary,
                linewidth=theme_config.line_width,
                label="Filtered",
                linestyle="--",
                zorder=2,
            )
            if filtered_cov is not None:
                filt_std = _extract_std(filtered_cov, state_idx)
                if filt_std is not None:
                    ax.fill_between(
                        t,
                        filt - z_score * filt_std,
                        filt + z_score * filt_std,
                        color=colors.secondary,
                        alpha=0.15,
                        label=f"Filtered {ci * 100:.0f}% CI",
                    )

        # Smoothed state
        if show_smoothed and smoothed is not None:
            sm = smoothed[:, state_idx]
            ax.plot(
                t,
                sm,
                color=colors.primary,
                linewidth=theme_config.line_width,
                label="Smoothed",
                zorder=3,
            )
            if smoothed_cov is not None:
                sm_std = _extract_std(smoothed_cov, state_idx)
                if sm_std is not None:
                    ax.fill_between(
                        t,
                        sm - z_score * sm_std,
                        sm + z_score * sm_std,
                        color=colors.primary,
                        alpha=0.15,
                        label=f"Smoothed {ci * 100:.0f}% CI",
                    )

        # Labels
        if state_names and panel_idx < len(state_names):
            label = state_names[panel_idx]
        else:
            label = f"State {state_idx}"
        ax.set_ylabel(label, fontsize=theme_config.fonts.label_size)
        ax.legend(
            loc="upper right",
            fontsize=theme_config.fonts.legend_size,
            framealpha=0.8,
        )

    axes[-1].set_xlabel("Time", fontsize=theme_config.fonts.label_size)

    fig_title = title or "State Estimates"
    fig.suptitle(fig_title, fontsize=theme_config.fonts.title_size + 2, y=1.0)
    fig.tight_layout()

    return fig


def _get_state_array(results: Any, kind: str) -> NDArray[np.float64] | None:
    """Get filtered or smoothed state array."""
    attr_name = f"{kind}_state"
    val = getattr(results, attr_name, None)
    if val is not None:
        return np.asarray(val, dtype=np.float64)
    return None


def _get_cov_array(results: Any, kind: str) -> NDArray[np.float64] | None:
    """Get filtered or smoothed state covariance array."""
    attr_name = f"{kind}_state_cov"
    val = getattr(results, attr_name, None)
    if val is not None:
        return np.asarray(val, dtype=np.float64)
    return None


def _get_observed(results: Any) -> NDArray[np.float64] | None:
    """Get observed data from results."""
    for name in ["observed", "endog", "y"]:
        val = getattr(results, name, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64)
    return None


def _extract_std(
    cov: NDArray[np.float64],
    state_idx: int,
) -> NDArray[np.float64] | None:
    """Extract standard deviation for a specific state from covariance array."""
    if cov.ndim == 3:
        # (T, n_states, n_states)
        variances = cov[:, state_idx, state_idx]
        return np.sqrt(np.maximum(variances, 0.0))
    elif cov.ndim == 2:
        # (T, n_states) - diagonal only
        if state_idx < cov.shape[1]:
            return np.sqrt(np.maximum(cov[:, state_idx], 0.0))
    return None
