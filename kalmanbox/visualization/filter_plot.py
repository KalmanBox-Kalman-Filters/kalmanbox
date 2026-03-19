"""Kalman filter internal diagnostics plots.

Plots for monitoring filter behavior:
- plot_prediction_errors: Prediction (innovation) errors with uncertainty bands
- plot_covariance_convergence: Convergence of state covariance matrix P
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.themes import get_theme


def plot_prediction_errors(
    results: Any,
    ci: float = 0.95,
    theme: str | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> Figure:
    """Plot prediction (innovation) errors with confidence bands.

    Parameters
    ----------
    results : FilterResults or similar
        Expected attributes:
        - `prediction_errors` or `innovations` or `forecast_errors`: (T,) array
        - `prediction_error_cov` or `innovation_var` (optional): for bands
    ci : float
        Confidence level for bands. Default 0.95.
    theme : str or None
        Theme name.
    figsize : tuple
        Figure size. Default (12, 5).
    title : str or None
        Figure title. Default 'Prediction Errors'.

    Returns
    -------
    matplotlib.figure.Figure
        The prediction errors plot.
    """
    theme_config = get_theme(theme)
    colors = theme_config.colors

    errors = _get_prediction_errors(results)
    if errors is None:
        msg = "No prediction errors found in results."
        raise ValueError(msg)

    error_var = _get_prediction_error_var(results)

    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(len(errors))

    # Plot errors
    ax.plot(
        t,
        errors,
        color=colors.primary,
        linewidth=theme_config.line_width * 0.7,
        alpha=0.8,
        label="Prediction Error",
    )
    ax.axhline(y=0, color=colors.text, linewidth=0.5, alpha=0.5)

    # CI bands
    if error_var is not None:
        from scipy import stats

        z = float(stats.norm.ppf((1 + ci) / 2))
        error_std = np.sqrt(np.maximum(error_var, 0.0))
        upper = z * error_std
        lower = -z * error_std
        ax.fill_between(
            t, lower, upper, color=colors.tertiary, alpha=0.3, label=f"{ci * 100:.0f}% CI"
        )
        ax.legend(loc="upper right", fontsize=theme_config.fonts.legend_size)

    ax.set_xlabel("Time", fontsize=theme_config.fonts.label_size)
    ax.set_ylabel("Error", fontsize=theme_config.fonts.label_size)

    fig_title = title or "Prediction Errors"
    ax.set_title(fig_title, fontsize=theme_config.fonts.title_size)
    fig.tight_layout()

    return fig


def plot_covariance_convergence(
    results: Any,
    states: Sequence[int] | None = None,
    theme: str | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
    state_names: Sequence[str] | None = None,
    log_scale: bool = False,
) -> Figure:
    """Plot convergence of diagonal elements of state covariance P.

    Shows how the uncertainty in each state estimate evolves over time,
    typically decreasing and converging as the filter processes more data.

    Parameters
    ----------
    results : FilterResults or similar
        Expected attributes:
        - `filtered_state_cov`: (T, n_states, n_states) covariance array
    states : list of int or None
        Which state diagonal elements to plot. Default all.
    theme : str or None
        Theme name.
    figsize : tuple
        Figure size. Default (12, 5).
    title : str or None
        Figure title. Default 'Covariance Convergence'.
    state_names : list of str or None
        Custom names for states.
    log_scale : bool
        Whether to use log scale for y-axis. Default False.

    Returns
    -------
    matplotlib.figure.Figure
        The convergence plot.
    """
    theme_config = get_theme(theme)
    colors = theme_config.colors

    cov = _get_filtered_cov(results)
    if cov is None:
        msg = "No filtered state covariance found in results."
        raise ValueError(msg)

    if cov.ndim == 3:
        n_time, n_states, _ = cov.shape
    elif cov.ndim == 2:
        n_time, n_states = cov.shape
    else:
        msg = f"Unexpected covariance shape: {cov.shape}"
        raise ValueError(msg)

    if states is None:
        states = list(range(n_states))

    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(n_time)

    series_colors = colors.series

    for idx, state_idx in enumerate(states):
        color = series_colors[idx % len(series_colors)]

        diag_vals = cov[:, state_idx, state_idx] if cov.ndim == 3 else cov[:, state_idx]

        if state_names and idx < len(state_names):
            label = state_names[idx]
        else:
            label = f"P[{state_idx},{state_idx}]"

        ax.plot(t, diag_vals, color=color, linewidth=theme_config.line_width, label=label)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Time", fontsize=theme_config.fonts.label_size)
    ax.set_ylabel("Variance (P diagonal)", fontsize=theme_config.fonts.label_size)
    ax.legend(loc="upper right", fontsize=theme_config.fonts.legend_size)

    fig_title = title or "Covariance Convergence"
    ax.set_title(fig_title, fontsize=theme_config.fonts.title_size)
    fig.tight_layout()

    return fig


def _get_prediction_errors(results: Any) -> NDArray[np.float64] | None:
    """Extract prediction errors from results."""
    for attr in [
        "prediction_errors",
        "innovations",
        "forecast_errors",
        "standardized_forecasts_error",
    ]:
        val = getattr(results, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64).ravel()
    return None


def _get_prediction_error_var(results: Any) -> NDArray[np.float64] | None:
    """Extract prediction error variance from results."""
    for attr in [
        "prediction_error_cov",
        "prediction_error_var",
        "innovation_var",
        "forecast_error_cov",
    ]:
        val = getattr(results, attr, None)
        if val is not None:
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 3:
                return arr[:, 0, 0]
            elif arr.ndim == 2:
                return arr[:, 0]
            return arr.ravel()
    return None


def _get_filtered_cov(results: Any) -> NDArray[np.float64] | None:
    """Extract filtered state covariance from results."""
    for attr in ["filtered_state_cov", "predicted_state_cov", "P_filtered", "P"]:
        val = getattr(results, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64)
    return None
