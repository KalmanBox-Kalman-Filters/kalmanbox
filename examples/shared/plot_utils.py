"""Plotting utilities for kalmanbox examples."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_filtered_state(
    y: np.ndarray,
    filtered_state: np.ndarray,
    filtered_var: Optional[np.ndarray] = None,
    title: str = "Filtered State",
    ylabel: str = "Value",
    dates: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot observations with filtered state and optional 95% confidence bands."""
    fig, ax = plt.subplots(figsize=figsize)
    x = dates if dates is not None else np.arange(len(y))

    ax.plot(x, y, "o", markersize=3, alpha=0.5, label="Observations", color="C0")
    ax.plot(x, filtered_state, linewidth=1.5, label="Filtered state", color="C1")

    if filtered_var is not None:
        std = np.sqrt(filtered_var)
        ax.fill_between(
            x,
            filtered_state - 1.96 * std,
            filtered_state + 1.96 * std,
            alpha=0.2,
            color="C1",
            label="95% CI",
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_smoothed_state(
    y: np.ndarray,
    smoothed_state: np.ndarray,
    smoothed_var: Optional[np.ndarray] = None,
    title: str = "Smoothed State",
    ylabel: str = "Value",
    dates: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot observations with smoothed state and optional 95% confidence bands."""
    fig, ax = plt.subplots(figsize=figsize)
    x = dates if dates is not None else np.arange(len(y))

    ax.plot(x, y, "o", markersize=3, alpha=0.5, label="Observations", color="C0")
    ax.plot(x, smoothed_state, linewidth=1.5, label="Smoothed state", color="C2")

    if smoothed_var is not None:
        std = np.sqrt(smoothed_var)
        ax.fill_between(
            x,
            smoothed_state - 1.96 * std,
            smoothed_state + 1.96 * std,
            alpha=0.2,
            color="C2",
            label="95% CI",
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_components(
    components_dict: dict[str, np.ndarray],
    dates: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot decomposed components (level, trend, seasonal, irregular)."""
    n_components = len(components_dict)
    fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True)
    if n_components == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, components_dict.items()):
        x = dates if dates is not None else np.arange(len(values))
        ax.plot(x, values, linewidth=1.2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Component Decomposition")
    fig.tight_layout()
    return fig


def plot_forecast(
    y: np.ndarray,
    forecast_mean: np.ndarray,
    forecast_var: Optional[np.ndarray] = None,
    n_forecast: int = 10,
    title: str = "Forecast",
    dates: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot series with out-of-sample forecast and prediction intervals."""
    fig, ax = plt.subplots(figsize=figsize)
    n_obs = len(y)

    if dates is not None:
        x_obs = dates[:n_obs]
        x_fcast = dates[n_obs : n_obs + n_forecast]
    else:
        x_obs = np.arange(n_obs)
        x_fcast = np.arange(n_obs, n_obs + n_forecast)

    ax.plot(x_obs, y, linewidth=1.2, label="Observed", color="C0")
    ax.plot(x_fcast, forecast_mean[:n_forecast], linewidth=1.5, label="Forecast", color="C3")

    if forecast_var is not None:
        std = np.sqrt(forecast_var[:n_forecast])
        ax.fill_between(
            x_fcast,
            forecast_mean[:n_forecast] - 1.96 * std,
            forecast_mean[:n_forecast] + 1.96 * std,
            alpha=0.2,
            color="C3",
            label="95% PI",
        )

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(
    residuals: np.ndarray,
    title: str = "Residual Diagnostics",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """4-panel residual diagnostic plot (time series, histogram, ACF, QQ)."""
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    # Panel 1: Time series of residuals
    ax = axes[0, 0]
    ax.plot(residuals, linewidth=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Standardized Residuals")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)

    # Panel 2: Histogram with normal overlay
    ax = axes[0, 1]
    ax.hist(residuals, bins=20, density=True, alpha=0.7, edgecolor="black")
    x_grid = np.linspace(residuals.min() - 0.5, residuals.max() + 0.5, 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid, loc=np.mean(residuals), scale=np.std(residuals)),
            "r-", linewidth=1.5)
    ax.set_title("Histogram")
    ax.grid(True, alpha=0.3)

    # Panel 3: ACF
    ax = axes[1, 0]
    n = len(residuals)
    max_lag = min(20, n - 1)
    r = residuals - np.mean(residuals)
    acf_vals = np.correlate(r, r, mode="full")[n - 1 :] / np.dot(r, r)
    acf_vals = acf_vals[: max_lag + 1]
    ax.bar(range(max_lag + 1), acf_vals, width=0.3, color="C0")
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("ACF")
    ax.set_xlabel("Lag")
    ax.grid(True, alpha=0.3)

    # Panel 4: QQ plot
    ax = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
