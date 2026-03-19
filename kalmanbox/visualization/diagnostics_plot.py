"""Diagnostic plots for state-space model residuals.

Generates a 2x2 panel with:
- Top-left: Standardized residuals vs time
- Top-right: ACF of residuals
- Bottom-left: QQ-plot against Normal distribution
- Bottom-right: Histogram with fitted Normal curve
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats as sp_stats

from kalmanbox.visualization.themes import get_theme


def plot_diagnostics(
    results: Any,
    lags: int = 20,
    theme: str | None = None,
    figsize: tuple[float, float] = (12, 9),
    title: str | None = None,
) -> Figure:
    """Plot residual diagnostics in a 2x2 panel.

    Parameters
    ----------
    results : StateSpaceResults or similar
        Fitted model results. Expected attributes:
        - `standardized_residuals` or `resid` or `residuals`: residual array
    lags : int
        Number of lags for ACF plot. Default 20.
    theme : str or None
        Theme name. If None, uses current active theme.
    figsize : tuple
        Figure size (width, height). Default (12, 9).
    title : str or None
        Figure title. Default 'Residual Diagnostics'.

    Returns
    -------
    matplotlib.figure.Figure
        The 2x2 diagnostic figure.
    """
    theme_config = get_theme(theme)
    colors = theme_config.colors

    residuals = _get_residuals(results)
    if residuals is None:
        msg = "No residuals found in results. Fit the model first."
        raise ValueError(msg)

    # Remove NaN values
    residuals_clean = residuals[~np.isnan(residuals)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- Top-left: Standardized residuals vs time ---
    ax = axes[0, 0]
    t = np.arange(len(residuals))
    ax.scatter(t, residuals, color=colors.primary, s=8, alpha=0.6, zorder=2)
    ax.axhline(y=0, color=colors.text, linewidth=0.8, linestyle="-", alpha=0.5)
    ax.axhline(y=2, color=colors.accent, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=-2, color=colors.accent, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("Standardized Residuals", fontsize=theme_config.fonts.label_size)
    ax.set_xlabel("Time", fontsize=theme_config.fonts.tick_size)
    ax.set_ylabel("Residual", fontsize=theme_config.fonts.tick_size)

    # --- Top-right: ACF ---
    ax = axes[0, 1]
    _plot_acf(
        ax,
        residuals_clean,
        lags=lags,
        color=colors.primary,
        ci_color=colors.tertiary,
        theme_config=theme_config,
    )
    ax.set_title("Autocorrelation (ACF)", fontsize=theme_config.fonts.label_size)

    # --- Bottom-left: QQ-plot ---
    ax = axes[1, 0]
    _plot_qq(
        ax,
        residuals_clean,
        color=colors.primary,
        line_color=colors.accent,
        theme_config=theme_config,
    )
    ax.set_title("Normal Q-Q Plot", fontsize=theme_config.fonts.label_size)

    # --- Bottom-right: Histogram ---
    ax = axes[1, 1]
    _plot_histogram(
        ax,
        residuals_clean,
        color=colors.primary,
        curve_color=colors.accent,
        theme_config=theme_config,
    )
    ax.set_title("Histogram of Residuals", fontsize=theme_config.fonts.label_size)

    fig_title = title or "Residual Diagnostics"
    fig.suptitle(fig_title, fontsize=theme_config.fonts.title_size + 2, y=1.02)
    fig.tight_layout()

    return fig


def _get_residuals(results: Any) -> NDArray[np.float64] | None:
    """Extract residuals from results object."""
    for attr in ["standardized_residuals", "std_residuals", "resid", "residuals"]:
        val = getattr(results, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64).ravel()
    return None


def _plot_acf(
    ax: plt.Axes,
    data: NDArray[np.float64],
    lags: int,
    color: str,
    ci_color: str,
    theme_config: Any,
) -> None:
    """Plot autocorrelation function."""
    n = len(data)
    max_lags = min(lags, n - 1)

    # Compute ACF manually
    mean = np.mean(data)
    centered = data - mean
    var = np.sum(centered**2) / n

    acf_values = np.zeros(max_lags + 1)
    acf_values[0] = 1.0
    for k in range(1, max_lags + 1):
        acf_values[k] = np.sum(centered[: n - k] * centered[k:]) / (n * var)

    lag_indices = np.arange(max_lags + 1)

    # Plot bars
    ax.bar(lag_indices, acf_values, width=0.3, color=color, alpha=0.7)

    # Significance bands (approximate 95%)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(y=ci, color=ci_color, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(y=-ci, color=ci_color, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xlabel("Lag", fontsize=theme_config.fonts.tick_size)
    ax.set_ylabel("ACF", fontsize=theme_config.fonts.tick_size)
    ax.set_xlim(-0.5, max_lags + 0.5)


def _plot_qq(
    ax: plt.Axes,
    data: NDArray[np.float64],
    color: str,
    line_color: str,
    theme_config: Any,
) -> None:
    """Plot Q-Q plot against normal distribution."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    theoretical = sp_stats.norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))

    ax.scatter(theoretical, sorted_data, color=color, s=12, alpha=0.6, zorder=2)

    # 45-degree reference line
    min_val = min(theoretical.min(), sorted_data.min())
    max_val = max(theoretical.max(), sorted_data.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=line_color,
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )

    ax.set_xlabel("Theoretical Quantiles", fontsize=theme_config.fonts.tick_size)
    ax.set_ylabel("Sample Quantiles", fontsize=theme_config.fonts.tick_size)


def _plot_histogram(
    ax: plt.Axes,
    data: NDArray[np.float64],
    color: str,
    curve_color: str,
    theme_config: Any,
) -> None:
    """Plot histogram with fitted normal curve."""
    n_bins = min(max(int(np.sqrt(len(data))), 10), 50)

    ax.hist(
        data,
        bins=n_bins,
        density=True,
        color=color,
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
    )

    # Fitted normal curve
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(data.min() - 0.5 * sigma, data.max() + 0.5 * sigma, 200)
    pdf = sp_stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, color=curve_color, linewidth=1.5, label=f"N({mu:.2f}, {sigma:.2f})")

    ax.set_xlabel("Residual", fontsize=theme_config.fonts.tick_size)
    ax.set_ylabel("Density", fontsize=theme_config.fonts.tick_size)
    ax.legend(fontsize=theme_config.fonts.legend_size)
