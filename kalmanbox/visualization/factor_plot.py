"""Factor model visualization.

Plots for DynamicFactorModel results:
- plot_factors: Time series of extracted factors
- plot_loadings: Heatmap of factor loadings matrix
- plot_variance_decomposition: Stacked bar chart of variance explained
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.themes import get_theme


def plot_factors(
    results: Any,
    factors: Sequence[int] | None = None,
    theme: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    factor_names: Sequence[str] | None = None,
) -> Figure:
    """Plot extracted factor time series.

    Parameters
    ----------
    results : DynamicFactorResults or similar
        Fitted model results. Expected attributes:
        - `factors` or `smoothed_factors`: (T, n_factors) array
    factors : list of int or None
        Which factor indices to plot. Default None plots all.
    theme : str or None
        Theme name. If None, uses current active theme.
    figsize : tuple or None
        Figure size. Default auto-calculated.
    title : str or None
        Figure title. Default 'Extracted Factors'.
    factor_names : list of str or None
        Custom names for factors. Default 'Factor 1', 'Factor 2', etc.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    theme_config = get_theme(theme)
    colors = theme_config.colors

    factor_data = _get_factors(results)
    if factor_data is None:
        msg = "No factor data found in results."
        raise ValueError(msg)

    if factor_data.ndim == 1:
        factor_data = factor_data.reshape(-1, 1)

    n_time, n_factors = factor_data.shape

    if factors is None:
        factors = list(range(n_factors))

    n_panels = len(factors)
    if figsize is None:
        figsize = (12, 2.5 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    t = np.arange(n_time)
    time_index = getattr(results, "time_index", None)
    if time_index is not None:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            t = np.asarray(time_index)

    series_colors = colors.series

    for panel_idx, factor_idx in enumerate(factors):
        ax = axes[panel_idx]
        color = series_colors[panel_idx % len(series_colors)]

        ax.plot(t, factor_data[:, factor_idx], color=color, linewidth=theme_config.line_width)
        ax.axhline(y=0, color=colors.text, linewidth=0.5, alpha=0.3)

        if factor_names and panel_idx < len(factor_names):
            label = factor_names[panel_idx]
        else:
            label = f"Factor {factor_idx + 1}"
        ax.set_ylabel(label, fontsize=theme_config.fonts.label_size)

    axes[-1].set_xlabel("Time", fontsize=theme_config.fonts.label_size)

    fig_title = title or "Extracted Factors"
    fig.suptitle(fig_title, fontsize=theme_config.fonts.title_size + 2, y=1.0)
    fig.tight_layout()

    return fig


def plot_loadings(
    results: Any,
    theme: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    series_names: Sequence[str] | None = None,
    factor_names: Sequence[str] | None = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
) -> Figure:
    """Plot factor loadings as a heatmap.

    Parameters
    ----------
    results : DynamicFactorResults or similar
        Fitted model results. Expected attributes:
        - `loadings` or `factor_loadings`: (n_series, n_factors) matrix
    theme : str or None
        Theme name. If None, uses current active theme.
    figsize : tuple
        Figure size. Default (10, 8).
    title : str or None
        Figure title. Default 'Factor Loadings'.
    series_names : list of str or None
        Names for y-axis (series). Default 'Series 1', etc.
    factor_names : list of str or None
        Names for x-axis (factors). Default 'Factor 1', etc.
    cmap : str
        Matplotlib colormap name. Default 'RdBu_r'.
    annotate : bool
        Whether to show values in cells. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The heatmap figure.
    """
    theme_config = get_theme(theme)

    loadings = _get_loadings(results)
    if loadings is None:
        msg = "No loadings matrix found in results."
        raise ValueError(msg)

    n_series, n_factors = loadings.shape

    if series_names is None:
        series_names = [f"Series {i + 1}" for i in range(n_series)]
    if factor_names is None:
        factor_names = [f"Factor {i + 1}" for i in range(n_factors)]

    fig, ax = plt.subplots(figsize=figsize)

    # Symmetric color range centered at zero
    vmax = np.abs(loadings).max()
    vmin = -vmax

    im = ax.imshow(loadings, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Ticks and labels
    ax.set_xticks(np.arange(n_factors))
    ax.set_xticklabels(factor_names[:n_factors], fontsize=theme_config.fonts.tick_size)
    ax.set_yticks(np.arange(n_series))
    ax.set_yticklabels(series_names[:n_series], fontsize=theme_config.fonts.tick_size)

    # Annotate cells
    if annotate:
        for i in range(n_series):
            for j in range(n_factors):
                val = loadings[i, j]
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=theme_config.fonts.annotation_size,
                )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Loading")

    fig_title = title or "Factor Loadings"
    ax.set_title(fig_title, fontsize=theme_config.fonts.title_size)
    fig.tight_layout()

    return fig


def plot_variance_decomposition(
    results: Any,
    theme: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    series_names: Sequence[str] | None = None,
    factor_names: Sequence[str] | None = None,
) -> Figure:
    """Plot variance decomposition as stacked bar chart.

    Shows the proportion of variance in each observed series explained
    by each factor plus the idiosyncratic component.

    Parameters
    ----------
    results : DynamicFactorResults or similar
        Expected attributes:
        - `variance_decomposition`: (n_series, n_factors+1) array
          where last column is idiosyncratic
        - OR `loadings` + `idiosyncratic_var` to compute it
    theme : str or None
        Theme name.
    figsize : tuple
        Figure size. Default (12, 6).
    title : str or None
        Figure title. Default 'Variance Decomposition'.
    series_names : list of str or None
        Names for x-axis (series).
    factor_names : list of str or None
        Names for legend (factors + idiosyncratic).

    Returns
    -------
    matplotlib.figure.Figure
        The stacked bar chart.
    """
    theme_config = get_theme(theme)
    colors = theme_config.colors

    var_decomp = _get_variance_decomposition(results)
    if var_decomp is None:
        msg = "No variance decomposition data found in results."
        raise ValueError(msg)

    n_series, n_components = var_decomp.shape

    if series_names is None:
        series_names = [f"Series {i + 1}" for i in range(n_series)]
    if factor_names is None:
        n_factors = n_components - 1
        factor_names = [f"Factor {i + 1}" for i in range(n_factors)] + ["Idiosyncratic"]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_series)
    bottom = np.zeros(n_series)

    series_colors = [*colors.series, "#cccccc"]  # gray for idiosyncratic

    for j in range(n_components):
        color = series_colors[j % len(series_colors)]
        label = factor_names[j] if j < len(factor_names) else f"Component {j + 1}"
        ax.bar(
            x,
            var_decomp[:, j],
            bottom=bottom,
            color=color,
            label=label,
            width=0.6,
            alpha=0.85,
        )
        bottom += var_decomp[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(
        series_names[:n_series],
        fontsize=theme_config.fonts.tick_size,
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Proportion of Variance", fontsize=theme_config.fonts.label_size)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=theme_config.fonts.legend_size)

    fig_title = title or "Variance Decomposition"
    ax.set_title(fig_title, fontsize=theme_config.fonts.title_size)
    fig.tight_layout()

    return fig


def _get_factors(results: Any) -> NDArray[np.float64] | None:
    """Extract factor data from results."""
    for attr in ["factors", "smoothed_factors", "filtered_factors", "extracted_factors"]:
        val = getattr(results, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64)
    return None


def _get_loadings(results: Any) -> NDArray[np.float64] | None:
    """Extract loadings matrix from results."""
    for attr in ["loadings", "factor_loadings", "loading_matrix"]:
        val = getattr(results, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64)
    return None


def _get_variance_decomposition(results: Any) -> NDArray[np.float64] | None:
    """Extract or compute variance decomposition."""
    val = getattr(results, "variance_decomposition", None)
    if val is not None:
        return np.asarray(val, dtype=np.float64)

    # Try to compute from loadings
    loadings = _get_loadings(results)
    idio_var = getattr(results, "idiosyncratic_var", None)
    if idio_var is None:
        idio_var = getattr(results, "idiosyncratic_variance", None)

    if loadings is not None and idio_var is not None:
        idio_var = np.asarray(idio_var, dtype=np.float64).ravel()
        # Variance from each factor (assuming unit factor variance)
        factor_var = loadings**2  # (n_series, n_factors)
        total_var = factor_var.sum(axis=1) + idio_var
        # Proportions
        return np.column_stack(
            [
                factor_var / total_var[:, np.newaxis],
                idio_var / total_var,
            ]
        )

    return None
