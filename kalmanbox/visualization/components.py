"""Component decomposition plots for state-space models.

Generates multi-panel vertical layout showing observed data, trend,
slope, seasonal, cycle, and irregular components.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.themes import get_theme

# Default ordered list of components to look for
_DEFAULT_COMPONENTS = ["observed", "trend", "slope", "seasonal", "cycle", "irregular"]


def plot_components(
    results: Any,
    which: Sequence[str] | None = None,
    ci: float = 0.95,
    theme: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot component decomposition from model results.

    Creates a multi-panel vertical plot showing the decomposition of the
    observed series into its structural components (trend, seasonal, etc.).

    Parameters
    ----------
    results : StateSpaceResults or similar
        Fitted model results. Expected attributes:
        - `observed` or `endog`: array of observed values
        - `filtered_state` or `smoothed_state`: state arrays
        - `smoothed_state_cov` (optional): covariance for CI bands
        - Component-specific attributes: `trend`, `slope`, `seasonal`,
          `cycle`, `irregular` (arrays or None)
    which : list of str or None
        Which components to plot. Default None plots all available.
        Options: 'observed', 'trend', 'slope', 'seasonal', 'cycle', 'irregular'.
    ci : float
        Confidence interval level for bands. Default 0.95.
    theme : str or None
        Theme name. If None, uses current active theme.
    figsize : tuple or None
        Figure size (width, height). Default auto-calculated based on panels.
    title : str or None
        Figure title. Default 'Component Decomposition'.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    theme_config = get_theme(theme)
    components = _extract_components(results, which)

    if not components:
        msg = "No components found in results. Check model type and available attributes."
        raise ValueError(msg)

    n_panels = len(components)
    if figsize is None:
        figsize = (12, 2.5 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    z_score = _ci_to_z(ci)
    colors = theme_config.colors

    for idx, (name, data) in enumerate(components.items()):
        ax = axes[idx]
        values = data["values"]
        t = data.get("time", np.arange(len(values)))
        std = data.get("std", None)

        if name == "observed":
            # Observed: plot raw data + fitted if available
            ax.plot(
                t,
                values,
                color=colors.primary,
                linewidth=theme_config.line_width,
                label="Observed",
                alpha=0.7,
            )
            if "fitted" in data and data["fitted"] is not None:
                ax.plot(
                    t,
                    data["fitted"],
                    color=colors.accent,
                    linewidth=theme_config.line_width * 0.8,
                    label="Fitted",
                    linestyle="--",
                )
                ax.legend(loc="upper right", framealpha=0.8)
        elif name == "irregular":
            # Irregular: scatter-like with zero line
            ax.plot(
                t,
                values,
                color=colors.secondary,
                linewidth=theme_config.line_width * 0.7,
                alpha=0.8,
            )
            ax.axhline(y=0, color=colors.text, linewidth=0.5, linestyle="-", alpha=0.5)
        else:
            # Standard component: line + CI band
            ax.plot(t, values, color=colors.primary, linewidth=theme_config.line_width)
            if std is not None:
                upper = values + z_score * std
                lower = values - z_score * std
                ax.fill_between(
                    t,
                    lower,
                    upper,
                    color=colors.tertiary,
                    alpha=0.3,
                    label=f"{ci * 100:.0f}% CI",
                )
                ax.legend(loc="upper right", framealpha=0.8)

        ax.set_ylabel(name.capitalize(), fontsize=theme_config.fonts.label_size)
        ax.tick_params(labelsize=theme_config.fonts.tick_size)

    axes[-1].set_xlabel("Time", fontsize=theme_config.fonts.label_size)

    fig_title = title or "Component Decomposition"
    fig.suptitle(fig_title, fontsize=theme_config.fonts.title_size + 2, y=1.0)
    fig.tight_layout()

    return fig


def _extract_components(
    results: Any,
    which: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract component data from results object."""
    components: dict[str, dict[str, Any]] = {}
    available = which or _DEFAULT_COMPONENTS

    time_index = _get_time_index(results)

    for comp_name in available:
        data = _get_component_data(results, comp_name, time_index)
        if data is not None:
            components[comp_name] = data

    return components


def _get_component_data(
    results: Any,
    name: str,
    time_index: NDArray[np.float64] | None,
) -> dict[str, Any] | None:
    """Try to extract a single component from results."""
    data: dict[str, Any] = {}

    if name == "observed":
        values = _get_attr(results, ["observed", "endog", "y"])
        if values is None:
            return None
        data["values"] = np.asarray(values, dtype=np.float64)
        fitted = _get_attr(results, ["fitted_values", "fittedvalues", "fitted"])
        if fitted is not None:
            data["fitted"] = np.asarray(fitted, dtype=np.float64)
    elif name == "trend":
        values = _get_attr(results, ["trend", "level", "smoothed_level"])
        if values is None:
            return None
        data["values"] = np.asarray(values, dtype=np.float64)
        std = _get_component_std(results, name)
        if std is not None:
            data["std"] = std
    elif name == "slope":
        values = _get_attr(results, ["slope", "smoothed_slope"])
        if values is None:
            return None
        data["values"] = np.asarray(values, dtype=np.float64)
        std = _get_component_std(results, name)
        if std is not None:
            data["std"] = std
    elif name == "seasonal":
        values = _get_attr(results, ["seasonal", "smoothed_seasonal"])
        if values is None:
            return None
        data["values"] = np.asarray(values, dtype=np.float64)
        std = _get_component_std(results, name)
        if std is not None:
            data["std"] = std
    elif name == "cycle":
        values = _get_attr(results, ["cycle", "smoothed_cycle"])
        if values is None:
            return None
        data["values"] = np.asarray(values, dtype=np.float64)
        std = _get_component_std(results, name)
        if std is not None:
            data["std"] = std
    elif name == "irregular":
        values = _get_attr(
            results,
            [
                "irregular",
                "resid",
                "residuals",
                "standardized_residuals",
            ],
        )
        if values is None:
            return None
        data["values"] = np.asarray(values, dtype=np.float64)
    else:
        return None

    if time_index is not None and len(time_index) == len(data["values"]):
        data["time"] = time_index

    return data


def _get_attr(obj: Any, names: list[str]) -> Any:
    """Try to get an attribute from an object, trying multiple names."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return None


def _get_component_std(results: Any, name: str) -> NDArray[np.float64] | None:
    """Try to extract standard deviation for a component."""
    std_attr = _get_attr(results, [f"{name}_std", f"{name}_se", f"smoothed_{name}_std"])
    if std_attr is not None:
        return np.asarray(std_attr, dtype=np.float64)

    # Try extracting from covariance if available
    cov = _get_attr(results, ["smoothed_state_cov", "filtered_state_cov"])
    if cov is not None:
        component_index = _get_attr(results, [f"{name}_index", f"{name}_idx"])
        if component_index is not None:
            idx = int(component_index)
            if cov.ndim == 3:
                return np.sqrt(cov[:, idx, idx])
            elif cov.ndim == 2:
                return np.sqrt(np.diag(cov))

    return None


def _get_time_index(results: Any) -> NDArray[np.float64] | None:
    """Try to get a time index from results."""
    idx = _get_attr(results, ["time_index", "index", "dates"])
    if idx is not None:
        try:
            return np.asarray(idx)
        except (ValueError, TypeError):
            return None
    return None


def _ci_to_z(ci: float) -> float:
    """Convert confidence interval level to z-score."""
    from scipy import stats

    return float(stats.norm.ppf((1 + ci) / 2))
