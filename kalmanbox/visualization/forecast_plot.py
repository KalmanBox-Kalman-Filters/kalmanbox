"""Fan chart forecast plots in Bank of England style.

Creates graduated confidence interval plots for forecasts with
degradee colors from dark (narrow intervals) to light (wide intervals).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kalmanbox.visualization.themes import get_theme


def plot_forecast(
    results: Any,
    steps: int | None = None,
    ci_levels: list[float] | None = None,
    n_history: int | None = None,
    theme: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    ylabel: str | None = None,
) -> Figure:
    """Plot fan chart with graduated confidence intervals.

    Creates a Bank of England style fan chart showing historical data,
    forecast mean, and graduated confidence bands from dark (narrow)
    to light (wide).

    Parameters
    ----------
    results : StateSpaceResults or similar
        Fitted model results. Expected attributes:
        - `observed` or `endog`: historical data
        - `forecast_mean` or method `forecast(steps)`: point forecasts
        - `forecast_cov` or `forecast_se`: forecast uncertainty
    steps : int or None
        Number of forecast steps. If None, uses forecast already in results.
    ci_levels : list of float or None
        Confidence levels for bands. Default [0.50, 0.70, 0.90, 0.95].
    n_history : int or None
        Number of historical observations to show. Default all.
    theme : str or None
        Theme name. If None, uses current active theme.
    figsize : tuple
        Figure size (width, height). Default (12, 6).
    title : str or None
        Figure title. Default 'Forecast'.
    ylabel : str or None
        Y-axis label. Default None.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    theme_config = get_theme(theme)
    colors = theme_config.colors

    if ci_levels is None:
        ci_levels = [0.50, 0.70, 0.90, 0.95]
    # Sort widest first for proper layering
    ci_levels_sorted = sorted(ci_levels, reverse=True)

    # Extract data
    observed = _get_observed(results)
    forecast_data = _extract_forecast(results, steps)

    if observed is None:
        msg = "No observed data found in results."
        raise ValueError(msg)
    if forecast_data is None:
        msg = "No forecast data found in results. Call results.forecast(steps) first."
        raise ValueError(msg)

    fc_mean = forecast_data["mean"]
    fc_se = forecast_data.get("se", None)
    n_forecast = len(fc_mean)

    # Trim history if requested
    if n_history is not None:
        observed = observed[-n_history:]

    n_obs = len(observed)
    t_hist = np.arange(n_obs)
    t_fc = np.arange(n_obs - 1, n_obs + n_forecast)

    # Extend forecast to connect with last observation
    fc_mean_ext = np.concatenate([[observed[-1]], fc_mean])

    fig, ax = plt.subplots(figsize=figsize)

    # Plot CI bands (widest first, darkest last)
    band_colors = colors.get_band_colors(len(ci_levels_sorted))

    if fc_se is not None:
        from scipy import stats

        fc_se_ext = np.concatenate([[0.0], fc_se])

        for i, ci_level in enumerate(ci_levels_sorted):
            z = float(stats.norm.ppf((1 + ci_level) / 2))
            upper = fc_mean_ext + z * fc_se_ext
            lower = fc_mean_ext - z * fc_se_ext

            # Reverse index for color: widest band gets lightest color
            color_idx = i
            alpha = 0.25 + 0.15 * (len(ci_levels_sorted) - 1 - i)

            ax.fill_between(
                t_fc,
                lower,
                upper,
                color=band_colors[color_idx],
                alpha=alpha,
                label=f"{ci_level * 100:.0f}% CI",
                zorder=1,
            )

    # Historical data
    ax.plot(
        t_hist,
        observed,
        color=colors.primary,
        linewidth=theme_config.line_width,
        label="Observed",
        zorder=3,
    )

    # Forecast mean
    ax.plot(
        t_fc,
        fc_mean_ext,
        color=colors.accent,
        linewidth=theme_config.line_width,
        linestyle="--",
        label="Forecast",
        zorder=4,
    )

    # Vertical line at forecast origin
    ax.axvline(
        x=n_obs - 1,
        color=colors.text,
        linewidth=0.8,
        linestyle=":",
        alpha=0.5,
        zorder=2,
    )

    # Labels and legend
    ax.set_xlabel("Time", fontsize=theme_config.fonts.label_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=theme_config.fonts.label_size)

    ax.legend(
        loc="upper left",
        fontsize=theme_config.fonts.legend_size,
        framealpha=0.8,
    )

    fig_title = title or "Forecast"
    ax.set_title(fig_title, fontsize=theme_config.fonts.title_size)
    fig.tight_layout()

    return fig


def _get_observed(results: Any) -> NDArray[np.float64] | None:
    """Get observed data from results."""
    for name in ["observed", "endog", "y"]:
        val = getattr(results, name, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64).ravel()
    return None


def _extract_forecast(
    results: Any,
    steps: int | None = None,
) -> dict[str, NDArray[np.float64]] | None:
    """Extract forecast data from results.

    Tries multiple approaches:
    1. Direct attributes (forecast_mean, forecast_se)
    2. Calling results.forecast(steps)
    3. Calling results.get_forecast(steps)
    """
    # Try direct attributes
    fc_mean = getattr(results, "forecast_mean", None)
    fc_se = getattr(results, "forecast_se", None)
    if fc_mean is not None:
        data: dict[str, NDArray[np.float64]] = {
            "mean": np.asarray(fc_mean, dtype=np.float64).ravel(),
        }
        if fc_se is not None:
            data["se"] = np.asarray(fc_se, dtype=np.float64).ravel()
        return data

    # Try calling forecast method
    if steps is not None:
        for method_name in ["forecast", "get_forecast"]:
            method = getattr(results, method_name, None)
            if method is not None:
                try:
                    fc_result = method(steps=steps)
                    return _parse_forecast_result(fc_result)
                except (TypeError, AttributeError):
                    try:
                        fc_result = method(steps)
                        return _parse_forecast_result(fc_result)
                    except (TypeError, AttributeError):
                        continue

    return None


def _parse_forecast_result(fc_result: Any) -> dict[str, NDArray[np.float64]] | None:
    """Parse various forecast result formats."""
    if fc_result is None:
        return None

    # If it's a dict-like
    if isinstance(fc_result, dict):
        mean = fc_result.get("mean", fc_result.get("forecast"))
        if mean is not None:
            data: dict[str, NDArray[np.float64]] = {
                "mean": np.asarray(mean, dtype=np.float64).ravel(),
            }
            se = fc_result.get("se", fc_result.get("se_mean", fc_result.get("std")))
            if se is not None:
                data["se"] = np.asarray(se, dtype=np.float64).ravel()
            return data

    # If it has attributes
    mean = getattr(fc_result, "forecast_mean", None)
    if mean is None:
        mean = getattr(fc_result, "mean", None)
    if mean is None:
        mean = getattr(fc_result, "predicted_mean", None)
    if mean is None:
        # Maybe it's just an array
        try:
            mean = np.asarray(fc_result, dtype=np.float64).ravel()
            return {"mean": mean}
        except (ValueError, TypeError):
            return None

    data = {"mean": np.asarray(mean, dtype=np.float64).ravel()}
    se = getattr(fc_result, "forecast_se", None)
    if se is None:
        se = getattr(fc_result, "se_mean", None)
    if se is None:
        se = getattr(fc_result, "se", None)
    if se is not None:
        data["se"] = np.asarray(se, dtype=np.float64).ravel()

    return data
