"""Visualization module for kalmanbox.

Provides plotting functions for state-space model results including
component decomposition, diagnostics, forecasts, and more.

Themes
------
Three built-in themes are available:
- 'professional': Blue/gray palette, suitable for reports
- 'academic': Black/white minimal, suitable for papers
- 'presentation': Vibrant colors, suitable for slides

Usage
-----
>>> from kalmanbox.visualization import set_theme, plot_components
>>> set_theme('professional')
>>> fig = plot_components(results)
"""

from kalmanbox.visualization.components import plot_components
from kalmanbox.visualization.diagnostics_plot import plot_diagnostics
from kalmanbox.visualization.export import (
    close_figure,
    export_figure,
    figure_to_bytes,
)
from kalmanbox.visualization.factor_plot import (
    plot_factors,
    plot_loadings,
    plot_variance_decomposition,
)
from kalmanbox.visualization.filter_plot import (
    plot_covariance_convergence,
    plot_prediction_errors,
)
from kalmanbox.visualization.forecast_plot import plot_forecast
from kalmanbox.visualization.state_plot import plot_states
from kalmanbox.visualization.themes import (
    get_theme,
    list_themes,
    register_theme,
    reset_theme,
    set_theme,
)

__all__ = [
    # Themes
    "get_theme",
    "set_theme",
    "reset_theme",
    "list_themes",
    "register_theme",
    # Export
    "export_figure",
    "figure_to_bytes",
    "close_figure",
    # Plots
    "plot_components",
    "plot_states",
    "plot_forecast",
    "plot_diagnostics",
    "plot_factors",
    "plot_loadings",
    "plot_variance_decomposition",
    "plot_prediction_errors",
    "plot_covariance_convergence",
]
