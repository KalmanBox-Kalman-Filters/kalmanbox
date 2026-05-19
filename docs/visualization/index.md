# Visualization

The `kalmanbox.visualization` module provides a complete set of plotting utilities designed around the state-space modeling workflow. Every function produces publication-ready figures with a consistent visual language, and all return standard `matplotlib.Figure` objects that compose naturally with the rest of the matplotlib ecosystem.

---

## Philosophy

Visualization in kalmanbox is not an afterthought added to inspect results after the fact. It is woven into the modeling workflow so that plots are available at every stage — from initial data exploration through filter diagnostics, model comparison, and final presentation.

Three principles guide the design:

1. **Workflow integration** — plot functions accept filter and smoother result objects directly, so no manual data extraction is needed.
2. **Sensible defaults** — every plot is readable out of the box, with axis labels, confidence shading, and legends configured automatically.
3. **Full composability** — all functions return `matplotlib.Figure`, making it straightforward to add custom annotations, embed plots in larger layouts, or save to any format supported by matplotlib.

---

## Quick start

```python
from kalmanbox import LocalLevelModel
from kalmanbox.visualization import plot_filtered_state, plot_diagnostic_panel, set_theme

# Apply a consistent theme across all plots
set_theme("publication")

# Fit a model
model = LocalLevelModel(data)
result = model.fit()

# Plot the filtered state with 95 % credible band
fig = plot_filtered_state(result, ci=0.95)
fig.savefig("state.png", dpi=150)

# Full diagnostic panel in one call
fig = plot_diagnostic_panel(result)
fig.savefig("diagnostics.pdf")
```

---

## Plot categories

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **State Plots**

    ---

    Visualise filtered and smoothed state estimates, multi-dimensional state
    trajectories, and one-step-ahead predictions together with observation data.

    [:octicons-arrow-right-24: State plots reference](state-plots.md)

-   :material-puzzle-edit-outline:{ .lg .middle } **Component Plots**

    ---

    Decompose a structural model into its constituent parts — trend, seasonal,
    cycle, and irregular components — each plotted on its own panel or overlaid.

    [:octicons-arrow-right-24: Component plots reference](component-plots.md)

-   :material-test-tube:{ .lg .middle } **Innovation Plots**

    ---

    Assess filter correctness through QQ plots, ACF of residuals, histograms,
    CUSUM charts, and a combined diagnostic panel that surfaces all standard
    tests at a glance.

    [:octicons-arrow-right-24: Innovation plots reference](innovation-plots.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Filter Comparison Plots**

    ---

    Compare two or more filters side-by-side using NEES trajectories, covariance
    trace evolution, and innovation statistics to guide filter selection.

    [:octicons-arrow-right-24: Filter comparison reference](filter-plots.md)

-   :material-chart-areaspline:{ .lg .middle } **Forecast Fan Charts**

    ---

    Project future uncertainty with multi-band fan charts that show 50 %, 80 %,
    and 95 % prediction intervals over the forecast horizon.

    [:octicons-arrow-right-24: Forecast fan charts reference](state-plots.md#forecast-fan-charts)

-   :material-table-multiple:{ .lg .middle } **Factor Plots**

    ---

    Visualise DFM common factors, inspect the loading matrix as a heatmap,
    assess explained variance, and select the number of factors with a scree plot.

    [:octicons-arrow-right-24: Factor plots reference](factor-plots.md)

-   :material-chart-timeline-variant:{ .lg .middle } **TVP Plots**

    ---

    Plot time-varying coefficients with posterior bands, compare against OLS,
    shade significance windows, and run parameter-stability tests.

    [:octicons-arrow-right-24: TVP plots reference](tvp-plots.md)

-   :material-palette-outline:{ .lg .middle } **Themes**

    ---

    Control the visual style of every kalmanbox plot from a single call to
    `set_theme()`. Four built-in themes and full custom-theme support.

    [:octicons-arrow-right-24: Themes reference](themes.md)

</div>

---

## Theming

All kalmanbox plots share a unified theme controlled by `set_theme()`. Call it once at the start of a session and every subsequent plot function inherits the settings.

```python
from kalmanbox.visualization import set_theme

set_theme("default")       # balanced, screen-friendly palette
set_theme("dark")          # dark background for presentations
set_theme("publication")   # grayscale-safe, serif fonts, tight margins
set_theme("minimal")       # no gridlines, white background, thin lines
```

You can also pass keyword arguments to override individual settings without changing the base theme:

```python
set_theme("publication", font_size=10, line_width=1.2, palette="tab10")
```

!!! tip "Jupyter notebooks"
    In a Jupyter environment, `set_theme()` also sets the default figure DPI to 144
    so that plots look crisp on high-resolution displays. Override with
    `set_theme("default", dpi=96)` if you prefer smaller inline figures.

The four built-in themes cover the most common presentation contexts:

| Theme | Background | Font | Grid | Use case |
|---|---|---|---|---|
| `"default"` | White | Sans-serif | Light gray | General exploration |
| `"dark"` | `#1e1e2e` | Sans-serif | Subtle | Presentations, slide decks |
| `"publication"` | White | Serif | None | Journal figures, PDFs |
| `"minimal"` | White | Sans-serif | None | Dashboards, web embedding |

---

## State plots

State plots show how the hidden state evolves over time together with its
uncertainty. The simplest entry point is `plot_filtered_state`, which draws
the filtered mean and a shaded credible band on top of the raw observations.

```python
from kalmanbox.visualization import (
    plot_filtered_state,
    plot_smoothed_state,
    plot_filtered_vs_smoothed,
    plot_state_trajectory,
    plot_prediction,
)

# Filtered state with 90 % and 95 % bands
fig = plot_filtered_state(result, ci=[0.90, 0.95], obs=True)

# Smoothed state (retrospective estimate)
fig = plot_smoothed_state(result, ci=0.95)

# Side-by-side comparison on the same axes
fig = plot_filtered_vs_smoothed(result, component=0)

# Phase-plane trajectory for a 2-D state vector
fig = plot_state_trajectory(result, x_dim=0, y_dim=1, colorbar="time")

# One-step-ahead predictions vs observations
fig = plot_prediction(result, horizon=1, ci=0.95)
```

!!! info "Multi-dimensional states"
    For models with more than one state dimension, pass `component=<int>` to
    select a single element, or omit it to produce a grid of subplots — one per
    state component.

---

## Component plots

Structural models decompose an observed series into interpretable components.
The component plot functions extract each part from a fitted result and display
them in a stacked panel layout.

```python
from kalmanbox.visualization import (
    plot_components,
    plot_trend,
    plot_seasonal,
    plot_cycle,
    plot_decomposition,
)

# All components stacked vertically
fig = plot_components(result, ci=0.95)

# Individual component panels
fig = plot_trend(result, ci=0.95)
fig = plot_seasonal(result, freq="monthly")
fig = plot_cycle(result, ci=0.95)

# X-13-ARIMA-style decomposition table + chart
fig = plot_decomposition(result, include_irregular=True)
```

The `plot_decomposition` function generates a four-panel figure — observed,
trend, seasonal, and irregular — mirroring the layout familiar from classical
decomposition tools, but with full uncertainty quantification from the Kalman
smoother.

---

## Innovation plots

Innovations (one-step-ahead prediction errors) should be zero-mean, serially
uncorrelated, and approximately Gaussian if the filter is correctly specified.
The innovation plot functions make it easy to verify all three properties.

```python
from kalmanbox.visualization import (
    plot_innovations,
    plot_qq,
    plot_acf_residuals,
    plot_histogram,
    plot_cusum,
    plot_diagnostic_panel,
)

# Raw standardised innovations over time
fig = plot_innovations(result, standardised=True)

# Normal QQ plot with 95 % simulation envelope
fig = plot_qq(result, envelope=True)

# ACF and PACF of standardised innovations
fig = plot_acf_residuals(result, lags=40, pacf=True)

# Histogram with fitted normal density overlay
fig = plot_histogram(result, bins="auto", kde=True)

# CUSUM and CUSUMSQ for structural break detection
fig = plot_cusum(result, alpha=0.05)

# Everything in one 2 x 3 panel
fig = plot_diagnostic_panel(result)
```

!!! info "Diagnostic panel layout"
    `plot_diagnostic_panel` arranges six subplots: innovations over time,
    histogram with normal overlay, QQ plot, ACF, CUSUM, and CUSUMSQ.
    It is the recommended first stop when assessing a fitted model.

---

## Filter comparison plots

When evaluating competing filters — for example KF vs UKF, or models with
different covariance structures — the comparison functions plot all candidates
on shared axes with a consistent colour scheme.

```python
from kalmanbox.visualization import (
    plot_filter_comparison,
    plot_covariance_trace,
    plot_innovation_comparison,
    plot_nees,
)

results = {"KF": result_kf, "UKF": result_ukf, "EKF": result_ekf}

# Overlay filtered state estimates from multiple filters
fig = plot_filter_comparison(results, component=0, ci=0.95)

# Posterior covariance trace (sum of diagonal) over time
fig = plot_covariance_trace(results, log_scale=False)

# Innovation magnitudes across filters
fig = plot_innovation_comparison(results)

# Normalised Estimation Error Squared (NEES) with chi-squared bounds
fig = plot_nees(results, alpha=0.05, n_states=2)
```

The NEES plot is especially useful when ground truth is available (e.g. in
simulation studies). Values consistently outside the $\chi^2$ bounds signal
filter inconsistency — either over- or under-confidence in the state estimates.

---

## Forecast fan charts

Fan charts communicate forecast uncertainty intuitively by shading progressively
darker intervals around the predictive mean. kalmanbox generates multi-band fan
charts from the predictive distribution returned by any state-space model.

```python
from kalmanbox.visualization import plot_forecast, plot_ppc

# 50 / 80 / 95 % prediction bands over a 24-step horizon
fig = plot_forecast(
    result,
    horizon=24,
    ci=[0.50, 0.80, 0.95],
    history=True,       # show in-sample fit alongside forecast
    freq="M",           # pandas offset alias for x-axis formatting
)

# Posterior predictive check — observed vs simulated replications
fig = plot_ppc(result, n_rep=200, alpha=0.05)
```

---

## Additional specialised plots

Two additional functions cover outputs specific to factor models and
time-varying parameter models:

```python
from kalmanbox.visualization import plot_factors, plot_tvp_coefficients

# Common factors extracted by a Dynamic Factor Model
fig = plot_factors(result, n_factors=3, loadings=True)

# Time-varying regression coefficients with posterior bands
fig = plot_tvp_coefficients(result, coeff_names=["gdp", "cpi", "rate"])
```

---

## Quick reference

### State plots

| Function | Description |
|---|---|
| `plot_filtered_state` | Filtered mean and credible band, with optional observations |
| `plot_smoothed_state` | Smoothed (retrospective) state estimate and band |
| `plot_filtered_vs_smoothed` | Overlay of filtered and smoothed estimates |
| `plot_state_trajectory` | Phase-plane trajectory for 2-D state vectors |
| `plot_prediction` | One-step-ahead predictions vs observations |

### Component plots

| Function | Description |
|---|---|
| `plot_components` | All structural components in a stacked panel |
| `plot_trend` | Trend component with uncertainty band |
| `plot_seasonal` | Seasonal component (annual, monthly, or custom) |
| `plot_cycle` | Stochastic cycle component |
| `plot_decomposition` | Four-panel decomposition: observed, trend, seasonal, irregular |

### Innovation plots

| Function | Description |
|---|---|
| `plot_innovations` | Standardised innovations over time |
| `plot_qq` | Normal QQ plot with simulation envelope |
| `plot_acf_residuals` | ACF and PACF of standardised innovations |
| `plot_histogram` | Innovation histogram with normal density overlay |
| `plot_cusum` | CUSUM and CUSUMSQ charts for structural stability |
| `plot_diagnostic_panel` | Combined 2 x 3 diagnostic panel |

### Filter comparison plots

| Function | Description |
|---|---|
| `plot_filter_comparison` | Overlay filtered states from multiple filters |
| `plot_covariance_trace` | Posterior covariance trace over time |
| `plot_innovation_comparison` | Innovation magnitudes across filters |
| `plot_nees` | NEES trajectory with $\chi^2$ consistency bounds |

### Forecast and general plots

| Function | Description |
|---|---|
| `plot_forecast` | Multi-band fan chart over a forecast horizon |
| `plot_ppc` | Posterior predictive check against observed data |
| `plot_factors` | Common factors from a Dynamic Factor Model |
| `plot_tvp_coefficients` | Time-varying regression coefficients |

---

## Saving and exporting

All plot functions return a `matplotlib.Figure` object. Use the standard
`savefig` interface to export to any format that matplotlib supports.

```python
fig = plot_diagnostic_panel(result)

# High-resolution PNG for web or presentations
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")

# Vector PDF for print or LaTeX inclusion
fig.savefig("diagnostics.pdf", bbox_inches="tight")

# SVG for embedding in HTML documents
fig.savefig("diagnostics.svg", bbox_inches="tight")

# Tight layout applied manually before saving
fig.tight_layout()
fig.savefig("diagnostics.png", dpi=300)
```

!!! tip "Batch export"
    When generating many figures in a pipeline, call `plt.close(fig)` after
    each `savefig` to release memory, especially for long time series with
    many observations.

```python
import matplotlib.pyplot as plt

figures = {
    "state": plot_filtered_state(result),
    "components": plot_components(result),
    "diagnostics": plot_diagnostic_panel(result),
}

for name, fig in figures.items():
    fig.savefig(f"report/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
```

---

## Integration with reports

Because every kalmanbox plot function returns a standard `matplotlib.Figure`,
the figures compose naturally with any Python-based reporting workflow.

**Jupyter notebooks** — figures render inline when the notebook backend is
active. No extra configuration is required.

```python
# In a Jupyter cell — the figure displays automatically
plot_diagnostic_panel(result)
```

**PDF reports** — use `matplotlib.backends.backend_pdf.PdfPages` to assemble
multiple figures into a single multi-page PDF.

```python
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("kalmanbox_report.pdf") as pdf:
    pdf.savefig(plot_filtered_state(result))
    pdf.savefig(plot_components(result))
    pdf.savefig(plot_diagnostic_panel(result))
```

**Automated pipelines** — the return value can be passed to any function that
accepts a `matplotlib.Figure`, including tools from `reportlab`, `plotly` (via
`plotly.tools.mpl_to_plotly`), and web frameworks that serve figures as image
streams.

!!! info "Matplotlib API compatibility"
    All plots return `matplotlib.Figure` — composable with the standard matplotlib
    API. You can add axes, adjust subplots, change colours, or apply any standard
    matplotlib operation after the kalmanbox function returns, before saving or
    displaying.

```python
fig = plot_filtered_state(result, ci=0.95)

# Add a custom annotation using standard matplotlib
ax = fig.axes[0]
ax.axvline(x="2008-09-15", color="red", linestyle="--", label="Lehman")
ax.legend()

fig.savefig("annotated_state.pdf", bbox_inches="tight")
```

---

## Related pages

- [State plots](state-plots.md) — filtered states, smoothed states, and forecast fan charts
- [Component plots](component-plots.md) — trend, seasonal, cycle, and irregular decomposition
- [Innovation plots](innovation-plots.md) — QQ, ACF, CUSUM, and the full diagnostic panel
- [Filter comparison plots](filter-plots.md) — NEES, covariance trace, and multi-filter overlays
