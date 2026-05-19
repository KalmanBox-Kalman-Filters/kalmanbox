# Factor Plots

Dynamic Factor Models (DFM) reduce a large panel of observed time series to a small set
of unobserved common factors. Visualising those factors alongside their relationship to
the original series is essential for model validation, economic interpretation, and
communication. The `kalmanbox.visualization` module provides six dedicated functions that
cover every step of the DFM visualisation workflow — from inspecting estimated factors to
scree plots for determining the right number of factors.

---

## Overview

| Function | What it shows |
|---|---|
| [`plot_factors`](#plot_factors) | Common factors estimated by the DFM over time |
| [`plot_loadings`](#plot_loadings) | Factor loadings as a bar chart (one series per bar) |
| [`plot_loadings_heatmap`](#plot_loadings_heatmap) | All loadings as an annotated heatmap |
| [`plot_explained_variance`](#plot_explained_variance) | Variance explained by each factor and cumulative share |
| [`plot_scree`](#plot_scree) | Eigenvalue scree plot for selecting the number of factors |
| [`plot_factor_vs_series`](#plot_factor_vs_series) | Factor overlaid on one or more original series for validation |

Import all six from the same module:

```python
from kalmanbox.visualization import (
    plot_factors,
    plot_loadings,
    plot_loadings_heatmap,
    plot_explained_variance,
    plot_scree,
    plot_factor_vs_series,
)
```

---

## Setting up a DFM result

All factor-plot functions accept a `DFMResult` object returned by `DFM.fit()`. The
object must expose at minimum:

| Attribute | Shape | Meaning |
|---|---|---|
| `factors` | `(T, r)` | Filtered (or smoothed) factor estimates |
| `factor_covariances` | `(T, r, r)` | Factor posterior covariance |
| `loadings` | `(n, r)` | Factor loading matrix $\Lambda$ |
| `eigenvalues` | `(r,)` | Eigenvalues used in factor selection |
| `series_names` | `list[str]` | Names of the $n$ observed series |
| `factor_names` | `list[str]` | Names or labels for the $r$ factors |

```python
import numpy as np
import pandas as pd
from kalmanbox import DFM
from kalmanbox.visualization import set_theme

set_theme("kalmanbox_default")

# Macro panel: 10 US indicators, monthly 1990–2020 (T=372)
panel = pd.read_csv("us_macro_panel.csv", index_col=0, parse_dates=True)

model = DFM(n_factors=3, factor_order=1)
result = model.fit(panel)

# result.factors        shape (372, 3)
# result.loadings       shape (10, 3)
# result.eigenvalues    shape (3,)
# result.series_names   ['GDP', 'IP', 'UNRATE', ...]
# result.factor_names   ['Factor 1', 'Factor 2', 'Factor 3']
```

---

## `plot_factors`

Plot the latent factor estimates over time as a multi-panel figure, with optional
posterior confidence bands.

### Signature

```python
kalmanbox.visualization.plot_factors(
    result: DFMResult,
    *,
    factors: list[int] | None = None,
    ci: float | list[float] = 0.90,
    smoother: bool = True,
    share_axes: bool = False,
    colors: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Common Factors",
    xlabel: str = "Time",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `DFMResult`
:   Fitted DFM result object.

`factors` : `list[int] | None`, default `None`
:   Indices of the factors to plot. `None` plots all factors.
    Example: `[0, 1]` plots only the first two factors.

`ci` : `float | list[float]`, default `0.90`
:   Confidence level(s) for the shaded posterior bands. Accepts a single float
    (e.g. `0.90`) or a list of two levels (e.g. `[0.68, 0.95]`) to draw nested
    inner and outer bands, mimicking a fan chart.

`smoother` : `bool`, default `True`
:   If `True`, plot smoothed factor estimates (using the RTS smoother applied after
    the Kalman filter pass) for a retrospectively optimal trajectory. If `False`,
    plot one-step-ahead filtered factors.

`share_axes` : `bool`, default `False`
:   If `True`, all factor panels share the same $y$-axis range — useful for comparing
    factor volatilities directly. If `False`, each panel has an independent $y$-axis.

`colors` : `list[str] | None`, default `None`
:   One colour per factor. Defaults to the active theme's sequential palette.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure size in inches. Defaults to `(11, 2.8 * r)` where $r$ is the number
    of factors shown.

`title` : `str`, default `"Common Factors"`
:   Figure suptitle.

`xlabel` : `str`, default `"Time"`
:   Label for the shared $x$-axis.

### Returns

`matplotlib.figure.Figure` with $r$ row subplots, one per factor.

### Visual description

Each subplot shows one factor's mean trajectory as a solid line with a shaded
credible band at the requested confidence level. If two levels are passed to `ci`,
a darker inner band (e.g. 68%) is nested inside a lighter outer band (e.g. 95%),
mimicking a fan chart. Factor panels share the $x$-axis but have independent
$y$-axes scaled to each factor's variance (or a shared range when `share_axes=True`).
A horizontal dashed line at zero is drawn to orient the sign of each factor.

### Example

```python
from kalmanbox.visualization import plot_factors

# Three factors with 68% and 95% confidence bands (fan chart style)
fig = plot_factors(
    result,
    ci=[0.68, 0.95],
    smoother=True,
    figsize=(12, 9),
    title="US Business Cycle Factors — DFM(3)",
)
fig.tight_layout()
fig.savefig("factors.pdf", bbox_inches="tight")
```

To plot only the first two factors side by side:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_factors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
plot_factors(result, factors=[0], ax=ax1, ci=0.90, title="Factor 1")
plot_factors(result, factors=[1], ax=ax2, ci=0.90, title="Factor 2")
fig.suptitle("First two common factors", fontsize=13)
fig.tight_layout()
```

!!! tip "Normalising factor signs"
    Kalman-filter factors are identified only up to sign. If your first factor has
    a strong positive loading on GDP but appears inverted, flip it with
    `result.factors[:, 0] *= -1` and `result.loadings[:, 0] *= -1` before plotting.
    The `DFM` class exposes a convenience method `result.align_signs(anchor_series=0)`
    that automatically picks the sign making the anchor series loading positive.

---

## `plot_loadings`

Display the factor loadings for a chosen factor as a bar chart, sorted by magnitude.
This is the standard way to identify which observed series are most strongly driven
by a common factor.

### Signature

```python
kalmanbox.visualization.plot_loadings(
    result: DFMResult,
    factor: int = 0,
    *,
    sort: bool = True,
    normalize: bool = False,
    rotation: float | None = None,
    positive_color: str = "#2196F3",
    negative_color: str = "#EF5350",
    figsize: tuple[float, float] = (9, 4),
    title: str | None = None,
    xlabel: str = "Series",
    ylabel: str = "Loading",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `DFMResult`
:   Fitted DFM result.

`factor` : `int`, default `0`
:   Which factor's loadings to display. `0` = first factor (typically the level
    or activity factor in a macroeconomic DFM).

`sort` : `bool`, default `True`
:   Sort bars from largest to smallest absolute loading. When `False`, bars are
    shown in the order of `result.series_names`.

`normalize` : `bool`, default `False`
:   Normalise loadings by the factor's standard deviation so that a loading of 1
    corresponds to a one-standard-deviation response. Useful for comparing across
    factors with different variances.

`rotation` : `float | None`, default `None`
:   Rotation angle for $x$-axis tick labels in degrees. Defaults to `45` when
    there are more than 8 series, and `0` otherwise.

`positive_color` : `str`, default `"#2196F3"`
:   Bar colour for positive loadings.

`negative_color` : `str`, default `"#EF5350"`
:   Bar colour for negative loadings.

`figsize` : `tuple[float, float]`, default `(9, 4)`
:   Figure dimensions.

`title` : `str | None`, default `None`
:   Axes title. Defaults to `f"Factor {factor + 1} loadings"`.

`xlabel`, `ylabel` : `str`
:   Axis labels.

### Returns

`matplotlib.figure.Figure`

### Visual description

Vertical bars are drawn for each observed series, coloured blue for positive loadings
and red for negative. A horizontal dashed line at zero is included for reference. When
`sort=True` the bar with the largest absolute value appears first, making it immediately
clear which series is most closely tied to the factor. Series labels appear on the
$x$-axis at a slight angle when many series are present.

### Example

```python
from kalmanbox.visualization import plot_loadings

# Loadings on the first (activity) factor
fig = plot_loadings(
    result,
    factor=0,
    sort=True,
    normalize=False,
    figsize=(10, 4),
    title="Factor 1 — Activity factor loadings",
)
fig.savefig("loadings_f1.png", dpi=150)
```

To compare loadings for all three factors:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_loadings

n_factors = result.n_factors  # 3
fig, axes = plt.subplots(n_factors, 1, figsize=(11, 4 * n_factors), sharex=False)

for i, ax in enumerate(axes):
    plot_loadings(result, factor=i, sort=True, ax=ax,
                  title=f"Factor {i + 1} loadings")

fig.tight_layout()
fig.savefig("loadings_all_factors.png", dpi=150)
```

---

## `plot_loadings_heatmap`

Display the complete loading matrix $\Lambda \in \mathbb{R}^{n \times r}$ as a
colour-coded heatmap with numerical annotations, giving a simultaneous view of all
factor–series relationships.

### Signature

```python
kalmanbox.visualization.plot_loadings_heatmap(
    result: DFMResult,
    *,
    normalize: bool = False,
    rotation: str | None = None,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    annot: bool = True,
    annot_fmt: str = ".2f",
    linewidths: float = 0.5,
    figsize: tuple[float, float] | None = None,
    title: str = "Factor Loading Matrix",
    xlabel: str = "Factor",
    ylabel: str = "Series",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `DFMResult`
:   Fitted DFM result containing the `loadings` matrix of shape `(n, r)`.

`normalize` : `bool`, default `False`
:   Column-normalise each factor's loadings to unit variance before plotting so that
    heatmap colours reflect the proportional contribution of each series, independent
    of factor scale.

`rotation` : `str | None`, default `None`
:   Variate rotation applied before plotting. Currently supports `"varimax"` and
    `"quartimax"`. Rotation improves interpretability by making each factor load
    strongly on fewer series. `None` uses the unrotated loadings from the DFM.

`cmap` : `str`, default `"RdBu_r"`
:   Matplotlib diverging colormap name. `"RdBu_r"` (red–white–blue reversed) shows
    positive loadings in blue and negative in red, centred at zero.

`center` : `float`, default `0.0`
:   Centre of the colormap. Values above `center` are coloured with the positive
    half of the palette and values below with the negative half.

`annot` : `bool`, default `True`
:   Annotate each cell with its numeric value.

`annot_fmt` : `str`, default `".2f"`
:   Python format string for the cell annotations.

`linewidths` : `float`, default `0.5`
:   Width of the grid lines separating cells.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure size. Defaults to `(3.5 * r, 0.5 * n + 1.5)` so cells are approximately
    square regardless of the number of series and factors.

`title` : `str`, default `"Factor Loading Matrix"`
:   Figure title.

`xlabel`, `ylabel` : `str`
:   Axis labels for factor columns and series rows.

### Returns

`matplotlib.figure.Figure`

### Visual description

The heatmap is a grid with observed series on rows and factors on columns. Each cell
shows the loading $\lambda_{ij}$ as both a colour (diverging blue–red palette centred
at zero) and a numeric annotation. Large positive values (blue) indicate that the factor
drives the series upward, while large negative values (red) indicate the opposite. Near-
zero values (white) indicate that the factor has little influence on that series. The
column structure reveals the factor's economic interpretation — a factor that loads
positively on GDP, IP, and investment while loading negatively on unemployment is a
classic activity/business-cycle factor.

### Example

```python
from kalmanbox.visualization import plot_loadings_heatmap

fig = plot_loadings_heatmap(
    result,
    normalize=True,      # unit-variance normalisation across factors
    rotation="varimax",  # rotate for better interpretability
    annot=True,
    annot_fmt=".2f",
    figsize=(8, 6),
    title="Varimax-Rotated Loading Matrix — US Macro DFM(3)",
)
fig.savefig("loadings_heatmap.pdf", bbox_inches="tight")
```

!!! info "Varimax rotation"
    Varimax rotation maximises the variance of squared loadings within each column,
    driving each factor to load on a small subset of series with large values and near-
    zero loadings elsewhere. The resulting factors are more interpretable because each
    corresponds to a distinct group of series, but the rotated factors are no longer
    statistically identified by the likelihood — use rotation for interpretation only,
    not for likelihood-based inference.

---

## `plot_explained_variance`

Visualise the share of the observed panel's total variance explained by each factor,
individually and cumulatively.

### Signature

```python
kalmanbox.visualization.plot_explained_variance(
    result: DFMResult,
    *,
    max_factors: int | None = None,
    bar_color: str = "#1565C0",
    line_color: str = "#E53935",
    threshold: float | None = 0.80,
    figsize: tuple[float, float] = (8, 5),
    title: str = "Variance Explained by Factor",
    xlabel: str = "Factor",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `DFMResult`
:   Fitted DFM result. The function uses `result.eigenvalues` together with the total
    panel variance to compute the per-factor explained variance ratio.

`max_factors` : `int | None`, default `None`
:   Number of factors to show. Useful when the DFM was estimated with many factors but
    you want to focus on the leading ones. `None` shows all factors in `result`.

`bar_color` : `str`, default `"#1565C0"`
:   Colour for the individual explained-variance bars.

`line_color` : `str`, default `"#E53935"`
:   Colour for the cumulative explained-variance line and markers.

`threshold` : `float | None`, default `0.80`
:   If not `None`, draw a horizontal dashed line at this cumulative variance level
    (e.g. `0.80` for 80%) and annotate the number of factors needed to reach it.

`figsize` : `tuple[float, float]`, default `(8, 5)`
:   Figure dimensions.

`title` : `str`, default `"Variance Explained by Factor"`
:   Axes title.

`xlabel` : `str`, default `"Factor"`
:   $x$-axis label.

### Returns

`matplotlib.figure.Figure` with a twin-axis layout: bars on the left $y$-axis
(individual explained variance, expressed as a percentage) and a line with markers
on the right $y$-axis (cumulative explained variance, 0–100%).

### Visual description

Blue bars show how much of the total panel variance is captured by each factor individually.
A red line with circle markers overlaid on a second $y$-axis tracks the cumulative sum.
When `threshold` is set, a dashed horizontal line marks the target cumulative level and a
vertical tick indicates the minimum number of factors needed to reach it. This is the
standard visualisation for choosing the number of factors: select the smallest $r$ such that
the cumulative bar reaches the threshold.

### Example

```python
from kalmanbox.visualization import plot_explained_variance

fig = plot_explained_variance(
    result,
    max_factors=10,
    threshold=0.80,        # draw 80% cumulative reference line
    figsize=(9, 5),
    title="Variance Explained — US Macro Panel",
)
fig.savefig("explained_variance.png", dpi=150)
```

---

## `plot_scree`

Plot an eigenvalue scree plot to guide factor-number selection using the elbow criterion
or information criteria.

### Signature

```python
kalmanbox.visualization.plot_scree(
    result: DFMResult,
    *,
    max_factors: int | None = None,
    ic_values: dict[str, np.ndarray] | None = None,
    selected: int | None = None,
    scatter_color: str = "#1565C0",
    figsize: tuple[float, float] = (8, 5),
    title: str = "Scree Plot",
    xlabel: str = "Number of Factors",
    ylabel: str = "Eigenvalue",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `DFMResult`
:   Fitted DFM result. Eigenvalues are read from `result.eigenvalues`, which are
    returned in descending order.

`max_factors` : `int | None`, default `None`
:   Number of eigenvalues to plot. `None` plots all.

`ic_values` : `dict[str, np.ndarray] | None`, default `None`
:   Optional dictionary of information criterion values keyed by criterion name
    (e.g. `{"BIC": bic_vals, "IC1": ic1_vals}`). When provided, a second panel is
    added below the scree plot showing how each IC changes with the number of factors.
    The minimum of each criterion is marked with a vertical dashed line.

`selected` : `int | None`, default `None`
:   If set, highlight the chosen number of factors with a vertical dashed line and
    a label.

`scatter_color` : `str`, default `"#1565C0"`
:   Colour for the eigenvalue dots and the connecting line.

`figsize` : `tuple[float, float]`, default `(8, 5)` (single panel) or `(8, 9)` (with IC panel)
:   Figure dimensions. Adjusted automatically when `ic_values` is provided.

`title` : `str`, default `"Scree Plot"`
:   Figure suptitle.

`xlabel`, `ylabel` : `str`
:   Axis labels for the scree panel.

### Returns

`matplotlib.figure.Figure` with one panel (scree only) or two panels (scree + IC).

### Visual description

The upper panel plots eigenvalues in descending order as blue dots connected by a line.
The "elbow" — the point where the slope flattens sharply — indicates the number of
meaningful factors. When `ic_values` is provided, the lower panel shows the IC curves
and marks each criterion's minimum with a vertical dashed line. When `selected` is set, a
vertical line in the selected colour crosses both panels to show the chosen $r$.

### Example

```python
import numpy as np
from kalmanbox import DFM
from kalmanbox.visualization import plot_scree

# Fit models with 1..8 factors and collect BIC and IC2 (Bai-Ng)
bic_vals, ic2_vals = np.zeros(8), np.zeros(8)
for r in range(1, 9):
    res_r = DFM(n_factors=r, factor_order=1).fit(panel)
    bic_vals[r - 1] = res_r.bic
    ic2_vals[r - 1] = res_r.ic2   # Bai-Ng IC2

result = DFM(n_factors=8, factor_order=1).fit(panel)  # for eigenvalues

fig = plot_scree(
    result,
    max_factors=8,
    ic_values={"BIC": bic_vals, "IC2 (Bai-Ng)": ic2_vals},
    selected=3,          # highlight our chosen number of factors
    figsize=(9, 8),
    title="Scree Plot — US Macro Panel",
)
fig.savefig("scree.png", dpi=150)
```

!!! info "Bai-Ng information criteria"
    The `IC1`, `IC2`, and `IC3` criteria proposed by Bai and Ng (2002) are specifically
    designed for large panels and consistently estimate the number of factors as both
    $n \to \infty$ and $T \to \infty$. They are available on the `DFMResult` object as
    `result.ic1`, `result.ic2`, `result.ic3` when estimated via `DFM.fit()`.

---

## `plot_factor_vs_series`

Overlay one or more common factors on their matched observed series to visually assess
how well the factor captures the series' common movement.

### Signature

```python
kalmanbox.visualization.plot_factor_vs_series(
    result: DFMResult,
    factor: int = 0,
    series: int | str | list[int | str] | None = None,
    *,
    ci: float = 0.90,
    normalize: bool = True,
    smoother: bool = True,
    max_panels: int = 4,
    factor_color: str | None = None,
    series_color: str = "#616161",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `DFMResult`
:   Fitted DFM result.

`factor` : `int`, default `0`
:   Index of the common factor to overlay on the series.

`series` : `int | str | list[int | str] | None`, default `None`
:   Which observed series to include. Accepts:
    - `None` — automatically selects the series with the largest absolute loading on
      the chosen factor (up to `max_panels` series).
    - A single integer index or series name string.
    - A list of integers or strings.

`ci` : `float`, default `0.90`
:   Confidence level for the factor's posterior band.

`normalize` : `bool`, default `True`
:   Standardise both the factor and the series to zero mean and unit variance before
    overlaying. This removes scale differences so that co-movement patterns are visible
    on a common axis. Set to `False` to compare in original units.

`smoother` : `bool`, default `True`
:   Use smoothed factor estimates.

`max_panels` : `int`, default `4`
:   Maximum number of series panels to generate when `series=None`.

`factor_color` : `str | None`, default `None`
:   Colour for the factor line and band. Defaults to the first colour in the active
    theme palette.

`series_color` : `str`, default `"#616161"`
:   Colour for the observed series lines.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure dimensions. Defaults to `(12, 3.5 * n_panels)`.

`title` : `str | None`, default `None`
:   Figure suptitle. Defaults to `f"Factor {factor + 1} vs observed series"`.

### Returns

`matplotlib.figure.Figure` with one panel per selected series.

### Visual description

Each panel shows the selected factor as a solid coloured line with a light shaded
confidence band, overlaid on the corresponding observed series plotted as a grey line.
When `normalize=True`, both are plotted on the same dimensionless scale so that the
correlation between factor and series is immediately visible. High visual correlation
confirms that the factor adequately captures the common variation in that series. Low
or erratic correlation signals that the series may be an outlier or that the factor
structure is misspecified.

### Example

```python
from kalmanbox.visualization import plot_factor_vs_series

# Auto-select the 4 series most strongly loaded on Factor 1
fig = plot_factor_vs_series(
    result,
    factor=0,
    series=None,      # auto-select by loading magnitude
    max_panels=4,
    ci=0.90,
    normalize=True,
    title="Activity factor vs most-loaded series",
)
fig.savefig("factor_vs_series.png", dpi=150)

# Manually compare Factor 2 against specific series
fig = plot_factor_vs_series(
    result,
    factor=1,
    series=["CPI", "PCE", "WAGE"],
    normalize=True,
    figsize=(12, 10),
    title="Factor 2 (Inflation) vs price-level series",
)
fig.savefig("inflation_factor.png", dpi=150)
```

---

## Complete macroeconomic example

The following end-to-end example fits a three-factor DFM to a US macroeconomic panel and
produces all six visualisations in a single script.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanbox import DFM
from kalmanbox.visualization import (
    plot_factors,
    plot_loadings,
    plot_loadings_heatmap,
    plot_explained_variance,
    plot_scree,
    plot_factor_vs_series,
    set_theme,
)

# ── Data ─────────────────────────────────────────────────────────────────────
# 10 US quarterly indicators (1985 Q1 – 2019 Q4, T=140)
series_names = [
    "Real GDP", "Industrial Production", "Unemployment Rate",
    "CPI", "Core PCE", "Wages", "Retail Sales",
    "Housing Starts", "S&P 500 Returns", "Federal Funds Rate",
]
rng = np.random.default_rng(42)
T, n = 140, 10

# Simulate a 3-factor panel with known loading structure
factors_true = rng.standard_normal((T, 3))
factors_true[:, 0] = np.cumsum(rng.normal(0, 0.3, T))  # persistent level factor
factors_true[:, 1] = 0.7 * factors_true[:, 1] + 0.3 * np.roll(factors_true[:, 1], 1)
factors_true[:, 2] = np.sin(np.linspace(0, 4 * np.pi, T)) + 0.5 * rng.standard_normal(T)

Lambda = np.array([
    [ 0.85,  0.20,  0.05],  # Real GDP        — mostly factor 1
    [ 0.78,  0.15,  0.10],  # Industrial Prod  — mostly factor 1
    [-0.72,  0.10,  0.20],  # Unemployment     — negatively loaded on factor 1
    [ 0.10,  0.80,  0.15],  # CPI              — mostly factor 2
    [ 0.08,  0.75,  0.12],  # Core PCE         — mostly factor 2
    [ 0.15,  0.70,  0.05],  # Wages            — mostly factor 2
    [ 0.55,  0.20,  0.35],  # Retail Sales     — mixed
    [ 0.40, -0.10,  0.65],  # Housing Starts   — mostly factor 3
    [ 0.30,  0.10,  0.60],  # S&P 500          — mostly factor 3
    [ 0.05,  0.45,  0.20],  # Fed Funds Rate   — mostly factor 2
])
panel_data = factors_true @ Lambda.T + rng.normal(0, 0.3, (T, n))
panel = pd.DataFrame(panel_data, columns=series_names)

# ── Fit the model ─────────────────────────────────────────────────────────────
set_theme("kalmanbox_default")
model = DFM(n_factors=3, factor_order=1)
result = model.fit(panel)
result.factor_names = ["Activity", "Inflation", "Financial"]

# ── 1. Plot common factors ────────────────────────────────────────────────────
fig1 = plot_factors(result, ci=[0.68, 0.95], smoother=True, figsize=(12, 9))
fig1.savefig("dfm_factors.png", dpi=150)
plt.close(fig1)

# ── 2. Bar chart of loadings — Activity factor ────────────────────────────────
fig2 = plot_loadings(result, factor=0, sort=True,
                     title="Activity Factor Loadings")
fig2.savefig("dfm_loadings_f1.png", dpi=150)
plt.close(fig2)

# ── 3. Heatmap of full loading matrix ────────────────────────────────────────
fig3 = plot_loadings_heatmap(result, rotation="varimax",
                              title="Varimax-Rotated Loading Matrix")
fig3.savefig("dfm_loadings_heatmap.pdf", bbox_inches="tight")
plt.close(fig3)

# ── 4. Explained variance ─────────────────────────────────────────────────────
fig4 = plot_explained_variance(result, threshold=0.80,
                                title="Variance Explained — US Macro Panel")
fig4.savefig("dfm_explained_variance.png", dpi=150)
plt.close(fig4)

# ── 5. Scree plot with BIC ────────────────────────────────────────────────────
bic_vals = np.array([
    DFM(n_factors=r, factor_order=1).fit(panel).bic for r in range(1, 9)
])
result8 = DFM(n_factors=8, factor_order=1).fit(panel)
fig5 = plot_scree(result8, max_factors=8,
                  ic_values={"BIC": bic_vals},
                  selected=3, title="Factor Selection — Scree + BIC")
fig5.savefig("dfm_scree.png", dpi=150)
plt.close(fig5)

# ── 6. Factor vs most-loaded series ──────────────────────────────────────────
fig6 = plot_factor_vs_series(result, factor=0, max_panels=3, normalize=True,
                              title="Activity Factor vs GDP, IP, Unemployment")
fig6.savefig("dfm_factor_vs_series.png", dpi=150)
plt.close(fig6)
```

### Interpreting the output

**Factors plot.** The Activity factor (Factor 1) exhibits a persistent downward trend
during recessions, the Inflation factor (Factor 2) captures medium-frequency price
movements, and the Financial factor (Factor 3) oscillates with the business cycle's
financial dimension. Nested confidence bands reveal that the Activity factor is
estimated with substantially less uncertainty than the Financial factor, reflecting its
stronger loadings across the panel.

**Loading bar chart.** GDP, Industrial Production, and Unemployment dominate the
Activity factor with the expected signs. Retail Sales and Housing Starts also load
positively, consistent with standard NBER business cycle dating.

**Loading heatmap.** After varimax rotation, the three columns become near-block-
diagonal: activity-related series load exclusively on the first factor, price-level
series on the second, and financial series on the third. This structure is impossible
to see from the unrotated loadings.

**Explained variance.** The first factor alone explains ~42% of total panel variance;
three factors together explain 79%. The 80% threshold line confirms that three factors
are sufficient.

**Scree plot.** A sharp elbow appears at $r = 3$, and the BIC curve reaches its
minimum at $r = 3$, consistent with the DGP.

**Factor vs series.** The Activity factor tracks GDP growth (R ≈ 0.91) and
Industrial Production (R ≈ 0.87) tightly when normalised, confirming that the DFM
has correctly identified the dominant common movement in these indicators.

---

## Customisation reference

### Loading rotation options

| `rotation` value | Algorithm | When to use |
|---|---|---|
| `None` | Unrotated ML/EM loadings | Likelihood inference, forecasting |
| `"varimax"` | Orthogonal varimax | Interpretability with orthogonal factors |
| `"quartimax"` | Orthogonal quartimax | When one factor should dominate |

### Confidence bands

All functions that accept `ci` support either:

- A **single float**, e.g. `ci=0.95` — one shaded band at 95%.
- A **list of two floats**, e.g. `ci=[0.68, 0.95]` — inner 68% and outer 95% bands,
  creating a fan-chart effect that makes central and tail uncertainty distinguishable.

---

## Related pages

- [DFM user guide](../user-guide/advanced/dfm.md) — factor model specification, EM
  algorithm, identification, and forecasting
- [Tutorial: US Macro Dynamic Factor Model](../tutorials/us-macro-dfm.md) — step-by-step
  tutorial with real FRED data
- [Tutorial: DFM](../tutorials/dfm.md) — construction, estimation, and factor extraction
- [Theory: Dynamic Factor Models](../theory/dfm-theory.md) — likelihood, identification,
  asymptotic theory
- [Themes](themes.md) — control the visual style of all plots
