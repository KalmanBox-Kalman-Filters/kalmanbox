# TVP Plots

Time-Varying Parameter (TVP) models allow regression coefficients to evolve over time
as random walks rather than remaining fixed. Visualising this evolution — together with
posterior uncertainty bands and comparisons to OLS benchmarks — is central to interpreting
TVP results. The five plot functions documented here cover the full TVP visualisation
workflow: from coefficient time-paths to significance shading, stability tests, and
side-by-side OLS comparison.

---

## Overview

| Function | What it shows |
|---|---|
| [`plot_tvp_coefficients`](#plot_tvp_coefficients) | Time-varying coefficients with posterior bands |
| [`plot_tvp_heatmap`](#plot_tvp_heatmap) | All coefficients across time as a colour-coded heatmap |
| [`plot_tvp_vs_ols`](#plot_tvp_vs_ols) | TVP estimates overlaid on OLS fixed estimate |
| [`plot_tvp_significance`](#plot_tvp_significance) | Periods where each coefficient is significantly non-zero |
| [`plot_tvp_stability`](#plot_tvp_stability) | Coefficient stability / structural break test over time |

Import all five from the visualisation module:

```python
from kalmanbox.visualization import (
    plot_tvp_coefficients,
    plot_tvp_heatmap,
    plot_tvp_vs_ols,
    plot_tvp_significance,
    plot_tvp_stability,
)
```

---

## Setting up a TVP result

All TVP plot functions accept a `TVPResult` object returned by `TVP.fit()`. The object
must expose:

| Attribute | Shape | Meaning |
|---|---|---|
| `coefficients` | `(T, k)` | Posterior mean of the time-varying coefficients |
| `coefficient_covariances` | `(T, k, k)` | Posterior covariance of coefficients |
| `coeff_names` | `list[str]` | Names of the $k$ regressors |
| `observations` | `(T,)` | Dependent variable |
| `regressors` | `(T, k)` | Regressor matrix |

```python
import numpy as np
import pandas as pd
from kalmanbox import TVP
from kalmanbox.visualization import set_theme

set_theme("kalmanbox_default")

# Quarterly US data: output gap and lagged inflation as regressors
df = pd.read_csv("us_phillips_quarterly.csv", index_col=0, parse_dates=True)
y = df["CPI_inflation"].values
X = df[["output_gap", "lagged_inflation", "import_prices"]].values

model = TVP(coeff_names=["Output Gap", "Lagged Inflation", "Import Prices"])
result = model.fit(y, X)

# result.coefficients          shape (T, 3)
# result.coefficient_covariances shape (T, 3, 3)
# result.coeff_names            ['Output Gap', 'Lagged Inflation', 'Import Prices']
```

---

## `plot_tvp_coefficients`

The primary TVP visualisation: plot each regression coefficient as a function of time,
with posterior credible bands showing uncertainty around the estimated trajectory.

### Signature

```python
kalmanbox.visualization.plot_tvp_coefficients(
    result: TVPResult,
    *,
    coefficients: list[int | str] | None = None,
    ci: float | list[float] = 0.90,
    smoother: bool = True,
    zero_line: bool = True,
    share_y: bool = False,
    colors: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Time-Varying Coefficients",
    xlabel: str = "Time",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `TVPResult`
:   Fitted TVP result object.

`coefficients` : `list[int | str] | None`, default `None`
:   Subset of coefficients to plot, specified by integer index or name string.
    `None` plots all coefficients. Example: `["Output Gap", "Lagged Inflation"]`.

`ci` : `float | list[float]`, default `0.90`
:   Confidence level(s) for the posterior bands. Pass a single float for one band,
    or two floats (e.g. `[0.68, 0.95]`) for nested inner and outer bands (fan chart).

`smoother` : `bool`, default `True`
:   If `True`, plot the RTS-smoothed posterior coefficient estimates, which use all
    observations and are retrospectively optimal. If `False`, plot the filtered
    (one-step-ahead) estimates.

`zero_line` : `bool`, default `True`
:   Draw a dashed horizontal line at zero on each panel. This makes it visually
    immediate whether the coefficient crosses zero — i.e., whether its sign changes
    over time.

`share_y` : `bool`, default `False`
:   Share the $y$-axis range across all coefficient panels. Useful for comparing the
    magnitudes of different coefficients on a common scale, but sacrifices within-panel
    readability for small coefficients.

`colors` : `list[str] | None`, default `None`
:   One colour per coefficient panel. Defaults to the active theme palette.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure dimensions. Defaults to `(11, 3.2 * k)` where $k$ is the number of
    coefficients shown.

`title` : `str`, default `"Time-Varying Coefficients"`
:   Figure suptitle.

`xlabel` : `str`, default `"Time"`
:   Shared $x$-axis label.

### Returns

`matplotlib.figure.Figure` with one panel per coefficient.

### Visual description

Each panel displays the posterior mean coefficient trajectory as a solid line with a
shaded credible band. When two confidence levels are specified, a darker inner band
(typically 68%) is nested within a lighter outer band (typically 95%), giving an
intuitive representation of the full posterior. The zero line (when `zero_line=True`)
immediately shows periods where the coefficient's sign changes, which in a Phillips
curve context indicates when the slope relationship between the output gap and inflation
inverts.

### Example

```python
from kalmanbox.visualization import plot_tvp_coefficients

# All three coefficients with inner 68% and outer 95% bands
fig = plot_tvp_coefficients(
    result,
    ci=[0.68, 0.95],
    smoother=True,
    zero_line=True,
    figsize=(12, 10),
    title="Phillips Curve — Time-Varying Coefficients",
)
fig.tight_layout()
fig.savefig("tvp_coefficients.pdf", bbox_inches="tight")
```

To plot a single coefficient using an existing axes:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_tvp_coefficients

fig, ax = plt.subplots(figsize=(12, 4))
plot_tvp_coefficients(
    result,
    coefficients=["Output Gap"],
    ci=0.90,
    ax=ax,
    title="Phillips curve slope — output gap coefficient",
)
ax.axvspan("2008-01-01", "2009-12-01", color="#FFCDD2", alpha=0.4,
           label="GFC recession")
ax.legend()
fig.savefig("output_gap_coeff.png", dpi=150)
```

!!! tip "Recession shading"
    Overlay NBER recession shading on any TVP coefficient plot using `ax.axvspan(start,
    end, color="lightgray", alpha=0.3)`. This immediately reveals whether coefficients
    change behavior during contractions.

---

## `plot_tvp_heatmap`

Display all coefficients across all time periods simultaneously as a colour-coded
heatmap, enabling a compact view of the full coefficient matrix.

### Signature

```python
kalmanbox.visualization.plot_tvp_heatmap(
    result: TVPResult,
    *,
    coefficients: list[int | str] | None = None,
    smoother: bool = True,
    normalize: bool = False,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    annot: bool = False,
    annot_step: int = 10,
    annot_fmt: str = ".2f",
    figsize: tuple[float, float] | None = None,
    title: str = "TVP Coefficient Heatmap",
    xlabel: str = "Time",
    ylabel: str = "Coefficient",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `TVPResult`
:   Fitted TVP result.

`coefficients` : `list[int | str] | None`, default `None`
:   Subset of coefficients to include. `None` includes all.

`smoother` : `bool`, default `True`
:   Use RTS-smoothed coefficient estimates.

`normalize` : `bool`, default `False`
:   Normalise each coefficient row to zero mean and unit variance before plotting,
    so that the heatmap shows relative changes over time rather than absolute magnitudes.
    Useful when coefficients have very different scales.

`cmap` : `str`, default `"RdBu_r"`
:   Matplotlib diverging colormap. `"RdBu_r"` encodes positive values in blue and
    negative values in red, centred at zero.

`center` : `float`, default `0.0`
:   Colormap centre value.

`annot` : `bool`, default `False`
:   Annotate cells with numeric values. Because the time dimension $T$ can be large,
    annotations are shown only at every `annot_step` time periods.

`annot_step` : `int`, default `10`
:   Spacing between annotated time columns when `annot=True`.

`annot_fmt` : `str`, default `".2f"`
:   Number format for annotations.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure size. Defaults to `(min(T / 4, 16), max(k * 1.2, 4))`.

`title` : `str`, default `"TVP Coefficient Heatmap"`
:   Figure title.

`xlabel`, `ylabel` : `str`
:   Axis labels.

### Returns

`matplotlib.figure.Figure`

### Visual description

The heatmap has time on the $x$-axis and coefficients on the $y$-axis. Each cell's
colour encodes the posterior mean coefficient at that period: blue for positive, red
for negative, white for near-zero. This compact representation makes it easy to
spot structural breaks (sudden colour shifts), persistent sign changes, and periods
where multiple coefficients move together. The heatmap is most informative when
`normalize=True`, because it then shows each coefficient's deviation from its own
time-average.

### Example

```python
from kalmanbox.visualization import plot_tvp_heatmap

fig = plot_tvp_heatmap(
    result,
    normalize=True,        # highlight changes relative to each coeff's mean
    cmap="RdBu_r",
    figsize=(14, 4),
    title="TVP Heatmap — Normalised Coefficient Evolution",
)
fig.savefig("tvp_heatmap.pdf", bbox_inches="tight")
```

---

## `plot_tvp_vs_ols`

Overlay the TVP time-varying coefficient estimate on the corresponding OLS fixed
estimate to quantify how much parameter variation is missed by a static regression.

### Signature

```python
kalmanbox.visualization.plot_tvp_vs_ols(
    result: TVPResult,
    *,
    coefficients: list[int | str] | None = None,
    ci: float = 0.90,
    ols_color: str = "#E53935",
    tvp_color: str | None = None,
    ols_ci: bool = True,
    smoother: bool = True,
    zero_line: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "TVP vs OLS Coefficients",
    xlabel: str = "Time",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `TVPResult`
:   Fitted TVP result. The function extracts the OLS estimates from `result.ols`
    (automatically computed by `TVP.fit()` using `numpy.linalg.lstsq`).

`coefficients` : `list[int | str] | None`, default `None`
:   Coefficients to display. `None` shows all.

`ci` : `float`, default `0.90`
:   Confidence level for the TVP posterior band.

`ols_color` : `str`, default `"#E53935"`
:   Colour for the OLS horizontal reference line and (if `ols_ci=True`) for the OLS
    confidence band.

`tvp_color` : `str | None`, default `None`
:   Colour for the TVP trajectory. Defaults to the first theme colour.

`ols_ci` : `bool`, default `True`
:   Shade the OLS 95% confidence interval as a horizontal band. This gives a visual
    reference for the uncertainty of the static estimate against which the TVP
    time-variation is assessed.

`smoother` : `bool`, default `True`
:   Use RTS-smoothed TVP estimates.

`zero_line` : `bool`, default `True`
:   Draw a dashed line at zero.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure dimensions.

`title` : `str`, default `"TVP vs OLS Coefficients"`
:   Figure suptitle.

`xlabel` : `str`, default `"Time"`
:   $x$-axis label.

### Returns

`matplotlib.figure.Figure`

### Visual description

Each panel contains:
- A solid coloured line for the TVP posterior mean with a shaded credible band.
- A horizontal dashed red line for the OLS point estimate.
- (When `ols_ci=True`) A lightly shaded horizontal band for the OLS 95% CI.

Time periods where the TVP estimate falls outside the OLS band signal statistically
meaningful parameter instability that OLS cannot capture. The visual gap between the
time-varying and static estimates quantifies the cost of imposing coefficient
constancy.

### Example

```python
from kalmanbox.visualization import plot_tvp_vs_ols

fig = plot_tvp_vs_ols(
    result,
    ci=0.90,
    ols_ci=True,
    figsize=(12, 10),
    title="Phillips Curve — TVP vs OLS (1985–2020)",
)
fig.tight_layout()
fig.savefig("tvp_vs_ols.pdf", bbox_inches="tight")
```

For a single-panel inline comparison:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_tvp_vs_ols

fig = plot_tvp_vs_ols(
    result,
    coefficients=["Output Gap"],
    ci=[0.68, 0.95],
    ols_ci=True,
    figsize=(12, 4),
    title="Output gap coefficient: TVP vs static OLS",
)
fig.savefig("gap_tvp_vs_ols.png", dpi=150)
```

!!! info "Interpreting a flat TVP trajectory"
    If the TVP estimate is nearly constant over time and falls well within the OLS
    confidence band, the data are consistent with a fixed coefficient and the more
    parsimonious OLS model is preferable. Use a likelihood-ratio test
    (`TVP.lr_test_constant()`) to formalise this.

---

## `plot_tvp_significance`

Shade time periods where a coefficient is significantly different from zero at a chosen
credible level, providing a quick visual summary of when each regressor's effect is
statistically active.

### Signature

```python
kalmanbox.visualization.plot_tvp_significance(
    result: TVPResult,
    *,
    coefficients: list[int | str] | None = None,
    ci: float = 0.90,
    smoother: bool = True,
    sig_color: str = "#1565C0",
    insig_color: str = "#BDBDBD",
    background: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "TVP Coefficient Significance",
    xlabel: str = "Time",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `TVPResult`
:   Fitted TVP result.

`coefficients` : `list[int | str] | None`, default `None`
:   Subset of coefficients. `None` plots all.

`ci` : `float`, default `0.90`
:   Credible level used to define significance. A coefficient at time $t$ is
    classified as **significant** if the `(1 - ci) / 2` and `(1 + ci) / 2`
    posterior quantiles have the same sign — i.e., zero is excluded from the
    credible interval.

`smoother` : `bool`, default `True`
:   Use smoothed coefficient estimates.

`sig_color` : `str`, default `"#1565C0"`
:   Colour for the coefficient line during significant periods.

`insig_color` : `str`, default `"#BDBDBD"`
:   Colour for the coefficient line during insignificant periods.

`background` : `bool`, default `True`
:   Shade the background of significant periods with a very light version of
    `sig_color` to make significance windows immediately distinguishable.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure size.

`title` : `str`, default `"TVP Coefficient Significance"`
:   Figure suptitle.

`xlabel` : `str`, default `"Time"`
:   $x$-axis label.

### Returns

`matplotlib.figure.Figure`

### Visual description

Each panel draws the coefficient trajectory, colouring the line segment in `sig_color`
(blue) during periods where zero is excluded from the credible interval, and in
`insig_color` (grey) during insignificant periods. When `background=True`, significant
windows receive a light blue background fill. The dashed zero line is always present for
reference. This layout immediately answers "when did this regressor matter?" — a question
that cannot be answered by OLS or any constant-coefficient model.

### Example

```python
from kalmanbox.visualization import plot_tvp_significance

fig = plot_tvp_significance(
    result,
    ci=0.90,
    smoother=True,
    figsize=(12, 10),
    title="Phillips Curve — Coefficient Significance Periods",
)
fig.savefig("tvp_significance.png", dpi=150)
```

For a single coefficient with a tighter credible interval:

```python
from kalmanbox.visualization import plot_tvp_significance

fig = plot_tvp_significance(
    result,
    coefficients=["Output Gap"],
    ci=0.95,   # more conservative significance threshold
    figsize=(12, 4),
    title="Output gap — significant periods at 95% credibility",
)
fig.savefig("gap_significance.png", dpi=150)
```

---

## `plot_tvp_stability`

Test and visualise the temporal stability of the TVP coefficients using a
parameter-constancy test, displaying the test statistic over time alongside
critical-value bounds.

### Signature

```python
kalmanbox.visualization.plot_tvp_stability(
    result: TVPResult,
    *,
    coefficients: list[int | str] | None = None,
    test: str = "nyblom",
    alpha: float = 0.05,
    smoother: bool = True,
    colors: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Coefficient Stability Test",
    xlabel: str = "Time",
) -> matplotlib.figure.Figure
```

### Parameters

`result` : `TVPResult`
:   Fitted TVP result.

`coefficients` : `list[int | str] | None`, default `None`
:   Subset of coefficients. `None` shows all.

`test` : `str`, default `"nyblom"`
:   Stability test to use. Available options:

    - `"nyblom"` — Nyblom (1989) parameter-constancy test. The test statistic is the
      cumulative sum of outer products of score contributions and is asymptotically
      distributed as a Cramér–von Mises distribution under the null of constancy.
    - `"cusum"` — Recursive CUSUM of squared innovations. Exceeding the critical-value
      band indicates instability in the innovation variance, which reflects coefficient
      instability in a state-space context.
    - `"quandt"` — Supremum Wald / Quandt Likelihood Ratio (QLR) statistic over all
      possible break dates in a trimmed central region [15%, 85%]. The maximum of the
      sequence identifies the most likely break date.

`alpha` : `float`, default `0.05`
:   Significance level for the critical-value bounds. Critical values are taken from
    tables tabulated for the selected test.

`smoother` : `bool`, default `True`
:   Use smoothed coefficient estimates as the basis for the test statistic.

`colors` : `list[str] | None`, default `None`
:   Per-coefficient colours.

`figsize` : `tuple[float, float] | None`, default `None`
:   Figure dimensions. Defaults to `(11, 3.5 * k)`.

`title` : `str`, default `"Coefficient Stability Test"`
:   Figure suptitle.

`xlabel` : `str`, default `"Time"`
:   $x$-axis label.

### Returns

`matplotlib.figure.Figure`

### Visual description

Each panel shows the sequential stability test statistic for one coefficient plotted
over time. Dashed horizontal lines mark the critical values at significance level
`alpha`. Periods where the statistic exceeds the upper critical value indicate that the
coefficient's constancy hypothesis is rejected at that point — evidence of structural
change. When `test="quandt"`, the location of the supremum is annotated as the most
likely break date.

### Example

```python
from kalmanbox.visualization import plot_tvp_stability

# Nyblom parameter-constancy test for all three Phillips curve coefficients
fig = plot_tvp_stability(
    result,
    test="nyblom",
    alpha=0.05,
    figsize=(12, 10),
    title="Parameter Stability — Nyblom Test",
)
fig.savefig("tvp_stability_nyblom.png", dpi=150)

# QLR test to identify the most likely structural break date
fig = plot_tvp_stability(
    result,
    test="quandt",
    alpha=0.05,
    figsize=(12, 10),
    title="Quandt LR Test — Break Date Identification",
)
fig.savefig("tvp_stability_qlr.png", dpi=150)
```

---

## Complete Phillips curve example

The following end-to-end script fits a three-regressor TVP model to the US Phillips
curve and produces all five visualisation types.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanbox import TVP
from kalmanbox.visualization import (
    plot_tvp_coefficients,
    plot_tvp_heatmap,
    plot_tvp_vs_ols,
    plot_tvp_significance,
    plot_tvp_stability,
    set_theme,
)

# ── Data ─────────────────────────────────────────────────────────────────────
# Quarterly US data 1985 Q1 – 2020 Q4 (T = 144)
rng = np.random.default_rng(42)
T = 144
dates = pd.date_range("1985-01-01", periods=T, freq="QS")

# True DGP: time-varying output-gap slope with a post-2008 flattening
beta_true = np.zeros((T, 3))
beta_true[:, 0] = 0.5 - 0.3 * (np.arange(T) / T)      # declining output-gap slope
beta_true[:, 1] = 0.6 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, T))  # persistent inflation
beta_true[:, 2] = 0.1 * np.ones(T)                       # stable import-price pass-through

output_gap    = rng.normal(0, 1.0, T)
lag_inflation = np.zeros(T)
import_prices = rng.normal(0, 0.5, T)

y = np.zeros(T)
for t in range(1, T):
    lag_inflation[t] = y[t - 1]
    y[t] = (beta_true[t, 0] * output_gap[t]
            + beta_true[t, 1] * lag_inflation[t]
            + beta_true[t, 2] * import_prices[t]
            + rng.normal(0, 0.3))

X = np.column_stack([output_gap, lag_inflation, import_prices])

# ── Fit TVP model ─────────────────────────────────────────────────────────────
set_theme("kalmanbox_default")
model = TVP(
    coeff_names=["Output Gap Slope", "Inflation Persistence", "Import Pass-through"],
    process_variance=1e-4,   # small random-walk variance — smooth evolution
)
result = model.fit(y, X, index=dates)

# ── 1. Coefficient time paths ─────────────────────────────────────────────────
fig1 = plot_tvp_coefficients(
    result,
    ci=[0.68, 0.95],
    smoother=True,
    zero_line=True,
    title="Phillips Curve — Time-Varying Coefficients (1985–2020)",
    figsize=(12, 11),
)
fig1.tight_layout()
fig1.savefig("tvp_coefficients.pdf", bbox_inches="tight")
plt.close(fig1)

# ── 2. Heatmap overview ───────────────────────────────────────────────────────
fig2 = plot_tvp_heatmap(
    result,
    normalize=True,
    figsize=(14, 4),
    title="TVP Coefficient Heatmap — Normalised",
)
fig2.savefig("tvp_heatmap.png", dpi=150)
plt.close(fig2)

# ── 3. TVP vs OLS comparison ──────────────────────────────────────────────────
fig3 = plot_tvp_vs_ols(
    result,
    ci=0.90,
    ols_ci=True,
    title="Phillips Curve — TVP vs OLS (1985–2020)",
    figsize=(12, 11),
)
fig3.tight_layout()
fig3.savefig("tvp_vs_ols.pdf", bbox_inches="tight")
plt.close(fig3)

# ── 4. Significance periods ───────────────────────────────────────────────────
fig4 = plot_tvp_significance(
    result,
    ci=0.90,
    figsize=(12, 11),
    title="Significant Periods — 90% Credible Interval",
)
fig4.savefig("tvp_significance.png", dpi=150)
plt.close(fig4)

# ── 5. Stability tests ────────────────────────────────────────────────────────
fig5 = plot_tvp_stability(
    result,
    test="nyblom",
    alpha=0.05,
    figsize=(12, 11),
    title="Parameter Stability — Nyblom Test",
)
fig5.savefig("tvp_stability.png", dpi=150)
plt.close(fig5)
```

### Interpreting the output

**Coefficient time paths.** The output-gap slope declines from ~0.5 in the late 1980s
to near zero after the Great Financial Crisis, consistent with the widely-documented
flattening of the Phillips curve. The inflation-persistence coefficient oscillates
around 0.6, spiking briefly above 0.7 during the early 2000s. Import pass-through
remains near 0.1 throughout, unsurprisingly given the long-run exchange-rate stability
in the sample.

**Heatmap.** After normalisation, the declining output-gap slope shows as a colour
gradient from blue (large positive) in the early sample to white (near-zero) in the
post-GFC period. The inflation-persistence row shows a mid-sample blue concentration
corresponding to the 2000s.

**TVP vs OLS.** The OLS point estimates for the output-gap slope and inflation
persistence are significant averages of the time-varying paths. However, the OLS
confidence band in the late post-GFC period does not overlap with the TVP estimate for
the output-gap coefficient, confirming statistically that the slope has changed in a
way OLS cannot represent.

**Significance.** The output-gap slope is significant throughout the 1985–2007 period
but loses significance after 2009. Inflation persistence is significant for the entire
sample. Import pass-through is borderline throughout, consistent with the small
estimated coefficient.

**Stability test.** The Nyblom statistic for the output-gap coefficient exceeds the 5%
critical value starting around 2007–2008, pinpointing the GFC as the structural break
date, consistent with the academic literature on Phillips curve instability.

---

## Market beta example

TVP models are also used in finance to estimate time-varying CAPM betas. The following
snippet shows the most common pattern:

```python
import numpy as np
import pandas as pd
from kalmanbox import TVP
from kalmanbox.visualization import plot_tvp_coefficients, plot_tvp_vs_ols

# Weekly excess returns — tech stock vs market, 2005–2020 (T=782)
stock_returns = pd.read_csv("tech_stock.csv", index_col=0, parse_dates=True)
y_excess = (stock_returns["AAPL"] - stock_returns["RF"]).values
X_market  = (stock_returns["MKT"] - stock_returns["RF"]).values.reshape(-1, 1)

model = TVP(coeff_names=["Market Beta"], process_variance=1e-5)
result = model.fit(y_excess, X_market, index=stock_returns.index)

# Time-varying beta with fan chart
fig = plot_tvp_coefficients(
    result,
    ci=[0.68, 0.95],
    title="AAPL Market Beta — TVP-CAPM (2005–2020)",
    figsize=(12, 4),
)

# Overlay CAPM static beta
fig2 = plot_tvp_vs_ols(
    result,
    ci=0.90,
    ols_ci=True,
    title="Time-Varying vs Static CAPM Beta",
    figsize=(12, 4),
)
fig2.savefig("capm_beta.png", dpi=150)
```

---

## Customisation

### Significance definition

The significance classification used by `plot_tvp_significance` is Bayesian: a
coefficient $\beta_t$ is significant if zero lies outside the credible interval:

$$
P\!\left(\beta_t > 0 \mid y_{1:T}\right) > \frac{1 + \text{ci}}{2}
\quad\text{or}\quad
P\!\left(\beta_t < 0 \mid y_{1:T}\right) > \frac{1 + \text{ci}}{2}
$$

This is equivalent to checking whether the posterior mean coefficient lies more than
$z_{(1-\text{ci})/2}$ posterior standard deviations away from zero.

### Combining plots

Because all functions return standard `matplotlib.Figure` objects, you can assemble
custom multi-panel layouts:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_tvp_coefficients, plot_tvp_significance

# Create a figure with coefficients above and significance below
fig, axes = plt.subplots(6, 1, figsize=(13, 18), sharex=True)

for i in range(3):
    plot_tvp_coefficients(result, coefficients=[i], ci=0.90, ax=axes[i])
    plot_tvp_significance(result, coefficients=[i], ci=0.90, ax=axes[i + 3])

fig.suptitle("TVP Coefficients and Significance", fontsize=14)
fig.tight_layout()
fig.savefig("combined_tvp.pdf", bbox_inches="tight")
```

---

## Related pages

- [TVP user guide](../user-guide/advanced/tvp.md) — model specification, random-walk
  vs AR(1) coefficients, identification, MLE vs Bayesian estimation
- [Tutorial: TVP — Time-Varying Parameters](../tutorials/tvp.md) — step-by-step
  Phillips curve analysis
- [Tutorial: Time-Varying CAPM](../tutorials/tvp-capm.md) — market beta estimation
- [Theory: State-Space and Kalman filter](../theory/state-space.md) — TVP as a
  state-space model
- [Diagnostics: Likelihood-ratio test](../diagnostics/likelihood-ratio.md) — formal
  test for coefficient constancy
- [Themes](themes.md) — control visual style across all plots
