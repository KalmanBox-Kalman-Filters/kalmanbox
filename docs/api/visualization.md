# Visualization API

`kalmanbox.visualization`

`kalmanbox.visualization` provides a complete plotting toolkit for state-space model output.
All functions return `matplotlib.figure.Figure` (or `matplotlib.axes.Axes`) and accept an
optional `ax` / `axes` keyword argument for embedding plots into existing figures. A global
theme can be set with [`set_theme()`](#set_theme) and applies to every subsequent plot call.

| Group | Functions |
|---|---|
| [State Plots](#state-plots) | [`plot_filtered_state`](#plot_filtered_state), [`plot_smoothed_state`](#plot_smoothed_state), [`plot_prediction`](#plot_prediction) |
| [Component Plots](#component-plots) | [`plot_components`](#plot_components), [`plot_trend`](#plot_trend), [`plot_seasonal`](#plot_seasonal), [`plot_cycle`](#plot_cycle) |
| [Innovation Diagnostics](#innovation-diagnostic-plots) | [`plot_innovations`](#plot_innovations), [`plot_qq`](#plot_qq), [`plot_acf_residuals`](#plot_acf_residuals), [`plot_diagnostic_panel`](#plot_diagnostic_panel) |
| [Filter Comparison](#filter-comparison-plots) | [`plot_filter_comparison`](#plot_filter_comparison), [`plot_nees`](#plot_nees) |
| [Factor Plots](#factor-plots) | [`plot_factors`](#plot_factors), [`plot_loadings`](#plot_loadings), [`plot_scree`](#plot_scree) |
| [TVP Plots](#tvp-plots) | [`plot_tvp_coefficients`](#plot_tvp_coefficients), [`plot_tvp_heatmap`](#plot_tvp_heatmap) |
| [Themes](#themes) | [`set_theme`](#set_theme), [`get_theme`](#get_theme), [`register_theme`](#register_theme), [`ThemeConfig`](#themeconfig) |

---

## State Plots

### `plot_filtered_state`

`kalmanbox.visualization.plot_filtered_state`

Plot filtered state means $a_{t|t}$ with marginal credible bands
$a_{t|t} \pm z_{\alpha/2} \, P_{t|t,ii}^{1/2}$ for each selected state
dimension. When `state_idx` is `None` all state dimensions are plotted in
separate panels stacked vertically on a shared figure.

```python
plot_filtered_state(
    result,
    state_idx=None,
    ci=0.95,
    ax=None,
    figsize=(10, 4),
    obs=True,
    color="#2166ac",
    band_alpha=0.2,
    title=None,
    ylabel=None,
    legend=True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FilterResult \| FitResult` | required | Output of `KalmanFilter.filter()` or any `.fit()` call. Must contain `a_filt` and `P_filt` arrays. |
| `state_idx` | `int \| list[int] \| None` | `None` | State dimension(s) to plot. A single integer draws one panel; a list draws one panel per element; `None` draws all dimensions. |
| `ci` | `float` | `0.95` | Credible interval coverage probability. Converted to a z-score as `scipy.stats.norm.ppf((1 + ci) / 2)`. Set to `None` to suppress the shaded band entirely. |
| `ax` | `Axes \| None` | `None` | Existing `matplotlib.axes.Axes` to draw into. Ignored when `state_idx` requests multiple panels; a new figure is always created in that case. |
| `figsize` | `tuple[int, int]` | `(10, 4)` | Figure size in inches `(width, height)`. Height is multiplied by the number of panels when multiple state dimensions are plotted. Only used when `ax` is `None`. |
| `obs` | `bool` | `True` | Overlay the observed series `y` as grey scatter points for a direct visual comparison. |
| `color` | `str` | `"#2166ac"` | Primary colour for the state mean line. The CI band is derived from the same colour at reduced opacity. |
| `band_alpha` | `float` | `0.2` | Opacity of the credible interval band (0 = fully transparent, 1 = opaque). |
| `title` | `str \| None` | `None` | Custom figure title. Defaults to `"Filtered State"` when `None`. |
| `ylabel` | `str \| None` | `None` | Y-axis label. Defaults to `"State"` or the state name from the model if available. |
| `legend` | `bool` | `True` | Display a legend identifying the filtered state, CI band, and observations. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
import numpy as np
from kalmanbox import LocalLevel
from kalmanbox.visualization import plot_filtered_state

# Nile river flow data (annual, 1871-1970)
nile = np.array([
    1120, 1160,  963, 1210, 1160, 1160,  813, 1230, 1370, 1140,
     995,  935, 1110,  994, 1020,  960, 1180,  799,  958, 1140,
    1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100,  774,  840,
     874,  694,  940,  833,  701,  916,  692, 1020, 1050,  969,
     831,  726,  456,  824,  702, 1120,  990,  950,  867, 1010,
     818,  801,  961,  878,  935, 1110,  994, 1020,  960, 1180,
     799,  958, 1140, 1100, 1210, 1150, 1250, 1260, 1220, 1030,
    1100,  774,  840,  874,  694,  940,  833,  701,  916,  692,
    1020, 1050,  969,  831,  726,  456,  824,  702, 1120,  990,
     950,  867, 1010,  818,  801,  961,  878,  935, 1110,  994,
])

model = LocalLevel(level_var=1469.1, obs_var=15099.0)
result = model.fit(nile)

fig = plot_filtered_state(
    result,
    ci=0.95,
    obs=True,
    color="#2166ac",
    title="Nile: Filtered Level",
    ylabel="Flow (10⁸ m³)",
)
fig.savefig("nile_filtered.png", dpi=150, bbox_inches="tight")
```

---

### `plot_smoothed_state`

`kalmanbox.visualization.plot_smoothed_state`

Plot smoothed state means $a_{t|n}$ with marginal credible bands derived
from the smoothed covariance $P_{t|n}$. Smoothed estimates use information
from the entire sample, producing narrower bands than the filtered output.

```python
plot_smoothed_state(
    result,
    state_idx=None,
    ci=0.95,
    ax=None,
    figsize=(10, 4),
    obs=True,
    color="#2166ac",
    band_alpha=0.2,
    title=None,
    ylabel=None,
    legend=True,
    compare_filtered=False,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FilterResult \| FitResult` | required | Must contain `a_smooth` and `P_smooth` produced by [`RTSSmoother`](core.md#rtssmoother) or a `.fit()` call with `smooth=True`. |
| `state_idx` | `int \| list[int] \| None` | `None` | State dimension(s) to plot. Same behaviour as [`plot_filtered_state`](#plot_filtered_state). |
| `ci` | `float` | `0.95` | Credible interval coverage. Set to `None` to suppress bands. |
| `ax` | `Axes \| None` | `None` | Existing axes. Ignored for multi-panel plots. |
| `figsize` | `tuple[int, int]` | `(10, 4)` | Figure size when a new figure is created. |
| `obs` | `bool` | `True` | Overlay observed series. |
| `color` | `str` | `"#2166ac"` | Primary line colour. |
| `band_alpha` | `float` | `0.2` | Opacity of the CI band. |
| `title` | `str \| None` | `None` | Custom title. Defaults to `"Smoothed State"`. |
| `ylabel` | `str \| None` | `None` | Y-axis label. |
| `legend` | `bool` | `True` | Show legend. |
| `compare_filtered` | `bool` | `False` | When `True`, overlay the filtered state mean as a dashed line for a direct filtered-vs-smoothed comparison. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox import LocalLinearTrend
from kalmanbox.visualization import plot_smoothed_state

model = LocalLinearTrend()
result = model.fit(nile, smooth=True)

fig = plot_smoothed_state(
    result,
    compare_filtered=True,
    title="Nile: Smoothed vs Filtered Level",
)
```

---

### `plot_prediction`

`kalmanbox.visualization.plot_prediction`

Fan-chart forecast plot combining the observed history with a
`steps`-ahead predictive distribution. Multiple shaded bands correspond
to distinct coverage levels supplied in `ci`. Bands are stacked from
wide (outermost) to narrow (innermost) in alternating opacity.

```python
plot_prediction(
    result,
    steps=12,
    ci=(0.5, 0.95),
    ax=None,
    figsize=(10, 5),
    history=None,
    color="#d7191c",
    title=None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. Must support `.forecast(steps)` to obtain predictive means and variances. |
| `steps` | `int` | `12` | Number of steps ahead to forecast. |
| `ci` | `tuple[float, ...]` | `(0.5, 0.95)` | One or more coverage probabilities for the shaded fan. Each level produces an additional band. Levels are sorted descending so the widest band is drawn first. |
| `ax` | `Axes \| None` | `None` | Existing axes to draw into. |
| `figsize` | `tuple` | `(10, 5)` | Figure size when a new figure is created. |
| `history` | `int \| None` | `None` | Number of historical observations to display before the forecast origin. `None` shows the full observed series. |
| `color` | `str` | `"#d7191c"` | Primary colour used for the forecast mean line and band fill. |
| `title` | `str \| None` | `None` | Custom title. Defaults to `"Forecast"`. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox import BasicStructuralModel
from kalmanbox.visualization import plot_prediction

model = BasicStructuralModel(seasonal_period=12)
result = model.fit(airline_passengers)

fig = plot_prediction(
    result,
    steps=24,
    ci=(0.5, 0.80, 0.95),
    history=36,
    color="#d7191c",
    title="Airline Passengers: 24-Month Forecast",
)
fig.savefig("airline_forecast.png", dpi=150, bbox_inches="tight")
```

---

## Component Plots

### `plot_components`

`kalmanbox.visualization.plot_components`

Multi-panel decomposition showing each structural component — trend,
slope, seasonal, cycle, irregular — in its own subplot. Only components
present in the fitted model are displayed; absent components are silently
skipped. Panels share a common x-axis when `share_x=True`.

```python
plot_components(
    result,
    components=None,
    figsize=None,
    share_x=True,
    ci=0.95,
    colors=None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Result from a structural model such as `BasicStructuralModel`, `UnobservedComponents`, or `LocalLevel`. Must expose a `.components` attribute mapping component names to `(mean, variance)` arrays. |
| `components` | `list[str] \| None` | `None` | Subset of components to plot. Valid names are `"trend"`, `"slope"`, `"seasonal"`, `"cycle"`, `"irregular"`. `None` plots all components detected in the result. |
| `figsize` | `tuple \| None` | `None` | Figure size. When `None`, height is computed automatically as `3 × n_panels` inches and width is `10` inches. |
| `share_x` | `bool` | `True` | Share the x-axis across all component panels for aligned zooming. |
| `ci` | `float` | `0.95` | Credible interval coverage for each component band. |
| `colors` | `dict[str, str] \| None` | `None` | Custom colour map keyed by component name, e.g. `{"trend": "#1a9641", "seasonal": "#d7191c"}`. Unmapped components fall back to the theme defaults. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
import numpy as np
from kalmanbox import BasicStructuralModel
from kalmanbox.visualization import plot_components

# Box-Jenkins airline passenger data (monthly, 1949-1960)
airline = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
])

model = BasicStructuralModel(seasonal_period=12)
result = model.fit(np.log(airline))

fig = plot_components(
    result,
    components=["trend", "seasonal", "cycle", "irregular"],
    colors={"trend": "#1a9641", "seasonal": "#d7191c"},
    ci=0.90,
)
fig.savefig("airline_components.png", dpi=150, bbox_inches="tight")
```

---

### `plot_trend`

`kalmanbox.visualization.plot_trend`

Plot the trend component only. This is a convenience wrapper around
[`plot_components`](#plot_components) with `components=["trend"]` and an
option to overlay the original observations for comparison.

```python
plot_trend(
    result,
    ci=0.95,
    ax=None,
    figsize=(10, 3),
    compare_obs=True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted structural model result containing a `"trend"` component. |
| `ci` | `float` | `0.95` | Credible interval coverage for the trend band. |
| `ax` | `Axes \| None` | `None` | Existing axes to draw into. |
| `figsize` | `tuple` | `(10, 3)` | Figure size when a new figure is created. |
| `compare_obs` | `bool` | `True` | Overlay the observed data as faint scatter points behind the trend line. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_trend

fig = plot_trend(result, compare_obs=True, ci=0.90)
```

---

### `plot_seasonal`

`kalmanbox.visualization.plot_seasonal`

Plot the seasonal component extracted from a structural model. When
`period` is provided, a second axes (or inset) shows the mean seasonal
pattern aggregated over one cycle, making it easy to inspect the
amplitude and shape of the seasonal variation.

```python
plot_seasonal(
    result,
    period=None,
    ci=0.95,
    ax=None,
    figsize=(10, 3),
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted structural model result containing a `"seasonal"` component. |
| `period` | `int \| None` | `None` | Seasonal period (e.g. `12` for monthly, `4` for quarterly). When provided, an additional seasonal-pattern panel is appended showing the mean effect over one cycle. If `None`, auto-detected from the model's `seasonal_period` attribute. |
| `ci` | `float` | `0.95` | Credible interval coverage. |
| `ax` | `Axes \| None` | `None` | Existing axes to draw into. When `period` is given and `ax` is not `None`, the period panel is skipped. |
| `figsize` | `tuple` | `(10, 3)` | Figure size when a new figure is created. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_seasonal

fig = plot_seasonal(result, period=12, ci=0.90)
```

---

### `plot_cycle`

`kalmanbox.visualization.plot_cycle`

Plot the stochastic cycle component of a structural model. This is a
convenience wrapper around [`plot_components`](#plot_components) with
`components=["cycle"]`.

```python
plot_cycle(
    result,
    ci=0.95,
    ax=None,
    figsize=(10, 3),
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted structural model result containing a `"cycle"` component. |
| `ci` | `float` | `0.95` | Credible interval coverage for the cycle band. |
| `ax` | `Axes \| None` | `None` | Existing axes to draw into. |
| `figsize` | `tuple` | `(10, 3)` | Figure size when a new figure is created. |

**Returns** `matplotlib.figure.Figure`

---

## Innovation Diagnostic Plots

### `plot_innovations`

`kalmanbox.visualization.plot_innovations`

Time-series plot of one-step-ahead prediction errors (innovations)
$v_t = y_t - Z_t a_{t|t-1}$. When `standardise=True` the innovations
are divided by their standard deviation $F_t^{1/2}$ so the resulting
series should resemble i.i.d. $\mathcal{N}(0,1)$ under a correctly
specified model.

```python
plot_innovations(
    result,
    standardise=True,
    ax=None,
    figsize=(10, 3),
    zero_line=True,
    color="#555555",
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult \| FilterResult` | required | Fitted or filtered result. Must contain `v` (innovations) and `F` (innovation variances). |
| `standardise` | `bool` | `True` | Divide each innovation by $F_t^{1/2}$ before plotting. Standardised innovations should lie within ±1.96 roughly 95 % of the time under Gaussianity. |
| `ax` | `Axes \| None` | `None` | Existing axes to draw into. |
| `figsize` | `tuple` | `(10, 3)` | Figure size when a new figure is created. |
| `zero_line` | `bool` | `True` | Draw a solid horizontal line at zero for visual reference. |
| `color` | `str` | `"#555555"` | Line colour for the innovation series. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_innovations

fig = plot_innovations(result, standardise=True)
```

---

### `plot_qq`

`kalmanbox.visualization.plot_qq`

Quantile-quantile plot of standardised innovations against the
$\mathcal{N}(0,1)$ theoretical quantiles. Significant departure from
the 45° reference line indicates non-Gaussianity in the residuals.

```python
plot_qq(
    result,
    ax=None,
    figsize=(5, 5),
    line=True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult \| FilterResult` | required | Must contain standardised innovations or raw `(v, F)` arrays from which they are computed. |
| `ax` | `Axes \| None` | `None` | Existing axes to draw into. |
| `figsize` | `tuple` | `(5, 5)` | Figure size when a new figure is created. Square figures are recommended. |
| `line` | `bool` | `True` | Draw the 45° reference diagonal. The line passes through the first and third quartiles of the theoretical distribution. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_qq

fig = plot_qq(result)
```

---

### `plot_acf_residuals`

`kalmanbox.visualization.plot_acf_residuals`

Side-by-side autocorrelation function (ACF) and partial autocorrelation
function (PACF) of the standardised innovations. Dashed significance
lines at ±1.96 / √n mark the approximate 95 % bounds under the null of
no autocorrelation.

```python
plot_acf_residuals(
    result,
    lags=20,
    ax=None,
    figsize=(10, 4),
    significance=0.05,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult \| FilterResult` | required | Result containing innovations. |
| `lags` | `int` | `20` | Maximum lag to display on both ACF and PACF panels. |
| `ax` | `list[Axes] \| None` | `None` | List of exactly two `Axes` objects: `ax[0]` for the ACF panel, `ax[1]` for the PACF panel. When `None` a new figure with two side-by-side panels is created. |
| `figsize` | `tuple` | `(10, 4)` | Figure size when a new figure is created. |
| `significance` | `float` | `0.05` | Significance level for the dashed threshold lines. Thresholds are drawn at `±scipy.stats.norm.ppf(1 - significance/2) / sqrt(n)`. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_acf_residuals

fig = plot_acf_residuals(result, lags=30)
```

---

### `plot_diagnostic_panel`

`kalmanbox.visualization.plot_diagnostic_panel`

4-panel composite diagnostic figure arranged in a 2 × 2 grid:

- **Top-left** — standardised innovations over time
- **Top-right** — ACF of standardised innovations
- **Bottom-left** — Normal Q-Q plot
- **Bottom-right** — squared standardised innovations (ARCH / heteroskedasticity check)

When `test_results` is provided, p-values from normality and
independence tests are annotated in the corner of the relevant panels.

```python
plot_diagnostic_panel(
    result,
    figsize=(12, 8),
    title=None,
    test_results=None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult \| FilterResult` | required | Fitted or filtered result with innovations and innovation variances. |
| `figsize` | `tuple` | `(12, 8)` | Overall figure size. |
| `title` | `str \| None` | `None` | Super-title placed above all panels. Defaults to `"Residual Diagnostics"`. |
| `test_results` | `dict \| None` | `None` | Optional dictionary of test statistics and p-values to annotate. Expected keys: `"normality"` (Jarque-Bera), `"independence"` (Ljung-Box), `"heteroskedasticity"` (Goldfeld-Quandt). Values are `(statistic, p_value)` tuples. |

**Returns** `matplotlib.figure.Figure`

!!! tip "Combining with DiagnosticSuite"

    Pass the output of `DiagnosticSuite(result).run()` directly as
    `test_results` to annotate all four panels automatically.

**Example**

```python
from kalmanbox.diagnostics import DiagnosticSuite
from kalmanbox.visualization import plot_diagnostic_panel

suite = DiagnosticSuite(result)
tests = suite.run()

fig = plot_diagnostic_panel(
    result,
    title="BSM Airline — Residual Diagnostics",
    test_results=tests,
)
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
```

---

## Filter Comparison Plots

### `plot_filter_comparison`

`kalmanbox.visualization.plot_filter_comparison`

Overlay filtered state estimates from multiple filter objects on the
same axes to facilitate direct visual comparison of accuracy and
uncertainty. Each filter's result is labelled by its key in
`results_dict`.

```python
plot_filter_comparison(
    results_dict,
    state_idx=0,
    ci=0.95,
    figsize=(12, 5),
    obs=True,
    colors=None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results_dict` | `dict[str, FilterResult]` | required | Mapping from filter label to `FilterResult`. Example: `{"KF": kf_res, "UKF": ukf_res, "EnKF": enkf_res}`. |
| `state_idx` | `int` | `0` | Which state dimension to plot on the shared axes. |
| `ci` | `float` | `0.95` | Credible interval coverage drawn for each filter. |
| `figsize` | `tuple` | `(12, 5)` | Figure size. |
| `obs` | `bool` | `True` | Overlay the true observations as grey scatter for reference. |
| `colors` | `list[str] \| None` | `None` | Colours assigned to filters in the iteration order of `results_dict`. When `None` the current theme's palette is used. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation, KalmanFilter
from kalmanbox.filters import UnscentedKalmanFilter, EnsembleKalmanFilter
from kalmanbox.visualization import plot_filter_comparison

# Shared linear model: KF is exact; UKF/EnKF are approximate
ss = StateSpaceRepresentation(
    Z=np.array([[1.0, 0.0]]),
    T=np.array([[1.0, 1.0], [0.0, 1.0]]),
    R=np.eye(2),
    H=np.array([[4.0]]),
    Q=np.diag([0.1, 0.01]),
)

rng = np.random.default_rng(0)
n = 200
a_true = np.zeros((n + 1, 2))
y = np.zeros((n, 1))
for t in range(n):
    noise = rng.multivariate_normal(np.zeros(2), ss.Q)
    a_true[t+1] = ss.T @ a_true[t] + noise
    y[t] = ss.Z @ a_true[t+1] + rng.normal(0, 2.0)

kf  = KalmanFilter(ss)
kf_res = kf.filter(y)

def f(a, t): return ss.T @ a
def h(a, t): return ss.Z @ a
ukf = UnscentedKalmanFilter(2, 1, f, h, ss.Q, ss.H)
ukf_res = ukf.filter(y, np.zeros(2), np.eye(2))

fig = plot_filter_comparison(
    {"KF (exact)": kf_res, "UKF": ukf_res},
    state_idx=0,
    ci=0.95,
    obs=True,
)
```

---

### `plot_nees`

`kalmanbox.visualization.plot_nees`

Plot the Normalised Estimation Error Squared (NEES) over time:

$$
\varepsilon_t = (a_t^* - a_{t|t})' P_{t|t}^{-1} (a_t^* - a_{t|t})
$$

where $a_t^*$ is the true state. Under a correctly specified filter,
$\varepsilon_t \sim \chi^2(m)$ at each step. The 95 % chi-squared bounds
are drawn as dashed horizontal lines for the specified significance level.
Persistent exceedance indicates filter inconsistency.

```python
plot_nees(
    filter_result,
    a_true,
    significance=0.05,
    figsize=(10, 4),
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter_result` | `FilterResult` | required | Filtered result containing `a_filt` and `P_filt`. |
| `a_true` | `np.ndarray` | required | True state sequence. Shape `(n, m)`. Must align with the filter's output time axis. |
| `significance` | `float` | `0.05` | Significance level for the chi-squared bounds. Bounds are drawn at the `significance/2` and `1 - significance/2` quantiles of the $\chi^2(m)$ distribution. |
| `figsize` | `tuple` | `(10, 4)` | Figure size. |

**Returns** `matplotlib.figure.Figure`

!!! note

    `plot_nees` requires the ground-truth state sequence and is therefore
    only applicable to simulation studies or datasets where the true
    latent state is known (e.g. GPS ground truth in tracking experiments).

**Example**

```python
from kalmanbox.visualization import plot_nees

fig = plot_nees(kf_res, a_true=a_true[1:], significance=0.05)
```

---

## Factor Plots

### `plot_factors`

`kalmanbox.visualization.plot_factors`

Plot estimated common factors $f_t$ extracted by a
`DynamicFactorModel`, each in its own panel with marginal 95 % credible
bands. Factors can optionally be standardised to unit variance before
plotting, which makes their relative dynamics comparable regardless of
identification normalisation.

```python
plot_factors(
    result,
    factor_idx=None,
    ci=0.95,
    figsize=None,
    standardise=True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Result from `DynamicFactorModel.fit()`. Must expose `factors` (mean trajectories) and `factor_cov` (variance trajectories). |
| `factor_idx` | `int \| list[int] \| None` | `None` | Which factor(s) to plot. `None` plots all `r` factors in separate stacked panels. |
| `ci` | `float` | `0.95` | Credible interval coverage for each factor band. |
| `figsize` | `tuple \| None` | `None` | Figure size. Defaults to `(10, 3 × r)` where `r` is the number of panels. |
| `standardise` | `bool` | `True` | Divide each factor by its sample standard deviation before plotting so that all factors share a comparable scale. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
import numpy as np
from kalmanbox import DynamicFactorModel
from kalmanbox.visualization import plot_factors

rng = np.random.default_rng(1)
n, N = 200, 10
f_true = rng.standard_normal((n, 2))          # 2 latent factors
Lambda = rng.standard_normal((N, 2))           # loadings (N × r)
y = f_true @ Lambda.T + rng.standard_normal((n, N)) * 0.5

model = DynamicFactorModel(n_factors=2)
result = model.fit(y)

fig = plot_factors(result, standardise=True, ci=0.90)
fig.savefig("dfm_factors.png", dpi=150, bbox_inches="tight")
```

---

### `plot_loadings`

`kalmanbox.visualization.plot_loadings`

Heatmap of the factor loading matrix $\Lambda$ (shape $N \times r$) where
$N$ is the number of observed series and $r$ is the number of factors.
Positive loadings are red, negative loadings are blue (under the default
`"RdBu_r"` colormap). Cell values are annotated when `annotate=True`.

```python
plot_loadings(
    result,
    figsize=(8, 5),
    annotate=True,
    cmap="RdBu_r",
    series_labels=None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Result from `DynamicFactorModel.fit()`. Must expose a `.loadings` attribute of shape `(N, r)`. |
| `figsize` | `tuple` | `(8, 5)` | Figure size. |
| `annotate` | `bool` | `True` | Overlay the numerical loading value in each heatmap cell, formatted to 2 decimal places. |
| `cmap` | `str` | `"RdBu_r"` | Matplotlib colormap name. Diverging maps centred at zero are recommended. |
| `series_labels` | `list[str] \| None` | `None` | Labels for the N observed series (y-axis tick labels). Defaults to `["y0", "y1", ...]`. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_loadings

fig = plot_loadings(
    result,
    series_labels=[f"Series {i+1}" for i in range(N)],
    annotate=True,
)
```

---

### `plot_scree`

`kalmanbox.visualization.plot_scree`

Scree plot of eigenvalues and explained variance ratio from the PCA
initialisation step of a `DynamicFactorModel`. A cumulative variance
line is overlaid when `cumulative=True`, aiding in the selection of the
number of factors.

```python
plot_scree(
    result,
    figsize=(6, 4),
    cumulative=True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Result from `DynamicFactorModel.fit()`. Must expose `.pca_eigenvalues` and `.pca_explained_variance_ratio` from the initialisation PCA. |
| `figsize` | `tuple` | `(6, 4)` | Figure size. |
| `cumulative` | `bool` | `True` | Overlay a secondary y-axis showing cumulative explained variance. A dashed horizontal line is drawn at 90 %. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_scree

fig = plot_scree(result, cumulative=True)
```

---

## TVP Plots

### `plot_tvp_coefficients`

`kalmanbox.visualization.plot_tvp_coefficients`

Plot time-varying regression coefficients $\beta_t$ estimated by a
`TimeVaryingParameters` model, with marginal credible bands. Each
selected coefficient is drawn in its own panel.

```python
plot_tvp_coefficients(
    result,
    param_idx=None,
    ci=0.95,
    figsize=None,
    zero_line=True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Result from `TimeVaryingParameters.fit()`. Must expose `beta` (coefficient trajectories, shape `(n, k)`) and `beta_cov` (marginal variances, shape `(n, k)`). |
| `param_idx` | `int \| list[int] \| None` | `None` | Which coefficient(s) to plot by column index. `None` plots all `k` coefficients in separate stacked panels. |
| `ci` | `float` | `0.95` | Credible interval coverage for each coefficient band. |
| `figsize` | `tuple \| None` | `None` | Figure size. Defaults to `(10, 3 × k)` where `k` is the number of panels. |
| `zero_line` | `bool` | `True` | Draw a dashed horizontal reference line at $\beta = 0$ in each panel to help judge statistical significance. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
import numpy as np
from kalmanbox import TimeVaryingParameters
from kalmanbox.visualization import plot_tvp_coefficients

rng = np.random.default_rng(2)
n = 300
beta_true = np.cumsum(rng.standard_normal((n, 2)) * 0.02, axis=0)
X = rng.standard_normal((n, 2))
y = (X * beta_true).sum(axis=1) + rng.standard_normal(n) * 0.5

model = TimeVaryingParameters()
result = model.fit(y, X)

fig = plot_tvp_coefficients(
    result,
    param_idx=[0, 1],
    ci=0.90,
    zero_line=True,
)
fig.savefig("tvp_coefficients.png", dpi=150, bbox_inches="tight")
```

---

### `plot_tvp_heatmap`

`kalmanbox.visualization.plot_tvp_heatmap`

Heatmap showing the evolution of all TVP coefficients over time. Time
runs along the x-axis and each row corresponds to one regression
coefficient $\beta_{k,t}$. This compact view is useful when the number
of coefficients is large and individual line plots become cluttered.

```python
plot_tvp_heatmap(
    result,
    figsize=(10, 6),
    cmap="RdBu_r",
    symmetric=True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Result from `TimeVaryingParameters.fit()`. Must expose `beta` of shape `(n, k)`. |
| `figsize` | `tuple` | `(10, 6)` | Figure size. |
| `cmap` | `str` | `"RdBu_r"` | Matplotlib colormap. Diverging maps centred at zero are recommended. |
| `symmetric` | `bool` | `True` | When `True`, the colormap is centred at zero by setting `vmin = -vmax`. When `False`, the range spans the actual data minimum and maximum. |

**Returns** `matplotlib.figure.Figure`

**Example**

```python
from kalmanbox.visualization import plot_tvp_heatmap

fig = plot_tvp_heatmap(result, symmetric=True)
```

---

## Themes

### `set_theme`

`kalmanbox.visualization.set_theme`

Set the global plot theme applied to all subsequent `kalmanbox.visualization`
function calls. The theme modifies `matplotlib` `rcParams` and is stored
in module-level state; it persists until `set_theme` is called again.

```python
set_theme(name: str) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of the theme to activate. Must be one of the built-in themes listed below or a name previously registered via [`register_theme`](#register_theme). |

**Returns** `None`. Modifies the global theme state as a side effect.

**Available built-in themes**

| Name | Description |
|---|---|
| `"default"` | Clean white background, blue accent colours, 100 dpi. Suitable for interactive notebooks and exploratory analysis. |
| `"publication"` | Grayscale-safe palette, serif fonts, 300 dpi, tight figure margins. Produces figures suitable for journal submission. |
| `"dark"` | Dark grey background with neon accent colours. Designed for dashboards, slide decks, and dark-mode notebooks. |
| `"minimal"` | No grid lines, thin lines, muted palette with generous whitespace. Suitable for presentations and infographics. |

**Example**

```python
from kalmanbox.visualization import set_theme, plot_components

set_theme("publication")
fig = plot_components(result)
fig.savefig("figure_1.pdf", bbox_inches="tight")

set_theme("default")   # restore for interactive use
```

---

### `get_theme`

`kalmanbox.visualization.get_theme`

Return the name of the currently active theme.

```python
get_theme() -> str
```

**Parameters** — none.

**Returns** `str` — name of the active theme (e.g. `"default"`).

**Example**

```python
from kalmanbox.visualization import get_theme

print(get_theme())  # "default"
```

---

### `register_theme`

`kalmanbox.visualization.register_theme`

Register a custom theme so that it can subsequently be activated by
name with [`set_theme`](#set_theme). Custom themes are stored in
module-level state for the duration of the Python session.

```python
register_theme(name: str, config: ThemeConfig) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique name for the custom theme. Raises `ValueError` if the name collides with a built-in theme. |
| `config` | `ThemeConfig` | required | `ThemeConfig` dataclass instance specifying all style properties. See [`ThemeConfig`](#themeconfig) for the full field list. |

**Returns** `None`.

**Example**

```python
from kalmanbox.visualization import register_theme, set_theme, ThemeConfig

corporate = ThemeConfig(
    bg_color="#f5f5f5",
    primary_color="#003366",
    secondary_color="#cc0000",
    font_family="Arial",
    font_size=12,
    dpi=150,
)

register_theme("corporate", corporate)
set_theme("corporate")
```

---

### `ThemeConfig`

`kalmanbox.visualization.ThemeConfig`

Dataclass that fully specifies the visual properties of a kalmanbox plot
theme. All fields have defaults matching the `"default"` built-in theme.

```python
ThemeConfig(
    bg_color: str = "#ffffff",
    text_color: str = "#222222",
    primary_color: str = "#2166ac",
    secondary_color: str = "#d7191c",
    grid_color: str = "#dddddd",
    font_family: str = "sans-serif",
    font_size: int = 11,
    dpi: int = 100,
    line_width: float = 1.5,
    fig_facecolor: str = "#ffffff",
    ax_facecolor: str = "#ffffff",
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bg_color` | `str` | `"#ffffff"` | Background colour applied to the figure and axes faces when `fig_facecolor` and `ax_facecolor` are not overridden. |
| `text_color` | `str` | `"#222222"` | Default text colour for axis labels, tick labels, and titles. |
| `primary_color` | `str` | `"#2166ac"` | Primary accent colour used for the first plotted series, state mean lines, and CI bands. |
| `secondary_color` | `str` | `"#d7191c"` | Secondary accent colour used for the second plotted series, forecast lines, and comparison overlays. |
| `grid_color` | `str` | `"#dddddd"` | Colour of major grid lines. Grid lines are drawn at half this opacity for minor ticks. |
| `font_family` | `str` | `"sans-serif"` | Matplotlib font family string. Pass `"serif"` for publication-style figures. |
| `font_size` | `int` | `11` | Base font size in points. Titles are scaled to `font_size + 2`. |
| `dpi` | `int` | `100` | Default resolution in dots per inch used when saving figures. |
| `line_width` | `float` | `1.5` | Default line width in points for all plotted series. |
| `fig_facecolor` | `str` | `"#ffffff"` | Figure background colour (passed to `fig.patch.set_facecolor`). |
| `ax_facecolor` | `str` | `"#ffffff"` | Axes background colour (passed to `ax.set_facecolor`). |

---

## Complete Visualization Example

The following end-to-end example fits a `BasicStructuralModel` to the
Box-Jenkins airline passenger data, inspects the structural components,
runs the full diagnostic panel, switches to the publication theme, and
saves publication-ready figures to disk.

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import BasicStructuralModel
from kalmanbox.diagnostics import DiagnosticSuite
from kalmanbox.visualization import (
    set_theme,
    plot_components,
    plot_diagnostic_panel,
    plot_prediction,
)

# ── 1. Data ──────────────────────────────────────────────────────────────
airline = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
])

# Log-transform to stabilise variance
log_airline = np.log(airline)

# ── 2. Fit BSM ────────────────────────────────────────────────────────────
model = BasicStructuralModel(seasonal_period=12)
result = model.fit(log_airline, smooth=True)

print(f"Log-likelihood : {result.loglikelihood:.4f}")
print(f"AIC            : {result.aic:.4f}")
print(f"BIC            : {result.bic:.4f}")

# ── 3. Component decomposition (interactive / notebook) ──────────────────
fig_comp = plot_components(
    result,
    components=["trend", "seasonal", "irregular"],
    colors={"trend": "#1a9641", "seasonal": "#d7191c"},
    ci=0.95,
)
fig_comp.suptitle("Airline Passengers — BSM Decomposition (log scale)", y=1.02)
plt.tight_layout()
plt.show()

# ── 4. Diagnostic panel ───────────────────────────────────────────────────
suite = DiagnosticSuite(result)
test_results = suite.run()   # returns dict with normality, independence, etc.

fig_diag = plot_diagnostic_panel(
    result,
    title="Airline BSM — Residual Diagnostics",
    test_results=test_results,
)
plt.show()

# ── 5. Switch to publication theme and save ───────────────────────────────
set_theme("publication")

fig_comp_pub = plot_components(
    result,
    components=["trend", "seasonal", "irregular"],
    ci=0.95,
)
fig_comp_pub.savefig(
    "airline_components_pub.pdf",
    dpi=300,
    bbox_inches="tight",
)

fig_diag_pub = plot_diagnostic_panel(
    result,
    title="Airline BSM — Residual Diagnostics",
    test_results=test_results,
)
fig_diag_pub.savefig(
    "airline_diagnostics_pub.pdf",
    dpi=300,
    bbox_inches="tight",
)

# ── 6. Forecast fan chart ─────────────────────────────────────────────────
fig_fc = plot_prediction(
    result,
    steps=24,
    ci=(0.50, 0.80, 0.95),
    history=36,
    color="#d7191c",
    title="Airline Passengers — 24-Month Forecast (log scale)",
)
fig_fc.savefig(
    "airline_forecast_pub.pdf",
    dpi=300,
    bbox_inches="tight",
)

print("Figures saved: airline_components_pub.pdf, airline_diagnostics_pub.pdf, airline_forecast_pub.pdf")
```

---

## See Also

- [User Guide: Visualization Overview](../user-guide/visualization/index.md)
- [User Guide: State Plots](../user-guide/visualization/state-plots.md)
- [User Guide: Component Plots](../user-guide/visualization/component-plots.md)
- [User Guide: Innovation Plots](../user-guide/visualization/innovation-plots.md)
- [User Guide: Filter Comparison Plots](../user-guide/visualization/filter-plots.md)
- [User Guide: Factor Plots](../user-guide/visualization/factor-plots.md)
- [User Guide: TVP Plots](../user-guide/visualization/tvp-plots.md)
- [User Guide: Themes](../user-guide/visualization/themes.md)
- [Tutorials: Complete Workflow](../tutorials/complete-workflow.md)
- [API: Core (KalmanFilter, RTSSmoother)](core.md)
- [API: Structural Models](structural.md)
- [API: Advanced Models](advanced.md)
