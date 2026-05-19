# State Plots

The `kalmanbox.visualization` module provides five dedicated functions for
visualising state estimates produced by the Kalman filter and the
RTS smoother. Understanding which to use requires distinguishing between
the two types of estimate:

- **Filtered estimate** $a_{t|t}$ — a *one-sided*, real-time estimate
  conditioned only on observations up to and including time $t$, i.e.
  $\mathbb{E}[\alpha_t \mid y_{1:t}]$.  It is the natural output when
  the model runs online.

- **Smoothed estimate** $a_{t|n}$ — a *two-sided* estimate conditioned
  on the full sample $y_{1:n}$, i.e.
  $\mathbb{E}[\alpha_t \mid y_{1:n}]$.  The RTS backward pass revises
  every time step in the light of data that had not yet arrived during
  the forward filter.  Smoothed estimates are the standard for
  retrospective analysis.

The corresponding posterior variances are denoted $P_{t|t}$ (filtered)
and $P_{t|n}$ (smoothed).  Because $P_{t|n} \le P_{t|t}$ at every $t$,
smoothed confidence bands are always at least as tight as filtered bands,
often substantially so at the beginning and end of the sample.

!!! info "Which estimate should I use?"
    Use **filtered** estimates for real-time or recursive applications
    (signal detection, online learning, sequential forecasting).  Use
    **smoothed** estimates whenever you want the best-possible retrospective
    picture of the latent states — trend/cycle decomposition, parameter
    identification, publication charts.

---

## Confidence bands

All state-plot functions shade a confidence band around the point
estimate.  The half-width of the band is

$$
w_t = z_{\alpha/2} \sqrt{P_{t|t}}
\qquad \text{(filtered)}
$$

$$
w_t = z_{\alpha/2} \sqrt{P_{t|n}}
\qquad \text{(smoothed)}
$$

where $z_{\alpha/2}$ is the upper $\alpha/2$ quantile of the standard
normal distribution.

Two sigma conventions are supported:

| `sigma` | $z$ value | Nominal coverage |
|---------|-----------|-----------------|
| `1`     | 1.000     | 68.3 %           |
| `2`     | 1.960     | 95.0 %           |

The `alpha` parameter provides direct control over the coverage level
via the inverse-normal quantile, independently of `sigma`:

| `alpha` | $z_{\alpha/2}$ | Coverage |
|---------|----------------|----------|
| `0.32`  | 1.000          | 68 %     |
| `0.10`  | 1.645          | 90 %     |
| `0.05`  | 1.960          | 95 %     |
| `0.01`  | 2.576          | 99 %     |

!!! note "Precedence"
    When both `sigma` and `alpha` are supplied, `alpha` takes
    precedence.  Pass only one of them to avoid confusion.

---

## `plot_filtered_state()`

Plot the **filtered** state mean $a_{t|t}$ together with its marginal
confidence band at each time step.

### Signature

```python
from kalmanbox.visualization import plot_filtered_state

def plot_filtered_state(
    results: FilterResults | SmootherResults,
    component: str | list[str] | None = None,
    alpha: float = 0.05,
    sigma: int | None = None,
    figsize: tuple[float, float] | None = None,
    color: str = "C0",
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show_data: bool = False,
) -> matplotlib.figure.Figure:
    ...
```

### Parameters

| Parameter   | Type                          | Default | Description |
|-------------|-------------------------------|---------|-------------|
| `results`   | `FilterResults` or `SmootherResults` | — | Object returned by `.filter()` or `.smooth()`. Must expose `.filtered_states` and `.filtered_states_cov`. |
| `component` | `str`, `list[str]`, or `None` | `None`  | State component(s) to plot. `None` plots all components. For univariate models a single panel is shown; for multivariate models one panel per component. |
| `alpha`     | `float`                       | `0.05`  | Significance level for the confidence band.  `0.05` yields a 95 % band. |
| `sigma`     | `int` or `None`               | `None`  | If `1` or `2`, overrides `alpha` with the corresponding $\pm 1\sigma$ or $\pm 2\sigma$ (95 %) band. |
| `figsize`   | `tuple[float, float]` or `None` | `None` | Figure size in inches `(width, height)`.  When `None` the size is chosen automatically based on the number of panels. |
| `color`     | `str`                         | `"C0"`  | Matplotlib colour string for the state line and shaded band. |
| `title`     | `str` or `None`               | `None`  | Figure-level title.  When `None`, a default title is generated from the model name. |
| `ax`        | `matplotlib.axes.Axes` or `None` | `None` | Existing axes to draw into.  Ignored when `component` requests multiple panels. |
| `show_data` | `bool`                        | `False` | If `True`, overlays the observed series $y_t$ as grey dots. |

### Returns

`matplotlib.figure.Figure` — the figure object.  Call `.savefig()` or
pass to `IPython.display.display()` as needed.

### Visual output

The function produces one panel per requested state component.  Each
panel contains:

1. A solid line for the filtered mean $a_{t|t}$.
2. A filled band between $a_{t|t} - z_{\alpha/2}\sqrt{P_{t|t}}$ and
   $a_{t|t} + z_{\alpha/2}\sqrt{P_{t|t}}$.
3. (Optional) the observed data $y_t$ as small grey scatter points.

Bands are typically widest near $t = 1$ when the diffuse initialisation
has not yet been overcome by data, and stabilise after roughly $m$ to
$2m$ observations (where $m$ is the state dimension).

### Example

```python
import numpy as np
from kalmanbox import KalmanFilter
from kalmanbox.visualization import plot_filtered_state

rng = np.random.default_rng(0)
n = 120
level = np.cumsum(rng.normal(0, 0.3, n))
y = level + rng.normal(0, 0.5, n)

# local level: Z=1, T=1, Q=sigma_eta^2, H=sigma_eps^2
kf = KalmanFilter(
    transition_matrices=[[1.0]],
    observation_matrices=[[1.0]],
    transition_covariance=[[0.09]],
    observation_covariance=[[0.25]],
    initial_state_mean=[0.0],
    initial_state_covariance=[[10.0]],
)
results = kf.filter(y)

fig = plot_filtered_state(
    results,
    alpha=0.05,
    color="steelblue",
    title="Local Level — filtered state $a_{t|t}$",
    show_data=True,
)
fig.savefig("filtered_state.png", dpi=150)
```

!!! tip "Interpreting band width"
    The filtered band widens whenever $y_t$ is missing.  See
    [Missing data](../user-guide/kalman/missing-data.md) for how
    `kalmanbox` handles gaps.

---

## `plot_smoothed_state()`

Plot the **smoothed** state mean $a_{t|n}$ together with its marginal
confidence band.  The RTS smoother must have been run first (call
`.smooth()` instead of `.filter()`).

### Signature

```python
from kalmanbox.visualization import plot_smoothed_state

def plot_smoothed_state(
    results: SmootherResults,
    component: str | list[str] | None = None,
    alpha: float = 0.05,
    sigma: int | None = None,
    figsize: tuple[float, float] | None = None,
    color: str = "C1",
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show_data: bool = False,
) -> matplotlib.figure.Figure:
    ...
```

### Parameters

| Parameter   | Type                          | Default | Description |
|-------------|-------------------------------|---------|-------------|
| `results`   | `SmootherResults`             | —       | Object returned by `.smooth()`.  Must expose `.smoothed_states` and `.smoothed_states_cov`. |
| `component` | `str`, `list[str]`, or `None` | `None`  | State component(s) to plot. |
| `alpha`     | `float`                       | `0.05`  | Significance level; `0.05` → 95 % band. |
| `sigma`     | `int` or `None`               | `None`  | `1` or `2` to use fixed $\sigma$ convention instead of `alpha`. |
| `figsize`   | `tuple[float, float]` or `None` | `None` | Figure size in inches. |
| `color`     | `str`                         | `"C1"`  | Colour for the state line and band.  Default is orange. |
| `title`     | `str` or `None`               | `None`  | Figure-level title. |
| `ax`        | `matplotlib.axes.Axes` or `None` | `None` | Existing axes to draw into (single-panel only). |
| `show_data` | `bool`                        | `False` | Overlay observed data $y_t$ as grey dots. |

### Returns

`matplotlib.figure.Figure`

### Visual output

Visually identical to `plot_filtered_state` but uses $a_{t|n}$ and
$P_{t|n}$.  In practice the smoothed bands are noticeably narrower in
the interior of the sample.  The endpoint variance $P_{n|n}$ equals the
filtered variance (the smoother has no backward information at the final
observation), so the bands converge at the right margin.

### Example

```python
from kalmanbox.visualization import plot_smoothed_state

# Fit BSM and smooth
from kalmanbox.models import BasicStructuralModel

bsm = BasicStructuralModel(
    endog=y,
    seasonal_periods=12,
    stochastic_level=True,
    stochastic_slope=True,
    stochastic_seasonal=True,
)
results = bsm.fit().smooth()

fig = plot_smoothed_state(
    results,
    component="level",          # plot only the level component
    alpha=0.05,
    color="darkorange",
    show_data=True,
    title="BSM — smoothed level $a_{t|n}^{(\mu)}$",
)
```

!!! note "Smoother required"
    Calling `plot_smoothed_state` on a `FilterResults` object (i.e. one
    that has not been smoothed) raises `SmootherNotRunError`.  Run
    `.smooth()` first.

---

## `plot_filtered_vs_smoothed()`

Side-by-side comparison of the filtered estimate $a_{t|t}$ and the
smoothed estimate $a_{t|n}$ on the same axes, making the **information
gain** of the backward pass immediately visible.

### Signature

```python
from kalmanbox.visualization import plot_filtered_vs_smoothed

def plot_filtered_vs_smoothed(
    results: SmootherResults,
    component: str | list[str] | None = None,
    alpha: float = 0.05,
    sigma: int | None = None,
    figsize: tuple[float, float] | None = None,
    color: tuple[str, str] = ("C0", "C1"),
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show_data: bool = False,
) -> matplotlib.figure.Figure:
    ...
```

### Parameters

| Parameter   | Type                          | Default            | Description |
|-------------|-------------------------------|--------------------|-------------|
| `results`   | `SmootherResults`             | —                  | Must contain both filtered and smoothed arrays. |
| `component` | `str`, `list[str]`, or `None` | `None`             | Component(s) to compare.  Multi-component → one column of panels per component. |
| `alpha`     | `float`                       | `0.05`             | Significance level for both bands. |
| `sigma`     | `int` or `None`               | `None`             | Fixed $\sigma$ convention. |
| `figsize`   | `tuple[float, float]` or `None` | `None`           | Figure size.  Auto-scales with number of panels. |
| `color`     | `tuple[str, str]`             | `("C0", "C1")`     | Two-element tuple: `(filtered_color, smoothed_color)`. |
| `title`     | `str` or `None`               | `None`             | Figure-level title. |
| `ax`        | `matplotlib.axes.Axes` or `None` | `None`          | Existing axes (single-panel only). |
| `show_data` | `bool`                        | `False`            | Overlay observed $y_t$ as grey dots. |

### Returns

`matplotlib.figure.Figure`

### Visual output

Each panel shows:

- **Filtered** mean as a solid blue line with a lightly shaded band.
- **Smoothed** mean as a dashed orange line with a more opaque band.
- Both bands share the same $\alpha$ level so the variance reduction is
  visually apparent: the orange (smoothed) band is contained within the
  blue (filtered) band except near the endpoints.

For multi-component models the function lays out one subplot per
component, each with the same dual-trace layout.

### Example

```python
from kalmanbox.visualization import plot_filtered_vs_smoothed

results = bsm.fit().smooth()   # smoother must be run

fig = plot_filtered_vs_smoothed(
    results,
    component="level",
    alpha=0.05,
    color=("royalblue", "tomato"),
    show_data=True,
    title="Filtered vs Smoothed — BSM level",
)
```

!!! tip "Reading the gap"
    The vertical distance between the two means shows how much the
    backward pass **revises** the filter's online estimate.  Large
    revisions indicate that the signal-to-noise ratio is low and that
    observations far from $t$ carry substantial information about
    $\alpha_t$.

---

## `plot_state_trajectory()`

Plot a 2-D **phase-space trajectory** of two state components, where
the horizontal axis is one component and the vertical axis is another.
This is the standard tool for visualising the joint evolution of two
state dimensions — for example, the level–slope phase plane of a
Local Linear Trend or BSM model.

### Signature

```python
from kalmanbox.visualization import plot_state_trajectory

def plot_state_trajectory(
    results: FilterResults | SmootherResults,
    component: list[str] | list[int],
    alpha: float = 0.05,
    sigma: int | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    color: str = "C2",
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show_data: bool = False,
) -> matplotlib.figure.Figure:
    ...
```

### Parameters

| Parameter   | Type                          | Default     | Description |
|-------------|-------------------------------|-------------|-------------|
| `results`   | `FilterResults` or `SmootherResults` | — | Filter or smoother output. |
| `component` | `list[str]` or `list[int]`    | —           | **Required.** Exactly two component names or integer indices, e.g. `["level", "slope"]` or `[0, 1]`. The first element maps to the x-axis, the second to the y-axis. |
| `alpha`     | `float`                       | `0.05`      | Significance level for the ellipsoidal confidence regions. |
| `sigma`     | `int` or `None`               | `None`      | Fixed $\sigma$ override. |
| `figsize`   | `tuple[float, float]`         | `(6.0, 6.0)` | Figure size in inches (square default for phase plots). |
| `color`     | `str`                         | `"C2"`      | Colour for the trajectory line and confidence ellipses. |
| `title`     | `str` or `None`               | `None`      | Figure title. |
| `ax`        | `matplotlib.axes.Axes` or `None` | `None`   | Existing axes to draw into. |
| `show_data` | `bool`                        | `False`     | Not applicable to phase plots; silently ignored. |

### Returns

`matplotlib.figure.Figure`

### Visual output

The figure shows:

1. A **trajectory line** connecting consecutive state-mean points
   $(a_{t|t}^{(1)},\, a_{t|t}^{(2)})$ in chronological order, coloured
   from light (early) to dark (late) along a gradient so that direction
   of time is immediately apparent.
2. **Confidence ellipses** at the requested $\alpha$ level, drawn from
   the $2 \times 2$ sub-block of $P_{t|t}$ corresponding to the two
   selected components.
3. A **start marker** (circle) and **end marker** (diamond) at $t = 1$
   and $t = n$.
4. Axis labels set to the component names.

!!! note "Correlation structure"
    Phase plots expose the **correlation** between state components that
    marginal time-series plots hide.  A tightly elongated ellipse
    indicates that the two components are nearly linearly constrained;
    a circular ellipse indicates independence.

### Example

See [Example: BSM state trajectory](#example-bsm-state-trajectory) below.

---

## `plot_prediction()`

Plot **in-sample one-step-ahead predictions** $\hat y_{t|t-1}$ together
with prediction intervals and the observed data.  This function uses
the *filter* prediction step (not the smoother) and is the correct tool
for checking model fit and detecting outliers.

The predicted observation is

$$
\hat y_{t|t-1} = Z_t \, a_{t|t-1}
$$

and the prediction variance is

$$
F_t = Z_t P_{t|t-1} Z_t' + H_t
$$

so the prediction interval is $\hat y_{t|t-1} \pm z_{\alpha/2} \sqrt{F_t}$.

### Signature

```python
from kalmanbox.visualization import plot_prediction

def plot_prediction(
    results: FilterResults | SmootherResults,
    component: str | list[str] | None = None,
    alpha: float = 0.05,
    sigma: int | None = None,
    figsize: tuple[float, float] | None = None,
    color: str = "C3",
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show_data: bool = True,
) -> matplotlib.figure.Figure:
    ...
```

### Parameters

| Parameter   | Type                          | Default | Description |
|-------------|-------------------------------|---------|-------------|
| `results`   | `FilterResults` or `SmootherResults` | — | Filter or smoother output. |
| `component` | `str`, `list[str]`, or `None` | `None`  | Observation component to plot.  Relevant for multivariate observation models. |
| `alpha`     | `float`                       | `0.05`  | Significance level for the prediction interval. |
| `sigma`     | `int` or `None`               | `None`  | Fixed $\sigma$ override. |
| `figsize`   | `tuple[float, float]` or `None` | `None` | Figure size. |
| `color`     | `str`                         | `"C3"`  | Colour for the predicted mean line and band. |
| `title`     | `str` or `None`               | `None`  | Figure title. |
| `ax`        | `matplotlib.axes.Axes` or `None` | `None` | Existing axes (single-panel only). |
| `show_data` | `bool`                        | `True`  | Overlay observed data.  Defaults to `True` because comparison with $y_t$ is the primary purpose. |

### Returns

`matplotlib.figure.Figure`

### Visual output

The figure contains:

1. The observed series $y_t$ as black dots or a thin line.
2. The one-step-ahead predicted mean $\hat y_{t|t-1}$ as a solid coloured
   line.
3. A shaded band $\hat y_{t|t-1} \pm z_{\alpha/2}\sqrt{F_t}$.
4. Observations that fall outside the band are flagged with a red
   cross — the number of such points is reported in the legend if it
   exceeds 5 % of the sample (suggesting model misspecification).

!!! tip "Prediction vs fitted values"
    `plot_prediction` shows **one-step-ahead predictions** $\hat y_{t|t-1}$
    — not the smoothed fitted values $Z_t a_{t|n}$.  One-step predictions
    are causal (they use only past data) and are therefore appropriate
    for residual diagnostics and information-criterion computation.
    Use [Component decomposition](components.md) to plot the smoothed fit.

### Example

```python
from kalmanbox.visualization import plot_prediction

# After fitting any model
results = bsm.fit()

fig = plot_prediction(
    results,
    alpha=0.05,
    color="crimson",
    title="BSM — one-step-ahead predictions",
    show_data=True,
)
```

---

## Customization

All five functions share a consistent customisation API.  The most
common adjustments are shown below.

### Use a specific colour

```python
fig = plot_filtered_state(results, color="steelblue")
fig = plot_smoothed_state(results, color="seagreen")
```

Any string accepted by `matplotlib.colors` works, including hex codes
(`"#2196F3"`), CSS names (`"tomato"`), and the CN notation (`"C4"`).

### Add to existing axes

Embed a state plot into a pre-constructed figure by passing the target
`Axes` object:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_filtered_state(results, ax=ax1, title="Filtered")
plot_smoothed_state(results, ax=ax2, title="Smoothed")

fig.suptitle("State estimates — Local Level Model")
fig.tight_layout()
```

!!! note "Multi-panel constraint"
    The `ax` parameter is honoured only when a single panel is produced
    (i.e. `component` selects exactly one state dimension).  For
    multi-component plots the function allocates its own `Figure` and
    ignores `ax`.

### Overlay the observed data

```python
fig = plot_smoothed_state(results, show_data=True)
```

The data is drawn as small grey circles with `zorder=0` so it sits
behind the state line without obscuring it.

### Change the confidence level

```python
# 90 % band
fig = plot_filtered_state(results, alpha=0.10)

# 99 % band
fig = plot_filtered_state(results, alpha=0.01)

# 1-sigma (68 %) band using the sigma shorthand
fig = plot_filtered_state(results, sigma=1)
```

### Save at publication quality

```python
fig = plot_smoothed_state(results, figsize=(8, 4))
fig.savefig("figure1.pdf", dpi=300, bbox_inches="tight")
```

Alternatively, use the built-in export helper:

```python
from kalmanbox.visualization.export import save_figure

save_figure(fig, "figure1", fmt="pdf", dpi=300)
```

---

## Multivariate states

Structural models (BSM, UCM, Local Linear Trend) carry a multi-dimensional
state vector $\alpha_t = (\mu_t, \beta_t, \gamma_t, \ldots)'$.  Each
component has its own time series of means and variances.

### Component names

`kalmanbox` assigns string names to each state dimension.  The standard
names are:

| Model type  | Component names |
|-------------|----------------|
| Local Level | `"level"` |
| Local Linear Trend | `"level"`, `"slope"` |
| BSM (monthly) | `"level"`, `"slope"`, `"seasonal"` |
| UCM + cycle | `"level"`, `"slope"`, `"cycle"`, `"seasonal"` |
| DFM ($r$ factors) | `"factor_1"`, …, `"factor_r"` |

You can inspect the names available on a fitted result via:

```python
print(results.state_names)
# ['level', 'slope', 'seasonal']
```

### Plotting a subset of components

```python
# One component
fig = plot_smoothed_state(results, component="slope")

# Two components in one figure
fig = plot_smoothed_state(results, component=["level", "slope"])
```

### Iterating over components

When you need a separate figure per component — for example when
submitting to a journal with strict column widths — loop explicitly:

```python
for name in results.state_names:
    fig = plot_smoothed_state(
        results,
        component=name,
        figsize=(6, 3),
        title=f"BSM — smoothed {name}",
    )
    fig.savefig(f"state_{name}.pdf", dpi=300, bbox_inches="tight")
```

---

## Example: Local Level Model

The following self-contained example fits a Local Level Model to the
Nile river data and produces three state plots.

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import KalmanFilter
from kalmanbox.visualization import (
    plot_filtered_state,
    plot_smoothed_state,
    plot_filtered_vs_smoothed,
)

# ── Load data ──────────────────────────────────────────────────────────
from kalmanbox.datasets import load_nile

nile = load_nile()           # annual flow volumes, 1871–1970
y = nile.values.astype(float)
n = len(y)

# ── Build and fit a Local Level Model via MLE ──────────────────────────
from kalmanbox.models import LocalLevelModel

llm = LocalLevelModel(endog=y)
fit = llm.fit(method="L-BFGS-B", disp=False)
print(fit.summary())

# ── Run the filter and the smoother ────────────────────────────────────
filter_res = fit.filter()
smooth_res = fit.smooth()

# ── Plot 1: filtered state ─────────────────────────────────────────────
fig1 = plot_filtered_state(
    filter_res,
    alpha=0.05,
    color="steelblue",
    show_data=True,
    title="Nile — filtered level $a_{t|t}$",
    figsize=(10, 4),
)

# ── Plot 2: smoothed state ─────────────────────────────────────────────
fig2 = plot_smoothed_state(
    smooth_res,
    alpha=0.05,
    color="darkorange",
    show_data=True,
    title="Nile — smoothed level $a_{t|n}$",
    figsize=(10, 4),
)

# ── Plot 3: filtered vs smoothed comparison ────────────────────────────
fig3 = plot_filtered_vs_smoothed(
    smooth_res,                # smoother results contain both arrays
    alpha=0.05,
    color=("steelblue", "darkorange"),
    show_data=True,
    title="Nile — filtered vs smoothed level",
    figsize=(10, 4),
)

plt.show()
```

!!! tip "What to look for"
    In the Nile data there is a well-known **structural break** around
    1899 (completion of the Aswan dam).  The filtered estimate reacts to
    the break gradually as new data arrive, while the smoothed estimate
    places the break sharply because it conditions on all 100 years of
    data simultaneously.  `plot_filtered_vs_smoothed` makes this
    contrast vivid.

---

## Example: BSM state trajectory

The **phase-space trajectory** of the level–slope plane is a compact
diagnostic for assessing whether the trend is accelerating, decelerating,
or oscillating.

```python
import numpy as np
from kalmanbox.models import BasicStructuralModel
from kalmanbox.datasets import load_airline
from kalmanbox.visualization import (
    plot_smoothed_state,
    plot_state_trajectory,
    plot_prediction,
)

# ── Load data ──────────────────────────────────────────────────────────
airline = load_airline()     # Box-Jenkins airline passengers, 1949–1960
y = np.log(airline.values.astype(float))

# ── Fit BSM with monthly seasonal ─────────────────────────────────────
bsm = BasicStructuralModel(
    endog=y,
    seasonal_periods=12,
    stochastic_level=True,
    stochastic_slope=True,
    stochastic_seasonal=True,
)
fit = bsm.fit(method="L-BFGS-B", disp=False)
smooth_res = fit.smooth()

print("State names:", smooth_res.state_names)
# ['level', 'slope', 'seasonal_0', ..., 'seasonal_10']

# ── Panel A: smoothed level and slope ─────────────────────────────────
fig_states = plot_smoothed_state(
    smooth_res,
    component=["level", "slope"],
    alpha=0.05,
    show_data=False,
    figsize=(10, 6),
    title="Airline BSM — smoothed level and slope",
)

# ── Panel B: level–slope phase trajectory ─────────────────────────────
fig_traj = plot_state_trajectory(
    smooth_res,
    component=["level", "slope"],   # x-axis = level, y-axis = slope
    alpha=0.05,
    color="teal",
    figsize=(6, 6),
    title="Level–slope phase space (log passengers)",
)
# The trajectory drifts rightward (rising level) with a positive
# slope that gradually compresses — characteristic of a decelerating
# growth trend.

# ── Panel C: in-sample predictions ────────────────────────────────────
fig_pred = plot_prediction(
    smooth_res,
    alpha=0.05,
    show_data=True,
    color="crimson",
    title="Airline BSM — one-step-ahead predictions (log scale)",
    figsize=(10, 4),
)

import matplotlib.pyplot as plt
plt.show()
```

!!! info "Log-scale interpretation"
    The model is fitted to $\log y_t$, so the level state represents
    log-scale trend and the slope represents the instantaneous log-growth
    rate.  A slope of 0.02 corresponds to approximately 2 % monthly
    growth.

---

## Related

- [Filtered states](filtered-states.md) — quick-start page for
  `plot_filtered`
- [Smoothed states](smoothed-states.md) — quick-start page for
  `plot_smoothed`
- [Component decomposition](components.md) — decompose BSM/UCM into
  level + slope + seasonal
- [Forecast fan charts](forecasts.md) — multi-step-ahead predictive
  intervals
- [Diagnostics plots](diagnostics.md) — residual ACF, Q-Q, CUSUM
- [Visualization API](../api/visualization.md) — complete API reference
- [User guide: Kalman filter](../user-guide/kalman/kalman-filter.md)
- [User guide: RTS smoother](../user-guide/kalman/rts-smoother.md)
- [User guide: BSM](../user-guide/structural/bsm.md)
- [Tutorial: Nile Local Level](../tutorials/nile-local-level.md)
- [Tutorial: Airline BSM](../tutorials/airline-bsm.md)
