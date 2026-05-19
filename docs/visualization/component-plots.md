# Component Plots

Structural models such as BSM and UCM decompose an observed series $y_t$ into
a set of **interpretable, unobserved components** — a smooth trend, a periodic
seasonal pattern, a medium-frequency stochastic cycle, and a white-noise
irregular term. Component plots make these latent signals visible: they draw
each smoothed component in its own panel, together with pointwise credible
bands, so that you can inspect the decomposition at a glance, communicate it
to non-technical audiences, and diagnose potential misspecification.

`kalmanbox.visualization` ships five dedicated component-plotting functions:

| Function | Purpose |
|---|---|
| `plot_components` | Full panel of all estimated components |
| `plot_trend` | Trend component isolated (level + optional slope) |
| `plot_seasonal` | Seasonal component, optionally folded by period |
| `plot_cycle` | Stochastic cycle with damping envelope |
| `plot_decomposition` | Stacked full-decomposition panel with observed series |

All five functions return a `matplotlib.Figure` object that you can further
customise or save with `fig.savefig(...)`.

---

## Component taxonomy

Every structural model expresses the observation equation as a sum of
components. The notation used throughout this page is:

$$
y_t = \mu_t + \gamma_t + \psi_t + \varepsilon_t
$$

where each symbol represents one interpretable part of the signal.

| Component | Symbol | State model | Interpretation |
|---|---|---|---|
| Level | $\mu_t$ | Random walk (or with slope) | The smoothly evolving mean of the series |
| Slope | $\beta_t$ | Random walk | The rate of change of the level per period |
| Seasonal | $\gamma_t$ | Dummy or trigonometric constraint | Periodic fluctuations repeating every $s$ periods |
| Cycle | $\psi_t$ | Damped harmonic oscillator | Medium-frequency oscillation (e.g., business cycle) |
| Irregular | $\varepsilon_t$ | White noise $\mathcal{N}(0, \sigma_\varepsilon^2)$ | Unexplained residual noise |

### Level and slope

The **trend** is the combination of a stochastic level and a stochastic slope:

$$
\begin{aligned}
\mu_{t+1} &= \mu_t + \beta_t + \eta_t, &\quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2) \\
\beta_{t+1} &= \beta_t + \zeta_t, &\quad \zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2)
\end{aligned}
$$

When $\sigma_\zeta^2 = 0$ the slope is deterministic (fixed linear drift).
When $\sigma_\eta^2 = \sigma_\zeta^2 = 0$ the trend is entirely deterministic.

### Seasonal

The **dummy-seasonal** specification constrains seasonal effects to sum to zero
over each complete cycle:

$$
\sum_{j=0}^{s-1} \gamma_{t-j} = \omega_t, \qquad \omega_t \sim \mathcal{N}(0, \sigma_\omega^2)
$$

A stochastic seasonal ($\sigma_\omega^2 > 0$) allows the seasonal pattern to
evolve slowly over time. The **trigonometric** form (used by UCM) is equivalent
but parameterises the seasonal as a sum of $\lfloor s/2 \rfloor$ harmonic pairs.

### Cycle

The **stochastic cycle** is a damped harmonic oscillator:

$$
\begin{pmatrix}\psi_{t+1} \\ \psi_{t+1}^*\end{pmatrix}
= \rho
\begin{pmatrix}\cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c\end{pmatrix}
\begin{pmatrix}\psi_t \\ \psi_t^*\end{pmatrix}
+
\begin{pmatrix}\kappa_t \\ \kappa_t^*\end{pmatrix}
$$

where $\rho \in (0,1)$ is the damping factor, $\lambda_c \in (0, \pi)$ is the
cycle frequency (radians per period), and $\kappa_t, \kappa_t^*$ are
independent $\mathcal{N}(0, \sigma_\kappa^2)$ disturbances.

The implied period of the cycle is $2\pi / \lambda_c$ periods.

### Irregular

The **irregular** is pure observation noise added on top of the state-space
signal:

$$
\varepsilon_t \sim \mathcal{N}(0,\, \sigma_\varepsilon^2), \qquad
\text{independent of all state disturbances.}
$$

In a well-specified model the smoothed irregular looks like white noise.
Autocorrelation in the irregular is a sign of missing structure (e.g., an
omitted cycle or AR term).

---

## `plot_components`

Renders all estimated components side-by-side in a vertical stack of subplots.
This is the primary decomposition view for BSM and UCM results.

### Signature

```python
def plot_components(
    results: SmootherResults,
    components: list[str] | None = None,
    figsize: tuple[float, float] = (10, 8),
    sharey: bool = False,
    color_map: dict[str, str] | None = None,
    show_bands: bool = True,
    alpha: float = 0.05,
    title: str | None = None,
    ax: list[matplotlib.axes.Axes] | None = None,
) -> matplotlib.figure.Figure:
```

### Parameters

`results` : `SmootherResults`
:   Output of `.fit()` on any structural model (BSM, UCM, LocalLevel, LocalLinearTrend).
    Must contain smoothed state arrays (`smoothed_states`, `smoothed_states_cov`).

`components` : `list[str] | None`, default `None`
:   Which components to draw. When `None` all components detected in `results`
    are drawn. Valid component names are `"level"`, `"slope"`, `"seasonal"`,
    `"cycle"`, `"irregular"`. Unrecognised names raise `ValueError`.

`figsize` : `tuple[float, float]`, default `(10, 8)`
:   Width and height of the returned figure in inches. The height is shared
    across all subplots; increase it if component labels overlap.

`sharey` : `bool`, default `False`
:   When `True` all component subplots share the same y-axis scale, which
    makes amplitude differences easier to compare. When `False` each subplot
    auto-scales independently, which is usually better for components that
    operate on very different scales (e.g., a large trend alongside a small
    irregular).

`color_map` : `dict[str, str] | None`, default `None`
:   Maps component names to matplotlib colour strings. Keys are the same
    names accepted by `components`. Missing keys fall back to the default
    palette (`C0` for level, `C1` for slope, `C2` for seasonal, `C3` for
    cycle, `C4` for irregular).

`show_bands` : `bool`, default `True`
:   Whether to draw the $(1 - \alpha)$ pointwise confidence band around each
    component. The band is computed from the diagonal of the smoothed state
    covariance matrix (or component sub-covariance for multi-component
    models).

`alpha` : `float`, default `0.05`
:   Significance level for confidence bands. The coverage is $1 - \alpha$;
    the default $\alpha = 0.05$ gives 95% bands. Must satisfy $0 < \alpha < 1$.

`title` : `str | None`, default `None`
:   Super-title placed above all subplots via `fig.suptitle`. When `None` a
    default title derived from the model class name is used.

`ax` : `list[matplotlib.axes.Axes] | None`, default `None`
:   Pre-existing axes to draw into. When provided, the list must have at
    least as many elements as there are components to draw. When `None` a
    new figure and axes are created internally. Using `ax` lets you embed
    component panels inside a larger figure layout.

### Returns

`matplotlib.figure.Figure`
:   The figure containing all component subplots. Use `.savefig()` or
    display it in a Jupyter notebook.

### Visual output

Each subplot shows:

- A **solid line** for the smoothed component estimate.
- A **shaded band** (when `show_bands=True`) spanning
  $\hat{c}_t \pm z_{\alpha/2}\,\widehat{\mathrm{se}}(c_t)$
  where $z_{0.025} = 1.96$ for the default 95% coverage.
- A **horizontal zero line** (thin grey dashes) for reference.
- The component name and symbol as the subplot title.

### Example

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.visualization import plot_components, set_theme

set_theme()  # kalmanbox house style

rng = np.random.default_rng(42)
t = np.arange(120)
y = (0.3 * t + 50) + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120)

results = BSM(y, seasonal_periods=12).fit()

fig = plot_components(results)
fig.savefig("bsm_components.png", dpi=150)
```

Showing a subset with custom colours:

```python
fig = plot_components(
    results,
    components=["level", "seasonal"],
    color_map={"level": "steelblue", "seasonal": "coral"},
    figsize=(10, 5),
    title="BSM — trend and seasonal only",
)
```

---

## `plot_trend`

Focuses on the **trend component** alone. By default it overlays both the level
$\mu_t$ and the slope $\beta_t$ (on a shared figure with two subplots), making
it easy to see whether the growth rate is accelerating or decelerating over time.

### Signature

```python
def plot_trend(
    results: SmootherResults,
    include_slope: bool = True,
    figsize: tuple[float, float] = (10, 5),
    color_map: dict[str, str] | None = None,
    show_bands: bool = True,
    alpha: float = 0.05,
    title: str | None = None,
    ax: matplotlib.axes.Axes | list[matplotlib.axes.Axes] | None = None,
) -> matplotlib.figure.Figure:
```

### Parameters

`results` : `SmootherResults`
:   Smoother output from a model that includes a trend component (BSM, UCM
    with `level=True`, LocalLinearTrend, etc.).

`include_slope` : `bool`, default `True`
:   When `True` a second subplot is added below the level showing the
    smoothed slope $\beta_t$. When `False` only the level is drawn in a
    single subplot.

`figsize` : `tuple[float, float]`, default `(10, 5)`
:   Figure width and height in inches. When `include_slope=True` you may
    want to increase the height (e.g., `(10, 7)`) so both panels are
    comfortable.

`color_map` : `dict[str, str] | None`, default `None`
:   Colour overrides for `"level"` and `"slope"` keys. Falls back to the
    default palette when `None`.

`show_bands` : `bool`, default `True`
:   Draws $(1 - \alpha)$ confidence bands around the level and slope estimates.

`alpha` : `float`, default `0.05`
:   Significance level for confidence bands (default: 95% bands).

`title` : `str | None`, default `None`
:   Figure super-title. Defaults to `"Trend decomposition"`.

`ax` : `Axes | list[Axes] | None`, default `None`
:   One or two pre-existing axes. When `include_slope=True` and two axes are
    supplied, the level goes into `ax[0]` and the slope into `ax[1]`.
    When a single `Axes` is supplied (or `include_slope=False`), the level
    goes into that axis.

### Returns

`matplotlib.figure.Figure`

### Visual output

- **Top panel** (always): smoothed level $\mu_t$ as a solid line; shaded 95%
  band; the observed data $y_t$ overlaid as a semi-transparent scatter in the
  background so you can judge how closely the trend tracks the data.
- **Bottom panel** (when `include_slope=True`): smoothed slope $\beta_t$ with
  band; a dashed horizontal zero line; positive values indicate an upswing,
  negative values a downturn.

### Example

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.visualization import plot_trend

rng = np.random.default_rng(7)
t = np.arange(120)
y = (0.3 * t + 50) + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120)

results = BSM(y, seasonal_periods=12).fit()

# Level + slope view
fig = plot_trend(results, include_slope=True, figsize=(10, 6))

# Level only (single panel)
fig = plot_trend(results, include_slope=False, title="Smoothed trend")
```

!!! tip "Deterministic slope"
    When the model is fitted with a fixed (deterministic) slope
    (`slope_variance=0` in UCM), the slope panel is a flat line. Use
    `include_slope=False` in that case to avoid a redundant subplot.

---

## `plot_seasonal`

Plots the **seasonal component** $\gamma_t$ extracted by the smoother.
An optional *period-folding* view collapses all years (or cycles) on top of
each other, producing a seasonal-profile chart that shows how the average
pattern within a cycle has evolved over the sample.

### Signature

```python
def plot_seasonal(
    results: SmootherResults,
    period: int | None = None,
    figsize: tuple[float, float] = (10, 5),
    color_map: dict[str, str] | None = None,
    show_bands: bool = True,
    alpha: float = 0.05,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
```

### Parameters

`results` : `SmootherResults`
:   Smoother output from a model that includes a seasonal component.
    Raises `ValueError` if no seasonal state is detected.

`period` : `int | None`, default `None`
:   When `None` (the default) the seasonal component is plotted as a plain
    time series, with all $n$ time points on the x-axis. When set to an
    integer $s$, the function produces a **polar / overlay** view: $\lfloor
    n/s \rfloor$ overlapping lines, one per complete cycle, with the
    within-cycle index ($1, \ldots, s$) on the x-axis. This makes
    long-run changes in the seasonal pattern immediately visible.
    Common values: `12` (monthly), `4` (quarterly), `52` (weekly).

`figsize` : `tuple[float, float]`, default `(10, 5)`
:   Figure dimensions in inches.

`color_map` : `dict[str, str] | None`, default `None`
:   Accepts a `"seasonal"` key for the main line colour. When using the
    period-overlay view, each cycle line is coloured by a gradient from the
    colormap specified under `"seasonal_cmap"` (default `"Blues"`).

`show_bands` : `bool`, default `True`
:   In the plain time-series view, draws the $(1 - \alpha)$ confidence band.
    In the period-overlay view, draws a shaded band for the **average** seasonal
    profile (average across all complete cycles in the sample).

`alpha` : `float`, default `0.05`
:   Significance level for confidence bands.

`title` : `str | None`, default `None`
:   Figure title. Defaults to `"Seasonal component"`.

`ax` : `Axes | None`, default `None`
:   Optional pre-existing axes. When provided the plot is drawn into these
    axes and the owning figure is returned.

### Returns

`matplotlib.figure.Figure`

### Visual output

**Plain view** (`period=None`):

- Solid coloured line tracing $\hat\gamma_t$ across the full sample.
- Shaded $(1 - \alpha)$ confidence band.
- Horizontal zero reference line.

**Period-overlay view** (`period=s`):

- One line per complete cycle (year, quarter, …), coloured from light to
  dark so that the most recent cycles are darker.
- A bold line for the average profile across all cycles.
- Shaded band around the average profile.
- X-axis labelled with within-cycle position (e.g., months Jan–Dec).

### Example

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.visualization import plot_seasonal

rng = np.random.default_rng(3)
t = np.arange(120)
seasonal_signal = 8 * np.sin(2 * np.pi * t / 12)
y = (0.3 * t + 50) + seasonal_signal + rng.normal(0, 3, 120)

results = BSM(y, seasonal_periods=12).fit()

# Plain time-series view
fig = plot_seasonal(results)

# Period-overlay view: each of the 10 years as a separate line
fig = plot_seasonal(
    results,
    period=12,
    color_map={"seasonal_cmap": "viridis"},
    title="Monthly seasonal profile (10 years)",
)
```

!!! note "Trigonometric seasonal (UCM)"
    For UCM models with `seasonal_harmonics=k`, the seasonal component is
    reconstructed from the harmonic pairs before plotting. The result is
    identical in interpretation to the dummy-seasonal output.

---

## `plot_cycle`

Renders the **stochastic cycle** component $\psi_t$ and overlays its
**damping envelope** — the time-varying amplitude implied by the estimated
cycle variance $\sigma_\kappa^2$ and damping parameter $\rho$.

### Signature

```python
def plot_cycle(
    results: SmootherResults,
    figsize: tuple[float, float] = (10, 4),
    color_map: dict[str, str] | None = None,
    show_bands: bool = True,
    alpha: float = 0.05,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
```

### Parameters

`results` : `SmootherResults`
:   Smoother output from a UCM model fitted with `cycle=True`. Raises
    `ValueError` if no cycle state is detected.

`figsize` : `tuple[float, float]`, default `(10, 4)`
:   Figure dimensions in inches.

`color_map` : `dict[str, str] | None`, default `None`
:   Accepts a `"cycle"` key for the cycle line colour and an
    `"envelope"` key for the damping-envelope colour (default `"grey"`).

`show_bands` : `bool`, default `True`
:   When `True`, draws the pointwise $(1 - \alpha)$ confidence band around
    $\hat\psi_t$ (from smoothed state covariance) **and** the damping envelope
    $\pm A_t$ where $A_t$ is the smoothed cycle amplitude.

`alpha` : `float`, default `0.05`
:   Significance level for the confidence bands.

`title` : `str | None`, default `None`
:   Figure title. Defaults to `"Stochastic cycle"` with the estimated
    period appended (e.g., `"Stochastic cycle — period ≈ 40 quarters"`).

`ax` : `Axes | None`, default `None`
:   Optional pre-existing axes.

### Returns

`matplotlib.figure.Figure`

### Visual output

- **Solid line**: smoothed cycle $\hat\psi_t$.
- **Shaded band**: pointwise 95% confidence band from the smoothed state covariance.
- **Dashed envelope** (positive and negative): the time-varying amplitude
  $\pm A_t$ derived from $\sigma_\kappa^2 / (1 - \rho^2)$. This shows the
  theoretical maximum swing of the cycle at each point.
- A subtitle or annotation with the estimated parameters
  ($\hat\rho$, $\hat\lambda_c$, implied period).

### Example

```python
import numpy as np
from kalmanbox import UCM
from kalmanbox.visualization import plot_cycle

rng = np.random.default_rng(17)
n = 200

# Simulate series with a ~40-period business cycle
t = np.arange(n)
cycle = 5 * np.cos(2 * np.pi * t / 40)
y = 50 + 0.1 * t + cycle + rng.normal(0, 2, n)

results = UCM(
    y,
    level=True,
    slope=True,
    cycle=True,
    irregular=True,
).fit()

fig = plot_cycle(results, title="Business cycle component")
fig.savefig("cycle_component.png", dpi=150)
```

!!! tip "Interpreting the envelope"
    A narrow, constant envelope indicates a stable cycle variance. A widening
    envelope toward the end of the sample usually means the smoother has
    less information there (boundary effect). Check whether the envelope
    width collapses to near zero — that would indicate the model estimated
    $\rho \approx 0$, meaning no persistent cycle is supported by the data.

---

## `plot_decomposition`

Produces a **comprehensive stacked decomposition** figure with the observed
series at the top and each estimated component below it. This is the
publication-ready version of the decomposition, combining all components
into a single self-contained figure.

### Signature

```python
def plot_decomposition(
    results: SmootherResults,
    components: list[str] | None = None,
    figsize: tuple[float, float] = (12, 10),
    sharey: bool = False,
    color_map: dict[str, str] | None = None,
    show_bands: bool = True,
    alpha: float = 0.05,
    title: str | None = None,
    ax: list[matplotlib.axes.Axes] | None = None,
) -> matplotlib.figure.Figure:
```

### Parameters

`results` : `SmootherResults`
:   Smoother output from any structural model.

`components` : `list[str] | None`, default `None`
:   Components to include in the lower panels. The top panel always shows
    the observed series $y_t$ and the reconstructed fit. When `None`, all
    available components are shown.

`figsize` : `tuple[float, float]`, default `(12, 10)`
:   Figure dimensions. The default is wider and taller than `plot_components`
    because an extra panel is added for the observed series.

`sharey` : `bool`, default `False`
:   Share the y-axis across the component panels (not the observed series
    panel, which always auto-scales).

`color_map` : `dict[str, str] | None`, default `None`
:   Colour overrides. An additional `"observed"` key controls the colour of
    the observed series scatter; `"fitted"` controls the reconstructed fit
    line.

`show_bands` : `bool`, default `True`
:   Draws confidence bands on all component panels. The observed panel does
    not carry bands.

`alpha` : `float`, default `0.05`
:   Significance level for confidence bands.

`title` : `str | None`, default `None`
:   Figure super-title. Defaults to `"Full decomposition — <ModelClass>"`.

`ax` : `list[Axes] | None`, default `None`
:   Pre-existing axes list. Must have `1 + len(components)` elements: the
    first is used for the observed series, the rest for the components.

### Returns

`matplotlib.figure.Figure`

### Visual output

- **Panel 0 (Observed)**: dots for $y_t$ (dark grey) and a solid line for
  the smoothed reconstructed fit $\hat y_t = Z\hat\alpha_{t|n}$. The fit
  line and its confidence band are drawn in the same colour as the level
  component.
- **Panel 1 — Level** $\mu_t$: smooth trend line with 95% band.
- **Panel 2 — Slope** $\beta_t$ (if present): rate-of-change line with band.
- **Panel 3 — Seasonal** $\gamma_t$ (if present): periodic oscillation.
- **Panel 4 — Cycle** $\psi_t$ (if present): medium-frequency oscillation.
- **Panel 5 — Irregular** $\varepsilon_t$ (if present): residual noise,
  expected to look like white noise.

### Example

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.visualization import plot_decomposition, set_theme

set_theme()

rng = np.random.default_rng(0)
t = np.arange(120)
y = (0.3 * t + 50) + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120)

results = BSM(y, seasonal_periods=12).fit()

fig = plot_decomposition(results, figsize=(12, 10))
fig.savefig("full_decomposition.png", dpi=150, bbox_inches="tight")
```

---

## BSM decomposition example

This end-to-end example fits a Basic Structural Model to simulated monthly
data with a linear trend and a 12-month seasonal pattern, then visualises the
full decomposition.

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.visualization import plot_components, plot_decomposition, set_theme

set_theme()  # apply kalmanbox house style

# -----------------------------------------------------------------
# Simulate 10 years of monthly data
# -----------------------------------------------------------------
rng = np.random.default_rng(0)
t = np.arange(120)

trend    = 0.3 * t + 50                        # linear trend
seasonal = 8 * np.sin(2 * np.pi * t / 12)     # 12-month seasonal
y        = trend + seasonal + rng.normal(0, 3, 120)

# -----------------------------------------------------------------
# Fit BSM
# -----------------------------------------------------------------
results = BSM(y, seasonal_periods=12).fit()
print(results.summary())

# -----------------------------------------------------------------
# Component panel (4 subplots)
# -----------------------------------------------------------------
fig = plot_components(results)
```

### What each subplot shows

After fitting, `plot_components(results)` produces four stacked panels:

**Level $\mu_t$**
:   A smooth upward curve that tracks the 0.3-per-period drift. The 95% band
    is narrow in the centre of the sample and slightly wider at both ends
    (the Kalman smoother boundary effect). The level absorbs the long-run
    mean of the series.

**Slope $\beta_t$**
:   A near-horizontal line around $+0.3$, reflecting the true deterministic
    drift. Because the true slope variance $\sigma_\zeta^2$ is small relative
    to the observation noise, the estimated slope does not fluctuate much.
    A time-varying slope would appear as a curved trajectory.

**Seasonal $\gamma_t$**
:   A repeating sinusoidal pattern with amplitude approximately $\pm 8$,
    completing one full cycle every 12 observations. The stochastic seasonal
    may show slight inter-year variation in amplitude if the MLE estimate of
    $\sigma_\omega^2 > 0$.

**Irregular $\varepsilon_t$**
:   White noise centred on zero with standard deviation close to 3 (the true
    observation noise). No systematic pattern should be visible. Autocorrelation
    in this panel would suggest a missing component.

```python
# -----------------------------------------------------------------
# Full stacked decomposition
# -----------------------------------------------------------------
fig = plot_decomposition(results, figsize=(12, 10))
fig.savefig("bsm_full.png", dpi=150, bbox_inches="tight")
```

!!! info "Reconstructed fit"
    The top panel of `plot_decomposition` shows both the raw data and the
    smoothed fit $\hat y_t = \hat\mu_t + \hat\gamma_t$. The sum of the
    components in the lower panels equals this fit line exactly.

---

## UCM decomposition example

UCM generalises BSM by allowing a **stochastic business cycle** component
$\psi_t$ in addition to (or instead of) a seasonal. The following example
fits a UCM with level, slope, seasonal, cycle, and irregular to a longer
quarterly series.

=== "Fit and plot"

    ```python
    import numpy as np
    from kalmanbox import UCM
    from kalmanbox.visualization import plot_decomposition, plot_cycle, set_theme

    set_theme()

    rng = np.random.default_rng(5)
    n = 200  # 50 years of quarterly data

    t           = np.arange(n)
    trend_q     = 0.05 * t + 100
    seasonal_q  = 3 * np.sin(2 * np.pi * t / 4)    # 4-quarter seasonal
    cycle_q     = 6 * np.cos(2 * np.pi * t / 32)   # ~8-year business cycle
    y_q         = trend_q + seasonal_q + cycle_q + rng.normal(0, 2, n)

    results_ucm = UCM(
        y_q,
        level=True,
        slope=True,
        seasonal=4,        # dummy seasonal with period 4
        cycle=True,        # stochastic business cycle
        irregular=True,
    ).fit()

    print(results_ucm.summary())

    # Full stacked decomposition
    fig = plot_decomposition(results_ucm, figsize=(12, 14))
    fig.savefig("ucm_decomposition.png", dpi=150, bbox_inches="tight")
    ```

=== "Cycle only"

    ```python
    from kalmanbox.visualization import plot_cycle

    # Zoom in on the business cycle component
    fig = plot_cycle(
        results_ucm,
        title="Business cycle — UCM (quarterly, ~8-year period)",
        figsize=(11, 4),
    )
    fig.savefig("ucm_cycle.png", dpi=150)
    ```

=== "Seasonal profile"

    ```python
    from kalmanbox.visualization import plot_seasonal

    # Fold the seasonal across all 50 years
    fig = plot_seasonal(
        results_ucm,
        period=4,
        title="Quarterly seasonal profile",
        color_map={"seasonal_cmap": "RdYlGn"},
    )
    fig.savefig("ucm_seasonal_profile.png", dpi=150)
    ```

The UCM decomposition produces five panels:

| Panel | Component | Expected appearance |
|---|---|---|
| 0 — Observed + fit | $y_t$ and $\hat y_t$ | Data cloud with smooth fit |
| 1 — Level | $\mu_t$ | Slow upward drift |
| 2 — Slope | $\beta_t$ | Near-constant ~0.05/quarter |
| 3 — Seasonal | $\gamma_t$ | 4-period oscillation |
| 4 — Cycle | $\psi_t$ | ~32-period oscillation with damping |
| 5 — Irregular | $\varepsilon_t$ | White noise $\approx \mathcal{N}(0, 4)$ |

---

## Customisation

### Selecting a component subset

Pass `components` to limit which panels are drawn:

```python
fig = plot_components(
    results,
    components=["level", "seasonal"],
)
```

This is useful when a model has many components but you want to highlight
only the trend and the seasonal pattern in a report figure.

### Custom colours

Use `color_map` to override the default palette. Keys must match valid
component names.

```python
fig = plot_components(
    results,
    color_map={
        "level":    "navy",
        "slope":    "royalblue",
        "seasonal": "coral",
        "cycle":    "forestgreen",
        "irregular": "slategrey",
    },
)
```

### Suppressing confidence bands

Set `show_bands=False` to remove all shading. This is useful for clean
publication figures where bands would clutter the layout:

```python
fig = plot_components(results, show_bands=False)
```

### Sharing the y-axis

When you want to emphasise the *relative* amplitudes of components, set
`sharey=True`. All component panels will use the same y-axis limits, making
small components appear compressed next to large ones:

```python
fig = plot_components(results, sharey=True)
```

### Changing the confidence level

The default $\alpha = 0.05$ gives 95% coverage. For 90% or 99% bands:

```python
fig = plot_components(results, alpha=0.10)  # 90% bands
fig = plot_components(results, alpha=0.01)  # 99% bands
```

### Embedding in a larger figure

Pass pre-existing axes to integrate component panels into a custom layout:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_components

fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
plot_components(results, ax=axes)

# Add your own annotation on top
axes[0].axvline(60, color="red", linestyle="--", label="Structural break?")
axes[0].legend()

fig.savefig("custom_layout.png", dpi=150)
```

---

## Reconstructed fit

The smoothed component estimates sum to the **in-sample fitted values**.
You can extract and overlay them manually for full control:

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import BSM

rng = np.random.default_rng(0)
t = np.arange(120)
y = (0.3 * t + 50) + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120)

results = BSM(y, seasonal_periods=12).fit()

# Extract fitted values and 95% prediction interval
fitted = results.fitted_values()                # array of length n
se_fit = results.fitted_se()                    # standard error of fit
z = 1.96
lower = fitted - z * se_fit
upper = fitted + z * se_fit

# Manual overlay
fig, ax = plt.subplots(figsize=(11, 4))
ax.scatter(t, y, color="grey", s=12, alpha=0.6, label="Observed $y_t$")
ax.plot(t, fitted, color="C0", linewidth=1.8, label=r"Fitted $\hat{y}_t$")
ax.fill_between(t, lower, upper, color="C0", alpha=0.15,
                label="95% prediction interval")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title("BSM — observed data and reconstructed fit")
ax.legend(framealpha=0.9)
fig.tight_layout()
```

!!! note "Fitted vs. filtered vs. smoothed"
    `results.fitted_values()` uses the **smoothed** state estimates
    ($a_{t|n}$), not the filtered estimates ($a_{t|t}$). The smoothed fit
    will always be closer to the data than the filtered fit because it
    conditions on all observations. For a proper one-step-ahead in-sample
    fit, use `results.filtered_values()` instead.

### Checking reconstruction accuracy

Verify that the sum of all smoothed components equals the smoothed fit:

```python
mu_hat    = results.smoothed_component("level")
gamma_hat = results.smoothed_component("seasonal")
eps_hat   = results.smoothed_component("irregular")

reconstruction = mu_hat + gamma_hat + eps_hat
residual       = fitted - reconstruction

print(f"Max reconstruction error: {np.abs(residual).max():.2e}")
# Expected: < 1e-10 (machine precision)
```

A non-negligible reconstruction error would indicate that you have omitted
a component from the sum (e.g., a cycle or exogenous regressor contribution).

---

## Parameter summary after fitting

All component plots become more meaningful when you first inspect the
estimated hyperparameters. Call `.summary()` on the results object:

```python
results = BSM(y, seasonal_periods=12).fit()
print(results.summary())
```

Typical output:

```
BSM — MLE estimates (n = 120)
======================================================
           Coef     SE      z      p      [95% CI]
------------------------------------------------------
sigma2_eta  0.087  0.031   2.83  0.005  [0.027, 0.148]
sigma2_zeta 0.002  0.004   0.46  0.642  [0.000, 0.010]
sigma2_omega 0.011 0.008   1.37  0.170  [0.000, 0.026]
sigma2_eps  8.711  1.214   7.18  <.001  [6.332, 11.09]
------------------------------------------------------
Log-likelihood: -338.41   AIC: 684.82   BIC: 696.30
```

The ratio $\sigma_\eta^2 / \sigma_\varepsilon^2$ is the **signal-to-noise
ratio** (SNR) for the level. A very small SNR means the level barely moves
relative to the observation noise — the trend will appear flat. A large SNR
means the level tracks the data closely.

---

## Related pages

- [User guide: BSM](../user-guide/structural/bsm.md) — full specification and state-space matrices
- [User guide: UCM](../user-guide/structural/ucm.md) — component configuration and cycle specification
- [State plots](state-plots.md) — filtered and smoothed state plots (pre-decomposition view)
- [Filtered states](filtered-states.md) — one-sided online estimates $a_{t|t}$
- [Smoothed states](smoothed-states.md) — two-sided estimates $a_{t|n}$
- [Forecast fan charts](forecasts.md) — forward-looking component fans
- [Diagnostics](../diagnostics/residuals.md) — checking the irregular for white noise
