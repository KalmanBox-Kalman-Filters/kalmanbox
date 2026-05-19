# Innovation Plots

Innovations — the **one-step-ahead prediction errors** produced by the Kalman filter — are
the primary diagnostic residual in a state-space model. Under correct specification they
should be an i.i.d. Gaussian white-noise sequence. The functions in this page make that
claim visually testable.

---

## Innovation theory

At each time step $t$ the Kalman filter produces the innovation

$$
v_t = y_t - Z_t\, a_{t|t-1}
$$

where $a_{t|t-1} = \mathbb{E}[x_t \mid y_1, \ldots, y_{t-1}]$ is the one-step-ahead state
prediction and $Z_t$ is the observation matrix. The associated innovation variance is

$$
F_t = Z_t P_{t|t-1} Z_t^\top + H_t
$$

where $P_{t|t-1}$ is the state prediction covariance and $H_t$ is the observation noise
covariance. Under correct model specification, the innovations satisfy:

$$
v_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0,\; F_t), \qquad t = 1, \ldots, n
$$

and the **standardised innovations**

$$
e_t = \frac{v_t}{\sqrt{F_t}}
$$

are i.i.d. $\mathcal{N}(0,1)$.

!!! info "Why standardise?"
    Raw innovations $v_t$ have time-varying variance $F_t$. During the **diffuse
    initialisation** phase (typically the first $d$ observations) $F_t$ can be very large,
    inflating the raw innovations. Missing observations have the opposite effect — the filter
    skips those time steps entirely. Standardised innovations $e_t$ remove this
    heteroscedasticity and are directly comparable across time, making them the preferred
    residual for all diagnostic purposes.

The log-likelihood is itself expressed in terms of innovations:

$$
\log L = -\frac{np}{2}\log(2\pi)
           - \frac{1}{2}\sum_{t=1}^{n}\!\left(\log|F_t| + v_t^\top F_t^{-1} v_t\right)
$$

Every diagnostic test on innovations is therefore testing the assumptions that underpin MLE
and Bayesian inference alike.

### Key properties to check

| Property | What it means | Plot that tests it |
|---|---|---|
| Zero mean | $\mathbb{E}[e_t] = 0$ | Innovation time-series, histogram |
| Serial independence | $\text{Cov}(e_t, e_s) = 0,\; t \neq s$ | ACF |
| Homoscedasticity | $\text{Var}(e_t) = 1$ after standardisation | Time-series scatter, CUSUMSQ |
| Normality | $e_t \sim \mathcal{N}(0,1)$ | QQ-plot, histogram |
| Parameter stability | no structural break in mean/variance | CUSUM, CUSUMSQ |

---

## `plot_innovations`

Plot the innovation series $v_t$ (raw or standardised) over time, with
$\pm 2\sigma$ reference bands.

### Signature

```python
from kalmanbox.visualization import plot_innovations

def plot_innovations(
    results,
    standardize: bool = True,
    figsize: tuple[float, float] = (10, 4),
    color: str = "C0",
    band_color: str = "C0",
    band_alpha: float = 0.15,
    sigma: float = 2.0,
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `KalmanResults` | — | Fitted model results object |
| `standardize` | `bool` | `True` | Plot $e_t = v_t/\sqrt{F_t}$ instead of raw $v_t$ |
| `figsize` | `tuple` | `(10, 4)` | Figure size in inches `(width, height)` |
| `color` | `str` | `"C0"` | Line colour for the innovation series |
| `band_color` | `str` | `"C0"` | Colour of the $\pm\sigma$ shaded band |
| `band_alpha` | `float` | `0.15` | Opacity of the shaded band |
| `sigma` | `float` | `2.0` | Number of standard deviations for the reference band |
| `title` | `str \| None` | `None` | Axes title; defaults to `"Standardised innovations"` or `"Innovations"` |
| `ax` | `Axes \| None` | `None` | Existing axes to draw on; a new figure is created if `None` |

### Returns

`matplotlib.figure.Figure` — the figure containing the axes.

### Interpretation guide

- **Points within the band**: expected behaviour under a correctly specified model;
  roughly 95 % of $e_t$ should fall inside $\pm 2\sigma$ bands.
- **Outliers outside the band**: isolated outliers are common; a *cluster* of outliers
  suggests a missing intervention variable or a misspecified error distribution.
- **Trend or drift in the series**: the mean is not zero — the model may be missing a
  level component.
- **Changing spread over time**: variance is not constant — consider adding a stochastic
  volatility or Student-$t$ error component.

!!! tip "Standardisation default"
    Leave `standardize=True` (the default) unless you specifically want to inspect the
    raw innovations on the original scale of $y_t$.

### Example

```python
import numpy as np
from kalmanbox import LocalLevel
from kalmanbox.visualization import plot_innovations

rng = np.random.default_rng(0)
y = np.cumsum(rng.normal(0, 1, 120)) + rng.normal(0, 0.5, 120)

results = LocalLevel(y).fit()

fig = plot_innovations(results, standardize=True, sigma=2.0, figsize=(11, 3))
fig.suptitle("Local Level — Standardised Innovations", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("innovations.png", dpi=150)
```

The horizontal dashed lines at $\pm 2$ mark the 95 % band. Under the null of white-noise
innovations, approximately 5 % of points are expected to exceed these lines by chance.

---

## `plot_qq`

QQ-plot of the standardised innovations against the theoretical quantiles of the standard
normal distribution.

### Signature

```python
from kalmanbox.visualization import plot_qq

def plot_qq(
    results,
    line: str = "45",
    ci: bool = True,
    ci_alpha: float = 0.05,
    color: str = "C0",
    line_color: str = "C3",
    figsize: tuple[float, float] = (5, 5),
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `KalmanResults` | — | Fitted model results object |
| `line` | `str` | `"45"` | Reference line style: `"45"` for the 45° identity line, `"s"` for standardised, `"q"` for quartile fit, `"r"` for regression fit |
| `ci` | `bool` | `True` | Draw pointwise 95 % confidence band around the reference line |
| `ci_alpha` | `float` | `0.05` | Significance level for the confidence band |
| `color` | `str` | `"C0"` | Colour of the scatter points |
| `line_color` | `str` | `"C3"` | Colour of the reference line and confidence band |
| `figsize` | `tuple` | `(5, 5)` | Figure size in inches |
| `title` | `str \| None` | `None` | Axes title; defaults to `"QQ-Plot — Standardised Innovations"` |
| `ax` | `Axes \| None` | `None` | Existing axes to draw on |

### Returns

`matplotlib.figure.Figure`

### Interpretation guide

- **Points on the 45° line**: the standardised innovations are normally distributed —
  the model fits well.
- **S-shaped curve (light tails)**: the empirical distribution has lighter tails than
  the normal; can arise from censored data or misspecified variance structure.
- **Reverse S-shape (heavy tails)**: empirical distribution has heavier tails; suggests
  the observation noise should be modelled with a Student-$t$ or mixture distribution.
- **Points curve upward at the right tail only**: positive skewness; the model may be
  missing a log-transformation or a positive outlier correction.
- **Points systematically above/below the 45° line**: non-zero mean; a level shift or
  intercept is missing.

### Example

```python
from kalmanbox.visualization import plot_qq

fig = plot_qq(results, line="45", ci=True, figsize=(5, 5))
fig.tight_layout()
```

---

## `plot_acf_residuals`

Bar plot of the autocorrelation function (ACF) of the standardised innovations, with
optional partial ACF (PACF) panel.

### Signature

```python
from kalmanbox.visualization import plot_acf_residuals

def plot_acf_residuals(
    results,
    nlags: int = 40,
    pacf: bool = False,
    alpha: float = 0.05,
    color: str = "C0",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `KalmanResults` | — | Fitted model results object |
| `nlags` | `int` | `40` | Maximum lag to compute and display |
| `pacf` | `bool` | `False` | If `True`, add a second panel with the PACF |
| `alpha` | `float` | `0.05` | Significance level for the $\pm 1.96/\sqrt{n}$ Bartlett bounds |
| `color` | `str` | `"C0"` | Bar colour |
| `figsize` | `tuple \| None` | `None` | Figure size; defaults to `(10, 3)` for ACF-only and `(10, 6)` when PACF is added |
| `title` | `str \| None` | `None` | Axes title |

### Returns

`matplotlib.figure.Figure`

### Interpretation guide

The significance bounds are $\pm z_{1-\alpha/2}/\sqrt{n}$, where $n$ is the number of
non-missing standardised innovations. Under the white-noise null, 95 % of lags should fall
inside these bounds by chance.

- **All bars inside the bounds**: no detectable serial correlation — good.
- **Spike at lag 1 only**: first-order MA structure remains unexplained; verify the model
  includes all moving-average terms.
- **Significant spike at lag $s$ (seasonal)**: a seasonal component at period $s$ is
  missing from the model.
- **Slow geometric decay**: AR structure in the innovations; the model may need an
  additional autoregressive component.
- **Spike at lag 1 in PACF, nothing beyond**: consistent with an MA(1) error process.

!!! tip "Degrees of freedom correction"
    The `nlags` argument controls how many lags are plotted. For model-selection purposes,
    the Ljung–Box portmanteau test (see
    [Innovation Tests](../diagnostics/innovation-tests.md)) is a formal complement to the
    ACF visual.

### Example

```python
from kalmanbox.visualization import plot_acf_residuals

# ACF only
fig = plot_acf_residuals(results, nlags=36)

# ACF + PACF
fig2 = plot_acf_residuals(results, nlags=36, pacf=True)
fig2.tight_layout()
```

---

## `plot_histogram`

Histogram of the standardised innovations overlaid with the fitted standard normal density,
for a quick visual normality check.

### Signature

```python
from kalmanbox.visualization import plot_histogram

def plot_histogram(
    results,
    bins: int | str = "auto",
    density: bool = True,
    color: str = "C0",
    density_color: str = "C3",
    density_lw: float = 2.0,
    kde: bool = False,
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `KalmanResults` | — | Fitted model results object |
| `bins` | `int \| str` | `"auto"` | Number of bins or a NumPy bin-selection strategy (`"auto"`, `"fd"`, `"scott"`, `"rice"`, …) |
| `density` | `bool` | `True` | Normalise the histogram to a probability density |
| `color` | `str` | `"C0"` | Bar colour |
| `density_color` | `str` | `"C3"` | Colour of the overlaid $\mathcal{N}(0,1)$ density curve |
| `density_lw` | `float` | `2.0` | Line width of the density curve |
| `kde` | `bool` | `False` | Overlay a kernel density estimate in addition to the parametric $\mathcal{N}(0,1)$ curve |
| `figsize` | `tuple` | `(6, 4)` | Figure size in inches |
| `title` | `str \| None` | `None` | Axes title; defaults to `"Innovation Distribution"` |
| `ax` | `Axes \| None` | `None` | Existing axes to draw on |

### Returns

`matplotlib.figure.Figure`

### Interpretation guide

- **Bars track the red N(0,1) curve closely**: innovations are approximately Gaussian.
- **Fat tails** (bars exceed the curve at the extremes): leptokurtic distribution;
  consider a Student-$t$ or mixture of normals observation model.
- **Asymmetry**: one tail is heavier than the other; investigate outliers or missing
  intervention dummies.
- **Bimodal shape**: the model may be fitting two regimes simultaneously; consider a
  regime-switching or Markov-switching SSM.

### Example

```python
from kalmanbox.visualization import plot_histogram

fig = plot_histogram(results, bins=20, kde=True, figsize=(6, 4))
fig.tight_layout()
```

---

## `plot_cusum`

CUSUM (cumulative sum) and CUSUMSQ (cumulative sum of squares) stability charts. These
plots detect whether the model parameters or error variance change over the sample.

### Signature

```python
from kalmanbox.visualization import plot_cusum

def plot_cusum(
    results,
    kind: str = "cusum",
    alpha: float = 0.05,
    color: str = "C0",
    boundary_color: str = "C3",
    boundary_lw: float = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `KalmanResults` | — | Fitted model results object |
| `kind` | `str` | `"cusum"` | Which statistic to plot: `"cusum"` for the level CUSUM, `"cusumsq"` for the squared-residual CUSUM, or `"both"` for side-by-side panels |
| `alpha` | `float` | `0.05` | Significance level for the boundary lines |
| `color` | `str` | `"C0"` | Colour of the CUSUM path |
| `boundary_color` | `str` | `"C3"` | Colour of the critical boundary lines |
| `boundary_lw` | `float` | `1.5` | Line width of the boundaries |
| `figsize` | `tuple \| None` | `None` | Figure size; defaults to `(8, 4)` for a single panel and `(14, 4)` for `kind="both"` |
| `title` | `str \| None` | `None` | Figure super-title |

### Returns

`matplotlib.figure.Figure`

### CUSUM statistic

The CUSUM statistic accumulates the standardised innovations:

$$
\text{CS}_t = \sum_{s=1}^{t} e_s, \qquad t = 1, \ldots, n
$$

Under parameter stability, $\text{CS}_t / \sqrt{n}$ converges to a standard Brownian motion
on $[0,1]$, giving asymptotic critical boundaries

$$
\pm\!\left(c_\alpha\sqrt{n} + 2c_\alpha \frac{t}{\sqrt{n}}\right)
$$

where $c_{0.05} \approx 1.358$.

### CUSUMSQ statistic

The CUSUMSQ statistic tests for *variance* instability:

$$
\text{CSQ}_t = \frac{\displaystyle\sum_{s=1}^{t} e_s^2}{\displaystyle\sum_{s=1}^{n} e_s^2}
$$

Under stability this is approximately uniform on $[0,1]$, with critical bounds derived from
the Brownian bridge distribution.

!!! warning "CUSUM boundary crossing"
    If the CUSUM path exits the shaded boundary region, reject the null of parameter
    stability at level `alpha`. The *time* of crossing indicates when the break occurred.
    A CUSUM violation typically signals a structural break in the *level* (mean), while a
    CUSUMSQ violation indicates a break in the *variance*. A single boundary crossing near
    the end of the sample can arise from a single outlier — always inspect the innovation
    time-series plot alongside the CUSUM.

### Example

```python
from kalmanbox.visualization import plot_cusum

# CUSUM only (default)
fig = plot_cusum(results, kind="cusum", alpha=0.05)

# Both statistics side by side
fig2 = plot_cusum(results, kind="both", alpha=0.05, figsize=(14, 4))
fig2.suptitle("Structural Stability Tests", fontsize=12)
fig2.tight_layout()
```

---

## `plot_diagnostic_panel`

A single composite 2×2 figure combining all four key diagnostic plots:

1. **Top-left** — Standardised innovation time series with $\pm 1.96$ bands.
2. **Top-right** — QQ-plot with 45° reference line and pointwise confidence band.
3. **Bottom-left** — ACF of standardised innovations with Bartlett significance bounds.
4. **Bottom-right** — Histogram of standardised innovations with $\mathcal{N}(0,1)$ density overlay.

This is the **recommended first-pass diagnostic** after fitting any kalmanbox model.

### Signature

```python
from kalmanbox.visualization import plot_diagnostic_panel

def plot_diagnostic_panel(
    results,
    figsize: tuple[float, float] = (12, 8),
    nlags: int = 24,
    bins: int | str = "auto",
    alpha: float = 0.05,
    color: str = "C0",
    accent_color: str = "C3",
    suptitle: str | None = None,
    tight_layout: bool = True,
) -> "matplotlib.figure.Figure":
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `KalmanResults` | — | Fitted model results object |
| `figsize` | `tuple` | `(12, 8)` | Overall figure size in inches `(width, height)` |
| `nlags` | `int` | `24` | Number of ACF lags to display in panel 3 |
| `bins` | `int \| str` | `"auto"` | Histogram bins for panel 4; passed directly to `plot_histogram` |
| `alpha` | `float` | `0.05` | Significance level for QQ confidence band and ACF bounds; also controls `±z_{1-\alpha/2}` bands in the time-series panel |
| `color` | `str` | `"C0"` | Primary colour used in all panels |
| `accent_color` | `str` | `"C3"` | Accent colour for reference lines (QQ 45° line, ACF bounds, density curve) |
| `suptitle` | `str \| None` | `None` | Figure super-title; set explicitly or use `fig.suptitle()` afterwards |
| `tight_layout` | `bool` | `True` | Call `fig.tight_layout()` before returning |

### Returns

`matplotlib.figure.Figure` — A 2×2 panel figure with axes accessible via
`fig.axes` in row-major order `[ts, qq, acf, hist]`.

### Panel layout

```
┌──────────────────────┬──────────────────────┐
│  Panel 1             │  Panel 2             │
│  Standardised        │  QQ-plot vs          │
│  innovations         │  N(0,1)              │
│  (time series)       │                      │
├──────────────────────┼──────────────────────┤
│  Panel 3             │  Panel 4             │
│  ACF of              │  Histogram +         │
│  innovations         │  N(0,1) density      │
└──────────────────────┴──────────────────────┘
```

**Panel 1 (top-left): Innovation time series**

The standardised innovations $e_t$ are plotted as a line (or scatter for short samples)
with a horizontal zero line and dashed $\pm 1.96$ reference bands. Colour-coded dots
outside the bands draw immediate attention to potential outliers.

**Panel 2 (top-right): QQ-plot**

Empirical quantiles of $e_t$ versus theoretical quantiles of $\mathcal{N}(0,1)$.  
The red 45° reference line represents perfect normality; the shaded region is the
pointwise $(1-\alpha)$ confidence band computed via the order-statistic distribution.

**Panel 3 (bottom-left): ACF**

Autocorrelation function of $e_t$ at lags $1, \ldots, \texttt{nlags}$.  
The dashed horizontal lines at $\pm z_{1-\alpha/2}/\sqrt{n}$ are the Bartlett asymptotic
significance bounds. The lag-0 autocorrelation (always 1) is suppressed.

**Panel 4 (bottom-right): Histogram**

A normalised histogram of $e_t$ with the $\mathcal{N}(0,1)$ density overlaid as a solid
red curve. The histogram bars should track the bell curve closely for a well-specified model.

### Example

```python
from kalmanbox.visualization import plot_diagnostic_panel

fig = plot_diagnostic_panel(
    results,
    figsize=(12, 8),
    nlags=24,
    bins=20,
    alpha=0.05,
)
fig.suptitle("Model Residual Diagnostics", fontsize=14, y=1.01)
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
```

### Accessing individual axes

```python
ax_ts, ax_qq, ax_acf, ax_hist = fig.axes

# Add a custom annotation to the time-series panel
ax_ts.axvline(x=60, color="grey", linestyle="--", label="Intervention")
ax_ts.legend()
```

---

## Interpreting the diagnostic panel

### What good residuals look like

A well-specified model produces standardised innovations that are:

- **Centred at zero** with no visible drift or trend (panel 1).
- **Homoscedastic**: the spread is constant throughout the sample (panel 1).
- **Serially uncorrelated**: all ACF bars fall inside the significance bounds (panel 3).
- **Normally distributed**: QQ-plot points follow the 45° line and the histogram bars
  match the $\mathcal{N}(0,1)$ density (panels 2 and 4).

### Pathological patterns and remedies

The table below catalogs the most common misspecification signatures and their likely causes.

| Pattern | Likely cause | Suggested remedy |
|---|---|---|
| Spike in ACF at lag 1 | Under-differenced data or missing MA(1) component | Add an AR/MA term; check integration order |
| Spike in ACF at lag $s$ (seasonal) | Missing seasonal component at period $s$ | Add `seasonal_periods=s` to BSM / UCM |
| Slow geometric decay in ACF | Strong AR structure unexplained by the model | Add AR component or use ARIMA-SSM |
| Positive skewness in histogram | Outlier or asymmetric error distribution | Add intervention dummy; consider log transform |
| Heavy tails in QQ-plot | Leptokurtic errors (non-Gaussian) | Use Student-$t$ observation model |
| Light tails in QQ-plot | Over-dispersed model or data censoring | Review $H_t$ specification |
| Trend or drift in innovation series | Missing level or slope component | Add stochastic level/trend |
| Sudden variance shift in panel 1 | Heteroscedastic errors or structural break | Use stochastic volatility; check CUSUM |
| Bimodal histogram | Regime switching or data mixture | Markov-switching SSM |
| Innovations correlated with fitted values | Non-linear dynamics | EKF / UKF; linearise model |
| Large isolated outliers | Measurement error or data error | Outlier-robust filter; add intervention |

!!! warning "One diagnostic rarely tells the full story"
    A clean ACF does not guarantee normality; a near-normal histogram does not rule out
    serial correlation at seasonal frequencies. Always inspect all four panels together,
    and back them up with the formal tests in
    [Innovation Tests](../diagnostics/innovation-tests.md).

### Multiple testing caveat

With $n = 40$ ACF lags plotted, you expect approximately $40 \times 0.05 = 2$ bars outside
the bounds by pure chance under the null. A single marginally significant spike should not
trigger immediate model revision; a *cluster* of spikes or a spike at a theoretically
motivated lag (e.g., lag 12 for monthly data) deserves attention.

---

## CUSUM interpretation

The CUSUM and CUSUMSQ plots complement the four-panel diagnostic by examining **parameter
stability** rather than distributional properties.

### Reading the CUSUM chart

The CUSUM path starts at zero and accumulates the standardised innovations step by step.
Under the null of parameter stability the path performs a zero-mean random walk and stays
within the wedge-shaped critical boundaries with probability $1 - \alpha$.

- **Path within the boundaries**: no evidence of a structural break at level $\alpha$.
- **Path crosses the upper boundary**: a *positive* mean shift has occurred; the level of
  $y_t$ has jumped upward.
- **Path crosses the lower boundary**: a *negative* mean shift; the level has dropped.
- **Path oscillates before crossing**: the break may be gradual (parameter drift) rather
  than abrupt.
- **The time at which the path first exits the boundary** is the estimated break date.

!!! warning "CUSUM boundary crossing"
    A confirmed CUSUM boundary crossing is strong evidence of model misspecification or a
    true structural change in the data-generating process. Possible responses:
    
    1. Add an **intervention variable** (level shift dummy) at the estimated break date.
    2. Extend the model with a **time-varying parameter** (TVP) for the relevant coefficient.
    3. Split the sample at the break date and estimate separate models for each sub-period.
    4. Use a **Markov-switching** or **threshold** state-space model.

### Reading the CUSUMSQ chart

The CUSUMSQ path is the cumulative squared-innovation share — it should rise linearly from 0
to 1 if the error variance is constant throughout the sample.

- **Path hugs the diagonal** $\text{CSQ}_t \approx t/n$: variance is stable.
- **Path bulges above the diagonal early**: variance is *higher* in the first part of the
  sample than the second (decreasing volatility).
- **Path bulges below the diagonal early**: variance is *lower* initially and increases later
  (increasing volatility or a volatility break at some interior point).
- **Path exits the confidence band**: reject variance stability at level $\alpha$.

CUSUMSQ crossing without CUSUM crossing is a pure variance break; both crossing simultaneously
suggests a mean *and* variance break (e.g., a level shift with accompanying increase in
uncertainty).

### Example: CUSUM for BSM

```python
from kalmanbox.visualization import plot_cusum

# Single CUSUM plot
fig_cs = plot_cusum(results, kind="cusum", alpha=0.05)

# Both CUSUM and CUSUMSQ for comprehensive stability analysis
fig_both = plot_cusum(results, kind="both", figsize=(14, 4))
fig_both.suptitle("Parameter Stability Tests", fontsize=12)
fig_both.tight_layout()
```

!!! info "Relation to formal tests"
    The CUSUM plot is the graphical version of the **Brown–Durbin–Evans** test. The
    associated p-value from `results.summary()` uses the asymptotic Brownian motion
    distribution. For small samples ($n < 50$), bootstrap critical values may be more
    reliable — see [CUSUM and Structural-Break Tests](../diagnostics/cusum.md).

---

## Complete diagnostic workflow

The recommended workflow after fitting any kalmanbox model is to run the four-panel
diagnostic panel, inspect the CUSUM plots, and then optionally run the formal tests for
a quantitative assessment.

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.visualization import (
    plot_diagnostic_panel,
    plot_cusum,
    plot_innovations,
    set_theme,
)

# ── 1. Set a consistent visual theme ────────────────────────────────────────
set_theme("whitegrid")          # seaborn-v0_8-whitegrid under the hood

# ── 2. Simulate monthly data with a trend break at observation 60 ────────────
rng = np.random.default_rng(42)
n = 120
t = np.arange(n)
level = np.where(t < 60, 0.05 * t, 3.0 + 0.02 * t)   # slope changes at t=60
y = level + rng.normal(0, 0.8, n)

# ── 3. Fit a Basic Structural Model ─────────────────────────────────────────
model = BSM(y, seasonal_periods=12)
results = model.fit(method="mle", verbose=False)

print(results.summary())
```

```
                        BSM Model Results
=================================================================
Dep. Variable:                  y   Log Likelihood:    -142.31
No. Observations:             120   AIC:               294.62
State dimension:                5   BIC:               307.94
──────────────────────────────────────────────────────────────────
Component            Variance    Std Err   [0.025    0.975]
──────────────────────────────────────────────────────────────────
Irregular             0.5893      0.1021    0.3892    0.7893
Level                 0.0012      0.0004    0.0005    0.0020
Slope                 0.0008      0.0003    0.0003    0.0014
Seasonal (12)         0.0003      0.0001    0.0001    0.0005
=================================================================
```

The log-likelihood and information criteria give a point estimate of model quality.
The real diagnostic value comes from the residual plots.

```python
# ── 4. Four-panel diagnostic ─────────────────────────────────────────────────
fig = plot_diagnostic_panel(
    results,
    figsize=(12, 8),
    nlags=24,      # 2 years of monthly lags
    bins=20,
    alpha=0.05,
)
fig.suptitle("BSM Residual Diagnostics — Monthly Data", fontsize=14, y=1.01)
fig.savefig("bsm_diagnostics.png", dpi=150, bbox_inches="tight")
```

The four-panel figure tells a quick story:

- **Panel 1** (innovation time series): Look for any systematic trend or step change.
  In this example, a subtle mean shift around observation 60 may be visible after
  standardisation — the model partially absorbs it through the stochastic slope, but
  residual evidence may remain.
- **Panel 2** (QQ-plot): Points along the 45° line indicate the $\mathcal{N}(0,1)$
  assumption is met. Extreme points curling away from the reference line in both tails
  indicate leptokurtosis.
- **Panel 3** (ACF): All bars within the $\pm 1.96/\sqrt{120} \approx \pm 0.18$ bounds
  confirm the model has absorbed the autocorrelation structure. A spike at lag 12 here
  would indicate a missing or poorly estimated seasonal component.
- **Panel 4** (histogram): The histogram bars should approximate the red $\mathcal{N}(0,1)$
  bell curve. A bimodal shape or heavy tails require model revision.

```python
# ── 5. Stability check with CUSUM ────────────────────────────────────────────
fig_cusum = plot_cusum(results, kind="both", alpha=0.05, figsize=(14, 4))
fig_cusum.suptitle("BSM Structural Stability", fontsize=12)
fig_cusum.tight_layout()
fig_cusum.savefig("bsm_cusum.png", dpi=150, bbox_inches="tight")
```

!!! warning "Interpreting the CUSUM output for this example"
    Because the true data-generating process has a slope change at $t = 60$, the CUSUM
    path is expected to show a departure from zero near that break date. If the BSM slope
    component has captured the change adequately, the path should remain within the
    boundaries. If not, an intervention or a TVP extension is warranted.

```python
# ── 6. Zoom in on the innovation series ──────────────────────────────────────
fig_ts = plot_innovations(
    results,
    standardize=True,
    sigma=2.0,
    figsize=(11, 3),
)
fig_ts.suptitle("Standardised Innovations", fontsize=12, y=1.02)
fig_ts.tight_layout()
```

```python
# ── 7. Formal tests to back up the visual ────────────────────────────────────
from kalmanbox.diagnostics import innovation_tests

tests = innovation_tests(results)
print(tests)
```

```
Innovation Diagnostics
──────────────────────────────────────────────────
  Jarque-Bera (normality)   statistic=1.83  p=0.401
  Ljung-Box (Q, lag=24)     statistic=22.1  p=0.571
  Heteroscedasticity (H)    statistic=1.21  p=0.318
  CUSUM (Brown-Durbin)      p=0.083
──────────────────────────────────────────────────
```

!!! info "Reading the formal tests"
    All four p-values exceed 0.05, so the null hypotheses of normality, no autocorrelation,
    constant variance, and parameter stability are not rejected at the 5 % level. This
    corroborates the visual diagnostics. If a test is borderline (e.g., the CUSUM p = 0.083),
    inspect the plots in detail before deciding whether to revise the model.

---

## Related

- [Diagnostics: Innovation Tests](../diagnostics/innovation-tests.md) — formal Jarque-Bera,
  Ljung–Box, and heteroscedasticity tests.
- [Diagnostics: CUSUM and Structural Breaks](../diagnostics/cusum.md) — theory and
  critical values for CUSUM / CUSUMSQ.
- [Diagnostics: Prediction Error Decomposition](../diagnostics/prediction-error.md) —
  likelihood-based diagnostics using the innovation decomposition.
- [Visualization: Filtered States](filtered-states.md) — plot $a_{t|t}$ alongside $y_t$.
- [Visualization: Smoothed States](smoothed-states.md) — plot $a_{t|n}$ after the smoother.
- [Choosing a model](../getting-started/choosing-model.md) — decision guide when diagnostics
  reveal systematic misspecification.
