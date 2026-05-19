# Filter Comparison Plots

When you apply multiple filters — Kalman, EKF, UKF, EnKF — to the same data, their filtered
states, covariance traces, and innovations will differ because each filter makes different
approximations. `kalmanbox.visualization` provides four dedicated plotting functions that
overlay these results on shared axes so you can immediately see where the filters agree, where
they diverge, and why.

Comparing filters is especially important for **nonlinear problems**: the standard Kalman
filter assumes linearity and will produce biased estimates when that assumption fails; the EKF
corrects for mild nonlinearity via first-order Taylor linearization; the UKF corrects to
third order via sigma points; and the EnKF handles high-dimensional strongly nonlinear systems
via Monte Carlo. Plotting their outputs side by side makes linearization errors, covariance
inflation, and divergence visible without running a formal benchmark.

---

## When to compare filters

| Scenario | What you learn from comparison |
|---|---|
| **Linear Gaussian** | KF is optimal; EKF and UKF should match KF exactly — divergence signals a coding error |
| **Mildly nonlinear** ($\delta \ll 1$) | EKF vs UKF: small but measurable difference in covariance; NEES shows whether linearization matters |
| **Moderately nonlinear** | EKF vs UKF: visible state estimate gap; UKF covariance tighter and better calibrated |
| **Strongly nonlinear** | UKF vs EKF: EKF may diverge; UKF or EnKF remains consistent |
| **High-dimensional nonlinear** | EnKF vs UKF: UKF scales as $O(k^3)$ and becomes infeasible; EnKF is the only practical option |
| **Tuning process noise** | `plot_nees` over a grid of $Q$ values pinpoints the setting that achieves consistency |

!!! tip "Start with `plot_filter_comparison`"
    A single overlay plot of filtered states is usually the first diagnostic. If KF and UKF
    agree visually, the system is linear enough that the simpler filter is adequate. If they
    differ, proceed to `plot_covariance_trace` and `plot_nees` to understand the cause.

---

## Setting up `results_dict`

All four functions accept a `results_dict` argument: a plain Python `dict` that maps a
**human-readable filter name** (the legend label) to a `FilterResults` object returned by
that filter's `.filter()` method.

```python
import numpy as np
from kalmanbox import KalmanFilter
from kalmanbox.filters import EKF, UKF
from kalmanbox.models import StateSpaceModel

# ── Define the state-space model ──────────────────────────────────────────────
# For a nonlinear range/bearing example all three share the same Q and R.
Q = np.diag([0.1, 0.1, 0.05, 0.05])   # 4-state: position (x,y), velocity (vx,vy)
R = np.diag([1.0, np.deg2rad(2.0)])    # range [m] and bearing [rad]

# Transition and observation functions (nonlinear)
def f(x, dt=1.0):
    """Constant-velocity transition."""
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    return F @ x

def h(x):
    """Range / bearing observation."""
    rng = np.sqrt(x[0]**2 + x[1]**2)
    brg = np.arctan2(x[1], x[0])
    return np.array([rng, brg])

# ── Run three filters on the same observation sequence y ──────────────────────
ssm = StateSpaceModel(...)   # linear approximation for KF baseline
kf_res  = KalmanFilter(ssm).filter(y)
ekf_res = EKF(f, h, Q, R).filter(y)
ukf_res = UKF(f, h, Q, R, alpha=1e-3, beta=2, kappa=0).filter(y)

# ── Build results_dict ────────────────────────────────────────────────────────
results = {"Kalman": kf_res, "EKF": ekf_res, "UKF": ukf_res}
```

Every value in `results_dict` must expose at minimum:

| Attribute | Shape | Meaning |
|---|---|---|
| `filtered_states` | `(T, k)` | Filtered mean $a_{t\|t}$ |
| `filtered_covariances` | `(T, k, k)` | Filtered covariance $P_{t\|t}$ |
| `innovations` | `(T, p)` | Innovation $v_t = y_t - \hat{y}_t$ |
| `innovation_covariances` | `(T, p, p)` | Innovation covariance $F_t$ |

`FilterResults` objects returned by all built-in filters satisfy this interface
automatically.

---

## `plot_filter_comparison`

Overlay the filtered state estimates from two or more filters on a single axes so that
differences in the estimated mean trajectory are immediately visible.

### Signature

```python
kalmanbox.visualization.plot_filter_comparison(
    results_dict: dict[str, FilterResults],
    component: int | str = 0,
    *,
    observations: np.ndarray | None = None,
    true_states: np.ndarray | None = None,
    show_bands: bool = True,
    alpha_fill: float = 0.15,
    ci_level: float = 0.95,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure
```

### Parameters

`results_dict` : `dict[str, FilterResults]`
:   Maps a filter label to its `FilterResults` object.
    Example: `{"KF": kf_res, "EKF": ekf_res, "UKF": ukf_res}`.
    Dict insertion order determines drawing order (and therefore legend order).

`component` : `int | str`, default `0`
:   Which state component to display.
    Pass an integer index (e.g. `2` for the third state), or a named component
    string that is looked up via `FilterResults.state_names` (if available).

`observations` : `np.ndarray | None`, default `None`
:   Raw observation vector, shape `(T,)` or `(T, p)`. When provided, column 0
    is plotted as grey dots behind the filter lines for context.

`true_states` : `np.ndarray | None`, default `None`
:   Ground-truth state trajectory from simulation, shape `(T, k)`. When provided,
    the selected component is drawn as a dashed black line.

`show_bands` : `bool`, default `True`
:   Draw $\pm z_{\alpha/2}\sqrt{P_{t|t,ii}}$ shaded bands around each filter's mean.
    Bands make covariance differences between filters immediately visible — a wide
    band from the EKF and a narrow band from the UKF indicate linearization
    inflates EKF uncertainty.

`alpha_fill` : `float`, default `0.15`
:   Opacity of the shaded confidence bands. Reduce to `0.08` when many filters
    overlap; increase to `0.3` for two-filter plots.

`ci_level` : `float`, default `0.95`
:   Confidence level for the shaded bands (default gives $z_{0.025} = 1.96$
    standard deviations).

`colors` : `list[str] | None`, default `None`
:   Custom colours for each filter in the same order as `results_dict.keys()`.
    Defaults to matplotlib's `tab10` palette.

`linestyles` : `list[str] | None`, default `None`
:   Line styles for each filter, e.g. `["-", "--", ":"]`. Useful for
    black-and-white figures.

`figsize` : `tuple[float, float]`, default `(10, 4)`
:   Figure width and height in inches.

`title` : `str | None`, default `None`
:   Axes title. Defaults to `"Filtered state — component {component}"`.

`xlabel`, `ylabel` : `str`
:   Axis labels. `ylabel` defaults to the component name or `f"State {component}"`.

`ax` : `matplotlib.axes.Axes | None`, default `None`
:   Existing axes to draw into. When `None` a new `Figure` is created and returned.
    When an axes is supplied the function still returns the parent `Figure`.

### Returns

`matplotlib.figure.Figure` — the figure containing the overlay plot.

### Visual description

The axes shows one line per filter, drawn in a distinct colour, with a shaded confidence
band of the same colour at reduced opacity. If `observations` is provided, raw data
appear as small grey dots. If `true_states` is provided, the ground truth appears as a
dashed black line behind all filter lines. The legend identifies each filter by its key
in `results_dict`.

### Example

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_filter_comparison

# results is the dict built in "Setting up results_dict"
fig = plot_filter_comparison(
    results,
    component=0,            # x-position
    observations=y,
    true_states=true_x,
    show_bands=True,
    alpha_fill=0.12,
    figsize=(11, 4),
    title="Range/Bearing Tracking — x position",
)
fig.tight_layout()
fig.savefig("filter_comparison.png", dpi=150)
plt.show()
```

To compare all state components in a grid:

```python
from kalmanbox.visualization import plot_filter_comparison
import matplotlib.pyplot as plt

state_names = ["x [m]", "y [m]", "vx [m/s]", "vy [m/s]"]
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for ax, (comp, name) in zip(axes.flat, enumerate(state_names)):
    plot_filter_comparison(
        results,
        component=comp,
        true_states=true_x,
        show_bands=True,
        title=name,
        ax=ax,
    )

fig.suptitle("Filter comparison — all state components", fontsize=14)
fig.tight_layout()
fig.savefig("filter_comparison_grid.png", dpi=150)
```

!!! tip "Choosing `component` by name"
    If you set `state_names` on your model (e.g. `ssm.state_names = ["x", "y", "vx", "vy"]`),
    pass the string directly: `plot_filter_comparison(results, component="vx")`. This is
    especially helpful in scripts where integer indices are error-prone.

---

## `plot_covariance_trace`

Plot the **trace of the filtered covariance matrix**, $\operatorname{tr}(P_{t|t})$, over
time for each filter in `results_dict`. The trace is the sum of all marginal variances and
serves as a scalar summary of the total estimation uncertainty.

### Signature

```python
kalmanbox.visualization.plot_covariance_trace(
    results_dict: dict[str, FilterResults],
    *,
    log_scale: bool = False,
    normalize: bool = False,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = r"$\operatorname{tr}(P_{t|t})$",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure
```

### Parameters

`results_dict` : `dict[str, FilterResults]`
:   Maps filter name to results. The function extracts `filtered_covariances`
    of shape `(T, k, k)` and computes `np.trace(P[t])` for each $t$.

`log_scale` : `bool`, default `False`
:   Use a logarithmic $y$-axis. Useful when the trace spans several orders of
    magnitude (e.g. during diffuse initialisation or after a sudden observation gap).

`normalize` : `bool`, default `False`
:   Normalise each filter's trace by its initial value $\operatorname{tr}(P_{0|0})$
    so that all curves start at 1. Highlights *relative* uncertainty reduction rather
    than absolute scale.

`colors` : `list[str] | None`, default `None`
:   Per-filter colours. Defaults to `tab10`.

`linestyles` : `list[str] | None`, default `None`
:   Per-filter line styles.

`figsize` : `tuple[float, float]`, default `(10, 4)`
:   Figure dimensions in inches.

`title` : `str | None`, default `None`
:   Axes title. Defaults to `"Covariance trace — total estimation uncertainty"`.

`xlabel`, `ylabel` : `str`
:   Axis labels.

`ax` : `matplotlib.axes.Axes | None`, default `None`
:   Existing axes to draw into. When `None` a new `Figure` is created.

### Returns

`matplotlib.figure.Figure`

### Visual description

Each filter produces a line showing how its total uncertainty $\operatorname{tr}(P_{t|t})$
evolves over the observation window. Key visual patterns:

- **Rapid initial drop**: all filters shed uncertainty quickly once early observations
  arrive, regardless of the initialisation.
- **Steady-state plateau**: for time-invariant systems the trace converges to a fixed
  value — the Riccati solution. Filters that reach the same steady state are
  asymptotically equivalent.
- **Persistent gap between filters**: if the EKF trace is uniformly higher than the
  UKF trace, linearisation is inflating the EKF's uncertainty at every step.
- **Spike at missing observations**: the trace rises during observation gaps (no update
  is made) and falls when observations resume.

### Example

```python
from kalmanbox.visualization import plot_covariance_trace

fig = plot_covariance_trace(
    results,
    log_scale=False,
    normalize=True,      # compare relative decay rates
    figsize=(10, 4),
    title="Relative uncertainty decay — KF vs EKF vs UKF",
)
fig.savefig("cov_trace.png", dpi=150)
```

For a two-panel figure showing both raw and normalised traces:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_covariance_trace

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
plot_covariance_trace(results, ax=ax1, title="Absolute trace")
plot_covariance_trace(results, normalize=True, ax=ax2, title="Normalised trace")
fig.tight_layout()
fig.savefig("cov_trace_panels.png", dpi=150)
```

!!! tip "Steady-state covariance"
    For linear time-invariant systems you can also compute the analytical steady-state
    covariance by solving the discrete algebraic Riccati equation with
    `scipy.linalg.solve_discrete_are`. Superimposing that constant level on the
    `plot_covariance_trace` figure immediately shows how many time steps each filter
    needs to converge.

---

## `plot_innovation_comparison`

Plot the innovation sequences $v_t = y_t - \hat{y}_{t|t-1}$ from multiple filters on
shared axes. Innovations that are white and centred on zero are the hallmark of a
consistent filter; systematic patterns reveal model misspecification or filter divergence.

### Signature

```python
kalmanbox.visualization.plot_innovation_comparison(
    results_dict: dict[str, FilterResults],
    obs_component: int = 0,
    *,
    standardize: bool = True,
    show_zero: bool = True,
    show_bands: bool = True,
    band_sigma: float = 2.0,
    colors: list[str] | None = None,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure
```

### Parameters

`results_dict` : `dict[str, FilterResults]`
:   Maps filter name to results. The function reads `innovations` (shape `(T, p)`)
    and, when `standardize=True`, also reads `innovation_covariances` (shape `(T, p, p)`).

`obs_component` : `int`, default `0`
:   Which observation dimension to display. For a scalar observation pass `0`.
    For a multivariate observation, call the function once per component or loop over
    components.

`standardize` : `bool`, default `True`
:   Divide each innovation $v_{t,j}$ by its one-step-ahead standard deviation
    $\sqrt{F_{t,jj}}$ before plotting, so all filters are on a common $N(0,1)$ scale.
    This is almost always preferable because it removes the dependence on the
    observation noise scale and makes cross-filter comparison fair.

`show_zero` : `bool`, default `True`
:   Draw a horizontal dashed line at zero.

`show_bands` : `bool`, default `True`
:   Draw horizontal dashed lines at $\pm$`band_sigma` (default $\pm 2$).
    When `standardize=True` these are $\pm 2\sigma$ bounds; approximately 5% of
    standardised innovations should exceed these bounds under the null.

`band_sigma` : `float`, default `2.0`
:   Number of standard deviations for the reference bands.

`colors` : `list[str] | None`, default `None`
:   Per-filter colours.

`figsize` : `tuple[float, float]`, default `(10, 4)`
:   Figure dimensions.

`title` : `str | None`, default `None`
:   Axes title. Defaults to `"Standardised innovations"` or `"Raw innovations"`.

`xlabel`, `ylabel` : `str`
:   Axis labels. `ylabel` defaults to `"Std. innovation"` or `"Innovation"`.

`ax` : `matplotlib.axes.Axes | None`, default `None`
:   Existing axes to plot into.

### Returns

`matplotlib.figure.Figure`

### Visual description

Each filter's (standardised) innovation sequence is plotted as a thin line in its
assigned colour. Reference lines at 0 and $\pm 2\sigma$ are drawn in dashed grey.
Under a consistent filter the innovations should scatter randomly around zero within
the bands. Visual patterns to look for:

- **Persistent positive or negative bias**: the filter's predicted observation
  $\hat{y}_{t|t-1}$ is systematically off — the process model may be incorrect.
- **Slow oscillation or autocorrelation**: the filter is not extracting all
  predictable signal; check for missed dynamics or incorrect $Q$.
- **One filter's innovations centred closer to zero**: that filter's state prediction
  better captures the system's dynamics.
- **Different scale before standardisation**: a filter with a larger raw innovation
  variance is less confident about its predictions — usually the EKF vs UKF case.

### Example

```python
from kalmanbox.visualization import plot_innovation_comparison

# Standardised innovations for observation component 0 (range)
fig = plot_innovation_comparison(
    results,
    obs_component=0,
    standardize=True,
    band_sigma=2.0,
    figsize=(10, 4),
    title="Standardised range innovations — KF vs EKF vs UKF",
)
fig.savefig("innovations.png", dpi=150)
```

For a multivariate observation (range and bearing), compare both components:

```python
import matplotlib.pyplot as plt
from kalmanbox.visualization import plot_innovation_comparison

obs_labels = ["Range [m]", "Bearing [rad]"]
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for ax, (comp, label) in zip(axes, enumerate(obs_labels)):
    plot_innovation_comparison(
        results,
        obs_component=comp,
        standardize=True,
        title=label,
        ax=ax,
    )

fig.suptitle("Standardised innovations — all observation components", fontsize=13)
fig.tight_layout()
fig.savefig("innovations_multivariate.png", dpi=150)
```

!!! warning "Do not compare raw innovations across filters"
    Raw innovation magnitudes depend on $F_t = H_t P_{t|t-1} H_t^T + R$, which differs
    between filters because their $P_{t|t-1}$ differ. Always set `standardize=True` for
    cross-filter comparison; use `standardize=False` only when inspecting a single filter
    in the units of the original observation.

---

## `plot_nees`

Plot the **Normalized Estimation Error Squared (NEES)** over time for one or more filters,
together with chi-squared consistency bounds. NEES is the gold-standard scalar metric for
assessing whether a filter's covariance is correctly calibrated relative to the actual
estimation errors.

### Signature

```python
kalmanbox.visualization.plot_nees(
    results_dict: dict[str, FilterResults],
    true_states: np.ndarray,
    *,
    dof: int | None = None,
    ci_level: float = 0.95,
    window: int | None = None,
    time_average: bool = False,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "NEES",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure
```

### Parameters

`results_dict` : `dict[str, FilterResults]`
:   Maps filter name to results. The function reads `filtered_states` and
    `filtered_covariances` and combines them with `true_states` to compute NEES.

`true_states` : `np.ndarray`, shape `(T, k)`
:   Ground-truth state trajectory from simulation. NEES is only computable when the
    truth is known; this restricts the function to simulation studies and
    Monte Carlo evaluation scenarios.

`dof` : `int | None`, default `None`
:   Degrees of freedom for the chi-squared bounds, i.e. the state dimension $k$.
    If `None`, inferred from `true_states.shape[1]`.

`ci_level` : `float`, default `0.95`
:   Confidence level for the chi-squared consistency band. The function draws the
    $((1-\text{ci\_level})/2)$-th and $(1+\text{ci\_level})/2$-th quantiles of
    $\chi^2(k)$ as horizontal dashed lines.

`window` : `int | None`, default `None`
:   When set to a positive integer, plot the **rolling NEES** — the mean of NEES over
    a sliding window of `window` time steps. Rolling NEES smooths the highly variable
    pointwise NEES and makes trends in filter consistency visible.
    When `None`, the pointwise NEES is plotted.

`time_average` : `bool`, default `False`
:   If `True`, also draw the time-averaged NEES
    $\bar{\epsilon} = \frac{1}{T}\sum_t \epsilon_t$ as a horizontal solid line
    (per filter). Under consistency $\bar{\epsilon}$ should lie within the
    $\chi^2(k)$ confidence interval. Setting `window` and `time_average=True`
    simultaneously is useful: the rolling curve shows temporal trends while the
    horizontal line gives the overall verdict.

`colors` : `list[str] | None`, default `None`
:   Per-filter colours.

`linestyles` : `list[str] | None`, default `None`
:   Per-filter line styles.

`figsize` : `tuple[float, float]`, default `(10, 4)`
:   Figure dimensions.

`title` : `str | None`, default `None`
:   Axes title. Defaults to `"NEES — filter consistency"`.

`xlabel`, `ylabel` : `str`
:   Axis labels.

`ax` : `matplotlib.axes.Axes | None`, default `None`
:   Existing axes.

### Returns

`matplotlib.figure.Figure`

### Example

```python
from kalmanbox.visualization import plot_nees

# true_x is the ground-truth trajectory used to generate y
fig = plot_nees(
    results,
    true_states=true_x,
    dof=4,          # 4-state: (x, y, vx, vy)
    ci_level=0.95,
    window=20,      # rolling 20-step NEES
    time_average=True,
    figsize=(10, 4),
    title="NEES — range/bearing tracking (window=20)",
)
fig.savefig("nees.png", dpi=150)
```

---

## NEES: theory and interpretation

### Definition

Given the true state $x_t$ and the filter's estimate $\hat{x}_{t|t}$ with covariance
$P_{t|t}$, the **Normalized Estimation Error Squared** at time $t$ is:

$$
\epsilon_t = (x_t - \hat{x}_{t|t})^T P_{t|t}^{-1} (x_t - \hat{x}_{t|t})
$$

For a **consistent** filter — one whose covariance correctly describes the uncertainty
in its estimates — the error $x_t - \hat{x}_{t|t}$ is Gaussian with mean zero and
covariance $P_{t|t}$. It follows that:

$$
\epsilon_t \sim \chi^2(k)
$$

where $k$ is the state dimension (the `dof` parameter). The expected value under
consistency is therefore:

$$
\mathbb{E}[\epsilon_t] = k
$$

### Consistency bounds

The $100(1-\alpha)\%$ consistency interval for the **pointwise** NEES at a single time
step is:

$$
\left[\chi^2_{k,\,\alpha/2},\; \chi^2_{k,\,1-\alpha/2}\right]
$$

For example, with $k = 4$ and $\alpha = 0.05$:

$$
\left[\chi^2_{4,\,0.025},\; \chi^2_{4,\,0.975}\right] = [0.484,\; 11.14]
$$

Because pointwise NEES is highly variable, it is common to average over $N$ Monte Carlo
runs (or over a time window of length $W$), giving the **time-averaged NEES**
$\bar\epsilon$. Its consistency interval is:

$$
\left[\frac{\chi^2_{Nk,\,\alpha/2}}{N},\; \frac{\chi^2_{Nk,\,1-\alpha/2}}{N}\right]
$$

which narrows as $N$ or $W$ increases. The `window` parameter in `plot_nees` approximates
this by computing a rolling mean over $W$ consecutive time steps.

### Reading the NEES plot

| NEES behaviour | Interpretation | Likely cause | Remedy |
|---|---|---|---|
| All filters within bounds | Consistent | — | None needed |
| $\epsilon_t \gg \chi^2_{k,\,0.975}$ persistently | **Overconfident** — filter underestimates uncertainty | $Q$ too small, $R$ too small, or nonlinearity ignored | Increase $Q$; switch to UKF/EnKF |
| $\epsilon_t \ll \chi^2_{k,\,0.025}$ persistently | **Underconfident** — filter overestimates uncertainty | $Q$ too large, or covariance inflation applied excessively | Decrease $Q$; remove artificial inflation |
| $\epsilon_t$ rising over time | **Filter diverging** — errors compound | Linearization error (EKF), $Q$ misspecification | Switch to UKF; re-estimate $Q$ via MLE |
| EKF overconfident, UKF consistent | Linearization inflates EKF variance but not means | Moderate nonlinearity | Prefer UKF for this system |

!!! warning "NEES requires known truth"
    NEES is only meaningful in **simulation studies** where the ground truth $x_t$ is
    available. On real data, use the **NIS** (Normalized Innovation Squared) instead:
    $\eta_t = v_t^T F_t^{-1} v_t \sim \chi^2(p)$. The NIS is computable from filter
    output alone and tests the same consistency hypothesis on the observation side.
    See [Diagnostics: NEES/NIS](../diagnostics/consistency.md) for details.

!!! warning "Single-run NEES is noisy"
    A single simulation run produces one NEES sequence with high variance. A NEES value
    above the upper bound at a single time step does not indicate filter failure — only
    a persistent or systematic exceedance does. Use `window` to smooth the sequence, and
    repeat over many Monte Carlo runs to obtain statistically reliable conclusions.

---

## Complete comparison example

The following self-contained example simulates a nonlinear range/bearing tracking problem,
runs three filters, and produces all four comparison plots.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from kalmanbox import KalmanFilter
from kalmanbox.filters import EKF, UKF
from kalmanbox.models import StateSpaceModel
from kalmanbox.visualization import (
    plot_filter_comparison,
    plot_covariance_trace,
    plot_innovation_comparison,
    plot_nees,
    set_theme,
)

set_theme("publication")
rng = np.random.default_rng(0)

# ── System definition ─────────────────────────────────────────────────────────
T   = 120          # time steps
dt  = 1.0
k   = 4            # state dim: (x, y, vx, vy)
p   = 2            # obs dim:   (range, bearing)

Q = np.diag([0.0, 0.0, 0.5, 0.5])   # velocity noise
R = np.diag([2.0, np.deg2rad(3)])    # range [m], bearing [rad]

F_lin = np.array([[1, 0, dt, 0],     # linear transition matrix
                   [0, 1, 0, dt],
                   [0, 0, 1,  0],
                   [0, 0, 0,  1]])

def f(x):
    return F_lin @ x

def h(x):
    return np.array([np.sqrt(x[0]**2 + x[1]**2),
                     np.arctan2(x[1], x[0])])

# ── Simulate true trajectory ──────────────────────────────────────────────────
true_x = np.zeros((T, k))
true_x[0] = [100.0, 200.0, -2.0, 1.5]
for t in range(1, T):
    true_x[t] = f(true_x[t - 1]) + rng.multivariate_normal(np.zeros(k), Q)

y = np.array([h(true_x[t]) + rng.multivariate_normal(np.zeros(p), R)
              for t in range(T)])

# ── Linearised H matrix for KF (at nominal range 220 m, bearing 63°) ─────────
x0 = true_x[0]
r0 = np.sqrt(x0[0]**2 + x0[1]**2)
H_lin = np.array([[ x0[0]/r0,   x0[1]/r0,  0, 0],
                   [-x0[1]/r0**2, x0[0]/r0**2, 0, 0]])

ssm = StateSpaceModel(
    transition=F_lin,
    observation=H_lin,
    process_noise=Q,
    observation_noise=R,
    initial_mean=np.array([90.0, 190.0, -1.8, 1.2]),
    initial_cov=np.diag([50.0, 50.0, 5.0, 5.0]),
)

# ── Run filters ───────────────────────────────────────────────────────────────
kf_res  = KalmanFilter(ssm).filter(y)
ekf_res = EKF(f, h, Q, R,
              initial_mean=ssm.initial_mean,
              initial_cov=ssm.initial_cov).filter(y)
ukf_res = UKF(f, h, Q, R,
              alpha=1e-3, beta=2, kappa=0,
              initial_mean=ssm.initial_mean,
              initial_cov=ssm.initial_cov).filter(y)

results = {"KF": kf_res, "EKF": ekf_res, "UKF": ukf_res}

# ── Plot 1: filtered state comparison ────────────────────────────────────────
fig1 = plot_filter_comparison(
    results,
    component=0,          # x-position
    true_states=true_x,
    observations=y[:, 0], # range as surrogate context
    show_bands=True,
    alpha_fill=0.12,
    title="Filtered x-position — KF vs EKF vs UKF",
)
fig1.savefig("fig1_filter_comparison.png", dpi=150)

# ── Plot 2: covariance trace ──────────────────────────────────────────────────
fig2 = plot_covariance_trace(
    results,
    normalize=False,
    title="Total estimation uncertainty — tr(P)",
)
fig2.savefig("fig2_cov_trace.png", dpi=150)

# ── Plot 3: innovation comparison ─────────────────────────────────────────────
fig3 = plot_innovation_comparison(
    results,
    obs_component=0,     # range innovation
    standardize=True,
    title="Standardised range innovations",
)
fig3.savefig("fig3_innovations.png", dpi=150)

# ── Plot 4: NEES ──────────────────────────────────────────────────────────────
fig4 = plot_nees(
    results,
    true_states=true_x,
    dof=k,
    ci_level=0.95,
    window=15,
    time_average=True,
    title="NEES (rolling 15-step) — consistency assessment",
)
fig4.savefig("fig4_nees.png", dpi=150)

plt.show()
```

### Interpreting each figure

**Figure 1 — Filtered state comparison.** For the first 20–30 time steps the three filter
estimates visibly differ: the linearised KF pulls toward the direction implied by the
constant $H_\text{lin}$ matrix, while the EKF and UKF update nonlinearly and stay closer
to the true trajectory. From $t \approx 40$ onward all three converge as accumulated
observations overwhelm the prior. The UKF confidence band is typically 5–15% narrower
than the EKF band, reflecting the higher-order accuracy of sigma-point propagation.

**Figure 2 — Covariance trace.** All three traces drop sharply in the first 10 steps.
The UKF trace reaches a lower steady-state value than the EKF, indicating that the UKF
correctly identifies less residual uncertainty after updating. If the EKF and KF traces
were identical (linear case), the plot would show a single overlapping pair of lines — a
useful sanity check.

**Figure 3 — Standardised innovations.** Under a consistent filter, standardised
innovations should scatter around zero within the $\pm 2$ bands with no autocorrelation.
The KF innovations drift positive in the first 30 steps because its linear $H_\text{lin}$
approximation is too coarse near the initial position. The EKF and UKF innovations are
better centred throughout. A Ljung-Box test on each innovation sequence quantifies this;
see [Diagnostics: innovation tests](../diagnostics/innovation-tests.md).

**Figure 4 — NEES.** The rolling NEES for the KF exceeds the 95% upper bound
($\chi^2_{4,\,0.975} \approx 11.1$) for the first 25 steps, confirming the KF is
overconfident during its linearisation phase. The EKF rolling NEES remains near the
upper bound throughout — borderline consistency. The UKF rolling NEES stays well within
the bounds from $t \approx 10$ onward. The time-averaged NEES horizontal lines
(drawn solid) immediately summarise which filter is consistent over the full run.

---

## Interpreting filter differences

### State estimates agree but covariances differ

If `plot_filter_comparison` shows nearly identical state trajectories while
`plot_covariance_trace` shows a persistent gap, the difference is **covariance inflation
due to linearisation**. This is the most common EKF vs UKF pattern for mildly nonlinear
systems:

- The EKF Jacobians approximate the true nonlinear function; any curvature ignored by
  the first-order Taylor expansion reappears as spurious uncertainty in $P_{t|t}$.
- The UKF captures curvature to third order, so its covariance is tighter and more
  accurately reflects true uncertainty.
- Consequence: EKF confidence intervals are too wide, leading to overly conservative
  decisions downstream (e.g. wider prediction bands, less aggressive control).

!!! tip "When state estimates agree, choose the smaller covariance"
    If both filters produce the same filtered mean but different covariances, prefer the
    filter whose NEES is closest to the expected value $k$. A NEES of $k$ is the gold
    standard of calibration: not over-confident, not under-confident.

### EKF diverges, UKF remains consistent

Divergence occurs when the linearisation error at each step is large enough that the
Jacobian-based update drags the state estimate away from the truth. Once the state
estimate is far from the true state, future Jacobians are evaluated at a wrong operating
point, compounding the error. Signs:

- `plot_filter_comparison` shows the EKF line separating from both the UKF line and the
  true trajectory.
- `plot_covariance_trace` may show the EKF covariance collapsing (the filter becomes
  overconfident in the wrong location — a numerical sign of divergence).
- `plot_nees` shows EKF NEES rising without bound while UKF NEES stays within bounds.

Remedies:
1. Switch from EKF to UKF (or EnKF for high-dimensional systems).
2. Increase $Q$ to inject artificial process noise and prevent covariance collapse.
3. Re-parameterise the model to reduce the effective nonlinearity degree (e.g. work in
   log-polar coordinates for range/bearing tracking).

### NEES consistently above the upper bound

Persistent overconfidence — $\epsilon_t > \chi^2_{k,\,1-\alpha/2}$ — means the filter
believes its estimates are more accurate than they actually are.

!!! warning "Overconfidence is dangerous in safety-critical applications"
    An overconfident filter rejects genuine sensor data as outliers, because the
    Mahalanobis distance $v_t^T F_t^{-1} v_t$ exceeds the expected range. In tracking
    applications this causes **track loss**. In control applications it leads to
    insufficient actuation because the controller believes the state is well-known when
    it is not.

Common causes and cures:

| Cause | Remedy |
|---|---|
| $Q$ too small — process noise underestimated | Re-estimate $Q$ via expectation-maximisation (see [EM estimation](../user-guide/advanced/em.md)) |
| $R$ too small — observation noise underestimated | Calibrate sensor noise from static measurements; increase $R$ |
| Linearisation error (EKF in strongly nonlinear regime) | Switch to UKF or EnKF |
| Model structural mismatch | Add missing dynamic components; run [LRT](../diagnostics/likelihood-ratio.md) to identify them |

### NEES consistently below the lower bound

Persistent underconfidence — $\epsilon_t < \chi^2_{k,\,\alpha/2}$ — means the filter's
covariance is inflated relative to the actual errors.

Common causes:
- $Q$ too large (excessive covariance inflation used as a tuning heuristic).
- The model has too many degrees of freedom for the available data (identifiability
  issue; see [Diagnostics: identifiability](../diagnostics/consistency.md)).
- Correct model but erroneous initial covariance $P_0$ that is never fully absorbed.

---

## Related pages

- [EKF user guide](../user-guide/filters/ekf.md) — full EKF derivation, Jacobians, API
- [UKF user guide](../user-guide/filters/ukf.md) — sigma points, UT parameters, API
- [Filter comparison — feature matrix](../user-guide/filters/comparison.md) — tabular
  comparison of all six built-in filters
- [Diagnostics: NEES/NIS consistency tests](../diagnostics/consistency.md) — statistical
  tests, single-run vs Monte Carlo, NIS for real data
- [Diagnostics: innovation tests](../diagnostics/innovation-tests.md) — Ljung-Box,
  Jarque-Bera, and heteroscedasticity tests on innovations
- [Visualization: diagnostics](diagnostics.md) — residual ACF, QQ-plot, CUSUM
- [Visualization: filtered states](filtered-states.md) — single-filter plotting helpers
