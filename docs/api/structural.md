# Structural Models API

`kalmanbox.models`

Structural time-series models wrap a
[`StateSpaceRepresentation`](core.md#statespacerepresentation) in a
convenient high-level interface. Each class:

1. Constructs the system matrices `(Z, T, R, H, Q)` from interpretable
   parameters such as variances and seasonal periods.
2. Provides `.fit()`, `.filter()`, `.smooth()`, `.forecast()`, and
   `.components()` methods.
3. Exposes the underlying `StateSpaceRepresentation` via `.ss` for
   direct access to the Kalman filter machinery.

All structural models are ready-made state-space formulations;
for building **custom** component combinations see
[`UnobservedComponents`](#unobservedcomponents).

---

## LocalLevel

`kalmanbox.models.LocalLevel`

The local level (random walk plus noise) model:

$$
y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

$$
\mu_{t+1} = \mu_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
$$

The state-space matrices are:

$$
Z = [1], \quad T = [1], \quad R = [1], \quad
H = [\sigma_\varepsilon^2], \quad Q = [\sigma_\eta^2]
$$

The signal-to-noise ratio $q = \sigma_\eta^2 / \sigma_\varepsilon^2$
controls the smoothness of the estimated level.

### Constructor

```python
LocalLevel(
    sigma_eps: float | None = None,
    sigma_eta: float | None = None,
    diffuse_init: bool = True,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sigma_eps` | `float \| None` | `None` | Standard deviation of the observation noise `ε_t`. If `None`, estimated by `.fit()`. |
| `sigma_eta` | `float \| None` | `None` | Standard deviation of the level disturbance `η_t`. If `None`, estimated by `.fit()`. |
| `diffuse_init` | `bool` | `True` | Use exact diffuse initialisation for the non-stationary level state. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Properties

| Property | Type | Description |
|---|---|---|
| `ss` | `StateSpaceRepresentation` | The state-space representation. |
| `signal_noise_ratio` | `float` | Signal-to-noise ratio `q = σ_η² / σ_ε²`. `None` before `.fit()`. |
| `params` | `dict[str, float]` | Current parameter values. |

### Methods

#### `fit(y, method="L-BFGS-B", maxiter=1000)`

```python
def fit(
    y: np.ndarray,
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> FitResult
```

Maximise the prediction-error log-likelihood to estimate `sigma_eps`
and `sigma_eta`. Sets `self.sigma_eps` and `self.sigma_eta` in-place.

**Returns** [`FitResult`](core.md#fitresult).

---

#### `filter(y)`

```python
def filter(y: np.ndarray) -> FilterResult
```

Run the Kalman filter. Requires at least one of `sigma_eps`,
`sigma_eta` to be set (either via constructor or `.fit()`).

**Returns** [`FilterResult`](core.md#filterresult).

---

#### `smooth(y)`

```python
def smooth(y: np.ndarray) -> SmoothResult
```

Run the Kalman filter followed by the RTS smoother.

**Returns** [`SmoothResult`](core.md#smoothresult).

---

#### `forecast(y, n_ahead, confidence=0.95)`

```python
def forecast(
    y: np.ndarray,
    n_ahead: int,
    confidence: float = 0.95,
) -> ForecastResult
```

Forecast `n_ahead` steps beyond the end of `y`.

**Returns** [`ForecastResult`](core.md#forecastresult).

---

### Example

```python
import numpy as np
from kalmanbox.models import LocalLevel
from kalmanbox.datasets import load_dataset

# Nile river annual flow (Durbin & Koopman, example 2.1)
nile = load_dataset("nile")

model = LocalLevel()
result = model.fit(nile)

print(result.summary())
# sigma_eps: 15099.8   sigma_eta: 1469.1   SNR: 0.097

smooth = model.smooth(nile)
fc = model.forecast(nile, n_ahead=10)
print(f"10-year forecast: {fc.forecast.ravel()}")
```

---

## LocalLinearTrend

`kalmanbox.models.LocalLinearTrend`

Adds a stochastic slope to the local level:

$$
y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

$$
\mu_{t+1} = \mu_t + \nu_t + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \sigma_\xi^2)
$$

$$
\nu_{t+1} = \nu_t + \zeta_t, \quad \zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2)
$$

State vector: $a_t = (\mu_t, \nu_t)'$.  Special cases:

- $\sigma_\xi^2 = 0$: smooth trend (integrated random walk).
- $\sigma_\zeta^2 = 0$: deterministic slope.
- $\sigma_\xi^2 = \sigma_\zeta^2 = 0$: linear trend.

### Constructor

```python
LocalLinearTrend(
    sigma_eps: float | None = None,
    sigma_level: float | None = None,
    sigma_slope: float | None = None,
    diffuse_init: bool = True,
    fix_level_var: float | None = None,
    fix_slope_var: float | None = None,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sigma_eps` | `float \| None` | `None` | Observation noise std. |
| `sigma_level` | `float \| None` | `None` | Level disturbance std (`σ_ξ`). |
| `sigma_slope` | `float \| None` | `None` | Slope disturbance std (`σ_ζ`). |
| `diffuse_init` | `bool` | `True` | Exact diffuse initialisation for level and slope. |
| `fix_level_var` | `float \| None` | `None` | Fix `σ_ξ²` to this value (not estimated). |
| `fix_slope_var` | `float \| None` | `None` | Fix `σ_ζ²` to this value (not estimated). |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Methods

Same interface as [`LocalLevel`](#locallevel): `.fit()`, `.filter()`,
`.smooth()`, `.forecast()`.

#### `components(smooth_result)`

```python
def components(smooth_result: SmoothResult) -> dict[str, np.ndarray]
```

Extract labelled component arrays from a `SmoothResult`.

**Returns** `dict` with keys:

| Key | Shape | Description |
|---|---|---|
| `"level"` | `(n,)` | Smoothed level `μ_{t|n}`. |
| `"slope"` | `(n,)` | Smoothed slope `ν_{t|n}`. |
| `"level_std"` | `(n,)` | Standard deviation of the level estimate. |
| `"slope_std"` | `(n,)` | Standard deviation of the slope estimate. |

---

### Example

```python
import numpy as np
from kalmanbox.models import LocalLinearTrend
from kalmanbox.datasets import load_dataset

gdp = load_dataset("us_gdp")

model = LocalLinearTrend()
fit = model.fit(gdp)
print(fit.summary())

smooth = model.smooth(gdp)
comps = model.components(smooth)

print(f"Final trend level:  {comps['level'][-1]:.2f}")
print(f"Final trend slope:  {comps['slope'][-1]:.4f}")
```

---

## BasicStructuralModel

`kalmanbox.models.BasicStructuralModel`

The BSM decomposes an observed series into trend, seasonal, cycle, and
irregular components:

$$
y_t = \mu_t + \gamma_t + \psi_t + \varepsilon_t
$$

where `μ_t` is a local linear trend, `γ_t` is a stochastic seasonal
component with period `s`, `ψ_t` is an optional stochastic cycle, and
`ε_t` is the irregular disturbance.

!!! info "State-space dimension"

    The state dimension is `m = 2 + (s - 1) + 2·n_cycles` where `s` is the
    seasonal period. For monthly data with one cycle: `m = 2 + 11 + 2 = 15`.

### Constructor

```python
BasicStructuralModel(
    seasonal_period: int,
    sigma_eps: float | None = None,
    sigma_level: float | None = None,
    sigma_slope: float | None = None,
    sigma_seasonal: float | None = None,
    sigma_cycle: float | None = None,
    include_cycle: bool = False,
    cycle_frequency: float | None = None,
    cycle_damping: float = 0.9,
    seasonal_type: str = "trigonometric",
    diffuse_init: bool = True,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seasonal_period` | `int` | required | Number of seasons per cycle (e.g., 12 for monthly, 4 for quarterly). |
| `sigma_eps` | `float \| None` | `None` | Observation noise std. |
| `sigma_level` | `float \| None` | `None` | Level disturbance std. |
| `sigma_slope` | `float \| None` | `None` | Slope disturbance std. |
| `sigma_seasonal` | `float \| None` | `None` | Seasonal disturbance std. If 0, the seasonal pattern is fixed. |
| `sigma_cycle` | `float \| None` | `None` | Cycle disturbance std. Only used when `include_cycle=True`. |
| `include_cycle` | `bool` | `False` | Add a stochastic trigonometric cycle component. |
| `cycle_frequency` | `float \| None` | `None` | Cycle frequency in radians per period (0, π). Estimated if `None`. |
| `cycle_damping` | `float` | `0.9` | Cycle damping factor `ρ ∈ (0, 1)`. |
| `seasonal_type` | `str` | `"trigonometric"` | Seasonal specification: `"trigonometric"` or `"dummy"`. |
| `diffuse_init` | `bool` | `True` | Exact diffuse initialisation for non-stationary states. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Methods

#### `fit(y, method="L-BFGS-B", maxiter=2000)`

```python
def fit(
    y: np.ndarray,
    method: str = "L-BFGS-B",
    maxiter: int = 2000,
    tol: float = 1e-8,
    disp: bool = False,
) -> FitResult
```

Estimate all free variance parameters by maximum likelihood.

**Returns** [`FitResult`](core.md#fitresult).

---

#### `filter(y)` / `smooth(y)`

Same signature as [`LocalLevel`](#locallevel).

---

#### `forecast(y, n_ahead, confidence=0.95)`

```python
def forecast(
    y: np.ndarray,
    n_ahead: int,
    confidence: float = 0.95,
) -> ForecastResult
```

Produce `n_ahead`-step-ahead forecasts including the seasonal projection.

---

#### `components(smooth_result)`

```python
def components(smooth_result: SmoothResult) -> dict[str, np.ndarray]
```

Return the smoothed component decomposition.

**Returns** `dict` with keys:

| Key | Shape | Description |
|---|---|---|
| `"trend"` | `(n,)` | Smoothed trend `μ_{t|n}`. |
| `"slope"` | `(n,)` | Smoothed slope `ν_{t|n}`. |
| `"seasonal"` | `(n,)` | Smoothed seasonal `γ_{t|n}`. |
| `"cycle"` | `(n,)` | Smoothed cycle `ψ_{t|n}` (zeros if `include_cycle=False`). |
| `"irregular"` | `(n,)` | Residual irregular `ε̂_t`. |
| `"trend_std"` | `(n,)` | Trend estimation uncertainty. |
| `"seasonal_std"` | `(n,)` | Seasonal estimation uncertainty. |

---

#### `snr()`

```python
def snr() -> dict[str, float]
```

Return signal-to-noise ratios for each component relative to the
observation variance `σ_ε²`.

---

### Example

```python
import numpy as np
from kalmanbox.models import BasicStructuralModel
from kalmanbox.datasets import load_dataset

airline = load_dataset("airline")  # Box-Jenkins airline passengers (logged)

model = BasicStructuralModel(
    seasonal_period=12,
    seasonal_type="trigonometric",
    include_cycle=False,
)
fit = model.fit(np.log(airline))
print(fit.summary())

smooth = model.smooth(np.log(airline))
comps = model.components(smooth)

# Forecast 24 months ahead
fc = model.forecast(np.log(airline), n_ahead=24)
import numpy as np
print(f"24-month forecast (log): {fc.forecast.ravel()[-1]:.4f}")
print(f"Back-transformed:        {np.exp(fc.forecast.ravel()[-1]):.1f}")
```

---

## UnobservedComponents

`kalmanbox.models.UnobservedComponents`

A flexible builder for arbitrary combinations of structural components.
Internally assembles a block-diagonal `StateSpaceRepresentation` by
stacking the state vectors of each component.

!!! tip "Use `BasicStructuralModel` for standard models"

    `UnobservedComponents` is best for models that fall outside the
    pre-defined BSM structure (e.g., multiple cycles, mixed seasonal
    periods, or regression components).

### Constructor

```python
UnobservedComponents(
    level: bool = True,
    trend: bool = False,
    seasonal: int | None = None,
    cycle: bool = False,
    irregular: bool = True,
    autoregressive: int | None = None,
    exog: np.ndarray | None = None,
    diffuse_init: bool = True,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | `bool` | `True` | Include a stochastic level (local level). |
| `trend` | `bool` | `False` | Include a stochastic slope (local linear trend). |
| `seasonal` | `int \| None` | `None` | Seasonal period. Includes a trigonometric seasonal component. |
| `cycle` | `bool` | `False` | Include a stochastic trigonometric cycle. |
| `irregular` | `bool` | `True` | Include an i.i.d. irregular component. |
| `autoregressive` | `int \| None` | `None` | AR order for the irregular component. |
| `exog` | `np.ndarray \| None` | `None` | Exogenous regressors (design matrix). Shape `(n, k)`. |
| `diffuse_init` | `bool` | `True` | Diffuse initialisation for non-stationary components. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Methods

#### `add_component(component)`

```python
def add_component(component: SSMComponent) -> UnobservedComponents
```

Append an arbitrary `SSMComponent` to the model. Returns `self` for
method chaining.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `component` | `SSMComponent` | Any object implementing the `SSMComponent` protocol (`.state_dim`, `.obs_matrix`, `.transition`, `.selection`, `.noise_cov`). |

---

#### `fit(y, method="L-BFGS-B", maxiter=2000)`

Same interface as [`BasicStructuralModel.fit()`](#basicstructuralmodel).

---

#### `filter(y)` / `smooth(y)` / `forecast(y, n_ahead, confidence=0.95)`

Same interface as [`BasicStructuralModel`](#basicstructuralmodel).

---

#### `components(smooth_result)`

```python
def components(smooth_result: SmoothResult) -> dict[str, np.ndarray]
```

Return a `dict` mapping each named component to its smoothed time path.
Keys correspond to the components added at construction time.

---

### Example

```python
import numpy as np
from kalmanbox.models import UnobservedComponents
from kalmanbox.datasets import load_dataset

# Monthly industrial production: trend + monthly seasonal + AR(1) irregular
ip = load_dataset("industrial_production")

model = UnobservedComponents(
    level=True,
    trend=True,
    seasonal=12,
    irregular=True,
    autoregressive=1,
)
fit = model.fit(ip)
print(fit.summary())

smooth = model.smooth(ip)
comps = model.components(smooth)
print(f"Trend at last obs: {comps['level'][-1]:.2f}")
print(f"Seasonal at last obs: {comps['seasonal'][-1]:.2f}")
```

---

## Cycle

`kalmanbox.models.Cycle`

A stochastic trigonometric cycle component:

$$
\begin{pmatrix} \psi_{t+1} \\ \psi_{t+1}^* \end{pmatrix}
= \rho
\begin{pmatrix} \cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c \end{pmatrix}
\begin{pmatrix} \psi_t \\ \psi_t^* \end{pmatrix}
+
\begin{pmatrix} \kappa_t \\ \kappa_t^* \end{pmatrix}
$$

where `λ_c ∈ (0, π)` is the frequency in radians, `ρ ∈ (0, 1)` is the
damping factor, and `κ_t ~ N(0, σ_κ²)`.

`Cycle` is typically used as a sub-component of `UnobservedComponents`
via `.add_component()`, but it can also be used standalone.

### Constructor

```python
Cycle(
    frequency: float | None = None,
    damping: float = 0.9,
    sigma_cycle: float | None = None,
    period: float | None = None,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `frequency` | `float \| None` | `None` | Frequency `λ_c` in radians per period. If `None`, estimated from data or derived from `period`. |
| `damping` | `float` | `0.9` | Damping factor `ρ ∈ (0, 1)`. Values close to 1 produce persistent cycles. |
| `sigma_cycle` | `float \| None` | `None` | Cycle disturbance std `σ_κ`. |
| `period` | `float \| None` | `None` | Alternative specification: period in time steps. Sets `λ_c = 2π / period`. Overrides `frequency`. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Properties

| Property | Type | Description |
|---|---|---|
| `frequency` | `float` | Cycle frequency `λ_c` in radians. |
| `period_steps` | `float` | Implied period `2π / λ_c`. |
| `state_dim` | `int` | Always `2` (ψ_t and ψ_t*). |

### Methods

#### `as_ssm()`

```python
def as_ssm() -> StateSpaceRepresentation
```

Return the 2×2 state-space representation for the cycle component
alone. Useful for inspecting matrices or embedding in custom models.

**Returns** `StateSpaceRepresentation` with `m = 2`, `p = 1`.

---

#### `fit(y, method="L-BFGS-B")`

```python
def fit(
    y: np.ndarray,
    method: str = "L-BFGS-B",
    fix_frequency: bool = False,
) -> FitResult
```

Fit `frequency`, `damping`, and `sigma_cycle` to `y` by maximum
likelihood.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` | required | Scalar time series. |
| `method` | `str` | `"L-BFGS-B"` | Optimiser. |
| `fix_frequency` | `bool` | `False` | Hold `frequency` fixed at the constructor value. |

**Returns** [`FitResult`](core.md#fitresult).

---

### Example

```python
import numpy as np
from kalmanbox.models import UnobservedComponents, Cycle
from kalmanbox.datasets import load_dataset

# US unemployment: trend + two business cycles
unemp = load_dataset("us_unemployment")

short_cycle = Cycle(period=16, damping=0.85)
long_cycle  = Cycle(period=40, damping=0.95)

model = UnobservedComponents(level=True, trend=True, irregular=True)
model.add_component(short_cycle)
model.add_component(long_cycle)

fit = model.fit(unemp)
smooth = model.smooth(unemp)
comps  = model.components(smooth)
print(comps.keys())
# dict_keys(['level', 'slope', 'irregular', 'cycle_0', 'cycle_1'])
```

---

## See Also

- [User Guide: Structural Models](../user-guide/structural/index.md)
- [User Guide: Local Level](../user-guide/structural/local-level.md)
- [User Guide: BSM](../user-guide/structural/bsm.md)
- [User Guide: UCM](../user-guide/structural/ucm.md)
- [Tutorial: BSM Structural Decomposition](../tutorials/bsm.md)
- [API: Advanced Models](advanced.md)
- [API: Core](core.md)
