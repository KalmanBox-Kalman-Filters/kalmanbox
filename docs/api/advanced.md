# Advanced Models API

`kalmanbox.models` · `kalmanbox.estimation`

This page documents the three advanced model and estimation classes:

- **[`DynamicFactorModel`](#dynamicfactormodel)** — multi-factor
  dynamic factor models estimated by PCA + EM or maximum likelihood.
- **[`TimeVaryingParameters`](#timevaryingparameters)** — time-varying
  coefficient regression cast as a state-space model.
- **[`EMEstimator`](#emestimator)** — stand-alone EM algorithm that
  can fit any `StateSpaceRepresentation` with free parameters.

---

## DynamicFactorModel

`kalmanbox.models.DynamicFactorModel`

A Dynamic Factor Model (DFM) decomposes a panel of `N` observed series
into `r` common latent factors plus series-specific idiosyncratic
disturbances:

$$
y_t = \Lambda\, f_t + \varepsilon_t, \quad
\varepsilon_t \sim \mathcal{N}(0, \Psi)
$$

$$
f_t = A_1\, f_{t-1} + \cdots + A_p\, f_{t-p} + \eta_t, \quad
\eta_t \sim \mathcal{N}(0, Q)
$$

where `Λ` is the `(N × r)` factor loading matrix, `f_t` is the
`(r × 1)` factor vector, and `Ψ = diag(ψ_1, …, ψ_N)` is a diagonal
idiosyncratic covariance.

The model is cast in state-space form with:

$$
a_t = (f_t', f_{t-1}', \ldots, f_{t-p+1}')',\quad m = r \cdot p
$$

!!! info "Identification"

    The loading matrix `Λ` is identified by fixing the upper `(r × r)`
    block to a lower-triangular matrix with positive diagonal entries.
    Use `identification="lower_triangular"` (the default) or provide a
    custom constraint matrix via `constraint`.

### Constructor

```python
DynamicFactorModel(
    n_factors: int,
    factor_order: int = 1,
    idiosyncratic_ar: int = 0,
    estimation_method: str = "em",
    identification: str = "lower_triangular",
    constraint: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    em_init: str = "pca",
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_factors` | `int` | required | Number of latent factors `r`. |
| `factor_order` | `int` | `1` | VAR lag order `p` for factor dynamics. |
| `idiosyncratic_ar` | `int` | `0` | AR order for each idiosyncratic disturbance `ε_i,t`. `0` = i.i.d. |
| `estimation_method` | `str` | `"em"` | Estimation strategy: `"em"` (EM algorithm), `"mle"` (direct numerical MLE), or `"two_step"` (PCA then MLE). |
| `identification` | `str` | `"lower_triangular"` | Loading identification restriction: `"lower_triangular"` or `"none"`. |
| `constraint` | `np.ndarray \| None` | `None` | Custom `(N, r)` binary constraint matrix specifying which loadings are free. |
| `max_iter` | `int` | `500` | Maximum EM iterations (used when `estimation_method="em"`). |
| `tol` | `float` | `1e-6` | EM convergence tolerance on log-likelihood improvement. |
| `em_init` | `str` | `"pca"` | EM initialisation strategy: `"pca"` or `"random"`. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Properties

| Property | Type | Description |
|---|---|---|
| `n_factors` | `int` | Number of latent factors. |
| `n_series` | `int` | Number of observed series `N`. Set after `.fit()`. |
| `ss` | `StateSpaceRepresentation` | Underlying state-space representation. |
| `factor_loadings` | `np.ndarray` | Estimated loading matrix `Λ`. Shape `(N, r)`. `None` before `.fit()`. |
| `idiosyncratic_variances` | `np.ndarray` | Diagonal of `Ψ`. Shape `(N,)`. `None` before `.fit()`. |
| `factor_var` | `np.ndarray` | VAR coefficient matrix `A`. Shape `(r, r·p)`. `None` before `.fit()`. |
| `loglikelihood_history` | `np.ndarray` | Log-likelihood per EM iteration. Empty for MLE. |

### Methods

#### `fit(Y, sample_weight=None)`

```python
def fit(
    Y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> FitResult
```

Estimate all model parameters (`Λ`, `Ψ`, `A`, `Q`) from the `(n × N)`
observation matrix `Y`. Missing values are handled via the EM update
rules.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Y` | `np.ndarray` | required | Panel data. Shape `(n, N)`. `np.nan` encodes missing observations. |
| `sample_weight` | `np.ndarray \| None` | `None` | Per-observation weights. Shape `(n,)`. |

**Returns** [`FitResult`](core.md#fitresult) with additional attribute `loglikelihood_history`.

---

#### `filter(Y)`

```python
def filter(Y: np.ndarray) -> FilterResult
```

Run the Kalman filter on `Y` using current parameter estimates.

**Returns** [`FilterResult`](core.md#filterresult) where `a_filt` contains the
filtered factor estimates with shape `(n, r·p)`.

---

#### `smooth(Y)`

```python
def smooth(Y: np.ndarray) -> SmoothResult
```

Run filter + RTS smoother. Smoothed factors are in
`smooth_result.a_smooth[:, :r]`.

---

#### `factors(Y, method="smooth")`

```python
def factors(
    Y: np.ndarray,
    method: str = "smooth",
) -> np.ndarray
```

Extract estimated common factors from `Y`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Y` | `np.ndarray` | required | Panel data. Shape `(n, N)`. |
| `method` | `str` | `"smooth"` | `"smooth"` (RTS smoother) or `"filter"` (one-step-ahead). |

**Returns** `np.ndarray` of shape `(n, r)` — the factor time series.

---

#### `loadings()`

```python
def loadings() -> np.ndarray
```

Return the estimated factor loading matrix `Λ`. Shape `(N, r)`.

---

#### `forecast(Y, n_ahead, confidence=0.95)`

```python
def forecast(
    Y: np.ndarray,
    n_ahead: int,
    confidence: float = 0.95,
) -> ForecastResult
```

Forecast `n_ahead` steps for all `N` series jointly.

**Returns** [`ForecastResult`](core.md#forecastresult) where `forecast`
has shape `(n_ahead, N)`.

---

#### `score_obs(Y)`

```python
def score_obs(Y: np.ndarray) -> np.ndarray
```

Return the per-observation contribution to the log-likelihood. Shape
`(n,)`. Useful for outlier detection and cross-validation.

---

### Example

```python
import numpy as np
from kalmanbox.models import DynamicFactorModel
from kalmanbox.datasets import load_dataset

# US macro panel: GDP growth, CPI inflation, unemployment, fed funds rate
macro = load_dataset("us_macro")  # shape (240, 4), monthly 2003–2023

dfm = DynamicFactorModel(
    n_factors=2,
    factor_order=1,
    estimation_method="em",
    max_iter=1000,
)
fit = dfm.fit(macro)
print(f"LL at convergence: {fit.loglikelihood:.2f}")
print(f"EM iterations:     {len(dfm.loglikelihood_history)}")

# Extract smoothed factors
f = dfm.factors(macro, method="smooth")   # (240, 2)
print(f"Factor 1 loading on GDP: {dfm.loadings()[0, 0]:.4f}")
print(f"Factor 2 loading on CPI: {dfm.loadings()[1, 1]:.4f}")

# Forecast 6 months ahead
fc = dfm.forecast(macro, n_ahead=6)
print(f"GDP 6-month forecast: {fc.forecast[:, 0]}")
```

---

## TimeVaryingParameters

`kalmanbox.models.TimeVaryingParameters`

Casts a linear regression with time-varying coefficients into state-space
form (Cooley & Prescott 1973; Kim & Nelson 1999):

$$
y_t = x_t'\, \beta_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

$$
\beta_{t+1} = \beta_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)
$$

The state vector is `a_t = β_t` (k-dimensional), with:

$$
Z_t = x_t', \quad T = I_k, \quad R = I_k, \quad H = \sigma^2, \quad Q = \text{diag}(q_1, \ldots, q_k)
$$

!!! tip "Detecting structural breaks"

    Large innovations `η_t` indicate coefficient instability. Use
    `.disturbance_smooth()` or the [CUSUM diagnostics](../diagnostics/cusum.md)
    to test for structural change.

### Constructor

```python
TimeVaryingParameters(
    exog: np.ndarray,
    evolution_variance: np.ndarray | float | None = None,
    sigma_eps: float | None = None,
    initial_state: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
    random_walk_coefficients: list[bool] | None = None,
    diffuse_init: bool = True,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `exog` | `np.ndarray` | required | Regressor matrix. Shape `(n, k)`. Include an intercept column manually if desired. |
| `evolution_variance` | `np.ndarray \| float \| None` | `None` | Diagonal entries of `Q`. Scalar applies the same variance to all coefficients. If `None`, all `q_i` are estimated. |
| `sigma_eps` | `float \| None` | `None` | Observation noise std. Estimated if `None`. |
| `initial_state` | `np.ndarray \| None` | `None` | Initial coefficient vector `β_1|0`. Shape `(k,)`. Defaults to OLS estimate. |
| `initial_covariance` | `np.ndarray \| None` | `None` | Initial state covariance `P_1|0`. Shape `(k, k)`. |
| `random_walk_coefficients` | `list[bool] \| None` | `None` | Boolean mask: `True` → coefficient follows a random walk; `False` → fixed. `None` = all time-varying. |
| `diffuse_init` | `bool` | `True` | Exact diffuse initialisation. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Properties

| Property | Type | Description |
|---|---|---|
| `n_params` | `int` | Number of coefficients `k`. |
| `ss` | `StateSpaceRepresentation` | Underlying time-varying state-space representation. |
| `params` | `dict[str, float]` | Current parameter estimates. |

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

Maximise the log-likelihood to estimate `Q` (or its free diagonal
entries) and `σ²`.

**Returns** [`FitResult`](core.md#fitresult).

---

#### `filter(y)`

```python
def filter(y: np.ndarray) -> FilterResult
```

Run the Kalman filter. `a_filt` contains filtered coefficient estimates.

---

#### `smooth(y)`

```python
def smooth(y: np.ndarray) -> SmoothResult
```

Run filter + RTS smoother for smoothed coefficient paths.

---

#### `coefficients(y, method="smooth")`

```python
def coefficients(
    y: np.ndarray,
    method: str = "smooth",
) -> tuple[np.ndarray, np.ndarray]
```

Extract time-varying coefficient estimates.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` | required | Observation series. Shape `(n,)` or `(n, 1)`. |
| `method` | `str` | `"smooth"` | `"smooth"` (RTS) or `"filter"` (one-step-ahead). |

**Returns** `(beta, beta_std)` where:

| Return | Shape | Description |
|---|---|---|
| `beta` | `(n, k)` | Coefficient time paths `β_{t|n}` (smoothed) or `β_{t|t}` (filtered). |
| `beta_std` | `(n, k)` | Standard deviations from the diagonal of `P_{t|n}` or `P_{t|t}`. |

---

#### `forecast(y, x_future, n_ahead, confidence=0.95)`

```python
def forecast(
    y: np.ndarray,
    x_future: np.ndarray,
    n_ahead: int,
    confidence: float = 0.95,
) -> ForecastResult
```

Forecast the dependent variable using future regressor values.

| Parameter | Type | Description |
|---|---|---|
| `y` | `np.ndarray` | Historical observations. |
| `x_future` | `np.ndarray` | Future regressors. Shape `(n_ahead, k)`. |
| `n_ahead` | `int` | Forecast horizon. |
| `confidence` | `float` | Coverage for prediction intervals. |

**Returns** [`ForecastResult`](core.md#forecastresult).

---

### Example

```python
import numpy as np
from kalmanbox.models import TimeVaryingParameters
from kalmanbox.datasets import load_dataset

# Time-varying CAPM: R_i,t = alpha_t + beta_t * R_m,t + eps_t
returns = load_dataset("equity_returns")  # columns: asset, market
y = returns[:, 0]                          # asset excess returns
X = np.column_stack([                      # intercept + market
    np.ones(len(y)),
    returns[:, 1],
])

tvp = TimeVaryingParameters(exog=X)
fit = tvp.fit(y)
print(fit.summary())

beta, beta_std = tvp.coefficients(y, method="smooth")
print(f"Alpha range: [{beta[:, 0].min():.3f}, {beta[:, 0].max():.3f}]")
print(f"Beta range:  [{beta[:, 1].min():.3f}, {beta[:, 1].max():.3f}]")

# Forecast assuming market return = 0 next month
x_future = np.array([[1.0, 0.0]])
fc = tvp.forecast(y, x_future=x_future, n_ahead=1)
print(f"1-month forecast: {fc.forecast[0, 0]:.4f}")
```

---

## EMEstimator

`kalmanbox.estimation.EMEstimator`

A stand-alone Expectation–Maximisation estimator for arbitrary
`StateSpaceRepresentation` models. The EM algorithm iterates between:

- **E-step**: run the Kalman smoother to compute the smoothed states
  and their cross-covariances required by the M-step.
- **M-step**: update free parameters analytically using the smoothed
  sufficient statistics.

!!! info "When to use EM vs MLE"

    | Method | Use when |
    |---|---|
    | **EM** | DFM / missing data; closed-form M-step is available; robust initialisation needed |
    | **MLE** | Gradient-based optimiser is well-conditioned; small number of free parameters |
    | **Both** | EM to find a good starting point, then MLE for exact Hessian and standard errors |

### Constructor

```python
EMEstimator(
    max_iter: int = 500,
    tol: float = 1e-6,
    print_level: int = 0,
    missing_data: bool = True,
    enforce_positive_definite: bool = True,
    update_Z: bool = True,
    update_T: bool = True,
    update_H: bool = True,
    update_Q: bool = True,
    update_obs_intercept: bool = False,
    update_state_intercept: bool = False,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_iter` | `int` | `500` | Maximum number of EM iterations. |
| `tol` | `float` | `1e-6` | Convergence criterion on log-likelihood improvement `|ℓ_{k+1} - ℓ_k|`. |
| `print_level` | `int` | `0` | Verbosity: `0` = silent, `1` = per-10 iterations, `2` = every iteration. |
| `missing_data` | `bool` | `True` | Handle `np.nan` entries via the EM missing-data M-step. |
| `enforce_positive_definite` | `bool` | `True` | Project `H` and `Q` updates onto the PSD cone after each M-step. |
| `update_Z` | `bool` | `True` | Update the observation matrix `Z` in the M-step. |
| `update_T` | `bool` | `True` | Update the transition matrix `T` in the M-step. |
| `update_H` | `bool` | `True` | Update the observation noise covariance `H`. |
| `update_Q` | `bool` | `True` | Update the state noise covariance `Q`. |
| `update_obs_intercept` | `bool` | `False` | Update the observation intercept `d`. |
| `update_state_intercept` | `bool` | `False` | Update the state intercept `c`. |

### Properties

| Property | Type | Description |
|---|---|---|
| `loglikelihood_history` | `np.ndarray` | Log-likelihood at each iteration. Shape `(n_iter,)`. |
| `n_iter` | `int` | Number of EM iterations performed. |
| `converged` | `bool` | `True` if convergence criterion was satisfied. |

### Methods

#### `fit(ss, y, a0=None, P0=None)`

```python
def fit(
    ss: StateSpaceRepresentation,
    y: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> StateSpaceRepresentation
```

Run the EM algorithm to update `ss` in-place until convergence or
`max_iter` is reached.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ss` | `StateSpaceRepresentation` | required | The state-space representation to fit. Free matrices are updated; fixed matrices are held constant. |
| `y` | `np.ndarray` | required | Observations. Shape `(n, p)`. |
| `a0` | `np.ndarray \| None` | `None` | Initial state mean. Shape `(m,)`. |
| `P0` | `np.ndarray \| None` | `None` | Initial state covariance. Shape `(m, m)`. |

**Returns** The updated `StateSpaceRepresentation` (same object,
updated in-place). Access fitted matrices directly via `ss.Z`, `ss.T`,
`ss.H`, `ss.Q`.

---

#### `e_step(ss, y, a0=None, P0=None)`

```python
def e_step(
    ss: StateSpaceRepresentation,
    y: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> EMSufficientStats
```

Run the Kalman smoother and compute sufficient statistics for the
M-step. Returns an `EMSufficientStats` object with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `S_11` | `(m, m)` | `∑_t E[a_t a_t' | y_{1:n}]` |
| `S_10` | `(m, m)` | `∑_t E[a_t a_{t-1}' | y_{1:n}]` |
| `S_00` | `(m, m)` | `∑_t E[a_{t-1} a_{t-1}' | y_{1:n}]` |
| `Syy` | `(p, p)` | `∑_t y_t y_t'` (observed only) |
| `Sya` | `(p, m)` | `∑_t y_t E[a_t' | y_{1:n}]` |
| `loglikelihood` | `float` | Current log-likelihood. |

---

#### `m_step(ss, stats, y)`

```python
def m_step(
    ss: StateSpaceRepresentation,
    stats: EMSufficientStats,
    y: np.ndarray,
) -> None
```

Update the free matrices of `ss` in-place using the closed-form
M-step equations given sufficient statistics `stats`.

---

### Example

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation
from kalmanbox.estimation import EMEstimator
from kalmanbox.datasets import load_dataset

# DFM via raw EM — illustrates the building-block API
Y = load_dataset("us_macro")  # (240, 4)
n, N = Y.shape
r = 2  # factors

# Random initialisation
rng = np.random.default_rng(0)
Lambda_init = rng.standard_normal((N, r)) * 0.1
Psi_init    = np.eye(N) * 0.5
T_init      = np.eye(r) * 0.8
Q_init      = np.eye(r) * 0.1

# Build state-space representation
ss = StateSpaceRepresentation(
    Z=Lambda_init,
    T=T_init,
    R=np.eye(r),
    H=Psi_init,
    Q=Q_init,
)

em = EMEstimator(max_iter=300, tol=1e-6, print_level=1)
ss_fitted = em.fit(ss, Y)

import matplotlib.pyplot as plt
plt.plot(em.loglikelihood_history)
plt.xlabel("EM Iteration")
plt.ylabel("Log-likelihood")
plt.title(f"EM convergence (converged={em.converged}, n_iter={em.n_iter})")
plt.tight_layout()
plt.show()

print("Fitted Z (factor loadings):")
print(ss_fitted.Z.round(4))
```

---

## See Also

- [User Guide: Dynamic Factor Model](../user-guide/advanced/dfm.md)
- [User Guide: Time-Varying Parameters](../user-guide/advanced/tvp.md)
- [User Guide: EM Algorithm](../user-guide/advanced/em.md)
- [Tutorial: DFM](../tutorials/dfm.md)
- [Tutorial: TVP](../tutorials/tvp.md)
- [Theory: DFM Theory](../theory/dfm-theory.md)
- [API: Structural Models](structural.md)
- [API: Core](core.md)
