# Core API

The `kalmanbox.core` and `kalmanbox.filters` / `kalmanbox.smoothers`
modules provide the fundamental building blocks of the library:

- **[`StateSpaceRepresentation`](#statespacerepresentation)** — the
  system matrices `(Z, T, R, H, Q)` that define any linear Gaussian
  state-space model.
- **[`KalmanFilter`](#kalmanfilter)** — the forward recursion that
  computes filtered states, innovations, and the prediction-error
  log-likelihood.
- **[`RTSSmoother`](#rtssmoother)** — the Rauch–Tung–Striebel backward
  pass that conditions on the full observation sequence.

Higher-level model classes (`LocalLevel`, `DynamicFactorModel`, …)
all construct a `StateSpaceRepresentation` internally and delegate
filtering / smoothing to these classes.

---

## StateSpaceRepresentation

`kalmanbox.core.StateSpaceRepresentation`

Holds the matrices that define a linear Gaussian state-space model in
Durbin & Koopman (2012) notation:

$$
y_t = Z_t\, a_t + d_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H_t)
$$

$$
a_{t+1} = T_t\, a_t + c_t + R_t\, \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)
$$

Matrices may be time-invariant (`ndim == 2`) or time-varying
(`ndim == 3`, with the leading axis indexing time).

### Constructor

```python
StateSpaceRepresentation(
    Z: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    obs_intercept: np.ndarray | None = None,
    state_intercept: np.ndarray | None = None,
    time_varying: bool = False,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Z` | `np.ndarray` | required | Observation (design) matrix. Shape `(p, m)` for time-invariant or `(n, p, m)` for time-varying. |
| `T` | `np.ndarray` | required | State transition matrix. Shape `(m, m)` or `(n, m, m)`. |
| `R` | `np.ndarray` | required | Noise selection (loading) matrix. Shape `(m, r)` or `(n, m, r)`. |
| `H` | `np.ndarray` | required | Observation noise covariance. Shape `(p, p)` or `(n, p, p)`. Must be positive semi-definite. |
| `Q` | `np.ndarray` | required | State disturbance covariance. Shape `(r, r)` or `(n, r, r)`. Must be positive semi-definite. |
| `obs_intercept` | `np.ndarray \| None` | `None` | Observation intercept `d_t`. Shape `(p,)` or `(n, p)`. Defaults to zero vector. |
| `state_intercept` | `np.ndarray \| None` | `None` | State intercept `c_t`. Shape `(m,)` or `(n, m)`. Defaults to zero vector. |
| `time_varying` | `bool` | `False` | If `True`, the leading axis of 3-D arrays is treated as the time index. Must be consistent with the array shapes provided. |

### Properties

| Property | Type | Description |
|---|---|---|
| `state_dim` | `int` | State dimension `m`. |
| `obs_dim` | `int` | Observation dimension `p`. |
| `disturbance_dim` | `int` | Disturbance dimension `r`. |
| `is_time_varying` | `bool` | `True` when any matrix has a leading time axis. |

### Methods

#### `validate()`

```python
def validate() -> None
```

Check that all matrices are internally consistent (correct shapes,
positive semi-definiteness of `H` and `Q`, compatible dimensions).
Raises `ValueError` on failure.

---

#### `update_matrices(**matrices)`

```python
def update_matrices(
    Z: np.ndarray | None = None,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    H: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    obs_intercept: np.ndarray | None = None,
    state_intercept: np.ndarray | None = None,
) -> None
```

Replace one or more system matrices in-place. Only keyword arguments
that are not `None` are updated; the rest remain unchanged.

**Parameters** — same names and shapes as the constructor.

---

#### `copy()`

```python
def copy() -> StateSpaceRepresentation
```

Return a deep copy of the representation with independent NumPy arrays.

---

### Example

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation

# Local level model: y_t = mu_t + eps_t,  mu_{t+1} = mu_t + eta_t
p, m, r = 1, 1, 1
sigma_eps, sigma_eta = 1.0, 0.5

ss = StateSpaceRepresentation(
    Z=np.array([[1.0]]),            # (1, 1)
    T=np.array([[1.0]]),            # (1, 1)
    R=np.array([[1.0]]),            # (1, 1)
    H=np.array([[sigma_eps**2]]),   # (1, 1)
    Q=np.array([[sigma_eta**2]]),   # (1, 1)
)
ss.validate()
print(ss.state_dim, ss.obs_dim, ss.disturbance_dim)  # 1 1 1
```

---

## KalmanFilter

`kalmanbox.filters.KalmanFilter`

Implements the Kalman prediction–update recursion for linear Gaussian
state-space models. Operates on a
[`StateSpaceRepresentation`](#statespacerepresentation) and returns a
`FilterResult` containing filtered states, innovations, gain matrices,
and the prediction-error log-likelihood.

!!! info "Diffuse initialisation"

    When `diffuse_init=True` the filter uses the Kalman–de Jong diffuse
    recursion for the first `d` steps until all diffuse components have
    been collapsed (`F_inf < diffuse_threshold`). This is the recommended
    approach for models with unit-root states such as `LocalLevel` or
    `LocalLinearTrend`.

### Constructor

```python
KalmanFilter(
    ss: StateSpaceRepresentation,
    diffuse_init: bool = False,
    diffuse_threshold: float = 1e6,
    missing_obs_method: str = "skip",
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ss` | `StateSpaceRepresentation` | required | State-space system matrices. |
| `diffuse_init` | `bool` | `False` | Use exact diffuse initialisation for non-stationary states. |
| `diffuse_threshold` | `float` | `1e6` | Threshold below which a diffuse variance `F_inf` is considered collapsed. |
| `missing_obs_method` | `str` | `"skip"` | How to handle `np.nan` observations. `"skip"` propagates the state without an update; `"impute"` uses the filter density to fill in the missing value. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision for all internal arrays. |

### Properties

| Property | Type | Description |
|---|---|---|
| `ss` | `StateSpaceRepresentation` | The state-space representation attached to this filter. |
| `state_dim` | `int` | Alias for `ss.state_dim`. |
| `obs_dim` | `int` | Alias for `ss.obs_dim`. |

### Methods

#### `filter(y, a0=None, P0=None)`

```python
def filter(
    y: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> FilterResult
```

Run the forward Kalman filter on the observation sequence `y`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` | required | Observations. Shape `(n,)`, `(n, 1)`, or `(n, p)`. `np.nan` encodes missing values. |
| `a0` | `np.ndarray \| None` | `None` | Initial state mean. Shape `(m,)`. Defaults to zeros. |
| `P0` | `np.ndarray \| None` | `None` | Initial state covariance. Shape `(m, m)`. Defaults to `1e6 * I` when `diffuse_init=False`, or the exact diffuse form `κ → ∞` when `diffuse_init=True`. |

**Returns** `FilterResult` with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `a_pred` | `(n+1, m)` | One-step-ahead predicted state means `a_{t|t-1}`. |
| `P_pred` | `(n+1, m, m)` | Predicted state covariances `P_{t|t-1}`. |
| `a_filt` | `(n, m)` | Filtered state means `a_{t|t}`. |
| `P_filt` | `(n, m, m)` | Filtered state covariances `P_{t|t}`. |
| `v` | `(n, p)` | Innovation vectors `v_t = y_t - Z_t a_{t|t-1} - d_t`. |
| `F` | `(n, p, p)` | Innovation covariance matrices `F_t = Z_t P_{t|t-1} Z_t' + H_t`. |
| `K` | `(n, m, p)` | Kalman gain matrices `K_t = T_t P_{t|t-1} Z_t' F_t^{-1}`. |
| `loglikelihood` | `float` | Prediction-error decomposition log-likelihood. |
| `n_diffuse` | `int` | Number of diffuse initialisation steps (0 when `diffuse_init=False`). |

---

#### `predict(n_ahead, result, exog=None)`

```python
def predict(
    n_ahead: int,
    result: FilterResult,
    exog: np.ndarray | None = None,
) -> ForecastResult
```

Compute `n_ahead`-step-ahead point forecasts and confidence intervals
starting from the last filtered state in `result`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_ahead` | `int` | required | Number of future steps to forecast. |
| `result` | `FilterResult` | required | Output of a prior `.filter()` call. |
| `exog` | `np.ndarray \| None` | `None` | Future exogenous regressors included in `obs_intercept`. Shape `(n_ahead, k)`. |

**Returns** `ForecastResult` with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `forecast` | `(n_ahead, p)` | Point forecasts `E[y_{n+h} | y_{1:n}]`. |
| `forecast_states` | `(n_ahead, m)` | Predicted state means for each horizon. |
| `lower` | `(n_ahead, p)` | Lower bound of 95 % prediction interval. |
| `upper` | `(n_ahead, p)` | Upper bound of 95 % prediction interval. |
| `confidence` | `float` | Confidence level used (default 0.95). |

---

#### `loglikelihood(y, a0=None, P0=None, diffuse=False)`

```python
def loglikelihood(
    y: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    diffuse: bool = False,
) -> float
```

Compute the Gaussian prediction-error log-likelihood

$$
\ell = -\frac{1}{2} \sum_{t=1}^{n} \left( p \ln 2\pi + \ln|F_t| + v_t' F_t^{-1} v_t \right)
$$

without storing the full `FilterResult`. Useful as an objective in
numerical optimisation. When `diffuse=True` the diffuse portion of
the likelihood is excluded from the sum.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` | required | Observations. Same shape rules as `.filter()`. |
| `a0` | `np.ndarray \| None` | `None` | Initial state mean. |
| `P0` | `np.ndarray \| None` | `None` | Initial state covariance. |
| `diffuse` | `bool` | `False` | Exclude diffuse initialisation steps from likelihood. |

**Returns** `float` — the scalar log-likelihood value.

---

#### `fit(y, method="L-BFGS-B", maxiter=1000, tol=1e-8, callback=None)`

```python
def fit(
    y: np.ndarray,
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-8,
    callback: Callable | None = None,
) -> FitResult
```

Maximise the prediction-error log-likelihood over the free parameters
of the attached `StateSpaceRepresentation`. Parameters are those whose
values are `None` at construction time or wrapped in `Param(...)`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` | required | Observations used for estimation. |
| `method` | `str` | `"L-BFGS-B"` | SciPy optimiser name. Common choices: `"L-BFGS-B"`, `"Nelder-Mead"`, `"BFGS"`. |
| `maxiter` | `int` | `1000` | Maximum number of optimiser iterations. |
| `tol` | `float` | `1e-8` | Convergence tolerance on the log-likelihood improvement. |
| `callback` | `Callable \| None` | `None` | Optional function called at each iteration with the current parameter vector. |

**Returns** `FitResult` with attributes:

| Attribute | Type | Description |
|---|---|---|
| `params` | `dict[str, float]` | Estimated parameter values keyed by name. |
| `loglikelihood` | `float` | Log-likelihood at the optimum. |
| `aic` | `float` | Akaike information criterion. |
| `bic` | `float` | Bayesian information criterion. |
| `hqic` | `float` | Hannan–Quinn information criterion. |
| `converged` | `bool` | `True` if the optimiser reported convergence. |
| `n_iter` | `int` | Number of optimiser iterations taken. |
| `filter_result` | `FilterResult` | `FilterResult` at the estimated parameters. |

---

#### `update(ss)`

```python
def update(ss: StateSpaceRepresentation) -> None
```

Replace the attached state-space representation in-place. Useful when
iteratively updating matrices during EM or Bayesian sampling.

---

### Example

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

rng = np.random.default_rng(0)
n = 200

# Simulate a local level
sigma_eps, sigma_eta = 1.0, 0.3
mu = np.cumsum(rng.normal(0, sigma_eta, n))
y = mu + rng.normal(0, sigma_eps, n)

# Build state-space representation
ss = StateSpaceRepresentation(
    Z=np.array([[1.0]]),
    T=np.array([[1.0]]),
    R=np.array([[1.0]]),
    H=np.array([[sigma_eps**2]]),
    Q=np.array([[sigma_eta**2]]),
)

# Run filter
kf = KalmanFilter(ss, diffuse_init=True)
result = kf.filter(y)

print(f"Log-likelihood: {result.loglikelihood:.3f}")
print(f"Filtered state mean at t=100: {result.a_filt[100, 0]:.4f}")

# Forecast 12 steps ahead
fc = kf.predict(n_ahead=12, result=result)
print(f"12-step-ahead forecast: {fc.forecast.ravel()[:3]}")
```

---

## RTSSmoother

`kalmanbox.smoothers.RTSSmoother`

The Rauch–Tung–Striebel (RTS) smoother runs a single backward pass
after the Kalman filter to compute state estimates conditioned on
the full observation sequence:

$$
J_t = P_{t|t}\, T_t'\, P_{t+1|t}^{-1}
$$

$$
a_{t|n} = a_{t|t} + J_t\,(a_{t+1|n} - a_{t+1|t})
$$

$$
P_{t|n} = P_{t|t} + J_t\,(P_{t+1|n} - P_{t+1|t})\, J_t'
$$

The smoother is exact and runs in $\mathcal{O}(n m^3)$ time.

!!! tip "When to use RTS vs other smoothers"

    | Smoother | Preferred when |
    |---|---|
    | `RTSSmoother` | Default for all linear Gaussian models |
    | `FixedIntervalSmoother` | `P_{t|t}` is near-singular; uses information-filter backward pass |
    | `FixedLagSmoother` | Online smoothing with bounded latency |
    | `DisturbanceSmoother` | Need smoothed disturbances `η_t`, `ε_t` |

### Constructor

```python
RTSSmoother(
    ss: StateSpaceRepresentation,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ss` | `StateSpaceRepresentation` | required | The state-space representation to smooth over. Must match the one used for filtering. |

### Methods

#### `smooth(filter_result)`

```python
def smooth(
    filter_result: FilterResult,
) -> SmoothResult
```

Run the RTS backward pass on the output of a `KalmanFilter.filter()`
call.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `filter_result` | `FilterResult` | Output of `KalmanFilter.filter()`. |

**Returns** `SmoothResult` with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `a_smooth` | `(n, m)` | Smoothed state means `a_{t|n}`. |
| `P_smooth` | `(n, m, m)` | Smoothed state covariances `P_{t|n}`. |
| `V_smooth` | `(n, m, m)` | Covariance of smoothing error `Var(a_t - a_{t|n})`. |
| `J` | `(n, m, m)` | Smoother gain matrices `J_t`. |
| `loglikelihood` | `float` | Carried over from `filter_result`. |

---

#### `disturbance_smooth(filter_result)`

```python
def disturbance_smooth(
    filter_result: FilterResult,
) -> DisturbanceResult
```

Compute smoothed state and observation disturbances via the
de Jong (1989) recursion:

$$
\hat{\eta}_t = Q\, R'\, r_t, \qquad
\hat{\varepsilon}_t = H\, Z'\, r_t - H\, K_t'\, L_t'\, r_{t+1\ldots n}
$$

where `r_t` is the smoothing residual vector.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `filter_result` | `FilterResult` | Output of `KalmanFilter.filter()`. |

**Returns** `DisturbanceResult` with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `eta_smooth` | `(n, r)` | Smoothed state disturbances `η̂_t`. |
| `eps_smooth` | `(n, p)` | Smoothed observation disturbances `ε̂_t`. |
| `eta_var` | `(n, r, r)` | Variance of `η̂_t`. |
| `eps_var` | `(n, p, p)` | Variance of `ε̂_t`. |
| `r` | `(n, m)` | Smoothing residuals `r_t`. |
| `N` | `(n, m, m)` | Precision of smoothing residuals `N_t`. |

---

### Example

```python
import numpy as np
from kalmanbox import KalmanFilter, RTSSmoother, StateSpaceRepresentation

rng = np.random.default_rng(42)
n = 300
sigma_eps, sigma_eta = 2.0, 0.5

# Simulate local level
mu = np.cumsum(rng.normal(0, sigma_eta, n))
y = mu + rng.normal(0, sigma_eps, n)

ss = StateSpaceRepresentation(
    Z=np.array([[1.0]]),
    T=np.array([[1.0]]),
    R=np.array([[1.0]]),
    H=np.array([[sigma_eps**2]]),
    Q=np.array([[sigma_eta**2]]),
)

kf = KalmanFilter(ss, diffuse_init=True)
filter_result = kf.filter(y)

smoother = RTSSmoother(ss)
smooth_result = smoother.smooth(filter_result)

# Smoothed estimates are less variable than filtered
a_filt_var = filter_result.P_filt[:, 0, 0].mean()
a_smooth_var = smooth_result.P_smooth[:, 0, 0].mean()
print(f"Mean filtered variance:  {a_filt_var:.4f}")
print(f"Mean smoothed variance:  {a_smooth_var:.4f}")  # smaller

# Recover disturbances for outlier analysis
dist_result = smoother.disturbance_smooth(filter_result)
standardised_eps = (
    dist_result.eps_smooth.ravel()
    / np.sqrt(dist_result.eps_var[:, 0, 0])
)
print(f"Potential outliers (|z| > 3): {(np.abs(standardised_eps) > 3).sum()}")
```

---

## Result Containers

The result containers are lightweight data classes; all attributes are
NumPy arrays and scalars.

### FilterResult

Returned by `KalmanFilter.filter()`.

| Attribute | Shape | Description |
|---|---|---|
| `a_pred` | `(n+1, m)` | Predicted state means (includes `a_{1|0}` at index 0). |
| `P_pred` | `(n+1, m, m)` | Predicted state covariances. |
| `a_filt` | `(n, m)` | Filtered state means `a_{t|t}`. |
| `P_filt` | `(n, m, m)` | Filtered state covariances `P_{t|t}`. |
| `v` | `(n, p)` | Innovations `v_t`. |
| `F` | `(n, p, p)` | Innovation covariances `F_t`. |
| `K` | `(n, m, p)` | Kalman gain matrices. |
| `loglikelihood` | `float` | Prediction-error log-likelihood. |
| `n_diffuse` | `int` | Diffuse initialisation step count. |

### SmoothResult

Returned by `RTSSmoother.smooth()`.

| Attribute | Shape | Description |
|---|---|---|
| `a_smooth` | `(n, m)` | Smoothed state means `a_{t|n}`. |
| `P_smooth` | `(n, m, m)` | Smoothed state covariances `P_{t|n}`. |
| `V_smooth` | `(n, m, m)` | Var`(a_t - a_{t|n})`. |
| `J` | `(n, m, m)` | RTS smoother gain. |
| `loglikelihood` | `float` | Carried from `FilterResult`. |

### ForecastResult

Returned by `KalmanFilter.predict()`.

| Attribute | Shape | Description |
|---|---|---|
| `forecast` | `(n_ahead, p)` | Point forecasts. |
| `forecast_states` | `(n_ahead, m)` | Predicted state means at each horizon. |
| `lower` | `(n_ahead, p)` | Lower confidence bound. |
| `upper` | `(n_ahead, p)` | Upper confidence bound. |
| `confidence` | `float` | Coverage level (default 0.95). |

### FitResult

Returned by `KalmanFilter.fit()` and high-level model `.fit()`.

| Attribute | Type | Description |
|---|---|---|
| `params` | `dict[str, float]` | Estimated parameters. |
| `loglikelihood` | `float` | Log-likelihood at the optimum. |
| `aic` | `float` | AIC = −2ℓ + 2k. |
| `bic` | `float` | BIC = −2ℓ + k ln n. |
| `hqic` | `float` | HQIC = −2ℓ + 2k ln ln n. |
| `converged` | `bool` | Optimiser convergence flag. |
| `n_iter` | `int` | Number of iterations taken. |
| `filter_result` | `FilterResult` | Filter output at optimal parameters. |
| `std_errors` | `dict[str, float]` | Asymptotic standard errors. |
| `summary()` | — | Print a formatted parameter table. |

### DisturbanceResult

Returned by `RTSSmoother.disturbance_smooth()`.

| Attribute | Shape | Description |
|---|---|---|
| `eta_smooth` | `(n, r)` | Smoothed state disturbances `η̂_t`. |
| `eps_smooth` | `(n, p)` | Smoothed observation disturbances `ε̂_t`. |
| `eta_var` | `(n, r, r)` | Variance of `η̂_t`. |
| `eps_var` | `(n, p, p)` | Variance of `ε̂_t`. |
| `r` | `(n, m)` | de Jong smoothing residuals. |
| `N` | `(n, m, m)` | Precision of smoothing residuals. |

---

## KalmanBoxConfig

`kalmanbox.core.KalmanBoxConfig`

Global configuration singleton for library-wide defaults.

```python
from kalmanbox.core import KalmanBoxConfig

KalmanBoxConfig.set(
    dtype=np.float64,
    diffuse_threshold=1e6,
    default_confidence=0.95,
    numba_enabled=True,
    warn_singular=True,
)
```

| Option | Type | Default | Description |
|---|---|---|---|
| `dtype` | `np.dtype` | `np.float64` | Default floating-point type for all internal arrays. |
| `diffuse_threshold` | `float` | `1e6` | Default threshold for diffuse collapse detection. |
| `default_confidence` | `float` | `0.95` | Default coverage for prediction intervals. |
| `numba_enabled` | `bool` | `True` | Use Numba-JIT kernels when available. |
| `warn_singular` | `bool` | `True` | Emit a warning when `F_t` is near-singular. |

---

## See Also

- [User Guide: Kalman Filter](../user-guide/kalman/kalman-filter.md)
- [User Guide: RTS Smoother](../user-guide/kalman/rts-smoother.md)
- [Theory: Kalman Filter Derivation](../theory/kalman-filter-derivation.md)
- [Theory: Smoothing Theory](../theory/smoothing-theory.md)
- [API: Structural Models](structural.md)
- [API: Alternative Filters](filters.md)
