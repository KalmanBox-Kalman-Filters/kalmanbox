# Alternative Filters API

`kalmanbox.filters`

This page documents the nonlinear and numerically-specialised filter
variants. For the standard linear Gaussian filter see
[`KalmanFilter`](core.md#kalmanfilter).

| Filter | Best for |
|---|---|
| [`ExtendedKalmanFilter`](#extendedkalmanfilter) | Mildly nonlinear models; fast, first-order approximation |
| [`UnscentedKalmanFilter`](#unscentedkalmanfilter) | Moderately nonlinear models; second-order accuracy without Jacobians |
| [`SquareRootFilter`](#squarerootfilter) | Near-singular covariance; numerically ill-conditioned linear models |
| [`InformationFilter`](#informationfilter) | Sensor fusion; diffuse initialisation; sparse precision matrices |
| [`EnsembleKalmanFilter`](#ensemblekalmanfilter) | Very high-dimensional systems; highly nonlinear dynamics |

See [Filter Comparison](../user-guide/filters/comparison.md) for a
detailed analysis of accuracy, speed, and numerical robustness.

---

## ExtendedKalmanFilter

`kalmanbox.filters.ExtendedKalmanFilter`

The Extended Kalman Filter (EKF) linearises a nonlinear state-space model
around the current state estimate at each time step using first-order
Taylor expansion:

$$
y_t = h(a_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R_t)
$$

$$
a_{t+1} = f(a_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)
$$

The Jacobians `H_t = ∂h/∂a |_{a_{t|t-1}}` and `F_t = ∂f/∂a |_{a_{t|t}}`
replace the constant matrices `Z` and `T` of the linear Kalman filter.

!!! warning "EKF limitations"

    The EKF can diverge when the nonlinearity is severe or the initial
    state uncertainty is large. Use the
    [`UnscentedKalmanFilter`](#unscentedkalmanfilter) or
    [`EnsembleKalmanFilter`](#ensemblekalmanfilter) for highly nonlinear
    systems.

### Constructor

```python
ExtendedKalmanFilter(
    state_dim: int,
    obs_dim: int,
    transition_fn: Callable,
    observation_fn: Callable,
    Q: np.ndarray,
    R: np.ndarray,
    jacobian_transition: Callable | None = None,
    jacobian_observation: Callable | None = None,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `state_dim` | `int` | required | State dimension `m`. |
| `obs_dim` | `int` | required | Observation dimension `p`. |
| `transition_fn` | `Callable` | required | Nonlinear state transition `f(a_t, t) → a_{t+1}`. Signature: `(a: ndarray shape (m,), t: int) -> ndarray shape (m,)`. |
| `observation_fn` | `Callable` | required | Nonlinear observation model `h(a_t, t) → y_t`. Signature: `(a: ndarray shape (m,), t: int) -> ndarray shape (p,)`. |
| `Q` | `np.ndarray` | required | State noise covariance. Shape `(m, m)`. |
| `R` | `np.ndarray` | required | Observation noise covariance. Shape `(p, p)`. |
| `jacobian_transition` | `Callable \| None` | `None` | Analytical Jacobian of `f`. Signature: `(a: ndarray, t: int) -> ndarray shape (m, m)`. If `None`, computed by finite differences. |
| `jacobian_observation` | `Callable \| None` | `None` | Analytical Jacobian of `h`. Signature: `(a: ndarray, t: int) -> ndarray shape (p, m)`. If `None`, computed by finite differences. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Methods

#### `filter(y, a0, P0)`

```python
def filter(
    y: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
) -> FilterResult
```

Run the EKF forward pass.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y` | `np.ndarray` | Observations. Shape `(n, p)`. `np.nan` for missing. |
| `a0` | `np.ndarray` | Initial state mean. Shape `(m,)`. |
| `P0` | `np.ndarray` | Initial state covariance. Shape `(m, m)`. |

**Returns** [`FilterResult`](core.md#filterresult).

!!! note

    `FilterResult.K` contains the linearised Kalman gain matrices at
    each step. The `loglikelihood` is the Gaussian pseudo-log-likelihood
    of the linearised model — it is an approximation for nonlinear systems.

---

#### `predict(n_ahead, result)`

```python
def predict(
    n_ahead: int,
    result: FilterResult,
) -> ForecastResult
```

Propagate the last filtered state through the nonlinear transition
function `n_ahead` times to form a point forecast.

**Returns** [`ForecastResult`](core.md#forecastresult).

---

#### `update(a_pred, P_pred, y_t, t)`

```python
def update(
    a_pred: np.ndarray,
    P_pred: np.ndarray,
    y_t: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray]
```

Perform a single EKF update step for observation `y_t`.

**Returns** `(a_filt, P_filt)` — the updated state mean and covariance.

---

### Example

```python
import numpy as np
from kalmanbox.filters import ExtendedKalmanFilter

# Constant-velocity tracking with nonlinear bearing observation
# State: [x, vx, y, vy], Observation: [range, bearing]
def transition(a, t):
    dt = 1.0
    F = np.array([
        [1, dt, 0,  0],
        [0,  1, 0,  0],
        [0,  0, 1, dt],
        [0,  0, 0,  1],
    ])
    return F @ a

def observation(a, t):
    x, _, y, _ = a
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.array([r, theta])

def jac_obs(a, t):
    x, _, y, _ = a
    r2 = x**2 + y**2
    r  = np.sqrt(r2)
    return np.array([
        [ x/r,  0, y/r, 0],
        [-y/r2, 0, x/r2, 0],
    ])

Q = np.diag([0.1, 0.01, 0.1, 0.01])
R = np.diag([0.5, 0.01])

ekf = ExtendedKalmanFilter(
    state_dim=4,
    obs_dim=2,
    transition_fn=transition,
    observation_fn=observation,
    Q=Q,
    R=R,
    jacobian_observation=jac_obs,
)

# Simulate and filter
rng = np.random.default_rng(0)
n   = 100
a_true = np.zeros((n + 1, 4))
a_true[0] = [10.0, 0.5, 5.0, 0.3]
y = np.zeros((n, 2))
for t in range(n):
    a_true[t+1] = transition(a_true[t], t) + rng.multivariate_normal(np.zeros(4), Q)
    y[t] = observation(a_true[t+1], t) + rng.multivariate_normal(np.zeros(2), R)

a0 = np.array([10.0, 0.5, 5.0, 0.3])
P0 = np.eye(4) * 5.0

result = ekf.filter(y, a0, P0)
rmse = np.sqrt(np.mean((result.a_filt[:, [0, 2]] - a_true[1:, [0, 2]])**2))
print(f"Position RMSE: {rmse:.4f}")
```

---

## UnscentedKalmanFilter

`kalmanbox.filters.UnscentedKalmanFilter`

The Unscented Kalman Filter (UKF) propagates a set of deterministically
chosen **sigma points** through the nonlinear functions to capture the
true mean and covariance to second-order accuracy without computing
Jacobians.

Given state mean `a` and covariance `P`, the 2m+1 sigma points are:

$$
\mathcal{X}_0 = a, \quad
\mathcal{X}_i = a + \left(\sqrt{(m+\lambda)P}\right)_i, \quad
\mathcal{X}_{m+i} = a - \left(\sqrt{(m+\lambda)P}\right)_i
$$

with `λ = α²(m + κ) − m`.

!!! tip "UKF vs EKF"

    The UKF matches the EKF accuracy on linear models but achieves
    second-order accuracy on nonlinear models without user-supplied
    Jacobians. It is typically preferred over the EKF when the
    nonlinearity is moderate.

### Constructor

```python
UnscentedKalmanFilter(
    state_dim: int,
    obs_dim: int,
    transition_fn: Callable,
    observation_fn: Callable,
    Q: np.ndarray,
    R: np.ndarray,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `state_dim` | `int` | required | State dimension `m`. |
| `obs_dim` | `int` | required | Observation dimension `p`. |
| `transition_fn` | `Callable` | required | State transition `f(a_t, t) → a_{t+1}`. |
| `observation_fn` | `Callable` | required | Observation function `h(a_t, t) → y_t`. |
| `Q` | `np.ndarray` | required | State noise covariance. Shape `(m, m)`. |
| `R` | `np.ndarray` | required | Observation noise covariance. Shape `(p, p)`. |
| `alpha` | `float` | `1e-3` | Spread of sigma points around the mean. Typically `1e-4 ≤ α ≤ 1`. |
| `beta` | `float` | `2.0` | Prior knowledge of the distribution. `β = 2` is optimal for Gaussian. |
| `kappa` | `float` | `0.0` | Secondary scaling. Typically `0` or `3 - m`. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Properties

| Property | Type | Description |
|---|---|---|
| `lambda_` | `float` | Composite scaling parameter `λ = α²(m + κ) − m`. |
| `weights_mean` | `np.ndarray` | Mean weights for sigma points. Shape `(2m+1,)`. |
| `weights_cov` | `np.ndarray` | Covariance weights for sigma points. Shape `(2m+1,)`. |

### Methods

#### `filter(y, a0, P0)`

```python
def filter(
    y: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
) -> FilterResult
```

Run the UKF forward pass.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y` | `np.ndarray` | Observations. Shape `(n, p)`. |
| `a0` | `np.ndarray` | Initial state mean. Shape `(m,)`. |
| `P0` | `np.ndarray` | Initial state covariance. Shape `(m, m)`. |

**Returns** [`FilterResult`](core.md#filterresult).

---

#### `sigma_points(a, P)`

```python
def sigma_points(
    a: np.ndarray,
    P: np.ndarray,
) -> np.ndarray
```

Compute the 2m+1 sigma points for given state mean and covariance.

**Returns** `np.ndarray` of shape `(2m+1, m)`.

---

#### `unscented_transform(sigma_pts, weights_mean, weights_cov, noise_cov=None)`

```python
def unscented_transform(
    sigma_pts: np.ndarray,
    weights_mean: np.ndarray,
    weights_cov: np.ndarray,
    noise_cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]
```

Compute the weighted mean and covariance from transformed sigma points.

**Returns** `(mean, covariance)`.

---

#### `predict(n_ahead, result)`

```python
def predict(
    n_ahead: int,
    result: FilterResult,
) -> ForecastResult
```

Monte Carlo forecast by propagating sigma points through `transition_fn`.

**Returns** [`ForecastResult`](core.md#forecastresult).

---

### Example

```python
import numpy as np
from kalmanbox.filters import UnscentedKalmanFilter

# Stochastic volatility model: log-volatility as latent state
# y_t  = exp(h_t / 2) * eps_t,  eps_t ~ N(0,1)
# h_{t+1} = mu + phi*(h_t - mu) + sigma * eta_t

mu, phi, sigma_h = -0.5, 0.97, 0.15

def transition(h, t):
    return np.array([mu + phi * (h[0] - mu)])

def observation(h, t):
    return np.array([0.0])   # mean zero; variance absorbed in R below

Q = np.array([[sigma_h**2]])
R = np.array([[1.0]])  # normalised; actual variance is exp(h_t)

ukf = UnscentedKalmanFilter(
    state_dim=1,
    obs_dim=1,
    transition_fn=transition,
    observation_fn=observation,
    Q=Q,
    R=R,
    alpha=1e-3,
    beta=2.0,
    kappa=0.0,
)

# Simulate
rng = np.random.default_rng(7)
n = 500
h = np.zeros(n + 1)
h[0] = mu
for t in range(n):
    h[t+1] = mu + phi * (h[t] - mu) + sigma_h * rng.standard_normal()
y = np.exp(h[1:] / 2) * rng.standard_normal(n)

a0 = np.array([mu])
P0 = np.array([[sigma_h**2 / (1 - phi**2)]])

result = ukf.filter(y[:, np.newaxis], a0, P0)
print(f"Filtered log-vol at t=100: {result.a_filt[100, 0]:.4f}")
print(f"True log-vol at t=100:     {h[101]:.4f}")
```

---

## SquareRootFilter

`kalmanbox.filters.SquareRootFilter`

The Square-Root Kalman Filter propagates the **Cholesky factor** of the
covariance `S_t = chol(P_t)` instead of `P_t` itself, using QR
decompositions in the prediction and update steps:

$$
\begin{bmatrix} S_{t+1|t}' \\ 0 \end{bmatrix} = \text{qr}\left(
\begin{bmatrix} S_{t|t}' T_t' \\ S_\eta' R_t' \end{bmatrix}
\right)
$$

This guarantees positive semi-definiteness of `P_t` to machine precision
and roughly halves the condition number relative to the standard Kalman
filter, which is critical for:

- Models with near-unit-root states (very small innovations)
- Long time series where numerical errors accumulate
- Systems where `P_{t|t}` is close to rank-deficient

### Constructor

```python
SquareRootFilter(
    ss: StateSpaceRepresentation,
    cholesky_method: str = "upper",
    clip_negative: bool = True,
    diffuse_init: bool = False,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ss` | `StateSpaceRepresentation` | required | State-space system matrices. |
| `cholesky_method` | `str` | `"upper"` | Convention for the triangular factor: `"upper"` (`P = U'U`) or `"lower"` (`P = LL'`). |
| `clip_negative` | `bool` | `True` | Clip negative eigenvalues to zero before the Cholesky factorisation during update. Improves robustness at the cost of exact symmetry. |
| `diffuse_init` | `bool` | `False` | Use a square-root form of the diffuse initialisation recursion. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Methods

#### `filter(y, a0=None, S0=None)`

```python
def filter(
    y: np.ndarray,
    a0: np.ndarray | None = None,
    S0: np.ndarray | None = None,
) -> SquareRootFilterResult
```

Run the square-root Kalman filter.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` | required | Observations. Shape `(n,)` or `(n, p)`. |
| `a0` | `np.ndarray \| None` | `None` | Initial state mean. |
| `S0` | `np.ndarray \| None` | `None` | Initial Cholesky factor `S_0 = chol(P_0)`. Shape `(m, m)`. Defaults to `chol(1e6 * I)`. |

**Returns** `SquareRootFilterResult` with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `a_filt` | `(n, m)` | Filtered state means. |
| `S_filt` | `(n, m, m)` | Cholesky factors `S_{t|t}`. |
| `a_pred` | `(n+1, m)` | Predicted state means. |
| `S_pred` | `(n+1, m, m)` | Predicted Cholesky factors `S_{t|t-1}`. |
| `v` | `(n, p)` | Innovations. |
| `loglikelihood` | `float` | Log-likelihood. |
| `P_filt` | `(n, m, m)` | Reconstructed `P_{t|t} = S_{t|t}' S_{t|t}` (computed on demand). |
| `P_pred` | `(n+1, m, m)` | Reconstructed `P_{t|t-1}`. |

---

### Example

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation
from kalmanbox.filters import SquareRootFilter

# Ill-conditioned model: near-singular observation noise
n, m, p = 500, 4, 2
rng = np.random.default_rng(1)

T = np.eye(m) * 0.99
Z = rng.standard_normal((p, m))
R = np.eye(m)
H = np.diag([1e-6, 1e-6])   # very small: standard filter may lose P.D.
Q = np.eye(m) * 0.01

ss = StateSpaceRepresentation(Z=Z, T=T, R=R, H=H, Q=Q)

sqf = SquareRootFilter(ss, cholesky_method="upper")
y = rng.standard_normal((n, p))
result = sqf.filter(y)

# P_filt should remain positive definite throughout
min_eigenvalue = np.linalg.eigvalsh(result.P_filt).min()
print(f"Minimum eigenvalue of P_filt: {min_eigenvalue:.2e}")  # > 0
```

---

## InformationFilter

`kalmanbox.filters.InformationFilter`

The Information Filter is the dual form of the Kalman filter, propagating
the **information matrix** `Ω_t = P_t^{-1}` and the **information vector**
`ξ_t = P_t^{-1} a_t` instead of `(a_t, P_t)`:

**Update** (trivially additive):

$$
\Omega_{t|t} = \Omega_{t|t-1} + Z_t' H_t^{-1} Z_t, \qquad
\xi_{t|t}   = \xi_{t|t-1}   + Z_t' H_t^{-1} y_t
$$

**Prediction** (requires matrix inversion):

$$
\Omega_{t+1|t} = \bigl(T_t \Omega_{t|t}^{-1} T_t' + R_t Q_t R_t'\bigr)^{-1}
$$

!!! info "Advantages of the information form"

    - Exact diffuse initialisation: start with `Ω_0 = 0` (infinite variance).
    - Sensor fusion: updates from multiple sensors are additive.
    - Sparse systems: `Ω_t` may be sparse even when `P_t` is dense.

### Constructor

```python
InformationFilter(
    ss: StateSpaceRepresentation,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ss` | `StateSpaceRepresentation` | required | State-space system matrices. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Methods

#### `filter(y, Omega0=None, xi0=None)`

```python
def filter(
    y: np.ndarray,
    Omega0: np.ndarray | None = None,
    xi0: np.ndarray | None = None,
) -> InformationFilterResult
```

Run the information filter forward pass.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` | required | Observations. Shape `(n,)` or `(n, p)`. |
| `Omega0` | `np.ndarray \| None` | `None` | Initial information matrix `P_0^{-1}`. Shape `(m, m)`. Defaults to `0` (diffuse: `P_0 → ∞`). |
| `xi0` | `np.ndarray \| None` | `None` | Initial information vector `P_0^{-1} a_0`. Shape `(m,)`. Defaults to zeros. |

**Returns** `InformationFilterResult` with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `Omega_filt` | `(n, m, m)` | Filtered information matrices `Ω_{t|t}`. |
| `xi_filt` | `(n, m)` | Filtered information vectors `ξ_{t|t}`. |
| `a_filt` | `(n, m)` | Filtered state means `Ω^{-1} ξ` (computed on demand). |
| `P_filt` | `(n, m, m)` | Filtered covariances `Ω^{-1}` (computed on demand). |
| `v` | `(n, p)` | Innovations (for models with `Ω_0 > 0`). |
| `loglikelihood` | `float` | Prediction-error log-likelihood. |

---

#### `fuse(y_list, R_list, t)`

```python
def fuse(
    y_list: list[np.ndarray],
    R_list: list[np.ndarray],
    t: int,
) -> tuple[np.ndarray, np.ndarray]
```

Fuse multiple sensor observations at a single time step using the
additive property of the information update. Returns the updated
`(Omega, xi)` after incorporating all sensors.

| Parameter | Type | Description |
|---|---|---|
| `y_list` | `list[np.ndarray]` | List of sensor observations at time `t`. |
| `R_list` | `list[np.ndarray]` | List of sensor noise covariances. |
| `t` | `int` | Time index. |

---

### Example

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation
from kalmanbox.filters import InformationFilter

# Diffuse initialisation via information filter
n, m = 200, 2
rng = np.random.default_rng(3)

ss = StateSpaceRepresentation(
    Z=np.array([[1.0, 0.0]]),
    T=np.array([[1.0, 1.0], [0.0, 1.0]]),
    R=np.eye(2),
    H=np.array([[4.0]]),
    Q=np.diag([0.1, 0.01]),
)

# Exact diffuse initialisation: Omega0 = 0
inf_filter = InformationFilter(ss)
y = rng.standard_normal((n, 1)) * 2

result = inf_filter.filter(y, Omega0=np.zeros((2, 2)), xi0=np.zeros(2))

# Extract state means from information quantities
a_filt = result.a_filt
print(f"Filtered position at t=100: {a_filt[100, 0]:.4f}")
print(f"Filtered velocity at t=100: {a_filt[100, 1]:.4f}")
```

---

## EnsembleKalmanFilter

`kalmanbox.filters.EnsembleKalmanFilter`

The Ensemble Kalman Filter (EnKF) represents the state distribution as
a finite ensemble of `N_e` particles `{a_t^{(i)}}_{i=1}^{N_e}` and
approximates the Kalman equations by sample covariances:

$$
\hat{P}_{t|t-1} = \frac{1}{N_e - 1} \sum_{i=1}^{N_e}
  (a_t^{(i)} - \bar{a}_t)(a_t^{(i)} - \bar{a}_t)'
$$

The update perturbs each ensemble member with a sample from the
observation noise (stochastic EnKF) or uses the deterministic square-root
update (ETKF). Localization and inflation prevent ensemble collapse in
large systems.

!!! info "EnKF vs particle filter"

    The EnKF is **not** a particle filter: it assumes Gaussian noise and
    uses a linear update rule even for nonlinear models. For strongly
    non-Gaussian posteriors, prefer `bootstrap_filter` in
    `kalmanbox.simulation`.

### Constructor

```python
EnsembleKalmanFilter(
    transition_fn: Callable,
    observation_fn: Callable,
    ensemble_size: int,
    Q: np.ndarray,
    R: np.ndarray,
    localization_radius: float | None = None,
    inflation_factor: float = 1.0,
    update_type: str = "stochastic",
    state_dim: int | None = None,
    obs_dim: int | None = None,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transition_fn` | `Callable` | required | Nonlinear transition `f(ensemble, t) → ensemble`. Signature: `(A: ndarray shape (Ne, m), t: int) -> ndarray shape (Ne, m)`. |
| `observation_fn` | `Callable` | required | Nonlinear observation `h(ensemble, t) → obs`. Signature: `(A: ndarray shape (Ne, m), t: int) -> ndarray shape (Ne, p)`. |
| `ensemble_size` | `int` | required | Number of ensemble members `N_e`. Typically 50–500. |
| `Q` | `np.ndarray` | required | State noise covariance. Shape `(m, m)`. |
| `R` | `np.ndarray` | required | Observation noise covariance. Shape `(p, p)`. |
| `localization_radius` | `float \| None` | `None` | Gaspari–Cohn localization radius in grid-space units. `None` disables localization. |
| `inflation_factor` | `float` | `1.0` | Multiplicative inflation `ρ ≥ 1` applied to ensemble anomalies before update. |
| `update_type` | `str` | `"stochastic"` | Update scheme: `"stochastic"` (perturbed-observation) or `"etkf"` (ensemble transform, deterministic). |
| `state_dim` | `int \| None` | `None` | State dimension `m`. Inferred from the first ensemble draw if `None`. |
| `obs_dim` | `int \| None` | `None` | Observation dimension `p`. Inferred from `observation_fn` if `None`. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Properties

| Property | Type | Description |
|---|---|---|
| `ensemble_size` | `int` | Number of ensemble members `N_e`. |
| `inflation_factor` | `float` | Current multiplicative inflation coefficient. |

### Methods

#### `filter(y, ensemble0)`

```python
def filter(
    y: np.ndarray,
    ensemble0: np.ndarray,
) -> EnsembleFilterResult
```

Run the EnKF on the observation sequence `y`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y` | `np.ndarray` | Observations. Shape `(n, p)`. |
| `ensemble0` | `np.ndarray` | Initial ensemble. Shape `(N_e, m)`. Typically drawn from the prior `N(a_0, P_0)`. |

**Returns** `EnsembleFilterResult` with attributes:

| Attribute | Shape | Description |
|---|---|---|
| `ensemble_mean` | `(n, m)` | Per-step sample mean of the filtered ensemble. |
| `ensemble_cov` | `(n, m, m)` | Per-step sample covariance. |
| `ensemble` | `(n, Ne, m)` | Full ensemble trajectory (stored if `store_ensemble=True`). |
| `loglikelihood` | `float` | Approximate Gaussian log-likelihood from ensemble statistics. |

---

#### `ensemble_mean(t=None)`

```python
def ensemble_mean(t: int | None = None) -> np.ndarray
```

Return the ensemble mean at step `t` (all steps if `t` is `None`).

---

#### `ensemble_cov(t=None)`

```python
def ensemble_cov(t: int | None = None) -> np.ndarray
```

Return the ensemble sample covariance at step `t`.

---

### Example

```python
import numpy as np
from kalmanbox.filters import EnsembleKalmanFilter

# Lorenz-96 attractor — 40-dimensional chaotic system
M = 40   # state dimension
F = 8.0  # forcing

def lorenz96(a, t):
    """Lorenz-96 RK4 step with dt=0.05."""
    def rhs(x):
        return np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + F
    dt = 0.05
    k1 = rhs(a);        k2 = rhs(a + 0.5*dt*k1)
    k3 = rhs(a + 0.5*dt*k2); k4 = rhs(a + dt*k3)
    return a + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def transition_ensemble(A, t):
    return np.array([lorenz96(A[i], t) for i in range(A.shape[0])])

# Observe every other grid point
obs_idx = np.arange(0, M, 2)  # 20 observations
H_mat = np.zeros((len(obs_idx), M))
for i, j in enumerate(obs_idx):
    H_mat[i, j] = 1.0

def obs_ensemble(A, t):
    return A @ H_mat.T

Q = np.eye(M) * 0.01
R = np.eye(len(obs_idx)) * 0.5

enkf = EnsembleKalmanFilter(
    transition_fn=transition_ensemble,
    observation_fn=obs_ensemble,
    ensemble_size=100,
    Q=Q,
    R=R,
    localization_radius=4.0,
    inflation_factor=1.05,
    update_type="etkf",
)

rng = np.random.default_rng(42)
n = 200
# Simulate true trajectory and noisy observations
a_true = np.zeros((n+1, M))
a_true[0] = F + rng.standard_normal(M) * 0.01
for t in range(n):
    a_true[t+1] = lorenz96(a_true[t], t) + rng.multivariate_normal(np.zeros(M), Q)
y = a_true[1:, obs_idx] + rng.multivariate_normal(np.zeros(len(obs_idx)), R, size=n)

ensemble0 = F + rng.standard_normal((100, M)) * 1.0
result = enkf.filter(y, ensemble0)

rmse = np.sqrt(np.mean((result.ensemble_mean - a_true[1:])**2))
print(f"Analysis RMSE: {rmse:.4f}")
```

---

## See Also

- [User Guide: Alternative Filters](../user-guide/filters/index.md)
- [User Guide: EKF](../user-guide/filters/ekf.md)
- [User Guide: UKF](../user-guide/filters/ukf.md)
- [User Guide: Square-Root Filter](../user-guide/filters/square-root.md)
- [User Guide: Information Filter](../user-guide/filters/information.md)
- [User Guide: Ensemble Kalman Filter](../user-guide/filters/ensemble.md)
- [User Guide: Filter Comparison](../user-guide/filters/comparison.md)
- [Theory: Nonlinear Filter Theory](../theory/nonlinear-theory.md)
- [API: Core (KalmanFilter)](core.md)
