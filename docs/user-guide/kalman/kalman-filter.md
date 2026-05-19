# Kalman Filter

The [`KalmanFilter`][kalmanbox.filters.kalman.KalmanFilter] class implements the
**forward recursion** for a linear Gaussian state-space model. It is the core
computational engine of kalmanbox and the foundation of every higher-level model.

---

## Mathematical formulation

### The state-space model

$$
\begin{aligned}
\alpha_{t+1} &= T_t\,\alpha_t + c_t + R_t\,\eta_t, &
\eta_t &\sim \mathcal{N}(0,\,Q_t),\quad t = 1,\ldots,n \\
y_t &= Z_t\,\alpha_t + d_t + \varepsilon_t, &
\varepsilon_t &\sim \mathcal{N}(0,\,H_t)
\end{aligned}
$$

$\alpha_t \in \mathbb{R}^k$ is the latent state, $y_t \in \mathbb{R}^p$ the
observation, and $\eta_t \perp \varepsilon_t$ for all $t$.

### Prediction step

Given $a_{t-1|t-1}$ and $P_{t-1|t-1}$, propagate through the transition equation:

$$
\boxed{
\begin{aligned}
a_{t|t-1} &= T_{t-1}\,a_{t-1|t-1} + c_{t-1} \\
P_{t|t-1} &= T_{t-1}\,P_{t-1|t-1}\,T_{t-1}' + R_{t-1}\,Q_{t-1}\,R_{t-1}'
\end{aligned}
}
$$

The predicted state $a_{t|t-1}$ is the MMSE estimate of $\alpha_t$ given only
$y_{1:t-1}$.

### Update step

When $y_t$ arrives, compute the **innovation** (prediction error) and the
**innovation covariance**, then update:

$$
\boxed{
\begin{aligned}
v_t &= y_t - Z_t\,a_{t|t-1} - d_t \\[4pt]
F_t &= Z_t\,P_{t|t-1}\,Z_t' + H_t \\[4pt]
K_t &= P_{t|t-1}\,Z_t'\,F_t^{-1} \quad\text{(Kalman gain)} \\[4pt]
a_{t|t} &= a_{t|t-1} + K_t\,v_t \\[4pt]
P_{t|t} &= \bigl(I - K_t Z_t\bigr)\,P_{t|t-1}\,\bigl(I - K_t Z_t\bigr)'
          + K_t\,H_t\,K_t' \quad\text{(Joseph form)}
\end{aligned}
}
$$

!!! info "Why the Joseph form?"

    The algebraically equivalent expression $P_{t|t} = (I - K_t Z_t) P_{t|t-1}$
    is slightly cheaper but can accumulate asymmetry and lose positive-definiteness
    under finite-precision arithmetic. The Joseph form is symmetric by construction
    and is used throughout kalmanbox. See
    [Numerical stability](../../theory/numerical-stability.md).

### Log-likelihood

The Kalman recursion produces the **prediction-error decomposition** of the
log-likelihood as a by-product:

$$
\log p(y_{1:n} \mid \theta)
= -\frac{1}{2}\sum_{t=1}^{n}
  \Bigl[p\log 2\pi + \log\!\det F_t + v_t'\,F_t^{-1}\,v_t\Bigr]
$$

where $p = \dim y_t$ and $\theta$ collects the free system parameters.
This is the objective used by [`MLEEstimator`][kalmanbox.estimation.mle.MLEEstimator]
and [`EMEstimator`][kalmanbox.advanced.EMEstimator] for parameter estimation.

!!! note "Diffuse observations excluded"

    When using exact diffuse initialisation, the first $d$ observations
    contribute to the *diffuse* log-likelihood ($\ell_d$) separately and
    are excluded from the Gaussian term above. `out.loglike` always returns
    the correct combined value. See
    [Diffuse initialisation](diffuse-initialization.md).

---

## System matrices

Every call to `KalmanFilter` requires a
[`StateSpaceRepresentation`][kalmanbox.core.StateSpaceRepresentation] (SSR)
object that bundles the system matrices.

| Argument | Shape                   | Required | Description                                  |
|----------|-------------------------|:--------:|----------------------------------------------|
| `T`      | `(k, k)` or `(n, k, k)` | Yes      | State transition matrix                      |
| `Z`      | `(p, k)` or `(n, p, k)` | Yes      | Observation (design) matrix                  |
| `R`      | `(k, g)` or `(n, k, g)` | Yes      | Selection matrix for state disturbances      |
| `Q`      | `(g, g)` or `(n, g, g)` | Yes      | State disturbance covariance                 |
| `H`      | `(p, p)` or `(n, p, p)` | Yes      | Observation noise covariance                 |
| `c`      | `(k,)` or `(n, k)`      | No       | State intercept (default: zeros)             |
| `d`      | `(p,)` or `(n, p)`      | No       | Observation intercept (default: zeros)       |

A **time-invariant** system passes 2-D arrays; a **time-varying** system passes
3-D arrays with first dimension $n$ (number of observations).

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation

k, p, g = 2, 1, 2      # state dim, obs dim, shock dim

# Time-invariant system
T = np.array([[1.0, 1.0],
              [0.0, 1.0]])   # local linear trend
Z = np.array([[1.0, 0.0]])
R = np.eye(k)
Q = np.diag([0.5, 0.05])
H = np.array([[1.0]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

# Time-varying observation matrix (n=200 time steps)
n = 200
X = np.random.randn(n, 3)                      # 3 regressors
Z_tv = X[:, np.newaxis, :]                     # shape (200, 1, 3)
ssr_tv = StateSpaceRepresentation(T=..., Z=Z_tv, R=..., Q=..., H=...)
```

---

## Initialization

The filter requires a starting value $a_{1|0}$ and $P_{1|0}$. kalmanbox supports
three initialization strategies.

### Exact diffuse initialization (recommended default)

For models with non-stationary components (random walks, integrated processes),
the initial state distribution is improper ($P_{1|0} \to \infty$ for the
diffuse components). kalmanbox implements the **exact diffuse algorithm** of
Durbin & Koopman (2012, §5.7) which handles this rigorously without numerical
substitution of a large constant.

```python
from kalmanbox import KalmanFilter

kf = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)
```

!!! tip

    `"diffuse"` is the safe default whenever the state contains a random walk
    or unit-root component (local level, local linear trend, BSM, etc.).

### Stationary initialization

If all eigenvalues of $T$ lie strictly inside the unit circle, the unconditional
distribution $\alpha_1 \sim \mathcal{N}(0, P_\infty)$ exists where $P_\infty$
is the solution to the **discrete Lyapunov equation**:

$$
P_\infty = T\,P_\infty\,T' + R\,Q\,R'
$$

kalmanbox solves this via `scipy.linalg.solve_discrete_lyapunov`.

```python
kf = KalmanFilter(ssr, initialization="stationary")
```

### Custom initialization

Provide your own prior mean and covariance:

```python
a1  = np.zeros(k)
P1  = np.eye(k) * 1e4    # vague but proper prior

kf = KalmanFilter(ssr, initialization="custom", a1=a1, P1=P1)
```

!!! warning "Avoid P1 = kappa * I with large kappa"

    This approximates the diffuse prior but pollutes the log-likelihood for
    the first few observations. Prefer `initialization="diffuse"` for an
    exact treatment.

---

## Step-by-step filtering

### Filtering a complete series

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

# ── Local Level model ─────────────────────────────────────────────────────────
#   alpha_{t+1} = alpha_t + eta_t,   eta_t ~ N(0, sigma_eta^2)
#   y_t         = alpha_t + eps_t,   eps_t ~ N(0, sigma_eps^2)

sigma_eta = 0.5
sigma_eps = 1.0

T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[sigma_eta**2]])
H = np.array([[sigma_eps**2]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
kf  = KalmanFilter(ssr, initialization="diffuse")

# Simulate data
rng = np.random.default_rng(42)
n   = 300
eta = rng.normal(scale=sigma_eta, size=n)
eps = rng.normal(scale=sigma_eps, size=n)
alpha_true = np.cumsum(eta)
y = alpha_true + eps

# Run the filter
out = kf.run(y)

print(f"Log-likelihood        : {out.loglike:.4f}")
print(f"Predicted states      : {out.a.shape}")          # (301, 1) — includes t=0
print(f"Filtered states       : {out.a_filtered.shape}") # (300, 1)
print(f"Innovations           : {out.v.shape}")           # (300, 1)
print(f"Innovation covariance : {out.F.shape}")           # (300, 1, 1)
print(f"Kalman gain           : {out.K.shape}")           # (300, 1, 1)
```

### Online (step-by-step) filtering

For streaming or real-time applications, drive the filter one observation at a
time using the low-level step methods:

```python
from kalmanbox import KalmanFilter, StateSpaceRepresentation

kf = KalmanFilter(ssr, initialization="diffuse")
kf.initialize()    # sets a_{1|0}, P_{1|0}

for t, y_t in enumerate(y_stream):
    state = kf.predict_step(t)          # returns (a_{t|t-1}, P_{t|t-1})
    update = kf.update_step(t, y_t)    # returns (a_{t|t}, P_{t|t}, v_t, F_t)

    # Use update.a_filtered, update.v, update.F in real time
    if should_raise_alert(update.v, update.F):
        handle_outlier(t, y_t, update)
```

---

## Accessing results

`kf.run(y)` returns a [`FilterOutput`][kalmanbox.filters.kalman.FilterOutput]
dataclass with the following attributes:

```python
out = kf.run(y)

# ── State estimates ───────────────────────────────────────────────────────────
out.a            # predicted means     np.ndarray (n+1, k)  — a_{t|t-1}, t=1..n+1
out.P            # predicted covs      np.ndarray (n+1, k, k)
out.a_filtered   # filtered means      np.ndarray (n, k)    — a_{t|t}
out.P_filtered   # filtered covs       np.ndarray (n, k, k)

# ── Innovations ───────────────────────────────────────────────────────────────
out.v            # innovations         np.ndarray (n, p)    — y_t - Z_t a_{t|t-1}
out.F            # innovation covs     np.ndarray (n, p, p)
out.F_inv        # F_t^{-1}           np.ndarray (n, p, p)

# ── Gains and likelihood ─────────────────────────────────────────────────────
out.K            # Kalman gain         np.ndarray (n, k, p)
out.loglike      # log p(y | theta)    float

# ── Convenience properties ───────────────────────────────────────────────────
out.filtered_state       # alias for out.a_filtered
out.filtered_state_cov   # alias for out.P_filtered
out.predicted_state      # alias for out.a
out.predicted_state_cov  # alias for out.P
out.standardized_residuals  # v_t / sqrt(F_t), useful for diagnostics
```

---

## Examples

### Example 1: Random walk with drift

A random walk with drift (constant mean growth $\mu$):

$$
\alpha_{t+1} = \alpha_t + \mu + \eta_t, \qquad y_t = \alpha_t + \varepsilon_t
$$

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

# Augment state: [alpha_t, 1] to absorb the drift
T = np.array([[1.0, 1.0],   # alpha_{t+1} = alpha_t + 1*drift
              [0.0, 1.0]])   # drift is constant
Z = np.array([[1.0, 0.0]])  # observe alpha only
R = np.array([[1.0],
              [0.0]])        # shock only enters alpha
Q = np.array([[0.3]])        # sigma_eta^2
H = np.array([[1.0]])        # sigma_eps^2

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

# Diffuse init: both alpha_0 and drift are non-stationary
kf  = KalmanFilter(ssr, initialization="diffuse")

rng  = np.random.default_rng(7)
n    = 150
true_drift = 0.2
alpha_true = np.cumsum(true_drift + rng.normal(scale=np.sqrt(0.3), size=n))
y = alpha_true + rng.normal(scale=1.0, size=n)

out = kf.run(y)

# out.a_filtered[:, 0] — filtered trend
# out.a_filtered[:, 1] — filtered drift estimate (should ≈ 0.2)
print(f"Estimated drift: {out.a_filtered[-1, 1]:.4f}  (true: {true_drift})")
```

### Example 2: Local Linear Trend

The canonical Local Linear Trend model:

$$
\begin{aligned}
\mu_{t+1} &= \mu_t + \nu_t + \eta_t^{(\mu)}, &\eta_t^{(\mu)} &\sim \mathcal{N}(0,\sigma_\mu^2) \\
\nu_{t+1} &= \nu_t + \eta_t^{(\nu)},          &\eta_t^{(\nu)} &\sim \mathcal{N}(0,\sigma_\nu^2) \\
y_t        &= \mu_t + \varepsilon_t,           &\varepsilon_t   &\sim \mathcal{N}(0,\sigma_\varepsilon^2)
\end{aligned}
$$

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

sigma_mu  = 0.3   # level disturbance
sigma_nu  = 0.05  # slope disturbance
sigma_eps = 1.0   # observation noise

T = np.array([[1.0, 1.0],
              [0.0, 1.0]])
Z = np.array([[1.0, 0.0]])
R = np.eye(2)
Q = np.diag([sigma_mu**2, sigma_nu**2])
H = np.array([[sigma_eps**2]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
kf  = KalmanFilter(ssr, initialization="diffuse")

# Simulate
rng = np.random.default_rng(0)
n   = 200
mu  = np.zeros(n + 1)
nu  = np.zeros(n + 1)
nu[0] = 0.1
for t in range(n):
    nu[t+1] = nu[t] + rng.normal(scale=sigma_nu)
    mu[t+1] = mu[t] + nu[t] + rng.normal(scale=sigma_mu)
y = mu[1:] + rng.normal(scale=sigma_eps, size=n)

out = kf.run(y)

mu_filtered = out.a_filtered[:, 0]   # filtered level
nu_filtered = out.a_filtered[:, 1]   # filtered slope
print(f"Final filtered slope: {nu_filtered[-1]:.4f}")
```

### Example 3: Dynamic regression (time-varying coefficients)

A regression $y_t = x_t'\beta_t + \varepsilon_t$ where the coefficient vector
$\beta_t$ follows a random walk:

$$
\beta_{t+1} = \beta_t + \eta_t, \qquad y_t = x_t'\beta_t + \varepsilon_t
$$

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

rng = np.random.default_rng(1)
n, k = 250, 3     # observations, regressors

# Simulate slowly drifting coefficients
beta_true = np.zeros((n + 1, k))
beta_true[0] = [1.0, -0.5, 0.8]
for t in range(n):
    beta_true[t+1] = beta_true[t] + rng.normal(scale=0.02, size=k)

X = rng.standard_normal((n, k))              # regressors
y = np.einsum("ti,ti->t", X, beta_true[1:]) + rng.normal(scale=0.5, size=n)

# Time-varying Z_t = x_t'  (shape n x 1 x k)
Z_tv = X[:, np.newaxis, :]   # (n, 1, k)
T    = np.eye(k)              # random walk coefficients
R    = np.eye(k)
Q    = np.eye(k) * 0.02**2   # coefficient drift variance
H    = np.array([[0.25]])     # observation variance

ssr = StateSpaceRepresentation(T=T, Z=Z_tv, R=R, Q=Q, H=H)
kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)

# out.a_filtered[t] ≈ beta_true[t+1]  (k-vector)
print(f"Final coefficient estimates: {out.a_filtered[-1]}")
print(f"True final coefficients    : {beta_true[-1]}")
```

---

## Known vs. unknown parameters

### Known parameters

When all system matrices are fixed, run the filter directly:

```python
kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)
loglike = out.loglike
```

### Unknown parameters: MLE

Estimate free parameters by maximizing the prediction-error log-likelihood:

```python
from kalmanbox.estimation import MLEEstimator

def build_ssr(params):
    sigma_eta, sigma_eps = np.exp(params)   # log-parameterization
    return StateSpaceRepresentation(
        T=np.array([[1.0]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma_eta**2]]),
        H=np.array([[sigma_eps**2]]),
    )

mle = MLEEstimator(
    build_ssr=build_ssr,
    initialization="diffuse",
    params0=np.log([0.5, 1.0]),   # starting values on log scale
)
result = mle.fit(y)

print(result.summary())
sigma_eta_hat, sigma_eps_hat = np.exp(result.params)
```

### Unknown parameters: EM algorithm

For models where the M-step has a closed form (e.g., DFM), the EM algorithm
is often faster and more stable than gradient-based MLE:

```python
from kalmanbox.advanced import EMEstimator

em = EMEstimator(model, max_iter=500, tol=1e-6)
result = em.fit(y)
```

---

## Performance and optimization tips

!!! tip "Time-invariant systems"

    If all system matrices are constant (no `t` subscript), pass 2-D arrays.
    kalmanbox detects this and avoids re-indexing on every step, giving a
    significant speed-up for long series.

!!! tip "Batch vs. online"

    `kf.run(y)` is vectorized (NumPy operations over the full sample) and is
    faster than calling `predict_step` / `update_step` in a Python loop.
    Use the step methods only when you need to react to each observation.

!!! tip "Large state vectors"

    For $k \gg p$ (many state variables, few observations), the standard
    update inverts $F_t \in \mathbb{R}^{p \times p}$ rather than
    $P_{t|t-1} \in \mathbb{R}^{k \times k}$ — already optimal. If $p$ is
    also large, consider the [Information Filter](../filters/information.md)
    which works with the precision matrix $F_t^{-1}$.

!!! tip "Numerical instability"

    If $P_{t|t}$ becomes non-positive-definite, switch to the
    [Square-Root Filter](../filters/square-root.md), which propagates the
    Cholesky factor of $P_t$ instead of $P_t$ itself.

!!! tip "Non-Gaussian or nonlinear systems"

    The standard Kalman filter is optimal only under Gaussianity and linearity.
    For nonlinear observation or transition functions, use the
    [EKF](../filters/ekf.md) or [UKF](../filters/ukf.md).

---

## Diagnostics

After filtering, inspect the **standardized innovations** for model adequacy:

```python
from kalmanbox.diagnostics import innovation_diagnostics

diag = innovation_diagnostics(out)
print(diag.ljung_box(lags=10))     # serial correlation
print(diag.jarque_bera())          # normality
print(diag.heteroskedasticity())   # variance homogeneity
```

A well-specified model should have standardized innovations that are
approximately i.i.d. $\mathcal{N}(0, 1)$. See
[Diagnostics: residual analysis](../../diagnostics/residuals.md).

---

## Related

- [RTS Smoother](rts-smoother.md) — backward pass for full-sample estimates
- [Forecasting](forecasting.md) — multi-step-ahead predictions
- [Missing data](missing-data.md) — handling `NaN` in $y_t$
- [Diffuse initialisation](diffuse-initialization.md) — non-stationary states
- [Theory: Kalman filter derivation](../../theory/kalman-filter-derivation.md)
- [Theory: Numerical stability](../../theory/numerical-stability.md)
- [API: KalmanFilter](../../api/filters.md)
- [Diagnostics: residuals](../../diagnostics/residuals.md)
