# Extended Kalman Filter (EKF)

The [`EKF`][kalmanbox.filters.EKF] class implements the **Extended Kalman Filter**, the
classical approach to nonlinear state estimation. It extends the standard Kalman filter to
systems with differentiable nonlinear transition and observation functions by **linearizing
each function around the current state estimate** using a first-order Taylor expansion.

!!! note "When to use the EKF"
    Reach for the EKF when your model has smooth, differentiable nonlinearities and analytical
    Jacobians are available (or automatic differentiation is acceptable). For strongly nonlinear
    problems or when Jacobians are intractable, prefer the [UKF](ukf.md) instead.

---

## 1. Nonlinear state-space model

The EKF applies to the **nonlinear Gaussian state-space model**:

$$
\begin{aligned}
\alpha_{t+1} &= f(\alpha_t,\, u_t) + R\,\eta_t, &
\eta_t &\sim \mathcal{N}(0,\, Q) \\
y_t &= h(\alpha_t) + \varepsilon_t, &
\varepsilon_t &\sim \mathcal{N}(0,\, H)
\end{aligned}
$$

where:

- $f : \mathbb{R}^k \to \mathbb{R}^k$ is the **nonlinear state transition function**
- $h : \mathbb{R}^k \to \mathbb{R}^p$ is the **nonlinear observation function**
- $u_t$ is an optional control input or exogenous variable
- $\eta_t \perp \varepsilon_t$ (process and observation noise are independent)

Unlike the linear SSM, $f$ and $h$ are arbitrary smooth functions — no linearity assumption
is imposed. The price for this generality is approximation: the EKF is no longer the *optimal*
estimator but rather a computationally tractable approximation.

---

## 2. Linearization via Jacobians

The central idea of the EKF is to **linearize** $f$ and $h$ around the current best estimate
using a **first-order Taylor expansion**:

$$
f(\alpha_t) \approx f(a_{t|t}) + F_t\,(\alpha_t - a_{t|t})
$$

$$
h(\alpha_t) \approx h(a_{t|t-1}) + H_t\,(\alpha_t - a_{t|t-1})
$$

where the **Jacobian matrices** are:

$$
\boxed{
F_t = \left.\frac{\partial f}{\partial \alpha}\right|_{\alpha = a_{t|t}}
\in \mathbb{R}^{k \times k}, \qquad
H_t = \left.\frac{\partial h}{\partial \alpha}\right|_{\alpha = a_{t|t-1}}
\in \mathbb{R}^{p \times k}
}
$$

$F_t$ is evaluated at the **filtered** estimate (after incorporating $y_t$), and $H_t$ is
evaluated at the **predicted** estimate (before incorporating $y_t$). This linearization
schedule minimizes propagation error at each step.

---

## 3. EKF recursion

### 3.1 Initialization

$$
a_{0|0} = \mathbb{E}[\alpha_0], \qquad P_{0|0} = \operatorname{Var}[\alpha_0]
$$

### 3.2 Prediction step

Propagate the state mean through the **full nonlinear** transition (not its linearization):

$$
\boxed{
\begin{aligned}
a_{t|t-1} &= f(a_{t-1|t-1}) \\
P_{t|t-1} &= F_{t-1}\,P_{t-1|t-1}\,F_{t-1}' + R\,Q\,R'
\end{aligned}
}
$$

The mean $a_{t|t-1}$ uses $f$ itself (higher accuracy), while the covariance propagation
uses the Jacobian $F_{t-1}$ (first-order approximation). This is the standard *first-order EKF*.

### 3.3 Update step

$$
\boxed{
\begin{aligned}
v_t &= y_t - h(a_{t|t-1}) \quad \text{(innovation)} \\
S_t &= H_t\,P_{t|t-1}\,H_t' + H \quad \text{(innovation covariance)} \\
K_t &= P_{t|t-1}\,H_t'\,S_t^{-1} \quad \text{(Kalman gain)} \\
a_{t|t} &= a_{t|t-1} + K_t\,v_t \\
P_{t|t} &= (I - K_t\,H_t)\,P_{t|t-1}\,(I - K_t\,H_t)' + K_t\,H\,K_t'
\end{aligned}
}
$$

The last covariance update uses the **Joseph form** $(I - K_t H_t)\,P\,(I - K_t H_t)' + K_t H K_t'$
to preserve symmetry and positive-definiteness under finite-precision arithmetic. This is the
same convention used throughout kalmanbox (see
[Numerical Stability](../../theory/numerical-stability.md)).

### 3.4 Log-likelihood

The approximate log-likelihood contribution from observation $t$ is:

$$
\ell_t = -\frac{p}{2}\ln(2\pi) - \frac{1}{2}\ln|S_t| - \frac{1}{2}\,v_t'\,S_t^{-1}\,v_t
$$

This is exact under the Gaussian approximation induced by linearization.

---

## 4. Jacobian computation

### Analytical Jacobians (recommended)

For maximum accuracy and speed, provide analytical Jacobians:

```python
import numpy as np

def f(x: np.ndarray) -> np.ndarray:
    """Constant-velocity transition in 2D."""
    # state: [px, py, vx, vy]
    return np.array([x[0] + x[2], x[1] + x[3], x[2], x[3]])

def F_jac(x: np.ndarray) -> np.ndarray:
    """Jacobian of f — constant for a linear transition."""
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

def h(x: np.ndarray) -> np.ndarray:
    """Range-bearing observation."""
    px, py = x[0], x[1]
    return np.array([np.sqrt(px**2 + py**2), np.arctan2(py, px)])

def H_jac(x: np.ndarray) -> np.ndarray:
    """Jacobian of h."""
    px, py = x[0], x[1]
    r2 = px**2 + py**2
    r = np.sqrt(r2)
    return np.array([
        [ px / r,   py / r,  0, 0],
        [-py / r2,  px / r2, 0, 0],
    ])
```

### Numerical Jacobians (automatic fallback)

When `transition_jac` and `observation_jac` are `None`, kalmanbox computes Jacobians via
**central finite differences** with step size $\delta$:

$$
\frac{\partial f_i}{\partial \alpha_j}\bigg|_{\alpha=a}
\approx \frac{f_i(a + \delta e_j) - f_i(a - \delta e_j)}{2\delta}
$$

This requires $2k$ evaluations of $f$ and $2k$ evaluations of $h$ per time step.

```python
from kalmanbox.filters import EKF

ekf = EKF(
    transition_fn=f,
    observation_fn=h,
    transition_jac=None,   # triggers automatic finite-difference Jacobian
    observation_jac=None,
    Q=Q, H=H, x0=x0, P0=P0,
    fd_step=1e-5,          # finite-difference step size
)
```

### Automatic differentiation

For differentiable functions defined in JAX, pass the autodiff Jacobian directly:

```python
import jax
import jax.numpy as jnp

def f_jax(x):
    return jnp.array([x[0] + x[2], x[1] + x[3], x[2], x[3]])

F_jac_auto = jax.jacobian(f_jax)  # returns a callable

ekf = EKF(transition_fn=f_jax, transition_jac=F_jac_auto, ...)
```

---

## 5. Limitations of the EKF

!!! warning "EKF limitations"

    1. **Strong nonlinearity** — when the curvature of $f$ or $h$ is large relative to the
       spread of the prior, the first-order approximation introduces systematic bias in both
       mean and covariance
    2. **Non-Gaussian noise** — the EKF assumes Gaussian posteriors; multimodal or heavy-tailed
       distributions are not captured
    3. **Filter divergence** — if the initial estimate $a_{0|0}$ is poor, the linearization
       point may be far from the true state, causing covariance collapse and divergence
    4. **Discontinuous or non-smooth $f$, $h$** — Jacobians do not exist; use the UKF or EnKF

### Second-order EKF

The second-order EKF corrects the mean bias by including Hessian terms:

$$
a_{t|t-1} \approx f(a_{t-1|t-1}) + \frac{1}{2}\sum_{j=1}^{k}\operatorname{tr}\!\left[\nabla^2 f_j(a_{t-1|t-1})\, P_{t-1|t-1}\right]\,e_j
$$

This is available in kalmanbox as `EKF(order=2)` but requires Hessian functions and is
rarely used in practice — the [UKF](ukf.md) achieves comparable accuracy more conveniently.

---

## 6. API reference

```python
from kalmanbox.filters import EKF

ekf = EKF(
    transition_fn,           # Callable[[np.ndarray], np.ndarray] — nonlinear f
    observation_fn,          # Callable[[np.ndarray], np.ndarray] — nonlinear h
    Q,                       # process noise cov, shape (k, k)
    H,                       # observation noise cov, shape (p, p)
    x0,                      # initial state mean, shape (k,)
    P0,                      # initial state cov, shape (k, k)
    transition_jac=None,     # Callable or None → finite differences
    observation_jac=None,    # Callable or None → finite differences
    R=None,                  # selection matrix; defaults to I_k
    fd_step=1e-5,            # finite-difference step if jac is None
    order=1,                 # 1 = standard EKF, 2 = second-order EKF
)
```

### Key methods

| Method | Description |
|--------|-------------|
| `ekf.filter(y)` | Run forward EKF pass over observations `y` of shape `(T, p)` |
| `ekf.smooth(y)` | EKF forward pass then EKF-based RTS smoother backward pass |
| `ekf.log_likelihood(y)` | Compute total approximate log-likelihood |
| `ekf.predict(n_steps)` | Propagate forward `n_steps` using the transition function |

### FilterResult attributes

```python
result = ekf.filter(y)

result.filtered_states        # shape (T, k): a_{t|t}
result.filtered_covariances   # shape (T, k, k): P_{t|t}
result.predicted_states       # shape (T, k): a_{t|t-1}
result.predicted_covariances  # shape (T, k, k): P_{t|t-1}
result.innovations            # shape (T, p): v_t = y_t - h(a_{t|t-1})
result.innovation_covariances # shape (T, p, p): S_t
result.log_likelihood         # scalar: sum of ell_t
```

---

## 7. Examples

### Example 1: Bearing-range tracking

A radar tracks a target moving at constant velocity. The state is $(p_x, p_y, v_x, v_y)$;
observations are range and bearing — a nonlinear function of position.

```python
import numpy as np
from kalmanbox.filters import EKF

dt = 1.0  # seconds

def f(x: np.ndarray) -> np.ndarray:
    """Constant-velocity transition."""
    return np.array([x[0] + dt*x[2], x[1] + dt*x[3], x[2], x[3]])

def F_jac(x: np.ndarray) -> np.ndarray:
    return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)

def h(x: np.ndarray) -> np.ndarray:
    """Range and bearing observations."""
    return np.array([np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])])

def H_jac(x: np.ndarray) -> np.ndarray:
    px, py = x[0], x[1]
    r2 = px**2 + py**2
    r = np.sqrt(r2)
    return np.array([[px/r, py/r, 0, 0], [-py/r2, px/r2, 0, 0]])

Q = 0.1**2 * np.eye(4)
H_obs = np.diag([5.0**2, 0.02**2])   # range noise 5m, bearing noise ~1.1 deg
x0 = np.array([100.0, 50.0, -2.0, 1.0])
P0 = np.diag([10.0, 10.0, 1.0, 1.0])

ekf = EKF(
    transition_fn=f, observation_fn=h,
    transition_jac=F_jac, observation_jac=H_jac,
    Q=Q, H=H_obs, x0=x0, P0=P0,
)

# Simulate
np.random.seed(42)
T = 100
true_states = np.zeros((T, 4))
observations = np.zeros((T, 2))
x = x0.copy()
for t in range(T):
    x = f(x) + np.linalg.cholesky(Q) @ np.random.randn(4)
    true_states[t] = x
    observations[t] = h(x) + np.linalg.cholesky(H_obs) @ np.random.randn(2)

result = ekf.filter(observations)
print(f"Log-likelihood: {result.log_likelihood:.2f}")
rmse_pos = np.sqrt(np.mean((result.filtered_states[:, :2] - true_states[:, :2])**2))
print(f"Position RMSE: {rmse_pos:.2f} m")
```

### Example 2: Stochastic volatility

The stochastic volatility (SV) model has a nonlinear observation equation:

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta\,\eta_t, \qquad
y_t = \exp(h_t / 2)\,\varepsilon_t
$$

The EKF can be applied directly after a log-squaring transformation $y_t^* = \ln(y_t^2)$,
which linearizes the observation equation at the cost of adding $\chi^2$ noise:

```python
import numpy as np
from kalmanbox.filters import EKF

mu, phi, sigma_eta = -1.0, 0.95, 0.1

def f_sv(x: np.ndarray) -> np.ndarray:
    """AR(1) log-variance."""
    return np.array([mu + phi * (x[0] - mu)])

def F_sv(x: np.ndarray) -> np.ndarray:
    return np.array([[phi]])

def h_sv(x: np.ndarray) -> np.ndarray:
    """Approximate observation: E[log y^2] = h + log(2) + psi(1/2)."""
    return np.array([x[0] - 1.2704])   # psi(1/2) ≈ -1.9635, log(2) ≈ 0.6931

def H_sv(x: np.ndarray) -> np.ndarray:
    return np.array([[1.0]])

Q_sv = np.array([[sigma_eta**2]])
H_obs_sv = np.array([[np.pi**2 / 2]])   # variance of log chi^2(1)
x0_sv = np.array([mu])
P0_sv = np.array([[sigma_eta**2 / (1 - phi**2)]])   # stationary variance

ekf_sv = EKF(
    transition_fn=f_sv, observation_fn=h_sv,
    transition_jac=F_sv, observation_jac=H_sv,
    Q=Q_sv, H=H_obs_sv, x0=x0_sv, P0=P0_sv,
)

# Simulate returns and filter
np.random.seed(0)
T = 500
log_var = np.zeros(T)
log_var[0] = mu
for t in range(1, T):
    log_var[t] = mu + phi * (log_var[t-1] - mu) + sigma_eta * np.random.randn()

returns = np.exp(log_var / 2) * np.random.randn(T)
y_star = np.log(returns**2 + 1e-10).reshape(-1, 1)

result_sv = ekf_sv.filter(y_star)
print(f"SV model log-likelihood: {result_sv.log_likelihood:.2f}")
```

---

## 8. Practical tips

### Tuning $P_0$

For the linearized system with transition matrix $F$, the stationary covariance satisfies
the discrete Lyapunov equation $P_\infty = F P_\infty F' + R Q R'$:

```python
from kalmanbox.utils import solve_discrete_lyapunov
P_stationary = solve_discrete_lyapunov(F_evaluated, Q)
```

### Consistency check

Normalized innovations $\tilde{v}_t = S_t^{-1/2} v_t$ should be approximately i.i.d.
$\mathcal{N}(0, I)$ for a correctly specified and tuned filter:

```python
from kalmanbox.diagnostics import normalized_innovation_squared
nis = normalized_innovation_squared(result)
# NIS should follow chi-squared(p); empirical mean should be ≈ p
```

### Preventing divergence

1. **Increase $P_0$** — ensure wide enough initial uncertainty
2. **Inflate $Q$** — add artificial process noise to prevent overconfidence
3. **Switch to UKF** — better for strong nonlinearities
4. **Covariance inflation** — multiply $P_{t|t}$ by a scalar factor $\rho > 1$ after each update

---

## See also

- [UKF](ukf.md) — derivative-free alternative; better for strong nonlinearity
- [Square-Root Filter](square-root.md) — numerical stability for linear models
- [Ensemble Kalman Filter](enkf.md) — Monte Carlo alternative for very high dimensions
- [Nonlinear Tracking Tutorial](../../tutorials/nonlinear-tracking.md) — end-to-end EKF/UKF comparison
- [Numerical Stability](../../theory/numerical-stability.md) — Joseph form and covariance conditioning
- [API Reference: Filters](../../api/filters.md)
