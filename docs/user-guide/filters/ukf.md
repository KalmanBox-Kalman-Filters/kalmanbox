# Unscented Kalman Filter (UKF)

The [`UKF`][kalmanbox.filters.UKF] class implements the **Unscented Kalman Filter**, a
derivative-free nonlinear filter that avoids the linearization errors of the
[EKF](ekf.md) by applying the **Unscented Transform** — a deterministic sampling strategy
that propagates a minimal set of carefully chosen points through the nonlinear functions
and then reconstructs the mean and covariance of the output.

!!! note "When to use the UKF"
    Prefer the UKF over the EKF when the model is **moderately to strongly nonlinear**,
    Jacobians are **difficult to compute**, or you need **higher-order accuracy** without the
    full Monte Carlo cost of particle filters or the [Ensemble Kalman Filter](enkf.md).

---

## 1. The core idea: Unscented Transform

The fundamental problem in nonlinear filtering is: given $\alpha_t \sim \mathcal{N}(a, P)$,
what is the distribution of $z = g(\alpha_t)$ for a nonlinear function $g$?

The EKF answers this by expanding $g$ to first order around $a$. The UKF takes a different
approach: **sample a small, deterministic set of points** (sigma points) from $\mathcal{N}(a, P)$,
propagate each point through $g$ exactly, then fit a Gaussian to the propagated cloud.

!!! abstract "Why sigma points beat linearization"
    A first-order Taylor expansion is exact only for linear $g$. For nonlinear $g$, it ignores
    curvature and introduces bias proportional to $\operatorname{tr}[\nabla^2 g \cdot P]$.
    The Unscented Transform captures the mean to **third order** and the variance to **third
    order** (for Gaussian inputs) using only $2k+1$ function evaluations — no derivatives
    required.

---

## 2. Unscented Transform

### Sigma points

Given mean $a \in \mathbb{R}^k$ and covariance $P \in \mathbb{R}^{k \times k}$, construct
$2k+1$ sigma points $\{\mathcal{X}_i\}_{i=0}^{2k}$:

$$
\boxed{
\begin{aligned}
\mathcal{X}_0 &= a \\
\mathcal{X}_i &= a + \left(\sqrt{(k + \lambda)\,P}\right)_{\!i}, \quad i = 1,\ldots,k \\
\mathcal{X}_{k+i} &= a - \left(\sqrt{(k + \lambda)\,P}\right)_{\!i}, \quad i = 1,\ldots,k
\end{aligned}
}
$$

where $\left(\sqrt{(k+\lambda)\,P}\right)_{\!i}$ is the $i$-th **column** of the matrix
square root (lower Cholesky factor) of $(k+\lambda)\,P$, and:

$$
\lambda = \alpha^2(k + \kappa) - k
$$

### Weights

$$
\boxed{
\begin{aligned}
W_0^m &= \frac{\lambda}{k + \lambda} \\[4pt]
W_0^c &= \frac{\lambda}{k + \lambda} + (1 - \alpha^2 + \beta) \\[4pt]
W_i^m &= W_i^c = \frac{1}{2(k + \lambda)}, \quad i = 1, \ldots, 2k
\end{aligned}
}
$$

Superscripts $m$ and $c$ denote weights for **mean** and **covariance** reconstruction.
Note $W_0^m \neq W_0^c$ — the covariance weight for the central point absorbs the correction
$(1 - \alpha^2 + \beta)$ that encodes prior knowledge about the distribution's kurtosis.

### Reconstructed statistics

After propagating all sigma points through $g$:

$$
\mathcal{Z}_i = g(\mathcal{X}_i), \quad i = 0, \ldots, 2k
$$

the output mean and covariance are:

$$
\bar{z} = \sum_{i=0}^{2k} W_i^m\,\mathcal{Z}_i, \qquad
P_{zz} = \sum_{i=0}^{2k} W_i^c\,(\mathcal{Z}_i - \bar{z})(\mathcal{Z}_i - \bar{z})'
$$

---

## 3. UKF parameters

The three scaling parameters $(\alpha, \beta, \kappa)$ govern the shape and accuracy of the
sigma-point approximation:

| Parameter | Role | Typical range | Default |
|-----------|------|:-------------:|:-------:|
| $\alpha$ | Spread of sigma points around the mean | $[10^{-4},\, 1]$ | $10^{-3}$ |
| $\beta$ | Prior knowledge of distribution kurtosis | $\beta \geq 0$ | $2$ (Gaussian) |
| $\kappa$ | Secondary scaling | $\kappa \geq 0$ | $0$ |

**$\alpha$ (spread):** smaller $\alpha$ concentrates sigma points closer to the mean, reducing
the effect of terms beyond the local neighborhood. For strongly nonlinear $g$, larger $\alpha$
captures more of the global behavior at the cost of possible numerical issues.

**$\beta$ (kurtosis correction):** for a Gaussian input, $\beta = 2$ causes the covariance
reconstruction to exactly match the true covariance for quadratic functions. For heavier-tailed
distributions, increase $\beta$.

**$\kappa$ (secondary scaling):** $\kappa = 3 - k$ is sometimes recommended to keep $W_0^c$
positive, avoiding potential negative-definiteness of the reconstructed covariance. For large
$k$, the safer default is $\kappa = 0$.

!!! warning "Negative $W_0^m$ is normal"
    For $\lambda < 0$ (small $\alpha$ and large $k$), $W_0^m$ can be negative. This is
    mathematically valid — sigma-point weights are not probabilities. The reconstruction
    remains consistent as long as $P_{zz}$ is positive-definite.

---

## 4. UKF recursion

### 4.1 Initialization

$$
a_{0|0} = \mathbb{E}[\alpha_0], \qquad P_{0|0} = \operatorname{Var}[\alpha_0]
$$

### 4.2 Prediction step

**Generate sigma points** from the filtered distribution:

$$
\{\mathcal{X}_{t-1}^{(i)}\} \leftarrow \operatorname{SigmaPoints}\!\left(a_{t-1|t-1},\, P_{t-1|t-1}\right)
$$

**Propagate** each sigma point through $f$:

$$
\hat{\mathcal{X}}_{t}^{(i)} = f\!\left(\mathcal{X}_{t-1}^{(i)}\right), \quad i = 0, \ldots, 2k
$$

**Reconstruct** predicted mean and covariance:

$$
\boxed{
\begin{aligned}
a_{t|t-1} &= \sum_{i=0}^{2k} W_i^m\,\hat{\mathcal{X}}_{t}^{(i)} \\[6pt]
P_{t|t-1} &= \sum_{i=0}^{2k} W_i^c\,\bigl(\hat{\mathcal{X}}_{t}^{(i)} - a_{t|t-1}\bigr)
              \bigl(\hat{\mathcal{X}}_{t}^{(i)} - a_{t|t-1}\bigr)' + R\,Q\,R'
\end{aligned}
}
$$

### 4.3 Update step

**Generate new sigma points** from the predicted distribution:

$$
\{\mathcal{X}_{t}^{(i)}\} \leftarrow \operatorname{SigmaPoints}\!\left(a_{t|t-1},\, P_{t|t-1}\right)
$$

**Propagate** through $h$:

$$
\mathcal{Y}_{t}^{(i)} = h\!\left(\mathcal{X}_{t}^{(i)}\right)
$$

**Reconstruct** predicted observation, innovation covariance, and cross-covariance:

$$
\hat{y}_t = \sum_{i=0}^{2k} W_i^m\,\mathcal{Y}_{t}^{(i)}
$$

$$
S_t = \sum_{i=0}^{2k} W_i^c\,\bigl(\mathcal{Y}_{t}^{(i)} - \hat{y}_t\bigr)\bigl(\mathcal{Y}_{t}^{(i)} - \hat{y}_t\bigr)' + H
$$

$$
C_t = \sum_{i=0}^{2k} W_i^c\,\bigl(\mathcal{X}_{t}^{(i)} - a_{t|t-1}\bigr)\bigl(\mathcal{Y}_{t}^{(i)} - \hat{y}_t\bigr)'
$$

**Kalman gain and update:**

$$
\boxed{
\begin{aligned}
K_t &= C_t\,S_t^{-1} \\[4pt]
v_t &= y_t - \hat{y}_t \\[4pt]
a_{t|t} &= a_{t|t-1} + K_t\,v_t \\[4pt]
P_{t|t} &= P_{t|t-1} - K_t\,S_t\,K_t'
\end{aligned}
}
$$

The cross-covariance $C_t$ plays the role that $P_{t|t-1} H_t'$ plays in the linear
Kalman filter and EKF — it measures how strongly the predicted state correlates with
the predicted observation. Here it is computed from sigma points, not from a Jacobian.

---

## 5. UKF vs EKF: accuracy comparison

### Taylor expansion perspective

For a scalar nonlinear function $g : \mathbb{R} \to \mathbb{R}$ with input
$x \sim \mathcal{N}(a, P)$, let $g'$, $g''$ denote derivatives at $a$:

| Statistic | True value | EKF approximation | UKF approximation |
|-----------|-----------|------------------|------------------|
| $\mathbb{E}[g(x)]$ | $g(a) + \frac{1}{2}g''P + O(P^2)$ | $g(a)$ | $g(a) + \frac{1}{2}g''P + O(P^2)$ |
| $\operatorname{Var}[g(x)]$ | $(g')^2 P + \frac{1}{2}(g'')^2 P^2 + O(P^3)$ | $(g')^2 P$ | $(g')^2 P + O(P^2)$ |

The EKF ignores the $\frac{1}{2}g''P$ correction to the mean. For multivariate $g$, the UKF
captures mean and covariance to **third order** for Gaussian inputs — equivalent to a full
third-order Taylor expansion at only $O(k)$ function evaluations.

### Practical performance

| Scenario | EKF | UKF |
|----------|-----|-----|
| Mild nonlinearity, Jacobians available | Slightly faster | Marginally better accuracy |
| Strong nonlinearity (angles, ranges) | May diverge | Robust |
| Jacobians unavailable or expensive | Requires autodiff or finite diff | No Jacobians needed |
| Non-Gaussian but unimodal posterior | Biased | Less biased |
| Multimodal posterior | Fails | Still fails — use particle filter |

---

## 6. API reference

```python
from kalmanbox.filters import UKF

ukf = UKF(
    transition_fn,         # Callable[[np.ndarray], np.ndarray] — nonlinear f
    observation_fn,        # Callable[[np.ndarray], np.ndarray] — nonlinear h
    Q,                     # process noise cov, shape (k, k)
    H,                     # observation noise cov, shape (p, p)
    x0,                    # initial state mean, shape (k,)
    P0,                    # initial state cov, shape (k, k)
    R=None,                # selection matrix; defaults to I_k
    alpha=1e-3,            # sigma-point spread
    beta=2.0,              # kurtosis parameter (2 = Gaussian)
    kappa=0.0,             # secondary scaling
)
```

### Key methods

| Method | Description |
|--------|-------------|
| `ukf.filter(y)` | Forward UKF pass over `y` of shape `(T, p)` |
| `ukf.smooth(y)` | UKF forward pass then Unscented RTS smoother backward pass |
| `ukf.sigma_points(a, P)` | Return the $2k+1$ sigma points and weights for `(a, P)` |
| `ukf.log_likelihood(y)` | Compute total approximate log-likelihood |
| `ukf.predict(n_steps)` | Propagate forward `n_steps` using sigma-point prediction |

### FilterResult attributes

```python
result = ukf.filter(y)

result.filtered_states        # shape (T, k): a_{t|t}
result.filtered_covariances   # shape (T, k, k): P_{t|t}
result.predicted_states       # shape (T, k): a_{t|t-1}
result.predicted_covariances  # shape (T, k, k): P_{t|t-1}
result.innovations            # shape (T, p): v_t = y_t - hat_y_t
result.innovation_covariances # shape (T, p, p): S_t
result.log_likelihood         # scalar
```

---

## 7. Examples

### Example 1: Bearings-only tracking

A passive sonar tracks a submarine using **bearing measurements only** — no range information.
The state is $(p_x, p_y, v_x, v_y)$ and the observation is the angle
$\theta_t = \arctan(p_y / p_x)$ — a classic UKF benchmark with high nonlinearity.

```python
import numpy as np
from kalmanbox.filters import UKF

dt = 1.0

def f(x: np.ndarray) -> np.ndarray:
    """Constant-velocity dynamics."""
    return np.array([x[0] + dt*x[2], x[1] + dt*x[3], x[2], x[3]])

def h(x: np.ndarray) -> np.ndarray:
    """Bearing-only measurement."""
    return np.array([np.arctan2(x[1], x[0])])

sigma_proc = 0.05    # process noise (m/s^2)
sigma_bear = 0.017   # bearing noise (≈ 1 degree)

Q = sigma_proc**2 * np.diag([dt**2, dt**2, 1.0, 1.0])
H_obs = np.array([[sigma_bear**2]])

x0 = np.array([1000.0, 300.0, -10.0, 5.0])
P0 = np.diag([100.0, 100.0, 25.0, 25.0])

ukf = UKF(
    transition_fn=f, observation_fn=h,
    Q=Q, H=H_obs, x0=x0, P0=P0,
    alpha=1e-2, beta=2.0, kappa=1.0,
)

np.random.seed(0)
T = 60
true_states = np.zeros((T, 4))
observations = np.zeros((T, 1))
x = x0.copy()
for t in range(T):
    x = f(x) + sigma_proc * np.random.randn(4)
    true_states[t] = x
    observations[t] = h(x) + sigma_bear * np.random.randn(1)

result = ukf.filter(observations)
rmse_pos = np.sqrt(np.mean((result.filtered_states[:, :2] - true_states[:, :2])**2))
print(f"UKF position RMSE: {rmse_pos:.2f} m")
print(f"Log-likelihood: {result.log_likelihood:.2f}")
```

### Example 2: Nonlinear pendulum dynamics

A pendulum with angle $\theta$ and angular velocity $\dot{\theta}$ evolves according to
the nonlinear ODE $\ddot{\theta} = -(g/L)\sin\theta$. The EKF linearizes $\sin\theta \approx \theta$;
the UKF propagates sigma points through the exact nonlinear dynamics.

```python
import numpy as np
from kalmanbox.filters import UKF, EKF

g_grav, L = 9.81, 1.0
dt = 0.05

def f_pendulum(x: np.ndarray) -> np.ndarray:
    """Euler-discretized pendulum (nonlinear)."""
    theta, omega = x
    alpha = -(g_grav / L) * np.sin(theta)
    return np.array([theta + dt * omega, omega + dt * alpha])

def h_pendulum(x: np.ndarray) -> np.ndarray:
    """Observe angle directly."""
    return np.array([x[0]])

Q = np.diag([1e-5, 1e-4])
H_obs = np.array([[0.01]])
x0 = np.array([np.pi / 3, 0.0])   # start at 60 degrees
P0 = np.diag([0.1, 0.01])

ukf = UKF(
    transition_fn=f_pendulum, observation_fn=h_pendulum,
    Q=Q, H=H_obs, x0=x0, P0=P0,
    alpha=0.1, beta=2.0, kappa=0.0,
)

np.random.seed(1)
T = 200
states = np.zeros((T, 2))
obs = np.zeros((T, 1))
x = x0.copy()
for t in range(T):
    x = f_pendulum(x) + np.random.multivariate_normal(np.zeros(2), Q)
    states[t] = x
    obs[t] = x[0] + np.sqrt(0.01) * np.random.randn()

result_ukf = ukf.filter(obs)
angle_rmse = np.sqrt(np.mean((result_ukf.filtered_states[:, 0] - states[:, 0])**2))
print(f"UKF angle RMSE: {angle_rmse:.4f} rad")
```

### Example 3: Inspecting sigma points

```python
import numpy as np
from kalmanbox.filters import UKF

ukf = UKF(
    transition_fn=lambda x: x,
    observation_fn=lambda x: x[:1],
    Q=np.eye(3), H=np.eye(1),
    x0=np.zeros(3), P0=np.eye(3),
    alpha=1e-3, beta=2.0, kappa=0.0,
)

a = np.array([1.0, 2.0, 0.5])
P = np.diag([0.4, 0.9, 0.25])

sigma_pts, Wm, Wc = ukf.sigma_points(a, P)
print(f"Number of sigma points: {len(sigma_pts)}")   # 7 = 2*3 + 1
print(f"Sum of mean weights:    {Wm.sum():.10f}")    # must equal 1.0
print(f"W_0^m: {Wm[0]:.6f},  W_0^c: {Wc[0]:.6f}")
# Verify: weighted mean recovers a
reconstructed_mean = sigma_pts.T @ Wm
print(f"Reconstructed mean:     {reconstructed_mean}")   # should match a
```

---

## 8. Unscented RTS smoother

The UKF can be extended to a full smoother using the **Unscented RTS (URTS)** backward pass.
The smoothing gain is computed from the cross-covariances stored during the forward pass:

$$
D_t = C_{t+1|t}\,P_{t+1|t}^{-1}
$$

$$
a_{t|n} = a_{t|t} + D_t\,(a_{t+1|n} - a_{t+1|t})
$$

$$
P_{t|n} = P_{t|t} + D_t\,(P_{t+1|n} - P_{t+1|t})\,D_t'
$$

where $C_{t+1|t} = \sum_i W_i^c\,(\mathcal{X}_i - a_{t|t})(\hat{\mathcal{X}}_i - a_{t+1|t})'$
is the state-prediction cross-covariance stored during the prediction step.

```python
smooth_result = ukf.smooth(obs)

smooth_result.smoothed_states      # shape (T, k): a_{t|n}
smooth_result.smoothed_covariances # shape (T, k, k): P_{t|n}

angle_rmse_smooth = np.sqrt(
    np.mean((smooth_result.smoothed_states[:, 0] - states[:, 0])**2)
)
print(f"URTS angle RMSE: {angle_rmse_smooth:.4f} rad")  # better than filter
```

---

## See also

- [EKF](ekf.md) — linearization-based alternative; faster when Jacobians are available
- [Ensemble Kalman Filter](enkf.md) — Monte Carlo alternative for very high-dimensional or
  extremely nonlinear problems
- [Nonlinear Tracking Tutorial](../../tutorials/nonlinear-tracking.md) — EKF vs UKF benchmarks
- [API Reference: Filters](../../api/filters.md)
- [Numerical Stability](../../theory/numerical-stability.md)
