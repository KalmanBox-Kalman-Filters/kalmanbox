# Information Filter

The [`InformationFilter`][kalmanbox.filters.InformationFilter] reformulates the Kalman recursion
in terms of the **inverse** of the state covariance matrix and the corresponding **information
vector**. Where the standard Kalman filter propagates the pair $(a_t, P_t)$, the Information
Filter propagates the dual pair $(\xi_t, \Lambda_t)$:

$$
\boxed{\Lambda_t = P_t^{-1}, \qquad \xi_t = P_t^{-1}\,a_t = \Lambda_t\,a_t}
$$

$\Lambda_t \in \mathbb{R}^{k \times k}$ is the **information matrix** (also called the precision
matrix) and $\xi_t \in \mathbb{R}^k$ is the **information vector**. The state mean is recovered
as $a_t = \Lambda_t^{-1}\xi_t$ only when explicitly needed.

!!! note "When to use the Information Filter"
    The Information Filter is preferred when:

    - The problem has a **diffuse or improper prior** — total ignorance is represented *exactly*
      by $\Lambda_0 = 0$, with no need for an artificial large constant $P_0 = \kappa I$
    - **Multiple sensors** report simultaneously — each sensor's information contribution adds
      directly, with no matrix inversions required
    - The **observation dimension $p$ is large** relative to state dimension $k$ — the update
      step costs $O(pk^2)$ instead of $O(p^3 + k^2 p)$

    For standard problems with a well-defined prior and a single observation stream, the
    [classical Kalman filter](../kalman/kalman-filter.md) is simpler. For nonlinear systems,
    consider the [EKF](ekf.md) or [UKF](ukf.md).

---

## 1. The natural parameterization of a Gaussian

Consider the standard linear Gaussian SSM:

$$
\begin{aligned}
\alpha_{t+1} &= T\,\alpha_t + R\,\eta_t, & \eta_t &\sim \mathcal{N}(0,\, Q) \\
y_t &= Z_t\,\alpha_t + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0,\, H_t)
\end{aligned}
$$

The Kalman filter works with the **moment parameterization** $(a_t, P_t)$ of the posterior
$p(\alpha_t \mid \mathcal{F}_t)$. The Information Filter instead uses the **natural (canonical)
parameterization** of the Gaussian exponential family:

$$
p(\alpha_t \mid \mathcal{F}_t) \propto \exp\!\left(
    -\tfrac{1}{2}\,\alpha_t'\,\Lambda_t\,\alpha_t + \xi_t'\,\alpha_t
\right)
$$

The two parameterizations are equivalent: given $(\Lambda_t, \xi_t)$, recover moments via
$a_t = \Lambda_t^{-1}\xi_t$ and $P_t = \Lambda_t^{-1}$. The key advantage of the natural
form is that the **Fisher information adds** when conditionally independent observations are
incorporated:

$$
p(\alpha \mid y_1, y_2) \propto p(\alpha)\, p(y_1 \mid \alpha)\, p(y_2 \mid \alpha)
\quad\Rightarrow\quad
\Lambda \text{ gets the sum of all information contributions}
$$

---

## 2. Information filter recursion

### 2.1 Initialization

The Information Filter handles two types of initialization naturally:

=== "Informative prior"

    When the initial state has mean $a_0$ and covariance $P_0$:

    $$
    \Lambda_{0|0} = P_0^{-1}, \qquad \xi_{0|0} = P_0^{-1}\,a_0
    $$

=== "Diffuse prior (total ignorance)"

    When no prior information is available — equivalently $P_0 \to \infty\,I_k$:

    $$
    \boxed{\Lambda_{0|0} = 0_k, \qquad \xi_{0|0} = 0_k}
    $$

    This is the decisive advantage over the standard Kalman filter: the diffuse prior is
    represented *exactly* by $\Lambda_0 = 0$. See [Section 4](#4-connection-to-diffuse-initialization)
    and [Diffuse Initialization](../kalman/diffuse.md) for comparison with the classical approach.

### 2.2 Update step (observation incorporation)

Given the **predicted** pair $(\xi_{t|t-1}, \Lambda_{t|t-1})$ and observation $y_t$, the
**update** is additive and inversion-free:

$$
\boxed{
\begin{aligned}
\Lambda_{t|t} &= \Lambda_{t|t-1} + \underbrace{Z_t'\,H_t^{-1}\,Z_t}_{\displaystyle\Omega_t} \\[6pt]
\xi_{t|t} &= \xi_{t|t-1} + \underbrace{Z_t'\,H_t^{-1}\,y_t}_{\displaystyle\zeta_t}
\end{aligned}
}
$$

The quantities $\Omega_t = Z_t' H_t^{-1} Z_t \in \mathbb{R}^{k \times k}$ and
$\zeta_t = Z_t' H_t^{-1} y_t \in \mathbb{R}^k$ are the **observation information matrix** and
**observation information vector** respectively. They depend only on the observation model
$(Z_t, H_t)$ and the data $y_t$ — not on the current state estimate.

!!! tip "Sensor fusion via parallel updates"
    If $M$ sensors observe the state simultaneously at time $t$, with observation matrices
    $Z_t^{(i)}$ and noise covariances $H_t^{(i)}$, the joint update is simply:

    $$
    \Lambda_{t|t} = \Lambda_{t|t-1} + \sum_{i=1}^{M} (Z_t^{(i)})'(H_t^{(i)})^{-1} Z_t^{(i)},
    \qquad
    \xi_{t|t} = \xi_{t|t-1} + \sum_{i=1}^{M} (Z_t^{(i)})'(H_t^{(i)})^{-1} y_t^{(i)}
    $$

    Each sensor contributes independently to the information sum. The order of incorporation
    does not matter and no intermediate inversions are required.

### 2.3 Prediction step (time propagation)

Given the **filtered** pair $(\xi_{t|t}, \Lambda_{t|t})$, the **prediction** step propagates
the information forward one time step. Let $Q_* = R\,Q\,R'$ denote the effective state noise
covariance. The predicted information matrix satisfies:

$$
P_{t+1|t} = T\,\Lambda_{t|t}^{-1}\,T' + Q_*
\quad\Rightarrow\quad
\boxed{\Lambda_{t+1|t} = \bigl(T\,\Lambda_{t|t}^{-1}\,T' + Q_*\bigr)^{-1}}
$$

and the predicted information vector is:

$$
\boxed{\xi_{t+1|t} = \Lambda_{t+1|t}\,T\,\Lambda_{t|t}^{-1}\,\xi_{t|t}}
$$

Note that the prediction step requires **one inversion of $\Lambda_{t|t}$** (to recover $P_{t|t}$)
and one inversion to compute $\Lambda_{t+1|t}$. This is the main computational cost of the
Information Filter and the price paid for the inversion-free update step.

**Woodbury form (when $Q_*$ is invertible):**

Applying the matrix inversion lemma to $\Lambda_{t+1|t} = (Q_* + T \Lambda_{t|t}^{-1} T')^{-1}$:

$$
\Lambda_{t+1|t}
= Q_*^{-1}
  - Q_*^{-1}\,T\,\bigl(\Lambda_{t|t} + T'\,Q_*^{-1}\,T\bigr)^{-1}\,T'\,Q_*^{-1}
$$

This avoids computing $P_{t|t} = \Lambda_{t|t}^{-1}$ explicitly when $Q_*$ is diagonal or
sparse — an advantage when the state dimension $k$ is large but $Q_*$ is structured.

### 2.4 State recovery (lazy)

The posterior mean and covariance are recovered **on demand**:

$$
a_t = \Lambda_t^{-1}\,\xi_t, \qquad P_t = \Lambda_t^{-1}
$$

In kalmanbox, recovery is lazy — these inversions are only performed when `.filtered_states`
or `.filtered_covariances` are accessed.

### 2.5 Log-likelihood

The log-likelihood is computed via the standard innovation decomposition. Defining the predicted
mean $a_{t|t-1} = \Lambda_{t|t-1}^{-1}\,\xi_{t|t-1}$ (one inversion per time step):

$$
\ell_t = -\frac{p}{2}\ln(2\pi) - \frac{1}{2}\ln\lvert S_t \rvert - \frac{1}{2}\,v_t'\,S_t^{-1}\,v_t
$$

where $v_t = y_t - Z_t\,a_{t|t-1}$ and $S_t = Z_t\,\Lambda_{t|t-1}^{-1}\,Z_t' + H_t$.

During the diffuse initialization phase (when $\Lambda_{t|t-1}$ is rank-deficient), the
log-likelihood contribution is computed using the generalized innovation rather than the
standard formula. The total log-likelihood excludes the diffuse contributions to match the
convention of the exact diffuse Kalman filter (Durbin & Koopman 2012, Chapter 5).

---

## 3. Duality with the standard Kalman filter

The Information Filter and the Kalman Filter are **mathematically equivalent** — they differ
only in parameterization. The Kalman gain in the standard filter:

$$
K_t = P_{t|t-1}\,Z_t'\,S_t^{-1} = \Lambda_{t|t-1}^{-1}\,Z_t'\,(Z_t\,\Lambda_{t|t-1}^{-1}\,Z_t' + H_t)^{-1}
$$

The Joseph form update $P_{t|t} = (I - K_t Z_t)P_{t|t-1}(I-K_tZ_t)' + K_t H_t K_t'$ is
algebraically equivalent to $\Lambda_{t|t} = \Lambda_{t|t-1} + Z_t' H_t^{-1} Z_t$ via the
Woodbury identity.

The choice between the two parameterizations comes down to **which step is the bottleneck**:

| Step | Standard Kalman Filter | Information Filter |
|------|----------------------|-------------------|
| Update (incorporate $y_t$) | $O(p^3 + k^2 p)$ — invert $S_t \in \mathbb{R}^{p \times p}$ | $O(pk^2)$ — additive, **no inversion** |
| Predict (propagate time) | $O(k^3)$ — matrix multiply | $O(k^3)$ — requires inversion of $\Lambda_{t|t}$ |

**Rule of thumb:** When $p \gg k$ (many simultaneous observations, low-dimensional state),
the Information Filter wins. When $p \ll k$ (few observations, high-dimensional state),
the standard Kalman filter wins.

---

## 4. Connection to diffuse initialization

In the standard Kalman filter, handling a completely uninformative prior requires either:

1. **Approximate:** set $P_0 = \kappa I$ for large $\kappa$ (distorts the likelihood for early
   observations and introduces an arbitrary tuning constant)
2. **Exact diffuse filter:** a two-phase recursion that handles the diffuse and stationary
   components separately (Koopman 1997; Durbin & Koopman 2012, Chapter 5)

The Information Filter sidesteps both approaches. Setting $\Lambda_0 = 0$ and $\xi_0 = 0$
represents **total ignorance** exactly:

$$
\Lambda_{0|0} = 0 \implies P_{0|0} \to \infty\,I_k \text{ (improper prior, no information)}
$$

The filter starts with zero information and accumulates it as observations arrive. After the
first $d \le k$ observations (where $d$ is determined by the observability of the model),
$\Lambda_{t|t}$ becomes full-rank and the state is fully identified:

```python
# Track rank of Lambda_{t|t} over time
result = inf_filt.filter(y, track_rank=True)
print(result.rank[:10])   # [0, 1, 1, 1, 1, 1, 1, 1, 1, 1] for k=1 (rank 1 after t=0)
```

!!! example "Diffuse initialization comparison"

    === "Standard KF — approximate diffuse"

        ```python
        P0 = 1e6 * np.eye(k)   # Large but finite — distorts early log-likelihood
        a0 = np.zeros(k)
        kf = KalmanFilter(T, Z, R, Q, H, a0=a0, P0=P0)
        result = kf.filter(y)
        # Log-likelihood affected by the arbitrary choice of kappa = 1e6
        ```

    === "Information Filter — exact diffuse"

        ```python
        from kalmanbox.filters import InformationFilter

        inf_filt = InformationFilter(T, Z, R, Q, H)
        # Lambda_0 = 0, xi_0 = 0 set automatically — truly diffuse
        result = inf_filt.filter(y)
        # Log-likelihood is exact (excludes improper diffuse contribution)
        ```

---

## 5. API reference

```python
from kalmanbox.filters import InformationFilter

inf_filt = InformationFilter(
    T,                   # Transition matrix, shape (k, k)
    Z,                   # Observation matrix, shape (p, k)
    R,                   # Noise selection matrix, shape (k, r)
    Q,                   # State noise covariance, shape (r, r)
    H,                   # Observation noise covariance, shape (p, p)
    a0=None,             # Initial state mean; None → exact diffuse (Lambda_0 = 0)
    P0=None,             # Initial covariance; None → exact diffuse (Lambda_0 = 0)
    time_varying=False,  # If True, T/Z/R/Q/H may be lists or arrays indexed by t
)
```

### Key methods

| Method | Returns | Description |
|--------|---------|-------------|
| `filter(y)` | `InformationResult` | Forward Information Filter pass |
| `smooth(y)` | `SmootherResult` | Forward IF + backward RTS smoother |
| `log_likelihood(y)` | `float` | Total log-likelihood via innovation decomposition |
| `filter_multi(ys, Zs, Hs)` | `InformationResult` | Fuse multiple simultaneous sensor streams |

### InformationResult attributes

```python
result = inf_filt.filter(y)

result.Lambdas                # shape (T, k, k): Lambda_{t|t} — filtered information matrices
result.xis                    # shape (T, k):    xi_{t|t}     — filtered information vectors
result.filtered_states        # shape (T, k):    a_{t|t} = Lambda^{-1} xi  (lazy)
result.filtered_covariances   # shape (T, k, k): P_{t|t} = Lambda^{-1}     (lazy)
result.innovations            # shape (T, p):    v_t = y_t - Z a_{t|t-1}
result.innovation_covariances # shape (T, p, p): S_t
result.log_likelihood         # scalar
result.rank                   # shape (T,): rank of Lambda_{t|t} per step (if track_rank=True)
```

---

## 6. Examples

### Example 1: Diffuse initialization for the Local Level model

The [Local Level model](../structural/local-level.md) has no informative prior on the initial
state. The Information Filter handles this natively with exact diffuse initialization:

```python
import numpy as np
from kalmanbox.filters import InformationFilter
from kalmanbox.datasets import load_nile

y = load_nile().values.reshape(-1, 1)   # Nile annual flow, 100 observations

# Local level: alpha_{t+1} = alpha_t + eta_t,  y_t = alpha_t + eps_t
T_mat = np.array([[1.0]])
Z_mat = np.array([[1.0]])
R_mat = np.array([[1.0]])
Q_mat = np.array([[1469.1]])    # signal noise variance (MLE estimate)
H_mat = np.array([[15099.8]])   # observation noise variance (MLE estimate)

# Exact diffuse initialization: a0=None, P0=None → Lambda_0 = 0, xi_0 = 0
inf_filt = InformationFilter(T=T_mat, Z=Z_mat, R=R_mat, Q=Q_mat, H=H_mat)
result = inf_filt.filter(y)

print(f"Log-likelihood: {result.log_likelihood:.4f}")

# Examine how information accumulates over time
print(f"Lambda after t=0: {result.Lambdas[0, 0, 0]:.6e}")   # nonzero after first obs
print(f"Lambda after t=9: {result.Lambdas[9, 0, 0]:.6e}")   # stabilizes

# Recover filtered states (lazy inversion of Lambda)
states = result.filtered_states   # shape (100, 1)
stds = np.sqrt(result.filtered_covariances[:, 0, 0])

print(f"\nFinal filtered level:    {states[-1, 0]:.1f}")
print(f"Final posterior std:     {stds[-1]:.2f}")
```

### Example 2: Multi-sensor fusion

Three sensors observe the same position, with different noise levels. The Information Filter
fuses their contributions in parallel via additive updates:

```python
import numpy as np
from kalmanbox.filters import InformationFilter

np.random.seed(42)
T_steps = 200
k = 2   # state: [position, velocity]

# Constant-velocity model
T_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
R_mat = np.array([[0.5], [1.0]])
Q_mat = np.array([[0.01]])

# Three sensors with different accuracies and observation matrices
sensors = [
    {"Z": np.array([[1.0, 0.0]]), "H": np.array([[2.0]])},    # high-accuracy GPS
    {"Z": np.array([[1.0, 0.0]]), "H": np.array([[10.0]])},   # medium-accuracy altimeter
    {"Z": np.array([[1.0, 0.0]]), "H": np.array([[50.0]])},   # low-accuracy barometer
]

# Simulate true trajectory
alpha = np.zeros((T_steps + 1, k))
alpha[0] = [0.0, 0.5]
for t in range(T_steps):
    alpha[t+1] = T_mat @ alpha[t] + (R_mat * np.random.randn()).ravel()

# Simulate observations from each sensor
observations = []
for s in sensors:
    y_s = (s["Z"] @ alpha[1:].T).T
    y_s += np.sqrt(s["H"][0, 0]) * np.random.randn(T_steps, 1)
    observations.append(y_s)

# --- Multi-sensor Information Filter ---
inf_filt = InformationFilter(
    T=T_mat, Z=sensors[0]["Z"], R=R_mat, Q=Q_mat, H=sensors[0]["H"]
)
result_fused = inf_filt.filter_multi(
    ys=observations,
    Zs=[s["Z"] for s in sensors],
    Hs=[s["H"] for s in sensors],
)

# --- Single-sensor baseline ---
result_single = inf_filt.filter(observations[0])

true_pos = alpha[1:, 0]
fused_pos = result_fused.filtered_states[:, 0]
single_pos = result_single.filtered_states[:, 0]

rmse_fused  = np.sqrt(np.mean((fused_pos  - true_pos)**2))
rmse_single = np.sqrt(np.mean((single_pos - true_pos)**2))

print(f"Multi-sensor RMSE (position): {rmse_fused:.4f} m")
print(f"Single-sensor RMSE (position): {rmse_single:.4f} m")
print(f"Improvement factor: {rmse_single / rmse_fused:.2f}x")
```

### Example 3: Large-$p$ observation model

When $p \gg k$, the Information Filter's $O(pk^2)$ update outperforms the standard Kalman
filter's $O(p^3)$ update. Here $p = 100$ and $k = 10$:

```python
import numpy as np
import time
from kalmanbox import KalmanFilter
from kalmanbox.filters import InformationFilter

np.random.seed(0)
T_steps = 500
k = 10    # state dimension
p = 100   # observation dimension — much larger than k

# Random SSM matrices with p >> k
T_mat = 0.95 * np.eye(k)
Z_mat = np.random.randn(p, k) / np.sqrt(k)
Q_mat = 0.01 * np.eye(k)
H_mat = np.eye(p)

# Simulate data
alpha = np.zeros((T_steps + 1, k))
y = np.zeros((T_steps, p))
for t in range(T_steps):
    alpha[t+1] = T_mat @ alpha[t] + 0.1 * np.random.randn(k)
    y[t] = Z_mat @ alpha[t+1] + np.random.randn(p)

a0 = np.zeros(k)
P0 = np.eye(k)

# Standard Kalman Filter (must invert p×p innovation covariance S_t)
kf = KalmanFilter(T=T_mat, Z=Z_mat, R=np.eye(k), Q=Q_mat, H=H_mat, a0=a0, P0=P0)
t0 = time.perf_counter()
r_kf = kf.filter(y)
t1 = time.perf_counter()

# Information Filter (additive update; inverts k×k matrices only)
inf_filt = InformationFilter(T=T_mat, Z=Z_mat, R=np.eye(k), Q=Q_mat, H=H_mat)
r_if = inf_filt.filter(y)
t2 = time.perf_counter()

print(f"Standard KF  — log-lik: {r_kf.log_likelihood:.2f}  time: {t1-t0:.3f}s")
print(f"Information  — log-lik: {r_if.log_likelihood:.2f}  time: {t2-t1:.3f}s")
print(f"Speedup (p={p}, k={k}): {(t1-t0)/(t2-t1):.1f}x")
```

---

## 7. Practical considerations

### Handling a rank-deficient information matrix

Near the start of a diffuse filter, $\Lambda_{t|t}$ may be rank-deficient (the state is not yet
fully observable). kalmanbox handles this via a **pseudoinverse** fallback:

```python
result = inf_filt.filter(y, handle_rank_deficiency=True)
result.rank   # shape (T,): rank of Lambda_{t|t} at each time step
```

The filter transitions from the diffuse phase to the stationary phase automatically when
$\operatorname{rank}(\Lambda_{t|t}) = k$.

### Numerical stability of the additive update

The update $\Lambda_{t|t} = \Lambda_{t|t-1} + \Omega_t$ can accumulate asymmetry from finite
precision arithmetic. kalmanbox symmetrizes after each update:

```python
Lambda = 0.5 * (Lambda + Lambda.T)
```

For very long time series ($T > 10^4$) or near-singular $\Lambda$, consider the Square-Root
Information Filter (SRIF), which propagates the Cholesky factor of $\Lambda_t$.

### Time-varying system matrices

When $Z_t$, $H_t$, or $T$ vary over time, pass them as lists or 3D arrays:

```python
inf_filt = InformationFilter(
    T=T_list,        # list of k×k arrays, or shape (T, k, k)
    Z=Z_list,        # list of p_t×k arrays (p_t may vary)
    R=R_list,
    Q=Q_list,
    H=H_list,
    time_varying=True,
)
```

---

## 8. Information Filter vs. exact diffuse Kalman filter

The exact diffuse Kalman filter (Durbin & Koopman 2012, Chapter 5) handles the diffuse phase by
splitting the covariance into a finite part $P_*$ and an infinite diffuse component $P_\infty$,
processing them separately until $P_\infty = 0$, then switching to the standard recursion. The
Information Filter achieves the same result with a **unified recursion**:

| Approach | Pros | Cons |
|----------|------|------|
| Exact diffuse KF | Fast once stationary; standard in statistical software | Two-phase; complex state machine; hard to extend |
| Information Filter | Unified recursion; natural diffuse init; parallel sensor fusion | Prediction step requires inversions throughout |

For large $p$ or multi-sensor settings, the Information Filter is preferred. For the typical
econometric SSM with $p \le k$ and a single observation stream, the exact diffuse KF is equally
accurate and simpler to implement.

---

## See also

- [Diffuse Initialization](../kalman/diffuse.md) — classical exact diffuse Kalman filter
- [Kalman Filter](../kalman/kalman-filter.md) — standard moment-form filter
- [Square-Root Filter](square-root.md) — numerically stable Cholesky-factor filter
- [Filter Comparison](comparison.md) — choosing between all filters
- [Numerical Stability](../../theory/numerical-stability.md)
- [API Reference: Filters](../../api/filters.md)
