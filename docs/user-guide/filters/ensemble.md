# Ensemble Kalman Filter (EnKF)

The [`EnKF`][kalmanbox.filters.EnKF] implements the **Ensemble Kalman Filter**, a Monte Carlo
approximation of the Kalman recursion designed for **very high-dimensional state spaces** where
storing and propagating the full covariance matrix $P_t \in \mathbb{R}^{k \times k}$ is
computationally intractable. Instead of propagating $P_t$ explicitly, the EnKF maintains an
**ensemble** of $N$ state realizations $\{\alpha_t^{(i)}\}_{i=1}^N$ whose empirical covariance
approximates $P_t$.

!!! note "When to use the EnKF"
    - State dimension $k \gtrsim 10^3$ (geophysics, fluid dynamics, high-dimensional economic
      models) where forming $P_t$ is prohibitively expensive in memory or time
    - The transition function $f$ is a complex nonlinear simulator without a closed-form Jacobian
    - Mildly to strongly nonlinear dynamics where EKF/UKF approximations are too inaccurate
    - The model evaluates naturally in parallel (each ensemble member is independent)

    For moderate dimensions ($k \lesssim 100$) with smooth nonlinearity, prefer the [UKF](ukf.md).
    For linear models, use the [standard Kalman filter](../kalman/kalman-filter.md).

---

## 1. Ensemble representation of the posterior

The standard Kalman filter represents the posterior at time $t$ as:

$$
p(\alpha_t \mid \mathcal{F}_t) = \mathcal{N}(a_{t|t},\, P_{t|t})
$$

The EnKF replaces this with an **empirical distribution** over $N$ ensemble members:

$$
\hat{p}(\alpha_t \mid \mathcal{F}_t) = \frac{1}{N} \sum_{i=1}^N \delta(\alpha_t - \alpha_{t|t}^{(i)})
$$

The ensemble approximates the Gaussian moments as:

$$
a_{t|t} \approx \bar{\alpha}_{t|t} = \frac{1}{N} \sum_{i=1}^N \alpha_{t|t}^{(i)}
$$

$$
P_{t|t} \approx \hat{P}_{t|t} = \frac{1}{N-1} \sum_{i=1}^N
\bigl(\alpha_{t|t}^{(i)} - \bar{\alpha}_{t|t}\bigr)
\bigl(\alpha_{t|t}^{(i)} - \bar{\alpha}_{t|t}\bigr)'
$$

The approximation error decays at the Monte Carlo rate $O(N^{-1/2})$.

The key computational saving: the empirical covariance $\hat{P}_{t|t}$ is **never formed
explicitly**. It is represented implicitly by the $k \times N$ **anomaly matrix** of rank
$\min(N-1, k)$, reducing storage from $O(k^2)$ to $O(Nk)$.

---

## 2. EnKF algorithm

### 2.1 Initialization

Draw $N$ ensemble members from the initial distribution:

$$
\alpha_{0|0}^{(i)} \overset{\text{iid}}{\sim} \mathcal{N}(a_0,\, P_0), \quad i = 1, \ldots, N
$$

```python
import numpy as np

k = 50     # state dimension
N = 200    # ensemble size

a0 = np.zeros(k)
P0 = np.eye(k)
L0 = np.linalg.cholesky(P0)

alpha_ensemble = a0[:, None] + L0 @ np.random.randn(k, N)   # shape (k, N)
```

For diffuse initialization, draw from a broad distribution or initialize with a single state
and let the ensemble spread develop through the first few time steps with added noise.

### 2.2 Forecast step (prediction)

Propagate each ensemble member through the (potentially nonlinear) transition function with
independent noise realizations:

$$
\boxed{
\alpha_{t|t-1}^{(i)} = f\!\left(\alpha_{t-1|t-1}^{(i)},\, u_t\right) + R\,\eta_t^{(i)},
\quad \eta_t^{(i)} \overset{\text{iid}}{\sim} \mathcal{N}(0,\, Q)
}
$$

This step is **embarrassingly parallel** — each member evolves independently.

**Compute forecast ensemble statistics:**

$$
\bar{\alpha}_{t|t-1} = \frac{1}{N} \sum_{i=1}^N \alpha_{t|t-1}^{(i)}
$$

$$
A_{t|t-1} = \frac{1}{\sqrt{N-1}}
\Bigl[\alpha_{t|t-1}^{(1)} - \bar{\alpha}_{t|t-1} \;\Big\vert\; \cdots \;\Big\vert\;
\alpha_{t|t-1}^{(N)} - \bar{\alpha}_{t|t-1}\Bigr]
\in \mathbb{R}^{k \times N}
$$

The forecast covariance is approximated implicitly as:

$$
\hat{P}_{t|t-1} = A_{t|t-1}\,A_{t|t-1}' \approx P_{t|t-1}
$$

### 2.3 Analysis step (update)

**Project ensemble into observation space:**

$$
y_{t|t-1}^{(i)} = H_t\,\alpha_{t|t-1}^{(i)}
\quad (\text{linear}) \qquad\text{or}\qquad
y_{t|t-1}^{(i)} = h\!\left(\alpha_{t|t-1}^{(i)}\right)
\quad (\text{nonlinear})
$$

$$
\bar{y}_{t|t-1} = \frac{1}{N} \sum_{i=1}^N y_{t|t-1}^{(i)}
$$

$$
B_{t|t-1} = \frac{1}{\sqrt{N-1}}
\Bigl[y_{t|t-1}^{(1)} - \bar{y}_{t|t-1} \;\Big\vert\; \cdots \;\Big\vert\;
y_{t|t-1}^{(N)} - \bar{y}_{t|t-1}\Bigr]
\in \mathbb{R}^{p \times N}
$$

**Ensemble innovation covariance:**

$$
\hat{S}_t = B_{t|t-1}\,B_{t|t-1}' + H_t \approx S_t = Z_t P_{t|t-1} Z_t' + H_t
$$

**Ensemble Kalman gain:**

$$
\boxed{\hat{K}_t = A_{t|t-1}\,B_{t|t-1}'\,\hat{S}_t^{-1}}
$$

The gain is computed via products of rank-$N$ anomaly matrices — no explicit $k \times k$
covariance is formed. The dominant cost is $\hat{S}_t^{-1}$, which is $p \times p$.

**Stochastic update (Burgers, van Leeuwen & Evensen 1998):**

Draw **perturbed observations** to preserve ensemble spread:

$$
\tilde{y}_t^{(i)} = y_t + e_t^{(i)}, \qquad e_t^{(i)} \overset{\text{iid}}{\sim} \mathcal{N}(0,\, H_t)
$$

Update each member:

$$
\boxed{
\alpha_{t|t}^{(i)} = \alpha_{t|t-1}^{(i)} + \hat{K}_t\,\bigl(\tilde{y}_t^{(i)} - y_{t|t-1}^{(i)}\bigr)
}
$$

The perturbed-observation trick ensures that the updated ensemble covariance approximates
$P_{t|t} = (I - \hat{K}_t Z_t)\,\hat{P}_{t|t-1}$ in expectation over the perturbations.

!!! info "Deterministic EnKF variants"
    The **Ensemble Square-Root Filter (EnSRF)** and **Ensemble Transform Kalman Filter (ETKF)**
    update the anomaly matrix directly, avoiding observation perturbations:

    $$
    A_{t|t} = A_{t|t-1}\,\Bigl(I_N - \tfrac{1}{2}\,B_{t|t-1}'\,\hat{S}_t^{-1}\,B_{t|t-1}\Bigr)^{1/2}
    $$

    These deterministic variants eliminate the $O(N^{-1/2})$ sampling noise from observation
    perturbations and are available in kalmanbox as `EnKF(variant="etkf")` or `"ensrf"`.

---

## 3. Covariance localization

With a finite ensemble of size $N \ll k$, the empirical covariance has rank at most $N-1$
and exhibits **spurious long-range correlations** between physically unrelated state components.
Localization suppresses these artifacts by tapering the covariance to zero beyond a physical
distance threshold.

### B-localization (covariance localization)

Apply a **Schur (elementwise) product** of the ensemble covariance with a distance-based taper:

$$
\tilde{P}_{t|t-1} = \rho \circ \hat{P}_{t|t-1}
$$

where $\rho_{ij}$ decays from 1 to 0 as the distance between state components $i$ and $j$
increases. The standard choice is the **Gaspari–Cohn compactly supported function**:

$$
\rho(r) = \begin{cases}
-\dfrac{1}{4}r^5 + \dfrac{1}{2}r^4 + \dfrac{5}{8}r^3 - \dfrac{5}{3}r^2 + 1
& 0 \le r \le 1 \\[8pt]
\dfrac{1}{12}r^5 - \dfrac{1}{2}r^4 + \dfrac{5}{8}r^3 + \dfrac{5}{3}r^2 - 5r + 4 - \dfrac{2}{3r}
& 1 < r \le 2 \\[8pt]
0 & r > 2
\end{cases}
$$

where $r = d / r_c$, $d$ is the physical distance between components, and $r_c$ is the
**localization radius** (tuning parameter).

### R-localization (observation-space localization)

Alternatively, inflate the observation error covariance for observations far from each state
component. This is more natural for geographically distributed observation networks:

$$
\tilde{H}_{ij} = H_{ij}\, /\, \rho(d_{ij} / r_c)
$$

where $d_{ij}$ is the distance from observation $j$ to state component $i$.

```python
from kalmanbox.filters import EnKF

enkf = EnKF(
    model=model,
    ensemble_size=100,
    H=H,
    localization=True,
    localization_radius=10.0,           # in state grid spacing units
    localization_fn="gaspari-cohn",     # or "exponential", "boxcar", callable
    localization_type="B",              # "B" (covariance) or "R" (observation)
)
```

---

## 4. Covariance inflation

With finite $N$, the ensemble tends to **underestimate true uncertainty** (covariance collapse)
over many analysis cycles. Inflation counteracts this by artificially increasing ensemble spread.

### Multiplicative inflation

Rescale anomalies by a factor $(1 + \delta)$ before each analysis step:

$$
\alpha_{t|t-1}^{(i)} \leftarrow \bar{\alpha}_{t|t-1} + (1 + \delta)\,\bigl(\alpha_{t|t-1}^{(i)} - \bar{\alpha}_{t|t-1}\bigr)
$$

This increases the ensemble covariance by $(1+\delta)^2$. Typical values: $\delta \in [0.01, 0.1]$.

### Additive inflation

Add independent perturbations drawn from the model error covariance:

$$
\alpha_{t|t-1}^{(i)} \leftarrow \alpha_{t|t-1}^{(i)} + b^{(i)}, \qquad b^{(i)} \sim \mathcal{N}(0,\, \sigma_b^2\,Q)
$$

Additive inflation is preferable when the model error distribution is well-characterized and
multiplicative inflation alone cannot compensate for systematic underestimation.

### Adaptive inflation

kalmanbox supports **adaptive covariance inflation** (Anderson 2009), which estimates the
optimal inflation factor from the data by matching observed innovation statistics to their
expected values under a correct filter:

```python
enkf = EnKF(
    model=model,
    ensemble_size=100,
    H=H,
    inflation="adaptive",        # estimate delta online
    inflation_prior=(1.0, 0.6),  # Gaussian prior: (mean, std) of inflation factor
)
```

---

## 5. Computational complexity

| Operation | Standard Kalman Filter | EnKF |
|-----------|:---------------------:|:----:|
| Covariance storage | $O(k^2)$ | $O(Nk)$ — anomaly matrix only |
| Forecast step | $O(k^3)$ — matrix multiply | $O(Nk)$ per member, fully parallel |
| Analysis (Kalman gain) | $O(k^3 + p^3)$ | $O(Nkp + p^3)$ — rank-$N$ products |
| Per-step total | $O(k^3 + p^3)$ | $O(Nk^2 + p^3)$ |

When $k \gg N$, the EnKF reduces covariance storage from $O(k^2)$ to $O(Nk)$ — a factor of
$k/N$ saving. For $k = 10^6$ and $N = 100$, storage drops from terabytes to megabytes.

### Parallelism

The forecast step is **embarrassingly parallel**: each ensemble member evolves independently.
kalmanbox dispatches ensemble members across available CPU cores automatically:

```python
enkf = EnKF(model=model, ensemble_size=500, H=H, n_jobs=-1)  # use all cores
```

For GPU-accelerated models, pass a model that accepts batched inputs:

```python
class BatchedModel(EnKFModel):
    def forecast(self, ensemble, t):
        # ensemble shape: (k, N) — vectorized over the N dimension
        return jax.vmap(self._single_step)(ensemble.T).T
```

---

## 6. API reference

```python
from kalmanbox.filters import EnKF, EnKFModel
import numpy as np


class MyModel(EnKFModel):
    """Subclass EnKFModel to define the transition and observation functions."""

    def forecast(
        self,
        ensemble: np.ndarray,   # shape (k, N): current ensemble
        t: int,                 # current time index
    ) -> np.ndarray:
        """Propagate ensemble forward one time step. Return shape (k, N)."""
        ...

    def observe(
        self,
        ensemble: np.ndarray,   # shape (k, N): forecast ensemble
        t: int,
    ) -> np.ndarray:
        """Map ensemble to observation space. Return shape (p, N)."""
        ...


enkf = EnKF(
    model,                           # EnKFModel instance
    ensemble_size=200,               # N — number of ensemble members
    H=H,                             # observation noise covariance, shape (p, p)
    localization=False,              # enable covariance localization
    localization_radius=None,        # localization radius r_c
    localization_fn="gaspari-cohn",  # "gaspari-cohn" | "exponential" | "boxcar" | callable
    localization_type="B",           # "B" (covariance-space) | "R" (observation-space)
    inflation=1.0,                   # float (multiplicative) or "adaptive"
    inflation_prior=None,            # (mean, std) tuple for adaptive inflation prior
    variant="stochastic",            # "stochastic" | "etkf" | "ensrf"
    n_jobs=1,                        # parallel ensemble members; -1 = all cores
    rng=None,                        # np.random.Generator for reproducibility
)
```

### Key methods

| Method | Returns | Description |
|--------|---------|-------------|
| `filter(y, alpha0_ensemble)` | `EnsembleResult` | Forward EnKF pass over all observations |
| `smooth(y, alpha0_ensemble)` | `EnsembleResult` | Forward filter + ensemble backward smoother |
| `log_likelihood(y, alpha0_ensemble)` | `float` | Approximate log-likelihood |
| `forecast(n_steps)` | `EnsembleResult` | Ensemble forecast `n_steps` into the future |

### EnsembleResult attributes

```python
result = enkf.filter(y, alpha0_ensemble)

result.ensemble              # shape (T, k, N): full ensemble history (all members)
result.ensemble_mean         # shape (T, k):    bar{alpha}_{t|t}
result.ensemble_covariance   # shape (T, k, k): hat{P}_{t|t} (computed on demand)
result.ensemble_spread       # shape (T, k):    per-component standard deviation
result.innovations_mean      # shape (T, p):    bar{v}_t = y_t - bar{y}_{t|t-1}
result.inflation_history     # shape (T,):      adaptive inflation factor per step
result.log_likelihood        # scalar: approximate log-likelihood
```

---

## 7. Examples

### Example 1: Nonlinear 1D tracking (Kitagawa model)

The Kitagawa (1996) nonlinear model is a standard benchmark for nonlinear filters. The state
transition and observation are both nonlinear:

$$
\alpha_{t+1} = \frac{\alpha_t}{2} + \frac{25\,\alpha_t}{1 + \alpha_t^2} + 8\cos(1.2\,t) + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1)
$$

$$
y_t = \frac{\alpha_t^2}{20} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)
$$

```python
import numpy as np
from kalmanbox.filters import EnKF, EnKFModel


class KitagawaModel(EnKFModel):
    def forecast(self, ensemble: np.ndarray, t: int) -> np.ndarray:
        # ensemble shape: (1, N)
        x = ensemble[0]
        x_next = 0.5*x + 25*x/(1 + x**2) + 8*np.cos(1.2*t) + np.random.randn(len(x))
        return x_next[None, :]   # shape (1, N)

    def observe(self, ensemble: np.ndarray, t: int) -> np.ndarray:
        return (ensemble**2 / 20)   # nonlinear; shape (1, N)


np.random.seed(42)
T = 100
N = 300

# Simulate true trajectory and observations
x_true = np.zeros(T + 1)
x_true[0] = 0.0
for t in range(T):
    x_true[t+1] = (0.5*x_true[t] + 25*x_true[t]/(1 + x_true[t]**2)
                   + 8*np.cos(1.2*(t+1)) + np.random.randn())
y = x_true[1:]**2 / 20 + np.random.randn(T)

# EnKF
H_obs = np.array([[1.0]])
model = KitagawaModel()
enkf = EnKF(model=model, ensemble_size=N, H=H_obs, inflation=1.05, variant="etkf")

alpha0_ens = np.random.randn(1, N)   # diffuse initialization
result = enkf.filter(y.reshape(-1, 1), alpha0_ens)

rmse = np.sqrt(np.mean((result.ensemble_mean[:, 0] - x_true[1:])**2))
print(f"EnKF RMSE (N={N}): {rmse:.3f}")
print(f"Ensemble mean (final step): {result.ensemble_mean[-1, 0]:.3f}")
print(f"Ensemble spread (final step): {result.ensemble_spread[-1, 0]:.3f}")
```

### Example 2: High-dimensional data assimilation (Lorenz-96)

The Lorenz (1996) model is the standard benchmark for large-scale data assimilation. It
represents $k$ atmosphere variables on a periodic ring with chaotic dynamics:

$$
\frac{d\alpha_j}{dt} = (\alpha_{j+1} - \alpha_{j-2})\,\alpha_{j-1} - \alpha_j + F,
\quad j = 1, \ldots, k
$$

with $F = 8$ (chaotic regime). With $k = 100$ state variables and $N = 50$ ensemble members,
the standard Kalman filter ($O(k^3) = O(10^6)$ per step) is still feasible but the EnKF
demonstrates the localization and inflation machinery:

```python
import numpy as np
from kalmanbox.filters import EnKF, EnKFModel


class Lorenz96(EnKFModel):
    def __init__(self, k: int = 100, F: float = 8.0, dt: float = 0.05):
        self.k = k
        self.F = F
        self.dt = dt
        self.sigma_q = 0.1

    def _rhs(self, x: np.ndarray) -> np.ndarray:
        """Lorenz-96 RHS, vectorized over ensemble. x shape: (k, N)."""
        return (
            (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0)
            - x + self.F
        )

    def forecast(self, ensemble: np.ndarray, t: int) -> np.ndarray:
        """4th-order Runge-Kutta, one time step."""
        dt = self.dt
        k1 = self._rhs(ensemble)
        k2 = self._rhs(ensemble + 0.5 * dt * k1)
        k3 = self._rhs(ensemble + 0.5 * dt * k2)
        k4 = self._rhs(ensemble + dt * k3)
        x_next = ensemble + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next + self.sigma_q * np.random.randn(*x_next.shape)

    def observe(self, ensemble: np.ndarray, t: int) -> np.ndarray:
        """Observe every other grid point."""
        return ensemble[::2, :]   # shape (k//2, N)


np.random.seed(0)
k_state = 100
N_ens = 50
T_steps = 200

model = Lorenz96(k=k_state)
p_obs = k_state // 2
H_obs = np.eye(p_obs)   # direct observation, unit noise

enkf = EnKF(
    model=model,
    ensemble_size=N_ens,
    H=H_obs,
    localization=True,
    localization_radius=5.0,    # localize over 5 grid points
    inflation=1.05,             # 5% multiplicative inflation
    variant="etkf",             # deterministic — no observation perturbations
    rng=np.random.default_rng(seed=42),
)

# Warm-up true state from F equilibrium
x_truth = np.random.randn(k_state) + model.F
for _ in range(200):
    x_truth = x_truth + model.dt * model._rhs(x_truth[:, None])[:, 0]

# Simulate observations
y_all = np.zeros((T_steps, p_obs))
for t in range(T_steps):
    x_truth = x_truth + model.dt * model._rhs(x_truth[:, None])[:, 0]
    y_all[t] = x_truth[::2] + np.random.randn(p_obs)

# Initial ensemble
alpha0_ens = np.random.randn(k_state, N_ens) + model.F

result = enkf.filter(y_all, alpha0_ens)

final_rmse = np.sqrt(np.mean((result.ensemble_mean[-1] - x_truth)**2))
mean_spread = result.ensemble_spread[-1].mean()
print(f"Final RMSE:             {final_rmse:.3f}")
print(f"Final ensemble spread:  {mean_spread:.3f}")
print(f"Spread/RMSE ratio:      {mean_spread / final_rmse:.2f}  (ideal ≈ 1)")
```

!!! tip "Spread/RMSE consistency check"
    A well-calibrated ensemble filter should have a spread-to-RMSE ratio near 1. A ratio
    much less than 1 indicates filter divergence; much greater than 1 indicates over-inflation.

---

## 8. Limitations and failure modes

!!! warning "Common pitfalls"

    **Ensemble collapse (filter divergence)**
    : When $N$ is too small relative to the effective uncertainty rank, the ensemble
      degenerates to a single point. The filter becomes overconfident, ignoring new
      observations. **Fix:** increase $N$, add multiplicative inflation, apply localization.

    **Spurious long-range correlations**
    : With small $N$, the empirical covariance has many statistically insignificant
      off-diagonal entries that couple unrelated state components. **Fix:** apply
      B-localization or R-localization with a physically motivated radius.

    **Monte Carlo noise in the log-likelihood**
    : The EnKF log-likelihood estimate has $O(N^{-1/2})$ variance. For parameter estimation
      via MLE, use large $N$ (≥ 500) or switch to the particle filter (available in
      [particlefilterbox](https://particlefilterbox.nodesecon.com)) for accurate likelihood.

    **Non-Gaussian posteriors**
    : The EnKF analysis step assumes Gaussian posteriors and is not consistent for
      multimodal or heavy-tailed distributions. For strongly non-Gaussian posteriors,
      use the particle filter instead.

---

## 9. Practical guidelines

### Choosing the ensemble size $N$

| State dimension $k$ | Minimum $N$ | Recommended $N$ |
|---------------------|:-----------:|:---------------:|
| $\le 100$ | 50 | 100–500 |
| $10^2$–$10^3$ | 100 | 500–2,000 |
| $10^3$–$10^6$ | 50–200 | 200–1,000 + localization |
| $> 10^6$ | 20–100 | 50–500 + strong localization |

The key rule: $N$ should exceed the **effective rank** of the dominant uncertainty subspace.
With localization, this is typically much smaller than $k$, enabling $N \ll k$.

### Tuning the localization radius $r_c$

Start with $r_c \approx \sqrt{k/p}$ grid spacings as an initial guess. Then tune by
cross-validation (hold out observations and minimize validation RMSE) or by tracking the
spread/RMSE ratio — a ratio far from 1 indicates mis-calibration that localization can correct.

### Variant selection

| Variant | Pros | Cons |
|---------|------|------|
| `"stochastic"` (default) | Simple; standard in literature | $O(N^{-1/2})$ observation-perturbation noise |
| `"etkf"` | Deterministic; no perturbation noise; ensemble exactly updated | Slightly more complex; may require re-centering |
| `"ensrf"` | Processes observations sequentially; efficient for large $p$ | Order-dependent; requires $H$ to be diagonal or decomposable |

---

## See also

- [EKF](ekf.md) — deterministic nonlinear filter for moderate dimensions
- [UKF](ukf.md) — sigma-point filter for moderate nonlinearity without Jacobians
- [Information Filter](information.md) — inverse-covariance form; natural for sensor fusion
- [Filter Comparison](comparison.md) — side-by-side comparison of all six filters
- [Nonlinear Tracking Tutorial](../../tutorials/nonlinear-tracking.md)
- [Theory: Numerical Stability](../../theory/numerical-stability.md)
- [API Reference: Filters](../../api/filters.md)
