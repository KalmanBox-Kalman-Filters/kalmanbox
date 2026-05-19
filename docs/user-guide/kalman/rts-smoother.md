# RTS Smoother

The [`RTSSmoother`][kalmanbox.smoothers.rts.RTSSmoother] implements the
**Rauch–Tung–Striebel (RTS) two-pass smoother**. It refines the filtered
estimates $a_{t|t}$ into **smoothed** estimates $a_{t|n}$ that incorporate
the information in the full sample $y_1, \ldots, y_n$ — not just
$y_1, \ldots, y_t$.

---

## Mathematical formulation

### Why smooth?

The filter is causal: $a_{t|t} = E[\alpha_t \mid y_{1:t}]$ uses only past
and present observations. After collecting the complete dataset, future
observations $y_{t+1}, \ldots, y_n$ contain information about the state at
time $t$ through the transition dynamics. The smoother harvests this
additional information.

The result is always at least as precise as the filter:

$$
P_{t|n} \leq P_{t|t} \quad \text{(positive semi-definite ordering)}
$$

### Forward pass (Kalman filter)

Run the standard Kalman filter forward to obtain $\{a_{t|t},\, P_{t|t},\, a_{t|t-1},\, P_{t|t-1}\}_{t=1}^{n}$.

### Backward pass (RTS smoother)

Initialize at the terminal condition: $a_{n|n},\, P_{n|n}$ (already from the filter).

For $t = n-1, n-2, \ldots, 1$, iterate **backward**:

$$
\boxed{
\begin{aligned}
J_t &= P_{t|t}\,T_t'\,P_{t+1|t}^{-1} \quad\text{(smoothing gain)} \\[6pt]
a_{t|n} &= a_{t|t} + J_t\,\bigl(a_{t+1|n} - a_{t+1|t}\bigr) \\[4pt]
P_{t|n} &= P_{t|t} + J_t\,\bigl(P_{t+1|n} - P_{t+1|t}\bigr)\,J_t'
\end{aligned}
}
$$

The smoothed mean $a_{t|n}$ is the filtered mean corrected by the **revision**
$a_{t+1|n} - a_{t+1|t}$, weighted by the smoothing gain $J_t$ which captures
how much the state at $t$ predicts the state at $t+1$.

### Cross-covariance (lag-1 smoother)

Some applications (EM algorithm, disturbance smoothing) require the lag-1
cross-covariance $\text{Cov}(\alpha_{t+1}, \alpha_t \mid y_{1:n})$:

$$
P_{t+1,t|n} = J_t'\,P_{t+1|n}
$$

This is computed automatically when `compute_cross_cov=True` is passed.

---

## Disturbance smoothing

Beyond the state, we often want estimates of the **latent disturbances**
$\eta_t$ (state noise) and $\varepsilon_t$ (observation noise).

The **disturbance smoother** (Koopman 1993) provides:

$$
\begin{aligned}
\hat{\eta}_t &= E[\eta_t \mid y_{1:n}] = Q_t R_t' r_t \\[4pt]
\hat{\varepsilon}_t &= E[\varepsilon_t \mid y_{1:n}] = H_t (F_t^{-1} v_t - K_t' r_t)
\end{aligned}
$$

where $r_t$ is the **smoothed score** computed by the backward recursion:

$$
\begin{aligned}
r_{t-1} &= Z_t' F_t^{-1} v_t + L_t' r_t \\[4pt]
L_t &= T_t - K_t Z_t \\[4pt]
r_n &= 0
\end{aligned}
$$

and their covariances:

$$
\begin{aligned}
\text{Var}(\hat{\eta}_t) &= Q_t - Q_t R_t' N_t R_t Q_t \\[4pt]
\text{Var}(\hat{\varepsilon}_t) &= H_t - H_t (F_t^{-1} + K_t' N_t K_t) H_t
\end{aligned}
$$

where $N_t = Z_t' F_t^{-1} Z_t + L_t' N_{t+1} L_t$ and $N_n = 0$.

These quantities are used for **outlier detection** (large $\hat\varepsilon_t$
relative to its variance), **structural break tests**, and the **EM algorithm**.

---

## API

### Basic usage

```python
import numpy as np
from kalmanbox import KalmanFilter, RTSSmoother, StateSpaceRepresentation

# ── Define and run the forward filter ─────────────────────────────────────────
T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[0.5]])
H = np.array([[1.0]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)

# ── Run the backward smoother ─────────────────────────────────────────────────
smoother = RTSSmoother(out, ssr)
sm = smoother.run()

sm.a_smoothed   # E[alpha_t | y_{1:n}]   shape (n, k)
sm.P_smoothed   # Var(alpha_t | y_{1:n}) shape (n, k, k)
sm.J            # smoothing gains         shape (n, k, k)
```

### With disturbance smoothing

```python
from kalmanbox.smoothers import DisturbanceSmoother

ds = DisturbanceSmoother(out, ssr)
dout = ds.run()

dout.eta_hat       # E[eta_t | y_{1:n}]       shape (n, g)
dout.eps_hat       # E[eps_t | y_{1:n}]       shape (n, p)
dout.eta_var       # Var(eta_t | y_{1:n})     shape (n, g, g)
dout.eps_var       # Var(eps_t | y_{1:n})     shape (n, p, p)
dout.r             # smoothed score r_t        shape (n, k)
dout.N             # smoothed precision N_t    shape (n, k, k)
```

### With lag-1 cross-covariance (for EM)

```python
smoother = RTSSmoother(out, ssr, compute_cross_cov=True)
sm = smoother.run()

sm.P_cross     # Cov(alpha_{t+1}, alpha_t | y_{1:n})  shape (n-1, k, k)
```

---

## Smoothed vs. filtered states

### Conceptual comparison

| Property                     | Filtered $a_{t\mid t}$        | Smoothed $a_{t\mid n}$              |
|------------------------------|-------------------------------|-------------------------------------|
| Conditioning set             | $y_1, \ldots, y_t$            | $y_1, \ldots, y_n$ (full sample)    |
| Uncertainty ($P$)            | Larger                        | Smaller or equal                    |
| Computation                  | Forward pass only             | Forward + backward pass             |
| Available in real time       | Yes                           | No (needs future data)              |
| Suitable for forecasting     | Yes (at $t = n$)              | No                                  |
| Suitable for decomposition   | No (one-sided)                | Yes (two-sided)                     |

### Variance reduction

The difference in uncertainty is largest in the **middle** of the sample and
vanishes at $t = n$ (terminal condition: $P_{n|n} = P_{n|n}$):

```
Variance
 │
 │  ╔══ Filtered P_{t|t} ══════════════════════════════╗
 │  ║                                                   ║
 │  ║    ┌── Smoothed P_{t|n} ──────────────────────┐  ║
 │  ║    │                                           │  ║
 │  ╚════╩═══════════════════════════════════════════╩══╝
 └────────────────────────────────────────────────────────► t
    1                         n/2                      n
```

At $t = n$ both coincide. Early in the sample, the smoother benefits most
from future observations.

---

## When to smooth vs. filter

```
Do you need estimates as soon as data arrives (real time)?
├─ Yes → Use the Kalman Filter (online, causal)
└─ No  → Do you have the complete dataset?
         ├─ Yes → Use the RTS Smoother (full-sample, two-sided)
         └─ Partial (bounded lag) → Use FixedLagSmoother
```

| Use case                               | Use filter | Use smoother |
|----------------------------------------|:----------:|:------------:|
| Streaming / online estimation          | ✅         | ❌           |
| Historical re-analysis                 | ❌         | ✅           |
| Signal extraction (trend, cycle)       | ❌         | ✅           |
| Forecasting beyond sample end          | ✅         | ❌ (use filter at $t=n$) |
| Parameter estimation via EM algorithm  | Both       | ✅ (M-step)  |
| Disturbance / outlier estimation       | ❌         | ✅           |
| State initialization in mixed models   | ❌         | ✅           |

---

## Examples

### Example 1: Trend extraction with the Local Linear Trend

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import KalmanFilter, RTSSmoother, StateSpaceRepresentation

sigma_mu, sigma_nu, sigma_eps = 0.3, 0.05, 1.5

T = np.array([[1.0, 1.0], [0.0, 1.0]])
Z = np.array([[1.0, 0.0]])
R = np.eye(2)
Q = np.diag([sigma_mu**2, sigma_nu**2])
H = np.array([[sigma_eps**2]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

# Simulate a slowly trending series
rng = np.random.default_rng(5)
n   = 250
mu  = np.zeros(n + 1)
nu  = np.zeros(n + 1)
nu[0] = 0.1
for t in range(n):
    nu[t+1] = nu[t] + rng.normal(scale=sigma_nu)
    mu[t+1] = mu[t] + nu[t] + rng.normal(scale=sigma_mu)
y = mu[1:] + rng.normal(scale=sigma_eps, size=n)

# Filter and smooth
kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)
sm  = RTSSmoother(out, ssr).run()

# Compare filtered vs smoothed trend
mu_filtered = out.a_filtered[:, 0]
mu_smoothed = sm.a_smoothed[:, 0]

# Smoothed trend is typically less noisy and lags less at turning points
rmse_filter  = np.sqrt(np.mean((mu_filtered - mu[1:])**2))
rmse_smoother = np.sqrt(np.mean((mu_smoothed - mu[1:])**2))
print(f"RMSE filtered : {rmse_filter:.4f}")
print(f"RMSE smoothed : {rmse_smoother:.4f}")   # should be smaller
```

### Example 2: Component estimation with BSM

The Basic Structural Model decomposes $y_t$ into trend, seasonal, and irregular:

```python
from kalmanbox.structural import BSM
from kalmanbox import RTSSmoother

model  = BSM(period=12)               # monthly data
result = model.fit(y, method="mle")   # estimates all variances

# Access smoothed components
sm = result.smoother_output           # RTSSmoother run internally

trend    = result.trend               # mu_{t|n}  (smoothed)
seasonal = result.seasonal            # gamma_{t|n} (smoothed)
irregular = y - trend - seasonal      # residual

print(f"Trend range   : [{trend.min():.2f}, {trend.max():.2f}]")
print(f"Seasonal range: [{seasonal.min():.2f}, {seasonal.max():.2f}]")
```

### Example 3: Disturbance smoothing for outlier detection

Large smoothed observation disturbances $\hat\varepsilon_t$ relative to their
standard deviation indicate potential outliers or level shifts:

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation
from kalmanbox.smoothers import DisturbanceSmoother

# Local Level with a planted outlier at t=100
T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[0.5]])
H = np.array([[1.0]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

rng = np.random.default_rng(9)
n   = 200
y   = rng.normal(scale=1.0, size=n)
y[100] += 8.0    # plant an outlier

kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)

ds   = DisturbanceSmoother(out, ssr)
dout = ds.run()

# Standardized smoothed disturbances
eps_hat = dout.eps_hat[:, 0]
eps_std = np.sqrt(dout.eps_var[:, 0, 0])
standardized = eps_hat / eps_std

# Flag observations with |standardized| > 3
outliers = np.where(np.abs(standardized) > 3)[0]
print(f"Suspected outliers at t = {outliers}")   # should contain 100
```

### Example 4: EM algorithm using smoothed statistics

The M-step of the EM algorithm for a Local Level model requires the smoothed
sufficient statistics:

```python
import numpy as np
from kalmanbox import KalmanFilter, RTSSmoother, StateSpaceRepresentation

def em_local_level(y, n_iter=50):
    n = len(y)
    # Starting values
    sigma_eta2 = 1.0
    sigma_eps2 = 1.0

    for _ in range(n_iter):
        # E-step: filter + smooth
        ssr = StateSpaceRepresentation(
            T=np.array([[1.0]]), Z=np.array([[1.0]]),
            R=np.array([[1.0]]),
            Q=np.array([[sigma_eta2]]), H=np.array([[sigma_eps2]]),
        )
        kf  = KalmanFilter(ssr, initialization="diffuse")
        out = kf.run(y)
        sm  = RTSSmoother(out, ssr, compute_cross_cov=True).run()

        # Smoothed sufficient statistics
        # E[alpha_t^2 | y] = P_{t|n}[0,0] + a_{t|n}[0]^2
        Ealpha2      = sm.P_smoothed[:, 0, 0] + sm.a_smoothed[:, 0]**2
        # E[alpha_{t+1} alpha_t | y] = P_{t+1,t|n}[0,0] + a_{t+1|n}[0]*a_{t|n}[0]
        Ealpha_cross = (sm.P_cross[:, 0, 0]
                        + sm.a_smoothed[1:, 0] * sm.a_smoothed[:-1, 0])

        # M-step: closed-form updates
        sigma_eta2 = (
            np.sum(Ealpha2[1:]) - 2 * np.sum(Ealpha_cross) + np.sum(Ealpha2[:-1])
        ) / (n - 1)
        sigma_eps2 = (
            np.sum(y**2) - 2 * np.sum(y * sm.a_smoothed[:, 0]) + np.sum(Ealpha2)
        ) / n

    return sigma_eta2, sigma_eps2

sigma_eta2_hat, sigma_eps2_hat = em_local_level(y)
print(f"EM estimated: sigma_eta={np.sqrt(sigma_eta2_hat):.4f}, "
      f"sigma_eps={np.sqrt(sigma_eps2_hat):.4f}")
```

---

## Related smoothers

| Class | Description | When to use |
|-------|-------------|-------------|
| [`RTSSmoother`][kalmanbox.smoothers.rts.RTSSmoother] | Standard two-pass RTS | Fixed interval, full sample |
| [`FixedIntervalSmoother`][kalmanbox.smoothers.fixed_interval.FixedIntervalSmoother] | Alternative fixed-interval form | More memory-efficient formulation |
| [`FixedLagSmoother`][kalmanbox.smoothers.fixed_lag.FixedLagSmoother] | Bounded-lag smoothing | Online with latency $\ell$ |
| [`DisturbanceSmoother`][kalmanbox.smoothers.disturbance.DisturbanceSmoother] | Smoothed disturbances $\hat\eta_t$, $\hat\varepsilon_t$ | Outlier detection, EM |

---

## Related

- [Kalman Filter](kalman-filter.md) — forward pass
- [Forecasting](forecasting.md) — out-of-sample prediction from $a_{n|n}$
- [Theory: RTS derivation](../../theory/rts-derivation.md)
- [Theory: Kalman filter derivation](../../theory/kalman-filter-derivation.md)
- [Visualization: smoothed states](../../visualization/smoothed-states.md)
- [API: RTSSmoother](../../api/smoothers.md)
- [Diagnostics: residuals](../../diagnostics/residuals.md)
