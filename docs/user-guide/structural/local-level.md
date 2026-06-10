# Local Level Model

The **Local Level** model — also called the *random walk plus noise* model or
*integrated moving-average IMA(1,1)* in the ARIMA family — is the simplest
non-trivial state-space model. Despite its simplicity it is a useful forecasting
benchmark, the building block of every more complex structural model, and the
canonical example for learning state-space methods.

---

## Mathematical formulation

The model has two equations. The **measurement equation** links the observed
$y_t$ to the latent level $\mu_t$:

$$
y_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\,\sigma_\varepsilon^2)
$$

The **state equation** describes how the level evolves over time as a random
walk:

$$
\mu_{t+1} = \mu_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\,\sigma_\eta^2)
$$

The two disturbances are independent: $\eta_t \perp \varepsilon_s$ for all
$t, s$.

!!! info "Indexing convention"

    kalmanbox uses the Durbin & Koopman (2012) convention where the state
    equation defines $\mu_{t+1}$ given $\mu_t$.  Some textbooks write
    $\mu_t = \mu_{t-1} + \eta_{t-1}$, which is equivalent but shifts the
    subscripts.

---

## State-space representation

In kalmanbox's general state-space form

$$
\begin{aligned}
\alpha_{t+1} &= T\,\alpha_t + R\,\eta_t \\
y_t          &= Z\,\alpha_t + \varepsilon_t
\end{aligned}
$$

the Local Level model maps to scalar matrices:

$$
T = \begin{bmatrix} 1 \end{bmatrix}, \quad
Z = \begin{bmatrix} 1 \end{bmatrix}, \quad
R = \begin{bmatrix} 1 \end{bmatrix}, \quad
Q = \begin{bmatrix} \sigma_\eta^2 \end{bmatrix}, \quad
H = \begin{bmatrix} \sigma_\varepsilon^2 \end{bmatrix}
$$

The state vector is $\alpha_t = \mu_t \in \mathbb{R}^1$. The transition matrix
$T = 1$ encodes the unit root (pure random walk). $R = 1$ means the shock
enters the state directly (no selection matrix reduction).

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

sigma_eta = 25.0    # level shock std dev (not yet estimated)
sigma_eps = 15.0    # measurement noise std dev

ssr = StateSpaceRepresentation(
    T=np.array([[1.0]]),
    Z=np.array([[1.0]]),
    R=np.array([[1.0]]),
    Q=np.array([[sigma_eta**2]]),
    H=np.array([[sigma_eps**2]]),
)

kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)
```

---

## Signal-to-noise ratio

The two variance parameters are not separately identified from the **shape** of
the filtered path — only their ratio matters for the smoothness of the level
estimate. Define the **signal-to-noise ratio**:

$$
q = \frac{\sigma_\eta^2}{\sigma_\varepsilon^2}
$$

| $q$ | Behaviour |
|-----|-----------|
| $q \to 0$ | Level barely moves; $\hat\mu_t \approx \bar{y}$ (global mean) |
| $q \approx 0.01$–$0.1$ | Slow, smooth drifts |
| $q \approx 1$ | Level tracks $y_t$ moderately |
| $q \to \infty$ | Level follows $y_t$ exactly (no smoothing) |

!!! example "Intuition"

    With Nile river discharge data ($n=100$), MLE returns
    $\hat{\sigma}_\eta^2 \approx 1469$ and
    $\hat{\sigma}_\varepsilon^2 \approx 471$, giving $q \approx 3.1$.
    The level is *highly reactive* — each year's measurement shifts the
    estimated level substantially — reflecting the well-known 1899 Aswan Dam
    intervention visible in the raw data.

The Kalman gain at steady state converges to a constant $k_\infty$ that depends
only on $q$:

$$
k_\infty = \frac{-1 + \sqrt{1 + 4/q}}{2/q}
$$

The smoothed level is then equivalent to an EWMA with discount factor
$1 - k_\infty$.

---

## Parameter estimation (MLE)

### High-level API

```python
from kalmanbox.structural import LocalLevel
from kalmanbox.datasets import load_nile

nile    = load_nile()
y       = nile["volume"].to_numpy()      # annual discharge, m³/s × 10⁸

model   = LocalLevel(y)
results = model.fit(method="mle", n_starts=10, disp=True)

print(results.summary())
```

```
                 Local Level Model Results
=============================================================
Dep. Variable:   volume     Log-Likelihood:  -632.537
No. Observations: 100       AIC:             1269.073
Df Model:          2        BIC:             1274.284
                             HQIC:            1271.184
=============================================================
             Estimate   Std.Err    z-stat    p-value
sigma2_eta   1469.01    363.15     4.044    0.0001
sigma2_eps    471.44    181.96     2.591    0.0096
=============================================================
Signal-to-noise  q = 3.12
```

### Low-level MLE from scratch

```python
import numpy as np
from scipy.optimize import minimize
from kalmanbox import KalmanFilter, StateSpaceRepresentation
from kalmanbox.datasets import load_nile

y = load_nile()["volume"].to_numpy()

def make_kf(psi: np.ndarray) -> KalmanFilter:
    """Local Level KF; psi = [log σ²_η, log σ²_ε] (unconstrained)."""
    sigma2_eta, sigma2_eps = np.exp(psi)
    ssr = StateSpaceRepresentation(
        T=np.array([[1.0]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma2_eta]]),
        H=np.array([[sigma2_eps]]),
    )
    return KalmanFilter(ssr, initialization="diffuse")

best, rng = None, np.random.default_rng(0)
for _ in range(10):
    psi0 = rng.uniform(-2, 10, size=2)
    res  = minimize(lambda p: -make_kf(p).run(y).loglike, psi0,
                    method="L-BFGS-B", options={"ftol": 1e-12})
    if best is None or res.fun < best.fun:
        best = res

sigma2_eta, sigma2_eps = np.exp(best.x)
print(f"σ²_η = {sigma2_eta:.1f},  σ²_ε = {sigma2_eps:.1f}")
print(f"q    = {sigma2_eta / sigma2_eps:.3f}")
print(f"Log-likelihood = {-best.fun:.4f}")
```

---

## Filtered vs. smoothed level

After fitting, you can extract two estimates of $\mu_t$:

| Estimate | Notation | Uses | When |
|----------|----------|------|------|
| **Filtered** | $\hat\mu_{t\mid t}$ | $y_1, \ldots, y_t$ | Real-time monitoring |
| **Smoothed** | $\hat\mu_{t\mid n}$ | $y_1, \ldots, y_n$ | Retrospective analysis |

The smoothed estimate always has lower variance because it conditions on the
full sample. The two estimates coincide at $t = n$ (the final observation).

```python
from kalmanbox.structural import LocalLevel
from kalmanbox.datasets import load_nile
import matplotlib.pyplot as plt

nile    = load_nile()
y       = nile["volume"].to_numpy()
results = LocalLevel(y).fit()

filt = results.filter()
smot = results.smooth()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(nile["year"], y, "k.", alpha=0.5, label="Nile discharge (observed)")
ax.plot(nile["year"], filt.a_filtered[:, 0], "b-",
        linewidth=1.5, label=r"Filtered level $\hat\mu_{t|t}$")
ax.plot(nile["year"], smot.a_smoothed[:, 0], "r-",
        linewidth=2.0, label=r"Smoothed level $\hat\mu_{t|n}$")

# 95% confidence band on the smoothed level
se_smooth = np.sqrt(smot.V_smoothed[:, 0, 0])
ax.fill_between(nile["year"],
                smot.a_smoothed[:, 0] - 1.96 * se_smooth,
                smot.a_smoothed[:, 0] + 1.96 * se_smooth,
                alpha=0.15, color="red", label="95% CI (smoothed)")

ax.axvline(1899, color="gray", ls="--", label="Aswan Dam (1899)")
ax.set_xlabel("Year"); ax.set_ylabel("Discharge (10⁸ m³/s)")
ax.legend(); ax.set_title("Nile River: Local Level decomposition")
plt.tight_layout()
```

**Expected output:** The smoothed level shows a pronounced step-down around
1899 (Aswan Dam construction), while the filtered level reacts to this
intervention one year at a time, creating a staircase pattern.

---

## Examples

### Example 1: Nile river annual discharge

The Nile river dataset ($n = 100$, 1871–1970) is the canonical benchmark for
the Local Level model (Harvey 1989, Ch. 3). The visible structural break in
1899 makes it ideal for illustrating how the smoother retrospectively
re-estimates the pre-intervention level.

```python
from kalmanbox.structural import LocalLevel
from kalmanbox.datasets import load_nile
from kalmanbox.diagnostics import innovation_diagnostics

nile    = load_nile()
y       = nile["volume"].to_numpy()
results = LocalLevel(y).fit(n_starts=10)

# Model summary
print(results.summary())

# Innovation diagnostics
diag = innovation_diagnostics(results.filter())
print(diag.ljung_box(lags=10))       # H0: no serial correlation
print(diag.jarque_bera())            # H0: normality
print(diag.heteroskedasticity())     # H0: homoskedasticity

# Smoothed level
sm_level = results.smooth().a_smoothed[:, 0]
print(f"Avg level 1871–1898 : {sm_level[:28].mean():.1f}")
print(f"Avg level 1900–1970 : {sm_level[29:].mean():.1f}")
```

**Typical output:**

```
Avg level 1871–1898 : 1097.3
Avg level 1900–1970 :  846.1
```

### Example 2: Real GDP growth (simulated)

A slowly drifting mean growth rate for GDP per capita, where the level shift
corresponds to a structural change in productivity:

```python
import numpy as np
from kalmanbox.structural import LocalLevel

rng = np.random.default_rng(42)
n   = 120    # 30 years of quarterly data

# Simulate: mean growth shifts from 0.8% to 0.3% after period 60
mu_true = np.concatenate([
    np.full(60, 0.8) + np.cumsum(rng.normal(scale=0.05, size=60)),
    np.full(60, 0.3) + np.cumsum(rng.normal(scale=0.05, size=60)),
])
y_gdp = mu_true + rng.normal(scale=0.3, size=n)    # quarterly growth rate (%)

results = LocalLevel(y_gdp).fit(n_starts=5)
sm      = results.smooth()

# q < 1 indicates smooth level changes
q = results.params["sigma2_eta"] / results.params["sigma2_eps"]
print(f"Signal-to-noise q = {q:.3f}")
print(f"MLE σ²_η = {results.params['sigma2_eta']:.4f}")
print(f"MLE σ²_ε = {results.params['sigma2_eps']:.4f}")

# Forecast 8 quarters ahead
fc = results.forecast(steps=8, alpha=0.05)
print(fc[["mean", "lower_95", "upper_95"]].round(3))
```

---

## Forecasting

The Local Level model produces forecasts by propagating the state forward.
Since $T = 1$, the forecast mean is constant for all horizons:

$$
\hat{y}_{n+h \mid n} = \hat\mu_{n \mid n}, \qquad h = 1, 2, \ldots
$$

The forecast variance grows with horizon $h$:

$$
\mathrm{Var}(\hat{y}_{n+h \mid n}) = P_{n+1 \mid n} + (h-1)\,\sigma_\eta^2 + \sigma_\varepsilon^2
$$

```python
results = LocalLevel(y).fit()
fc = results.forecast(steps=10, alpha=0.05)
#   fc.mean        — point forecast (constant for Local Level)
#   fc.lower_95    — lower 95% interval
#   fc.upper_95    — upper 95% interval
```

!!! tip "When to use Local Level for forecasting"

    The Local Level forecast is a sensible baseline when you believe the
    series has no systematic trend. For trending series, use
    [Local Linear Trend](local-linear-trend.md) or
    [BSM](bsm.md) instead.

---

## Diagnostics

After fitting, inspect the **standardized innovations** for model adequacy:

```python
from kalmanbox.diagnostics import innovation_diagnostics, plot_diagnostics

filt = results.filter()
diag = innovation_diagnostics(filt)

print(diag.ljung_box(lags=10))   # auto-correlation test
print(diag.jarque_bera())         # normality test
print(diag.heteroskedasticity())  # variance stability

plot_diagnostics(filt)            # 2×2 panel: residuals, ACF, QQ, periodogram
```

A well-specified Local Level model should produce standardized innovations
$v_t / \sqrt{F_t}$ that are approximately i.i.d. $\mathcal{N}(0,1)$. Serial
correlation in innovations suggests a missing trend or seasonal component.

---

## Connection to ARIMA

It can be shown that the Local Level model is equivalent to an ARIMA(0,1,1)
process. The MA(1) coefficient is:

$$
\theta = \frac{1 - \sqrt{1 + 4q} - q^{-1}}{2}
\quad \in [-1, 0)
$$

This equivalence means that MLE of the Local Level yields the same
log-likelihood as ARIMA(0,1,1) fitted by exact methods. The state-space form
is preferred because it directly delivers the filtered level, exact
diffuse initialization, and exact missing-data handling.

---

## When to use

- A slowly-drifting mean with **no trend** and **no seasonality**.
- As a **baseline** to compare against LLT or BSM.
- As a **building block** inside a larger custom model.
- When you want the simplest possible state-space specification to explain
  to stakeholders.

For a series with a drifting slope, use [Local Linear Trend](local-linear-trend.md).
For seasonal series, go directly to [BSM](bsm.md).

---

## API reference

::: kalmanbox.models.local_level.LocalLevel
    options:
      heading_level: 3
      show_source: false

---

## Related

- [Local Linear Trend](local-linear-trend.md) — adds a stochastic slope
- [BSM](bsm.md) — adds seasonality on top of the Local Linear Trend
- [Kalman Filter](../kalman/kalman-filter.md) — the forward recursion that
  powers `LocalLevel.filter()`
- [RTS Smoother](../kalman/rts-smoother.md) — the backward pass for smoothed
  estimates
- [MLE](../kalman/mle.md) — parameter estimation details
- [Tutorial: Nile with Local Level](../../tutorials/nile-local-level.md)
- [Theory: state-space foundations](../../theory/state-space-theory.md)
- [API: structural models](../../api/models.md)

### References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter.* Cambridge University Press. Ch. 2–3.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press. §2.1–2.3.
- Commandeur, J. J. F. & Koopman, S. J. (2007). *An Introduction to State
  Space Time Series Analysis.* Oxford University Press. Ch. 2.
