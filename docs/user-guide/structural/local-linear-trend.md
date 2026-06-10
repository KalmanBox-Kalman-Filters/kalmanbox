# Local Linear Trend Model

The **Local Linear Trend** (LLT) model extends the [Local Level](local-level.md)
model by adding a **stochastic slope** $\beta_t$ that can itself drift over
time. The result is a flexible trend component capable of capturing series
whose rate of growth gradually accelerates or decelerates — a pervasive feature
of macroeconomic indicators.

---

## Mathematical formulation

The model has three equations. The **measurement equation** is unchanged from
the Local Level:

$$
y_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\,\sigma_\varepsilon^2)
$$

The **level** evolves as a random walk *with drift* $\beta_t$:

$$
\mu_{t+1} = \mu_t + \beta_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\,\sigma_\eta^2)
$$

The **slope** itself is a random walk:

$$
\beta_{t+1} = \beta_t + \zeta_t, \qquad \zeta_t \sim \mathcal{N}(0,\,\sigma_\zeta^2)
$$

All three disturbances are mutually independent. The model has three free
parameters: $\sigma_\varepsilon^2,\, \sigma_\eta^2,\, \sigma_\zeta^2$.

---

## State-space representation

The state vector is $\alpha_t = (\mu_t, \beta_t)'$. In the general form
$\alpha_{t+1} = T\,\alpha_t + R\,\eta_t$, $y_t = Z\,\alpha_t + \varepsilon_t$:

$$
T = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
Z = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad
R = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

$$
Q = \begin{bmatrix} \sigma_\eta^2 & 0 \\ 0 & \sigma_\zeta^2 \end{bmatrix}, \quad
H = \begin{bmatrix} \sigma_\varepsilon^2 \end{bmatrix}
$$

- $T$ is the companion matrix of a double-integrated process. Both eigenvalues
  equal 1, so the system is non-stationary and requires **diffuse
  initialization**.
- $Z = [1, 0]$ means only the level $\mu_t$ is observed; the slope $\beta_t$
  is entirely latent.
- $R = I_2$ because both state components receive their own shock.

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

sigma_eta = 0.5
sigma_zeta = 0.05
sigma_eps  = 1.0

ssr = StateSpaceRepresentation(
    T=np.array([[1.0, 1.0],
                [0.0, 1.0]]),
    Z=np.array([[1.0, 0.0]]),
    R=np.eye(2),
    Q=np.diag([sigma_eta**2, sigma_zeta**2]),
    H=np.array([[sigma_eps**2]]),
)

kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)

mu_filtered   = out.a_filtered[:, 0]   # filtered level
beta_filtered = out.a_filtered[:, 1]   # filtered slope
```

---

## Special cases

The LLT nests several important sub-models controlled by the variance parameters.

### 1. Deterministic linear trend ($\sigma_\eta^2 = 0$, $\sigma_\zeta^2 = 0$)

Both the level and the slope are fixed constants. The model reduces to:

$$
y_t = \mu + \beta \cdot t + \varepsilon_t
$$

a classical linear regression on time. Estimation returns OLS-equivalent
estimates for $(\mu, \beta)$.

```python
from kalmanbox.structural import LocalLinearTrend

# Fix both innovation variances at zero to enforce deterministic trend
results = LocalLinearTrend(y, stochastic_level=False, stochastic_slope=False).fit()
```

### 2. Fixed slope, stochastic level ($\sigma_\zeta^2 = 0$, $\sigma_\eta^2 > 0$)

The slope is deterministic but the level is allowed to deviate from the trend
line by a random walk. This is the **random walk with fixed drift** model:

$$
\mu_{t+1} = \mu_t + \beta + \eta_t, \qquad y_t = \mu_t + \varepsilon_t
$$

Useful when you believe the long-run growth rate is stable but short-run
fluctuations exist.

```python
results = LocalLinearTrend(y, stochastic_slope=False).fit()
```

### 3. Stochastic slope, smooth level ($\sigma_\eta^2 = 0$, $\sigma_\zeta^2 > 0$)

The level is forced to lie exactly on the trend line (no level shock), but the
slope can drift. This is the **integrated random walk** (IRW) or
**smoothing spline** model:

$$
\mu_{t+1} = \mu_t + \beta_t, \quad
\beta_{t+1} = \beta_t + \zeta_t
$$

The IRW trend is very smooth — it interpolates through the data like a cubic
spline with the penalty controlled by $\sigma_\zeta^2 / \sigma_\varepsilon^2$.

```python
results = LocalLinearTrend(y, stochastic_level=False).fit()
```

### 4. Full stochastic LLT ($\sigma_\eta^2 > 0$, $\sigma_\zeta^2 > 0$)

The default: both level and slope are stochastic. Suitable when both the
position and direction of the trend change gradually.

```python
results = LocalLinearTrend(y).fit()   # both stochastic by default
```

---

## Special case comparison table

| Variant | $\sigma_\eta^2$ | $\sigma_\zeta^2$ | Trend shape | Best for |
|---------|:---------------:|:-----------------:|-------------|----------|
| Deterministic linear | 0 | 0 | Exact straight line | Pure regression baseline |
| Random walk with drift | $>0$ | 0 | Noisy line with fixed slope | Stable growth rate |
| Integrated random walk | 0 | $>0$ | Smooth spline | Filtering/interpolation |
| Full LLT | $>0$ | $>0$ | Slowly drifting trend | Macro series with structural shifts |

---

## Parameter estimation (MLE)

### High-level API

```python
from kalmanbox.structural import LocalLinearTrend
from kalmanbox.datasets import load_dataset

y = load_dataset("us_gdp_growth")["growth"].to_numpy()   # quarterly, 1960–2023

results = LocalLinearTrend(y).fit(method="mle", n_starts=10, disp=True)
print(results.summary())
```

```
             Local Linear Trend Model Results
=============================================================
Dep. Variable:   growth     Log-Likelihood:  -189.422
No. Observations: 252       AIC:              384.844
Df Model:          3        BIC:              395.661
                             HQIC:             389.194
=============================================================
             Estimate   Std.Err    z-stat    p-value
sigma2_eta    0.0412    0.0183     2.251    0.0244
sigma2_zeta   0.0018    0.0009     2.007    0.0448
sigma2_eps    0.2635    0.0381     6.916    0.0000
=============================================================
```

### Interpreting estimates

- A small $\hat\sigma_\zeta^2$ relative to $\hat\sigma_\eta^2$ means the slope
  changes very slowly — the trend direction is stable.
- Both variances near zero indicate the trend is nearly deterministic.
- A large $\hat\sigma_\eta^2$ relative to $\hat\sigma_\varepsilon^2$ (high
  signal-to-noise) means the level tracks the data closely.

---

## Filtered and smoothed components

```python
from kalmanbox.structural import LocalLinearTrend
import matplotlib.pyplot as plt
import numpy as np

y       = ...   # your series
results = LocalLinearTrend(y).fit()

sm = results.smooth()
mu_sm   = sm.a_smoothed[:, 0]    # smoothed level
beta_sm = sm.a_smoothed[:, 1]    # smoothed slope
V_mu    = sm.V_smoothed[:, 0, 0] # variance of smoothed level
V_beta  = sm.V_smoothed[:, 1, 1] # variance of smoothed slope

t = np.arange(len(y))

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(t, y, "k.", alpha=0.4, label="Observed")
axes[0].plot(t, mu_sm, "r-", lw=2, label="Smoothed level")
axes[0].fill_between(t,
    mu_sm - 1.96 * np.sqrt(V_mu),
    mu_sm + 1.96 * np.sqrt(V_mu),
    alpha=0.15, color="red")
axes[0].set_title("Level $\\hat{\\mu}_{t|n}$"); axes[0].legend()

axes[1].plot(t, beta_sm, "b-", lw=2)
axes[1].fill_between(t,
    beta_sm - 1.96 * np.sqrt(V_beta),
    beta_sm + 1.96 * np.sqrt(V_beta),
    alpha=0.15, color="blue")
axes[1].axhline(0, color="k", ls="--", alpha=0.4)
axes[1].set_title("Slope $\\hat{\\beta}_{t|n}$")

axes[2].plot(t, sm.residuals, "g.", alpha=0.6)
axes[2].axhline(0, color="k", ls="--", alpha=0.4)
axes[2].set_title("Smoothed irregular $\\hat{\\varepsilon}_{t|n}$")

plt.tight_layout()
```

**Expected output:** Three panels showing (1) the data with a smooth trend
overlay and confidence band, (2) the time-varying growth rate (slope) with its
uncertainty, and (3) the irregular component.

---

## Examples

### Example 1: US GDP per capita (simulated macro data)

```python
import numpy as np
from kalmanbox.structural import LocalLinearTrend

rng = np.random.default_rng(7)
n   = 240   # 20 years of monthly data

# Simulate log GDP with a slowly decelerating growth rate
beta_true = np.zeros(n + 1)
mu_true   = np.zeros(n + 1)
beta_true[0] = 0.25   # initial monthly growth ≈ 0.25%

for t in range(n):
    beta_true[t+1] = beta_true[t] + rng.normal(scale=0.005)
    mu_true[t+1]   = mu_true[t] + beta_true[t] + rng.normal(scale=0.1)

y_gdp = mu_true[1:] + rng.normal(scale=0.3, size=n)

results = LocalLinearTrend(y_gdp).fit(n_starts=8)
sm      = results.smooth()

print(results.summary())
print(f"Initial smoothed slope : {sm.a_smoothed[0, 1]:.4f}")
print(f"Final smoothed slope   : {sm.a_smoothed[-1, 1]:.4f}")
print(f"True initial slope     : {beta_true[1]:.4f}")
print(f"True final slope       : {beta_true[-1]:.4f}")
```

### Example 2: Unemployment rate

```python
import numpy as np
from kalmanbox.structural import LocalLinearTrend
from kalmanbox.datasets import load_dataset

# Monthly US unemployment rate (seasonally adjusted)
y = load_dataset("us_unemployment")["rate"].to_numpy()

# Compare LLT against local level using AIC
from kalmanbox.structural import LocalLevel

r_ll  = LocalLevel(y).fit(disp=False)
r_llt = LocalLinearTrend(y).fit(disp=False)

print(f"Local Level  AIC = {r_ll.aic:.2f}")
print(f"LLT          AIC = {r_llt.aic:.2f}")

# The LLT almost always wins for unemployment: the trend direction shifts
# significantly across business cycles.

# Extract the trend and slope
sm   = r_llt.smooth()
trend = sm.components["level"]
slope = sm.components["slope"]
```

### Example 3: Log-linearizing a nonlinear trend

Many economic series exhibit exponential growth best modelled on a log scale:

```python
import numpy as np
from kalmanbox.structural import LocalLinearTrend

y_raw  = load_dataset("world_co2")["ppm"].to_numpy()
y_log  = np.log(y_raw)   # log transform; LLT on log scale = geometric trend

results = LocalLinearTrend(y_log).fit(n_starts=10)
sm      = results.smooth()

# Back-transform to original scale
trend_ppm = np.exp(sm.a_smoothed[:, 0])
slope_pct = sm.a_smoothed[:, 1] * 100   # slope in % per period

print(f"Current annual growth rate: {slope_pct[-1]:.3f}%")
```

---

## Forecasting

For an $h$-step-ahead forecast, the level and slope propagate as:

$$
\begin{aligned}
\hat\mu_{n+h \mid n} &= \hat\mu_{n \mid n} + h\,\hat\beta_{n \mid n} \\
\hat\beta_{n+h \mid n} &= \hat\beta_{n \mid n}
\end{aligned}
$$

The forecast is a **straight line** extrapolation of the current level and
slope, but the uncertainty grows quadratically:

$$
\mathrm{Var}(\hat{y}_{n+h \mid n}) \approx
  P_{\mu,n+1|n} + h^2\, P_{\beta,n+1|n} + \sigma_\varepsilon^2
$$

```python
results = LocalLinearTrend(y).fit()
fc = results.forecast(steps=24, alpha=0.05)

# fc.mean          — linear extrapolation of current trend
# fc.lower_95      — lower 95% prediction interval
# fc.upper_95      — upper 95% prediction interval
```

!!! warning "Long-horizon extrapolation"

    Because the LLT extrapolates the current slope indefinitely, forecast
    intervals widen rapidly at long horizons. For multi-year forecasts, consider
    constraining $\sigma_\zeta^2 = 0$ (fixed slope) or using a damped-trend
    variant via [UCM](ucm.md) with a cycle component.

---

## Model selection: LLT vs. Local Level

Use the likelihood-ratio test or information criteria to decide:

```python
from kalmanbox.structural import LocalLevel, LocalLinearTrend
import scipy.stats as stats

r_ll  = LocalLevel(y).fit(disp=False)
r_llt = LocalLinearTrend(y).fit(disp=False)

# AIC comparison
print(f"Local Level  AIC = {r_ll.aic:.2f}")
print(f"LLT          AIC = {r_llt.aic:.2f}")

# Likelihood-ratio test (Local Level nested in LLT with σ_ζ²=0)
lr_stat = 2 * (r_llt.loglike - r_ll.loglike)
p_val   = 1 - stats.chi2.cdf(lr_stat, df=1)   # 1 extra parameter
print(f"LR stat = {lr_stat:.3f},  p-value = {p_val:.4f}")
```

!!! note "Boundary issue"

    The LRT p-value is only approximate here because the null hypothesis
    $\sigma_\zeta^2 = 0$ is on the boundary of the parameter space. The true
    distribution under $H_0$ is a mixture of $\chi^2(0)$ and $\chi^2(1)$,
    so the tabulated p-value is conservative (too large). This is acceptable
    for model selection purposes.

---

## API reference

::: kalmanbox.models.local_linear_trend.LocalLinearTrend
    options:
      heading_level: 3
      show_source: false

---

## Related

- [Local Level](local-level.md) — simpler model without slope
- [BSM](bsm.md) — adds a seasonal component to LLT
- [UCM](ucm.md) — fully configurable, includes damped trend and cycle
- [MLE](../kalman/mle.md) — parameter estimation
- [Kalman Filter](../kalman/kalman-filter.md) — forward recursion
- [RTS Smoother](../kalman/rts-smoother.md) — backward pass
- [Theory: structural models](../../theory/structural-models.md)

### References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter.* Cambridge University Press. §2.3–2.4.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). §3.2.
- Watson, M. W. (1986). Univariate detrending methods with stochastic trends.
  *Journal of Monetary Economics*, 18(1), 49–75.
