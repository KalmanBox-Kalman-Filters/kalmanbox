# Unobserved Components Model (UCM)

The **Unobserved Components Model** (UCM) is the most general structural time-series
model in kalmanbox. Where [BSM](bsm.md) fixes the decomposition to level + slope +
dummy-seasonal + irregular, UCM lets you **opt components in and out** — you decide
which components are present, whether they are stochastic or deterministic, and what
seasonal form to use. BSM is a strict special case of UCM.

UCM is the model that `chronobox` uses when the automatic model-selection step picks
a seasonal period larger than 12 or requests a cycle component.

---

## General UCM formulation

Every UCM is a linear Gaussian state-space model:

$$
\begin{aligned}
y_t &= Z \alpha_t + \varepsilon_t, &\quad \varepsilon_t &\sim \mathcal{N}(0, H) \\
\alpha_{t+1} &= T \alpha_t + R \eta_t, &\quad \eta_t &\sim \mathcal{N}(0, Q)
\end{aligned}
$$

The state vector $\alpha_t$ is the **concatenation of the active component sub-states**:

$$
\alpha_t = \bigl[\alpha_t^{(\ell)},\; \alpha_t^{(\text{seas})},\; \alpha_t^{(\psi)},\; \alpha_t^{(\text{ar})}\bigr]'
$$

The system matrices $T$, $Z$, $R$, $Q$ are block-diagonal (or block-structured) with
one block per component. kalmanbox assembles these blocks automatically at construction
time based on the flags you supply.

---

## Available components

| Component | Parameter | State dim | Free variances | Notes |
|-----------|-----------|:---------:|:--------------:|-------|
| Level | `level=True` | 1 | 1 ($\sigma_\eta^2$) | Random-walk level $\ell_t$ |
| Slope | `slope=True` | +1 = 2 | +1 ($\sigma_\zeta^2$) | Stochastic slope $\beta_t$; requires `level` |
| Seasonal (dummy) | `seasonal=s` | $s-1$ | 1 ($\sigma_\omega^2$) | Default for $s \le 12$ |
| Seasonal (trig) | `seasonal_harmonics=k` | $2k$ | 1 or $k$ | Parsimonious for large $s$ |
| Cycle | `cycle=True` | 2 | 1 ($\sigma_\kappa^2$) + 2 hyper ($\rho, \lambda$) | Stochastic cycle |
| AR($p$) | `ar=p` | $p$ | 1 ($\sigma_\xi^2$) + $p$ ($\phi_i$) | Stationary AR |
| Irregular | `irregular=True` | 0 | 1 ($\sigma_\varepsilon^2$) | Observation noise |
| Exogenous | `exog=X` | 0 | 0 | Regressed out in obs. eq. |

### Special cases

The following UCM configurations recover the simpler named models exactly:

| UCM configuration | Equivalent named model |
|-------------------|------------------------|
| `level=True, irregular=True` | `LocalLevel` |
| `level=True, slope=True, irregular=True` | `LocalLinearTrend` |
| `level=True, slope=True, seasonal=s, irregular=True` | `BSM(period=s)` |

---

## Component specifications

### Level and slope

Level and slope follow the **Local Linear Trend** dynamics:

$$
\begin{aligned}
\ell_{t+1} &= \ell_t + \beta_t + \eta_t^\ell, &\quad \eta_t^\ell &\sim \mathcal{N}(0, \sigma_\eta^2) \\
\beta_{t+1} &= \beta_t + \eta_t^\beta, &\quad \eta_t^\beta &\sim \mathcal{N}(0, \sigma_\zeta^2)
\end{aligned}
$$

Setting $\sigma_\zeta^2 = 0$ fixes the slope (integrated random walk level).
Setting both $\sigma_\eta^2 = 0$ and $\sigma_\zeta^2 = 0$ gives a deterministic linear trend.

### Trigonometric seasonal

UCM always uses the **trigonometric** seasonal form (unlike `BSM`, which defaults to dummy):

$$
\gamma_t = \sum_{j=1}^{k} \gamma_{j,t}
$$

where the $j$-th harmonic pair evolves as:

$$
\begin{bmatrix} \gamma_{j,t+1} \\ \gamma_{j,t+1}^* \end{bmatrix}
=
\begin{bmatrix} \cos\lambda_j & \sin\lambda_j \\ -\sin\lambda_j & \cos\lambda_j \end{bmatrix}
\begin{bmatrix} \gamma_{j,t} \\ \gamma_{j,t}^* \end{bmatrix}
+
\begin{bmatrix} \omega_{j,t} \\ \omega_{j,t}^* \end{bmatrix},
\qquad
\omega_{j,t}, \omega_{j,t}^* \sim \mathcal{N}(0, \sigma_{\omega,j}^2)
$$

with seasonal frequency $\lambda_j = 2\pi j / s$. By default all harmonics share
one variance ($\sigma_{\omega,1}^2 = \cdots = \sigma_{\omega,k}^2 \equiv \sigma_\omega^2$).

The number of harmonics defaults to $k = \lfloor s/2 \rfloor$ (the maximum that preserves
identification). Reducing $k$ via `seasonal_harmonics` produces a parsimonious seasonal.

### Cycle

See the dedicated [Cycle component](cycle.md) page for the full specification.
The cycle state pair $(\psi_t, \psi_t^*)$ is appended to $\alpha_t$; only $\psi_t$
enters the observation equation via $Z$.

### AR($p$) component

The AR($p$) component adds a stationary autoregressive process as a structural block:

$$
\xi_t = \phi_1 \xi_{t-1} + \cdots + \phi_p \xi_{t-p} + \eta_t^\xi,
\qquad \eta_t^\xi \sim \mathcal{N}(0, \sigma_\xi^2)
$$

In state-space form this is the standard companion representation (see [ARIMA-SSM](arima-ssm.md)).
It is useful for modelling residual autocorrelation that is too regular for the
irregular component but too stationary for a cycle.

### Exogenous regressors

Regressors enter the **observation equation** without being part of the state:

$$
y_t = Z \alpha_t + x_t' \delta + \varepsilon_t
$$

where $x_t \in \mathbb{R}^k$ and $\delta \in \mathbb{R}^k$ are estimated coefficients.
This is equivalent to adding deterministic regressors after the structural decomposition.
For time-varying coefficients, use the [TVP](../advanced/tvp.md) model instead.

---

## UCM vs BSM: when to use which

| Situation | Recommendation |
|-----------|----------------|
| Monthly / quarterly series, stable seasonal | **BSM** — simpler, fewer params |
| Seasonal period $s > 12$ (weekly, daily) | **UCM** with trigonometric seasonal |
| Series with a medium-frequency cycle | **UCM** with `cycle=True` |
| Want to suppress slope entirely | **UCM** with `slope=False` |
| Need AR residuals instead of white-noise irregular | **UCM** with `ar=p` |
| Regressors with structural decomposition | **UCM** with `exog=X` |
| Full compositional flexibility | **UCM** |

!!! warning "Model identifiability"

    Not all component combinations are identified. In particular:

    - A **cycle and a seasonal** at the same frequency are not jointly identified.
    - A **slope and an AR(1) with unit root** are close to collinear.
    - Always inspect the Hessian at the MLE and check
      [identifiability diagnostics](../../theory/identifiability.md).

---

## Fitting a UCM

### Example 1: trend + cycle decomposition

GDP growth often exhibits a long-run trend plus a business cycle of 6–10 years.

```python
import numpy as np
from kalmanbox.structural import UCM
from kalmanbox.datasets import load_gdp

gdp = load_gdp()
y   = np.log(gdp["gdp"].to_numpy())          # log real GDP

# Trend (level + slope) + business cycle + irregular
model = UCM(
    y,
    level=True,
    slope=True,
    cycle=True,
    cycle_period_bounds=(24, 120),            # cycle period between 2 and 10 years
    irregular=True,
)
results = model.fit(method="mle", n_starts=15, disp=True)

print(results.summary())
```

```
          Unobserved Components Model Results
==============================================================
Dep. Variable:   log_gdp   Log-Likelihood:   384.219
No. Observations: 240      AIC:             -758.437
Df Model:          5        BIC:             -741.112
==============================================================
             Estimate    Std.Err    z-stat    p-value
sigma2_eta   3.11e-06   1.4e-06     2.222    0.0263
sigma2_zeta  8.02e-08   4.9e-08     1.636    0.1020
sigma2_kappa 4.91e-05   1.7e-05     2.888    0.0039
rho_cycle    0.9712      0.0124    78.323    0.0000
lambda_cycle 0.0621      0.0041    15.146    0.0000   (period ≈ 101 months)
sigma2_eps   1.88e-05   5.3e-06     3.547    0.0004
==============================================================
```

!!! note "Interpreting the cycle parameters"
    - `rho_cycle = 0.97` — the cycle is highly persistent (near-unit damping).
    - `lambda_cycle = 0.062 rad/month` → period ≈ $2\pi / 0.062 \approx 101$ months
      ≈ 8.4 years, consistent with a standard business cycle.

### Decompose components

```python
sm = results.smooth()

trend  = sm.components["level"]       # smoothed ℓ_t
slope  = sm.components["slope"]       # smoothed β_t
cycle  = sm.components["cycle"]       # smoothed ψ_t
irreg  = sm.components["irregular"]   # ε_t

import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(gdp.index, y, "k-", lw=0.8, alpha=0.7, label="log GDP")
axes[0].plot(gdp.index, trend, "r-", lw=2,  label="Trend")
axes[0].legend(); axes[0].set_title("Observed vs. trend")

axes[1].plot(gdp.index, slope, "b-")
axes[1].axhline(0, color="grey", ls="--", alpha=0.4)
axes[1].set_title("Slope (quarterly growth rate)")

axes[2].plot(gdp.index, cycle, "g-")
axes[2].axhline(0, color="grey", ls="--", alpha=0.4)
axes[2].set_title("Business cycle")

axes[3].plot(gdp.index, irreg, "m.", alpha=0.5)
axes[3].axhline(0, color="grey", ls="--", alpha=0.4)
axes[3].set_title("Irregular")

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
```

---

### Example 2: seasonal decomposition with exogenous regressor

Monthly electricity consumption is driven by temperature in addition to the
structural components.

```python
import numpy as np
from kalmanbox.structural import UCM
from kalmanbox.datasets import load_electricity

data   = load_electricity()
y      = data["consumption"].to_numpy()   # MWh, monthly
temp   = data["temperature"].to_numpy()   # mean monthly temperature (°C)
exog   = temp.reshape(-1, 1)

model = UCM(
    y,
    level=True,
    slope=True,
    seasonal=12,                          # trigonometric monthly seasonal
    seasonal_harmonics=6,                 # all 6 harmonics
    irregular=True,
    exog=exog,                            # temperature regressor
)
results = model.fit(n_starts=10, disp=False)

print(f"Temperature coefficient: {results.params['delta_0']:.4f} MWh/°C")
print(f"AIC: {results.aic:.2f}")
```

The regression coefficient `delta_0` measures the average effect of a 1°C temperature
increase on monthly consumption, **after** removing the structural trend and seasonal.

---

### Example 3: trend + seasonal + cycle (full decomposition)

Some macroeconomic series — retail sales, industrial production — exhibit both a
seasonal pattern and a medium-frequency cycle superimposed on the trend.

```python
import numpy as np
from kalmanbox.structural import UCM
from kalmanbox.datasets import load_dataset

y = load_dataset("industrial_production")["index"].to_numpy()

model = UCM(
    y,
    level=True,
    slope=True,
    seasonal=12,                          # monthly seasonal
    cycle=True,
    cycle_period_bounds=(18, 84),         # cycle 1.5 – 7 years
    irregular=True,
)
results = model.fit(n_starts=20, disp=True)

sm    = results.smooth()
cycle = sm.components["cycle"]
seas  = sm.components["seasonal"]
trend = sm.components["level"]

# Seasonally adjusted series
sa = y - seas

# Cycle amplitude over time
import numpy as np
cycle_amp = np.abs(cycle)
print(f"Mean cycle amplitude: {cycle_amp.mean():.4f}")
```

!!! tip "Seasonally adjusted series"

    `y - sm.components['seasonal']` is the **structural seasonal adjustment**.
    This is the same principle used by X-13ARIMA-SEATS, but cast in a
    fully-probabilistic state-space framework.

---

## Forecasting

```python
# 24-step ahead forecast with 80% and 95% prediction intervals
fc = results.forecast(steps=24, alpha=[0.05, 0.20])

print(fc[["mean", "lower_80", "upper_80", "lower_95", "upper_95"]].head(6))
```

Forecasts from UCM automatically propagate all component uncertainties. The trend
component widens prediction intervals with horizon; the seasonal and cycle
contributions remain bounded.

---

## Model selection within UCM

kalmanbox provides an automatic search over UCM component combinations ranked by AIC:

```python
from kalmanbox.structural import ucm_search

best = ucm_search(
    y,
    components=["level", "slope", "seasonal", "cycle", "irregular"],
    seasonal_periods=[12],
    n_starts=10,
    criterion="aic",
)

print(best.summary_table())
# Prints ranked table of configurations
```

---

## Diagnostics

After fitting a UCM with cycle or AR components, check the innovation diagnostics as
for any structural model:

```python
from kalmanbox.diagnostics import innovation_diagnostics

diag = innovation_diagnostics(results.filter())
print(diag.ljung_box(lags=24))
print(diag.jarque_bera())
```

A well-specified UCM should have white-noise innovations. Residual autocorrelation
at seasonal lags suggests a seasonal misspecification; autocorrelation at intermediate
lags (5–20) suggests a missing or mis-specified cycle.

---

## State-space representation (compact form)

For a UCM with level, slope, trigonometric seasonal ($k$ harmonics), and cycle,
the state vector is:

$$
\alpha_t = \underbrace{(\ell_t,\; \beta_t)}_{\text{trend}} \oplus
           \underbrace{(\gamma_{1,t}, \gamma_{1,t}^*, \ldots, \gamma_{k,t}, \gamma_{k,t}^*)}_{\text{seasonal}} \oplus
           \underbrace{(\psi_t, \psi_t^*)}_{\text{cycle}}
$$

$$
\dim(\alpha_t) = 2 + 2k + 2
$$

The system matrices are block-diagonal:

$$
T = T^{(\ell\beta)} \oplus T^{(\text{seas})} \oplus T^{(\psi)},
\quad
Q = Q^{(\ell\beta)} \oplus Q^{(\text{seas})} \oplus Q^{(\psi)}
$$

with blocks as defined in [BSM](bsm.md#state-space-representation) for the trend/seasonal
and in [Cycle](cycle.md#state-space-representation) for the cycle.

---

## API reference

::: kalmanbox.models.ucm.UnobservedComponents
    options:
      heading_level: 3
      show_source: false

---

## Related

- [BSM](bsm.md) — simplified special case; uses dummy seasonal
- [Cycle component](cycle.md) — detailed cycle specification and standalone usage
- [ARIMA-SSM](arima-ssm.md) — ARIMA as a special UCM with no structural components
- [Local Linear Trend](local-linear-trend.md) — trend sub-model used by UCM
- [TVP](../advanced/tvp.md) — time-varying regression coefficients
- [MLE](../kalman/mle.md) — parameter estimation
- [Missing data](../kalman/missing-data.md) — UCM handles gaps natively
- [Choosing a model](../../getting-started/choosing-model.md)
- [Theory: structural models](../../theory/structural-models.md)
- [API: structural models](../../api/models.md)

### References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter.* Cambridge University Press. Ch. 2–3.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press. §3.2–3.6.
- Harvey, A. C. & Shephard, N. (1993). Structural time series models. In
  *Handbook of Statistics*, Vol. 11, pp. 261–301.
