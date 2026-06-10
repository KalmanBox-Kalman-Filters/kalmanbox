# Basic Structural Model (BSM)

The **Basic Structural Model** (Harvey 1989) is the standard structural
decomposition for seasonal economic and business time series. It extends the
[Local Linear Trend](local-linear-trend.md) model with a **seasonal component**
and is the workhorse of applied structural time-series analysis. BSM is the
model that `chronobox` and `forecastbox` use internally for their
trend/seasonal pipelines.

---

## Full formulation

A BSM decomposes the observed series into four unobserved components:

$$
y_t = \underbrace{\mu_t}_{\text{trend}} + \underbrace{\gamma_t}_{\text{seasonal}} + \underbrace{\varepsilon_t}_{\text{irregular}}
$$

with the trend itself split into level and slope:

$$
\mu_t = \underbrace{\ell_t}_{\text{level}} + \underbrace{\beta_t \cdot t}_{\text{slope contribution}}
$$

### Trend component

$$
\begin{aligned}
\ell_{t+1} &= \ell_t + \beta_t + \eta_t, &\quad \eta_t &\sim \mathcal{N}(0,\,\sigma_\eta^2) \\
\beta_{t+1} &= \beta_t + \zeta_t,         &\quad \zeta_t &\sim \mathcal{N}(0,\,\sigma_\zeta^2)
\end{aligned}
$$

### Irregular component

$$
\varepsilon_t \sim \mathcal{N}(0,\,\sigma_\varepsilon^2)
$$

### Seasonal component

kalmanbox supports two seasonal specifications. The **dummy** form is the
default for the `BSM` class; the **trigonometric** form is available via
[UCM](ucm.md) and is preferred for long seasonal periods.

---

## Dummy seasonal specification

The dummy seasonal constraint requires that seasonal effects sum to zero across
each complete cycle of length $s$:

$$
\sum_{j=0}^{s-1} \gamma_{t-j} = 0 \quad \text{for all } t
$$

The stochastic version lets this constraint be violated by a small shock each
period:

$$
\gamma_{t+1} = -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t, \qquad
\omega_t \sim \mathcal{N}(0,\,\sigma_\omega^2)
$$

When $\sigma_\omega^2 = 0$ the seasonal is **deterministic** (fixed pattern);
when $\sigma_\omega^2 > 0$ the seasonal **evolves slowly** over time.

The state vector carries the $s-1$ most recent seasonal values:

$$
\alpha_t^{(\text{seas})} = (\gamma_t,\, \gamma_{t-1},\, \ldots,\, \gamma_{t-s+2})'
\in \mathbb{R}^{s-1}
$$

---

## Trigonometric seasonal specification

The trigonometric form expresses seasonality as a sum of $\lfloor s/2 \rfloor$
harmonic pairs:

$$
\gamma_t = \sum_{j=1}^{\lfloor s/2 \rfloor} \gamma_{j,t}
$$

where each harmonic $j$ evolves as:

$$
\begin{bmatrix} \gamma_{j,t+1} \\ \gamma_{j,t+1}^* \end{bmatrix}
=
\begin{bmatrix}
  \cos\lambda_j & \sin\lambda_j \\
  -\sin\lambda_j & \cos\lambda_j
\end{bmatrix}
\begin{bmatrix} \gamma_{j,t} \\ \gamma_{j,t}^* \end{bmatrix}
+
\begin{bmatrix} \kappa_{j,t} \\ \kappa_{j,t}^* \end{bmatrix}
$$

with seasonal frequency $\lambda_j = 2\pi j / s$ and shocks
$\kappa_{j,t},\, \kappa_{j,t}^* \sim \mathcal{N}(0, \sigma_{\kappa,j}^2)$.

| Specification | State vars for seasonal | Requires $s$ params per harmonic | Notes |
|---------------|:-----------------------:|:---------------------------------:|-------|
| Dummy | $s - 1$ | 1 ($\sigma_\omega^2$) | Simple, natural for small $s$ |
| Trigonometric | $2\lfloor s/2 \rfloor$ | 1 per harmonic or shared | Parsimonious for large $s$ |

For monthly data ($s=12$) the dummy form needs 11 state variables and 1
seasonal variance; the trigonometric form with a common variance also needs
1 seasonal variance but 12 state variables (6 pairs). For $s=52$ (weekly),
the trigonometric form with grouped variances is dramatically more parsimonious.

!!! tip "When to use trigonometric seasonality"

    Use the dummy form (default `BSM`) when $s \leq 12$ and the seasonal
    pattern is expected to be stable or only slowly evolving. Switch to the
    trigonometric form via `UCM(seasonal_harmonics=...)` when $s > 12$ or when
    you want to allow different harmonics to evolve at different rates.

---

## State-space representation

The full BSM state vector concatenates level, slope, and seasonal:

$$
\alpha_t = \bigl(\ell_t,\; \beta_t,\; \gamma_t,\; \gamma_{t-1},\; \ldots,\; \gamma_{t-s+2}\bigr)'
\in \mathbb{R}^{s+1}
$$

For monthly data ($s = 12$), $\dim\alpha_t = 13$. The system matrices are:

### Transition matrix $T$ (size $(s+1) \times (s+1)$)

$$
T = \begin{bmatrix}
1 & 1 &  0 &  0 & \cdots &  0 &  0 \\
0 & 1 &  0 &  0 & \cdots &  0 &  0 \\
0 & 0 & -1 & -1 & \cdots & -1 & -1 \\
0 & 0 &  1 &  0 & \cdots &  0 &  0 \\
0 & 0 &  0 &  1 & \cdots &  0 &  0 \\
\vdots & & & & \ddots & & \vdots \\
0 & 0 &  0 &  0 & \cdots &  1 &  0
\end{bmatrix}
$$

The top-left $2\times2$ block is the LLT transition. The seasonal block in
rows 3 onward implements the dummy constraint
$\gamma_{t+1} = -\gamma_t - \gamma_{t-1} - \cdots - \gamma_{t-s+2} + \omega_t$.

### Observation matrix $Z$ (size $1 \times (s+1)$)

$$
Z = \begin{bmatrix} 1 & 0 & 1 & 0 & \cdots & 0 \end{bmatrix}
$$

Picks out the level $\ell_t$ (position 1) and the current seasonal $\gamma_t$
(position 3). The slope and lagged seasonals are not directly observed.

### Selection matrix $R$ (size $(s+1) \times 3$)

$$
R = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0 \\
\vdots & \vdots & \vdots \\
0 & 0 & 0
\end{bmatrix}
$$

Only the three active shocks ($\eta_t$, $\zeta_t$, $\omega_t$) enter the state;
the lagged seasonal slots have no shock.

### State disturbance covariance $Q$ (size $3 \times 3$)

$$
Q = \begin{bmatrix}
\sigma_\eta^2 & 0 & 0 \\
0 & \sigma_\zeta^2 & 0 \\
0 & 0 & \sigma_\omega^2
\end{bmatrix}
$$

### Observation noise $H$ (size $1 \times 1$)

$$
H = \begin{bmatrix} \sigma_\varepsilon^2 \end{bmatrix}
$$

---

## Parameters

| Parameter | Symbol | Interpretation |
|-----------|--------|---------------|
| Level variance | $\sigma_\eta^2$ | How quickly the trend level shifts |
| Slope variance | $\sigma_\zeta^2$ | How quickly the growth rate changes |
| Seasonal variance | $\sigma_\omega^2$ | How quickly the seasonal pattern evolves |
| Irregular variance | $\sigma_\varepsilon^2$ | Short-run unexplained noise |

Setting a variance to zero makes the corresponding component **deterministic**.
The most common restriction is $\sigma_\zeta^2 = 0$ (fixed slope), reducing
to a "smooth-trend + seasonal" model.

---

## Fitting a BSM

### Airline passengers — the canonical example

The Box-Jenkins airline passengers dataset ($n=144$, January 1949 – December
1960) is the standard benchmark for seasonal structural models. The data exhibit
a clear upward trend and strong, growing seasonal swings — motivating a
log-transform before modelling.

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.structural import BSM
from kalmanbox.datasets import load_airline

airline = load_airline()
y_raw   = airline["passengers"].to_numpy()  # monthly, n=144
y_log   = np.log(y_raw)                     # log-transform stabilises variance

# ── Fit BSM with monthly seasonality ─────────────────────────────────────────
model   = BSM(y_log, period=12)
results = model.fit(method="mle", n_starts=10, disp=True)

print(results.summary())
```

```
                Basic Structural Model Results
=============================================================
Dep. Variable:   passengers  Log-Likelihood:   213.041
No. Observations: 144        AIC:             -418.082
Df Model:          4         BIC:             -405.482
                              HQIC:            -412.953
=============================================================
             Estimate    Std.Err    z-stat    p-value
sigma2_eta   1.41e-04   5.8e-05     2.430    0.0151
sigma2_zeta  2.33e-07   1.9e-07     1.228    0.2196
sigma2_omega 3.05e-05   1.2e-05     2.532    0.0113
sigma2_eps   2.12e-04   6.7e-05     3.165    0.0016
=============================================================
```

**Interpreting the output:** The slope variance $\hat\sigma_\zeta^2$ is small
relative to its standard error, suggesting the growth rate is nearly constant
over this 12-year period. The seasonal variance $\hat\sigma_\omega^2 > 0$
indicates the seasonal pattern evolves slowly — consistent with the growing
amplitude visible in the raw data (before log-transform).

---

## Component decomposition

After fitting, the RTS smoother decomposes the log-passengers series into its
structural components:

```python
sm = results.smooth()

trend    = sm.components["level"]      # smoothed ℓ_t + β_t contribution
slope    = sm.components["slope"]      # smoothed β_t
seasonal = sm.components["seasonal"]   # smoothed γ_t
irregular = y_log - trend - seasonal   # residual ε_t

print(f"Trend   range  : [{trend.min():.3f}, {trend.max():.3f}]")
print(f"Seasonal range : [{seasonal.min():.3f}, {seasonal.max():.3f}]")
print(f"Irregular std  : {irregular.std():.4f}")

# Visualise
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

dates = airline.index

axes[0].plot(dates, y_log, "k-", lw=0.8, alpha=0.7, label="log(passengers)")
axes[0].plot(dates, trend, "r-", lw=2, label="Trend")
axes[0].set_title("Observed vs. trend"); axes[0].legend()

axes[1].plot(dates, slope, "b-", lw=1.5)
axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
axes[1].set_title("Slope (monthly growth rate)")

axes[2].plot(dates, seasonal, "g-", lw=1.5)
axes[2].axhline(0, color="gray", ls="--", alpha=0.5)
axes[2].set_title("Seasonal component")

axes[3].plot(dates, irregular, "m.", alpha=0.6)
axes[3].axhline(0, color="gray", ls="--", alpha=0.5)
axes[3].set_title("Irregular")

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
```

**Expected output:** Four panels showing:

1. Log-passengers with a smooth upward trend overlay.
2. Nearly flat slope (≈ 0.01 per month = ~12% per year), slightly increasing.
3. A strong sinusoidal seasonal pattern with July/August peak and
   January/February trough; the pattern is stable because $\hat\sigma_\omega^2$
   is small.
4. Small, mean-zero irregular residuals with no apparent structure.

---

## Forecasting with BSM

The BSM naturally projects trend and seasonal forward simultaneously:

```python
# Forecast 24 months ahead (on log scale)
fc_log = results.forecast(steps=24, alpha=0.05)

# Back-transform to original scale (log-normal mean correction)
fc_mean_raw = np.exp(fc_log.mean + 0.5 * fc_log.variance)
fc_lower     = np.exp(fc_log.lower_95)
fc_upper     = np.exp(fc_log.upper_95)

print(fc_log[["mean", "lower_95", "upper_95"]].head(12))

# Plot forecast
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(airline.index, y_raw, "k-", label="Observed")
fc_dates = pd.date_range(airline.index[-1], periods=25, freq="MS")[1:]
ax.plot(fc_dates, fc_mean_raw, "r-", lw=2, label="Forecast (back-transformed)")
ax.fill_between(fc_dates, fc_lower, fc_upper,
                alpha=0.2, color="red", label="95% PI")
ax.set_title("BSM forecast — airline passengers"); ax.legend()
plt.tight_layout()
```

**Expected output:** The forecast replicates the seasonal pattern and upward
trend seen in the historical data, with the 95% prediction interval widening
modestly at the 2-year horizon.

---

## Example 2: Monthly retail sales

```python
import numpy as np
from kalmanbox.structural import BSM
from kalmanbox.datasets import load_dataset

# Monthly retail sales index, not seasonally adjusted
y_raw   = load_dataset("us_retail_sales")["index"].to_numpy()
y_log   = np.log(y_raw)

model   = BSM(y_log, period=12, stochastic_slope=True, stochastic_seasonal=True)
results = model.fit(n_starts=10, disp=False)

sm = results.smooth()

# The seasonal component captures Black Friday / Christmas effects
seasonal = sm.components["seasonal"]
months   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Average seasonal effect by month (last full year)
avg_seasonal = seasonal[-12:]
for m, s in zip(months, avg_seasonal):
    direction = "↑" if s > 0 else "↓"
    print(f"{m}: {s:+.4f} {direction}")
```

---

## Stochastic vs. deterministic seasonal

Use the `stochastic_seasonal` flag to switch between the two specifications:

```python
from kalmanbox.structural import BSM
import numpy as np

y = ...   # monthly series

# Deterministic seasonal (σ_ω² = 0): faster convergence, fewer parameters
r_det = BSM(y, period=12, stochastic_seasonal=False).fit(disp=False)

# Stochastic seasonal (default): σ_ω² estimated
r_sto = BSM(y, period=12, stochastic_seasonal=True).fit(disp=False)

print(f"Deterministic seasonal AIC = {r_det.aic:.2f}")
print(f"Stochastic seasonal    AIC = {r_sto.aic:.2f}")

# Likelihood-ratio test
lr = 2 * (r_sto.loglike - r_det.loglike)
from scipy import stats
p = 1 - stats.chi2.cdf(lr, df=1)
print(f"LR = {lr:.3f},  p = {p:.4f}")
```

!!! tip "When to use stochastic seasonality"

    Choose stochastic seasonal (`stochastic_seasonal=True`) when:

    - The series spans many years and the seasonal pattern may have shifted
      (e.g., changing consumer habits, new public holidays).
    - Residual diagnostics show seasonal autocorrelation in the deterministic
      case.

    Choose deterministic seasonal when:
    
    - The series is short (few complete cycles).
    - The seasonal pattern appears stable over time.
    - You want fewer parameters for a simple benchmark.

---

## Diagnostics after fitting BSM

```python
from kalmanbox.diagnostics import innovation_diagnostics, seasonal_diagnostics

filt = results.filter()
diag = innovation_diagnostics(filt)

# Standard innovation tests
print(diag.ljung_box(lags=24))        # test up to 2 full seasonal cycles
print(diag.jarque_bera())
print(diag.heteroskedasticity())

# Seasonal-specific diagnostics (test for residual seasonal patterns)
sdiag = seasonal_diagnostics(filt, period=12)
print(sdiag.seasonal_ljung_box())     # LB at seasonal lags 12, 24, ...
print(sdiag.spectral_test())          # Fisher's test for periodicity
```

For a well-specified BSM, the standardized innovations should be i.i.d.
$\mathcal{N}(0,1)$ with no residual autocorrelation at seasonal lags.
A significant Ljung-Box statistic at lag 12 (or 24) suggests the seasonal
component is mis-specified — try switching to the trigonometric form via
[UCM](ucm.md), or check whether additional harmonics are needed.

---

## Comparing BSM with ARIMA

```python
# BSM vs ARIMA(0,1,1)(0,1,1)_12 — the "airline model" — on the airline data
from kalmanbox.structural import BSM
from kalmanbox.arima_ssm import ARIMA
import numpy as np

y_log = np.log(load_airline()["passengers"].to_numpy())

r_bsm   = BSM(y_log, period=12).fit(disp=False)
r_arima = ARIMA(y_log, order=(0,1,1), seasonal_order=(0,1,1,12)).fit(disp=False)

print(f"BSM   AIC = {r_bsm.aic:.2f}")
print(f"ARIMA AIC = {r_arima.aic:.2f}")
```

The BSM is typically comparable to or better than the classical airline ARIMA
model, and additionally delivers an interpretable decomposition.

---

## API reference

::: kalmanbox.models.bsm.BasicStructuralModel
    options:
      heading_level: 3
      show_source: false

---

## Related

- [Local Linear Trend](local-linear-trend.md) — the trend sub-model embedded in BSM
- [UCM](ucm.md) — fully configurable; exposes trigonometric seasonal and cycles
- [MLE](../kalman/mle.md) — parameter estimation details
- [Missing data](../kalman/missing-data.md) — handling gaps in seasonal series
- [Diagnostics](../../diagnostics/residuals.md) — model checking
- [Visualization: component plots](../../visualization/components.md)
- [Tutorial: Airline passengers with BSM](../../tutorials/airline-bsm.md)
- [Theory: structural models](../../theory/structural-models.md)
- [API: structural models](../../api/models.md)

### References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter.* Cambridge University Press. Ch. 2–3.
- Harvey, A. C. & Todd, P. H. J. (1983). Forecasting economic time series
  with structural and Box-Jenkins models: a case study. *Journal of Business &
  Economic Statistics*, 1(4), 299–307.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press. §3.2–3.4.
- Commandeur, J. J. F. & Koopman, S. J. (2007). *An Introduction to State
  Space Time Series Analysis.* Oxford University Press. Ch. 3–4.
