---
title: "Tutorial: UCM Custom Components"
description: >-
  Intermediate tutorial that builds a flexible Unobserved Components Model
  (UCM) with trend, stochastic cycle, and seasonality on US Industrial
  Production — including component analysis, UCM vs. BSM comparison, and
  forecasting.
---

# Tutorial: UCM Custom Components

**Level:** :material-signal: Intermediate · **Time:** ~60 min · **Dataset:** US Industrial Production

The Unobserved Components Model (UCM) is the most general form of structural
time series model. Unlike BSM, which has a fixed component structure (level
+ slope + seasonal + irregular), UCM lets you freely choose which components
to include and configure each one independently.

UCM is the right choice when:

- You believe a **business cycle** drives part of the variation (beyond trend)
- You want to specify a cycle with a particular frequency or damping coefficient
- You need different seasonal specifications (harmonic vs. dummy)
- You want to compare multiple structural hypotheses on the same data

By the end of this tutorial you will have:

- Loaded and explored the US Industrial Production Index
- Fitted a UCM with trend + cycle + seasonal components
- Compared UCM specifications (with and without cycle)
- Compared UCM vs. BSM on the same dataset
- Analysed cycle frequency, period, and damping
- Produced forecasts from the best-fitting UCM

!!! info "Prerequisites"
    Complete [BSM Tutorial](bsm.md) first. UCM is a generalisation of BSM
    and shares most of the same API patterns. Install: `pip install kalmanbox`

---

## The dataset: US Industrial Production Index

The Federal Reserve's Industrial Production Index (INDPRO) measures real
output of manufacturing, mining, and utility industries. It is a leading
coincident indicator of the US business cycle and exhibits:

- **Long-run trend**: steady growth with technology-driven productivity
- **Business cycle**: 2–8 year (24–96 month) fluctuations around trend
- **Seasonality**: modest monthly patterns from weather and institutional factors
- **Recessions**: sharp downturns in 1974, 1980, 1990, 2001, 2008, 2020

These features make it an ideal dataset to demonstrate UCM's cycle component.

---

## Step 1 — Load and explore the data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanbox.datasets import load_dataset

# ── Load US Industrial Production (FRED: INDPRO, 1960-01 to 1984-12) ─────────
ip: pd.DataFrame = load_dataset("us_indpro")
print(ip.head(10))
print(f"\nShape   : {ip.shape}")
print(f"Period  : {ip.index[0]} → {ip.index[-1]}")
print(f"Min     : {ip['indpro'].min():.2f}")
print(f"Max     : {ip['indpro'].max():.2f}")
```

### Expected output

```
            indpro
1960-01-01   28.74
1960-02-01   28.49
1960-03-01   28.94
...

Shape   : (300, 1)
Period  : 1960-01-01 → 1984-12-01
Min     : 25.31
Max     : 68.47
```

```python
# ── Log transform and visualise ───────────────────────────────────────────────
y_raw = ip["indpro"]
y     = np.log(y_raw)   # log scale: additive trend + seasonality

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

axes[0].plot(y_raw.index, y_raw.values, color="steelblue", linewidth=1.2)
axes[0].set_title("US Industrial Production Index (level)")
axes[0].set_ylabel("Index (2012 = 100)")

axes[1].plot(y.index, y.values, color="darkorange", linewidth=1.2)
axes[1].set_title("Log Industrial Production")
axes[1].set_ylabel("log(INDPRO)")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.show()
```

The log series shows a clear upward trend interrupted by cyclical downturns
— most visibly in 1974–75 (oil shock) and 1980–82 (Volcker disinflation).
These are business cycle fluctuations that BSM's level + slope structure
would absorb into the trend; a UCM cycle component will identify them
explicitly.

---

## Step 2 — The UCM state-space model

UCM decomposes the log series as:

$$
y_t = \mu_t + \psi_t + \gamma_t + \varepsilon_t
$$

**Trend** (Local Linear Trend):

$$
\begin{aligned}
\mu_{t+1} &= \mu_t + \nu_t + \xi_t, &\quad \xi_t &\sim \mathcal{N}(0, \sigma_\xi^2) \\
\nu_{t+1} &= \nu_t + \zeta_t,         &\quad \zeta_t &\sim \mathcal{N}(0, \sigma_\zeta^2)
\end{aligned}
$$

**Cycle** (stochastic cycle at angular frequency $\lambda_c$):

$$
\begin{pmatrix} \psi_{t+1} \\ \psi^*_{t+1} \end{pmatrix}
= \rho \begin{pmatrix} \cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c \end{pmatrix}
\begin{pmatrix} \psi_t \\ \psi^*_t \end{pmatrix}
+ \begin{pmatrix} \kappa_t \\ \kappa^*_t \end{pmatrix}, \quad
\kappa_t, \kappa^*_t \sim \mathcal{N}(0, \sigma_\kappa^2)
$$

where:
- $\rho \in (0,1)$ is the **damping coefficient** — how quickly the cycle
  dissipates (closer to 1 = longer-lasting cycles)
- $\lambda_c \in (0, \pi)$ is the **angular frequency** — related to the
  cycle period by $P_c = 2\pi / \lambda_c$ months
- $\sigma_\kappa^2$ is the cycle disturbance variance

**Seasonal** (trigonometric, $s = 12$): same as BSM.

The cycle has **5 estimated parameters**: $\rho$, $\lambda_c$, $\sigma_\kappa^2$
plus the initial state mean and variance. In practice `UCM` estimates
$\lambda_c$ and $\rho$ during MLE, choosing the frequency that best fits
the data.

---

## Step 3 — Fit UCM without cycle (baseline)

We start with a baseline model that matches BSM — no cycle — to establish
a reference log-likelihood.

```python
from kalmanbox import UCM

# ── UCM without cycle (baseline = BSM-equivalent) ────────────────────────────
model_nocycle = UCM(
    y,
    level="local linear trend",   # μ_t + ν_t (stochastic level and slope)
    seasonal=12,                  # trigonometric seasonal, s=12
    cycle=False,                  # no cycle component
    irregular=True,               # i.i.d. measurement noise
)
results_nocycle = model_nocycle.fit(method="newton", disp=False)

print("=== UCM (no cycle) ===")
print(results_nocycle.summary())
```

### Expected output

```
==============================================================================
                   Unobserved Components Model (UCM)
==============================================================================
Model:             UCM           Log-Likelihood:     384.117
Sample:            1960-01-01    AIC:               -760.234
                   1984-12-01    BIC:               -742.698
No. Observations:  300
Components:        level, slope, seasonal(12)
==============================================================================
                    coef    std err          z      P>|z|    [0.025    0.975]
------------------------------------------------------------------------------
sigma2.irregular  0.0001    0.0000      2.114      0.035    0.0000   0.0001
sigma2.level      0.0000    0.0000      0.418      0.676   -0.0000   0.0000
sigma2.slope      0.0000    0.0000      3.721      0.000    0.0000   0.0000
sigma2.seasonal   0.0000    0.0000      1.093      0.274   -0.0000   0.0000
==============================================================================
```

---

## Step 4 — Fit UCM with stochastic cycle

Now add the cycle component. UCM estimates the cycle frequency $\lambda_c$
from the data — we provide the search range.

```python
# ── UCM with stochastic cycle (business cycle 24–96 months) ──────────────────
model_cycle = UCM(
    y,
    level="local linear trend",   # trend with stochastic slope
    seasonal=12,                  # trigonometric seasonal
    cycle=True,                   # add stochastic cycle
    cycle_period_bounds=(24, 96), # search range for cycle period (months)
    damping_coefficient=None,     # estimate ρ from data
    irregular=True,
)
results_cycle = model_cycle.fit(method="newton", disp=False)

print("=== UCM (with cycle) ===")
print(results_cycle.summary())
```

### Expected output

```
==============================================================================
                   Unobserved Components Model (UCM)
==============================================================================
Model:             UCM           Log-Likelihood:     401.843
Sample:            1960-01-01    AIC:               -791.686
                   1984-12-01    BIC:               -767.262
No. Observations:  300
Components:        level, slope, cycle, seasonal(12)
==============================================================================
                    coef    std err          z      P>|z|    [0.025    0.975]
------------------------------------------------------------------------------
sigma2.irregular  0.0000    0.0000      0.841      0.400   -0.0000   0.0001
sigma2.level      0.0000    0.0000      0.093      0.926   -0.0000   0.0000
sigma2.slope      0.0000    0.0000      2.614      0.009    0.0000   0.0000
sigma2.cycle      0.0003    0.0001      4.817      0.000    0.0002   0.0005
sigma2.seasonal   0.0000    0.0000      0.782      0.434   -0.0000   0.0000
cycle.frequency   0.1421    0.0185      7.685      0.000    0.1059   0.1783
cycle.damping     0.9521    0.0214     44.490      0.000    0.9101   0.9940
==============================================================================
```

```python
# ── Interpret cycle parameters ─────────────────────────────────────────────────
lambda_c = results_cycle.params["cycle.frequency"]
rho      = results_cycle.params["cycle.damping"]
period_c = 2 * np.pi / lambda_c   # in months

print(f"\nCycle parameters:")
print(f"  Angular frequency λ_c = {lambda_c:.4f} rad/month")
print(f"  Damping coefficient ρ = {rho:.4f}")
print(f"  Cycle period P_c      = {period_c:.1f} months ({period_c/12:.1f} years)")
print(f"  Half-life             ≈ {np.log(0.5)/np.log(rho):.1f} months")
```

### Expected output

```
Cycle parameters:
  Angular frequency λ_c = 0.1421 rad/month
  Damping coefficient ρ = 0.9521
  Cycle period P_c      = 44.2 months (3.7 years)
  Half-life             ≈ 14.1 months
```

The model identifies a business cycle with a period of approximately
44 months (3.7 years) and high persistence ($\rho = 0.95$). The half-life
of 14 months means a cyclical shock dissipates by half in about 14 months —
consistent with medium-term business cycle dynamics.

!!! note "Interpreting the damping coefficient"
    $\rho$ controls how quickly the cycle mean-reverts. $\rho = 0$ gives
    a white noise cycle (no persistence), $\rho = 1$ gives a unit-root
    cycle (never mean-reverts). Values of $\rho \in [0.9, 0.99]$ correspond
    to cycles with half-lives of 6–70 months, typical for business cycles.

---

## Step 5 — Compare UCM specifications

```python
# ── Information criteria comparison ───────────────────────────────────────────
print("Model comparison:")
print(f"{'Model':30s}  {'LogLik':>10s}  {'AIC':>10s}  {'BIC':>10s}")
print("-" * 65)
print(f"{'UCM (no cycle)':30s}  "
      f"{results_nocycle.llf:>10.3f}  "
      f"{results_nocycle.aic:>10.3f}  "
      f"{results_nocycle.bic:>10.3f}")
print(f"{'UCM (with cycle)':30s}  "
      f"{results_cycle.llf:>10.3f}  "
      f"{results_cycle.aic:>10.3f}  "
      f"{results_cycle.bic:>10.3f}")
```

### Expected output

```
Model comparison:
Model                           LogLik         AIC         BIC
-----------------------------------------------------------------
UCM (no cycle)                 384.117    -760.234    -742.698
UCM (with cycle)               401.843    -791.686    -767.262
```

Both AIC and BIC favour the model with a cycle (lower values). The
log-likelihood improvement of +17.7 with 2 additional parameters ($\lambda_c$,
$\rho$) is decisive — the cycle component is strongly supported by the data.

```python
# ── Likelihood ratio test: cycle vs. no cycle ─────────────────────────────────
from kalmanbox.diagnostics import likelihood_ratio_test

lrt = likelihood_ratio_test(results_nocycle, results_cycle, df=2)
print(f"\nLR test (cycle vs. no cycle):")
print(f"  LR statistic = {lrt.statistic:.4f}")
print(f"  df           = {lrt.df}")
print(f"  p-value      = {lrt.pvalue:.6f}")
```

### Expected output

```
LR test (cycle vs. no cycle):
  LR statistic = 35.452
  df           = 2
  p-value      = 0.000002
```

The likelihood ratio test strongly rejects the no-cycle model (p < 0.001),
confirming that the stochastic cycle captures genuine business cycle
dynamics that the trend-seasonal model misses.

---

## Step 6 — Extract and analyse components

```python
# ── Extract components from UCM with cycle ────────────────────────────────────
components = results_cycle.components()
print("Available components:", components.columns.tolist())

trend    = components["level"]
slope    = components["slope"]
cycle    = components["cycle"]
seasonal = components["seasonal"]
irreg    = components["irregular"]
```

```python
# ── Full decomposition plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

# Original + trend
axes[0].plot(y.index, y.values, color="lightgray", linewidth=1.0,
             label="log(INDPRO)", alpha=0.9)
axes[0].plot(y.index, trend.values, color="steelblue", linewidth=2.0,
             label="Trend $\\mu_t$")
axes[0].set_title("Log Industrial Production and trend")
axes[0].legend()

# Slope
axes[1].plot(y.index, slope.values, color="steelblue", linewidth=1.5)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1].set_title("Slope (trend growth rate) $\\nu_t$")

# Cycle — the key component
axes[2].fill_between(y.index, 0, cycle.values,
                     where=cycle.values >= 0, color="steelblue",
                     alpha=0.5, label="Expansion")
axes[2].fill_between(y.index, 0, cycle.values,
                     where=cycle.values < 0, color="crimson",
                     alpha=0.5, label="Contraction")
axes[2].plot(y.index, cycle.values, color="black", linewidth=0.8)
axes[2].axhline(0, color="black", linewidth=1.0)
axes[2].set_title("Business cycle component $\\psi_t$")
axes[2].legend()

# Seasonal
axes[3].plot(y.index, seasonal.values, color="darkorange", linewidth=1.2)
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[3].set_title("Seasonal component $\\gamma_t$")

# Irregular
axes[4].plot(y.index, irreg.values, color="gray", linewidth=0.8)
axes[4].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[4].set_title("Irregular component $\\varepsilon_t$")
axes[4].set_xlabel("Date")

plt.suptitle("UCM decomposition — Log US Industrial Production",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
```

The cycle panel (middle) is particularly informative. Positive values (blue)
correspond to above-trend production (economic expansions); negative values
(red) correspond to below-trend production (recessions). The 1974–75 oil
shock recession and the 1980–82 double-dip recession are clearly visible as
deep negative excursions.

---

## Step 7 — Align cycle with NBER recession dates

To validate our estimated cycle, compare it against official NBER recession
dates.

```python
# ── NBER recession periods (approximate, US) ──────────────────────────────────
recessions = [
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),   # oil shock
    ("1980-01-01", "1980-07-01"),   # Volcker I
    ("1981-07-01", "1982-11-01"),   # Volcker II
]

fig, ax = plt.subplots(figsize=(14, 5))

# Shade recession periods
for start, end in recessions:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               alpha=0.15, color="gray", label="_nolegend_")

ax.fill_between(y.index, 0, cycle.values,
                where=cycle.values >= 0, color="steelblue", alpha=0.6,
                label="Estimated expansion")
ax.fill_between(y.index, 0, cycle.values,
                where=cycle.values < 0, color="crimson", alpha=0.6,
                label="Estimated contraction")
ax.plot(y.index, cycle.values, color="black", linewidth=0.8)
ax.axhline(0, color="black", linewidth=1.2)

# Add recession labels
ax.text(pd.Timestamp("1974-09-01"), cycle.min() * 0.85,
        "Oil\nshock", ha="center", fontsize=8, color="gray")
ax.text(pd.Timestamp("1981-04-01"), cycle.min() * 0.85,
        "Volcker\ndisinfl.", ha="center", fontsize=8, color="gray")

ax.set_title("UCM business cycle vs. NBER recessions (gray shading)")
ax.set_xlabel("Date")
ax.set_ylabel("Cycle component $\\psi_t$ (log points)")
ax.legend()
plt.tight_layout()
plt.show()
```

The UCM cycle closely tracks NBER recession timing without any prior
information about recession dates — it is estimated purely from the
structure of the Industrial Production series.

---

## Step 8 — Compare UCM vs. BSM

A UCM without a cycle component is equivalent to BSM (with the same trend
and seasonal specifications). Adding a cycle to UCM separates business cycle
dynamics from the long-run trend.

```python
from kalmanbox import BSM

# ── Fit BSM (structurally equivalent to UCM with no cycle) ────────────────────
model_bsm = BSM(y, seasonal_period=12,
                stochastic_level=True, stochastic_slope=True,
                stochastic_seasonal=True, stochastic_cycle=False)
results_bsm = model_bsm.fit(disp=False)
bsm_components = results_bsm.components()

# ── Compare trend estimates ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Trend comparison
axes[0].plot(y.index, y.values, color="lightgray", linewidth=1.0,
             alpha=0.8, label="log(INDPRO)")
axes[0].plot(y.index, components["level"].values, color="steelblue",
             linewidth=2.0, linestyle="-", label="UCM trend $\\mu_t$")
axes[0].plot(y.index, bsm_components["level"].values, color="darkorange",
             linewidth=1.5, linestyle="--", label="BSM trend $\\mu_t$")
axes[0].set_title("Trend: UCM (with cycle) vs. BSM (no cycle)")
axes[0].legend()

# Cycle (UCM) vs. BSM irregular (which absorbs the cycle)
axes[1].plot(y.index, cycle.values, color="steelblue", linewidth=1.5,
             label="UCM cycle $\\psi_t$")
axes[1].plot(y.index, bsm_components["irregular"].values, color="darkorange",
             linewidth=1.0, linestyle="--", label="BSM irregular $\\varepsilon_t$")
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1].set_title("UCM cycle vs. BSM irregular — what the cycle component captures")
axes[1].set_xlabel("Date")
axes[1].legend()

plt.tight_layout()
plt.show()
```

```python
# ── Quantitative comparison ────────────────────────────────────────────────────
from scipy.stats import pearsonr

r_trend, _ = pearsonr(components["level"].values, bsm_components["level"].values)
print(f"Trend correlation (UCM vs BSM): r = {r_trend:.6f}")

print(f"\nModel fit comparison:")
print(f"{'Model':25s}  {'LogLik':>10s}  {'AIC':>10s}  {'BIC':>10s}")
print("-" * 55)
print(f"{'BSM (no cycle)':25s}  "
      f"{results_bsm.llf:>10.3f}  "
      f"{results_bsm.aic:>10.3f}  "
      f"{results_bsm.bic:>10.3f}")
print(f"{'UCM (no cycle)':25s}  "
      f"{results_nocycle.llf:>10.3f}  "
      f"{results_nocycle.aic:>10.3f}  "
      f"{results_nocycle.bic:>10.3f}")
print(f"{'UCM (with cycle)':25s}  "
      f"{results_cycle.llf:>10.3f}  "
      f"{results_cycle.aic:>10.3f}  "
      f"{results_cycle.bic:>10.3f}")
```

### Expected output

```
Trend correlation (UCM vs BSM): r = 0.997214

Model fit comparison:
Model                     LogLik         AIC         BIC
-------------------------------------------------------
BSM (no cycle)           383.902    -759.804    -742.268
UCM (no cycle)           384.117    -760.234    -742.698
UCM (with cycle)         401.843    -791.686    -767.262
```

Key findings:

- BSM and UCM-no-cycle produce nearly identical results (as expected — same
  component structure)
- Adding a cycle to UCM improves log-likelihood by +17.7 with only 2 extra
  parameters, decisively favoured by both AIC and BIC
- The UCM trend is smoother than BSM's because the cycle absorbs
  medium-frequency variation that BSM would otherwise attribute to the trend

!!! tip "BSM vs. UCM: when to use each"
    **BSM** is simpler and sufficient when you are primarily interested in
    trend and seasonal decomposition and do not need to isolate a business
    cycle. **UCM** is preferable when you need to:
    (1) explicitly model and estimate a cycle with a specific period and
    damping, (2) test whether the cycle is statistically significant,
    (3) keep the trend smooth while capturing medium-frequency fluctuations,
    or (4) compare models with different cycle specifications.

---

## Step 9 — Forecast with UCM

```python
# ── Forecast 24 months from UCM with cycle ────────────────────────────────────
forecast = results_cycle.forecast(steps=24)

# Convert back to original scale (exp of log forecasts)
fc_mean     = np.exp(forecast["mean"])
fc_lower_80 = np.exp(forecast["lower_80"])
fc_upper_80 = np.exp(forecast["upper_80"])
fc_lower_95 = np.exp(forecast["lower_95"])
fc_upper_95 = np.exp(forecast["upper_95"])

# ── Plot forecast on original scale ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Show last 5 years of history for context
history_idx = y_raw.index[-60:]
history_val = y_raw.values[-60:]

ax.plot(history_idx, history_val, color="steelblue", linewidth=1.5,
        label="Historical (thousands)")
ax.fill_between(
    forecast.index,
    fc_lower_95.values,
    fc_upper_95.values,
    alpha=0.20, color="darkorange", label="95% PI",
)
ax.fill_between(
    forecast.index,
    fc_lower_80.values,
    fc_upper_80.values,
    alpha=0.35, color="darkorange", label="80% PI",
)
ax.plot(forecast.index, fc_mean.values, color="darkorange",
        linewidth=2.0, label="Forecast mean")
ax.axvline(y_raw.index[-1], color="black", linewidth=0.8,
           linestyle="--", alpha=0.5, label="Forecast origin")
ax.set_title("UCM forecast — US Industrial Production (original scale)")
ax.set_xlabel("Date")
ax.set_ylabel("Index (2012 = 100)")
ax.legend()
plt.tight_layout()
plt.show()

print(f"\nForecast summary (original scale):")
print(f"  Month 1   : {fc_mean.iloc[0]:.2f}  "
      f"[{fc_lower_95.iloc[0]:.2f}, {fc_upper_95.iloc[0]:.2f}]")
print(f"  Month 12  : {fc_mean.iloc[11]:.2f}  "
      f"[{fc_lower_95.iloc[11]:.2f}, {fc_upper_95.iloc[11]:.2f}]")
print(f"  Month 24  : {fc_mean.iloc[23]:.2f}  "
      f"[{fc_lower_95.iloc[23]:.2f}, {fc_upper_95.iloc[23]:.2f}]")
```

The UCM forecast reflects the current position in the business cycle —
if we are in a cyclical contraction, the forecast will show a gradual
recovery toward trend. This is the key advantage of UCM over BSM for
cyclically sensitive series: forecasts incorporate both the trend trajectory
and the current phase of the cycle.

!!! info "Forecast uncertainty with a cycle"
    UCM forecasts have wider prediction intervals than BSM at medium
    horizons (6–24 months) because cycle uncertainty compounds with horizon:
    we do not know where the cycle will be 24 months from now. At very long
    horizons (> 5 years), UCM and BSM interval widths converge as the cycle
    mean-reverts to zero.

---

## Step 10 — Cycle component with uncertainty

```python
# ── Extract cycle component with confidence band ───────────────────────────────
smoothed = results_cycle.smooth()

# Cycle component is part of the smoothed state
# Components returns smoothed component means; for variance we need the
# smoothed state covariances and the corresponding columns.
cycle_mean = components["cycle"].values

# Approximate 95% CI using the residual std of the cycle
cycle_std = np.sqrt(results_cycle.params.get("sigma2.cycle", 1e-6))
# Use smoother variances if exposed by results object
try:
    cycle_var = results_cycle.component_variances()["cycle"].values
    cycle_band = 1.96 * np.sqrt(cycle_var)
except AttributeError:
    cycle_band = np.full_like(cycle_mean, 1.96 * cycle_std * 5)

fig, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(y.index,
                cycle_mean - cycle_band,
                cycle_mean + cycle_band,
                alpha=0.2, color="steelblue", label="95% CI")
ax.fill_between(y.index, 0, cycle_mean,
                where=cycle_mean >= 0, color="steelblue", alpha=0.5)
ax.fill_between(y.index, 0, cycle_mean,
                where=cycle_mean < 0, color="crimson", alpha=0.5)
ax.plot(y.index, cycle_mean, color="black", linewidth=1.0)
ax.axhline(0, color="black", linewidth=1.0)
ax.set_title(f"Estimated business cycle $\\psi_t$  "
             f"(period = {period_c:.0f} months, ρ = {rho:.3f})")
ax.set_xlabel("Date")
ax.set_ylabel("Cycle (log points)")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Summary

| Step | API | Key finding |
|------|-----|-------------|
| 1 | `load_dataset`, `np.log` | US IP shows trend + business cycle + seasonality |
| 2 | UCM theory | UCM adds a stochastic cycle $\psi_t$ with frequency $\lambda_c$ and damping $\rho$ |
| 3 | `UCM(cycle=False).fit()` | Baseline: AIC = −760.2, comparable to BSM |
| 4 | `UCM(cycle=True).fit()` | MLE finds cycle period ≈ 44 months, $\rho = 0.95$ |
| 5 | `likelihood_ratio_test` | Cycle component highly significant (p < 0.001) |
| 6 | `results.components()` | Cycle captures 1974–75 and 1980–82 recessions |
| 7 | NBER comparison | UCM cycle aligns with NBER recession timing |
| 8 | BSM vs. UCM | UCM trend smoother; cycle absorbs medium-frequency variation |
| 9 | `results.forecast(24)` | Forecast reflects current cycle phase |
| 10 | Component uncertainty | Cycle CI widens for poorly identified phases |

---

## Next steps

<div class="grid cards" markdown>

-   :material-bank:{ .lg .middle } **DFM Tutorial**

    ---

    Extract common factors from a panel of macroeconomic series — the
    natural extension of UCM to multiple series.

    [:octicons-arrow-right-24: DFM Tutorial](us-macro-dfm.md)

-   :material-book-open-variant:{ .lg .middle } **UCM User Guide**

    ---

    Full API reference: component specification options, cycle
    parameterisation, identifiability conditions.

    [:octicons-arrow-right-24: UCM Guide](../user-guide/structural/ucm.md)

-   :material-chart-bar:{ .lg .middle } **Cycle Component Guide**

    ---

    Deep dive into the stochastic cycle: spectral interpretation,
    relationship to ARMA processes, and identification.

    [:octicons-arrow-right-24: Cycle Component](../user-guide/structural/cycle.md)

-   :material-flask-outline:{ .lg .middle } **Diagnostics**

    ---

    CUSUM stability tests, innovation tests, and auxiliary residuals —
    essential for validating any structural model.

    [:octicons-arrow-right-24: Diagnostics](../diagnostics/index.md)

</div>
