---
title: "Tutorial: BSM Structural Decomposition"
description: >-
  Intermediate tutorial that decomposes the international airline passengers
  series into trend, seasonal, and irregular components using the Basic
  Structural Model (BSM), with MLE estimation, forecasting, and comparison
  against STL decomposition.
---

# Tutorial: BSM Structural Decomposition

**Level:** :material-signal: Intermediate · **Time:** ~60 min · **Dataset:** Airline Passengers

The Basic Structural Model (BSM) decomposes a time series into interpretable
components — trend, seasonal, and irregular — estimated simultaneously by
maximum likelihood via the Kalman filter. Unlike STL (Seasonal-Trend
decomposition using Loess), BSM is a **probabilistic model** that:

- Provides uncertainty estimates for each component
- Allows components to be **stochastic** (they evolve over time)
- Produces proper forecast prediction intervals
- Supports formal hypothesis testing via likelihood ratio tests

By the end of this tutorial you will have:

- Fitted a BSM to the classic airline passengers dataset
- Extracted and plotted all decomposed components
- Diagnosed residuals and validated the model
- Produced a 24-month forecast with prediction intervals
- Compared BSM vs. STL decomposition

!!! info "Prerequisites"
    Complete [Fundamentals](fundamentals.md) first or have basic familiarity with
    `KalmanFilter` and `MLEstimator`. Install: `pip install kalmanbox statsmodels`

---

## The dataset: international airline passengers

Box and Jenkins (1976) introduced the airline passengers dataset as the
canonical example of a seasonal time series. It records monthly totals of
international airline passengers (in thousands) from January 1949 to
December 1960 — 144 monthly observations.

**Key features:**
- **Trend**: strong upward trend (passengers roughly tripled over 12 years)
- **Seasonality**: clear 12-month seasonal pattern (peaks in summer)
- **Variance growth**: seasonal swings grow with the level (multiplicative)

Because variance grows with the level, we work on the **log scale** — this
converts multiplicative seasonality to additive seasonality, making BSM
appropriate.

---

## Step 1 — Load and explore the data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanbox.datasets import load_dataset

# ── Load the airline passengers dataset ───────────────────────────────────────
air: pd.DataFrame = load_dataset("airpassengers")
print(air.head(10))
print(f"\nShape   : {air.shape}")
print(f"Period  : {air.index[0]} → {air.index[-1]}")
print(f"Min     : {air['passengers'].min()}")
print(f"Max     : {air['passengers'].max()}")
```

### Expected output

```
            passengers
1949-01-01         112
1949-02-01         118
1949-03-01         132
...

Shape   : (144, 1)
Period  : 1949-01-01 → 1960-12-01
Min     : 104
Max     : 622
```

```python
# ── Log transform: stabilise variance ─────────────────────────────────────────
y_raw = air["passengers"]
y     = np.log(y_raw)          # work on log scale throughout

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

axes[0].plot(y_raw.index, y_raw.values, color="steelblue", linewidth=1.2)
axes[0].set_title("Raw airline passengers (thousands)")
axes[0].set_ylabel("Passengers")

axes[1].plot(y.index, y.values, color="darkorange", linewidth=1.2)
axes[1].set_title("Log airline passengers (log scale)")
axes[1].set_ylabel("log(Passengers)")

plt.tight_layout()
plt.show()
```

The raw series shows multiplicative seasonality — the amplitude of seasonal
swings grows with the trend. On the log scale the seasonal amplitude is
roughly constant, confirming that the log transformation is appropriate.

---

## Step 2 — The BSM state-space model

The Basic Structural Model decomposes the log series as:

$$
\log y_t = \mu_t + \gamma_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

**Trend component** (Local Linear Trend):

$$
\begin{aligned}
\mu_{t+1} &= \mu_t + \nu_t + \xi_t, &\quad \xi_t &\sim \mathcal{N}(0, \sigma_\xi^2) \\
\nu_{t+1} &= \nu_t + \zeta_t,         &\quad \zeta_t &\sim \mathcal{N}(0, \sigma_\zeta^2)
\end{aligned}
$$

**Seasonal component** (trigonometric, $s = 12$ harmonics):

$$
\gamma_t = \sum_{j=1}^{[s/2]} \gamma_{j,t}, \quad
\begin{pmatrix} \gamma_{j,t+1} \\ \gamma^*_{j,t+1} \end{pmatrix}
= \begin{pmatrix} \cos\lambda_j & \sin\lambda_j \\ -\sin\lambda_j & \cos\lambda_j \end{pmatrix}
\begin{pmatrix} \gamma_{j,t} \\ \gamma^*_{j,t} \end{pmatrix}
+ \begin{pmatrix} \omega_{j,t} \\ \omega^*_{j,t} \end{pmatrix}
$$

where $\lambda_j = 2\pi j/s$ and $\omega_{j,t} \sim \mathcal{N}(0, \sigma_\omega^2)$.

The model has **four variance parameters** to estimate by MLE:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Irregular variance | $\sigma_\varepsilon^2$ | i.i.d. measurement noise |
| Level variance | $\sigma_\xi^2$ | Stochastic drift in the level |
| Slope variance | $\sigma_\zeta^2$ | Stochastic changes in slope |
| Seasonal variance | $\sigma_\omega^2$ | Evolution of seasonal pattern |

Setting any variance to zero makes the corresponding component
**deterministic** (fixed shape, no evolution).

---

## Step 3 — Configure and fit the BSM

```python
from kalmanbox import BSM

# ── Configure BSM with monthly seasonality ────────────────────────────────────
model = BSM(
    y,
    seasonal_period=12,           # monthly: 12-month seasonal cycle
    stochastic_level=True,        # level drifts stochastically
    stochastic_slope=True,        # slope evolves stochastically
    stochastic_seasonal=True,     # seasonal pattern can change over time
    stochastic_cycle=False,       # no business cycle component for now
)

# ── Fit by MLE using Newton-Raphson ───────────────────────────────────────────
results = model.fit(method="newton", disp=False)

print(results.summary())
```

### Expected output

```
==============================================================================
                     Basic Structural Model (BSM)
==============================================================================
Model:             BSM           Log-Likelihood:     244.696
Sample:            1949-01-01    AIC:               -481.392
                   1960-12-01    BIC:               -467.212
No. Observations:  144
Seasonal period:   12            Stochastic level:  True
                                 Stochastic slope:  True
                                 Stochastic seasonal: True
==============================================================================
                    coef    std err          z      P>|z|    [0.025    0.975]
------------------------------------------------------------------------------
sigma2.irregular  0.0000    0.0001      0.086      0.932   -0.0001   0.0001
sigma2.level      0.0000    0.0000      0.126      0.900   -0.0000   0.0000
sigma2.slope      0.0001    0.0000      2.318      0.020    0.0000   0.0001
sigma2.seasonal   0.0000    0.0000      0.182      0.856   -0.0000   0.0000
==============================================================================
```

!!! info "Interpreting near-zero variances"
    The near-zero estimates for `sigma2.irregular`, `sigma2.level`, and
    `sigma2.seasonal` indicate that these components are effectively
    **deterministic** for this dataset. The significant `sigma2.slope`
    tells us the trend slope changes stochastically — the growth rate of
    airline travel was not constant over this period.

    This is a common pattern for the airline dataset: the BSM concentrates
    almost all stochasticity in the slope, producing a smooth trend with a
    stable seasonal shape.

```python
# ── Inspect estimated parameters ──────────────────────────────────────────────
print("\nEstimated parameters:")
for name, val in results.params.items():
    print(f"  {name:25s} = {val:.6f}")

print(f"\nLog-likelihood : {results.llf:.4f}")
print(f"AIC            : {results.aic:.4f}")
print(f"BIC            : {results.bic:.4f}")
```

---

## Step 4 — Extract and plot decomposed components

```python
# ── Extract components ────────────────────────────────────────────────────────
components = results.components()
print("Available components:", components.columns.tolist())
print(components.head())
```

### Expected output

```
Available components: ['level', 'slope', 'seasonal', 'irregular']

                level     slope  seasonal  irregular
1949-01-01  4.709382  0.010283 -0.060221        0.0
1949-02-01  4.719665  0.010283 -0.001732        0.0
1949-03-01  4.729948  0.010283  0.082451        0.0
...
```

```python
# ── Full decomposition plot ───────────────────────────────────────────────────
from kalmanbox.visualization import plot_components

fig = plot_components(
    results,
    title="BSM decomposition — Log airline passengers",
    figsize=(13, 12),
)
plt.show()
```

```python
# ── Manual decomposition plot for full control ────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)

# Original + trend
axes[0].plot(y.index, y.values, color="lightgray", linewidth=1.0,
             label="log(passengers)")
axes[0].plot(y.index, components["level"].values, color="steelblue",
             linewidth=2.0, label="Trend $\\mu_t$")
axes[0].set_title("Original series and trend component")
axes[0].legend()

# Slope (trend growth rate)
axes[1].plot(y.index, components["slope"].values, color="darkorange",
             linewidth=1.5, label="Slope $\\nu_t$")
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1].set_title("Slope (trend growth rate)")
axes[1].legend()

# Seasonal component
axes[2].plot(y.index, components["seasonal"].values, color="crimson",
             linewidth=1.5, label="Seasonal $\\gamma_t$")
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[2].set_title("Seasonal component")
axes[2].legend()

# Irregular (residual)
axes[3].plot(y.index, components["irregular"].values, color="gray",
             linewidth=0.8, label="Irregular $\\varepsilon_t$")
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[3].set_title("Irregular component")
axes[3].set_xlabel("Date")
axes[3].legend()

plt.suptitle("BSM decomposition — Log airline passengers", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
```

The four-panel plot reveals the story of airline travel 1949–1960:

- **Trend** ($\mu_t$): a smooth upward trajectory accelerating through the
  period — the aviation boom
- **Slope** ($\nu_t$): the trend growth rate itself, increasing from ~0.8%
  per month early in the sample to ~1.2% by 1960
- **Seasonal** ($\gamma_t$): a stable 12-month pattern peaking in July-August
  and troughing in November-January
- **Irregular** ($\varepsilon_t$): negligible — virtually all variation is
  explained by trend and season

---

## Step 5 — Examine seasonal pattern in detail

```python
# ── Monthly seasonal factors (average seasonal component by month) ────────────
import calendar

seasonal_vals = components["seasonal"].values
months = y.index.month

avg_seasonal = np.array([seasonal_vals[months == m].mean() for m in range(1, 13)])
month_names  = [calendar.month_abbr[m] for m in range(1, 13)]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["steelblue" if v >= 0 else "crimson" for v in avg_seasonal]
ax.bar(month_names, avg_seasonal, color=colors, alpha=0.8, edgecolor="black",
       linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Average seasonal factors by month (log scale)")
ax.set_ylabel("Seasonal effect (log points)")

# Add value labels
for i, (name, val) in enumerate(zip(month_names, avg_seasonal)):
    ax.text(i, val + 0.002 * np.sign(val), f"{val:.3f}",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

plt.tight_layout()
plt.show()

print("\nMonthly seasonal factors (log scale):")
for m, s in zip(month_names, avg_seasonal):
    pct = (np.exp(s) - 1) * 100
    print(f"  {m}: {s:+.4f} ({pct:+.1f}% vs trend)")
```

### Expected output

```
Monthly seasonal factors (log scale):
  Jan: -0.0697 (-6.7% vs trend)
  Feb: -0.0018 (-0.2% vs trend)
  Mar: +0.0879 (+9.2% vs trend)
  Apr: +0.0424 (+4.3% vs trend)
  May: +0.0271 (+2.7% vs trend)
  Jun: +0.1560 (+16.9% vs trend)
  Jul: +0.2415 (+27.3% vs trend)
  Aug: +0.2300 (+25.9% vs trend)
  Sep: +0.0799 (+8.3% vs trend)
  Oct: -0.0246 (-2.4% vs trend)
  Nov: -0.1730 (-15.9% vs trend)
  Dec: -0.0951 (-9.1% vs trend)
```

The seasonal factors confirm the travel pattern: July is the peak month
(27% above trend), while November is the trough (16% below trend).

---

## Step 6 — Diagnose the model

```python
from kalmanbox.diagnostics import residual_diagnostics

diag = residual_diagnostics(results)
print(diag)
```

### Expected output

```
Residual diagnostics
--------------------
Ljung-Box Q(10):    7.32   p-value: 0.695   (no autocorrelation detected)
Jarque-Bera:        0.95   p-value: 0.622   (residuals appear Gaussian)
Heteroscedasticity: 1.14   p-value: 0.341   (no heteroscedasticity detected)
```

```python
# ── Visual diagnostics ────────────────────────────────────────────────────────
from kalmanbox.visualization import plot_diagnostic_panel

fig = plot_diagnostic_panel(results, figsize=(14, 10))
plt.suptitle("BSM residual diagnostics — airline passengers", fontsize=12, y=1.01)
plt.show()
```

All diagnostic tests pass. The innovations are serially uncorrelated,
approximately Gaussian, and homoscedastic — confirming that the BSM
adequately captures the systematic variation in the series.

---

## Step 7 — Forecast 24 months

```python
# ── Forecast 24 months (Jan 1961 – Dec 1962) ──────────────────────────────────
n_forecast = 24
forecast = results.forecast(steps=n_forecast)

print(f"Forecast period: {forecast.index[0]} → {forecast.index[-1]}")
print("\nFirst 6 months of forecast (log scale):")
print(forecast[["mean", "lower_95", "upper_95"]].head(6))
```

### Expected output

```
Forecast period: 1961-01-01 → 1962-12-01

                mean  lower_95  upper_95
1961-01-01  6.186432  6.060119  6.312745
1961-02-01  6.201821  6.062131  6.341511
1961-03-01  6.295508  6.144124  6.446893
1961-04-01  6.248959  6.086426  6.411491
1961-05-01  6.234477  6.060976  6.407978
1961-06-01  6.392497  6.208295  6.576698
```

```python
# ── Convert forecast back to original scale ────────────────────────────────────
fc_mean     = np.exp(forecast["mean"])
fc_lower_80 = np.exp(forecast["lower_80"])
fc_upper_80 = np.exp(forecast["upper_80"])
fc_lower_95 = np.exp(forecast["lower_95"])
fc_upper_95 = np.exp(forecast["upper_95"])

# ── Plot history + forecast on original scale ──────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_raw.index, y_raw.values, color="steelblue", linewidth=1.5,
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
           linestyle="--", alpha=0.6, label="Forecast origin")
ax.set_title("BSM forecast — airline passengers (original scale, thousands)")
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")
ax.legend()
plt.tight_layout()
plt.show()

# ── Summary statistics ─────────────────────────────────────────────────────────
print(f"\n24-month forecast summary (thousands):")
print(f"  Jan 1961 forecast : {fc_mean.iloc[0]:.1f}  "
      f"[{fc_lower_95.iloc[0]:.1f}, {fc_upper_95.iloc[0]:.1f}]")
print(f"  Jul 1961 forecast : {fc_mean.iloc[6]:.1f}  "
      f"[{fc_lower_95.iloc[6]:.1f}, {fc_upper_95.iloc[6]:.1f}]")
print(f"  Dec 1962 forecast : {fc_mean.iloc[-1]:.1f}  "
      f"[{fc_lower_95.iloc[-1]:.1f}, {fc_upper_95.iloc[-1]:.1f}]")
```

The forecast reproduces the seasonal pattern while extrapolating the trend
upward. Prediction intervals widen with the forecast horizon, capturing the
compounding uncertainty from slope drift.

---

## Step 8 — Compare BSM vs. STL decomposition

STL (Seasonal-Trend decomposition using Loess) is a popular non-parametric
alternative to BSM. The key differences:

| Feature | BSM | STL |
|---------|-----|-----|
| **Model** | Probabilistic state-space | Non-parametric smoothing |
| **Parameters** | Estimated by MLE | Bandwidth chosen heuristically |
| **Uncertainty** | Full uncertainty intervals | None by default |
| **Forecasting** | Native (extrapolate SSM) | Requires separate model |
| **Missing data** | Handled automatically | Requires interpolation |
| **Stochastic seasonality** | Yes (via $\sigma_\omega^2$) | No (fixed seasonal each year) |
| **Inference** | Likelihood ratio tests | No formal tests |

```python
from statsmodels.tsa.seasonal import STL

# ── STL decomposition ─────────────────────────────────────────────────────────
stl = STL(y, seasonal=13, robust=True)
stl_result = stl.fit()

# ── BSM smoothed states ────────────────────────────────────────────────────────
bsm_smoothed = results.smooth()
bsm_trend    = components["level"].values
bsm_seasonal = components["seasonal"].values
bsm_irreg    = components["irregular"].values

stl_trend    = stl_result.trend
stl_seasonal = stl_result.seasonal
stl_irreg    = stl_result.resid

# ── Side-by-side comparison ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

for col, (label, trend, seasonal, irreg) in enumerate([
    ("BSM",  bsm_trend,  bsm_seasonal,  bsm_irreg),
    ("STL",  stl_trend,  stl_seasonal,  stl_irreg),
]):
    axes[0, col].plot(y.index, trend, color="steelblue", linewidth=1.5)
    axes[0, col].set_title(f"{label} — Trend")

    axes[1, col].plot(y.index, seasonal, color="darkorange", linewidth=1.0)
    axes[1, col].axhline(0, color="gray", linewidth=0.7, linestyle="--")
    axes[1, col].set_title(f"{label} — Seasonal")

    axes[2, col].plot(y.index, irreg, color="gray", linewidth=0.8)
    axes[2, col].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2, col].set_title(f"{label} — Irregular")
    axes[2, col].set_xlabel("Date")

plt.suptitle("BSM vs STL decomposition — Log airline passengers",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
```

```python
# ── Quantitative comparison ────────────────────────────────────────────────────
print("Decomposition comparison:")
print(f"\n  STL trend:    min={stl_trend.min():.4f}  max={stl_trend.max():.4f}")
print(f"  BSM trend:    min={bsm_trend.min():.4f}  max={bsm_trend.max():.4f}")
print(f"\n  STL seasonal variance: {stl_seasonal.var():.6f}")
print(f"  BSM seasonal variance: {bsm_seasonal.var():.6f}")
print(f"\n  STL irregular std: {stl_irreg.std():.6f}")
print(f"  BSM irregular std: {bsm_irreg.std():.6f}")

# Correlation between trend estimates
from scipy.stats import pearsonr
r, _ = pearsonr(bsm_trend, stl_trend)
print(f"\n  Trend correlation (BSM vs STL): r = {r:.6f}")
```

### Expected output

```
Decomposition comparison:

  STL trend:    min=4.6939  max=6.3141
  BSM trend:    min=4.7016  max=6.2897

  STL seasonal variance: 0.019104
  BSM seasonal variance: 0.018972

  STL irregular std: 0.016281
  BSM irregular std: 0.000143

  Trend correlation (BSM vs STL): r = 0.999721
```

The two methods produce nearly identical trend estimates (correlation > 0.999)
and very similar seasonal patterns. The key difference is in the irregular:
BSM assigns almost nothing to the irregular component (the model explains
nearly all variation through trend and seasonal), while STL's irregular is
larger because STL's smoother is not constrained to a parametric model.

!!! tip "When to choose BSM over STL"
    Choose BSM when you need: (1) prediction intervals on components,
    (2) a probabilistic forecast, (3) formal model comparison via AIC/BIC,
    (4) to test whether seasonality is deterministic or stochastic, or
    (5) to handle missing values without pre-imputation. Choose STL for
    quick exploratory decomposition without distributional assumptions.

---

## Step 9 — Test deterministic vs. stochastic seasonality

A key advantage of BSM is the ability to formally test whether the seasonal
pattern evolves over time using a likelihood ratio test.

```python
from kalmanbox.diagnostics import likelihood_ratio_test

# ── Model with stochastic seasonal (unrestricted) ─────────────────────────────
model_unrestricted = BSM(y, seasonal_period=12,
                         stochastic_level=True, stochastic_slope=True,
                         stochastic_seasonal=True)
results_unrestr = model_unrestricted.fit(disp=False)

# ── Model with deterministic seasonal (restricted: σ²_ω = 0) ─────────────────
model_restricted = BSM(y, seasonal_period=12,
                       stochastic_level=True, stochastic_slope=True,
                       stochastic_seasonal=False)
results_restr = model_restricted.fit(disp=False)

# ── Likelihood ratio test ─────────────────────────────────────────────────────
lrt = likelihood_ratio_test(results_restr, results_unrestr, df=1)
print(lrt)

print(f"\nRestricted   loglik = {results_restr.llf:.4f}  (deterministic seasonal)")
print(f"Unrestricted loglik = {results_unrestr.llf:.4f}  (stochastic seasonal)")
print(f"LR statistic        = {lrt.statistic:.4f}")
print(f"p-value             = {lrt.pvalue:.4f}")
```

### Expected output

```
Restricted   loglik = 244.412  (deterministic seasonal)
Unrestricted loglik = 244.696  (stochastic seasonal)
LR statistic        = 0.568
p-value             = 0.451
```

The p-value of 0.451 means we cannot reject the null hypothesis of
deterministic seasonality. For the airline dataset, the seasonal pattern
is stable over the 12-year period — a finding consistent with the near-zero
`sigma2.seasonal` estimated in Step 3.

---

## Summary

| Step | API | Key finding |
|------|-----|-------------|
| 1 | `load_dataset`, `np.log` | Log transform stabilises multiplicative variance |
| 2 | BSM theory | 4-parameter SSM: irregular, level, slope, seasonal variances |
| 3 | `BSM.fit()` | MLE estimates: slope is the only significant stochastic component |
| 4 | `results.components()` | Trend, slope, seasonal, irregular extracted cleanly |
| 5 | Monthly aggregation | July = peak (+27%), November = trough (-16%) |
| 6 | `residual_diagnostics()` | All diagnostics pass — model is well-specified |
| 7 | `results.forecast(24)` | 24-month seasonal forecast with widening intervals |
| 8 | `STL` comparison | Trend correlates > 0.999 with STL; BSM adds uncertainty + inference |
| 9 | `likelihood_ratio_test()` | Cannot reject deterministic seasonality (p = 0.45) |

---

## Next steps

<div class="grid cards" markdown>

-   :material-tune-variant:{ .lg .middle } **UCM Tutorial**

    ---

    Build a more flexible Unobserved Components Model with a stochastic
    cycle and compare it against BSM on the same dataset.

    [:octicons-arrow-right-24: UCM Tutorial](ucm.md)

-   :material-book-open-variant:{ .lg .middle } **BSM User Guide**

    ---

    Full API reference: all constructor parameters, component extraction
    options, and identifiability conditions for BSM.

    [:octicons-arrow-right-24: BSM Guide](../user-guide/structural/bsm.md)

-   :material-chart-bar:{ .lg .middle } **Information Criteria**

    ---

    Use AIC, BIC, and HQIC to formally compare BSM specifications and
    choose the best-fitting model.

    [:octicons-arrow-right-24: Information Criteria](../diagnostics/information-criteria.md)

-   :material-flask-outline:{ .lg .middle } **Likelihood Ratio Test**

    ---

    Test nested hypotheses: is the slope stochastic? Is seasonality
    stochastic? All via the Kalman filter log-likelihood.

    [:octicons-arrow-right-24: Likelihood Ratio Test](../diagnostics/likelihood-ratio.md)

</div>
