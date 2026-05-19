# Tutorial — Time-Varying CAPM with TVP Regression

The Capital Asset Pricing Model (CAPM) assumes a **constant market
beta**. In practice, the sensitivity of a stock to the market shifts
with business cycles, leverage, and liquidity regimes. A TVP regression
casts alpha and beta as random-walk states, letting the data reveal how
they move.

## Model

**Observation equation**

$$
r_t^e = \alpha_t + \beta_t\, r_{m,t}^e + \varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_\varepsilon^2)
$$

**State transition equations**

$$
\alpha_t = \alpha_{t-1} + \eta_t^\alpha, \qquad
\eta_t^\alpha \sim \mathcal{N}(0,\, \sigma_\alpha^2)
$$

$$
\beta_t = \beta_{t-1} + \eta_t^\beta, \qquad
\eta_t^\beta \sim \mathcal{N}(0,\, \sigma_\beta^2)
$$

The state vector is $\theta_t = (\alpha_t,\, \beta_t)'$. The regressor
matrix (design matrix for TVP) is $Z_t = (1,\, r_{m,t}^e)$.

## 1. Load monthly excess returns

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanbox import TimeVaryingParameters as TVP

rng = np.random.default_rng(0)
T = 360          # 30 years of monthly data
dates = pd.date_range("1995-01", periods=T, freq="MS")

# Simulate a time-varying beta that rises, then falls
true_alpha = 0.002 * np.ones(T)
true_beta = np.linspace(0.7, 1.4, T // 2).tolist() + \
            np.linspace(1.4, 0.9, T - T // 2).tolist()
true_beta = np.array(true_beta)

# Market excess return: persistent AR(1), roughly 5% annualised Sharpe
r_mkt = np.zeros(T)
for t in range(1, T):
    r_mkt[t] = 0.05 * r_mkt[t - 1] + rng.normal(0, 0.045)

# Stock excess return generated from the TVP CAPM
sigma_eps = 0.025
r_stock = true_alpha + true_beta * r_mkt + rng.normal(0, sigma_eps, T)

# Package as pandas Series
df = pd.DataFrame(
    {"r_stock": r_stock, "r_mkt": r_mkt},
    index=dates,
)
print(df.describe().round(4))
```

## 2. Fit TVP regression in state-space form

```python
import numpy as np
from kalmanbox import TimeVaryingParameters as TVP

# Regressors: intercept + market return
X = np.column_stack([np.ones(T), df["r_mkt"].values])
y = df["r_stock"].values

model = TVP(y, exog=X)
results = model.fit()
print(results.summary())
```

Typical summary:

```
         Time-Varying Parameters Results
=====================================================
Method                  MLE (L-BFGS-B)
Log-likelihood          1047.3
AIC                    -2086.6
BIC                    -2070.8
sigma2_eps               6.24e-04
sigma2_alpha             1.12e-07
sigma2_beta              3.41e-05
=====================================================
```

!!! note "Identifiability"
    `sigma2_alpha` is near zero — the intercept barely moves.
    This is consistent with the data-generating process above and a
    common finding in real equity data (the CAPM alpha is close to zero
    and stable).

## 3. Plot time-varying beta

```python
smoothed = results.smooth()
# State vector: column 0 = alpha_t, column 1 = beta_t
alpha_t = smoothed.a_smoothed[:, 0]
beta_t  = smoothed.a_smoothed[:, 1]
# 90 % confidence bands
se_beta = smoothed.P_smoothed[:, 1, 1] ** 0.5

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates, beta_t, color="C0", linewidth=1.5, label=r"$\hat\beta_t$")
ax.fill_between(dates,
                beta_t - 1.645 * se_beta,
                beta_t + 1.645 * se_beta,
                alpha=0.20, color="C0", label="90 % CI")
ax.plot(dates, true_beta, color="C3", linewidth=1.0,
        linestyle="--", label="True $\\beta_t$")
ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")
ax.set_title("Time-Varying Market Beta")
ax.set_ylabel(r"$\beta_t$")
ax.legend()
plt.tight_layout()
```

The smoother recovers the true arc-shaped beta trajectory closely,
with uncertainty bands widening at the tails of the sample.

## 4. Test for parameter constancy (Nyblom test)

The Nyblom (1989) test checks whether the coefficients are constant
against the alternative of a martingale variation:

```python
from kalmanbox.diagnostics import nyblom_test

ntest = nyblom_test(results)
print(ntest)
```

Example output:

```
Nyblom Parameter Constancy Test
----------------------------------------
           Statistic  p-value  Reject H0
alpha_t       0.082    0.743    False
beta_t        1.847    0.001    True
Joint         2.914    0.002    True
----------------------------------------
Critical values (5%): individual 0.47, joint 1.07
```

Beta is strongly time-varying; alpha is not — consistent with the
simulation design.

## 5. Compare to OLS constant-coefficient CAPM

```python
from kalmanbox import RegressionSSM   # static coefficients
import numpy as np

ols_model = RegressionSSM(y, exog=X)
ols_results = ols_model.fit()

# OLS estimates
beta_ols = ols_results.params["beta"][1]   # market beta
alpha_ols = ols_results.params["beta"][0]  # alpha

print(f"OLS beta  = {beta_ols:.4f}")
print(f"OLS alpha = {alpha_ols:.6f}")

# Overlay on the TVP beta plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates, beta_t, color="C0", linewidth=1.5,
        label=r"TVP $\hat\beta_t$")
ax.fill_between(dates,
                beta_t - 1.645 * se_beta,
                beta_t + 1.645 * se_beta,
                alpha=0.15, color="C0")
ax.axhline(beta_ols, color="C1", linewidth=1.5,
           linestyle="--", label=f"OLS $\\hat\\beta$ = {beta_ols:.2f}")
ax.set_title("TVP Beta vs OLS Constant Beta")
ax.legend()
plt.tight_layout()
```

The OLS estimate is a **time-averaged** beta and misses the full arc,
leading to:

- Mis-stated systematic risk during the high-beta period.
- Overestimated risk during the low-beta period.
- Biased standard errors due to coefficient instability.

| Metric                      | OLS CAPM | TVP CAPM |
|-----------------------------|:--------:|:--------:|
| Log-likelihood              | 982.1    | 1047.3   |
| AIC                         | −1960.2  | −2086.6  |
| RMSE (in-sample)            | 0.0268   | 0.0241   |
| Nyblom (beta) rejected?     | —        | Yes      |

!!! tip "When to prefer OLS"
    If `sigma2_beta` is estimated near zero and the Nyblom test does
    not reject, parameter variation is negligible. OLS is then both
    simpler and more efficient.

## What we learned

- TVP regression detects and tracks the slow drift in the market beta
  that OLS obscures.
- The Nyblom test provides a formal criterion for whether variation is
  statistically significant.
- Precision of $\hat\beta_t$ depends on the signal-to-noise ratio
  $\sigma_\beta^2 / \sigma_\varepsilon^2$; low SNR produces wide
  confidence bands.

## Next

- [User guide: Time-Varying Parameters](../user-guide/advanced/tvp.md)
- [Bayesian TVP with shrinkage priors](../user-guide/bayesian/priors.md)
- [Stability diagnostics (Nyblom, CUSUM)](../diagnostics/stability.md)
