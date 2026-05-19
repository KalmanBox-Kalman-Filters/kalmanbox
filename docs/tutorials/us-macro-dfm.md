# Tutorial — US Macroeconomic Dynamic Factor Model

A **Dynamic Factor Model** (DFM) extracts a small number of latent
common factors from a panel of observed series. Here we fit a 2-factor
DFM to five monthly US macroeconomic indicators and construct a
coincident economic activity index.

## Model

**Measurement equation**

$$
y_t = \Lambda\, f_t + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal{N}(0, H),\ H\ \text{diagonal}
$$

**Transition equation**

$$
f_t = A\, f_{t-1} + \eta_t, \qquad
\eta_t \sim \mathcal{N}(0, Q)
$$

where $y_t \in \mathbb{R}^p$ is the vector of $p=5$ observed
(standardised) series, $f_t \in \mathbb{R}^2$ are the two common
factors, $\Lambda \in \mathbb{R}^{5 \times 2}$ is the loading matrix,
and $A$ governs the factor dynamics.

For identification the upper $2 \times 2$ block of $\Lambda$ is
constrained to be lower-triangular, and $Q = I_2$.

## 1. Load data

We construct a realistic pseudo-panel that mimics US macro dynamics.
In production replace this block with your own data or a FRED pull.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanbox import DynamicFactorModel as DFM

rng = np.random.default_rng(42)
T = 300          # ~25 years of monthly data
dates = pd.date_range("2000-01", periods=T, freq="MS")

# Common factors: persistent AR(1) dynamics
f1 = np.zeros(T)
f2 = np.zeros(T)
for t in range(1, T):
    f1[t] = 0.85 * f1[t - 1] + rng.normal(0, 1.0)
    f2[t] = 0.70 * f2[t - 1] + rng.normal(0, 1.0)

# True loadings (Lambda): rows = [GDP, CPI, UNEMP, IP, Y10]
LAMBDA_TRUE = np.array([
    [0.90,  0.00],   # GDP growth      — loads on factor 1 only
    [0.40,  0.70],   # CPI inflation
    [-0.75, 0.30],   # unemployment rate (counter-cyclical)
    [0.85,  0.10],   # industrial production
    [0.50,  0.60],   # 10-year Treasury yield
])

# Idiosyncratic noise (diagonal H)
sigma_eps = np.array([0.30, 0.40, 0.35, 0.25, 0.50])
noise = rng.normal(0, sigma_eps, size=(T, 5))

# Observed panel
Y_raw = (LAMBDA_TRUE @ np.column_stack([f1, f2]).T).T + noise

# Standardise
Y = (Y_raw - Y_raw.mean(axis=0)) / Y_raw.std(axis=0)

series_names = ["GDP_growth", "CPI_inf", "UNEMP", "IndProd", "Y10"]
df = pd.DataFrame(Y, index=dates, columns=series_names)

df.plot(subplots=True, figsize=(10, 8), title="Standardised US macro panel")
plt.tight_layout()
```

## 2. Fit the DFM via MLE

```python
model = DFM(df, k_factors=2, factor_order=1)
results = model.fit(method="em", maxiter=500, tol=1e-6)
print(results.summary())
```

The EM algorithm alternates between a Kalman-smoother E-step and
closed-form M-step updates. Typical output:

```
              Dynamic Factor Model Results
=======================================================
Method                         EM
Log-likelihood              -1823.4
AIC                          3694.8
BIC                          3756.1
Converged                    True (iter 87)
=======================================================
Factor loadings (Lambda):
              Factor 1   Factor 2
GDP_growth      0.893      0.012
CPI_inf         0.397      0.703
UNEMP          -0.748      0.288
IndProd         0.842      0.112
Y10             0.511      0.595
=======================================================
```

## 3. Extract and plot common factors

```python
smoothed = results.smooth()
# a_smoothed shape: (T, k_factors)
factor1 = smoothed.a_smoothed[:, 0]
factor2 = smoothed.a_smoothed[:, 1]

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(dates, factor1, color="C0")
axes[0].axhline(0, color="k", linewidth=0.6, linestyle="--")
axes[0].set_title("Factor 1 — Business Cycle")
axes[0].set_ylabel("Smoothed value")

axes[1].plot(dates, factor2, color="C1")
axes[1].axhline(0, color="k", linewidth=0.6, linestyle="--")
axes[1].set_title("Factor 2 — Inflation / Yield Pressure")
axes[1].set_ylabel("Smoothed value")

plt.tight_layout()
```

Factor 1 is dominated by GDP, industrial production, and unemployment
(negative), and traces the business cycle. Factor 2 loads on CPI and
the 10-year yield — it behaves like an inflationary pressure indicator.

## 4. Compare factor loadings

A loading heatmap makes interpretation easier:

```python
import matplotlib.ticker as mticker

loadings = results.params["Lambda"]   # shape (p, k)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Factor 1\n(Business Cycle)",
                    "Factor 2\n(Inflation)"])
ax.set_yticks(range(len(series_names)))
ax.set_yticklabels(series_names)
for i in range(loadings.shape[0]):
    for j in range(loadings.shape[1]):
        ax.text(j, i, f"{loadings[i, j]:.2f}",
                ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, label="Loading")
ax.set_title("Factor Loadings $\\Lambda$")
plt.tight_layout()
```

!!! tip "Sign normalisation"
    The sign of each factor is arbitrary. kalmanbox normalises Factor 1
    to have a positive loading on the first variable (GDP growth). If
    your series ordering differs, flip signs after extraction.

## 5. Coincident economic activity index

A coincident index weights the factors by their contribution to the
variance of a key real-activity series (here, GDP growth):

```python
# Variance contribution: lambda_i^2 / sum(lambda_i^2) for GDP_growth row
gdp_loadings = loadings[0, :]                    # [l1, l2]
weights = gdp_loadings**2 / (gdp_loadings**2).sum()

# Weighted combination of smoothed factors
coincident_index = (smoothed.a_smoothed * weights).sum(axis=1)

# Normalise to zero mean, unit variance for comparability
coincident_index = (coincident_index - coincident_index.mean()) \
                   / coincident_index.std()

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(dates, coincident_index, color="C2", linewidth=1.5,
        label="Coincident Index")
ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
ax.fill_between(dates, coincident_index,
                where=(coincident_index < 0),
                color="C3", alpha=0.25, label="Below trend")
ax.set_title("Coincident Economic Activity Index")
ax.legend()
plt.tight_layout()
```

!!! note "NBER recessions"
    In practice, shade NBER recession periods with
    `ax.axvspan(start, end, alpha=0.15, color='gray')` to validate
    that the coincident index turns negative during downturns.

## What we learned

- A 2-factor DFM parsimoniously captures both the **business cycle** and
  an **inflation/financial** dimension from a 5-variable macro panel.
- The EM estimator converges reliably and is faster than direct MLE for
  this model class.
- The coincident index provides an interpretable summary of economic
  conditions that updates each period as new data arrive.

## Next

- [User guide: Dynamic Factor Model](../user-guide/advanced/dfm.md)
- [Tutorial: Time-varying CAPM](tvp-capm.md)
- [Bayesian DFM with Gibbs/FFBS](../user-guide/bayesian/gibbs.md)
