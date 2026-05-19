# Dynamic Factor Model (DFM)

A **Dynamic Factor Model** represents a panel of $p$ time series as being driven by a small
number $k \ll p$ of unobserved **common factors**. The key insight is that most co-movement
in multivariate economic, financial, or sensor data can be attributed to a handful of latent
drivers, while each series also carries its own idiosyncratic noise.

!!! note "When to use DFM"
    Reach for a DFM when you have **multivariate panel data** (many series, common time index)
    and you believe a low-dimensional latent structure explains much of the cross-sectional
    co-movement. Typical applications: macroeconomic nowcasting, yield-curve modelling, sensor
    fusion, portfolio risk factor analysis.

---

## 1. Concept

Consider $p$ observed time series $y_{1,t}, y_{2,t}, \ldots, y_{p,t}$ stacked into the column
vector $y_t \in \mathbb{R}^p$. In a DFM, the observed series share $k$ **common factors**
$f_t \in \mathbb{R}^k$ (with $k \ll p$) linked to the observations through the **factor
loading matrix** $\Lambda \in \mathbb{R}^{p \times k}$:

$$
y_t = \Lambda\, f_t + \varepsilon_t
$$

Each series $i$ is driven by the same set of factors, weighted by the $i$-th row of $\Lambda$,
plus a **idiosyncratic component** $\varepsilon_{i,t}$ that is uncorrelated across series.
The factors themselves evolve dynamically — typically as a vector autoregression — so that
today's factor values carry information about tomorrow's.

The central appeal of the DFM is **dimension reduction**: rather than modelling
$p(p+1)/2$ covariance parameters, the cross-sectional dependence is captured by
$p \times k$ loading parameters plus $k$ factor variances. For a panel with $p = 20$ series
and $k = 2$ factors this reduces 210 covariance parameters to 42.

---

## 2. State-Space Formulation

### Observation and transition equations

The DFM is specified by two equations:

**Observation equation** (measurement):

$$
y_t = \Lambda\, f_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, H)
$$

where $H = \operatorname{diag}(\sigma_1^2, \ldots, \sigma_p^2)$ is **diagonal** — this is
the "approximate factor model" assumption of Bai & Ng (2002): idiosyncratic shocks are
cross-sectionally uncorrelated.

**Factor transition equation**:

$$
f_t = \Phi\, f_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)
$$

where $\Phi \in \mathbb{R}^{k \times k}$ governs the factor dynamics (often diagonal for
parsimony) and $Q$ is the factor innovation covariance.

### Dimensions

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $y_t$ | $p \times 1$ | observed panel at time $t$ |
| $f_t$ | $k \times 1$ | latent common factors |
| $\Lambda$ | $p \times k$ | factor loading matrix |
| $\Phi$ | $k \times k$ | factor VAR(1) transition |
| $H$ | $p \times p$ | idiosyncratic noise covariance (diagonal) |
| $Q$ | $k \times k$ | factor innovation covariance |

### Cast as standard SSM

The Kalman filter operates on a **standard linear Gaussian SSM** with state vector
$\alpha_t$, transition matrix $T$, design matrix $Z$, selection matrix $R$, and
covariance matrices $Q$ and $H$. For the DFM the mapping is direct:

$$
\alpha_t = f_t \in \mathbb{R}^k
$$

$$
Z = \Lambda \in \mathbb{R}^{p \times k}
$$

$$
T = \Phi \in \mathbb{R}^{k \times k}
$$

$$
R = I_k \in \mathbb{R}^{k \times k}
$$

So the full SSM in standard `kalmanbox` notation is:

$$
\begin{aligned}
\alpha_{t+1} &= T\,\alpha_t + R\,\eta_t, & \eta_t &\sim \mathcal{N}(0, Q) \\
y_t          &= Z\,\alpha_t + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, H)
\end{aligned}
$$

with $T = \Phi$, $Z = \Lambda$, $R = I_k$, and $H$ diagonal.

### Higher-order factor dynamics

For a factor VAR of order $r > 1$, the $k \times 1$ factor is stacked into a companion
state of dimension $k \cdot r$:

$$
\tilde{f}_t =
\begin{pmatrix} f_t \\ f_{t-1} \\ \vdots \\ f_{t-r+1} \end{pmatrix},
\qquad
T =
\begin{pmatrix}
\Phi_1 & \Phi_2 & \cdots & \Phi_r \\
I_k   & 0      & \cdots & 0      \\
0     & I_k    & \cdots & 0      \\
\vdots &       & \ddots & \vdots \\
0     & \cdots & I_k    & 0
\end{pmatrix},
\quad
Z = \begin{pmatrix} \Lambda & 0 & \cdots & 0 \end{pmatrix}
$$

`kalmanbox` handles this expansion automatically when `factor_order > 1` is passed.

---

## 3. Identification Constraints

The DFM suffers from a fundamental **rotation indeterminacy**: for any invertible
$k \times k$ matrix $M$, the reparameterisation

$$
\Lambda^* = \Lambda M^{-1}, \qquad f_t^* = M f_t
$$

gives exactly the same observation $y_t = \Lambda^* f_t^*$. Neither $\Lambda$ nor $f_t$
is individually identified without additional restrictions.

### Normalisation strategies

Two standard conventions resolve this:

**Strategy A — Fix the factor covariance:**

$$
Q = I_k, \quad \Lambda \text{ unrestricted}
$$

The factors are standardised to unit variance and uncorrelated innovations. The loadings
absorb the scale. This is the default in `kalmanbox`.

**Strategy B — Lower-triangular restriction on $\Lambda$:**

$$
\Lambda_{ij} = 0 \text{ for } i < j, \quad Q \text{ free (positive definite)}
$$

The upper triangle of $\Lambda$ is zeroed out. This pins the rotation but requires a
specific ordering of the series: the first series loads only on factor 1, the second on
factors 1 and 2, and so on.

### Comparison of identification strategies

| | Strategy A | Strategy B |
|---|---|---|
| Factor variance | $Q = I_k$ (fixed) | $Q$ estimated |
| Loadings | unrestricted | lower-triangular |
| Series ordering | irrelevant | matters — re-order if factors are interpretable |
| Interpretation | factors have unit scale | factors have natural scale |
| Default in kalmanbox | yes (`identification="variance"`) | `identification="triangular"` |

### Practical conventions

Even after fixing the strategy above, a **sign** ambiguity remains (replacing $f_t$ with
$-f_t$ and $\Lambda$ with $-\Lambda$ leaves the likelihood unchanged). `kalmanbox`
resolves this by normalising each factor to have a positive loading on the first series that
loads on it.

A **diagonal** $\Phi$ is the most common practical choice. It reduces
$k^2$ VAR parameters to $k$, avoids Granger-causality between factors, and aids
interpretability.

!!! warning "Identification matters for inference"
    Standard errors and Wald tests for individual loadings $\Lambda_{ij}$ are only valid
    once a complete set of identification restrictions is imposed. Running DFM without
    identification constraints will produce redundant parameters and a singular Hessian.
    Always check `results.identification_valid` before interpreting parameter estimates.

---

## 4. Number of Factors Selection

Choosing $k$ is arguably the most consequential modelling decision in a DFM. Too few
factors leave common variation in the residuals; too many produce factors that are
essentially noise.

### Bai & Ng (2002) information criteria

Bai & Ng (2002) derive three panel information criteria that are consistent for fixed
$k$ as both $n$ (sample size) and $p$ (number of series) grow:

$$
\mathrm{IC}_1(k) = \log \hat\sigma^2(k) + k \cdot \frac{n+p}{np} \ln\!\left(\frac{np}{n+p}\right)
$$

$$
\mathrm{IC}_2(k) = \log \hat\sigma^2(k) + k \cdot \frac{n+p}{np} \ln\!\left(\min(n, p)\right)
$$

$$
\mathrm{IC}_3(k) = \log \hat\sigma^2(k) + k \cdot \frac{\ln(\min(n, p))}{\min(n, p)}
$$

where $\hat\sigma^2(k) = \frac{1}{np}\sum_{i=1}^{p}\sum_{t=1}^{n}
\hat\varepsilon_{it}^2(k)$ is the average residual variance when $k$ factors are
extracted.

Select $k$ as the minimiser of $\mathrm{IC}_j(k)$ over $k = 1, \ldots, k_{\max}$.
$\mathrm{IC}_2$ is the most commonly used and tends to be robust across panel dimensions.

### Scree plot approach

Compute the eigenvalues of the sample covariance matrix $\hat\Sigma_y$ of the
standardised panel. Plot them in decreasing order. The point at which the eigenvalues
"level off" (the elbow) suggests the number of true factors. This is an informal but
fast diagnostic, especially useful before running the full EM.

### Cross-validation

For forecasting applications, select $k$ by minimising the out-of-sample mean squared
forecast error. Hold out the last $h$ periods, fit the DFM on the training window for
$k = 1, \ldots, k_{\max}$, and compare $h$-step-ahead forecast accuracy across values
of $k$.

### Typical values in practice

| Application | Typical $k$ | Notes |
|-------------|------------|-------|
| Macro panel (FRED-MD, ~100 variables) | 3–8 | First 3 often capture cycle, inflation, financial |
| Small macro panel (5–15 variables) | 1–3 | Parsimony is key |
| Yield curve (10 maturities) | 2–3 | Level, slope, curvature |
| Sector returns panel | 1–5 | Market + sector factors |
| IoT sensor panel | 1–4 | Depends on physical structure |

### Code example: `factor_ic()`

```python
import numpy as np
import pandas as pd
from kalmanbox.advanced import DFM, factor_ic
from kalmanbox.datasets import load_macro_panel

panel = load_macro_panel()   # 120 months x 15 variables
y = panel.to_numpy()

# Compute Bai-Ng ICs for k = 1 ... 6
ics = factor_ic(y, k_max=6, criterion=["IC1", "IC2", "IC3"])
print(ics)
```

```
   k       IC1       IC2       IC3
0  1  0.312451  0.287634  0.301122
1  2  0.198763  0.176210  0.193540
2  3  0.197854  0.177892  0.195311
3  4  0.202341  0.189012  0.204567
4  5  0.210987  0.201234  0.218930
5  6  0.221456  0.215678  0.231204
```

```python
# Select k minimising IC2
k_opt = int(ics["IC2"].idxmin()) + 1
print(f"Optimal number of factors (IC2): {k_opt}")
# Optimal number of factors (IC2): 2
```

!!! tip "Use multiple criteria"
    When IC1, IC2, and IC3 disagree, prefer IC2. If they all agree, that is a strong
    signal. When uncertain, err on the side of one extra factor — a redundant near-zero
    factor is easier to detect post-estimation than omitted common variation.

---

## 5. Estimation

### EM algorithm (default, recommended)

`kalmanbox` estimates DFMs by default with the **Expectation-Maximisation (EM) algorithm**.
The EM exploits the fact that, conditional on the factors $f_{1:n}$, the M-step updates
for $\Lambda$, $\Phi$, $Q$, and $H$ are all available in **closed form** — no numerical
optimisation is needed.

**E-step:** Run the Kalman filter forward and the RTS smoother backward to compute:

$$
\hat{f}_{t|n} = \mathbb{E}[f_t \mid y_{1:n}], \qquad
P_{t|n} = \operatorname{Var}(f_t \mid y_{1:n}), \qquad
P_{t,t-1|n} = \operatorname{Cov}(f_t, f_{t-1} \mid y_{1:n})
$$

**M-step:** Update parameters analytically:

$$
\hat\Lambda = \left(\sum_t (y_t - \bar y)\,\hat{f}_{t|n}'\right)
              \left(\sum_t (\hat{f}_{t|n}\hat{f}_{t|n}' + P_{t|n})\right)^{-1}
$$

$$
\hat\Phi = \left(\sum_{t=2}^{n} P_{t,t-1|n} + \hat{f}_{t|n}\hat{f}_{t-1|n}'\right)
           \left(\sum_{t=2}^{n} P_{t-1|n} + \hat{f}_{t-1|n}\hat{f}_{t-1|n}'\right)^{-1}
$$

$$
\hat\sigma_i^2 = \frac{1}{n}\sum_{t=1}^{n}
\left[(y_{it} - \hat\Lambda_i \hat{f}_{t|n})^2 + \hat\Lambda_i\, P_{t|n}\, \hat\Lambda_i'\right]
$$

See [EM Algorithm](em.md) for the full derivation and convergence theory.

### Direct MLE via gradient optimisation

For small panels or when the EM is slow to converge, `kalmanbox` supports direct
maximisation of the prediction-error log-likelihood using gradient-based methods
(`method="mle"`). This calls `scipy.optimize.minimize` with L-BFGS-B internally.
See [MLE estimation](../kalman/mle.md) for the general framework.

Direct MLE is more sensitive to starting values and scaling than EM, but can be
faster for $k = 1$ or $p < 5$.

### Convergence monitoring

The EM iteration is stopped when the relative log-likelihood improvement falls below
`tol`:

$$
\frac{|\ell^{(m+1)} - \ell^{(m)}|}{1 + |\ell^{(m)}|} < \mathrm{tol}
$$

```python
results = model.fit(method="em", maxiter=500, tol=1e-8)
print(f"Converged: {results.converged}")
print(f"Iterations: {results.n_iter}")
print(f"Final log-likelihood: {results.loglike:.4f}")
```

A useful convergence diagnostic is to plot the log-likelihood across EM iterations:

```python
import matplotlib.pyplot as plt

plt.plot(results.llf_trace)
plt.xlabel("EM iteration")
plt.ylabel("Log-likelihood")
plt.title("EM convergence")
plt.tight_layout()
```

!!! tip "EM vs. direct MLE"
    For DFMs with $p > 10$ series and $k \leq 5$ factors, the EM algorithm almost always
    converges faster and more reliably than direct gradient optimisation. Use
    `method="em"` as your default and switch to `method="mle"` only if EM stalls.

---

## 6. Usage — Basic Example (Macroeconomic Indicators)

```python
import numpy as np
import pandas as pd
from kalmanbox.advanced import DFM
from kalmanbox.datasets import load_macro_panel

# Load a 120-month x 15-variable macro panel (standardised)
panel = load_macro_panel()   # 120 months x 15 variables
y = panel.to_numpy()         # shape (120, 15)

# Fit a 2-factor DFM with AR(1) factor dynamics
model = DFM(y, k_factors=2, factor_order=1)
results = model.fit(method="em", maxiter=300, tol=1e-8)

print(results.summary())
```

```
              Dynamic Factor Model Results
========================================================
Method                         EM
No. Observations               120
No. Series (p)                 15
No. Factors (k)                2
Factor order                   1
Log-likelihood             -1243.7
AIC                         2551.4
BIC                         2673.8
Converged                    True (iter 112)
========================================================
Factor loadings (Lambda):
             Factor 1  Factor 2
GDP_growth      0.887     0.015
CPI_inf         0.411     0.698
UNEMP          -0.752     0.313
IndProd         0.839     0.098
Y10             0.503     0.581
...
========================================================
```

```python
# Extract smoothed factors and loadings
sm = results.smooth()
factors = sm.factors_smoothed          # shape (120, 2)
loadings = results.params["Lambda"]    # shape (15, 2)

# Factor interpretation via loadings
df_loadings = pd.DataFrame(
    loadings,
    index=panel.columns,
    columns=["Factor 1", "Factor 2"],
)
print(df_loadings.sort_values("Factor 1", ascending=False))
```

```
            Factor 1  Factor 2
GDP_growth    0.887     0.015
IndProd       0.839     0.098
CPI_inf       0.411     0.698
Y10           0.503     0.581
UNEMP        -0.752     0.313
```

```python
# Factor transition coefficients (Phi) and idiosyncratic variances
phi = results.params["Phi"]              # shape (2, 2) — diagonal by default
h_diag = results.params["H_diag"]       # shape (15,)

print(f"Factor 1 persistence: {phi[0, 0]:.3f}")
print(f"Factor 2 persistence: {phi[1, 1]:.3f}")
```

---

## 7. Usage — Factor Selection Example

```python
from kalmanbox.advanced import DFM, factor_ic
from kalmanbox.datasets import load_macro_panel
import pandas as pd

panel = load_macro_panel()
y = panel.to_numpy()

# Compute Bai-Ng information criteria for k = 1 to 6
ics = factor_ic(y, k_max=6, criterion=["IC1", "IC2", "IC3"])
print(ics)

# Select k that minimises IC2
k_opt = int(ics["IC2"].idxmin()) + 1
print(f"\nIC2-optimal number of factors: {k_opt}")

# Fit the optimal model
model = DFM(y, k_factors=k_opt, factor_order=1)
results = model.fit(method="em", maxiter=300, tol=1e-8)
print(results.summary())
```

Visual scree plot for informal selection:

```python
import numpy as np
import matplotlib.pyplot as plt

# Eigenvalues of the sample correlation matrix
Sigma_y = np.corrcoef(y.T)              # (15, 15) correlation matrix
eigvals = np.sort(np.linalg.eigvalsh(Sigma_y))[::-1]

plt.figure(figsize=(6, 3))
plt.plot(range(1, len(eigvals) + 1), eigvals, "o-", color="C0")
plt.axvline(k_opt, color="C3", linestyle="--", label=f"IC2 optimal k={k_opt}")
plt.xlabel("Factor index")
plt.ylabel("Eigenvalue")
plt.title("Scree plot — macro panel")
plt.legend()
plt.tight_layout()
```

---

## 8. Factor Loadings Interpretation

Factor loadings are the primary tool for interpreting what each latent factor represents.

### Economic interpretation patterns

**Factor 1 — "Global" or "Level" factor:**
In macro panels, the first factor typically has large loadings of the same sign on most
real-activity series (GDP, industrial production, employment) and a negative loading on
unemployment. This pattern reflects the **business cycle** — a broad expansion or
contraction that affects all series simultaneously.

**Factor 2 — "Differential" or "Spread" factor:**
The second factor often loads positively on financial or price variables (long rates, CPI)
and weakly or negatively on real variables. It represents a different dimension of
variation — inflationary pressure, a risk premium, or a sector-specific driver.

### Rotation for interpretability

The factors extracted by EM are only identified up to an orthogonal rotation. For
interpretation, **varimax** rotation maximises the sum of variances of squared loadings,
producing sparse loadings that are easier to label:

```python
from scipy.stats import ortho_group
from sklearn.preprocessing import normalize

# Varimax rotation (requires scikit-learn >= 1.3 or scipy)
from kalmanbox.utils import varimax_rotation

Lambda_rot, R_rot = varimax_rotation(results.params["Lambda"])

df_rot = pd.DataFrame(
    Lambda_rot,
    index=panel.columns,
    columns=["Factor 1 (rotated)", "Factor 2 (rotated)"],
)
print(df_rot)
```

!!! note "Rotation does not change fit"
    Any orthogonal rotation of the factor space gives the same log-likelihood, the same
    fitted values, and the same idiosyncratic residuals. Rotation is a purely
    interpretive step; it does not affect model selection or forecasting accuracy.

### Loading heatmap

```python
import matplotlib.pyplot as plt
import numpy as np

loadings = results.params["Lambda"]    # shape (p, k)
k = loadings.shape[1]

fig, ax = plt.subplots(figsize=(3 * k, 0.5 * len(panel.columns) + 1))
im = ax.imshow(loadings, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
ax.set_xticks(range(k))
ax.set_xticklabels([f"Factor {j+1}" for j in range(k)])
ax.set_yticks(range(len(panel.columns)))
ax.set_yticklabels(panel.columns)
for i in range(loadings.shape[0]):
    for j in range(loadings.shape[1]):
        ax.text(j, i, f"{loadings[i, j]:.2f}",
                ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Loading")
ax.set_title("Factor Loadings $\\Lambda$")
plt.tight_layout()
```

---

## 9. Forecasting with DFM

### Multi-step ahead forecasts

The DFM forecasts all $p$ series jointly, propagating factor uncertainty through $\Lambda$:

```python
# h-step ahead forecasts for all p series
fc = results.forecast(steps=12)

print(fc["mean"].shape)        # (12, 15)
print(fc["lower_95"].shape)    # (12, 15)
print(fc["upper_95"].shape)    # (12, 15)

# Forecasts for one specific series
import pandas as pd
import numpy as np

series_idx = 0    # GDP growth
fc_series = pd.DataFrame({
    "mean"    : fc["mean"][:, series_idx],
    "lower_95": fc["lower_95"][:, series_idx],
    "upper_95": fc["upper_95"][:, series_idx],
})
print(fc_series)
```

### Nowcasting with ragged-edge data

A key advantage of the DFM state-space form is its ability to handle **ragged-edge
panels** — situations where some series are released with a delay and are not yet
available for the most recent periods. Missing observations at the end of the sample
are treated like any other missing data: the Kalman filter skips those equations in the
measurement update.

```python
import numpy as np
from kalmanbox.advanced import DFM

# Some series arrive with a 2-month publication lag
y_ragged = y.copy()
y_ragged[-2:, [3, 7, 11]] = np.nan   # series 3, 7, 11 missing for last 2 months

model_now = DFM(y_ragged, k_factors=2, factor_order=1)
results_now = model_now.fit(method="em")

# Filtered state at the last period: best estimate of current-period factor
kf_out = results_now.filter()
nowcast_factor = kf_out.a_filtered[-1]   # shape (2,) — current factor estimate

# Implied nowcast for any series (e.g. series 3 which was missing)
Lambda = results_now.params["Lambda"]
nowcast_gdp = Lambda[3, :] @ nowcast_factor
print(f"Nowcast for series 3: {nowcast_gdp:.4f}")
```

!!! tip "Publication-lag calendars"
    In operational nowcasting, build a lag calendar specifying the exact publication
    delay of each series and populate `y_ragged` accordingly at each release date.
    Re-running `filter()` after each new data release gives an updated nowcast.

### Forecast evaluation

```python
from kalmanbox.diagnostics import DieboldMariano

# Evaluate 1-step-ahead forecast errors out-of-sample
fc_errors = results.forecast_errors(steps=1, burn=24)   # skip first 24 months
print(fc_errors["rmse"])    # shape (15,) — RMSE per series
print(fc_errors["mae"])     # shape (15,) — MAE per series
```

---

## 10. Activity Index Example

A common use of DFM results is to construct a **coincident economic activity index**
that summarises the state of the economy in a single number.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanbox.advanced import DFM
from kalmanbox.datasets import load_macro_panel

panel = load_macro_panel()
y = panel.to_numpy()

model = DFM(y, k_factors=2, factor_order=1)
results = model.fit(method="em", maxiter=300, tol=1e-8)

sm = results.smooth()
factors = sm.factors_smoothed          # shape (120, 2)
loadings = results.params["Lambda"]    # shape (15, 2)

# --- Approach 1: First common factor as index ---
# The first factor captures the bulk of common variation
activity_index = factors[:, 0]

# --- Approach 2: Variance-weighted combination ---
# Weight each factor by its share of explained variance in the panel
var_explained = np.sum(loadings**2, axis=0)   # shape (k,)
weights = var_explained / var_explained.sum()
activity_index_weighted = factors @ weights   # scalar time series

# --- Approach 3: First-loading weighted sum ---
# Weighted by loadings of the first (most-loaded) series
w = loadings[:, 0] / np.sum(np.abs(loadings[:, 0]))
activity_index_loaded = factors[:, 0]         # factor 1 per loading convention

# Normalise to zero mean, unit variance for comparability
activity_index = (activity_index - activity_index.mean()) / activity_index.std()

dates = pd.date_range("2015-01", periods=len(activity_index), freq="MS")
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(dates, activity_index, color="C0", linewidth=1.5,
        label="Activity Index (Factor 1)")
ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
ax.fill_between(dates, activity_index,
                where=(activity_index < 0),
                color="C3", alpha=0.25, label="Below trend")
ax.set_title("Coincident Economic Activity Index")
ax.set_ylabel("Standardised units")
ax.legend()
plt.tight_layout()
```

!!! note "Interpreting the index"
    Values above zero indicate above-trend economic activity; values below zero indicate
    below-trend activity. The index is only meaningful in relative terms (higher = better)
    unless anchored to an external reference series such as real GDP growth.

---

## 11. Model Diagnostics

After fitting, check three aspects of model adequacy:

### Idiosyncratic residual autocorrelation

The idiosyncratic component $\varepsilon_{it}$ is assumed i.i.d. by the DFM. If common
factors are insufficient, autocorrelation will appear in the residuals:

```python
from kalmanbox.diagnostics import ljung_box
import pandas as pd

# Idiosyncratic residuals: observed minus factor-fitted values
sm = results.smooth()
Lambda = results.params["Lambda"]
fitted = sm.factors_smoothed @ Lambda.T       # shape (n, p)
resid = y - fitted                            # shape (n, p)

# Ljung-Box test for each series
lb_results = ljung_box(resid, lags=12, return_df=True)
print(lb_results)   # p-values per series; reject H0 = autocorrelation present
```

Series with significant Ljung-Box p-values (< 0.05) suggest either:

1. More factors are needed to capture common dynamics, or
2. The idiosyncratic component for that series is itself autocorrelated — consider
   augmenting the model with an AR(1) idiosyncratic component.

### Factor autocorrelation

The estimated factors should be well-described by the VAR($r$) transition. Check the
autocorrelation of factor **innovations** $\hat\eta_t = \hat f_t - \hat\Phi \hat f_{t-1}$:

```python
from kalmanbox.diagnostics import plot_acf

factor_innov = (sm.factors_smoothed[1:] -
                sm.factors_smoothed[:-1] @ results.params["Phi"].T)

for j in range(model.k_factors):
    plot_acf(factor_innov[:, j], lags=20, title=f"Factor {j+1} innovation ACF")
```

Significant autocorrelation in factor innovations suggests increasing `factor_order`.

### Cross-sectional residual correlation

If the diagonal-$H$ assumption is violated, residuals will be correlated across series:

```python
import numpy as np

resid_corr = np.corrcoef(resid.T)   # (p, p) correlation matrix
# Large off-diagonal entries suggest H should be non-diagonal or
# that a sector/block factor structure is needed
print(f"Max off-diagonal |corr|: {np.max(np.abs(resid_corr - np.eye(resid_corr.shape[0]))):.3f}")
```

!!! warning "Approximate factor model assumption"
    The DFM with diagonal $H$ is an *approximate* factor model — cross-sectional
    correlation in $\varepsilon_t$ is allowed to be small but not zero in large panels.
    For small panels ($p < 10$), departures from diagonality can significantly bias
    the factor estimates. Consider a block-diagonal $H$ or a factor-augmented VAR
    (FAVAR) if cross-sectional residual correlations are large.

---

## 12. API Reference

::: kalmanbox.advanced.DFM
    options:
      heading_level: 3
      show_source: false

---

## 13. Related

- [EM Algorithm](em.md) — detailed derivation of E-step and M-step updates for DFM
- [MLE estimation](../kalman/mle.md) — direct gradient-based optimisation alternative
- [Multivariate models](multivariate.md) — full-covariance multivariate SSMs
- [Bayesian estimation](../bayesian/index.md) — Gibbs / FFBS for DFM posterior inference
- [Tutorial: US Macro DFM](../../tutorials/us-macro-dfm.md) — end-to-end worked example

---

## References

- Stock, J. H., & Watson, M. W. (2002). Macroeconomic forecasting using diffusion
  indexes. *Journal of Business & Economic Statistics*, 20(2), 147–162.

- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor
  models. *Econometrica*, 70(1), 191–221.

- Doz, C., Giannone, D., & Reichlin, L. (2012). A quasi-maximum likelihood approach
  for large, approximate dynamic factor models. *Review of Economics and Statistics*,
  94(4), 1014–1024.

- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
  (2nd ed.). Oxford University Press. Chapter 8.
