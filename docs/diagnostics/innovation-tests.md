# Innovation Tests

Innovations — also called **one-step-ahead prediction errors** — are the primary diagnostic
residual in a state-space model. When the model is correctly specified, the innovations form
a Gaussian white-noise sequence. Innovation tests verify this claim.

---

## 1. Innovations: definition and properties

The Kalman filter produces innovations at each time step:

$$
v_t = y_t - Z_t a_t, \qquad t = 1, \ldots, n
$$

where $a_t = \mathbb{E}[x_t \mid y_1, \ldots, y_{t-1}]$ is the one-step-ahead state
prediction and $Z_t$ is the observation matrix.

Under correct model specification, the innovations satisfy:

$$
v_t \sim \mathcal{N}(0,\; F_t), \qquad F_t = Z_t P_t Z_t^\top + H_t
$$

where $F_t$ is the **innovation variance** (or covariance for multivariate models) and $P_t$
is the state prediction covariance. The key probabilistic properties are:

1. **Zero mean**: $\mathbb{E}[v_t] = 0$ for all $t$.
2. **Serially uncorrelated**: $\mathbb{E}[v_t v_s^\top] = 0$ for $t \neq s$.
3. **Gaussian**: $v_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0, F_t)$.
4. **Correctly scaled**: the standardised innovations $\tilde{v}_t = F_t^{-1/2} v_t$ are
   i.i.d. $\mathcal{N}(0, I)$.

!!! note "Standardised vs raw innovations"
    Most tests operate on the **standardised innovations** $\tilde{v}_t$ to remove the
    heteroscedasticity introduced by time-varying $F_t$ (common in diffuse initialisation
    and with missing observations). kalmanbox standardises automatically when `standardize=True`
    (the default).

The **log-likelihood** itself is constructed from innovations:

$$
\log L = -\frac{np}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^{n}\left(\log|F_t| + v_t^\top F_t^{-1} v_t\right)
$$

so diagnostic tests on innovations are directly testing the assumptions underlying MLE.

---

## 2. Normality tests

### 2.1 Jarque-Bera test

The most widely used omnibus test for normality based on skewness $S$ and excess kurtosis $K$:

$$
JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \xrightarrow{d} \chi^2_2
$$

where

$$
S = \frac{\frac{1}{n}\sum_t \tilde{v}_t^3}{\left(\frac{1}{n}\sum_t \tilde{v}_t^2\right)^{3/2}}, \qquad
K = \frac{\frac{1}{n}\sum_t \tilde{v}_t^4}{\left(\frac{1}{n}\sum_t \tilde{v}_t^2\right)^{2}}
$$

**Interpretation**: A large $JB$ statistic (small $p$-value) indicates departures from
normality. Check the QQ-plot to distinguish skewness (bowed QQ) from heavy tails (S-shaped
QQ). Heavy tails are common in financial data; they do not necessarily invalidate the model
but inflate standard errors of parameter estimates.

**Typical fixes**:

- Add an **outlier indicator** variable for isolated spikes.
- Use a **Student-$t$** observation error distribution.
- Switch to particle filtering for strongly non-Gaussian noise.

### 2.2 Shapiro-Wilk test

Based on the ratio of the best linear unbiased estimator of $\sigma$ to the sample standard
deviation:

$$
W = \frac{\left(\sum_t a_t \tilde{v}_{(t)}\right)^2}{\sum_t (\tilde{v}_t - \bar{\tilde{v}})^2}
$$

where $\tilde{v}_{(t)}$ are order statistics and $a_t$ are pre-computed constants.

**When to prefer Shapiro-Wilk**: higher power than Jarque-Bera for small samples
($n < 50$). Loses some power for $n > 200$.

### 2.3 Doornik-Hansen test

An improved omnibus test that corrects the finite-sample distributions of skewness and
kurtosis before combining them:

$$
DH = z_1^2 + z_2^2 \xrightarrow{d} \chi^2_2
$$

where $z_1$ and $z_2$ are normalised transforms of skewness and kurtosis respectively
(Doornik & Hansen, 2008). This test is better calibrated than Jarque-Bera for moderate
sample sizes and is the recommended normality test in kalmanbox.

---

## 3. Independence tests

Serial correlation in standardised innovations is the most serious violation — it signals
a missing dynamic component (an omitted AR lag, seasonal pattern, or state equation).

### 3.1 Ljung-Box test

$$
Q_{LB}(h) = n(n+2)\sum_{k=1}^{h}\frac{\hat{\rho}_k^2}{n-k} \xrightarrow{d} \chi^2_{h-q}
$$

where $\hat{\rho}_k = \text{corr}(\tilde{v}_t, \tilde{v}_{t-k})$ is the sample
autocorrelation at lag $k$ and $q$ is the number of estimated parameters (a degrees-of-freedom
correction recommended by Box & Pierce).

**Lag selection guidelines**:

| Sample size $n$ | Recommended $h$ |
|-----------------|----------------|
| $< 50$ | $h = \min(10, n/5)$ |
| $50$–$200$ | $h = \lfloor\log n\rfloor$ |
| $> 200$ | $h = \lfloor 2\log n\rfloor$ |

**Interpretation**: Significant $Q_{LB}$ at low lags (1–3) indicates an AR/MA component
is missing from the observation or state equation. Significance only at seasonal lags (e.g.
lag 12 for monthly data) suggests a missing seasonal pattern.

### 3.2 Box-Pierce test

A simpler variant that is asymptotically equivalent to Ljung-Box but has lower power in
small samples:

$$
Q_{BP}(h) = n\sum_{k=1}^{h}\hat{\rho}_k^2 \xrightarrow{d} \chi^2_{h-q}
$$

Ljung-Box is preferred unless you specifically need comparison with older software.

---

## 4. Homoscedasticity tests

Variance that changes over time can be caused by genuine volatility clustering (use a
stochastic volatility model) or by a missing variance-break (use TVP or a structural break
model).

### 4.1 ARCH-LM test

Engle's (1982) Lagrange Multiplier test regresses squared standardised innovations on their
own lags:

$$
\tilde{v}_t^2 = \alpha_0 + \alpha_1 \tilde{v}_{t-1}^2 + \cdots + \alpha_q \tilde{v}_{t-q}^2 + u_t
$$

The test statistic is $LM = nR^2$ from this OLS regression, distributed $\chi^2_q$ under
the null of no ARCH effects.

**Interpretation**: Rejection means volatility clustering is present. Options:

- Use the **Ensemble Kalman Filter** with an estimated variance schedule.
- Extend to a **stochastic volatility** model (available in `particlefilterbox`).
- Allow $H_t$ to be time-varying via the TVP framework.

### 4.2 Goldfeld-Quandt test

Splits the sample into three parts (typically 40 % / 20 % / 40 %), computes OLS variances
on the outer thirds, and tests the ratio:

$$
GQ = \frac{S_2^2 / df_2}{S_1^2 / df_1} \sim F(df_2, df_1)
$$

Useful for detecting a single variance break at a known or unknown breakpoint. Pair it with
CUSUM-of-squares for unknown-breakpoint detection.

---

## 5. Visual diagnostics

Beyond formal tests, always examine the following four-panel plot:

| Panel | What to look for |
|-------|-----------------|
| **Time plot** of $\tilde{v}_t$ | No trend, no clusters of large residuals, no visible pattern |
| **ACF of $\tilde{v}_t$** | All bars inside the 95 % confidence band |
| **ACF of $\tilde{v}_t^2$** | All bars inside band (no ARCH) |
| **QQ-plot** | Points on the 45° line; deviations at tails indicate fat tails |
| **Histogram** | Symmetric, bell-shaped; compare with $\mathcal{N}(0,1)$ overlay |

```python
from kalmanbox.visualization import plot_innovation_diagnostics

plot_innovation_diagnostics(results)
# Produces the 5-panel figure described above
```

---

## 6. API reference

### `innovation_tests()`

```python
from kalmanbox.diagnostics import innovation_tests

InnovationTests = innovation_tests(
    results,                 # KalmanResults object
    lags: int = None,        # Ljung-Box lags (auto if None)
    arch_lags: int = 5,      # ARCH-LM lags
    standardize: bool = True # Use F_t^{-1/2} v_t
)
```

Returns an `InnovationTestResults` object with attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.doornik_hansen` | `(stat, pval)` | Doornik-Hansen normality test |
| `.jarque_bera` | `(stat, pval)` | Jarque-Bera test |
| `.shapiro_wilk` | `(stat, pval)` | Shapiro-Wilk test |
| `.ljung_box` | `DataFrame` | LB statistic and p-value per lag |
| `.box_pierce` | `DataFrame` | BP statistic and p-value per lag |
| `.arch_lm` | `(stat, pval)` | ARCH-LM test |
| `.goldfeld_quandt` | `(stat, pval)` | Goldfeld-Quandt test |
| `.skewness` | `float` | Sample skewness |
| `.kurtosis` | `float` | Sample excess kurtosis |

```python
itr = innovation_tests(results)
print(itr.summary())
```

### `normality_test()`

```python
from kalmanbox.diagnostics import normality_test

norm = normality_test(
    results,
    method: str = "doornik-hansen",  # "doornik-hansen" | "jarque-bera" | "shapiro-wilk"
    standardize: bool = True,
)
stat, pval = norm
```

### `independence_test()`

```python
from kalmanbox.diagnostics import independence_test

ind = independence_test(
    results,
    lags: int = 20,
    method: str = "ljung-box",  # "ljung-box" | "box-pierce"
    dof_correction: int = 0,    # estimated parameters for df adjustment
    standardize: bool = True,
)
# Returns DataFrame with columns: lag, statistic, pvalue
print(ind.head())
```

### `heteroscedasticity_test()`

```python
from kalmanbox.diagnostics import heteroscedasticity_test

het = heteroscedasticity_test(
    results,
    method: str = "arch",  # "arch" | "goldfeld-quandt"
    lags: int = 5,
    standardize: bool = True,
)
stat, pval = het
```

---

## 7. Examples

### Example 1: diagnosing a Local Level model

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import innovation_tests
from kalmanbox.visualization import plot_innovation_diagnostics

# Nile river data (Harvey 1989)
from kalmanbox.datasets import load_nile
nile = load_nile()

# Fit
model = LocalLevelModel()
results = model.fit(nile)

# Run all innovation tests
itr = innovation_tests(results, lags=15, arch_lags=5)
print(itr.summary())
```

**Expected output (well-specified model)**:

```
Innovation Diagnostic Tests
===========================
Observations : 100
Parameters   : 2

Normality
  Doornik-Hansen : stat=1.23, p=0.540  [PASS]
  Jarque-Bera    : stat=2.47, p=0.290  [PASS]

Serial Independence
  Ljung-Box  Q(5)  : stat= 4.12, p=0.532  [PASS]
  Ljung-Box  Q(10) : stat= 9.87, p=0.452  [PASS]
  Ljung-Box  Q(15) : stat=16.32, p=0.362  [PASS]

Homoscedasticity
  ARCH-LM (q=5)    : stat= 3.41, p=0.636  [PASS]

Skewness : -0.14    Kurtosis (excess) : 0.27
```

**Interpretation**: All tests pass. The Local Level model is well specified for the Nile
data. The innovations are approximately normally distributed with no serial correlation
or heteroscedasticity.

---

### Example 2: diagnosing a BSM with a missing seasonal component

```python
import numpy as np
from kalmanbox import BSM, LocalLevelModel
from kalmanbox.diagnostics import innovation_tests

# Airline passengers (strongly seasonal)
from kalmanbox.datasets import load_airline
y = load_airline()

# Fit a mis-specified model (no seasonal)
bad_model = LocalLevelModel()
bad_results = bad_model.fit(np.log(y))

# Fit the correct model
good_model = BSM(period=12, stochastic_seasonal=True)
good_results = good_model.fit(np.log(y))

for label, res in [("LocalLevel (bad)", bad_results), ("BSM (good)", good_results)]:
    itr = innovation_tests(res, lags=24)
    lb_12 = itr.ljung_box.loc[12]
    jb = itr.jarque_bera
    print(f"\n{label}")
    print(f"  LB(12): Q={lb_12['statistic']:.1f}, p={lb_12['pvalue']:.3f}")
    print(f"  JB    : stat={jb[0]:.2f}, p={jb[1]:.3f}")
```

**Expected output**:

```
LocalLevel (bad)
  LB(12): Q=87.3, p=0.000   ← strong seasonal autocorrelation
  JB    : stat=14.8, p=0.001  ← non-normality from residual seasonality

BSM (good)
  LB(12): Q=11.2, p=0.511   ← seasonal structure captured
  JB    : stat=2.1, p=0.350  ← innovations approximately normal
```

**Interpretation**: The LB statistic at lag 12 collapses from 87 (strong seasonal
autocorrelation in the bad model) to 11 (no evidence of autocorrelation in the BSM). Adding
the seasonal component fixes both the independence and normality violations simultaneously,
because the residual seasonal pattern was driving both.

---

## 8. Multivariate case

For models with $p > 1$ observation series, run tests on each standardised innovation series
individually and also inspect the cross-correlation matrix:

```python
from kalmanbox.diagnostics import multivariate_innovation_tests

mitr = multivariate_innovation_tests(results)
print(mitr.per_series_summary())    # test results for each series
print(mitr.cross_correlation_matrix())  # off-diagonal = unexplained co-movement
```

Significant off-diagonal autocorrelations indicate unmodelled common dynamics — consider
adding a Dynamic Factor Model layer.

---

## Related

- [Residual analysis](residuals.md)
- [CUSUM and stability tests](cusum.md)
- [Prediction error analysis](prediction-error.md)
- [Theory: Kalman filter derivation](../theory/kalman-filter-derivation.md)
- [Theory: identifiability](../theory/identifiability.md)
- [API: diagnostics module](../api/diagnostics.md)
- [Visualization: diagnostic plots](../visualization/diagnostics.md)
