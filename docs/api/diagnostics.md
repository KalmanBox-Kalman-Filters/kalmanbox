# Diagnostics API

`kalmanbox.diagnostics`

`kalmanbox.diagnostics` provides a complete toolkit for model validation and
selection after fitting a state-space model. The module is organised into
eight functional groups that cover every stage of the diagnostic workflow,
from raw innovation tests through to forecast accuracy evaluation.

| Group | Functions | Purpose |
|---|---|---|
| [Innovation Tests](#innovation-tests) | `normality_test`, `independence_test`, `heteroscedasticity_test`, `run_all_tests` | Formal tests on the standardised innovation sequence |
| [CUSUM Tests](#cusum-tests) | `cusum`, `cusum_sq`, `plot_cusum` | Structural-stability monitoring via cumulative sums |
| [Residual Diagnostics](#residual-diagnostics) | `auxiliary_residuals`, `observation_residuals`, `state_residuals`, `state_smoothness`, `roughness` | Outlier detection, break detection, and smoothness metrics |
| [Model Selection](#model-selection) | `aic`, `bic`, `hqic`, `aicc`, `compare_models`, `likelihood_ratio_test` | Information criteria and likelihood-based comparison |
| [Cross-Validation](#cross-validation) | `cross_validate`, `rolling_cv`, `expanding_cv` | Time-series out-of-sample evaluation |
| [Filter Comparison](#filter-comparison) | `compare_filters`, `nees`, `nis`, `consistency_test` | Benchmarking and statistical consistency of competing filters |
| [Prediction Metrics](#prediction-metrics) | `prediction_errors`, `forecast_metrics`, `dm_test` | Point forecast accuracy and the Diebold–Mariano test |

See also the [Diagnostics user guide](../diagnostics/index.md) for
worked examples.

---

## Innovation Tests

The standardised one-step-ahead prediction errors (innovations) of a correctly
specified state-space model should form a sequence of independent, identically
distributed standard normals:

$$
e_t = F_t^{-1/2}\, v_t \;\overset{\text{iid}}{\sim}\; \mathcal{N}(0, 1),
\qquad t = d+1, \ldots, n
$$

where $v_t = y_t - Z_t a_{t|t-1}$ are the raw innovations, $F_t$ is the
innovation covariance, and $d$ is the number of diffuse periods discarded
during initialisation.  The three tests below check normality, serial
independence, and homoscedasticity, respectively.

---

### `normality_test`

```python
kalmanbox.diagnostics.normality_test(
    result,
    method="jarque_bera",
    lag_correction=True,
    significance=0.05,
)
```

Tests the null hypothesis that the standardised innovations are Gaussian.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult \| FilterResult` | required | Fitted model result object returned by `.fit()` or `.filter()`. |
| `method` | `str` | `"jarque_bera"` | Test statistic to use. One of `"jarque_bera"`, `"shapiro_wilk"`, `"ks"` (Kolmogorov–Smirnov), `"anderson_darling"`. |
| `lag_correction` | `bool` | `True` | Apply Bonferroni correction for the number of estimated parameters when computing the effective sample size. Recommended when `result` comes from MLE. |
| `significance` | `float` | `0.05` | Significance level used to set the `passed` flag in the result. Must be in `(0, 1)`. |

**Returns** `NormalityTestResult` — a named result object with the following fields:

| Field | Type | Description |
|---|---|---|
| `statistic` | `float` | Value of the test statistic. |
| `pvalue` | `float` | Two-tailed p-value. |
| `method` | `str` | Name of the test used. |
| `passed` | `bool` | `True` if `pvalue > significance` (fail to reject H₀). |
| `n_obs` | `int` | Number of non-diffuse innovations used. |
| `message` | `str` | Human-readable interpretation string. |

!!! note "Jarque–Bera statistic"

    The Jarque–Bera statistic combines skewness $S$ and excess kurtosis $K$:

    $$
    \text{JB} = \frac{n}{6}\left(S^2 + \frac{K^2}{4}\right)
    \;\overset{H_0}{\sim}\; \chi^2(2)
    $$

    It is the default because it has good power against the heavy-tailed
    alternatives that typically arise when the model is misspecified.

!!! warning "Small samples"

    The Shapiro–Wilk test is more powerful for $n < 50$ but is limited
    to $n \leq 5000$.  For very long series ($n > 5000$), prefer
    `"jarque_bera"` or `"anderson_darling"`.

**Example**

```python
import numpy as np
from kalmanbox import LocalLevel
from kalmanbox.diagnostics import normality_test

# Fit a Local Level model to the Nile discharge series
rng = np.random.default_rng(0)
nile = np.array([1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370,
                 1140, 995, 935, 1110, 994, 1020, 960, 1180, 799,
                 958, 1140, 1100, 1210, 1150, 1250, 1260, 1220, 1030,
                 1100, 774, 840, 874, 694, 940, 833, 701, 916, 692,
                 1020, 1050, 969, 831, 726, 456, 824, 702, 1120, 1100,
                 832, 764, 821, 768, 845, 864, 862, 698, 845, 744,
                 796, 1040, 759, 781, 865, 845, 944, 984, 897, 822,
                 1010, 771, 676, 649, 846, 812, 742, 801, 1040, 860,
                 874, 848, 890, 744, 749, 838, 1050, 918, 986, 797,
                 923, 975, 815, 1020, 906, 901, 1170, 912, 746, 919,
                 718, 714, 740])

model  = LocalLevel()
result = model.fit(nile)

test = normality_test(result, method="jarque_bera")
print(f"JB statistic : {test.statistic:.4f}")
print(f"p-value      : {test.pvalue:.4f}")
print(f"Passed H0    : {test.passed}")
print(test.message)
```

---

### `independence_test`

```python
kalmanbox.diagnostics.independence_test(
    result,
    lags=None,
    method="ljung_box",
    significance=0.05,
)
```

Tests the null hypothesis of no serial correlation in the standardised
innovations.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult \| FilterResult` | required | Fitted model result. |
| `lags` | `int \| list[int] \| None` | `None` | Lag(s) to test. A single integer tests all lags from 1 to that value; a list tests only the specified lags. If `None`, uses `min(n // 5, 10)`. |
| `method` | `str` | `"ljung_box"` | Test to apply. One of `"ljung_box"`, `"box_pierce"`, `"durbin_watson"`. |
| `significance` | `float` | `0.05` | Significance level for the `passed` verdict. |

**Returns** `IndependenceTestResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `statistic` | `float \| np.ndarray` | Test statistic at each tested lag. |
| `pvalue` | `float \| np.ndarray` | P-value(s). Array when multiple lags are tested. |
| `lags` | `np.ndarray` | Lags at which the test was evaluated. |
| `passed` | `bool` | `True` if all p-values exceed `significance`. |
| `method` | `str` | Name of the test used. |

!!! note "Degrees-of-freedom correction"

    Both the Ljung–Box and Box–Pierce statistics adjust the degrees of
    freedom for the number of estimated parameters `k`:

    $$
    Q_{\text{LB}}(h) = n(n+2)\sum_{j=1}^{h}\frac{\hat{\rho}_j^2}{n - j}
    \;\overset{H_0}{\sim}\; \chi^2(h - k)
    $$

    This correction is applied automatically when `result` carries a
    `k` attribute from MLE estimation.

**Example**

```python
from kalmanbox.diagnostics import independence_test

ind = independence_test(result, lags=[1, 5, 10], method="ljung_box")
for lag, stat, pval in zip(ind.lags, ind.statistic, ind.pvalue):
    flag = "OK" if pval > 0.05 else "FAIL"
    print(f"  lag={lag:2d}  Q={stat:6.2f}  p={pval:.3f}  [{flag}]")
```

---

### `heteroscedasticity_test`

```python
kalmanbox.diagnostics.heteroscedasticity_test(
    result,
    lags=None,
    method="arch_lm",
    significance=0.05,
)
```

Tests the null hypothesis of no ARCH effects (conditional
heteroscedasticity) in the squared standardised innovations.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |
| `lags` | `int \| None` | `None` | Number of ARCH lags to include in the auxiliary regression. Defaults to `min(n // 5, 5)`. |
| `method` | `str` | `"arch_lm"` | Test method. One of `"arch_lm"` (Engle's LM test) or `"goldfeld_quandt"` (variance ratio across sub-samples). |
| `significance` | `float` | `0.05` | Significance level. |

**Returns** `HeteroscedasticityTestResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `statistic` | `float` | LM statistic (ARCH-LM) or F statistic (Goldfeld–Quandt). |
| `pvalue` | `float` | P-value under the null of no ARCH effects. |
| `passed` | `bool` | `True` if `pvalue > significance`. |
| `method` | `str` | Test name. |
| `lags` | `int` | Number of lags used. |

**Example**

```python
from kalmanbox.diagnostics import heteroscedasticity_test

het = heteroscedasticity_test(result, lags=5)
print(f"ARCH-LM({het.lags}): stat={het.statistic:.4f}  p={het.pvalue:.4f}")
print("Homoscedastic:", het.passed)
```

---

### `run_all_tests`

```python
kalmanbox.diagnostics.run_all_tests(result)
```

Convenience wrapper that runs normality (Jarque–Bera), independence
(Ljung–Box, `lags=5`), and heteroscedasticity (ARCH-LM, `lags=5`) tests
in a single call and returns a consolidated summary table.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult \| FilterResult` | required | Fitted model result. |

**Returns** `pd.DataFrame` with columns:

| Column | Description |
|---|---|
| `test` | Test name string. |
| `statistic` | Scalar test statistic. |
| `pvalue` | P-value. |
| `passed` | Boolean verdict. |

**Example**

```python
import pandas as pd
from kalmanbox.diagnostics import run_all_tests

summary = run_all_tests(result)
print(summary.to_string(index=False))
# test                statistic   pvalue  passed
# jarque_bera             1.432   0.4888    True
# ljung_box(5)            4.211   0.5192    True
# arch_lm(5)              2.874   0.7194    True
```

!!! tip

    Call `run_all_tests` first to get a quick overview, then dive into
    individual test functions for detailed output when any test fails.

---

## CUSUM Tests

CUSUM (cumulative sum) statistics monitor whether the innovation sequence is
structurally stable over time.  A model whose parameters have shifted
mid-sample will produce innovations with a non-zero mean in the post-break
segment, causing the CUSUM path to drift outside the significance bounds.

The recursive CUSUM statistic is defined as:

$$
W_t = \frac{1}{\hat{\sigma}} \sum_{s=d+1}^{t} e_s, \qquad t = d+1, \ldots, n
$$

where $\hat{\sigma}$ is the sample standard deviation of the full innovation
sequence and $d$ is the number of diffuse periods. Under the null hypothesis
of structural stability, $W_t$ stays within the 5 % significance bounds:

$$
\pm\, c_\alpha \sqrt{n - d}, \qquad c_{0.05} = 0.948,\quad c_{0.01} = 1.143,\quad c_{0.10} = 0.850
$$

---

### `cusum`

```python
kalmanbox.diagnostics.cusum(
    result,
    significance=0.05,
)
```

Computes the recursive CUSUM of standardised innovations and tests whether
the path exceeds the significance bounds.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result providing the innovation sequence. |
| `significance` | `float` | `0.05` | Significance level for the critical bounds. Supported values: `0.01`, `0.05`, `0.10`. |

**Returns** `CUSUMResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `cusum_values` | `np.ndarray` shape `(n,)` | Cumulative sum path $W_t$, one entry per observation. |
| `upper_bound` | `float` | Positive significance boundary $+c_\alpha\sqrt{n-d}$. |
| `lower_bound` | `float` | Negative significance boundary $-c_\alpha\sqrt{n-d}$. |
| `break_detected` | `bool` | `True` if the path exits the bounds at any point. |
| `break_index` | `int \| None` | Index of the first crossing, or `None` if no break. |
| `significance` | `float` | Significance level used. |

**Example**

```python
from kalmanbox.diagnostics import cusum, plot_cusum

cs = cusum(result, significance=0.05)
print(f"Break detected : {cs.break_detected}")
if cs.break_index is not None:
    print(f"First crossing at index: {cs.break_index}")

fig = plot_cusum(cs)
fig.savefig("cusum.png", dpi=150, bbox_inches="tight")
```

---

### `cusum_sq`

```python
kalmanbox.diagnostics.cusum_sq(
    result,
    significance=0.05,
)
```

CUSUM-of-squares test for time-varying variance (heteroscedasticity).
Accumulates squared standardised innovations and compares the path against
the Brown–Durbin–Evans bounds for a variance change-point.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |
| `significance` | `float` | `0.05` | Significance level. Supported values: `0.01`, `0.05`, `0.10`. |

**Returns** `CUSUMResult` with the same fields as `cusum()`. The
`cusum_values` field now contains the normalised squared-residual path
$\sum_{s} e_s^2 / \sum_{s=d+1}^{n} e_s^2$.

!!! warning "Interpretation"

    A break in `cusum_sq` indicates a change in innovation variance, not
    necessarily a change in the model parameters. Use together with
    `cusum()` to distinguish the two types of instability.

**Example**

```python
from kalmanbox.diagnostics import cusum_sq

cs2 = cusum_sq(result)
print(f"Variance instability: {cs2.break_detected}")
```

---

### `plot_cusum`

```python
kalmanbox.diagnostics.plot_cusum(
    cusum_result,
    ax=None,
    figsize=(10, 4),
    title=None,
)
```

Plot the CUSUM (or CUSUM-of-squares) path together with the significance
bounds as shaded bands.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cusum_result` | `CUSUMResult` | required | Result object from `cusum()` or `cusum_sq()`. |
| `ax` | `matplotlib.axes.Axes \| None` | `None` | Axes to draw on. If `None`, a new figure is created with the given `figsize`. |
| `figsize` | `tuple[int, int]` | `(10, 4)` | Figure size in inches, used only when `ax=None`. |
| `title` | `str \| None` | `None` | Plot title. Auto-generated from the result type when `None`. |

**Returns** `matplotlib.figure.Figure` — the parent figure of the axes,
whether newly created or pre-existing.

---

## Residual Diagnostics

The functions in this group decompose the residuals into observation-level
and state-level components, enabling outlier detection and structural-break
localisation.

---

### `auxiliary_residuals`

```python
kalmanbox.diagnostics.auxiliary_residuals(result)
```

Computes Koopman's (1993) auxiliary residuals from the disturbance smoother.
These residuals are designed to have unit variance under the null hypothesis
of no outliers or breaks, making them directly comparable across time and
across state components.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. The model must have been estimated with a full smoothing pass (`smooth=True`, which is the default). |

**Returns** `AuxResidualResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `observation_residuals` | `np.ndarray` shape `(n,)` | Standardised observation disturbance residuals. Values with absolute value > 3 flag potential outliers. |
| `state_residuals` | `np.ndarray` shape `(n, m)` | Standardised state disturbance residuals. Spikes flag potential structural breaks. |
| `observation_residuals_std` | `np.ndarray` shape `(n,)` | Standard deviations of the observation disturbance residuals. |
| `state_residuals_std` | `np.ndarray` shape `(n, m)` | Standard deviations of the state disturbance residuals. |
| `observation_outliers` | `np.ndarray[bool]` shape `(n,)` | `True` at time points where `|obs_residual| > 3`. |
| `state_breaks` | `np.ndarray[bool]` shape `(n, m)` | `True` at time points where `|state_residual| > 3` for each state component. |

!!! note "Koopman auxiliary residuals"

    The observation auxiliary residual at time $t$ is:

    $$
    \tilde{u}_t = \frac{H_t \varepsilon_t^*}{\sqrt{D_t}}
    $$

    where $\varepsilon_t^*$ is the smoothed observation disturbance and
    $D_t$ is its variance as computed by the disturbance smoother.
    See Koopman (1993) for the full derivation.

**Example**

```python
import matplotlib.pyplot as plt
from kalmanbox.diagnostics import auxiliary_residuals

aux = auxiliary_residuals(result)

print(f"Observation outliers at: {np.where(aux.observation_outliers)[0]}")
print(f"State breaks at: {np.where(aux.state_breaks)[0]}")

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(aux.observation_residuals, label="Obs residuals")
axes[0].axhline( 3, color="red", linestyle="--")
axes[0].axhline(-3, color="red", linestyle="--")
axes[0].set_title("Observation Auxiliary Residuals")

axes[1].plot(aux.state_residuals[:, 0], label="State 0")
axes[1].axhline( 3, color="red", linestyle="--")
axes[1].axhline(-3, color="red", linestyle="--")
axes[1].set_title("State Auxiliary Residuals (component 0)")
plt.tight_layout()
```

---

### `observation_residuals`

```python
kalmanbox.diagnostics.observation_residuals(
    result,
    standardise=True,
)
```

Returns the one-step-ahead prediction errors $v_t = y_t - Z_t a_{t|t-1}$,
optionally standardised by the square root of the innovation covariance.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |
| `standardise` | `bool` | `True` | If `True`, returns $e_t = F_t^{-1/2} v_t$. If `False`, returns the raw innovations $v_t$. |

**Returns** `np.ndarray` of shape `(n,)` for univariate models or
`(n, p)` for multivariate models.

**Example**

```python
from kalmanbox.diagnostics import observation_residuals
import matplotlib.pyplot as plt

e = observation_residuals(result, standardise=True)
plt.figure(figsize=(10, 3))
plt.plot(e)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Standardised innovations")
plt.show()
```

---

### `state_residuals`

```python
kalmanbox.diagnostics.state_residuals(
    result,
    standardise=True,
)
```

Returns the smoothed state disturbance residuals $\hat{\eta}_t = E[\eta_t | Y_n]$
from the disturbance smoother.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result with smoothing. |
| `standardise` | `bool` | `True` | Divide by the square root of the smoothed disturbance variance $\text{Var}(\hat{\eta}_t)$. |

**Returns** `np.ndarray` of shape `(n, r)` where `r` is the number of
state disturbance components.

---

### `state_smoothness`

```python
kalmanbox.diagnostics.state_smoothness(result)
```

Measures the smoothness of the estimated state trajectory using first and
second differences.  A very rough trajectory may indicate an over-fitted
model or a misspecified state noise covariance $Q$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result with smoothing pass. |

**Returns** `SmoothnessResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `roughness` | `float` | Mean squared second difference of the smoothed state. Lower is smoother. |
| `roughness_ratio` | `float` | Roughness relative to the variance of the state; dimensionless. |
| `effective_knots` | `int` | Approximate number of effective changepoints in the state trajectory. |

---

### `roughness`

```python
kalmanbox.diagnostics.roughness(state_sequence)
```

Compute the roughness index of a state or signal sequence, defined as the
mean squared second difference:

$$
\mathcal{R}(a) = \frac{1}{n-2} \sum_{t=3}^{n} \bigl(\Delta^2 a_t\bigr)^2,
\qquad \Delta^2 a_t = a_t - 2a_{t-1} + a_{t-2}
$$

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `state_sequence` | `np.ndarray` | required | State trajectory. Shape `(n,)` for a scalar state or `(n, m)` for a vector state. |

**Returns** `float` (scalar state) or `np.ndarray` of shape `(m,)` (vector
state) — the per-component roughness indices.

**Example**

```python
from kalmanbox.diagnostics import roughness

a_smooth = result.a_smooth  # shape (n, m)
r = roughness(a_smooth)
print(f"Roughness per component: {r}")
```

---

## Model Selection

Information criteria penalise the log-likelihood for model complexity.  All
criteria below are computed from the prediction-error decomposition
log-likelihood of the state-space model, using only the non-diffuse
observations.

$$
\ell = -\frac{n^*}{2}\ln(2\pi) - \frac{1}{2}\sum_{t=d+1}^{n}\bigl(\ln|F_t| + v_t'F_t^{-1}v_t\bigr)
$$

where $n^* = n - d$ and $k$ is the number of free parameters.

---

### `aic`

```python
kalmanbox.diagnostics.aic(result)
```

Akaike Information Criterion: $\text{AIC} = -2\ell + 2k$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result carrying `loglikelihood` and `k` attributes. |

**Returns** `float` — AIC value.  Lower is better.

---

### `bic`

```python
kalmanbox.diagnostics.bic(result)
```

Bayesian Information Criterion: $\text{BIC} = -2\ell + k\ln n^*$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |

**Returns** `float` — BIC value.  Lower is better.

!!! note "BIC vs AIC"

    The BIC imposes a stronger penalty than the AIC for large $n$, making
    it more consistent (selects the true model asymptotically if it is in
    the candidate set) but potentially over-parsimonious for finite samples.

---

### `hqic`

```python
kalmanbox.diagnostics.hqic(result)
```

Hannan–Quinn Information Criterion: $\text{HQIC} = -2\ell + 2k\ln(\ln n^*)$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |

**Returns** `float` — HQIC value.  Lower is better.

---

### `aicc`

```python
kalmanbox.diagnostics.aicc(result)
```

Bias-corrected AIC for small samples:

$$
\text{AIC}_c = \text{AIC} + \frac{2k(k+1)}{n^* - k - 1}
$$

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |

**Returns** `float` — AICc value.  Lower is better.

!!! tip "When to use AICc"

    Use AICc whenever $n^* / k < 40$.  The correction is negligible for
    large samples but can be substantial when the model is large relative
    to the sample.

**Example — computing all four criteria**

```python
from kalmanbox.diagnostics import aic, bic, hqic, aicc

print(f"AIC  = {aic(result):.4f}")
print(f"BIC  = {bic(result):.4f}")
print(f"HQIC = {hqic(result):.4f}")
print(f"AICc = {aicc(result):.4f}")
```

---

### `compare_models`

```python
kalmanbox.diagnostics.compare_models(
    *results,
    criterion="aic",
    sort=True,
)
```

Construct a comparison table for a collection of fitted models, including
Akaike weights that quantify relative model support.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `*results` | `FitResult` | required | Two or more fitted model result objects. Each must have a `name` attribute or the model class name is used. |
| `criterion` | `str` | `"aic"` | Criterion used to compute $\Delta$ and sort the table. One of `"aic"`, `"bic"`, `"hqic"`, `"aicc"`, `"loglik"`. |
| `sort` | `bool` | `True` | Sort the table in ascending order of the criterion (descending for `"loglik"`). |

**Returns** `pd.DataFrame` with one row per model and the following columns:

| Column | Description |
|---|---|
| `model` | Model name. |
| `loglik` | Log-likelihood $\ell$. |
| `k` | Number of free parameters. |
| `n` | Effective number of observations $n^*$. |
| `aic` | AIC value. |
| `bic` | BIC value. |
| `hqic` | HQIC value. |
| `aicc` | AICc value. |
| `Δ_criterion` | Difference from the best model on the selected criterion. |
| `weight` | Akaike weight $w_i = \exp(-\Delta_i/2) / \sum_j \exp(-\Delta_j/2)$. |

**Example**

```python
from kalmanbox import LocalLevel, LocalLinearTrend, BSM
from kalmanbox.diagnostics import compare_models

ll_result  = LocalLevel().fit(nile)
llt_result = LocalLinearTrend().fit(nile)
bsm_result = BSM(period=12).fit(nile)

table = compare_models(ll_result, llt_result, bsm_result, criterion="bic")
print(table[["model", "loglik", "k", "bic", "Δ_criterion", "weight"]].to_string(index=False))
```

---

### `likelihood_ratio_test`

```python
kalmanbox.diagnostics.likelihood_ratio_test(
    result_restricted,
    result_unrestricted,
    df=None,
)
```

Wilks likelihood ratio test for comparing a restricted model against its
unrestricted (larger) counterpart.  Under $H_0$ (restricted model is true):

$$
\Lambda = -2\bigl(\ell_r - \ell_u\bigr) \;\overset{H_0}{\sim}\; \chi^2(q)
$$

where $q$ is the number of restrictions (difference in free parameters).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result_restricted` | `FitResult` | required | Result from the smaller (restricted) model. |
| `result_unrestricted` | `FitResult` | required | Result from the larger (unrestricted) model. |
| `df` | `int \| None` | `None` | Degrees of freedom for the $\chi^2$ distribution. If `None`, computed as `k_unrestricted - k_restricted`. |

**Returns** `LRTResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `statistic` | `float` | LR statistic $\Lambda = -2(\ell_r - \ell_u)$. |
| `pvalue` | `float` | P-value from the $\chi^2(q)$ distribution. |
| `df` | `int` | Degrees of freedom used. |
| `passed` | `bool` | `True` if `pvalue > 0.05` (restricted model is not rejected). |

!!! warning "Boundary testing"

    When a restriction places a parameter on the boundary of its feasible
    region (e.g. testing $\sigma_\eta^2 = 0$), the standard $\chi^2$
    distribution is conservative.  Use a mixture distribution
    $\tfrac{1}{2}\chi^2(0) + \tfrac{1}{2}\chi^2(1)$ in that case.

**Example**

```python
from kalmanbox.diagnostics import likelihood_ratio_test

# Is the slope variance in the LLT significantly different from zero?
lrt = likelihood_ratio_test(ll_result, llt_result)
print(f"LR stat = {lrt.statistic:.4f},  df = {lrt.df},  p = {lrt.pvalue:.4f}")
print("Reject local level in favour of LLT:", not lrt.passed)
```

---

## Cross-Validation

Time-series cross-validation produces honest out-of-sample error estimates
by training on past data only and evaluating on future observations.
All functions in this group refit the model on each training fold using
the same optimisation settings as the original `model.fit()` call.

---

### `cross_validate`

```python
kalmanbox.diagnostics.cross_validate(
    model,
    y,
    method="rolling",
    min_train=None,
    step=1,
    horizon=1,
    metrics=None,
)
```

General time-series cross-validation driver supporting both rolling and
expanding window schemes.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | kalmanbox model | required | An unfitted kalmanbox model object (e.g. `LocalLevel()`, `BSM(period=12)`). The model will be cloned and refitted on each fold. |
| `y` | `np.ndarray` shape `(n,)` | required | Full observation sequence. |
| `method` | `str` | `"rolling"` | CV scheme. `"rolling"` uses a fixed-size training window; `"expanding"` grows the window from `min_train` to `n - horizon`. |
| `min_train` | `int \| None` | `None` | Minimum training set size. Defaults to `max(20, int(0.2 * n))`. |
| `step` | `int` | `1` | Number of observations to advance the window between folds. Larger values reduce computation at the cost of fewer folds. |
| `horizon` | `int` | `1` | Forecast horizon; predictions are made `horizon` steps ahead of the training window. |
| `metrics` | `list[str] \| None` | `None` | Metrics to compute. Defaults to `["rmse", "mae", "mape"]`. Supported: `"rmse"`, `"mae"`, `"mape"`, `"smape"`, `"coverage_95"`. |

**Returns** `CVResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `scores` | `dict[str, np.ndarray]` | Per-fold scores for each metric. Keys are metric names; values are 1-D arrays of length `n_splits`. |
| `mean_scores` | `dict[str, float]` | Mean score across folds for each metric. |
| `std_scores` | `dict[str, float]` | Standard deviation of scores across folds. |
| `n_splits` | `int` | Total number of CV folds evaluated. |

**Example**

```python
from kalmanbox import LocalLevel
from kalmanbox.diagnostics import cross_validate

model = LocalLevel()
cv = cross_validate(
    model, nile,
    method="expanding",
    min_train=50,
    horizon=1,
    metrics=["rmse", "mae"],
)
print(f"CV RMSE: {cv.mean_scores['rmse']:.2f} ± {cv.std_scores['rmse']:.2f}")
print(f"CV MAE : {cv.mean_scores['mae']:.2f} ± {cv.std_scores['mae']:.2f}")
print(f"Folds  : {cv.n_splits}")
```

---

### `rolling_cv`

```python
kalmanbox.diagnostics.rolling_cv(
    model,
    y,
    window,
    horizon=1,
)
```

Fixed-window rolling cross-validation: the training window has exactly
`window` observations and slides forward by one step at each fold.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | kalmanbox model | required | Unfitted model object. |
| `y` | `np.ndarray` shape `(n,)` | required | Observation sequence. |
| `window` | `int` | required | Fixed training window size. Must satisfy `window < n - horizon`. |
| `horizon` | `int` | `1` | Forecast horizon in steps. |

**Returns** `CVResult` (same structure as `cross_validate`).

**Example**

```python
from kalmanbox.diagnostics import rolling_cv

rcv = rolling_cv(LocalLevel(), nile, window=50, horizon=3)
print(f"Rolling CV RMSE (h=3): {rcv.mean_scores['rmse']:.2f}")
```

---

### `expanding_cv`

```python
kalmanbox.diagnostics.expanding_cv(
    model,
    y,
    min_train=None,
    horizon=1,
)
```

Expanding-window (pseudo-out-of-sample) cross-validation.  The training
set grows from `min_train` to `n - horizon`, adding one observation per fold.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | kalmanbox model | required | Unfitted model object. |
| `y` | `np.ndarray` shape `(n,)` | required | Observation sequence. |
| `min_train` | `int \| None` | `None` | Initial training size. Defaults to `max(20, int(0.2 * n))`. |
| `horizon` | `int` | `1` | Forecast horizon in steps. |

**Returns** `CVResult`.

!!! tip "Expanding vs rolling"

    Use expanding windows when you believe the full historical record is
    informative (e.g. long-run trend models).  Use rolling windows when
    the data-generating process may have changed and older data would
    hurt forecast accuracy.

---

## Filter Comparison

These utilities benchmark multiple filter objects on the same observation
sequence and compute statistical metrics that assess whether a filter's
uncertainty estimates are consistent with the true errors.

---

### `compare_filters`

```python
kalmanbox.diagnostics.compare_filters(
    y,
    filters,
    metrics=None,
    a0=None,
    P0=None,
)
```

Run a set of filter objects on the same observation sequence and collect
performance metrics into a comparison table.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` shape `(n, p)` | required | Observation sequence. Use shape `(n, 1)` for univariate. |
| `filters` | `dict[str, filter_object]` | required | Mapping from a label string to an initialised (but not yet run) filter object. Example: `{"KF": kf, "UKF": ukf, "SR": sqf}`. |
| `metrics` | `list[str] \| None` | `None` | Metrics to compute. If `None`, uses `["rmse", "loglik", "time_ms", "nees"]`. Supported values: `"rmse"`, `"loglik"`, `"time_ms"`, `"nees"`, `"nis"`, `"max_eigenvalue"`. |
| `a0` | `np.ndarray \| None` | `None` | Shared initial state mean. Shape `(m,)`. If `None`, each filter uses its own default. |
| `P0` | `np.ndarray \| None` | `None` | Shared initial covariance. Shape `(m, m)`. If `None`, each filter uses its own default. |

**Returns** `pd.DataFrame` with one row per filter and one column per
metric.  Rows are sorted by log-likelihood in descending order by default.

**Example**

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation
from kalmanbox.filters import KalmanFilter, SquareRootFilter, InformationFilter
from kalmanbox.diagnostics import compare_filters

ss = StateSpaceRepresentation(
    Z=np.array([[1.0, 0.0]]),
    T=np.array([[1.0, 1.0], [0.0, 1.0]]),
    R=np.eye(2),
    H=np.array([[4.0]]),
    Q=np.diag([0.5, 0.05]),
)
rng = np.random.default_rng(1)
y = rng.standard_normal((200, 1)) * 2.0

filters = {
    "KalmanFilter"    : KalmanFilter(ss),
    "SquareRoot"      : SquareRootFilter(ss),
    "InformationFilter": InformationFilter(ss),
}
table = compare_filters(y, filters, metrics=["loglik", "time_ms"])
print(table)
```

---

### `nees`

```python
kalmanbox.diagnostics.nees(a_true, a_filt, P_filt)
```

Normalised Estimation Error Squared (NEES) measures how well the filter's
covariance calibrates the estimation error:

$$
\text{NEES}_t = (a_t - \hat{a}_{t|t})' P_{t|t}^{-1} (a_t - \hat{a}_{t|t})
\;\overset{H_0}{\sim}\; \chi^2(m)
$$

A consistent filter has $E[\text{NEES}_t] = m$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `a_true` | `np.ndarray` shape `(m,)` or `(n, m)` | required | True state vector(s). |
| `a_filt` | `np.ndarray` shape `(m,)` or `(n, m)` | required | Filtered state estimate(s). Same shape as `a_true`. |
| `P_filt` | `np.ndarray` shape `(m, m)` or `(n, m, m)` | required | Filtered covariance matrix (or sequence of matrices). |

**Returns** `float` (single time step) or `np.ndarray` of shape `(n,)` (sequence).

**Example**

```python
from kalmanbox.diagnostics import nees

# a_true: ground-truth states from a simulation
nees_seq = nees(a_true[1:], result.a_filt, result.P_filt)
print(f"Mean NEES: {nees_seq.mean():.3f}  (expected {m:.1f})")
```

---

### `nis`

```python
kalmanbox.diagnostics.nis(y, y_pred, F)
```

Normalised Innovation Squared (NIS) measures whether the predicted
observation covariance $F_t$ correctly calibrates the forecast error:

$$
\text{NIS}_t = v_t' F_t^{-1} v_t
\;\overset{H_0}{\sim}\; \chi^2(p)
$$

A consistent filter has $E[\text{NIS}_t] = p$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray` shape `(p,)` or `(n, p)` | required | Observed value(s). |
| `y_pred` | `np.ndarray` shape `(p,)` or `(n, p)` | required | Predicted observation(s). Same shape as `y`. |
| `F` | `np.ndarray` shape `(p, p)` or `(n, p, p)` | required | Innovation covariance matrix or sequence. |

**Returns** `float` (single time step) or `np.ndarray` of shape `(n,)`.

---

### `consistency_test`

```python
kalmanbox.diagnostics.consistency_test(
    filter_result,
    a_true=None,
    significance=0.05,
)
```

Chi-squared test for filter consistency using NEES (when true states are
known) or NIS (from the filter's own innovations and predicted observation
covariances).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter_result` | `FilterResult` | required | Result from `filter.filter()`, carrying `a_filt`, `P_filt`, `v`, and `F`. |
| `a_true` | `np.ndarray \| None` | `None` | True state sequence, shape `(n, m)`. If provided, computes NEES; otherwise computes NIS from innovations stored in `filter_result`. |
| `significance` | `float` | `0.05` | Two-sided significance level for the chi-squared test. |

**Returns** `ConsistencyTestResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `statistic` | `float` | Mean NEES or NIS over the non-diffuse period. |
| `pvalue` | `float` | P-value from the $\chi^2$ test. |
| `passed` | `bool` | `True` if the mean statistic falls within the `(1-significance)` chi-squared confidence interval. |
| `bounds` | `tuple[float, float]` | Lower and upper acceptance bounds at the given significance level. |

**Example**

```python
from kalmanbox.diagnostics import consistency_test

ct = consistency_test(filter_result, a_true=a_true[1:])
print(f"NEES = {ct.statistic:.3f}  bounds = {ct.bounds}  consistent = {ct.passed}")
```

!!! warning "NIS-only consistency"

    When `a_true` is unavailable (the typical operational case), only
    NIS-based consistency can be assessed.  NIS consistency is a necessary
    but not sufficient condition for NEES consistency.

---

## Prediction Metrics

---

### `prediction_errors`

```python
kalmanbox.diagnostics.prediction_errors(
    result,
    y_test=None,
    horizon=1,
)
```

Compute multi-step prediction errors from a fitted model.  When `y_test`
is provided, the model is used to forecast `horizon` steps beyond the
training data and the errors are computed against the held-out observations.
When `y_test` is `None`, in-sample one-step-ahead prediction errors are
returned (equivalent to `observation_residuals(result, standardise=False)`).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |
| `y_test` | `np.ndarray \| None` | `None` | Held-out observations of shape `(n_test,)`. If `None`, uses in-sample innovations. |
| `horizon` | `int` | `1` | Forecast horizon when `y_test` is provided. Must satisfy `horizon >= 1`. |

**Returns** `dict` with the following keys:

| Key | Type | Description |
|---|---|---|
| `"errors"` | `np.ndarray` | Signed prediction errors $\hat{y} - y$. |
| `"abs_errors"` | `np.ndarray` | Absolute errors $|\hat{y} - y|$. |
| `"sq_errors"` | `np.ndarray` | Squared errors $(\hat{y} - y)^2$. |
| `"coverage_95"` | `float` | Fraction of observations falling within the 95 % prediction interval. |

---

### `forecast_metrics`

```python
kalmanbox.diagnostics.forecast_metrics(
    result,
    y_test,
    horizon=1,
)
```

Compute a standard battery of point-forecast accuracy metrics and
prediction-interval coverage.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | Fitted model result. |
| `y_test` | `np.ndarray` shape `(n_test,)` | required | Held-out observations. |
| `horizon` | `int` | `1` | Forecast horizon in steps. |

**Returns** `dict[str, float]` with the following keys:

| Key | Description |
|---|---|
| `"rmse"` | Root mean squared error $\sqrt{\text{MSE}}$. |
| `"mae"` | Mean absolute error. |
| `"mape"` | Mean absolute percentage error (percent). |
| `"smape"` | Symmetric MAPE (percent). |
| `"coverage_50"` | Fraction of observations within the 50 % prediction interval. |
| `"coverage_95"` | Fraction of observations within the 95 % prediction interval. |

**Example**

```python
from kalmanbox import LocalLevel
from kalmanbox.diagnostics import forecast_metrics

y_train, y_test = nile[:80], nile[80:]

result = LocalLevel().fit(y_train)
metrics = forecast_metrics(result, y_test, horizon=1)

for name, val in metrics.items():
    print(f"  {name:<14} {val:.4f}")
```

---

### `dm_test`

```python
kalmanbox.diagnostics.dm_test(
    e1,
    e2,
    horizon=1,
    method="harvey",
)
```

Diebold–Mariano test for the null hypothesis of equal predictive accuracy
between two competing forecast models.

The test statistic is based on the loss differential $d_t = L(e_{1,t}) - L(e_{2,t})$
where $L(\cdot) = (\cdot)^2$ (squared-error loss by default).  Under $H_0$,
$E[d_t] = 0$.

$$
\text{DM} = \frac{\bar{d}}{\sqrt{(LRV_d + \delta_n) / n}}
\;\overset{H_0}{\longrightarrow}\; \mathcal{N}(0,1)
$$

where $LRV_d$ is the long-run variance of $d_t$ estimated via Newey–West,
and $\delta_n$ is Harvey's (1997) small-sample correction.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `e1` | `np.ndarray` shape `(n,)` | required | Forecast errors from model 1. |
| `e2` | `np.ndarray` shape `(n,)` | required | Forecast errors from model 2. Must be the same length as `e1`. |
| `horizon` | `int` | `1` | Forecast horizon used to set the Newey–West bandwidth to `horizon - 1`. |
| `method` | `str` | `"harvey"` | `"harvey"` applies the small-sample $t$-correction of Harvey et al. (1997); `"original"` uses the asymptotic normal distribution. |

**Returns** `DMTestResult` — named result object:

| Field | Type | Description |
|---|---|---|
| `statistic` | `float` | DM test statistic. Negative values favour model 1. |
| `pvalue` | `float` | Two-sided p-value under $H_0$ of equal accuracy. |
| `better_model` | `int` | `1` if model 1 is significantly more accurate, `2` if model 2, `0` if no significant difference. |

!!! note "Loss function"

    The default loss is squared error. To use absolute-error loss, pass
    the absolute forecast errors directly: `dm_test(np.abs(e1), np.abs(e2))`.
    Ensure both error arrays are computed on the same held-out sample.

**Example**

```python
import numpy as np
from kalmanbox import LocalLevel, LocalLinearTrend
from kalmanbox.diagnostics import dm_test

y_train, y_test = nile[:80], nile[80:]

r1 = LocalLevel().fit(y_train)
r2 = LocalLinearTrend().fit(y_train)

# h-step-ahead predictions
from kalmanbox.diagnostics import prediction_errors
e1 = prediction_errors(r1, y_test, horizon=1)["errors"]
e2 = prediction_errors(r2, y_test, horizon=1)["errors"]

dm = dm_test(e1, e2, horizon=1, method="harvey")
print(f"DM stat   = {dm.statistic:.4f}")
print(f"p-value   = {dm.pvalue:.4f}")
print(f"Better    = model {dm.better_model} (0 = no significant difference)")
```

---

## Full Diagnostic Workflow

The example below demonstrates a complete diagnostic cycle: fitting a Basic
Structural Model (BSM) to the classic airline passenger series, running all
innovation tests, comparing candidate models by BIC, and saving a CUSUM plot.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kalmanbox import LocalLevel, LocalLinearTrend, BSM
from kalmanbox.diagnostics import (
    run_all_tests,
    cusum,
    cusum_sq,
    plot_cusum,
    auxiliary_residuals,
    compare_models,
    forecast_metrics,
)

# ------------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------------
# Airline monthly passengers 1949–1960, log-transformed
airline_log = np.log(np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
], dtype=float))

y_full  = airline_log
y_train = y_full[:-12]   # hold out last year for evaluation
y_test  = y_full[-12:]

# ------------------------------------------------------------------
# 2. Fit candidate models
# ------------------------------------------------------------------
ll_result  = LocalLevel().fit(y_train)
llt_result = LocalLinearTrend().fit(y_train)
bsm_result = BSM(period=12).fit(y_train)

# ------------------------------------------------------------------
# 3. Model comparison table
# ------------------------------------------------------------------
table = compare_models(ll_result, llt_result, bsm_result, criterion="bic")
print("=" * 60)
print("Model comparison (sorted by BIC)")
print("=" * 60)
print(table[["model", "loglik", "k", "bic", "Δ_criterion", "weight"]].to_string(index=False))

# ------------------------------------------------------------------
# 4. Innovation diagnostics on the best model (BSM)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Innovation diagnostics — BSM")
print("=" * 60)
summary = run_all_tests(bsm_result)
print(summary.to_string(index=False))

# ------------------------------------------------------------------
# 5. CUSUM stability tests
# ------------------------------------------------------------------
cs  = cusum(bsm_result,    significance=0.05)
cs2 = cusum_sq(bsm_result, significance=0.05)

print(f"\nCUSUM    break detected: {cs.break_detected}")
print(f"CUSUM-sq break detected: {cs2.break_detected}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
plot_cusum(cs,  ax=axes[0], title="CUSUM — level/slope stability")
plot_cusum(cs2, ax=axes[1], title="CUSUM-of-squares — variance stability")
plt.tight_layout()
fig.savefig("airline_cusum.png", dpi=150, bbox_inches="tight")
print("CUSUM plot saved to airline_cusum.png")

# ------------------------------------------------------------------
# 6. Auxiliary residuals for outlier/break identification
# ------------------------------------------------------------------
aux = auxiliary_residuals(bsm_result)
outlier_idx = np.where(aux.observation_outliers)[0]
break_idx   = np.where(aux.state_breaks)[0]

print(f"\nPotential outlier observations : {outlier_idx.tolist()}")
print(f"Potential state breaks         : {break_idx.tolist()}")

# ------------------------------------------------------------------
# 7. Out-of-sample forecast evaluation
# ------------------------------------------------------------------
metrics = forecast_metrics(bsm_result, y_test, horizon=1)

print("\n" + "=" * 60)
print("Out-of-sample metrics (last 12 months, h=1)")
print("=" * 60)
for name, val in metrics.items():
    print(f"  {name:<16} {val:.6f}")
```

---

## See Also

- [Diagnostics User Guide](../diagnostics/index.md)
- [Innovation Tests](../diagnostics/innovation-tests.md)
- [CUSUM](../diagnostics/cusum.md)
- [Auxiliary Residuals](../diagnostics/auxiliary-residuals.md)
- [State Smoothness](../diagnostics/state-smoothness.md)
- [Information Criteria](../diagnostics/information-criteria.md)
- [Cross-Validation](../diagnostics/cross-validation.md)
- [Filter Comparison](../diagnostics/filter-comparison.md)
- [Consistency](../diagnostics/consistency.md)
- [API: Core (KalmanFilter)](core.md)
- [API: Alternative Filters](filters.md)
- [Theory: MLE Theory](../theory/mle-theory.md)
- [Theory: Kalman Theory](../theory/kalman-theory.md)
