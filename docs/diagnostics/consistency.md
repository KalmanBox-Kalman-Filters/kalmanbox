# Filter Consistency Tests

A filter is **consistent** if its state estimates and uncertainty bounds accurately
reflect the true estimation error — the filter neither over- nor under-confidently
describes its own uncertainty. Consistency is a necessary condition for a filter to
be probabilistically valid: an inconsistent filter is wrong even when its point estimates
look reasonable.

The two standard consistency metrics are the **Normalized Estimation Error Squared (NEES)**
and the **Normalized Innovation Squared (NIS)**. Both are chi-squared statistics that
compare actual errors against the filter's predicted covariance.

---

## 1. Foundations: what consistency means

Let the true state at time $t$ be $\alpha_t^\star$ and the filter estimate be
$\hat{\alpha}_{t|t}$ with covariance $P_{t|t}$.

**Consistent filter**: the estimation error $\tilde{\alpha}_t = \alpha_t^\star - \hat{\alpha}_{t|t}$
satisfies

$$
\tilde{\alpha}_t \sim \mathcal{N}(0,\; P_{t|t})
$$

i.e., the filter correctly characterises its own uncertainty. The covariance is
neither inflated (over-conservative) nor deflated (over-confident).

!!! info "Consistency vs accuracy"
    A filter can be **accurate but inconsistent** (correct estimates, wrong covariances),
    or **consistent but inaccurate** (miscalibrated estimates with correctly reported
    large covariance). Both failure modes are diagnostically distinct.

### 1.1 Two perspectives on consistency

- **NEES** tests consistency from the **state estimation** perspective: does $P_{t|t}$
  correctly describe the spread of $\hat{\alpha}_{t|t}$ around $\alpha_t^\star$?
  Requires ground truth.
- **NIS** tests consistency from the **measurement prediction** perspective: does $F_t$
  correctly describe the spread of $v_t$ around zero? Does **not** require ground truth.

These two tests are complementary:

| Test | Requires ground truth? | What it measures |
|------|------------------------|------------------|
| NEES | Yes (simulation studies) | State covariance accuracy |
| NIS | No (real data) | Innovation covariance accuracy |

---

## 2. Normalized Estimation Error Squared (NEES)

### 2.1 Definition

At each time step $t$, the **NEES** is:

$$
\varepsilon_t = \tilde{\alpha}_t^\top P_{t|t}^{-1} \tilde{\alpha}_t
$$

where $\tilde{\alpha}_t = \alpha_t^\star - \hat{\alpha}_{t|t}$ is the estimation error.

Under correct filter specification:

$$
\varepsilon_t \mid \alpha_t^\star \sim \chi^2(m)
$$

where $m$ is the state dimension (the dimension of $\alpha_t$).

**Intuition**: NEES is a scaled squared distance. Under the null, the estimation error
lies in an $m$-dimensional Gaussian ellipsoid defined by $P_{t|t}$. The NEES statistic
measures how many "standard deviations" the true state is from the estimate, normalised
by the reported uncertainty.

### 2.2 Average NEES

In practice, a single realisation $\varepsilon_t$ is noisy. The **average NEES**
over $T$ time steps and $N$ independent Monte Carlo runs provides a stable test:

$$
\bar{\varepsilon} = \frac{1}{N \cdot T}\sum_{i=1}^{N}\sum_{t=1}^{T} \varepsilon_t^{(i)}
$$

Under the null hypothesis of consistency:

$$
N \cdot T \cdot \bar{\varepsilon} \sim \chi^2(N \cdot T \cdot m)
$$

and equivalently, $\bar{\varepsilon}$ has mean $m$ and variance $2m/(N \cdot T)$.

### 2.3 Confidence interval for average NEES

A $100(1-\alpha)\%$ confidence interval for $\bar{\varepsilon}$ under the null:

$$
\left[\frac{\chi^2_{N \cdot T \cdot m,\; \alpha/2}}{N \cdot T},\; \frac{\chi^2_{N \cdot T \cdot m,\; 1-\alpha/2}}{N \cdot T}\right]
$$

For large $N \cdot T$, the interval approximates:

$$
m \pm z_{\alpha/2} \sqrt{\frac{2m}{N \cdot T}}
$$

The filter is declared **inconsistent** if $\bar{\varepsilon}$ falls outside this interval.

### 2.4 Interpreting NEES

| $\bar{\varepsilon}$ vs $m$ | Interpretation |
|---------------------------|----------------|
| $\bar{\varepsilon} \approx m$ | Consistent — covariance correctly characterises error |
| $\bar{\varepsilon} > m$ | **Optimistic** (overconfident) — $P_{t|t}$ too small |
| $\bar{\varepsilon} \ll m$ | **Pessimistic** (underconfident) — $P_{t|t}$ too large |

**Overconfident filter** ($\bar{\varepsilon} > m$): the filter underestimates its own
uncertainty. The true state is further from the estimate than the covariance suggests.
Common causes: underestimated process noise $Q$, poor initialisation of $P_1$.

**Underconfident filter** ($\bar{\varepsilon} < m$): the filter overestimates uncertainty.
Common causes: overestimated process noise, inflated $P_1$, or a deliberate conservative
tuning.

---

## 3. Normalized Innovation Squared (NIS)

### 3.1 Definition

The **NIS** at time $t$ uses the observable innovation $v_t$ and its predicted
covariance $F_t$:

$$
\delta_t = v_t^\top F_t^{-1} v_t
$$

where $v_t = y_t - Z_t \hat{\alpha}_{t|t-1}$ is the innovation (prediction error)
and $F_t = Z_t P_{t|t-1} Z_t^\top + H_t$ is the innovation covariance.

Under correct specification:

$$
\delta_t \sim \chi^2(p)
$$

where $p$ is the observation dimension.

### 3.2 Relationship to log-likelihood

The NIS statistic is the **quadratic form** term in the log-likelihood:

$$
\log L = -\frac{np}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^{n}\!\left(\log|F_t| + \underbrace{v_t^\top F_t^{-1} v_t}_{\delta_t}\right)
$$

A filter with consistently large $\delta_t$ generates a low likelihood regardless of
the log-determinant term. NIS monitoring is therefore a real-time signal of filter
degradation.

### 3.3 Average NIS

$$
\bar{\delta} = \frac{1}{n}\sum_{t=1}^{n} \delta_t
$$

Under the null: $\bar{\delta} \sim \chi^2(np) / n$, with mean $p$ and variance $2p/n$.

Confidence interval:

$$
\left[\frac{\chi^2_{np,\;\alpha/2}}{n},\; \frac{\chi^2_{np,\;1-\alpha/2}}{n}\right]
$$

### 3.4 NEES vs NIS: the relationship

In a linear Gaussian model, NEES and NIS are connected through the
**information matrix**. Define the filter gain $K_t = P_{t|t-1} Z_t^\top F_t^{-1}$.
Then:

$$
\varepsilon_t = (1 - K_t Z_t)^{-1} \delta_t \quad \text{(approximately)}
$$

In practice, NIS can serve as a *proxy* for NEES when ground truth is unavailable:
if NIS is within bounds, NEES is likely consistent too (for linear models).

For nonlinear models (EKF/UKF), this relationship breaks down — both tests should
be run independently.

---

## 4. Chi-squared test procedure

### 4.1 Single-run test (NIS only — real data)

For a single time series of length $n$:

1. Compute $\delta_t = v_t^\top F_t^{-1} v_t$ for $t = 1, \ldots, n$.
2. Compute $\bar{\delta} = \frac{1}{n}\sum_t \delta_t$.
3. Under $H_0$: $n\bar{\delta} \sim \chi^2(np)$.
4. Reject at level $\alpha$ if $n\bar{\delta} > \chi^2_{np,\;1-\alpha}$ (overconfident)
   or $< \chi^2_{np,\;\alpha}$ (underconfident).

### 4.2 Monte Carlo test (NEES — simulation study)

For $N$ independent runs, each of length $T$:

1. Simulate $N$ realisations $(\alpha_{1:T}^{(i)}, y_{1:T}^{(i)})$ from the model.
2. Run the filter on each realisation.
3. Compute $\varepsilon_t^{(i)}$ for each $(i, t)$.
4. Pool: $\bar{\varepsilon} = \frac{1}{NT}\sum_{i,t} \varepsilon_t^{(i)}$.
5. Under $H_0$: $NT\bar{\varepsilon} \sim \chi^2(NTm)$.
6. Construct the $95\%$ interval and check if $\bar{\varepsilon}$ falls inside.

!!! tip "How many Monte Carlo runs?"
    The width of the NEES confidence interval shrinks as $1/\sqrt{NT}$. A common
    minimum is $N \geq 20$ runs of length $T \geq 50$, giving $NT \geq 1000$. For
    tight tests with $m \leq 4$, this yields a CI width of $\approx \pm 0.13m$.

### 4.3 Time-averaged NEES profile

Rather than pooling all time steps, plot $\bar{\varepsilon}_t$ as a function of $t$
(averaged over $N$ Monte Carlo runs) to detect where consistency breaks:

$$
\bar{\varepsilon}_t^{(N)} = \frac{1}{N}\sum_{i=1}^{N} \varepsilon_t^{(i)} \sim \chi^2(Nm) / N
$$

Regions where $\bar{\varepsilon}_t^{(N)}$ persistently exceeds the CI indicate
model misspecification at those time points (e.g., a wrong noise model after a
structural break).

---

## 5. Causes and remedies for inconsistency

### 5.1 Overconfident filter ($\bar{\varepsilon} > m$, $\bar{\delta} > p$)

**The filter underestimates its own uncertainty.** The true state is consistently
further from the estimate than the reported covariance implies.

| Cause | Diagnostic signal | Remedy |
|-------|-------------------|--------|
| Process noise $Q$ too small | $\bar{\varepsilon}$ grows over time | Increase $Q$ or re-estimate via MLE |
| Observation noise $H$ too small | $\bar{\delta}$ elevated immediately | Re-estimate $H$ |
| Bad initial covariance $P_1$ | $\bar{\varepsilon}_t$ high at $t=1$, then recovers | Increase $P_1$ or use diffuse initialisation |
| Nonlinear model fitted with EKF | Grows with trajectory length | Switch to UKF or EnKF |
| Missing state component | NEES drift correlates with missing component | Add state variable to model |
| Model misspecification | Both NEES and NIS elevated | Re-specify model; run innovation tests |

### 5.2 Underconfident filter ($\bar{\varepsilon} < m$, $\bar{\delta} < p$)

**The filter overestimates uncertainty.** More benign than overconfidence but leads to
over-wide credible intervals and excessively slow adaptation.

| Cause | Diagnostic signal | Remedy |
|-------|-------------------|--------|
| Process noise $Q$ too large | Filter slow to adapt | Decrease $Q$ or re-estimate |
| Over-inflated $P_1$ | High early $\bar{\varepsilon}_t$, declines | Reduce $P_1$ |
| Covariance inflation (EnKF) | Persistent underconfidence | Reduce inflation factor |
| Duplicate observations | $\bar{\delta}$ very low | Check for repeated rows in $y$ |

### 5.3 Time-varying inconsistency

If NEES or NIS is consistent globally but inconsistent in a specific time window:

- **Structural break** in the data that the model does not capture.
- **Missing seasonal component** causing periodic inconsistency.
- **Outlier cluster** inflating the chi-squared statistic locally.

Diagnose with the time-profile plot of $\bar{\varepsilon}_t$ and the
[auxiliary residuals](auxiliary-residuals.md) for the same window.

### 5.4 Inconsistency from linearisation (EKF)

For nonlinear models, EKF overconfidence is systematic, not random:

$$
P_{t|t}^{\text{EKF}} = (I - K_t Z_t) P_{t|t-1} \approx (I - K_t \nabla h|_{\hat{\alpha}}) P_{t|t-1}
$$

The Jacobian $\nabla h$ only captures first-order uncertainty propagation. If $h$
is highly curved, second-order terms dominate and $P_{t|t}^{\text{EKF}}$ is
systematically too small. The result: NEES $> m$ that grows with nonlinearity.

**Remedy**: Switch to UKF (captures third-order terms) or EnKF (Monte Carlo convergence).

---

## 6. API reference

### `nees()`

```python
from kalmanbox.diagnostics import nees

NEESResult = nees(
    filter_results,                      # KalmanResults or list[KalmanResults] for MC
    true_states: np.ndarray,             # shape (n, m) or (N, n, m) for MC
    alpha: float = 0.05,                 # significance level for CI
    per_timestep: bool = True,           # return ε_t in addition to ε̄
)
```

**`NEESResult` attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.nees_sequence` | `ndarray` | $\varepsilon_t$ for each time step, shape $(n,)$ or $(N, n)$ |
| `.average_nees` | `float` | $\bar{\varepsilon}$ pooled over time (and runs) |
| `.state_dim` | `int` | $m$ — expected value under consistency |
| `.ci_lower` | `float` | Lower bound of $100(1-\alpha)\%$ CI |
| `.ci_upper` | `float` | Upper bound of CI |
| `.is_consistent` | `bool` | True if $\bar{\varepsilon} \in [\text{ci\_lower}, \text{ci\_upper}]$ |
| `.chi2_stat` | `float` | Chi-squared test statistic $NT\bar{\varepsilon}$ |
| `.pvalue` | `float` | Two-sided $p$-value |
| `.n_mc_runs` | `int` | Number of Monte Carlo runs ($N$) |
| `.summary()` | method | Print formatted result |
| `.plot()` | method | Plot $\bar{\varepsilon}_t$ with CI bands |

### `nis()`

```python
from kalmanbox.diagnostics import nis

NISResult = nis(
    filter_results,                      # KalmanResults (single run, real data OK)
    alpha: float = 0.05,
    per_timestep: bool = True,
    skip_diffuse: bool = True,           # exclude diffuse initialisation steps
)
```

**`NISResult` attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.nis_sequence` | `ndarray` | $\delta_t$ for each time step, shape $(n,)$ |
| `.average_nis` | `float` | $\bar{\delta}$ |
| `.obs_dim` | `int` | $p$ — expected value under consistency |
| `.ci_lower` | `float` | Lower CI bound |
| `.ci_upper` | `float` | Upper CI bound |
| `.is_consistent` | `bool` | True if $\bar{\delta} \in [\text{ci\_lower}, \text{ci\_upper}]$ |
| `.chi2_stat` | `float` | Test statistic $n\bar{\delta}$ |
| `.pvalue` | `float` | Two-sided $p$-value |
| `.summary()` | method | Print formatted result |
| `.plot()` | method | Plot $\delta_t$ time series with CI bands |

### `consistency_test()`

```python
from kalmanbox.diagnostics import consistency_test

ConsistencyReport = consistency_test(
    filter_results,                      # KalmanResults or list[KalmanResults]
    true_states: np.ndarray = None,      # required for NEES; None = NIS only
    alpha: float = 0.05,
    n_mc_runs: int = None,               # auto from list length if list provided
)
```

Runs **both** NEES (if `true_states` provided) and NIS, and returns a combined
`ConsistencyReport` with:

| Attribute | Description |
|-----------|-------------|
| `.nees` | `NEESResult` (or `None` if no ground truth) |
| `.nis` | `NISResult` |
| `.overall_consistent` | `bool` — True if both NEES and NIS pass |
| `.diagnosis` | `str` — plain-text interpretation of results |
| `.summary()` | Print combined report |

### `plot_consistency()`

```python
from kalmanbox.diagnostics import plot_consistency

fig = plot_consistency(
    consistency_report,                  # ConsistencyReport
    dates: pd.DatetimeIndex = None,
    show_nees: bool = True,
    show_nis: bool = True,
    figsize: tuple = (14, 8),
    alpha_band: float = 0.15,            # CI band transparency
)
```

Produces a two-panel figure:

1. **NEES panel** (if ground truth available): $\bar{\varepsilon}_t^{(N)}$ over time
   with horizontal lines at the CI bounds and the null value $m$.
2. **NIS panel**: $\delta_t$ over time with CI bounds and null value $p$, plus a
   rolling 10-step average to reveal trends.

---

## 7. Examples

### Example 1: consistency test on a Kalman filter — well-specified model

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import consistency_test, plot_consistency

rng = np.random.default_rng(42)
n, N = 200, 50   # 50 Monte Carlo runs of length 200

model = LocalLevelModel(sigma_level=1.0, sigma_obs=2.0)

# Monte Carlo simulation
mc_results = []
mc_true_states = []

for seed in range(N):
    true_alpha, y = model.simulate(n=n, random_seed=seed)
    res = model.filter(y)
    mc_results.append(res)
    mc_true_states.append(true_alpha)

true_states_array = np.stack(mc_true_states)   # shape (N, n, 1)

report = consistency_test(
    mc_results,
    true_states=true_states_array,
    alpha=0.05,
)
print(report.summary())
plot_consistency(report)
```

**Expected output**:

```
Consistency Test Report
=======================
Model: LocalLevel  |  n=200, m=1, p=1  |  N=50 Monte Carlo runs

NEES (Normalized Estimation Error Squared)
  Average NEES    : 1.021
  Expected (m=1)  : 1.000
  95% CI          : [0.937, 1.063]
  χ²(10000) stat  : 10210.0, p=0.172
  Verdict         : CONSISTENT  ✓

NIS (Normalized Innovation Squared)
  Average NIS     : 0.997  (using single representative run)
  Expected (p=1)  : 1.000
  95% CI          : [0.913, 1.087]
  χ²(200) stat    : 199.4, p=0.491
  Verdict         : CONSISTENT  ✓

Overall: CONSISTENT — filter correctly characterises its uncertainty.

Diagnosis:
  Both NEES (1.021) and NIS (0.997) lie within the 95% CI.
  The LocalLevel filter accurately reports its state estimation uncertainty
  for this model specification.
```

**Interpretation**: The average NEES of 1.021 falls within the confidence interval
$[0.937, 1.063]$, and NIS of 0.997 is essentially 1.0. The filter is consistent:
its reported covariance $P_{t|t}$ accurately reflects the true estimation error.

---

### Example 2: diagnosing an overconfident filter

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import consistency_test, plot_consistency

rng = np.random.default_rng(0)
n, N = 300, 50

# True model: sigma_level=2.0 (large process noise)
# Fitted model: sigma_level=0.5 (underestimated — overconfident)
true_model = LocalLevelModel(sigma_level=2.0, sigma_obs=1.0)
fitted_model = LocalLevelModel(sigma_level=0.5, sigma_obs=1.0)

mc_results = []
mc_true_states = []

for seed in range(N):
    true_alpha, y = true_model.simulate(n=n, random_seed=seed)
    res = fitted_model.filter(y)    # fit wrong model
    mc_results.append(res)
    mc_true_states.append(true_alpha)

true_states_array = np.stack(mc_true_states)

report = consistency_test(mc_results, true_states=true_states_array, alpha=0.05)
print(report.summary())
```

**Expected output**:

```
Consistency Test Report
=======================
Model: LocalLevel (fitted)  |  n=300, m=1, p=1  |  N=50 Monte Carlo runs

NEES (Normalized Estimation Error Squared)
  Average NEES    : 5.847
  Expected (m=1)  : 1.000
  95% CI          : [0.951, 1.049]
  χ²(15000) stat  : 87705.0, p≈0.000
  Verdict         : INCONSISTENT — OVERCONFIDENT  ✗

NIS (Normalized Innovation Squared)
  Average NIS     : 4.123
  Expected (p=1)  : 1.000
  95% CI          : [0.912, 1.088]
  χ²(300) stat    : 1236.9, p≈0.000
  Verdict         : INCONSISTENT — OVERCONFIDENT  ✗

Overall: INCONSISTENT

Diagnosis:
  NEES = 5.85 >> m=1  → filter is OVERCONFIDENT (underestimates uncertainty).
  NIS  = 4.12 >> p=1  → innovation covariance F_t too small.

  Most likely cause: process noise Q underestimated.
    True σ_level ≈ 2.0, fitted σ_level = 0.5.
    True Q = 4.0, fitted Q = 0.25 — factor of 16 difference.

  Recommended action:
    1. Re-estimate parameters via MLE: model.fit(y) using full data.
    2. Or increase Q manually until NEES converges to m=1.
    3. Check: plot_consistency() shows NEES growing over time (not stationary)
       which confirms systematic, not random, overconfidence.
```

**Interpretation**: NEES of 5.85 is nearly 6× the expected value of 1.0 — the filter
is severely overconfident because it underestimates how much the state wanders
(small $Q$ → small $P_{t|t}$ → NEES high). NIS of 4.12 confirms the same from the
observation side: the innovation variances $F_t$ are too small, so the innovations
appear larger than expected.

**Recommended fix**: Use MLE to re-estimate $\sigma_\text{level}$:

```python
from kalmanbox import LocalLevelModel

model_mle = LocalLevelModel()
results_mle = model_mle.fit(y)          # MLE re-estimates sigma_level
print(results_mle.params)               # should recover sigma_level ≈ 2.0

# Re-run consistency test with MLE-estimated model
mc_results_mle = [model_mle.filter(yi) for yi in [true_model.simulate(n=n, random_seed=s)[1] for s in range(N)]]
report_mle = consistency_test(mc_results_mle, true_states=true_states_array)
print(report_mle.summary())             # NEES should recover to ≈ 1.0
```

---

### Example 3: NIS monitoring on real data (no ground truth)

```python
import numpy as np
import pandas as pd
from kalmanbox import BSM
from kalmanbox.datasets import load_airline
from kalmanbox.diagnostics import nis, plot_consistency

y = np.log(load_airline())
dates = pd.date_range("1949-01", periods=len(y), freq="ME")

model = BSM(period=12, stochastic_seasonal=True)
results = model.fit(y)

nis_result = nis(results, alpha=0.05, skip_diffuse=True)
print(nis_result.summary())
plot_consistency(nis_result, dates=dates)
```

**Expected output**:

```
NIS (Normalized Innovation Squared)
====================================
Model: BSM  |  n=132 (12 diffuse steps excluded)  |  p=1

Average NIS     : 1.043
Expected (p=1)  : 1.000
95% CI          : [0.851, 1.149]
χ²(132) stat    : 137.7, p=0.348
Verdict         : CONSISTENT  ✓

NIS sequence statistics:
  Min: 0.001   Q1: 0.183   Median: 0.601   Q3: 1.412   Max: 7.231
  Large NIS (>χ²₁,₀.₉₅=3.84) at 6 steps: t=7,12,24,36,48,60  [4.5%]
  Expected exceedances at α=0.05: 5% → 6.6 steps — within range.

Diagnosis:
  Global NIS is consistent. The 6 elevated NIS values at annual lags
  are consistent with random variation (expected 5% exceedance rate).
  No evidence of systematic overconfidence or underconfidence.
```

**Interpretation**: The BSM passes the NIS test with a mean of 1.043 and a $p$-value
of 0.348. The 6 elevated NIS values at annual lags (every 12 months) deserve
attention but fall within the expected 5 % exceedance rate for $n = 132$ observations.
If a persistent pattern at lag 12 appeared in every year, it would signal a missing
seasonal component — but here the exceedances are scattered, confirming adequate model
specification.

---

### Example 4: comparing EKF vs UKF consistency on a nonlinear model

```python
import numpy as np
from kalmanbox.filters import EKF, UKF
from kalmanbox.diagnostics import consistency_test
from kalmanbox.models import BearingsOnlyTracking

model = BearingsOnlyTracking(process_noise=0.1, obs_noise=0.05, dt=0.1)

N, n = 30, 200
ekf_results, ukf_results, true_states_list = [], [], []

for seed in range(N):
    true_states, y = model.simulate(n=n, random_seed=seed)
    ekf_results.append(EKF().run(model, y))
    ukf_results.append(UKF(alpha=1e-3, beta=2.0, kappa=0.0).run(model, y))
    true_states_list.append(true_states)

true_arr = np.stack(true_states_list)   # (N, n, 4)

for label, results in [("EKF", ekf_results), ("UKF", ukf_results)]:
    report = consistency_test(results, true_states=true_arr, alpha=0.05)
    nees_val = report.nees.average_nees
    nis_val  = report.nis.average_nis
    status   = "CONSISTENT" if report.overall_consistent else "INCONSISTENT"
    print(f"{label}: NEES={nees_val:.3f} (exp 4.0), NIS={nis_val:.3f} (exp 1.0) → {status}")
```

**Expected output**:

```
EKF: NEES=7.423 (exp 4.0), NIS=2.841 (exp 1.0) → INCONSISTENT
UKF: NEES=4.112 (exp 4.0), NIS=1.034 (exp 1.0) → CONSISTENT
```

**Interpretation**:

- **EKF NEES = 7.42** against the expected $m = 4$: the filter is nearly 2× overconfident.
  The first-order linearisation of the nonlinear bearing function loses second-order terms,
  causing $P_{t|t}^{\text{EKF}}$ to systematically underestimate the true estimation error.
- **UKF NEES = 4.11** ≈ 4: the unscented transform propagates uncertainty through the
  nonlinearity accurately, producing consistent covariances.
- The NIS values tell the same story from the observation side: EKF NIS = 2.84 means the
  filter's predicted innovation variance is less than half the actual innovation variance.

For this nonlinear model, **UKF is both more accurate and more consistent** than EKF.
EKF should only be used if computational constraints prohibit UKF.

---

## 8. Consistency diagnostics summary table

| Test | Statistic | Null distribution | Requires ground truth | Detects |
|------|-----------|-------------------|-----------------------|---------|
| NEES | $\bar{\varepsilon}$ | $\chi^2(NTm) / (NT)$ | Yes | State covariance mis-specification |
| NIS | $\bar{\delta}$ | $\chi^2(np) / n$ | No | Innovation covariance mis-specification |
| NEES profile | $\bar{\varepsilon}_t^{(N)}$ | $\chi^2(Nm) / N$ | Yes | Time-localised inconsistency |
| NIS profile | $\delta_t$ | $\chi^2(p)$ | No | Real-time filter degradation |

---

## Related

- [Filter Comparison](filter-comparison.md) — accuracy and efficiency benchmarks across filter algorithms
- [Innovation Tests](innovation-tests.md) — white-noise tests on Kalman filter innovations
- [Auxiliary Residuals](auxiliary-residuals.md) — smoother-based outlier and break detection
- [Prediction Error Analysis](prediction-error.md) — out-of-sample forecast accuracy
- [User guide: EKF](../user-guide/filters/ekf.md) — Extended Kalman Filter
- [User guide: UKF](../user-guide/filters/ukf.md) — Unscented Kalman Filter
- [User guide: Ensemble KF](../user-guide/filters/ensemble.md)
- [Theory: Nonlinear Filter Theory](../theory/nonlinear-theory.md) — accuracy order analysis
- [Theory: Kalman Filter Theory](../theory/kalman-theory.md) — optimality and covariance recursions
- [API: diagnostics module](../api/diagnostics.md)
