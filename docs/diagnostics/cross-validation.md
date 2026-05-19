# Cross-Validation for State-Space Models

Cross-validation (CV) evaluates **out-of-sample predictive performance** by repeatedly
fitting a model on a subset of the data and measuring its accuracy on the held-out
portion. For time series and state-space models, CV must respect the temporal ordering
of observations — future data cannot be used to fit models that are then evaluated on past
data.

---

## 1. Why CV for state-space models?

Information criteria (AIC, BIC) estimate the expected out-of-sample log-likelihood
analytically, under assumptions that may not hold for finite samples or misspecified
models. Cross-validation makes no distributional assumptions beyond stationarity of the
predictive problem and is therefore:

- **Robust to misspecification**: measures actual forecast accuracy, not asymptotic
  approximations.
- **Metric-flexible**: can optimise for RMSE, MAE, quantile loss, or any user-defined
  metric.
- **Task-aligned**: a $h$-step CV directly measures $h$-step forecast accuracy, which
  AIC does not.

!!! warning "Time ordering"
    Standard $k$-fold CV shuffles observations, which **must not** be done for time
    series. All kalmanbox CV methods enforce the constraint that the test set always
    lies entirely after the training set in time.

---

## 2. CV schemes for time series

### 2.1 Rolling-window cross-validation

The training window has **fixed width** $w$; both the training and test windows slide
forward one step at a time.

$$
\text{Fold } k: \quad
\text{Train} = \{y_{k}, \ldots, y_{k+w-1}\}, \quad
\text{Test}  = \{y_{k+w}, \ldots, y_{k+w+h-1}\}
$$

for $k = 1, 2, \ldots, n - w - h + 1$.

```
Time:  1  2  3  4  5  6  7  8  9 10 11 12
Fold1: [train: 1-6] [test: 7-8]
Fold2:    [train: 2-7] [test: 8-9]
Fold3:       [train: 3-8] [test: 9-10]
...
```

**Use when**: the data-generating process may be non-stationary or
regime-changing and old data should not influence recent-period fits.

### 2.2 Expanding-window cross-validation

The training window grows from a minimum size $w_0$ to the full sample; the test window
is always at the right edge.

$$
\text{Fold } k: \quad
\text{Train} = \{y_1, \ldots, y_{w_0+k-1}\}, \quad
\text{Test}  = \{y_{w_0+k}, \ldots, y_{w_0+k+h-1}\}
$$

for $k = 1, 2, \ldots, n - w_0 - h + 1$.

```
Time:  1  2  3  4  5  6  7  8  9 10 11 12
Fold1: [train: 1-6      ] [test: 7-8]
Fold2: [train: 1-7        ] [test: 8-9]
Fold3: [train: 1-8          ] [test: 9-10]
...
```

**Use when**: the model is assumed to be stationary and all historical data is
informative. This is the standard approach when the training set is already small.

### 2.3 One-step-ahead cross-validation

A special case of expanding-window CV with $h = 1$. The model is re-estimated at each
step and the one-step-ahead forecast error is recorded.

$$
\text{CV}_{1} = \frac{1}{n-w_0}\sum_{t=w_0}^{n-1}\mathcal{L}(y_{t+1},\hat{y}_{t+1|t})
$$

One-step-ahead CV has a particularly useful property for state-space models: the
**standardised innovations** from a single Kalman filter pass are approximately
equivalent to leaving each observation out one at a time (Bernardo & Smith, 1994,
§6.4). This makes one-step-ahead CV nearly free to compute.

```python
from kalmanbox.diagnostics import one_step_ahead_cv

cv1 = one_step_ahead_cv(model, y, metric="rmse")
print(f"RMSE (1-step): {cv1.rmse:.4f}")
print(f"MAE  (1-step): {cv1.mae:.4f}")
print(f"MASE (1-step): {cv1.mase:.4f}")
```

### 2.4 Multi-step cross-validation

Evaluates $h$-step-ahead forecast accuracy. At each fold the model is re-estimated and
then iterated forward $h$ steps without updating the filter:

$$
\text{CV}_{h} = \frac{1}{K}\sum_{k=1}^{K}\mathcal{L}(y_{t_k+h}, \hat{y}_{t_k+h \mid t_k})
$$

Multi-step CV reveals model weaknesses that one-step CV can hide: a model with excellent
one-step accuracy may degrade rapidly at $h = 6$ or $h = 12$ if its trend or seasonal
assumptions are incorrect.

### 2.5 Leave-future-out cross-validation (LFO-CV)

LFO-CV approximates the leave-one-out CV score using importance sampling from the
sequential predictive densities. Unlike LOO-CV (which conditions on the whole sample),
LFO-CV conditions on the past, making it appropriate for time series:

$$
\text{LFO-CV} = \sum_{t=L}^{n} \log p(y_t \mid y_{1:t-1}, \hat\theta)
$$

where $L$ is a minimum training size and $p(y_t \mid y_{1:t-1}, \hat\theta)$ is the
one-step-ahead predictive density evaluated at the MLE (or posterior mean).

The **log-predictive density (LPD)** sum is related to AIC/BIC but requires no
asymptotic approximation. It can be approximated cheaply from innovations:

$$
\log p(y_t \mid y_{1:t-1}) = -\frac{1}{2}\left(\log|F_t| + v_t^\top F_t^{-1} v_t + p\log(2\pi)\right)
$$

which is exactly the contribution of observation $t$ to the total log-likelihood.

---

## 3. Metrics

### 3.1 Point-forecast metrics

| Metric | Formula | Scale-dependent? | Outlier-sensitive? |
|--------|---------|-----------------|-------------------|
| RMSE | $\sqrt{\frac{1}{K}\sum e_t^2}$ | Yes | Yes |
| MAE | $\frac{1}{K}\sum|e_t|$ | Yes | Less |
| MAPE | $\frac{100}{K}\sum\left|\frac{e_t}{y_t}\right|$ | No | Very |
| MASE | $\frac{\text{MAE}}{\text{MAE}_\text{naive}}$ | No | Less |
| Theil U | $\frac{\text{RMSE}}{\text{RMSE}_\text{naive}}$ | No | Yes |

where $e_t = y_t - \hat{y}_{t \mid t-h}$ are the $h$-step forecast errors and the naive
benchmark is the random walk: $\hat{y}_{t \mid t-h}^\text{naive} = y_{t-h}$.

$\text{MASE} < 1$ means the model beats the naive benchmark; it is the recommended
metric for comparing models across different series with different scales.

### 3.2 Distributional metrics

When comparing models on **probabilistic** forecast accuracy:

$$
\text{Log-predictive density (LPD)} = \sum_{t \in \mathcal{T}} \log p(y_t \mid y_{1:t-1}, \hat\theta)
$$

Higher LPD is better. The LPD is directly computed from Kalman filter innovations and
does not require simulation.

**Continuous Ranked Probability Score (CRPS)**:

$$
\text{CRPS}(\hat{F}_t, y_t) = \int_{-\infty}^{\infty}\left(\hat{F}_t(z) - \mathbf{1}[z \geq y_t]\right)^2 dz
$$

For Gaussian forecasts $\hat{F}_t = \mathcal{N}(\hat{y}_{t|t-h},\, F_t)$, the CRPS has
a closed form:

$$
\text{CRPS} = \sigma_t\left[z\left(2\Phi(z)-1\right) + 2\phi(z) - \pi^{-1/2}\right], \quad z = \frac{y_t - \hat{y}_{t|t-h}}{\sigma_t}
$$

where $\phi$ and $\Phi$ are the standard normal PDF and CDF, $\sigma_t = \sqrt{F_t}$.

---

## 4. Computational considerations

State-space models have two computational costs per fold:

1. **MLE re-estimation**: runs an optimiser over the parameter space (typically $O(n\,k)$
   per evaluation, with $O(n)$ being the Kalman filter and $k$ the state dimension).
2. **Kalman filter evaluation**: $O(n\,k^3)$ per fold for the full filter pass.

### 4.1 Cost of different CV schemes

| CV scheme | Model re-fitted? | Folds | Total filter passes | Relative cost |
|-----------|-----------------|-------|---------------------|---------------|
| One-step-ahead (filter-only) | No | 1 | 1 | ~$1\times$ |
| Expanding window ($h=1$, no refit) | No | $n-w_0$ | $n-w_0$ | ~$n\times$ |
| Expanding window (with refit) | Yes | $n-w_0$ | $n-w_0$ | ~$n\cdot n_\text{opt}\times$ |
| Rolling window (with refit) | Yes | $n-w-h$ | $n-w-h$ | ~$(n-w)\cdot n_\text{opt}\times$ |

For large $n$ or expensive models (DFM, Bayesian), expanding-window CV with full
re-estimation becomes prohibitive. kalmanbox provides two cost-reduction strategies:

### 4.2 Warm-start optimisation

Use the parameter estimates from the previous fold as the starting point for the next
optimisation. This typically reduces the number of function evaluations by $5$–$10\times$
when parameters change slowly across folds:

```python
cv = rolling_cv(model, y, window=100, horizon=12, refit=True, warm_start=True)
```

### 4.3 Fixed-parameter CV

Estimate parameters once on the full sample, then evaluate forecast errors without
re-estimating. This is appropriate when the goal is to evaluate state tracking
(filter performance) rather than parameter stability:

```python
results = model.fit(y)  # estimate once
cv = rolling_cv(results, y, window=100, horizon=12, refit=False)
```

### 4.4 Approximate LFO-CV via PSIS

Vehtari, Mononen, Tran et al. (2017) show that Pareto-smoothed importance sampling
(PSIS) can approximate LFO-CV in $O(n)$ time without refitting, using sequential
importance weights computed from filter quantities. kalmanbox implements this as
`lfo_cv(results, approximate=True)`.

---

## 5. API reference

### `cross_validate()`

```python
from kalmanbox.diagnostics import cross_validate

cv = cross_validate(
    model,                           # unfitted model or KalmanResults
    y: np.ndarray,                   # time series (n,) or (n, p)
    method: str = "expanding",       # "expanding" | "rolling" | "lfo"
    window: int | None = None,       # min/fixed training window
    horizon: int = 1,                # forecast horizon h
    step: int = 1,                   # step between folds
    refit: bool = True,              # re-estimate parameters each fold
    warm_start: bool = True,         # initialise from previous fold estimates
    metrics: list[str] = ["rmse", "mae", "lpd"],
)
```

Returns a `CVResults` object.

### `rolling_cv()`

```python
from kalmanbox.diagnostics import rolling_cv

cv = rolling_cv(
    model,
    y: np.ndarray,
    window: int,                     # fixed training window width
    horizon: int = 1,
    step: int = 1,
    refit: bool = True,
    warm_start: bool = True,
    metrics: list[str] = ["rmse", "mae", "lpd"],
)
```

### `expanding_cv()`

```python
from kalmanbox.diagnostics import expanding_cv

cv = expanding_cv(
    model,
    y: np.ndarray,
    min_window: int,                 # minimum training size
    horizon: int = 1,
    step: int = 1,
    refit: bool = True,
    warm_start: bool = True,
    metrics: list[str] = ["rmse", "mae", "lpd"],
)
```

### `CVResults` object

| Attribute | Type | Description |
|-----------|------|-------------|
| `.errors` | `DataFrame` | Forecast errors per fold and horizon |
| `.metrics` | `dict[str, float]` | Aggregated metric values |
| `.rmse` | `float` | Root mean squared error |
| `.mae` | `float` | Mean absolute error |
| `.mase` | `float` | Mean absolute scaled error |
| `.lpd` | `float` | Total log-predictive density |
| `.crps` | `float` | Average CRPS (Gaussian) |
| `.n_folds` | `int` | Number of CV folds |
| `.fold_metrics` | `DataFrame` | Per-fold metric breakdown |
| `.params_history` | `DataFrame` | Estimated params per fold (if `refit=True`) |

```python
print(cv.summary())
cv.plot_errors()           # time plot of forecast errors
cv.plot_metrics_by_fold()  # metric trajectory over folds
cv.plot_params_stability() # parameter estimates over folds (requires refit=True)
```

---

## 6. Examples

### Example 1: one-step-ahead CV from filter innovations

This is the most efficient form of CV for state-space models because the innovations
are produced as a by-product of a single Kalman filter pass.

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.datasets import load_airline

y = np.log(load_airline())

results = BSM(period=12, stochastic_seasonal=True).fit(y)

# One-step-ahead metrics directly from innovations
from kalmanbox.diagnostics import one_step_ahead_cv

cv1 = one_step_ahead_cv(results)
print(cv1.summary())
```

**Expected output**:

```
One-Step-Ahead Cross-Validation
================================
Model  : BSM(period=12, stochastic_seasonal=True)
Series : 144 observations

Metrics (h=1):
  RMSE  : 0.0412
  MAE   : 0.0317
  MASE  : 0.381  (vs. naive RW)
  LPD   : 136.24
  CRPS  : 0.0221

Note: metrics computed from standardised Kalman filter innovations
      (equivalent to leave-one-out CV under Gaussian state-space)
```

### Example 2: comparing BSM and UCM via rolling CV

```python
import numpy as np
from kalmanbox import BSM, UCM
from kalmanbox.diagnostics import rolling_cv, model_cv_comparison
from kalmanbox.datasets import load_airline

y = np.log(load_airline())

# Rolling window CV: train on 96 months, forecast 12 months ahead
cv_bsm = rolling_cv(
    BSM(period=12, stochastic_seasonal=True),
    y, window=96, horizon=12, refit=True, warm_start=True,
)

cv_ucm = rolling_cv(
    UCM(period=12, stochastic_cycle=True),
    y, window=96, horizon=12, refit=True, warm_start=True,
)

# Aggregate comparison
table = model_cv_comparison(
    [cv_bsm, cv_ucm],
    names=["BSM", "UCM"],
)
print(table)
```

**Expected output**:

```
CV Comparison (rolling window=96, horizon=12)
=============================================

  Model  n_folds   RMSE    MAE   MASE    LPD   CRPS
    BSM       36  0.058  0.044  0.421  412.3  0.031
    UCM       36  0.063  0.049  0.469  397.8  0.035

Winner: BSM (lower RMSE, MAE, CRPS; higher LPD)
```

**Interpretation**: The BSM outperforms the UCM on all metrics in rolling CV. The UCM's
extra cycle component does not improve out-of-sample forecast accuracy, consistent with
the BIC evidence ($\Delta_\text{BIC} = 11.3$, see [information-criteria.md](information-criteria.md)).
CV and IC agree here; when they disagree, CV (measuring actual forecast accuracy) is
preferred for forecasting tasks.

### Example 3: horizon profile — how accuracy degrades with $h$

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.diagnostics import expanding_cv
from kalmanbox.datasets import load_airline
import matplotlib.pyplot as plt

y = np.log(load_airline())

# Evaluate horizons h=1..12 separately
rmse_by_h = {}
for h in range(1, 13):
    cv = expanding_cv(
        BSM(period=12, stochastic_seasonal=True),
        y, min_window=72, horizon=h, step=3, refit=False,
    )
    rmse_by_h[h] = cv.rmse

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(rmse_by_h.keys(), rmse_by_h.values())
ax.set_xlabel("Forecast horizon $h$")
ax.set_ylabel("RMSE")
ax.set_title("BSM forecast accuracy by horizon (log airline passengers)")
plt.tight_layout()
plt.savefig("horizon_profile.png", dpi=150)
```

The horizon profile reveals how quickly forecast accuracy degrades. A flat profile
(accuracy stable across horizons) is characteristic of models with a strong seasonal
component. A rapidly rising RMSE indicates the trend is uncertain at longer horizons.

### Example 4: parameter stability over folds

```python
from kalmanbox import BSM
from kalmanbox.diagnostics import expanding_cv
from kalmanbox.datasets import load_airline
import numpy as np

y = np.log(load_airline())

cv = expanding_cv(
    BSM(period=12, stochastic_seasonal=True),
    y, min_window=60, horizon=1, refit=True, warm_start=True,
)

# Inspect how parameter estimates evolve as more data is added
params = cv.params_history
print(params[["fold", "sigma_eps", "sigma_eta", "sigma_omega"]].to_string())

cv.plot_params_stability()
```

Parameter stability over expanding-window folds is a useful diagnostic in its own right:
large jumps in parameter estimates as new data arrives suggest model misspecification or
structural instability. Complement with [CUSUM tests](cusum.md).

### Example 5: leave-future-out CV with PSIS approximation

```python
from kalmanbox import UCM
from kalmanbox.diagnostics import lfo_cv
from kalmanbox.datasets import load_gdp

y = load_gdp()

results = UCM(period=4, stochastic_cycle=True).fit(y)  # quarterly GDP

# Approximate LFO-CV — O(n), no refitting
lfo = lfo_cv(results, min_window=40, approximate=True)
print(f"LFO-LPD (approx): {lfo.lpd:.2f}")
print(f"Pareto-k diagnostics: {lfo.pareto_k_summary()}")
```

The Pareto-$k$ diagnostic flags folds where the importance weights are unreliable.
Folds with $\hat{k} > 0.7$ should be computed exactly (set `approximate=False` to
fall back to exact re-estimation for those folds automatically).

---

## 7. CV vs. information criteria: when to use each

| Situation | Recommended |
|-----------|-------------|
| Large sample, nested models, quick answer | BIC / LR test |
| Small sample, $n/k < 40$ | AICc |
| Forecasting task, care about $h$-step accuracy | Rolling or expanding CV |
| Non-Gaussian model, no asymptotic guarantees | CV |
| Comparing many models efficiently | AIC/BIC first; CV to confirm top-2 |
| Evaluating probabilistic calibration | CRPS or LPD via CV |
| Parameter stability analysis | Expanding-window CV with refit |
| Computationally expensive model (MCMC) | PSIS-LFO approximate CV |

!!! tip "Triangulate"
    For important model choices, use **all three** tools: BIC for a quick ranking,
    the likelihood ratio test for pairwise nested comparisons, and rolling CV to
    verify the winner actually performs better out-of-sample. When all three agree,
    you can be confident in the selection.

---

## Related

- [Information Criteria](information-criteria.md) — AIC, BIC, HQIC for in-sample model selection
- [Likelihood Ratio Test](likelihood-ratio.md) — formal nested model testing
- [Prediction Error Analysis](prediction-error.md) — forecast metrics on a fixed test set
- [CUSUM & Stability](cusum.md) — detect structural breaks driving parameter instability
- [Theory: MLE theory](../theory/mle-theory.md) — log-likelihood and innovations
- [Experiment framework](../user-guide/experiment.md) — automated multi-model benchmark
- [API: diagnostics module](../api/diagnostics.md)
