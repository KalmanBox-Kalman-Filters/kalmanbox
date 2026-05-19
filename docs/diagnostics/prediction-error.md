# Prediction Error Analysis

Prediction error analysis quantifies how accurately the fitted state-space model forecasts
new observations. Unlike in-sample residual tests (which measure fit), forecast metrics
measure **generalisation** — the ability of the model to predict data it was not trained on.

---

## 1. One-step-ahead prediction errors

### Definition

At each time $t$, the Kalman filter produces the **one-step-ahead forecast** of $y_t$
before observing it:

$$
\hat{y}_{t|t-1} = Z_t a_t
$$

where $a_t = \mathbb{E}[x_t \mid y_1, \ldots, y_{t-1}]$. The corresponding **prediction
error** (raw innovation) is:

$$
e_t = y_t - \hat{y}_{t|t-1} = v_t
$$

and the **prediction error variance** is:

$$
\text{Var}(e_t) = F_t = Z_t P_t Z_t^\top + H_t
$$

One-step-ahead errors are the most natural measure of model fit because:

1. They use only **past** observations — no look-ahead bias.
2. They are directly related to the **log-likelihood**: maximising $\log L$ is equivalent to
   minimising the sum of squared standardised prediction errors.
3. They are **recursive residuals** — statistically independent under correct specification.

### Log-likelihood connection

$$
\log L(\theta) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^{n}\left(\log F_t + \frac{e_t^2}{F_t}\right)
$$

A model with smaller prediction errors necessarily achieves a higher likelihood.

---

## 2. Multi-step prediction errors

### Definition

The $h$-step-ahead forecast at origin $t$ is obtained by iterating the state transition:

$$
a_{t+h|t} = T^h a_{t|t}, \qquad P_{t+h|t} = T^h P_{t|t} (T^\top)^h + \sum_{j=0}^{h-1} T^j Q (T^\top)^j
$$

The $h$-step forecast of $y_{t+h}$ is:

$$
\hat{y}_{t+h|t} = Z_{t+h}\, a_{t+h|t}
$$

and the forecast error is:

$$
e_{t+h|t} = y_{t+h} - \hat{y}_{t+h|t}
$$

with variance $F_{t+h|t} = Z_{t+h} P_{t+h|t} Z_{t+h}^\top + H_{t+h}$.

!!! note "Multi-step errors are not innovations"
    Multi-step errors $\{e_{t+h|t}\}_{t}$ (fixed horizon $h$, varying origin $t$) are
    *serially correlated* with autocorrelation up to order $h-1$ under correct specification.
    This is expected — unlike one-step innovations they are not innovations of a martingale.
    Tests for serial correlation on multi-step errors must account for this.

### Forecast error decomposition

The mean squared forecast error (MSFE) decomposes into three components (Theil 1966):

$$
\text{MSFE} = \underbrace{\bar{e}^2}_{\text{Bias}^2} + \underbrace{(s_e - s_y \cdot r_{ey})^2}_{\text{Variance component}} + \underbrace{2(1-r_{ey})s_e s_y}_{\text{Covariance component}}
$$

where $\bar{e}$ is the mean error, $s_e$ and $s_y$ are standard deviations of errors and
actuals, and $r_{ey}$ is their correlation. Dividing by MSFE gives **Theil's proportions**:

| Proportion | Ideal value | High value indicates |
|-----------|-------------|----------------------|
| Bias (UM) | 0 | Systematic over/under-forecasting |
| Variance (UR) | 0 | Model misses the amplitude |
| Covariance (UC) | 1 | Acceptable unsystematic error |

---

## 3. Point forecast metrics

### RMSE — Root Mean Squared Error

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n} e_t^2}
$$

- **Scale-dependent**: in the same units as $y_t$.
- **Penalises large errors** quadratically — sensitive to outliers.
- Equivalent to the standard deviation of errors when the mean error is zero.

### MAE — Mean Absolute Error

$$
\text{MAE} = \frac{1}{n}\sum_{t=1}^{n} |e_t|
$$

- **Scale-dependent**, more robust to outliers than RMSE.
- When RMSE $\gg$ MAE, the error distribution is heavy-tailed (a few large errors dominate).
- When RMSE $\approx$ MAE, errors are roughly uniform.

### MAPE — Mean Absolute Percentage Error

$$
\text{MAPE} = \frac{100}{n}\sum_{t=1}^{n} \left|\frac{e_t}{y_t}\right|
$$

- **Scale-free** — comparable across series with different units or magnitudes.
- **Undefined** when $y_t = 0$; heavily distorted when $y_t$ is close to zero.
- Asymmetric: a 50 % under-forecast contributes less than a 50 % over-forecast.

### sMAPE — Symmetric MAPE

$$
\text{sMAPE} = \frac{200}{n}\sum_{t=1}^{n} \frac{|e_t|}{|y_t| + |\hat{y}_t|}
$$

Addresses the asymmetry of MAPE. Bounded in $[0 \%, 200 \%]$.

### Theil's U statistic

$$
U_1 = \frac{\text{RMSE}(\hat{y})}{\text{RMSE}(\hat{y}^{\text{naive}})}
$$

where the naive benchmark is the random walk: $\hat{y}^{\text{naive}}_{t|t-1} = y_{t-1}$.

| $U_1$ value | Interpretation |
|-------------|---------------|
| $U_1 < 1$ | Model beats the random walk |
| $U_1 = 1$ | Equal to the random walk |
| $U_1 > 1$ | Worse than the random walk — reconsider the model |

Theil's $U_2$ (the corrected version) is:

$$
U_2 = \frac{\sqrt{\sum_{t=1}^{n-1}\left(\frac{\hat{y}_{t+1|t} - y_{t+1}}{y_t}\right)^2}}{\sqrt{\sum_{t=1}^{n-1}\left(\frac{y_{t+1} - y_t}{y_t}\right)^2}}
$$

kalmanbox computes both $U_1$ and $U_2$ in `forecast_metrics()`.

---

## 4. Rolling window evaluation

Fixed-origin ("recursive") evaluation trains on all data up to $t$ and forecasts $h$ steps
ahead. Rolling-window evaluation uses a fixed-length training window:

```
Training window: [t - W + 1, t]   →   Forecast: [t+1, t+h]
```

This is the recommended protocol for comparing models because:

- It mimics the actual forecasting process.
- It reveals whether performance degrades over the sample (non-stationarity).
- It uses more evaluation points than a single train/test split.

```python
from kalmanbox.diagnostics import rolling_forecast_evaluation

rfev = rolling_forecast_evaluation(
    model,
    y,
    window: int = 60,    # training window length
    horizon: int = 12,   # steps ahead to forecast
    step: int = 1,       # roll by 1 observation each time
    metrics: list[str] = ["rmse", "mae", "mape", "theil_u"],
)

print(rfev.summary())    # average metrics across all origins
rfev.plot()              # RMSE over time by horizon
```

---

## 5. Diebold-Mariano test

The **Diebold-Mariano (DM) test** (Diebold & Mariano 1995) tests whether two competing
models have equal predictive accuracy. It is the standard tool for comparing state-space
forecasters.

### Formulation

Let $L(e_t^{(1)})$ and $L(e_t^{(2)})$ be loss functions applied to the errors of models
1 and 2 respectively. Common choices:

- Squared error: $L(e) = e^2$
- Absolute error: $L(e) = |e|$

Define the **loss differential**:

$$
d_t = L(e_t^{(1)}) - L(e_t^{(2)})
$$

The null hypothesis is equal predictive accuracy: $H_0: \mathbb{E}[d_t] = 0$.

### DM statistic

$$
DM = \frac{\bar{d}}{\sqrt{\hat{\gamma}_0 / n + 2\sum_{j=1}^{h-1}\hat{\gamma}_j / n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

where $\bar{d}$ is the sample mean of $d_t$, $\hat{\gamma}_j = \text{Cov}(d_t, d_{t-j})$
are sample autocovariances, and $h$ is the forecast horizon. The long-run variance
estimator in the denominator (a HAC estimator) accounts for the serial correlation in
multi-step loss differentials.

**Harvey, Leybourne & Newbold (1997) correction** adjusts the DM statistic for small
samples by multiplying by $\sqrt{(n+1-2h+h(h-1)/n)/n}$ and comparing to a $t$-distribution
with $n-1$ degrees of freedom. This correction is applied by default in kalmanbox.

### Interpretation

| DM $p$-value | Conclusion |
|-------------|------------|
| $p < 0.05$, $DM < 0$ | Model 1 is significantly better |
| $p < 0.05$, $DM > 0$ | Model 2 is significantly better |
| $p \geq 0.05$ | No significant difference in forecast accuracy |

!!! warning "DM test requires genuine out-of-sample errors"
    The DM test is valid only for **out-of-sample** forecast errors on data not used for
    estimation. In-sample residuals are too optimistic and will produce misleading DM
    statistics. Use rolling or recursive evaluation schemes.

---

## 6. API reference

### `prediction_errors()`

```python
from kalmanbox.diagnostics import prediction_errors

PredictionErrorResult = prediction_errors(
    results,                        # KalmanResults object
    horizon: int = 1,               # Forecast horizon h
    standardize: bool = False,      # Divide by sqrt(F_{t+h|t})
)
```

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.errors` | `ndarray` | Raw errors $e_{t+h|t}$, shape $(n - h,)$ |
| `.std_errors` | `ndarray` | Standardised errors (if `standardize=True`) |
| `.variances` | `ndarray` | Forecast error variances $F_{t+h|t}$ |
| `.horizon` | `int` | Forecast horizon |

### `forecast_metrics()`

```python
from kalmanbox.diagnostics import forecast_metrics

metrics = forecast_metrics(
    results,
    horizon: int = 1,
    y_actual: np.ndarray | None = None,    # Supply if not in results
    benchmark: str = "rw",                  # Benchmark for Theil U ("rw" = random walk)
    metrics: list[str] = ["rmse", "mae", "mape", "smape", "theil_u1", "theil_u2",
                           "bias", "bias_pct", "uc", "ur", "um"],
)
```

Returns a `pd.Series` with all requested metrics.

```python
m = forecast_metrics(results, horizon=1)
print(m)
# rmse         0.183
# mae          0.141
# mape         4.21
# smape        4.18
# theil_u1     0.72
# theil_u2     0.68
# bias        -0.003
# bias_pct    -0.08
# uc           0.89
# ur           0.06
# um           0.05
```

### `dm_test()`

```python
from kalmanbox.diagnostics import dm_test

DMTestResult = dm_test(
    errors_1: np.ndarray,           # Forecast errors from model 1
    errors_2: np.ndarray,           # Forecast errors from model 2
    horizon: int = 1,               # Forecast horizon (for HAC adjustment)
    loss: str = "squared",          # "squared" | "absolute"
    correction: str = "hn97",       # "hn97" (Harvey-Newbold) | "none"
    alternative: str = "two-sided", # "two-sided" | "less" | "greater"
)
```

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.statistic` | `float` | DM test statistic |
| `.pvalue` | `float` | $p$-value |
| `.mean_diff` | `float` | $\bar{d}$ = mean loss differential |
| `.reject` | `bool` | Whether to reject equal accuracy at 5 % |
| `.better_model` | `int | None` | 1 or 2 if one is significantly better |

### `rolling_forecast_evaluation()`

```python
from kalmanbox.diagnostics import rolling_forecast_evaluation

rolling_forecast_evaluation(
    model,                          # Unfitted kalmanbox model
    y: np.ndarray,
    window: int,                    # Fixed training window length
    horizon: int = 1,               # Max forecast horizon
    step: int = 1,                  # Roll step size
    refit: bool = True,             # Re-estimate parameters each window
    metrics: list[str] = ["rmse", "mae", "mape"],
)
```

---

## 7. Examples

### Example 1: evaluate one-step-ahead BSM forecasts

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.diagnostics import prediction_errors, forecast_metrics

# UK quarterly GDP log-growth (simulated)
from kalmanbox.datasets import load_uk_gdp
y = load_uk_gdp(log=True)

# Fit the BSM
model = BSM(period=4)
results = model.fit(y)

# Compute one-step prediction errors
pe = prediction_errors(results, horizon=1)

# Print accuracy metrics
m = forecast_metrics(results, horizon=1)
print(m)
```

**Expected output**:

```
rmse        0.0042
mae         0.0031
mape        0.4123    # percent
smape       0.4119
theil_u1    0.681     # beats random walk
theil_u2    0.674
bias       -0.0001
bias_pct   -0.0012
uc          0.923     # most error is unsystematic
ur          0.045
um          0.032
```

**Interpretation**:

- Theil $U_1 = 0.68 < 1$: the BSM is substantially better than a naive random walk.
- Bias proportion $= 0.032$ (3.2 %): almost no systematic bias — model forecasts are
  centred.
- Covariance proportion $= 0.923$ (92.3 %): most of the error is unsystematic and
  irreducible.

---

### Example 2: comparing BSM vs Local Level with Diebold-Mariano

```python
import numpy as np
from kalmanbox import BSM, LocalLevelModel
from kalmanbox.diagnostics import dm_test, rolling_forecast_evaluation

from kalmanbox.datasets import load_airline
y = np.log(load_airline())    # log of monthly airline passengers (144 obs)

# Rolling out-of-sample evaluation: train on 96 obs, forecast next 1 month
results_bsm = rolling_forecast_evaluation(BSM(period=12), y, window=96, horizon=1)
results_ll  = rolling_forecast_evaluation(LocalLevelModel(), y, window=96, horizon=1)

e_bsm = results_bsm.errors[1]    # 1-step-ahead errors, shape (48,)
e_ll  = results_ll.errors[1]

# Diebold-Mariano test: BSM (1) vs LocalLevel (2)
dm = dm_test(e_bsm, e_ll, horizon=1, loss="squared")

print(f"DM statistic : {dm.statistic:.3f}")
print(f"p-value      : {dm.pvalue:.4f}")
print(f"Mean loss diff: {dm.mean_diff:.6f}")
if dm.reject:
    print(f"Model {dm.better_model} has significantly better forecast accuracy.")
else:
    print("No significant difference in forecast accuracy at 5% level.")
```

**Expected output**:

```
DM statistic : -3.847
p-value      : 0.0002
Mean loss diff: -0.000312
Model 1 has significantly better forecast accuracy.
```

**Interpretation**: The BSM significantly outperforms the Local Level model for the airline
data. This is expected because the airline series has a strong deterministic seasonal pattern
that the BSM captures via the stochastic seasonal component, while the Local Level model
cannot. The DM $p$-value of 0.0002 provides strong evidence against equal predictive
accuracy.

---

### Example 3: rolling window RMSE by horizon

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.diagnostics import rolling_forecast_evaluation

from kalmanbox.datasets import load_airline
y = np.log(load_airline())

rfev = rolling_forecast_evaluation(
    BSM(period=12), y,
    window=96,
    horizon=12,    # evaluate 1–12 step ahead
    step=1,
    refit=True,
    metrics=["rmse", "mape", "theil_u1"],
)

# Print per-horizon summary
print(rfev.summary())
```

**Expected output**:

```
Rolling Forecast Evaluation (window=96, origins=48)
====================================================
Horizon  RMSE     MAPE(%)  Theil U1
-------  -------  -------  --------
1        0.00418  0.413    0.648
2        0.00591  0.584    0.712
3        0.00734  0.726    0.773
4        0.00821  0.812    0.801
6        0.01023  1.013    0.844
12       0.01347  1.337    0.891
```

**Interpretation**: RMSE grows with the horizon — expected for any forecasting model as
uncertainty accumulates over time. Theil $U_1$ remains below 1 at all horizons, confirming
the BSM beats the random-walk benchmark even at 12 steps ahead.

---

## 8. Interval forecast accuracy

Beyond point forecasts, kalmanbox computes the theoretical forecast intervals from the
innovation variance $F_{t+h|t}$ and evaluates their empirical coverage:

```python
from kalmanbox.diagnostics import interval_forecast_metrics

ifm = interval_forecast_metrics(
    results,
    horizon: int = 1,
    alpha: float = 0.05,    # Nominal 95% prediction interval
)

print(f"Nominal coverage: {(1 - alpha)*100:.0f}%")
print(f"Empirical coverage: {ifm.coverage*100:.1f}%")
print(f"Mean interval width: {ifm.mean_width:.4f}")
print(f"Winkler score: {ifm.winkler_score:.4f}")
```

| Metric | Description |
|--------|-------------|
| **Coverage** | Fraction of actuals inside the prediction interval |
| **Mean width** | Average interval length — narrower is better (holding coverage fixed) |
| **Winkler score** | Width + penalty for misses; lower is better |

A well-calibrated model achieves empirical coverage close to the nominal level (e.g.
94–96 % for a nominal 95 % interval).

---

## Related

- [Innovation tests](innovation-tests.md)
- [CUSUM and stability tests](cusum.md)
- [Information criteria](information-criteria.md)
- [User guide: forecasting](../user-guide/kalman/forecasting.md)
- [User guide: BSM](../user-guide/structural/bsm.md)
- [Theory: MLE theory](../theory/mle-theory.md)
- [API: diagnostics module](../api/diagnostics.md)
- [Tutorials: airline passengers BSM](../tutorials/airline-bsm.md)
