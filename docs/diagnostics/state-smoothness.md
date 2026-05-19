# State Smoothness Diagnostics

State smoothness diagnostics assess whether the estimated states from a Kalman smoother
are appropriately smooth — neither too rigid (over-smoothing) nor too noisy
(under-smoothing). Since the degree of smoothness is directly controlled by the
**signal-to-noise ratio** (SNR), these diagnostics also serve as indirect checks on
hyperparameter calibration.

---

## 1. What does "smoothness" mean for state estimates?

In a state-space model, the smoothed state $\hat{\alpha}_t = \mathbb{E}[\alpha_t \mid y_{1:n}]$
balances two forces:

- **Fidelity to the data**: the states should track the observed series closely.
- **Regularisation from the dynamics**: the state transition $\alpha_{t+1} = T_t \alpha_t + R_t \eta_t$ penalises rapid changes by the innovation variance $Q$.

The trade-off is governed by the **signal-to-noise ratio**:

$$
q = \frac{\sigma^2_\eta}{\sigma^2_\varepsilon}
$$

where $\sigma^2_\eta$ is the state innovation variance and $\sigma^2_\varepsilon$ is the
observation noise variance. A large $q$ allows the state to move rapidly (less smooth);
a small $q$ forces the state to stay nearly flat (more smooth).

!!! note "Smoothness is a spectrum, not a binary"
    There is no single "correct" level of smoothness. The goal of these diagnostics is to
    detect **pathological extremes** — states that are so smooth they miss genuine signal, or
    so volatile they fit observation noise.

---

## 2. Roughness metrics

### 2.1 First-difference variance

The simplest roughness metric for a univariate smoothed state sequence
$\{\hat{\alpha}_t\}_{t=1}^n$ is the **variance of first differences**:

$$
R_1 = \frac{1}{n-1} \sum_{t=2}^{n} (\hat{\alpha}_t - \hat{\alpha}_{t-1})^2
$$

Under the local-level model at the MLE of $q$, this quantity satisfies:

$$
\mathbb{E}[R_1] \approx 2\,\sigma^2_\eta \left(1 - \rho_1\right)
$$

where $\rho_1$ is the first autocorrelation of the smoothed state. Comparing the
empirical $R_1$ against the model-implied expectation provides an informal calibration
check.

### 2.2 Generalised roughness penalty

For higher-order smoothers (e.g. local linear trend, BSM), the appropriate roughness
is measured by the second or higher difference:

$$
R_d = \frac{1}{n-d} \sum_{t=d+1}^{n} (\Delta^d \hat{\alpha}_t)^2,
\qquad \Delta^d = (1 - B)^d
$$

| Model | Natural roughness order $d$ |
|-------|----------------------------|
| Local Level | 1 (random walk) |
| Local Linear Trend | 2 (integrated random walk) |
| BSM with trend | 2 (trend component) |
| Spline smoother | $d$ matching the spline degree |

### 2.3 Normalised roughness ratio

To compare roughness across series with different scales, normalise by the
observation variance:

$$
\tilde{R} = \frac{R_d}{\widehat{\text{Var}}(y_t)}
$$

A normalised ratio $\tilde{R} \approx 0$ indicates a near-flat estimated state
(potential over-smoothing). A ratio $\tilde{R} \approx 1$ means the state tracks
the raw observations almost exactly (potential under-smoothing).

### 2.4 Effective degrees of freedom

The **hat matrix** $H$ of the smoother maps observations to fitted values:
$\hat{y} = H y$. The trace $\text{tr}(H)$ gives the **effective degrees of freedom**
(EDF) — a scale-free smoothness measure:

$$
\text{EDF} = \text{tr}(H) \in [1,\; n]
$$

- $\text{EDF} \approx 1$: state is a constant (maximum smoothing).
- $\text{EDF} \approx n$: state replicates each observation (no smoothing).

kalmanbox approximates EDF efficiently without forming the full $H$ matrix:

$$
\text{EDF} \approx \sum_{t=1}^{n} \frac{P_{t|t-1}}{F_t}
$$

where $P_{t|t-1}$ is the one-step prediction variance and $F_t = P_{t|t-1} + H_t$ is
the innovation variance.

---

## 3. Filtered vs smoothed state comparison

The gap between the **filtered** state $a_t = \mathbb{E}[\alpha_t \mid y_{1:t}]$ and
the **smoothed** state $\hat{\alpha}_t = \mathbb{E}[\alpha_t \mid y_{1:n}]$ reveals
how much future information revises each estimate:

$$
\delta_t = \hat{\alpha}_t - a_t
$$

### 3.1 Revision magnitude

$$
\text{MeanRevision} = \frac{1}{n} \sum_{t=1}^{n} |\delta_t|, \qquad
\text{MaxRevision} = \max_t |\delta_t|
$$

Large revisions at particular time points identify periods where the smoother has
substantially corrected the filter — often near genuine structural shifts or outliers.

### 3.2 Revision variance ratio

$$
\text{RVR} = \frac{\text{Var}(\hat{\alpha})}{\text{Var}(a)}
$$

Under correct specification: $\text{RVR} < 1$ always (the smoother reduces uncertainty).
An $\text{RVR} \approx 1$ suggests the smoother adds almost no information — the series
may be too noisy or the SNR too low for retrospective smoothing to help.

### 3.3 Smoothing gain

The theoretical smoothing gain at each time point is:

$$
G_t = I - \frac{P_{t|n}}{P_{t|t}}
$$

where $P_{t|n}$ is the smoothed covariance and $P_{t|t}$ is the filtered covariance.
$G_t = 0$ means no smoothing (occurs when $q \to \infty$); $G_t \to I$ means
maximum revision (occurs when $q \to 0$).

---

## 4. Over-smoothing and under-smoothing

### 4.1 Over-smoothing

**Definition**: The estimated state is excessively constrained, missing genuine
variation in the underlying signal.

**Causes**:
- The SNR $q$ is estimated too small (relative to the true value).
- An overly restrictive model (e.g. forcing a constant level when a trend is present).
- Diffuse initialisation that has not yet converged to the stationary distribution.

**Symptoms**:
- Very low roughness ratio $\tilde{R} \ll 1$.
- Large, patterned innovation residuals (the model is forced to explain via noise what
  should be captured by the state).
- Significant autocorrelation in innovations at low lags.
- Smoothed state nearly flat despite visible trends in the data.
- Large residuals clustered around turning points.

!!! warning "Over-smoothing is the more common failure mode"
    Over-smoothing often goes undetected because the model may still pass normality
    tests. Always inspect the roughness ratio and the filtered/smoothed comparison
    plot alongside innovation tests.

**Diagnostic checklist**:

```
□  Roughness ratio R̃ < 0.05 for a model with visible trend
□  EDF < 5 for a series with n > 100
□  Innovation ACF shows significant autocorrelation at lags 1–3
□  Max smoothed revision > 3σ at multiple time points
```

### 4.2 Under-smoothing

**Definition**: The state tracks the observations too closely, fitting observation
noise rather than genuine signal.

**Causes**:
- The SNR $q$ is estimated too large.
- Insufficient regularisation in the state equation.
- Misspecified observation noise variance ($H$ too small).
- Near-unit-root state with insufficient stationarising penalty.

**Symptoms**:
- Roughness ratio $\tilde{R} \approx 1$.
- Smoothed state visually indistinguishable from raw data.
- Very small innovations (the state tracks the data almost exactly).
- Low EDF relative improvement from filtering to smoothing.

**Diagnostic checklist**:

```
□  Roughness ratio R̃ > 0.8
□  EDF > 0.9·n
□  Innovations are negligibly small compared to the observation scale
□  Filtered and smoothed states are nearly identical (RVR ≈ 1)
```

---

## 5. Signal-to-noise ratio as a smoothness indicator

The SNR $q$ is the primary tuning parameter for smoothness. kalmanbox provides
a diagnostic breakdown of the estimated SNR alongside its uncertainty:

### Estimated SNR from MLE

After fitting via MLE:

$$
\hat{q} = \frac{\hat{\sigma}^2_\eta}{\hat{\sigma}^2_\varepsilon}
$$

The **smoothing weight** $k^*$ — the steady-state Kalman gain — is related to $q$ by:

$$
k^* = \frac{-q + \sqrt{q^2 + 4q}}{2}
$$

| $\hat{q}$ | $k^*$ | Interpretation |
|-----------|--------|----------------|
| $< 0.01$ | $< 0.1$ | Nearly constant state (heavy smoothing) |
| $0.01$–$0.1$ | $0.1$–$0.27$ | Moderate smoothing |
| $0.1$–$1.0$ | $0.27$–$0.62$ | Balanced signal tracking |
| $> 1.0$ | $> 0.62$ | State tracks data closely (light smoothing) |

### SNR profile likelihood

To assess uncertainty in $\hat{q}$, kalmanbox provides a profile likelihood plot over
a grid of $q$ values:

```python
from kalmanbox.diagnostics import snr_profile

profile = snr_profile(results, q_grid=np.logspace(-3, 1, 100))
profile.plot()  # log-likelihood vs log(q) with 95% CI band
```

---

## 6. Cross-validation for smoothness calibration

When the MLE estimate of $q$ is uncertain or the model is specified to use a fixed
smoothness level, **leave-one-out cross-validation** (LOO-CV) provides an
empirical calibration criterion.

### Leave-one-out prediction error

For each held-out observation $y_t$, the LOO prediction is the filtered prediction
$\hat{y}_{t|t-1} = Z_t a_t$ from the model fitted on all other observations. The
LOO criterion is:

$$
\text{LOOCV}(q) = \frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_{t|t-1}(q))^2
$$

Because the Kalman filter is sequential, the LOO residuals are exactly the
**innovations** $v_t = y_t - Z_t a_t$, so no refitting is required:

$$
\text{LOOCV}(q) = \frac{1}{n} \sum_{t=1}^{n} v_t^2
$$

!!! tip "LOO-CV is free for state-space models"
    Unlike cross-section regression, the innovations from a single Kalman filter pass
    are already the LOO residuals. You only need to refit when varying $q$ on a grid.

### K-fold cross-validation

For longer series, $k$-fold CV avoids refitting $n$ times:

```python
from kalmanbox.diagnostics import cross_validate

cv_result = cross_validate(
    model,
    y,
    cv=TimeSeriesSplit(n_splits=5),
    q_grid=np.logspace(-3, 1, 20),
    scoring="mse",
)
cv_result.plot()          # CV score vs q
print(cv_result.optimal_q)
```

---

## 7. API reference

### `state_smoothness()`

```python
from kalmanbox.diagnostics import state_smoothness

SmoothnessResult = state_smoothness(
    results,                      # KalmanResults object (must include smoother output)
    diff_order: int = 1,          # Differencing order d for roughness (1 or 2)
    normalize: bool = True,       # Normalise roughness by Var(y)
    compute_edf: bool = True,     # Compute effective degrees of freedom
)
```

**`SmoothnessResult` attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.roughness` | `float` | $R_d$ — raw roughness statistic |
| `.roughness_ratio` | `float` | $\tilde{R}$ — normalised roughness ratio |
| `.edf` | `float` | Effective degrees of freedom |
| `.mean_revision` | `ndarray` | $|\delta_t|$ for each time point |
| `.revision_variance_ratio` | `float` | $\text{Var}(\hat{\alpha}) / \text{Var}(a)$ |
| `.snr` | `float` | Estimated SNR $\hat{q}$ |
| `.smoothing_gain` | `ndarray` | $G_t$ for each time point |
| `.over_smooth_flag` | `bool` | Heuristic flag for over-smoothing |
| `.under_smooth_flag` | `bool` | Heuristic flag for under-smoothing |
| `.summary()` | method | Print diagnostic summary |

### `roughness()`

```python
from kalmanbox.diagnostics import roughness

R = roughness(
    states: np.ndarray,           # Smoothed states array, shape (n, m) or (n,)
    diff_order: int = 1,          # Order of differencing
    normalize_by: float | None = None,  # Denominator for normalisation (Var(y) if None)
)
# Returns: float (scalar) or ndarray (per state component)
```

### `plot_state_smoothness()`

```python
from kalmanbox.diagnostics import plot_state_smoothness

fig = plot_state_smoothness(
    results,
    state_index: int = 0,         # Which state component to plot
    dates: pd.DatetimeIndex | None = None,
    show_revisions: bool = True,  # Shade |δ_t| = |smoothed - filtered|
    show_gain: bool = True,       # Plot smoothing gain G_t
    figsize: tuple = (14, 8),
)
```

Produces a three-panel figure:
1. **Top**: filtered and smoothed state with $\pm 2\sigma$ bands.
2. **Middle**: pointwise revision magnitude $|\delta_t|$.
3. **Bottom**: smoothing gain $G_t$ over time.

### `snr_profile()`

```python
from kalmanbox.diagnostics import snr_profile

SNRProfile = snr_profile(
    results,
    q_grid: np.ndarray | None = None,   # Default: logspace(-3, 2, 100)
    ci_level: float = 0.95,             # Confidence interval for profile CI
)
# Returns SNRProfile with .plot() and .confidence_interval attributes
```

---

## 8. Examples

### Example 1: diagnosing a Local Level model on Nile data

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.datasets import load_nile
from kalmanbox.diagnostics import state_smoothness, plot_state_smoothness

nile = load_nile()  # Annual Nile flow 1871–1970 (n=100)

model = LocalLevelModel()
results = model.fit(nile)

sm = state_smoothness(results, diff_order=1, normalize=True)
print(sm.summary())
```

**Expected output**:

```
State Smoothness Diagnostics
============================
Model         : LocalLevel
Observations  : 100
SNR (q)       : 0.076   (moderate smoothing)
Steady-state gain k* : 0.236

Roughness
  R_1 (raw)        :  821.4
  R̃  (normalised)  :  0.083    [moderate — typical for q=0.076]
  EDF              : 19.3 / 100

Revision Statistics
  Mean |revision|  : 18.2 (obs units)
  Max  |revision|  : 62.4 at t=29 (1899)
  Revision Var Ratio: 0.71

Flags
  Over-smoothing   : No
  Under-smoothing  : No

Conclusion: Smoothness is well-calibrated for this model.
```

**Interpretation**: The normalised roughness $\tilde{R} = 0.083$ and the EDF of 19.3
indicate moderate smoothing consistent with the estimated SNR. The large revision at
$t = 29$ (year 1899) warrants investigation — this is likely the period following the
completion of the Aswan Dam, which permanently altered the Nile flow regime. The
auxiliary residuals diagnostic (`auxiliary-residuals.md`) can pinpoint whether this is
a genuine outlier or a structural break.

---

### Example 2: detecting over-smoothing in a misspecified model

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import state_smoothness, snr_profile, plot_state_smoothness

# Simulate a series with high SNR (state changes a lot)
rng = np.random.default_rng(42)
n = 150
true_states = np.cumsum(rng.normal(0, 3.0, n))    # large state innovations
y = true_states + rng.normal(0, 1.0, n)            # small observation noise
# True q = 9.0, expected lots of state variation

# Fit with q constrained too small (over-smoothed)
model_bad = LocalLevelModel()
results_bad = model_bad.fit(y, q_fixed=0.01)       # force near-zero SNR

sm_bad = state_smoothness(results_bad, normalize=True)
print("Misspecified (q=0.01):")
print(sm_bad.summary())

# Fit with MLE
model_good = LocalLevelModel()
results_good = model_good.fit(y)

sm_good = state_smoothness(results_good, normalize=True)
print("\nMLE-fitted:")
print(sm_good.summary())

# Profile plot to see the log-likelihood surface
profile = snr_profile(results_good)
profile.plot()
```

**Expected output (misspecified)**:

```
Misspecified (q=0.01):
  SNR (q)           : 0.010   (heavy smoothing — likely misspecified)
  R̃ (normalised)   : 0.004   ← suspiciously low
  EDF               : 2.1 / 150  ← near-constant state
  Over-smoothing    : YES  ← flagged
  Conclusion: State is nearly constant despite large observed variation.
              Consider increasing q or using MLE estimation.
```

**Expected output (MLE)**:

```
MLE-fitted:
  SNR (q)           : 8.73   (light smoothing — state tracks data closely)
  R̃ (normalised)   : 0.851
  EDF               : 127.4 / 150
  Over-smoothing    : No
  Under-smoothing   : No
  Conclusion: Smoothness calibrated by MLE. Verify with innovation tests.
```

---

### Example 3: cross-validation to select SNR for a monthly series

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import cross_validate

# Monthly industrial production index (simulated)
rng = np.random.default_rng(0)
n = 240
trend = np.linspace(100, 130, n)
y = trend + np.cumsum(rng.normal(0, 0.5, n)) + rng.normal(0, 2.0, n)

model = LocalLevelModel()

cv_result = cross_validate(
    model,
    y,
    cv=TimeSeriesSplit(n_splits=5),
    q_grid=np.logspace(-3, 1, 30),
    scoring="mse",
)

cv_result.plot()
print(f"Optimal q (CV): {cv_result.optimal_q:.4f}")
print(f"Optimal q (MLE): {model.fit(y).snr:.4f}")
```

**Interpretation**: The CV curve typically has a broad minimum, meaning the MSE is
relatively insensitive to $q$ over a range. If the CV-optimal $q$ differs substantially
from the MLE estimate, this signals either a non-Gaussian series (where MLE is less
efficient) or a misspecified model structure.

---

## 9. Smoothness diagnostics decision guide

```
Start: fit smoother, call state_smoothness()
       │
       ├─ over_smooth_flag = True?
       │    ├─ Roughness ratio R̃ < 0.05?    →  SNR is severely under-estimated
       │    │   Action: refit with larger q_init, check MLE convergence
       │    ├─ EDF < 5?                       →  state is nearly constant
       │    │   Action: check if model needs a trend component
       │    └─ Innovation ACF significant?    →  model structure missing dynamics
       │        Action: add AR component or switch to local linear trend
       │
       ├─ under_smooth_flag = True?
       │    ├─ Roughness ratio R̃ > 0.85?    →  state tracks observations
       │    │   Action: increase H (observation noise floor)
       │    └─ EDF > 0.9·n?                  →  effectively no smoothing
       │        Action: add regularisation, check H > 0 constraint
       │
       └─ Neither flag?
            Run cross_validate() to confirm SNR is stable
            Proceed to auxiliary residuals for outlier detection
```

---

## Related

- [Auxiliary Residuals](auxiliary-residuals.md) — smoother-based residuals for outlier and break detection
- [Innovation Tests](innovation-tests.md) — white-noise tests on filter residuals
- [CUSUM & Stability](cusum.md) — structural-break detection
- [Prediction Error Analysis](prediction-error.md) — out-of-sample performance
- [Theory: Smoothing](../theory/smoothing-theory.md) — RTS smoother derivation and fixed-interval algorithm
- [User guide: Local Level](../user-guide/structural/local-level.md)
- [API: diagnostics module](../api/diagnostics.md)
