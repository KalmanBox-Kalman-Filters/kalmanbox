# Auxiliary Residuals

Auxiliary residuals, introduced by Koopman, Harvey & Doornik (1999), are smoother-based
diagnostics that reveal **where** in the sample a model fails — identifying specific
time points with outlying observations or abrupt state shifts. Unlike innovations
(which are filter-based and one-sided), auxiliary residuals condition on the full sample
and therefore have higher power for detecting isolated departures.

---

## 1. Background and motivation

The Kalman innovations $v_t = y_t - Z_t a_t$ are optimal one-step-ahead residuals but
are "blurred" by the filter's recursive nature: a single outlier at time $t$ contaminates
innovations at $t$, $t+1$, $\ldots$ as the filter slowly adjusts its state estimate.

**Auxiliary residuals** overcome this by using the full-information smoother. They
localise the departure to the time point where it actually occurred, making them far
more effective for:

- **Outlier detection** in the observation equation.
- **Level-shift and slope-break detection** in the state equation.
- **Model influence analysis** — identifying which observations drive the estimated trajectory.

!!! info "Reference"
    S.J. Koopman, A.C. Harvey & J.A. Doornik (1999). *Stamp 5.0: Structural Time Series
    Analyser, Modeller and Predictor*. London: Timberlake Consultants. The theory is also
    presented in Harvey (1989) *Forecasting, Structural Time Series Models and the Kalman
    Filter*, Chapter 5.

---

## 2. Observation auxiliary residuals

### 2.1 Definition

Let $\hat{\alpha}_t = \mathbb{E}[\alpha_t \mid y_{1:n}]$ be the smoothed state. The
**observation auxiliary residual** (or smoother observation residual) at time $t$ is:

$$
\hat{e}_t = y_t - Z_t \hat{\alpha}_t
$$

This is the residual between the observed $y_t$ and what the *smoothed* state predicts
at that time — unlike the innovation $v_t$, it uses the full-sample estimate of $\alpha_t$.

### 2.2 Variance and distribution

The variance of $\hat{e}_t$ is:

$$
\text{Var}(\hat{e}_t) = D_t = H_t - Z_t P_{t|n} Z_t^\top
$$

where $P_{t|n} = \text{Var}(\hat{\alpha}_t \mid y_{1:n})$ is the smoothed state
covariance. Note that $D_t \leq H_t$ always: smoothing reduces residual variance
compared to the observation noise.

Under correct model specification:

$$
\hat{e}_t \sim \mathcal{N}(0,\; D_t)
$$

The standardised observation residuals are:

$$
\tilde{e}_t = D_t^{-1/2}\, \hat{e}_t \sim \mathcal{N}(0, I)
$$

### 2.3 Recursive computation via the disturbance smoother

The observation auxiliary residuals can be computed efficiently via the
**disturbance smoother** of Koopman (1993) without forming $\hat{\alpha}_t$ explicitly.
The smoother runs backward through the filter output, producing:

$$
\hat{e}_t = H_t r_{t-1}
$$

$$
D_t = H_t - H_t N_{t-1} H_t
$$

where $r_t$ and $N_t$ satisfy the backward recursions (see
[Theory: Smoothing](../theory/smoothing-theory.md) for full derivations):

$$
r_{t-1} = Z_t^\top F_t^{-1} v_t + L_t^\top r_t, \qquad r_n = 0
$$

$$
N_{t-1} = Z_t^\top F_t^{-1} Z_t + L_t^\top N_t L_t, \qquad N_n = 0
$$

with $L_t = T_t - K_t Z_t$ and $K_t = T_t P_t Z_t^\top F_t^{-1}$ (the Kalman gain).

!!! note "Connection to the innovation smoother"
    The quantity $r_t$ is the **smoothed innovation vector** — it accumulates future
    innovation information backward through the sample, converting one-sided filter
    residuals into two-sided smoother residuals.

---

## 3. State auxiliary residuals

### 3.1 Definition

The **state auxiliary residual** at time $t$ measures the unexpected change in the
state between $t$ and $t+1$, conditioned on all observations:

$$
\hat{u}_t = \text{Var}(\eta_t)^{-1/2} \cdot \hat{\eta}_t
= Q_t^{-1/2} \hat{\eta}_t
$$

where $\hat{\eta}_t = \mathbb{E}[\eta_t \mid y_{1:n}]$ is the smoothed state disturbance
and $Q_t$ is the state innovation covariance.

Equivalently:

$$
\hat{\eta}_t = Q_t R_t^\top r_t
$$

with variance:

$$
\text{Var}(\hat{\eta}_t) = Q_t - Q_t R_t^\top N_t R_t Q_t
$$

The standardised state residual is:

$$
\tilde{u}_t = \text{Var}(\hat{\eta}_t)^{-1/2} \hat{\eta}_t \sim \mathcal{N}(0, I)
$$

under correct specification.

### 3.2 Interpretation

The state residual $\hat{\eta}_t$ represents the estimated shock to the state at
time $t$, using all available data. A large $|\tilde{u}_t|$ at time $t$ signals
that an unusually large state transition occurred — a **level shift**, **slope break**,
or other abrupt structural change — precisely at time $t$.

**Key contrast with observation residuals**:

| Residual type | Detects | Localised to |
|---------------|---------|--------------|
| $\hat{e}_t$ — observation | Outliers, measurement errors | Observation equation |
| $\hat{u}_t$ — state | Level shifts, trend breaks | State transition equation |

---

## 4. Standardisation and testing

### 4.1 Standardised residuals

Both types are standardised by their respective square-root variances to yield
unit-variance sequences under the null:

$$
\tilde{e}_t = D_t^{-1/2} \hat{e}_t, \quad
\tilde{u}_t = \text{Var}(\hat{\eta}_t)^{-1/2} \hat{\eta}_t
$$

For a scalar model these reduce to simple ratios:

$$
\tilde{e}_t = \frac{\hat{e}_t}{\sqrt{D_t}}, \qquad
\tilde{u}_t = \frac{\hat{\eta}_t}{\sqrt{\text{Var}(\hat{\eta}_t)}}
$$

### 4.2 Outlier detection threshold

The most direct use of auxiliary residuals is comparing the standardised values
against normal critical values:

| $|\tilde{e}_t|$ or $|\tilde{u}_t|$ | Signal |
|--------------------------------------|--------|
| $< 2.0$ | No evidence of anomaly |
| $2.0$–$3.3$ | Mild anomaly — investigate |
| $3.3$–$4.0$ | Probable outlier / level shift |
| $> 4.0$ | Strong outlier / structural break |

These thresholds correspond to the Bonferroni-corrected normal critical values for
simultaneous testing over $n$ time points:

$$
z_{\alpha/(2n)} \approx \Phi^{-1}\!\left(1 - \frac{\alpha}{2n}\right)
$$

At $\alpha = 0.05$ and $n = 100$: $z_{\alpha/(2n)} \approx 3.29$.

### 4.3 Normality test on auxiliary residuals

Under correct specification, both $\{\tilde{e}_t\}$ and $\{\tilde{u}_t\}$ should be
i.i.d. $\mathcal{N}(0, 1)$. Apply the Doornik-Hansen test:

$$
DH = z_1^2 + z_2^2 \xrightarrow{d} \chi^2_2
$$

where $z_1$ and $z_2$ are normalised skewness and kurtosis transforms. A rejection
indicates persistent non-Gaussianity — multiple outliers, fat tails, or structural
changes not captured by the model.

### 4.4 Independence of auxiliary residuals

Unlike innovations, auxiliary residuals are **not** generally uncorrelated — they
are produced by the smoother which uses all observations, introducing some dependence.
Serial correlation tests should therefore not be applied directly to auxiliary residuals
for model specification testing (use innovations for that purpose).

However, **clustering of large residuals** — several consecutive time points with
$|\tilde{e}_t| > 2$ — does indicate a missing structural component.

---

## 5. Types of anomaly: observation vs state residuals

### 5.1 Additive outlier (AO)

An **additive outlier** at time $\tau$ is an isolated spike in $y_\tau$ that does not
affect the underlying state:

$$
y_\tau = Z_\tau \alpha_\tau + \varepsilon_\tau + \omega \cdot \mathbf{1}_{t=\tau}
$$

**Signature in residuals**:
- $|\tilde{e}_\tau|$ large (the observation residual flags the affected time point).
- $|\tilde{u}_\tau|$ and $|\tilde{u}_{\tau-1}|$ remain small (state undisturbed).

### 5.2 Level shift (LS) — innovation outlier

A **level shift** is a permanent jump in the state at time $\tau$:

$$
\alpha_{\tau+1} = T_\tau \alpha_\tau + \omega \cdot \mathbf{1}_{t=\tau} + R_\tau \eta_\tau
$$

**Signature in residuals**:
- $|\tilde{u}_\tau|$ large (the state residual identifies the shift).
- $|\tilde{e}_t|$ may be elevated near $\tau$ but the effect is diffuse.

### 5.3 Transient outlier (TO)

A transient outlier decays exponentially after time $\tau$, captured by a
first-order decay in the state equation.

**Signature**: Elevated $|\tilde{u}_\tau|$ with smaller but still significant
$|\tilde{e}_\tau|$.

### 5.4 Slope change

An abrupt change in the trend slope produces large $|\tilde{u}_\tau|$ in the
**slope component** of a local linear trend or BSM model.

```
Anomaly type    │  ẽ_t large  │  ũ_t large  │  Pattern
────────────────┼─────────────┼─────────────┼─────────────────────
Additive outlier│  at τ       │  no         │  spike in obs residual
Level shift     │  near τ     │  at τ       │  step in state residual
Transient outlier│ at τ       │  at τ       │  both spike, decays fast
Slope change    │  no         │  at τ (slope│  ramp in state residual
                │             │  component) │
```

---

## 6. Influence analysis

### 6.1 Observation influence on smoothed states

The influence of observation $y_s$ on the smoothed state $\hat{\alpha}_t$ is:

$$
\frac{\partial \hat{\alpha}_t}{\partial y_s} = P_{t|n} Z_t^\top F_s^{-1} \cdot \mathbf{1}_{s=t} + \text{cross-time terms}
$$

kalmanbox computes the **leverage** of each observation — the diagonal of the
hat matrix $H$ — which summarises total influence:

$$
h_t = Z_t P_{t|t-1} Z_t^\top F_t^{-1}
$$

Observations with $h_t > 2p/n$ (where $p$ is the state dimension) are
high-leverage points.

### 6.2 Cook's distance analogue

A state-space analogue of Cook's distance measures how much the smoothed trajectory
changes if observation $t$ is removed:

$$
CD_t = \frac{(\hat{\alpha}_{-t} - \hat{\alpha})^\top \Sigma^{-1} (\hat{\alpha}_{-t} - \hat{\alpha})}{p}
$$

where $\hat{\alpha}_{-t}$ is the smoothed state with $y_t$ treated as missing.
Large $CD_t$ identifies observations that substantially pull the estimated trajectory.

---

## 7. API reference

### `auxiliary_residuals()`

```python
from kalmanbox.diagnostics import auxiliary_residuals

AuxResiduals = auxiliary_residuals(
    results,                         # KalmanResults (smoother must have been run)
    standardize: bool = True,        # Divide by sqrt(D_t) and sqrt(Var(η̂_t))
    alpha: float = 0.05,             # Significance level for outlier thresholds
    bonferroni: bool = True,         # Apply Bonferroni correction across n tests
)
```

**`AuxResiduals` attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.obs_residuals` | `ndarray` | $\hat{e}_t$ raw observation residuals, shape $(n, p)$ |
| `.obs_variances` | `ndarray` | $D_t$ variances, shape $(n, p, p)$ |
| `.obs_std` | `ndarray` | $\tilde{e}_t$ standardised, shape $(n, p)$ |
| `.state_residuals` | `ndarray` | $\hat{\eta}_t$ raw state disturbances, shape $(n, m)$ |
| `.state_variances` | `ndarray` | $\text{Var}(\hat{\eta}_t)$, shape $(n, m, m)$ |
| `.state_std` | `ndarray` | $\tilde{u}_t$ standardised, shape $(n, m)$ |
| `.threshold` | `float` | Outlier detection threshold (Bonferroni or plain) |
| `.obs_outliers` | `ndarray[bool]` | Flags for observation residual outliers |
| `.state_outliers` | `ndarray[bool]` | Flags for state residual outliers |
| `.plot()` | method | Plot both residual series with threshold bands |
| `.summary()` | method | Print outlier and normality test results |

### `observation_residuals()`

```python
from kalmanbox.diagnostics import observation_residuals

ObsRes = observation_residuals(
    results,
    standardize: bool = True,
    alpha: float = 0.05,
    bonferroni: bool = True,
)
# Returns subset of AuxResiduals focused on observation equation only
```

### `state_residuals()`

```python
from kalmanbox.diagnostics import state_residuals

StateRes = state_residuals(
    results,
    standardize: bool = True,
    state_component: int | None = None,  # None = all components
    alpha: float = 0.05,
    bonferroni: bool = True,
)
```

### `plot_auxiliary_residuals()`

```python
from kalmanbox.diagnostics import plot_auxiliary_residuals

fig = plot_auxiliary_residuals(
    results,
    alpha: float = 0.05,
    bonferroni: bool = True,
    dates: pd.DatetimeIndex | None = None,
    annotate_outliers: bool = True,     # Label flagged time points
    figsize: tuple = (14, 8),
)
```

Produces a two-panel figure:
1. **Top**: standardised observation residuals with $\pm z_{\alpha/(2n)}$ bands.
2. **Bottom**: standardised state residuals (one row per state component) with bands.

---

## 8. Examples

### Example 1: detecting an additive outlier in Nile river data

The Nile annual flow series (Harvey 1989) contains a suspected outlier around 1899
coinciding with the construction of the first Aswan Dam.

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.datasets import load_nile
from kalmanbox.diagnostics import auxiliary_residuals, plot_auxiliary_residuals

nile = load_nile()  # shape (100,), years 1871–1970

model = LocalLevelModel()
results = model.fit(nile)

aux = auxiliary_residuals(results, alpha=0.05, bonferroni=True)
print(aux.summary())
plot_auxiliary_residuals(results, annotate_outliers=True)
```

**Expected output**:

```
Auxiliary Residual Diagnostics
==============================
Model         : LocalLevel
Observations  : 100
Threshold     : ±3.29  (Bonferroni α=0.05, n=100)

Observation Residuals (ê_t)
  Doornik-Hansen normality : stat=2.84, p=0.241  [PASS]
  Outliers detected        : 1

  t=29 (1899) : ẽ_t = -3.47  ← OUTLIER (additive)
                              Observation 912 vs predicted 974
                              Magnitude: -62 Mm³ (-6.4%)

State Residuals (û_t)
  Doornik-Hansen normality : stat=1.92, p=0.383  [PASS]
  Outliers detected        : 0

  No state residuals exceed threshold.
  Largest: t=29, û_t = -1.83  (below threshold 3.29)

Interpretation:
  Large OBSERVATION residual with no corresponding STATE residual at t=29
  → Consistent with an ADDITIVE OUTLIER (measurement anomaly or one-off
    event), not a permanent level shift in the Nile flow regime.
```

**Interpretation**: The 1899 observation has an unusually low observation residual
($\tilde{e}_{29} = -3.47$), exceeding the Bonferroni threshold of 3.29, while the
state residual at the same point is only $-1.83$. This pattern — large $|\tilde{e}_t|$
with small $|\tilde{u}_t|$ — is the signature of an **additive outlier**. The Aswan Dam
lowered the 1899 measurement but did not permanently shift the underlying mean flow level
(which would produce a large state residual).

**Recommended action**: Include an indicator variable for 1899 in the observation equation:

```python
import numpy as np
from kalmanbox import LocalLevelModel

nile_adj = nile.copy()
intervention = np.zeros(len(nile))
intervention[28] = 1.0   # t=29 (0-indexed: 28)

model_robust = LocalLevelModel()
results_robust = model_robust.fit(nile_adj, X=intervention.reshape(-1, 1))
print(results_robust.summary())  # coefficient on intervention = estimated outlier magnitude
```

---

### Example 2: detecting a level shift

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import auxiliary_residuals

rng = np.random.default_rng(42)
n = 120

# Generate data with a level shift at t=60
level = np.concatenate([
    np.zeros(60),
    np.full(60, 5.0)   # +5 unit permanent jump
])
y = level + np.cumsum(rng.normal(0, 0.3, n)) + rng.normal(0, 1.0, n)

model = LocalLevelModel()
results = model.fit(y)

aux = auxiliary_residuals(results, alpha=0.05, bonferroni=True)
print(aux.summary())
```

**Expected output**:

```
Observation Residuals (ê_t)
  Outliers detected: 0
  Largest: t=60, ẽ_t = 2.11  (below threshold 3.32)

State Residuals (û_t)
  Outliers detected: 1

  t=60 : û_t = 4.87  ← OUTLIER (level shift / innovation outlier)
                       Estimated shift: +4.9 units

Interpretation:
  Large STATE residual with no significant observation residual at t=60.
  → Consistent with a LEVEL SHIFT in the underlying state at t=60.
```

**Interpretation**: The large state residual ($\tilde{u}_{60} = 4.87$) with no
corresponding observation residual confirms a **level shift** at $t = 60$. The
observation residuals remain small because the filter and smoother adapt the state
trajectory to the shift; the unusual movement appears in the state disturbance, not the
measurement noise.

**Recommended action**: Incorporate the known break:

```python
from kalmanbox import LocalLevelModel

# Intervention indicator for state equation
model_with_break = LocalLevelModel()
results_with_break = model_with_break.fit(y, level_shift_at=60)
# Or use TimeVaryingParameters model to allow the level to shift freely
```

---

### Example 3: comprehensive diagnostic workflow

```python
import numpy as np
import pandas as pd
from kalmanbox import BSM
from kalmanbox.datasets import load_airline
from kalmanbox.diagnostics import (
    auxiliary_residuals,
    innovation_tests,
    plot_auxiliary_residuals,
)
from kalmanbox.visualization import plot_components

# UK airline passengers 1949–1960 (monthly, n=144)
y = np.log(load_airline())
dates = pd.date_range("1949-01", periods=len(y), freq="ME")

model = BSM(period=12, stochastic_seasonal=True)
results = model.fit(y)

# Step 1: Innovation tests (filter-based, check overall specification)
itr = innovation_tests(results, lags=24)
print(itr.summary())

# Step 2: Auxiliary residuals (smoother-based, pinpoint anomalies)
aux = auxiliary_residuals(results, alpha=0.05, bonferroni=True)
print(aux.summary())

# Step 3: Visual inspection
plot_auxiliary_residuals(results, dates=dates, annotate_outliers=True)
plot_components(results, dates=dates)
```

**Interpreting the full workflow**:

1. **Innovation tests** tell you *whether* the model is misspecified (overall).
2. **Auxiliary residuals** tell you *where* the anomalies are and *which equation*
   is affected (observation vs state).
3. **Component plots** show *how* each structural component (trend, seasonal, cycle)
   behaves — complementing the residual evidence.

---

## 9. Relationship to other diagnostics

| Diagnostic | Residual type | Conditioning set | Primary use |
|-----------|---------------|-----------------|-------------|
| Innovation $v_t$ | Filter | $y_{1:t-1}$ | Model specification, white-noise tests |
| Observation aux $\hat{e}_t$ | Smoother | $y_{1:n}$ | Additive outlier detection |
| State aux $\hat{u}_t$ | Smoother | $y_{1:n}$ | Level shift, slope-break detection |
| CUSUM | Filter (cumulative) | $y_{1:t}$ | Structural break location |
| Prediction error | Filter | $y_{1:t-1}$ | Out-of-sample accuracy |

---

## Related

- [State Smoothness Diagnostics](state-smoothness.md) — roughness metrics and over/under-smoothing checks
- [Innovation Tests](innovation-tests.md) — white-noise tests on filter residuals
- [CUSUM & Stability](cusum.md) — structural-break detection using cumulative innovations
- [Residual Analysis](residuals.md) — standardised innovation overview
- [Theory: Smoothing](../theory/smoothing-theory.md) — RTS and disturbance smoother derivations
- [User guide: Kalman Smoother](../user-guide/kalman/rts-smoother.md) — practical smoother usage
- [API: diagnostics module](../api/diagnostics.md)
