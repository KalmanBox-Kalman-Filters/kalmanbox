# Migration guide

Coming from another state-space library? This page maps common idioms onto
their `kalmanbox` equivalents.

## From `statsmodels.tsa.statespace`

| `statsmodels`                              | `kalmanbox`                                          |
|--------------------------------------------|------------------------------------------------------|
| `UnobservedComponents(y, level='llevel')`  | `LocalLevel(y)`                                      |
| `UnobservedComponents(y, level='lltrend')` | `LocalLinearTrend(y)`                                |
| `UnobservedComponents(y, ..., seasonal=4)` | `BasicStructuralModel(y, seasonal_periods=4)`        |
| `MLEModel.fit()`                           | `model.fit()` → `StateSpaceResults`                  |
| `results.smoother_results`                 | `results.smooth()` → `SmootherOutput`                |
| `results.get_forecast(h)`                  | `results.forecast(steps=h)`                          |
| `results.plot_diagnostics()`               | `kalmanbox.diagnostics.residual_diagnostics(results)`|

Key conventions are aligned: same matrix names ($T$, $Z$, $R$, $H$, $Q$),
same definition of filtered vs. smoothed states.

## From `pykalman`

```python
# pykalman
from pykalman import KalmanFilter
kf = KalmanFilter(transition_matrices=T, observation_matrices=Z)
state_means, _ = kf.filter(y)

# kalmanbox
from kalmanbox import KalmanFilter, StateSpaceRepresentation
ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
kf = KalmanFilter(ssr)
out = kf.run(y)        # FilterOutput with a, P, v, F, K, ...
```

Differences:

- `kalmanbox` separates the **representation** (matrices) from the
  **filter** (algorithm).
- Filter output is a structured `FilterOutput` rather than tuples.
- MLE / EM / Bayesian estimation is provided in `kalmanbox.estimation`.

## From `filterpy`

`filterpy` exposes a discrete Kalman filter at the level of single-step
`predict` / `update`. `kalmanbox` operates on whole series by default but
also exposes single-step methods on the filter classes for online use.

```python
# filterpy
from filterpy.kalman import KalmanFilter
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.predict(); kf.update(z)

# kalmanbox  (online)
from kalmanbox.filters.kalman import KalmanFilter
kf = KalmanFilter(ssr)
kf.predict_step(t)
kf.update_step(t, y_t)
```

## Naming alignment

`kalmanbox` follows the textbook conventions used by Durbin & Koopman
(2012):

- $\alpha_t$ — state vector
- $T_t$ — transition matrix
- $Z_t$ — design (observation) matrix
- $R_t$ — selection matrix
- $Q_t$ — state-disturbance covariance
- $H_t$ — observation-disturbance covariance
- $a_{t|t-1}$ — predicted state
- $a_{t|t}$  — filtered state
- $a_{t|n}$  — smoothed state
- $v_t$, $F_t$, $K_t$ — innovation, innovation covariance, Kalman gain

Variable names in `kalmanbox/filters/`, `kalmanbox/smoothers/` and
`kalmanbox/models/` use these symbols verbatim (see the `pyproject.toml`
ruff exceptions for `N803/N806`).

## Behavioural differences worth knowing

!!! warning "Diffuse initialisation"

    `kalmanbox` implements proper **exact diffuse initialisation** through
    [`DiffuseInitialization`][kalmanbox.estimation.diffuse.DiffuseInitialization].
    Some libraries default to a large-variance approximation that is
    less numerically stable.

!!! tip "Joseph form by default"

    The Kalman update uses the symmetric **Joseph form** to preserve
    positive-definiteness of $P_{t|t}$. You don't need to enable it.
