# Tutorial — Airline passengers with BSM

The Box & Jenkins airline series (monthly international airline
passengers, 1949–1960) is the canonical **trended seasonal** time
series. We fit a Basic Structural Model (BSM) with monthly seasonality.

## 1. Load and inspect

```python
import numpy as np, matplotlib.pyplot as plt
from kalmanbox import BasicStructuralModel as BSM
from kalmanbox.datasets import load_dataset

air = load_dataset("airline_passengers")
y = np.log(air["passengers"])              # multiplicative -> additive
y.plot(figsize=(9, 3), title="log(airline passengers)")
```

## 2. Fit

```python
model = BSM(y, seasonal_periods=12)
results = model.fit()
print(results.summary())
```

The fitted variances are:

| Parameter         | Meaning                          |
|-------------------|----------------------------------|
| $\sigma_\eta^2$   | Level innovation                 |
| $\sigma_\zeta^2$  | Slope innovation                 |
| $\sigma_\omega^2$ | Seasonal innovation              |
| $\sigma_\varepsilon^2$ | Irregular variance          |

Expect $\sigma_\zeta^2$ near zero — the slope is essentially constant.

## 3. Component decomposition

```python
from kalmanbox.visualization import plot_components

plot_components(results)
```

Three panels: smoothed level, smoothed slope, smoothed seasonal. The
seasonal panel shows the textbook Christmas peak / Q1 trough.

## 4. Forecast

```python
from kalmanbox.visualization import plot_forecast

plot_forecast(results, steps=24, levels=(0.5, 0.95))
```

The forecast tracks the trend extrapolation plus the periodic seasonal
profile, with bands widening at the seasonal frequency.

## 5. Diagnostics

```python
from kalmanbox.diagnostics import residual_diagnostics
print(residual_diagnostics(results))
```

In log space the residuals should be close to white noise with no
visible seasonality remaining.

## Variants to try

- Switch `seasonal_periods=12` to a UCM with `seasonal=12,
  seasonal_form="trig"` for a smoother seasonal.
- Add a stochastic cycle to capture multi-year fluctuations beyond the
  annual cycle.
- Compare AIC/BIC against `LocalLinearTrend` (no seasonal) and
  `ARIMA_SSM(order=(0,1,1), seasonal_order=(0,1,1,12))`.

## Next

- [User guide: BSM](../user-guide/structural/bsm.md)
- [US macro DFM](us-macro-dfm.md)
