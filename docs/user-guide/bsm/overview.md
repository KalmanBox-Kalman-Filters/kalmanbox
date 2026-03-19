# Basic Structural Model (BSM)

The BSM decomposes a time series into trend, seasonal, and irregular components.

## Model specification

$$y_t = \mu_t + \gamma_t + \varepsilon_t$$

**Trend** (local linear trend):

$$\mu_{t+1} = \mu_t + \nu_t + \xi_t, \quad \xi_t \sim N(0, \sigma^2_\xi)$$

$$\nu_{t+1} = \nu_t + \zeta_t, \quad \zeta_t \sim N(0, \sigma^2_\zeta)$$

**Seasonal** (trigonometric or dummy):

$$\gamma_{t+1} = -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t, \quad \omega_t \sim N(0, \sigma^2_\omega)$$

where $s$ is the seasonal period.

## Usage

```python
from kalmanbox import BSM
from kalmanbox.datasets import load_dataset

airline = load_dataset('airline')
model = BSM(airline['passengers'], seasonal_period=12)
results = model.fit()
print(results.summary())
```

## Example: Airline passengers

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.datasets import load_dataset

airline = load_dataset('airline')

# Log transform for multiplicative seasonality
y = np.log(airline['passengers'].to_numpy(dtype=np.float64))

model = BSM(y, seasonal_period=12)
results = model.fit()

# Components
smoothed = results.smoothed_state
print(f"Number of states: {smoothed.shape[1]}")

# Forecast 12 months ahead
fc = results.forecast(steps=12)
```

## Parameters

| Parameter | Description |
|:----------|:------------|
| `seasonal_period` | Number of seasons (e.g., 12 for monthly) |

## When to use

- Monthly or quarterly data with clear seasonality
- Need to decompose trend and seasonal components
- Forecasting seasonal time series
