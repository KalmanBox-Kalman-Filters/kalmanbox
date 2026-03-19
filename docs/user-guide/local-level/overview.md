# Local Level Model

The Local Level model (also called the random walk plus noise model) is the
simplest structural time series model.

## Model specification

$$y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2_\varepsilon)$$

$$\mu_{t+1} = \mu_t + \eta_t, \quad \eta_t \sim N(0, \sigma^2_\eta)$$

**Parameters:** $\sigma^2_\varepsilon$ (observation noise), $\sigma^2_\eta$ (level noise)

**States:** $\alpha_t = \mu_t$ (one state)

## Usage

```python
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset

nile = load_dataset('nile')
model = LocalLevel(nile['volume'])
results = model.fit()
print(results.summary())
```

## Example: Nile river flow

The classic Nile dataset (1871-1970) shows a level shift around 1898
when the Aswan dam was built:

```python
import numpy as np
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset

nile = load_dataset('nile')
model = LocalLevel(nile['volume'])
results = model.fit()

# Estimated parameters
print(f"sigma2_obs:   {results.params[0]:.1f}")   # ~15099
print(f"sigma2_level: {results.params[1]:.1f}")   # ~1469
print(f"Log-likelihood: {results.loglike:.2f}")    # ~-632.54

# Smoothed state (estimated level)
smoothed = results.smoothed_state
print(f"Level in 1871: {smoothed[0]:.1f}")
print(f"Level in 1970: {smoothed[-1]:.1f}")

# Forecast
fc = results.forecast(steps=10)
```

## Signal-to-noise ratio

The ratio $q = \sigma^2_\eta / \sigma^2_\varepsilon$ controls how quickly
the level adapts:

- $q \to 0$: level is nearly constant (all variation is noise)
- $q \to \infty$: level follows the data closely

For the Nile data, $q \approx 1469/15099 \approx 0.097$.

## Diagnostics

```python
# Residual analysis
residuals = results.residuals

# Ljung-Box test
from kalmanbox.diagnostics import ljung_box_test
lb = ljung_box_test(residuals)
print(f"Ljung-Box p-value: {lb.pvalue:.4f}")
```

## When to use

- Time series with a slowly changing level
- No trend or seasonality
- As a baseline model before trying more complex specifications
