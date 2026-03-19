# KalmanBox

**State-space models and Kalman filtering for time series analysis.**

KalmanBox is a Python library for building, estimating, and forecasting with
state-space models using the Kalman filter. It provides pre-built models for
common use cases and a flexible framework for custom models.

## Features

- **Pre-built models**: Local Level, Local Linear Trend, BSM, UCM, Dynamic Factor,
  TVP Regression, ARIMA-SSM
- **Kalman filter & smoother**: Numerically stable implementations with
  Joseph-form updates
- **Maximum likelihood estimation**: Automatic parameter estimation via MLE
- **Forecasting**: Point forecasts with confidence intervals
- **Missing data**: Automatic handling of missing observations
- **Diagnostics**: Residual tests, information criteria (AIC/BIC)
- **19 built-in datasets**: Classic time series and Brazilian economic data
- **CLI**: Command-line interface for quick model fitting
- **Experiment pattern**: Compare multiple models automatically

## Quick Example

```python
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset

# Load data
nile = load_dataset('nile')

# Fit model
model = LocalLevel(nile['volume'])
results = model.fit()

# View results
print(results.summary())

# Forecast
forecast = results.forecast(steps=10)
```

## Installation

```bash
pip install kalmanbox
```

See the [Installation Guide](getting-started/installation.md) for more options.

## License

MIT License. See [LICENSE](https://github.com/nodesecon/kalmanbox/blob/main/LICENSE).
