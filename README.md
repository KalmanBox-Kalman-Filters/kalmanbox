# kalmanbox

State-space models and Kalman filtering for time series analysis.

## Features

- **Models**: `LocalLevel`, `LocalLinearTrend`, `BasicStructuralModel` (BSM),
  `UnobservedComponents` (UCM), `DynamicFactorModel`, `TimeVaryingParameters` (TVP),
  `ARIMA_SSM`, and `CustomStateSpace` for arbitrary user-defined systems.
- **Filters**: standard Kalman, Extended (EKF), Unscented (UKF), Ensemble (EnKF),
  Square-Root, and Information filters.
- **Smoothers**: Rauch-Tung-Striebel (RTS), Fixed-Interval, Fixed-Lag, and
  Disturbance smoothers.
- Maximum likelihood estimation, diagnostics, visualization, and built-in datasets.

## Installation

```bash
pip install kalmanbox
```

### From source (development)

```bash
git clone https://github.com/nodesecon/kalmanbox
cd kalmanbox
pip install -e ".[dev]"
```

## Quick Start

```python
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset

nile = load_dataset('nile')
model = LocalLevel(nile['volume'])
results = model.fit()
print(results.summary())
```

## Documentation

Full documentation is available at https://kalmanbox.nodesecon.com.

## License

MIT
