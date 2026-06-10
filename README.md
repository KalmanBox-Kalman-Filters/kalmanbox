# kalmanbox

[![CI](https://github.com/KalmanBox-Kalman-Filters/kalmanbox/actions/workflows/ci.yml/badge.svg)](https://github.com/KalmanBox-Kalman-Filters/kalmanbox/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KalmanBox-Kalman-Filters/kalmanbox/branch/main/graph/badge.svg)](https://codecov.io/gh/KalmanBox-Kalman-Filters/kalmanbox)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/kalmanbox.svg)](https://badge.fury.io/py/kalmanbox)
[![Python versions](https://img.shields.io/pypi/pyversions/kalmanbox)](https://pypi.org/project/kalmanbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Development Status](https://img.shields.io/badge/development%20status-alpha-orange)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/kalmanbox?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/kalmanbox)
[![Documentation](https://readthedocs.org/projects/kalmanbox/badge/?version=latest)](https://kalmanbox.readthedocs.io/)

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
