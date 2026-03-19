# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-17

### Added

- **Core**: StateSpaceRepresentation with system matrices (T, Z, R, H, Q)
- **Kalman Filter**: Numerically stable implementation with Joseph-form updates
- **RTS Smoother**: Rauch-Tung-Striebel backward smoother
- **Models**:
  - LocalLevel (random walk + noise)
  - LocalLinearTrend (trend + level)
  - BSM (Basic Structural Model with seasonality)
  - UCM (Unobserved Components Model)
  - DynamicFactor (dynamic factor model)
  - TVPRegression (time-varying parameter regression)
  - ARIMASSM (ARIMA in state-space form)
- **Estimation**: Maximum Likelihood via scipy.optimize
- **Forecasting**: Point forecasts with confidence intervals
- **Diagnostics**: Ljung-Box test, Jarque-Bera test, residual analysis
- **Missing data**: Automatic handling of NaN observations
- **Reports**: HTML and LaTeX report generation
- **Datasets**: 19 built-in datasets (classic, macro, Brazilian)
- **CLI**: Command-line interface (estimate, info, forecast)
- **Experiment**: KalmanExperiment for model comparison and validation
- **Numba**: Optional JIT acceleration for core loops (5x+ speedup)
- **Documentation**: Full MkDocs site with tutorials, API reference, theory

### Dependencies

- Python >= 3.11
- NumPy >= 1.24
- SciPy >= 1.10
- pandas >= 2.0
