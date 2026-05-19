# Structural Models Examples

This directory contains examples of **Structural Time Series Models** (also known as
Unobserved Components Models) implemented with `kalmanbox`. These models decompose a
time series into interpretable components (trend, cycle, seasonal, regression effects)
estimated via the Kalman filter.

## Notebooks

| # | Notebook | Description | Dataset |
|---|----------|-------------|---------|
| 1 | **UCM - Unobserved Components Model** | Trend-cycle decomposition of Brazilian GDP using a local linear trend + stochastic cycle (~8 years) | `brazil_gdp.csv` |
| 2 | **Stochastic Cycle Model** | Extracting inflation cycles from Brazilian IPCA with time-varying amplitude and frequency | `brazil_ipca.csv` |
| 3 | **State-Space Regression** | Regression with time-varying coefficients and structural breaks for UK gas consumption (Harvey, 1989) | `uk_gas.csv` |

## Datasets

- **`data/brazil_gdp.csv`** - Brazilian quarterly GDP index (2000Q1-2023Q4, 96 obs).
  Synthetic data calibrated to real GDP dynamics: growth trend, ~8-year business cycle,
  and shocks (2008 crisis, 2015-16 recession, COVID-2020).

- **`data/brazil_ipca.csv`** - Brazilian monthly CPI inflation (IPCA, 12-month accumulated,
  2000-01 to 2023-12, 288 obs). Synthetic data with mean ~6%, long and short cycles,
  and inflationary spikes (2002-03, 2015-16, 2021-22).

- **`data/uk_gas.csv`** - UK quarterly gas consumption (1960Q1-1986Q4, 108 obs).
  Classic series with upward trend, strong seasonality (winter peak), and structural
  change after the 1973 oil crisis.

## Pre-requisites

```bash
pip install kalmanbox numpy pandas matplotlib
```

For cross-validation against reference implementations:

```bash
pip install statsmodels filterpy
```

## Directory Structure

```
02_structural_models/
├── data/                  # CSV datasets
├── solutions/             # Completed notebook solutions
├── validation/
│   ├── R/                 # R scripts (KFAS, dlm) for cross-validation
│   └── stata/             # Stata scripts (ucm) for cross-validation
└── README.md
```

## References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*.
  Cambridge University Press.
- Commandeur, J. J. F., & Koopman, S. J. (2007). *An Introduction to State Space Time
  Series Analysis*. Oxford University Press.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
  (2nd ed.). Oxford University Press.
- Kim, C.-J., & Nelson, C. R. (1999). *State-Space Models with Regime Switching*.
  MIT Press.
