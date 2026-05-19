# Local Level and Trend Models - Examples

Examples demonstrating local level and structural time series models using **kalmanbox**.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_local_level.ipynb` | Local level (random walk + noise) model applied to the Nile river flow dataset. Covers filtering, smoothing, parameter estimation via MLE, and diagnostics. |
| 2 | `02_local_linear_trend.ipynb` | Local linear trend model with stochastic level and slope. Applied to airline passengers (log-transformed). |
| 3 | `03_bsm.ipynb` | Basic Structural Model with level, trend, and seasonal components. Applied to UK drivers deaths with intervention analysis (seatbelt legislation, Feb 1983). |

## Datasets

- **nile.csv** — Annual flow of the Nile at Aswan, 1871-1970 (100 obs). Source: Durbin & Koopman (2012), Cobb (1978).
- **airline.csv** — Monthly international airline passengers, 1949-1960 (144 obs). Source: Box & Jenkins (1970).
- **uk_drivers.csv** — Monthly UK road casualties, 1969-1984 (192 obs). Source: Harvey & Durbin (1986).

## Prerequisites

```bash
pip install kalmanbox jupyter matplotlib statsmodels
```

## Running

```bash
cd examples/01_local_level_trend
jupyter notebook
```

## Cross-validation

The `validation/` directory contains scripts for comparing kalmanbox results against:
- **R**: KFAS (`SSModel`, `KFS`), dlm
- **Stata**: `sspace`

## References

- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Harvey, A.C. & Durbin, J. (1986). The effects of seat belt legislation on British road casualties. *Journal of the Royal Statistical Society A*, 149(3), 187-227.
- Cobb, G.W. (1978). The problem of the Nile: conditional solution to a changepoint problem. *Biometrika*, 65(2), 243-251.
