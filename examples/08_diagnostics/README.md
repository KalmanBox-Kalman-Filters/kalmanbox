# Diagnostics for State-Space Models — Examples

Diagnostic tools for validating state-space models fit with **kalmanbox**: standardized
residuals, CUSUM tests, auxiliary residuals, missing-data handling, and bootstrap
confidence intervals.

## Why diagnostics?

Once a state-space model is estimated by MLE, the innovation residuals (one-step-ahead
prediction errors from the Kalman filter) carry the diagnostic content of the fit:

- `v_t = y_t - E[y_t | y_{1:t-1}]`            (innovation)
- `F_t = Var(v_t | y_{1:t-1})`                (innovation variance)
- `e_t = v_t / sqrt(F_t)`                     (standardized residual)

If the model is correctly specified and Gaussian, the standardized residuals satisfy:

| Property                        | Test                                                        |
|---------------------------------|-------------------------------------------------------------|
| `E[e_t] = 0`                    | t-test on the sample mean                                   |
| `Var(e_t) = 1`                  | sample variance close to 1                                  |
| `Cov(e_t, e_s) = 0` for `t ≠ s` | Ljung–Box / Box–Pierce on residual autocorrelations         |
| `e_t ~ N(0, 1)`                 | Jarque–Bera, Shapiro–Wilk, QQ-plot                          |
| Stable parameters               | CUSUM and CUSUMSQ tests on cumulative residuals             |

### Auxiliary residuals (de Jong & Penzer, 1998)

Standardized residuals diagnose the *observation* equation. To detect outliers and
structural breaks affecting individual *unobserved components* (level, slope, seasonal),
de Jong & Penzer (1998) introduced **auxiliary residuals** built from the smoothed
state disturbances `eta_t`. Large standardized auxiliary residuals at time `t` flag:

- a likely **additive outlier** in the observation equation (`epsilon_t`),
- a **level shift** (jump in the level disturbance),
- a **slope change** (jump in the slope disturbance), or
- a **seasonal break**.

### Missing data

The Kalman filter handles missing observations naturally: at any `t` with `y_t = NaN`
the update step is skipped (filtered state equals predicted state, `F_t` is undefined),
while the prediction step continues. The smoother then interpolates the missing values
using all observations. The likelihood is computed only over observed periods.

### Bootstrap

Asymptotic standard errors from the MLE Hessian can be unreliable in small samples
or near the boundary of the parameter space (variances close to zero). Parametric
and non-parametric bootstrap (residual / wild bootstrap on innovations) provide
confidence intervals for parameters, smoothed states, and forecasts without relying
on asymptotic normality.

## Datasets

| File                  | Description                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| `nile.csv`            | Annual flow of the Nile at Aswan, 1871–1970 (100 obs). Source: Cobb (1978), Durbin & Koopman (2012).              |
| `airline.csv`         | Monthly international airline passengers, 1949–1960 (144 obs). Source: Box & Jenkins (1970).                      |
| `airline_missing.csv` | `airline.csv` with 22 of 144 observations (~15%) set to `NaN` in random blocks of 1–3 months (seed `20260416`).   |
| `nile_outliers.csv`   | `nile.csv` with 4 additive outliers injected (volume × 1.5) in years 1888, 1913, 1932, 1955; flagged in `is_outlier`. |

The synthetic datasets are produced by `data/_generate_datasets.py` (deterministic
under the recorded seed) so the patterns can be regenerated bit-for-bit.

### Outlier years in `nile_outliers.csv`

| Year | Original | Modified | Ratio |
|------|----------|----------|-------|
| 1888 | 799      | 1198     | 1.50  |
| 1913 | 456      | 684      | 1.50  |
| 1932 | 865      | 1298     | 1.50  |
| 1955 | 918      | 1377     | 1.50  |

These are the ground truth that auxiliary-residual diagnostics should recover.

## Notebooks

The `solutions/` directory contains two notebooks (built in subsequent subphases):

1. **Standardized residuals, CUSUM, and auxiliary residuals** — diagnostics on Nile
   (with and without injected outliers) and on a structural model fit to airline data.
2. **Missing data and bootstrap** — Kalman smoothing as interpolation on
   `airline_missing.csv`, and parametric bootstrap for confidence intervals on
   variance parameters.

## Cross-validation

The `validation/` directory contains scripts to reproduce the diagnostics in:

- **R / KFAS** — `KFS()` returns standardized and auxiliary residuals directly.
- **Stata / sspace** — `predict, residuals` and `estat` commands.

## Prerequisites

```bash
pip install kalmanbox jupyter matplotlib statsmodels scipy
```

## References

- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Cobb, G. W. (1978). The problem of the Nile: conditional solution to a changepoint problem. *Biometrika*, 65(2), 243–251.
- de Jong, P. & Penzer, J. (1998). Diagnosing shocks in time series. *Journal of the American Statistical Association*, 93(442), 796–806.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.), Chapter 7. Oxford University Press.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*, Chapter 5. Cambridge University Press.
