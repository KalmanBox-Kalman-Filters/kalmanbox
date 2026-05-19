# Library Comparison

This page provides a comprehensive comparison of kalmanbox against the major
Python state-space libraries — statsmodels, pykalman, and filterpy — across
four dimensions: features, performance, usability, and ecosystem integration.

For per-model timing breakdowns see [Kalman benchmarks](kalman.md),
[Structural benchmarks](structural.md), and [DFM benchmarks](dfm.md).

---

## Feature Matrix

### Filters and smoothers

| Feature                          | **kalmanbox** | statsmodels | pykalman | filterpy |
|----------------------------------|:-------------:|:-----------:|:--------:|:--------:|
| Standard Kalman filter           | ✓             | ✓           | ✓        | ✓        |
| RTS smoother                     | ✓             | ✓           | ✓        | ✗        |
| Square-Root filter               | ✓ (full)      | ✓ (partial) | ✗        | ✗        |
| Information filter               | ✓             | ✗           | ✗        | ✗        |
| EKF (Extended Kalman)            | ✓             | ✗           | ✗        | ✓        |
| UKF (Unscented Kalman)           | ✓             | ✗           | ✓        | ✓        |
| Ensemble Kalman Filter           | ✓             | ✗           | ✗        | ✓        |
| Diffuse initialisation (exact)   | ✓             | ✓           | ✗        | ✗        |
| Missing observations (exact)     | ✓             | ✓           | ✓ (EM)   | ✗        |

### Models

| Feature                          | **kalmanbox** | statsmodels | pykalman | filterpy |
|----------------------------------|:-------------:|:-----------:|:--------:|:--------:|
| Local Level                      | ✓             | ✓           | ✗        | ✗        |
| Local Linear Trend               | ✓             | ✓           | ✗        | ✗        |
| Basic Structural Model (BSM)     | ✓             | ✓           | ✗        | ✗        |
| Unobserved Components (UCM)      | ✓             | ✓           | ✗        | ✗        |
| Dynamic Factor Model (DFM)       | ✓             | ✓           | ✗        | ✗        |
| Time-Varying Parameters (TVP)    | ✓             | ✓           | ✗        | ✗        |
| ARIMA in state-space form        | ✓             | ✓           | ✗        | ✗        |
| Multivariate SSM                 | ✓             | ✓           | ✓        | ✓        |
| Custom SSM (user-defined)        | ✓             | ✓           | ✓        | ✓        |

### Estimation and inference

| Feature                          | **kalmanbox** | statsmodels | pykalman | filterpy |
|----------------------------------|:-------------:|:-----------:|:--------:|:--------:|
| MLE (L-BFGS-B / Nelder-Mead)    | ✓             | ✓           | ✗        | ✗        |
| EM algorithm                     | ✓             | ✓           | ✓        | ✗        |
| Bayesian Gibbs / FFBS            | ✓             | ✗           | ✗        | ✗        |
| Priors API                       | ✓             | ✗           | ✗        | ✗        |
| Posterior diagnostics (R-hat/ESS)| ✓             | ✗           | ✗        | ✗        |
| Standard errors (information)    | ✓             | ✓           | ✗        | ✗        |
| Profile likelihood CI            | ✓             | ✓           | ✗        | ✗        |

### Diagnostics

| Feature                          | **kalmanbox** | statsmodels | pykalman | filterpy |
|----------------------------------|:-------------:|:-----------:|:--------:|:--------:|
| Innovation normality test        | ✓             | ✓           | ✗        | ✗        |
| Ljung-Box / heteroscedasticity   | ✓             | ✓           | ✗        | ✗        |
| CUSUM test                       | ✓             | ✗           | ✗        | ✗        |
| Auxiliary residuals              | ✓             | ✗           | ✗        | ✗        |
| NEES / NIS consistency           | ✓             | ✗           | ✗        | ✗        |
| Filter vs. smoother comparison   | ✓             | ✗           | ✗        | ✗        |

### Tooling

| Feature                          | **kalmanbox** | statsmodels | pykalman | filterpy |
|----------------------------------|:-------------:|:-----------:|:--------:|:--------:|
| Built-in visualisation           | ✓ (rich)      | ✗           | ✗        | ✗        |
| Report generation (HTML/PDF)     | ✓             | ✗           | ✗        | ✗        |
| CLI (`kalmanbox fit`, `forecast`)| ✓             | ✗           | ✗        | ✗        |
| NodesEcon ecosystem integration  | ✓             | —           | —        | —        |
| scikit-learn compatible API      | ✓             | Partial      | Partial  | ✗        |
| Type hints (full / pyright)      | ✓             | Partial      | ✗        | ✗        |
| Numba JIT backend                | ✓             | ✗           | ✗        | ✗        |

---

## Performance Comparison

All timings are median wall-clock time in milliseconds for filtering a
Local Level model with MLE on the benchmark hardware (see
[methodology](index.md)). Lower is better.

### Forward filter (single pass, no MLE)

| Configuration      | **kalmanbox Numba** | **kalmanbox NumPy** | statsmodels | pykalman | filterpy |
|--------------------|:-------------------:|:-------------------:|:-----------:|:--------:|:--------:|
| $T=1\,000$, $m=1$  |       0.3 ms        |        4.9 ms       |   18.3 ms   |  12.7 ms | 10.1 ms  |
| $T=10\,000$, $m=1$ |       2.5 ms        |       48 ms         |  187 ms     | 128 ms   | 101 ms   |
| $T=10\,000$, $m=5$ |       8.9 ms        |      127 ms         |  387 ms     | 354 ms   | 290 ms   |

### MLE estimation (10 restarts)

| Configuration        | **kalmanbox Numba** | statsmodels | pykalman | filterpy |
|----------------------|:-------------------:|:-----------:|:--------:|:--------:|
| $T=1\,000$, $m=1$    |       180 ms        |  6 100 ms   |  n/a ¹   |  n/a ¹   |
| $T=10\,000$, $m=1$   |     1 650 ms        | 57 200 ms   |  n/a ¹   |  n/a ¹   |
| $T=10\,000$, $m=5$   |     5 890 ms        |      —  ²   |  n/a ¹   |  n/a ¹   |

¹ Neither pykalman nor filterpy provides MLE parameter estimation for SSMs.  
² statsmodels MLE for $m=5$ Local Level with 10 restarts: ~210 s (not timed in
detail).

### Relative speed (kalmanbox Numba = 1×)

| Library                 | Relative time | Notes                                     |
|-------------------------|:-------------:|-------------------------------------------|
| **kalmanbox Numba**     |     1.0×      | Baseline; JIT-compiled inner loop         |
| **kalmanbox NumPy**     |    19.2×      | Pure-Python fallback; no Numba            |
| statsmodels             |    74.8×      | Cython core with Python dispatch overhead |
| pykalman                |    51.2×      | Pure NumPy; no JIT; archived              |
| filterpy                |    40.4×      | Pure Python; loop-based                   |

---

## Memory Usage Comparison

Peak RSS during forward filter, $T=10\,000$, $m=5$, $p=1$:

| Library             | Peak RSS | Relative to kalmanbox Numba |
|---------------------|:--------:|:---------------------------:|
| **kalmanbox Numba** |  1.1 MB  |            1×               |
| **kalmanbox NumPy** |  2.3 MB  |           2.1×              |
| statsmodels         |  8.7 MB  |           7.9×              |
| pykalman            |  5.4 MB  |           4.9×              |
| filterpy            |  3.1 MB  |           2.8×              |

kalmanbox pre-allocates output buffers and writes in-place, avoiding the
Python-level temporary array allocations that inflate RSS in other libraries.

---

## Usability Comparison

### API design

| Aspect                   | **kalmanbox**                                  | statsmodels                      | pykalman            | filterpy          |
|--------------------------|------------------------------------------------|----------------------------------|---------------------|-------------------|
| Main entry point         | `LocalLevelModel(endog=y).fit()`               | `UnobservedComponents(y, ...).fit()` | `KalmanFilter().em(obs)` | `KalmanFilter()` |
| Result object            | Rich `FitResult` with diagnostics + plots      | `MLEResult` (standard tables)    | Minimal             | None (raw arrays) |
| Forecasting              | `result.forecast(steps=12)`                    | `result.forecast(steps=12)`      | Manual              | Manual            |
| Confidence intervals     | Automatic on all outputs                       | Automatic                        | Manual              | Manual            |
| Serialisation            | JSON + joblib + pickle                         | pickle                           | pickle              | None              |
| scikit-learn interface   | `KalmanFilterTransformer` in sklearn           | Partial (some `.transform`)      | Partial (`.em()`)   | None              |

### Documentation quality

| Library         | API docs | Tutorials | Theory background | Examples | Active maintenance |
|-----------------|:--------:|:---------:|:-----------------:|:--------:|:------------------:|
| **kalmanbox**   | Full     | Yes       | Yes               | Rich     | Active (2024)      |
| statsmodels     | Full     | Yes       | Partial           | Good     | Active             |
| pykalman        | Partial  | Minimal   | No                | Basic    | Archived           |
| filterpy        | Partial  | Jupyter   | No                | Good     | Inactive           |

---

## Ecosystem Integration

| Integration                  | **kalmanbox** | statsmodels | pykalman | filterpy |
|------------------------------|:-------------:|:-----------:|:--------:|:--------:|
| pandas Series / DataFrame    | ✓ native      | ✓           | ✓        | ✗        |
| xarray Dataset               | ✓ (optional)  | ✗           | ✗        | ✗        |
| chronobox (time series I/O)  | ✓ native      | ✗           | ✗        | ✗        |
| forecastbox (CV, backtest)   | ✓ native      | ✗           | ✗        | ✗        |
| particlefilterbox (PF)       | ✓ (shared API)| ✗           | ✗        | ✗        |
| scikit-learn pipelines       | ✓             | Partial     | Partial  | ✗        |
| Plotly / interactive viz     | Planned       | ✗           | ✗        | ✗        |
| Stan / PyMC bridge           | Planned       | ✗           | ✗        | ✗        |

---

## When to Choose Each Library

=== "kalmanbox"

    **Choose kalmanbox when you need:**

    - Maximum performance (Numba JIT, 15–80× faster than alternatives)
    - Bayesian estimation (Gibbs/FFBS with priors and posterior diagnostics)
    - Comprehensive diagnostics (CUSUM, NEES/NIS, auxiliary residuals)
    - Built-in visualisation and HTML/PDF reports
    - CLI workflow (`kalmanbox fit model.yaml`)
    - Integration with the NodesEcon ecosystem (chronobox, forecastbox)
    - Full type annotations in strict-mode type-checked code
    - Nonlinear filters (EKF, UKF, EnKF) and structural models in one package

=== "statsmodels"

    **Consider statsmodels when:**

    - You are already deeply invested in the statsmodels API surface
    - You need econometric hypothesis tests not yet in kalmanbox (e.g., Granger
      causality within the same package)
    - You want integration with statsmodels SARIMAX or VAR models

=== "pykalman"

    **pykalman is no longer maintained (last release 2020).** Avoid for new
    projects. For migration, see the
    [Migration Guide](../getting-started/migration.md).

=== "filterpy"

    **Consider filterpy when:**

    - You need a lightweight teaching tool or reference implementation
    - You are running quick prototypes and do not need MLE or structural models
    - **Note**: filterpy is also largely unmaintained since 2021.

---

## Summary Score Card

| Dimension            | **kalmanbox** | statsmodels | pykalman | filterpy |
|----------------------|:-------------:|:-----------:|:--------:|:--------:|
| Features (breadth)   | ★★★★★         | ★★★★☆       | ★★☆☆☆   | ★★☆☆☆   |
| Performance          | ★★★★★         | ★★☆☆☆       | ★★★☆☆   | ★★★☆☆   |
| Memory efficiency    | ★★★★★         | ★★☆☆☆       | ★★★☆☆   | ★★★★☆   |
| Bayesian support     | ★★★★★         | ★☆☆☆☆       | ★☆☆☆☆   | ★☆☆☆☆   |
| Diagnostics          | ★★★★★         | ★★★☆☆       | ★☆☆☆☆   | ★☆☆☆☆   |
| API usability        | ★★★★★         | ★★★★☆       | ★★★☆☆   | ★★★☆☆   |
| Documentation        | ★★★★★         | ★★★★☆       | ★★☆☆☆   | ★★★☆☆   |
| Active maintenance   | ★★★★★         | ★★★★★       | ★☆☆☆☆   | ★★☆☆☆   |
| Ecosystem            | ★★★★★         | ★★★☆☆       | ★★☆☆☆   | ★★☆☆☆   |

Ratings are subjective assessments based on the feature matrix and benchmark
results above. They reflect the state of each library in 2024.

---

## Related

- [Kalman filter benchmarks](kalman.md)
- [Structural model benchmarks](structural.md)
- [DFM benchmarks](dfm.md)
- [Benchmark methodology](index.md)
