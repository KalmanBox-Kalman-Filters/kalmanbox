# Changelog

All notable changes to **kalmanbox** are documented here.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Categories used in each release:

| Category | Meaning |
|----------|---------|
| **Added** | New features, models, classes, or functions |
| **Changed** | Changes to existing functionality |
| **Deprecated** | Features that will be removed in a future release |
| **Removed** | Features removed in this release |
| **Fixed** | Bug fixes |
| **Security** | Security patches |

---

## [Unreleased]

### Added

- `KalmanFilter.stream()`: online single-step update without re-running the
  full filter pass. Useful for real-time data ingestion.
- `EnsembleKalmanFilter.localize()`: distance-based covariance localization
  using a Gaspari–Cohn taper function.

### Changed

- `DFM.fit(method="em")` now defaults to `max_iter=1000` (was 500).
  Convergence tolerance unchanged at `1e-8`.

---

## [0.5.0] — 2026-05-01

### Added

- **Experiment Framework** (`kalmanbox.experiment`): six-stage pipeline for
  reproducible model comparison (data loading, model specification, filter
  execution, metric computation, cross-validation, reporting).
- `Experiment.from_yaml()`: load experiment configuration from a YAML file.
- `CVStrategy.expanding_window()` and `CVStrategy.rolling_window()`:
  time-series cross-validation strategies.
- **Ensemble Kalman Filter** (`EnKF`): Monte Carlo-based filter for
  high-dimensional nonlinear systems. Supports inflation and localization.
- **Factor Plots** (`kalmanbox.visualization.factor_plots`): six dedicated
  functions for visualizing Dynamic Factor Model output (factor loadings,
  idiosyncratic variance, factor time series, forecast fans).
- **TVP Coefficient Plots** (`kalmanbox.visualization.tvp_plots`): five
  functions for time-varying parameter visualization with credible bands.
- **Theme system** (`kalmanbox.visualization.themes`): four built-in
  publication-quality themes (`kalman_default`, `publication_grayscale`,
  `presentation_dark`, `colorblind_safe`) plus `register_theme()` API.
- `Reports.to_latex()`: export model comparison tables as LaTeX with
  `booktabs` formatting.
- CLI command `kalmanbox experiment run`: execute an experiment from YAML
  and save results.
- CLI command `kalmanbox report generate`: produce HTML/PDF reports from
  saved experiment results.
- **Datasets module** (`kalmanbox.datasets`): six curated datasets
  (`nile`, `airline`, `us_gdp`, `sp500_returns`, `temperature`, `macro_panel`)
  plus three simulators (`simulate_local_level`, `simulate_dfm`,
  `simulate_nonlinear`).

### Changed

- `KalmanFilter.filter()` now returns a `FilterResult` dataclass instead of
  a plain tuple. The old tuple unpacking still works via `__iter__` but is
  deprecated.
- `RTSSmoother.smooth()` signature changed: `result` parameter now accepts
  either a `FilterResult` or the legacy tuple. The tuple form will be removed
  in v0.6.0.
- `DFM.fit()` uses the Louis (1982) correction for EM standard errors by
  default (`se_method="louis"`).
- Minimum Python version raised from 3.10 to **3.11**.
- Minimum NumPy version raised from 1.24 to **1.26**.

### Fixed

- Square-Root filter no longer loses Cholesky symmetry after 10 000+ steps
  on ill-conditioned observation covariances (#412).
- `LocalLinearTrend.forecast()` incorrect interval width when
  `alpha != 0.05` (#398).
- `UCM` cycle component amplitude constraint not enforced during MLE (#387).
- `FFBS` backward sampler drew from the wrong conditional distribution when
  `R` was not square (#371).
- Missing-data imputation in `DFM` now handles runs of consecutive `NaN`
  values correctly (#356).

### Security

- Updated `aiohttp` (transitive) to address CVE-2024-23334.

---

## [0.4.0] — 2025-11-15

### Added

- **Bayesian estimation module** (`kalmanbox.bayesian`): Gibbs sampler,
  Forward-Filter Backward-Sample (FFBS), conjugate priors, and posterior
  diagnostics (R-hat, ESS, trace plots).
- **Information Filter** (`InformationFilter`): numerically preferable
  formulation that avoids explicit matrix inversion; supports diffuse
  initialization via the augmented information representation.
- **Filter Comparison** (`kalmanbox.diagnostics.filter_comparison`): unified
  API for comparing KF, EKF, UKF, Square-Root, and EnKF across RMSE,
  log-likelihood, runtime, and condition-number metrics.
- **Consistency Tests** (`kalmanbox.diagnostics.consistency`): NEES and NIS
  chi-squared tests with p-value reporting.
- **State Smoothness** (`kalmanbox.diagnostics.state_smoothness`): second
  differences, first-differences distribution, smoothness index, and
  Ljung–Box test on smoothed states.
- **Auxiliary Residuals** (`kalmanbox.diagnostics.auxiliary_residuals`):
  disturbance smoother, standardized residuals, outlier detection at
  user-specified significance levels.
- **Theory section** in documentation: five deep-dive pages covering
  State-Space theory, Kalman derivation, smoothing theory, MLE theory,
  and structural model theory (800–1 000 lines each with full LaTeX).
- **Advanced Theory** section: DFM theory, nonlinear filter theory, Bayesian
  SSM theory, diffuse initialization theory, and annotated references.

### Changed

- `KalmanFilter` constructor now accepts `dtype` parameter (default `float64`).
- `MLE.fit()` `method` parameter now accepts `"bfgs"`, `"lbfgsb"`,
  `"nelder-mead"`, and `"powell"` (previously only `"bfgs"`).

### Deprecated

- `KalmanFilter.filter_tuple()`: use `KalmanFilter.filter()` which returns
  a `FilterResult`. Will be removed in v0.6.0.

### Fixed

- Diffuse log-likelihood sign error when `H` has zero diagonal entries (#289).
- `RTSSmoother` gain computation unstable when `P_predicted` is near-singular;
  now uses `scipy.linalg.lstsq` fallback (#277).

---

## [0.3.0] — 2025-06-20

### Added

- **Alternative Filters**: Extended Kalman Filter (`EKF`), Unscented Kalman
  Filter (`UKF`), and Square-Root Kalman Filter.
- `UKF` supports the Merwe (2004) scaled sigma-point algorithm with
  configurable `alpha`, `beta`, `kappa` parameters.
- **Dynamic Factor Model** (`DFM`): EM and MLE estimation, Bai–Ng IC for
  factor-count selection, identification via diagonal loading.
- **Time-Varying Parameters** (`TVP`): random-walk and mean-reverting
  coefficient evolution; OLS warm-start.
- **EM Algorithm** (`kalmanbox.estimation.em`): Shumway–Stoffer two-step EM
  with Dempster–Laird–Rubin convergence monitoring.
- **Multivariate models** (`MultivariateDFM`): handles `NaN` patterns
  specific to unbalanced panels.
- `CUSUMTest` and `PredictionErrorDecomposition` diagnostics.
- `kalmanbox.visualization`: `plot_states`, `plot_components`,
  `plot_innovations`, `plot_filter_comparison` (four-panel diagnostic figure).

### Changed

- `BSM` seasonal component specification changed from `n_seasons` integer
  to explicit `cycle_length` parameter.
- `LocalLevel` and `LocalLinearTrend` now accept `sigma2_obs=None` to
  estimate from data.

### Fixed

- UCM cycle component variance constraint violated during grid search (#198).
- `plot_innovations` histogram bin count incorrect for short series (#187).

---

## [0.2.0] — 2025-01-10

### Added

- **Unobserved Components Model** (`UCM`): trend, cycle, seasonal, irregular
  decomposition with MLE and diagnostics.
- **Basic Structural Model** (`BSM`): level + slope + seasonal with Harvey
  (1989) parameterization.
- **ARIMA in State-Space form**: any ARIMA(p, d, q)(P, D, Q)_s representable
  as a state-space model.
- `MLE.profile_likelihood()` for grid search over variance ratios.
- Innovation diagnostic tests: Ljung–Box, Jarque–Bera, and Kolmogorov–
  Smirnov.
- `kalmanbox.datasets.nile()` and `kalmanbox.datasets.airline()` built-in
  datasets.
- CLI `kalmanbox fit`: fit a named model to a CSV file.
- CLI `kalmanbox diagnose`: run the full diagnostic suite.

### Changed

- `KalmanFilter` constructor parameters reorganized; `H` and `Q` are now
  keyword-only.

### Fixed

- Log-likelihood computation incorrect for multivariate observations when
  `n > 1` (#112).

---

## [0.1.0] — 2024-09-01

### Added

- `KalmanFilter`: predict–update Kalman recursion with support for
  time-varying system matrices.
- `RTSSmoother`: Rauch–Tung–Striebel smoother.
- `LocalLevel`: local level model with MLE via `scipy.optimize`.
- `LocalLinearTrend`: local linear trend model.
- Missing data handling via `NaN` masking.
- Diffuse initialization (exact and approximate).
- `kalmanbox.visualization.plot_filter()`: basic filtered-state plot.
- MkDocs Material documentation skeleton.
- GitHub Actions CI: test matrix (Python 3.10, 3.11), ruff, pyright, bandit.

---

## Template for new entries

When adding an entry to `[Unreleased]`, use this template:

```markdown
## [Unreleased]

### Added
- `ClassName.method_name()`: short description of what it does (#issue_number).

### Changed
- `ClassName.existing_method()`: describe the breaking or non-breaking change.

### Deprecated
- `old_function()`: use `new_function()` instead. Will be removed in vX.Y.0.

### Fixed
- `ClassName.buggy_method()`: describe what was wrong and what was fixed (#issue).

### Security
- Bumped `dependency>=X.Y.Z` to address CVE-YYYY-NNNNN.
```

!!! tip "Keep entries linkable"
    Every entry should reference a GitHub issue or PR number (`(#123)`)
    so users can find the full discussion and diff.

[Unreleased]: https://github.com/nodesecon/kalmanbox/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/nodesecon/kalmanbox/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/nodesecon/kalmanbox/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/nodesecon/kalmanbox/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nodesecon/kalmanbox/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nodesecon/kalmanbox/releases/tag/v0.1.0
