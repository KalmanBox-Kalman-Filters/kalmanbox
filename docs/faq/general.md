# FAQ — General

## What is kalmanbox?

kalmanbox is a **Python library for Kalman filtering and state-space modelling**,
designed as the foundational estimation layer of the NodesEcon ecosystem.

It provides:

- Forward Kalman filter and RTS smoother (standard, square-root, information, ensemble)
- Extended (EKF) and Unscented (UKF) Kalman filters for nonlinear systems
- Structural time-series models: Local Level, Local Linear Trend, Basic Structural
  Model (BSM), Unobserved Components Model (UCM)
- Dynamic Factor Models (DFM) and Time-Varying Parameter (TVP) models
- Maximum likelihood (MLE) and EM parameter estimation
- Bayesian Gibbs/FFBS posterior sampling
- Diagnostic tests (innovation tests, CUSUM, NEES/NIS)
- Built-in visualisation, reports, and a CLI

kalmanbox targets econometricians, forecasters, and engineers who need
**reliable, well-tested, well-documented** state-space tools in Python.

---

## How does kalmanbox differ from statsmodels?

statsmodels has a `tsa.statespace` sub-package that covers similar ground.
The main differences:

| Aspect                  | **kalmanbox**                          | statsmodels                             |
|-------------------------|----------------------------------------|-----------------------------------------|
| Focus                   | State-space models as first-class API  | General stats framework; SSM is one of many |
| Bayesian estimation     | Native Gibbs/FFBS sampler              | Not available                           |
| Nonlinear filters       | EKF, UKF, EnKF built-in               | Not available                           |
| Square-root filter      | Full Cholesky-factor update            | Partial (diffuse only)                  |
| Diagnostic suite        | CUSUM, NEES/NIS, auxiliary residuals   | Basic innovation tests                  |
| Visualisation           | Dedicated plot functions + themes      | Minimal (matplotlib wrappers)           |
| CLI                     | `kalmanbox fit`, `forecast`, `report`  | None                                    |
| NodesEcon integration   | Native (chronobox, forecastbox, …)     | Not applicable                          |
| Type hints              | Full, pyright-strict                   | Partial                                 |
| Speed (Numba JIT)       | 1× baseline                           | ~8× slower (Cython core, not JIT)       |

In practice, if you are already deep in the statsmodels ecosystem and only
need standard KF + MLE for a univariate structural model, statsmodels is a
reasonable choice. If you need Bayesian inference, nonlinear filters,
comprehensive diagnostics, or NodesEcon integration, use kalmanbox.

---

## How does kalmanbox differ from pykalman?

pykalman is a lightweight pure-NumPy library that implements the basic Kalman
filter and smoother with a scikit-learn-style API. It is easy to get started
with but lacks many features needed for serious time-series work:

| Aspect                        | **kalmanbox**      | pykalman         |
|-------------------------------|--------------------|------------------|
| Diffuse initialisation        | Yes                | No               |
| Missing observations          | Yes (native)       | Partial (EM only)|
| Square-root / Information KF  | Yes                | No               |
| EKF / UKF / EnKF              | Yes                | UKF only         |
| Structural models (BSM, UCM…) | Yes                | No               |
| DFM / TVP                     | Yes                | No               |
| Bayesian (Gibbs / FFBS)       | Yes                | No               |
| MLE via L-BFGS-B              | Yes                | No               |
| Type hints                    | Full               | No               |
| Maintained (2024)             | Active             | Archived         |

pykalman is no longer actively maintained. If you are using pykalman today,
consider migrating to kalmanbox — the [Migration Guide](../getting-started/migration.md)
covers the API differences.

---

## How does kalmanbox relate to chronobox, forecastbox, and particlefilterbox?

kalmanbox is the **foundational state estimation layer** of the NodesEcon ecosystem.
The other packages build on top of it:

```
                    ┌─────────────────────────────┐
                    │        forecastbox           │
                    │  Forecast API, CV, backtest  │
                    └──────────────┬──────────────┘
                                   │ wraps
               ┌───────────────────▼───────────────────┐
               │              kalmanbox                 │
               │  Kalman / SSM / EKF / UKF / Bayesian  │
               └──┬──────────────────────────────┬─────┘
                  │ shares model spec API         │ feeds data
     ┌────────────▼────────────┐    ┌─────────────▼────────┐
     │   particlefilterbox     │    │      chronobox        │
     │  Non-Gaussian / PF      │    │  Time series I/O      │
     └─────────────────────────┘    └──────────────────────┘
```

| Package             | Role                                                            |
|---------------------|-----------------------------------------------------------------|
| **kalmanbox**       | Core state-space estimation — all other packages depend on it   |
| chronobox           | Time-series I/O, alignment, resampling — provides clean input   |
| forecastbox         | Forecast API, cross-validation, backtesting — wraps outputs     |
| particlefilterbox   | Non-Gaussian / nonlinear particle filtering — sibling package   |

You can use kalmanbox standalone. The integrations are activated by installing
the respective packages.

---

## What models are available?

kalmanbox ships the following model families out of the box:

**Structural time-series models**

| Class                  | Description                                           |
|------------------------|-------------------------------------------------------|
| `LocalLevelModel`      | Random walk + noise (Harvey's LL)                     |
| `LocalLinearTrend`     | Level + slope, both stochastic                        |
| `BasicStructuralModel` | Trend + seasonal + cycle + irregular (BSM)            |
| `UnobservedComponents` | General UCM with plug-in components                   |

**Advanced models**

| Class                  | Description                                           |
|------------------------|-------------------------------------------------------|
| `DynamicFactorModel`   | DFM via EM or two-step estimation                     |
| `TVPModel`             | Time-Varying Parameters VAR-SSM                       |
| `ARIMAStateSpace`      | ARIMA/SARIMA in state-space form                      |

**Low-level filters**

| Class                  | Description                                           |
|------------------------|-------------------------------------------------------|
| `KalmanFilter`         | Standard forward filter                               |
| `RTSSmoother`          | Rauch–Tung–Striebel backward smoother                 |
| `SquareRootFilter`     | Cholesky-factor Kalman filter                         |
| `InformationFilter`    | Information-form filter (large $m$, sparse $P$)       |
| `EnsembleKalmanFilter` | Monte Carlo ensemble filter (large $m$)               |
| `ExtendedKalmanFilter` | EKF for nonlinear systems                             |
| `UnscentedKalmanFilter`| UKF / sigma-point filter for nonlinear systems        |

---

## Does kalmanbox support nonlinear models?

Yes. kalmanbox provides three nonlinear filter families:

- **EKF** (`ExtendedKalmanFilter`) — linearises the transition/measurement
  functions via first-order Taylor expansion. Fast, but accuracy degrades
  when nonlinearity is severe.
- **UKF** (`UnscentedKalmanFilter`) — propagates sigma-points through the
  exact nonlinear functions. Better accuracy than EKF at modest extra cost.
- **EnKF** (`EnsembleKalmanFilter`) — Monte Carlo ensemble approximation.
  Scales to very large state spaces ($m > 100$).

```python
from kalmanbox import UnscentedKalmanFilter
import numpy as np

def fx(x, dt):  # nonlinear transition: constant-velocity with drag
    return np.array([x[0] + x[1]*dt, x[1] * 0.98])

def hx(x):      # linear measurement of position
    return x[:1]

ukf = UnscentedKalmanFilter(
    dim_x=2, dim_z=1,
    fx=fx, hx=hx,
    dt=1.0,
    Q=np.eye(2) * 0.1,
    R=np.array([[1.0]]),
)
ukf.initialize(x0=np.zeros(2), P0=np.eye(2))
```

For fully non-Gaussian filtering (e.g., discrete state spaces, fat-tailed
likelihoods), use
[particlefilterbox](https://github.com/nodesecon/particlefilterbox),
which shares kalmanbox's model specification API.

---

## Does kalmanbox support Bayesian estimation?

Yes. kalmanbox includes a native Bayesian sampler based on the
**Gibbs / Forward-Filtering Backward-Sampling (FFBS)** algorithm for
conjugate-prior state-space models:

```python
from kalmanbox.bayesian import GibbsSampler
from kalmanbox.bayesian.priors import InverseGammaPrior, NormalPrior

model = LocalLevelModel(endog=y)
sampler = GibbsSampler(
    model=model,
    priors={
        "sigma2_eta": InverseGammaPrior(shape=3.0, scale=0.1),
        "sigma2_eps": InverseGammaPrior(shape=3.0, scale=1.0),
    },
    n_iter=4000,
    n_burnin=1000,
)
result = sampler.sample()
result.posterior_summary()          # mean, std, 5%–95% HDI
result.plot_trace(["sigma2_eta"])
```

Posterior diagnostics (R-hat, ESS, trace plots) are available via
`result.diagnostics()`. For non-conjugate priors or custom likelihoods,
kalmanbox can be coupled with PyMC or Stan via a thin bridge layer
(see [Bayesian guide](../user-guide/kalman/bayesian.md)).

---

## Can I use kalmanbox in production?

Yes. kalmanbox is designed for production use:

- **Semantic versioning**: breaking changes only in major releases; deprecation
  warnings in minor releases with at least one cycle grace period.
- **Type-annotated API**: full `py.typed` marker; works with mypy and pyright
  in strict mode, so type errors are caught at development time.
- **Tested**: over 600 unit and integration tests; CI runs on Python 3.11–3.13
  and Linux/macOS/Windows.
- **Reproducible**: pass `random_state` to stochastic methods for exact
  reproducibility across runs.
- **Serialisation**: models and results serialise to JSON/pickle for persistence
  (see [Saving models](#how-do-i-save-and-load-an-estimated-model) in the
  Advanced FAQ).
- **No heavy optional dependencies**: NumPy + SciPy are the only required
  runtime deps; Numba and Pandas are optional.

For high-frequency or embedded deployments, see the
[Performance FAQ](advanced.md#how-do-i-handle-very-long-series) for Numba
and chunked-filtering strategies.

---

## How do I cite kalmanbox?

If you use kalmanbox in academic work, please cite:

```bibtex
@software{kalmanbox,
  author  = {{NodesEcon Contributors}},
  title   = {kalmanbox: Kalman Filter and State-Space Models for Python},
  url     = {https://github.com/nodesecon/kalmanbox},
  version = {0.4.0},
  year    = {2024},
}
```

A CITATION.cff file is also included in the repository root for GitHub's
"Cite this repository" feature.

---

## What Python versions are supported?

kalmanbox requires **Python 3.11 or later**. The test suite runs against
3.11, 3.12, and 3.13 on every CI build. Python 3.10 and below are not
supported because kalmanbox uses `match`/`case` statements and several
`typing` features introduced in 3.11.

---

## Can I use kalmanbox for real-time streaming applications?

Yes. The Kalman filter is inherently **online** — call `kf.update(y_new)` to
incorporate a single new observation without reprocessing the full history.

```python
from kalmanbox import KalmanFilter

kf = KalmanFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)
kf.initialize_diffuse()

for y_t in stream:          # live data feed
    kf.update(y_t)
    state_now = kf.a        # current filtered state mean
    cov_now   = kf.P        # current filtered state covariance
```

MLE parameter estimates must be computed on a training window first;
re-estimating on every tick is not practical. For fully adaptive streaming,
consider recursive EM (on the roadmap).

---

## What is the license?

kalmanbox is released under the **MIT License**. You can use it in commercial
products, modify it, and redistribute it, provided the original copyright
notice is retained. The full text is in
[`LICENSE`](https://github.com/nodesecon/kalmanbox/blob/main/LICENSE).

---

## Where can I get help beyond the docs?

- [GitHub Discussions](https://github.com/nodesecon/kalmanbox/discussions) —
  community Q&A (search before posting).
- [NodesEcon Slack](https://nodesecon.slack.com) — `#kalmanbox` channel for
  real-time discussion.
- Stack Overflow — tag `[kalmanbox]` for general questions.
- [Issue tracker](https://github.com/nodesecon/kalmanbox/issues) — bug reports
  with a minimal reproducible example.
