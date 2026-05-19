---
title: kalmanbox
description: >-
  Foundational Kalman Filter and State-Space Models library —
  the base of the NodesEcon ecosystem (chronobox, forecastbox, particlefilterbox).
hide:
  - navigation
---

<div class="kb-hero" markdown>
<span class="kb-tagline">NodesEcon · Foundation Library</span>

# kalmanbox

**Kalman filtering and state-space modelling, done right — and built to be built upon.**

`kalmanbox` is the foundational library of the **NodesEcon** ecosystem.
It delivers a numerically robust Kalman recursion engine, a complete library of
structural time-series models, Dynamic Factor Models, Time-Varying Parameters,
Bayesian posterior samplers, and the shared `StateSpaceRepresentation` that
[`chronobox`](#ecosystem), [`forecastbox`](#ecosystem), and
[`particlefilterbox`](#ecosystem) all build on top of.

[![PyPI version](https://img.shields.io/pypi/v/kalmanbox)](https://pypi.org/project/kalmanbox/)
[![Python versions](https://img.shields.io/pypi/pyversions/kalmanbox)](https://pypi.org/project/kalmanbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/nodesecon/kalmanbox/actions/workflows/ci.yml/badge.svg)](https://github.com/nodesecon/kalmanbox/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/nodesecon/kalmanbox)](https://codecov.io/gh/nodesecon/kalmanbox)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://kalmanbox.nodesecon.com)
</div>

## What is kalmanbox?

`kalmanbox` is a complete Python toolkit for **Kalman filtering and state-space
modelling**. Its design follows three principles:

- **Unified API** — every model is a `StateSpaceRepresentation`; every filter
  and smoother works with any representation, so algorithms are fully
  interchangeable.
- **Numerical robustness** — Square-Root and Information filter variants,
  diffuse initialisation for non-stationary states, and Joseph-form covariance
  updates guard against floating-point degradation.
- **Full inference stack** — MLE, EM, and Bayesian (Gibbs / FFBS) estimation
  are first-class citizens, not afterthoughts.

The result is a library you can use stand-alone for rigorous state-space
analysis, *and* the stable low-level engine that the rest of the NodesEcon
ecosystem depends on.

---

## Capabilities

<div class="grid cards" markdown>

-   :material-filter:{ .lg .middle } **Kalman Filter & RTS Smoother**

    ---

    Exact linear Gaussian filter with diffuse initialisation for non-stationary
    state components. The Rauch-Tung-Striebel backward smoother delivers
    full-sample state estimates and their covariances in a single backward pass.

    Supports missing observations, time-varying system matrices, and mixed
    stationary / diffuse initial conditions (Koopman 1997 approach).

    [:octicons-arrow-right-24: Kalman Filter](user-guide/kalman/kalman-filter.md) ·
    [:octicons-arrow-right-24: RTS Smoother](user-guide/kalman/rts-smoother.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Structural Time-Series Models**

    ---

    Ready-to-use decomposition models that express trend, seasonality, cycle,
    and irregular components as latent states:

    - **Local Level** — random-walk signal + white noise
    - **Local Linear Trend** — integrated random-walk trend
    - **Basic Structural Model (BSM)** — trend + trigonometric seasonal + irregular
    - **Unobserved Components (UCM)** — fully customisable component composition

    [:octicons-arrow-right-24: Structural models](user-guide/structural/index.md)

-   :material-cube-outline:{ .lg .middle } **Dynamic Factor Models (DFM)**

    ---

    Multivariate DFMs that extract $r$ common latent factors from a panel of
    $p$ observed series. Handles mixed-frequency panels, EM-based
    initialisation, and Kalman-based factor estimation and smoothing.

    Well-suited for macroeconomic nowcasting and financial index construction.

    [:octicons-arrow-right-24: Dynamic Factor Models](user-guide/advanced/dfm.md) ·
    [:octicons-arrow-right-24: Tutorial: US macro DFM](tutorials/us-macro-dfm.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Time-Varying Parameters (TVP)**

    ---

    Regression with state-dependent coefficients that evolve as random walks or
    stationary AR(1) processes. Captures structural breaks, slow parameter
    drift, and time-varying risk exposures (e.g. rolling CAPM betas).

    [:octicons-arrow-right-24: Time-Varying Parameters](user-guide/advanced/tvp.md) ·
    [:octicons-arrow-right-24: Tutorial: TVP-CAPM](tutorials/tvp-capm.md)

-   :material-function-variant:{ .lg .middle } **Alternative Filters**

    ---

    All share the same `filter()` / `smooth()` interface as the classical filter:

    | Filter | Use case |
    |--------|----------|
    | **EKF** | Mildly nonlinear, locally linearised |
    | **UKF** | Strongly nonlinear, sigma-point propagation |
    | **Square-Root** | Numerically stable for ill-conditioned $P$ |
    | **Information** | Sparse / high-dimensional inverse-covariance form |
    | **EnKF** | Monte Carlo, very high-dimensional systems |

    [:octicons-arrow-right-24: Alternative filters](user-guide/filters/index.md)

-   :material-dice-multiple:{ .lg .middle } **Bayesian Estimation — Gibbs / FFBS**

    ---

    Full Bayesian posterior inference via **Gibbs sampling** with the
    **Forward-Filter Backward-Sample (FFBS)** algorithm. Draws from the exact
    joint posterior of state trajectories and hyperparameters under conjugate
    Inverse-Gamma / Normal-Wishart priors.

    Includes MCMC convergence diagnostics (Gelman-Rubin $\hat{R}$, effective
    sample size) and posterior predictive checks.

    [:octicons-arrow-right-24: Bayesian estimation](user-guide/bayesian/index.md) ·
    [:octicons-arrow-right-24: FFBS](user-guide/bayesian/ffbs.md)

-   :material-cog-outline:{ .lg .middle } **MLE & EM Estimation**

    ---

    **Maximum Likelihood Estimation** via gradient-based optimisation of the
    exact Kalman log-likelihood (Durbin & Koopman 2012, Ch. 7). Supports
    bounded and unconstrained optimisers from `scipy.optimize`.

    **Expectation-Maximisation** provides closed-form updates for
    model families with conjugate sufficient statistics — faster convergence
    for large samples.

    [:octicons-arrow-right-24: Estimation overview](user-guide/bayesian/index.md)

-   :material-stethoscope:{ .lg .middle } **Diagnostics**

    ---

    Comprehensive post-estimation diagnostics callable from `results.diagnostics()`:

    - Standardised innovation residuals: ACF, Ljung-Box, Jarque-Bera
    - Information criteria: AIC, BIC, HQIC
    - Parameter stability: CUSUM, Harvey-Collier, Nyblom-Hansen
    - Numerical health: condition numbers, rank deficiency flags

    [:octicons-arrow-right-24: Diagnostics guide](diagnostics/index.md)

-   :material-chart-line:{ .lg .middle } **Integrated Visualisation**

    ---

    One-line plots without boilerplate:

    - Filtered and smoothed state trajectories with uncertainty bands
    - Forecast fan charts at arbitrary confidence levels
    - Component decomposition (trend, seasonal, cycle, irregular)
    - Diagnostic plots (residual ACF, QQ, CUSUM)

    Matplotlib backend by default; optional Plotly for interactive output.

    [:octicons-arrow-right-24: Visualization guide](visualization/index.md)

-   :material-console:{ .lg .middle } **CLI for Automation**

    ---

    Run models, produce HTML / PDF reports, and export results without writing
    Python — ideal for batch pipelines, CI/CD workflows, and reproducible
    research notebooks.

    ```bash
    kalmanbox fit --model LocalLevel --data nile.csv --output results/
    kalmanbox report results/ --format html
    ```

    [:octicons-arrow-right-24: CLI reference](api/cli.md)

</div>

---

## Quick example

=== "Kalman Filter"

    ```python
    import numpy as np
    from kalmanbox import KalmanFilter

    # AR(1) plus noise in state-space form
    kf = KalmanFilter(
        transition_matrices=[[0.95]],        # T: state transition
        observation_matrices=[[1.0]],        # Z: measurement
        transition_covariance=[[0.1]],       # Q: state disturbance variance
        observation_covariance=[[1.0]],      # H: observation noise variance
        initial_state_mean=[0.0],
        initial_state_covariance=[[1.0]],
    )

    rng = np.random.default_rng(42)
    y = rng.standard_normal(300)           # 300 synthetic observations

    result = kf.filter(y)
    print(result.log_likelihood)           # total log-likelihood
    print(result.filtered_state.shape)     # (300, 1)

    # Smooth over the full sample
    smoothed = kf.smooth(y)
    print(smoothed.smoothed_state[-1])     # last smoothed state mean
    ```

=== "Basic Structural Model (BSM)"

    ```python
    from kalmanbox import BSM
    from kalmanbox.datasets import load_dataset

    # Monthly airline passengers — classic Harvey (1989) example
    airline = load_dataset("airline")

    model = BSM(
        airline["passengers"],
        trend="local-linear",              # integrated random-walk trend
        seasonal="trigonometric",          # smooth harmonic seasonality
        seasonal_period=12,
        cycle=False,
    )

    results = model.fit(method="mle")      # gradient-based log-likelihood
    print(results.summary())

    # Decompose into trend, seasonal, irregular
    results.plot_components(figsize=(12, 8))

    # 24-month ahead forecast with 90 % confidence bands
    forecast = results.forecast(steps=24, alpha=0.10)
    forecast.plot()
    ```

=== "Bayesian (Gibbs / FFBS)"

    ```python
    from kalmanbox import LocalLevel, GibbsSampler
    from kalmanbox.bayesian import InverseGammaPrior
    from kalmanbox.datasets import load_dataset

    nile = load_dataset("nile")

    model = LocalLevel(nile["volume"])

    # Specify conjugate Inverse-Gamma priors on both variances
    priors = {
        "sigma2_level": InverseGammaPrior(shape=2.5, scale=5e4),
        "sigma2_obs":   InverseGammaPrior(shape=2.5, scale=5e4),
    }

    sampler = GibbsSampler(model, priors=priors, n_iter=3000, n_burn=1000)
    trace = sampler.sample(seed=0)

    # Posterior summaries
    print(trace.summary())
    trace.plot_posterior(["sigma2_level", "sigma2_obs"])

    # Posterior predictive forecast
    pred = trace.forecast(steps=10, credible_interval=0.95)
    pred.plot()
    ```

---

## State-space representation

Every `kalmanbox` model is expressed in the standard linear Gaussian form:

$$
\begin{aligned}
\alpha_{t+1} &= T_t\,\alpha_t + c_t + R_t\,\eta_t,
    &\quad \eta_t &\sim \mathcal{N}(0,\,Q_t), \\[4pt]
y_t &= Z_t\,\alpha_t + d_t + \varepsilon_t,
    &\quad \varepsilon_t &\sim \mathcal{N}(0,\,H_t).
\end{aligned}
$$

| Symbol | Dim | Role |
|--------|-----|------|
| $\alpha_t$ | $m \times 1$ | Latent state vector |
| $T_t$ | $m \times m$ | Transition (state evolution) matrix |
| $Z_t$ | $p \times m$ | Measurement (observation) matrix |
| $R_t$ | $m \times g$ | State disturbance selector |
| $Q_t$ | $g \times g$ | State disturbance covariance |
| $H_t$ | $p \times p$ | Observation noise covariance |
| $y_t$ | $p \times 1$ | Observed data at time $t$ |

The
[`StateSpaceRepresentation`][kalmanbox.core.representation.StateSpaceRepresentation]
object stores these matrices and is the single contract between models and
algorithms — swap the filter without touching the model, or swap the model
without touching the filter.

[:octicons-arrow-right-24: State-space theory](theory/state-space-theory.md) ·
[:octicons-arrow-right-24: Kalman Filter derivation](theory/kalman-filter-derivation.md) ·
[:octicons-arrow-right-24: RTS Smoother derivation](theory/rts-derivation.md)

---

## Ecosystem { #ecosystem }

!!! ecosystem "kalmanbox is the NodesEcon foundation"

    `kalmanbox` is **the base layer**. The higher-level libraries import its
    filters, smoothers, and `StateSpaceRepresentation` directly — they do not
    re-implement Kalman recursions.

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        NodesEcon stack                          │
    │                                                                 │
    │   particlefilterbox   forecastbox       chronobox               │
    │   (Sequential MC)     (Forecasting)     (Time-series toolbox)   │
    │          │                  │                  │                │
    │          └──────────────────┴──────────────────┘                │
    │                             │                                   │
    │                       kalmanbox  ← YOU ARE HERE                 │
    │          (Kalman filters · structural models · estimation)      │
    │                             │                                   │
    │              NumPy · SciPy · pandas · Matplotlib                │
    └─────────────────────────────────────────────────────────────────┘
    ```

    | Library | Depends on | Role |
    |---------|------------|------|
    | **kalmanbox** | NumPy · SciPy · pandas | Kalman filters, smoothers, structural models, DFM, TVP, MLE & Bayesian estimation |
    | **chronobox** | kalmanbox | Time-series toolbox: STL decomposition, calendar adjustment, seasonality, anomaly detection |
    | **forecastbox** | kalmanbox · chronobox | Forecasting framework: model selection, cross-validation, backtesting, ensembles |
    | **particlefilterbox** | kalmanbox | Sequential Monte Carlo: particle filters for non-Gaussian / nonlinear state-space models |

    [:octicons-arrow-right-24: Ecosystem overview](getting-started/ecosystem.md)

---

## Install

=== "Standard"

    ```bash
    pip install kalmanbox
    ```

=== "With extras"

    ```bash
    # Numba JIT acceleration (10–30× faster inner loops)
    pip install "kalmanbox[numba]"

    # Development tools (pytest, ruff, mypy, pre-commit)
    pip install "kalmanbox[dev]"

    # MkDocs documentation stack
    pip install "kalmanbox[docs]"

    # Everything
    pip install "kalmanbox[dev,docs,numba]"
    ```

=== "From source"

    ```bash
    git clone https://github.com/nodesecon/kalmanbox.git
    cd kalmanbox
    pip install -e ".[dev]"
    ```

**Requires:** Python 3.10+, NumPy ≥ 1.24, SciPy ≥ 1.10, pandas ≥ 2.0.

[:octicons-arrow-right-24: Full installation guide](getting-started/installation.md)

---

## Navigate the docs

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Installation, quickstart, and key concepts. Be productive in under five
    minutes.

    [:octicons-arrow-right-24: Getting Started](getting-started/index.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    In-depth coverage of every model family, filter variant, and estimation
    method — with worked examples at each step.

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-flask:{ .lg .middle } **Tutorials**

    ---

    End-to-end notebooks with real datasets: Nile River, airline passengers,
    US macro DFM, time-varying CAPM, and nonlinear tracking.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-bookshelf:{ .lg .middle } **API Reference**

    ---

    Auto-generated reference for every public class and function, with
    numpy-style docstrings and type signatures.

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

---

## License

MIT License — see [`LICENSE`](https://github.com/nodesecon/kalmanbox/blob/main/LICENSE).
