---
title: Getting Started
description: >-
  Your guided path from pip install to a fitted state-space model.
  Learn installation, core concepts, quickstart examples, and model selection.
---

# Getting Started

Welcome to **kalmanbox** — the foundational Kalman filter and state-space
modelling library for the [NodesEcon ecosystem](ecosystem.md).

This section takes you from a fresh `pip install` to confidently selecting,
fitting, and diagnosing state-space models. Whether you are new to Kalman
filtering or migrating from another library, the pages here provide a clear
on-ramp into `kalmanbox`.

---

## Learning path

```
Installation ──► Quickstart ──► Key Concepts ──► User Guide ──► Tutorials
     │               │               │
  verify setup   first model    mental model
                 in < 10 min    state-space
```

Work through the pages in order the first time. After that, jump directly to
the section you need.

---

## What you will learn

<div class="grid cards" markdown>

-   :material-package-variant:{ .lg .middle } **Installation**

    ---

    Install `kalmanbox` from PyPI, add optional extras (`numba` for speed,
    `docs` for building locally), and set up a development environment.
    Includes a verification step and common troubleshooting tips.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Four focused examples — raw `KalmanFilter`, `LocalLevel`, `BSM` with
    seasonality, and `RTSSmoother` — each with full code, expected output,
    and links to deeper documentation.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Key concepts**

    ---

    The mental model behind state-space representations, the Kalman
    recursion, the prediction-error decomposition, and how filters,
    smoothers, and estimators relate to each other.

    [:octicons-arrow-right-24: Key concepts](key-concepts.md)

-   :material-graph-outline:{ .lg .middle } **Ecosystem**

    ---

    How `kalmanbox` serves as the engine for `chronobox`, `forecastbox`,
    and `particlefilterbox`. Understand the shared `StateSpaceRepresentation`
    that connects the whole stack.

    [:octicons-arrow-right-24: Ecosystem](ecosystem.md)

-   :material-swap-horizontal:{ .lg .middle } **Migration guide**

    ---

    Coming from `statsmodels`, `pykalman`, or `filterpy`?
    Side-by-side comparisons and a migration checklist to move your
    existing code to `kalmanbox`.

    [:octicons-arrow-right-24: Migration](migration.md)

</div>

---

## Prerequisites

`kalmanbox` is a Python library. You will need:

| Requirement | Version |
|-------------|---------|
| Python | >= 3.11 |
| NumPy | >= 1.24 |
| SciPy | >= 1.10 |
| pandas | >= 2.0 |

!!! tip "Virtual environments"
    Always install `kalmanbox` inside a virtual environment
    (`python -m venv .venv` or `conda create -n myenv`).
    This prevents dependency conflicts with other projects.

Familiarity with NumPy arrays and basic probability is helpful, but not
required to use the high-level model APIs (`LocalLevel`, `BSM`, etc.).
The [Key Concepts](key-concepts.md) page explains the statistical background
you need.

---

## Choosing a model

Not sure which model fits your problem? Use this table as a starting point.

| Your data / goal | Recommended model | Where to learn more |
|-----------------|-------------------|---------------------|
| Smooth a noisy signal, no trend | `LocalLevel` | [Local Level](../user-guide/structural/local-level.md) |
| Trend + level without seasonality | `LocalLinearTrend` | [Local Linear Trend](../user-guide/structural/local-linear-trend.md) |
| Trend + seasonality + cycle | `BSM` | [Basic Structural Model](../user-guide/structural/bsm.md) |
| Arbitrary components (irregular) | `UCM` | [Unobserved Components](../user-guide/structural/ucm.md) |
| Multiple correlated series, few latent factors | `DFM` | [Dynamic Factor Model](../user-guide/advanced/dfm.md) |
| Regression with time-varying coefficients | `TVP` | [Time-Varying Parameters](../user-guide/advanced/tvp.md) |
| Nonlinear / non-Gaussian dynamics | `EKF`, `UKF`, `EnKF` | [Alternative Filters](../user-guide/filters/index.md) |
| Full posterior over parameters | `GibbsSampler`, `FFBS` | [Bayesian Estimation](../user-guide/bayesian/index.md) |
| Custom state-space formulation | `KalmanFilter` directly | [Custom Models](../user-guide/advanced/custom.md) |

!!! note "When to use `KalmanFilter` directly"
    The `LocalLevel`, `BSM`, and other model classes are thin wrappers that
    build a `StateSpaceRepresentation` and hand it to `KalmanFilter`.
    If your system matrices do not fit any built-in model, use `KalmanFilter`
    directly with your own $(T, Z, R, Q, H)$ matrices.

---

!!! ecosystem "Where kalmanbox sits"

    `kalmanbox` is the **foundation** of NodesEcon. Every higher-level library
    — `chronobox` (time-series toolbox), `forecastbox` (forecasting framework),
    and `particlefilterbox` (sequential Monte Carlo) — imports from `kalmanbox`
    and depends on its `StateSpaceRepresentation`. Understanding the primitives
    here pays dividends across the entire stack.

    See [Ecosystem](ecosystem.md) for a full architectural walkthrough.
