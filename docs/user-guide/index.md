# User Guide

The User Guide is organized by **what you want to do**, not by which class to instantiate.
Find your problem in one of the sections below and follow the cross-references into the API.

---

## Sections

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } **Kalman Filtering**

    ---

    The core forward recursion that produces optimal MMSE state estimates, plus the
    RTS backward smoother, forecasting beyond the sample, handling gaps in the data,
    and diffuse (non-stationary) initialization.

    **Start here** if you already have a state-space representation.

    [:octicons-arrow-right-24: Kalman Filter & RTS Smoother](kalman/index.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Structural Time-Series Models**

    ---

    Ready-made state-space forms for univariate series:
    Local Level, Local Linear Trend, Basic Structural Model (BSM),
    Unobserved Components Model (UCM), Cycle, and ARIMA-SSM.

    [:octicons-arrow-right-24: Structural Models](structural/index.md)

-   :material-cube-outline:{ .lg .middle } **Advanced Models**

    ---

    Dynamic Factor Models (DFM) for high-dimensional panels,
    Time-Varying Parameter (TVP) regression, regression in state-space form,
    and fully custom system matrices.

    [:octicons-arrow-right-24: Advanced Models](advanced/index.md)

-   :material-filter-variant:{ .lg .middle } **Alternative Filters**

    ---

    Nonlinear and non-Gaussian extensions:
    Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF),
    Square-Root Filter, Information Filter, and Ensemble Kalman Filter (EnKF).

    [:octicons-arrow-right-24: Alternative Filters](filters/index.md)

-   :material-dice-multiple:{ .lg .middle } **Bayesian Estimation**

    ---

    Full Bayesian treatment via Gibbs sampling and the
    Forward-Filter Backward-Sampler (FFBS):
    prior specification, posterior diagnostics, and convergence checks.

    [:octicons-arrow-right-24: Bayesian Estimation](bayesian/index.md)

</div>

---

## Choosing where to start

```
Is your system linear and Gaussian?
│
├─ Yes → Do you have a ready-made model (local level, BSM, etc.)?
│         ├─ Yes → Structural Models ──────────────────────────────────────────────┐
│         └─ No  → Kalman Filter (custom system matrices) ──────────────────────┐  │
│                                                                               │  │
├─ No (nonlinear / non-Gaussian) ─────────────────────────────────────────────>│  │
│         → Alternative Filters (EKF / UKF / EnKF)                             │  │
│                                                                               ▼  ▼
└─ Do you want uncertainty on the parameters too?                        Run the Filter
          └─ Yes → Bayesian Estimation (Gibbs / FFBS)                          │
                                                                               ▼
                                              Need a full-sample view? → RTS Smoother
```

---

## Recommended reading order

!!! tip "If you are new to state-space models"

    1. Read [Core Concepts](../getting-started/core-concepts.md) for notation and
       the model definition.
    2. Work through [Kalman Filter](kalman/kalman-filter.md) — the recursion,
       system matrices, and initialization.
    3. Continue to [RTS Smoother](kalman/rts-smoother.md) to understand backward
       refinement.
    4. Pick a [Structural Model](structural/index.md) that fits your series, or
       build a [custom system](advanced/custom.md).
    5. Explore [Bayesian Estimation](bayesian/index.md) or
       [Alternative Filters](filters/index.md) for special needs.

!!! tip "If you are porting from another library"

    See [Choosing a Model](../getting-started/choosing-model.md) for a
    feature-comparison matrix, and [Migration](../getting-started/migration.md)
    for code-level translation notes.

---

## Quick-start snippets

=== "Kalman Filter"

    ```python
    import numpy as np
    from kalmanbox import KalmanFilter, StateSpaceRepresentation

    # Local Level: alpha_t = alpha_{t-1} + eta_t, y_t = alpha_t + eps_t
    n = 200
    T = np.array([[1.0]])        # state transition
    Z = np.array([[1.0]])        # observation matrix
    R = np.array([[1.0]])        # selection matrix
    Q = np.array([[0.5]])        # state noise variance
    H = np.array([[1.0]])        # observation noise variance

    ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
    kf  = KalmanFilter(ssr, initialization="diffuse")

    y   = np.random.randn(n)
    out = kf.run(y)

    print(f"Log-likelihood: {out.loglike:.4f}")
    print(f"Filtered states: {out.a_filtered.shape}")
    ```

=== "RTS Smoother"

    ```python
    from kalmanbox import KalmanFilter, RTSSmoother, StateSpaceRepresentation

    ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
    kf  = KalmanFilter(ssr, initialization="diffuse")
    out = kf.run(y)

    smoother = RTSSmoother(out, ssr)
    sm = smoother.run()

    print(f"Smoothed states: {sm.a_smoothed.shape}")
    # Smoothed uncertainty is always <= filtered uncertainty
    assert (sm.P_smoothed <= out.P_filtered).all()
    ```

=== "Structural Model"

    ```python
    from kalmanbox.structural import BSM

    model = BSM(period=12)          # monthly seasonality
    result = model.fit(y, method="mle")

    print(result.summary())
    trend, seasonal, irregular = result.components()
    ```

=== "DFM"

    ```python
    from kalmanbox.advanced import DFM

    model = DFM(n_factors=2, n_obs=Y.shape[1])
    result = model.fit(Y, method="em", max_iter=500)

    factors = result.smoothed_factors    # shape (n, 2)
    loadings = result.loadings           # shape (p, 2)
    ```

---

## Relationship to the NodesEcon ecosystem

kalmanbox is the **foundational layer** used by every other NodesEcon library:

```
┌─────────────────────────────────────────────────────────────┐
│  chronobox · forecastbox · particlefilterbox · …            │  ← domain libraries
└──────────────────────────┬──────────────────────────────────┘
                           │ builds on
         ┌─────────────────▼─────────────────┐
         │           kalmanbox               │  ← you are here
         │  KalmanFilter · RTSSmoother       │
         │  Structural · Advanced · Bayes    │
         └───────────────────────────────────┘
```

All higher-level libraries delegate their filtering and smoothing to `kalmanbox`
and rely on its numerically stable implementations and consistent `FilterOutput` /
`SmootherOutput` interfaces.

---

## Related sections

- [Getting Started](../getting-started/index.md) — installation, quickstart, key concepts
- [Theory](../theory/index.md) — mathematical derivations and references
- [API Reference](../api/index.md) — complete class and function documentation
- [Diagnostics](../diagnostics/index.md) — residual analysis and information criteria
- [Tutorials](../tutorials/index.md) — end-to-end worked examples
