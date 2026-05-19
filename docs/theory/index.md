# Theory

This section contains the **mathematical background** behind the
algorithms in `kalmanbox`. It is meant to be read alongside the
[User Guide](../user-guide/index.md) — the user guide tells you
*what* to call; the theory section tells you *why* it works.

## Comprehensive Theory Pages

<div class="grid cards" markdown>

-   :material-sigma:{ .lg .middle } **State-Space Theory**

    ---

    General linear Gaussian representation, system matrices, stability,
    observability, and connections to ARMA/VAR representations.

    [:octicons-arrow-right-24: State-Space Theory](state-space-theory.md)

-   :material-filter:{ .lg .middle } **Kalman Filter Theory**

    ---

    BLUP and Bayesian derivations, optimality proofs, prediction-error
    decomposition, steady-state filter, and Riccati equation.

    [:octicons-arrow-right-24: Kalman Filter Theory](kalman-theory.md)

-   :material-arrow-left-bold:{ .lg .middle } **Smoothing Theory**

    ---

    RTS and disturbance smoothers, fixed-interval/point/lag smoothers,
    lag-one covariance, and connection to the EM E-step.

    [:octicons-arrow-right-24: Smoothing Theory](smoothing-theory.md)

-   :material-chart-bell-curve:{ .lg .middle } **MLE Theory**

    ---

    Prediction-error log-likelihood, score and information matrix,
    asymptotic standard errors, optimization, and information criteria.

    [:octicons-arrow-right-24: MLE Theory](mle-theory.md)

-   :material-chart-timeline:{ .lg .middle } **Structural Models Theory**

    ---

    Local Level, Local Linear Trend, BSM, UCM, trigonometric seasonal,
    cycle components, and identification.

    [:octicons-arrow-right-24: Structural Theory](structural-theory.md)

</div>

## Derivations

<div class="grid cards" markdown>

-   :material-arrow-right-bold:{ .lg .middle } **Kalman filter derivation**

    ---

    Step-by-step derivation of the prediction/update recursion from
    first principles.

    [:octicons-arrow-right-24: KF derivation](kalman-filter-derivation.md)

-   :material-arrow-left-bold:{ .lg .middle } **RTS smoother derivation**

    ---

    Two derivations of the backward recursion: from the joint Gaussian,
    and via the smoothing gain.

    [:octicons-arrow-right-24: RTS derivation](rts-derivation.md)

-   :material-function:{ .lg .middle } **Likelihood**

    ---

    Prediction-error decomposition, diffuse log-likelihood, EM and
    score functions.

    [:octicons-arrow-right-24: Likelihood](likelihood.md)

</div>

## Advanced Topics

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } **Numerical stability**

    ---

    Joseph form, square-root forms, condition-number analysis,
    common pitfalls.

    [:octicons-arrow-right-24: Numerical](numerical-stability.md)

-   :material-puzzle:{ .lg .middle } **Identifiability**

    ---

    When are state-space parameters identified? Practical
    reparametrisations and prior-driven solutions.

    [:octicons-arrow-right-24: Identifiability](identifiability.md)

-   :material-layers-triple:{ .lg .middle } **DFM Theory**

    ---

    Dynamic Factor Models: formulation, identification, EM estimation,
    factor selection criteria (Bai-Ng IC), asymptotic theory.

    [:octicons-arrow-right-24: DFM Theory](dfm-theory.md)

-   :material-sine-wave:{ .lg .middle } **Nonlinear Filter Theory**

    ---

    EKF linearization, UKF sigma-point transform, Ensemble Kalman
    Filter derivation, and comparison with particle filters.

    [:octicons-arrow-right-24: Nonlinear Theory](nonlinear-theory.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Bayesian Theory**

    ---

    FFBS derivation, Gibbs sampler for state-space, conjugate priors,
    MCMC convergence, and connection to EM.

    [:octicons-arrow-right-24: Bayesian Theory](bayesian-theory.md)

-   :material-infinity:{ .lg .middle } **Diffuse Initialization Theory**

    ---

    Exact diffuse filter (De Jong 1991, Koopman 1997), diffuse
    log-likelihood, mixed initialization, and augmented Kalman filter.

    [:octicons-arrow-right-24: Diffuse Theory](diffuse-theory.md)

-   :material-bookshelf:{ .lg .middle } **References**

    ---

    Comprehensive bibliography: books, seminal papers, software
    systems, and the NodesEcon ecosystem.

    [:octicons-arrow-right-24: References](references.md)

</div>
