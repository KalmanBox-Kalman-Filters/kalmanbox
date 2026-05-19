# Advanced Models

Models that step beyond the standard univariate structural decomposition
— multivariate factor structures, time-varying coefficients, EM
estimation, multivariate systems, and fully custom state-space forms.

<div class="grid cards" markdown>

-   :material-share-variant:{ .lg .middle } **Dynamic Factor Model**

    ---

    Common factors driving multiple time series. Useful for
    macroeconomic nowcasting and yield-curve modelling.

    [:octicons-arrow-right-24: DFM](dfm.md)

-   :material-clock-time-eight-outline:{ .lg .middle } **Time-Varying Parameters**

    ---

    Regression coefficients that drift over time according to a random
    walk or AR(1) process.

    [:octicons-arrow-right-24: TVP](tvp.md)

-   :material-refresh:{ .lg .middle } **EM Algorithm**

    ---

    Expectation-Maximization for parameter estimation. E-step via Kalman
    smoother; M-step with closed-form updates. Preferred for DFMs and
    large multivariate models.

    [:octicons-arrow-right-24: EM Algorithm](em.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Multivariate Models**

    ---

    Kalman filter for $p > 1$ observation series. Full $H$ and $Q$
    covariance structures, VAR in state-space form, and mixed-frequency
    estimation.

    [:octicons-arrow-right-24: Multivariate](multivariate.md)

-   :material-vector-line:{ .lg .middle } **Regression-SSM**

    ---

    Static regression cast in state-space form — a trivial case that
    integrates with the rest of the framework.

    [:octicons-arrow-right-24: Regression-SSM](regression-ssm.md)

-   :material-tools:{ .lg .middle } **Custom**

    ---

    Build your own state-space model by specifying $T, Z, R, Q, H$.
    The escape hatch when the pre-built models are not enough.

    [:octicons-arrow-right-24: Custom](custom.md)

</div>

## Model complexity and prerequisites

| Model | Complexity | State dim | Typical use case |
|-------|-----------|-----------|-----------------|
| [Regression-SSM](regression-ssm.md) | Low | $k$ | Static OLS in SSM form |
| [TVP](tvp.md) | Medium | $k$ | Drifting regression coefficients |
| [DFM](dfm.md) | High | $k$ factors | Panel co-movement, nowcasting |
| [Multivariate](multivariate.md) | High | $m$ | Joint multivariate systems |
| [Custom](custom.md) | Variable | User-defined | Bespoke models from literature |
| [EM Algorithm](em.md) | — | — | Estimation method for any model above |

## When to reach for these

- You have **multivariate** data with shared drivers — go to [DFM](dfm.md).
- You have **multivariate** data with explicit system structure — go to [Multivariate](multivariate.md).
- You suspect your **regression coefficients drift** — go to [TVP](tvp.md).
- You need **robust parameter estimation** for a DFM or UCM — go to [EM Algorithm](em.md).
- You have a **bespoke model** from a paper or your own derivation — go to [Custom](custom.md).
