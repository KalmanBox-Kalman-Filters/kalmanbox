# Structural Time-Series Models

Structural time-series models decompose an observed series into a sum of
**unobserved, interpretable components** — level, trend, seasonal pattern,
cycle, and irregular noise — each governed by its own stochastic equation.
Unlike ARIMA models, which absorb all dynamics into a black-box polynomial,
structural models express the dynamics you believe in directly.

---

## Philosophy

The fundamental principle is **additive decomposition**:

$$
y_t = \mu_t + \beta_t + \gamma_t + \psi_t + \varepsilon_t
$$

| Symbol | Name | Describes |
|--------|------|-----------|
| $\mu_t$ | Level | The slowly-varying mean of the series |
| $\beta_t$ | Slope | The rate of change of the level |
| $\gamma_t$ | Seasonal | Repeating calendar patterns |
| $\psi_t$ | Cycle | Medium-frequency quasi-periodic variation |
| $\varepsilon_t$ | Irregular | Short-run noise, assumed i.i.d. Gaussian |

Each component is either **stochastic** (has its own shock) or **deterministic**
(fixed parameters). The choice of which components to include and whether they
are stochastic defines the model. kalmanbox casts every combination into a
unified state-space representation, so the Kalman filter handles all of them
with the same algorithm.

---

## Model hierarchy

Models in kalmanbox are nested from simple to complex. Each adds one or more
components to its predecessor:

```
LocalLevel
  └─ LocalLinearTrend    (adds stochastic slope β_t)
       └─ BSM            (adds seasonal component γ_t)
            └─ UCM       (full control: cycle, multiple seasonalities, …)
```

This hierarchy is also a **model-selection guide**: start with `LocalLevel` as
a baseline, add a slope if the series trends, add seasonality if calendar
patterns are evident, and use `UCM` when you need fine-grained control.

!!! tip "Mermaid decision tree"
    Not sure which model to use? See
    [Choosing a model](../../getting-started/choosing-model.md) for a
    step-by-step decision tree.

---

## Comparative table

| Model | Level | Slope | Seasonal | Cycle | Free params | State dim |
|-------|:-----:|:-----:|:--------:|:-----:|:-----------:|:---------:|
| `LocalLevel` | ✅ | ❌ | ❌ | ❌ | 2 | 1 |
| `LocalLinearTrend` | ✅ | ✅ | ❌ | ❌ | 3 | 2 |
| `BSM` (period $s$) | ✅ | ✅ | ✅ (dummy) | ❌ | 4 | $s+1$ |
| `UCM` | ⚙ | ⚙ | ⚙ | ⚙ | varies | varies |

*Free params* = number of variance parameters estimated by MLE.
*State dim* = dimension of the latent state vector $\alpha_t$.

---

## Models in this section

<div class="grid cards" markdown>

-   :material-trending-neutral:{ .lg .middle } **Local Level**

    ---

    The simplest structural model: a level that follows a random walk, observed
    with noise. The building block for everything else. Ideal for series with
    no trend or seasonality.

    **2 parameters** — $\sigma_\eta^2$ (level shock), $\sigma_\varepsilon^2$ (noise)

    [:octicons-arrow-right-24: Local Level](local-level.md)

-   :material-trending-up:{ .lg .middle } **Local Linear Trend**

    ---

    Adds a stochastic slope $\beta_t$ to the local level. The slope itself
    can drift over time, capturing series whose trend changes direction
    gradually. Special case: $\sigma_\zeta^2 = 0$ gives a fixed linear trend.

    **3 parameters** — $\sigma_\eta^2$, $\sigma_\zeta^2$, $\sigma_\varepsilon^2$

    [:octicons-arrow-right-24: Local Linear Trend](local-linear-trend.md)

-   :material-chart-areaspline:{ .lg .middle } **BSM**

    ---

    Basic Structural Model (Harvey 1989): trend + dummy-seasonal + irregular.
    The workhorse for monthly and quarterly economic series. Supports
    trigonometric seasonal form via `UCM`.

    **4 parameters** — level, slope, seasonal, irregular variances

    [:octicons-arrow-right-24: BSM](bsm.md)

-   :material-tune-variant:{ .lg .middle } **UCM**

    ---

    Fully configurable Unobserved Components Model. Mix and match level,
    slope, multiple seasonal frequencies (trigonometric), cycles with custom
    frequency and damping, and regressors.

    **Flexible** — any combination of stochastic components

    [:octicons-arrow-right-24: UCM](ucm.md)

-   :material-sine-wave:{ .lg .middle } **Cycle**

    ---

    Stochastic cycle component with free frequency $\lambda_c$, damping
    factor $\rho_c$, and shock variance $\sigma_\kappa^2$. Used as a
    stand-alone model or embedded inside UCM.

    [:octicons-arrow-right-24: Cycle](cycle.md)

-   :material-call-merge:{ .lg .middle } **ARIMA in state space**

    ---

    Cast any ARIMA$(p,d,q)$ into state-space form for unified likelihood
    evaluation, exact missing-data handling, and RTS smoothing.

    [:octicons-arrow-right-24: ARIMA-SSM](arima-ssm.md)

</div>

---

## Quick-start comparison

=== "Local Level"

    ```python
    from kalmanbox.structural import LocalLevel
    from kalmanbox.datasets import load_nile

    y = load_nile()["volume"].to_numpy()
    results = LocalLevel(y).fit()
    print(results.summary())
    ```

=== "Local Linear Trend"

    ```python
    from kalmanbox.structural import LocalLinearTrend
    import numpy as np

    y = np.log(load_gdp()["gdp"].to_numpy())
    results = LocalLinearTrend(y).fit()
    sm = results.smooth()
    trend = sm.components["level"]
    slope = sm.components["slope"]
    ```

=== "BSM"

    ```python
    from kalmanbox.structural import BSM
    from kalmanbox.datasets import load_airline
    import numpy as np

    y_log = np.log(load_airline()["passengers"].to_numpy())
    results = BSM(y_log, period=12).fit()
    decomp = results.smooth()
    print(decomp.components.keys())
    # dict_keys(['level', 'slope', 'seasonal', 'irregular'])
    ```

---

## Why structural models?

Structural models offer three advantages over reduced-form alternatives:

**1. Interpretability** — After fitting a BSM, you receive a decomposition into
trend, seasonal, and irregular. Each component has its own confidence band.
You can plot them, reason about them, and communicate them to non-specialists.

**2. Principled handling of missing data** — The Kalman filter propagates
uncertainty through gaps transparently. No imputation required. See
[Missing data](../kalman/missing-data.md).

**3. Seamless forecasting** — Projecting the state vector forward produces
multi-step-ahead forecasts with correct uncertainty that automatically widens
with horizon. The seasonal pattern and trend are projected jointly.

!!! ecosystem "Used by chronobox & forecastbox"

    `chronobox` uses BSM and UCM internally for trend/seasonal decomposition
    pipelines. `forecastbox` exposes them as forecasting families with
    automatic model selection. Both delegate the actual filtering, smoothing,
    and likelihood computation to `kalmanbox`.

---

## Related

- [Core concepts: state-space models](../../core-concepts.md)
- [Choosing a model](../../getting-started/choosing-model.md)
- [Theory: state-space foundations](../../theory/state-space-theory.md)
- [Tutorial: Nile with Local Level](../../tutorials/nile-local-level.md)
- [Tutorial: Airline passengers with BSM](../../tutorials/airline-bsm.md)
- [API: structural models](../../api/models.md)
