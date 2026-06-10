# Time-Varying Parameters (TVP)

A TVP regression replaces the static OLS coefficient vector $\beta$ with a
**time-varying state** $\beta_t$ that evolves stochastically. This captures
structural change, regime shifts, and slowly evolving economic or financial
relationships that a fixed-coefficient model cannot represent.

---

## 1. Concept

In a standard regression, $\beta$ is estimated once from the full sample and
treated as constant across all observations. When the true relationship between
$y_t$ and $x_t$ shifts over time — gradually or abruptly — the OLS estimate
becomes a weighted average of all regimes and loses interpretive power.

TVP models resolve this by allowing $\beta_t$ to be **a function of time**,
modelled as a latent stochastic process. Typical applications include:

- **Macroeconomics**: a Phillips curve whose inflation-unemployment slope
  flattens as central bank credibility improves.
- **Finance**: a CAPM market beta that changes as a firm's leverage or sector
  exposure evolves.
- **Structural change detection**: a relationship that shifts after a policy
  change, crisis, or technological disruption.

!!! note "TVP vs structural breaks"
    Discrete structural-break models (Chow test, Bai-Perron) assume $\beta$
    jumps at unknown break dates. TVP models assume $\beta_t$ drifts continuously,
    making them better suited when change is gradual. If a break is abrupt and
    well-located, a break model may be more parsimonious.

---

## 2. State-Space Formulation

### 2.1 Observation equation

$$
y_t = x_t'\,\beta_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\,\sigma^2)
$$

where:

- $y_t \in \mathbb{R}$ is the scalar observation at time $t$,
- $x_t \in \mathbb{R}^k$ is the regressor vector (design vector) at time $t$,
- $\beta_t \in \mathbb{R}^k$ is the latent coefficient vector at time $t$,
- $\varepsilon_t$ is the observation noise with variance $\sigma^2$.

### 2.2 State (coefficient) transition

**Random walk (default)**

$$
\beta_{t+1} = \beta_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\,Q)
$$

The coefficient vector performs a multivariate random walk. Setting $Q = 0$
recovers the static OLS fixed-$\beta$ model. Larger $Q$ lets $\beta_t$ drift
faster.

**AR(1) / mean-reverting**

$$
\beta_{t+1} = \mu + \Phi\,(\beta_t - \mu) + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\,Q)
$$

where $\Phi = \operatorname{diag}(\phi_1, \ldots, \phi_k)$ with $|\phi_i| < 1$
for all $i$. The coefficient vector mean-reverts to the long-run level $\mu
\in \mathbb{R}^k$ at rate $\Phi$. This is useful when economic theory implies
that coefficients should not wander arbitrarily far from a prior belief.

### 2.3 Standard SSM matrices

TVP is a special case of the linear Gaussian state-space model:

$$
\begin{aligned}
\alpha_{t+1} &= T\,\alpha_t + c + R\,\eta_t, \qquad \eta_t \sim \mathcal{N}(0,\,Q) \\
y_t          &= Z_t\,\alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\,H)
\end{aligned}
$$

with the following assignments:

| SSM symbol | TVP assignment | Notes |
|------------|---------------|-------|
| State $\alpha_t$ | $\beta_t$ | $k$-dimensional coefficient vector |
| $Z_t$ | $x_t'$ | Time-varying $1 \times k$ design row |
| $T$ | $I_k$ (random walk) or $\Phi$ (AR(1)) | $k \times k$ transition matrix |
| $c$ | $0$ (RW) or $(I - \Phi)\mu$ (AR(1)) | Constant offset |
| $R$ | $I_k$ | Full-rank noise loading |
| $Q$ | $\operatorname{diag}(q_1, \ldots, q_k)$ | Evolution variance (diagonal default) |
| $H$ | $\sigma^2$ | Scalar observation variance |

!!! tip "Time-varying design matrix"
    Unlike standard structural models where $Z$ is constant, here $Z_t = x_t'$
    varies at every time step. The Kalman filter handles this naturally — each
    step simply uses the contemporaneous regressor row.

---

## 3. Variants and Special Cases

| Variant | Transition $T$ | Evolution $Q$ | Behavior |
|---------|---------------|--------------|---------|
| Static OLS | $I_k$ | $0$ | $\beta_t \equiv \beta$ — reduces exactly to GLS/OLS |
| Random walk | $I_k$ | $\operatorname{diag}(q_1,\ldots,q_k)$ | Continuous drift without bound |
| AR(1) mean-reverting | $\Phi$ | $\operatorname{diag}(q_i)$ | Drift with pull back to long-run mean $\mu$ |
| Block-varying | $I_k$ | $\operatorname{blockdiag}(Q_{11}, 0)$ | Some coefficients drift, others are fixed |
| Full covariance | $I_k$ | $Q$ (dense) | All coefficients co-drift; $k(k+1)/2$ extra params |

The **block-varying** variant is particularly useful when theory indicates that
certain regressors (e.g.\ seasonal dummies) should have fixed coefficients while
others (e.g.\ a trend or policy variable) are allowed to evolve.

---

## 4. Estimation of Evolution Variances

The key unknown in TVP is the **evolution variance** $Q$. Its magnitude
controls the flexibility of $\beta_t$.

### 4.1 Signal-to-noise ratio

The ratio $q_i / \sigma^2$ is the signal-to-noise ratio (SNR) for coefficient $i$:

$$
\text{SNR}_i = \frac{q_i}{\sigma^2}
$$

- **Small SNR** ($q_i \ll \sigma^2$): $\beta_{i,t}$ barely moves — close to OLS.
- **Large SNR** ($q_i \gg \sigma^2$): $\beta_{i,t}$ tracks every fluctuation —
  high flexibility but possibly overfitting.

Typical macro applications find $\text{SNR} \in [10^{-4},\,10^{-1}]$, meaning
coefficients evolve slowly relative to observation noise.

### 4.2 Maximum likelihood (MLE)

The prediction-error decomposition of the Kalman filter yields the Gaussian
log-likelihood:

$$
\ell(\theta) = -\frac{n}{2}\log(2\pi)
               - \frac{1}{2}\sum_{t=1}^{n}\left(\log F_t + \frac{v_t^2}{F_t}\right)
$$

where $v_t = y_t - x_t'\,a_{t|t-1}$ is the one-step-ahead forecast error and
$F_t = x_t'\,P_{t|t-1}\,x_t + \sigma^2$ is its variance. The parameter vector
$\theta = (\sigma^2, q_1, \ldots, q_k)$ is optimized numerically. Because $q_i
\geq 0$, optimization is typically done over $\log q_i$ to avoid boundary issues.

!!! note "Flat likelihood surface"
    With many regressors, $\ell(\theta)$ can be flat in the $Q$ direction.
    Starting from a grid of initial values or using profile likelihood diagnostics
    (plotting $\ell$ as a function of each $q_i$) is recommended.

### 4.3 Bayesian estimation (Gibbs / FFBS)

Bayesian inference places priors on $Q$ and $\sigma^2$ and samples from the
joint posterior $p(\beta_{1:T}, Q, \sigma^2 \mid y_{1:T})$ using a
**Gibbs sampler**:

1. **State draw**: sample the entire trajectory $\beta_{1:T}$ conditional on
   $Q, \sigma^2, y_{1:T}$ using the **Forward Filter Backward Sampler (FFBS)**.
2. **Variance draw**: given $\beta_{1:T}$, the increments $\eta_t = \beta_{t+1}
   - \beta_t$ are observed. Each $q_i$ is drawn from its conjugate
   **Inverse-Gamma** posterior:

$$
q_i \mid \beta_{1:T} \sim \mathcal{IG}\!\left(
  a_0 + \frac{T-1}{2},\;\;
  b_0 + \frac{1}{2}\sum_{t=1}^{T-1}(\beta_{i,t+1} - \beta_{i,t})^2
\right)
$$

3. **Observation variance draw**: similarly, residuals $\varepsilon_t = y_t -
   x_t'\,\beta_t$ give the posterior for $\sigma^2$.

The Bayesian approach automatically regularizes $Q$ through the prior, avoiding
the flat-likelihood problem of MLE. See [Bayesian estimation](../bayesian/index.md)
and [FFBS](../bayesian/ffbs.md) for full details.

### 4.4 EM algorithm

The EM algorithm iterates between:

- **E-step**: run the Kalman smoother to compute $E[\beta_t \mid y]$ and
  $E[\beta_t \beta_t' \mid y]$.
- **M-step**: closed-form update for $Q$ using the smoothed cross-products
  of successive states.

EM is attractive because it avoids numerical Hessians and guarantees
monotone likelihood ascent. See [EM Algorithm](em.md) for details.

---

## 5. Basic Example

```python
import numpy as np
from kalmanbox.advanced import TVP
from kalmanbox.datasets import load_macro

data = load_macro()

y = data["inflation"].to_numpy()       # inflation rate
u = data["unemployment"].to_numpy()    # unemployment gap
X = np.column_stack([np.ones(len(y)), u])  # [intercept, unemployment]

model = TVP(y, exog=X)
results = model.fit()

print(results.summary())

sm = results.smooth()
beta_t = sm.a_smoothed    # shape (T, 2): [intercept_t, slope_t]
beta_ci = sm.confidence_intervals(alpha=0.05)  # 95% credible bands
```

The `a_smoothed` array has shape `(T, k)`, where each row is the smoothed
(retrospective) estimate of $\beta_t$. The `confidence_intervals` method
returns pointwise bands derived from the smoothed covariance `P_smoothed`.

!!! tip "Filtered vs smoothed coefficients"
    `results.filter().a_filtered` gives the **one-sided** (causal) estimate
    of $\beta_t$ using only $y_1, \ldots, y_t$.
    `results.smooth().a_smoothed` uses the **full sample** $y_1, \ldots, y_T$
    and is preferred for retrospective analysis. For real-time forecasting,
    use the filtered estimate.

---

## 6. Phillips Curve Example

The time-varying Phillips curve is a canonical application of TVP models.
It asks: how has the inflation-unemployment trade-off changed over time?

$$
\pi_t = \alpha_t + \beta_t\,u^{\text{gap}}_t + \gamma_t\,\pi_{t-1} + \varepsilon_t
$$

where $\alpha_t$ captures trend inflation, $\beta_t$ is the (possibly flattening)
unemployment sensitivity, and $\gamma_t$ measures inflation persistence.

```python
import numpy as np
from kalmanbox.advanced import TVP
from kalmanbox.datasets import load_us_macro

data = load_us_macro(freq="quarterly", start="1960Q1", end="2023Q4")

y = data["core_cpi_change"].to_numpy()          # quarterly inflation
u_gap = data["unemployment_gap"].to_numpy()      # unemployment gap
pi_lag = data["core_cpi_change"].shift(1).dropna().to_numpy()  # lagged inflation

# TVP Phillips curve: π_t = α_t + β_t * u_gap_t + γ_t * π_{t-1} + ε_t
X = np.column_stack([np.ones(len(y)-1), u_gap[1:], pi_lag])
y_trim = y[1:]

model = TVP(y_trim, exog=X)
results = model.fit(method="mle")

sm = results.smooth()
alpha_t = sm.a_smoothed[:, 0]   # time-varying intercept (trend inflation)
beta_t  = sm.a_smoothed[:, 1]   # time-varying slope (unemployment sensitivity)
gamma_t = sm.a_smoothed[:, 2]   # time-varying inflation persistence

print(f"Mean unemployment slope: {beta_t.mean():.4f}")
print(f"Slope range: [{beta_t.min():.4f}, {beta_t.max():.4f}]")
```

!!! note "Economic interpretation"
    The slope $\beta_t$ measuring inflation-unemployment sensitivity has
    flattened in recent decades (moving closer to zero), consistent with the
    "flattening Phillips curve" documented post-1990s. A negative $\beta_t$
    reflects the conventional trade-off: higher unemployment gaps are
    disinflationary. As the curve flattens, the Fed's ability to reduce
    inflation by tolerating unemployment diminishes.

---

## 7. Time-Varying CAPM Example

The Capital Asset Pricing Model (CAPM) assumes a constant market beta:
$r_{i,t} = \alpha + \beta\,r_{m,t} + \varepsilon_t$. In practice, a firm's
systematic risk exposure evolves with leverage, product mix, and market
conditions. TVP-CAPM allows $\beta_t$ to vary continuously.

```python
import numpy as np
from kalmanbox.advanced import TVP
from kalmanbox.datasets import load_returns

data = load_returns(["AAPL", "SPY"], start="2010-01-01", end="2023-12-31")

r_stock = data["AAPL"].to_numpy()
r_market = data["SPY"].to_numpy()

# CAPM: r_t = α_t + β_t * r_market_t + ε_t
X = np.column_stack([np.ones(len(r_stock)), r_market])
model = TVP(r_stock, exog=X)
results = model.fit()

sm = results.smooth()
alpha_t = sm.a_smoothed[:, 0]   # Jensen's alpha (time-varying)
beta_t  = sm.a_smoothed[:, 1]   # market beta (time-varying)

# Beta uncertainty bands
ci = sm.confidence_intervals(alpha=0.05)
beta_lower = ci["lower"][:, 1]
beta_upper = ci["upper"][:, 1]

print(f"Current beta: {beta_t[-1]:.3f} (95% CI: [{beta_lower[-1]:.3f}, {beta_upper[-1]:.3f}])")
```

!!! tip "Rolling window vs TVP"
    A common alternative is rolling OLS with a fixed window $w$. TVP is
    preferable because it:

    - Uses all data (not just the window), weighted by the Kalman gain.
    - Produces uncertainty bands from the state covariance, not just sampling
      error of OLS.
    - Does not require choosing an arbitrary window length — the evolution
      variance $Q$ is estimated from the data.

---

## 8. AR(1) / Mean-Reverting Variant

When theory implies that coefficients should revert to a known or estimated
long-run level rather than wandering unboundedly, use the AR(1) transition:

$$
\beta_{t+1} = \mu + \Phi\,(\beta_t - \mu) + \eta_t
$$

The unconditional mean of $\beta_t$ is $\mu$, and the autocorrelation at lag
$h$ is $\Phi^h$. With $\Phi$ close to 1 the process is highly persistent
(near-random-walk); with $\Phi$ close to 0 the coefficients rapidly revert to
$\mu$.

```python
# Mean-reverting TVP: β_t reverts to long-run mean
model_ar = TVP(
    y, exog=X,
    beta_transition="ar1",          # AR(1) transition
    phi_init=0.9,                   # initial AR coefficient
    mu_init=np.array([-0.2, 0.5])  # initial long-run mean per coefficient
)
results_ar = model_ar.fit()
```

The parameters $\Phi$ and $\mu$ are estimated jointly with $Q$ and $\sigma^2$
via MLE or Bayesian sampling. Constraining $\phi_i \in (-1,\,1)$ is enforced
internally by reparameterizing as $\phi_i = \tanh(\tilde\phi_i)$.

!!! warning "AR(1) identification"
    With $k$ coefficients, the AR(1) variant adds $2k$ parameters ($\phi_i$ and
    $\mu_i$) on top of $Q$. When the sample is short, prefer fixing $\mu = 0$ and
    estimating only $\Phi$ to avoid overparameterization.

---

## 9. Interpreting Time-Varying Coefficients

### 9.1 Plotting coefficient evolution

```python
import matplotlib.pyplot as plt

sm = results.smooth()
ci = sm.confidence_intervals(alpha=0.05)

dates = data.index  # datetime index from the dataset

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for i, (ax, name) in enumerate(zip(axes, ["Intercept α_t", "Slope β_t"])):
    ax.plot(dates, sm.a_smoothed[:, i], label="Smoothed estimate")
    ax.fill_between(
        dates,
        ci["lower"][:, i],
        ci["upper"][:, i],
        alpha=0.2, label="95% CI"
    )
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
```

The shaded credible bands widen in periods with sparse data or high observation
noise, reflecting genuine uncertainty about the coefficient at those times.

### 9.2 Structural break vs smooth change

TVP coefficient paths can reveal whether apparent change is:

- **Gradual drift**: $\beta_t$ moves slowly and monotonically — typical of
  evolving economic relationships.
- **Step change**: $\beta_t$ jumps quickly then stabilizes — suggests a discrete
  structural break. A Chow-test breakpoint model may be more parsimonious.
- **Oscillatory**: $\beta_t$ fluctuates around a mean — consistent with
  business-cycle variation; the AR(1) mean-reverting variant is appropriate.

### 9.3 Testing for time-variation

To test $H_0: Q = 0$ (static OLS) against $H_1: Q > 0$ (TVP), a standard
likelihood ratio (LR) test can be used:

$$
\text{LR} = 2\left[\ell(\hat\theta_{\text{TVP}}) - \ell(\hat\theta_{\text{OLS}})\right]
\;\xrightarrow{d}\; \bar\chi^2_k
$$

!!! warning "Boundary testing"
    Under $H_0$, $Q = 0$ is on the boundary of the parameter space, so the
    asymptotic null distribution is a **mixture of chi-squared distributions**
    ($\bar\chi^2$), not the standard $\chi^2_k$. Inference based on the naive
    $\chi^2_k$ critical values will be conservative (over-rejects). See
    Andrews (2001) for correct critical values.

A simpler diagnostic is to inspect the profile log-likelihood
$\ell(q_i)$ while fixing other parameters: a flat or non-concave profile
suggests $q_i$ is not well-identified from the data.

---

## 10. Practical Guidelines

### 10.1 Signal-to-noise ratio guidance

| SNR ($q_i / \sigma^2$) | Regime | Practical recommendation |
|------------------------|--------|------------------------|
| $< 10^{-4}$ | Near-static | Use OLS; TVP adds no information |
| $10^{-4}$ – $10^{-2}$ | Slow drift | Typical macro coefficients (decade-scale change) |
| $10^{-2}$ – $10^{-1}$ | Moderate drift | Business-cycle variation, financial betas |
| $> 10^{-1}$ | Rapid variation | Possible overfitting; check out-of-sample performance |

### 10.2 When to use TVP vs OLS

| Situation | Recommendation |
|-----------|---------------|
| Theory says $\beta$ is constant | OLS — fewer parameters, lower variance |
| Known structural break at date $\tau$ | Split-sample OLS or Chow model |
| Gradual, unknown structural change | TVP with random-walk transition |
| Coefficients expected to revert | TVP with AR(1) transition |
| High-frequency financial data | TVP — betas shift with regimes |

### 10.3 Computational cost

The Kalman filter pass is $\mathcal{O}(T k^3)$ per likelihood evaluation due to
matrix inversions of size $k \times k$. For large $k$, this is the dominant cost.
MLE with numerical gradient is $\mathcal{O}(T k^3 d)$ where $d$ is the number
of free parameters in $Q$ (at most $k(k+1)/2$ for a full matrix, just $k$
for diagonal $Q$). Prefer diagonal $Q$ unless strong a priori evidence exists
for cross-coefficient drift correlation.

!!! warning "Identification with many regressors"
    With $k$ regressors, a full $Q$ matrix has $k(k+1)/2$ free parameters.
    For $k = 10$ this is already 55 parameters to identify, often exceeding
    the information content of a typical macro dataset. Practical remedies:

    - **Diagonal $Q$**: $k$ parameters, default in `kalmanbox`.
    - **Scalar $Q = q I_k$**: 1 parameter — all coefficients evolve at the same
      rate.
    - **Block-diagonal $Q$**: fix some coefficients, allow others to drift.
    - **Bayesian shrinkage**: hierarchical prior that pools information across
      coefficients, effectively shrinking small $q_i$ toward zero. See
      [Priors](../bayesian/priors.md).

---

## 11. API Reference

::: kalmanbox.models.tvp.TimeVaryingParameters
    options:
      heading_level: 3
      show_source: false

---

## 12. Related Pages

- [EM Algorithm](em.md) — closed-form M-step for $Q$ estimation
- [Bayesian estimation](../bayesian/index.md) — Gibbs / FFBS for full posterior
  inference over $\beta_{1:T}$, $Q$, and $\sigma^2$
- [FFBS](../bayesian/ffbs.md) — Forward Filter Backward Sampler used in the
  Bayesian state draw
- [Dynamic Factor Model](dfm.md) — latent factor alternative when $y_t$ is
  multivariate
- [MLE](../kalman/mle.md) — prediction-error likelihood and optimization
- [Tutorial: time-varying CAPM](../../tutorials/tvp-capm.md)
- [Regression-SSM](regression-ssm.md) — static-coefficient regression in
  state-space form

---

## 13. References

Primiceri, G. E. (2005). Time Varying Structural Vector Autoregressions and
Monetary Policy. *Review of Economic Studies*, 72(3), 821–852.

Kim, C. J. & Nelson, C. R. (1999). *State-Space Models with Regime Switching*.
MIT Press.

Stock, J. H. & Watson, M. W. (1996). Evidence on Structural Instability in
Macroeconomic Time Series Relations. *Journal of Business & Economic Statistics*,
14(1), 11–30.

Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
Methods* (2nd ed.), Chapters 3 & 9. Oxford University Press.

Andrews, D. W. K. (2001). Testing When a Parameter Is on the Boundary of the
Maintained Hypothesis. *Econometrica*, 69(3), 683–734.
