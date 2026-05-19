# Gibbs Sampling for State-Space Models

**Gibbs sampling** is a Markov Chain Monte Carlo (MCMC) algorithm for drawing from a joint
distribution by iteratively sampling each variable (or block of variables) from its **full
conditional distribution** — the distribution of that variable given all others and the data.

For linear Gaussian state-space models, the joint posterior factorises in a way that makes two
conditional distributions analytically tractable:

$$
p\!\left(\theta, \{\alpha_t\} \mid y_{1:T}\right)
$$

- $p\!\left(\{\alpha_t\} \mid \theta, y_{1:T}\right)$ — multivariate Gaussian, sampled exactly
  by [FFBS](ffbs.md)
- $p\!\left(\theta \mid \{\alpha_t\}, y_{1:T}\right)$ — conjugate closed-form posteriors when
  priors are Inverse-Gamma (variances) and Normal (coefficients)

Alternating these two blocks produces a **valid MCMC sampler** for the joint posterior.

!!! note "Gibbs for non-Gaussian models"
    When the observation distribution is non-Gaussian (e.g., count data, stochastic volatility),
    the observation equation block is no longer conjugate. kalmanbox handles this via Metropolis-
    within-Gibbs steps. For strongly non-Gaussian or nonlinear dynamics, see
    [`particlefilterbox`](../../getting-started/ecosystem.md).

---

## 1. The state-space model

Consider the general linear Gaussian SSM:

$$
\begin{aligned}
y_t        &= Z\,\alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, H) \\
\alpha_{t+1} &= T\,\alpha_t + R\,\eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)
\end{aligned}
$$

with parameter vector $\theta = (T, Z, R, Q, H, a_0, P_0)$. The joint complete-data
log-likelihood over observations and states is:

$$
\log p(y_{1:T}, \{\alpha_t\} \mid \theta)
= -\frac{1}{2} \sum_{t=1}^{T} \varepsilon_t' H^{-1} \varepsilon_t
  -\frac{1}{2} \sum_{t=0}^{T-1} \eta_t' Q^{-1} \eta_t
  - \frac{T}{2}\log|H| - \frac{T}{2}\log|Q| + \text{const}
$$

where $\varepsilon_t = y_t - Z\alpha_t$ and $\eta_t = \alpha_{t+1} - T\alpha_t$.

---

## 2. Algorithm structure

### 2.1 Pseudocode

```text
Initialise:
    θ⁽⁰⁾  ← MLE estimate or prior mean
    α⁽⁰⁾  ← Kalman smoother means under θ⁽⁰⁾

For s = 1, 2, ..., S:
    ── State block (FFBS) ──────────────────────────────────────
    α⁽ˢ⁾  ~ p(α₁:T | y₁:T, θ⁽ˢ⁻¹⁾)     [forward filter + backward sample]

    ── Parameter block (conjugate draws) ───────────────────────
    H⁽ˢ⁾  ~ p(H   | y₁:T, α⁽ˢ⁾)        [Inverse-Wishart]
    Q⁽ˢ⁾  ~ p(Q   | y₁:T, α⁽ˢ⁾)        [Inverse-Wishart]
    T⁽ˢ⁾  ~ p(T   | y₁:T, α⁽ˢ⁾)        [Matrix-Normal, if T is free]
    ...

Discard: θ⁽¹⁾, ..., θ⁽ᴮ⁾    (burn-in)
Keep:    θ⁽ᴮ⁺¹⁾, θ⁽ᴮ⁺¹⁺ᵏ⁾, ...   (post-burn, thinned every k steps)
```

### 2.2 Why this converges

The Gibbs sampler produces a Markov chain $\{(\theta^{(s)}, \{\alpha_t\}^{(s)})\}$ whose
**stationary distribution is the joint posterior**. This follows from the theory of
**detailed balance**: each block update leaves the joint posterior invariant because we
draw exactly from the correct conditional distribution.

Under the linear Gaussian SSM, geometric ergodicity holds with a rate determined by the
spectral radius of the sampler's transition operator — typically fast for well-identified models.

---

## 3. Conjugate posterior conditionals

### 3.1 Observation noise variance $H$ (scalar case)

**Prior:** $H = \sigma_\varepsilon^2 \sim \mathcal{IG}(a_\varepsilon, b_\varepsilon)$

The complete-data likelihood contributes:

$$
p(y_{1:T} \mid \alpha_{1:T}, \sigma_\varepsilon^2) \propto
(\sigma_\varepsilon^2)^{-T/2} \exp\!\left(-\frac{1}{2\sigma_\varepsilon^2}
\sum_{t=1}^{T}(y_t - Z\alpha_t)^2\right)
$$

Combining with the Inverse-Gamma prior gives the **conjugate posterior**:

$$
\boxed{
\sigma_\varepsilon^2 \mid \alpha_{1:T}, y_{1:T} \sim \mathcal{IG}\!\left(
a_\varepsilon + \frac{T}{2},\;
b_\varepsilon + \frac{1}{2}\sum_{t=1}^{T}(y_t - Z\alpha_t)^2
\right)
}
$$

The updated shape adds $T/2$ observations; the updated scale adds half the residual sum of squares.

### 3.2 State noise variance $Q$ (scalar case)

**Prior:** $Q = \sigma_\eta^2 \sim \mathcal{IG}(a_\eta, b_\eta)$

$$
\boxed{
\sigma_\eta^2 \mid \alpha_{0:T}, y_{1:T} \sim \mathcal{IG}\!\left(
a_\eta + \frac{T}{2},\;
b_\eta + \frac{1}{2}\sum_{t=1}^{T}(\alpha_t - T\alpha_{t-1})^2
\right)
}
$$

### 3.3 Multivariate noise covariances (Inverse-Wishart)

For vector-valued noise with prior $\Sigma \sim \mathcal{IW}(\nu_0, \Psi_0)$:

$$
\Sigma \mid \alpha_{1:T}, y_{1:T} \sim \mathcal{IW}\!\left(
\nu_0 + T,\;
\Psi_0 + \sum_{t=1}^{T} r_t r_t'
\right)
$$

where $r_t$ is the relevant residual vector ($y_t - Z\alpha_t$ for $H$, or
$\alpha_t - T\alpha_{t-1}$ for $Q$).

### 3.4 Coefficient/loading block with Normal-Inverse-Gamma prior

For a regression-type block where the loading matrix $Z$ (or a subset of parameters) is free,
and the prior is $\text{vec}(Z) \sim \mathcal{N}(\mu_0, \Sigma_0)$:

$$
\text{vec}(Z) \mid \alpha_{1:T}, \sigma_\varepsilon^2, y_{1:T}
\sim \mathcal{N}\!\left(\bar{\mu}, \bar{\Sigma}\right)
$$

with:

$$
\bar{\Sigma}^{-1} = \Sigma_0^{-1} + \frac{1}{\sigma_\varepsilon^2}
\sum_{t=1}^T \alpha_t \alpha_t', \qquad
\bar{\mu} = \bar{\Sigma}\!\left(\Sigma_0^{-1}\mu_0
+ \frac{1}{\sigma_\varepsilon^2} \sum_{t=1}^T \alpha_t y_t'\right)
$$

### 3.5 AR coefficient block (TVP model)

For the [TVP model](../advanced/tvp.md), each coefficient follows a random walk $\beta_{j,t} =
\beta_{j,t-1} + \eta_{j,t}$. The state noise variance $\sigma_{\eta_j}^2$ for each coefficient
$j$ gets an independent Inverse-Gamma update:

$$
\sigma_{\eta_j}^2 \mid \beta_{j,1:T}, y_{1:T}
\sim \mathcal{IG}\!\left(a_j + \frac{T-1}{2},\;
b_j + \frac{1}{2}\sum_{t=2}^{T}(\beta_{j,t} - \beta_{j,t-1})^2\right)
$$

This per-coefficient update is the computational bottleneck of TVP Gibbs sampling:
$O(kT)$ operations for $k$ time-varying coefficients.

---

## 4. Configuring the sampler

### 4.1 Burn-in

The chain needs time to move from the initialisation $\theta^{(0)}$ to the high-probability
region of the posterior. Samples collected during this period are discarded as **burn-in**
(sometimes called *warm-up*).

Rule of thumb:

- Well-identified models, good initialisation: 500–1000 burn-in draws
- Weakly identified or high-dimensional models: 2000–5000 burn-in draws
- Always inspect trace plots to confirm the chain has stabilised

### 4.2 Thinning

Successive Gibbs draws are correlated. **Thinning** keeps every $k$-th draw to reduce
autocorrelation in the stored chain. Note that thinning reduces autocorrelation but does
*not* increase information — collecting more un-thinned draws and accepting the autocorrelation
is usually more efficient unless memory is constrained.

Use `thin=1` (no thinning) by default; only thin if the effective sample size diagnostic shows
$\text{ESS} \ll N_\text{draws}$ after running sufficient iterations.

### 4.3 Number of iterations and chains

The number of post-burn-in draws required depends on the effective sample size (ESS) target.
For stable posterior mean estimates, aim for ESS ≥ 400 per parameter. If your chain has
autocorrelation $\rho$ at lag 1, a rough approximation is:

$$
\text{ESS} \approx \frac{N_\text{draws}}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

Running **multiple chains** from different starting points is essential for:

1. Computing the **R-hat** convergence diagnostic
2. Detecting multimodality or poor mixing that single-chain diagnostics can miss
3. Embarrassing parallelism (chains run independently)

```python
from kalmanbox.bayesian import GibbsSampler
from kalmanbox.structural import BasicStructuralModel

sampler = GibbsSampler(
    model=BasicStructuralModel(),
    priors={
        "sigma2_obs":      InverseGamma(shape=2.0, scale=0.1),
        "sigma2_level":    InverseGamma(shape=2.0, scale=0.05),
        "sigma2_slope":    InverseGamma(shape=2.0, scale=0.01),
        "sigma2_seasonal": InverseGamma(shape=2.0, scale=0.05),
    },
    n_iter=6000,
    burn_in=2000,
    thin=1,            # store all post-burn-in draws
    n_chains=4,        # run 4 chains for R-hat
    n_jobs=-1,         # use all CPU cores for parallel chains
    random_state=42,
)
result = sampler.run(y)
```

---

## 5. Example: Gibbs for the Local Level model

The Local Level model has two scalar variance parameters — the simplest non-trivial case for
demonstrating the Gibbs sampler.

```python
import numpy as np
from kalmanbox.bayesian import GibbsSampler, InverseGamma
from kalmanbox.structural import LocalLevel

# ── Simulate data ──────────────────────────────────────────────────────────
np.random.seed(2024)
T = 300
sigma2_true_level = 0.1    # true state noise variance
sigma2_true_obs   = 0.5    # true observation noise variance

level = np.cumsum(np.random.randn(T) * np.sqrt(sigma2_true_level))
y     = level + np.random.randn(T) * np.sqrt(sigma2_true_obs)

# ── Specify priors ─────────────────────────────────────────────────────────
# Weakly informative: prior mean ≈ variance of first-difference of y
s2 = np.var(np.diff(y)) / 2
priors = {
    "sigma2_obs":   InverseGamma(shape=2.5, scale=s2),
    "sigma2_level": InverseGamma(shape=2.5, scale=s2 / 5),
}

# ── Run Gibbs sampler ──────────────────────────────────────────────────────
model   = LocalLevel(sigma2_obs=1.0, sigma2_level=0.1)
sampler = GibbsSampler(
    model=model, priors=priors,
    n_iter=4000, burn_in=1000, thin=2, n_chains=4, random_state=42,
)
result = sampler.run(y)

# ── Posterior summaries ────────────────────────────────────────────────────
print(result.posterior.summary())
#                  mean    std   2.5%  97.5%  r_hat   ess
# sigma2_obs      0.508  0.046  0.424  0.601  1.001  3811
# sigma2_level    0.103  0.019  0.070  0.145  1.002  3654

# Signal-to-noise ratio (derived quantity)
q_draws = (result.posterior.samples["sigma2_level"]
           / result.posterior.samples["sigma2_obs"])
print(f"SNR mean: {q_draws.mean():.3f}, 95% CI: [{np.quantile(q_draws,0.025):.3f}, "
      f"{np.quantile(q_draws,0.975):.3f}]")
```

---

## 6. Example: Gibbs for the Basic Structural Model (BSM)

The BSM adds slope and seasonal components, each with their own noise variance. The Gibbs
sampler draws each variance independently — a key advantage of conjugate priors.

```python
import numpy as np
from kalmanbox.bayesian import GibbsSampler, InverseGamma
from kalmanbox.structural import BasicStructuralModel

# Monthly airline passenger data (Box-Jenkins)
# y is log-transformed to stabilise variance
import pandas as pd
y = np.log(pd.read_csv("airline.csv")["passengers"].values)

priors = {
    "sigma2_obs":      InverseGamma(shape=3.0, scale=1e-3),
    "sigma2_level":    InverseGamma(shape=3.0, scale=1e-3),
    "sigma2_slope":    InverseGamma(shape=3.0, scale=1e-4),
    "sigma2_seasonal": InverseGamma(shape=3.0, scale=1e-3),
}

model   = BasicStructuralModel(period=12)
sampler = GibbsSampler(
    model=model, priors=priors,
    n_iter=8000, burn_in=2000, thin=2, n_chains=4, random_state=0,
)
result = sampler.run(y)

# Extract posterior trend component
trend_draws  = result.state_components["trend"]     # shape (n_chains, n_draws, T)
trend_mean   = trend_draws.mean(axis=(0, 1))
trend_90_ci  = np.quantile(trend_draws, [0.05, 0.95], axis=(0, 1))

# 24-month forecast with full parameter uncertainty
forecast = result.posterior_predictive(steps=24)
```

---

## 7. GibbsSampler API reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | SSM instance | — | kalmanbox structural model |
| `priors` | dict | — | `{param_name: Prior}` mapping |
| `n_iter` | int | 2000 | Total MCMC draws per chain |
| `burn_in` | int | 500 | Draws discarded before collection |
| `thin` | int | 1 | Keep every $k$-th draw |
| `n_chains` | int | 1 | Number of independent chains |
| `n_jobs` | int | 1 | Parallel workers (`-1` = all cores) |
| `random_state` | int | None | RNG seed for reproducibility |
| `init` | str or dict | `"mle"` | Initialisation strategy |

**Return value:** `GibbsResult` with attributes:

| Attribute | Description |
|-----------|-------------|
| `.posterior` | `PosteriorResult` with `.samples`, `.summary()`, `.quantile()` |
| `.state_trajectory` | State draws, shape `(n_chains, n_draws, T, k)` |
| `.state_components` | Named component draws (trend, seasonal, etc.) |
| `.log_likelihood` | Kalman log-likelihood at each draw |
| `.posterior_predictive(steps)` | Draw from $p(y_{T+1:T+h} \mid y_{1:T})$ |
| `.diagnostics` | `DiagnosticsResult` with R-hat, ESS, Geweke |

---

## Further reading

| Topic | Page |
|-------|------|
| FFBS — state sampling block | [FFBS](ffbs.md) |
| Prior specification | [Priors](priors.md) |
| Convergence diagnostics | [Posterior Diagnostics](posterior-diagnostics.md) |
| EM algorithm (alternative point estimation) | [EM Algorithm](../advanced/em.md) |
| TVP model with Gibbs | [Time-Varying Parameters](../advanced/tvp.md) |
| Identifiability theory | [Identifiability](../../theory/identifiability.md) |
