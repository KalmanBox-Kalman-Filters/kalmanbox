# Priors and Hyperparameters

A **prior distribution** encodes beliefs about a parameter before seeing the data. In Bayesian
state-space models, priors serve two complementary roles:

1. **Regularisation** — they prevent overfitting in high-dimensional models (TVP, DFM) and keep
   variances away from zero or infinity in weakly identified models.
2. **Domain knowledge** — economic or physical theory often constrains parameters (e.g., AR
   coefficients must be stationary; variance ratios are typically small).

kalmanbox provides a set of **conjugate and semi-conjugate prior classes** that integrate with the
Gibbs sampler's analytical conditional draws. Non-conjugate priors are supported via
Metropolis-within-Gibbs steps.

!!! tip "Prior sensitivity"
    Always re-fit with an alternative prior specification (e.g., an order of magnitude tighter
    and looser). If posterior conclusions change substantially, document the prior dependence
    explicitly. This is especially important for variance parameters in short time series.

---

## 1. Priors for variance parameters

Variance parameters $\sigma^2 > 0$ appear in the observation noise ($H$) and state noise ($Q$)
of every state-space model. The canonical conjugate prior for a scalar variance is the
**Inverse-Gamma** distribution.

### 1.1 Inverse-Gamma prior

The Inverse-Gamma distribution $\mathcal{IG}(\alpha, \beta)$ has density:

$$
p(\sigma^2) = \frac{\beta^\alpha}{\Gamma(\alpha)}\,
(\sigma^2)^{-(\alpha+1)}\,
\exp\!\left(-\frac{\beta}{\sigma^2}\right), \quad \sigma^2 > 0
$$

| Moment | Formula |
|--------|---------|
| Mean | $\beta / (\alpha - 1)$, for $\alpha > 1$ |
| Mode | $\beta / (\alpha + 1)$ |
| Variance | $\beta^2 / [(\alpha-1)^2(\alpha-2)]$, for $\alpha > 2$ |

**Conjugate Gibbs update** — see [Gibbs Sampling](gibbs.md) for the derivation. With $n$
residuals $e_1, \ldots, e_n$:

$$
\sigma^2 \mid \text{data} \sim \mathcal{IG}\!\left(
\alpha + \frac{n}{2},\;
\beta + \frac{1}{2}\sum_{i=1}^n e_i^2
\right)
$$

```python
from kalmanbox.bayesian import InverseGamma

# Weakly informative: mean ≈ sample variance, broad
sigma2_y = y.var()
prior_obs = InverseGamma(shape=2.0, scale=sigma2_y * 0.1)

# Mean-centred at known scale s²
s2 = 0.5
prior_level = InverseGamma(shape=3.0, scale=s2 * 2.0)  # mean = s²

# Inspect moments
print(f"Prior mean: {prior_obs.mean():.4f}")
print(f"Prior mode: {prior_obs.mode():.4f}")
print(f"Prior std:  {prior_obs.std():.4f}")

# Prior predictive draws
samples = prior_obs.rvs(size=10_000)
```

**Elicitation guide:**

| Situation | Recommended $\alpha$ | Recommended $\beta$ |
|-----------|---------------------|---------------------|
| Nearly no information | 1.01 | $\varepsilon$ small | Improper limit; use with care |
| Weakly informative | 2.0–2.5 | $0.1 \cdot \hat\sigma^2_y$ | Broad, mean ≈ $0.1\hat\sigma^2$ |
| Mean-centred at $s^2$ | 3.0 | $2 s^2$ | Mean = $s^2$, std = $s^2$ |
| Tightly centred at $s^2$ | 10.0 | $9 s^2$ | Mean = $s^2$, std ≈ $0.35 s^2$ |

### 1.2 Half-Cauchy prior

The **Half-Cauchy** prior $\sigma \sim \text{Half-Cauchy}(0, \tau)$ is a popular weakly
informative choice for standard deviations (not variances). It places heavy probability mass
near zero while having thick tails that permit large values — ideal when you suspect most
variances are small but allow for occasional large ones.

$$
p(\sigma) = \frac{2}{\pi\,\tau}\,\frac{1}{1 + (\sigma/\tau)^2}, \quad \sigma > 0
$$

The Half-Cauchy is **not conjugate** to the Gaussian likelihood; kalmanbox handles it via
a parameter expansion (data augmentation) technique that restores conjugacy:

$$
\sigma^2 \mid \xi \sim \mathcal{IG}(1/2,\; 1/\xi), \qquad
\xi \sim \mathcal{IG}(1/2,\; 1/\tau^2)
$$

This two-level hierarchy is equivalent to $\sigma \sim \text{Half-Cauchy}(0, \tau)$ but admits
exact Gibbs draws.

```python
from kalmanbox.bayesian import HalfCauchy

prior_sigma = HalfCauchy(scale=1.0)   # scale = τ; tail parameter
```

**When to use Half-Cauchy over Inverse-Gamma:**

- Variance could plausibly be zero (nearly static component), but you want to avoid hard
  boundary priors
- TVP models where most coefficients are time-invariant but a few vary substantially
- Hierarchical models where global-local shrinkage is desired

---

## 2. Priors for coefficient parameters

Regression coefficients, factor loadings, and observation equation parameters are naturally
modelled with **Normal priors**, which are conjugate to Gaussian likelihoods.

### 2.1 Normal prior

$$
\beta \sim \mathcal{N}(\mu_0, \Sigma_0)
$$

**Conjugate Gibbs update** (from the regression-with-prior formula):

$$
\beta \mid \text{data} \sim \mathcal{N}(\bar\mu, \bar\Sigma)
$$

$$
\bar\Sigma^{-1} = \Sigma_0^{-1} + \frac{X'X}{\sigma^2}, \qquad
\bar\mu = \bar\Sigma\!\left(\Sigma_0^{-1}\mu_0 + \frac{X'y}{\sigma^2}\right)
$$

where $X$ is the design matrix formed from the state trajectory.

```python
from kalmanbox.bayesian import NormalPrior
import numpy as np

k = 5  # number of coefficients

# Zero-mean, diffuse
prior_beta = NormalPrior(mean=np.zeros(k), cov=100 * np.eye(k))

# Informative: loadings near 1 with small variance
prior_loadings = NormalPrior(mean=np.ones(k), cov=0.25 * np.eye(k))
```

**Elicitation guide for $\Sigma_0$:**

| Model | Recommended prior |
|-------|------------------|
| DFM loadings (unknown sign/scale) | $\Sigma_0 = 10 \cdot I$ (diffuse) |
| TVP initial state $\beta_0$ | $\Sigma_0 = 100 \cdot I$ (very diffuse) |
| Observation matrix $Z$ (structural model) | $\Sigma_0 = I$ with $\mu_0 = 0$ |
| AR coefficients in UCM cycle | Normal truncated to $[-1, 1]$ (see §2.3) |

### 2.2 Laplace (double-exponential) prior — sparsity

When many loadings or coefficients are expected to be exactly or nearly zero, the **Laplace
prior** promotes sparsity similarly to the LASSO:

$$
p(\beta_j) = \frac{\lambda}{2}\,\exp(-\lambda|\beta_j|)
$$

The Laplace is implemented via the Normal scale-mixture representation:

$$
\beta_j \mid \tau_j^2 \sim \mathcal{N}(0, \tau_j^2), \qquad
\tau_j^2 \sim \mathcal{Exp}(\lambda^2 / 2)
$$

This restores conjugacy and allows standard Gibbs steps for each $\beta_j$ and $\tau_j^2$.

```python
from kalmanbox.bayesian import LaplacePrior

prior_sparse = LaplacePrior(scale=1.0)   # λ = 1.0; smaller = more shrinkage
```

Useful for DFM loadings when the factor structure is sparse (many zero loadings).

---

## 3. Priors for AR parameters

### 3.1 Truncated Normal for stationary AR coefficients

AR(1) coefficients $\phi \in (-1, 1)$ must be inside the unit circle for stationarity.
The **truncated Normal** restricts the support to $(-1, 1)$:

$$
\phi \sim \mathcal{N}(\mu_0, \sigma_0^2)\,\mathbf{1}(|\phi| < 1)
$$

The normalising constant is $\Phi\!\left(\frac{1-\mu_0}{\sigma_0}\right)
- \Phi\!\left(\frac{-1-\mu_0}{\sigma_0}\right)$, where $\Phi$ is the standard normal CDF.

```python
from kalmanbox.bayesian import TruncatedNormal

prior_phi = TruncatedNormal(
    mean=0.8,    # prior belief: moderate persistence
    std=0.2,     # uncertainty in that belief
    lower=-1.0,
    upper=1.0,
)
```

For the [UCM cycle component](../structural/cycle.md) with AR(2) dynamics, use:

```python
from kalmanbox.bayesian import StationaryAR

prior_ar2 = StationaryAR(p=2, mean=np.array([0.7, -0.2]), cov=0.1 * np.eye(2))
```

`StationaryAR` automatically enforces the stationarity region in $\mathbb{R}^p$ (a triangular
region for $p = 2$) via rejection sampling.

### 3.2 Minnesota prior for VAR/DFM transition matrices

For multi-equation VAR-type transition matrices, the **Minnesota prior** places more shrinkage
on higher lags and cross-variable effects:

$$
T_{ij}^\ell \sim \mathcal{N}\!\left(
\bar{T}_{ij}^\ell,\; \frac{\lambda_1}{\ell^2} \cdot \frac{\sigma_i^2}{\sigma_j^2}
\right)
$$

where $\bar{T}_{ij}^\ell = \mathbf{1}(i=j,\, \ell=1)$ (diagonal persistence prior for lag 1),
$\lambda_1$ controls overall tightness, and $\sigma_i^2$ scales by each equation's OLS residual
variance.

```python
from kalmanbox.bayesian import MinnesotaPrior

prior_var = MinnesotaPrior(
    n_vars=5,
    n_lags=2,
    lambda1=0.1,   # overall tightness; smaller = more shrinkage
    lambda2=0.5,   # cross-variable shrinkage relative to own-variable
)
```

---

## 4. Informative vs weakly informative priors

### 4.1 Weakly informative priors

A weakly informative prior is **not flat**, but it is broad enough that the data dominate the
posterior in most practical situations. The purpose is to prevent pathological behaviour
(boundary estimates, non-convergence) without strongly influencing the posterior.

**Recommended defaults for structural models:**

```python
from kalmanbox.bayesian import InverseGamma, NormalPrior

sigma2_y = y.var()

weakly_informative = {
    # Variances: IG(2, 10% of data variance) — broad, positive mean
    "sigma2_obs":      InverseGamma(shape=2.0, scale=0.1 * sigma2_y),
    "sigma2_level":    InverseGamma(shape=2.0, scale=0.05 * sigma2_y),
    "sigma2_slope":    InverseGamma(shape=2.0, scale=0.01 * sigma2_y),
    "sigma2_seasonal": InverseGamma(shape=2.0, scale=0.05 * sigma2_y),
}
```

### 4.2 Informative priors

Informative priors encode domain knowledge. For example:

- **Signal-to-noise ratio for the Local Level model** — in many macroeconomic series,
  the variance ratio $q = \sigma_\eta^2 / \sigma_\varepsilon^2$ is known to be small (0.01–0.1).
  Encode this via the ratio directly:

  ```python
  from kalmanbox.bayesian import InverseGamma

  # Signal-to-noise prior: σ_η² / σ_ε² ~ Beta reparametrised
  # Equivalently: place IG(5, 0.02) on σ_η² and IG(5, 0.5) on σ_ε²
  informative = {
      "sigma2_obs":   InverseGamma(shape=5.0, scale=0.5),
      "sigma2_level": InverseGamma(shape=5.0, scale=0.02),
  }
  ```

- **BSM seasonal variance** — seasonal patterns in many economic series are highly stable.
  A tight prior on $\sigma_\gamma^2$ near zero reflects this:

  ```python
  bsm_informative = {
      "sigma2_seasonal": InverseGamma(shape=10.0, scale=1e-4),
  }
  ```

---

## 5. Prior elicitation for structural models

### 5.1 Local Level model

| Parameter | Meaning | Starting point |
|-----------|---------|---------------|
| $\sigma_\varepsilon^2$ | Observation noise | $\approx \text{Var}(y)$ |
| $\sigma_\eta^2$ | Level noise | $\approx \text{Var}(\Delta y) / 2$ |

```python
sigma2_obs   = InverseGamma(shape=2.5, scale=y.var() * 0.5)
sigma2_level = InverseGamma(shape=2.5, scale=np.var(np.diff(y)) / 4)
```

### 5.2 Basic Structural Model (BSM)

The BSM has four variance parameters. A natural ordering of prior informativeness:

```
σ²_obs ≥ σ²_level ≥ σ²_seasonal ≥ σ²_slope
```

For monthly economic series:

```python
priors_bsm = {
    "sigma2_obs":      InverseGamma(shape=3.0, scale=y.var() * 0.3),
    "sigma2_level":    InverseGamma(shape=3.0, scale=y.var() * 0.05),
    "sigma2_slope":    InverseGamma(shape=3.0, scale=y.var() * 0.002),
    "sigma2_seasonal": InverseGamma(shape=3.0, scale=y.var() * 0.02),
}
```

### 5.3 Time-Varying Parameters (TVP) model

TVP models require priors on $\beta_0$ (initial coefficients) and $\sigma_{\eta_j}^2$
(random walk variance per coefficient). Tight priors on $\sigma_{\eta_j}^2$ implement
shrinkage toward constant coefficients.

```python
import numpy as np
from kalmanbox.bayesian import NormalPrior, InverseGamma, HalfCauchy

k = 4   # number of time-varying coefficients

priors_tvp = {
    # Initial state: diffuse Normal
    "beta0": NormalPrior(mean=np.zeros(k), cov=100 * np.eye(k)),

    # Per-coefficient random walk variance: Half-Cauchy for global-local shrinkage
    "sigma2_eta": [HalfCauchy(scale=0.1) for _ in range(k)],

    # Observation noise
    "sigma2_obs": InverseGamma(shape=3.0, scale=y.var() * 0.1),
}
```

---

## 6. Prior class API reference

| Class | Parameters | Distribution | Conjugate to |
|-------|-----------|-------------|-------------|
| `InverseGamma(shape, scale)` | $\alpha > 0$, $\beta > 0$ | $\mathcal{IG}(\alpha, \beta)$ | Gaussian variance |
| `NormalPrior(mean, cov)` | $\mu_0$, $\Sigma_0$ | $\mathcal{N}(\mu_0, \Sigma_0)$ | Gaussian coefficient |
| `HalfCauchy(scale)` | $\tau > 0$ | $\text{Half-Cauchy}(0, \tau)$ | Via augmentation |
| `LaplacePrior(scale)` | $\lambda > 0$ | $\text{Laplace}(0, 1/\lambda)$ | Via scale-mixture |
| `TruncatedNormal(mean, std, lower, upper)` | $\mu_0, \sigma_0, a, b$ | $\mathcal{N}(\mu_0, \sigma_0^2)\,\mathbf{1}(a < \cdot < b)$ | Rejection sampling |
| `StationaryAR(p, mean, cov)` | $p, \mu_0, \Sigma_0$ | Normal restricted to stationarity region | Rejection sampling |
| `InverseWishart(df, scale)` | $\nu > k-1$, $\Psi \succ 0$ | $\mathcal{IW}(\nu, \Psi)$ | Gaussian covariance matrix |
| `MinnesotaPrior(n_vars, n_lags, lambda1, lambda2)` | — | Normal with Minnesota structure | Gaussian VAR coefficients |

All prior classes share a common interface:

```python
prior = InverseGamma(shape=2.0, scale=1.0)

prior.mean()              # prior mean
prior.mode()              # prior mode
prior.std()               # prior standard deviation
prior.var()               # prior variance
prior.rvs(size=1000)      # draw samples
prior.logpdf(x)           # log density at x
prior.plot()              # visualise the prior density
```

---

## Further reading

| Topic | Page |
|-------|------|
| Gibbs sampling — uses these priors | [Gibbs Sampling](gibbs.md) |
| FFBS — state sampling conditioned on parameters | [FFBS](ffbs.md) |
| Posterior diagnostics | [Posterior Diagnostics](posterior-diagnostics.md) |
| TVP model full example | [Time-Varying Parameters](../advanced/tvp.md) |
| BSM structural decomposition | [Basic Structural Model](../structural/bsm.md) |
| API reference — prior classes | [api/bayesian](../../api/bayesian.md) |
