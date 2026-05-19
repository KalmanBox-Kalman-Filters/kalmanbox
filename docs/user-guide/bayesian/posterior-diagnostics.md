# Posterior Diagnostics

Before drawing any scientific conclusions from a Bayesian analysis, you must verify that:

1. The MCMC chains have **converged** to the target posterior distribution
2. The chains are **mixing** well — exploring the posterior efficiently
3. The **effective sample size** (ESS) is large enough for reliable inference
4. The model **fits the data** — posterior predictive checks

Failing to check these exposes you to making inferences from a biased, non-converged chain
that does not represent the posterior.

!!! warning "Convergence ≠ correctness"
    Convergence diagnostics can only tell you that the chains are *consistent* with each other
    and that the run is long enough. They cannot tell you whether the model is correctly
    specified. Always pair convergence checks with **posterior predictive checks** to assess
    model fit.

---

## 1. Trace plots

A **trace plot** shows the sampled value of a parameter against iteration number. It is
the first diagnostic to inspect — visual pathologies are often immediately obvious.

**What to look for:**

- **Stationarity** — the chain should oscillate around a stable level after burn-in, with no
  persistent upward/downward drift
- **Good mixing** — fast oscillation with no long stretches at one value (slow mixing) or
  "sticking" (proposal rejections in Metropolis steps)
- **Multi-chain agreement** — all chains should visit the same region after burn-in; chains
  that separate indicate multimodality or insufficient exploration

```python
from kalmanbox.bayesian import GibbsSampler, InverseGamma
from kalmanbox.structural import LocalLevel
from kalmanbox.bayesian import diagnostics

model   = LocalLevel(sigma2_obs=1.0, sigma2_level=0.1)
priors  = {
    "sigma2_obs":   InverseGamma(shape=2.5, scale=0.1),
    "sigma2_level": InverseGamma(shape=2.5, scale=0.05),
}
sampler = GibbsSampler(model=model, priors=priors,
                       n_iter=4000, burn_in=1000, n_chains=4, random_state=0)
result  = sampler.run(y)

# Trace plots for all parameters
diagnostics.plot_trace(result)

# Trace plot for a specific parameter with running mean overlay
diagnostics.plot_trace(result, params=["sigma2_obs"], show_running_mean=True)
```

**Interpreting trace plots:**

```
Good mixing:                 Slow mixing / high autocorrelation:
┌──────────────────────┐    ┌──────────────────────┐
│ ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱  │    │ ╭────────────╮       │
│╱  ╲  ╲  ╱  ╲  ╱    │    │╱             ╰──────── │
└──────────────────────┘    └──────────────────────┘

Non-stationarity (no convergence):   Multimodal (bad):
┌──────────────────────┐    ┌──────────────────────┐
│           ──────────  │    │ ─────      ─────     │
│──────────             │    │      ──────     ───── │
└──────────────────────┘    └──────────────────────┘
```

---

## 2. R-hat (Gelman-Rubin diagnostic)

The **potential scale reduction factor** $\hat{R}$ compares the variance *between* chains to
the variance *within* chains. If all chains have converged to the same distribution, the between
and within variances should be equal, giving $\hat{R} \approx 1$.

### 2.1 Classical Gelman-Rubin statistic

For $M$ chains each of length $N$ post-burn-in, with chain $m$ having draws
$\{\theta^{(m,s)}\}_{s=1}^N$:

$$
\bar{\theta}_{m\cdot} = \frac{1}{N}\sum_{s=1}^N \theta^{(m,s)}, \qquad
\bar{\theta}_{\cdot\cdot} = \frac{1}{M}\sum_{m=1}^M \bar{\theta}_{m\cdot}
$$

**Between-chain variance:**

$$
B = \frac{N}{M-1} \sum_{m=1}^M \left(\bar{\theta}_{m\cdot} - \bar{\theta}_{\cdot\cdot}\right)^2
$$

**Within-chain variance:**

$$
W = \frac{1}{M(N-1)} \sum_{m=1}^M \sum_{s=1}^N \left(\theta^{(m,s)} - \bar{\theta}_{m\cdot}\right)^2
$$

**Pooled posterior variance estimate:**

$$
\hat{V} = \frac{N-1}{N}\,W + \frac{1}{N}\,B
$$

**R-hat:**

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

### 2.2 Rank-normalised R-hat (recommended)

The classical $\hat{R}$ can fail to detect non-stationarity in heavy-tailed posteriors.
The **rank-normalised** version (Vehtari et al., 2021) transforms draws to their ranks
before computing $\hat{R}$, providing better detection of non-convergence:

$$
\hat{R}_\text{rank} = \hat{R}\!\left(\text{rank-normalise}(\theta^{(m,s)})\right)
$$

kalmanbox computes rank-normalised $\hat{R}$ by default.

**Interpretation:**

| $\hat{R}$ | Interpretation |
|-----------|---------------|
| $< 1.01$ | Convergence is strongly supported |
| $1.01$–$1.05$ | Possible convergence — check trace plots |
| $1.05$–$1.10$ | Suspect non-convergence — run longer |
| $> 1.10$ | Non-convergence — do not interpret results |

```python
# Compute R-hat for all parameters
rhat = diagnostics.rhat(result)
print(rhat)
#  sigma2_obs      1.0014
#  sigma2_level    1.0021

# Flag parameters with R-hat > 1.01
problematic = {k: v for k, v in rhat.items() if v > 1.01}
if problematic:
    print(f"WARNING: {list(problematic.keys())} show potential non-convergence")
```

---

## 3. Effective Sample Size (ESS)

Successive MCMC draws are **autocorrelated** — they are not independent samples from the
posterior. The **effective sample size** accounts for this autocorrelation:

$$
\text{ESS} = \frac{N\,M}{1 + 2\sum_{k=1}^{\infty} \rho_k}
$$

where $N \cdot M$ is the total number of draws (chains × iterations per chain) and $\rho_k$
is the autocorrelation at lag $k$, estimated from the pooled chain.

Intuitively, $\text{ESS}$ is the number of *independent* draws that would give the same
posterior estimation precision as the $N \cdot M$ autocorrelated draws.

### 3.1 Bulk ESS and tail ESS

| Metric | Measures | Target |
|--------|---------|--------|
| **Bulk ESS** | Efficiency for estimating posterior mean | ESS $\geq 400$ |
| **Tail ESS** | Efficiency for estimating quantiles (2.5%, 97.5%) | ESS $\geq 400$ |

Tail ESS is typically lower than bulk ESS because extreme quantiles are estimated from fewer
effective draws.

```python
ess = diagnostics.effective_sample_size(result)
print(ess)
#  sigma2_obs      bulk=3812, tail=3654
#  sigma2_level    bulk=3421, tail=3211

# Low ESS signals: increase n_iter or thin less aggressively
for param, (ess_bulk, ess_tail) in ess.items():
    if min(ess_bulk, ess_tail) < 400:
        print(f"WARNING: {param} has low ESS (bulk={ess_bulk}, tail={ess_tail}). "
              f"Run longer or reparametrise.")
```

### 3.2 ESS per unit time

When comparing different configurations, compute ESS per wall-clock second:

```python
print(result.ess_per_second())
#  sigma2_obs      bulk=1906 draws/s, tail=1827 draws/s
```

---

## 4. Autocorrelation of chains

The **autocorrelation function (ACF)** of the chain at lag $k$ measures the correlation between
draw $s$ and draw $s+k$. High autocorrelation at many lags indicates slow mixing.

```python
# ACF plots for all parameters
diagnostics.plot_autocorr(result, max_lag=50)

# ACF at specific lags
acf = diagnostics.autocorr(result, lags=range(1, 51))
print(f"Lag-1 ACF (sigma2_obs): {acf['sigma2_obs'][1]:.3f}")
```

**Rules of thumb:**

- Lag-1 ACF $< 0.1$ — excellent mixing
- Lag-1 ACF $0.1$–$0.5$ — acceptable; monitor ESS
- Lag-1 ACF $> 0.5$ — slow mixing; consider reparametrisation or longer thinning

---

## 5. Geweke diagnostic

The **Geweke test** checks stationarity within a single chain by comparing the posterior mean
in the first $f_a$ fraction of the chain to the last $f_b$ fraction (after burn-in). Under
stationarity, the difference should be zero:

$$
z = \frac{\bar\theta_A - \bar\theta_B}{\sqrt{\hat S_A / n_A + \hat S_B / n_B}}
\stackrel{H_0}{\sim} \mathcal{N}(0, 1)
$$

where $\hat S_A$ and $\hat S_B$ are spectral density estimates that account for autocorrelation
within each segment.

A $|z| > 2$ suggests the chain has not converged within the allocated iterations.

```python
geweke = diagnostics.geweke(result, first=0.1, last=0.5)
print(geweke)
#              z_score  p_value  converged
# sigma2_obs    -0.832    0.405       True
# sigma2_level   1.103    0.270       True
```

---

## 6. Posterior distribution plots

### 6.1 Marginal posterior histograms

```python
# Marginal posteriors for all parameters
diagnostics.plot_posterior(result)

# Custom parameter subset with kernel density estimate overlay
diagnostics.plot_posterior(
    result,
    params=["sigma2_obs", "sigma2_level"],
    kind="kde",            # "hist", "kde", or "both"
    point_estimate="mean", # "mean", "median", or "mode"
    ci=0.94,               # credible interval coverage
)
```

### 6.2 Pairplot (joint marginals)

For detecting parameter correlations and multimodality:

```python
diagnostics.plot_pair(result, params=["sigma2_obs", "sigma2_level"])
```

For the Local Level model, the classic **signal-to-noise** correlation appears: as
$\sigma_\varepsilon^2$ increases (more observation noise), the posterior for $\sigma_\eta^2$
shifts upward to maintain the SNR consistent with the data.

---

## 7. Posterior predictive checks

**Posterior predictive checks (PPCs)** test whether data generated from the fitted posterior
$p(y^\text{rep} \mid y_{1:T})$ look similar to the observed data $y_{1:T}$. Systematic
discrepancies reveal model misspecification that convergence diagnostics cannot detect.

### 7.1 In-sample PPC

```python
# Draw from in-sample posterior predictive
y_rep = result.posterior_predictive(n_draws=500, steps=0)  # shape (500, T)

# Fan chart: compare replications to observed
diagnostics.plot_posterior_predictive(
    y_obs=y,
    y_rep=y_rep,
    ci_levels=[0.50, 0.90, 0.95],
    title="Local Level — In-sample Posterior Predictive Check",
)
```

### 7.2 Test statistics

Compute a scalar test statistic $T(y)$ on both observed and replicated data, then check
whether the observed value falls within the posterior predictive distribution:

```python
import numpy as np

# Test statistic: standard deviation (tests variance fit)
T_obs = y.std()
T_rep = np.array([y_rep[s].std() for s in range(y_rep.shape[0])])

# Posterior predictive p-value
p_value = (T_rep >= T_obs).mean()
print(f"PPC p-value (std): {p_value:.3f}")
# Values near 0 or 1 indicate model misspecification

# Additional test statistics
diagnostics.ppc_stats(y, y_rep, stats=["mean", "std", "skewness", "kurtosis",
                                        "acf_lag1", "min", "max"])
```

### 7.3 Common PPC failure modes

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| Replications miss peak seasonal amplitude | Seasonal variance too small | Loosen $\sigma_\gamma^2$ prior |
| Replications over-smooth level breaks | Level noise too restricted | More diffuse $\sigma_\eta^2$ prior |
| Fat tails in $y$ not reproduced | Gaussian observation noise | Consider Student-$t$ observation model |
| Replications are too smooth overall | Missing trend or cycle component | Add slope or cycle component |

---

## 8. Complete diagnostic workflow

```python
from kalmanbox.bayesian import GibbsSampler, InverseGamma, diagnostics
from kalmanbox.structural import BasicStructuralModel
import numpy as np

y = ...  # your time series

model  = BasicStructuralModel(period=12)
priors = {
    "sigma2_obs":      InverseGamma(shape=2.5, scale=y.var() * 0.1),
    "sigma2_level":    InverseGamma(shape=2.5, scale=y.var() * 0.05),
    "sigma2_slope":    InverseGamma(shape=2.5, scale=y.var() * 0.005),
    "sigma2_seasonal": InverseGamma(shape=2.5, scale=y.var() * 0.03),
}

sampler = GibbsSampler(
    model=model, priors=priors,
    n_iter=8000, burn_in=2000, thin=2, n_chains=4, n_jobs=-1, random_state=42,
)
result = sampler.run(y)

# ── Step 1: R-hat ──────────────────────────────────────────────────────────
rhat = diagnostics.rhat(result)
assert all(v < 1.01 for v in rhat.values()), f"Non-convergence: {rhat}"

# ── Step 2: ESS ───────────────────────────────────────────────────────────
ess  = diagnostics.effective_sample_size(result)
for param, (bulk, tail) in ess.items():
    if min(bulk, tail) < 400:
        print(f"Low ESS for {param}: bulk={bulk}, tail={tail}")

# ── Step 3: Trace plots ────────────────────────────────────────────────────
diagnostics.plot_trace(result)

# ── Step 4: Autocorrelation ───────────────────────────────────────────────
diagnostics.plot_autocorr(result, max_lag=40)

# ── Step 5: Posterior marginals ───────────────────────────────────────────
diagnostics.plot_posterior(result, kind="kde", ci=0.94)

# ── Step 6: Geweke ────────────────────────────────────────────────────────
geweke = diagnostics.geweke(result)
print(geweke[["z_score", "converged"]])

# ── Step 7: Posterior predictive check ────────────────────────────────────
y_rep = result.posterior_predictive(n_draws=500, steps=0)
diagnostics.plot_posterior_predictive(y, y_rep)
diagnostics.ppc_stats(y, y_rep)
```

---

## 9. Common failure modes and remedies

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| Two variance chains anti-correlate ($\hat{R}$ inflated) | Identifiability: signal-to-noise ratio | Tighter or hierarchical prior; reparametrise as $q = \sigma_\eta^2/\sigma_\varepsilon^2$ |
| One chain stuck near zero variance | Mode at boundary; $\alpha \leq 1$ | Increase $\alpha$ in IG prior or use Half-Cauchy |
| ESS $< 100$ for slope variance | Slope barely identified from data | Run much longer; pre-initialise from MLE |
| Posterior predictive misses seasonality | Model lacks seasonal component | Switch to BSM or UCM |
| All $\hat{R}$ values high (> 1.1) | Fundamentally multimodal posterior | Increase burn-in, try multiple initialisations, check identifiability |
| Geweke $z$-score outside $[-2, 2]$ | Chain not stationary after burn-in | Extend burn-in; initialise with smoother output |
| ACF drops slowly (lag-1 > 0.7) | High parameter correlation | Reparametrise; use block updates |

---

## 10. Diagnostics API reference

| Function | Returns | Description |
|----------|---------|-------------|
| `diagnostics.rhat(result)` | `dict[str, float]` | Rank-normalised $\hat{R}$ per parameter |
| `diagnostics.effective_sample_size(result)` | `dict[str, tuple]` | Bulk and tail ESS per parameter |
| `diagnostics.geweke(result, first, last)` | `DataFrame` | Geweke z-scores and p-values |
| `diagnostics.autocorr(result, lags)` | `dict[str, ndarray]` | ACF values at specified lags |
| `diagnostics.plot_trace(result, params, ...)` | `Figure` | Trace plots with optional running mean |
| `diagnostics.plot_posterior(result, params, ...)` | `Figure` | Marginal posterior histograms / KDE |
| `diagnostics.plot_autocorr(result, max_lag)` | `Figure` | ACF plots per parameter |
| `diagnostics.plot_pair(result, params)` | `Figure` | Pairwise joint posteriors |
| `diagnostics.plot_posterior_predictive(y, y_rep)` | `Figure` | Fan chart of PPC replications |
| `diagnostics.ppc_stats(y, y_rep, stats)` | `DataFrame` | PPC p-values for test statistics |

---

## Further reading

| Topic | Page |
|-------|------|
| Gibbs sampler configuration (burn-in, thinning) | [Gibbs Sampling](gibbs.md) |
| Prior specification — avoiding boundary issues | [Priors](priors.md) |
| Identifiability theory for SSMs | [Identifiability](../../theory/identifiability.md) |
| General convergence diagnostics | [Diagnostics: Convergence](../../diagnostics/convergence.md) |
| API reference — diagnostics module | [api/bayesian](../../api/bayesian.md) |
