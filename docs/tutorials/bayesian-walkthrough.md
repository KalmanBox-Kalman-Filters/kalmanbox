# Tutorial — Bayesian estimation walkthrough

A full MCMC analysis of the Nile dataset using `BayesianSSM`. We fit a
Local Level model under conjugate Inverse-Gamma priors, run a
[Gibbs sampler](../user-guide/bayesian/gibbs.md) with
[FFBS](../user-guide/bayesian/ffbs.md) state draws, and compare the
posterior mean to the MLE estimate.

## 1. Load data

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset

nile = load_dataset("nile")
y    = nile["volume"].to_numpy(dtype=float)
```

## 2. Prior specification

The Local Level model has two variance parameters:

$$
\sigma_\eta^2 \sim \mathcal{IG}(a_\eta,\, b_\eta), \qquad
\sigma_\varepsilon^2 \sim \mathcal{IG}(a_\varepsilon,\, b_\varepsilon)
$$

An $\mathcal{IG}(a, b)$ prior has mean $b/(a-1)$ and is conjugate for a
Gaussian likelihood with known mean. We choose weakly informative values:

```python
from kalmanbox.estimation import BayesianSSM, InverseGamma

priors = {
    "sigma2_eta":  InverseGamma(a=3.0, b=500.0),   # mean ~ 250
    "sigma2_eps":  InverseGamma(a=3.0, b=5000.0),  # mean ~ 2500
}
```

!!! tip "Prior sensitivity"

    Run the sampler under two or three different $(a, b)$ pairs and
    overlay the posteriors to confirm the data are informative enough to
    swamp the prior. For the Nile series — 100 observations — they are.

## 3. Gibbs sampler setup

The sampler alternates two blocks each sweep:

1. **FFBS** — draw the full state trajectory
   $\alpha_{1:n} \mid \sigma^2, y$.
2. **Conjugate variance update** — draw each $\sigma^2 \mid \alpha, y$:

$$
\sigma^2 \mid \alpha, y \;\sim\;
\mathcal{IG}\!\left(
  a + \frac{n}{2},\;
  b + \frac{1}{2}\sum_{t=1}^{n} e_t^2
\right)
$$

where $e_t$ is the relevant squared residual (level innovation for
$\sigma_\eta^2$, measurement residual for $\sigma_\varepsilon^2$).

```python
sampler = BayesianSSM(
    model_cls=LocalLevel,
    y=y,
    prior=priors,
)

result = sampler.fit(
    n_iter=6000,     # total iterations
    n_burn=1000,     # discarded burn-in
    thin=1,
    n_chains=4,
    seed=0,
)
```

## 4. Trace plots and posterior diagnostics

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 6))

for chain_idx in range(4):
    draws = result.posterior.samples[chain_idx]  # shape (n_draws, n_params)
    axes[0, 0].plot(draws[:, 0], lw=0.6, alpha=0.7)
    axes[0, 1].plot(draws[:, 1], lw=0.6, alpha=0.7)

axes[0, 0].set_title(r"Trace — $\sigma_\eta^2$")
axes[0, 1].set_title(r"Trace — $\sigma_\varepsilon^2$")

# Marginal posteriors
all_draws = result.posterior.samples.reshape(-1, 2)
axes[1, 0].hist(all_draws[:, 0], bins=50, density=True, color="C0")
axes[1, 0].set_xlabel(r"$\sigma_\eta^2$")
axes[1, 1].hist(all_draws[:, 1], bins=50, density=True, color="C1")
axes[1, 1].set_xlabel(r"$\sigma_\varepsilon^2$")

plt.tight_layout()
```

Inspect convergence statistics:

```python
print(result.summary())
```

Expected output (values approximate):

```
                  mean      sd    2.5%   97.5%   R-hat   ESS
sigma2_eta      1461.3   387.4   873.1  2353.6    1.00  7823
sigma2_eps     14820.1  2104.8  11080.4 19320.5   1.00  7651
```

$\hat{R} \approx 1.00$ and ESS > 1000 per parameter confirm
convergence.

## 5. Posterior mean vs MLE

```python
# MLE reference
mle_result = LocalLevel(y).fit()
print("MLE estimates:")
print(f"  sigma2_eta = {mle_result.params['sigma2_eta']:.1f}")
print(f"  sigma2_eps = {mle_result.params['sigma2_eps']:.1f}")

# Posterior means
print("\nBayesian posterior mean:")
print(f"  sigma2_eta = {result.posterior.mean['sigma2_eta']:.1f}")
print(f"  sigma2_eps = {result.posterior.mean['sigma2_eps']:.1f}")
```

The posterior mean is pulled slightly toward the prior, but with 100
observations the difference is small (< 2 %).

## 6. Posterior predictive checks

Draw replicated datasets $y^{\text{rep}} \sim p(y \mid \theta^{(s)})$
from a random subset of posterior draws:

```python
ppc = result.posterior_predictive(steps=0)  # in-sample replication

fig, ax = plt.subplots(figsize=(9, 4))
for rep in ppc.samples[:200]:               # 200 random draws
    ax.plot(rep, color="C0", alpha=0.04, lw=0.8)
ax.plot(y, "k.", ms=4, label="observed", zorder=5)
ax.set_title("Posterior predictive check — Nile")
ax.legend()
```

The observed series should sit well within the predictive envelope.
Systematic deviations (e.g. the 1898 structural break appearing as an
outlier period) suggest model misspecification — consider a
Markov-switching extension.

!!! note "Forecasting"

    Pass `steps=20` to `posterior_predictive` to obtain posterior
    predictive forecasts with full parameter uncertainty propagated —
    wider than the MLE plug-in bands.

## What we learned

- Conjugate Inverse-Gamma priors enable **exact** Gibbs updates; no
  Metropolis–Hastings tuning required.
- FFBS produces exact draws from the state posterior, making each Gibbs
  iteration $O(n)$.
- Posterior means and MLE estimates agree closely for 100 observations;
  the Bayesian approach additionally quantifies **parameter uncertainty**
  in predictions.

## Next

- [User guide: Gibbs sampling](../user-guide/bayesian/gibbs.md)
- [User guide: FFBS](../user-guide/bayesian/ffbs.md)
- [Priors and hyperparameters](../user-guide/bayesian/priors.md)
- [Posterior diagnostics](../user-guide/bayesian/posterior-diagnostics.md)
