# Identifiability

A state-space model is **identified** if every parameter vector $\theta$
produces a distinct distribution of observations. Many natural-looking
specifications fail this test. Recognising and resolving identifiability
issues up front saves hours of "MLE doesn't converge".

## Common non-identification patterns

### 1. Indistinguishable variances

In a Local Level model

$$
\mu_{t+1} = \mu_t + \eta_t,\quad y_t = \mu_t + \varepsilon_t
$$

only the **ratio** $q = \sigma_\eta^2 / \sigma_\varepsilon^2$ enters the
likelihood through the steady-state Kalman gain. Both variances are
formally identified through the diffuse initialisation, but the
likelihood is **flat in their sum**, so MLE is slow.

**Remedy**: reparametrise as $(q, \sigma_\varepsilon^2)$, optimise on
log-scale, or use a Bayesian prior.

### 2. Sign / rotation symmetries in factor models

In a DFM, $\Lambda F_t = (\Lambda Q)(Q' F_t)$ for any orthogonal $Q$.
Without constraints, the factors are identified only up to rotation.

**Remedy**: lower-triangular $\Lambda$ with positive diagonal, or
$\operatorname{Var}(F_t) = I_k$.

### 3. Cycle / seasonal collision

A stochastic cycle with frequency $\lambda$ near $2\pi/s$ (the
fundamental seasonal frequency) competes with a seasonal component of
period $s$ for the same variation in $y_t$. Likelihood develops two
modes.

**Remedy**: drop one component, or place a prior pinning $\lambda$ away
from $2\pi/s$.

### 4. Slope vs. cycle in trending series

If $y_t$ has a near-deterministic trend, both `LocalLinearTrend`'s slope
and a UCM's cycle can absorb it. They are **observationally equivalent**
in finite samples.

**Remedy**: pick one based on theory; do not let MLE choose between
them.

## Diagnosing identifiability

```python
from kalmanbox.diagnostics import information_matrix_rank

results = model.fit()
rank = information_matrix_rank(results)
print(f"Rank: {rank}/{len(results.params)}")
# Less than full rank → identification failure
```

A rank-deficient observed information matrix is a smoking gun.

## In MCMC

Bayesian estimation papers over weak identification with informative
priors — but if the posterior is multi-modal or wandering across
non-identified manifolds, you should fix the model, not the sampler.

## Related

- [Bayesian estimation: priors](../user-guide/bayesian/priors.md)
- [Diagnostics: convergence](../diagnostics/convergence.md)
- Harvey & Koopman (2003) — *Computing observed information in the EM
  algorithm for state-space models.*
