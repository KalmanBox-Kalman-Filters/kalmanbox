# FAQ — Estimation

## What optimizer does MLE use by default?

The default method is **L-BFGS-B** from `scipy.optimize.minimize`.
L-BFGS-B uses gradient information (computed by finite differences
unless you supply analytical gradients) and supports box constraints,
which are used to enforce positivity of variance parameters through a
log-transform of the parameterisation.

To use a different optimizer:

```python
results = model.fit(method="mle", optimizer="Nelder-Mead")
# or pass scipy options directly
results = model.fit(method="mle",
                    optimizer_kwargs={"method": "SLSQP",
                                      "options": {"ftol": 1e-9}})
```

For DFMs and other high-dimensional models, `method="em"` is usually
faster and more robust than direct MLE.

## How many random restarts does the optimizer try?

By default, **1** — the optimizer starts from the internally computed
initial values (heuristic variances based on the series). You can
request multiple random restarts to reduce sensitivity to the starting
point:

```python
results = model.fit(n_restarts=10, random_state=42)
```

With `n_restarts=10` the optimizer is run 10 times from randomly
perturbed starting points; the run with the highest log-likelihood
is returned. The number of restarts is a trade-off between robustness
and computation time.

## How do I supply my own starting values?

Pass a dictionary of parameter names to their initial values via
`start_params`:

```python
results = model.fit(
    start_params={
        "sigma2_eta": 500.0,    # level innovation
        "sigma2_eps": 15000.0,  # measurement noise
    }
)
```

Parameter names match those printed by `model.param_names`. Unknown
keys are silently ignored so that you can supply a partial set.

## When should I use EM instead of direct MLE?

Use **EM** when:

- The model has many parameters (DFM, high-order UCM) — EM updates
  are closed-form per block and converge reliably even from poor
  starting points.
- The parameter space has many local maxima — EM takes monotone steps
  (the likelihood never decreases), reducing the risk of converging
  to a saddle point.

Use **direct MLE** (L-BFGS-B) when:

- The model is small (Local Level, BSM) — MLE converges in fewer
  iterations and the overhead of repeated Kalman passes in EM is
  not justified.
- You need the Hessian at convergence for standard error estimates
  — L-BFGS-B approximates the Hessian; the exact Hessian can be
  requested via `results.hessian()`.

EM standard errors require a Louis (1982) correction or a separate
numerical Hessian evaluation, which `results.standard_errors()` does
automatically.

## How does Gibbs sampling work for state-space models?

kalmanbox implements the **Carter & Kohn (1994)** algorithm:

1. **Forward-Filtering Backward-Sampling (FFBS)**: run the Kalman
   filter forward, then sample the full state trajectory
   $\{\alpha_t\}_{t=1}^T$ backward from the conditional Gaussian
   distributions.
2. **Parameter draw**: given the sampled states, draw the system
   matrices ($Q$, $H$, $\Lambda$, …) from their conjugate posterior
   distributions.
3. **Repeat** for `n_draws` iterations, discarding the first
   `n_burn` as warm-up.

```python
from kalmanbox.estimation import GibbsSampler

sampler = GibbsSampler(model, n_draws=2000, n_burn=500,
                       random_state=1)
posterior = sampler.sample()
print(posterior.summary())   # mean, std, HDI for each parameter
```

See [Gibbs / FFBS](../user-guide/bayesian/gibbs.md) for the full API.

## What are sensible priors for variance parameters?

For variance parameters ($\sigma^2$), the conjugate prior is the
**Inverse-Gamma**:

$$
\sigma^2 \sim \mathrm{IG}(a_0,\, b_0)
$$

- A **weakly informative** choice: $a_0 = 3$, $b_0 = s^2 \times
  (a_0 - 1)$ where $s^2$ is a rough prior guess at the variance.
  This concentrates prior mass around $s^2$ while leaving the tails
  heavy enough to accommodate surprises.
- For **near-zero variances** (e.g., a slope that you expect to be
  nearly deterministic): $a_0 = 1.5$, $b_0 = 0.01$ — this puts
  mass near zero without forcing it to zero (use a fixed parameter
  for that).

kalmanbox pre-defined priors live in
[`kalmanbox.estimation.priors`][kalmanbox.estimation.priors]:

```python
from kalmanbox.estimation.priors import InverseGamma

prior = InverseGamma(a=3.0, b=500.0)
```

See [Priors](../user-guide/bayesian/priors.md) for a full catalogue.

## How do I diagnose convergence of the MCMC chain?

kalmanbox computes standard diagnostics from
[`arviz`](https://python.arviz.org/):

```python
import arviz as az

idata = posterior.to_inference_data()   # ArviZ InferenceData
print(az.summary(idata))               # R-hat, ESS, HDI
az.plot_trace(idata)                   # traceplots
az.plot_autocorr(idata)                # within-chain autocorrelation
```

Key indicators:

| Metric      | Target          | Concern             |
|-------------|:---------------:|---------------------|
| $\hat R$    | $< 1.01$        | $> 1.05$ — chains not mixed |
| Bulk ESS    | $> 400$         | $< 100$ — high autocorrelation |
| Tail ESS    | $> 400$         | $< 100$ — tails poorly sampled |

If ESS is low, increase `n_draws` or thin the chain with the
`thin` argument to `GibbsSampler`.

See [Posterior diagnostics](../user-guide/bayesian/posterior-diagnostics.md).
