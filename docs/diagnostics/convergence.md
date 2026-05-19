# Convergence diagnostics

Whether you are running MLE or MCMC, convergence checks are
non-negotiable.

## MLE

```python
results = model.fit(method="lbfgs", maxiter=500)
print(results.optimizer.success)
print(results.optimizer.message)
print(results.optimizer.n_iter, results.optimizer.n_func_evals)
```

Quality checks:

- **Gradient norm** at solution: $\|\nabla \ell(\hat\theta)\|$ should be
  small (`results.optimizer.grad_norm`).
- **Observed information**: positive-definite Hessian at $\hat\theta$
  (`results.observed_information`).
- **Multi-start**: re-run from several initial values to check whether
  the optimum is global. `kalmanbox` exposes `n_starts` for this.

## EM

The auxiliary log-likelihood $\mathcal{Q}$ should be monotonically
non-decreasing. `EMEstimator` records its trace in `results.em_history`
— inspect the curve to confirm.

EM converges to the **nearest** local maximum, so multi-start is even
more important than for L-BFGS.

## MCMC

For Bayesian estimation use:

- **Trace plots** — `result.plot_trace()`.
- **$\hat R$** — Gelman-Rubin statistic across chains.
- **ESS** — effective sample size per parameter.
- **Posterior predictive check** — does sampled $\tilde y$ resemble
  $y$?

```python
result.summary()    # includes r_hat, ess_bulk, ess_tail
```

Aim for $\hat R < 1.01$ and ESS $> 400$.

## When chains disagree

Common causes:

| Cause                        | Remedy                                                  |
|------------------------------|---------------------------------------------------------|
| Multi-modal posterior        | Stronger prior, or tighter parameterisation.            |
| Non-identifiability          | Reparametrise; see [Identifiability](../theory/identifiability.md). |
| Initial values too dispersed | Start chains in a posterior-credible region.            |
| Numerical underflow in $F_t$ | Switch filter to square-root form.                       |

## Related

- [Bayesian: posterior diagnostics](../user-guide/bayesian/posterior-diagnostics.md)
- [Theory: numerical stability](../theory/numerical-stability.md)
