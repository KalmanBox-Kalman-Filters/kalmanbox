# Diagnostic plots

A four-panel diagnostic figure makes residual problems visible at a
glance.

## Quick plot

```python
from kalmanbox.visualization import plot_residual_diagnostics

fig = plot_residual_diagnostics(results)
```

Panels:

| Panel              | What to look for                                              |
|--------------------|---------------------------------------------------------------|
| Standardised residuals | Random scatter around zero, constant variance.            |
| ACF                | All bars within $\pm 1.96/\sqrt n$ for stable lags.           |
| QQ-plot            | Points along the 45° line; tails are the tell.                |
| Histogram + N(0,1) | Bell-shape, no skew, mean ~ 0, sd ~ 1.                        |

## CUSUM plot

```python
from kalmanbox.visualization import plot_cusum

plot_cusum(results)            # CUSUM and CUSUMSQ side by side
```

Lines crossing the bounds signal a structural break.

## Posterior predictive plot (Bayesian)

```python
from kalmanbox.visualization import plot_ppc

plot_ppc(bayes_result, n_samples=200)
```

200 simulated $\tilde y$ are over-plotted in light grey; the observed
$y$ should look like one of them. A clear systematic deviation
indicates model misspecification.

## Related

- [Diagnostics: residuals](../diagnostics/residuals.md)
- [Diagnostics: stability](../diagnostics/stability.md)
