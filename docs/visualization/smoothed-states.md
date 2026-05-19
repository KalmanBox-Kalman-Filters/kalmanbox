# Plotting smoothed states

The RTS smoother produces $a_{t|n}$ and $P_{t|n}$ — the **two-sided**
estimate of the latent state using the entire sample. Smoothed plots
are the standard output for retrospective analysis.

## Quick plot

```python
from kalmanbox.visualization import plot_smoothed

fig = plot_smoothed(results, components="all", alpha=0.05)
```

## Difference from filtered

| Quantity                | Filtered ($a_{t|t}$) | Smoothed ($a_{t|n}$)        |
|-------------------------|----------------------|-----------------------------|
| Conditioning set        | $y_{1:t}$            | $y_{1:n}$                   |
| Variance at boundaries  | Larger at $t = 1$    | Larger at $t = 1$ and $t = n$ |
| Use case                | Real-time            | Historical                   |

The two estimates **converge** in the middle of the sample where
forwards and backwards information overlap.

## Multivariate states

For multi-dimensional $\alpha_t$ (e.g. BSM), `plot_smoothed` lays out
one subplot per component using the names supplied by the model
(`level`, `slope`, `seasonal`, etc.).

## Decomposition view

For BSM / UCM fits, see [Component decomposition](components.md) —
which sums components back into $\hat y_t$.

## Related

- [Filtered states](filtered-states.md)
- [User guide: RTS smoother](../user-guide/kalman/rts-smoother.md)
