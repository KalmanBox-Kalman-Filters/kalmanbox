# Stability tests

State-space parameters are usually assumed **constant** over the
sample. Stability tests check whether that assumption holds.

## Recursive residuals

Standardised innovations $\tilde v_t$ are themselves the recursive
residuals; under stability they are i.i.d. $\mathcal{N}(0,1)$.

## CUSUM

$$
\text{CUSUM}_t = \sum_{s=1}^{t} \tilde v_s
$$

Plot $\text{CUSUM}_t$ against $t$. Under stability it stays inside
straight-line bounds derived from the Brownian-motion limit; crossings
indicate a **mean break**.

## CUSUM-of-squares

$$
\text{CUSUMSQ}_t = \frac{\sum_{s=1}^{t}\tilde v_s^2}{\sum_{s=1}^{n}\tilde v_s^2}
$$

Under stability, $\text{CUSUMSQ}_t \approx t/n$. Departures indicate a
**variance break**.

## Filter divergence

Watch for $F_t$ growing without bound, or for the filter producing
implausible state estimates after a regime change. Both are red flags
that the parameters are not constant.

## Time-varying alternatives

If stability tests fail and a structural reason is plausible:

- **TVP**: switch from a static regression to
  [`TimeVaryingParameters`](../user-guide/advanced/tvp.md).
- **Markov switching**: not in `kalmanbox` core; see
  [`forecastbox`](../getting-started/ecosystem.md).
- **Stochastic volatility**: typically handled by
  [`particlefilterbox`](../getting-started/ecosystem.md).

## Usage

```python
from kalmanbox.diagnostics import cusum, cusum_squared

c    = cusum(results)
csq  = cusum_squared(results)
```

## Related

- [Residual analysis](residuals.md)
- [TVP](../user-guide/advanced/tvp.md)
