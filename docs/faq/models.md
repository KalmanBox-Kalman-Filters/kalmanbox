# FAQ — Models

## What is the difference between LocalLevel and LocalLinearTrend?

**LocalLevel** (random-walk-plus-noise) has a single state — the
stochastic level $\mu_t$. It is appropriate when the series drifts
slowly but has no systematic slope.

**LocalLinearTrend** adds a second state — the slope $\nu_t$ — so the
level is driven by a drifting trend:

$$
\mu_{t+1} = \mu_t + \nu_t + \eta_t^{\mu}, \quad
\nu_{t+1} = \nu_t + \eta_t^{\nu}
$$

Use LocalLinearTrend when the series shows a persistent upward or
downward movement that itself changes over time. If the slope variance
$\sigma_\zeta^2$ is estimated near zero, the trend is essentially
linear and LocalLevel is usually sufficient.

See: [Local Level](../user-guide/structural/local-level.md) |
[Local Linear Trend](../user-guide/structural/local-linear-trend.md)

## When should I use BSM vs UCM?

The **Basic Structural Model (BSM)** is a fixed architecture:
level + slope + dummy seasonal + irregular. It is quick to specify and
well-suited to standard economic / demographic series with a single
regular seasonal period.

The **Unobserved Components Model (UCM)** is a modular builder: you
combine level, slope, trigonometric seasonal blocks, stochastic cycles,
regression terms, and an irregular in any combination. Use UCM when:

- You need **multiple seasonal periods** (e.g., daily data with weekly
  + annual seasonality).
- You want a **cycle** component distinct from the trend.
- You need to **fix** certain variances to zero (e.g., deterministic
  seasonality).

See: [BSM](../user-guide/structural/bsm.md) |
[UCM](../user-guide/structural/ucm.md)

## How do I add a custom component to a UCM?

Subclass `UCMComponent` and register it with the builder:

```python
from kalmanbox.models.ucm import UCMComponent, UnobservedComponents

class StochasticCycle(UCMComponent):
    def __init__(self, period: float, damping: float = 0.9) -> None:
        self.period = period
        self.damping = damping

    def state_matrices(self) -> dict:
        import numpy as np
        lam = 2 * np.pi / self.period
        T_block = self.damping * np.array([
            [np.cos(lam), np.sin(lam)],
            [-np.sin(lam), np.cos(lam)],
        ])
        return {"T": T_block, "n_states": 2}

model = UnobservedComponents(y)
model.add_component(StochasticCycle(period=40.0, damping=0.95))
results = model.fit()
```

The framework automatically concatenates state transition and
covariance matrices from all registered components.

## What is the maximum number of factors in a DFM?

There is no hard-coded limit. Practical constraints are:

1. **Identifiability**: $k$ factors require at least $2k + 1$ observed
   series (the Bai–Ng 2002 rule of thumb). With $p = 5$ series, $k \le 2$
   is the practical maximum.
2. **Memory**: the state vector has dimension $k \cdot r$ (where $r$ is
   the factor VAR order), so cost scales as $O(T k^2 r^2)$.
3. **Information criteria**: use `results.select_k_factors()` to
   compare AIC/BIC across $k = 1, 2, \ldots, k_{\max}$.

For high-dimensional panels ($p \gg k$), Bai–Ng (2002) information
criteria (`ICp1`, `ICp2`) are implemented in
[`kalmanbox.diagnostics.ic`][kalmanbox.diagnostics.ic].

## Can I fix some parameters during estimation?

Yes. Pass a `fixed_params` dictionary mapping parameter names to their
fixed values:

```python
model = BasicStructuralModel(y, seasonal_periods=12)
# Fix the slope variance to zero (deterministic slope)
results = model.fit(fixed_params={"sigma2_zeta": 0.0})
```

Fixed parameters are held constant during optimisation; only the
remaining free parameters are updated by MLE or EM.

## How do I specify observation-level missing data?

Pass `np.nan` in the observation array. kalmanbox detects NaN entries
and runs the **missing-data Kalman filter** (Durbin & Koopman §4.10):
the update step is skipped for missing periods, and the state is
propagated by the prediction equations only.

```python
import numpy as np
from kalmanbox import LocalLevel

y = nile["volume"].copy().astype(float)
y.iloc[10:15] = np.nan       # create a gap
model = LocalLevel(y)
results = model.fit()        # missing periods handled automatically
```

Multivariate models support **partially observed** measurement vectors:
only the non-NaN elements enter the observation equation at each $t$.
See [Missing data](../user-guide/kalman/missing-data.md).

## How does ARIMA_SSM compare to statsmodels ARIMA?

`ARIMA_SSM` casts an ARIMA(p, d, q) model in state-space form and
estimates it via the Kalman filter likelihood. It is equivalent to
`statsmodels.tsa.statespace.SARIMAX` in specification but:

- shares the kalmanbox parameter API and result objects,
- integrates with `forecastbox` cross-validation out of the box,
- supports missing observations in the same way as all other
  kalmanbox models.

For pure ARIMA work without state-space features, `statsmodels.ARIMA`
(the `statsmodels.tsa.arima.model.ARIMA` class) is equally capable and
slightly faster due to a more specialised implementation.

See: [ARIMA-SSM](../user-guide/structural/arima-ssm.md)
