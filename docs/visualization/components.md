# Component decomposition

For structural models (BSM, UCM, LLT), each smoothed component carries
substantive interpretation. `plot_components` shows them stacked.

## Quick plot

```python
from kalmanbox.visualization import plot_components

results = BSM(y, seasonal_periods=12).fit()
fig = plot_components(results)
```

Default panels (BSM):

1. **Level** $\mu_t$ — the underlying mean.
2. **Slope** $\beta_t$ — the rate of change.
3. **Seasonal** $\gamma_t$ — the periodic component.
4. **Irregular** $\varepsilon_t$ — the residual noise.

Each panel shows the smoothed estimate with its 95% band.

## Reconstructing the fit

The reconstructed in-sample fit is the sum of the **observation-equation
contributions** of all included components (level + seasonal in BSM,
plus any cycle/exog).

```python
fitted = results.fitted_values()         # smoothed mean of Z * alpha
residuals = y - fitted
```

## Decomposing forecast variance

Forecast uncertainty is the sum of:

- Future state-disturbance variances $R_t Q_t R_t'$.
- Initial state variance propagated through $T_t$.
- Observation noise $H_t$ at the forecast horizon.

`plot_components(results, kind="forecast")` shows this break-down.

## Related

- [User guide: BSM](../user-guide/structural/bsm.md)
- [User guide: UCM](../user-guide/structural/ucm.md)
- [Forecast fan charts](forecasts.md)
