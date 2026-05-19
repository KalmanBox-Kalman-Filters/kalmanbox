# Forecasting

Forecasting in a state-space framework is just the **filter recursion run
forward beyond the sample**, with no observations to update against.

## Mechanics

For $h = 1, \ldots, H$:

$$
\begin{aligned}
a_{n+h|n} &= T_{n+h-1}\,a_{n+h-1|n} + c_{n+h-1} \\
P_{n+h|n} &= T_{n+h-1}\,P_{n+h-1|n}\,T_{n+h-1}' + R_{n+h-1}\,Q_{n+h-1}\,R_{n+h-1}' \\
\hat{y}_{n+h} &= Z_{n+h}\,a_{n+h|n} + d_{n+h} \\
\operatorname{Var}(y_{n+h}) &= Z_{n+h}\,P_{n+h|n}\,Z_{n+h}' + H_{n+h}
\end{aligned}
$$

The forecast variance grows monotonically as $h$ increases (no further
information arrives).

## API

```python
results = LocalLevel(y).fit()

f = results.forecast(steps=12, alpha=0.05)

f["mean"]      # point forecast              shape (12,)
f["lower_95"]  # lower 95 % prediction band  shape (12,)
f["upper_95"]  # upper 95 % prediction band  shape (12,)
f["state"]     # state forecasts             shape (12, k)
f["P"]         # state covariances           shape (12, k, k)
```

## Decomposing forecast uncertainty

For BSM / UCM decompositions you can break the variance into
component-level contributions (level, trend, seasonal). See
[Component decomposition](../../visualization/components.md).

## Forecasting with exogenous regressors

If the model includes time-varying $Z_t$ or $d_t$ (e.g. regression
in state-space form), supply the **future** regressors:

```python
f = results.forecast(steps=12, exog=future_X)
```

## Related

- [Visualization: forecast fan charts](../../visualization/forecasts.md)
- [API: results](../../api/core.md)
