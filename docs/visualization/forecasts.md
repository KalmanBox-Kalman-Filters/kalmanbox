# Forecast fan charts

A fan chart visualises the **growing uncertainty** of multi-step-ahead
forecasts via nested confidence bands.

## Quick plot

```python
from kalmanbox.visualization import plot_forecast

fig = plot_forecast(
    results,
    steps=24,
    levels=(0.50, 0.80, 0.95),
)
```

This produces:

- The historical series in black.
- The point forecast in colour.
- Three nested bands: 50%, 80%, 95% prediction intervals.

## Annotations

```python
fig = plot_forecast(results, steps=24, levels=(0.5, 0.95),
                    title="Brazilian IPCA — 24-month forecast",
                    target_value=4.5,                # central bank target
                    target_label="BCB target")
```

## Component-level forecasts

For BSM / UCM, fan chart each component separately:

```python
plot_forecast(results, steps=24, components=["level", "seasonal"])
```

This is often more informative than the headline forecast — it shows
which component is driving the uncertainty.

## Bayesian forecast distributions

When `results` comes from a Bayesian fit, the forecast is **posterior
predictive** by default — the bands reflect parameter and state
uncertainty jointly:

```python
ppf = bayes_result.posterior_predictive(steps=24)
plot_forecast(ppf, levels=(0.5, 0.95))
```

## Related

- [User guide: forecasting](../user-guide/kalman/forecasting.md)
- [Bayesian estimation](../user-guide/bayesian/index.md)
