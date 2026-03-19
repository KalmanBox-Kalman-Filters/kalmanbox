# Unobserved Components Model (UCM)

The UCM provides a flexible framework for specifying structural time series
models with configurable components.

## Usage

```python
from kalmanbox.models.ucm import UCM

model = UCM(y, level=True, trend='damped', seasonal='trigonometric',
            seasonal_period=12)
results = model.fit()
```

## Components

| Component | Options |
|:----------|:--------|
| Level | `True` / `False` |
| Trend | `None`, `'deterministic'`, `'stochastic'`, `'damped'` |
| Seasonal | `None`, `'dummy'`, `'trigonometric'` |
| Cycle | `None`, `'stochastic'` |

## Example

See the [BSM guide](../bsm/overview.md) for a related example. UCM generalizes
BSM by allowing individual component configuration.
