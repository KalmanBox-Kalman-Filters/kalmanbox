# Dynamic Factor Model

Extract common factors from multivariate time series.

## Usage

```python
from kalmanbox.models.dfm import DynamicFactor

model = DynamicFactor(Y, k_factors=2, factor_order=1)
results = model.fit()
factors = results.smoothed_state[:, :2]
```

## When to use

- Multiple correlated time series
- Want to extract common driving factors
- Dimensionality reduction for macro data
