# Time-Varying Parameter Regression

Regression with coefficients that evolve over time.

## Usage

```python
from kalmanbox.models.tvp import TVPRegression

model = TVPRegression(y, exog=X)
results = model.fit()
betas = results.smoothed_state  # time-varying coefficients
```

## When to use

- Regression relationships that change over time
- Structural breaks in coefficient values
- Time-varying beta in finance (e.g., CAPM)
