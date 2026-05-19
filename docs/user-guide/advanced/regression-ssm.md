# Regression in state-space form

A static linear regression $y_t = x_t' \beta + \varepsilon_t$ becomes
trivial in state space: the state is constant and equal to $\beta$.

## Specification

$$
\begin{aligned}
\beta_{t+1} &= \beta_t,                & \\
y_t         &= x_t' \beta_t + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, \sigma^2).
\end{aligned}
$$

In matrix form: $T = I_k$, $R = 0$, $Q = 0$, $Z_t = x_t'$, $H = \sigma^2$,
with **diffuse** initialisation on $\beta_0$.

## Why bother?

Casting OLS as a state-space model is mostly a stepping stone:

1. **Sequential / online OLS** — the filter updates $\hat\beta_t$ one
   observation at a time without re-inverting the Gram matrix.
2. **Mixed regression / state models** — combine fixed coefficients
   with dynamic components in one unified framework.
3. **Reference baseline** for [TVP](tvp.md) — set $Q = 0$ and you
   recover this model.

## Usage

```python
from kalmanbox import RegressionSSM

model = RegressionSSM(y, exog=X)
results = model.fit()
print(results.params)   # beta-hat, sigma2
```

The point estimates equal OLS up to numerical precision.

## API

::: kalmanbox.models.regression_ssm.RegressionSSM
    options:
      heading_level: 3

## Related

- [TVP](tvp.md) — drift the coefficients over time.
- [UCM](../structural/ucm.md) — adds regressors to a structural model.
