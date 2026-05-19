# Custom State-Space Models

When the pre-built models do not match your problem, build a state-space
model from scratch with [`CustomStateSpace`][kalmanbox.models.custom.CustomStateSpace].

## Pattern

```python
from kalmanbox import CustomStateSpace, StateSpaceRepresentation
import numpy as np


class MyModel(CustomStateSpace):
    """Two-state local linear trend with constrained slope variance."""

    param_names = ("sigma2_eta", "sigma2_zeta", "sigma2_eps")

    def _build_ssm(self, params: dict) -> StateSpaceRepresentation:
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        Z = np.array([[1.0, 0.0]])
        R = np.eye(2)
        Q = np.diag([params["sigma2_eta"], params["sigma2_zeta"]])
        H = np.array([[params["sigma2_eps"]]])
        return StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

    def _initial_params(self):
        var_y = float(np.nanvar(self.endog))
        return {"sigma2_eta": 0.1 * var_y,
                "sigma2_zeta": 0.01 * var_y,
                "sigma2_eps": 0.5 * var_y}
```

Then:

```python
model = MyModel(y)
results = model.fit()
```

## What you get for free

By inheriting from `CustomStateSpace` you automatically get:

- Kalman filter & RTS smoother on your representation.
- MLE / EM / Bayesian estimation.
- Forecasting, missing data, diffuse initialisation.
- Diagnostics, visualisation, reports.
- The same `StateSpaceResults` API as the built-in models.

## Time-varying matrices

If your $T_t, Z_t, \ldots$ depend on $t$ (or on exogenous regressors),
return arrays with a leading time dimension:

```python
T = np.empty((n, k, k))
for t in range(n):
    T[t] = build_T(t, exog=X[t])
return StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
```

## Constraints

Override `_transform_params` and `_untransform_params` to enforce
positivity, stationarity, etc., during MLE optimisation.

## API

::: kalmanbox.models.custom.CustomStateSpace
    options:
      heading_level: 3

## Related

- [`StateSpaceRepresentation`][kalmanbox.core.representation.StateSpaceRepresentation]
- [Theory: identifiability](../../theory/identifiability.md)
- [Contributing: add a new model](../../contributing/setup.md)
