# Custom Models

Build custom state-space models by specifying system matrices directly.

## Usage

```python
from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.models.base import StateSpaceModel

# Define your own state-space matrices
rep = StateSpaceRepresentation(
    k_states=2, k_obs=1,
    T=..., Z=..., R=..., H=..., Q=...,
)

# Create and fit model
model = StateSpaceModel(y, representation=rep)
results = model.fit()
```

## When to use

- Non-standard model specifications
- Research applications
- Models not covered by built-in classes
