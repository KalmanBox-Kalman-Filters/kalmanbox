# Ensemble Kalman Filter (EnKF)

The EnKF approximates the Kalman recursion with an **ensemble** of
$N$ Monte Carlo state samples. The covariance $P_t$ is never formed
explicitly — it is replaced by the empirical covariance of the
ensemble.

## When to use

- Very high-dimensional state spaces (climate, geophysics, fluid
  dynamics) where storing $P_t \in \mathbb{R}^{k\times k}$ is
  prohibitive.
- Models where evaluating $f$ is expensive but easy to do in parallel.
- Mildly nonlinear dynamics that would otherwise need an EKF / UKF.

## Algorithm sketch

For an ensemble $\{\alpha_t^{(i)}\}_{i=1}^N$:

1. **Forecast**: propagate each member through $f$ with independent
   noise draws.
2. **Compute** ensemble mean $\bar\alpha_{t|t-1}$ and anomalies
   $A_{t|t-1}^{(i)} = \alpha_{t|t-1}^{(i)} - \bar\alpha_{t|t-1}$.
3. **Approximate** $P_{t|t-1} \approx \frac{1}{N-1} A A'$ implicitly.
4. **Stochastic update**: draw perturbed observations $y_t + e_t^{(i)}$
   and update each ensemble member with the empirical Kalman gain.

## Usage

```python
from kalmanbox.filters import EnsembleKalmanFilter, EnKFModel
import numpy as np


class MyEnKFModel(EnKFModel):
    def f(self, alpha, t, noise):
        return alpha + 0.1 * np.sin(alpha) + noise

    def h(self, alpha, t):
        return alpha[..., :1]    # observe first component


enkf = EnsembleKalmanFilter(MyEnKFModel(Q=Q, H=H), n_members=200)
out = enkf.run(y, alpha0_samples=initial_ensemble)
```

## Caveats

!!! numerical "Spurious correlations"

    With small $N$ relative to $k$ the empirical covariance has many
    spurious off-diagonal correlations. Mitigate with **localisation**
    (zero out entries beyond a distance threshold) or **inflation**
    (rescale anomalies to combat covariance collapse).

## API

::: kalmanbox.filters.enkf.EnsembleKalmanFilter
    options:
      heading_level: 3
