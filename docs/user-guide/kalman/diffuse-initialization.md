# Diffuse initialisation

When the state vector contains **non-stationary** components — random
walks, integrated processes, deterministic levels and trends — there is
no natural prior $a_0, P_0$. The textbook trick of "set $P_0 = \kappa I$
with $\kappa$ very large" is numerically unstable.

`kalmanbox` provides **exact diffuse initialisation** following Koopman
& Durbin (2003).

## The idea

Decompose the initial covariance:

$$
P_0 = P_*  + \kappa P_\infty, \qquad \kappa \to \infty
$$

where $P_\infty$ projects onto the diffuse (non-stationary) subspace and
$P_*$ is the proper prior on stationary directions. The recursion is
augmented with auxiliary quantities $a_t^{(\infty)}$, $P_t^{(\infty)}$,
$F_t^{(\infty)}$ that handle the limit analytically. After a small number
of "initialisation steps" (rank of $P_\infty$ at most), the diffuse
quantities collapse and the standard Kalman recursion takes over.

## API

```python
from kalmanbox.estimation import DiffuseInitialization
from kalmanbox.filters import KalmanFilter

init = DiffuseInitialization.from_representation(ssr)
kf = KalmanFilter(ssr, initialization=init)
out = kf.run(y)

out.diffuse_steps    # how many initialisation steps were used
out.loglike          # diffuse log-likelihood (correct large-sample limit)
```

The fitted models in `kalmanbox.models` automatically configure diffuse
initialisation for non-stationary components — you rarely need to set it
manually.

## When you must use it

| Component                | Stationary? | Diffuse?    |
|--------------------------|:-----------:|:-----------:|
| Local Level $\mu_t$      | ❌          | ✅           |
| Local Linear Trend slope | ❌          | ✅           |
| Stationary AR($p$)       | ✅          | ❌           |
| Cycle (damped, $\rho<1$) | ✅          | ❌           |
| Seasonal dummies         | ❌          | ✅           |

## Related

- [Theory: likelihood computation](../../theory/likelihood.md)
- [API: estimation.diffuse](../../api/estimation.md)
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods.* §5.
