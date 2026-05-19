# FAQ — Numerical Issues

## My covariance matrix became negative definite — what happened?

Filtered and predicted covariance matrices $P_{t|t}$ and $P_{t+1|t}$
must be symmetric positive semi-definite (SPSD) by construction. In
practice, rounding errors in the Joseph-form or naive KF update can
cause small negative eigenvalues after thousands of iterations.

Causes:

- Using the standard (non-Joseph) update $P_{t|t} = (I - K_t Z_t) P_{t|t-1}$
  instead of the numerically stable Joseph form
  $(I - K_t Z_t) P_{t|t-1} (I - K_t Z_t)' + K_t H K_t'$.
- Very small observation noise $H$ relative to the predicted covariance,
  causing the Kalman gain to approach 1.
- Accumulated floating-point errors over very long series ($T > 10^5$).

**Solutions**:

1. Use the **Square-Root filter** (`SquareRootFilter`) which maintains
   the Cholesky factor $S_t$ instead of $P_t$; definiteness is
   guaranteed.
2. Add a small symmetry correction after each update:
   `P = 0.5 * (P + P.T)`.
3. Periodically project $P$ onto the SPSD cone:
   `P = nearestSPD(P)` (available in `kalmanbox.utils.linalg`).

## When should I use the Square-Root filter?

Use the **Square-Root Kalman filter** when:

- State dimension $m$ is large ($m \gtrsim 20$) and the filter runs
  for many steps — rounding error accumulation is most severe here.
- Observation noise $H$ is very small or zero (exact observations).
- The condition number of $P_t$ is expected to be large (poorly
  observable states with very informative observations).

```python
from kalmanbox.filters import SquareRootFilter

sr = SquareRootFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)
result = sr.filter(y)
```

The Square-Root filter is slightly slower per step (Cholesky
factorisation at each update) but eliminates indefiniteness issues.
See [Square-Root filter](../user-guide/filters/square-root.md).

## What is the diffuse initialisation and when do I need it?

For **non-stationary** models (Local Level, Local Linear Trend, ARIMA
with $d > 0$), the unconditional variance of the initial state is
infinite. Setting $P_1 = \kappa I$ with large $\kappa$ (the "big
$\kappa$" approach) approximates this, but the choice of $\kappa$
affects the first few likelihood contributions.

The **exact diffuse initialisation** (Koopman 1997; Durbin & Koopman
2012 §5) splits the initial covariance into a finite part and an
infinite-variance diffuse component, and handles the first $d^\star$
diffuse periods analytically.

```python
from kalmanbox import LocalLevel

model = LocalLevel(y)
results = model.fit(initialization="diffuse")    # default
# or use big-kappa for compatibility with older code:
results = model.fit(initialization="approximate_diffuse", kappa=1e6)
```

kalmanbox uses exact diffuse initialisation by default for all
non-stationary models. See
[Diffuse initialisation](../user-guide/kalman/diffuse-initialization.md).

## My log-likelihood is −∞ — what's wrong?

A log-likelihood of $-\infty$ usually means the innovation covariance
matrix $F_t = Z_t P_{t|t-1} Z_t' + H$ became singular or indefinite
and its determinant evaluated to zero (or negative):

$$
\ell_t = -\tfrac{p}{2}\log(2\pi) - \tfrac{1}{2}\log|F_t| -
          \tfrac{1}{2} v_t' F_t^{-1} v_t
$$

Common causes:

| Cause                                   | Fix                                      |
|-----------------------------------------|------------------------------------------|
| $H = 0$ (exact observations) with $P_{t|t-1} \approx 0$ | Use diffuse init; or Information filter |
| $Q = 0$ with non-identifiable states    | Fix or remove unidentified states        |
| Duplicate observations (identical rows in multivariate $y$) | Check data for duplicates |
| Parameter step during MLE put $\sigma^2 < 0$ | Check param transform; use `bounds` |

Enable verbose output for diagnosis:

```python
results = model.fit(disp=True)   # prints likelihood at each iteration
```

## How does kalmanbox handle near-singular observation covariances?

When $F_t$ is near-singular, `np.linalg.solve(F_t, v_t)` is
numerically unstable. kalmanbox uses a **nugget** strategy by default:
a small diagonal term $\epsilon I$ is added to $F_t$ before inversion,
with $\epsilon = $ `machine_eps * trace(F_t)`. A `NumericalWarning` is
emitted if the nugget exceeds `1e-6` times the trace.

For **exact** zero-noise observations, switch to the
[Information filter](../user-guide/filters/information.md), which
operates on the precision matrix $F_t^{-1}$ directly and avoids
inversion of singular matrices.

## What is the difference between the Information filter and standard KF?

The **Information filter** (Bierman 1977) propagates the inverse
covariance (information matrix) $\Omega_t = P_t^{-1}$ and the
information vector $\xi_t = \Omega_t a_t$:

$$
\Omega_{t|t} = \Omega_{t|t-1} + Z_t' H^{-1} Z_t, \qquad
\xi_{t|t}    = \xi_{t|t-1}   + Z_t' H^{-1} y_t
$$

Advantages over standard KF:

- **Handles exact observations** ($H = 0$) without singularity.
- **Parallel update** across multiple sensors (information matrices
  are additive).
- Numerically efficient when $p \gg m$ (many observation dimensions,
  few states).

Disadvantages: requires $H$ to be invertible; filtered means
$a_{t|t} = \Omega_{t|t}^{-1} \xi_{t|t}$ need an additional solve.

```python
from kalmanbox.filters import InformationFilter

inf_f = InformationFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)
result = inf_f.filter(y)
```

## How can I speed up filtering for very long series?

1. **Enable Numba** (`pip install numba`): the core loop is JIT-compiled
   on first call; subsequent runs are 5–20× faster.
2. **Reduce state dimension**: simplify the model if possible — every
   extra state adds $O(m^2)$ work per step.
3. **Chunked filtering**: split the series and carry the terminal state
   forward (see [Memory profile](../benchmarks/memory.md#chunked-filtering)).
4. **Skip the smoother**: if you only need filtered (not smoothed)
   states, pass `store_history=False` to avoid allocating $O(T m^2)$
   arrays.
5. **Steady-state filter**: for time-invariant models the filter
   covariance $P_{t|t}$ converges to a fixed point (the Riccati
   equation solution). Solve for it once and apply the steady-state
   gain $K_\infty$ for the remainder of the series.

```python
from kalmanbox.filters import SteadyStateKalmanFilter

ss_kf = SteadyStateKalmanFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)
result = ss_kf.filter(y)   # solves Riccati once, then O(Tm) per step
```

See also: [Performance benchmarks](../benchmarks/performance.md).
