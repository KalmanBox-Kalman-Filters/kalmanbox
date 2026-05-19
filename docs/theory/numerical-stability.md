# Numerical stability

The Kalman recursion is mathematically simple but **numerically subtle**.
`kalmanbox` ships several safeguards by default; this page explains them
so you can choose alternatives when needed.

## Joseph form

The naive update $P_{t|t} = (I - K_t Z_t) P_{t|t-1}$ is exact in real
arithmetic but not symmetric in floating point. The Joseph form

$$
P_{t|t} = (I - K_t Z_t)\,P_{t|t-1}\,(I - K_t Z_t)' + K_t H_t K_t'
$$

is **symmetric by construction** and remains positive-definite for any
gain $K_t$, not just the optimal one. `KalmanFilter` uses it by default.

## Square-root forms

When $P_t$ becomes ill-conditioned (large dynamic range across
eigenvalues, deterministic states, long sample sizes), Joseph form may
not be enough. The
[`SquareRootKalmanFilter`](../user-guide/filters/square-root.md)
propagates the Cholesky factor $S_t$ via QR decompositions, ensuring
$P_t = S_t S_t'$ stays PSD throughout.

## Symmetrisation

Even with Joseph form, `kalmanbox` symmetrises $P_t \leftarrow (P_t +
P_t')/2$ after each update. Cheap, harmless when the matrix is already
symmetric, life-saving when accumulated drift is non-trivial.

## Innovation covariance inversion

$F_t^{-1}$ is solved via Cholesky factorisation, not explicit
inversion. When $F_t$ is rank-deficient (e.g. perfectly collinear
observations), the solver raises a clear error rather than producing
garbage.

## Diffuse handling

Replacing $\kappa I$ for diffuse states with **exact** diffuse
recursions removes the conditioning blow-up that the
"large-variance approximation" produces. See
[Diffuse initialisation](../user-guide/kalman/diffuse-initialization.md).

## When to escalate

| Symptom                                    | Fix                                          |
|--------------------------------------------|----------------------------------------------|
| `LinAlgError: not positive definite`       | Switch to Square-Root filter.                |
| Log-likelihood = $-\infty$ at start        | Use diffuse init for non-stationary states.  |
| Filter diverges after many steps           | Symmetrise + check $F_t$ condition number.   |
| Estimate / std error are NaN at MLE optimum| Re-run from multiple inits; check identifiability. |

## Practical tips

!!! numerical "Reproducibility"

    For reproducible MLE results across BLAS implementations, scale
    your data so that $\operatorname{Var}(y_t) \approx 1$. Filter
    matrices then have moderate dynamic range and float64 is plenty.

!!! tip "Profile before square-root"

    The square-root filter has a higher constant. If a quick run with
    Joseph form succeeds and parameter estimates are stable, you don't
    need it.

## Related

- [User guide: Square-Root filter](../user-guide/filters/square-root.md)
- [User guide: Information filter](../user-guide/filters/information.md)
- [Kalman filter derivation](kalman-filter-derivation.md)
