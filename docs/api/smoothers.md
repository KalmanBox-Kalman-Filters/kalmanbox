# Smoothers API

Smoothers compute $a_{t|n}$ and $P_{t|n}$ — the state mean and
covariance conditioned on the **full** observation sequence
$y_{1:n}$ — by running a backward pass after the forward Kalman filter.

All smoothers in `kalmanbox` accept a `FilterResult` returned by
`KalmanFilter.run()` and a `StateSpaceRepresentation`.

!!! info "Which smoother to use?"

    | Smoother | Use case |
    |----------|----------|
    | `RTSSmoother` | Default choice for linear Gaussian models. Exact, $O(n)$. |
    | `FixedIntervalSmoother` | Equivalent to RTS but expressed in the de Jong / information-filter form; numerically preferable when $P_{t\|t}$ is near-singular. |
    | `FixedLagSmoother` | Online smoothing over a rolling window of $L$ steps; lower latency than the full smoother. |
    | `DisturbanceSmoother` | Computes smoothed disturbances $\hat\eta_t$ and $\hat\varepsilon_t$ and their covariances; required for disturbance-based outlier detection and signal extraction. |

For the theoretical derivation of the RTS recursion see
[Theory: RTS Smoother Derivation](../theory/rts-derivation.md).

## RTSSmoother

The Rauch–Tung–Striebel smoother runs a single backward pass after the
Kalman filter using the gain

$$
J_t = P_{t|t}\, T'\, P_{t+1|t}^{-1}
$$

to compute

$$
a_{t|n} = a_{t|t} + J_t\,(a_{t+1|n} - a_{t+1|t}), \qquad
P_{t|n}  = P_{t|t}  + J_t\,(P_{t+1|n} - P_{t+1|t})\,J_t'.
$$

::: kalmanbox.smoothers.rts.RTSSmoother

## FixedIntervalSmoother

::: kalmanbox.smoothers.fixed_interval.FixedIntervalSmoother

## FixedLagSmoother

The fixed-lag smoother maintains a buffer of the last $L$ filter
densities and updates the smoothed estimate at lag $L$ as each new
observation arrives. It approximates the full smoother with bounded
memory and constant per-step cost.

::: kalmanbox.smoothers.fixed_lag.FixedLagSmoother

## DisturbanceSmoother

The disturbance smoother recovers the smoothed state and observation
disturbances

$$
\hat\eta_t = Q_t\, R_t'\, r_t, \qquad
\hat\varepsilon_t = H_t\, Z_t'\, r_t - H_t\, K_t'\, L_t' r_{t+1\cdots n}
$$

where $r_t$ is the smoothing residual vector from the de Jong recursion.
These disturbances are essential for the
[signal-extraction interpretation](../user-guide/structural/bsm.md) of
structural models and for Cook-distance style outlier detection.

::: kalmanbox.smoothers.disturbance.DisturbanceSmoother
