# Kalman Filter Derivation

## The filtering problem

Given observations $y_1, \ldots, y_t$, estimate the state $\alpha_t$.

The Kalman filter computes:

$$a_{t|t} = E[\alpha_t | y_1, \ldots, y_t]$$
$$P_{t|t} = Var[\alpha_t | y_1, \ldots, y_t]$$

## Prediction step

$$a_{t|t-1} = T_t a_{t-1|t-1} + c_t$$

$$P_{t|t-1} = T_t P_{t-1|t-1} T_t' + R_t Q_t R_t'$$

## Innovation

$$v_t = y_t - Z_t a_{t|t-1} - d_t$$

$$F_t = Z_t P_{t|t-1} Z_t' + H_t$$

## Update step

**Kalman gain:**

$$K_t = P_{t|t-1} Z_t' F_t^{-1}$$

**Filtered state:**

$$a_{t|t} = a_{t|t-1} + K_t v_t$$

**Filtered covariance (Joseph form for numerical stability):**

$$P_{t|t} = (I - K_t Z_t) P_{t|t-1} (I - K_t Z_t)' + K_t H_t K_t'$$

## Log-likelihood

The log-likelihood is computed as a by-product:

$$\log L = -\frac{1}{2} \sum_{t=1}^{T} \left[ p \log(2\pi) + \log|F_t| + v_t' F_t^{-1} v_t \right]$$

## RTS Smoother

The Rauch-Tung-Striebel smoother runs backward from $t = T$ to $t = 1$:

**Smoother gain:**

$$L_t = P_{t|t} T_{t+1}' P_{t+1|t}^{-1}$$

**Smoothed state:**

$$a_{t|T} = a_{t|t} + L_t (a_{t+1|T} - a_{t+1|t})$$

**Smoothed covariance:**

$$P_{t|T} = P_{t|t} + L_t (P_{t+1|T} - P_{t+1|t}) L_t'$$

**Key property:** $P_{t|T} \leq P_{t|t}$ (smoothing always reduces uncertainty).

## References

- Durbin, J. & Koopman, S.J. (2012). Chapters 4-5.
- Anderson, B.D.O. & Moore, J.B. (1979). *Optimal Filtering*. Chapter 7.
