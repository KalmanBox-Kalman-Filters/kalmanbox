# RTS smoother derivation

The Rauch–Tung–Striebel (RTS) smoother computes
$E[\alpha_t \mid y_{1:n}]$ and $\operatorname{Var}(\alpha_t \mid y_{1:n})$
for $t = 1, \ldots, n$. We give two equivalent derivations.

## Setup

Assume the linear Gaussian model

$$
\alpha_{t+1} = T_t \alpha_t + R_t \eta_t,\qquad
y_t = Z_t \alpha_t + \varepsilon_t,
$$

and that the Kalman filter has produced

$$
a_{t|t-1},\, P_{t|t-1},\, a_{t|t},\, P_{t|t} \quad \text{for } t = 1,\ldots,n.
$$

## Derivation 1 — joint Gaussian

The triple $(\alpha_t, \alpha_{t+1}, y_{1:n})$ is jointly Gaussian. Using
the conditional formula for partitioned Gaussians,

$$
\begin{pmatrix} \alpha_t \\ \alpha_{t+1} \end{pmatrix} \mid y_{1:t}
\sim \mathcal{N}\!\left(
\begin{pmatrix} a_{t|t} \\ T_t a_{t|t} \end{pmatrix},
\begin{pmatrix} P_{t|t} & P_{t|t} T_t' \\ T_t P_{t|t} & P_{t+1|t} \end{pmatrix}
\right).
$$

Conditioning further on the **future** $y_{t+1:n}$ acts only through
$\alpha_{t+1}$ (Markov property):

$$
\alpha_t \mid y_{1:n}
= \alpha_t \mid \alpha_{t+1}, y_{1:t}.
$$

Apply the conditional Gaussian formula with the **smoothing gain**
$J_t = P_{t|t} T_t' P_{t+1|t}^{-1}$:

$$
\boxed{\;
\begin{aligned}
a_{t|n} &= a_{t|t} + J_t \,(a_{t+1|n} - a_{t+1|t}), \\
P_{t|n} &= P_{t|t} + J_t\,(P_{t+1|n} - P_{t+1|t})\,J_t'.
\end{aligned}
\;}
$$

## Derivation 2 — via the BLUE projection

Treat $(\alpha_t)$ as random vectors and $a_{t|t}$ as the best linear
unbiased estimator (BLUE) given $y_{1:t}$. Adding $y_{t+1:n}$ injects
new information *only via* $\alpha_{t+1}$. The optimal update is the
projection of the residual $\alpha_{t+1} - a_{t+1|t}$ onto $\alpha_t$,
with regression coefficient exactly $J_t$. Same recursion drops out.

## Lag-one covariance

For EM and disturbance smoothing one needs

$$
P_{t,t-1|n} = \operatorname{Cov}(\alpha_t, \alpha_{t-1} \mid y_{1:n})
= P_{t|n}\, J_{t-1}'.
$$

This is what
[`compute_lag_one_covariance`][kalmanbox.estimation.em.compute_lag_one_covariance]
returns.

## Numerical notes

- The inverse $P_{t+1|t}^{-1}$ inside $J_t$ is the same matrix already
  inverted by the filter — store it from the forward pass instead of
  recomputing.
- A square-root smoother propagates Cholesky factors of $P_{t|n}$ for
  improved stability.

## Related

- [User guide: RTS smoother](../user-guide/kalman/rts-smoother.md)
- [Kalman filter derivation](kalman-filter-derivation.md)
- [Numerical stability](numerical-stability.md)
