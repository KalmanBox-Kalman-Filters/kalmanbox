# Square-Root Kalman Filter

The [`SquareRootFilter`][kalmanbox.filters.SquareRootFilter] implements a numerically superior
variant of the standard Kalman filter that maintains and propagates the **Cholesky factor**
$S_t$ of the state covariance matrix $P_t = S_t S_t'$, rather than $P_t$ directly.

!!! note "When to use the Square-Root Filter"
    Use this filter when the standard Kalman filter produces negative-definite or asymmetric
    covariance matrices, when the condition number of $P_t$ is very large, or when processing
    long time series where numerical drift accumulates. The model must still be **linear
    Gaussian** — for nonlinear models, see [EKF](ekf.md) or [UKF](ukf.md).

---

## 1. The numerical stability problem

The standard Kalman covariance update involves the recursion:

$$
P_{t|t} = (I - K_t Z)\,P_{t|t-1}\,(I - K_t Z)' + K_t H K_t' \quad \text{(Joseph form)}
$$

Even with the Joseph form (which is symmetric by construction), repeated matrix
multiplications over long time series can cause numerical issues:

- **Loss of symmetry**: floating-point rounding makes $P$ slightly non-symmetric after each
  step; over $T$ steps the asymmetry compounds
- **Loss of positive-definiteness**: small rounding errors accumulate until some eigenvalue
  of $P$ becomes negative, violating the fundamental property that covariance matrices are PSD
- **Ill-conditioning**: when the ratios of state uncertainties across dimensions differ by
  many orders of magnitude, the condition number
  $\kappa(P) = \lambda_{\max}(P) / \lambda_{\min}(P)$ can exceed $10^{12}$, causing
  catastrophic cancellation in subsequent operations

### Quantifying the precision loss

For a $k \times k$ covariance matrix in double precision with unit roundoff
$\epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$, the number of accurate decimal digits
in $P$ after $T$ steps satisfies approximately:

$$
\text{digits}(P_T) \approx 16 - \log_{10}(\kappa(P) \cdot T)
$$

For $\kappa(P) = 10^8$ and $T = 1000$, this leaves only $16 - 8 - 3 = 5$ accurate digits —
insufficient for reliable inference.

### The square-root solution

The Cholesky factor $S = \operatorname{chol}(P)$ satisfies:

$$
\kappa(S) = \sqrt{\kappa(P)}
$$

The condition number of the factor is the **square root** of the condition number of the full
matrix. By operating on $S$ rather than $P$:

- Numerical errors scale with $\sqrt{\kappa(P)}$ rather than $\kappa(P)$
- Positive-definiteness is guaranteed because $P = S S'$ is PD whenever $S$ is nonsingular
- Symmetry is guaranteed by the $S S'$ structure — no asymmetry can accumulate

---

## 2. Mathematical formulation

### State-space model

The Square-Root filter applies to the same linear Gaussian SSM as the standard filter:

$$
\begin{aligned}
\alpha_{t+1} &= T\,\alpha_t + R\,\eta_t, & \eta_t &\sim \mathcal{N}(0, Q) \\
y_t &= Z\,\alpha_t + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, H)
\end{aligned}
$$

### Notation

Throughout this page, $S_t$ denotes the **lower triangular Cholesky factor** such that
$P_t = S_t S_t'$. Similarly $S_Q = \operatorname{chol}(Q)$ and $S_H = \operatorname{chol}(H)$.

### 2.1 Initialization

$$
a_{0|0} = \mathbb{E}[\alpha_0], \qquad S_{0|0} = \operatorname{chol}(P_0)
$$

### 2.2 Prediction step via QR decomposition

The standard predicted covariance is:

$$
P_{t|t-1} = T\,P_{t-1|t-1}\,T' + R\,Q\,R' = T S_{t-1} S_{t-1}' T' + R S_Q S_Q' R'
$$

To extract $S_{t|t-1}$ without ever forming $P_{t|t-1}$, construct the block matrix:

$$
\boxed{
\begin{bmatrix}
S_{t-1|t-1}'\,T' \\[4pt]
S_Q'\,R'
\end{bmatrix}
\xrightarrow{\text{QR}}
Q_{\text{QR}}\,
\begin{bmatrix}
S_{t|t-1}' \\
0
\end{bmatrix}
}
$$

where $Q_{\text{QR}}$ is an orthogonal matrix (discarded) and $S_{t|t-1}'$ is the upper
triangular Cholesky factor of $P_{t|t-1}$, read from the first $k$ rows of the right factor.

This QR-based approach is exact: $P_{t|t-1} = S_{t|t-1} S_{t|t-1}'$ by construction.

The predicted state mean is unchanged:

$$
a_{t|t-1} = T\,a_{t-1|t-1}
$$

### 2.3 Update step via QR decomposition

The Square-Root update uses a single QR decomposition on the augmented matrix:

$$
\boxed{
\begin{bmatrix}
S_H' & Z\,S_{t|t-1} \\[4pt]
0   & S_{t|t-1}
\end{bmatrix}
\xrightarrow{\text{QR}}
\begin{bmatrix}
S_{S_t}' & \star \\
0       & S_{t|t}'
\end{bmatrix}
}
$$

The upper-left block $S_{S_t}'$ is the Cholesky factor of the **innovation covariance**
$\Sigma_t = Z\,P_{t|t-1}\,Z' + H$, and the lower-right block $S_{t|t}'$ is the Cholesky
factor of the **updated covariance** $P_{t|t}$ — both extracted simultaneously without
forming $P_{t|t-1}$ explicitly.

The Kalman gain is then computed via triangular solves (no matrix inversion):

$$
K_t = \left(S_{t|t-1}\,\left(S_{t|t-1}'\,Z'\,S_{S_t}^{-T}\right)\right)\,S_{S_t}^{-1}
$$

where $S_{S_t}^{-T}$ and $S_{S_t}^{-1}$ are backward/forward triangular solves —
numerically stable and $O(p^2)$ rather than $O(p^3)$.

### 2.4 Innovation and log-likelihood

$$
v_t = y_t - Z\,a_{t|t-1}
$$

$$
\ell_t = -\frac{p}{2}\ln(2\pi) - \sum_{j=1}^{p}\ln\bigl[S_{S_t}\bigr]_{jj}
         - \frac{1}{2}\,v_t'\,S_{S_t}^{-T}\,S_{S_t}^{-1}\,v_t
$$

The determinant $|Z P_{t|t-1} Z' + H| = \prod_j [S_{S_t}]_{jj}^2$ is computed cheaply from
the diagonal of the triangular factor.

---

## 3. Available algorithms

kalmanbox supports three Square-Root implementations, selectable via the `method` parameter:

=== "QR method (default)"

    **Bierman-Thornton** formulation using QR decompositions.

    - Handles arbitrary observation dimension $p$
    - Each step costs $O(k^2 p)$ for the QR
    - Numerically the most stable option

    ```python
    from kalmanbox.filters import SquareRootFilter

    sqf = SquareRootFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs,
                           a0=a0, P0=P0, method="qr")
    ```

=== "Potter algorithm"

    **Sequential scalar measurement** processing via Givens rotations (rank-1 Cholesky updates).

    - Most efficient for $p \ll k$ (few observations per step)
    - Each scalar update costs $O(k^2)$ for $p$ total updates per step
    - Slightly less stable than QR for large $p$

    ```python
    sqf = SquareRootFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs,
                           a0=a0, P0=P0, method="potter")
    ```

=== "UD factorization"

    **Carlson-Schmidt** $UDU'$ factorization where $U$ is unit upper triangular and $D$ is
    diagonal.

    - Avoids square roots in the scalar update (uses only multiply/add)
    - Well-suited for embedded or fixed-precision systems
    - $P_t = U_t D_t U_t'$ with $D_t$ diagonal positive

    ```python
    sqf = SquareRootFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs,
                           a0=a0, P0=P0, method="ud")
    ```

---

## 4. API reference

```python
from kalmanbox.filters import SquareRootFilter

sqf = SquareRootFilter(
    T,               # transition matrix, shape (k, k)
    Z,               # observation matrix, shape (p, k)
    Q,               # process noise cov, shape (q, q)
    H,               # observation noise cov, shape (p, p)
    a0,              # initial state mean, shape (k,)
    P0,              # initial state cov, shape (k, k) — stored as Cholesky internally
    R=None,          # selection matrix, shape (k, q); defaults to I_k
    c=None,          # constant state intercept, shape (k,)
    d=None,          # constant observation intercept, shape (p,)
    method="qr",     # "qr" | "potter" | "ud"
)
```

### Key methods

| Method | Description |
|--------|-------------|
| `sqf.filter(y)` | Forward Square-Root Kalman pass over `y` of shape `(T, p)` |
| `sqf.smooth(y)` | Forward pass then Square-Root RTS smoother backward pass |
| `sqf.log_likelihood(y)` | Exact log-likelihood via Cholesky determinants |
| `sqf.condition_numbers()` | Return $\kappa(S_t)$ for each $t$ — diagnostic for numerical health |

### FilterResult attributes

```python
result = sqf.filter(y)

result.filtered_states        # shape (T, k): a_{t|t}
result.filtered_covariances   # shape (T, k, k): P_{t|t} = S_{t|t} S_{t|t}'
result.filtered_sqrt_cov      # shape (T, k, k): S_{t|t} Cholesky factors
result.predicted_states       # shape (T, k): a_{t|t-1}
result.innovations            # shape (T, p): v_t
result.log_likelihood         # scalar
```

---

## 5. Numerical stability: standard vs Square-Root

### Example 1: Ill-conditioned covariance

```python
import numpy as np
from kalmanbox.filters import KalmanFilter, SquareRootFilter

np.random.seed(42)
k, p, T_len = 5, 2, 500

rng = np.random.default_rng(42)
A = rng.standard_normal((k, k))
T_mat = A @ np.diag(np.linspace(0.99, 0.95, k)) @ np.linalg.inv(A)
Z_mat = rng.standard_normal((p, k))

# Ill-conditioned: state uncertainties span 10 orders of magnitude
P0 = np.diag([1e0, 1e2, 1e4, 1e6, 1e8])
a0 = np.zeros(k)
Q = 0.01 * np.eye(k)
H_obs = 0.1 * np.eye(p)

x = a0.copy()
y = np.zeros((T_len, p))
for t in range(T_len):
    x = T_mat @ x + np.random.multivariate_normal(np.zeros(k), Q)
    y[t] = Z_mat @ x + np.random.multivariate_normal(np.zeros(p), H_obs)

kf  = KalmanFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs, a0=a0, P0=P0)
sqf = SquareRootFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs, a0=a0, P0=P0)

r_kf  = kf.filter(y)
r_sqf = sqf.filter(y)

evals_kf = np.linalg.eigvalsh(r_kf.filtered_covariances[-1])
evals_sq = np.linalg.eigvalsh(r_sqf.filtered_covariances[-1])

print(f"Standard KF — min eigenvalue: {evals_kf.min():.2e}")   # may be negative!
print(f"Square-Root — min eigenvalue: {evals_sq.min():.2e}")   # always positive

cond_numbers = sqf.condition_numbers()
print(f"Max cond number (S_t): {cond_numbers.max():.2e}")      # sqrt of kappa(P)
```

### Example 2: Long time series with drift

```python
import numpy as np
from kalmanbox.filters import KalmanFilter, SquareRootFilter

T_len = 10_000
sigma_level, sigma_obs = 1.0, 2.0

T_mat = np.array([[1.0]])
Z_mat = np.array([[1.0]])
Q = np.array([[sigma_level**2]])
H_obs = np.array([[sigma_obs**2]])
a0 = np.array([0.0])
P0 = np.array([[10.0]])

level = np.cumsum(sigma_level * np.random.randn(T_len))
y = (level + sigma_obs * np.random.randn(T_len)).reshape(-1, 1)

r_kf  = KalmanFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs, a0=a0, P0=P0).filter(y)
r_sqf = SquareRootFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs, a0=a0, P0=P0).filter(y)

P_kf  = r_kf.filtered_covariances[:, 0, 0]
P_sqf = r_sqf.filtered_covariances[:, 0, 0]
print(f"Max |P_kf - P_sqf|: {np.abs(P_kf - P_sqf).max():.2e}")
# For scalar case both agree; divergence appears when k > 1 is ill-conditioned
```

---

## 6. When does the Square-Root filter provide a decisive advantage?

| Scenario | Standard KF risk | Square-Root benefit |
|----------|:---------------:|:-------------------:|
| $\kappa(P_0) > 10^8$ | High | High |
| Long series ($T > 10^4$) | Moderate | Moderate |
| High state dimension ($k > 50$) | Moderate | Moderate to High |
| States with vastly different scales | High | High |
| Standard KF already well-conditioned | Low | Negligible |

!!! tip "Performance overhead"
    The Square-Root filter costs approximately **50% more** per time step than the standard
    Kalman filter (due to the QR decomposition). This overhead is negligible compared to the
    cost of unreliable results from a numerically unstable standard filter.

---

## 7. Square-Root RTS smoother

The backward smoother propagates Cholesky factors through the RTS equations:

$$
\boxed{
\begin{aligned}
J_t &= P_{t|t}\,T'\,P_{t+1|t}^{-1} = S_{t|t}\,\bigl(S_{t|t-1}^{-T} T' S_{t+1|t}^{-1}\bigr)' \\[4pt]
a_{t|n} &= a_{t|t} + J_t\,(a_{t+1|n} - a_{t+1|t}) \\[4pt]
S_{t|n} &= \operatorname{cholupdate}\!\left(S_{t|t},\, J_t,\, S_{t+1|n},\, S_{t+1|t}\right)
\end{aligned}
}
$$

The `smooth()` method applies this automatically via a QR-based backward sweep:

```python
smooth_result = sqf.smooth(y)

smooth_result.smoothed_states       # shape (T, k): a_{t|n}
smooth_result.smoothed_covariances  # shape (T, k, k): P_{t|n}
smooth_result.smoothed_sqrt_cov     # shape (T, k, k): S_{t|n} Cholesky factors
```

---

## 8. Condition number diagnostics

kalmanbox provides a unified diagnostic interface to compare filter numerical health:

```python
from kalmanbox.filters import KalmanFilter, SquareRootFilter
from kalmanbox.diagnostics import filter_condition_report

kf  = KalmanFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs, a0=a0, P0=P0)
sqf = SquareRootFilter(T=T_mat, Z=Z_mat, Q=Q, H=H_obs, a0=a0, P0=P0)

r_kf  = kf.filter(y)
r_sqf = sqf.filter(y)

print(filter_condition_report(r_kf,  label="Standard KF"))
print(filter_condition_report(r_sqf, label="Square-Root KF"))
# Report includes: max/mean condition number, minimum eigenvalue across all t,
# number of near-singular steps
```

---

## See also

- [Kalman Filter](../kalman/kalman-filter.md) — the standard linear filter (baseline)
- [Numerical Stability Theory](../../theory/numerical-stability.md) — full treatment of
  condition numbers, the Joseph form, and square-root methods
- [EKF](ekf.md) — nonlinear extension (uses standard covariance form by default)
- [Information Filter](information.md) — dual formulation; superior for high observation dimension
- [API Reference: Filters](../../api/filters.md)
