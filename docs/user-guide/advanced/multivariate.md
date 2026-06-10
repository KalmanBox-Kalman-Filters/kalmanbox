# Multivariate State-Space Models

The standard kalmanbox Kalman filter extends naturally to systems with **multiple
observation series**. The multivariate Kalman filter handles
$y_t \in \mathbb{R}^p$ for $p > 1$, with possibly correlated observation errors
and correlated state disturbances. This page covers the mathematical foundations,
implementation details, and practical guidance for building and estimating
multivariate state-space models in kalmanbox.

---

## Multivariate SSM Formulation

### General form

A time-invariant multivariate state-space model is written as:

$$
y_t = Z \alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, H)
$$

$$
\alpha_{t+1} = T \alpha_t + R \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\, Q)
$$

with $\varepsilon_t \perp \eta_s$ for all $t, s$ and $\alpha_t \perp \varepsilon_s,
\eta_s$ for $s < t$.

### Dimensions

| Symbol | Space | Description |
|--------|-------|-------------|
| $y_t$ | $\mathbb{R}^p$ | Observation vector ($p$ series) |
| $\alpha_t$ | $\mathbb{R}^m$ | State vector ($m$ states) |
| $Z$ | $\mathbb{R}^{p \times m}$ | Observation (loading) matrix |
| $T$ | $\mathbb{R}^{m \times m}$ | State transition matrix |
| $H$ | $\mathbb{R}^{p \times p}$ | Observation noise covariance — not necessarily diagonal |
| $R$ | $\mathbb{R}^{m \times r}$ | Selection matrix (maps disturbances to states) |
| $Q$ | $\mathbb{R}^{r \times r}$ | State disturbance covariance |

!!! note "Time-varying matrices"
    All matrices can carry a time index — $Z_t, T_t, H_t, R_t, Q_t$ — to
    represent structural breaks, regime switches, or models with time-varying
    regressors. kalmanbox handles time-varying matrices natively by accepting
    arrays with a leading time dimension.

### Relationship to the univariate case

When $p = 1$, the multivariate formulation collapses to the familiar univariate
filter: $Z \in \mathbb{R}^{1 \times m}$, $H = \sigma_\varepsilon^2 \in
\mathbb{R}$, and the innovation covariance $F_t$ is a scalar. The multivariate
generalisation replaces every scalar quantity with the appropriate matrix.

---

## Multivariate Kalman Filter Recursion

### Prediction step

Given the filtered distribution $\alpha_{t-1|t-1} \sim \mathcal{N}(a_{t-1|t-1},
P_{t-1|t-1})$, the one-step-ahead predictive distribution is:

$$
a_{t|t-1} = T\, a_{t-1|t-1}
$$

$$
P_{t|t-1} = T\, P_{t-1|t-1}\, T' + R\, Q\, R'
$$

### Innovation

The prediction error and its covariance are:

$$
v_t = y_t - Z\, a_{t|t-1}
$$

$$
F_t = Z\, P_{t|t-1}\, Z' + H \;\in\; \mathbb{R}^{p \times p}
$$

!!! warning "Key difference from the univariate case"
    $F_t$ is now a $p \times p$ matrix. Its inversion costs $O(p^3)$ per
    time step — see the [Performance](#performance-and-computational-complexity)
    section for strategies when $p$ is large.

### Update step

The Kalman gain and updated distribution are:

$$
K_t = P_{t|t-1}\, Z'\, F_t^{-1} \;\in\; \mathbb{R}^{m \times p}
$$

$$
a_{t|t} = a_{t|t-1} + K_t\, v_t
$$

$$
P_{t|t} = (I_m - K_t\, Z)\, P_{t|t-1}
$$

For numerical stability kalmanbox uses the **Joseph form**:

$$
P_{t|t} = (I - K_t Z)\, P_{t|t-1}\, (I - K_t Z)' + K_t\, H\, K_t'
$$

which keeps $P_{t|t}$ positive semi-definite even with finite-precision
arithmetic.

### Log-likelihood

The Gaussian log-likelihood evaluated through the prediction error decomposition
is:

$$
\ell(\theta) = -\frac{pT}{2}\log(2\pi)
  - \frac{1}{2} \sum_{t=1}^{T} \left(\log|F_t| + v_t'\, F_t^{-1}\, v_t\right)
$$

where $\theta$ collects all free parameters in $Z, T, H, Q, R$. This is
maximised numerically to obtain the MLE.

---

## Correlated Observation Errors (Full H)

By default, kalmanbox sets $H$ to be **diagonal**, implying that idiosyncratic
observation errors across series are uncorrelated at the same time $t$. A
**full** (unstructured) $H$ lifts this restriction.

### When to use a full H

- **Currency pairs or asset returns** where market microstructure induces
  contemporaneous correlation in measurement errors.
- **Yield curve models** where nearby maturities share liquidity noise.
- **Multiple sensors** measuring the same physical process from different
  angles.

### Identification warning

!!! warning "Identification with DFM"
    A full $H$ combined with a Dynamic Factor Model may be under-identified:
    the shared factors and the error correlations can explain the same
    cross-sectional covariance. Either constrain $H$ to diagonal, impose
    exclusion restrictions on the loading matrix $\Lambda$, or use
    informative priors.

### Cholesky parameterisation

kalmanbox parameterises an unconstrained positive-definite $H$ via its lower
Cholesky factor $L$ so that $H = L L'$. This ensures $H \succ 0$ during
optimisation without additional constraints. Only the lower-triangular elements
of $L$ are free parameters — $p(p+1)/2$ values in total.

### Example

```python
import numpy as np
from kalmanbox.advanced import MultivariateSSM

# 3-series system with correlated observation errors.
# True H = [[sigma1^2, rho12*s1*s2, 0],
#           [rho12*s1*s2, sigma2^2, 0],
#           [0,          0,         sigma3^2]]
# Parameterised internally via Cholesky: H = L @ L.T

model = MultivariateSSM(
    y,                           # (T, 3) array
    n_states=2,
    H_structure="unstructured",  # free p×p covariance
)
results = model.fit()

print(results.params["H"])       # (3, 3) estimated covariance matrix
```

The estimated $H$ can be inspected as a full $p \times p$ matrix. Off-diagonal
elements quantify contemporaneous error correlation between pairs of series.

---

## Correlated State Disturbances (Full Q)

The state disturbance covariance $Q$ governs how the latent states
**co-move over time**. A full $Q$ allows states to drift together, which is
appropriate when the underlying dynamics are driven by a common shock.

### When to use a full Q

- **Bivariate local level**: two series that trend together (e.g. wages and
  productivity).
- **Joint slope models**: two yield curves sharing a common slope factor.
- **Multi-country macro panels**: output gaps that respond to a common
  global business cycle.

### Example — bivariate local level with correlated innovations

```python
import numpy as np
from kalmanbox.advanced import MultivariateSSM

# Simulate two correlated random walks.
rho = 0.8
Sigma = np.array([[1.0, rho],
                  [rho, 1.0]])
rng = np.random.default_rng(42)
shocks = rng.multivariate_normal([0.0, 0.0], Sigma, size=200)
states = np.cumsum(shocks, axis=0)                          # (200, 2)
noise  = rng.multivariate_normal([0.0, 0.0], 0.1 * np.eye(2), size=200)
y = states + noise                                          # (200, 2)

model = MultivariateSSM(
    y,
    T=np.eye(2),           # random walk transitions
    Z=np.eye(2),           # direct (noisy) observations
    H_structure="diagonal",
    Q_structure="full",    # allow correlated state shocks
)
results = model.fit()

print("Estimated Q (state disturbance covariance):")
print(results.params["Q"])
# Expected: off-diagonal close to 0.8, diagonal close to 1.0
```

!!! tip "Q vs H"
    A common modelling question is whether to put correlation in $Q$ or in
    $H$. Correlation in $Q$ means the **permanent** (state) component is
    shared; correlation in $H$ means only the **transient** measurement
    errors co-move. If the cross-series co-movement persists over time,
    prefer a full $Q$.

---

## VAR in State-Space Form

A Vector Autoregression (VAR) is one of the most common multivariate time-series
models. Casting it in state-space form grants access to the Kalman filter,
missing-data handling, and a unified estimation framework.

### VAR(1)

$$
y_t = \Phi_1\, y_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma)
$$

State-space equivalence: set $\alpha_t = y_t$, $T = \Phi_1$, $Z = I_p$,
$Q = \Sigma$, $H = 0$.

### VAR(2) companion form

A VAR(2) with $p$ variables is rewritten as a VAR(1) in the $2p$-dimensional
companion state $\tilde{\alpha}_t = (y_t', y_{t-1}')'$:

$$
\begin{bmatrix} y_t \\ y_{t-1} \end{bmatrix}
=
\begin{bmatrix} \Phi_1 & \Phi_2 \\ I_p & 0 \end{bmatrix}
\begin{bmatrix} y_{t-1} \\ y_{t-2} \end{bmatrix}
+
\begin{bmatrix} \varepsilon_t \\ 0 \end{bmatrix}
$$

The companion state transition matrix has dimension $2p \times 2p$. The
observation equation maps back to the $p$ observed variables:
$Z = [I_p \;\; 0]$, $H = 0$ (observations are exact apart from the VAR shock).

### VAR(lag) in kalmanbox

```python
from kalmanbox.advanced import VARStateSpace

# VAR(2) in state-space form with 3 variables.
# y has shape (T, 3).
var_model = VARStateSpace(y, lags=2)
results = var_model.fit()

print("Phi_1:")
print(results.params["Phi"][0])   # (3, 3) first-lag coefficient matrix
print("Phi_2:")
print(results.params["Phi"][1])   # (3, 3) second-lag coefficient matrix
print("Sigma:")
print(results.params["Sigma"])    # (3, 3) innovation covariance

# Impulse responses via state-space recursion.
irf = results.impulse_response(steps=20, shock_size=1.0)
print(irf.shape)   # (20, 3, 3): [steps, response_variable, shock_variable]
```

### Advantages of the SSM representation of a VAR

- **Missing data**: ragged-edge panels or irregular sampling handled exactly
  by the filter — no listwise deletion.
- **Mixed-frequency VARs**: quarterly GDP and monthly CPI in a single system
  without aggregation.
- **Identification via restrictions**: impose structural restrictions on $\Phi$
  or $\Sigma$ through the `CustomStateSpace` interface.
- **Bayesian estimation**: Minnesota-style priors on $\Phi$ map naturally to
  Gaussian priors on the state-space parameters.

---

## MultivariateSSM — Custom System

For structural economic or financial models, you can supply the full system
matrices directly. kalmanbox validates dimensions, runs the filter, and
provides the standard results API.

### Worked example — 4-equation structural system

```python
import numpy as np
from kalmanbox.advanced import MultivariateSSM

# 4-equation structural macro system.
# States:  [output_gap, inflation, interest_rate, foreign_rate]
# Observations: [gdp_proxy, cpi, fed_funds, libor]

T = np.array([
    [0.90, -0.10,  0.00,  0.05],
    [0.30,  0.80, -0.20,  0.00],
    [0.10,  0.50,  0.70,  0.10],
    [0.00,  0.00,  0.30,  0.85],
])

Z = np.eye(4)          # all states observed with noise

H = np.diag([0.10, 0.05, 0.02, 0.08])   # diagonal measurement noise

Q = np.array([
    [0.40, 0.10, 0.00, 0.00],
    [0.10, 0.20, 0.10, 0.00],
    [0.00, 0.10, 0.30, 0.10],
    [0.00, 0.00, 0.10, 0.20],
])   # block-sparse state disturbance covariance

model = MultivariateSSM(y=data, T=T, Z=Z, H=H, Q=Q)
results = model.filter()

# Kalman-filtered state estimates and covariances.
a_filt = results.a_filtered    # (T, 4): filtered state means
P_filt = results.P_filtered    # (T, 4, 4): filtered state covariances

# RTS smoother for full-sample inference.
smoother = model.smooth()
a_smooth = smoother.a_smoothed  # (T, 4)
```

### Parameter estimation

When the system matrices contain unknown parameters, use `.fit()` instead of
`.filter()`:

```python
# Specify which elements are free via a mask or param_names.
model = MultivariateSSM(
    y=data,
    T=T,
    Z=Z,
    H="estimate_diagonal",   # estimate diagonal H
    Q="estimate_full",       # estimate full positive-definite Q
)
results = model.fit(method="mle")
print(results.summary())
```

---

## Performance and Computational Complexity

The dominant cost in the multivariate Kalman filter is the inversion of $F_t$
at each time step.

| Operation | Univariate ($p = 1$) | Multivariate ($p$ series) |
|-----------|---------------------|--------------------------|
| $F_t$ inversion | $O(1)$ | $O(p^3)$ |
| Kalman gain $K_t$ | $O(m)$ | $O(m \cdot p^2)$ |
| State update | $O(m^2)$ | $O(m^2)$ |
| Total per time step | $O(m^2)$ | $O(m^2 + p^3)$ |
| Memory | $O(m^2)$ | $O(m^2 + p^2)$ |

For panels with $p \lesssim 10$ the $O(p^3)$ cost is negligible. For large
panels it dominates.

### Woodbury identity for diagonal H

When $H$ is diagonal, the inversion of $F_t = Z P_{t|t-1} Z' + H$ can be
avoided using the **Woodbury matrix identity**:

$$
F_t^{-1}
= H^{-1}
  - H^{-1} Z\, P_{t|t-1}\, Z'
    \bigl(I_m + Z\, P_{t|t-1}\, Z'\, H^{-1}\bigr)^{-1}
    H^{-1}
$$

This reduces the per-step cost from $O(p^3)$ to $O(m^3 + p \cdot m^2)$,
which is favourable when $m \ll p$. kalmanbox applies this optimisation
automatically when `H_structure="diagonal"`.

!!! tip "Large panels ($p > 20$)"
    For very wide panels:

    - Use a **Dynamic Factor Model** (`DFM`) to reduce effective
      dimensionality from $p$ to $k \ll p$.
    - Ensure `H_structure="diagonal"` so the Woodbury optimisation activates.
    - For partially correlated observations, use a block-diagonal $H$ with
      `H_structure="block_diagonal"`.

```python
# Activate Woodbury optimisation explicitly.
model = MultivariateSSM(
    y,
    H_structure="diagonal",   # enables Woodbury for O(p·m²) instead of O(p³)
)
```

### Univariate treatment of multivariate series

An alternative to inverting $F_t \in \mathbb{R}^{p \times p}$ is to process
the $p$ observations **one at a time** within each period (the *univariate
treatment* of Koopman & Durbin, 2000). This requires $H$ to be diagonal.
The total cost becomes $p$ scalar filter steps per period, each $O(m^2)$,
giving $O(p \cdot m^2)$ overall — identical to the Woodbury result but
sometimes easier to implement. kalmanbox uses this approach when
`H_structure="diagonal"` and the state dimension is large.

---

## Missing Data in Multivariate Systems

Individual series can carry `NaN` values at different time points — **ragged
edges** (series start at different dates), **mixed frequency** (one series
observed quarterly, another monthly), or **unbalanced panels** (arbitrary
missing patterns).

### How the filter handles missing observations

At time $t$, let $\mathcal{O}_t \subseteq \{1, \ldots, p\}$ be the index
set of observed series. Define $Z_t^* = Z[\mathcal{O}_t, :]$ and
$H_t^* = H[\mathcal{O}_t, \mathcal{O}_t]$. The filter proceeds with the
reduced system $y_t^* = Z_t^* \alpha_t + \varepsilon_t^*$ — the prediction
step is unchanged and only the update uses the available observations. When
$\mathcal{O}_t = \emptyset$ (no observations), the update step is skipped
entirely and $a_{t|t} = a_{t|t-1}$, $P_{t|t} = P_{t|t-1}$.

!!! note "No imputation required"
    State estimation is **exact** under missing data — there is no need to
    impute missing observations beforehand. The uncertainty about the missing
    values is propagated automatically through $P_{t|t-1}$.

### Mixed-frequency example

```python
import numpy as np
from kalmanbox.advanced import MultivariateSSM

# Monthly panel: GDP (quarterly) + CPI (monthly), 36 months.
y_mixed = np.full((36, 2), np.nan)
y_mixed[2::3, 0] = gdp_quarterly   # GDP available at end of each quarter
y_mixed[:, 1]    = cpi_monthly     # CPI available every month

model = MultivariateSSM(y_mixed, ...)   # NaN handled natively
results = model.filter()

# a_filtered is available for every month, even where GDP is missing.
print(results.a_filtered.shape)   # (36, m)
```

### Ragged-edge nowcasting

```python
import numpy as np
from kalmanbox.advanced import MultivariateSSM

# Last observation of each indicator comes at a different date.
y = load_panel()           # (T, p), last few rows contain NaN for some series
model = MultivariateSSM(y, ...)
results = model.filter()

# Nowcast: one-step-ahead forecast at the ragged edge.
nowcast = results.a_predicted[-1]   # state mean at T+1
```

---

## Forecasting

The Kalman filter provides the optimal linear predictor for all future horizons
given the full data history. For a time-invariant system the $h$-step-ahead
state forecast is:

$$
a_{T+h|T} = T^h\, a_{T|T}
$$

and the associated covariance satisfies the Riccati recursion:

$$
P_{T+h|T} = T\, P_{T+h-1|T}\, T' + R\, Q\, R', \qquad h = 1, 2, \ldots
$$

The observation forecast and its covariance are:

$$
\hat{y}_{T+h} = Z\, a_{T+h|T}, \qquad
\operatorname{Var}(\hat{y}_{T+h}) = Z\, P_{T+h|T}\, Z' + H
$$

Because $\operatorname{Var}(\hat{y}_{T+h})$ is the **full** $p \times p$
covariance matrix, kalmanbox can report marginal intervals for each series
and joint prediction ellipsoids.

```python
# h-step ahead joint forecasts for all p series.
fc = results.forecast(steps=8)

means = fc["mean"]          # (8, p): point forecasts
covs  = fc["covariance"]    # (8, p, p): joint forecast covariance matrices
lower = fc["lower_95"]      # (8, p): 95% lower bound (marginal)
upper = fc["upper_95"]      # (8, p): 95% upper bound (marginal)

# Forecast correlation between series at horizon h=1.
cov_h1 = fc["covariance"][0]                    # (p, p)
std_h1 = np.sqrt(np.diag(cov_h1))              # (p,)
corr_h1 = cov_h1 / np.outer(std_h1, std_h1)   # (p, p)

print("1-step forecast correlation matrix:")
print(corr_h1)
```

!!! tip "Fan charts"
    The joint forecast covariance can be used to simulate forecast paths
    via `results.simulate_forecast(steps=h, n_paths=1000)`, which draws
    from the multivariate Gaussian forecast distribution — useful for
    fan charts and Value-at-Risk calculations.

---

## API Reference

Multivariate and VAR systems are built with the general-purpose
[`CustomStateSpace`][kalmanbox.models.custom.CustomStateSpace] model, which lets
you specify arbitrary `Z`, `T`, `R`, `Q`, and `H` matrices (including the
full-`H` and companion-form constructions shown above).

::: kalmanbox.models.custom.CustomStateSpace
    options:
      heading_level: 3
      show_source: false

---

## Related

- [Dynamic Factor Model](dfm.md) — dimensionality reduction for large panels
- [Kalman Filter](../kalman/kalman-filter.md) — univariate foundations
- [Missing data](../kalman/missing-data.md) — handling gaps in univariate settings
- [EM Algorithm](em.md) — parameter estimation for multivariate models
- [Time-Varying Parameters](tvp.md) — time-varying coefficients

---

## References

- Anderson, B. D. O. & Moore, J. B. (1979). *Optimal Filtering*. Prentice-Hall. Ch. 3–4.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press. Ch. 2, 6.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press. Ch. 4.
- Koopman, S. J. & Durbin, J. (2000). Fast filtering and smoothing for multivariate state space models. *Journal of Time Series Analysis*, 21(3), 281–296.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Ch. 9.
