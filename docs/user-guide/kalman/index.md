# Kalman Filtering

The Kalman filter is the **forward recursion** that produces optimal
minimum-mean-square-error (MMSE) state estimates for a linear Gaussian
state-space model. The Rauch–Tung–Striebel (RTS) smoother is the matching
**backward recursion** that refines those estimates using the full sample.

This section covers the mechanics, the API, and the practical knobs:
missing data, diffuse initialisation, and multi-step forecasting.

---

## Pages in this section

<div class="grid cards" markdown>

-   :material-arrow-right-bold-circle:{ .lg .middle } **Kalman Filter**

    ---

    The forward recursion step by step: prediction, update, innovations,
    log-likelihood, Joseph-form covariance update, initialization strategies,
    and system matrix configuration with code examples.

    [:octicons-arrow-right-24: Kalman Filter](kalman-filter.md)

-   :material-arrow-left-bold-circle:{ .lg .middle } **RTS Smoother**

    ---

    The backward recursion: smoothing gain, smoothed states vs filtered states,
    when to smooth rather than filter, disturbance smoothing, and examples
    with trend extraction and component estimation.

    [:octicons-arrow-right-24: RTS Smoother](rts-smoother.md)

-   :material-chart-line:{ .lg .middle } **Forecasting**

    ---

    Multi-step-ahead forecasts, prediction intervals, and forecast variance
    decomposition beyond the estimation sample.

    [:octicons-arrow-right-24: Forecasting](forecasting.md)

-   :material-database-off:{ .lg .middle } **Missing Data**

    ---

    How to handle gaps in $y_t$ — the filter adapts naturally by
    skipping the update step whenever $y_t$ is `NaN`.

    [:octicons-arrow-right-24: Missing data](missing-data.md)

-   :material-blur:{ .lg .middle } **Diffuse Initialisation**

    ---

    Starting the filter with non-stationary states (random walks, integrated
    processes). Exact diffuse initialisation, approximate diffuse, transition
    from diffuse to standard recursion, and the diffuse log-likelihood.

    [:octicons-arrow-right-24: Diffuse initialisation](diffuse.md)

-   :material-chart-scatter-plot:{ .lg .middle } **MLE Estimation**

    ---

    Maximum likelihood via prediction-error decomposition, parameter
    transformations, standard errors from the Hessian, convergence
    diagnostics, and information criteria (AIC, BIC, HQIC).

    [:octicons-arrow-right-24: MLE Estimation](mle.md)

</div>

---

## The state-space model

All pages in this section refer to the following model:

$$
\begin{aligned}
\alpha_{t+1} &= T_t\,\alpha_t + c_t + R_t\,\eta_t, &\eta_t &\sim \mathcal{N}(0, Q_t) \\
y_t &= Z_t\,\alpha_t + d_t + \varepsilon_t, &\varepsilon_t &\sim \mathcal{N}(0, H_t)
\end{aligned}
$$

where $\alpha_t \in \mathbb{R}^k$ is the latent state and $y_t \in \mathbb{R}^p$
is the observed vector.

| Matrix | Dim          | Role                                               |
|--------|--------------|----------------------------------------------------|
| $T_t$  | $k \times k$ | State transition (dynamics)                        |
| $Z_t$  | $p \times k$ | Observation (links state to data)                  |
| $R_t$  | $k \times g$ | Selection (routes shocks into the state)           |
| $Q_t$  | $g \times g$ | State disturbance covariance                       |
| $H_t$  | $p \times p$ | Observation noise covariance                       |
| $c_t$  | $k \times 1$ | State intercept (optional; default zero)           |
| $d_t$  | $p \times 1$ | Observation intercept (optional; default zero)     |

---

## Filter quantities at a glance

Running `KalmanFilter.run(y)` produces, for each $t = 1, \ldots, n$:

| Output field    | Symbol          | Meaning                                          |
|-----------------|-----------------|--------------------------------------------------|
| `a`             | $a_{t\mid t-1}$ | $E[\alpha_t \mid y_{1:t-1}]$ — predicted mean   |
| `P`             | $P_{t\mid t-1}$ | $\text{Var}(\alpha_t \mid y_{1:t-1})$            |
| `a_filtered`    | $a_{t\mid t}$   | $E[\alpha_t \mid y_{1:t}]$ — filtered mean      |
| `P_filtered`    | $P_{t\mid t}$   | $\text{Var}(\alpha_t \mid y_{1:t})$              |
| `v`             | $v_t$           | Innovation $y_t - Z_t a_{t\mid t-1}$            |
| `F`             | $F_t$           | Innovation covariance $Z_t P_{t\mid t-1} Z_t' + H_t$ |
| `K`             | $K_t$           | Kalman gain $P_{t\mid t-1} Z_t' F_t^{-1}$       |
| `loglike`       | $\log p(y\mid\theta)$ | Prediction-error log-likelihood           |

After running the smoother, `RTSSmoother.run()` adds:

| Output field    | Symbol          | Meaning                                          |
|-----------------|-----------------|--------------------------------------------------|
| `a_smoothed`    | $a_{t\mid n}$   | $E[\alpha_t \mid y_{1:n}]$ — smoothed mean      |
| `P_smoothed`    | $P_{t\mid n}$   | $\text{Var}(\alpha_t \mid y_{1:n})$              |
| `J`             | $J_t$           | Smoothing gain $P_{t\mid t} T_t' P_{t+1\mid t}^{-1}$ |

---

## Minimal working example

```python
import numpy as np
from kalmanbox import KalmanFilter, RTSSmoother, StateSpaceRepresentation

# ── System matrices for a Local Level model ──────────────────────────────────
T = np.array([[1.0]])   # random-walk state
Z = np.array([[1.0]])   # direct observation
R = np.array([[1.0]])   # full-rank selection
Q = np.array([[0.5]])   # signal variance
H = np.array([[1.0]])   # noise variance

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

# ── Simulate some data ────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
n   = 100
eps = rng.normal(scale=1.0, size=n)
eta = rng.normal(scale=np.sqrt(0.5), size=n)
alpha = np.cumsum(eta)         # random walk
y     = alpha + eps            # noisy observations

# ── Forward filter ────────────────────────────────────────────────────────────
kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)

print(f"Log-likelihood : {out.loglike:.4f}")
print(f"Filtered shape : {out.a_filtered.shape}")   # (100, 1)

# ── Backward smoother ─────────────────────────────────────────────────────────
smoother = RTSSmoother(out, ssr)
sm = smoother.run()

print(f"Smoothed shape : {sm.a_smoothed.shape}")    # (100, 1)
# Smoothing can only reduce uncertainty
assert np.all(sm.P_smoothed <= out.P_filtered)
```

---

## Related

- [Theory: state-space foundations](../../theory/state-space-theory.md)
- [Theory: Kalman filter derivation](../../theory/kalman-filter-derivation.md)
- [Theory: RTS smoother derivation](../../theory/rts-derivation.md)
- [Diagnostics: residual analysis](../../diagnostics/residuals.md)
- [API: KalmanFilter](../../api/filters.md)
- [API: RTSSmoother](../../api/smoothers.md)
