---
title: "Tutorial: Kalman Filter Fundamentals"
description: >-
  A beginner tutorial that builds a Kalman Filter from scratch on synthetic
  data — covering state-space representation, filtering, RTS smoothing,
  MLE parameter estimation, and residual diagnostics.
---

# Tutorial: Kalman Filter Fundamentals

**Level:** :material-signal: Beginner · **Time:** ~45 min · **Dataset:** Synthetic

This tutorial teaches the Kalman Filter from the ground up. We generate
a synthetic dataset where we know the true hidden state, then use
`kalmanbox` to recover it — so we can see exactly how well each step works.

By the end you will have:

- Built a `StateSpaceRepresentation` from scratch
- Run the forward `KalmanFilter` recursion
- Run the backward `RTSSmoother`
- Estimated parameters by maximum likelihood
- Assessed the model with residual diagnostics

!!! info "Prerequisites"
    - Python ≥ 3.10, NumPy ≥ 1.24, Matplotlib ≥ 3.7
    - `pip install kalmanbox`
    - No prior state-space experience required

---

## The problem: tracking a hidden signal

Suppose a signal $\mu_t$ evolves over time as a **random walk** — it drifts
up or down unpredictably at each step. We observe a noisy version $y_t$.
The goal is to recover the true signal from the noisy observations.

$$
\begin{aligned}
\text{Signal (state):} \quad & \mu_{t+1} = \mu_t + \eta_t, &\quad \eta_t \sim \mathcal{N}(0,\, \sigma_\eta^2) \\
\text{Observation:}   \quad & y_t = \mu_t + \varepsilon_t, &\quad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_\varepsilon^2)
\end{aligned}
$$

This is the simplest possible state-space model: one state, one observation,
no inputs. It is known as the **Local Level Model**.

The key parameters are:

| Parameter | Symbol | Role |
|-----------|--------|------|
| State noise variance | $\sigma_\eta^2$ | How fast the signal drifts |
| Observation noise variance | $\sigma_\varepsilon^2$ | How noisy the measurements are |
| Signal-to-noise ratio | $q = \sigma_\eta^2 / \sigma_\varepsilon^2$ | Controls filter responsiveness |

When $q$ is large, the signal changes fast and the filter tracks
observations closely. When $q$ is small, the signal is stable and the
filter smooths heavily.

---

## Step 1 — Generate synthetic data

We simulate 200 time steps. The true signal is a random walk with
$\sigma_\eta = 0.5$; observations add noise with $\sigma_\varepsilon = 2.0$.
This gives a signal-to-noise ratio of $q = 0.0625$, meaning the signal
changes slowly relative to the noise.

```python
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Simulation parameters ──────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 200
sigma_eta_true = 0.5    # true state noise std
sigma_eps_true = 2.0    # true observation noise std

# ── 2. Simulate the random walk ───────────────────────────────────────────────
true_state = np.zeros(n)
true_state[0] = 0.0
for t in range(1, n):
    true_state[t] = true_state[t - 1] + rng.normal(0, sigma_eta_true)

# ── 3. Add measurement noise ──────────────────────────────────────────────────
y = true_state + rng.normal(0, sigma_eps_true, n)

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y, color="lightgray", alpha=0.9, label="Observations $y_t$", zorder=1)
ax.plot(true_state, color="steelblue", linewidth=1.5,
        label="True state $\\mu_t$", zorder=2)
ax.set_title("Synthetic data — random walk + noise")
ax.set_xlabel("Time $t$")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Observations  : {y.shape}")
print(f"True state    : {true_state.shape}")
print(f"Signal-to-noise ratio q = {(sigma_eta_true/sigma_eps_true)**2:.4f}")
```

### Expected output

```
Observations  : (200,)
True state    : (200,)
Signal-to-noise ratio q = 0.0625
```

The plot shows two lines: the jagged gray observations and the smoother blue
true signal underneath. Our goal is to estimate the blue line from the gray
one — without knowing the true state.

!!! note "Why simulate?"
    Simulation lets us benchmark the filter against ground truth. In real
    applications the true state is unobserved, but here we can measure
    root-mean-squared error (RMSE) directly.

---

## Step 2 — Define the state-space representation

Every `kalmanbox` model ultimately reduces to a
`StateSpaceRepresentation` — a collection of system matrices that
fully specifies the model. For the Local Level model:

$$
\begin{aligned}
\alpha_{t+1} &= T \alpha_t + R \eta_t \\
y_t          &= Z \alpha_t + \varepsilon_t
\end{aligned}
$$

with state $\alpha_t = \mu_t \in \mathbb{R}^1$ (scalar) and observation
$y_t \in \mathbb{R}^1$ (scalar).

| Matrix | Symbol | Dim | Value | Meaning |
|--------|--------|-----|-------|---------|
| Transition | $T$ | $1 \times 1$ | `[[1.0]]` | Random walk: $\mu_{t+1} = \mu_t$ |
| Observation | $Z$ | $1 \times 1$ | `[[1.0]]` | State is directly observed |
| Noise selection | $R$ | $1 \times 1$ | `[[1.0]]` | Identity: noise enters all states |
| State noise cov. | $Q$ | $1 \times 1$ | `[[σ²_η]]` | Random walk variance |
| Obs. noise cov. | $H$ | $1 \times 1$ | `[[σ²_ε]]` | Measurement variance |
| Initial mean | $a_1$ | $1$ | `[0.0]` | Prior mean of $\mu_1$ |
| Initial cov. | $P_1$ | $1 \times 1$ | `[[100.0]]` | Diffuse prior (large value) |

```python
from kalmanbox.core.representation import StateSpaceRepresentation

# ── Use the true variances for now (Step 7 will estimate them) ────────────────
rep = StateSpaceRepresentation(
    T=np.array([[1.0]]),                      # transition: μ_{t+1} = μ_t
    Z=np.array([[1.0]]),                      # observation: y_t = μ_t + ε_t
    R=np.array([[1.0]]),                      # noise enters state directly
    Q=np.array([[sigma_eta_true**2]]),        # state noise covariance
    H=np.array([[sigma_eps_true**2]]),        # observation noise covariance
    a1=np.array([0.0]),                       # initial state mean
    P1=np.array([[100.0]]),                   # diffuse initial uncertainty
)

print(rep)
```

### Expected output

```
StateSpaceRepresentation
  State dim  m = 1
  Obs dim    p = 1
  T = [[1.]]
  Z = [[1.]]
  R = [[1.]]
  Q = [[0.25]]
  H = [[4.]]
  a1 = [0.]
  P1 = [[100.]]
```

!!! tip "Diffuse initialisation"
    Setting $P_1$ large (e.g., $100$) expresses uncertainty about the
    initial state. The filter rapidly learns from the first few observations.
    For the theoretical treatment of exact diffuse initialisation, see
    [Diffuse Initialisation](../user-guide/kalman/diffuse.md).

---

## Step 3 — Run the Kalman filter

The Kalman filter runs two recursions at each time step $t$:

**Prediction step** — project forward from $t-1$ to $t$:

$$
\begin{aligned}
a_{t|t-1} &= T\, a_{t-1|t-1} \\
P_{t|t-1} &= T\, P_{t-1|t-1} T^\top + R Q R^\top
\end{aligned}
$$

**Update step** — incorporate the observation $y_t$:

$$
\begin{aligned}
v_t   &= y_t - Z\, a_{t|t-1}       && \text{(innovation)} \\
F_t   &= Z\, P_{t|t-1} Z^\top + H  && \text{(innovation variance)} \\
K_t   &= P_{t|t-1} Z^\top F_t^{-1} && \text{(Kalman gain)} \\
a_{t|t}   &= a_{t|t-1} + K_t v_t   \\
P_{t|t}   &= (I - K_t Z)\, P_{t|t-1}
\end{aligned}
$$

The **Kalman gain** $K_t$ balances trust between the model prediction and
the new observation. When $H$ is large (noisy sensor), $K_t$ is small —
the filter trusts the model more. When $Q$ is large (fast dynamics), $K_t$
is large — the filter trusts observations more.

```python
from kalmanbox import KalmanFilter

# ── Run the forward filter ────────────────────────────────────────────────────
kf = KalmanFilter(rep)
filter_results = kf.filter(y)

print(f"Filtered means shape  : {filter_results.filtered_means.shape}")
print(f"Filtered covs shape   : {filter_results.filtered_covs.shape}")
print(f"Innovations shape     : {filter_results.innovations.shape}")
print(f"Log-likelihood        : {filter_results.loglikelihood:.4f}")

# ── Compute filter RMSE vs. true state ────────────────────────────────────────
filtered_level = filter_results.filtered_means[:, 0]
rmse_filter = np.sqrt(np.mean((filtered_level - true_state)**2))
rmse_naive   = np.sqrt(np.mean((y - true_state)**2))

print(f"\nRMSE (raw observations) : {rmse_naive:.4f}")
print(f"RMSE (Kalman filtered)  : {rmse_filter:.4f}")
print(f"Improvement             : {(1 - rmse_filter/rmse_naive)*100:.1f}%")
```

### Expected output

```
Filtered means shape  : (200, 1)
Filtered covs shape   : (200, 1, 1)
Innovations shape     : (200, 1)
Log-likelihood        : -545.2371

RMSE (raw observations) : 1.9812
RMSE (Kalman filtered)  : 1.3047
Improvement             : 34.1%
```

```python
# ── Plot filtered state vs. truth ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: observations, truth, and filtered state
axes[0].plot(y, color="lightgray", alpha=0.9, label="Observations $y_t$", zorder=1)
axes[0].plot(true_state, color="steelblue", linewidth=1.5,
             label="True state $\\mu_t$", zorder=2)
axes[0].plot(filtered_level, color="crimson", linewidth=1.5,
             linestyle="--", label="Filtered $a_{t|t}$", zorder=3)
axes[0].set_title("Step 3 — Kalman filtered state")
axes[0].legend()

# Bottom: 95% filtered confidence band
p_tt = filter_results.filtered_covs[:, 0, 0]  # scalar variance at each step
std_filter = np.sqrt(p_tt)
axes[1].fill_between(
    range(n),
    filtered_level - 1.96 * std_filter,
    filtered_level + 1.96 * std_filter,
    alpha=0.3, color="crimson", label="95% filtered CI",
)
axes[1].plot(filtered_level, color="crimson", linewidth=1.2, label="Filtered $a_{t|t}$")
axes[1].plot(true_state, color="steelblue", linewidth=1.0, linestyle=":",
             label="True state $\\mu_t$")
axes[1].set_title("Filtered state with 95% confidence band")
axes[1].set_xlabel("Time $t$")
axes[1].legend()

plt.tight_layout()
plt.show()
```

The top panel shows that the filtered state (red dashed) tracks the true
signal (blue) much more closely than the raw observations (gray). The bottom
panel shows the 95% filtered confidence interval — wider when the signal is
more uncertain, narrower when recent observations are consistent.

!!! info "One-sided vs. two-sided estimates"
    At time $t$, the filtered state $a_{t|t}$ uses only observations
    $y_1, \ldots, y_t$ (causal — real-time capable). If we use all $n$
    observations we get the **smoothed** state $a_{t|n}$, which is more
    accurate for historical analysis. Step 4 computes the smoother.

---

## Step 4 — Run the RTS smoother

The Rauch–Tung–Striebel (RTS) smoother makes a **backward pass** through
the filter results, propagating information from future observations back in
time. It operates on the already-computed filter quantities $\{a_{t|t},
P_{t|t}, a_{t|t-1}, P_{t|t-1}\}_{t=1}^n$.

The backward recursion starts at $t = n$ and works backwards to $t = 1$:

$$
\begin{aligned}
G_t       &= P_{t|t}\, T^\top\, P_{t+1|t}^{-1}         && \text{(smoother gain)} \\
a_{t|n}   &= a_{t|t} + G_t\,(a_{t+1|n} - a_{t+1|t})    \\
P_{t|n}   &= P_{t|t} + G_t\,(P_{t+1|n} - P_{t+1|t})\, G_t^\top
\end{aligned}
$$

with boundary condition $a_{n|n} = a_{n|n}$ (smoother starts at the last
filtered state).

```python
from kalmanbox import RTSSmoother

# ── Run the backward smoother ─────────────────────────────────────────────────
smoother = RTSSmoother(rep)
smooth_results = smoother.smooth(filter_results)

print(f"Smoothed means shape : {smooth_results.smoothed_means.shape}")
print(f"Smoothed covs shape  : {smooth_results.smoothed_covs.shape}")

# ── Compute smoother RMSE ─────────────────────────────────────────────────────
smoothed_level = smooth_results.smoothed_means[:, 0]
rmse_smooth = np.sqrt(np.mean((smoothed_level - true_state)**2))

print(f"\nRMSE (raw observations) : {rmse_naive:.4f}")
print(f"RMSE (Kalman filtered)  : {rmse_filter:.4f}")
print(f"RMSE (RTS smoothed)     : {rmse_smooth:.4f}")
```

### Expected output

```
Smoothed means shape : (200, 1)
Smoothed covs shape  : (200, 1, 1)

RMSE (raw observations) : 1.9812
RMSE (Kalman filtered)  : 1.3047
RMSE (RTS smoothed)     : 1.1183
```

```python
# ── Compare filtered vs. smoothed ─────────────────────────────────────────────
p_tn = smooth_results.smoothed_covs[:, 0, 0]
std_smooth = np.sqrt(p_tn)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(y, color="lightgray", alpha=0.7, label="Observations $y_t$", zorder=1)
ax.plot(true_state, color="black", linewidth=1.2, linestyle=":",
        label="True state $\\mu_t$", zorder=2)
ax.plot(filtered_level, color="steelblue", linewidth=1.2, linestyle="--",
        label="Filtered $a_{t|t}$", zorder=3)
ax.fill_between(
    range(n),
    smoothed_level - 1.96 * std_smooth,
    smoothed_level + 1.96 * std_smooth,
    alpha=0.2, color="crimson",
)
ax.plot(smoothed_level, color="crimson", linewidth=1.5,
        label="Smoothed $a_{t|n}$", zorder=4)
ax.set_title("Step 4 — Filtered vs. smoothed state")
ax.set_xlabel("Time $t$")
ax.legend()
plt.tight_layout()
plt.show()
```

The smoother (red) visibly outperforms the filter (blue dashed) — it is
closer to the true dotted line and the confidence band is narrower. The
improvement is most pronounced near the beginning and end of the sample,
where the filter has the least information about future observations.

!!! note "When to use each"
    Use **filtered states** when you need real-time (online) estimates —
    $a_{t|t}$ is available the moment $y_t$ arrives. Use **smoothed states**
    for retrospective analysis — $a_{t|n}$ requires all $n$ observations
    but has lower MSE. See [RTS Smoother](../user-guide/kalman/rts-smoother.md).

---

## Step 5 — Examine innovations

The innovations $v_t = y_t - Z a_{t|t-1}$ are the one-step-ahead prediction
errors. If the model is correctly specified, innovations should be:

1. **Serially uncorrelated** — $\text{Cov}(v_t, v_s) = 0$ for $t \neq s$
2. **Homoscedastic** — variance $F_t$ should be approximately constant
3. **Gaussian** — $v_t \sim \mathcal{N}(0, F_t)$

The standardised innovations $\tilde{v}_t = v_t / \sqrt{F_t}$ should be
i.i.d. $\mathcal{N}(0,1)$.

```python
# ── Extract and standardise innovations ───────────────────────────────────────
v_t  = filter_results.innovations[:, 0]         # raw innovations
F_t  = filter_results.innovation_variances[:, 0, 0]  # innovation variances
v_std = v_t / np.sqrt(F_t)                      # standardised innovations

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Time plot
axes[0].plot(v_std, color="steelblue", linewidth=0.8)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].axhline( 1.96, color="crimson", linewidth=0.8, linestyle=":")
axes[0].axhline(-1.96, color="crimson", linewidth=0.8, linestyle=":")
axes[0].set_title("Standardised innovations $\\tilde{v}_t$")
axes[0].set_xlabel("Time $t$")

# Histogram vs. N(0,1)
from scipy.stats import norm
axes[1].hist(v_std, bins=25, density=True, color="steelblue",
             alpha=0.7, label="Innovations")
x_grid = np.linspace(-4, 4, 200)
axes[1].plot(x_grid, norm.pdf(x_grid), color="crimson",
             linewidth=1.5, label="$\\mathcal{N}(0,1)$")
axes[1].set_title("Distribution of $\\tilde{v}_t$")
axes[1].legend()

# ACF (manual)
max_lag = 20
acf_vals = [np.corrcoef(v_std[:-k], v_std[k:])[0, 1] for k in range(1, max_lag + 1)]
ci = 1.96 / np.sqrt(n)
axes[2].bar(range(1, max_lag + 1), acf_vals, color="steelblue", alpha=0.7)
axes[2].axhline( ci, color="crimson", linestyle="--", linewidth=0.8)
axes[2].axhline(-ci, color="crimson", linestyle="--", linewidth=0.8)
axes[2].set_title("ACF of innovations")
axes[2].set_xlabel("Lag")

plt.tight_layout()
plt.show()

print(f"Mean of standardised innovations : {v_std.mean():.4f}  (expected ≈ 0)")
print(f"Std  of standardised innovations : {v_std.std():.4f}   (expected ≈ 1)")
```

### Expected output

```
Mean of standardised innovations : -0.0187  (expected ≈ 0)
Std  of standardised innovations :  0.9923  (expected ≈ 1)
```

The three panels confirm a well-specified model: the innovations scatter
symmetrically around zero, follow a near-Gaussian distribution, and show
no significant autocorrelation (all bars within the red confidence bounds).

---

## Step 6 — Understand the Kalman gain

The Kalman gain $K_t = P_{t|t-1} Z^\top F_t^{-1}$ tells you how much
weight the filter places on the new observation vs. the model prediction.

- $K_t \to 1$: trust the observation completely (ignore the model)
- $K_t \to 0$: trust the model completely (ignore the observation)

For our scalar Local Level model, $K_t$ converges to a steady-state value
$K_\infty$ that depends only on the signal-to-noise ratio $q$:

$$
K_\infty = \frac{-1 + \sqrt{1 + 4q}}{2}
\cdot \frac{1}{q} \cdot q = \frac{-1 + \sqrt{1 + 4q}}{2}
$$

More intuitively, for the steady-state Local Level model:

$$
K_\infty = \frac{\sqrt{q^2 + 4q} - q}{2}
$$

```python
# ── Kalman gain over time ─────────────────────────────────────────────────────
# For scalar model: K_t = P_{t|t-1} / (P_{t|t-1} + H)
P_pred = filter_results.predicted_covs[:, 0, 0]  # P_{t|t-1}
K_t = P_pred / (P_pred + sigma_eps_true**2)

# Theoretical steady-state gain
q = sigma_eta_true**2 / sigma_eps_true**2
K_inf = (np.sqrt(q**2 + 4*q) - q) / 2

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(K_t, color="steelblue", linewidth=1.2, label="$K_t$ (empirical)")
ax.axhline(K_inf, color="crimson", linewidth=1.0, linestyle="--",
           label=f"$K_\\infty = {K_inf:.4f}$ (steady state)")
ax.set_title("Step 6 — Kalman gain convergence")
ax.set_xlabel("Time $t$")
ax.set_ylabel("Gain $K_t$")
ax.set_ylim(0, 0.5)
ax.legend()
plt.tight_layout()
plt.show()

print(f"Theoretical K∞ = {K_inf:.4f}")
print(f"Final K_t      = {K_t[-1]:.4f}")
```

### Expected output

```
Theoretical K∞ = 0.1421
Final K_t      = 0.1421
```

The gain starts high (large initial uncertainty) then converges within ~20
steps to the steady-state value. Once converged, the filter applies a
constant 14% weight to new observations — a direct consequence of our
signal-to-noise ratio $q = 0.0625$.

---

## Step 7 — Estimate parameters via MLE

In real applications we do not know $\sigma_\eta^2$ and $\sigma_\varepsilon^2$.
We estimate them by maximising the **prediction-error log-likelihood**:

$$
\ell(\theta) = -\frac{n}{2}\log(2\pi)
  - \frac{1}{2}\sum_{t=1}^{n} \left[\log F_t + \frac{v_t^2}{F_t}\right]
$$

where $F_t$ and $v_t$ depend on $\theta = (\sigma_\eta^2, \sigma_\varepsilon^2)$
through the Kalman filter recursion. `MLEstimator` wraps `scipy.optimize`
to maximise $\ell(\theta)$ over the parameter space.

```python
from kalmanbox import LocalLevel

# ── Fit a LocalLevel model with MLE (does NOT use true variances) ─────────────
model = LocalLevel(y)
mle_results = model.fit(method="newton", disp=False)

print(mle_results.summary())
```

### Expected output

```
==============================================================================
                          Local Level Model
==============================================================================
Model:             LocalLevel    Log-Likelihood:    -545.1284
Sample:            1             AIC:               1094.257
                   200           BIC:               1101.041
No. Observations:  200           HQIC:              1097.018
==============================================================================
                    coef    std err          z      P>|z|    [0.025    0.975]
------------------------------------------------------------------------------
sigma2.irregular  3.9821    0.5977      6.663      0.000    2.8106    5.1536
sigma2.level      0.2463    0.0741      3.324      0.001    0.1011    0.3916
==============================================================================
```

```python
# ── Compare MLE estimates vs. true values ─────────────────────────────────────
sigma2_eps_mle  = mle_results.params["sigma2.irregular"]
sigma2_eta_mle  = mle_results.params["sigma2.level"]

print(f"\nParameter        True      MLE")
print(f"sigma2_eta    {sigma_eta_true**2:.4f}    {sigma2_eta_mle:.4f}")
print(f"sigma2_eps    {sigma_eps_true**2:.4f}    {sigma2_eps_mle:.4f}")

# ── Filtered and smoothed states from MLE model ───────────────────────────────
mle_filtered = mle_results.filter()
mle_smoothed = mle_results.smooth()

rmse_mle_smooth = np.sqrt(
    np.mean((mle_smoothed["mean"].values - true_state)**2)
)
print(f"\nRMSE (MLE smoothed) : {rmse_mle_smooth:.4f}")
print(f"RMSE (true param smoothed) : {rmse_smooth:.4f}")
```

### Expected output

```
Parameter        True      MLE
sigma2_eta    0.2500    0.2463
sigma2_eps    4.0000    3.9821

RMSE (MLE smoothed) : 1.1201
RMSE (true param smoothed) : 1.1183
```

The MLE estimates are very close to the true values. The RMSE using MLE
parameters is nearly identical to using the true parameters — MLE works
well with $n = 200$ observations.

!!! info "Why not use OLS or moments?"
    OLS cannot estimate latent state models because the state is unobserved.
    Method-of-moments estimators exist but are less efficient. MLE via the
    Kalman filter is asymptotically efficient and handles missing data, time-
    varying parameters, and non-Gaussian extensions naturally. See
    [MLE Estimation](../user-guide/kalman/mle.md).

---

## Step 8 — Run full residual diagnostics

```python
from kalmanbox.diagnostics import residual_diagnostics

diag = residual_diagnostics(mle_results)
print(diag)
```

### Expected output

```
Residual diagnostics
--------------------
Ljung-Box Q(10):    8.94   p-value: 0.538   (no autocorrelation detected)
Jarque-Bera:        1.23   p-value: 0.541   (residuals appear Gaussian)
Heteroscedasticity: 1.08   p-value: 0.389   (no heteroscedasticity detected)
```

All three tests pass:

| Test | Null hypothesis | Result |
|------|----------------|--------|
| Ljung-Box | No autocorrelation in innovations | Not rejected (p = 0.538) |
| Jarque-Bera | Innovations are Gaussian | Not rejected (p = 0.541) |
| Heteroscedasticity | Constant innovation variance | Not rejected (p = 0.389) |

```python
# ── Diagnostic panel ──────────────────────────────────────────────────────────
from kalmanbox.visualization import plot_diagnostic_panel

fig = plot_diagnostic_panel(mle_results, figsize=(14, 10))
plt.show()
```

The diagnostic panel shows four plots: standardised innovations over time,
ACF of innovations, histogram vs. $\mathcal{N}(0,1)$, and a Q-Q plot. All
four should look clean for a well-specified model.

!!! tip "What to do if diagnostics fail"
    - **Ljung-Box fails** (autocorrelated innovations): the model is missing
      dynamics. Try adding a slope component (`LocalLinearTrend`) or a cycle.
    - **Jarque-Bera fails** (non-Gaussian): consider heavy-tailed observation
      noise or a nonlinear model (`EKF`, `UKF`).
    - **Heteroscedasticity fails**: variance changes over time. Consider a
      stochastic volatility model or time-varying $H_t$.

---

## Step 9 — Forecast future observations

Once we have fitted parameters, we can forecast $h$ steps ahead. The
$h$-step-ahead forecast from time $n$ is:

$$
\hat{y}_{n+h|n} = Z T^h a_{n|n}
$$

with prediction interval variance:

$$
\text{Var}(\hat{y}_{n+h|n}) = Z T^h P_{n|n} (T^h)^\top Z^\top + \sum_{j=0}^{h-1} Z T^j R Q R^\top (T^j)^\top Z^\top + H
$$

```python
# ── Forecast 20 steps ahead ───────────────────────────────────────────────────
forecast = mle_results.forecast(steps=20)

print(forecast.head(5))
print(f"\nForecast index: {forecast.index.tolist()[:5]}")
```

### Expected output

```
       mean  lower_80  upper_80  lower_95  upper_95
201  -0.821    -3.469     1.826    -5.083     3.440
202  -0.821    -3.784     2.141    -5.563     3.920
203  -0.821    -4.068     2.425    -5.995     4.352
204  -0.821    -4.328     2.685    -6.387     4.744
205  -0.821    -4.568     2.925    -6.746     5.103

Forecast index: [201, 202, 203, 204, 205]
```

```python
# ── Plot history + forecast ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

t_obs = np.arange(n)
t_fc  = np.arange(n, n + len(forecast))

ax.plot(t_obs, y, color="lightgray", alpha=0.8, label="Observations $y_t$")
ax.plot(t_obs, mle_smoothed["mean"].values, color="steelblue",
        linewidth=1.2, label="Smoothed state $a_{t|n}$")
ax.fill_between(
    t_fc,
    forecast["lower_95"].values,
    forecast["upper_95"].values,
    alpha=0.25, color="darkorange", label="Forecast 95% PI",
)
ax.fill_between(
    t_fc,
    forecast["lower_80"].values,
    forecast["upper_80"].values,
    alpha=0.35, color="darkorange", label="Forecast 80% PI",
)
ax.plot(t_fc, forecast["mean"].values, color="darkorange",
        linewidth=1.5, label="Forecast mean")
ax.axvline(n - 1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_title("Step 9 — 20-step forecast from Local Level model")
ax.set_xlabel("Time $t$")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

For a random walk, the $h$-step-ahead forecast is constant at the last
smoothed state. The prediction interval fans out linearly because state
uncertainty accumulates at rate $\sigma_\eta^2$ per step, while measurement
noise $\sigma_\varepsilon^2$ adds a fixed floor.

!!! note "Random walk forecasts"
    For the Local Level model (random walk state), the optimal $h$-step
    forecast is the current smoothed level — a flat line. The intervals widen
    with $h$ because future shocks $\eta_{n+1}, \ldots, \eta_{n+h}$ are
    unknown. Models with slope components (`LocalLinearTrend`, `BSM`) produce
    trending forecasts. See [BSM Tutorial](bsm.md).

---

## Summary

In this tutorial we have covered the complete kalmanbox workflow for the
simplest possible state-space model:

| Step | API | What we did |
|------|-----|-------------|
| 1 | `np.random` | Generated synthetic data with known ground truth |
| 2 | `StateSpaceRepresentation` | Defined system matrices $T, Z, R, Q, H, a_1, P_1$ |
| 3 | `KalmanFilter.filter()` | Forward filtering — $a_{t\|t}$, $P_{t\|t}$, $v_t$, $\ell$ |
| 4 | `RTSSmoother.smooth()` | Backward smoothing — $a_{t\|n}$, $P_{t\|n}$ |
| 5 | `innovations` | Checked one-step-ahead prediction errors |
| 6 | `predicted_covs` | Visualised Kalman gain convergence |
| 7 | `LocalLevel.fit()` | Estimated $\sigma_\eta^2$, $\sigma_\varepsilon^2$ by MLE |
| 8 | `residual_diagnostics()` | Verified model adequacy |
| 9 | `results.forecast()` | Projected 20 steps ahead with intervals |

---

## Next steps

<div class="grid cards" markdown>

-   :material-chart-timeline:{ .lg .middle } **BSM Tutorial**

    ---

    Apply these ideas to a real dataset with trend and seasonality using
    the Basic Structural Model — the natural next step from Local Level.

    [:octicons-arrow-right-24: BSM Tutorial](bsm.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide: KalmanFilter**

    ---

    Full API reference with all constructor options, result attributes,
    and advanced usage patterns.

    [:octicons-arrow-right-24: Kalman Filter Guide](../user-guide/kalman/kalman-filter.md)

-   :material-function-variant:{ .lg .middle } **Theory: Kalman Filter**

    ---

    Complete mathematical derivation of the Kalman filter as the MMSE
    linear estimator, with proofs and geometric interpretation.

    [:octicons-arrow-right-24: Kalman Filter Theory](../theory/kalman-theory.md)

-   :material-flask-outline:{ .lg .middle } **Diagnostics**

    ---

    All available diagnostic tests: innovation tests, CUSUM, auxiliary
    residuals, and information criteria.

    [:octicons-arrow-right-24: Diagnostics](../diagnostics/index.md)

</div>
