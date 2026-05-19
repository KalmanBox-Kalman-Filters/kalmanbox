# EM Algorithm for State-Space Models

The **Expectation-Maximization (EM) algorithm** is an iterative procedure for
maximum likelihood estimation when the model contains **latent (unobserved)
variables**. In state-space models the latent variables are the hidden states
$\alpha_t$, which are never observed directly. EM alternates between two steps
until convergence:

- **E-step** — compute the *expected sufficient statistics* of the complete data
  given the current parameter estimates and all observations. For linear Gaussian
  SSMs this is exactly the **Kalman smoother**.
- **M-step** — update the parameters by maximizing the *expected
  complete-data log-likelihood*. For linear Gaussian SSMs this step has
  **closed-form solutions**.

Because each M-step is solved analytically, EM avoids the numerical gradient
evaluations that can make direct MLE fragile in high-dimensional problems — a
key advantage for Dynamic Factor Models with many series.

!!! note "Scope of this page"
    The derivations below cover the general linear Gaussian SSM and specialize
    to the Dynamic Factor Model (DFM) case for the M-step formulas, since that
    is the primary use case in `kalmanbox`. Structural models such as
    [UCM](../structural/ucm.md) can also be fitted via EM; an example is given
    in [Section 7](#em-for-ucm).

---

## 1. Complete-Data Log-Likelihood

Consider the general linear Gaussian state-space model

$$
\begin{aligned}
\alpha_{t+1} &= T\,\alpha_t + R\,\eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)\\
y_t          &= Z\,\alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H)
\end{aligned}
$$

with parameter vector $\theta = (T, Z, Q, H, a_0, P_0)$.

If the states $\{\alpha_t\}_{t=1}^{T}$ were **observed**, the complete-data
log-likelihood would factor over the measurement and transition densities:

$$
\ell_c(\theta)
= -\frac{1}{2}\sum_{t=1}^{T}\Bigl[
      \log|H| + \varepsilon_t' H^{-1} \varepsilon_t
   \Bigr]
  -\frac{1}{2}\sum_{t=1}^{T-1}\Bigl[
      \log|Q| + \eta_t' Q^{-1} \eta_t
   \Bigr]
  + \text{const}
$$

where

$$
\varepsilon_t = y_t - Z\,\alpha_t, \qquad \eta_t = \alpha_{t+1} - T\,\alpha_t.
$$

Since the states are latent, the EM algorithm maximizes the **Q-function** —
the expectation of $\ell_c(\theta)$ taken over the states conditional on the
observations and the current parameter iterate $\theta^{(j)}$:

$$
\mathcal{Q}(\theta,\theta^{(j)})
= E_{\alpha|\,Y_T;\,\theta^{(j)}}\!\left[\ell_c(\theta)\right].
$$

---

## 2. E-Step: Kalman Smoother Sufficient Statistics

The E-step evaluates the Q-function by running the **RTS smoother** (see
[Kalman Smoother](../kalman/rts-smoother.md)) to obtain the following
conditional moments, which appear as sufficient statistics in the Q-function.

**Smoothed states and covariances:**

$$
E[\alpha_t \mid Y_T;\,\theta^{(j)}] = \hat{\alpha}_{t|T}
$$

$$
E[\alpha_t\,\alpha_t' \mid Y_T;\,\theta^{(j)}]
= P_{t|T} + \hat{\alpha}_{t|T}\,\hat{\alpha}_{t|T}'
$$

**Cross-covariance (lag-one smoother):**

$$
E[\alpha_{t+1}\,\alpha_t' \mid Y_T;\,\theta^{(j)}]
= P_{t+1,\,t\,|\,T} + \hat{\alpha}_{t+1|T}\,\hat{\alpha}_{t|T}'
$$

where $P_{t+1,\,t\,|\,T}$ is the **lag-one smoother cross-covariance** obtained
from the backward RTS pass:

$$
P_{t+1,\,t\,|\,T} = P_{t+1|T}\,L_t' + J_{t+1}\,(P_{t+1,t|T}^{*} - T\,P_{t|t})\,J_t'
$$

with $J_t = P_{t|t}\,T'\,P_{t+1|t}^{-1}$ being the RTS gain.

!!! note "Why the smoother?"
    The Kalman *filter* only provides $E[\alpha_t \mid Y_t]$ (filtered
    estimates using data up to $t$). The EM algorithm needs
    $E[\alpha_t \mid Y_T]$ — expectations conditioned on **all** observations
    — which requires the full backward RTS smoother pass.

Collecting notation, define the aggregated sufficient statistics:

$$
S_{11} = \sum_{t=1}^{T}\bigl(P_{t|T} + \hat{\alpha}_{t|T}\,\hat{\alpha}_{t|T}'\bigr)
$$

$$
S_{10} = \sum_{t=2}^{T}\bigl(P_{t,t-1|T} + \hat{\alpha}_{t|T}\,\hat{\alpha}_{t-1|T}'\bigr)
$$

$$
S_{00} = \sum_{t=1}^{T-1}\bigl(P_{t|T} + \hat{\alpha}_{t|T}\,\hat{\alpha}_{t|T}'\bigr)
$$

$$
S_{y\alpha} = \sum_{t=1}^{T} y_t\,\hat{\alpha}_{t|T}'
$$

These four matrices fully characterize the Q-function and are the only
quantities passed from the E-step to the M-step.

---

## 3. M-Step: Closed-Form Parameter Updates

Given the sufficient statistics from the E-step, the M-step maximizes
$\mathcal{Q}(\theta,\theta^{(j)})$ analytically. For the **Dynamic Factor Model**
(DFM) with $Z \equiv \Lambda$ (factor loadings) and diagonal $H$, the updates
are as follows.

### Factor Loadings ($\Lambda$)

$$
\hat{\Lambda}
= \left(\sum_{t=1}^{T} y_t\,\hat{f}_{t|T}'\right)
  \left(\sum_{t=1}^{T}\bigl(P_{t|T}^{ff} + \hat{f}_{t|T}\,\hat{f}_{t|T}'\bigr)\right)^{-1}
= S_{y\alpha}\,S_{11}^{-1}
$$

where $P_{t|T}^{ff}$ is the factor block of the smoothed state covariance.

### Idiosyncratic Variance ($H$, diagonal)

Each diagonal element $H_{ii}$ is updated independently:

$$
\hat{H}_{ii}
= \frac{1}{T}\sum_{t=1}^{T}
  \left[
    y_{it}^2
    - 2\,y_{it}\,\hat{\Lambda}_{i\cdot}\,\hat{f}_{t|T}
    + \hat{\Lambda}_{i\cdot}\bigl(P_{t|T}^{ff} + \hat{f}_{t|T}\,\hat{f}_{t|T}'\bigr)\hat{\Lambda}_{i\cdot}'
  \right]
$$

Compactly, for the full diagonal:

$$
\hat{H} = \frac{1}{T}\,\operatorname{diag}\!\left(
  \sum_{t=1}^{T} y_t y_t'
  - \hat{\Lambda}\, S_{y\alpha}'
\right)
$$

### Factor Transition Matrix ($\Phi \equiv T$)

$$
\hat{\Phi}
= \left(\sum_{t=2}^{T}\bigl(P_{t,t-1|T}^{ff} + \hat{f}_{t|T}\,\hat{f}_{t-1|T}'\bigr)\right)
  \left(\sum_{t=2}^{T}\bigl(P_{t-1|T}^{ff} + \hat{f}_{t-1|T}\,\hat{f}_{t-1|T}'\bigr)\right)^{-1}
= S_{10}\,S_{00}^{-1}
$$

### State Noise Covariance ($Q$)

$$
\hat{Q}
= \frac{1}{T-1}\sum_{t=2}^{T}\left[
    P_{t|T}^{ff} + \hat{f}_{t|T}\hat{f}_{t|T}'
    - \hat{\Phi}\bigl(P_{t,t-1|T}^{ff} + \hat{f}_{t|T}\hat{f}_{t-1|T}'\bigr)
  \right]
$$

Equivalently in matrix form:

$$
\hat{Q}
= \frac{1}{T-1}\!\left(S_{11}^{(-1)} - \hat{\Phi}\,S_{10}'\right)
$$

where $S_{11}^{(-1)}$ denotes $S_{11}$ summed from $t=2$ to $T$.

### Initial State ($a_0$, $P_0$)

When the initial state is treated as an unknown parameter:

$$
\hat{a}_0 = \hat{\alpha}_{1|T}, \qquad
\hat{P}_0 = P_{1|T}
$$

For non-stationary models kalmanbox uses diffuse initialization and does not
update $P_0$ in the M-step; see [Diffuse Initialization](../kalman/diffuse.md).

!!! note "General SSM M-step"
    For a general (non-DFM) linear Gaussian SSM where $Z$ is not the factor
    loading matrix and $H$ need not be diagonal, the update for $Z$ is:
    $$\hat{Z} = \left(\sum_{t=1}^T y_t\,\hat{\alpha}_{t|T}'\right) S_{11}^{-1}$$
    and for $T$: $\hat{T} = S_{10}\,S_{00}^{-1}$. The structure of the
    M-step formulas is the same; the difference lies in which parameters are
    free and which constraints apply.

---

## 4. Convergence

### Monotone Likelihood Increase

A key theoretical property of EM is that **the observed-data log-likelihood is
non-decreasing at every iteration**:

$$
\ell(\theta^{(j+1)}; Y_T) \;\geq\; \ell(\theta^{(j)}; Y_T)
$$

This follows directly from the information inequality applied to the Q-function.
EM cannot decrease the likelihood — a major advantage over gradient-based methods
that can overshoot.

### Convergence Criterion

kalmanbox uses a **relative log-likelihood tolerance**:

$$
\left|\ell(\theta^{(j+1)}) - \ell(\theta^{(j)})\right|
< \texttt{tol} \times \left|\ell(\theta^{(j)})\right|
$$

Typical values:

| Use case | Recommended `tol` | Typical iterations |
|----------|-------------------|--------------------|
| Quick exploration | `1e-4` | 20–80 |
| Standard estimation | `1e-6` | 50–200 |
| Publication-quality | `1e-8` | 100–500 |
| Very large DFM ($p > 50$) | `1e-6` | 100–300 |

### Convergence Plot

A healthy EM run shows rapid early gains followed by slow final refinement.
The curve should be monotonically increasing and level off smoothly.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Full convergence trajectory
axes[0].plot(results.llf_history, color="steelblue")
axes[0].set_xlabel("EM iteration")
axes[0].set_ylabel("Log-likelihood")
axes[0].set_title("Convergence trajectory")
axes[0].grid(True, alpha=0.3)

# Per-iteration improvement (log scale)
deltas = [abs(results.llf_history[i+1] - results.llf_history[i])
          for i in range(len(results.llf_history) - 1)]
axes[1].semilogy(deltas, color="tomato")
axes[1].set_xlabel("EM iteration")
axes[1].set_ylabel(r"$|\Delta\,\ell|$  (log scale)")
axes[1].set_title("Per-iteration improvement")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Local Maxima

EM is guaranteed to converge to a **stationary point** of the likelihood, but
not necessarily the global maximum. Practical strategies:

- **Multiple random restarts** (`n_starts` parameter) — run EM from several
  different initializations and keep the solution with the highest
  log-likelihood.
- **PCA warm start** — kalmanbox initializes factor loadings via PCA, which
  typically lands near a good basin of attraction.
- **Annealing** — start with an inflated $Q$ and gradually tighten; this
  prevents early convergence to degenerate solutions.

---

## 5. EMEstimator API

### Fitting a DFM via EM

The simplest entry point is `DFM.fit(method="em")`, which internally
constructs an `EMEstimator`:

```python
import numpy as np
from kalmanbox.advanced import DFM, EMEstimator

# Simulate panel data: 200 obs, 10 series, 3 latent factors
rng = np.random.default_rng(42)
T, p, k = 200, 10, 3
F_true = rng.standard_normal((T, k))
Lambda_true = rng.standard_normal((p, k))
y = F_true @ Lambda_true.T + 0.5 * rng.standard_normal((T, p))

# Option 1: fit via model method (recommended)
model = DFM(y, k_factors=3, factor_order=1)
results = model.fit(method="em", maxiter=300, tol=1e-8, n_starts=5)

print(f"Log-likelihood : {results.llf:.4f}")
print(f"EM iterations  : {results.em_iterations}")
print(f"Converged      : {results.converged}")
print(f"Final params   :")
print(f"  Lambda shape : {results.params['Lambda'].shape}")
print(f"  Q diagonal   : {np.diag(results.params['Q'])}")
print(f"  H diagonal   : {np.diag(results.params['H'])}")
```

### Using EMEstimator Directly

`EMEstimator` can be applied to **any** `kalmanbox` state-space model, not
only DFMs:

```python
from kalmanbox.advanced import EMEstimator
from kalmanbox.structural import UCM

# Build a UCM
ucm_model = UCM(y, level=True, slope=True, seasonal=12, irregular=True)

# Wrap with EMEstimator and fit
em = EMEstimator(ucm_model)
results = em.fit(maxiter=200, tol=1e-7, verbose=True)

# Access results
print(f"Converged in {results.em_iterations} iterations")
print(f"Final log-likelihood: {results.llf:.6f}")

# Plot convergence
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(results.llf_history, marker="o", markersize=3)
plt.xlabel("EM iteration")
plt.ylabel("Log-likelihood")
plt.title("EM convergence — UCM")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. Monitoring Convergence

Use the `callback` parameter to inspect every iteration without storing all
intermediate parameters:

```python
from kalmanbox.advanced import EMEstimator

em = EMEstimator(model)

# Callback receives (iteration, current_llf, delta_llf)
results = em.fit(
    maxiter=500,
    tol=1e-8,
    callback=lambda it, llf, delta: print(
        f"Iter {it:4d}: llf = {llf:14.6f},  delta = {delta:.2e}"
    ),
)

# Inspect final state
if not results.converged:
    print(
        f"WARNING: EM did not converge after {results.em_iterations} iterations.\n"
        f"  Final delta = {results.llf_history[-1] - results.llf_history[-2]:.2e}\n"
        "  Consider increasing maxiter or checking the initialization."
    )
else:
    print(f"Converged in {results.em_iterations} iterations.")
    print(f"Log-likelihood: {results.llf:.6f}")
```

!!! warning "Non-convergence"
    If `results.converged` is `False`, the solution is still the best found so
    far — it may be usable but should be treated with caution. Common causes:
    (1) `maxiter` too small, (2) near-degenerate initialization, (3) model
    misspecification. Try increasing `n_starts` before raising `maxiter`.

---

## 7. EM vs. Direct MLE

| Aspect | EM Algorithm | Direct MLE (L-BFGS-B) |
|--------|-------------|-----------------------|
| **Convergence type** | Monotone, non-decreasing $\ell$ | Non-monotone, may overshoot |
| **Speed per iteration** | Slow — full smoother pass | Fast — gradient + Hessian step |
| **Iterations to convergence** | Many (50–500) | Few (10–50) |
| **Robustness** | High; closed-form M-step | Lower; sensitive to initialization |
| **Large $p$ (many series)** | Preferred | May hit numerical issues |
| **Large $k$ (many factors)** | Preferred | Gradient dim grows as $p \times k$ |
| **Closed-form M-step** | Yes (linear Gaussian) | No |
| **Global maximum** | Not guaranteed (local EM) | Not guaranteed |
| **Gradient required** | No | Yes |
| **Implementation complexity** | Moderate | Lower (off-the-shelf optimizer) |

!!! tip "Rule of thumb"
    Use **EM** for DFMs with $p > 5$ series or $k > 2$ factors — the
    closed-form M-step is far more stable than gradient optimization in high
    dimensions. Use **direct MLE** for simple structural models
    ([LocalLevel](../structural/local-level.md),
    [BSM](../structural/bsm.md), [UCM](../structural/ucm.md)) where the
    parameter space is small and gradient methods converge quickly.

---

## 8. EM for UCM — Full Example

```python
from kalmanbox.advanced import EMEstimator
from kalmanbox.structural import UCM
from kalmanbox.datasets import load_gdp

# Load quarterly GDP growth
y = load_gdp()["gdp_growth"].to_numpy()

# UCM: trend (level + slope) + quarterly seasonal + irregular
model = UCM(
    y,
    level=True,
    slope=True,
    seasonal=4,       # quarterly seasonality
    irregular=True,
)

# --- Fit with EM ---
em = EMEstimator(model)
results_em = em.fit(maxiter=300, tol=1e-8)

# --- Fit with direct MLE for comparison ---
results_mle = model.fit(method="mle", n_starts=10)

print(f"EM  log-lik: {results_em.llf:.4f}   "
      f"({results_em.em_iterations} iterations)")
print(f"MLE log-lik: {results_mle.llf:.4f}   "
      f"({'converged' if results_mle.success else 'not converged'})")

# Decompose the signal using EM estimates
smoothed = results_em.smooth()
trend    = smoothed.states["level"]
seasonal = smoothed.states["seasonal"]
cycle    = y - trend - seasonal

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(y,        label="Observed",  alpha=0.6)
axes[0].plot(trend,    label="Trend",     linewidth=2)
axes[0].legend(); axes[0].set_title("GDP growth — EM decomposition")
axes[1].plot(seasonal, color="orange"); axes[1].set_title("Seasonal")
axes[2].plot(cycle,    color="green");  axes[2].set_title("Irregular")
plt.tight_layout()
plt.show()
```

---

## 9. Initialization

Poor initialization is the most common source of problems with EM for
state-space models.

### Default: PCA Warm Start

kalmanbox initializes the DFM parameters using **Principal Component Analysis**:

1. Compute the first $k$ principal components of $Y$ as initial factor
   estimates $\hat{F}^{(0)}$.
2. Set $\hat{\Lambda}^{(0)}$ to the corresponding loadings (eigenvectors scaled
   by eigenvalues).
3. Fit a VAR(1) to $\hat{F}^{(0)}$ to initialize $\hat{\Phi}^{(0)}$.
4. Set $\hat{H}^{(0)} = \operatorname{diag}(\text{residual variances})$.

This gives EM a good starting basin and avoids the degenerate $Q = 0$ trap.

### Multiple Random Restarts

```python
model = DFM(y, k_factors=3, factor_order=1)

# n_starts=10: run EM 10 times with different random initializations,
# keep the solution with highest log-likelihood
results = model.fit(method="em", maxiter=200, tol=1e-8, n_starts=10)

print(f"Best llf across {results.n_starts} starts: {results.llf:.4f}")
print(f"Start that won: #{results.best_start}")
```

### Annealing Initialization

For particularly ill-conditioned problems, start with an inflated state noise
and gradually reduce it:

```python
from kalmanbox.advanced import EMEstimator

em = EMEstimator(model)

# Phase 1: loose tolerance, inflated Q — get into a good basin
results_warm = em.fit(maxiter=50, tol=1e-3, q_scale=10.0)

# Phase 2: refine from warm start
results_final = em.fit(
    maxiter=500,
    tol=1e-8,
    init_params=results_warm.params,
)
```

---

## 10. Degenerate Solutions and Pitfalls

!!! warning "Variance collapse ($Q \to 0$)"
    A common failure mode: one or more diagonal entries of $\hat{Q}$ collapse
    to zero, making the corresponding factor effectively **static**. The model
    then becomes a static factor model — often a reasonable fit but not what
    was intended.

    **Signs:** diagonal entries of $\hat{Q}$ are $< 10^{-8}$, log-likelihood
    barely increases after the first few iterations.

    **Solutions:**

    1. **Lower bound on $Q$** — pass `q_lower=1e-6` to `EMEstimator` to clip
       the M-step update.
    2. **Multiple starts** — try `n_starts=10` or more.
    3. **Informative initialization** — use `q_scale` to inflate the initial
       $Q$ and let EM find the right scale.
    4. **Bayesian EM** — place an inverse-Wishart prior on $Q$; see
       [Bayesian estimation](../bayesian/index.md).

!!! warning "Identification failures"
    DFMs are only identified up to rotation of the factor space. kalmanbox
    applies a standard normalization (lower-triangular $\Lambda$ with positive
    diagonal), but you may still see sign-flips or permutations across
    restarts. This does not affect the fitted values or the likelihood.

!!! note "Missing data"
    EM handles missing observations **automatically** — the E-step simply skips
    the update equations for time points where $y_t$ is missing (or partially
    missing in the multivariate case). No imputation is needed. See
    [Missing Data](../kalman/missing-data.md) for details.

---

## 11. API Reference

::: kalmanbox.advanced.EMEstimator
    options:
      heading_level: 3
      show_source: false

---

## 12. Related Pages

- [Dynamic Factor Model](dfm.md) — primary use case for EM in kalmanbox
- [MLE](../kalman/mle.md) — direct optimization alternative to EM
- [Kalman Smoother (RTS)](../kalman/rts-smoother.md) — the engine of the E-step
- [Diffuse Initialization](../kalman/diffuse.md) — handling non-stationary models
- [Missing Data](../kalman/missing-data.md) — automatic handling in EM
- [Bayesian Estimation](../bayesian/index.md) — alternative to frequentist EM
- [UCM](../structural/ucm.md) — structural models fittable via EM

---

## 13. References

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).
**Maximum Likelihood from Incomplete Data via the EM Algorithm.**
*Journal of the Royal Statistical Society, Series B*, 39(1), 1–38.

Shumway, R. H., & Stoffer, D. S. (1982).
**An Approach to Time Series Smoothing and Forecasting Using the EM Algorithm.**
*Journal of Time Series Analysis*, 3(4), 253–264.

Watson, M. W., & Engle, R. F. (1983).
**Alternative Algorithms for the Estimation of Dynamic Factor, MIMIC, and
Varying Coefficient Regression Models.**
*Journal of Econometrics*, 23(3), 385–400.

Doz, C., Giannone, D., & Reichlin, L. (2012).
**A Quasi-Maximum Likelihood Approach for Large, Approximate Dynamic Factor
Models.**
*Review of Economics and Statistics*, 94(4), 1014–1024.

Bork, L. (2009).
**Estimating US Monetary Policy Shocks Using a Factor-Augmented Vector
Autoregression: An EM Algorithm Approach.**
CREATES Research Paper 2009-11.
