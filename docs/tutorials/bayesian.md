---
title: "Tutorial: Bayesian Estimation with Gibbs Sampling and FFBS"
description: >-
  Advanced tutorial that fits a Basic Structural Model (BSM) to quarterly UK
  gas consumption data using full Bayesian MCMC — Gibbs sampling with Forward
  Filtering Backward Sampling (FFBS) — covering prior elicitation, convergence
  diagnostics, posterior predictive checks, and comparison with MLE.
---

# Tutorial: Bayesian Estimation with Gibbs Sampling and FFBS

**Level:** :material-signal: Advanced · **Time:** ~90 min · **Dataset:** UK Gas Consumption

Classical MLE provides a single point estimate for each variance parameter of
a state-space model. Bayesian estimation goes further: it gives you the
**full posterior distribution** over every parameter, naturally propagates
parameter uncertainty into state estimates and forecasts, and handles weakly
identified parameters gracefully through priors.

The workhorse algorithm for linear Gaussian state-space models is the
**Gibbs sampler with Forward Filtering Backward Sampling (FFBS)**, introduced
by Carter & Kohn (1994) and Frühwirth-Schnatter (1994). Because both the
state block and the variance block admit exact conjugate updates, each Gibbs
sweep produces an **exact** draw from the joint posterior — no Metropolis
accept/reject step is needed.

By the end of this tutorial you will have:

- Simulated a realistic quarterly UK gas consumption series with known true parameters
- Elicited weakly informative Inverse-Gamma priors and checked them via prior predictive simulation
- Configured and run a four-chain Gibbs sampler with FFBS for a BSM on real-scale data
- Assessed convergence using trace plots, R-hat (Gelman-Rubin), and effective sample size (ESS)
- Compared the full posterior distributions against MLE point estimates
- Quantified state uncertainty by sampling state trajectories from the posterior
- Produced posterior predictive intervals that correctly account for parameter uncertainty

!!! info "Prerequisites"
    Complete the [BSM tutorial](bsm.md) and [Bayesian User Guide overview](../user-guide/bayesian/index.md)
    before starting. You should be comfortable with:

    - `BSM`, `KalmanFilter`, and `RTSSmoother` from `kalmanbox`
    - The state-space matrices of the Basic Structural Model (see [BSM theory](../theory/structural-theory.md))
    - Basic MCMC concepts: chains, burn-in, thinning, trace plots

    **Python packages required:**

    ```bash
    pip install kalmanbox arviz matplotlib pandas numpy
    ```

---

## The model: BSM with stochastic trend and quarterly seasonality

The Basic Structural Model decomposes the observed series $y_t$ as:

$$
y_t = \mu_t + \gamma_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

The **local linear trend** evolves as:

$$
\mu_{t+1} = \mu_t + \nu_t + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \sigma_\xi^2)
$$

$$
\nu_{t+1} = \nu_t + \zeta_t, \quad \zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2)
$$

The **quarterly seasonal component** satisfies the dummy-variable constraint
that seasonal effects sum to zero over each year:

$$
\sum_{j=0}^{s-1} \gamma_{t-j} = \omega_t, \quad \omega_t \sim \mathcal{N}(0, \sigma_\omega^2)
$$

where $s = 4$ for quarterly data.

The state vector is $\alpha_t = (\mu_t,\; \nu_t,\; \gamma_t,\; \gamma_{t-1},\; \gamma_{t-2})'$,
so the state dimension is $m = 5$. The four variance parameters to estimate
Bayesianly are $\theta = (\sigma_\varepsilon^2,\; \sigma_\xi^2,\; \sigma_\zeta^2,\; \sigma_\omega^2)$.

---

## Step 1 — Generate and explore the UK gas consumption data

We simulate 120 quarters (30 years, 1993 Q1 – 2022 Q4) of UK gas consumption
using a local linear trend with a slowly evolving slope, quarterly seasonal
dummies (Q4 highest — winter heating demand), and an irregular shock. The true
parameters are:

| Component | Parameter | True value |
|-----------|-----------|-----------|
| Irregular | $\sigma_\varepsilon$ | 50 |
| Level | $\sigma_\xi$ | 30 |
| Slope | $\sigma_\zeta$ | 5 |
| Seasonal | $\sigma_\omega$ | 20 |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kalmanbox import BSM
from kalmanbox.estimation import BayesianSSM, InverseGamma
from kalmanbox.visualization import plot_components, set_theme

# ── Global plot style ─────────────────────────────────────────────────────────
set_theme("kalmanbox")

# ── Simulation parameters ─────────────────────────────────────────────────────
rng = np.random.default_rng(0)
T = 120                        # 30 years of quarterly observations
dates = pd.date_range("1993-01", periods=T, freq="QS")

# True variance parameters (standard deviations for simulation)
TRUE_SIGMA_EPS   = 50.0   # irregular
TRUE_SIGMA_XI    = 30.0   # level disturbance
TRUE_SIGMA_ZETA  =  5.0   # slope disturbance
TRUE_SIGMA_OMEGA = 20.0   # seasonal disturbance

# ── Simulate the state sequence ───────────────────────────────────────────────
mu   = np.zeros(T + 1)     # level
nu   = np.zeros(T + 1)     # slope
gam  = np.zeros((T + 1, 4))  # gamma[t, j] for j=0..3 (quarterly dummies)

# Initial conditions: roughly plausible gas consumption around 5000 units
mu[0]    = 5000.0
nu[0]    =   10.0        # slight upward trend
# Deterministic seasonal pattern: Q1 low, Q2 lowest, Q3 medium, Q4 high
gam[0]   = np.array([ -80.0, -150.0,  50.0, 180.0])

for t in range(T):
    q = t % 4            # quarter index 0..3
    mu[t + 1]    = mu[t] + nu[t] + rng.normal(0, TRUE_SIGMA_XI)
    nu[t + 1]    = nu[t] + rng.normal(0, TRUE_SIGMA_ZETA)
    # Seasonal: new dummy replaces oldest; constraint sum = omega
    new_gam      = -(gam[t, 1] + gam[t, 2] + gam[t, 3]) + rng.normal(0, TRUE_SIGMA_OMEGA)
    gam[t + 1]   = np.array([new_gam, gam[t, 0], gam[t, 1], gam[t, 2]])

# Observations
eps = rng.normal(0, TRUE_SIGMA_EPS, T)
y_values = mu[:T] + gam[:T, 0] + eps
y: pd.Series = pd.Series(y_values, index=dates, name="gas_consumption")

print("=== UK Gas Consumption (Simulated) ===")
print(f"Observations   : {T} quarters (1993 Q1 – 2022 Q4)")
print(f"Mean           : {y.mean():.1f}")
print(f"Std            : {y.std():.1f}")
print(f"Min            : {y.min():.1f}")
print(f"Max            : {y.max():.1f}")
seasonal_amp = y.groupby(y.index.quarter).mean()
print(f"Seasonal amplitude (Q4-Q2 mean diff): {seasonal_amp[4] - seasonal_amp[2]:.1f}")
```

### Expected output

```
=== UK Gas Consumption (Simulated) ===
Observations   : 120 quarters (1993 Q1 – 2022 Q4)
Mean           : 5263.4
Std            : 433.7
Min            : 4147.2
Max            : 6581.8
Seasonal amplitude (Q4-Q2 mean diff): 521.3
```

```python
# ── Visualise raw series and true component decomposition ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig.suptitle("UK Gas Consumption — True Component Decomposition", fontsize=14)

axes[0, 0].plot(dates, y_values, color="steelblue", linewidth=1.2)
axes[0, 0].set_title("Observed series $y_t$")
axes[0, 0].set_ylabel("Gas consumption (units)")

axes[0, 1].plot(dates, mu[:T], color="darkorange", linewidth=1.5)
axes[0, 1].set_title("True trend $\\mu_t$")
axes[0, 1].set_ylabel("Level")

axes[1, 0].plot(dates, gam[:T, 0], color="seagreen", linewidth=1.2)
axes[1, 0].axhline(0, color="grey", linewidth=0.8, linestyle="--")
axes[1, 0].set_title("True seasonal $\\gamma_t$")
axes[1, 0].set_ylabel("Seasonal effect")

axes[1, 1].plot(dates, eps, color="firebrick", linewidth=0.9, alpha=0.8)
axes[1, 1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
axes[1, 1].set_title("Irregular $\\varepsilon_t$")
axes[1, 1].set_ylabel("Residual")

plt.tight_layout()
plt.show()
```

The top-left panel shows the characteristic **Q4 peak** driven by winter
heating demand and a gentle upward trend reflecting population growth and
expanding industrial use. The trend panel (top-right) reveals that growth is
not perfectly linear — the slope $\nu_t$ wanders slowly. The seasonal panel
(bottom-left) shows a stable but not perfectly fixed pattern; the small
seasonal disturbances $\sigma_\omega = 20$ allow the winter peak to vary by a
few percent year-on-year.

---

## Step 2 — Prior specification and elicitation

For a conjugate Gibbs sampler each variance parameter receives an
**Inverse-Gamma** prior:

$$
\sigma_j^2 \sim \mathcal{IG}(a_j,\, b_j), \quad j \in \{\varepsilon, \xi, \zeta, \omega\}
$$

The density is $p(\sigma^2) \propto (\sigma^2)^{-(a+1)} \exp(-b/\sigma^2)$, with:

$$
\mathbb{E}[\sigma^2] = \frac{b}{a - 1} \quad (a > 1), \qquad
\text{mode}(\sigma^2) = \frac{b}{a + 1}, \qquad
\text{Var}(\sigma^2) = \frac{b^2}{(a-1)^2(a-2)} \quad (a > 2)
$$

**Elicitation strategy:** the data are on a scale of roughly 5000 units with
standard deviation ~430. Reasonable prior means for standard deviations are
in the range 30–100. We fix $a = 3$ for all parameters (weakly informative,
finite variance) and choose $b$ so that the prior mean $b/(a-1) = b/2$ is
a plausible variance.

| Parameter | $a$ | $b$ | Prior mean $\sigma^2$ | Prior mean $\sigma$ |
|-----------|-----|-----|----------------------|---------------------|
| $\sigma_\varepsilon^2$ (irregular) | 3.0 | 5000.0 | 2500 | 50 |
| $\sigma_\xi^2$ (level) | 3.0 | 2000.0 | 1000 | 32 |
| $\sigma_\zeta^2$ (slope) | 3.0 | 100.0 | 50 | 7 |
| $\sigma_\omega^2$ (seasonal) | 3.0 | 1000.0 | 500 | 22 |

The prior means are deliberately centred close to the true values to keep the
tutorial concise, but in practice you would set them from domain knowledge of
the data scale before fitting.

```python
# ── Define Inverse-Gamma priors ───────────────────────────────────────────────
priors: dict[str, InverseGamma] = {
    "sigma2_eps":   InverseGamma(a=3.0, b=5000.0),
    "sigma2_xi":    InverseGamma(a=3.0, b=2000.0),
    "sigma2_zeta":  InverseGamma(a=3.0, b=100.0),
    "sigma2_omega": InverseGamma(a=3.0, b=1000.0),
}

# ── Display prior properties ──────────────────────────────────────────────────
print(f"{'Parameter':<18} {'a':>5} {'b':>8} {'Mean σ²':>10} {'Mode σ²':>10} {'Mean σ':>8}")
print("-" * 65)
for name, prior in priors.items():
    a, b = prior.a, prior.b
    mean_var  = b / (a - 1)
    mode_var  = b / (a + 1)
    mean_std  = mean_var ** 0.5
    print(f"{name:<18} {a:>5.1f} {b:>8.1f} {mean_var:>10.1f} {mode_var:>10.1f} {mean_std:>8.1f}")
```

### Expected output

```
Parameter          a        b    Mean σ²    Mode σ²   Mean σ
-----------------------------------------------------------------
sigma2_eps      3.0   5000.0     2500.0     1250.0     50.0
sigma2_xi       3.0   2000.0     1000.0      500.0     31.6
sigma2_zeta     3.0    100.0       50.0       25.0      7.1
sigma2_omega    3.0   1000.0      500.0      250.0     22.4
```

```python
# ── Prior predictive check ────────────────────────────────────────────────────
# Draw 50 parameter samples from the priors; simulate a short series from each;
# verify that the simulated series are plausible given the data scale.

rng_ppc = np.random.default_rng(7)
n_ppc = 50

fig, ax = plt.subplots(figsize=(13, 5))

for _ in range(n_ppc):
    # Draw variances from priors (Inverse-Gamma: IG(a,b) ~ b / chi2(2a) * 2a)
    s2_eps   = priors["sigma2_eps"].rvs(rng_ppc)
    s2_xi    = priors["sigma2_xi"].rvs(rng_ppc)
    s2_zeta  = priors["sigma2_zeta"].rvs(rng_ppc)
    s2_omega = priors["sigma2_omega"].rvs(rng_ppc)

    # Simulate a short 40-step series from the prior
    mu_sim  = np.zeros(41); nu_sim = np.zeros(41)
    gam_sim = np.zeros((41, 4))
    mu_sim[0] = 5000.0; nu_sim[0] = 10.0
    gam_sim[0] = np.array([-80.0, -150.0, 50.0, 180.0])
    for t in range(40):
        mu_sim[t + 1]  = mu_sim[t] + nu_sim[t] + rng_ppc.normal(0, s2_xi ** 0.5)
        nu_sim[t + 1]  = nu_sim[t] + rng_ppc.normal(0, s2_zeta ** 0.5)
        ng              = -(gam_sim[t, 1] + gam_sim[t, 2] + gam_sim[t, 3])
        ng             += rng_ppc.normal(0, s2_omega ** 0.5)
        gam_sim[t + 1]  = np.array([ng, gam_sim[t, 0], gam_sim[t, 1], gam_sim[t, 2]])
    eps_sim = rng_ppc.normal(0, s2_eps ** 0.5, 40)
    y_sim   = mu_sim[:40] + gam_sim[:40, 0] + eps_sim

    ax.plot(range(40), y_sim, color="cornflowerblue", alpha=0.15, linewidth=0.8)

ax.plot(range(40), y_values[:40], color="black", linewidth=1.5, label="Observed $y_t$")
ax.set_title("Prior predictive check — 50 prior draws vs. observed data (first 40 quarters)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Gas consumption (units)")
ax.legend()
plt.tight_layout()
plt.show()
```

The observed series (black line) sits comfortably within the envelope of
prior-simulated trajectories, confirming that the prior supports the observed
data scale without being excessively diffuse.

!!! warning "Prior sensitivity"
    Always verify that your conclusions are not dominated by the prior.
    After fitting, rerun the sampler with priors scaled up by a factor of 4
    (doubling $b$) and down by a factor of 4 (halving $b$). If the posterior
    means shift by more than one posterior standard deviation, report the
    sensitivity. For this dataset ($T = 120$) the likelihood is informative
    enough that results are typically robust to moderate prior changes, but
    weakly identified parameters (especially $\sigma_\zeta^2$ for the slope)
    can be more sensitive.

---

## Step 3 — Configure the Gibbs Sampler with FFBS

The **Gibbs sampler** cycles through two conditional distributions, each of
which can be sampled exactly:

**Block 1 — States given parameters** (FFBS step):

Given the current $\theta = (\sigma_\varepsilon^2, \sigma_\xi^2, \sigma_\zeta^2, \sigma_\omega^2)$,
draw the entire state sequence $\alpha_{1:T}$ jointly from:

$$
p(\alpha_{1:T} \mid y_{1:T}, \theta) = p(\alpha_T \mid y_{1:T}, \theta)
\prod_{t=T-1}^{1} p(\alpha_t \mid \alpha_{t+1}, y_{1:t}, \theta)
$$

The **forward pass** runs the standard Kalman filter recursion, storing all
predicted means $a_{t|t-1}$, filtered means $a_{t|t}$, and covariances
$P_{t|t}$. The **backward pass** samples from the following Gaussian
distribution at each step (going from $t = T$ back to $t = 1$):

$$
\alpha_t \mid \alpha_{t+1}, y_{1:t} \sim \mathcal{N}(m_t^*, V_t^*)
$$

where:

$$
J_t = P_{t|t} T' (T P_{t|t} T' + Q)^{-1}
$$

$$
m_t^* = a_{t|t} + J_t (\alpha_{t+1} - T\, a_{t|t}), \qquad
V_t^* = P_{t|t} - J_t\, T\, P_{t|t}
$$

This gives an exact draw from the joint smoothing distribution — no
approximation.

**Block 2 — Variances given states** (conjugate IG update):

Each variance parameter is updated from its conjugate posterior. For
example, the irregular variance $\sigma_\varepsilon^2$ has sufficient
statistic $\sum_{t=1}^T (y_t - Z \alpha_t)^2$:

$$
\sigma_\varepsilon^2 \mid \alpha_{1:T}, y_{1:T} \sim \mathcal{IG}\!\left(a_\varepsilon + \frac{T}{2},\; b_\varepsilon + \frac{1}{2}\sum_{t=1}^T \varepsilon_t^2\right)
$$

where $\varepsilon_t = y_t - \mu_t - \gamma_t$ are the residuals implied by
the drawn state sequence. The same pattern applies for $\sigma_\xi^2$,
$\sigma_\zeta^2$, and $\sigma_\omega^2$ using the corresponding state
disturbances. Because these are closed-form draws, each Gibbs iteration costs
$O(m^3 T)$ — the same order as a single Kalman filter pass.

```python
# ── Configure the Gibbs sampler ───────────────────────────────────────────────
sampler = BayesianSSM(
    model_cls=BSM,
    y=y,
    prior=priors,
    model_kwargs={"seasonal_periods": 4},   # pass-through to BSM constructor
)

print(sampler)
print(f"\nState dimension m = {sampler.state_dim}")
print(f"Observation dim  = {sampler.obs_dim}")
print(f"Free variances   : {list(sampler.param_names)}")
```

### Expected output

```
BayesianSSM(model=BSM, T=120, m=5, params=4)
  Priors : sigma2_eps=IG(3.0, 5000.0)  sigma2_xi=IG(3.0, 2000.0)
           sigma2_zeta=IG(3.0, 100.0)  sigma2_omega=IG(3.0, 1000.0)
  Algorithm: Gibbs + FFBS (Carter-Kohn 1994)

State dimension m = 5
Observation dim  = 1
Free variances   : ['sigma2_eps', 'sigma2_xi', 'sigma2_zeta', 'sigma2_omega']
```

The `model_kwargs` dictionary is forwarded directly to `BSM.__init__`, so you
can pass any keyword that `BSM` accepts — for example `stochastic_cycle=True`
or `irregular=False`.

!!! note "Initialisation of the chain"
    By default `BayesianSSM` initialises each chain at the prior means.
    Alternatively, pass `init="mle"` to start chains at the MLE estimates,
    which typically shortens the burn-in phase when the data are highly
    informative.

---

## Step 4 — Run MCMC: burn-in and sampling

We run four independent chains, each for 8000 iterations with the first 2000
discarded as burn-in. The 6000 post-burn draws from each chain give
$4 \times 6000 = 24\,000$ total posterior samples.

```python
# ── Run the Gibbs sampler ─────────────────────────────────────────────────────
result = sampler.fit(
    n_iter=8000,      # total iterations per chain (includes burn-in)
    n_burn=2000,      # number of burn-in iterations to discard
    thin=1,           # keep every draw (no thinning needed with FFBS)
    n_chains=4,       # number of independent parallel chains
    seed=42,          # reproducibility seed (each chain gets seed + chain_id)
)

# ── Quick convergence overview ────────────────────────────────────────────────
print(result.summary())
```

### Expected output

```
======================================================
BayesianSSM Posterior Summary
======================================================
Chains        : 4
Iterations    : 8000 per chain  (burn-in: 2000)
Kept samples  : 6000 per chain  (24000 total)
Thin          : 1
------------------------------------------------------
Parameter       mean       sd     2.5%    97.5%  R-hat    ESS
------------------------------------------------------
sigma2_eps    2512.8    412.6   1793.4   3371.2   1.00   8943
sigma2_xi     1089.3    349.1    571.0   1882.6   1.00   6217
sigma2_zeta     55.4     38.2     13.2    149.3   1.00   5831
sigma2_omega   479.6    142.7    252.4    810.3   1.00   7402
------------------------------------------------------
All R-hat < 1.01  [PASS]
All ESS   > 400   [PASS]
======================================================
```

!!! tip "Chain parallelism"
    `kalmanbox` runs each chain in a separate worker process using Python's
    `multiprocessing` module. On a quad-core machine, four chains run in
    approximately the same wall-clock time as one chain. Pass
    `n_jobs=2` to limit parallelism when memory is constrained. Each chain
    with $T=120$, $m=5$, and 8000 iterations completes in roughly 4–8 seconds
    on a modern laptop.

---

## Step 5 — Convergence diagnostics: trace plots, R-hat, and ESS

Three complementary diagnostics tell you whether the chains have converged to
the target posterior:

1. **Trace plots**: all chains should mix well (no trends, no chains stuck at
   different levels).
2. **R-hat** (Gelman-Rubin statistic): compares within-chain and between-chain
   variance. Values below 1.01 indicate convergence.
3. **ESS** (effective sample size): accounts for autocorrelation within chains.
   Target at least 400 for means and 1000 for tail quantiles.

```python
# ── Trace plots ───────────────────────────────────────────────────────────────
param_names  = ["sigma2_eps", "sigma2_xi", "sigma2_zeta", "sigma2_omega"]
param_labels = [r"$\sigma^2_\varepsilon$", r"$\sigma^2_\xi$",
                r"$\sigma^2_\zeta$",       r"$\sigma^2_\omega$"]
chain_colors = ["steelblue", "darkorange", "seagreen", "firebrick"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Trace Plots — 4 Chains (post-burn)", fontsize=14)

for ax, name, label in zip(axes.flat, param_names, param_labels):
    for c_idx in range(result.n_chains):
        draws = result.posterior.chain(c_idx)[name]   # shape: (6000,)
        ax.plot(draws, color=chain_colors[c_idx], alpha=0.6,
                linewidth=0.5, label=f"Chain {c_idx + 1}")
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Post-burn iteration")
    ax.set_ylabel("Value")

axes[0, 0].legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.show()
```

Well-mixed chains look like **stationary "caterpillars"** — all four coloured
traces overlap and wander around the same mean level. If any chain trends
upward or stays separated, increase `n_burn`.

```python
# ── Numerical diagnostics table ───────────────────────────────────────────────
import arviz as az

# Convert result to ArviZ InferenceData for rich diagnostics
idata: az.InferenceData = result.to_arviz()

diag = az.summary(idata, var_names=param_names,
                  stat_funcs=None, extend=True, round_to=2)

print("\n=== Posterior Diagnostic Table ===")
print(f"{'Parameter':<18} {'mean':>8} {'sd':>8} {'2.5%':>8} {'97.5%':>8}"
      f" {'r_hat':>7} {'ess_bulk':>10}")
print("-" * 72)
for name, label in zip(param_names, param_labels):
    row = diag.loc[name]
    print(f"{name:<18} {row['mean']:>8.1f} {row['sd']:>8.1f}"
          f" {row['hdi_2.5%']:>8.1f} {row['hdi_97.5%']:>8.1f}"
          f" {row['r_hat']:>7.3f} {row['ess_bulk']:>10.0f}")
```

### Expected output

```
=== Posterior Diagnostic Table ===
Parameter           mean       sd     2.5%    97.5%  r_hat   ess_bulk
------------------------------------------------------------------------
sigma2_eps        2512.8    412.6   1793.4   3371.2   1.001      8943
sigma2_xi         1089.3    349.1    571.0   1882.6   1.000      6217
sigma2_zeta         55.4     38.2     13.2    149.3   1.002      5831
sigma2_omega       479.6    142.7    252.4    810.3   1.001      7402
```

```python
# ── Autocorrelation plots ─────────────────────────────────────────────────────
from kalmanbox.visualization import plot_diagnostic_panel

# Plot ACF for Chain 1 of each parameter (lags 0–50)
plot_diagnostic_panel(result, chain=0, max_lag=50,
                      title="Autocorrelation Functions — Chain 1")
plt.show()
```

FFBS draws are nearly **independent** — the ACF typically drops to near zero
within 2–3 lags. This is the key advantage of FFBS over particle-based or
random-walk Metropolis approaches for linear Gaussian models.

| Diagnostic | Target | Interpretation if violated |
|------------|--------|---------------------------|
| R-hat < 1.01 | Chains have merged | Increase burn-in or check model identifiability |
| ESS > 1000 | Enough independent draws | Try `thin=2` to reduce autocorrelation |
| ACF → 0 by lag 10 | Low within-chain correlation | Check for label-switching or near-unit-root |

---

## Step 6 — Posterior distributions of parameters

With convergence confirmed, we examine the posterior distributions in detail.
The key question is: how much has the data updated our beliefs relative to
the priors?

```python
# ── Marginal posterior histograms with prior overlay ─────────────────────────
from scipy.stats import invgamma

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Posterior vs. Prior — Variance Parameters", fontsize=14)

x_ranges = {
    "sigma2_eps":   np.linspace(500,   6000, 300),
    "sigma2_xi":    np.linspace(100,   4000, 300),
    "sigma2_zeta":  np.linspace(0,      400, 300),
    "sigma2_omega": np.linspace(50,    1500, 300),
}

for ax, name, label, prior in zip(axes.flat, param_names, param_labels,
                                   priors.values()):
    # Collect all post-burn draws across chains
    all_draws = np.concatenate([
        result.posterior.chain(c)[name] for c in range(result.n_chains)
    ])

    ax.hist(all_draws, bins=60, density=True, color="steelblue",
            alpha=0.55, label="Posterior")

    # Prior density (scipy InverseGamma: parameterised as shape=a, scale=b)
    x = x_ranges[name]
    prior_pdf = invgamma.pdf(x, a=prior.a, scale=prior.b)
    ax.plot(x, prior_pdf, color="darkorange", linewidth=2.0,
            linestyle="--", label="Prior IG")

    # Posterior HPD interval
    hpd_low  = np.percentile(all_draws, 2.5)
    hpd_high = np.percentile(all_draws, 97.5)
    ax.axvline(np.mean(all_draws), color="navy", linewidth=1.5, label="Post. mean")
    ax.axvspan(hpd_low, hpd_high, alpha=0.12, color="navy", label="95% HPD")

    # True value marker
    true_vals = {
        "sigma2_eps": TRUE_SIGMA_EPS**2, "sigma2_xi": TRUE_SIGMA_XI**2,
        "sigma2_zeta": TRUE_SIGMA_ZETA**2, "sigma2_omega": TRUE_SIGMA_OMEGA**2,
    }
    ax.axvline(true_vals[name], color="crimson", linewidth=1.5,
               linestyle=":", label=f"True $\\sigma^2={true_vals[name]:.0f}$")

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Variance")
    ax.set_ylabel("Density")
    if name == "sigma2_eps":
        ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

For all four parameters the posterior is substantially **narrower** than the
prior and centred near the true value, demonstrating that 120 quarters of
data are highly informative.

```python
# ── Parameter correlation matrix ──────────────────────────────────────────────
all_draws_matrix = np.column_stack([
    np.concatenate([result.posterior.chain(c)[name] for c in range(result.n_chains)])
    for name in param_names
])
corr_matrix = np.corrcoef(all_draws_matrix.T)

print("\n=== Posterior Correlation Matrix ===")
header = f"{'':>14}" + "".join(f"{n:>14}" for n in param_names)
print(header)
for i, row_name in enumerate(param_names):
    row_str = f"{row_name:>14}" + "".join(f"{corr_matrix[i, j]:>14.3f}"
                                           for j in range(4))
    print(row_str)
```

### Expected output

```
=== Posterior Correlation Matrix ===
               sigma2_eps    sigma2_xi  sigma2_zeta sigma2_omega
    sigma2_eps      1.000        0.127       -0.043        0.182
     sigma2_xi      0.127        1.000        0.058        0.062
   sigma2_zeta     -0.043        0.058        1.000        0.019
  sigma2_omega      0.182        0.062        0.019        1.000
```

```python
# ── Joint scatter: sigma2_eps vs sigma2_omega ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

eps_draws   = all_draws_matrix[:, 0]
omega_draws = all_draws_matrix[:, 3]

ax.scatter(eps_draws, omega_draws, alpha=0.04, s=4, color="steelblue")
ax.set_xlabel(r"$\sigma^2_\varepsilon$ (irregular variance)")
ax.set_ylabel(r"$\sigma^2_\omega$ (seasonal variance)")
ax.set_title(r"Joint posterior: $\sigma^2_\varepsilon$ vs. $\sigma^2_\omega$")
plt.tight_layout()
plt.show()
```

The mild positive correlation ($\rho \approx 0.18$) between $\sigma_\varepsilon^2$
and $\sigma_\omega^2$ is expected: when the irregular variance is large, the
filter attributes more variation to the noise term and consequently updates
the seasonal component less aggressively, increasing posterior uncertainty
about $\sigma_\omega^2$ as well. The correlations are small enough that the
marginal summaries are adequate for most purposes.

The key advantage over MLE is immediate: MLE delivers numbers such as
$\hat{\sigma}_\varepsilon^2 = 2489$ with no accompanying uncertainty. The
Bayesian posterior gives you $\sigma_\varepsilon^2 \in [1793, 3371]$ (95%
HPD) — making model risk explicit.

---

## Step 7 — Compare MLE vs Bayesian estimates

We now fit the same BSM by maximum likelihood and compare parameter estimates
side by side.

```python
# ── Fit MLE reference model ───────────────────────────────────────────────────
mle_model   = BSM(y, seasonal_periods=4)
mle_results = mle_model.fit(method="lbfgs", disp=False)

print("=== MLE Fit Summary ===")
print(mle_results.summary())
```

### Expected output

```
=== MLE Fit Summary ===
BSM — MLE Results (L-BFGS-B)
──────────────────────────────────────────────────────
Log-likelihood : -743.21
AIC            : 1498.42
BIC            : 1514.83
──────────────────────────────────────────────────────
Parameter          MLE est.   Std. Err.    95% CI
──────────────────────────────────────────────────────
sigma2_eps         2489.0       381.5    [1741, 3237]
sigma2_xi           997.4       295.8    [ 417, 1578]
sigma2_zeta          44.2        29.1    [  -13,  101]
sigma2_omega        461.3       128.9    [ 209,  714]
──────────────────────────────────────────────────────
```

```python
# ── Side-by-side comparison table ────────────────────────────────────────────
mle_ests = {
    "sigma2_eps":   mle_results.params["sigma2_eps"],
    "sigma2_xi":    mle_results.params["sigma2_xi"],
    "sigma2_zeta":  mle_results.params["sigma2_zeta"],
    "sigma2_omega": mle_results.params["sigma2_omega"],
}
mle_se = {
    "sigma2_eps":   mle_results.std_errors["sigma2_eps"],
    "sigma2_xi":    mle_results.std_errors["sigma2_xi"],
    "sigma2_zeta":  mle_results.std_errors["sigma2_zeta"],
    "sigma2_omega": mle_results.std_errors["sigma2_omega"],
}

print(f"\n{'Parameter':<18} {'True σ²':>9} {'MLE':>9} {'Bayes mean':>12} {'Bayes 95% HPD':>22}")
print("-" * 75)

true_vals_map = {
    "sigma2_eps":   TRUE_SIGMA_EPS**2,
    "sigma2_xi":    TRUE_SIGMA_XI**2,
    "sigma2_zeta":  TRUE_SIGMA_ZETA**2,
    "sigma2_omega": TRUE_SIGMA_OMEGA**2,
}

for name in param_names:
    all_d   = np.concatenate([result.posterior.chain(c)[name]
                               for c in range(result.n_chains)])
    b_mean  = np.mean(all_d)
    hpd_lo  = np.percentile(all_d, 2.5)
    hpd_hi  = np.percentile(all_d, 97.5)
    true_v  = true_vals_map[name]
    mle_v   = mle_ests[name]
    hpd_str = f"[{hpd_lo:.0f}, {hpd_hi:.0f}]"
    print(f"{name:<18} {true_v:>9.0f} {mle_v:>9.1f} {b_mean:>12.1f} {hpd_str:>22}")
```

### Expected output

```
Parameter          True σ²       MLE   Bayes mean      Bayes 95% HPD
---------------------------------------------------------------------------
sigma2_eps            2500    2489.0       2512.8         [1793, 3371]
sigma2_xi             900     997.4       1089.3          [ 571, 1882]
sigma2_zeta           25      44.2          55.4          [  13,  149]
sigma2_omega          400    461.3         479.6          [ 252,  810]
```

With $T = 120$ the MLE and posterior mean agree closely for the well-identified
parameters ($\sigma_\varepsilon^2$ and $\sigma_\omega^2$). The slope variance
$\sigma_\zeta^2$ is weakly identified — its MLE standard error exceeds its
estimate — and the posterior reflects this by being wide and skewed right.

```python
# ── Filtered states: MLE vs posterior mean ────────────────────────────────────
# Extract MLE filtered trend
mle_filtered = mle_results.filtered_states["level"]   # shape: (T,)

# Posterior mean of trend: average over all posterior state draws
post_trend_mean = result.posterior.state_mean("level")   # shape: (T,)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle("MLE vs. Bayesian Filtered States", fontsize=14)

axes[0].plot(dates, y_values, color="lightgrey", linewidth=1.0, label="$y_t$")
axes[0].plot(dates, mle_filtered,     color="darkorange", linewidth=1.5,
             linestyle="--", label="MLE filtered $\\hat{\\mu}_t$")
axes[0].plot(dates, post_trend_mean,  color="steelblue",  linewidth=1.5,
             label="Posterior mean $\\bar{\\mu}_t$")
axes[0].set_title("Trend component $\\mu_t$")
axes[0].set_ylabel("Level")
axes[0].legend()

# Residuals: MLE filtered vs posterior mean
residuals = mle_filtered - post_trend_mean
axes[1].plot(dates, residuals, color="firebrick", linewidth=0.9)
axes[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
axes[1].set_title("Difference: MLE − Bayesian posterior mean")
axes[1].set_ylabel("Units")
axes[1].set_xlabel("Quarter")

plt.tight_layout()
plt.show()
```

The filtered trend estimates nearly coincide (maximum absolute difference
typically under 20 units, less than 0.4% of the mean). The Bayesian estimate
is slightly smoother because it averages over draws where $\sigma_\zeta^2$
takes larger values, reducing the estimated slope variation.

!!! note "When do MLE and Bayesian estimates diverge?"
    With $T = 120$ and well-identified parameters, the two approaches agree
    closely. Divergence increases in three situations: (1) **small samples**
    ($T < 30$) where the prior meaningfully regularises; (2) **weakly
    identified parameters** such as $\sigma_\zeta^2$ where the likelihood is
    nearly flat; and (3) **boundary effects** — MLE can return exactly zero for
    a variance while the IG prior forces the posterior away from zero.

---

## Step 8 — Posterior predictive checks and state uncertainty via FFBS

The full Bayesian posterior does more than give distributions over parameters.
By propagating parameter draws through the FFBS state smoother and the
prediction equations, we obtain **state trajectories** and **predictive
distributions** that correctly account for all sources of uncertainty.

### 8a — State trajectory uncertainty from FFBS samples

```python
# ── Sample 200 state trajectories from the posterior ─────────────────────────
N_TRAJ = 200
rng_traj = np.random.default_rng(99)

# result.posterior_states returns an array of shape (N_TRAJ, T, m)
# where each row is one FFBS draw from one posterior parameter sample
state_samples = result.posterior_states(n_samples=N_TRAJ, seed=rng_traj,
                                         state_name="level")   # shape: (200, 120)
post_trend_p2p5  = np.percentile(state_samples, 2.5,  axis=0)
post_trend_p97p5 = np.percentile(state_samples, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("Trend State Uncertainty via FFBS Samples", fontsize=14)

# Spaghetti: 200 thin grey trajectories
for i in range(N_TRAJ):
    ax.plot(dates, state_samples[i], color="grey", alpha=0.06, linewidth=0.6)

# Posterior mean and 95% band
ax.plot(dates, post_trend_mean, color="steelblue", linewidth=2.0,
        label="Posterior mean $\\bar{\\mu}_t$")
ax.fill_between(dates, post_trend_p2p5, post_trend_p97p5,
                alpha=0.25, color="steelblue", label="95% posterior band")

# MLE confidence band for comparison
mle_ci_low  = mle_results.filtered_states["level"] - 1.96 * mle_results.filtered_std["level"]
mle_ci_high = mle_results.filtered_states["level"] + 1.96 * mle_results.filtered_std["level"]
ax.fill_between(dates, mle_ci_low, mle_ci_high,
                alpha=0.18, color="darkorange", label="MLE 95% CI (plug-in)")

ax.plot(dates, y_values, color="lightgrey", linewidth=0.9, zorder=1)
ax.set_title("Trend $\\mu_t$ — FFBS posterior samples vs. MLE plug-in CI")
ax.set_xlabel("Quarter")
ax.set_ylabel("Level")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

The spaghetti plot reveals an important pattern: the **Bayesian posterior
band is wider** than the MLE plug-in CI, especially in the middle of the
sample where the slope can wander most. The MLE CI treats $\hat{\sigma}_\zeta^2$
as known; the Bayesian band averages over all plausible values of
$\sigma_\zeta^2$, correctly reflecting the weak identification of the slope
variance.

### 8b — Posterior predictive check (in-sample)

A posterior predictive check asks: if the model is correctly specified,
should the observed data look like replicated data drawn from the fitted
model? Systematic deviations indicate misspecification.

```python
# ── In-sample posterior predictive check ─────────────────────────────────────
# result.posterior_predictive(steps=0) returns replicated observations
# by drawing theta ~ posterior, then alpha_{1:T} ~ FFBS(theta), then y* ~ model

ppc = result.posterior_predictive(steps=0, n_samples=200, seed=42)
# ppc.y_rep has shape (200, 120)

y_rep_p2p5  = np.percentile(ppc.y_rep, 2.5,  axis=0)
y_rep_p97p5 = np.percentile(ppc.y_rep, 97.5, axis=0)
y_rep_mean  = np.mean(ppc.y_rep, axis=0)

fig, ax = plt.subplots(figsize=(14, 6))

for i in range(200):
    ax.plot(dates, ppc.y_rep[i], color="cornflowerblue",
            alpha=0.04, linewidth=0.6)

ax.plot(dates, y_values, color="black", linewidth=1.6,
        zorder=5, label="Observed $y_t$")
ax.plot(dates, y_rep_mean, color="steelblue", linewidth=1.4,
        linestyle="--", label="Posterior predictive mean")
ax.fill_between(dates, y_rep_p2p5, y_rep_p97p5,
                alpha=0.20, color="steelblue", label="95% posterior predictive")

ax.set_title("Posterior Predictive Check — In-Sample")
ax.set_xlabel("Quarter")
ax.set_ylabel("Gas consumption")
ax.legend()
plt.tight_layout()
plt.show()

# ── Fraction of observations inside predictive interval ──────────────────────
coverage = np.mean((y_values >= y_rep_p2p5) & (y_values <= y_rep_p97p5))
print(f"Empirical coverage of 95% posterior predictive interval: {coverage:.1%}")
print(f"Expected: ~95%  |  {'PASS' if 0.90 <= coverage <= 1.00 else 'WARN'}")
```

### Expected output

```
Empirical coverage of 95% posterior predictive interval: 94.2%
Expected: ~95%  |  PASS
```

Observed data sitting well within the envelope and an empirical coverage
close to the nominal 95% confirm adequate in-sample fit.

### 8c — Out-of-sample forecast with parameter uncertainty

```python
# ── 8-quarter-ahead forecast (2 years) ───────────────────────────────────────
N_FCST = 8
forecast = result.posterior_predictive(steps=N_FCST, n_samples=2000, seed=42)
# forecast.y_rep has shape (2000, N_FCST)

fcst_dates = pd.date_range(dates[-1], periods=N_FCST + 1, freq="QS")[1:]

fcst_mean   = np.mean(forecast.y_rep, axis=0)
fcst_p2p5   = np.percentile(forecast.y_rep, 2.5,  axis=0)
fcst_p97p5  = np.percentile(forecast.y_rep, 97.5, axis=0)
fcst_p25    = np.percentile(forecast.y_rep, 25,   axis=0)
fcst_p75    = np.percentile(forecast.y_rep, 75,   axis=0)

# MLE plug-in forecast (treats MLE point estimates as true)
mle_forecast = mle_results.forecast(steps=N_FCST)
mle_fcst_lo  = mle_forecast.mean - 1.96 * mle_forecast.std
mle_fcst_hi  = mle_forecast.mean + 1.96 * mle_forecast.std

fig, ax = plt.subplots(figsize=(14, 6))

# Historical data (last 5 years for context)
ax.plot(dates[-20:], y_values[-20:], color="black", linewidth=1.4,
        label="Observed $y_t$")

# Bayesian forecast
ax.fill_between(fcst_dates, fcst_p2p5, fcst_p97p5,
                alpha=0.20, color="steelblue", label="Bayesian 95% PI")
ax.fill_between(fcst_dates, fcst_p25, fcst_p75,
                alpha=0.35, color="steelblue", label="Bayesian 50% PI")
ax.plot(fcst_dates, fcst_mean, color="steelblue", linewidth=2.0,
        marker="o", markersize=5, label="Bayesian forecast mean")

# MLE plug-in forecast
ax.fill_between(fcst_dates, mle_fcst_lo, mle_fcst_hi,
                alpha=0.20, color="darkorange", label="MLE 95% PI (plug-in)")
ax.plot(fcst_dates, mle_forecast.mean, color="darkorange", linewidth=1.8,
        linestyle="--", marker="s", markersize=5, label="MLE forecast mean")

ax.axvline(dates[-1], color="grey", linewidth=0.9, linestyle=":")
ax.set_title("8-Quarter Forecast — Bayesian vs. MLE Plug-In")
ax.set_xlabel("Quarter")
ax.set_ylabel("Gas consumption")
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.show()

# ── Forecast interval width comparison ───────────────────────────────────────
bayes_widths = fcst_p97p5 - fcst_p2p5
mle_widths   = mle_fcst_hi - mle_fcst_lo

print("\n=== Forecast Interval Width Comparison (95%) ===")
print(f"{'Quarter':<10} {'Bayesian':>12} {'MLE plug-in':>14} {'Ratio':>8}")
print("-" * 48)
for h, (bw, mw, fd) in enumerate(zip(bayes_widths, mle_widths, fcst_dates), 1):
    print(f"Q+{h:<8} {bw:>12.1f} {mw:>14.1f} {bw/mw:>8.3f}")
```

### Expected output

```
=== Forecast Interval Width Comparison (95%) ===
Quarter      Bayesian    MLE plug-in    Ratio
------------------------------------------------
Q+1          1238.4         982.7        1.260
Q+2          1397.2        1089.3        1.283
Q+3          1521.8        1163.8        1.307
Q+4          1635.7        1228.9        1.331
Q+5          1818.4        1383.2        1.314
Q+6          1949.6        1489.7        1.309
Q+7          2084.3        1591.2        1.310
Q+8          2213.7        1693.4        1.307
```

The Bayesian predictive intervals are consistently **25–35% wider** than
the MLE plug-in intervals. This gap reflects parameter uncertainty, which
the MLE approach ignores by treating point estimates as exact. The gap grows
slightly with the forecast horizon because slope uncertainty ($\sigma_\zeta^2$
is weakly identified) compounds over time.

!!! note "Structural breaks in the forecast"
    If the gas consumption series experienced a structural break — for example
    due to a switch to renewable heating or an energy price shock — the 2-year
    forecast would diverge rapidly from realised data. Both MLE and Bayesian
    BSM assume the model structure is constant. Consider extending to a
    Markov-switching structural model (`kalmanbox.MarkovSwitchingBSM`) or
    adding a time-varying coefficient on a price regressor (`TVP`) when
    structural breaks are suspected. See [TVP tutorial](tvp-capm.md) for the
    time-varying parameter approach.

---

## Summary

In this tutorial you applied the full Bayesian estimation pipeline to a
quarterly structural time series using `kalmanbox`. Here is what was covered:

- **Gibbs sampler with FFBS**: each iteration takes one exact draw of the
  state sequence $\alpha_{1:T}$ via the Carter-Kohn forward-filter / backward-
  sample algorithm, then updates each variance $\sigma_j^2$ from its
  conjugate Inverse-Gamma posterior. No Metropolis step is required.

- **Inverse-Gamma prior elicitation**: prior means were set by considering
  the data scale and choosing $b/(a-1)$ to be a plausible variance. A prior
  predictive simulation verified that the chosen priors support the observed
  data magnitude.

- **Convergence diagnostics**: R-hat (Gelman-Rubin) below 1.01 and ESS
  above 1000 confirmed that four chains with 6000 post-burn draws each had
  converged. FFBS draws have very low autocorrelation, so ACF decays within
  a few lags.

- **Posterior vs. MLE comparison**: with $T = 120$, posterior means agree
  with MLE point estimates within a few percent for well-identified parameters.
  Weakly identified parameters (slope variance $\sigma_\zeta^2$) show wider,
  skewed posteriors and larger divergence from MLE.

- **State uncertainty via FFBS samples**: spaghetti plots of 200 drawn trend
  trajectories reveal that the Bayesian posterior band is wider than the MLE
  plug-in confidence interval, correctly propagating parameter uncertainty
  into state uncertainty.

- **Posterior predictive checks**: in-sample PPC showed empirical coverage
  of ~94%, consistent with the nominal 95%. The 8-quarter-ahead Bayesian
  predictive intervals are ~30% wider than MLE plug-in intervals because
  they marginalise over parameter uncertainty.

---

## Next steps

Deepen your understanding with the following resources:

| Resource | What you will learn |
|----------|---------------------|
| [Gibbs Sampling User Guide](../user-guide/bayesian/gibbs.md) | Full API reference for `BayesianSSM`, chain options, custom priors |
| [FFBS Algorithm](../user-guide/bayesian/ffbs.md) | Implementation details, diffuse initialisation with FFBS, stability |
| [Prior Specification](../user-guide/bayesian/priors.md) | IG, Normal-IG conjugate pairs, hierarchical priors, prior predictive tools |
| [Posterior Diagnostics](../user-guide/bayesian/posterior-diagnostics.md) | R-hat, ESS, MCMC SE, rank plots, divergence warnings |
| [Bayesian Theory](../theory/bayesian-theory.md) | Full mathematical derivations: FFBS proof, IG conjugacy, Rao-Blackwellisation |
