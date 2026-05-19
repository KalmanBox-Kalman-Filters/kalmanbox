# Bayesian Estimation of State-Space Models

## Overview

This directory contains notebooks, datasets, and validation scripts that
demonstrate **Bayesian estimation** of linear Gaussian state-space models
using the `kalmanbox` library. The examples are built around the two
workhorses of Bayesian SSM inference:

- **FFBS** — the *Forward Filtering, Backward Sampling* state simulator of
  Carter & Kohn (1994) and Frühwirth-Schnatter (1994), which draws from
  :math:`p(\alpha_{1:T} \mid y_{1:T}, \theta)` in one pass through the
  Kalman filter.
- **Gibbs sampler** — an outer MCMC that alternates between sampling the
  latent states (via FFBS) and the hyperparameters (via conjugate
  inverse-Gamma / inverse-Wishart updates) as in Kim & Nelson (1999).

This complements the frequentist / MLE-based notebooks in
`01_local_level_trend/`, `02_structural_models/`, and `05_tvp/`: the same
models are refit with priors and diagnosed with standard MCMC convergence
tools (R-hat, ESS, HPD intervals).

## Why Bayesian?

Classical (MLE) estimation of SSMs has well-known pathologies:

- Variance components near zero lead to a **likelihood pile-up at the
  boundary** (so-called "pile-up problem"; see Shephard & Harvey, 1990).
- Small-sample standard errors for variance components are unreliable.
- Smoothed states are reported *conditional* on the MLE — uncertainty about
  the hyperparameters is ignored.

A Bayesian treatment resolves all three at the cost of choosing priors and
running MCMC. The posterior over states automatically integrates out the
parameter uncertainty, and proper priors regularise variance components.

---

## The Gibbs / FFBS Algorithm

Consider the generic linear Gaussian SSM

$$
\begin{aligned}
y_t &= Z_t\, \alpha_t + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, H_t) \\
\alpha_{t+1} &= T_t\, \alpha_t + R_t\, \eta_t, & \eta_t &\sim \mathcal{N}(0, Q_t)
\end{aligned}
$$

with hyperparameters :math:`\theta = \{H_t, Q_t, \ldots\}` collecting the
unknown variance components and any other fixed parameters.

The joint posterior :math:`p(\theta, \alpha_{1:T} \mid y_{1:T})` is generally
intractable, but the two *full conditionals* are both tractable:

### Block 1 — States given parameters (FFBS)

Given :math:`\theta`, draw the entire path :math:`\alpha_{1:T}` from

$$
p(\alpha_{1:T} \mid y_{1:T}, \theta) =
    p(\alpha_T \mid y_{1:T}, \theta) \prod_{t=1}^{T-1}
    p(\alpha_t \mid \alpha_{t+1}, y_{1:t}, \theta).
$$

Each factor is Gaussian, so the joint draw is obtained by:

1. **Forward pass.** Run the Kalman filter forward and store the filtered
   means :math:`a_{t|t}` and covariances :math:`P_{t|t}` for
   :math:`t = 1, \ldots, T`.
2. **Terminal draw.**
   :math:`\alpha_T \sim \mathcal{N}(a_{T|T}, P_{T|T})`.
3. **Backward recursion.** For :math:`t = T-1, T-2, \ldots, 1`, draw
   :math:`\alpha_t \mid \alpha_{t+1}, y_{1:t}, \theta
   \sim \mathcal{N}(a_{t|t,t+1},\, P_{t|t,t+1})` with

$$
\begin{aligned}
a_{t|t,t+1} &= a_{t|t} + J_t \big(\alpha_{t+1} - T_t\, a_{t|t}\big) \\
P_{t|t,t+1} &= P_{t|t} - J_t\, T_t\, P_{t|t} \\
J_t &= P_{t|t}\, T_t^\top\, P_{t+1|t}^{-1}.
\end{aligned}
$$

This is one-for-one the Carter & Kohn (1994) / Frühwirth-Schnatter (1994)
*simulation smoother*.

### Block 2 — Parameters given states (conjugate draws)

Given :math:`\alpha_{0:T}`, the complete-data likelihood for the variance
components factorises into independent Gaussian pieces, and with
**inverse-Gamma priors**

$$
\sigma_\varepsilon^2 \sim \mathcal{IG}(a_0, b_0), \qquad
\sigma_\eta^2 \sim \mathcal{IG}(c_0, d_0),
$$

the full conditionals are also inverse-Gamma:

$$
\begin{aligned}
\sigma_\varepsilon^2 \mid \alpha_{1:T}, y_{1:T}
  &\sim \mathcal{IG}\!\left(a_0 + \tfrac{T}{2},\;
    b_0 + \tfrac12 \sum_{t=1}^T (y_t - Z_t \alpha_t)^2 \right), \\
\sigma_\eta^2 \mid \alpha_{0:T}
  &\sim \mathcal{IG}\!\left(c_0 + \tfrac{T}{2},\;
    d_0 + \tfrac12 \sum_{t=1}^T (\alpha_t - T_{t-1}\alpha_{t-1})^2 \right).
\end{aligned}
$$

For multivariate states the analogous conjugate prior on :math:`Q` is the
**inverse-Wishart**.

### Gibbs loop

```text
Initialise theta^(0)
for s = 1, ..., S:
    alpha_{1:T}^(s) ~ p(alpha_{1:T} | y_{1:T}, theta^(s-1))   # FFBS
    theta^(s)       ~ p(theta | y_{1:T}, alpha_{1:T}^(s))      # conjugate
```

After a burn-in phase the draws :math:`\{\theta^{(s)}, \alpha^{(s)}_{1:T}\}`
form an approximate sample from the joint posterior.

---

## MCMC Diagnostics

The module `data/mcmc_utils.py` provides the standard tooling used across
all notebooks in this directory:

| Function                   | Purpose                                        |
|----------------------------|------------------------------------------------|
| `trace_plot`               | Trace plot (multi-chain aware, running mean).  |
| `autocorrelation_plot`     | Sample ACF with 95% white-noise bands.         |
| `gelman_rubin`             | :math:`\hat R` via split-chain variance ratio. |
| `effective_sample_size`    | ESS via Geyer's initial-positive-sequence.     |
| `hpd_interval`             | Chen-Shao HPD credible interval.               |

Convergence rule-of-thumb (Gelman et al., 2013):
:math:`\hat R < 1.01` and ESS per chain :math:`\gtrsim 400` before
trusting posterior summaries.

---

## Notebooks

Two notebooks (added in subphases F7.2–F7.5) exercise the material:

1. **Bayesian local level on the Nile dataset** — Gibbs/FFBS estimation of
   the classical :math:`\sigma_\varepsilon^2, \sigma_\eta^2` pair, with
   posterior credible bands on the smoothed level. Replicates the canonical
   application from Durbin & Koopman (2012, Chap. 2) in a Bayesian setting.
2. **Bayesian TVP Phillips curve** — FFBS inside a Gibbs sampler for a
   time-varying-coefficient regression of inflation on unemployment,
   following the Primiceri (2005) / Cogley & Sargent (2005) recipe with
   conjugate inverse-Wishart prior on :math:`Q`.

## Datasets

Both datasets are re-used from earlier subphases so results are directly
comparable to the MLE-based notebooks.

- **`data/nile.csv`** — Annual flow of the Nile at Aswan, 1871–1970
  (100 obs). Source: Durbin & Koopman (2012), Cobb (1978).
- **`data/us_inflation_unemployment.csv`** — Synthetic quarterly US macro
  panel, 1960Q1–2023Q4 (256 obs): `inflation`, `unemployment`, `gdp_gap`.
  See `examples/05_tvp/README.md` for the full calibration documentation.

## Validation

- **R** (`validation/R/`): `dlm` package with MCMC (`dlmGibbsDIG`) for the
  local-level model; `bvarsv` for Primiceri-style TVP Bayesian regressions.
- **Stata** (`validation/stata/`): reference only — native `bayes:` prefix
  for state-space models is limited; comparison is qualitative.

---

## References

- **Carter, C.K. and Kohn, R. (1994).** "On Gibbs Sampling for State Space
  Models." *Biometrika*, 81(3), 541–553. *(Original FFBS paper.)*
- **Frühwirth-Schnatter, S. (1994).** "Data Augmentation and Dynamic Linear
  Models." *Journal of Time Series Analysis*, 15(2), 183–202.
  *(Independent derivation of the simulation smoother.)*
- **Kim, C.-J. and Nelson, C.R. (1999).** *State-Space Models with Regime
  Switching: Classical and Gibbs-Sampling Approaches with Applications.*
  MIT Press. *(Standard textbook treatment of Bayesian SSMs.)*
- **Cogley, T. and Sargent, T.J. (2005).** "Drifts and Volatilities:
  Monetary Policies and Outcomes in the Post WWII US." *Review of Economic
  Dynamics*, 8(2), 262–302.
- **Primiceri, G.E. (2005).** "Time Varying Structural Vector
  Autoregressions and Monetary Policy." *Review of Economic Studies*,
  72(3), 821–852.
- **Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., and
  Rubin, D.B. (2013).** *Bayesian Data Analysis*, 3rd ed. Chapman & Hall.
- **Durbin, J. and Koopman, S.J. (2012).** *Time Series Analysis by State
  Space Methods*, 2nd ed. Oxford University Press.
- **Shephard, N. and Harvey, A.C. (1990).** "On the Probability of
  Estimating a Deterministic Component in the Local Level Model."
  *Journal of Time Series Analysis*, 11(4), 339–347.
