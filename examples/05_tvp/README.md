# Time-Varying Parameter (TVP) Regression

## Overview

Time-Varying Parameter (TVP) regressions generalise the standard linear model
by letting the regression coefficients drift over time as unobserved latent
states. They are the workhorse tool in empirical macro for tracking structural
change — the slope of the Phillips curve, the inflation persistence parameter,
the monetary policy rule, NAIRU, and expected inflation all show up as TVPs in
modern work.

A TVP model is a natural state-space model: the observation equation is a
regression at time `t`, and the state transition is a random walk on the
coefficients. The Kalman filter/smoother then produces a *history of
coefficients* conditional on the full sample.

## Model Specification

### Observation equation

$$y_t = x_t^\top \beta_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$

where $x_t \in \mathbb{R}^k$ is a vector of regressors (typically including a
constant), $\beta_t \in \mathbb{R}^k$ is the vector of time-varying
coefficients at time $t$, and $\sigma_\varepsilon^2$ is the variance of the
measurement noise.

### State transition (random walk coefficients)

$$\beta_t = \beta_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)$$

with $Q$ a $k \times k$ positive-semidefinite innovation covariance. Small
diagonal elements of $Q$ produce nearly-constant coefficients; large elements
allow rapid structural change. Random walks are the canonical specification
because they require no stationarity assumption on $\beta_t$ and impose
minimal structure on the time variation.

### State-space form

The TVP regression maps directly into the standard linear Gaussian SSM used
throughout `kalmanbox`:

- State: $\alpha_t \equiv \beta_t \in \mathbb{R}^k$
- Transition: $\alpha_t = T\, \alpha_{t-1} + R\, \eta_t$ with $T = I_k$ and
  $R = I_k$
- Observation: $y_t = Z_t\, \alpha_t + \varepsilon_t$ with the *time-varying*
  design matrix $Z_t = x_t^\top$
- Covariances: state innovation $Q$, observation noise $H = \sigma_\varepsilon^2$

The observation equation is scalar here; the only difference from a standard
local-level model is that $Z_t$ depends on the exogenous regressors $x_t$ at
each date.

### Estimation

1. **MLE via Kalman filter.** Evaluate the prediction-error decomposition of
   the log-likelihood as a function of $(\sigma_\varepsilon^2, Q)$ and
   numerically maximise. Typically $Q$ is parameterised as diagonal to keep
   the problem well-identified.
2. **Kalman smoother.** Given parameters, the RTS smoother returns the full
   posterior mean and covariance of $\beta_t$ for all $t$.
3. **Bayesian MCMC.** Cogley & Sargent (2005) and Primiceri (2005) sample
   $\beta_t$ via forward-filter/backward-sample (FFBS) and draw $Q$ from an
   inverse-Wishart prior.

## The Phillips Curve with Time-Varying NAIRU

The textbook application of the TVP framework is the *Phillips curve* with a
time-varying natural rate of unemployment (NAIRU):

$$\pi_t = \beta_{0,t} + \beta_{1,t}\, (u_t - u^*_t) + \varepsilon_t$$

where $\pi_t$ is inflation, $u_t$ is the unemployment rate, $u^*_t$ is the
(unobserved) NAIRU, and $\beta_{1,t}$ is the time-varying slope of the
Phillips curve.

Empirically, both $\beta_{0,t}$ (the inflation intercept / long-run
expectation) and $\beta_{1,t}$ (the slope) have drifted substantially in the
United States since 1960: the slope was visibly negative in the 1970s, appears
much flatter during the Great Moderation, and is the subject of ongoing
debate in the post-COVID period. NAIRU itself is typically filtered jointly
as an additional latent state (see Stock & Watson, 2007).

## Dataset — `data/us_inflation_unemployment.csv`

Synthetic quarterly US macro data, 1960Q1–2023Q4 (256 observations),
calibrated to reproduce the main historical regimes used in the TVP
literature:

| Column         | Description                                          |
|----------------|------------------------------------------------------|
| `date`         | Quarter start, `YYYY-MM-DD`                          |
| `inflation`    | CPI year-on-year inflation, percent                  |
| `unemployment` | Civilian unemployment rate, percent                  |
| `gdp_gap`      | Real output gap, percent of potential GDP            |

Stylised facts embedded in the calibration:

- **Great Inflation (1970s).** Trend inflation rises from ~1.5% in 1960 to
  peaks above 10% in the mid- to late-1970s.
- **Volcker disinflation (1980s).** Inflation falls from ~10% to ~4% across
  the decade, accompanied by the deep 1981–82 recession spike in
  unemployment.
- **Great Moderation (1990s–2007).** Low and stable inflation (~2–3%),
  declining NAIRU, and smaller cyclical fluctuations.
- **Global Financial Crisis (2008–09).** Unemployment jumps above 9%,
  inflation dips briefly.
- **COVID-19 (2020–2022).** Sharp unemployment spike in 2020, deflationary
  dip, followed by the 2021–2022 inflation surge.
- **Okun's-law style link.** The output gap is negatively correlated with
  cyclical unemployment (empirical correlation ≈ −0.86).

## Notebooks

Two notebooks (added in subphases F5.2–F5.5) exercise the dataset:

1. **TVP regression — Phillips curve with time-varying slope.** Fit a
   bivariate regression of inflation on a constant and unemployment with
   random-walk coefficients; estimate $(\sigma_\varepsilon^2, Q)$ by MLE and
   recover the smoothed path of $\beta_{1,t}$.
2. **TVP with time-varying NAIRU.** Augment the state with a latent NAIRU
   that follows its own random walk, jointly filtered with the Phillips-curve
   intercept and slope, following the logic of Stock & Watson (2007).

## Validation

Independent replication of the Python results is performed in two parallel
stacks:

- **R** (`validation/R/`): the `dlm` package for MLE estimation of the TVP
  regression; `bvarsv` for Primiceri-style Bayesian TVP-VARs when the
  notebooks cover that extension.
- **Stata** (`validation/stata/`): the built-in `sspace` command with
  time-varying coefficients.

## References

- Cogley, T. and Sargent, T.J. (2005). "Drifts and Volatilities: Monetary
  Policies and Outcomes in the Post WWII US." *Review of Economic Dynamics*,
  8(2), 262–302.
- Primiceri, G.E. (2005). "Time Varying Structural Vector Autoregressions
  and Monetary Policy." *Review of Economic Studies*, 72(3), 821–852.
- Stock, J.H. and Watson, M.W. (2007). "Why Has U.S. Inflation Become
  Harder to Forecast?" *Journal of Money, Credit and Banking*, 39(S1), 3–33.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.
- Durbin, J. and Koopman, S.J. (2012). *Time Series Analysis by State Space
  Methods*, 2nd ed. Oxford University Press.
