# Maximum Likelihood Estimation: Theory

This page provides the rigorous mathematical foundations for maximum likelihood
estimation (MLE) of state-space model parameters. It complements the practical
workflow in [`../user-guide/kalman/mle.md`](../user-guide/kalman/mle.md) and
should be read after the foundational material in
[`state-space-theory.md`](state-space-theory.md) and
[`kalman-filter-derivation.md`](kalman-filter-derivation.md) (referred to
below as KF-derivation).

Throughout this page the Durbin & Koopman (2012) convention is used:

| Symbol | Dimension | Meaning |
|:-------|:----------|:--------|
| $Z_t$ | $p \times m$ | Observation (design) matrix |
| $T_t$ | $m \times m$ | Transition matrix |
| $R_t$ | $m \times r$ | Noise selection matrix |
| $Q_t$ | $r \times r$ | State disturbance covariance |
| $H_t$ | $p \times p$ | Observation noise covariance |
| $d_t$ | $p \times 1$ | Observation intercept |
| $c_t$ | $m \times 1$ | State intercept |
| $\theta$ | $k \times 1$ | Unknown parameter vector |

Subscripts are omitted when the system matrices are time-invariant.

---

## 1. The MLE Problem in State-Space Models

### 1.1 Objective

Let $y = (y_1, \ldots, y_T)$ be the observed time series where $y_t \in
\mathbb{R}^p$. The joint density $p(y_{1:T} \mid \theta)$ is determined by the
parameter vector $\theta \in \Theta \subseteq \mathbb{R}^k$, which collects all
unknown elements of the system matrices. The **maximum likelihood estimator** is

$$
\hat\theta_{\mathrm{MLE}} = \arg\max_{\theta \in \Theta}\; \log p(y_{1:T} \mid \theta).
$$

For the linear Gaussian state-space model this objective is **exactly**
tractable: the Kalman filter yields the prediction-error decomposition that
evaluates $\log p(y_{1:T} \mid \theta)$ in $O(Tp^3)$ time.

### 1.2 Parameter Types

The elements of $\theta$ can be grouped by their structural role:

**Covariance parameters.**  The positive (semi-)definite matrices $H$ and $Q$
contribute $p(p+1)/2$ and $r(r+1)/2$ free elements respectively when
unrestricted. In practice most models impose diagonal or scalar structure,
reducing this count substantially.

**Transition and loading parameters.**  Elements of $T$ and $Z$ that govern
dynamics (e.g., autoregressive coefficients) and the mapping from states to
observations (factor loadings in DFMs) fall in this category. They may be free
real numbers or restricted to compact sets (stationarity, identifiability).

**Intercepts.**  The vectors $c$ and $d$ are typically either zero or estimated
freely.

**Signal-to-noise ratios.**  Many structural models are characterised by
ratios $q = \sigma^2_\eta / \sigma^2_\varepsilon$ rather than individual
variances. Optimising on the log scale of $q$ is often numerically superior.

### 1.3 Constraints

The feasible set $\Theta$ imposes:

1. **Positive definiteness** of $Q$ and $H$: all eigenvalues must be strictly
   positive (or zero for restricted components).
2. **Stability of $T$**: for stationary models all eigenvalues of $T$ must lie
   strictly inside the unit disk, $|\lambda_i(T)| < 1$.
3. **Identification**: discussed in [`identifiability.md`](identifiability.md).
   Without identification conditions the likelihood is flat along certain
   directions and the Hessian is singular at the MLE.

Constraints are most efficiently handled by **reparametrisation** — mapping
$\theta$ to an unconstrained vector $\tilde\theta \in \mathbb{R}^k$ — which is
covered in [Section 8](#8-parameter-constraints-and-transformations).

!!! warning "Local optima"
    The log-likelihood of a state-space model is generally **not** concave in
    $\theta$. Multiple local maxima may exist, particularly in models with many
    variance components. Multiple restarts (see Section 7) are strongly
    recommended.

---

## 2. Log-Likelihood via Prediction Error Decomposition

### 2.1 Chain-Rule Factorisation

By the chain rule of probability, the joint density factors as

$$
p(y_{1:T} \mid \theta) = \prod_{t=1}^{T} p(y_t \mid y_{1:t-1},\, \theta),
$$

with the convention $p(y_1 \mid y_{1:0}) \equiv p(y_1 \mid \theta)$. Taking
logarithms,

$$
\log p(y_{1:T} \mid \theta) = \sum_{t=1}^{T} \log p(y_t \mid y_{1:t-1},\, \theta).
$$

### 2.2 Gaussian Conditionals from the Kalman Filter

Under the linear Gaussian assumption the conditional distribution is

$$
y_t \mid y_{1:t-1},\, \theta \;\sim\; N\!\left(Z_t\, a_{t|t-1} + d_t,\; F_t\right),
$$

where:

- $a_{t|t-1} = E[\alpha_t \mid y_{1:t-1}]$ is the one-step-ahead predicted state
- $F_t = Z_t P_{t|t-1} Z_t' + H_t$ is the innovation covariance
- $v_t = y_t - Z_t a_{t|t-1} - d_t$ is the **innovation** (prediction error)

All three quantities are produced by the Kalman filter recursion (see
[KF-derivation](kalman-filter-derivation.md)) at essentially no additional
cost once the filter has been run.

### 2.3 Full Log-Likelihood

Substituting the Gaussian density for each conditional term:

$$
\boxed{
\log L(\theta)
= -\frac{1}{2}\sum_{t=1}^{T}
\left[\, p \log(2\pi) + \log|F_t| + v_t' F_t^{-1} v_t \,\right].
}
$$

The three components of each summand have clear interpretations:

| Term | Role |
|:-----|:-----|
| $p\log(2\pi)$ | Normalisation constant; contributes $-\frac{pT}{2}\log(2\pi)$ in total |
| $\log\lvert F_t\rvert$ | Volume penalty — large innovation variance reduces likelihood |
| $v_t' F_t^{-1} v_t$ | Mahalanobis distance — large normalised innovations reduce likelihood |

!!! tip "Computational cost"
    Computing $\log|F_t|$ from the Cholesky factor $F_t = L_t L_t'$ costs
    $O(p^3/3)$ flops (already paid during the Kalman gain computation).
    $F_t^{-1} v_t$ follows from forward substitution. The total cost of the
    log-likelihood is therefore dominated by the filter itself: $O(T p^2 m +
    T m^3)$.

### 2.4 Diffuse Log-Likelihood

When the initial state $\alpha_1$ is partially or fully non-stationary, the
exact diffuse initialisation of Durbin & Koopman (2012, Ch.\ 5) is used.
Let $d$ be the number of diffuse components. During the diffuse phase
($t = 1, \ldots, d$) the innovation covariance admits the decomposition

$$
F_t = F_{t,\infty} \kappa + F_{t,0} + O(\kappa^{-1}), \quad \kappa \to \infty,
$$

where $F_{t,\infty}$ captures the divergent part. The **diffuse log-likelihood**
is obtained by taking the limit $\kappa \to \infty$:

$$
\log L_d(\theta)
= -\frac{1}{2}\sum_{t=1}^{d}
  \bigl[\log|F_{t,\infty}|\bigr]
- \frac{1}{2}\sum_{t=d+1}^{T}
  \bigl[p\log(2\pi) + \log|F_t| + v_t' F_t^{-1} v_t\bigr].
$$

The correction subtracts the contribution of the $d$ diffuse steps to the
normalisation constant, effectively removing $\frac{d}{2}\log(2\pi)$ relative
to the standard likelihood and replacing the first $d$ log-determinant terms
with $\log|F_{t,\infty}|$. The remaining $T - d$ terms are standard.

!!! note "Koopman & Durbin (2003)"
    The exact computation of the diffuse log-likelihood, including the
    modification of the score recursion, is described in Koopman & Durbin
    (2003). `kalmanbox` accumulates the diffuse correction exactly through
    auxiliary diffuse quantities propagated during the filter sweep.

---

## 3. Concentrated Log-Likelihood

### 3.1 Variance Factorisation

Many structural models admit the factorisation

$$
\theta = (\sigma^2,\, \psi),
$$

where $\sigma^2 > 0$ is a **common variance scale** and $\psi$ collects
**dimensionless shape parameters** (e.g., signal-to-noise ratios, damping
factors, loadings). Concretely,

$$
H = \sigma^2\, \tilde{H}(\psi), \qquad Q = \sigma^2\, \tilde{Q}(\psi),
$$

with $\tilde{H}$ and $\tilde{Q}$ depending only on $\psi$.

### 3.2 Concentrating Out $\sigma^2$

Under this factorisation the innovation covariance decomposes as

$$
F_t = \sigma^2 \tilde{F}_t(\psi), \quad
\tilde{F}_t = Z_t \tilde{P}_{t|t-1} Z_t' + \tilde{H},
$$

where $\tilde{P}_{t|t-1}$ is the predicted state covariance computed with
$(\tilde{H}, \tilde{Q})$. The log-likelihood becomes

$$
\log L(\sigma^2, \psi)
= -\frac{pT}{2}\log(2\pi)
  - \frac{pT}{2}\log\sigma^2
  - \frac{1}{2}\sum_t \log|\tilde{F}_t|
  - \frac{1}{2\sigma^2}\sum_t v_t' \tilde{F}_t^{-1} v_t.
$$

Taking the first-order condition $\partial \log L / \partial \sigma^2 = 0$
yields the **closed-form MLE of the scale**:

$$
\hat\sigma^2(\psi)
= \frac{1}{pT}\sum_{t=1}^{T} v_t'\,\tilde{F}_t^{-1}\, v_t.
$$

Substituting back and simplifying produces the **concentrated log-likelihood**:

$$
\boxed{
\log L^*(\psi)
= -\frac{pT}{2}\!\left[\log\hat\sigma^2(\psi) + 1\right]
  - \frac{1}{2}\sum_{t=1}^{T}\log|\tilde{F}_t(\psi)|,
}
$$

which depends only on $\psi$. The optimisation problem dimension is reduced
by one, which is non-trivial when $k$ is small (e.g., the local level model
has $k = 2$ reduced to $k = 1$).

!!! note "When concentration applies"
    Concentration requires all variance matrices to share the same $\sigma^2$
    scale. For models with multiple independent variance scales (e.g., DFMs
    with heteroskedastic factors), concentration applies only to the sub-problem
    over which a single scale is shared.

---

## 4. Score Function (Gradient of the Log-Likelihood)

### 4.1 Notation and Per-Step Contribution

Define the **per-step log-likelihood contribution** (excluding the constant):

$$
\ell_t(\theta) = -\frac{1}{2}\bigl[\log|F_t| + v_t' F_t^{-1} v_t\bigr].
$$

The score vector is

$$
\frac{\partial \log L}{\partial \theta_i}
= \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \theta_i}.
$$

### 4.2 Analytic Score via Matrix Calculus

Differentiating $\ell_t$ with respect to a scalar $\theta_i$, and using the
identities $\partial \log|A|/\partial \theta = \mathrm{tr}(A^{-1} \partial A /
\partial \theta)$ and $\partial (x'Ax)/\partial \theta = x' (\partial A /
\partial \theta) x$ (for fixed $x$), one obtains the general expression

$$
\frac{\partial \ell_t}{\partial \theta_i}
= -\frac{1}{2}\,\mathrm{tr}\!\left(F_t^{-1}\frac{\partial F_t}{\partial \theta_i}\right)
  + \frac{1}{2}\,v_t' F_t^{-1}\frac{\partial F_t}{\partial \theta_i} F_t^{-1} v_t
  + v_t' F_t^{-1}\frac{\partial v_t}{\partial \theta_i}.
$$

The partial derivatives $\partial F_t / \partial \theta_i$ and $\partial v_t /
\partial \theta_i$ depend on $\theta$ through both the system matrices and the
predicted state $a_{t|t-1}$ and covariance $P_{t|t-1}$, which themselves satisfy
recursions in $\theta$. Specifically, for $\theta_i$ affecting only $H$ or $Q$:

$$
\frac{\partial v_t}{\partial \theta_i} = -Z_t \frac{\partial a_{t|t-1}}{\partial \theta_i},
\qquad
\frac{\partial F_t}{\partial \theta_i}
  = Z_t \frac{\partial P_{t|t-1}}{\partial \theta_i} Z_t'
    + \frac{\partial H}{\partial \theta_i}.
$$

The sensitivities $\partial a_{t|t-1}/\partial \theta_i$ and $\partial P_{t|t-1}/
\partial \theta_i$ satisfy **auxiliary forward recursions** that can be run
simultaneously with the Kalman filter (Koopman & Shephard, 1992).

### 4.3 Koopman & Shephard (1992) Score Recursion

Koopman & Shephard (1992) showed that the exact analytic score of the diffuse
log-likelihood can be computed without running separate sensitivity recursions.
They introduce the **score filter**: auxiliary vectors $r_t$ and matrices
$N_t$ (the same quantities used in the smoother) that propagate backward
through time. The gradient then reads

$$
\frac{\partial \log L}{\partial \theta_i}
= \frac{1}{2}\sum_{t=1}^{T}
  \mathrm{tr}\!\left[\left(F_t^{-1} v_t v_t' F_t^{-1} - F_t^{-1}\right)
  \frac{\partial F_t}{\partial \theta_i}\right]
  + \sum_{t=1}^{T} r_t' \frac{\partial (T_t a_{t-1|t-1})}{\partial \theta_i},
$$

where $r_t$ is the backward-pass vector already computed by the RTS smoother.
This formulation is exact, numerically stable, and costs $O(T k m^2)$ beyond
the smoother sweep.

### 4.4 Numerical Score in kalmanbox

By default, `kalmanbox` uses **central finite differences** to approximate
the score:

$$
\frac{\partial \log L}{\partial \theta_i}
\approx \frac{\log L(\theta + h e_i) - \log L(\theta - h e_i)}{2h},
\quad h = \epsilon^{1/3}\bigl(1 + |\theta_i|\bigr),
$$

where $\epsilon \approx 2.2 \times 10^{-16}$ is machine epsilon and $e_i$ is
the $i$-th unit vector. This requires $2k$ additional filter evaluations per
gradient computation but avoids the complexity of the analytic score and is
exact for practical purposes when the step size $h$ is chosen correctly.

!!! tip "When to use analytic gradients"
    For large models (many parameters, long series) the $2k$-filter overhead
    can dominate. Set `gradient='analytic'` in `model.fit()` to switch to
    the Koopman-Shephard recursion.

---

## 5. Information Matrix

### 5.1 Expected (Fisher) Information

The **Fisher information matrix** is defined as

$$
\mathcal{I}(\theta)
= -\,E\!\left[\frac{\partial^2 \log p(y_{1:T} \mid \theta)}
                     {\partial \theta\, \partial \theta'}\right]
= E\!\left[\frac{\partial \log L}{\partial \theta}
            \frac{\partial \log L}{\partial \theta'}\right],
$$

where both expectations are taken under the true parameter $\theta$. For
state-space models with ergodic dynamics, the information matrix accumulates
approximately linearly in $T$:

$$
\mathcal{I}(\theta) \approx T\, \mathcal{I}_1(\theta),
$$

where $\mathcal{I}_1$ is the per-observation information, reflecting the
asymptotic $\sqrt{T}$ consistency rate of the MLE.

### 5.2 Observed Information

The **observed information** matrix evaluates the Hessian at the MLE:

$$
\hat{\mathcal{I}}(\theta)
= -\frac{\partial^2 \log L}{\partial \theta\, \partial \theta'}\bigg|_{\theta=\hat\theta}.
$$

Unlike Fisher information, $\hat{\mathcal{I}}$ does not require knowledge of
the true $\theta$ and is directly available from the optimisation. It is the
standard basis for standard error computation in `kalmanbox`.

### 5.3 Cramér-Rao Lower Bound

For any unbiased estimator $\tilde\theta$ of $\theta$,

$$
\mathrm{Var}(\tilde\theta) \geq \mathcal{I}(\theta)^{-1}
$$

in the matrix (Loewner) ordering. The MLE achieves this bound asymptotically
(see Section 6). The inverse $\mathcal{I}(\theta)^{-1}$ is therefore the
**theoretically minimum achievable variance** for unbiased estimation.

### 5.4 Numerical Computation of the Observed Hessian

`kalmanbox` approximutes $\hat{\mathcal{I}}$ via the **outer-product of
gradients** (OPG) estimator or a numerical Hessian:

$$
\hat{\mathcal{I}}^{\mathrm{OPG}}
= \sum_{t=1}^{T} g_t g_t', \quad
g_t = \frac{\partial \ell_t}{\partial \theta},
$$

and by direct second differences

$$
[\hat{\mathcal{I}}^{\mathrm{Hess}}]_{ij}
\approx -\frac{\log L(\theta + h_i e_i + h_j e_j)
               - \log L(\theta + h_i e_i)
               - \log L(\theta + h_j e_j)
               + \log L(\theta)}
              {h_i h_j}.
$$

The OPG estimator is faster ($k$ gradient evaluations vs. $O(k^2)$ likelihood
evaluations for the full Hessian) but less accurate in finite samples. The
numerical Hessian is the default for standard error computation.

### 5.5 Singularity and Near-Identification

If $\hat{\mathcal{I}}$ is singular or nearly singular (small eigenvalues) at
the MLE, the model is **near-non-identified** in those directions. Flat
regions of the likelihood correspond to poorly determined combinations of
parameters — a sign that the model should be simplified or regularised. See
[`identifiability.md`](identifiability.md) for a systematic treatment.

---

## 6. Asymptotic Standard Errors and Inference

### 6.1 Asymptotic Normality of the MLE

Under standard regularity conditions (Durbin & Koopman, 2012, Appendix A),
the MLE is **consistent** and **asymptotically normal**:

$$
\sqrt{T}\,(\hat\theta - \theta_0)
\xrightarrow{\;d\;} N\!\left(0,\; \mathcal{I}_1(\theta_0)^{-1}\right),
\quad T \to \infty,
$$

where $\mathcal{I}_1(\theta_0) = \lim_{T\to\infty} T^{-1}\mathcal{I}(\theta_0)$
is the per-observation Fisher information evaluated at the true parameter
$\theta_0$. In finite samples, $\hat{\mathcal{I}}^{-1}/T$ is used as a
consistent estimator of the asymptotic covariance.

### 6.2 Standard Errors

Individual parameter standard errors are

$$
\mathrm{se}(\hat\theta_i)
= \sqrt{\bigl[\hat{\mathcal{I}}(\hat\theta)^{-1}\bigr]_{ii}},
$$

the square root of the $i$-th diagonal entry of the inverse observed
information matrix.

### 6.3 Confidence Intervals

An asymptotic $100(1-\alpha)\%$ confidence interval for $\theta_i$ is

$$
\hat\theta_i \;\pm\; z_{\alpha/2}\;\mathrm{se}(\hat\theta_i),
$$

where $z_{\alpha/2}$ is the $(1-\alpha/2)$ quantile of $N(0,1)$
(e.g., $z_{0.025} = 1.96$).

!!! warning "Small-sample validity"
    Asymptotic normality can be a poor approximation when $T$ is small (say,
    $T < 100$) or when parameters are near their boundary (e.g., a variance
    close to zero). Profile likelihood intervals (Section 10) are more
    reliable in those situations.

### 6.4 Delta Method for Functions of Parameters

For a smooth function $g: \mathbb{R}^k \to \mathbb{R}^q$, the delta method
gives

$$
\sqrt{T}\,\bigl(g(\hat\theta) - g(\theta_0)\bigr)
\xrightarrow{\;d\;} N\!\left(0,\;
  G(\theta_0)\,\mathcal{I}_1(\theta_0)^{-1}\,G(\theta_0)'\right),
$$

where $G(\theta) = \partial g / \partial \theta' \in \mathbb{R}^{q \times k}$
is the Jacobian evaluated at $\theta_0$ (approximated by $\hat\theta$ in
practice). The estimated variance of $g(\hat\theta)$ is therefore

$$
\widehat{\mathrm{Var}}\bigl[g(\hat\theta)\bigr]
= \hat{G}\;\hat{\mathcal{I}}^{-1}\;\hat{G}',
$$

with $\hat{G} = G(\hat\theta)$. This is used, for example, to obtain
standard errors for the signal-to-noise ratio $q = \sigma^2_\eta /
\sigma^2_\varepsilon$ given estimates of the two variances separately.

### 6.5 Hypothesis Tests

=== "Wald test"

    Test $H_0: R\theta = r$ against $H_1: R\theta \neq r$:

    $$
    W = (R\hat\theta - r)'\,
        \bigl[R\,\hat{\mathcal{I}}^{-1} R'\bigr]^{-1}\,
        (R\hat\theta - r)
    \xrightarrow{d} \chi^2_q
    $$

    where $q = \mathrm{rank}(R)$. Requires only the MLE under the
    **unrestricted** model.

=== "Likelihood Ratio test"

    Test of a restriction $H_0$ that reduces the parameter space from
    $\Theta$ (dimension $k$) to $\Theta_0$ (dimension $k - q$):

    $$
    LR = 2\!\left[\log L(\hat\theta) - \log L(\hat\theta_0)\right]
    \xrightarrow{d} \chi^2_q
    $$

    Requires fitting **both** the restricted and unrestricted models.
    Generally preferred over the Wald test for testing variance parameters
    near boundaries.

=== "Score (LM) test"

    Test based on the score evaluated at the **restricted** MLE
    $\hat\theta_0$:

    $$
    S = s(\hat\theta_0)'\,\hat{\mathcal{I}}(\hat\theta_0)^{-1}\,s(\hat\theta_0)
    \xrightarrow{d} \chi^2_q,
    $$

    where $s(\theta) = \partial \log L / \partial \theta$. Requires fitting
    only the restricted model — computationally attractive when the restricted
    model is simpler.

---

## 7. Optimisation Algorithms

### 7.1 Overview

Maximising $\log L(\theta)$ is a nonlinear, possibly non-convex, optimisation
problem. `kalmanbox` exposes the following solvers via the `method` argument
to `model.fit()`:

| Method | Class | Hessian | Box constraints | Recommended for |
|:-------|:------|:--------|:----------------|:----------------|
| `bfgs` | Quasi-Newton | Approximated (BFGS) | No | Smooth likelihoods, moderate $k$ |
| `lbfgsb` | Quasi-Newton | Low-rank approx. | Yes | Large $k$, transformed parameters |
| `nm` | Simplex | None | No | Fallback; non-smooth or fragile derivatives |
| `newton` | Newton-Raphson | Numerical Hessian | No | Near the optimum, high accuracy |

### 7.2 BFGS

The **Broyden-Fletcher-Goldfarb-Shanno** algorithm maintains a positive
definite approximation $B_t \approx \nabla^2(-\log L)$ updated by

$$
B_{t+1} = B_t
  - \frac{B_t s_t s_t' B_t}{s_t' B_t s_t}
  + \frac{y_t y_t'}{y_t' s_t},
\quad
s_t = \theta_{t+1} - \theta_t,\quad
y_t = g_{t+1} - g_t,
$$

with $g_t = -\partial \log L/\partial\theta\rvert_{\theta_t}$ the negated
gradient. The iterate update is

$$
\theta_{t+1} = \theta_t - \alpha_t\, B_t^{-1} g_t,
$$

where $\alpha_t$ is the step length from a Wolfe-condition line search.
BFGS achieves **superlinear convergence** near the optimum. Storage cost is
$O(k^2)$.

### 7.3 L-BFGS-B

The **Limited-memory BFGS with Bound constraints** variant stores only the
$m_H$ most recent curvature pairs $(s_i, y_i)$, giving $O(m_H k)$ storage
(typically $m_H = 10$). Box constraints $\ell_i \leq \theta_i \leq u_i$ are
handled by gradient projection. This is the **recommended default** when the
parameter vector is large or when box constraints on transformed parameters
are needed.

### 7.4 Nelder-Mead Simplex

The **Nelder-Mead** algorithm is a derivative-free simplex method. It
evaluates $\log L$ at $k+1$ vertices of a simplex and updates the worst
vertex by reflection, expansion, or contraction. It is **slow** (linear
convergence at best) but robust when:

- the gradient is unavailable or unreliable (discontinuous likelihood);
- the model is fragile to parameter perturbations during the filter;
- BFGS repeatedly fails due to indefinite Hessian approximations.

Use as a fallback with `method='nm'`.

### 7.5 Newton-Raphson

The **Newton-Raphson** step is

$$
\theta_{t+1} = \theta_t + \alpha_t\,\mathcal{I}(\theta_t)^{-1}\, s(\theta_t),
$$

where $s(\theta_t) = \partial \log L / \partial\theta\rvert_{\theta_t}$ is the
score and $\mathcal{I}(\theta_t)$ is the observed information. It achieves
**quadratic convergence** near the MLE but requires evaluating and inverting
the Hessian at each iteration ($O(k^3)$ per step). Use with `method='newton'`
only when analytic gradients and the Hessian are available.

### 7.6 Starting Values

Choice of starting values is critical for non-convex problems:

**Default strategy (kalmanbox).**  Set all variance parameters to the
unconditional variance of $y$ (or a fraction of it). Set AR parameters to
zero. Set loadings to one. Run the diffuse filter from these values and use
the result as the starting point for BFGS.

**Grid search.**  For one or two variance parameters, evaluate the likelihood
on a coarse grid and select the grid point with highest value as the starting
point. This is the most reliable strategy for scalar structural models.

**Random multiple starts.**  For $k > 4$ draw 5–10 starting vectors from a
distribution over $\Theta$ (e.g., log-normal for variances, uniform for
loadings) and run BFGS from each. Retain the global best. The
`n_starts` argument to `model.fit()` controls this.

!!! tip "Recommendation"
    For models with $k \leq 6$ and no strong prior knowledge of the true
    optimum, use `n_starts=10` with `method='lbfgsb'`. For larger models,
    a two-stage approach — Nelder-Mead for 100 iterations followed by BFGS
    — tends to be robust.

---

## 8. Parameter Constraints and Transformations

The key insight is that constrained optimisation is equivalent to
**unconstrained optimisation in a reparametrised space** with bijective maps.
Let $\tilde\theta \in \mathbb{R}^k$ be the unconstrained vector and
$\theta = g(\tilde\theta)$ the constrained original.

### 8.1 Log Transform for Positive Parameters

For $\sigma^2 > 0$:

$$
\sigma^2 = \exp(\tilde\sigma), \qquad \tilde\sigma \in \mathbb{R}.
$$

The Jacobian is $\partial \sigma^2 / \partial \tilde\sigma = \sigma^2$, so the
chain rule gives

$$
\frac{\partial \log L}{\partial \tilde\sigma}
= \frac{\partial \log L}{\partial \sigma^2} \cdot \sigma^2.
$$

This transformation maps the positive half-line to all of $\mathbb{R}$ and
typically produces a more symmetric, better-conditioned likelihood surface.

### 8.2 Tanh (Logit) Transform for Bounded Parameters

For an AR coefficient $\phi \in (-1, 1)$ (stationarity):

$$
\phi = \tanh(\tilde\phi) = \frac{e^{\tilde\phi} - e^{-\tilde\phi}}{e^{\tilde\phi} + e^{-\tilde\phi}},
\qquad \tilde\phi \in \mathbb{R}.
$$

The Jacobian is $\partial \phi / \partial \tilde\phi = 1 - \phi^2$.

For a damping factor $\rho_c \in (0,1)$ the logistic function is used:

$$
\rho_c = \sigma(\tilde\rho) = \frac{1}{1 + e^{-\tilde\rho}},
\qquad \tilde\rho \in \mathbb{R}.
$$

### 8.3 Cholesky Parametrisation for Covariance Matrices

For a full positive-definite covariance matrix $\Sigma \in \mathbb{R}^{r \times r}$:

$$
\Sigma = L L', \quad L \text{ lower triangular with positive diagonal},
$$

optimise the $r(r+1)/2$ free elements of $L$ (diagonal elements on the log
scale: $L_{ii} = \exp(\tilde\ell_{ii})$). This guarantees positive
definiteness by construction and avoids eigenvalue constraints.

### 8.4 Signal-to-Noise Ratio

For the local level model, the SNR $q = \sigma^2_\eta / \sigma^2_\varepsilon$
is sufficient for filtering. Optimise $\tilde{q} = \log q \in \mathbb{R}$ and
recover $q = e^{\tilde{q}}$.

### 8.5 Gradient Transformation (Jacobian Correction)

The chain rule of differentiation gives the general formula for transforming
gradients from the constrained to the unconstrained space:

$$
\frac{\partial \log L}{\partial \tilde\theta}
= J(\tilde\theta)'\, \frac{\partial \log L}{\partial \theta},
\quad
J(\tilde\theta) = \frac{\partial \theta}{\partial \tilde\theta'}.
$$

For diagonal Jacobians (element-wise transforms), this reduces to element-wise
scaling. Optimisers that exploit gradients must therefore receive
$\partial \log L / \partial \tilde\theta$, not $\partial \log L / \partial \theta$.

!!! warning "Gradient sign convention"
    Most optimisation libraries **minimise** objectives. `kalmanbox` internally
    minimises $-\log L(\theta)$ and reverses signs before returning values.
    Ensure any custom likelihood functions follow the same convention.

---

## 9. Information Criteria

Information criteria penalise the maximised log-likelihood for model
complexity, enabling comparison of non-nested models estimated on the same
data. Let $\hat\theta$ be the MLE with $k$ free parameters and $T$
observations.

### 9.1 AIC — Akaike Information Criterion

$$
\boxed{AIC = -2\log L(\hat\theta) + 2k}
$$

**Derivation (Akaike, 1974).**  The AIC is motivated by minimising the
expected Kullback-Leibler divergence from the true density $p_0$ to the
fitted density $p_{\hat\theta}$:

$$
KL\bigl(p_0 \,\|\, p_{\hat\theta}\bigr)
= E_0\!\left[\log p_0(y) - \log p_{\hat\theta}(y)\right].
$$

Akaike showed that, to first order in $T$,

$$
-2\,E_0\!\left[\log L(\hat\theta)\right]
\approx -2\log L(\hat\theta) + 2k,
$$

where the bias correction $2k$ arises because $\log L(\hat\theta)$ is an
optimistically biased estimator of the expected log-likelihood. The AIC
targets **predictive accuracy** and may select models that are slightly
over-specified if the true model is not in the candidate set.

**Key property.**  AIC does **not** consistently select the true model in
general: as $T \to \infty$ it can favour over-parametrised models.

### 9.2 BIC — Bayesian Information Criterion (Schwarz, 1978)

$$
\boxed{BIC = -2\log L(\hat\theta) + k\log T}
$$

**Derivation (Schwarz, 1978).**  BIC is derived as an approximation to the
**marginal likelihood** (Bayesian evidence) of each model $\mathcal{M}_j$
under a flat prior:

$$
\log p(y \mid \mathcal{M}_j)
\approx \log L(\hat\theta_j) - \frac{k_j}{2}\log T + O(1).
$$

Minimising $BIC$ is equivalent to selecting the model with the highest
approximate Bayesian evidence, i.e., the posterior-most model under equal
prior probability. The penalty $k \log T$ grows faster than $2k$ for $T > 7$,
so BIC imposes **stronger penalisation** than AIC.

**Key property.**  BIC is **consistent**: if the true model belongs to the
candidate set and $T \to \infty$, BIC selects the true model with probability
approaching one. It tends to under-select in small samples relative to AIC.

### 9.3 HQIC — Hannan-Quinn Information Criterion

$$
\boxed{HQIC = -2\log L(\hat\theta) + 2k\log\log T}
$$

Hannan & Quinn (1979) derived HQIC as the criterion with the minimal penalty
that still achieves **weak consistency** (almost sure model selection as
$T \to \infty$). Its penalty grows more slowly than BIC but faster than AIC:

$$
\underbrace{2k}_{\text{AIC}} < \underbrace{2k\log\log T}_{\text{HQIC}} < \underbrace{k\log T}_{\text{BIC}}.
$$

HQIC is particularly useful in **high-frequency** or **very long** time series
where BIC's penalty is extremely large and AIC's over-fitting bias accumulates.

### 9.4 Comparison Table

| Criterion | Penalty | Consistent? | Preferred when |
|:----------|:--------|:------------|:---------------|
| AIC | $2k$ | No | Forecasting, prediction, short $T$ |
| AICc | $\frac{2k(k+1)}{T-k-1} + 2k$ | No | Small samples ($T/k < 40$) |
| BIC | $k\log T$ | Yes | Model identification, large $T$ |
| HQIC | $2k\log\log T$ | Weakly | Long series, moderate complexity |

### 9.5 Corrected AIC (AICc) for Small Samples

In small samples the bias correction in AIC is insufficiently large. The
corrected AIC of Hurvich & Tsai (1989) is:

$$
\boxed{AICc = AIC + \frac{2k(k+1)}{T - k - 1}.}
$$

As $T \to \infty$, $AICc \to AIC$. The correction term diverges as $T \to k+2$,
becoming prohibitively large and thereby discouraging over-parametrised models
in small samples. **Use AICc whenever $T/k < 40$.**

!!! example "Penalty growth"
    For $k = 3$ and $T = 50$:
    - AIC penalty = 6
    - AICc penalty $\approx$ 6 + 1.04 = 7.04
    - HQIC penalty $\approx$ 7.73
    - BIC penalty $\approx$ 11.74

---

## 10. Practical Diagnostics

### 10.1 Convergence Checks

A numerical optimiser is deemed to have converged only when:

1. **Gradient norm**: $\|s(\hat\theta)\|_\infty < \epsilon_g$, typically
   $\epsilon_g = 10^{-5}$. A large residual gradient indicates the solver
   stopped prematurely.
2. **Positive definite Hessian**: all eigenvalues of $\hat{\mathcal{I}}$ must
   be strictly positive. Negative eigenvalues indicate a saddle point or a
   maximum of the **negated** objective was not found.
3. **Relative function change**: $|\log L(\theta_{t+1}) - \log L(\theta_t)|
   / (1 + |\log L(\theta_t)|) < \epsilon_f$, typically $\epsilon_f = 10^{-8}$.

`kalmanbox` reports `results.converged`, `results.gradient_norm`, and
`results.hess_inv_eigvals` for inspection.

### 10.2 Profile Likelihood

For parameter $\theta_i$, the **profile log-likelihood** is

$$
\log L_P(\theta_i)
= \max_{\theta_{-i}} \log L(\theta_i, \theta_{-i}),
$$

where $\theta_{-i}$ denotes all parameters except $\theta_i$. Profile
likelihood intervals are defined by

$$
CI_P = \left\{\theta_i :
  \log L_P(\theta_i) \geq \log L(\hat\theta) - \frac{1}{2}\chi^2_{1,\alpha}
\right\},
$$

with $\chi^2_{1,0.05} \approx 3.84$ for a 95% interval. These intervals are
transformation-invariant, asymmetric, and more reliable than Wald intervals
near boundaries.

### 10.3 Likelihood Surface Visualisation

For models with two dominant parameters (e.g., the local level with
$\theta = (\sigma^2_\varepsilon, \sigma^2_\eta)$) the likelihood surface can
be plotted over a grid:

$$
\mathcal{L}_{ij} = \log L(\sigma^2_{\varepsilon,i}, \sigma^2_{\eta,j}),
\quad i,j = 1,\ldots,G.
$$

Contour plots reveal:

- **Elongated ridges**: high correlation between parameters (near
  non-identification).
- **Flat regions**: poorly determined directions; regularisation needed.
- **Multiple peaks**: global optimum may differ from BFGS solution.

### 10.4 Innovation Diagnostics

The standardised innovations should be i.i.d. $N(0, I_p)$:

$$
e_t = F_t^{-1/2}\, v_t \;\stackrel{H_0}{\sim}\; N(0, I_p).
$$

Standard diagnostic checks include:

| Test | Null hypothesis | Statistic |
|:-----|:----------------|:----------|
| Ljung-Box | No autocorrelation | $Q(h) = T(T+2)\sum_{j=1}^h \hat\rho_j^2/(T-j)$ |
| Jarque-Bera | Normality | $JB = T[(S^2/6) + (K-3)^2/24]$ |
| Q-Q plot | Normality | Visual, Kolmogorov-Smirnov |
| ARCH-LM | No conditional heteroskedasticity | Regress $e_t^2$ on lags |

Failure of the normality test often indicates model misspecification — the
chosen state-space structure may be inadequate for the data. Autocorrelated
innovations suggest the transition equation is missing dynamics. See
[`../user-guide/kalman/mle.md`](../user-guide/kalman/mle.md) for the
corresponding `kalmanbox` API calls.

---

## Example: MLE in kalmanbox

```python
from kalmanbox.structural import LocalLevelModel
import numpy as np

model = LocalLevelModel()
results = model.fit(y, method='bfgs', start_params=[0.5, 1.0])
print(results.summary())
print(f"Log-likelihood: {results.llf:.4f}")
print(f"AIC: {results.aic:.4f}, BIC: {results.bic:.4f}")
print(f"Parameters: {results.params}")
print(f"Std errors: {results.bse}")
```

The `results` object exposes:

| Attribute | Meaning |
|:----------|:--------|
| `results.llf` | Maximised log-likelihood $\log L(\hat\theta)$ |
| `results.params` | MLE $\hat\theta$ in constrained space |
| `results.bse` | Standard errors $\mathrm{se}(\hat\theta_i)$ |
| `results.aic` | AIC score |
| `results.bic` | BIC score |
| `results.aicc` | Corrected AIC |
| `results.hqic` | HQIC score |
| `results.converged` | Boolean convergence flag |
| `results.gradient_norm` | $\|s(\hat\theta)\|_\infty$ at solution |
| `results.innovations` | Standardised innovations $F_t^{-1/2} v_t$ |

!!! example "Multiple starts"
    ```python
    results = model.fit(
        y, method='lbfgsb', n_starts=10, random_state=42
    )
    # kalmanbox runs 10 random initialisations and returns the best
    ```

---

## Summary of Key Formulae

| Quantity | Formula |
|:---------|:--------|
| Log-likelihood | $-\frac{1}{2}\sum_t [p\log 2\pi + \log\lvert F_t\rvert + v_t' F_t^{-1} v_t]$ |
| Concentrated MLE of $\sigma^2$ | $\hat\sigma^2 = \frac{1}{pT}\sum_t v_t' \tilde{F}_t^{-1} v_t$ |
| Concentrated log-likelihood | $-\frac{pT}{2}[\log\hat\sigma^2 + 1] - \frac{1}{2}\sum_t \log\lvert\tilde{F}_t\rvert$ |
| Score | $\sum_t [-\frac{1}{2}\mathrm{tr}(F_t^{-1}\dot{F}_t) + \frac{1}{2}v_t'F_t^{-1}\dot{F}_t F_t^{-1}v_t + v_t'F_t^{-1}\dot{v}_t]$ |
| Standard errors | $\sqrt{[\hat{\mathcal{I}}^{-1}]_{ii}}$ |
| AIC | $-2\log L(\hat\theta) + 2k$ |
| BIC | $-2\log L(\hat\theta) + k\log T$ |
| AICc | $AIC + 2k(k+1)/(T-k-1)$ |
| HQIC | $-2\log L(\hat\theta) + 2k\log\log T$ |

---

## References

- Kalman, R.E. (1960). A new approach to linear filtering and prediction
  problems. *Journal of Basic Engineering*, **82**(1), 35–45.

- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.

- Akaike, H. (1974). A new look at the statistical model identification.
  *IEEE Transactions on Automatic Control*, **19**(6), 716–723.

- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of
  Statistics*, **6**(2), 461–464.

- Hannan, E.J. & Quinn, B.G. (1979). The determination of the order of an
  autoregression. *Journal of the Royal Statistical Society: Series B*,
  **41**(2), 190–195.

- Koopman, S.J. & Shephard, N. (1992). Exact score for time series models in
  state space form. *Biometrika*, **79**(4), 823–826.

- Koopman, S.J. & Durbin, J. (2003). Filtering and smoothing of state vector
  for diffuse state-space models. *Journal of Time Series Analysis*,
  **24**(1), 85–98.

- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press.

- Hurvich, C.M. & Tsai, C.L. (1989). Regression and time series model
  selection in small samples. *Biometrika*, **76**(2), 297–307.
