# Bayesian Estimation Theory for State-Space Models

This page provides a self-contained, mathematically rigorous treatment of Bayesian
parameter and state estimation in Gaussian linear state-space models. It covers the
philosophical foundations, conjugate prior theory, the Forward Filter Backward Sampling
(FFBS) algorithm, Gibbs sampling, MCMC convergence diagnostics, Metropolis-Hastings
extensions, posterior inference, and the connection to EM. Companion pages are
[Kalman filter theory](kalman-theory.md), [Smoothing theory](smoothing-theory.md),
and [MLE theory](mle-theory.md). The practical user-facing guides are
[Gibbs sampler](../user-guide/bayesian/gibbs.md),
[FFBS](../user-guide/bayesian/ffbs.md), and
[Posterior diagnostics](../user-guide/bayesian/posterior-diagnostics.md).

Throughout this page the Durbin & Koopman (2012) convention is used, augmented with
Bayesian notation from Frühwirth-Schnatter (1994) and Carter & Kohn (1994).

| Symbol | Dimension | Meaning |
|:-------|:----------|:--------|
| $y_t$ | $p \times 1$ | Observed vector at time $t$ |
| $\alpha_t$ | $m \times 1$ | Latent state vector |
| $Z_t$ | $p \times m$ | Design (observation) matrix |
| $T_t$ | $m \times m$ | Transition matrix |
| $R_t$ | $m \times r$ | Noise-selection matrix |
| $H_t$ | $p \times p$ | Observation noise covariance |
| $Q_t$ | $r \times r$ | State noise covariance |
| $\theta$ | $k \times 1$ | Parameter vector (unknown) |
| $\alpha_{1:T}$ | — | Full state path $(\alpha_1, \ldots, \alpha_T)$ |
| $y_{1:T}$ | — | Full observation sequence $(y_1, \ldots, y_T)$ |
| $G$ | scalar | Number of MCMC draws |
| $\mathcal{IW}$ | — | Inverse-Wishart distribution |
| $\mathcal{MN}$ | — | Matrix-normal distribution |

---

## 1. The Bayesian Framework for State-Space Models

### 1.1 Classical vs Bayesian Paradigm

In the **classical (frequentist)** approach the parameter vector $\theta$ is a fixed but
unknown quantity. Inference proceeds by maximising the observed-data likelihood
$p(y_{1:T} \mid \theta)$ over $\theta$, yielding the MLE $\hat\theta$. Uncertainty
about $\hat\theta$ is quantified through the asymptotic Fisher information matrix (see
[MLE theory](mle-theory.md)).

In the **Bayesian approach** $\theta$ is treated as a random variable. Prior knowledge
about $\theta$ is encoded in a probability distribution $p(\theta)$ — the *prior* — and
inference updates that distribution using the observed data via Bayes' theorem:

$$
p(\theta \mid y_{1:T}) = \frac{p(y_{1:T} \mid \theta)\, p(\theta)}{p(y_{1:T})}.
\tag{1.1}
$$

The resulting *posterior* $p(\theta \mid y_{1:T})$ summarises all information about
$\theta$ after observing the data. In state-space models there is a further
complication: the latent state path $\alpha_{1:T}$ is also unknown and must be treated
as a random quantity alongside $\theta$.

!!! note "Why treat states as random?"
    Even when $\theta$ is known, $\alpha_{1:T}$ is never directly observed. The Kalman
    smoother gives the conditional mean $E[\alpha_t \mid y_{1:T}, \theta]$, but a
    full Bayesian analysis requires propagating the *entire posterior distribution*
    of the state path, not just its mean. This is essential when $\theta$ is also
    unknown, because ignoring state uncertainty leads to underestimated posterior
    variance for $\theta$.

### 1.2 Complete Data Likelihood

Let the model be defined by the Gaussian linear state-space system:

$$
y_t = Z_t \alpha_t + d_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, H_t),
\tag{1.2}
$$

$$
\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q_t),
\tag{1.3}
$$

with $\alpha_1 \sim \mathcal{N}(a_1, P_1)$ and all disturbances mutually independent.

The *complete data likelihood* — the joint density of observations **and** states —
factorises as

$$
p(y_{1:T},\, \alpha_{1:T} \mid \theta)
  = p(y_{1:T} \mid \alpha_{1:T}, \theta)\; p(\alpha_{1:T} \mid \theta).
\tag{1.4}
$$

Because observations are conditionally independent given states:

$$
p(y_{1:T} \mid \alpha_{1:T}, \theta)
  = \prod_{t=1}^T p(y_t \mid \alpha_t, \theta)
  = \prod_{t=1}^T \mathcal{N}(y_t;\; Z_t \alpha_t + d_t,\; H_t).
\tag{1.5}
$$

The state path has a Markov structure:

$$
p(\alpha_{1:T} \mid \theta)
  = p(\alpha_1 \mid \theta) \prod_{t=1}^{T-1} p(\alpha_{t+1} \mid \alpha_t, \theta)
  = \mathcal{N}(\alpha_1; a_1, P_1)
    \prod_{t=1}^{T-1} \mathcal{N}(\alpha_{t+1};\; T_t \alpha_t + c_t,\; R_t Q_t R_t^\top).
\tag{1.6}
$$

### 1.3 Target: Joint Posterior

The object of inference is the **joint posterior** of states and parameters:

$$
p(\alpha_{1:T}, \theta \mid y_{1:T})
  \propto
  p(y_{1:T} \mid \alpha_{1:T}, \theta)\;
  p(\alpha_{1:T} \mid \theta)\;
  p(\theta).
\tag{1.7}
$$

Marginalising over $\alpha_{1:T}$ yields the parameter posterior

$$
p(\theta \mid y_{1:T}) = \int p(\alpha_{1:T}, \theta \mid y_{1:T})\, d\alpha_{1:T},
\tag{1.8}
$$

while marginalising over $\theta$ gives the smoothed state distribution

$$
p(\alpha_{1:T} \mid y_{1:T}) = \int p(\alpha_{1:T}, \theta \mid y_{1:T})\, d\theta.
\tag{1.9}
$$

### 1.4 Why Direct Sampling is Intractable

Direct sampling from $p(\alpha_{1:T}, \theta \mid y_{1:T})$ is generally impossible
because:

1. **High dimensionality.** The state path lives in $\mathbb{R}^{mT}$. For $m = 5$,
   $T = 1000$, the state space is $\mathbb{R}^{5000}$.
2. **Non-standard form.** Even with Gaussian innovations the parameter posterior
   $p(\theta \mid y_{1:T})$ does not belong to a known conjugate family — the
   observed-data likelihood $p(y_{1:T} \mid \theta)$ is the output of a Kalman recursion
   and is a complicated nonlinear function of $\theta$.
3. **Coupled unknowns.** States depend on parameters through $T_t, Z_t, H_t, Q_t$, so
   integrating out $\alpha_{1:T}$ analytically while also marginalising over $\theta$
   is infeasible except in trivial special cases.

### 1.5 Data Augmentation: Treating States as Missing Data

The key insight, due to Tanner & Wong (1987) and applied to state-space models by
Frühwirth-Schnatter (1994), is to treat $\alpha_{1:T}$ as **missing data**. Define the
augmented complete-data problem and exploit the following conditional independence
structure:

$$
\theta \perp y_{1:T} \mid \alpha_{1:T},
\qquad
\alpha_{1:T} \perp \theta^{-Q} \mid \alpha_{1:T-1}, y_{1:T}, Q,
\tag{1.10}
$$

where the second statement uses the fact that, given the full state path, the
observation equations and transition equations decouple. This allows Gibbs sampling
to iterate between two tractable conditionals:

$$
p(\alpha_{1:T} \mid \theta, y_{1:T}) \qquad \text{(block draw via FFBS, Section 3)},
$$

$$
p(\theta \mid \alpha_{1:T}, y_{1:T}) \qquad \text{(conjugate closed-form draw, Section 4)}.
$$

!!! abstract "Frühwirth-Schnatter (1994) Framework"
    Sylvia Frühwirth-Schnatter's landmark paper "*Data Augmentation and Dynamic Linear
    Models*" (JTSA, 1994) established the theoretical basis for Bayesian inference in
    state-space models via data augmentation. The paper showed that treating the
    latent state path as missing data converts an otherwise intractable posterior into
    two conditionals that are standard distributions, making Gibbs sampling feasible
    for general dynamic linear models.

---

## 2. Conjugate Prior Theory

### 2.1 Gaussian State-Space and Natural Conjugates

When the model is Gaussian and linear, several parameter blocks admit **natural
conjugate priors** — priors in the same exponential family as the likelihood, so that
the posterior belongs to the same family as the prior. Given the complete data
$\{y_{1:T}, \alpha_{1:T}\}$, the sufficient statistics are fixed linear functions of
the data, and posterior hyperparameters update in closed form.

We consider a time-invariant system with constant $Z, T, H, Q$ for clarity. Extensions
to time-varying matrices follow analogously.

### 2.2 Prior on Observation Noise Covariance $H$

**Prior:** Inverse-Wishart with degrees-of-freedom $\nu_0$ and scale matrix $S_0$:

$$
H \sim \mathcal{IW}(\nu_0,\, S_0),
\qquad
p(H) \propto |H|^{-(\nu_0 + p + 1)/2}
  \exp\!\left(-\tfrac{1}{2}\operatorname{tr}(S_0 H^{-1})\right).
\tag{2.1}
$$

The prior mean is $E[H] = S_0 / (\nu_0 - p - 1)$ for $\nu_0 > p + 1$.

**Sufficient statistics given $\alpha_{1:T}$:** Define the observation residuals
$e_t = y_t - Z\alpha_t - d$. The scatter matrix is

$$
S_{ee} = \sum_{t=1}^T e_t e_t^\top.
\tag{2.2}
$$

**Posterior:**

$$
H \mid \alpha_{1:T}, y_{1:T} \sim \mathcal{IW}\!\left(\nu_0 + T,\; S_0 + S_{ee}\right).
\tag{2.3}
$$

This is a clean rank-$T$ update: each observation contributes one additional degree of
freedom and adds the outer product $e_t e_t^\top$ to the scale.

### 2.3 Prior on State Noise Covariance $Q$

**Prior:** Inverse-Wishart with degrees-of-freedom $\nu_0^Q$ and scale matrix $S_0^Q$:

$$
Q \sim \mathcal{IW}(\nu_0^Q,\, S_0^Q).
\tag{2.4}
$$

**Sufficient statistics given $\alpha_{1:T}$:** Define the one-step transition residuals
$\eta_t = R^{-1}(\alpha_{t+1} - T\alpha_t - c)$ (assuming $R$ is square and invertible;
the non-square case uses a pseudo-inverse). The scatter matrix is

$$
S_{\eta\eta} = \sum_{t=1}^{T-1} \eta_t \eta_t^\top.
\tag{2.5}
$$

**Posterior:**

$$
Q \mid \alpha_{1:T} \sim \mathcal{IW}\!\left(\nu_0^Q + T - 1,\; S_0^Q + S_{\eta\eta}\right).
\tag{2.6}
$$

!!! tip "Scalar variances"
    For univariate models ($p = r = 1$) the inverse-Wishart reduces to the
    **inverse-Gamma** distribution. If $H \sim \mathcal{IG}(a_0, b_0)$, then
    $H \mid \alpha_{1:T}, y_{1:T} \sim \mathcal{IG}(a_0 + T/2,\; b_0 + S_{ee}/2)$.

### 2.4 Prior on Loading Matrix $\Lambda$ (Factor Models)

In factor models the observation equation is $y_t = \Lambda f_t + \varepsilon_t$ where
$\Lambda \in \mathbb{R}^{p \times k}$ is the loading matrix and $f_t$ are factors.

**Prior:** Matrix-normal conditional on $H$:

$$
\Lambda \mid H \sim \mathcal{MN}(M_0,\, P_0,\, H),
\tag{2.7}
$$

with density

$$
p(\Lambda \mid H) \propto |H|^{-k/2}
  \exp\!\left(-\tfrac{1}{2}\operatorname{tr}\!\left[H^{-1}
    (\Lambda - M_0) P_0^{-1} (\Lambda - M_0)^\top\right]\right).
\tag{2.8}
$$

**Sufficient statistics given factor path $f_{1:T}$:**

$$
S_{ff} = \sum_{t=1}^T f_t f_t^\top, \qquad
S_{yf} = \sum_{t=1}^T y_t f_t^\top.
\tag{2.9}
$$

**Posterior:**

$$
\Lambda \mid H, f_{1:T}, y_{1:T}
  \sim \mathcal{MN}\!\left(M_T,\, P_T,\, H\right),
\tag{2.10}
$$

where the updated row-coefficient covariance and mean are

$$
P_T^{-1} = P_0^{-1} + S_{ff},
\qquad
M_T = (S_{yf} + M_0 P_0^{-1})\, P_T.
\tag{2.11}
$$

### 2.5 Prior on Transition Matrix $T$

**Prior:** Matrix-normal (row-by-row independent Gaussian):

$$
T \mid Q \sim \mathcal{MN}(T_0,\, V_0,\, Q).
\tag{2.12}
$$

**Sufficient statistics given state path $\alpha_{1:T}$:**

$$
S_{\alpha\alpha} = \sum_{t=1}^{T-1} \alpha_t \alpha_t^\top,
\qquad
S_{\alpha\alpha'} = \sum_{t=1}^{T-1} \alpha_{t+1} \alpha_t^\top.
\tag{2.13}
$$

**Posterior:**

$$
T \mid Q, \alpha_{1:T}
  \sim \mathcal{MN}\!\left(T_T,\, V_T,\, Q\right),
\tag{2.14}
$$

$$
V_T^{-1} = V_0^{-1} + S_{\alpha\alpha},
\qquad
T_T = (S_{\alpha\alpha'} + T_0 V_0^{-1})\, V_T.
\tag{2.15}
$$

!!! note "Closed-form posteriors: the key advantage"
    All four parameter blocks above have **closed-form posterior distributions** given
    the complete state path $\alpha_{1:T}$. This is the critical property that makes the
    Gibbs sampler practical: the "M-step" of Bayesian estimation is a simple draw from
    a known distribution, not an optimisation. The only non-trivial step is drawing a
    new state path, which is handled by FFBS (Section 3).

### 2.6 Prior Elicitation Guidelines

| Parameter | Prior | Weakly informative choice |
|:----------|:------|:--------------------------|
| $H$ | $\mathcal{IW}(\nu_0, S_0)$ | $\nu_0 = p + 2$, $S_0 = \hat\sigma^2 I$ |
| $Q$ | $\mathcal{IW}(\nu_0^Q, S_0^Q)$ | $\nu_0^Q = r + 2$, $S_0^Q = \kappa \hat\sigma^2 I$, $\kappa \ll 1$ |
| $\Lambda$ | $\mathcal{MN}(0, c I, H)$ | $c = 10$ (diffuse rows) |
| $T$ | $\mathcal{MN}(I, c I, Q)$ | Near-unit-root prior for persistence |

Here $\hat\sigma^2$ is a rough scale estimate from the data (e.g., $\mathrm{Var}(y_t)$).

---

## 3. Forward Filter Backward Sampling (FFBS)

### 3.1 The Goal: Draw from the Smoothing Distribution

Given $\theta$ and the data $y_{1:T}$, we want to draw a sample $\alpha_{1:T}^{(g)}$
from the **joint smoothing distribution**

$$
p(\alpha_{1:T} \mid y_{1:T}, \theta).
\tag{3.1}
$$

This is a high-dimensional Gaussian — it is a joint distribution over $mT$ variables —
but it has a specific Markov structure that FFBS exploits.

### 3.2 Factorisation via Markov Property

By the Markov property of the state-space model:

$$
p(\alpha_{1:T} \mid y_{1:T}, \theta)
  = p(\alpha_T \mid y_{1:T}, \theta)
    \prod_{t=T-1}^{1} p(\alpha_t \mid \alpha_{t+1}, y_{1:t}, \theta).
\tag{3.2}
$$

The factorisation runs **backwards** in time. The key identity is

$$
p(\alpha_t \mid \alpha_{t+1},\, y_{1:T},\, \theta)
  = p(\alpha_t \mid \alpha_{t+1},\, y_{1:t},\, \theta).
\tag{3.3}
$$

**Proof.** Given $\alpha_{t+1}$, the future observations $y_{t+1:T}$ are conditionally
independent of $\alpha_t$ (because the Markov chain screens off the past from the
future). Formally, $\alpha_t \perp y_{t+1:T} \mid \alpha_{t+1}, \theta$, which follows
from the d-separation in the state-space DAG. $\square$

### 3.3 Forward Pass: Kalman Filter

Run the standard Kalman filter (see [Kalman filter theory](kalman-theory.md)) forward
through $t = 1, \ldots, T$, storing at each step the **filtered mean and covariance**:

$$
a_{t|t} = E[\alpha_t \mid y_{1:t}, \theta],
\qquad
P_{t|t} = \mathrm{Var}(\alpha_t \mid y_{1:t}, \theta).
\tag{3.4}
$$

Also store the **one-step predicted** quantities:

$$
a_{t+1|t} = T_t a_{t|t} + c_t,
\qquad
P_{t+1|t} = T_t P_{t|t} T_t^\top + R_t Q_t R_t^\top.
\tag{3.5}
$$

At $t = T$, draw $\alpha_T \sim \mathcal{N}(a_{T|T}, P_{T|T})$.

### 3.4 Backward Sampling: Deriving $h_t$ and $H_t^*$

For $t = T-1, T-2, \ldots, 1$, we need the conditional distribution
$p(\alpha_t \mid \alpha_{t+1}, y_{1:t}, \theta)$. Because all quantities are Gaussian,
this is also Gaussian.

**Joint distribution of $(\alpha_t, \alpha_{t+1})$ given $y_{1:t}$:**

$$
\begin{pmatrix} \alpha_t \\ \alpha_{t+1} \end{pmatrix}
\Bigg| y_{1:t} \sim \mathcal{N}\!\left(
  \begin{pmatrix} a_{t|t} \\ a_{t+1|t} \end{pmatrix},
  \begin{pmatrix} P_{t|t} & P_{t|t} T_t^\top \\
                  T_t P_{t|t} & P_{t+1|t} \end{pmatrix}
\right).
\tag{3.6}
$$

**Conditional distribution** (from standard Gaussian conditioning):

$$
\alpha_t \mid \alpha_{t+1}, y_{1:t}
  \sim \mathcal{N}(h_t, H_t^*),
\tag{3.7}
$$

where the **backward gain** $J_t$, the **backward mean** $h_t$, and the **backward
covariance** $H_t^*$ are:

$$
\boxed{
J_t = P_{t|t} T_t^\top P_{t+1|t}^{-1},
}
\tag{3.8}
$$

$$
\boxed{
h_t = a_{t|t} + J_t(\alpha_{t+1} - a_{t+1|t}),
}
\tag{3.9}
$$

$$
\boxed{
H_t^* = P_{t|t} - J_t P_{t+1|t} J_t^\top.
}
\tag{3.10}
$$

Note that $H_t^*$ does not depend on $\alpha_{t+1}$; it can be precomputed in the
forward pass. This is the same backward gain that appears in the RTS smoother
(see [Smoothing theory](smoothing-theory.md)), making FFBS closely related to
the simulation smoother of Durbin & Koopman (2002).

!!! definition "FFBS Algorithm"
    **Forward pass** ($t = 1, \ldots, T$):

    1. Run the Kalman filter, storing $(a_{t|t}, P_{t|t}, a_{t+1|t}, P_{t+1|t})$ for all $t$.
    2. Precompute $J_t = P_{t|t} T_t^\top P_{t+1|t}^{-1}$ and $H_t^* = P_{t|t} - J_t P_{t+1|t} J_t^\top$ for $t = 1, \ldots, T-1$.

    **Backward sampling** ($t = T, T-1, \ldots, 1$):

    1. Draw $\alpha_T \sim \mathcal{N}(a_{T|T}, P_{T|T})$.
    2. For $t = T-1, \ldots, 1$:
       - Compute $h_t = a_{t|t} + J_t(\alpha_{t+1} - a_{t+1|t})$.
       - Draw $\alpha_t \sim \mathcal{N}(h_t, H_t^*)$.
    3. Return the sampled path $\alpha_{1:T}^{(g)} = (\alpha_1, \ldots, \alpha_T)$.

### 3.5 Properties of FFBS

**Exactness.** FFBS draws **exactly** from $p(\alpha_{1:T} \mid y_{1:T}, \theta)$ in
the Gaussian linear case. Unlike MCMC for the states, FFBS produces i.i.d. draws (each
call produces an independent sample), which has crucial implications for mixing when
used inside a Gibbs sampler.

**Computational complexity.** The forward pass costs $O(Tm^3)$ (Kalman filter with
$m \times m$ matrix inversions). The backward pass costs $O(Tm^2)$ (matrix-vector
products). Total cost is $O(Tm^3)$ per FFBS draw.

**Numerical stability.** The backward covariance $H_t^*$ can become indefinite due
to floating-point errors. In practice, enforce symmetry: $H_t^* \leftarrow (H_t^* +
H_t^{*\top})/2$ and use Cholesky-based sampling.

### 3.6 Connection to Simulation Smoother

The **simulation smoother** of Durbin & Koopman (2002) achieves the same goal —
sampling from $p(\alpha_{1:T} \mid y_{1:T}, \theta)$ — but operates in disturbance
space rather than state space. It samples the noise sequences $(\varepsilon_{1:T},
\eta_{1:T})$ and then constructs $\alpha_{1:T}$ deterministically. The two algorithms
produce identical output distributions; the simulation smoother is often preferred for
numerical stability when $m$ is large (see [Smoothing theory](smoothing-theory.md) for
the full derivation).

!!! abstract "Historical Note: Carter & Kohn (1994) and Frühwirth-Schnatter (1994)"
    FFBS was independently derived by Carter & Kohn ("On Gibbs Sampling for State-Space
    Models", *Biometrika*, 1994) and Frühwirth-Schnatter ("Data Augmentation and Dynamic
    Linear Models", *JTSA*, 1994). Carter & Kohn emphasised the algorithm and its use
    inside Gibbs sampling; Frühwirth-Schnatter provided the broader data-augmentation
    framework. Both papers remain foundational references in Bayesian time-series
    econometrics.

---

## 4. Gibbs Sampler for State-Space Models

### 4.1 Full Conditional Decomposition

The Gibbs sampler (Geman & Geman, 1984; Gelfand & Smith, 1990) constructs a Markov
chain that converges to the target joint posterior by iteratively sampling each
parameter block from its full conditional distribution — the distribution of that block
given all other blocks and the data.

For the Gaussian linear state-space model, the joint posterior

$$
p(\alpha_{1:T}, H, Q, T_{\mathrm{mat}}, \Lambda \mid y_{1:T})
$$

decomposes into the following **full conditionals**:

| Block | Full conditional | Sampler |
|:------|:----------------|:--------|
| $\alpha_{1:T}$ | $p(\alpha_{1:T} \mid \theta, y_{1:T})$ | FFBS (Section 3) |
| $H$ | $p(H \mid \alpha_{1:T}, \theta_{-H}, y_{1:T})$ | Inverse-Wishart draw |
| $Q$ | $p(Q \mid \alpha_{1:T}, \theta_{-Q}, y_{1:T})$ | Inverse-Wishart draw |
| $T_{\mathrm{mat}}$ | $p(T_{\mathrm{mat}} \mid \alpha_{1:T}, Q, y_{1:T})$ | Matrix-normal draw |
| $\Lambda$ | $p(\Lambda \mid \alpha_{1:T}, H, y_{1:T})$ | Matrix-normal draw |

Here $\theta_{-X}$ denotes all parameters except $X$.

### 4.2 Complete Gibbs Algorithm

!!! definition "Gibbs Sampler for State-Space Models"
    **Inputs:** Observed data $y_{1:T}$; prior hyperparameters $\{\nu_0, S_0,
    \nu_0^Q, S_0^Q, M_0, P_0, T_0, V_0\}$; initial parameter values $\theta^{(0)}$;
    number of iterations $G$.

    **Initialise:** Set $g = 0$. Choose starting values
    $H^{(0)}, Q^{(0)}, T^{(0)}, \Lambda^{(0)}$ (e.g., MLE estimates or prior means).

    **For** $g = 1, 2, \ldots, G$:

    **Step 1 — Draw states (FFBS):**

    $$\alpha_{1:T}^{(g)} \sim p(\alpha_{1:T} \mid \theta^{(g-1)}, y_{1:T})$$

    using the Forward Filter Backward Sampling algorithm (Section 3) with the
    current parameter values $\theta^{(g-1)} = (H^{(g-1)}, Q^{(g-1)}, \ldots)$.

    **Step 2a — Draw observation noise covariance:**

    Compute $S_{ee}^{(g)} = \sum_{t=1}^T (y_t - Z\alpha_t^{(g)} - d)(y_t - Z\alpha_t^{(g)} - d)^\top$.

    $$H^{(g)} \sim \mathcal{IW}\!\left(\nu_0 + T,\; S_0 + S_{ee}^{(g)}\right).$$

    **Step 2b — Draw state noise covariance:**

    Compute $S_{\eta\eta}^{(g)} = \sum_{t=1}^{T-1} \eta_t^{(g)} \eta_t^{(g)\top}$
    where $\eta_t^{(g)} = R^{-1}(\alpha_{t+1}^{(g)} - T^{(g-1)}\alpha_t^{(g)} - c)$.

    $$Q^{(g)} \sim \mathcal{IW}\!\left(\nu_0^Q + T - 1,\; S_0^Q + S_{\eta\eta}^{(g)}\right).$$

    **Step 2c — Draw transition matrix (if unknown):**

    Compute $S_{\alpha\alpha}^{(g)} = \sum_{t=1}^{T-1}\alpha_t^{(g)}\alpha_t^{(g)\top}$,
    $S_{\alpha\alpha'}^{(g)} = \sum_{t=1}^{T-1}\alpha_{t+1}^{(g)}\alpha_t^{(g)\top}$.

    $$V_T^{-1} = V_0^{-1} + S_{\alpha\alpha}^{(g)}, \qquad T_T = (S_{\alpha\alpha'}^{(g)} + T_0 V_0^{-1}) V_T.$$

    $$T^{(g)} \sim \mathcal{MN}\!\left(T_T,\; V_T,\; Q^{(g)}\right).$$

    **Step 2d — Draw loading matrix (factor models):**

    $$P_T^{-1} = P_0^{-1} + S_{ff}^{(g)}, \qquad M_T = (S_{yf}^{(g)} + M_0 P_0^{-1}) P_T.$$

    $$\Lambda^{(g)} \sim \mathcal{MN}\!\left(M_T,\; P_T,\; H^{(g)}\right).$$

    **Update:** Set $\theta^{(g)} = (H^{(g)}, Q^{(g)}, T^{(g)}, \Lambda^{(g)})$.

    **Output:** Posterior draws $\{(\alpha_{1:T}^{(g)}, \theta^{(g)})\}_{g=B+1}^G$
    after discarding the first $B$ burn-in iterations.

### 4.3 Why Conjugacy Makes the Parameter Draws Trivial

Given the complete state path $\alpha_{1:T}^{(g)}$, the **sufficient statistics**
$(S_{ee}, S_{\eta\eta}, S_{\alpha\alpha}, S_{\alpha\alpha'}, S_{ff}, S_{yf})$ are
fixed numbers. The parameter posteriors then become standard distributions:

- **No optimisation required.** In the EM algorithm (see Section 8), the M-step
  maximises the expected complete-data log-likelihood, which requires solving a
  system of equations. In Gibbs sampling, the analogous step is a **single random
  draw** from the posterior, implemented in two lines of code.
- **Uncertainty propagated automatically.** Each parameter draw accounts for the
  uncertainty in $\alpha_{1:T}^{(g)}$ through the random realisation of states.
- **Block independence.** Given $\alpha_{1:T}^{(g)}$, the posteriors of $H$, $Q$,
  $T$, and $\Lambda$ are independent of each other (they depend on different
  sufficient statistics), so Steps 2a–2d can be performed in any order or in
  parallel.

### 4.4 Sufficient Statistics: Summary

For the time-invariant model, all sufficient statistics are $O(T)$ running sums that
can be accumulated in a single pass through $\alpha_{1:T}^{(g)}$:

$$
S_{ee} = \sum_{t=1}^T e_t e_t^\top, \quad e_t = y_t - Z\alpha_t - d,
\tag{4.1}
$$

$$
S_{\alpha\alpha} = \sum_{t=1}^{T-1} \alpha_t \alpha_t^\top, \quad
S_{\alpha\alpha'} = \sum_{t=1}^{T-1} \alpha_{t+1} \alpha_t^\top.
\tag{4.2}
$$

This $O(Tm^2)$ cost for accumulation is dominated by the $O(Tm^3)$ cost of FFBS.

---

## 5. MCMC Convergence Theory

### 5.1 Ergodicity and Stationarity

The Gibbs sampler generates a Markov chain $\{(\alpha_{1:T}^{(g)}, \theta^{(g)})\}$.
For this chain to be useful, it must converge to the target posterior. Three conditions
are sufficient:

1. **Irreducibility.** The chain can reach any region of positive posterior probability
   from any starting point. Guaranteed when each full conditional has full support on
   the relevant parameter space (e.g., $\mathcal{IW}$ has support on the positive
   definite cone).
2. **Aperiodicity.** The chain does not cycle deterministically. Satisfied for
   continuous-valued Gibbs samplers.
3. **Positive recurrence.** The chain returns to any region in finite expected time.
   Guaranteed for proper posteriors (i.e., when $p(\theta)$ is a proper prior and the
   likelihood is bounded).

Under these conditions, the empirical distribution of $\{(\alpha_{1:T}^{(g)},
\theta^{(g)})\}_{g=1}^G$ converges to the target joint posterior as $G \to \infty$
(Tierney, 1994).

!!! warning "Improper priors"
    Using improper priors (e.g., $p(\theta) \propto 1$) can lead to improper posteriors
    in state-space models — the posterior may not integrate to a finite constant, making
    Gibbs sampling invalid. Always verify posterior propriety analytically or use weakly
    informative proper priors. See Hobert & Casella (1996) for a rigorous treatment.

### 5.2 Burn-in and Thinning

**Burn-in.** The initial $B$ draws are discarded to reduce dependence on the starting
values $\theta^{(0)}$. Typical choices:

| Setting | Recommended burn-in |
|:--------|:-------------------|
| Simple model, good starting values | $B = 500$ |
| Complex model, diffuse start | $B = 2{,}000$–$5{,}000$ |
| High autocorrelation detected | $B = 10{,}000$+ |

**Thinning.** Every $k$-th draw is retained to reduce storage and autocorrelation.
While thinning does not increase statistical efficiency per unit of computation, it
reduces memory requirements for large $m$ and $T$. A common choice is $k = 5$–$10$,
but **thinning is not necessary for valid inference** — using all post-burn-in draws
is statistically optimal (Link & Eaton, 2012).

### 5.3 Gelman-Rubin $\hat{R}$ Statistic

The Gelman-Rubin statistic (Gelman & Rubin, 1992) assesses convergence by running
$C \geq 2$ independent chains from over-dispersed starting values and comparing
within-chain to between-chain variance.

For a scalar quantity of interest $\psi$ with $C$ chains of length $G$ each:

**Between-chain variance:**

$$
B = \frac{G}{C - 1} \sum_{c=1}^C (\bar\psi_{c\cdot} - \bar\psi_{\cdot\cdot})^2,
\qquad
\bar\psi_{c\cdot} = \frac{1}{G}\sum_{g=1}^G \psi^{(c,g)},
\quad
\bar\psi_{\cdot\cdot} = \frac{1}{C}\sum_{c=1}^C \bar\psi_{c\cdot}.
\tag{5.1}
$$

**Within-chain variance:**

$$
W = \frac{1}{C} \sum_{c=1}^C s_c^2,
\qquad
s_c^2 = \frac{1}{G-1}\sum_{g=1}^G (\psi^{(c,g)} - \bar\psi_{c\cdot})^2.
\tag{5.2}
$$

**Potential scale reduction factor:**

$$
\hat{R} = \sqrt{\frac{\hat V}{W}},
\qquad
\hat V = \frac{G-1}{G} W + \frac{1}{G} B.
\tag{5.3}
$$

!!! definition "Convergence criterion"
    If all chains have converged to the same stationary distribution, then $B \approx W$
    and $\hat R \approx 1$. The conventional threshold is $\hat R < 1.1$ for each
    parameter (Gelman et al., 2013, Chapter 11). Values $\hat R > 1.2$ indicate
    serious convergence problems.

### 5.4 Effective Sample Size

MCMC draws are autocorrelated, which reduces the effective number of independent
samples. The **effective sample size** (ESS) is

$$
n_{\mathrm{eff}} = \frac{G}{1 + 2\sum_{k=1}^\infty \rho_k},
\tag{5.4}
$$

where $\rho_k = \mathrm{Corr}(\psi^{(g)}, \psi^{(g+k)})$ is the lag-$k$
autocorrelation of the chain. In practice $\rho_k$ is estimated from the chain and the
infinite sum is truncated when $\hat\rho_k + \hat\rho_{k+1} < 0$ (the Geyer monotone
sequence estimator).

A rule of thumb: target $n_{\mathrm{eff}} \geq 400$ for reliable posterior mean and
standard deviation estimates; $n_{\mathrm{eff}} \geq 1000$ for reliable 95% credible
intervals.

**Implications for FFBS-based Gibbs:** Because FFBS draws states exactly and
independently, the state path component of the chain mixes in a single step. The
bottleneck for $n_{\mathrm{eff}}$ is typically the parameter blocks $\theta$, which
mix more slowly when parameters and states are highly correlated (e.g., when $Q$ is
small relative to $H$, the posterior of $\alpha_{1:T}$ is tightly concentrated and
$Q$ is poorly identified).

### 5.5 Geweke Z-Test

The Geweke (1992) diagnostic tests convergence of a single chain by comparing the mean
of the first fraction $f_A$ (e.g., 10%) of draws to the mean of the last fraction $f_B$
(e.g., 50%) of draws, after burn-in:

$$
Z = \frac{\bar\psi_A - \bar\psi_B}{\sqrt{\hat S_A / n_A + \hat S_B / n_B}}
  \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{under convergence},
\tag{5.5}
$$

where $\hat S_A$ and $\hat S_B$ are spectral density estimates at frequency zero
(accounting for autocorrelation within each segment). A p-value below 0.05 indicates
that the chain has not converged.

### 5.6 Known Problems

**Label switching (mixture models).** In mixture or regime-switching models the
posterior is multimodal with symmetric modes corresponding to relabelling of components.
Naive MCMC mixes across modes, making the marginal draws uninterpretable. Solutions
include post-hoc relabelling, identifiability constraints, or complete-data likelihood
penalisation. See Kim & Nelson (1999) for the state-space approach.

**Slow mixing for high-dimensional states.** When $m$ and $T$ are large, the joint
state posterior is very high-dimensional. FFBS handles this efficiently, but if
non-Gaussian elements are introduced that require element-wise MH steps for states,
mixing degrades severely. Blocked samplers and particle MCMC offer alternatives.

**Near-unit-root dynamics.** When $T$ has eigenvalues near 1, the state path is
highly persistent and the posterior of $Q$ may be bimodal (near-zero vs positive). The
Gibbs sampler may get trapped in one mode for many iterations.

---

## 6. Metropolis-Hastings within Gibbs

### 6.1 When Conjugacy Fails

Conjugate posteriors are available only for Gaussian linear models. Several
practically important extensions break conjugacy:

- **Non-Gaussian innovations.** Student-$t$ or skewed innovations for outlier
  robustness.
- **Regime-switching models.** Discrete latent regime variables (Hamilton, 1989;
  Kim & Nelson, 1999).
- **Stochastic volatility.** Log-volatility state with non-Gaussian log-square
  observation density.
- **Nonlinear state transitions.** $\alpha_{t+1} = f(\alpha_t, \eta_t)$ for
  nonlinear $f$.

In these cases, the full conditional for some parameter $\psi_j$ may be known up to
a normalising constant but lack a closed-form expression. **Metropolis-Hastings (MH)**
steps can be embedded within the Gibbs sampler to handle such blocks.

### 6.2 Random Walk Metropolis-Hastings

The Random Walk MH (RWMH) algorithm for a scalar non-conjugate parameter $\psi$:

!!! definition "RWMH Step"
    Given current value $\psi^{(g-1)}$:

    1. **Propose:** $\psi^* = \psi^{(g-1)} + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma_{\mathrm{prop}}^2)$.
    2. **Acceptance probability:**

    $$\alpha(\psi^*, \psi^{(g-1)}) = \min\!\left(1,\;
      \frac{p(\psi^* \mid \text{rest})}{p(\psi^{(g-1)} \mid \text{rest})}\right).$$

    3. **Accept/reject:** Set $\psi^{(g)} = \psi^*$ with probability $\alpha$;
       otherwise $\psi^{(g)} = \psi^{(g-1)}$.

For a $d$-dimensional parameter vector, the optimal proposal covariance (Roberts &
Rosenthal, 2001) scales as

$$
\Sigma_{\mathrm{prop}} = \frac{2.38^2}{d} \Sigma_{\mathrm{post}},
\tag{6.1}
$$

where $\Sigma_{\mathrm{post}}$ is the posterior covariance (estimated adaptively during
burn-in). This yields the target acceptance rate.

**Acceptance rate guidelines:**

| Dimension | Target acceptance rate |
|:----------|:----------------------|
| $d = 1$ | 44% |
| $d = 2$ | 35% |
| $d = 5$ | 28% |
| $d \geq 10$ | 23% |

Acceptance rates outside the range 15%–50% indicate the proposal scale needs
adjustment. Adaptive MCMC methods (Haario et al., 2001) tune $\sigma_{\mathrm{prop}}$
automatically during burn-in.

### 6.3 Slice Sampling

**Slice sampling** (Neal, 2003) is a gradient-free, self-tuning alternative to RWMH
that avoids manual proposal tuning. For a scalar $\psi$ with log-posterior $\ell(\psi)$:

1. **Vertical draw:** Sample auxiliary variable $u \sim \mathrm{Uniform}(0,\, e^{\ell(\psi^{(g-1)})})$.
2. **Horizontal draw:** Find the **slice** $S = \{\psi : e^{\ell(\psi)} \geq u\}$,
   then draw $\psi^{(g)} \sim \mathrm{Uniform}(S)$.

The slice is found via a stepping-out and shrinkage procedure. Slice sampling is
particularly useful for heavy-tailed or multimodal full conditionals that are
problematic for RWMH.

### 6.4 Log-Posterior for Common Non-Conjugate Cases

**Student-$t$ observation noise** ($\nu$ degrees of freedom, scale $\sigma^2$):

The Student-$t$ can be written as a scale mixture of Gaussians: $\varepsilon_t \mid
\lambda_t \sim \mathcal{N}(0, \sigma^2 \lambda_t^{-1})$, $\lambda_t \sim
\mathrm{Gamma}(\nu/2, \nu/2)$. This introduces auxiliary precision variables
$\lambda_t$, all of which have conjugate full conditionals:

$$
\lambda_t \mid \alpha_t, \sigma^2 \sim \mathrm{Gamma}\!\left(
  \frac{\nu + p}{2},\;
  \frac{\nu + e_t^\top e_t / \sigma^2}{2}
\right).
\tag{6.2}
$$

This data-augmentation trick restores conjugacy for $\sigma^2$ and avoids the need
for MH steps — a powerful technique due to West (1987).

---

## 7. Posterior Inference

### 7.1 Point Estimates

Given $G$ post-burn-in draws $\{\theta^{(g)}\}_{g=B+1}^G$, standard point estimates
are:

**Posterior mean** (minimum MSE under squared loss):

$$
\hat\theta_{\mathrm{mean}} = \frac{1}{G - B} \sum_{g=B+1}^G \theta^{(g)}.
\tag{7.1}
$$

**Posterior median** (minimum expected absolute error):

$$
\hat\theta_{\mathrm{median}} = \mathrm{quantile}_{0.5}\!\left(\{\theta^{(g)}\}\right).
\tag{7.2}
$$

**MAP estimate** (posterior mode; not directly from MCMC draws): Requires separate
optimisation of $\log p(\theta \mid y_{1:T})$ and coincides with the penalised MLE
when the prior is log-concave.

For symmetric, unimodal posteriors, all three estimates are approximately equal. For
skewed or multimodal posteriors, the posterior mean may be misleading; the median is
more robust.

### 7.2 Credible Intervals

**Equal-tails credible interval** (ETI): The interval $[q_{\alpha/2}, q_{1-\alpha/2}]$
where $q_p$ is the $p$-th sample quantile. Contains $(1-\alpha)$ of the posterior mass.
Simple to compute but may not be the shortest interval.

**Highest Posterior Density (HPD) region**: The shortest interval $C$ such that
$P(\theta \in C \mid y_{1:T}) = 1 - \alpha$. For a unimodal posterior, the HPD
region is unique and obtained by finding the level $\ell^*$ such that

$$
\int_{\{\theta : p(\theta \mid y_{1:T}) \geq \ell^*\}} p(\theta \mid y_{1:T})\, d\theta = 1 - \alpha.
\tag{7.3}
$$

For symmetric posteriors, ETI and HPD coincide. For skewed posteriors, HPD is generally
preferable as it has the smallest volume for a given coverage probability.

**Computational note.** HPD regions are computed from MCMC draws using kernel density
estimation followed by bisection on the density level $\ell^*$.

### 7.3 Posterior Predictive Distribution

The $h$-step-ahead **posterior predictive distribution** integrates over all
uncertainty in both states and parameters:

$$
p(y_{T+h} \mid y_{1:T})
  = \int\!\int p(y_{T+h} \mid \alpha_{T+h}, \theta)\;
    p(\alpha_{T+h}, \theta \mid y_{1:T})\, d\alpha_{T+h}\, d\theta.
\tag{7.4}
$$

This is generally non-Gaussian even in the linear-Gaussian case because it marginalises
over the parameter uncertainty encoded in $p(\theta \mid y_{1:T})$.

**Monte Carlo evaluation.** For each posterior draw $(\theta^{(g)}, \alpha_T^{(g)})$:

1. Simulate state path forward: $\alpha_{T+k}^{(g)} \sim p(\alpha_{T+k} \mid
   \alpha_{T+k-1}^{(g)}, \theta^{(g)})$ for $k = 1, \ldots, h$.
2. Draw observation: $y_{T+h}^{(g)} \sim \mathcal{N}(Z\alpha_{T+h}^{(g)} + d, H^{(g)})$.

The empirical distribution of $\{y_{T+h}^{(g)}\}_{g=B+1}^G$ approximates the
posterior predictive. Its mean is the Bayesian point forecast; its quantiles give
predictive intervals that are typically wider than those from a single point-estimate
of $\theta$.

### 7.4 Model Comparison

**Marginal likelihood.** The evidence for model $M_k$ is

$$
p(y_{1:T} \mid M_k) = \int p(y_{1:T} \mid \theta, M_k)\, p(\theta \mid M_k)\, d\theta.
\tag{7.5}
$$

This is generally intractable but can be estimated by:

- **Harmonic mean estimator** (Newton & Raftery, 1994): $\hat p(y \mid M) = \left[G^{-1} \sum_g p(y \mid \theta^{(g)})^{-1}\right]^{-1}$ — unstable in practice.
- **Chib's method** (Chib, 1995): Exact evaluation at a single parameter value using output from the Gibbs sampler.
- **Bridge sampling** (Meng & Wong, 1996): More numerically stable than harmonic mean.

**Bayes factor:**

$$
\mathrm{BF}_{12} = \frac{p(y_{1:T} \mid M_1)}{p(y_{1:T} \mid M_2)}.
\tag{7.6}
$$

$\mathrm{BF}_{12} > 10$ is considered strong evidence for $M_1$ (Jeffreys, 1961).

**Deviance Information Criterion:**

$$
\mathrm{DIC} = \bar D + p_D,
\qquad
\bar D = E_\theta[-2\log p(y \mid \theta) \mid y],
\quad
p_D = \bar D - D(\bar\theta),
\tag{7.7}
$$

where $p_D$ is the effective number of parameters. DIC penalises model complexity and
is easily computed from MCMC output (Spiegelhalter et al., 2002). Lower DIC indicates
a better predictive model.

---

## 8. Connection to EM

### 8.1 EM as a Point-Mass Approximation

The EM algorithm (Dempster, Laird & Rubin, 1977) — described in detail in
[MLE theory](mle-theory.md) — maximises the observed-data log-likelihood
$\log p(y_{1:T} \mid \theta)$. In the Bayesian interpretation, EM computes a MAP
estimate by iterating:

- **E-step:** Compute $Q(\theta \mid \theta^{(k)}) = E_{\alpha_{1:T} \mid y, \theta^{(k)}}[\log p(y, \alpha_{1:T} \mid \theta)]$ using the Kalman smoother.
- **M-step:** Maximise $Q(\theta \mid \theta^{(k)}) + \log p(\theta)$ over $\theta$.

This is equivalent to Bayesian inference with a **point-mass prior at the starting
value $\theta^{(0)}$** — or, equivalently, EM approximates the posterior
$p(\theta \mid y_{1:T})$ by a Dirac mass at $\hat\theta_{\mathrm{MAP}}$.

### 8.2 Formal Comparison

| Property | EM | Gibbs Sampling |
|:---------|:---|:---------------|
| Converges to | MAP estimate | Full posterior distribution |
| Parameter uncertainty | Asymptotic Hessian (post hoc) | Propagated automatically |
| Small-sample behavior | Overconfident (ignores estimation uncertainty) | Correctly reflects uncertainty |
| Computational cost | $O(G_{\mathrm{EM}} \cdot Tm^3)$ | $O(G \cdot Tm^3)$ |
| State uncertainty | Integrated out (E-step averages over states) | Fully propagated via FFBS |
| Multi-modality | May converge to local mode | Can explore multiple modes (with long chains) |
| Implementation | Deterministic recursion | Stochastic sampling |
| Predictive intervals | Gaussian approximation | Exact (non-Gaussian across $\theta$) |

### 8.3 When to Use EM vs Gibbs Sampling

**Prefer EM (MLE) when:**

- $T$ is large (e.g., $T > 10{,}000$) and the parameter posterior is approximately
  Gaussian by Bernstein-von Mises.
- Parameter uncertainty is not the primary interest; the goal is state filtering and
  smoothing at a given $\hat\theta$.
- Computational budget does not permit thousands of Kalman filter runs.
- A quick, reproducible point estimate is needed (e.g., for production forecasting
  systems).

**Prefer Gibbs sampling (Bayesian) when:**

- $T$ is small or moderate and parameter uncertainty is non-negligible.
- Predictive intervals need to reflect full uncertainty (including parameter
  uncertainty).
- Prior information is available and should be formally incorporated.
- The model includes discrete latent variables (regimes, mixture components) that
  interact with continuous states.
- Model comparison via Bayes factors or DIC is required.

!!! tip "Hybrid approach"
    A practical strategy is to use EM to find a good starting point
    $\theta^{(0)} = \hat\theta_{\mathrm{MLE}}$ and then run Gibbs sampling to explore
    the posterior around it. This avoids slow burn-in from a diffuse starting point
    and leverages the speed of EM for initialisation.

---

## 9. kalmanbox Implementation

### 9.1 Gibbs Sampler for the Local Level Model

The Local Level model (random walk plus noise) is the simplest non-trivial
state-space model and an ideal first example for Bayesian inference:

$$
y_t = \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2),
\qquad
\alpha_{t+1} = \alpha_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2).
\tag{9.1}
$$

The parameters are $\theta = (\sigma_\varepsilon^2, \sigma_\eta^2)$.

```python
import numpy as np
from kalmanbox.bayesian import GibbsSampler
from kalmanbox.models import LocalLevel
from kalmanbox.diagnostics import posterior_diagnostics

# --- Simulate data ---
rng = np.random.default_rng(42)
T = 300
sigma_eps_true = 1.0
sigma_eta_true = 0.3

alpha_true = np.cumsum(rng.normal(0, sigma_eta_true, T))
y = alpha_true + rng.normal(0, sigma_eps_true, T)

# --- Define model and priors ---
model = LocalLevel()

# Inverse-Gamma priors: IG(a0, b0) -> equivalent IW with nu0=2*a0, S0=2*b0 (scalar)
priors: dict[str, tuple[float, float]] = {
    "sigma_eps_sq": ("inverse_gamma", {"a": 2.0, "b": 1.0}),   # weakly informative
    "sigma_eta_sq": ("inverse_gamma", {"a": 2.0, "b": 0.1}),   # smaller scale prior
}

# --- Run Gibbs sampler ---
sampler = GibbsSampler(
    model=model,
    priors=priors,
    n_iter=3000,          # total draws including burn-in
    n_burnin=1000,        # discard first 1000
    n_chains=4,           # independent chains for convergence diagnostics
    thin=1,               # no thinning
    random_state=0,
)

result = sampler.fit(y)

# result.theta_draws: shape (n_chains, n_post_iter, n_params)
# result.state_draws: shape (n_chains, n_post_iter, T)
print(result.summary())
```

Expected output:

```
Gibbs Sampler Results — LocalLevel
═══════════════════════════════════════════════════════════
Chains: 4   Post-burn-in draws per chain: 2000   Thinning: 1

Parameter           Mean     Std    2.5%    50%   97.5%   R-hat   n_eff
─────────────────────────────────────────────────────────────────────────
sigma_eps_sq       1.023    0.087   0.861  1.019   1.205   1.002   3842
sigma_eta_sq       0.091    0.019   0.057  0.090   0.132   1.004   2917
─────────────────────────────────────────────────────────────────────────
All R-hat < 1.1: PASS
Min n_eff = 2917 (target >= 400): PASS
═══════════════════════════════════════════════════════════
```

### 9.2 Accessing Posterior Draws

```python
import matplotlib.pyplot as plt

# --- Posterior parameter draws (all chains combined) ---
theta_all: np.ndarray = result.theta_draws_combined  # shape (n_post_total, n_params)
sigma_eps_draws: np.ndarray = theta_all[:, 0]
sigma_eta_draws: np.ndarray = theta_all[:, 1]

# --- Posterior state draws ---
# Posterior mean of the smoothed state (column = time point)
state_mean: np.ndarray = result.state_draws_combined.mean(axis=0)  # shape (T,)
state_q05:  np.ndarray = np.quantile(result.state_draws_combined, 0.05, axis=0)
state_q95:  np.ndarray = np.quantile(result.state_draws_combined, 0.95, axis=0)

# --- Plot: smoothed state with credible band ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(y, color="steelblue", alpha=0.4, label="Observed $y_t$")
axes[0].plot(state_mean, color="darkred", lw=1.5, label="Posterior mean $E[\\alpha_t | y]$")
axes[0].fill_between(
    range(T), state_q05, state_q95,
    color="darkred", alpha=0.15, label="90% credible band"
)
axes[0].legend(fontsize=10)
axes[0].set_title("Local Level: Bayesian Smoothed State")

# --- Plot: parameter trace ---
axes[1].plot(sigma_eps_draws, color="steelblue", alpha=0.6, lw=0.5, label="$\\sigma_\\varepsilon^2$")
axes[1].plot(sigma_eta_draws, color="firebrick", alpha=0.6, lw=0.5, label="$\\sigma_\\eta^2$")
axes[1].set_xlabel("MCMC iteration")
axes[1].set_title("Parameter trace plots (post burn-in)")
axes[1].legend()

plt.tight_layout()
plt.savefig("local_level_bayesian.png", dpi=150)
```

### 9.3 Standalone FFBS Usage

FFBS can be used independently — for example, to draw from the smoothing distribution
at a fixed $\theta$ without running a full Gibbs sampler:

```python
from kalmanbox.bayesian import FFBS
from kalmanbox.models import LocalLinearTrend

# --- Build model with fixed parameters ---
model = LocalLinearTrend(
    sigma_eps_sq=1.0,
    sigma_level_sq=0.04,
    sigma_slope_sq=0.001,
)

# --- Fit Kalman filter (forward pass) ---
kf_result = model.filter(y)  # stores a_tt, P_tt, a_t1t, P_t1t for all t

# --- FFBS: draw N independent state paths ---
ffbs = FFBS(model=model, kf_result=kf_result)

n_draws: int = 1000
state_paths: np.ndarray = ffbs.sample(n_draws=n_draws, random_state=7)
# state_paths.shape == (n_draws, T, m)
# For LocalLinearTrend, m=2: [level, slope]

level_draws: np.ndarray = state_paths[:, :, 0]  # shape (1000, T)
slope_draws: np.ndarray = state_paths[:, :, 1]  # shape (1000, T)

print(f"Level posterior mean at t=T: {level_draws[:, -1].mean():.4f}")
print(f"Slope posterior std  at t=T: {slope_draws[:, -1].std():.6f}")
```

### 9.4 Posterior Diagnostics

```python
from kalmanbox.diagnostics import posterior_diagnostics

# --- Run diagnostics on Gibbs output ---
diag = posterior_diagnostics(
    draws=result.theta_draws,           # shape (n_chains, n_post_iter, n_params)
    param_names=["sigma_eps_sq", "sigma_eta_sq"],
    methods=["rhat", "ess", "geweke", "autocorrelation"],
)

# --- Print summary table ---
print(diag.summary_table())
```

```
Convergence Diagnostics
════════════════════════════════════════════════════════
Parameter       R-hat   n_eff   Geweke-Z   p-value   ACF-lag10
────────────────────────────────────────────────────────────────
sigma_eps_sq    1.002   3842     -0.83      0.407      0.012
sigma_eta_sq    1.004   2917      1.21      0.226      0.034
════════════════════════════════════════════════════════
All diagnostics pass at the 5% level.
```

```python
# --- HPD credible intervals ---
from kalmanbox.diagnostics import hpd_interval

for name, draws in zip(["sigma_eps_sq", "sigma_eta_sq"],
                        [sigma_eps_draws, sigma_eta_draws]):
    lo, hi = hpd_interval(draws, credible_mass=0.95)
    print(f"{name}: 95% HPD = [{lo:.4f}, {hi:.4f}]")
```

```
sigma_eps_sq: 95% HPD = [0.8583, 1.2009]
sigma_eta_sq: 95% HPD = [0.0563, 0.1304]
```

```python
# --- Posterior predictive draws ---
h: int = 12  # forecast horizon

predictive_draws: np.ndarray = result.posterior_predictive(
    h=h,
    n_draws=500,
    random_state=99,
)
# predictive_draws.shape == (500, h)

forecast_mean = predictive_draws.mean(axis=0)
forecast_lo   = np.quantile(predictive_draws, 0.025, axis=0)
forecast_hi   = np.quantile(predictive_draws, 0.975, axis=0)

print("12-step posterior predictive forecast:")
for k in range(h):
    print(f"  T+{k+1:02d}: {forecast_mean[k]:7.3f}  "
          f"[{forecast_lo[k]:7.3f}, {forecast_hi[k]:7.3f}]")
```

!!! tip "Performance note"
    For long series ($T > 5{,}000$), use `ffbs_backend="numba"` in `GibbsSampler` to
    enable JIT-compiled Kalman recursions. This provides a 10–50x speedup over pure
    NumPy implementations. Alternatively, `ffbs_backend="jax"` enables GPU
    acceleration for very large batch sizes.

---

## 10. Summary

The table below provides a concise reference for the key formulas developed in this
page.

| Concept | Formula | Section |
|:--------|:--------|:--------|
| Complete data likelihood | $p(y,\alpha\mid\theta) = p(y\mid\alpha,\theta)\,p(\alpha\mid\theta)$ | 1.2 |
| Target joint posterior | $p(\alpha_{1:T},\theta\mid y) \propto p(y\mid\alpha,\theta)\,p(\alpha\mid\theta)\,p(\theta)$ | 1.3 |
| IW posterior for $H$ | $H\mid\alpha,y \sim \mathcal{IW}(\nu_0+T,\; S_0+S_{ee})$ | 2.2 |
| IW posterior for $Q$ | $Q\mid\alpha \sim \mathcal{IW}(\nu_0^Q+T-1,\; S_0^Q+S_{\eta\eta})$ | 2.3 |
| FFBS backward gain | $J_t = P_{t\mid t}T_t^\top P_{t+1\mid t}^{-1}$ | 3.4 |
| FFBS backward mean | $h_t = a_{t\mid t} + J_t(\alpha_{t+1} - a_{t+1\mid t})$ | 3.4 |
| FFBS backward covariance | $H_t^* = P_{t\mid t} - J_t P_{t+1\mid t} J_t^\top$ | 3.4 |
| Gelman-Rubin $\hat R$ | $\hat R = \sqrt{[(G-1)W/G + B/G]\,/\,W}$ | 5.3 |
| Effective sample size | $n_{\mathrm{eff}} = G/(1 + 2\sum_k \rho_k)$ | 5.4 |
| Posterior predictive | $p(y_{T+h}\mid y) = \iint p(y_{T+h}\mid\alpha_{T+h},\theta)\,p(\alpha_{T+h},\theta\mid y)\,d\alpha\,d\theta$ | 7.3 |
| DIC | $\mathrm{DIC} = \bar D + p_D$ | 7.4 |

---

## References

**Foundational FFBS papers:**

- Carter, C.K. and Kohn, R. (1994). On Gibbs Sampling for State Space Models.
  *Biometrika*, 81(3), 541–553.
- Frühwirth-Schnatter, S. (1994). Data Augmentation and Dynamic Linear Models.
  *Journal of Time Series Analysis*, 15(2), 183–202.

**Bayesian state-space models:**

- Kim, C.-J. and Nelson, C.R. (1999). *State-Space Models with Regime Switching*.
  MIT Press.
- West, M. and Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*,
  2nd ed. Springer.

**MCMC theory and diagnostics:**

- Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A. and Rubin, D.B.
  (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
- Gelman, A. and Rubin, D.B. (1992). Inference from Iterative Simulation Using
  Multiple Sequences. *Statistical Science*, 7(4), 457–472.
- Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the
  Calculation of Posterior Moments. In *Bayesian Statistics 4*, Oxford University Press.
- Tierney, L. (1994). Markov Chains for Exploring Posterior Distributions.
  *Annals of Statistics*, 22(4), 1701–1728.

**Simulation smoother:**

- Durbin, J. and Koopman, S.J. (2002). A Simple and Efficient Simulation Smoother for
  State Space Time Series Analysis. *Biometrika*, 89(3), 603–616.

**Data augmentation:**

- Tanner, M.A. and Wong, W.H. (1987). The Calculation of Posterior Distributions by
  Data Augmentation. *Journal of the American Statistical Association*, 82(398), 528–540.

**EM algorithm:**

- Dempster, A.P., Laird, N.M. and Rubin, D.B. (1977). Maximum Likelihood from
  Incomplete Data via the EM Algorithm. *Journal of the Royal Statistical Society B*,
  39(1), 1–38.

**Student-$t$ scale mixture:**

- West, M. (1987). On Scale Mixtures of Normal Distributions.
  *Biometrika*, 74(3), 646–648.

**Model comparison:**

- Spiegelhalter, D.J., Best, N.G., Carlin, B.P. and Van der Linde, A. (2002).
  Bayesian Measures of Model Complexity and Fit. *Journal of the Royal Statistical
  Society B*, 64(4), 583–639.

**Adaptive MCMC:**

- Haario, H., Saksman, E. and Tamminen, J. (2001). An Adaptive Metropolis Algorithm.
  *Bernoulli*, 7(2), 223–242.
- Roberts, G.O. and Rosenthal, J.S. (2001). Optimal Scaling for Various
  Metropolis-Hastings Algorithms. *Statistical Science*, 16(4), 351–367.
