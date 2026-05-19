# Diffuse Initialization Theory

This page provides a rigorous mathematical treatment of **diffuse initialization** for
linear Gaussian state-space models. It covers the initialization problem in depth, exact
and approximate solutions, the modified diffuse log-likelihood, mixed initialization for
structural models, and the augmented Kalman filter approach. Companion pages are
[State-space foundations](state-space-theory.md), [Kalman filter theory](kalman-theory.md),
[Likelihood and MLE](likelihood.md), [Structural models](structural-theory.md), and
the user-facing [diffuse initialization guide](../user-guide/kalman/diffuse.md).

---

## 1. The Initialization Problem

### 1.1 Why Initialization Matters

Every application of the Kalman filter begins with a prior distribution for the initial
state $\alpha_1$. In the linear Gaussian state-space model,

$$
\begin{aligned}
y_t &= Z_t \alpha_t + d_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, H_t) \tag{1}\\
\alpha_{t+1} &= T_t \alpha_t + c_t + R_t \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q_t) \tag{2}
\end{aligned}
$$

the initialization is specified as

$$
\alpha_1 \sim \mathcal{N}(a_1, P_1). \tag{3}
$$

The choice of $a_1$ and $P_1$ directly enters the log-likelihood through the prediction-error
decomposition. The Kalman filter initializes with $a_{1|0} = a_1$ and $P_{1|0} = P_1$, and
the first-step innovation is

$$
v_1 = y_1 - Z_1 a_1 - d_1, \qquad F_1 = Z_1 P_1 Z_1' + H_1.
$$

A misspecified $P_1$ distorts $F_1$ and hence the log-likelihood, biases the filtered
estimates, and corrupts smoothed states. For long series the influence of $P_1$ on the
likelihood decays geometrically for stationary models, but for non-stationary models (random
walk trends, integrated processes) the effect persists throughout the entire sample.

!!! note "Practical Consequence"
    An incorrectly specified $P_1$ is not merely an aesthetic issue — it changes the
    numerical value of the log-likelihood, which changes the MLE of all model parameters.
    Getting initialization right is a prerequisite for correct inference.

### 1.2 Stationary Components: Unconditional Variance

When the state vector $\alpha_t$ is **covariance-stationary** — that is, all eigenvalues of
$T$ lie strictly inside the unit circle — the marginal distribution of $\alpha_t$ converges
to a fixed Gaussian as $t \to \infty$, and the unconditional covariance $P_\infty$ satisfies
the **discrete Lyapunov equation**:

$$
P_\infty = T P_\infty T' + R Q R'. \tag{4}
$$

This is the **Yule-Walker (algebraic Riccati) equation** for the state covariance. Taking
the vectorisation of both sides,

$$
\operatorname{vec}(P_\infty) = (T \otimes T)\, \operatorname{vec}(P_\infty) + \operatorname{vec}(RQR'),
$$

which rearranges to

$$
\boxed{
  P_1 = \operatorname{vec}^{-1}\!\Big( \big(I_{m^2} - T \otimes T\big)^{-1} \operatorname{vec}(RQR') \Big).
} \tag{5}
$$

**Derivation.** Starting from (4), apply the vectorisation identity
$\operatorname{vec}(ABC) = (C' \otimes A)\operatorname{vec}(B)$:

$$
\operatorname{vec}(T P_\infty T') = (T \otimes T)\operatorname{vec}(P_\infty).
$$

Hence $(I_{m^2} - T \otimes T)\operatorname{vec}(P_\infty) = \operatorname{vec}(RQR')$.
Stationarity guarantees that every eigenvalue of $T \otimes T$ is $\lambda_i \lambda_j$ where
$\lambda_i, \lambda_j$ are eigenvalues of $T$. Since $|\lambda_i| < 1$ and $|\lambda_j| < 1$,
all eigenvalues of $T \otimes T$ are strictly less than one in modulus, so $I_{m^2} - T \otimes T$
is non-singular and equation (5) has a unique positive-semidefinite solution. $\blacksquare$

For stationary models, equation (5) gives the **exact optimal initialization**, and one sets
$a_1 = (I - T)^{-1} c$ (the unconditional mean when a constant offset $c$ is present) and
$P_1$ from (5). This is the `initialization='stationary'` option in kalmanbox.

### 1.3 Non-Stationary Components

Many economically important state-space models include **non-stationary components**: random
walks, integrated-of-order-$d$ processes, ARIMA$(p, d, q)$ models in state-space form. For
these components the unconditional variance does not exist — the solution to (4) diverges.

**Examples of non-stationary components:**

| Component | Model | Eigenvalue of $T$ |
|:----------|:------|:------------------|
| Local level (random walk) | $\mu_{t+1} = \mu_t + \eta_t$ | $\lambda = 1$ |
| Local linear trend (slope) | $\nu_{t+1} = \nu_t + \zeta_t$ | $\lambda = 1$ |
| ARIMA$(p, d, q)$, $d \geq 1$ | Companion form | $d$ unit roots |
| Integrated cycle | $\psi_{t+1} = \rho_c \psi_t + \kappa_t$ with $\rho_c = 1$ | $\lambda = e^{\pm i\lambda_c}$, $|\lambda| = 1$ |

For such components the correct prior has **infinite variance**: the initial state is
completely uninformative, which is the diffuse prior $\alpha_{1,j} \sim \mathcal{N}(0, \kappa)$
as $\kappa \to \infty$.

### 1.4 The Finite-Arithmetic Problem

The theoretically correct diffuse prior $P_1 = \kappa I$ with $\kappa \to \infty$ cannot be
implemented directly in floating-point arithmetic. Two approaches resolve this:

1. **Approximate diffuse initialization:** choose a large finite $\kappa$ (e.g.,$\kappa = 10^6$).
2. **Exact diffuse initialization:** propagate the $\kappa \to \infty$ limit analytically using a
   modified filter (De Jong 1991; Koopman 1997).

!!! warning "Numerical Danger"
    Setting $\kappa$ too large causes numerical overflow or catastrophic cancellation in the
    Kalman gain and covariance updates. Setting $\kappa$ too small under-weights the prior
    uncertainty. Neither choice is entirely safe. **Exact diffuse initialization eliminates
    this dilemma.**

---

## 2. Approximate Diffuse Initialization

### 2.1 The $\kappa$-Prior Method

The simplest approach — due to Harvey & Phillips (1979) — is to set

$$
P_1 = \kappa I_m, \qquad \kappa \gg 1, \tag{6}
$$

and run the standard Kalman filter. As $\kappa$ grows, the first several filtered estimates
converge to their diffuse counterparts. This is the `initialization='approximate_diffuse'`
option in kalmanbox, with $\kappa = 10^6$ as the default.

**How the approximation works.** For large $\kappa$, the Kalman gain at $t = 1$ becomes

$$
K_1 = T_1 P_1 Z_1' F_1^{-1} \approx \kappa T_1 Z_1' (Z_1 \kappa Z_1')^{-1} = T_1 (Z_1' Z_1)^{-1} Z_1' \cdot \frac{1}{\kappa^0},
$$

so the filter quickly "learns" the state and the influence of $\kappa$ decays over subsequent
periods. Formally, the finite-$\kappa$ filter covariance $P_t(\kappa)$ satisfies

$$
P_t(\kappa) = P_t^* + O(\kappa^{-(t-1)}),
$$

where $P_t^*$ is the covariance of the exact diffuse filter after the diffuse period has
ended. The approximation error thus decays geometrically, but it is never exactly zero.

### 2.2 Problems with Approximate Initialization

The approximate approach suffers from several practical deficiencies:

**Numerical instability.** For large $\kappa$, the matrix $F_1 = Z_1 (\kappa I) Z_1' + H_1$
is ill-conditioned if $H_1$ is small. The ratio $F_1^{-1}$ magnifies rounding errors in the
update step, potentially causing $P_2$ to become indefinite.

**Sensitivity to $\kappa$.** The log-likelihood value depends on $\kappa$ through the first
few innovation variances $F_1, \ldots, F_d$. Different choices of $\kappa$ yield different
likelihood values and therefore different MLE parameter estimates. There is no principled way
to choose $\kappa$.

**Incorrect effective degrees of freedom.** The AIC and BIC corrections assume a known number
of estimated parameters. Approximate initialization conflates the contribution of diffuse
initial conditions with genuine model parameters, inflating the effective parameter count.

### 2.3 When Approximate Initialization is Adequate

Despite these problems, approximate diffuse initialization is acceptable when:

- The series is long ($T \gg d$), so the diffuse periods contribute negligibly to the total
  log-likelihood.
- The goal is computing filtered/smoothed states, not maximum-likelihood estimation.
- All non-stationary components become effectively stationary within a few periods (e.g.,
  models with near-unit-root behaviour but $|\lambda| < 1$).
- Quick prototyping where exact likelihood values are not required.

!!! tip "Recommendation"
    Use `initialization='approximate_diffuse'` for rapid exploration. Switch to
    `initialization='diffuse'` (exact) for final parameter estimation and likelihood-based
    model comparison (AIC/BIC).

---

## 3. Exact Diffuse Initialization: De Jong (1991)

### 3.1 The Partitioned Covariance Matrix

De Jong (1991) derives a modified Kalman filter that handles the limiting case
$\kappa \to \infty$ exactly by decomposing the state covariance into two components:

$$
\boxed{P_1 = P_1^* + \kappa P_1^\infty} \tag{7}
$$

where:

- $P_1^*$ is the **stationary covariance** — finite, positive-semidefinite, capturing the
  known uncertainty in the initial state.
- $P_1^\infty$ is the **diffuse covariance matrix** — a selection matrix (rank $m_d$, where
  $m_d$ is the number of diffuse components) encoding which state components are non-stationary.
- $\kappa \to \infty$ is taken analytically.

The corresponding decomposition of the initial state is

$$
\alpha_1 = a_1^* + \kappa^{1/2} \delta, \qquad \delta \sim \mathcal{N}(0, P_1^\infty), \tag{8}
$$

where $a_1^*$ is the best available prior mean and $\delta$ represents the unknown (diffuse)
component of the initial state.

**Structure of $P_1^\infty$.** For a model with $m_d$ diffuse components at positions
$\{i_1, \ldots, i_{m_d}\} \subset \{1, \ldots, m\}$, the matrix $P_1^\infty$ is the
$m \times m$ matrix with ones in positions $(i_j, i_j)$ and zeros elsewhere:

$$
P_1^\infty = \sum_{j=1}^{m_d} e_{i_j} e_{i_j}', \tag{9}
$$

where $e_i$ is the $i$-th standard basis vector in $\mathbb{R}^m$.

!!! definition "Diffuse Period"
    The **diffuse period** $\mathcal{D} = \{1, \ldots, d\}$ is the set of time indices during
    which the diffuse covariance $P_t^\infty \neq 0$. After period $d$, all diffuse uncertainty
    has been absorbed by the observations, and the filter reduces to the standard Kalman filter.

### 3.2 Modified Kalman Filter Recursion

The De Jong (1991) filter propagates two covariance matrices simultaneously. At each time $t$
in the diffuse period, partition the innovation covariance as

$$
F_t = F_t^* + \kappa F_t^\infty, \tag{10}
$$

where

$$
F_t^\infty = Z_t P_t^\infty Z_t', \qquad F_t^* = Z_t P_t^* Z_t' + H_t. \tag{11}
$$

**Case 1: $F_t^\infty > 0$ (diffuse period, $t \leq d$).**

The diffuse Kalman gain is

$$
K_t^\infty = T_t P_t^\infty Z_t' (F_t^\infty)^{-1}. \tag{12}
$$

The standard gain for the finite part involves a correction:

$$
K_t^* = T_t \Big( P_t^* Z_t' - K_t^\infty F_t^* \Big) (F_t^\infty)^{-1}. \tag{13}
$$

The state update uses the standard Kalman formula with gain $K_t^\infty$:

$$
a_{t+1} = T_t a_t + c_t + K_t^\infty v_t, \tag{14}
$$

where $v_t = y_t - Z_t a_t - d_t$ is the innovation. The covariance updates are:

$$
\begin{aligned}
P_{t+1}^\infty &= T_t P_t^\infty T_t' - K_t^\infty F_t^\infty (K_t^\infty)' \tag{15} \\
P_{t+1}^* &= T_t P_t^* T_t' + R_t Q_t R_t' - K_t^\infty F_t^* (K_t^\infty)' - K_t^* F_t^\infty (K_t^\infty)' - K_t^\infty F_t^\infty (K_t^*)'. \tag{16}
\end{aligned}
$$

Equation (15) shows that $P_t^\infty$ is monotonically non-increasing (in the positive-semidefinite
order) — once a diffuse component is "observed", its infinite uncertainty collapses.

**Case 2: $F_t^\infty = 0$ (stationary period, $t > d$).**

When $P_t^\infty = 0$, the filter reduces exactly to the **standard Kalman filter** applied
to $P_t^*$:

$$
K_t = T_t P_t^* Z_t' (F_t^*)^{-1}, \tag{17}
$$

$$
a_{t+1} = T_t a_t + c_t + K_t v_t, \tag{18}
$$

$$
P_{t+1}^* = T_t P_t^* T_t' + R_t Q_t R_t' - K_t F_t^* K_t'. \tag{19}
$$

### 3.3 Termination of the Diffuse Period

The diffuse period terminates at time $d$ when

$$
\operatorname{rank}(P_{d+1}^\infty) = 0, \quad \text{i.e.,} \quad P_{d+1}^\infty = 0. \tag{20}
$$

In practice, termination is detected numerically using a tolerance:
$\|P_{t+1}^\infty\|_F < \epsilon$ for a small $\epsilon > 0$ (kalmanbox uses $\epsilon = 10^{-10}$).

**Minimum diffuse period.** The diffuse period satisfies $d \geq m_d$, the number of
diffuse components. In the univariate case ($p = 1$), each observation resolves at most one
diffuse component, so $d = m_d$ exactly when all diffuse components are observable. For
multivariate systems ($p > 1$), the diffuse period may be shorter: $d \leq \lceil m_d / p \rceil$
periods suffice if the system is fully observable.

**Observability condition.** Let $\mathcal{O}_k$ be the $k$-step observability matrix. The
diffuse period is finite if and only if the diffuse subspace spanned by $P_1^\infty$ is
contained in the observable subspace:

$$
\operatorname{col}(P_1^\infty) \subseteq \operatorname{col}(\mathcal{O}_{m_d}'). \tag{21}
$$

If this condition fails, some diffuse components are unobservable and the likelihood is not
defined. This typically indicates a model identification failure.

### 3.4 Derivation of the Diffuse Filter Equations

We sketch the derivation of equations (12)-(16) from De Jong (1991).

**Setup.** Substitute (7) into the standard Kalman prediction:

$$
P_{1|0} = P_1^* + \kappa P_1^\infty.
$$

At time $t = 1$, the innovation variance is $F_1 = Z_1 P_1^* Z_1' + H_1 + \kappa Z_1 P_1^\infty Z_1' = F_1^* + \kappa F_1^\infty$.

**Gain expansion in $\kappa$.** The Kalman gain is

$$
K_1 = T_1 P_{1|0} Z_1' F_1^{-1} = T_1 (P_1^* + \kappa P_1^\infty) Z_1' (F_1^* + \kappa F_1^\infty)^{-1}.
$$

When $F_1^\infty > 0$, expand $(F_1^* + \kappa F_1^\infty)^{-1}$ in powers of $\kappa^{-1}$:

$$
F_1^{-1} = \frac{1}{\kappa} (F_1^\infty)^{-1} - \frac{1}{\kappa^2} (F_1^\infty)^{-1} F_1^* (F_1^\infty)^{-1} + O(\kappa^{-3}).
$$

Substituting and collecting terms:

$$
K_1 = T_1 P_1^\infty Z_1' (F_1^\infty)^{-1} + \kappa^{-1} T_1 \Big( P_1^* Z_1' (F_1^\infty)^{-1} - P_1^\infty Z_1' (F_1^\infty)^{-1} F_1^* (F_1^\infty)^{-1} \Big) + O(\kappa^{-2}).
$$

The $O(1)$ term is $K_1^\infty$; the $O(\kappa^{-1})$ term, when multiplied by $\kappa$ in the
covariance update, gives $K_1^*$.

**Covariance update.** The standard covariance update is
$P_{2} = (T_1 - K_1 Z_1) P_{1|0} T_1' + R_1 Q_1 R_1'$. Expanding in $\kappa$ and matching
terms of order $\kappa^1$ gives equation (15), and terms of order $\kappa^0$ give equation (16).

The derivation generalizes to all $t$ in the diffuse period by induction. $\blacksquare$

---

## 4. Exact Diffuse Initialization: Koopman (1997)

### 4.1 Augmented State Vector Approach

Koopman (1997) offers an alternative derivation using **state augmentation**. Let the original
state be $\alpha_t \in \mathbb{R}^m$ and introduce an auxiliary vector $\delta \in \mathbb{R}^{m_d}$
representing the unknown diffuse initial conditions. Define the augmented state vector

$$
\tilde{\alpha}_t = \begin{pmatrix} \alpha_t \\ \delta \end{pmatrix} \in \mathbb{R}^{m + m_d}, \tag{22}
$$

where $\delta$ is constant over time (it does not evolve — it represents fixed but unknown
initial conditions).

### 4.2 Augmented System Matrices

The augmented system matrices are constructed as follows. Define $A_1 = P_1^\infty$ (the
$m \times m_d$ matrix whose columns span the diffuse subspace). The augmented system is

$$
\begin{aligned}
y_t &= \tilde{Z}_t \tilde{\alpha}_t + d_t + \varepsilon_t, \\
\tilde{\alpha}_{t+1} &= \tilde{T}_t \tilde{\alpha}_t + \tilde{c}_t + \tilde{R}_t \eta_t,
\end{aligned}
$$

with augmented matrices

$$
\tilde{Z}_t = \begin{pmatrix} Z_t & Z_t A_t \end{pmatrix}, \quad
\tilde{T}_t = \begin{pmatrix} T_t & T_t A_t - A_{t+1} \\ 0 & I_{m_d} \end{pmatrix}, \tag{23}
$$

$$
\tilde{R}_t = \begin{pmatrix} R_t \\ 0 \end{pmatrix}, \quad
\tilde{c}_t = \begin{pmatrix} c_t \\ 0 \end{pmatrix}, \tag{24}
$$

where $A_t$ evolves as $A_{t+1} = T_t A_t$ with $A_1 = [I_{m_d}; 0]$ (the first $m_d$
columns corresponding to diffuse components).

The augmented initial conditions are

$$
\tilde{a}_{1|0} = \begin{pmatrix} a_1^* \\ 0 \end{pmatrix}, \qquad
\tilde{P}_{1|0} = \begin{pmatrix} P_1^* & 0 \\ 0 & \kappa I_{m_d} \end{pmatrix}, \tag{25}
$$

and the limit $\kappa \to \infty$ is taken analytically by running the Koopman (1997)
filter — which tracks the leading $\kappa$ terms exactly.

### 4.3 Equivalence to De Jong (1991)

The Koopman (1997) augmented filter produces **identical numerical results** to the De Jong
(1991) partitioned filter in the limit $\kappa \to \infty$. This can be verified by
substituting (25) into the augmented Kalman recursion and extracting the $\kappa^0$ and
$\kappa^1$ components — they recover equations (12)-(16) exactly.

!!! abstract "Key Result"
    The De Jong (1991) and Koopman (1997) approaches are mathematically equivalent. De Jong's
    formulation is more parsimonious (it avoids enlarging the state vector), while Koopman's
    formulation is conceptually cleaner (diffuse components appear explicitly as state variables)
    and more convenient for extending to regression effects (see Section 9).

### 4.4 Computational Properties

The Koopman (1997) augmented filter:

- Is a **single-pass algorithm** — no two-phase structure or special handling of the
  diffuse-to-stationary transition.
- Handles the diffuse and stationary periods uniformly within a single recursion.
- Introduces $m_d$ extra state components, increasing the cost from $O(m^3)$ to
  $O((m + m_d)^3)$ per time step during the diffuse period.
- Simplifies the implementation of the diffuse smoother, which requires only the standard
  RTS smoother applied to the augmented system.

---

## 5. Diffuse Log-Likelihood

### 5.1 Standard Log-Likelihood

The standard prediction-error log-likelihood for a Gaussian state-space model is

$$
\log L(\theta) = -\frac{1}{2} \sum_{t=1}^T \Big( p \log 2\pi + \log|F_t| + v_t' F_t^{-1} v_t \Big), \tag{26}
$$

where $v_t = y_t - Z_t a_t - d_t$ is the innovation and $F_t = Z_t P_t Z_t' + H_t$ is the
innovation covariance. This formula is valid when the initialization is proper ($P_1$ finite
and well-specified).

### 5.2 Divergence During the Diffuse Period

When the $\kappa$-prior is used, the innovation variance at $t \leq d$ is

$$
F_t(\kappa) = F_t^* + \kappa F_t^\infty. \tag{27}
$$

As $\kappa \to \infty$, two terms in (26) diverge for $t \in \mathcal{D}$:

1. $\log|F_t(\kappa)| = \log|F_t^* + \kappa F_t^\infty| \approx m_d^{(t)} \log \kappa + \log|F_t^\infty| + O(\kappa^{-1})$,
   where $m_d^{(t)} = \operatorname{rank}(F_t^\infty)$.

2. The quadratic form $v_t' F_t(\kappa)^{-1} v_t \to v_t^{*'} (F_t^\infty)^{-1} v_t^*$ where
   $v_t^* = v_t - F_t^* (F_t^\infty)^{-1} v_t$ is the adjusted innovation.

The $\kappa$-dependent terms cancel in the limit, leaving a finite diffuse log-likelihood.

### 5.3 Modified Diffuse Log-Likelihood

Following Durbin & Koopman (2012, eq. 7.6), the **exact diffuse log-likelihood** is

$$
\boxed{
\ell_d = -\frac{1}{2} \sum_{t \in \mathcal{D}} \Big( \log|F_t^\infty| + v_t^{*'} (F_t^\infty)^{-1} v_t^* \Big)
         - \frac{1}{2} \sum_{t \notin \mathcal{D}} \Big( p \log 2\pi + \log|F_t^*| + v_t' (F_t^*)^{-1} v_t \Big),
} \tag{28}
$$

where $\mathcal{D} = \{1, \ldots, d\}$ is the diffuse period and $v_t^*$ is the modified
innovation adjusted for the diffuse component:

$$
v_t^* = v_t - F_t^* (F_t^\infty)^{-1} v_t. \tag{29}
$$

The $p \log 2\pi$ term is absent from the diffuse-period sum because the effective degrees of
freedom are reduced — the diffuse observations constrain the initial conditions rather than
the model parameters.

**Correction term.** The diffuse log-likelihood differs from the conditional log-likelihood
(conditioning on the first $d$ observations) by the correction

$$
\Delta \ell = -\frac{1}{2} \sum_{t=1}^d \log|F_t^\infty|. \tag{30}
$$

This correction accounts for the information content of the diffuse-period observations about
the initial conditions, and ensures that $\ell_d$ is invariant to reparameterizations of the
diffuse components.

### 5.4 AIC and BIC Correction

Because the diffuse initial conditions are not estimated by MLE (they are integrated out), the
effective number of parameters for information criterion calculations is reduced. If the model
has $k$ structural parameters and $m_d$ diffuse components, the AIC uses

$$
\text{AIC} = -2\ell_d + 2k, \tag{31}
$$

and the BIC uses

$$
\text{BIC} = -2\ell_d + k \log(T - d), \tag{32}
$$

where the sample size is adjusted to $T - d$ (the number of non-diffuse observations).

!!! warning "Common Error"
    Some implementations add $m_d$ to the parameter count when computing AIC/BIC under diffuse
    initialization. This is incorrect: the diffuse initial conditions are nuisance parameters
    that are analytically marginalized, not estimated. The correct penalty is $2k$ (AIC) or
    $k \log(T - d)$ (BIC), with $k$ counting only the structural variance parameters.

---

## 6. Mixed Initialization

### 6.1 Stationary and Non-Stationary Components

In practice, state-space models for structural time series contain **a mixture of stationary
and non-stationary components**. For example, the Basic Structural Model (BSM) has:

- **Trend** $(\mu_t, \nu_t)$: two diffuse components (level and slope are random walks).
- **Seasonal** $(\gamma_t, \ldots)$: $s - 1$ seasonal states; typically treated as stationary
  with unconditional variance from equation (5), or as diffuse if the seasonal variances are
  unknown.
- **Cycle** $(\psi_t, \psi_t^*)$: stationary if $|\rho_c| < 1$, initialized from (5).
- **Irregular** $\varepsilon_t$: no state, no initialization needed.

### 6.2 Correct Partitioning

Let $\alpha_t = (\alpha_t^{(S)'}, \alpha_t^{(D)'})' \in \mathbb{R}^{m_S + m_D}$ where
superscripts $S$ and $D$ denote stationary and diffuse subvectors. The correct initialization
is the block-diagonal structure

$$
P_1 = \begin{pmatrix} P_1^{(S)} & 0 \\ 0 & \kappa I_{m_D} \end{pmatrix}, \tag{33}
$$

where $P_1^{(S)}$ is the unconditional variance from (5) applied to the stationary sub-system,
and $\kappa I_{m_D} \to \infty$ for the non-stationary sub-system. In the exact diffuse
formulation, this corresponds to

$$
P_1^* = \begin{pmatrix} P_1^{(S)} & 0 \\ 0 & 0 \end{pmatrix}, \qquad
P_1^\infty = \begin{pmatrix} 0 & 0 \\ 0 & I_{m_D} \end{pmatrix}. \tag{34}
$$

### 6.3 Initialization by Model Component

The following table summarizes the standard initialization choices for common structural
components:

| Component | State dimension | Diffuse components | Recommended initialization |
|:----------|:----------------|:-------------------|:--------------------------|
| Local level | 1 | 1 (level $\mu_t$) | Exact diffuse |
| Local linear trend | 2 | 2 (level $\mu_t$, slope $\nu_t$) | Exact diffuse |
| BSM (level + slope) | 2 | 2 | Exact diffuse |
| BSM seasonal ($s-1$ harmonics) | $s-1$ | 0 (if $\sigma_\omega^2 > 0$: stationary) | Stationary from (5) |
| Trigonometric seasonal | $2\lfloor s/2 \rfloor$ | 0 | Stationary from (5) |
| Stationary AR$(p)$ | $p$ | 0 | Stationary from (5) |
| ARIMA$(p, d, q)$ | $p + d$ | $d$ | $d$ exact diffuse + $p$ stationary |
| Stationary cycle | 2 | 0 | Stationary from (5) |
| TVP regression coefficients | $k$ | $k$ (if vague priors) | Exact diffuse |

### 6.4 kalmanbox Automatic Partitioning

kalmanbox implements automatic mixed initialization via the `initialization='auto'` option.
The algorithm:

1. Computes the eigenvalues of $T$.
2. Identifies state components associated with unit-modulus eigenvalues as diffuse.
3. Computes $P_1^{(S)}$ from (5) for the stationary sub-block.
4. Sets $P_1^\infty$ as in (34) for the non-stationary sub-block.

!!! note "Auto-initialization Limitations"
    Automatic partitioning based on eigenvalues of $T$ may fail for:
    - Near-unit-root models (eigenvalues close to but not exactly on the unit circle).
    - Models where the user specifies a non-standard initialization strategy.
    - Seasonal components with structural breaks.
    In these cases, use `initialization='diffuse'` or `initialization='stationary'` explicitly.

---

## 7. Augmented Kalman Filter (AKF)

### 7.1 State Augmentation for Diffuse Initialization

The **Augmented Kalman Filter (AKF)**, developed formally by Francke, Koopman & de Vos (2010),
generalizes the Koopman (1997) approach to handle diffuse initialization, regression effects,
and exogenous variables in a unified framework.

The key idea is to absorb all unknown initial conditions and regression coefficients into an
augmented state vector. Define:

$$
\tilde{\alpha}_t = \begin{pmatrix} \alpha_t \\ \mu \end{pmatrix}, \tag{35}
$$

where $\mu \in \mathbb{R}^q$ collects all unknown fixed effects (initial conditions plus
regression coefficients). The augmented state (35) evolves as

$$
\tilde{\alpha}_{t+1} = \begin{pmatrix} T_t & A_{t+1} - T_t A_t \\ 0 & I_q \end{pmatrix} \tilde{\alpha}_t + \begin{pmatrix} R_t \eta_t \\ 0 \end{pmatrix}, \tag{36}
$$

where $A_t \in \mathbb{R}^{m \times q}$ satisfies the recursion $A_{t+1} = T_t A_t + G_t$,
with $G_t = 0$ for pure diffuse initialization (no time-varying regression terms).

### 7.2 AKF and Regression Effects

The AKF is particularly powerful when regression variables $x_t$ appear in the observation
equation with unknown coefficients $\beta$:

$$
y_t = Z_t \alpha_t + x_t' \beta + \varepsilon_t. \tag{37}
$$

By treating $\beta$ as part of the augmented state $\mu$, the AKF eliminates $\beta$ from the
likelihood in a single filter pass, avoiding the need for concentrated likelihood techniques
(see Section 9). This corresponds to placing a diffuse prior on $\beta$, which is equivalent
to the frequentist treatment of $\beta$ as an unknown constant.

### 7.3 Connection to Francke, Koopman & de Vos (2010)

Francke, Koopman & de Vos (2010) establish several important results for the AKF:

1. The AKF log-likelihood equals the diffuse log-likelihood (28) plus a correction for the
   regression effects.
2. The smoother for the AKF coincides with the GLS estimator of $\mu$ given the Kalman
   residuals.
3. The AKF handles time-varying regression coefficients naturally by allowing $G_t \neq 0$.
4. The AKF is computationally equivalent to the De Jong (1991) filter with the diffuse
   subspace extended to include the regression regressors.

!!! abstract "Practical Relevance"
    In kalmanbox, the AKF is used internally when `exog` variables are present and
    `initialization='diffuse'` is set. Users do not interact with the augmented state
    directly — the augmentation is handled transparently.

### 7.4 Single-Pass Algorithm

The AKF performs diffuse initialization and handles regression effects in a **single forward
pass**, avoiding the two-step procedures sometimes used in classical approaches (compute
concentrated likelihood, then estimate $\beta$ by GLS). The filter equations are the standard
Kalman equations applied to the augmented system (36), with the augmented initial conditions

$$
\tilde{a}_{1|0} = \begin{pmatrix} a_1^* \\ 0 \end{pmatrix}, \qquad
\tilde{P}_{1|0} = \begin{pmatrix} P_1^* & 0 \\ 0 & \kappa I_q \end{pmatrix} \xrightarrow{\kappa \to \infty} \text{(exact diffuse)}. \tag{38}
$$

---

## 8. Practical Guidance

### 8.1 Choosing an Initialization Strategy

The following table summarizes the recommended initialization strategy by model type:

| Model | Diffuse components $m_d$ | Recommended strategy | API option |
|:------|:------------------------:|:---------------------|:-----------|
| Stationary AR$(p)$ | 0 | Unconditional variance (5) | `'stationary'` |
| Local Level | 1 | Exact diffuse | `'diffuse'` |
| Local Linear Trend | 2 | Exact diffuse | `'diffuse'` |
| BSM (level + slope) | 2 | Exact diffuse for trend | `'diffuse'` |
| BSM (full, with seasonal) | 2 | Auto-partitioned | `'auto'` |
| ARIMA$(p, d, q)$, $d \geq 1$ | $d$ | $d$ exact diffuse | `'diffuse'` |
| UCM with cycle ($|\rho_c| < 1$) | 2 | Exact diffuse (trend) + stationary (cycle) | `'auto'` |
| TVP regression | $k$ | Exact diffuse | `'diffuse'` |
| DFM (stationary factors) | 0 | Stationary | `'stationary'` |
| General unknown | — | Auto-partition | `'auto'` |

### 8.2 Number of Diffuse Periods

The number of diffuse periods $d$ depends on:

- The number of diffuse components $m_d$ (lower bound: $d \geq m_d / p$).
- The observability structure of the model.
- The presence of missing data during the diffuse period (each missing observation fails to
  reduce $\operatorname{rank}(P_t^\infty)$ and extends the diffuse period).

**Warning: Large $d$ relative to $T$.** If $d > T / 4$, the diffuse likelihood (28) is
unreliable because too few stationary-period observations contribute to the log-likelihood
sum. In this regime:

- Consider a shorter model (fewer non-stationary components).
- Use Bayesian priors on the initial state instead of diffuse initialization.
- If possible, collect more data.

!!! warning "Missing Data During Diffuse Period"
    If observations are missing during the diffuse period $\mathcal{D}$, those time points
    do not reduce $\operatorname{rank}(P_t^\infty)$. The diffuse period may therefore last
    longer than $m_d$ periods. kalmanbox handles this correctly by skipping the rank-reduction
    step at missing observations and extending the diffuse period accordingly.

### 8.3 Numerical Implementation Notes

kalmanbox uses the following safeguards:

- **Rank detection for $F_t^\infty$:** The effective rank of $F_t^\infty$ is determined using
  a tolerance of $10^{-10}$ times the largest singular value, preventing false detection of
  diffuse-period termination due to floating-point noise.
- **Cholesky-based inversion:** When $F_t^\infty$ is positive definite, it is inverted via
  Cholesky factorization rather than LU decomposition, improving numerical stability.
- **Smooth transition:** At the boundary $t = d + 1$, kalmanbox checks
  $\|P_{d+1}^\infty\|_F < 10^{-10}$ before switching to the standard filter, avoiding
  contamination of the stationary-period covariance by residual diffuse uncertainty.

### 8.4 kalmanbox API Summary

| API parameter | Effect |
|:--------------|:-------|
| `initialization='diffuse'` | Exact De Jong (1991) / Koopman (1997) filter |
| `initialization='approximate_diffuse'` | $\kappa$-prior with $\kappa = 10^6$ (default) |
| `initialization='stationary'` | Unconditional variance from Lyapunov equation (5) |
| `initialization='auto'` | Automatic partitioning: diffuse for unit-root components, stationary for rest |
| `initialization=<array>` | User-specified $P_1$ matrix |

Results attributes after fitting:

| Attribute | Description |
|:----------|:------------|
| `results.diffuse_periods` | Integer $d$, number of diffuse periods |
| `results.diffuse_loglikelihood` | Diffuse log-likelihood $\ell_d$ from (28) |
| `results.filter_results.predicted_diffuse_state_cov` | Array of $P_t^\infty$ matrices |
| `results.filter_results.predicted_state_cov` | Array of $P_t^*$ matrices |

---

## 9. Connection to Regression Effects

### 9.1 Unknown Constants via Diffuse Initialization

In many economic models, the observation equation contains **deterministic regressors** with
unknown coefficients. The standard approach (Durbin & Koopman 2012, Ch. 5) is to treat these
coefficients as part of the state vector with a diffuse (improper) prior.

Consider the model

$$
y_t = Z_t \alpha_t + x_t' \beta + \varepsilon_t, \tag{39}
$$

where $x_t$ is a vector of known regressors and $\beta$ is unknown. Two equivalent approaches:

**Approach 1: Concentrated likelihood.** Eliminate $\beta$ analytically by minimizing the
log-likelihood over $\beta$ for fixed structural parameters $\theta$. This yields the GLS
estimator $\hat{\beta}(\theta)$ and the concentrated log-likelihood $\ell_c(\theta)$.

**Approach 2: Diffuse state augmentation.** Augment the state with $\beta$ and apply
diffuse initialization: $\beta \sim \mathcal{N}(0, \kappa I)$ as $\kappa \to \infty$.

**Equivalence.** Both approaches yield the same log-likelihood value asymptotically, and
the same MLE $\hat{\theta}$. The diffuse augmentation approach additionally provides smoothed
estimates of $\beta$ from the Kalman smoother, which may be of independent interest.

### 9.2 Practical Equivalence

For the practically common case of a constant intercept $\mu$ in the local level model:

$$
y_t = \alpha_t + \mu + \varepsilon_t,
$$

treating $\mu$ as a diffuse state component (with $\mu_{t+1} = \mu_t$, $\eta_t^{(\mu)} = 0$)
is equivalent to demeaning the series before fitting. The diffuse log-likelihood automatically
accounts for the estimation of $\mu$.

!!! tip "When to Use Diffuse vs. Concentrated"
    Use concentrated likelihood when regressors are strictly exogenous and the number of
    regression parameters is large (many $\beta$s). Use diffuse augmentation when the
    regression coefficients may be time-varying (TVP) or when you want Bayesian-style
    smoothed estimates of $\beta$.

### 9.3 Reference: Durbin & Koopman (2012), Chapters 5 and 7

Chapter 5 of Durbin & Koopman (2012) provides a comprehensive treatment of regression effects
in state-space models, including the exact diffuse likelihood for regression models, the
connection to GLS, and the handling of initial effects in the smoother. Chapter 7 extends
this to non-Gaussian models.

---

## 10. kalmanbox Implementation Examples

### 10.1 Local Level Model with Diffuse Initialization

```python
import numpy as np
from kalmanbox.models import LocalLevel

# Simulate a random walk plus noise series
rng = np.random.default_rng(42)
T: int = 200
sigma_eta: float = 1.0
sigma_eps: float = 0.5

level: np.ndarray = np.cumsum(rng.normal(0, sigma_eta, T))
y: np.ndarray = level + rng.normal(0, sigma_eps, T)

# Fit with exact diffuse initialization (default for LocalLevel)
model = LocalLevel(y)
results = model.fit(initialization="diffuse")

print(f"Estimated sigma_eta: {results.params['sigma_eta']:.4f}")
print(f"Estimated sigma_eps: {results.params['sigma_eps']:.4f}")
print(f"Diffuse periods: {results.diffuse_periods}")
print(f"Diffuse log-likelihood: {results.diffuse_loglikelihood:.4f}")
```

### 10.2 Inspecting the Diffuse Period and $F_t^\infty$

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.models import LocalLevel
from kalmanbox import KalmanFilter

# Access the filter results to inspect diffuse covariances
model = LocalLevel(y)
results = model.fit(initialization="diffuse")

filter_res = results.filter_results

# P_t^inf matrices — shape (T, m, m)
P_inf: np.ndarray = filter_res.predicted_diffuse_state_cov

# F_t^inf = Z P_t^inf Z' — shape (T, p, p)
# For local level: Z = [1], so F_t^inf is a scalar
F_inf: np.ndarray = np.array([
    filter_res.design[t] @ P_inf[t] @ filter_res.design[t].T
    for t in range(filter_res.nobs)
])

# Plot rank of P_t^inf over time (shows when diffuse period ends)
ranks: list[int] = [
    int(np.linalg.matrix_rank(P_inf[t], tol=1e-10))
    for t in range(filter_res.nobs)
]

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(F_inf[:, 0, 0], label=r"$F_t^\infty$")
axes[0].axvline(results.diffuse_periods, color="red", linestyle="--",
                label=f"End of diffuse period ($d={results.diffuse_periods}$)")
axes[0].set_ylabel(r"$F_t^\infty$")
axes[0].legend()

axes[1].step(range(filter_res.nobs), ranks, label=r"rank($P_t^\infty$)")
axes[1].set_ylabel(r"Rank of $P_t^\infty$")
axes[1].set_xlabel("Time")
axes[1].legend()
plt.tight_layout()
plt.show()
```

### 10.3 Comparing Diffuse vs. Approximate Initialization

```python
import numpy as np
from kalmanbox.models import LocalLevel
from kalmanbox import KalmanFilter

# Generate data
rng = np.random.default_rng(0)
T: int = 100
y: np.ndarray = np.cumsum(rng.normal(0, 1.0, T)) + rng.normal(0, 0.5, T)

model = LocalLevel(y)

# Exact diffuse initialization
res_exact = model.fit(initialization="diffuse")

# Approximate diffuse with kappa = 1e6 (default)
res_approx = model.fit(initialization="approximate_diffuse")

# Large kappa: less stable but observe sensitivity
res_large = model.fit(initialization="approximate_diffuse", initial_variance=1e8)

print("Initialization comparison:")
print(f"{'Method':<30} {'Log-lik':>12} {'sigma_eta':>12} {'sigma_eps':>12}")
print("-" * 68)
print(f"{'Exact diffuse':<30} {res_exact.llf:>12.4f} "
      f"{res_exact.params['sigma_eta']:>12.4f} "
      f"{res_exact.params['sigma_eps']:>12.4f}")
print(f"{'Approximate (kappa=1e6)':<30} {res_approx.llf:>12.4f} "
      f"{res_approx.params['sigma_eta']:>12.4f} "
      f"{res_approx.params['sigma_eps']:>12.4f}")
print(f"{'Approximate (kappa=1e8)':<30} {res_large.llf:>12.4f} "
      f"{res_large.params['sigma_eta']:>12.4f} "
      f"{res_large.params['sigma_eps']:>12.4f}")

# Check filtered state mean differences
diff_state: np.ndarray = np.abs(
    res_exact.filtered_state[0] - res_approx.filtered_state[0]
)
print(f"\nMax |filtered state difference| (exact vs approx): {diff_state.max():.2e}")
print(f"Max |filtered state difference| in stationary period: "
      f"{diff_state[res_exact.diffuse_periods:].max():.2e}")
```

### 10.4 BSM with Mixed Initialization

```python
import numpy as np
from kalmanbox.models import UnobservedComponents

# Quarterly GDP growth series (simulated)
rng = np.random.default_rng(7)
T: int = 120  # 30 years of quarterly data
seasonal_period: int = 4

# Simulate BSM: trend (diffuse) + seasonal (stationary) + irregular
trend: np.ndarray = np.cumsum(rng.normal(0, 0.5, T))  # diffuse level
seasonal: np.ndarray = np.tile(
    rng.normal(0, 0.3, seasonal_period), T // seasonal_period + 1
)[:T]
y: np.ndarray = trend + seasonal + rng.normal(0, 0.2, T)

# Fit BSM with automatic mixed initialization
model = UnobservedComponents(
    y,
    level="local level",
    seasonal=seasonal_period,
    initialization="auto",  # diffuse for trend, stationary for seasonal
)
results = model.fit()

print(f"Diffuse periods: {results.diffuse_periods}")
print(f"(Expected: 1 for local level, plus overlap with seasonal)")
print(f"\nParameter estimates:")
for name, val in results.params.items():
    print(f"  {name}: {val:.4f}")
```

### 10.5 ARIMA(1,1,1) Diffuse Initialization

```python
import numpy as np
from kalmanbox import KalmanFilter

# ARIMA(1,1,1) in state-space form
# State: alpha_t = (y_t - y_{t-1}, psi_t)' where psi_t is the MA component
# The first component is the differenced series, initialized diffusely

phi: float = 0.7  # AR coefficient
theta: float = 0.4  # MA coefficient

# State-space matrices for ARIMA(1,1,1)
T_mat: np.ndarray = np.array([[1 + phi, 1.0], [0.0, 0.0]])
Z_mat: np.ndarray = np.array([[1.0, theta]])
R_mat: np.ndarray = np.array([[1.0], [1.0]])

# Simulate
rng = np.random.default_rng(13)
T: int = 200
eta: np.ndarray = rng.normal(0, 1.0, T)
# (series generation omitted for brevity)

# Fit with 1 diffuse component (the integrated component)
kf = KalmanFilter(
    k_endog=1,
    k_states=2,
    k_posdef=1,
    transition=T_mat,
    design=Z_mat,
    selection=R_mat,
    state_cov=np.eye(1),
    obs_cov=np.array([[0.0]]),  # pure ARIMA: no observation noise
    initialization="diffuse",
)
# ... attach data and fit
print("ARIMA(1,1,1) diffuse periods: 1 (one unit root => d=1)")
```

---

## References

**Primary references:**

- **De Jong, P. (1991)**. "The diffuse Kalman filter." *Annals of Statistics*, 19(2), 1073–1083.
  The foundational paper establishing the exact diffuse initialization via partitioned covariance
  matrices and the modified Kalman recursion.

- **Koopman, S.J. (1997)**. "Exact initial Kalman filtering and smoothing for nonstationary
  time series models." *Journal of the American Statistical Association*, 92(440), 1630–1638.
  Alternative derivation via state augmentation; also establishes the exact diffuse smoother.

- **Durbin, J. and Koopman, S.J. (2012)**. *Time Series Analysis by State Space Methods*,
  2nd edition. Oxford University Press.
  Chapters 5 (regression effects), 7 (exact diffuse likelihood), and 11 (implementation).
  The standard reference for practical implementation of diffuse initialization.

- **Harvey, A.C. and Phillips, G.D.A. (1979)**. "Maximum likelihood estimation of regression
  models with autoregressive-moving average disturbances." *Biometrika*, 66(1), 49–58.
  Early treatment of approximate diffuse initialization using large $\kappa$.

- **Harvey, A.C. (1989)**. *Forecasting, Structural Time Series Models and the Kalman Filter*.
  Cambridge University Press. Chapter 3 covers approximate diffuse initialization in structural
  models.

- **Francke, M.K., Koopman, S.J., and de Vos, A.F. (2010)**. "Likelihood functions for state
  space models with diffuse initial conditions." *Journal of Time Series Analysis*, 31(6),
  407–414. Establishes the augmented Kalman filter framework for unified treatment of diffuse
  initialization and regression effects.

**Additional reading:**

- **Anderson, B.D.O. and Moore, J.B. (1979)**. *Optimal Filtering*. Prentice-Hall. Chapter 4
  on initialization and steady-state behaviour.

- **Hamilton, J.D. (1994)**. *Time Series Analysis*. Princeton University Press. Section 13.4
  on diffuse initialization for ARIMA models.

- **Shumway, R.H. and Stoffer, D.S. (2017)**. *Time Series Analysis and Its Applications*,
  4th edition. Springer. Section 6.3 on non-stationary state-space models.
