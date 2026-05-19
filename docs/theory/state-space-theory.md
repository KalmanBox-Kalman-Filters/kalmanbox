# State-Space Model Theory

The **linear Gaussian state-space model** is the unifying framework behind every algorithm in
`kalmanbox`. This page develops the theory from first principles: the equations themselves, the
algebra of system matrices, initialization strategies, stability theory, and the canonical
representations of common time-series models.

Cross-references:

- [Kalman filter derivation](kalman-filter-derivation.md) — how the filtering recursion is derived from the
  model below.
- [RTS smoother derivation](rts-derivation.md) — backward pass for smoothed state estimates.
- [Structural model theory](structural-theory.md) — local level, linear trend, BSM as special cases.
- [Kalman filter user guide](../user-guide/kalman/kalman-filter.md) — practical API usage.

---

## 1. General Linear Gaussian State-Space Representation

### 1.1 The Two-Equation Form

Let $t = 1, 2, \ldots, n$ index discrete time. A **general linear Gaussian state-space model**
(LG-SSM) consists of two stochastic difference equations:

**Observation equation**

$$
y_t = Z_t \alpha_t + d_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, H_t),
\tag{1}
$$

**Transition (state) equation**

$$
\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q_t).
\tag{2}
$$

The **initial state** is distributed as

$$
\alpha_1 \sim \mathcal{N}(a_1, P_1),
\tag{3}
$$

with $a_1$ and $P_1$ (possibly diffuse — see §3) treated as part of the model specification.

### 1.2 Dimensions

| Symbol | Meaning | Dimension |
|:------:|:--------|:---------:|
| $y_t$ | observation vector at time $t$ | $p \times 1$ |
| $\alpha_t$ | state vector at time $t$ | $m \times 1$ |
| $\eta_t$ | state disturbance | $r \times 1$ |
| $\varepsilon_t$ | observation disturbance | $p \times 1$ |

Throughout this page: $p$ = number of observable series, $m$ = number of latent states, $r$ =
number of independent state disturbances, with $r \leq m$ in general.

### 1.3 Independence Assumptions

The following mutual independence conditions must hold for the model to be well-defined:

1. **Cross-independence:** $\varepsilon_t \perp \eta_s$ for all $t, s = 1, \ldots, n$.
2. **Serial independence of observation noise:** $\varepsilon_t \perp \varepsilon_s$ for $t \neq s$.
3. **Serial independence of state noise:** $\eta_t \perp \eta_s$ for $t \neq s$.
4. **Initial-state independence:** $\alpha_1 \perp \varepsilon_t$ and $\alpha_1 \perp \eta_t$ for
   all $t$.

These four conditions make $\{y_t\}$ a **conditionally Gaussian Markov process** with one-step
ahead predictive distribution $y_t \mid y_{1:t-1} \sim \mathcal{N}(Z_t a_{t|t-1} + d_t,\; F_t)$,
which underpins the prediction-error decomposition of the log-likelihood.

!!! note "Correlated disturbances"
    Some formulations allow $\mathrm{Cov}(\varepsilon_t, \eta_t) = S_t \neq 0$. This correlation
    can always be eliminated by a change of variables: replace $\varepsilon_t$ with
    $\varepsilon_t - S_t Q_t^{-1} \eta_t$ and absorb the cross-term into $Z_t$ and $H_t$.
    `kalmanbox` supports the $S_t$ form directly via the `cross_cov` argument; internally it
    applies this transformation before running the standard recursion.

### 1.4 The Joint Distribution

Because both equations are linear and all distributions are Gaussian, the joint distribution of
$(\alpha_1, \ldots, \alpha_n, y_1, \ldots, y_n)$ is multivariate normal. The entire model is
parametrised by the sequence of system matrices

$$
\Theta = \{Z_t, d_t, H_t, T_t, c_t, R_t, Q_t\}_{t=1}^{n}
$$

together with the initial conditions $(a_1, P_1)$.

---

## 2. System Matrices and Their Properties

### 2.1 Comprehensive Matrix Table

| Matrix | Dimension | Name | Role |
|:------:|:---------:|:-----|:-----|
| $Z_t$ | $p \times m$ | Design (observation) | Maps latent state to observable space. Each row specifies which linear combination of state components is measured by the corresponding observable. |
| $T_t$ | $m \times m$ | Transition (state) | Propagates the state one step forward. Its eigenvalues determine system stability. |
| $R_t$ | $m \times r$ | Selection | Routes the $r$-dimensional disturbance $\eta_t$ into the $m$-dimensional state. Typically sparse (e.g., unit columns selecting which states are stochastic). |
| $H_t$ | $p \times p$ | Observation noise covariance | Variance of the measurement error. Must be **positive semi-definite** ($H_t \succeq 0$). |
| $Q_t$ | $r \times r$ | State noise covariance | Variance of state innovations. Must be **positive semi-definite** ($Q_t \succeq 0$). |
| $d_t$ | $p \times 1$ | Observation intercept | Deterministic offset applied after the state-to-output mapping. Often zero. |
| $c_t$ | $m \times 1$ | State intercept | Deterministic drift added to each state transition. Encodes constants or exogenous inputs. |

### 2.2 Positive Semi-Definiteness Requirements

Both covariance matrices must satisfy the semi-definiteness constraint:

$$
H_t \succeq 0 \quad \Longleftrightarrow \quad x' H_t x \geq 0 \quad \forall\, x \in \mathbb{R}^p,
$$

$$
Q_t \succeq 0 \quad \Longleftrightarrow \quad x' Q_t x \geq 0 \quad \forall\, x \in \mathbb{R}^r.
$$

In practice, `kalmanbox` enforces these constraints by parameterising each covariance matrix via
its Cholesky factor $L$ so that $\Sigma = L L'$; this guarantees $\Sigma \succeq 0$ throughout
optimisation without requiring constrained optimisation.

!!! note "Zero entries on the diagonal of $Q_t$"
    A zero diagonal entry $Q_{t,ii} = 0$ means the $i$-th state disturbance has zero variance,
    i.e., that component of the state evolves deterministically. This is mathematically valid but
    may lead to rank-deficient covariance matrices in the filter recursion; `kalmanbox` uses the
    **square-root** (Cholesky) filter in such cases to avoid numerical issues.

### 2.3 Effective State Noise Covariance

Although $Q_t$ is $r \times r$, the effective state noise covariance entering the prediction step
is the $m \times m$ matrix

$$
\tilde{Q}_t \;=\; R_t Q_t R_t'.
$$

This matrix has rank at most $r$. When $r < m$, some state components have no direct stochastic
forcing and their uncertainty propagates only through deterministic coupling in $T_t$.

### 2.4 Time-Invariant Shorthand

When the system matrices do not depend on $t$, we write $\{Z, T, R, H, Q, d, c\}$ without
subscripts. The LTI (Linear Time-Invariant) case admits a **steady-state** Kalman gain $K_\infty$
and a steady-state error covariance $P_\infty$ (see §5).

---

## 3. Initial Conditions

The Kalman filter requires a starting distribution for the state:

$$
\alpha_1 \sim \mathcal{N}(a_1, P_1).
$$

The choice of $(a_1, P_1)$ is non-trivial and has a substantial effect on the first few
filtered estimates and on the log-likelihood. Three strategies are in common use.

### 3.1 Stationary Initialization

If the transition matrix $T$ has all eigenvalues strictly inside the unit circle (i.e., the system
is asymptotically stable), the state process $\{\alpha_t\}$ has an unconditional stationary
distribution. Setting $a_1 = (I - T)^{-1} c$ and choosing $P_1$ as the stationary covariance is
then the natural and fully efficient initialization.

The stationary covariance solves the **discrete Lyapunov equation**:

$$
P_1 = T P_1 T' + R Q R'.
\tag{4}
$$

Applying the $\text{vec}$ operator (which stacks matrix columns) and using the Kronecker identity
$\text{vec}(AXB) = (B' \otimes A)\,\text{vec}(X)$, equation (4) becomes

$$
\text{vec}(P_1) = (T \otimes T)\,\text{vec}(P_1) + \text{vec}(R Q R'),
$$

which rearranges to

$$
\boxed{
\text{vec}(P_1) = \bigl(I_{m^2} - T \otimes T\bigr)^{-1} \text{vec}(R Q R').
}
\tag{5}
$$

The matrix $(I_{m^2} - T \otimes T)$ is invertible if and only if no product $\lambda_i \lambda_j$
of eigenvalues of $T$ equals 1, i.e., equivalently, $T \otimes T$ has no eigenvalue equal to 1.
For a stable $T$ this is automatically satisfied.

!!! tip "Numerical solution"
    Equation (5) requires inverting an $m^2 \times m^2$ matrix, which is expensive for large $m$.
    In practice `kalmanbox` uses the Bartels-Stewart algorithm (via `scipy.linalg.solve_discrete_lyapunov`)
    with $O(m^3)$ complexity, which is far more efficient than the naive Kronecker approach.

### 3.2 Diffuse Initialization

When some or all eigenvalues of $T$ lie on or outside the unit circle (non-stationary components
such as random walks or integrated processes), the stationary covariance does not exist. The
standard solution is **diffuse initialization**:

$$
a_1 = 0, \qquad P_1 = \kappa I_m, \quad \kappa \to \infty.
\tag{6}
$$

In the limit $\kappa \to \infty$, the filter becomes uninformative about the initial state. For
finite implementation one must either:

- **Approximate diffuse:** Set $\kappa$ to a large finite value (e.g., $10^6$). Simple but can
  cause numerical ill-conditioning.
- **Exact diffuse (Koopman & Durbin, 2003):** Decompose $P_1 = P_1^\star + \kappa P_1^\infty$,
  where $P_1^\infty$ identifies the diffuse directions and $P_1^\star$ captures any prior
  information. Run two parallel filter recursions tracking $P_t^\star$ and $P_t^\infty$
  separately, switching to the ordinary filter once the diffuse part has collapsed (i.e., once
  $P_t^\infty = 0$ to numerical precision).

The **exact diffuse log-likelihood** differs from the ordinary one by terms that account for the
diffuse components; `kalmanbox` computes this correctly. See the [likelihood page](likelihood.md)
for the formula.

### 3.3 Mixed Initialization

Many practical models are **partially non-stationary**: some state components (e.g., a local
level) are non-stationary while others (e.g., a stationary AR component) are stationary. The
**mixed initialization** strategy handles this by partitioning the state:

$$
\alpha_1 = \begin{pmatrix} \alpha_1^{(s)} \\ \alpha_1^{(d)} \end{pmatrix},
\quad
P_1 = \begin{pmatrix} P_1^{(s)} & 0 \\ 0 & \kappa I \end{pmatrix},
$$

where $P_1^{(s)}$ is the stationary covariance for the stationary block (computed via equation
(5) restricted to that block) and $\kappa I$ is diffuse for the non-stationary block.

!!! example "Local Linear Trend initialization"
    The local linear trend model has state $\alpha_t = (\mu_t, \nu_t)'$ where $\mu_t$ is the
    level (random walk) and $\nu_t$ is the slope (also a random walk). Both components are
    non-stationary, so the fully diffuse initialization is used:
    $$P_1 = \kappa I_2, \quad a_1 = (0, 0)'.$$

---

## 4. Stability and Observability Conditions

### 4.1 Asymptotic Stability

The LTI system is **asymptotically stable** if and only if all eigenvalues of $T$ lie strictly
inside the complex unit disc:

$$
\rho(T) \;:=\; \max_i |\lambda_i(T)| \;<\; 1,
\tag{7}
$$

where $\rho(T)$ is the **spectral radius** of $T$. Under this condition:

- The stationary distribution (§3.1) exists and is unique.
- The Kalman filter gain converges to a steady-state value.
- The state prediction error covariance $P_{t|t-1}$ converges to a finite limit $P_\infty$.

When $\rho(T) = 1$, the system has **unit-root** components (e.g., random walk level or
integrated trend). These are handled via diffuse initialization and exact diffuse recursions.
When $\rho(T) > 1$ the system is **explosive** — still representable in state-space form but
rarely encountered in econometric applications.

### 4.2 Observability

The system $(Z, T)$ is **observable** if the initial state $\alpha_1$ can be uniquely recovered
from a finite sequence of observations $y_1, \ldots, y_m$ in the noise-free case.

The **observability matrix** is

$$
\mathcal{O} = \begin{pmatrix}
Z \\ ZT \\ ZT^2 \\ \vdots \\ ZT^{m-1}
\end{pmatrix} \in \mathbb{R}^{pm \times m}.
\tag{8}
$$

**Theorem (Kalman, 1960):** The system is observable if and only if

$$
\operatorname{rank}(\mathcal{O}) = m.
\tag{9}
$$

Observability has important implications for identifiability and filter convergence:

- If the system is **not observable**, some linear combination of state components cannot be
  estimated from observations regardless of how much data is available.
- Lack of observability typically signals an over-parameterised model or a redundant state.

!!! example "Testing observability for the Local Level model"
    With $Z = [1]$ and $T = [1]$, the observability matrix is $\mathcal{O} = [1]$, which
    has rank 1 = $m$. The model is observable.

### 4.3 Controllability (Reachability)

The system $(T, R)$ is **controllable** if any state can be reached from any initial condition
through a suitable sequence of inputs $\eta_t$.

The **controllability matrix** is

$$
\mathcal{C} = \begin{pmatrix} R & TR & T^2R & \cdots & T^{m-1}R \end{pmatrix} \in \mathbb{R}^{m \times rm}.
\tag{10}
$$

**Theorem:** The system is controllable if and only if $\operatorname{rank}(\mathcal{C}) = m$.

In state-space filtering, the more relevant concept is **stabilizability** — whether the unstable
modes of $T$ can be influenced by the noise input through $R$. Stabilizability is necessary for
the filter covariance to converge.

### 4.4 Detectability and Stabilizability

**Detectability** is a weaker condition than observability: the system $(Z, T)$ is detectable if
all **unstable** modes (eigenvalues of $T$ with $|\lambda| \geq 1$) are observable.

**Stabilizability** is the dual: the system $(T, R)$ is stabilizable if all unstable modes are
controllable.

**Theorem (Riccati convergence):** Under the LTI assumption, the discrete algebraic Riccati
equation

$$
P_\infty = T P_\infty T' + R Q R' - T P_\infty Z' (Z P_\infty Z' + H)^{-1} Z P_\infty T'
\tag{11}
$$

has a unique positive semi-definite stabilizing solution $P_\infty$, and the Kalman filter
covariance $P_{t|t-1}$ converges to $P_\infty$ from any initial $P_1 \succeq 0$, **if and only
if** $(Z, T)$ is detectable and $(T, R\sqrt{Q})$ is stabilizable.

!!! note "Practical relevance"
    In practice, `kalmanbox` checks eigenvalues of $T$ and warns the user if the spectral radius
    exceeds 1 and no diffuse initialization is specified, since this typically indicates a
    mis-specified model or a missing unit-root transformation.

### 4.5 Relation to Riccati Equation Convergence

The steady-state Kalman gain is

$$
K_\infty = T P_\infty Z' F_\infty^{-1}, \qquad F_\infty = Z P_\infty Z' + H,
\tag{12}
$$

and the closed-loop transition matrix is

$$
T_{\mathrm{cl}} = T - K_\infty Z,
\tag{13}
$$

whose eigenvalues are all strictly inside the unit circle when the detectability/stabilizability
conditions hold. This guarantees that the **innovation process** $\{v_t\}$ is asymptotically
stationary even when the state process is not.

---

## 5. Time-Invariant vs Time-Varying Models

### 5.1 Linear Time-Invariant (LTI) Case

When all system matrices are constant over time ($Z_t = Z$, $T_t = T$, etc.), the model is
**linear time-invariant**. Key properties:

- The steady-state covariance $P_\infty$ (solution to (11)) and gain $K_\infty$ (equation (12))
  exist under the detectability/stabilizability conditions.
- For $t$ large enough, $P_{t|t-1} \approx P_\infty$ and the filter requires only a single
  matrix inverse per step.
- The innovation variance $F_\infty$ is constant, which simplifies likelihood evaluation.
- The spectral density of $\{y_t\}$ is available in closed form (see §6.3).

`kalmanbox` automatically detects the LTI case and pre-computes $P_\infty$ via the DARE solver,
then switches to the steady-state filter after a user-configurable burn-in period.

### 5.2 Time-Varying Models

System matrices may vary with $t$ for several reasons:

**Periodically varying (seasonal) models:**
The seasonal BSM with trigonometric seasonality uses fixed matrices throughout, but alternative
dummy-variable seasonal specifications use matrices that cycle with period $s$ (e.g., $s = 12$
for monthly data). One can always augment the state to render such systems LTI.

**Structurally changing models:**
Regime changes, interventions, or parameter instability lead to models where $T_t$ or $Z_t$
change at known break dates. These are handled naturally in the SSM framework by updating the
relevant matrices at the break date.

**Piecewise time-invariant:**
A practical strategy for long series with structural breaks is **piecewise time-invariant**
modeling: run the steady-state filter within each segment and reinitialize at each break point.
This gives $O(n)$ complexity (same as fully time-invariant) with exact treatment of breaks.

**Observation-driven (GARCH-in-SSM):**
When $H_t = h(\alpha_{t-1}, y_{t-1})$ depends on lagged quantities, the model is no longer
strictly linear Gaussian. However, conditional on the path, the standard filter applies.
`kalmanbox` supports this via the `time_varying_H` callback interface.

### 5.3 Efficient Computation for Time-Varying Systems

For a generic time-varying LG-SSM with $n$ time steps and $m$ states, the Kalman filter has
complexity $O(nm^3 + npm^2 + np^3)$. In the LTI case the dominant $m^3$ term is paid only once
(for the DARE solve), reducing per-step cost to $O(m^2 p + p^3)$.

---

## 6. Connections to Other Representations

### 6.1 ARMA(p, q) in Companion Form

An ARMA($p$, $q$) process satisfies

$$
y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}.
\tag{14}
$$

Let $r = \max(p, q+1)$. Define the state vector

$$
\alpha_t = \begin{pmatrix}
y_t \\ \phi_2 y_{t-1} + \cdots + \phi_r y_{t-r+1} + \theta_1 \varepsilon_{t-1} + \cdots + \theta_{r-1} \varepsilon_{t-r+1}
\\ \vdots \\ \text{(shifted predictors)}
\end{pmatrix} \in \mathbb{R}^r.
$$

More precisely, using the **innovation form** (also called the controllable canonical form):

$$
\alpha_t = \begin{pmatrix} \alpha_{1,t} \\ \alpha_{2,t} \\ \vdots \\ \alpha_{r,t} \end{pmatrix},
\qquad
\alpha_{1,t} = y_t,
\quad
\alpha_{j,t} = \phi_j y_{t-1} + \alpha_{j+1,t-1} + \theta_{j-1} \varepsilon_{t-1},
\; j = 2, \ldots, r,
$$

leading to the companion state-space representation:

$$
Z = \begin{pmatrix} 1 & 0 & 0 & \cdots & 0 \end{pmatrix}, \qquad
T = \begin{pmatrix}
\phi_1 & 1 & 0 & \cdots & 0 \\
\phi_2 & 0 & 1 & \cdots & 0 \\
\vdots & & & \ddots & \vdots \\
\phi_{r-1} & 0 & 0 & \cdots & 1 \\
\phi_r & 0 & 0 & \cdots & 0
\end{pmatrix},
\tag{15}
$$

$$
R = \begin{pmatrix} 1 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_{r-1} \end{pmatrix}, \qquad
Q = \sigma_\varepsilon^2, \qquad
H = 0.
\tag{16}
$$

!!! example "ARMA(2,1) fully specified"
    For $y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \varepsilon_t + \theta_1 \varepsilon_{t-1}$,
    we have $r = \max(2, 2) = 2$ and:

    $$
    Z = \begin{pmatrix} 1 & 0 \end{pmatrix}, \quad
    T = \begin{pmatrix} \phi_1 & 1 \\ \phi_2 & 0 \end{pmatrix}, \quad
    R = \begin{pmatrix} 1 \\ \theta_1 \end{pmatrix}, \quad
    Q = \sigma_\varepsilon^2, \quad H = 0.
    $$

    The state vector is $\alpha_t = (y_t,\; \phi_2 y_{t-1} + \theta_1 \varepsilon_{t-1})'$.
    The transition eigenvalues $\lambda_1, \lambda_2$ are the roots of $\lambda^2 - \phi_1
    \lambda - \phi_2 = 0$; stationarity requires both roots inside the unit circle.

### 6.2 VAR(p) in Companion Form

A vector autoregression of order $p$ for an $\ell$-dimensional series $y_t$:

$$
y_t = A_1 y_{t-1} + \cdots + A_p y_{t-p} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma),
\tag{17}
$$

is rewritten by stacking $p$ lags into the **companion state** $\alpha_t = (y_t', y_{t-1}', \ldots, y_{t-p+1}')'$:

$$
Z = \begin{pmatrix} I_\ell & 0 & \cdots & 0 \end{pmatrix}, \quad
T = \begin{pmatrix}
A_1 & A_2 & \cdots & A_{p-1} & A_p \\
I_\ell & 0 & \cdots & 0 & 0 \\
0 & I_\ell & \cdots & 0 & 0 \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & I_\ell & 0
\end{pmatrix}, \quad
R = \begin{pmatrix} I_\ell \\ 0 \\ \vdots \\ 0 \end{pmatrix},
\tag{18}
$$

$$
Q = \Sigma, \qquad H = 0.
$$

The companion state has dimension $m = p\ell$. Stationarity of the VAR requires all eigenvalues
of the $p\ell \times p\ell$ companion matrix $T$ to lie strictly inside the unit circle —
equivalently, all roots of $\det(I_\ell - A_1 z - \cdots - A_p z^p) = 0$ must satisfy $|z| > 1$.

### 6.3 Transfer Function and Spectral Representation

For an LTI state-space model (time-invariant matrices, $d = 0$, $c = 0$), the **transfer
function** from the state disturbance $\eta_t$ to the output $y_t$ is

$$
G(z) = Z (zI - T)^{-1} R,
\tag{19}
$$

where $z$ is the complex frequency variable. The **spectral density matrix** of $\{y_t\}$ is

$$
\mathbf{f}(\omega) = G(e^{i\omega}) Q G(e^{i\omega})^* + H, \qquad \omega \in [-\pi, \pi],
\tag{20}
$$

where $(\cdot)^*$ denotes Hermitian transpose. This is the **spectral factorization** perspective:
the LTI SSM provides a rational spectral density for $\{y_t\}$ whose poles are determined by the
eigenvalues of $T$ and whose zeros are determined by the zeros of $G(z)$.

!!! tip "Checking model frequency content"
    For seasonal models, the spectral density should have peaks at the seasonal frequencies
    $\omega_k = 2\pi k / s$, $k = 1, \ldots, \lfloor s/2 \rfloor$. If the estimated spectral
    density is flat at seasonal frequencies, this suggests insufficient seasonal variance $q_\omega$.

### 6.4 Unobserved Components Models

An **unobserved components** (UC) model decomposes $y_t$ additively into latent components:

$$
y_t = \mu_t + \psi_t + \gamma_t + \varepsilon_t,
\tag{21}
$$

where $\mu_t$ is a trend, $\psi_t$ is a cycle, $\gamma_t$ is a seasonal, and $\varepsilon_t$ is
irregular noise. Each component evolves according to its own stochastic equation. The UC model is
a state-space model with:

- State vector: $\alpha_t = (\mu_t, \nu_t, \psi_t, \psi_t^*, \gamma_{1,t}, \ldots, \gamma_{s/2-1,t}, \gamma_{s/2-1,t}^*)'$
  (stacking trend states, cycle states, and seasonal states).
- Block-diagonal system matrices $Z$, $T$, $R$, $Q$ (each block corresponds to a component).
- Observation equation: $Z = (1, 0, 1, 0, 1, 0, \ldots, 1, 0)$ picks out the relevant state
  for each component.

The additive structure means the overall state transition is block-diagonal, which is numerically
efficient and aids interpretability. See [structural model theory](structural-theory.md) for the
full derivation.

---

## 7. Canonical Examples with Full Matrix Specifications

This section provides complete model specifications for the most common state-space models.

### 7.1 Local Level Model

**Setting:** $m = 1$, $p = 1$, $r = 1$.

The simplest non-trivial SSM: a random walk level $\mu_t$ observed with noise.

$$
y_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2),
$$
$$
\mu_{t+1} = \mu_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2).
$$

**System matrices:**

$$
Z = [1], \quad T = [1], \quad R = [1], \quad H = [\sigma_\varepsilon^2], \quad Q = [\sigma_\eta^2],
\quad d = [0], \quad c = [0].
$$

**Parameters:** Two variances $\sigma_\varepsilon^2 > 0$ and $\sigma_\eta^2 \geq 0$. The
**signal-to-noise ratio** $q = \sigma_\eta^2 / \sigma_\varepsilon^2$ governs the smoothness of
the extracted trend. When $q = 0$, $\mu_t$ is constant; as $q \to \infty$, $\mu_t$ becomes a
random walk indistinguishable from $y_t$.

**Initialization:** Diffuse, $P_1 = \kappa$ as $\kappa \to \infty$.

### 7.2 Local Linear Trend Model

**Setting:** $m = 2$, $p = 1$, $r = 2$.

Extends the local level with a stochastic slope (drift):

$$
y_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2),
$$
$$
\mu_{t+1} = \mu_t + \nu_t + \xi_t, \qquad \xi_t \sim \mathcal{N}(0, \sigma_\xi^2),
$$
$$
\nu_{t+1} = \nu_t + \zeta_t, \qquad \zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2),
$$

with state $\alpha_t = (\mu_t, \nu_t)'$.

**System matrices:**

$$
Z = \begin{pmatrix} 1 & 0 \end{pmatrix}, \quad
T = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad
R = I_2, \quad
H = [\sigma_\varepsilon^2], \quad
Q = \begin{pmatrix} \sigma_\xi^2 & 0 \\ 0 & \sigma_\zeta^2 \end{pmatrix}.
$$

**Initialization:** Diffuse for both components, $P_1 = \kappa I_2$.

**Special cases:**

- $\sigma_\zeta^2 = 0$: deterministic slope (slope constant over time).
- $\sigma_\xi^2 = 0$: integrated random walk slope (smooth trend, used in cubic spline smoothing).

### 7.3 Basic Structural Model with Trigonometric Seasonality

**Setting:** Let $s$ be the seasonal period (e.g., $s = 12$ for monthly). The number of
trigonometric harmonics is $k = \lfloor s/2 \rfloor$.

**Component structure** (period $s = 12$ example):

- Trend: $\alpha^{(\mu)}_t = (\mu_t, \nu_t)' \in \mathbb{R}^2$
- Seasonal: $\alpha^{(\gamma)}_t = (\gamma_{1,t}, \gamma_{1,t}^*, \ldots, \gamma_{6,t})' \in \mathbb{R}^{s-1}$
  (using $k = 5$ pairs and 1 singleton for $s = 12$, total $s-1 = 11$ states)
- Irregular: absorbed into $H$

**Total state dimension:** $m = 2 + (s-1) = s + 1$.

The $j$-th harmonic block in $T$ is the rotation matrix

$$
T_j = \begin{pmatrix} \cos \lambda_j & \sin \lambda_j \\ -\sin \lambda_j & \cos \lambda_j \end{pmatrix},
\qquad \lambda_j = \frac{2\pi j}{s}, \quad j = 1, \ldots, \lfloor s/2 \rfloor - 1.
$$

For the Nyquist frequency ($j = s/2$ when $s$ is even), the block degenerates to a scalar $T_{s/2} = [-1]$.

**Full system matrices ($s = 12$):**

$$
Z = \begin{pmatrix} 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix},
$$

$$
T = \mathrm{blockdiag}\!\left(
\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix},\;
T_1,\; T_2,\; T_3,\; T_4,\; T_5,\; [-1]
\right),
$$

$$
R = I_{13}, \quad Q = \mathrm{diag}(\sigma_\xi^2, \sigma_\zeta^2, \sigma_{\omega_1}^2, \sigma_{\omega_1}^2, \ldots, \sigma_{\omega_5}^2, \sigma_{\omega_5}^2, \sigma_{\omega_6}^2).
$$

!!! note "Restricted BSM"
    The standard BSM further restricts $\sigma_{\omega_j}^2 = \sigma_\omega^2$ for all harmonics
    $j$, reducing seasonal parameters from $\lfloor s/2 \rfloor$ to 1. This is the default in
    `kalmanbox.BSM`.

### 7.4 AR(2) in Companion Form

**Setting:** $m = 2$, $p = 1$, $r = 1$.

Scalar AR(2): $y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \varepsilon_t$, $\sigma_\varepsilon^2$.

$$
Z = \begin{pmatrix} 1 & 0 \end{pmatrix}, \quad
T = \begin{pmatrix} \phi_1 & 1 \\ \phi_2 & 0 \end{pmatrix}, \quad
R = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad
H = [0], \quad Q = [\sigma_\varepsilon^2].
$$

Note $H = 0$: all randomness enters through the state. The state vector is
$\alpha_t = (y_t,\; \phi_2 y_{t-1})'$.

**Stationarity region:** $|\phi_2| < 1$, $\phi_1 + \phi_2 < 1$, $\phi_2 - \phi_1 < 1$
(the interior of the triangular parameter region in the $(\phi_1, \phi_2)$ plane).

**Initialization:** Stationary covariance from equation (5), which for the scalar ARMA case
reduces to

$$
\begin{pmatrix} \gamma(0) & \gamma(1) \\ \gamma(1) & \gamma(0) \end{pmatrix},
$$

where $\gamma(h)$ are the autocovariances of the AR(2) process.

### 7.5 Static Factor Model

**Setting:** $k$ latent factors, $p$ observable series, $m = k$, $r = k$.

$$
y_t = \Lambda f_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Psi),
$$
$$
f_{t+1} = A f_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, I_k),
$$

where $\Lambda \in \mathbb{R}^{p \times k}$ is the **factor loading matrix**, $f_t \in
\mathbb{R}^k$ are the latent factors, $\Psi = \mathrm{diag}(\psi_1^2, \ldots, \psi_p^2)$ is the
**idiosyncratic variance** matrix (diagonal by assumption), and $A$ is a $k \times k$ factor
transition matrix (often set to $I_k$ for the static factor model or to a diagonal/companion
matrix for a dynamic factor model).

**System matrices:**

$$
Z = \Lambda, \quad T = A, \quad R = I_k, \quad H = \Psi, \quad Q = I_k.
$$

**Identification:** The factor model is not identified without restrictions. Standard approaches:

1. **Lower-triangular $\Lambda$:** Fix the upper triangle of $\Lambda$ to zero and the diagonal to
   positive values (eliminates rotation freedom).
2. **Orthonormal factors:** $Q = I_k$ and the first $k \times k$ block of $\Lambda$ is
   lower-triangular.
3. **Maximum likelihood:** Additional constraints may be imposed to fix scale and rotation.

!!! example "Dynamic Factor Model (DFM)"
    Setting $A = \mathrm{diag}(\rho_1, \ldots, \rho_k)$ with $|\rho_i| < 1$ gives stationary
    AR(1) factors. Setting $A = I_k$ gives random walk factors (integrated DFM). The state
    dimension is $m = k$ in either case; forecasting exploits the state recursion directly.

---

## 8. Summary of Model Properties

The table below summarises the key properties of each canonical example.

| Model | $m$ | $p$ | $r$ | Stationary? | Diffuse dims | Parameters |
|:------|:---:|:---:|:---:|:-----------:|:------------:|:-----------|
| Local Level | 1 | 1 | 1 | No | 1 | $\sigma_\varepsilon^2, \sigma_\eta^2$ |
| Local Linear Trend | 2 | 1 | 2 | No | 2 | $\sigma_\varepsilon^2, \sigma_\xi^2, \sigma_\zeta^2$ |
| BSM ($s=12$) | 13 | 1 | 13 | No | 2 | $\sigma_\varepsilon^2, \sigma_\xi^2, \sigma_\zeta^2, \sigma_\omega^2$ |
| AR(2) | 2 | 1 | 1 | Yes | 0 | $\phi_1, \phi_2, \sigma_\varepsilon^2$ |
| ARMA(2,1) | 2 | 1 | 1 | Yes | 0 | $\phi_1, \phi_2, \theta_1, \sigma_\varepsilon^2$ |
| VAR(2), $\ell=3$ | 6 | 3 | 3 | Yes | 0 | $A_1, A_2, \Sigma$ |
| Static Factor, $k=2$, $p=10$ | 2 | 10 | 2 | Yes | 0 | $\Lambda, \Psi, A$ |

---

## 9. Implementation Notes for `kalmanbox`

The classes in `kalmanbox` translate directly to the framework above:

```python
from kalmanbox import KalmanFilter
import numpy as np

# AR(2) in companion form
phi1, phi2, sigma2 = 0.5, 0.2, 1.0

kf = KalmanFilter(
    Z=np.array([[1.0, 0.0]]),          # (1 x 2)
    T=np.array([[phi1, 1.0],
                [phi2, 0.0]]),          # (2 x 2)
    R=np.array([[1.0], [0.0]]),         # (2 x 1)
    H=np.array([[0.0]]),                # (1 x 1), no observation noise
    Q=np.array([[sigma2]]),             # (1 x 1)
    init="stationary",                  # solve Lyapunov equation
)
```

```python
from kalmanbox import LocalLevel

# Local Level Model — diffuse initialization by default
ll = LocalLevel(sigma_eps=1.0, sigma_eta=0.3)

# Under the hood this sets:
# Z = [[1]], T = [[1]], R = [[1]]
# H = [[sigma_eps**2]], Q = [[sigma_eta**2]]
# init = "diffuse"
```

```python
from kalmanbox import BSM

# Basic Structural Model, monthly data (s=12)
bsm = BSM(
    period=12,
    stochastic_trend=True,
    stochastic_slope=True,
    stochastic_seasonal=True,
)
# Internally builds m = 13 state vector with block-diagonal T
```

!!! tip "Checking your model specification"
    Use `kf.summary()` to print the dimensions and a condensed view of all system matrices.
    Use `kf.check_stability()` to verify eigenvalues of $T$ and warn if the model is non-stationary
    but stationary initialization was requested.

---

## 10. Mathematical Notation Glossary

| Symbol | Meaning |
|:------:|:--------|
| $y_t$ | Observation vector, $p \times 1$ |
| $\alpha_t$ | State vector, $m \times 1$ |
| $\varepsilon_t$ | Observation disturbance, $p \times 1$ |
| $\eta_t$ | State disturbance, $r \times 1$ |
| $Z_t$ | Design matrix, $p \times m$ |
| $T_t$ | Transition matrix, $m \times m$ |
| $R_t$ | Selection matrix, $m \times r$ |
| $H_t$ | Observation noise covariance, $p \times p$ |
| $Q_t$ | State noise covariance, $r \times r$ |
| $d_t$ | Observation intercept, $p \times 1$ |
| $c_t$ | State intercept, $m \times 1$ |
| $a_1$ | Prior mean of initial state, $m \times 1$ |
| $P_1$ | Prior covariance of initial state, $m \times m$ |
| $\mathcal{O}$ | Observability matrix, $pm \times m$ |
| $\mathcal{C}$ | Controllability matrix, $m \times rm$ |
| $P_\infty$ | Steady-state prediction error covariance, $m \times m$ |
| $K_\infty$ | Steady-state Kalman gain, $m \times p$ |
| $\rho(T)$ | Spectral radius of $T$ |
| $\otimes$ | Kronecker product |
| $\text{vec}(\cdot)$ | Column-stacking vectorisation operator |

---

## References

- **Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space Methods*, 2nd
  edition. Oxford University Press. — The primary reference for the formulation used in
  `kalmanbox`. Chapters 2–4 cover the model, initialization, and diffuse recursions in depth.

- **Harvey, A.C. (1989).** *Forecasting, Structural Time Series Models and the Kalman Filter*.
  Cambridge University Press. — Excellent treatment of unobserved components models and their
  state-space representations; the standard reference for the BSM.

- **Anderson, B.D.O. & Moore, J.B. (1979).** *Optimal Filtering*. Prentice-Hall. — The control-
  theory perspective on state-space models, observability, controllability, and the Riccati
  equation; rigorous treatment of stability and convergence.

- **Shumway, R.H. & Stoffer, D.S. (2017).** *Time Series Analysis and Its Applications: With R
  Examples*, 4th edition. Springer. — Accessible introduction to state-space models with EM
  estimation; useful complement to Durbin & Koopman for practitioners.

- **Kalman, R.E. (1960).** A new approach to linear filtering and prediction problems. *Journal of
  Basic Engineering*, 82(1), 35–45. — The original paper establishing the filter and the rank
  conditions for observability and controllability.

- **Koopman, S.J. & Durbin, J. (2003).** Filtering and smoothing of state vector for diffuse
  state-space models. *Journal of Time Series Analysis*, 24(1), 85–98. — The exact diffuse
  initialization algorithm implemented in `kalmanbox`.
