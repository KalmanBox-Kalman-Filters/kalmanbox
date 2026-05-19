# Nonlinear Filter Theory: EKF, UKF, and EnKF

This page develops the mathematical foundations of **nonlinear Bayesian filtering**
as implemented in `kalmanbox`. The central challenge is that the Kalman filter, which
is exact for linear Gaussian models, must be approximated when either the state
transition function $f$ or the observation function $h$ is nonlinear. Three principal
approximation strategies are covered here: **local linearization** (Extended Kalman
Filter), **deterministic sigma-point sampling** (Unscented Kalman Filter), and
**Monte Carlo ensemble methods** (Ensemble Kalman Filter).

Cross-references:

- [Kalman filter theory](kalman-theory.md) — the exact linear Gaussian recursion
  that nonlinear filters generalise.
- [State-space theory](state-space-theory.md) — the general model framework.
- [EKF user guide](../user-guide/filters/ekf.md) — practical API and examples.
- [UKF user guide](../user-guide/filters/ukf.md) — sigma-point tuning and usage.
- [Ensemble filter guide](../user-guide/filters/ensemble.md) — high-dimensional
  data assimilation.
- [Filter comparison](../user-guide/filters/comparison.md) — empirical benchmarks
  across filter types.

---

## 1. Nonlinear State-Space Models

### 1.1 General Form

A **nonlinear Gaussian state-space model** generalises the linear model by replacing
the matrix multiplications $Z_t \alpha_t$ and $T_t \alpha_t$ with arbitrary smooth
functions. The two-equation form is:

**Observation equation:**

$$
y_t = h(\alpha_t) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, H_t)
\tag{1}
$$

**State transition equation:**

$$
\alpha_{t+1} = f(\alpha_t) + R_t \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\, Q_t)
\tag{2}
$$

where $f : \mathbb{R}^m \to \mathbb{R}^m$ and $h : \mathbb{R}^m \to \mathbb{R}^p$
are known, (possibly) nonlinear, differentiable functions, and $R_t \in \mathbb{R}^{m
\times r}$ is the noise-selection matrix.

The more general form, which allows noise to enter nonlinearly, is:

$$
y_t = h(\alpha_t,\, \varepsilon_t), \qquad \alpha_{t+1} = f(\alpha_t,\, \eta_t)
\tag{3}
$$

`kalmanbox` handles the additive-noise form (1)–(2) natively; the fully general form
(3) can be accommodated by augmenting the state vector with the noise terms.

| Symbol | Dimension | Meaning |
|:-------|:----------|:--------|
| $y_t$ | $p \times 1$ | Observed vector at time $t$ |
| $\alpha_t$ | $m \times 1$ | Latent state vector |
| $f(\cdot)$ | $\mathbb{R}^m \to \mathbb{R}^m$ | Nonlinear transition function |
| $h(\cdot)$ | $\mathbb{R}^m \to \mathbb{R}^p$ | Nonlinear observation function |
| $H_t$ | $p \times p$ | Observation noise covariance |
| $Q_t$ | $r \times r$ | State noise covariance |
| $R_t$ | $m \times r$ | Noise-selection matrix |

### 1.2 Why the Kalman Filter Fails for Nonlinear Models

The classical Kalman filter is **optimal** for the linear Gaussian model: it computes
the exact conditional distribution $p(\alpha_t \mid \mathcal{Y}_t) = \mathcal{N}(a_{t|t},
P_{t|t})$ in closed form. Two key properties make this possible:

1. **Linearity preserves Gaussianity.** If $\alpha_t \mid \mathcal{Y}_{t-1}$ is
   Gaussian and $f$ is linear, then $\alpha_{t+1} \mid \mathcal{Y}_{t-1}$ is also
   Gaussian. Similarly for $h$.
2. **The conditional is characterised by its mean and covariance.** Under Gaussianity
   only two moments need tracking — a finite-dimensional sufficient statistic.

When $f$ or $h$ is nonlinear, both properties fail:

- **Non-Gaussian propagation.** A Gaussian random variable $\alpha_t$ pushed through
  a nonlinear function $f$ yields a non-Gaussian distribution. The conditional
  $p(\alpha_{t+1} \mid \mathcal{Y}_t)$ is no longer Gaussian.
- **Infinite-dimensional filtering distribution.** The exact posterior
  $p(\alpha_t \mid \mathcal{Y}_t)$ cannot, in general, be represented by a finite
  set of sufficient statistics. Tracking it exactly requires infinite memory.

!!! warning "The Closure Problem"
    Without linearity, the Gaussian closure fails: posterior moments beyond order two
    are no longer zero, and ignoring them introduces a systematic bias. All practical
    nonlinear filters — EKF, UKF, EnKF — make some form of Gaussian approximation
    to achieve tractability.

### 1.3 Overview of Approximation Strategies

Three broad families of approximations are used in practice:

| Strategy | Method | Key idea |
|:---------|:-------|:---------|
| **Local linearization** | EKF | Approximate $f, h$ by first-order Taylor expansion |
| **Sigma-point / deterministic sampling** | UKF, CKF | Propagate carefully chosen deterministic points |
| **Monte Carlo ensemble** | EnKF, PF | Represent posterior by weighted/equal-weight particles |

### 1.4 The Optimal (Intractable) Bayesian Filter

The exact solution to the filtering problem is given by two equations.

**Prediction step** (Chapman-Kolmogorov equation):

$$
p(\alpha_t \mid \mathcal{Y}_{t-1})
= \int p(\alpha_t \mid \alpha_{t-1})\; p(\alpha_{t-1} \mid \mathcal{Y}_{t-1})\;
  d\alpha_{t-1}
\tag{4}
$$

This marginalises over the previous state, propagating the posterior distribution
forward through the transition density $p(\alpha_t \mid \alpha_{t-1})$.

**Update step** (Bayes theorem):

$$
p(\alpha_t \mid \mathcal{Y}_t)
= \frac{p(y_t \mid \alpha_t)\; p(\alpha_t \mid \mathcal{Y}_{t-1})}
       {p(y_t \mid \mathcal{Y}_{t-1})}
\tag{5}
$$

where the normalising constant is:

$$
p(y_t \mid \mathcal{Y}_{t-1})
= \int p(y_t \mid \alpha_t)\; p(\alpha_t \mid \mathcal{Y}_{t-1})\; d\alpha_t
\tag{6}
$$

For the linear Gaussian model, (4)–(6) yield closed-form Gaussian updates — the
Kalman filter. For nonlinear $f$ or $h$, the integrals in (4) and (6) are intractable
in general.

!!! note "The Role of Gaussian Approximations"
    All three algorithms treated in this page — EKF, UKF, and EnKF — differ only in
    *how* they approximate the intractable integrals (4) and (6). The EKF uses first-order
    Taylor expansions; the UKF uses a set of deterministic quadrature points; the EnKF
    uses a Monte Carlo ensemble. All three ultimately represent the posterior as a Gaussian
    $\mathcal{N}(a_{t|t}, P_{t|t})$.

---

## 2. Extended Kalman Filter (EKF)

### 2.1 First-Order Taylor Linearization

The EKF approximates $f$ and $h$ by their first-order Taylor expansions around the
current best state estimate.

**Linearization of the transition function** around the filtered estimate
$a_{t|t} = \hat\alpha_{t|t}$:

$$
f(\alpha_t) \approx f(a_{t|t}) + F_t\,(\alpha_t - a_{t|t})
\tag{7}
$$

**Linearization of the observation function** around the predicted estimate
$a_{t|t-1} = \hat\alpha_{t|t-1}$:

$$
h(\alpha_t) \approx h(a_{t|t-1}) + H_t\,(\alpha_t - a_{t|t-1})
\tag{8}
$$

The **Jacobian matrices** are:

$$
F_t = \left.\frac{\partial f}{\partial \alpha}\right|_{\alpha = a_{t|t}}
    \in \mathbb{R}^{m \times m}
\tag{9}
$$

$$
H_t = \left.\frac{\partial h}{\partial \alpha}\right|_{\alpha = a_{t|t-1}}
    \in \mathbb{R}^{p \times m}
\tag{10}
$$

The $(i,j)$-th entry of $F_t$ is $\partial f_i / \partial \alpha_j$ evaluated at
$a_{t|t}$, and analogously for $H_t$.

### 2.2 The EKF Predict-Update Recursion

Substituting the linearized approximations (7)–(8) into the Kalman filter recursion
yields the complete EKF algorithm. All five equations are listed below.

**Predict step:**

$$
a_{t|t-1} = f(a_{t-1|t-1})
\tag{11}
$$

$$
P_{t|t-1} = F_{t-1}\, P_{t-1|t-1}\, F_{t-1}^\top + R_{t-1}\, Q_{t-1}\, R_{t-1}^\top
\tag{12}
$$

**Innovation and its covariance:**

$$
v_t = y_t - h(a_{t|t-1})
\tag{13}
$$

$$
S_t = H_t\, P_{t|t-1}\, H_t^\top + H_t
\tag{14}
$$

!!! note "Notation"
    In (14) the second $H_t$ on the right is the **observation noise covariance**
    matrix $H_t \in \mathbb{R}^{p \times p}$ from equation (1). The first $H_t$ and
    its transpose are the **Jacobian** from (10). `kalmanbox` uses `obs_jac` for the
    Jacobian and `obs_cov` for the noise covariance to avoid ambiguity.

**Update step:**

$$
K_t = P_{t|t-1}\, H_t^\top\, S_t^{-1}
\tag{15}
$$

$$
a_{t|t} = a_{t|t-1} + K_t\, v_t
\tag{16}
$$

$$
P_{t|t} = (I - K_t H_t)\, P_{t|t-1}
\tag{17}
$$

!!! tip "Joseph Form for Numerical Stability"
    The symmetric **Joseph stabilized form** of the covariance update,
    $(I - K_t H_t) P_{t|t-1} (I - K_t H_t)^\top + K_t H_t K_t^\top$, ensures
    $P_{t|t}$ remains positive semi-definite even under floating-point rounding.
    `kalmanbox` uses this form by default. See
    [numerical-stability.md](numerical-stability.md) for the derivation.

### 2.3 Second-Order EKF

The first-order EKF ignores terms $O(\|\alpha_t - a_{t|t}\|^2)$ in the Taylor
expansion, which introduces a **bias** in the predicted mean. The second-order EKF
corrects this by including the Hessian:

$$
a_{t|t-1}^{\mathrm{(2nd)}} = f(a_{t-1|t-1})
+ \frac{1}{2} \sum_{j=1}^{m} [Q_{t-1}]_{jj}\; \nabla^2 f_j(a_{t-1|t-1})
\tag{18}
$$

where $\nabla^2 f_j$ is the Hessian of the $j$-th component of $f$. The covariance
update also receives a correction term. In practice the second-order EKF is rarely
used because computing and storing $m$ Hessian matrices costs $O(m^3)$ additional
memory and the improvement is modest for mildly nonlinear problems.

### 2.4 Linearization Error and When the EKF Fails

The EKF approximation error is controlled by the **degree of nonlinearity** of $f$
and $h$ relative to the uncertainty $P_{t|t}$. More precisely, the EKF is
accurate when:

$$
\left\|\frac{\partial^2 f_i}{\partial \alpha_j \partial \alpha_k}\right\|\;
\sqrt{[P_{t|t}]_{jj} [P_{t|t}]_{kk}}
\ll 1 \quad \forall\, i, j, k
\tag{19}
$$

that is, when the curvature of $f$ multiplied by the current state uncertainty is
small. The EKF systematically fails in several scenarios:

- **Highly curved manifolds.** When $f$ or $h$ bends strongly over the uncertainty
  ellipsoid, the linear approximation is poor across the full probability mass.
- **Large initial uncertainty.** At startup, $P_{1|0}$ may be large, making the
  local approximation unreliable regardless of the curvature.
- **Multi-modal posteriors.** Any unimodal Gaussian approximation fails when the
  true posterior has multiple modes — common in bearing-only tracking and some
  SLAM problems.
- **Discontinuous functions.** $f$ or $h$ must be differentiable for the Jacobian
  to exist. Piecewise-linear or indicator-based observation functions require
  alternative approaches.

### 2.5 Computational Cost

Each EKF step requires:

- **Jacobian computation:** $O(mp)$ operations for $H_t$ and $O(m^2)$ for $F_t$,
  using either analytic formulas or automatic differentiation.
- **Covariance propagation:** $O(m^3)$ for the matrix products in (12).
- **Kalman gain:** $O(p^3)$ for the inversion $S_t^{-1}$ plus $O(m^2 p)$ for the
  gain product.

For typical state-space models with $m \sim p \sim O(10)$, each EKF step runs in
microseconds. The dominant cost is $O(m^3)$ per time step.

---

## 3. Unscented Kalman Filter (UKF)

### 3.1 Motivation: Deterministic Sampling

The EKF linearizes first and then propagates moments — an approximation that is
accurate only to first order. The **Unscented Kalman Filter** (Julier & Uhlmann,
1997) inverts this order: it propagates a set of carefully chosen deterministic
**sigma points** through the exact nonlinear functions, then reconstructs the mean
and covariance from the transformed points.

!!! definition "Key Insight (Julier & Uhlmann, 1997)"
    "It is easier to approximate a probability distribution than to approximate an
    arbitrary nonlinear function." The Unscented Transform uses $2m+1$ deterministic
    sigma points to capture the Gaussian distribution to **third order** in the
    Taylor series — two orders higher than the EKF — without requiring any Jacobians.

### 3.2 The Unscented Transform

Let $\mu \in \mathbb{R}^m$ and $P \in \mathbb{R}^{m \times m}$ (positive
semi-definite) represent a Gaussian random variable $x \sim \mathcal{N}(\mu, P)$.
The **Unscented Transform** of the nonlinear function $g : \mathbb{R}^m \to
\mathbb{R}^n$ proceeds as follows.

**Step 1: Generate sigma points.** Form $2m+1$ sigma points:

$$
\mathcal{X}_0 = \mu
\tag{20}
$$

$$
\mathcal{X}_i = \mu + \left(\sqrt{(m+\lambda)\,P}\right)_i, \quad i = 1, \ldots, m
\tag{21}
$$

$$
\mathcal{X}_{m+i} = \mu - \left(\sqrt{(m+\lambda)\,P}\right)_i, \quad i = 1, \ldots, m
\tag{22}
$$

where $\left(\sqrt{(m+\lambda)\,P}\right)_i$ denotes the $i$-th **column** of the
matrix square root (Cholesky factor), and:

$$
\lambda = \alpha^2(m + \kappa) - m
\tag{23}
$$

**Step 2: Compute weights.** The mean and covariance weights are:

$$
W_0^m = \frac{\lambda}{m + \lambda}
\tag{24}
$$

$$
W_i^m = \frac{1}{2(m + \lambda)}, \quad i = 1, \ldots, 2m
\tag{25}
$$

$$
W_0^c = \frac{\lambda}{m + \lambda} + (1 - \alpha^2 + \beta)
\tag{26}
$$

$$
W_i^c = \frac{1}{2(m + \lambda)}, \quad i = 1, \ldots, 2m
\tag{27}
$$

Note that $\sum_{i=0}^{2m} W_i^m = 1$ always holds, but $W_i^c \neq W_i^m$ for
the central point ($i=0$) due to the $\beta$ correction.

**Step 3: Propagate through the nonlinear function.**

$$
\mathcal{Y}_i = g(\mathcal{X}_i), \quad i = 0, 1, \ldots, 2m
\tag{28}
$$

**Step 4: Reconstruct mean and covariance.**

$$
\bar{y} = \sum_{i=0}^{2m} W_i^m\, \mathcal{Y}_i
\tag{29}
$$

$$
P_{yy} = \sum_{i=0}^{2m} W_i^c\, (\mathcal{Y}_i - \bar{y})(\mathcal{Y}_i - \bar{y})^\top
\tag{30}
$$

### 3.3 UKF Tuning Parameters

The three tuning parameters $\alpha, \beta, \kappa$ control the shape and spread of
the sigma-point distribution.

| Parameter | Typical value | Role |
|:----------|:-------------|:-----|
| $\alpha$ | $10^{-3}$ to $1$ | Spread of sigma points around mean; small $\alpha$ gives tightly clustered points |
| $\beta$ | $2$ (optimal for Gaussian) | Incorporates prior knowledge of the distribution type; $\beta = 2$ minimises fourth-order error for Gaussians |
| $\kappa$ | $0$ or $3 - m$ | Secondary scaling; $\kappa = 0$ is the most common choice |

!!! tip "Default Settings in kalmanbox"
    `kalmanbox` defaults to $\alpha = 10^{-3}$, $\beta = 2$, $\kappa = 0$, which
    gives $\lambda \approx -m + \alpha^2 m \approx -m$ and keeps sigma points
    near the mean. For heavily non-Gaussian tails, increase $\alpha$ toward $1$.

### 3.4 Complete UKF Algorithm

**Initialisation:** $a_{0|0} = \mu_0$, $P_{0|0} = P_0$.

---

**For $t = 1, 2, \ldots, T$:**

**Predict step — generate and propagate sigma points through $f$:**

$$
\left\{ \mathcal{X}_{t-1}^{(i)} \right\} = \operatorname{SigmaPoints}(a_{t-1|t-1},\, P_{t-1|t-1})
\tag{31}
$$

$$
\mathcal{X}_{t|t-1}^{(i)} = f\!\left(\mathcal{X}_{t-1}^{(i)}\right), \quad i = 0, \ldots, 2m
\tag{32}
$$

**Predicted mean and covariance:**

$$
a_{t|t-1} = \sum_{i=0}^{2m} W_i^m\, \mathcal{X}_{t|t-1}^{(i)}
\tag{33}
$$

$$
P_{t|t-1} = \sum_{i=0}^{2m} W_i^c\,
  \left(\mathcal{X}_{t|t-1}^{(i)} - a_{t|t-1}\right)
  \left(\mathcal{X}_{t|t-1}^{(i)} - a_{t|t-1}\right)^\top
  + R_{t-1} Q_{t-1} R_{t-1}^\top
\tag{34}
$$

**Update step — generate new sigma points from predicted distribution, propagate through $h$:**

$$
\left\{ \mathcal{X}_{t}^{(i)} \right\} = \operatorname{SigmaPoints}(a_{t|t-1},\, P_{t|t-1})
\tag{35}
$$

$$
\mathcal{Y}_{t}^{(i)} = h\!\left(\mathcal{X}_{t}^{(i)}\right), \quad i = 0, \ldots, 2m
\tag{36}
$$

**Predicted observation mean and innovation covariance:**

$$
\hat{y}_t = \sum_{i=0}^{2m} W_i^m\, \mathcal{Y}_{t}^{(i)}
\tag{37}
$$

$$
S_t = \sum_{i=0}^{2m} W_i^c\,
  \left(\mathcal{Y}_{t}^{(i)} - \hat{y}_t\right)
  \left(\mathcal{Y}_{t}^{(i)} - \hat{y}_t\right)^\top
  + H_t
\tag{38}
$$

**Cross-covariance between predicted state and observation:**

$$
C_t = \sum_{i=0}^{2m} W_i^c\,
  \left(\mathcal{X}_{t}^{(i)} - a_{t|t-1}\right)
  \left(\mathcal{Y}_{t}^{(i)} - \hat{y}_t\right)^\top
\tag{39}
$$

**Kalman gain:**

$$
K_t = C_t\, S_t^{-1}
\tag{40}
$$

**Updated state mean and covariance:**

$$
a_{t|t} = a_{t|t-1} + K_t\,(y_t - \hat{y}_t)
\tag{41}
$$

$$
P_{t|t} = P_{t|t-1} - K_t\, S_t\, K_t^\top
\tag{42}
$$

---

### 3.5 Accuracy Comparison: UKF vs. EKF

The UKF achieves higher accuracy than the EKF by capturing more of the nonlinear
transformation. For a Gaussian random variable $x \sim \mathcal{N}(\mu, P)$ passed
through a smooth nonlinear function $g$:

- **EKF:** Mean accurate to **first order**, covariance accurate to **first order** in
  the Taylor expansion of $g$.
- **UKF:** Mean accurate to **third order**, covariance accurate to **third order** for
  Gaussian inputs, with the $\beta$ parameter providing additional fourth-order
  correction for the covariance.

This two-order improvement can be decisive for moderately nonlinear problems, such as
the Lorenz system, reentry vehicle tracking, and stochastic volatility models.

!!! note "Accuracy Statement (Wan & van der Merwe, 2000)"
    The UKF mean approximation error is $O(h^4)$ where $h$ is the step size of the
    quadrature rule induced by the sigma points, compared to $O(h^2)$ for the EKF.
    For Gaussian distributions this translates to third-order Taylor accuracy.

### 3.6 Computational Cost

Each UKF iteration requires:
- **Sigma point generation:** $O(m^3)$ for the Cholesky decomposition of $P_{t|t-1}$.
- **Propagation:** $2m+1$ evaluations of $f$ and $h$ — no Jacobians needed.
- **Moment reconstruction:** $O(m^2)$ matrix operations.
- **Gain computation:** $O(p^3)$ inversion plus $O(m^2 p)$ products.

The dominant cost is $O(m^3)$, the same asymptotic order as the EKF. However, the
constant factor is larger because $2m+1$ function evaluations are required versus one
Jacobian computation (which may be cheaper for sparse structures). For problems where
Jacobians are expensive or unavailable, the UKF is substantially preferable.

---

## 4. Square-Root UKF

### 4.1 Motivation and Formulation

The standard UKF recomputes the Cholesky factor $\sqrt{P_{t|t}}$ at each step, which
costs $O(m^3)$ and can introduce numerical errors when $P_{t|t}$ is near-singular.
The **Square-Root UKF** (SR-UKF, van der Merwe & Wan, 2001) propagates the Cholesky
factor $S_{t|t} = \mathrm{chol}(P_{t|t})$ directly, avoiding repeated decomposition
and guaranteeing positive semi-definiteness throughout.

The key identity used is the **QR decomposition rank-1 update**. Let
$S_{t|t-1} = \mathrm{chol}(P_{t|t-1})$; then:

$$
S_{t|t-1} = \mathrm{qr}\!\left(
  \left[\sqrt{W_1^c}\,(\mathcal{X}^{(1:2m)}_{t|t-1} - a_{t|t-1})\;\Big|\;
  \sqrt{Q_{t-1}}\right]^\top
\right)
\tag{43}
$$

followed by a rank-1 Cholesky update for the $i=0$ sigma point (using
`cholupdate`).

The **update step** covariance correction is similarly handled via a Cholesky
downdate:

$$
S_{t|t} = \mathrm{choldowndate}\!\left(S_{t|t-1},\; K_t \sqrt{S_t}\right)
\tag{44}
$$

where $S_t$ here is the Cholesky factor of the innovation covariance.

!!! tip "Connection to Square-Root Kalman Filter"
    The SR-UKF is the nonlinear analogue of the square-root Kalman filter described
    in [square-root.md](../user-guide/filters/square-root.md). Both propagate
    Cholesky factors rather than full covariance matrices, achieving $O(m^3 / 6)$
    savings in the triangular factor propagation and improved conditioning. The
    SR-UKF is the recommended implementation for problems with $m \gtrsim 50$ or
    near-degenerate noise covariances.

---

## 5. Ensemble Kalman Filter (EnKF)

### 5.1 Monte Carlo Derivation

The **Ensemble Kalman Filter** (Evensen, 1994) represents the filtering distribution
$p(\alpha_t \mid \mathcal{Y}_t)$ by a finite ensemble of $N$ state vectors:

$$
\left\{\alpha_t^{(i)}\right\}_{i=1}^{N} \approx p(\alpha_t \mid \mathcal{Y}_t)
\tag{45}
$$

Each ensemble member is an independent draw that evolves according to the model
dynamics. The ensemble mean and covariance approximate the posterior moments:

$$
\bar\alpha_t = \frac{1}{N} \sum_{i=1}^{N} \alpha_t^{(i)}
\tag{46}
$$

$$
P_t = \frac{1}{N-1} \sum_{i=1}^{N}
  (\alpha_t^{(i)} - \bar\alpha_t)(\alpha_t^{(i)} - \bar\alpha_t)^\top
\tag{47}
$$

Note the $N-1$ denominator (Bessel's correction) for an unbiased sample covariance.
Define the **anomaly matrix** $A_t \in \mathbb{R}^{m \times N}$ whose columns are
the centred ensemble members:

$$
A_t = \frac{1}{\sqrt{N-1}}
  \left[\alpha_t^{(1)} - \bar\alpha_t,\; \ldots,\; \alpha_t^{(N)} - \bar\alpha_t\right]
\tag{48}
$$

Then:

$$
P_t = A_t A_t^\top
\tag{49}
$$

This factored representation is central to the EnKF's computational efficiency: the
full $m \times m$ covariance matrix need never be formed explicitly.

### 5.2 Stochastic EnKF (Burgers, van Leeuwen & Evensen, 1998)

The original EnKF of Evensen (1994) had a systematic covariance underestimation bias.
Burgers, van Leeuwen & Evensen (1998) showed that this is corrected by **perturbing
the observations** with independent noise draws:

$$
y_t^{(i)} = y_t + \varepsilon_t^{(i)}, \quad \varepsilon_t^{(i)} \sim \mathcal{N}(0,\, H_t),
\quad i = 1, \ldots, N
\tag{50}
$$

**Forecast step:** propagate each ensemble member through the model:

$$
\alpha_t^{(i,f)} = f\!\left(\alpha_{t-1}^{(i,a)}\right) + R_{t-1}\, \eta_{t-1}^{(i)},
\quad \eta_{t-1}^{(i)} \sim \mathcal{N}(0,\, Q_{t-1})
\tag{51}
$$

**Analysis step:** update each member using the perturbed observation:

$$
\alpha_t^{(i,a)} = \alpha_t^{(i,f)} + K_t^{\mathrm{ens}}\,
  \left(y_t^{(i)} - H_t\, \alpha_t^{(i,f)}\right)
\tag{52}
$$

where the **ensemble Kalman gain** is:

$$
K_t^{\mathrm{ens}} = P_t^f H_t^\top \left(H_t P_t^f H_t^\top + H_t\right)^{-1}
\tag{53}
$$

with $P_t^f = A_t^f (A_t^f)^\top$ the **forecast covariance** computed from the
forecast ensemble anomalies:

$$
P_t^f = \frac{1}{N-1} \sum_{i=1}^{N}
  (\alpha_t^{(i,f)} - \bar\alpha_t^f)(\alpha_t^{(i,f)} - \bar\alpha_t^f)^\top
\tag{54}
$$

The stochastic EnKF is unbiased in expectation (Burgers et al., 1998), but the
observation perturbations introduce additional Monte Carlo sampling variance of
order $O(1/N)$.

!!! warning "Observation Perturbation Necessity"
    Skipping the perturbation in equation (50) — using the same $y_t$ for all
    members — leads to a systematic underestimate of the analysis covariance.
    This was the deficiency of the original 1994 EnKF that Burgers et al. (1998)
    corrected. The deterministic square-root variants below avoid this issue without
    adding sampling noise.

### 5.3 Deterministic EnKF (Ensemble Square-Root Filter)

**Ensemble square-root filters** (EnSRF, ETKF, LETKF) avoid observation perturbations
by computing an exact, deterministic analysis ensemble that has the correct second
moments. The idea is to transform the forecast anomaly matrix $A_t^f$ directly.

In the **Ensemble Transform Kalman Filter** (Bishop et al., 2001), the analysis
anomalies are:

$$
A_t^a = A_t^f\, \mathbf{T}_t
\tag{55}
$$

where the transform matrix $\mathbf{T}_t \in \mathbb{R}^{N \times N}$ satisfies:

$$
A_t^a (A_t^a)^\top = P_t^a = (I - K_t^{\mathrm{ens}} H_t)\, P_t^f
\tag{56}
$$

A symmetric square root of the right-hand side is used for $\mathbf{T}_t$, computed
via eigendecomposition of $I + (A_t^f)^\top H_t^\top H_t^{-1} H_t A_t^f$.

The key properties of the deterministic EnKF are:
- Exactly reproduces the Kalman update covariance for finite $N$ (in the linear case).
- No additional sampling noise from observation perturbations.
- Computationally more expensive per step than the stochastic EnKF but typically
  requires fewer ensemble members.

### 5.4 Inflation and Localization

Two essential practical techniques address known deficiencies of the ensemble
approximation at finite $N$:

**Covariance inflation** counteracts the tendency of ensemble covariances to
collapse toward zero over time (filter divergence). The simplest multiplicative
inflation replaces:

$$
P_t^f \leftarrow (1 + \delta)\, P_t^f, \quad \delta > 0
\tag{57}
$$

with $\delta$ typically chosen in the range $0.01$–$0.1$. Adaptive inflation
schemes estimate $\delta$ from the data.

**Covariance localization** suppresses spurious long-range correlations that arise
from finite-ensemble sampling. A localization function $\rho(d_{ij}) \in [0,1]$
(e.g., the Gaspari-Cohn compactly-supported correlation function) is applied element-wise:

$$
[P_t^f]_{ij}^{\mathrm{loc}} = \rho(d_{ij}) \cdot [P_t^f]_{ij}
\tag{58}
$$

where $d_{ij}$ is the physical distance between state components $i$ and $j$.
Localization allows practical EnKF implementations with $N \ll m$ — for instance,
$N = 40$ members for an atmospheric model with $m = 10^7$ state variables.

### 5.5 Convergence as $N \to \infty$

For linear Gaussian models, the stochastic EnKF is **consistent**: as $N \to \infty$,
the ensemble mean and covariance converge to the Kalman filter mean and covariance
almost surely. Evensen (2003) shows that the convergence rate is:

$$
\mathbb{E}\!\left[\left\| a_{t|t}^{\mathrm{EnKF}} - a_{t|t}^{\mathrm{KF}} \right\|^2\right]
= O\!\left(\frac{1}{N}\right)
\tag{59}
$$

For nonlinear models, the EnKF converges to the optimal filter only if the true
posterior is Gaussian — the same condition that makes the Kalman filter optimal.
When the posterior is significantly non-Gaussian, particle filters are preferable
(see Section 6).

### 5.6 Computational Cost

The dominant operations in the EnKF are:

| Operation | Cost |
|:----------|:-----|
| Forecast: propagate $N$ ensemble members | $O(N \cdot \text{cost}(f))$ |
| Compute ensemble covariance $P^f$ | $O(Nm^2)$ or $O(N^2 m)$ via anomaly matrix |
| Innovation covariance $S = H P^f H^\top + H$ | $O(N p^2)$ or $O(p^2 N)$ |
| Gain $K = P^f H^\top S^{-1}$ | $O(mp^2)$ or $O(mNp)$ with low-rank |
| Analysis update of $N$ members | $O(Nm)$ |

For high-dimensional systems ($m \gg p$, $N \ll m$), the dominant cost is
$O(Nm^2)$, which is far cheaper than the $O(m^3)$ cost of the standard Kalman
filter when $N \ll m$. This makes the EnKF the method of choice for large-scale
geophysical data assimilation (atmospheric, oceanographic, and subsurface models).

---

## 6. Particle Filter Connection

### 6.1 Sequential Importance Sampling

Particle filters (also called sequential Monte Carlo) represent the full posterior
$p(\alpha_t \mid \mathcal{Y}_t)$ by a weighted empirical measure:

$$
p(\alpha_t \mid \mathcal{Y}_t) \approx \sum_{i=1}^{N} w_t^{(i)}\,
  \delta\!\left(\alpha_t - \alpha_t^{(i)}\right)
\tag{60}
$$

where $\{(\alpha_t^{(i)}, w_t^{(i)})\}_{i=1}^N$ are **particles** with
associated **importance weights** that satisfy $\sum_i w_t^{(i)} = 1$.

The weights are updated sequentially by the likelihood of the new observation:

$$
\tilde{w}_t^{(i)} \propto w_{t-1}^{(i)}\;
  \frac{p(y_t \mid \alpha_t^{(i)})\; p(\alpha_t^{(i)} \mid \alpha_{t-1}^{(i)})}
       {q(\alpha_t^{(i)} \mid \alpha_{t-1}^{(i)}, y_t)}
\tag{61}
$$

where $q(\cdot)$ is the **importance (proposal) distribution**. The simplest
choice $q = p(\alpha_t^{(i)} \mid \alpha_{t-1}^{(i)})$ gives the **bootstrap
particle filter** (Gordon, Salmond & Smith, 1993).

### 6.2 Sequential Importance Resampling (SIR)

Without resampling, all weight concentrates on a single particle (weight
degeneracy) within a few steps. The **SIR** algorithm adds a resampling step:

1. **Propagate:** draw $\alpha_t^{(i)} \sim p(\alpha_t \mid \alpha_{t-1}^{(i)})$.
2. **Weight:** compute $w_t^{(i)} \propto p(y_t \mid \alpha_t^{(i)})$.
3. **Resample:** draw $N$ particles with replacement from $\{(\alpha_t^{(i)},
   w_t^{(i)})\}$, setting weights to $1/N$ after resampling.

The effective sample size $N_{\mathrm{eff}} = 1 / \sum_i (w_t^{(i)})^2$ monitors
degeneracy; resampling is triggered when $N_{\mathrm{eff}}$ falls below a threshold
(commonly $N/2$).

### 6.3 EnKF vs. Particle Filter

| Property | EnKF | Particle Filter |
|:---------|:-----|:----------------|
| Posterior representation | Gaussian approximation | Fully nonparametric |
| Optimal for linear Gaussian? | Yes (as $N \to \infty$) | Yes (as $N \to \infty$) |
| Handles multimodal posteriors? | No | Yes |
| Curse of dimensionality | Weak (via localization) | Severe ($N \sim \exp(m)$) |
| Typical ensemble size $N$ | $10$–$10^3$ | $10^3$–$10^6$ |
| Covariance approximation | Low-rank ($N-1$ rank) | Nonparametric |
| Resampling needed? | No | Yes (to avoid degeneracy) |

The fundamental distinction is that the **EnKF imposes a Gaussian approximation**
at each analysis step, while the **particle filter is fully nonparametric**. For
nearly Gaussian, high-dimensional systems (geophysical data assimilation), the
EnKF's Gaussian assumption is benign and its scalability is decisive. For
low-dimensional, strongly non-Gaussian problems (robot localization, SLAM,
multi-target tracking), the particle filter is superior.

!!! note "Dimension Curse for Particle Filters"
    In dimension $m$, a particle filter requires $N = O(\exp(m))$ particles for a
    fixed approximation accuracy (Snyder et al., 2008). This makes pure particle
    filters impractical for $m \gtrsim 20$ without importance function engineering
    (e.g., EKF-proposal or UKF-proposal particle filters). The EnKF's localization
    technique effectively reduces the local problem dimension to the localization
    radius.

### 6.4 Connection to the particlefilterbox Ecosystem

`kalmanbox` focuses on Gaussian filter approximations (EKF, UKF, EnKF). For
fully nonparametric sequential Monte Carlo methods, the companion package
`particlefilterbox` provides:

- Bootstrap particle filter (SIR)
- Auxiliary particle filter (Pitt & Shephard, 1999)
- Rao-Blackwellized particle filter for conditionally linear substructures
- Interacting Multiple Model (IMM) filter

The `kalmanbox.filters.EnsembleKalmanFilter` ensemble can be used as a
computationally efficient alternative to particle filters when $m \gg 20$ and
the posterior is approximately Gaussian.

---

## 7. Theoretical Comparison

### 7.1 Filter Comparison Table

| Property | EKF | UKF | EnKF | Particle Filter |
|:---------|:----|:----|:-----|:----------------|
| **Posterior form** | Gaussian | Gaussian | Gaussian | Nonparametric |
| **Taylor accuracy** | 1st order | 3rd order | N/A (Monte Carlo) | Exact (as $N\to\infty$) |
| **Jacobians required?** | Yes | No | No | No |
| **Ensemble size** | 1 (implicit) | $2m+1$ | $N$ (user-set) | $N$ (large) |
| **Per-step cost** | $O(m^3)$ | $O(m^3)$ | $O(Nm^2)$ | $O(Np^m)$ |
| **Scalability ($m$ large)** | Poor ($m^3$) | Poor ($m^3$) | Good (localization) | Very poor |
| **Nonlinear robustness** | Moderate | Good | Moderate (Gaussian) | Excellent |
| **Multimodal posteriors** | No | No | No | Yes |
| **Geophysical use** | Rare | Rare | Standard | Rare |
| **Robotics / SLAM use** | Common | Common | Rare | Common |
| **Reference** | Gelb (1974) | Julier & Uhlmann (1997) | Evensen (2003) | Gordon et al. (1993) |

### 7.2 Stability Analysis for the EKF

A key concern for the EKF is **filter stability**: does the estimation error
remain bounded over time? Anderson & Moore (1979) establish stability conditions
for the linear Kalman filter under observability and controllability. For the EKF
the situation is more delicate.

Let $e_t = \alpha_t - a_{t|t}$ be the estimation error. The EKF error satisfies:

$$
e_{t+1} \approx (F_t - K_t H_t)\, e_t + \text{noise} + O(\|e_t\|^2)
\tag{62}
$$

The EKF is **locally asymptotically stable** if the spectral radius of the
true (not estimated) Jacobian satisfies:

$$
\rho(F_t - K_t H_t) < 1 \quad \forall\, t
\tag{63}
$$

under the true Kalman gain computed from the true Jacobian. In practice this
condition is checked numerically by monitoring the maximum eigenvalue of
$P_{t|t}$: if $P_{t|t}$ remains bounded, the filter is stable.

!!! warning "EKF Divergence"
    The EKF can diverge when linearization errors compound over time, driving
    $P_{t|t}$ to collapse (overconfidence) while the true error $e_t$ grows.
    This is detected by monitoring the **normalized innovation squared** (NIS):
    $\mathrm{NIS}_t = v_t^\top S_t^{-1} v_t \sim \chi^2_p$ under correct
    specification. Persistent $\mathrm{NIS}_t > \chi^2_{p,0.95}$ signals
    filter inconsistency.

### 7.3 Consistency: The NEES Test

The **Normalized Estimation Error Squared** (NEES) is the standard diagnostic
for filter consistency:

$$
\mathrm{NEES}_t = e_t^\top P_{t|t}^{-1} e_t \sim \chi^2_m
\tag{64}
$$

Under a consistent filter, $\mathrm{NEES}_t$ follows a chi-squared distribution
with $m$ degrees of freedom, so $\mathbb{E}[\mathrm{NEES}_t] = m$.

In Monte Carlo simulations with $M$ independent runs, the time-averaged NEES:

$$
\overline{\mathrm{NEES}}_t = \frac{1}{M} \sum_{j=1}^{M} \mathrm{NEES}_t^{(j)}
\sim \chi^2_{mM} / M
\tag{65}
$$

provides a $(1-\alpha)$ confidence interval for filter consistency. The filter
is declared **inconsistent** (overconfident) if
$\overline{\mathrm{NEES}}_t > \chi^2_{mM, 1-\alpha/2} / M$ and
**inconsistent** (underconfident) if
$\overline{\mathrm{NEES}}_t < \chi^2_{mM, \alpha/2} / M$.

The analogous test using innovations is the **NIS test**, which applies when
the true state is not available (online operation).

For practical filter comparison, see [../user-guide/filters/comparison.md](../user-guide/filters/comparison.md).

---

## 8. Applications in kalmanbox

### 8.1 EKF for Stochastic Volatility

The **stochastic volatility (SV) model** expresses the log-variance $h_t =
\log \sigma_t^2$ as a latent AR(1) process. The observation $y_t$ is a return,
conditionally Gaussian with variance $\exp(h_t)$:

$$
y_t = \exp(h_t / 2)\, \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)
$$

$$
h_{t+1} = \mu + \phi(h_t - \mu) + \sigma_\eta \eta_t, \quad \eta_t \sim \mathcal{N}(0,1)
$$

The observation function $h(\alpha_t) = 0$ with noise $\exp(h_t)$ is multiplicative;
the EKF handles this by squaring and log-transforming:

```python
import numpy as np
from kalmanbox.filters import ExtendedKalmanFilter

# Stochastic volatility: state = log-variance h_t
# y_t = exp(h_t/2) * eps_t  =>  y_t^2 ~ LogNormal(h_t, sigma_eps^2)
# Transformed observation: log(y_t^2) = h_t + log(eps_t^2)
# log(eps_t^2) ~ log(chi^2_1): mean = -1.2704, var = 4.9348

def sv_obs_fn(state: np.ndarray) -> np.ndarray:
    """Observation function: identity on log-variance."""
    return state  # after transforming y_t -> log(y_t^2)

def sv_obs_jac(state: np.ndarray) -> np.ndarray:
    """Jacobian of observation function."""
    return np.array([[1.0]])

def sv_trans_fn(state: np.ndarray, phi: float, mu: float) -> np.ndarray:
    """AR(1) transition for log-variance."""
    return np.array([mu + phi * (state[0] - mu)])

def sv_trans_jac(state: np.ndarray, phi: float) -> np.ndarray:
    """Jacobian of transition function."""
    return np.array([[phi]])

# Model parameters
phi = 0.97      # persistence
mu = -0.5       # unconditional log-variance
sigma_eta = 0.2 # state noise std

returns = np.random.randn(500)  # simulated returns
log_y2 = np.log(returns**2 + 1e-8)  # transformed observations

ekf = ExtendedKalmanFilter(
    state_dim=1,
    obs_dim=1,
    transition_fn=lambda s: sv_trans_fn(s, phi, mu),
    transition_jac=lambda s: sv_trans_jac(s, phi),
    observation_fn=sv_obs_fn,
    observation_jac=sv_obs_jac,
    transition_cov=np.array([[sigma_eta**2]]),
    observation_cov=np.array([[4.9348]]),   # variance of log(chi^2_1)
    observation_offset=np.array([-1.2704]), # mean of log(chi^2_1)
)

results = ekf.filter(log_y2[:, None])
log_vol_filtered = results.state_mean[:, 0]
print(f"Filtered log-variance range: [{log_vol_filtered.min():.2f}, "
      f"{log_vol_filtered.max():.2f}]")
```

### 8.2 UKF for a Nonlinear Pendulum

The pendulum ODE $\ddot\theta + (g/L)\sin\theta = 0$ is nonlinear in the angle
$\theta$. Discretized with the Euler-Maruyama scheme with state
$\alpha_t = (\theta_t, \dot\theta_t)^\top$:

```python
import numpy as np
from kalmanbox.filters import UnscentedKalmanFilter

# Pendulum parameters
g = 9.81  # m/s^2
L = 1.0   # pendulum length in meters
dt = 0.05 # time step in seconds

def pendulum_f(state: np.ndarray) -> np.ndarray:
    """Nonlinear pendulum transition: Euler-Maruyama discretization."""
    theta, theta_dot = state
    theta_new = theta + dt * theta_dot
    theta_dot_new = theta_dot - dt * (g / L) * np.sin(theta)
    return np.array([theta_new, theta_dot_new])

def pendulum_h(state: np.ndarray) -> np.ndarray:
    """Observe angle only."""
    return np.array([state[0]])

# Noise covariances
Q = np.diag([1e-4, 1e-3])  # process noise (angle, angular velocity)
H_obs = np.array([[0.01]])  # observation noise (angle sensor)

# Initial state: theta=pi/4, theta_dot=0, with uncertainty
a0 = np.array([np.pi / 4, 0.0])
P0 = np.diag([0.1, 0.1])

ukf = UnscentedKalmanFilter(
    state_dim=2,
    obs_dim=1,
    transition_fn=pendulum_f,
    observation_fn=pendulum_h,
    transition_cov=Q,
    observation_cov=H_obs,
    alpha=1e-3,
    beta=2.0,
    kappa=0.0,
)

# Simulate and filter
T = 200
true_state = np.zeros((T, 2))
true_state[0] = a0
np.random.seed(42)
for t in range(1, T):
    true_state[t] = pendulum_f(true_state[t - 1])
    true_state[t] += np.random.multivariate_normal(np.zeros(2), Q)

obs = true_state[:, 0:1] + np.random.normal(0, np.sqrt(H_obs[0, 0]), (T, 1))

results = ukf.filter(obs, initial_state=a0, initial_cov=P0)

rmse_angle = np.sqrt(np.mean((results.state_mean[:, 0] - true_state[:, 0])**2))
print(f"UKF angle RMSE: {rmse_angle:.4f} rad")
```

### 8.3 EnKF for High-Dimensional Data Assimilation

The Lorenz-96 model is a standard benchmark for data assimilation, with $m = 40$
coupled ODEs that simulate atmospheric wave dynamics:

$$
\frac{d\alpha_i}{dt} = (\alpha_{i+1} - \alpha_{i-2})\alpha_{i-1} - \alpha_i + F,
\quad i = 1, \ldots, m
$$

```python
import numpy as np
from kalmanbox.filters import EnsembleKalmanFilter

# Lorenz-96 model
m = 40    # state dimension
F = 8.0   # forcing constant (chaotic regime)
dt = 0.05 # time step

def lorenz96_rhs(x: np.ndarray, F: float = 8.0) -> np.ndarray:
    """Lorenz-96 right-hand side."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def lorenz96_step(x: np.ndarray) -> np.ndarray:
    """4th-order Runge-Kutta step."""
    k1 = lorenz96_rhs(x)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2)
    k4 = lorenz96_rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Observe every other variable (20 observations)
p = 20
H_mat = np.zeros((p, m))
for i in range(p):
    H_mat[i, 2 * i] = 1.0

obs_noise_std = 1.0
H_obs = obs_noise_std**2 * np.eye(p)
Q_proc = 0.01 * np.eye(m)

# EnKF configuration
N = 50  # ensemble size

enkf = EnsembleKalmanFilter(
    state_dim=m,
    obs_dim=p,
    transition_fn=lorenz96_step,
    observation_matrix=H_mat,
    transition_cov=Q_proc,
    observation_cov=H_obs,
    ensemble_size=N,
    inflation_factor=1.05,      # 5% multiplicative inflation
    localization_radius=5.0,    # Gaspari-Cohn localization
)

# Generate synthetic observations
T = 500
true_state = np.zeros((T, m))
true_state[0] = F + 0.01 * np.random.randn(m)
for t in range(1, T):
    true_state[t] = lorenz96_step(true_state[t - 1])
    true_state[t] += np.random.multivariate_normal(np.zeros(m), Q_proc)

observations = true_state @ H_mat.T
observations += np.random.multivariate_normal(np.zeros(p), H_obs, size=T)

results = enkf.filter(observations)

# Compute time-mean RMSE after spinup
spinup = 100
rmse = np.sqrt(np.mean(
    (results.state_mean[spinup:] - true_state[spinup:])**2
))
print(f"EnKF mean RMSE (Lorenz-96, N={N}): {rmse:.4f}")
print(f"Effective rank of ensemble: {N - 1}")
```

---

## References

The following references are cited in this page. For the full bibliography see
[references.md](references.md).

**Foundational texts:**

- **Anderson, B.D.O. & Moore, J.B. (1979).** *Optimal Filtering.* Prentice-Hall,
  Englewood Cliffs, NJ. The definitive treatment of Kalman filter stability and
  optimality theory.
- **Gelb, A. (ed.) (1974).** *Applied Optimal Estimation.* MIT Press, Cambridge,
  MA. The practical engineering reference for EKF design, including stability
  analysis and worked examples.

**Extended Kalman Filter:**

- **Jazwinski, A.H. (1970).** *Stochastic Processes and Filtering Theory.*
  Academic Press, New York. Rigorous treatment of continuous-discrete and
  continuous-continuous EKF.

**Unscented Kalman Filter:**

- **Julier, S.J. & Uhlmann, J.K. (1997).** *A New Extension of the Kalman Filter
  to Nonlinear Systems.* Proceedings of SPIE 3068, Signal Processing, Sensor
  Fusion, and Target Recognition VI, 182–193. Original UKF paper; introduces the
  unscented transform.
- **Julier, S.J. & Uhlmann, J.K. (2004).** Unscented Filtering and Nonlinear
  Estimation. *Proceedings of the IEEE*, 92(3), 401–422. Comprehensive review and
  theory.
- **Wan, E.A. & van der Merwe, R. (2000).** The Unscented Kalman Filter for
  Nonlinear Estimation. In *Proceedings of the IEEE 2000 Adaptive Systems for
  Signal Processing, Communications, and Control Symposium*, 153–158. Key accuracy
  analysis and comparison with EKF.
- **van der Merwe, R. & Wan, E.A. (2001).** The Square-Root Unscented Kalman
  Filter for State and Parameter-Estimation. In *ICASSP 2001*, 3461–3464.
  SR-UKF formulation via QR and Cholesky updates.

**Ensemble Kalman Filter:**

- **Evensen, G. (1994).** Sequential data assimilation with a nonlinear
  quasi-geostrophic model using Monte Carlo methods to forecast error statistics.
  *Journal of Geophysical Research: Oceans*, 99(C5), 10143–10162. Original EnKF
  paper.
- **Burgers, G., van Leeuwen, P.J. & Evensen, G. (1998).** Analysis scheme in the
  Ensemble Kalman Filter. *Monthly Weather Review*, 126(6), 1719–1724.
  Corrects covariance underestimation via perturbed observations.
- **Evensen, G. (2003).** The Ensemble Kalman Filter: theoretical formulation and
  practical implementation. *Ocean Dynamics*, 53(4), 343–367. Comprehensive
  formulation, convergence, and practical guide.

**Particle Filters:**

- **Gordon, N.J., Salmond, D.J. & Smith, A.F.M. (1993).** Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2),
  107–113. Bootstrap particle filter.
- **Snyder, C., Bengtsson, T., Bickel, P. & Anderson, J. (2008).** Obstacles to
  high-dimensional particle filtering. *Monthly Weather Review*, 136(12),
  4629–4640. Dimension curse analysis.
