# Structural Time Series: Theory

This page develops the mathematical foundations of **structural (unobserved-components)
time series models** as implemented in `kalmanbox`.  
Every model is cast in the Durbin & Koopman (2012) state-space convention:

$$
\begin{aligned}
y_t &= Z_t \alpha_t + \varepsilon_t, &&\varepsilon_t \sim N(0, H_t)
    &&\text{(observation equation)}\\
\alpha_{t+1} &= T_t \alpha_t + R_t \eta_t, &&\eta_t \sim N(0, Q_t)
    &&\text{(state transition equation)}
\end{aligned}
$$

See [`state-space-theory.md`](state-space-theory.md) for the full general formulation,
[`kalman-theory.md`](kalman-filter-derivation.md) for the filter derivation, and
[`mle-theory.md`](likelihood.md) for estimation.

---

## 1. Classical Decomposition of Time Series

### 1.1 The additive unobserved-components representation

The fundamental premise of structural time series analysis is that any observed
series $\{y_t\}$ can be expressed as the **additive superposition of latent
processes**, each of which captures a qualitatively distinct mode of variation:

$$
\boxed{y_t = \mu_t + \gamma_t + \psi_t + \varepsilon_t}
$$

| Symbol | Name | Economic interpretation |
|:-------|:-----|:------------------------|
| $\mu_t$ | **Trend** | Long-run level and growth |
| $\gamma_t$ | **Seasonal** | Periodic within-year pattern |
| $\psi_t$ | **Cycle** | Medium-run business/economic cycle |
| $\varepsilon_t$ | **Irregular** | Short-run idiosyncratic noise |

!!! definition "Unobserved Components"
    Each component is **unobserved** (latent). The series $y_t$ alone is observed,
    and the Kalman smoother is used to recover optimal estimates
    $\hat\mu_t$, $\hat\gamma_t$, $\hat\psi_t$ conditional on the full sample.

### 1.2 Stochastic vs. deterministic decomposition

The unobserved-components (UC) approach is philosophically distinct from purely
deterministic or regression-based decompositions:

- **Deterministic methods** (e.g., X-11, Census X-13-ARIMA-SEATS, classical
  moving-average filters) treat seasonal and trend as fixed periodic functions
  or polynomial functions of time. They are computationally convenient but
  cannot adapt to structural breaks in seasonality or slope.

- **UC / structural methods** (Harvey, 1989; Durbin & Koopman, 2012) allow
  *each component to evolve stochastically* according to its own innovation.
  The variances $\sigma^2_\xi, \sigma^2_\zeta, \sigma^2_\omega$ govern how
  much each component is allowed to change. Setting any variance to zero
  recovers a deterministic special case.

### 1.3 Multiplicative form

When $y_t > 0$ throughout and the seasonal amplitude grows proportionally with
the trend level, the **multiplicative decomposition**

$$
y_t = \mu_t \cdot \gamma_t \cdot \psi_t \cdot \varepsilon_t
$$

is more appropriate. Taking logarithms reduces this to the additive form
applied to $\log y_t$, preserving all the state-space machinery developed below.

!!! note "Reference"
    The foundational treatment of this framework is Harvey (1989), Chapter 1,
    which introduces the philosophy of "modelling in terms of components that
    have a direct interpretation" rather than reduced-form ARIMA representations.

---

## 2. Local Level Model

### 2.1 Model equations

The simplest non-trivial structural model is the **Local Level Model (LLM)**,
also known as the *random walk plus noise* model:

$$
\begin{aligned}
y_t &= \mu_t + \varepsilon_t, &&\varepsilon_t \sim N(0, \sigma^2_\varepsilon)\\
\mu_{t+1} &= \mu_t + \eta_t, &&\eta_t \sim N(0, \sigma^2_\eta)
\end{aligned}
$$

where $\varepsilon_t$ and $\eta_s$ are mutually independent for all $t, s$.

### 2.2 State-space matrices

The LLM is already in the Durbin-Koopman state-space form with scalar state
$\alpha_t = \mu_t$:

$$
Z = [1], \quad T = [1], \quad R = [1], \quad H = [\sigma^2_\varepsilon],
\quad Q = [\sigma^2_\eta]
$$

### 2.3 Signal-to-noise ratio and steady-state Kalman gain

Define the **signal-to-noise ratio**

$$
q = \frac{\sigma^2_\eta}{\sigma^2_\varepsilon}.
$$

The Kalman filter gain at step $t$ is $K_t = T_t P_{t|t-1} Z_t' F_t^{-1}$,
which simplifies to $K_t = P_{t|t-1} / F_t$ with $F_t = P_{t|t-1} + \sigma^2_\varepsilon$.
In steady state, $P_{t|t-1} \to P_\infty$ satisfies the **algebraic Riccati equation**:

$$
P_\infty = P_\infty - \frac{P_\infty^2}{P_\infty + \sigma^2_\varepsilon} + \sigma^2_\eta.
$$

Dividing through by $\sigma^2_\varepsilon$ and writing $\pi = P_\infty / \sigma^2_\varepsilon$:

$$
\pi = \pi - \frac{\pi^2}{\pi + 1} + q
\quad\Longrightarrow\quad
\frac{\pi^2}{\pi + 1} = q
\quad\Longrightarrow\quad
\pi^2 = q(\pi + 1)
\quad\Longrightarrow\quad
\pi^2 - q\pi - q = 0.
$$

Applying the quadratic formula and discarding the negative root:

$$
\boxed{k_\infty = \frac{P_\infty}{P_\infty + \sigma^2_\varepsilon}
      = \frac{-q + \sqrt{q^2 + 4q}}{2q + 2 - q}
      = \frac{-1 + \sqrt{1 + 4q}}{2\left(\frac{1}{q} + 1\right)}}
$$

A cleaner derivation writes $k = P_\infty / F_\infty$ directly.
From $\pi^2 - q\pi - q = 0$ the positive root is

$$
\pi_+ = \frac{q + \sqrt{q^2 + 4q}}{2},
$$

so the **steady-state Kalman gain** is

$$
k_\infty = \frac{\pi_+}{\pi_+ + 1}
         = \frac{q + \sqrt{q^2 + 4q}}{q + \sqrt{q^2 + 4q} + 2}.
$$

For $q \to 0$ (nearly constant level), $k_\infty \to 0$; for $q \to \infty$
(very noisy level), $k_\infty \to 1$.

### 2.4 Equivalence to EWMA

In steady state, the filtered level update is

$$
\hat\mu_{t|t} = \hat\mu_{t|t-1} + k_\infty v_t
              = (1 - k_\infty)\hat\mu_{t-1|t-1} + k_\infty y_t,
$$

which is exactly an **exponentially weighted moving average (EWMA)** with
smoothing parameter $\lambda = k_\infty$. The structural model thus provides a
*principled derivation* of EWMA from first principles, with $\lambda$ determined
by the ratio $q$ rather than chosen arbitrarily.

### 2.5 Equivalence to ARIMA(0,1,1)

The *innovations representation* of the LLM is obtained by expressing $y_t$ in
terms of its one-step forecast errors $v_t = y_t - \hat{y}_{t|t-1}$:

$$
y_t - y_{t-1} = v_t - (1 - k_\infty) v_{t-1}.
$$

This is an **MA(1) process applied to first differences**, i.e., ARIMA(0,1,1)
with MA coefficient $\theta_1 = -(1 - k_\infty) \in (-1, 0]$. The mapping is:

$$
\theta_1 = -(1 - k_\infty), \qquad \sigma^2_v = F_\infty = \frac{\sigma^2_\eta}{k_\infty}.
$$

!!! note "Invertibility"
    $k_\infty \in (0,1)$ guarantees $\theta_1 \in (-1, 0)$, so the ARIMA(0,1,1)
    representation is always invertible for $q > 0$.

### 2.6 Forecasting

Because $T = [1]$, the $h$-step-ahead forecast from time $T$ is

$$
a_{T+h|T} = \hat\mu_{T|T} \quad \text{for all } h \geq 1,
$$

with forecast variance growing linearly:

$$
P_{T+h|T} = P_{T|T} + h\,\sigma^2_\eta.
$$

Prediction intervals therefore *widen without bound* as $h \to \infty$, reflecting
the permanent nature of level shocks.

---

## 3. Local Linear Trend Model

### 3.1 Model equations

The **Local Linear Trend (LLT)** model adds a stochastic slope $\nu_t$ to the
local level:

$$
\begin{aligned}
y_t &= \mu_t + \varepsilon_t,
    &&\varepsilon_t \sim N(0, \sigma^2_\varepsilon)\\
\mu_{t+1} &= \mu_t + \nu_t + \xi_t,
    &&\xi_t \sim N(0, \sigma^2_\xi)\\
\nu_{t+1} &= \nu_t + \zeta_t,
    &&\zeta_t \sim N(0, \sigma^2_\zeta)
\end{aligned}
$$

All disturbances $\varepsilon_t$, $\xi_t$, $\zeta_t$ are mutually independent.
The state vector is $\alpha_t = (\mu_t, \nu_t)'$.

### 3.2 State-space matrices

$$
Z = \begin{bmatrix}1 & 0\end{bmatrix}, \quad
T = \begin{bmatrix}1 & 1 \\ 0 & 1\end{bmatrix}, \quad
R = I_2, \quad
Q = \begin{bmatrix}\sigma^2_\xi & 0 \\ 0 & \sigma^2_\zeta\end{bmatrix},
\quad H = \sigma^2_\varepsilon
$$

The transition matrix $T$ encodes $\mu_{t+1} = \mu_t + \nu_t$ (plus noise $\xi_t$)
and $\nu_{t+1} = \nu_t$ (plus noise $\zeta_t$). Since both state components are
diffuse (non-stationary), diffuse initialization is required for the filter; see
[`state-space-theory.md`](state-space-theory.md).

### 3.3 Special cases

| Condition | Model | ARIMA equiv. |
|:----------|:------|:-------------|
| $\sigma^2_\xi = 0$, $\sigma^2_\zeta > 0$ | *Integrated random walk + noise* | ARIMA(0,2,2) (limit) |
| $\sigma^2_\zeta = 0$, $\sigma^2_\xi > 0$ | *Local level + deterministic slope* | ARIMA(0,1,1) plus drift |
| $\sigma^2_\xi = \sigma^2_\zeta = 0$ | *Deterministic linear trend* | Regression on $t$ |
| $\sigma^2_\varepsilon = 0$ | *Smooth trend* | Double integrated random walk |

### 3.4 Equivalence to ARIMA(0,2,2)

For the LLT with $\sigma^2_\xi > 0$ and $\sigma^2_\zeta > 0$, the reduced-form
ARIMA representation satisfies $\Delta^2 y_t = (1 + \theta_1 L + \theta_2 L^2) a_t$
where $a_t$ is white noise and the MA coefficients are functions of the three
variance parameters. This equivalence — first established by Harvey & Todd (1983)
— shows that ARIMA model selection can serve as a diagnostic for the structural
model, but the structural model offers a *richer parametrisation* since three
independent variances map to two MA coefficients (one degree of freedom is
absorbed by the overall scale).

### 3.5 H-step-ahead forecasts

Because $T^h = \begin{bmatrix}1 & h \\ 0 & 1\end{bmatrix}$, the $h$-step forecast is

$$
\begin{bmatrix}a_{\mu,T+h|T} \\ a_{\nu,T+h|T}\end{bmatrix}
= \begin{bmatrix}1 & h \\ 0 & 1\end{bmatrix}
  \begin{bmatrix}\hat\mu_{T|T} \\ \hat\nu_{T|T}\end{bmatrix}
= \begin{bmatrix}\hat\mu_{T|T} + h\,\hat\nu_{T|T} \\ \hat\nu_{T|T}\end{bmatrix}.
$$

The $h$-step-ahead point forecast for $y_{T+h}$ is therefore **linear in $h$**:

$$
\hat{y}_{T+h|T} = \hat\mu_{T|T} + h\,\hat\nu_{T|T}.
$$

The forecast variance grows as $O(h^3)$ for the level forecast, reflecting
cumulative uncertainty in both level and slope innovations.

---

## 4. Stochastic Seasonal Component — Dummy Variable Form

### 4.1 Motivation

A seasonal component $\gamma_t$ with period $s$ satisfies an approximate
*zero-sum constraint* over each window of $s$ consecutive observations:

$$
\sum_{j=0}^{s-1} \gamma_{t-j} \approx 0.
$$

Making this constraint stochastic introduces a disturbance $\omega_t$:

$$
\sum_{j=0}^{s-1} \gamma_{t-j} = \omega_t, \qquad \omega_t \sim N(0, \sigma^2_\omega).
$$

### 4.2 State recursion

Solving for $\gamma_t$ in terms of past values:

$$
\gamma_{t+1} = -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t
             = -\gamma_t - \gamma_{t-1} - \cdots - \gamma_{t-s+2} + \omega_t.
$$

The state vector collecting the current and $s-2$ lagged seasonal values is

$$
\gamma^{(t)} = (\gamma_t,\; \gamma_{t-1},\; \ldots,\; \gamma_{t-s+2})' \in \mathbb{R}^{s-1}.
$$

### 4.3 State-space matrices

The transition matrix $T_s \in \mathbb{R}^{(s-1)\times(s-1)}$ has the form

$$
T_s =
\begin{bmatrix}
-1 & -1 & -1 & \cdots & -1 & -1 \\
 1 &  0 &  0 & \cdots &  0 &  0 \\
 0 &  1 &  0 & \cdots &  0 &  0 \\
 \vdots & & \ddots & & & \vdots \\
 0 &  0 &  0 & \cdots &  1 &  0
\end{bmatrix}
$$

with the first row equal to $(-1, -1, \ldots, -1)$, identity sub-diagonal, and
zeros elsewhere. The selection vector $R_s = e_1 = (1, 0, \ldots, 0)'$ maps the
scalar disturbance $\omega_t$ onto the first state. The noise covariance is
$Q_s = [\sigma^2_\omega]$.

!!! definition "Seasonal State Observation"
    Only the first element of $\gamma^{(t)}$ enters the observation equation:
    $Z_s = e_1' = (1, 0, \ldots, 0)$.

### 4.4 Deterministic seasonal

When $\sigma^2_\omega = 0$, the constraint $\sum_{j=0}^{s-1} \gamma_{t-j} = 0$
holds exactly at every time point. The seasonal pattern is then **fixed across
all years** — identical to including $s - 1$ seasonal dummy regressors with
coefficients summing to zero.

---

## 5. Stochastic Seasonal Component — Trigonometric Form

### 5.1 Spectral decomposition of seasonality

An alternative parametrisation decomposes the seasonal effect into $\lfloor s/2 \rfloor$
trigonometric harmonics. At the $j$-th harmonic frequency

$$
\lambda_j = \frac{2\pi j}{s}, \qquad j = 1, 2, \ldots, \left\lfloor \frac{s}{2} \right\rfloor,
$$

a pair of latent state variables $(\gamma_{j,t},\, \gamma^*_{j,t})'$ evolves via
a **rotation matrix** perturbed by noise:

$$
\begin{bmatrix}\gamma_{j,t+1} \\ \gamma^*_{j,t+1}\end{bmatrix}
=
\underbrace{\begin{bmatrix}\cos\lambda_j & \sin\lambda_j \\ -\sin\lambda_j & \cos\lambda_j\end{bmatrix}}_{C(\lambda_j)}
\begin{bmatrix}\gamma_{j,t} \\ \gamma^*_{j,t}\end{bmatrix}
+
\begin{bmatrix}\omega_{j,t} \\ \omega^*_{j,t}\end{bmatrix}
$$

with $\omega_{j,t}, \omega^*_{j,t} \sim N(0, \sigma^2_{\omega_j})$ i.i.d. and
independent across harmonics. The rotation matrix $C(\lambda_j)$ is orthogonal,
$C(\lambda_j)' C(\lambda_j) = I_2$.

### 5.2 Total seasonal

The observed seasonal effect is the sum over harmonics:

$$
\gamma_t = \sum_{j=1}^{\lfloor s/2 \rfloor} \gamma_{j,t}.
$$

This decomposition is exact: the trigonometric basis $\{1, \cos\lambda_j t, \sin\lambda_j t\}$
spans the same space as the $(s-1)$-dimensional dummy seasonal.

### 5.3 Nyquist frequency

When $s$ is even, the highest harmonic $j = s/2$ has $\lambda_{s/2} = \pi$,
giving $C(\pi) = \text{diag}(-1, -1)$. The pair collapses to a **scalar equation**:

$$
\gamma_{s/2,\,t+1} = -\gamma_{s/2,\,t} + \omega_{s/2,\,t}, \qquad
\omega_{s/2,\,t} \sim N(0, \sigma^2_{\omega_{s/2}}).
$$

The total state dimension for the trigonometric seasonal is therefore
$2\lfloor s/2 \rfloor - [s \text{ even}] = s - 1$, matching the dummy form.

### 5.4 Advantages over the dummy form

| Property | Dummy form | Trigonometric form |
|:---------|:-----------|:-------------------|
| State dimension | $s - 1$ | $s - 1$ |
| Per-harmonic variance | No (shared $\sigma^2_\omega$) | Yes ($\sigma^2_{\omega_j}$ per harmonic) |
| Spectral interpretation | Indirect | Direct: peak at $\lambda_j$ |
| Smooth seasonal | Constrainable | Natural via $\sigma^2_{\omega_j}$ hierarchy |
| Missing period handling | Standard | Identical |

!!! note "Harvey (1989), Section 2.3"
    The trigonometric form is particularly useful when different harmonics of
    the seasonal pattern evolve at different speeds — for instance, when the
    semi-annual component is stable but the annual component drifts over time.

---

## 6. Basic Structural Model (BSM)

### 6.1 Full model

The **Basic Structural Model** (Harvey, 1989, Chapter 2) combines a Local Linear
Trend with a dummy-form stochastic seasonal:

$$
y_t = \mu_t + \gamma_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma^2_\varepsilon)
$$

with $\mu_t$ following the LLT equations (Section 3) and $\gamma_t$ the seasonal
recursion (Section 4).

### 6.2 Parameter vector

The BSM has four variance parameters:

$$
\theta = (\sigma^2_\xi,\; \sigma^2_\zeta,\; \sigma^2_\omega,\; \sigma^2_\varepsilon).
$$

### 6.3 Full state-space matrices (monthly data, $s = 12$)

For monthly data the state vector is
$\alpha_t = (\mu_t, \nu_t, \gamma_t, \gamma_{t-1}, \ldots, \gamma_{t-10})'$
of dimension $m = 2 + (s-1) = 13$.

**Observation matrix** $Z \in \mathbb{R}^{1 \times 13}$:

$$
Z = \bigl[\underbrace{1,\; 0}_{\text{trend}},\;
          \underbrace{1,\; 0,\; \ldots,\; 0}_{11 \text{ seasonal}}\bigr]
$$

**Transition matrix** $T \in \mathbb{R}^{13 \times 13}$ (block diagonal):

$$
T = \begin{bmatrix} T_\mu & 0 \\ 0 & T_s \end{bmatrix}
$$

where

$$
T_\mu = \begin{bmatrix}1 & 1 \\ 0 & 1\end{bmatrix} \in \mathbb{R}^{2\times 2},
\qquad
T_s = \begin{bmatrix}
-1 & -1 & \cdots & -1 & -1 \\
 1 &  0 & \cdots &  0 &  0 \\
 0 &  1 & \cdots &  0 &  0 \\
 \vdots & & \ddots & & \vdots \\
 0 &  0 & \cdots &  1 &  0
\end{bmatrix} \in \mathbb{R}^{11\times 11}
$$

**Selection matrix** $R \in \mathbb{R}^{13 \times 3}$:

$$
R = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0 \\
\vdots & \vdots & \vdots \\
0 & 0 & 0
\end{bmatrix}
$$

mapping disturbances $(\xi_t, \zeta_t, \omega_t)'$ to the first three state
positions $(\mu_t, \nu_t, \gamma_t)$.

**State noise covariance** $Q \in \mathbb{R}^{3 \times 3}$:

$$
Q = \begin{bmatrix}
\sigma^2_\xi & 0 & 0 \\
0 & \sigma^2_\zeta & 0 \\
0 & 0 & \sigma^2_\omega
\end{bmatrix}
$$

**Observation noise covariance**: $H = \sigma^2_\varepsilon$.

### 6.4 Signal-to-noise ratios

For estimation it is convenient to work with ratios relative to $\sigma^2_\varepsilon$:

$$
q_\xi = \frac{\sigma^2_\xi}{\sigma^2_\varepsilon}, \quad
q_\zeta = \frac{\sigma^2_\zeta}{\sigma^2_\varepsilon}, \quad
q_\omega = \frac{\sigma^2_\omega}{\sigma^2_\varepsilon}.
$$

The shape of the decomposition is entirely governed by $(q_\xi, q_\zeta, q_\omega)$;
the overall scale $\sigma^2_\varepsilon$ is a nuisance parameter for point estimates.

### 6.5 Special cases

| Condition | Resulting model |
|:----------|:----------------|
| $\sigma^2_\zeta = 0$ | LLM + seasonal (constant slope) |
| $\sigma^2_\xi = \sigma^2_\zeta = 0$ | Deterministic trend + stochastic seasonal |
| $\sigma^2_\omega = 0$ | LLT + deterministic seasonal |
| $\sigma^2_\varepsilon = 0$ | Exact signal-plus-noise (smooth trend) |

!!! warning "Boundary Solutions in MLE"
    MLE routinely returns $\hat\sigma^2 = 0$ for one or more components. This is
    **not a numerical failure** — it means the data support a deterministic
    component. The likelihood surface is valid at the boundary, and the
    interpretation is simply that the corresponding component does not evolve
    stochastically. See Section 9 for identification details.

---

## 7. Stochastic Cycle Component

### 7.1 Rotation-damping representation

The **stochastic cycle** at frequency $\lambda_c \in (0, \pi)$ with damping
factor $\rho \in (0, 1)$ is defined by the two-dimensional state recursion
(Harvey, 1989, Section 2.4):

$$
\begin{bmatrix}\psi_{t+1} \\ \psi^*_{t+1}\end{bmatrix}
= \rho
\underbrace{\begin{bmatrix}\cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c\end{bmatrix}}_{C(\lambda_c)}
\begin{bmatrix}\psi_t \\ \psi^*_t\end{bmatrix}
+
\begin{bmatrix}\kappa_t \\ \kappa^*_t\end{bmatrix}
$$

with $\kappa_t, \kappa^*_t \sim N(0, \sigma^2_\kappa)$ i.i.d.

### 7.2 Stationarity and variance

For $\rho < 1$, the cycle is **stationary**. The eigenvalues of the companion
matrix $\rho C(\lambda_c)$ are $\rho e^{\pm i\lambda_c}$ with modulus $\rho < 1$.
In steady state,

$$
\text{Var}(\psi_t) = \text{Var}(\psi^*_t) = \frac{\sigma^2_\kappa}{1 - \rho^2}.
$$

### 7.3 Period and spectral properties

The **period** of the cycle is $p_c = 2\pi / \lambda_c$ in time units. For
quarterly data and a typical 10-year business cycle:

$$
\lambda_c = \frac{2\pi}{40} \approx 0.157 \text{ rad/quarter}, \quad
p_c = 40 \text{ quarters} = 10 \text{ years}.
$$

The spectral density of $\psi_t$ has a **peak at $\lambda_c$** with bandwidth
controlled by $\rho$: as $\rho \to 1$ the peak sharpens; as $\rho \to 0$ the
spectrum flattens and the cycle degenerates to white noise.

### 7.4 Non-stationary cycle ($\rho = 1$)

When $\rho = 1$, the cycle is a **random walk on the circle**: the state rotates
at fixed frequency $\lambda_c$ but the amplitude follows a random walk. This
case requires diffuse initialization and is appropriate for modelling persistent
cyclical phenomena such as long-run demographic cycles.

!!! example "Business Cycle Parametrisation"
    A standard macroeconomic application sets $\lambda_c = 2\pi / 32$ for
    quarterly data (8-year cycle), $\rho = 0.9$ (moderate persistence), and
    estimates $\sigma^2_\kappa$ via MLE. The resulting filtered cycle
    $\hat\psi_{t|t}$ tracks NBER recession shading closely without requiring
    any auxiliary data.

---

## 8. Unobserved Components Model — General Framework

### 8.1 Full UCM

The **Unobserved Components Model (UCM)** assembles any subset of the components
described in Sections 2–7:

$$
y_t = \mu_t + \psi_t + \gamma_t + \varepsilon_t.
$$

The framework is **modular**: each component is a self-contained mini state-space
system. To combine them, stack their state vectors and form block-diagonal matrices.

### 8.2 Modular assembly

Let component $k$ have state $\alpha^{(k)}_t \in \mathbb{R}^{m_k}$, matrices
$(Z^{(k)}, T^{(k)}, R^{(k)}, Q^{(k)})$. The combined model has state
$\alpha_t = (\alpha^{(1)'}_t, \ldots, \alpha^{(K)'}_t)'$ of dimension
$m = \sum_k m_k$, with:

$$
Z = \bigl[Z^{(1)},\; Z^{(2)},\; \ldots,\; Z^{(K)}\bigr]
$$

$$
T = \mathrm{diag}\!\bigl(T^{(1)}, T^{(2)}, \ldots, T^{(K)}\bigr)
$$

$$
R = \mathrm{diag}\!\bigl(R^{(1)}, R^{(2)}, \ldots, R^{(K)}\bigr)
$$

$$
Q = \mathrm{diag}\!\bigl(Q^{(1)}, Q^{(2)}, \ldots, Q^{(K)}\bigr)
$$

!!! definition "Additivity and Block Structure"
    The additive nature of the UCM directly implies the block-diagonal structure
    of $T$, $R$, $Q$: components do not interact through the state transition.
    They interact only through the shared observation $y_t$, which is expressed
    through the concatenated $Z$.

### 8.3 Intervention analysis

Structural breaks are incorporated by adding **dummy state variables**:

- **Level shift** at time $t^*$: add state $\delta_t$ with $T_\delta = 1$,
  $Z_\delta = 1$, $Q_\delta = 0$, and initialize $\delta_{t^*} = 1$.
- **Slope change**: add state with $T_\delta = 1$, enter $Z$ through trend accumulation.
- **Seasonal break**: add replacement seasonal state from $t^*$ onwards.

These interventions leave the likelihood decomposition intact and allow
formal likelihood-ratio tests for structural breaks.

### 8.4 Regression components

Fixed regressors $x_t$ are absorbed into the observation intercept:

$$
y_t = x_t' \beta + \mu_t + \gamma_t + \psi_t + \varepsilon_t.
$$

In state-space form this enters as $d_t = x_t' \beta$ (or as additional rows
in $Z$ if $\beta$ is treated as a time-varying state — the TVP regression case;
see [`../user-guide/structural/ucm.md`](../user-guide/structural/ucm.md)).

---

## 9. Identification and Estimation Issues

### 9.1 The signal extraction problem

With only $y_t$ observed, the decomposition into components is **generically
non-unique**: for any given likelihood value $\ell(\theta^*)$, there may exist
other parameter vectors $\tilde\theta \neq \theta^*$ with the same likelihood.
This is the *fundamental identification challenge* of unobserved-components
analysis.

### 9.2 Canonical decomposition and admissibility

Harvey & Todd (1983) establish that identification can be achieved by imposing
the **admissibility condition**: the spectral density of each component must be
non-negative at every frequency. For the BSM, this is automatically satisfied
when $\sigma^2 \geq 0$ for each component.

The **canonical decomposition** sets the irregular variance to its minimum
compatible value (possibly zero), yielding the unique admissible decomposition
with maximum signal variance assigned to the trend and seasonal.

### 9.3 Harvey–Todd identification theorem

!!! note "Harvey–Todd (1983)"
    The BSM is identified if and only if each structural component has at least
    one positive variance parameter. Concretely: the model
    $y_t = \mu_t + \gamma_t + \varepsilon_t$
    is identified provided $\sigma^2_\varepsilon > 0$ OR $\sigma^2_\xi > 0$ (or both),
    and similarly for the seasonal. The parameters cannot be jointly zero unless
    a component is entirely absent.

### 9.4 Boundary solutions in MLE

The parameter space is $\Theta = \{\theta : \sigma^2_k \geq 0\}$. MLE subject
to this constraint frequently returns **boundary solutions** $\hat\sigma^2_k = 0$
for one or more $k$. Standard regularity conditions for asymptotic normality of
MLE fail at boundary points, so inference must rely on:

- **Likelihood ratio tests** (comparing nested models),
- **Profile likelihood confidence intervals** computed over the interior,
- **Bayesian posteriors** with support on $(0, \infty)$ (see
  [`../user-guide/structural/bsm.md`](../user-guide/structural/bsm.md)).

### 9.5 Variance parameterisation for optimization

To enforce $\sigma^2_k \geq 0$ without constrained optimisation, transform:

$$
\sigma^2_k = \exp(2\phi_k), \qquad \phi_k \in \mathbb{R}
$$

and optimise over $\phi_k$ unconstrained. The log-likelihood gradient with
respect to $\phi_k$ is

$$
\frac{\partial \ell}{\partial \phi_k}
= 2\sigma^2_k \frac{\partial \ell}{\partial \sigma^2_k}.
$$

This reparametrisation prevents the optimizer from evaluating at negative variances
but prevents boundary solutions — to recover them, supplement with a constrained
run at $\sigma^2_k = 0$ and compare likelihoods.

---

## 10. Forecasting with Structural Models

### 10.1 H-step-ahead state forecast

For a time-invariant model, the $h$-step-ahead predicted state from time $T$ is
obtained by iterating the transition equation:

$$
a_{T+h|T} = T^h a_{T|T}.
$$

This follows from the tower property of conditional expectations and the Markov
structure of the state.

### 10.2 H-step-ahead forecast variance

Propagating the uncertainty from both the filtered state and the accumulated
transition noise:

$$
P_{T+h|T} = T^h P_{T|T} (T^h)' + \sum_{j=0}^{h-1} T^j R Q R' (T^j)'.
$$

The second term is the **innovation variance accumulated over $h$ steps**, growing
polynomially in $h$ with degree determined by the integration order of the
fastest-growing component.

### 10.3 Component-level forecasts

For the BSM, the block-diagonal structure of $T$ means $T^h = \mathrm{diag}(T_\mu^h, T_s^h)$.

**Trend**: $T_\mu^h = \begin{bmatrix}1 & h \\ 0 & 1\end{bmatrix}$, so
$\hat\mu_{T+h|T} = \hat\mu_{T|T} + h\,\hat\nu_{T|T}$ (linear extrapolation).

**Seasonal**: $T_s^h$ has period $s$, so $\hat\gamma_{T+h|T}$ repeats the last
estimated seasonal pattern. If $\sigma^2_\omega > 0$, the innovation term in
$P_{T+h|T}$ inflates forecast uncertainty for seasonal components at multiples
of $s$.

### 10.4 Long-run behaviour

| Component | $\hat{y}_{T+h|T}$ as $h \to \infty$ |
|:----------|:--------------------------------------|
| Local level | Flat: $\hat\mu_{T|T}$ |
| LLT | Linear: $\hat\mu_{T|T} + h\,\hat\nu_{T|T}$ |
| Cycle ($\rho < 1$) | Damps to zero |
| Seasonal | Periodic with period $s$ |

!!! warning "Forecast Fan Charts"
    Prediction intervals based on $P_{T+h|T}$ assume Gaussian innovations. For
    non-Gaussian series (e.g., counts, proportions), the structural model should
    be extended to a non-Gaussian state-space form; see Durbin & Koopman (2012),
    Chapters 10–11.

---

## Python Example

!!! example "Fitting the BSM with `kalmanbox`"

    ```python
    from kalmanbox.structural import BSM
    import pandas as pd

    # Monthly time series (e.g., airline passengers)
    y = pd.read_csv("data/monthly_series.csv", index_col=0, parse_dates=True)["value"]

    # Specify BSM with s=12 monthly seasonal and an optional stochastic cycle
    model = BSM(seasonal_period=12, cycle=True)
    results = model.fit(y)

    # Extract the estimated decomposition
    decomp = results.get_decomposition()
    trend    = decomp["trend"]
    seasonal = decomp["seasonal"]
    cycle    = decomp["cycle"]
    irregular = decomp["irregular"]

    print(f"Trend variance:    {results.params['sigma2_level']:.6f}")
    print(f"Slope variance:    {results.params['sigma2_slope']:.6f}")
    print(f"Seasonal variance: {results.params['sigma2_seasonal']:.6f}")
    print(f"Cycle variance:    {results.params['sigma2_cycle']:.6f}")
    print(f"Irregular var:     {results.params['sigma2_irregular']:.6f}")
    print(f"Log-likelihood:    {results.llf:.4f}")
    print(f"AIC:               {results.aic:.4f}")

    # Signal-to-noise ratios
    sig2_eps = results.params["sigma2_irregular"]
    for name in ("sigma2_level", "sigma2_slope", "sigma2_seasonal"):
        q = results.params[name] / sig2_eps
        print(f"q_{name}: {q:.4f}")

    # 24-step-ahead forecast with 95 % prediction intervals
    forecast = results.forecast(steps=24, alpha=0.05)
    print(forecast[["mean", "lower_ci", "upper_ci"]].head(12))
    ```

    Full documentation: [`../user-guide/structural/bsm.md`](../user-guide/structural/bsm.md)

---

## Summary of State Dimensions

| Component | Parameters | State dim | ARIMA equiv. |
|:----------|:-----------|:---------:|:-------------|
| Local Level | $\sigma^2_\eta, \sigma^2_\varepsilon$ | 1 | ARIMA(0,1,1) |
| Local Linear Trend | $\sigma^2_\xi, \sigma^2_\zeta, \sigma^2_\varepsilon$ | 2 | ARIMA(0,2,2) |
| Seasonal (dummy) | $\sigma^2_\omega$ | $s-1$ | — |
| Seasonal (trig.) | $\sigma^2_{\omega_j}$ per harmonic | $s-1$ | — |
| Stochastic cycle | $\lambda_c, \rho, \sigma^2_\kappa$ | 2 | ARMA(2,1) |
| BSM (monthly) | 4 variances | 13 | Approx. ARIMA(0,1,1)$\times$(0,1,1)$_{12}$ |
| UCM (general) | $\sum$ component params | $\sum m_k$ | — |

---

## Cross-References

- **State-space foundations**: [`state-space-theory.md`](state-space-theory.md)
- **Kalman filter derivation**: [`kalman-theory.md`](kalman-filter-derivation.md)
- **MLE and likelihood**: [`mle-theory.md`](likelihood.md)
- **BSM user guide**: [`../user-guide/structural/bsm.md`](../user-guide/structural/bsm.md)
- **UCM user guide**: [`../user-guide/structural/ucm.md`](../user-guide/structural/ucm.md)

---

## References

**Harvey, A. C. (1989).** *Forecasting, Structural Time Series Models and the
Kalman Filter.* Cambridge University Press, Cambridge.  
The primary reference for the BSM, trigonometric seasonality, stochastic cycles,
and the equivalence between structural models and ARIMA representations.

**Harvey, A. C. & Todd, P. H. J. (1983).** "Forecasting economic time series
with structural and Box-Jenkins models: A case study." *Journal of Business &
Economic Statistics*, 1(4), 299–307.  
Establishes the ARIMA equivalences for the LLM and LLT, and the identification
theorem for the BSM.

**Durbin, J. & Koopman, S. J. (2012).** *Time Series Analysis by State Space
Methods* (2nd ed.). Oxford University Press, Oxford.  
The authoritative modern treatment of state-space methods, covering diffuse
initialization, exact likelihood, missing data, and non-Gaussian extensions.
The matrix conventions used throughout this page follow Durbin & Koopman (2012).

**West, M. & Harrison, J. (1997).** *Bayesian Forecasting and Dynamic Models*
(2nd ed.). Springer, New York.  
Develops dynamic linear models (DLMs) — the Bayesian counterpart of structural
time series — with sequential updating, discount factors, and prior specification
for variance parameters.
