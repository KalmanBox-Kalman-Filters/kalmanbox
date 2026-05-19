# Smoothing Theory

This page provides a self-contained, mathematically rigorous treatment of
**state smoothing** for linear Gaussian state-space models. The Kalman filter
produces estimates that use only observations up to the current time; smoothing
algorithms use the **entire** data set and are indispensable for signal
extraction, parameter estimation via the EM algorithm, and diagnostic residual
analysis.

Prerequisites: [State-Space Foundations](state-space-theory.md),
[Kalman Filter Derivation](kalman-filter-derivation.md).

See also: [RTS Smoother (User Guide)](../user-guide/kalman/rts-smoother.md),
[MLE Theory](mle-theory.md).

---

## 1. The Smoothing Problem

### 1.1 Filtering versus Smoothing

The **Kalman filter** computes, at each time $t$, the minimum mean-squared-error
(MMSE) linear estimate of the state given all observations up to and including
time $t$:

$$a_{t|t} = E[\alpha_t \mid y_1, \ldots, y_t], \qquad
P_{t|t} = \operatorname{Var}(\alpha_t \mid y_1, \ldots, y_t).$$

This estimate is *causal*: it cannot exploit any information in $y_{t+1},
\ldots, y_T$, even if that information is available. Once the full time series
$y_{1:T}$ has been collected, we can compute the **smoothed** estimate

$$a_{t|T} = E[\alpha_t \mid y_1, \ldots, y_T], \qquad
P_{t|T} = \operatorname{Var}(\alpha_t \mid y_1, \ldots, y_T),$$

for every $t = 1, \ldots, T$. Because $\{y_1, \ldots, y_T\}$ is a strictly
richer information set than $\{y_1, \ldots, y_t\}$ for $t < T$, the smoothed
estimator is at least as efficient as the filtered estimator in the sense of
positive semi-definite ordering:

$$P_{t|T} \leq P_{t|t} \quad \text{for all } t = 1, \ldots, T-1.$$

Equality holds only when all future observations are uninformative about the
current state — a situation that occurs asymptotically in stable systems but
not at finite samples.

!!! note "Smoothing at the terminal point"
    At $t = T$ the two information sets coincide, so $a_{T|T}$ and $P_{T|T}$
    are already the smoothed quantities. The smoother recursion is initialised
    with these values and runs backward toward $t = 1$.

### 1.2 Three Types of Smoother

Three variants of the smoothing problem arise in practice, depending on how the
target time and the data window are defined.

**Fixed-interval smoother** (most common)
: Compute $a_{t|T}$ and $P_{t|T}$ for *all* $t = 1, \ldots, T$ given the fixed
data set $y_{1:T}$. This is the standard offline post-processing problem.

**Fixed-point smoother**
: Fix a single time point $t^*$ and compute $a_{t^*|t}$ as $t$ increases
beyond $t^*$. Useful for refining a single historical estimate as new data
arrive.

**Fixed-lag smoother**
: For each current time $t$, compute $a_{t-L|t}$ for a fixed lag $L \geq 1$.
Provides a real-time estimate of the state $L$ periods in the past. As $t$
advances, the lag window rolls forward.

This page treats all three, with emphasis on the fixed-interval case because it
underlies the EM algorithm and most offline inference tasks.

---

## 2. RTS Smoother — Full Derivation

### 2.1 State-Space Model and Filter Output

We work with the Durbin & Koopman (2012) convention throughout. The model is

$$y_t = Z_t \alpha_t + d_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, H_t),$$
$$\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t, \qquad \eta_t \sim N(0, Q_t),$$

with $\varepsilon_t \perp \eta_s$ for all $t, s$, and $\alpha_1 \sim N(a_1, P_1)$
independent of all disturbances.

The forward Kalman filter pass produces, for $t = 1, \ldots, T$:

| Quantity | Notation | Description |
|:---------|:---------|:------------|
| Predicted state | $a_{t|t-1}$ | $E[\alpha_t \mid y_{1:t-1}]$ |
| Predicted variance | $P_{t|t-1}$ | $\operatorname{Var}(\alpha_t \mid y_{1:t-1})$ |
| Filtered state | $a_{t|t}$ | $E[\alpha_t \mid y_{1:t}]$ |
| Filtered variance | $P_{t|t}$ | $\operatorname{Var}(\alpha_t \mid y_{1:t})$ |
| Innovation | $v_t = y_t - Z_t a_{t|t-1} - d_t$ | Prediction error |
| Innovation variance | $F_t = Z_t P_{t|t-1} Z_t' + H_t$ | |
| Kalman gain | $K_t = P_{t|t-1} Z_t' F_t^{-1}$ | |

### 2.2 The Markov Property and the Joint Distribution

The key structural property of the state process is the **Markov property**:
given $\alpha_{t+1}$, the past $\alpha_t$ is conditionally independent of the
future observations $y_{t+1:T}$. Formally,

$$\alpha_t \perp y_{t+1:T} \mid \alpha_{t+1}.$$

This follows immediately from the transition equation: conditional on
$\alpha_{t+1}$ (and hence on $T_t \alpha_t + c_t + R_t \eta_t$), the residual
information carried by $y_{t+1:T}$ about $\alpha_t$ is exhausted by
$\alpha_{t+1}$.

Now consider the **joint distribution** of $(\alpha_t, \alpha_{t+1})$ given the
past $y_{1:t}$. Both $\alpha_t \mid y_{1:t}$ and $\alpha_{t+1} \mid y_{1:t}$
are Gaussian (a standard result for linear Gaussian systems), so the joint is
multivariate Gaussian with the following mean and block covariance:

$$\begin{pmatrix} \alpha_t \\ \alpha_{t+1} \end{pmatrix} \Bigg| y_{1:t}
\sim \mathcal{N}\!\left(
  \begin{pmatrix} a_{t|t} \\ T_t a_{t|t} + c_t \end{pmatrix},
  \begin{pmatrix} P_{t|t} & P_{t|t} T_t' \\
                  T_t P_{t|t} & P_{t+1|t} \end{pmatrix}
\right).$$

The diagonal blocks are immediate from the filter. The off-diagonal block
$\operatorname{Cov}(\alpha_t, \alpha_{t+1} \mid y_{1:t})$ is derived as follows.
From the transition equation,

$$\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t,$$

and $\eta_t \perp \alpha_t \mid y_{1:t}$ (the disturbance is independent of all
past observations), so

$$\operatorname{Cov}(\alpha_t, \alpha_{t+1} \mid y_{1:t})
= \operatorname{Cov}(\alpha_t, T_t \alpha_t \mid y_{1:t})
= P_{t|t} T_t'.$$

The lower-right block follows by the variance formula for a linear transformation:

$$\operatorname{Var}(\alpha_{t+1} \mid y_{1:t})
= T_t P_{t|t} T_t' + R_t Q_t R_t' = P_{t+1|t}.$$

### 2.3 Deriving the Smoothing Recursion

Conditioning the joint distribution of $(\alpha_t, \alpha_{t+1})$ given
$y_{1:t}$ further on $(\alpha_{t+1}, y_{t+1:T})$, and using the Markov property,

$$\alpha_t \mid \alpha_{t+1}, y_{1:T} = \alpha_t \mid \alpha_{t+1}, y_{1:t}.$$

By the standard formula for conditioning a multivariate Gaussian, the
conditional mean and variance of $\alpha_t$ given $\alpha_{t+1}$ and $y_{1:t}$
are:

$$E[\alpha_t \mid \alpha_{t+1}, y_{1:t}]
= a_{t|t} + P_{t|t} T_t' P_{t+1|t}^{-1} \bigl(\alpha_{t+1} - a_{t+1|t}\bigr),$$

$$\operatorname{Var}(\alpha_t \mid \alpha_{t+1}, y_{1:t})
= P_{t|t} - P_{t|t} T_t' P_{t+1|t}^{-1} T_t P_{t|t}.$$

Define the **smoothing gain**

$$\boxed{J_t = P_{t|t} T_t' P_{t+1|t}^{-1}.}$$

Taking expectations over $\alpha_{t+1} \mid y_{1:T}$, we obtain the
**RTS backward recursion**:

$$\boxed{a_{t|T} = a_{t|t} + J_t \bigl(a_{t+1|T} - a_{t+1|t}\bigr),}$$

$$\boxed{P_{t|T} = P_{t|t} + J_t \bigl(P_{t+1|T} - P_{t+1|t}\bigr) J_t'.}$$

The recursion runs for $t = T-1, T-2, \ldots, 1$, initialized with the terminal
filter output $a_{T|T}$ and $P_{T|T}$.

!!! info "Interpretation of the smoothing gain"
    The gain $J_t = P_{t|t} T_t' P_{t+1|t}^{-1}$ is a regression coefficient.
    It measures how much of the revision $a_{t+1|T} - a_{t+1|t}$ (the
    improvement in the one-step-ahead state estimate when all data are used)
    should be propagated back to $a_{t|t}$. If $T_t = 0$ (states are
    uncorrelated across time), then $J_t = 0$ and smoothing provides no
    benefit.

### 2.4 Proof of Monotone Variance Reduction

To prove $P_{t|T} \leq P_{t|t}$ formally, note that the recursion can be
rewritten as

$$P_{t|T}
= P_{t|t} - J_t P_{t+1|t} J_t' + J_t P_{t+1|T} J_t'.$$

!!! abstract "Proof"
    Expand $P_{t|t} + J_t(P_{t+1|T} - P_{t+1|t})J_t'$:

    $$P_{t|T} = P_{t|t} - J_t(P_{t+1|t} - P_{t+1|T})J_t'.$$

    Since $P_{t+1|T} \leq P_{t+1|t}$ (assumed by induction with base case
    $P_{T|T} \leq P_{T|T}$ trivially true), the matrix $P_{t+1|t} - P_{t+1|T}$
    is positive semi-definite. Hence $J_t(P_{t+1|t} - P_{t+1|T})J_t' \geq 0$,
    which gives $P_{t|T} \leq P_{t|t}$. By induction, the inequality holds for
    all $t$. $\square$

    The inequality is strict (positive definite difference) unless
    $P_{t+1|T} = P_{t+1|t}$, which occurs only when $T = t$ (no future data)
    or when $J_t = 0$ (states are dynamically uncoupled).

---

## 3. Lag-One Covariance Smoother

### 3.1 Definition and Motivation

Several algorithms — most notably the EM algorithm for maximum likelihood
estimation (see Section 9) — require not only the smoothed means and variances
but also the **lag-one cross-covariance**:

$$P_{t,t-1|T} = \operatorname{Cov}(\alpha_t, \alpha_{t-1} \mid y_{1:T}).$$

This is the covariance between the state at time $t$ and the state at time
$t-1$, both conditioned on all observations.

### 3.2 Recursion

The lag-one smoother can be computed as a by-product of the RTS backward pass.
The key identity is

$$\boxed{P_{t,t-1|T} = P_{t|T}\, J_{t-1}',}$$

valid for $t = 2, \ldots, T$.

**Derivation.** Write the conditional regression of $\alpha_{t-1}$ on
$\alpha_t$ given $y_{1:T}$:

$$E[\alpha_{t-1} \mid \alpha_t, y_{1:T}]
= a_{t-1|T} + J_{t-1}(\alpha_t - a_{t|T}).$$

Hence

$$\operatorname{Cov}(\alpha_t, \alpha_{t-1} \mid y_{1:T})
= \operatorname{Var}(\alpha_t \mid y_{1:T})\, J_{t-1}'
= P_{t|T}\, J_{t-1}'.$$

The symmetric relation is

$$P_{t,t+1|T} = \operatorname{Cov}(\alpha_t, \alpha_{t+1} \mid y_{1:T})
= J_t P_{t+1|T}.$$

### 3.3 Use in the EM E-Step

The E-step of the EM algorithm for state-space models requires computing the
**expected sufficient statistics** of the complete-data log-likelihood. These
are:

$$E[\alpha_t \alpha_{t-1}' \mid y_{1:T}]
= P_{t,t-1|T} + a_{t|T}\, a_{t-1|T}'.$$

This quantity involves $P_{t,t-1|T}$ directly and is not obtainable from the
RTS smoother alone without the lag-one extension. The full set of sufficient
statistics required by the E-step is documented in Section 9.

---

## 4. Fixed-Interval Smoother — Durbin-Koopman Formulation

### 4.1 The Disturbance Approach

An alternative but mathematically equivalent approach to fixed-interval
smoothing, due to Koopman (1993) and fully developed in Durbin & Koopman
(2012, Ch. 4), introduces auxiliary backward variables $r_t$ and $N_t$ and
operates on the **predicted** quantities $a_{t|t-1}$ and $P_{t|t-1}$ rather
than on the filtered quantities $a_{t|t}$ and $P_{t|t}$.

This formulation has computational advantages: it avoids storing the filtered
covariance sequence and is more naturally suited to diffuse initialization and
disturbance smoothing.

### 4.2 Backward Auxiliary Variables

Define the terminal conditions

$$r_T = 0_{m \times 1}, \qquad N_T = 0_{m \times m},$$

where $m$ is the state dimension. The backward recursion for $t = T, T-1, \ldots, 2$
proceeds as follows.

First, define the **companion matrix**

$$L_t = T_t - K_t Z_t,$$

where $K_t = T_t P_{t|t-1} Z_t' F_t^{-1}$ is the gain in the Koopman
(1993) convention. Note: in the Durbin & Koopman (2012) book the gain is
sometimes written as $K_t = T_t P_{t|t-1} Z_t' F_t^{-1}$ so that
$L_t = T_t(I - P_{t|t-1} Z_t' F_t^{-1} Z_t)$; both forms are equivalent. We
adopt $L_t = T_t - K_t Z_t$ throughout.

The backward recursion is then

$$\boxed{r_{t-1} = Z_t' F_t^{-1} v_t + L_t' r_t,}$$

$$\boxed{N_{t-1} = Z_t' F_t^{-1} Z_t + L_t' N_t L_t.}$$

### 4.3 Recovering Smoothed States

Given the backward quantities, the smoothed state and variance are recovered
from the **predicted** filter quantities alone:

$$\boxed{a_{t|T} = a_{t|t-1} + P_{t|t-1}\, r_{t-1},}$$

$$\boxed{P_{t|T} = P_{t|t-1} - P_{t|t-1}\, N_{t-1}\, P_{t|t-1}.}$$

!!! tip "Why predicted, not filtered, quantities?"
    The DK formulation is numerically cleaner for sparse models and missing
    data. The predicted quantities $a_{t|t-1}$ and $P_{t|t-1}$ are always
    well-defined even when $y_t$ is missing, whereas the filtered quantities
    $a_{t|t}$ require special-casing. The backward pass simply skips the
    $Z_t' F_t^{-1} v_t$ term for missing observations.

### 4.4 Equivalence to the RTS Smoother

The two formulations (RTS and DK) produce identical smoothed outputs. The
equivalence can be established by substituting the filter update equations into
the DK recursions and showing they reduce to the RTS recursion. Specifically,
the substitution $a_{t|t} = a_{t|t-1} + K_t v_t$ and
$P_{t|t} = P_{t|t-1} - K_t Z_t P_{t|t-1}$ in the RTS formula recovers the
DK formula. The two representations differ only in whether they store and
process filtered or predicted quantities.

---

## 5. Disturbance Smoother (Koopman, 1993)

### 5.1 Smoothing Disturbances Directly

In some applications it is more informative to smooth the **disturbances**
$\varepsilon_t$ and $\eta_t$ directly, rather than the states $\alpha_t$.
Smoothed disturbances are essential for:

- Auxiliary residuals and influence diagnostics (Harvey & Koopman, 1992);
- Detecting outliers and structural breaks;
- Simulation smoothing and Bayesian computation.

The disturbance smoother uses the same backward auxiliary variables $r_t$ and
$N_t$ from Section 4.

### 5.2 Smoothed Observation Disturbance

The smoothed observation disturbance is

$$\boxed{\hat{\varepsilon}_t = H_t \bigl(F_t^{-1} v_t - K_t' r_t\bigr),}$$

where $K_t' = F_t^{-1} Z_t P_{t|t-1} T_t'$ maps backward from the gain. (Here
$K_t$ is as in Section 4.2; the transpose accounts for the directional
convention.) Its conditional variance is

$$\operatorname{Var}(\varepsilon_t - \hat{\varepsilon}_t \mid y_{1:T})
= H_t - H_t F_t^{-1} H_t - H_t K_t' N_t K_t H_t.$$

### 5.3 Smoothed State Disturbance

The smoothed state disturbance is

$$\boxed{\hat{\eta}_t = Q_t R_t' r_t,}$$

with conditional variance

$$\operatorname{Var}(\eta_t - \hat{\eta}_t \mid y_{1:T})
= Q_t - Q_t R_t' N_t R_t Q_t.$$

!!! abstract "Derivation sketch"
    Both results follow from the general formula for the smoothed disturbance
    in a linear model. The complete-data score for $\varepsilon_t$ evaluated at
    $\hat{\varepsilon}_t = H_t F_t^{-1} v_t - H_t K_t' r_t$ satisfies the
    first-order optimality condition. The variance formula is derived from
    $(D-K, 2012, \S 4.5)$: $\operatorname{Var}(\varepsilon_t - \hat{\varepsilon}_t)$
    equals $H_t$ (prior variance) minus the information contributed by $v_t$
    (the $H_t F_t^{-1} H_t$ term) minus the information contributed by all
    future observations (the $H_t K_t' N_t K_t H_t$ term). The state
    disturbance variance follows analogously with $R_t Q_t$ in place of
    $H_t$.

### 5.4 Auxiliary Residuals

Define the **auxiliary residuals** (Harvey & Koopman, 1992)

$$e_t^{(\varepsilon)} = \hat{\varepsilon}_t \big/
  \sqrt{\operatorname{Var}(\varepsilon_t - \hat{\varepsilon}_t)_{ii}},$$

$$e_t^{(\eta)} = \hat{\eta}_t \big/
  \sqrt{\operatorname{Var}(\eta_t - \hat{\eta}_t)_{ii}},$$

where the denominator is the square root of the $i$-th diagonal element. Under
correct model specification these residuals are, asymptotically, independent
$N(0,1)$ random variables. Large values indicate potential outliers in $y_t$
(observation disturbance residuals) or structural breaks in $\alpha_t$ (state
disturbance residuals).

---

## 6. State Smoothing versus Disturbance Smoothing

The following table compares the two smoothing paradigms.

| Property | State Smoother (RTS / DK) | Disturbance Smoother |
|:---------|:--------------------------|:---------------------|
| Primary output | $\hat{\alpha}_t = a_{t|T}$, $P_{t|T}$ | $\hat{\varepsilon}_t$, $\hat{\eta}_t$ |
| Secondary output | $P_{t,t-1|T}$ via lag-1 extension | $\operatorname{Var}(\varepsilon_t - \hat{\varepsilon}_t)$, $\operatorname{Var}(\eta_t - \hat{\eta}_t)$ |
| Storage required | $\{a_{t|t}, P_{t|t}\}_{t=1}^T$ or $\{a_{t|t-1}, P_{t|t-1}\}_{t=1}^T$ | $\{v_t, F_t, K_t\}_{t=1}^T$ plus backward $r_t$, $N_t$ |
| Computational cost | $O(T m^2)$ backward pass | $O(T m^2)$ backward pass |
| When to use | Signal extraction, EM, forecasting | Diagnostics, outlier detection, simulation smoothing |
| Outlier detection | Indirect (large revision in $a_{t|T}$) | Direct (large $e_t^{(\varepsilon)}$ or $e_t^{(\eta)}$) |
| Diffuse handling | Requires modified recursion | Requires modified recursion |

!!! warning "Common misconception"
    The disturbance smoother is **not** an approximation to the state smoother;
    it solves a different but equally well-defined problem. The two are
    complementary: compute the state smoother for inference about latent states,
    the disturbance smoother for inference about shocks and model adequacy.

---

## 7. Fixed-Point Smoother

### 7.1 Problem Statement

Fix a time point $t^* \in \{1, \ldots, T\}$ and suppose observations arrive
sequentially. As each new $y_t$ is received for $t > t^*$, we wish to update
our estimate of $\alpha_{t^*}$ without re-running the full backward smoother.
This is the **fixed-point smoothing** problem.

### 7.2 Forward Recursion

Run the Kalman filter up to $t^*$ in the usual way. At time $t^*$ the estimate
is $a_{t^*|t^*}$ with variance $P_{t^*|t^*}$. To update as new observations
$y_{t^*+1}, y_{t^*+2}, \ldots$ arrive, maintain the **cross-covariance** matrix

$$P_{t^*,t|t} = \operatorname{Cov}(\alpha_{t^*}, \alpha_t \mid y_{1:t})$$

and update it forward.

At time $t = t^*$ initialise: $P_{t^*,t^*|t^*} = P_{t^*|t^*}$.

For $t = t^*, t^*+1, \ldots, T-1$, the recursion is:

$$\boxed{a_{t^*|t+1} = a_{t^*|t} + P_{t^*,t|t}\, Z_t' F_t^{-1} v_t,}$$

$$\boxed{P_{t^*,t+1|t+1} = P_{t^*,t|t}\, T_t' \bigl(I - P_{t|t-1} Z_t' F_t^{-1} Z_t\bigr).}$$

The simplified form of the cross-covariance update is often written as

$$P_{t^*,t+1|t+1} = P_{t^*,t|t}\, T_t' - P_{t^*,t|t}\, Z_t' F_t^{-1} Z_t P_{t|t-1} T_t'.$$

!!! tip "Simplified form using $L_t$"
    Using the companion matrix $L_t = T_t - K_t Z_t$ from Section 4, the
    cross-covariance update reduces to

    $$P_{t^*,t+1|t+1} = P_{t^*,t|t}\, L_t'.$$

    This compact form makes the implementation straightforward and avoids
    explicit computation of $(I - K_t Z_t)$.

### 7.3 Fixed-Point Variance

The smoothed variance $P_{t^*|t}$ satisfies

$$P_{t^*|t+1} = P_{t^*|t} - P_{t^*,t|t}\, F_t^{-1}\, P_{t,t^*|t}.$$

### 7.4 Computational Cost

Each time step beyond $t^*$ requires one $m \times m$ matrix multiplication
and one rank-1 (or rank-$p$) update, giving a cost of $O((m + p) \cdot m)$ per
observation. Over $T - t^*$ steps the total cost is $O((T - t^*)(m + p)m)$,
which is significantly cheaper than re-running the full RTS smoother at cost
$O(Tm^2)$ when $t^*$ is near $T$.

---

## 8. Fixed-Lag Smoother

### 8.1 Problem Statement

In the **fixed-lag** setting, we smooth the state $\alpha_{t-L}$ using
observations $y_1, \ldots, y_t$ for a fixed lag $L \geq 1$. As $t$ advances,
the smoothing window rolls forward: at time $t$ we estimate $\alpha_{t-L}$, at
time $t+1$ we estimate $\alpha_{t+1-L}$, and so on. The lag $L$ is chosen to
balance estimation precision (larger $L$ is better) against allowable latency
in a real-time application.

### 8.2 Efficient Implementation

The fixed-lag smoother can be implemented by running the fixed-point smoother
for the current lag-$L$ target at each time step, but this would be expensive.
A more efficient approach maintains $L$ additional cross-covariance matrices:

$$P_{t-L,t|t},\; P_{t-L+1,t|t},\; \ldots,\; P_{t,t|t} = P_{t|t}.$$

When a new observation $y_{t+1}$ arrives:

1. Run the standard Kalman filter update to obtain $a_{t+1|t+1}$ and
   $P_{t+1|t+1}$.
2. Update each of the $L$ cross-covariance matrices using the fixed-point
   recursion: $P_{s,t+1|t+1} = P_{s,t|t}\, L_t'$ for $s = t-L, \ldots, t$.
3. The lag-$L$ smoothed state update is:

$$a_{t-L|t+1} = a_{t-L|t} + P_{t-L,t|t}\, Z_t' F_t^{-1} v_t.$$

4. Drop the oldest cross-covariance $P_{t-L,t|t}$ and initialise
   $P_{t+1,t+1|t+1} = P_{t+1|t+1}$ for the next step.

The total additional memory requirement is $O(L \cdot m^2)$ and the per-step
cost is $O(L \cdot m^2)$.

### 8.3 Applications

Fixed-lag smoothing arises in:

- **Real-time signal extraction**: seasonal adjustment with a lag of one
  quarter, monetary policy analysis with a two-quarter lag.
- **Communications**: decoding with a bounded-latency constraint.
- **Tracking**: target state estimation in a data-association pipeline where
  a few-step lag is acceptable.

As $L \to \infty$ the fixed-lag smoother converges to the fixed-interval
smoother, so $L$ can be chosen by inspecting the gain in precision as a
function of lag (a rapidly decaying gain curve indicates a small effective
information horizon).

---

## 9. Relationship to the EM Algorithm (E-Step)

### 9.1 The EM Framework for State-Space Models

Let $\theta$ denote the vector of unknown parameters (elements of $T_t$, $Z_t$,
$H_t$, $Q_t$, $R_t$, $d_t$, $c_t$, and possibly the initial conditions
$a_1$, $P_1$). The **EM algorithm** (Shumway & Stoffer, 1982; Dempster et al.,
1977) iterates between:

- **E-step**: Compute $Q(\theta \mid \theta^{(k)}) =
  E_{\alpha_{1:T} \mid y_{1:T}, \theta^{(k)}}[\log p(y_{1:T}, \alpha_{1:T} \mid \theta)]$.
- **M-step**: Maximise $Q$ over $\theta$ to obtain $\theta^{(k+1)}$.

For linear Gaussian state-space models, the E-step reduces to running the
**smoother** under $\theta^{(k)}$: the entire conditional distribution
$p(\alpha_{1:T} \mid y_{1:T}, \theta^{(k)})$ is Gaussian with mean and
covariance given by the smoother.

### 9.2 Sufficient Statistics for the E-Step

The complete-data log-likelihood $\log p(y_{1:T}, \alpha_{1:T} \mid \theta)$
is a quadratic function of $\alpha_{1:T}$. Therefore, the E-step requires only
the **first and second moments** of $\alpha_{1:T}$ under the posterior
$p(\alpha_{1:T} \mid y_{1:T}, \theta^{(k)})$. These moments are:

$$E[\alpha_t \mid y_{1:T}] = a_{t|T},$$

$$E[\alpha_t \alpha_t' \mid y_{1:T}] = P_{t|T} + a_{t|T} a_{t|T}',$$

$$E[\alpha_t \alpha_{t-1}' \mid y_{1:T}] = P_{t,t-1|T} + a_{t|T} a_{t-1|T}'.$$

The first two are provided by the standard RTS smoother. The third requires
the **lag-one covariance smoother** of Section 3 and is the primary reason that
quantity is needed in practice.

The EM sufficient statistics aggregated over time are:

$$\Phi_{11} = \sum_{t=2}^{T} E[\alpha_t \alpha_t' \mid y_{1:T}]
= \sum_{t=2}^{T} \bigl(P_{t|T} + a_{t|T} a_{t|T}'\bigr),$$

$$\Phi_{10} = \sum_{t=2}^{T} E[\alpha_t \alpha_{t-1}' \mid y_{1:T}]
= \sum_{t=2}^{T} \bigl(P_{t,t-1|T} + a_{t|T} a_{t-1|T}'\bigr),$$

$$\Phi_{00} = \sum_{t=1}^{T-1} E[\alpha_t \alpha_t' \mid y_{1:T}]
= \sum_{t=1}^{T-1} \bigl(P_{t|T} + a_{t|T} a_{t|T}'\bigr).$$

### 9.3 M-Step Updates (Time-Invariant Case)

For a time-invariant model with fixed $T$, $Z$, $R$ and free $Q$, $H$, the
M-step closed-form updates are:

$$Q^{(k+1)} = \frac{1}{T-1}\bigl(\Phi_{11} - \Phi_{10} T^{(k)'} - T^{(k)} \Phi_{10}' + T^{(k)} \Phi_{00} T^{(k)'}\bigr),$$

$$H^{(k+1)} = \frac{1}{T}\sum_{t=1}^{T}\bigl(y_t y_t' - y_t a_{t|T}' Z^{(k)'} - Z^{(k)} a_{t|T} y_t' + Z^{(k)} E[\alpha_t \alpha_t'] Z^{(k)'}\bigr).$$

These updates are the essence of the Shumway & Stoffer (1982) EM algorithm for
state-space models, and the smoother output is the **only** input required.

!!! note "Connection to MLE"
    Under regularity conditions, the EM sequence $\{\theta^{(k)}\}$ converges
    to a local maximum of the observed-data log-likelihood
    $\log p(y_{1:T} \mid \theta)$, which is also the MLE. See
    [MLE Theory](mle-theory.md) for details.

---

## 10. Smoothing in the Diffuse Case

### 10.1 Motivation

When one or more state components have a completely diffuse (flat) prior —
either because they represent nonstationary components (random walks, integrated
processes) or because no meaningful prior information is available — the initial
variance matrix $P_1$ contains an infinite component. The standard Kalman
filter and RTS smoother recursions are then undefined at the start.

Durbin & Koopman (2012, Ch. 5) develop an **exact diffuse smoother** that
handles this case rigorously, without resorting to large-$\kappa$ approximations.

### 10.2 Augmented Filter Quantities

During the **diffuse initialisation period** (periods $t = 1, \ldots, d$,
where $d$ is the number of observations required to exhaust the diffuse
initial information), the Kalman filter maintains two covariance matrices:

- $P_{t|t-1}^{(\infty)}$: the part of the prediction variance associated with
  the diffuse component;
- $P_{t|t-1}^{(*)}$: the stationary (finite) part.

The innovations variance is decomposed as

$$F_t = F_t^{(\infty)} + F_t^{(*)} \quad
\text{where } F_t^{(\infty)} = Z_t P_{t|t-1}^{(\infty)} Z_t'.$$

The diffuse period ends at the first $t$ for which $F_t^{(\infty)} = 0$.

### 10.3 Diffuse Backward Recursion

The DK backward auxiliary variables are augmented:

$$r_{t-1}^{(\infty)} = Z_t' \bigl(F_t^{(\infty)}\bigr)^{-1} v_t
  + \bigl(T_t - K_t^{(\infty)} Z_t\bigr)' r_t^{(\infty)},$$

$$r_{t-1}^{(*)} = -Z_t' M_t^{(\infty)} v_t
  + \bigl(T_t - K_t^{(*)} Z_t\bigr)' r_t^{(*)}
  + \bigl(T_t - K_t^{(\infty)} Z_t\bigr)' r_t^{(\infty)} \text{ (correction terms)},$$

where $K_t^{(\infty)} = T_t P_{t|t-1}^{(\infty)} Z_t' (F_t^{(\infty)})^{-1}$
and $M_t^{(\infty)}$ is an auxiliary matrix defined by the diffuse recursion.

The terminal conditions remain $r_T^{(\infty)} = r_T^{(*)} = 0$ and
$N_T^{(\infty)} = N_T^{(*)} = N_T^{(**)} = 0$.

### 10.4 Transition from Diffuse to Stationary Phase

At the end of the diffuse period (time $d$), the filter switches from the
augmented diffuse recursion to the standard recursion. The smoothed quantities
at time $t \leq d$ are:

$$a_{t|T} = a_{t|t-1}^{(*)} + P_{t|t-1}^{(*)} r_{t-1}^{(*)}
            + P_{t|t-1}^{(\infty)} r_{t-1}^{(\infty)},$$

$$P_{t|T} = P_{t|t-1}^{(*)} - P_{t|t-1}^{(*)} N_{t-1}^{(*)} P_{t|t-1}^{(*)}
            - P_{t|t-1}^{(\infty)} N_{t-1}^{(\infty)} P_{t|t-1}^{(\infty)}
            \;\text{(simplified form)}.$$

For $t > d$ the standard DK formula of Section 4.3 applies unchanged.

!!! warning "Numerical implementation"
    Exact diffuse smoothing requires careful bookkeeping of which filter
    quantities belong to the diffuse and stationary components. An incorrect
    transition — for example, starting the standard backward pass too early —
    leads to inconsistent smoothed covariances that may violate positive
    semi-definiteness. The `kalmanbox` implementation tracks the diffuse flag
    per time period and per state dimension. See
    [Diffuse Initialization (User Guide)](../user-guide/kalman/diffuse-initialization.md)
    for the API.

### 10.5 Practical Recommendation

For models with integrated components (local level, local linear trend, BSM),
always use the exact diffuse smoother. The large-$\kappa$ approximation
($P_1 = \kappa I$, $\kappa = 10^6$) introduces errors that propagate through
the entire smoothed sequence and distort variance estimates, particularly for
short time series.

---

## 11. Summary: Algorithm Selection Guide

The following decision tree guides algorithm selection.

```
Is the full data set available offline?
  YES → Fixed-interval smoother (RTS or DK)
    Do you need disturbance residuals?
      YES → Disturbance smoother (Section 5)
      NO  → RTS smoother (Section 2) or DK state smoother (Section 4)
    Do you need lag-one cross-covariances (e.g., for EM)?
      YES → Append lag-one extension (Section 3)
  NO → Online / sequential setting
    Fixed time point to refine?
      YES → Fixed-point smoother (Section 7)
    Rolling lag L?
      YES → Fixed-lag smoother (Section 8)
```

For offline inference the RTS smoother is the standard choice: it is
numerically stable, parallelisable over independent series, and its output
directly feeds both the EM E-step and the diagnostic residual framework.

---

## 12. Notation Summary

| Symbol | Definition |
|:-------|:-----------|
| $m$ | State dimension |
| $p$ | Observation dimension |
| $r$ | State disturbance dimension |
| $T$ | Sample size |
| $\alpha_t$ | State vector, $m \times 1$ |
| $y_t$ | Observation vector, $p \times 1$ |
| $a_{t|s}$ | $E[\alpha_t \mid y_{1:s}]$ |
| $P_{t|s}$ | $\operatorname{Var}(\alpha_t \mid y_{1:s})$ |
| $P_{t,s|T}$ | $\operatorname{Cov}(\alpha_t, \alpha_s \mid y_{1:T})$ |
| $v_t$ | Innovation $y_t - Z_t a_{t|t-1} - d_t$ |
| $F_t$ | Innovation variance $Z_t P_{t|t-1} Z_t' + H_t$ |
| $K_t$ | Kalman gain $P_{t|t-1} Z_t' F_t^{-1}$ |
| $J_t$ | Smoothing gain $P_{t|t} T_t' P_{t+1|t}^{-1}$ |
| $L_t$ | Companion matrix $T_t - K_t Z_t$ |
| $r_t$ | Backward smoothing vector, $m \times 1$ |
| $N_t$ | Backward smoothing matrix, $m \times m$ |
| $\hat{\varepsilon}_t$ | Smoothed observation disturbance |
| $\hat{\eta}_t$ | Smoothed state disturbance |

---

## References

- **Rauch, H.E., Tung, F., & Striebel, C.T. (1965).** Maximum likelihood
  estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445–1450.
  The original paper introducing the RTS backward recursion.

- **Koopman, S.J. (1993).** Disturbance smoother for state space models.
  *Biometrika*, 80(1), 117–126. Introduces the disturbance smoother and
  auxiliary residuals; derives the DK backward auxiliary variables.

- **Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press. The definitive reference for the
  DK formulation, exact diffuse smoothing, and disturbance smoother. Chapters
  4–5.

- **Shumway, R.H. & Stoffer, D.S. (1982).** An approach to time series
  smoothing and forecasting using the EM algorithm. *Journal of Time Series
  Analysis*, 3(4), 253–264. First derivation of the EM algorithm for
  state-space models using the lag-one covariance smoother.

- **Shumway, R.H. & Stoffer, D.S. (2017).** *Time Series Analysis and Its
  Applications* (4th ed.). Springer. Chapter 6 covers state-space smoothing
  and the EM algorithm with accessible treatment.

- **Harvey, A.C. & Koopman, S.J. (1992).** Diagnostic checking of unobserved-
  components time series models. *Journal of Business & Economic Statistics*,
  10(4), 377–389. Auxiliary residuals from the disturbance smoother.

- **Anderson, B.D.O. & Moore, J.B. (1979).** *Optimal Filtering*. Prentice
  Hall. Chapter 8 covers fixed-point and fixed-lag smoothing.
