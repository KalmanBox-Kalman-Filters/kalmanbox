# Dynamic Factor Model Theory

This page develops the complete mathematical theory of **Dynamic Factor Models (DFMs)**:
their probabilistic foundations, identification conditions, asymptotic properties, and the
EM algorithm used to estimate them. The treatment is self-contained but assumes familiarity
with the linear Gaussian state-space model developed in
[`state-space-theory.md`](state-space-theory.md).

Practical API usage and code-first examples are in the
[DFM user guide](../user-guide/advanced/dfm.md). Maximum-likelihood foundations are
reviewed in [`mle-theory.md`](mle-theory.md).

---

## 1. Introduction and Motivation

### 1.1 Factor Model Intuition

Economic and financial datasets routinely consist of hundreds or thousands of time series
that move together in ways that cannot plausibly be attributed to independent shocks.
Monthly industrial production indices across sectors, sovereign bond yields at different
maturities, and equity returns within an industry group all exhibit **pervasive
co-movement** — a feature that sparse VAR specifications handle poorly because they
require $O(n^2)$ parameters to model $n$ series.

The **factor model** captures this co-movement parsimoniously: it posits that the
$n$-dimensional observed vector $y_t$ is driven, to a large extent, by an
$r$-dimensional vector of **common factors** $f_t$ with $r \ll n$. The remaining
variation is assigned to **idiosyncratic components** $e_t$ that are (approximately)
cross-sectionally uncorrelated. Formally:

$$
y_t = \Lambda f_t + e_t
$$

where $\Lambda \in \mathbb{R}^{n \times r}$ is the **loading matrix** and $e_t$ is the
idiosyncratic noise. The central idea is that after projecting out the common factor
space $\text{col}(\Lambda)$, the residuals are nearly uncorrelated across $i$.

This parsimony has a direct computational dividend: the full $n \times n$ covariance
matrix of $y_t$ is approximated by the $r \times r$ covariance of $f_t$ plus a diagonal
(or block-diagonal) remainder. For $n = 100$ series and $r = 3$ factors the number of
covariance parameters drops from 5050 to 303.

### 1.2 Connection to Principal Component Analysis

The **static factor model** is intimately related to principal component analysis (PCA).
Given the eigendecomposition

$$
\Sigma_y = \Lambda \Sigma_f \Lambda' + \Sigma_e \approx V_r D_r V_r'
$$

where $V_r$ collects the $r$ leading eigenvectors of the sample covariance $\hat\Sigma_y$
and $D_r$ is the diagonal matrix of the $r$ largest eigenvalues, the PC estimator
$\hat f_t = V_r' y_t$ is the maximum-likelihood estimator of the factors under the
strict (spherical idiosyncratic) factor model. Under the **approximate** factor model the
PC estimator remains consistent as $n, T \to \infty$ at a rate determined by
$\delta_{nT} = \min(\sqrt{n}, \sqrt{T})$ (Bai & Ng, 2002; Bai, 2003).

PCA, however, ignores the **serial dependence** of $f_t$. The dynamic factor model
extends the static approach by explicitly specifying the time-series structure of the
factors, enabling Kalman-filter-based inference and sequential updating — the features
that make DFMs valuable for real-time forecasting and nowcasting.

### 1.3 Key Applications

| Application | $n$ | $r$ | Dynamic structure |
|:------------|:----|:----|:------------------|
| Macroeconomic nowcasting (Giannone et al., 2008) | 100–700 | 2–5 | VAR(1) factors |
| Nelson-Siegel yield curve | 10–30 maturities | 3 | AR(1) level/slope/curvature |
| Asset pricing (Fama-French style) | 500+ | 3–6 | AR(1) or random walk |
| Sensor fusion / IoT panel | 50–1000 | 5–20 | system-specific |
| Mixed-frequency macro models | 20–100 | 2–4 | Mariano-Murasawa calendar |

---

## 2. General DFM Formulation

### 2.1 Static Factor Representation

The **static factor model** (Stock & Watson, 2002) writes

$$
y_t = \Lambda F_t + e_t, \qquad t = 1, \ldots, T
\tag{2.1}
$$

where $y_t \in \mathbb{R}^n$, $\Lambda \in \mathbb{R}^{n \times q}$, and
$F_t \in \mathbb{R}^q$ is a **static factor** that may stack current and lagged dynamic
factors. The idiosyncratic errors satisfy

$$
E[e_t e_t'] = \Sigma_e
$$

Under the **strict** factor model $\Sigma_e$ is diagonal ($\Sigma_e = \text{diag}(\sigma_1^2,
\ldots, \sigma_n^2)$); under the **approximate** factor model (Chamberlain & Rothschild,
1983) $\Sigma_e$ may have weak cross-sectional dependence but its largest eigenvalue
is bounded as $n \to \infty$.

### 2.2 Dynamic Factor Representation

The **dynamic factor model** explicitly models the factors as a stationary vector
autoregression:

$$
\begin{aligned}
y_t &= \Lambda f_t + e_t, & e_t &\sim \mathcal{N}(0, H) \tag{2.2a} \\
f_t &= \Phi_1 f_{t-1} + \cdots + \Phi_p f_{t-p} + \eta_t, & \eta_t &\sim \mathcal{N}(0, Q)
\tag{2.2b}
\end{aligned}
$$

where $f_t \in \mathbb{R}^r$ is the **dynamic factor** of dimension $r$ and $\Lambda \in
\mathbb{R}^{n \times r}$ are the contemporaneous loadings. All lags up to order $p$ are
present in the factor transition (2.2b).

!!! note "Notation conventions"
    Throughout this page:

    - $n$ = number of observed series
    - $r$ = number of dynamic factors (latent dimension)
    - $p$ = VAR order for factor dynamics
    - $T$ = number of time periods

    The **state dimension** is $m = r \cdot p$ after writing the companion form.

### 2.3 Companion Form and State-Space Representation

To cast (2.2) as a standard linear Gaussian SSM, stack the dynamic factors and their
lags into the augmented state vector:

$$
\tilde{f}_t =
\begin{pmatrix} f_t \\ f_{t-1} \\ \vdots \\ f_{t-p+1} \end{pmatrix} \in \mathbb{R}^{m},
\qquad m = r \cdot p
\tag{2.3}
$$

The **companion-form transition matrix** is

$$
\mathbf{T} =
\begin{pmatrix}
\Phi_1 & \Phi_2 & \cdots & \Phi_{p-1} & \Phi_p \\
I_r    & 0      & \cdots & 0          & 0      \\
0      & I_r    & \cdots & 0          & 0      \\
\vdots &        & \ddots &            & \vdots \\
0      & 0      & \cdots & I_r        & 0
\end{pmatrix}
\in \mathbb{R}^{m \times m}
\tag{2.4}
$$

and the noise-selection matrix is

$$
\mathbf{R} =
\begin{pmatrix} I_r \\ 0 \\ \vdots \\ 0 \end{pmatrix}
\in \mathbb{R}^{m \times r}
\tag{2.5}
$$

The **design matrix** only picks out the contemporaneous factors from the augmented
state:

$$
\mathbf{Z} = \begin{pmatrix} \Lambda & 0 & \cdots & 0 \end{pmatrix}
\in \mathbb{R}^{n \times m}
\tag{2.6}
$$

The resulting **state-space system** (in Durbin-Koopman notation) is:

$$
\begin{aligned}
\tilde{f}_{t+1} &= \mathbf{T}\, \tilde{f}_t + \mathbf{R}\, \eta_t,
& \eta_t &\sim \mathcal{N}(0, Q) \tag{2.7a} \\
y_t &= \mathbf{Z}\, \tilde{f}_t + e_t,
& e_t &\sim \mathcal{N}(0, H) \tag{2.7b}
\end{aligned}
$$

### 2.4 System Matrix Dimensions

| Symbol | Dimension | Role |
|:-------|:----------|:-----|
| $\mathbf{Z}$ | $n \times m$ | Design (observation) matrix; equals $[\Lambda \; 0_{n \times r(p-1)}]$ |
| $\mathbf{T}$ | $m \times m$ | Companion-form transition matrix |
| $\mathbf{R}$ | $m \times r$ | Noise-selection matrix; first $r$ rows are $I_r$ |
| $Q$ | $r \times r$ | Factor innovation covariance |
| $H$ | $n \times n$ | Idiosyncratic noise covariance (diagonal in exact model) |
| $\tilde{f}_t$ | $m \times 1$ | Augmented state = $(f_t', \ldots, f_{t-p+1}')'$ |

The Kalman filter then applies directly to (2.7), delivering optimal factor estimates
$\hat{f}_{t|t}$ and smoothed estimates $\hat{f}_{t|T}$ with no modification.

### 2.5 Exact vs Approximate Factor Models

The distinction is in the assumed structure of $H$:

**Exact (strict) factor model:**

$$
H = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)
$$

Idiosyncratic shocks are mutually uncorrelated. This is the standard DFM assumption
and ensures that the Kalman filter's $O(n^3)$ update reduces to $O(nr^2)$ due to the
Woodbury identity when $r \ll n$.

**Approximate factor model:**

$$
H = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2) + \Delta
$$

where $\Delta$ captures weak cross-sectional dependence. This encompasses
serially correlated idiosyncratic components modelled as block-structured SSMs. The
largest eigenvalue of $H$ must remain $O(1)$ as $n \to \infty$.

!!! warning "Computational impact of non-diagonal $H$"
    If $H$ is not diagonal, the Kalman update step requires inverting the $n \times n$
    innovation covariance $F_t = ZP_{t|t-1}Z' + H$. For large $n$ this is expensive.
    The diagonal assumption enables the Woodbury matrix identity to reduce the cost to
    $O(n r^2)$. `kalmanbox` exploits this by default when `idiosyncratic='diagonal'`.

---

## 3. Identification

### 3.1 The Rotation Problem

The DFM is **not identified** without normalisation restrictions. For any invertible
$r \times r$ matrix $M$, the reparameterisation

$$
\Lambda^* = \Lambda M^{-1}, \qquad f_t^* = M f_t
\tag{3.1}
$$

satisfies

$$
\Lambda^* f_t^* = \Lambda M^{-1} M f_t = \Lambda f_t
$$

so the likelihood is unchanged. Neither $\Lambda$ nor $f_t$ is individually identified.
The problem has two distinct components:

1. **Scale indeterminacy:** multiply any factor by $c$ and divide the corresponding
   loading column by $c$.
2. **Rotation indeterminacy:** apply any orthogonal transformation $Q$ (with $Q'Q = I_r$)
   and neither the likelihood nor the covariance structure changes.

### 3.2 Standard Normalisations

**Normalisation A — Fixed factor covariance (default in `kalmanbox`):**

$$
Q = I_r, \qquad \Lambda \text{ unrestricted}
\tag{3.2a}
$$

This pins the scale of each factor to unit innovation variance and forces uncorrelated
innovations. Rotation indeterminacy remains — it is removed by the lower-triangular
restriction below.

**Normalisation B — Lower-triangular loading matrix:**

$$
\Lambda = \begin{pmatrix} \lambda_{11} & 0 & \cdots & 0 \\
\lambda_{21} & \lambda_{22} & \cdots & 0 \\
\vdots & & \ddots & \vdots \\
\lambda_{r1} & \lambda_{r2} & \cdots & \lambda_{rr} \\
\lambda_{r+1,1} & \lambda_{r+1,2} & \cdots & \lambda_{r+1,r} \\
\vdots & & & \vdots \end{pmatrix}
\tag{3.2b}
$$

with $\lambda_{ii} > 0$ for $i = 1, \ldots, r$ (sign normalisation). The leading
$r \times r$ block is lower triangular with positive diagonal. Combined with (3.2a),
this achieves **global identification**.

**Normalisation C — PC normalisation:**

$$
\frac{1}{n}\Lambda'\Lambda = I_r, \qquad \frac{1}{T}\sum_{t=1}^T f_t f_t' \text{ diagonal}
\tag{3.2c}
$$

This is the convention used in the PC estimator of Bai & Ng (2002). It is convenient
for asymptotic theory but less natural in the Bayesian/EM context.

### 3.3 Sign Normalisation

Even after resolving rotation indeterminacy, the sign of each factor is not identified:
multiplying $f_t$ by $-1$ and $\Lambda$ by $-1$ leaves the likelihood unchanged. The
convention $\lambda_{ii} > 0$ (positive diagonal of the lower-triangular block) resolves
this.

!!! note "Economic sign conventions"
    In macroeconomic applications, a common practice is to require that the first factor
    loads positively on GDP or industrial production. This is a structural identification
    restriction that goes beyond the statistical normalisation.

### 3.4 Econometric Conditions (Bai & Ng, 2013)

Formal identification of the loadings and factors in large panels requires:

1. **Rank condition:** $\text{rank}(\Lambda) = r$ (full column rank).
2. **Moment condition:** $\frac{1}{n}\Lambda'\Lambda \to D$ where $D$ is a positive
   definite diagonal matrix (eigenvalues distinct and positive).
3. **Idiosyncratic weak dependence:** the $n \times n$ matrix
   $E[e_t e_t']/n \to 0$ in spectral norm as $n \to \infty$ (Chamberlain &
   Rothschild, 1983 approximate factor condition).
4. **Factor non-degeneracy:** the $r \times r$ second-moment matrix of $f_t$ is
   non-singular.

Under these conditions, the rotation from the PC estimator to the true factor space
converges at rate $\delta_{nT}^{-1} = 1/\min(\sqrt{n}, \sqrt{T})$.

### 3.5 Restrictions That Resolve Rotation Indeterminacy

Several economically motivated restrictions are used in practice:

| Restriction | Interpretation | Residual indeterminacy |
|:------------|:---------------|:-----------------------|
| Diagonal $\Phi$ | Factors evolve independently | Scale + sign |
| Lower-triangular $\Lambda$, $\lambda_{ii} > 0$ | Recursive structure: series $i$ loads on factors $1, \ldots, i$ only | None |
| Block $\Lambda$ (grouped series) | Regional/sectoral factor structure | Sign per block |
| Fixed rows of $\Lambda$ to $I_r$ | Observed reference series define factors | None |

---

## 4. Static Factors vs Dynamic Factors

### 4.1 Stock & Watson (2002) Static Factor Representation

Stock & Watson (2002) showed that a dynamic factor model with $r$ dynamic factors of
VAR order $p$ has a static representation with $q = r \cdot p$ static factors. The
static factor $F_t = (f_t', f_{t-1}', \ldots, f_{t-p+1}')' \in \mathbb{R}^q$
satisfies

$$
y_t = \tilde{\Lambda} F_t + e_t
\tag{4.1}
$$

where $\tilde{\Lambda} = [\Lambda\; 0\; \cdots\; 0] \in \mathbb{R}^{n \times q}$ and
$F_t$ follows a VAR(1) in companion form.

The advantage of the static representation is that PC applied to $y_t$ consistently
estimates the column space of $\tilde{\Lambda}$ (and hence $\Lambda$) without knowing
the factor dynamics. This is the basis of the two-step estimator.

### 4.2 Forni et al. (2000) Dynamic Factor Representation

Forni, Hallin, Lippi & Reichlin (2000) adopt the **spectral** approach, defining the
factor model in the frequency domain. For a (weakly) stationary panel, the spectral
density matrix is decomposed as

$$
\Sigma_y(\omega) = \Lambda(\omega) \Sigma_f(\omega) \Lambda(\omega)^* + \Sigma_e(\omega)
\tag{4.2}
$$

where $\Lambda(\omega) = \sum_{j=-\infty}^{\infty} \Lambda_j e^{-i\omega j}$ is a
one-sided or two-sided filter of loadings, and $*$ denotes conjugate transpose. The
**generalised dynamic factor** $\chi_{it} = \sum_j \lambda_{ij} f_{t-j}$ is a
two-sided filtered version of the primitive factor $f_t$.

The two-sided filter exploits future information and is not causal; a **one-sided**
(causal) approximation is used in real-time applications. The Forni et al. estimator
requires $n, T \to \infty$ for consistency.

### 4.3 PC Estimators of Static Factors

Given the $T \times n$ data matrix $Y$, the sample covariance is
$\hat\Sigma_y = Y'Y / T$. Let $\hat{V}_r$ be the $n \times r$ matrix of the $r$
leading eigenvectors (normalised so that $\hat{V}_r'\hat{V}_r = I_r$). The
**PC factor estimates** are

$$
\hat{F} = Y \hat{V}_r \in \mathbb{R}^{T \times r}
\tag{4.3}
$$

and the loading matrix is recovered by OLS of $y_t$ on $\hat{F}$:

$$
\hat{\Lambda} = Y' \hat{F} / T \in \mathbb{R}^{n \times r}
\tag{4.4}
$$

Under the Bai-Ng (2002) assumptions these are consistent up to a rotation matrix
$H_0$ that converges to a fixed invertible matrix as $n, T \to \infty$.

### 4.4 Relationship Between Static and Dynamic Representations

If $r$ dynamic factors of order $p$ generate the data, then the static representation
has $q \leq r \cdot p$ static factors. PC on $y_t$ consistently estimates the space
spanned by $(f_t', \ldots, f_{t-p+1}')'$. The dynamic structure must be recovered in a
second step (e.g., by regressing $\hat{f}_t$ on its own lags).

Importantly, the **number of static factors $q$ exceeds the number of dynamic factors
$r$** whenever $p > 1$. IC applied to the static representation overestimates $r$
when used to select the number of dynamic factors. This is why Bai & Ng (2007)
propose separate criteria for $r$ and $q$.

### 4.5 When Each Representation Is Preferred

| Criterion | Static PC | Dynamic (Kalman) |
|:----------|:----------|:-----------------|
| Computational cost | $O(nT + q^2 T)$ | $O(T m^3 + n r^2 T)$ |
| Handles missing data | Requires imputation | Natively via Kalman filter |
| Real-time updating | Requires full re-estimation | Sequential update |
| Mixed-frequency data | Not directly | Yes (Mariano-Murasawa) |
| Parameter uncertainty | Not quantified | Posterior intervals available |
| Sample size requirement | $n, T$ both large | Works for moderate $n, T$ |

---

## 5. Estimation: Two-Step (PCA + Kalman)

The two-step estimator (Doz, Giannone & Reichlin, 2012) combines the consistency
of PC at large $n$ with the optimality of the Kalman filter conditional on parameters.

### 5.1 Step 1: PCA Initialisation

**Standardise** each series to zero mean and unit variance (unless the data are already
comparable in scale):

$$
\tilde{y}_{it} = \frac{y_{it} - \bar{y}_i}{\hat\sigma_i}
\tag{5.1}
$$

Compute the eigendecomposition of the sample covariance:

$$
\hat\Sigma_{\tilde{y}} = \frac{1}{T}\tilde{Y}'\tilde{Y} = \sum_{j=1}^n \hat\lambda_j \hat{v}_j \hat{v}_j'
\tag{5.2}
$$

The initial loading estimate is

$$
\hat\Lambda^{(0)} = \sqrt{n}\,\hat{V}_r, \qquad \hat{V}_r = [\hat{v}_1, \ldots, \hat{v}_r]
\tag{5.3}
$$

(the $\sqrt{n}$ normalisation aligns with the PC convention $\Lambda'\Lambda/n = I_r$).
The initial factor estimates are

$$
\hat{f}_t^{(0)} = \frac{1}{n}\hat\Lambda^{(0)'} \tilde{y}_t
\tag{5.4}
$$

### 5.2 Initialising the Factor Dynamics

Regress $\hat{f}_t^{(0)}$ on its own lags to obtain initial VAR coefficients:

$$
\hat{f}_t^{(0)} = \hat\Phi_1 \hat{f}_{t-1}^{(0)} + \cdots + \hat\Phi_p \hat{f}_{t-p}^{(0)} + \hat\eta_t
\tag{5.5}
$$

The initial factor covariance is $\hat{Q}^{(0)} = \frac{1}{T-p}\sum_t \hat\eta_t \hat\eta_t'$.
The initial idiosyncratic variance is

$$
\hat\sigma_i^{2,(0)} = \frac{1}{T}\sum_t (y_{it} - \hat\Lambda_i^{(0)} \hat{f}_t^{(0)})^2
\tag{5.6}
$$

### 5.3 Step 2: Kalman Smoother with Fixed Parameters

Given the parameters $\theta^{(0)} = \{\hat\Lambda^{(0)}, \hat\Phi^{(0)}, \hat{Q}^{(0)},
\hat{H}^{(0)}\}$, run the **Kalman filter** forward and the **RTS smoother** backward
(see [`kalman-filter-derivation.md`](kalman-filter-derivation.md)) to obtain

$$
\hat{f}_{t|T}^{(0)} = E[f_t \mid y_{1:T}; \theta^{(0)}]
\tag{5.7}
$$

These smoothed factors are the two-step estimator. No parameter updates are performed
in the basic two-step version; the parameters remain fixed at the PC estimates.

### 5.4 Consistency of the Two-Step Estimator

!!! definition "Theorem (Doz, Giannone & Reichlin, 2012)"
    Under the approximate factor model assumptions with $\text{rank}(\Lambda) = r$,
    bounded idiosyncratic spectral density, and $\Phi$ having all roots outside the
    unit circle, the two-step estimator satisfies

    $$
    \frac{1}{\sqrt{T}} \sum_{t=1}^T \|\hat{f}_{t|T}^{(0)} - H_0 f_t\|^2
    \xrightarrow{p} 0
    $$

    as $n, T \to \infty$ where $H_0$ is a fixed $r \times r$ invertible rotation matrix.

The rate of convergence is $O_p(\delta_{nT}^{-1})$ with $\delta_{nT} = \min(\sqrt{n}, \sqrt{T})$.
When $n$ is large relative to $T$, the PC initialisation dominates; when $T$ is large,
the Kalman smoother's time-series efficiency dominates.

### 5.5 Bai & Ng (2002) Asymptotic Theory

Bai & Ng (2002) establish, under the assumptions:

(A1) $\frac{1}{n}\Lambda'\Lambda \to \Sigma_\Lambda > 0$ (full rank)

(A2) $E\|f_t\|^4 < \infty$ and $f_t$ is ergodic stationary

(A3) The idiosyncratic errors satisfy $E[e_{it}] = 0$, $E|e_{it}|^8 < \infty$, and
the largest eigenvalue of $E[e_t e_t']$ is bounded above by $M < \infty$ for all $n$

(A4) $E[f_t e_{js}] = 0$ for all $t, s, j$

that the PC factor estimates satisfy

$$
\left(\frac{\hat{f}_t^{(0)}}{H_0} - f_t\right) = O_p\left(\frac{1}{\delta_{nT}}\right)
\qquad \text{uniformly in } t
\tag{5.8}
$$

This is the key result that justifies using PC estimates as starting values for the EM
algorithm, even for moderate $n$.

---

## 6. Estimation: MLE via the EM Algorithm

Full maximum-likelihood estimation of the DFM — treating the factors as missing data —
is implemented via the **Expectation-Maximisation (EM) algorithm**. Let $\theta =
\{\Lambda, \Phi_1, \ldots, \Phi_p, Q, H\}$ be the full parameter vector.

### 6.1 Complete-Data Log-Likelihood

The **complete-data log-likelihood** conditions on the latent factor path
$f_{1:T} = (f_1, \ldots, f_T)$:

$$
\ell_c(\theta; y, f) = -\frac{T}{2}\log|H| - \frac{1}{2}\sum_{t=1}^T (y_t - \Lambda f_t)' H^{-1} (y_t - \Lambda f_t)
$$
$$
\qquad - \frac{T-p}{2}\log|Q| - \frac{1}{2}\sum_{t=p+1}^T \eta_t(\theta)' Q^{-1} \eta_t(\theta)
+ \text{const}
\tag{6.1}
$$

where $\eta_t(\theta) = f_t - \Phi_1 f_{t-1} - \cdots - \Phi_p f_{t-p}$.

### 6.2 E-Step: Kalman Smoother

At iteration $k$, the E-step computes the **expected complete-data log-likelihood**
$Q(\theta; \theta^{(k)}) = E[\ell_c(\theta; y, f) \mid y; \theta^{(k)}]$ by running the
Kalman filter and RTS smoother to obtain the following sufficient statistics:

$$
P_{t|T} = \text{Var}(f_t \mid y_{1:T}; \theta^{(k)})
\tag{6.2a}
$$

$$
\hat{f}_{t|T} = E[f_t \mid y_{1:T}; \theta^{(k)}]
\tag{6.2b}
$$

$$
P_{t,t-1|T} = \text{Cov}(f_t, f_{t-1} \mid y_{1:T}; \theta^{(k)})
\tag{6.2c}
$$

The three cross-product matrices required by the M-step are:

$$
\mathbf{A} = \sum_{t=1}^T E[f_t f_t' \mid y; \theta^{(k)}]
= \sum_{t=1}^T \left(\hat{f}_{t|T} \hat{f}_{t|T}' + P_{t|T}\right)
\tag{6.2d}
$$

$$
\mathbf{B} = \sum_{t=2}^T E[f_t f_{t-1}' \mid y; \theta^{(k)}]
= \sum_{t=2}^T \left(\hat{f}_{t|T} \hat{f}_{t-1|T}' + P_{t,t-1|T}\right)
\tag{6.2e}
$$

$$
\mathbf{C} = \sum_{t=1}^T y_t \hat{f}_{t|T}'
\tag{6.2f}
$$

### 6.3 M-Step: Closed-Form Parameter Updates

Given the E-step sufficient statistics, the M-step maximises $Q(\theta; \theta^{(k)})$
separately for each parameter block. All updates are in **closed form**.

**Loading matrix $\Lambda$:**

$$
\hat\Lambda^{(k+1)} = \mathbf{C} \cdot \mathbf{A}^{-1}
\tag{6.3a}
$$

This is a generalised least-squares regression of $y_t$ on the expected factors.

**Idiosyncratic variance $H = \text{diag}(h_1, \ldots, h_n)$:**

$$
\hat{h}_i^{(k+1)} = \frac{1}{T}\left[\sum_{t=1}^T y_{it}^2
- \hat\Lambda_i^{(k+1)} \sum_{t=1}^T \hat{f}_{t|T} y_{it}\right]
\tag{6.3b}
$$

In matrix form: $\hat{H}^{(k+1)} = \frac{1}{T}\left[\sum_t y_t y_t' - \hat\Lambda^{(k+1)} \mathbf{C}'\right]$
with off-diagonal elements discarded.

**VAR coefficients $[\Phi_1, \ldots, \Phi_p]$ (VAR(1) case, $p = 1$):**

$$
\hat\Phi^{(k+1)} = \mathbf{B} \cdot \mathbf{A}_{-1}^{-1}
\tag{6.3c}
$$

where $\mathbf{A}_{-1} = \sum_{t=2}^T E[f_{t-1} f_{t-1}' \mid y]$. For $p > 1$ the
update generalises to a multivariate Yule-Walker-type equation involving the stacked
cross-products from the companion state.

**Factor innovation covariance $Q$:**

$$
\hat{Q}^{(k+1)} = \frac{1}{T-1}\left[\mathbf{A}_{2:T} - \hat\Phi^{(k+1)} \mathbf{B}'\right]
\tag{6.3d}
$$

where $\mathbf{A}_{2:T} = \sum_{t=2}^T E[f_t f_t' \mid y]$.

!!! definition "EM fixed-point property"
    At convergence $\theta^*$ the EM iterates satisfy the **score equations**:

    $$
    \frac{\partial \ell(\theta; y)}{\partial \theta}\bigg|_{\theta = \theta^*} = 0
    $$

    Since each M-step maximises $Q(\theta; \theta^{(k)})$ over the full parameter space,
    the EM algorithm guarantees $\ell(\theta^{(k+1)}) \geq \ell(\theta^{(k)})$ — the
    observed-data likelihood is non-decreasing at every iteration.

### 6.4 EM with Missing Data

One of the greatest practical advantages of the EM approach is transparent handling of
**missing observations**. When $y_{it}$ is missing at time $t$:

1. **E-step:** The Kalman filter uses only the available observations at time $t$ by
   zeroing out the corresponding rows of $Z_t$ and $d_t$ (or equivalently setting
   $H_{ii} \to \infty$ for missing series). The filter proceeds normally.

2. **M-step:** The M-step formulas remain identical because the expectation over
   missing data is automatically absorbed into $\hat{f}_{t|T}$ and $P_{t|T}$.

In the Kalman filter, at time $t$ with $n_t$ available observations (out of $n$), the
update step operates on a reduced system of dimension $n_t$. The innovation and its
covariance are formed only from the observed elements:

$$
v_t = y_t^{\text{obs}} - Z_t^{\text{obs}} \hat{f}_{t|t-1}
\qquad F_t = Z_t^{\text{obs}} P_{t|t-1} Z_t^{\text{obs}'} + H_t^{\text{obs}}
\tag{6.4}
$$

This allows **ragged-edge** panels (different series observed at different frequencies
or with different publication lags) to be handled natively — a key feature for
macroeconomic nowcasting (Giannone, Reichlin & Small, 2008).

### 6.5 Convergence Properties

!!! note "Convergence of EM for DFMs"
    The EM algorithm for Gaussian state-space models is guaranteed to converge to a
    **stationary point** of the observed-data likelihood. For DFMs:

    - Convergence is **monotone** in the likelihood.
    - The limit point is typically a **local maximum**; multiple restarts from different
      initialisations (e.g., different PC rotations) are recommended.
    - Convergence is declared when $|\ell^{(k+1)} - \ell^{(k)}| / |\ell^{(k)}| < \epsilon$
      (default $\epsilon = 10^{-6}$ in `kalmanbox`).
    - Near-singular $\hat{H}$ is a common pathology: applying a floor
      $h_i \geq \delta \cdot \hat\sigma_i^2$ prevents numerical collapse.

---

## 7. Number of Factors: Selection Criteria

Choosing $r$ — the number of dynamic factors — is the central model selection problem
in the DFM literature. Several approaches are available.

### 7.1 Bai & Ng (2002) Information Criteria

Let $V(r, \hat{F}^r)$ be the mean squared residual when $r$ factors are extracted by PC:

$$
V(r, \hat{F}^r) = \frac{1}{nT}\sum_{i=1}^n \sum_{t=1}^T \hat{e}_{it}^2
= \frac{1}{nT}\|Y - \hat{F}^r \hat{\Lambda}^{r'}\|_F^2
\tag{7.1}
$$

The Bai-Ng criteria add a penalty for each additional factor:

$$
\text{IC}_{p1}(r) = \ln V(r, \hat{F}^r) + r\, g_1(n, T)
\tag{7.2a}
$$

$$
\text{IC}_{p2}(r) = \ln V(r, \hat{F}^r) + r\, g_2(n, T)
\tag{7.2b}
$$

$$
\text{IC}_{p3}(r) = \ln V(r, \hat{F}^r) + r\, g_3(n, T)
\tag{7.2c}
$$

where the penalty functions are

$$
g_1(n, T) = \left(\frac{n + T}{nT}\right)\ln\!\left(\frac{nT}{n+T}\right)
\tag{7.3a}
$$

$$
g_2(n, T) = \left(\frac{n + T}{nT}\right)\ln(\delta_{nT}^2), \qquad \delta_{nT} = \min(\sqrt{n}, \sqrt{T})
\tag{7.3b}
$$

$$
g_3(n, T) = \frac{\ln(\delta_{nT}^2)}{\delta_{nT}^2}
\tag{7.3c}
$$

The estimator $\hat{r} = \arg\min_{0 \leq r \leq r_{\max}} \text{IC}_{pj}(r)$ is
consistent: $\hat{r} \xrightarrow{p} r_0$ as $n, T \to \infty$.

!!! note "Practical recommendation"
    Bai & Ng (2002) recommend $\text{IC}_{p2}$ in most settings. $\text{IC}_{p1}$ tends
    to select more factors in small samples; $\text{IC}_{p3}$ is more conservative.
    Setting $r_{\max} \leq \lfloor\min(n, T)^{1/3}\rfloor$ as a crude upper bound is
    a safe default.

### 7.2 Scree Plot Interpretation

The scree plot of ordered eigenvalues $\hat\lambda_1 \geq \hat\lambda_2 \geq \cdots \geq
\hat\lambda_n$ of $\hat\Sigma_y$ shows a characteristic "elbow" at $r = r_0$ under the
factor model: the first $r_0$ eigenvalues diverge as $n \to \infty$ (they are $O(n)$)
while the remaining eigenvalues are $O(1)$.

The **scree test** chooses $\hat{r}$ at the point where the gap $\hat\lambda_j -
\hat\lambda_{j+1}$ is largest relative to subsequent gaps. This is a graphical
heuristic; the IC criteria provide a formal analogue.

### 7.3 Onatski (2010) Edge Distribution Estimator

Onatski (2010) proposes an estimator based on adjacent eigenvalue differences,
motivated by random matrix theory. Define

$$
\delta_j = \hat\lambda_j - \hat\lambda_{j+1}, \qquad j = 1, \ldots, r_{\max}
\tag{7.4}
$$

Under the null of $r_0$ factors, the standardised differences $\{\delta_j : j > r_0\}$
follow the Tracy-Widom distribution at the edge of the Marchenko-Pastur law. The
estimator $\hat{r}$ is the number of eigenvalues that "stick out" above the noise
edge, determined by an iterative thresholding algorithm:

$$
\hat{r} = \max\{j \leq r_{\max} : \hat\lambda_j - \hat\lambda_{j+1} \geq \delta\}
\tag{7.5}
$$

where $\delta$ is chosen based on the Tracy-Widom critical value. This estimator is
consistent when $n/T \to c \in (0, \infty)$.

### 7.4 Ahn & Horenstein (2013) Eigenvalue Ratio Test

Ahn & Horenstein (2013) propose two criteria based on **ratios** of adjacent
eigenvalues:

$$
\widehat{ER}(r) = \frac{\hat\mu_r}{\hat\mu_{r+1}}, \qquad r = 1, \ldots, r_{\max} - 1
\tag{7.6a}
$$

$$
\widehat{GR}(r) = \frac{\ln(\hat\mu_r) - \ln(\hat\mu_{r+1})}{\ln(\hat\mu_{r+1}) - \ln(\hat\mu_{r+2})}
\tag{7.6b}
$$

where $\hat\mu_j = \hat\lambda_j / \hat\lambda_{r_{\max}+1}$ are normalised eigenvalues.
The estimators are $\hat{r}_{ER} = \arg\max_{r} \widehat{ER}(r)$ and
$\hat{r}_{GR} = \arg\max_r \widehat{GR}(r)$. Both are consistent under similar
conditions to the Bai-Ng criteria.

### 7.5 Practical Guidance

!!! example "Factor selection workflow"
    A robust practical strategy combines multiple criteria:

    1. **Scree plot:** visual elbow at $r = r_{\text{scree}}$.
    2. **Bai-Ng IC$_{p2}$:** formal minimiser $r_{\text{BN}}$.
    3. **Eigenvalue ratio (ER):** $r_{\text{ER}}$ as robustness check.
    4. **Out-of-sample fit:** compare pseudo-out-of-sample $R^2$ for $r = r_{\text{BN}} \pm 1$.
    5. **Economic interpretability:** do the factor estimates correspond to recognisable
       aggregates (GDP, inflation, risk premium)?

    If all criteria agree, choose that $r$. If they diverge, prefer the criterion with
    the best out-of-sample track record for your specific application domain.

---

## 8. Asymptotic Properties

### 8.1 Consistency of PC Factor Estimates

The foundational consistency result for PC factor estimates is (Bai, 2003):

!!! definition "Theorem (Bai, 2003, Theorem 1)"
    Under Assumptions (A1)–(A4) of Section 5.5 and with $r$ factors correctly specified,
    there exists a sequence of $r \times r$ rotation matrices $H_{nT}$ such that

    $$
    \frac{1}{T}\sum_{t=1}^T \|\hat{f}_t - H_{nT} f_t\|^2 = O_p\left(\frac{1}{\delta_{nT}^2}\right)
    \tag{8.1}
    $$

    where $\delta_{nT} = \min(\sqrt{n}, \sqrt{T})$.

This means PC-estimated factors converge to the true factors (up to rotation) at rate
$\delta_{nT}$. The rate is the minimum of $\sqrt{n}$ and $\sqrt{T}$: cross-sectional
dimension reduces approximation error, but so does temporal dimension.

### 8.2 Rate of Convergence

The rate $\delta_{nT}^{-1}$ has two regimes:

- **$n \gg T$ (wide panel):** $\delta_{nT} = \sqrt{T}$ and estimation error is driven
  by time-series length. Adding more cross-sectional units beyond $T^2$ does not
  improve the factor estimates.
- **$T \gg n$ (long panel):** $\delta_{nT} = \sqrt{n}$ and estimation error is driven
  by the number of series. Adding more time periods beyond $n^2$ does not improve
  factor estimates.
- **$n \approx T$ (square panel):** both dimensions contribute equally and $\delta_{nT}
  = \sqrt{n} = \sqrt{T}$.

The rate $\delta_{nT}^{-1}$ is **faster than $T^{-1/2}$** whenever $n \to \infty$:
more cross-sectional units reduce idiosyncratic contamination, acting as additional
instruments for the factors.

### 8.3 Central Limit Theorem for Factor Estimates

!!! definition "Theorem (Bai, 2003, Theorem 3)"
    Under the conditions of Theorem 1, if $\sqrt{T}/n \to 0$, then

    $$
    \sqrt{T}\left(\hat{f}_t - H_{nT} f_t\right)
    \xrightarrow{d} \mathcal{N}\!\left(0,\; V^{-1} \Phi_{ft} V^{-1}\right)
    \tag{8.2}
    $$

    where $V = \text{plim}\left(\frac{\Lambda'\Lambda}{n}\right) \cdot D^{-1}$ involves the
    leading eigenvectors and $\Phi_{ft}$ is the asymptotic covariance of the
    time-average score.

The condition $\sqrt{T}/n \to 0$ ensures the cross-sectional dimension dominates,
eliminating the bias from idiosyncratic noise. In practice this means $n$ should be
substantially larger than $\sqrt{T}$.

### 8.4 Consistency of Loading Estimates

For the loading matrix, Bai (2003) shows:

$$
\sqrt{T}\left(\hat\Lambda_i - H_{nT}^{-1'} \Lambda_i\right)
\xrightarrow{d} \mathcal{N}(0, \Phi_{\Lambda_i})
\tag{8.3}
$$

where $\Phi_{\Lambda_i}$ depends on the fourth-order moments of the factors and
idiosyncratic errors. Inference on $\Lambda_i$ after PC estimation is thus valid with
conventional standard errors when $T$ is large.

### 8.5 Factor Rotation Convergence

The rotation matrix $H_{nT}$ converges to a limit $H_0$ at rate $\delta_{nT}^{-1}$:

$$
\|H_{nT} - H_0\| = O_p\!\left(\frac{1}{\delta_{nT}}\right)
\tag{8.4}
$$

This means structural interpretation of rotated factors (e.g., identifying factor 1 as
the "business cycle") becomes more reliable as both $n$ and $T$ grow.

---

## 9. Extensions

### 9.1 DFM with Stochastic Volatility

In financial applications, factor volatility is time-varying. The **DFM-SV** model
replaces the constant $Q$ with a diagonal stochastic volatility process:

$$
f_t = \Phi f_{t-1} + \Sigma_t^{1/2} \eta_t, \qquad \eta_t \sim \mathcal{N}(0, I_r)
\tag{9.1a}
$$

$$
\log h_{jt} = \mu_j + \phi_j (\log h_{j,t-1} - \mu_j) + \xi_{jt},
\qquad \xi_{jt} \sim \mathcal{N}(0, \sigma_{\xi j}^2)
\tag{9.1b}
$$

where $h_{jt}$ is the conditional variance of factor $j$ at time $t$. Estimation
requires particle filtering or MCMC (the log-normal form makes the Kalman filter
inapplicable directly). A common approximation linearises the log-variance equation
using the Kim-Shephard-Chib (1998) mixture approximation.

### 9.2 Mixed-Frequency DFM

The **Mariano-Murasawa (2003)** model handles the common case where some variables
(e.g., quarterly GDP) are observed at lower frequency than others (monthly surveys,
daily financial data). The key insight is to represent the low-frequency variable as
a **temporal aggregation** (weighted average or flow sum) of a latent high-frequency
counterpart, then include both in the state vector.

For a quarterly variable $y_t^Q$ as the sum of three monthly latent values:

$$
y_t^Q = \frac{1}{3}(y_t^* + y_{t-1}^* + y_{t-2}^*)
\tag{9.2}
$$

Stacking $y_t^* = \lambda' f_t + e_t$ in the state vector and treating quarterly
observations as missing except at the observation months yields a standard SSM solvable
by the Kalman filter. This is the foundation of the Atlanta Fed GDPNow and ECB EuroCOIN
nowcasting models.

!!! note "Mixed-frequency in `kalmanbox`"
    The `MixedFrequencyDFM` subclass in `kalmanbox` handles monthly/quarterly and
    weekly/monthly panels automatically. Specify `freq_map={'GDP': 'Q', 'IP': 'M'}`
    and the state vector augmentation is constructed internally.

### 9.3 DFM with Blocks (Structural Factors)

In large international or regional datasets, a natural hierarchical factor structure
exists: a **global factor** drives all series, **regional factors** drive country
clusters, and **idiosyncratic shocks** are country-specific. The **block factor model**
(Kose, Otrok & Whiteman, 2003; Banbura et al., 2010) structures the loading matrix as:

$$
y_{it} = \lambda_i^{(g)} f_t^{(g)} + \lambda_i^{(b_i)} f_t^{(b_i)} + e_{it}
\tag{9.3}
$$

where $f_t^{(g)}$ is the global factor, $f_t^{(b_i)}$ is the block factor for group
$b_i$ containing series $i$, and the cross-loading restrictions ($\lambda_i^{(b)}=0$
for $b \neq b_i$) are imposed directly in the loading matrix.

Identification within blocks requires the same lower-triangular restrictions described
in Section 3, applied separately within each block. The SSM form remains valid with a
suitably partitioned loading matrix.

### 9.4 Large DFM with Shrinkage

For very large $n$ (hundreds of series), even the diagonal $H$ has $n$ free parameters.
Banbura, Giannone & Reichlin (2010) apply **Minnesota-style shrinkage priors** on the
loading matrix to regularise estimation:

$$
p(\text{vec}(\Lambda)) = \mathcal{N}(0, \nu^{-1} I_{nr})
\tag{9.4}
$$

The EM M-step for $\Lambda$ then becomes a ridge regression:

$$
\hat\Lambda = \mathbf{C}\left(\mathbf{A} + \nu H\right)^{-1}
\tag{9.5}
$$

where $\nu > 0$ is the shrinkage parameter (selected by cross-validation or
empirical Bayes). For large $n$ this significantly improves finite-sample performance
by preventing overfitting to idiosyncratic variation.

---

## 10. kalmanbox Implementation Notes

### 10.1 Class Overview

```python
from kalmanbox.models import DynamicFactorModel

help(DynamicFactorModel)
```

The `DynamicFactorModel` class inherits from `kalmanbox.models.StateSpaceModel` and
provides:

- EM estimation with diagonal or unrestricted $H$
- PCA initialisation (default) or user-supplied starting values
- Automatic companion-form construction for `factor_order > 1`
- Missing-data handling via masked arrays or `np.nan` sentinels
- Bai-Ng IC$_{p1}$, IC$_{p2}$, IC$_{p3}$ for automatic factor selection
- RTS-smoothed factor estimates with uncertainty bands

### 10.2 Basic Usage: 3 Factors, 20 Observables

```python
import numpy as np
from kalmanbox.models import DynamicFactorModel

# Simulate a 3-factor panel (n=20 series, T=200 periods)
rng = np.random.default_rng(42)
T, n, r = 200, 20, 3

# True parameters
Lambda_true = rng.standard_normal((n, r))
Phi_true = np.diag([0.8, 0.6, 0.4])       # diagonal VAR(1)
Q_true = np.eye(r)
H_true = np.diag(rng.uniform(0.5, 1.5, n))

# Simulate factors and observations
f = np.zeros((T, r))
for t in range(1, T):
    f[t] = Phi_true @ f[t - 1] + rng.multivariate_normal(np.zeros(r), Q_true)
y = f @ Lambda_true.T + rng.multivariate_normal(np.zeros(n), H_true, size=T)

# Fit DFM via EM
model = DynamicFactorModel(
    n_factors=3,          # number of dynamic factors r
    factor_order=1,       # VAR order p
    idiosyncratic='diagonal',   # exact factor model
    em_max_iter=200,
    em_tol=1e-6,
    init='pca',           # PCA initialisation
)
result = model.fit(y)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Converged in {result.n_iter} iterations")
print(f"Loading matrix shape: {result.loadings.shape}")  # (20, 3)
```

### 10.3 Automatic Factor Selection

```python
from kalmanbox.models import DynamicFactorModel
import matplotlib.pyplot as plt

# Select number of factors using Bai-Ng IC_p2
ic_values = {}
for r in range(1, 8):
    m = DynamicFactorModel(n_factors=r, factor_order=1)
    res = m.fit(y)
    ic_values[r] = res.ic_p2

best_r = min(ic_values, key=ic_values.get)
print(f"Selected r = {best_r} factors by IC_p2")

# Also available: res.ic_p1, res.ic_p3
# Scree plot via res.eigenvalues
plt.plot(range(1, n + 1), result.eigenvalues, 'o-')
plt.axvline(best_r, color='r', linestyle='--', label=f'r={best_r}')
plt.xlabel("Factor index")
plt.ylabel("Eigenvalue")
plt.title("Scree plot")
plt.legend()
```

### 10.4 Nowcasting with Missing Data (Ragged-Edge Panel)

A key use case is **macroeconomic nowcasting**: estimate the current-quarter value of
GDP using monthly hard and soft indicators, some of which are not yet released.

```python
import numpy as np
from kalmanbox.models import DynamicFactorModel

# y_ragged: (T, n) array with np.nan for unreleased observations
# Typical structure: last 1-3 months have missing entries for lagging indicators
y_ragged = y.copy()
y_ragged[-1, [0, 2, 5, 7, 11]] = np.nan   # last period: 5 series not yet published
y_ragged[-2, [0, 2]] = np.nan              # second-to-last: 2 series still missing

# Fit model — missing data handled transparently in Kalman filter
model = DynamicFactorModel(n_factors=3, factor_order=1, idiosyncratic='diagonal')
result = model.fit(y_ragged)

# Smoothed factor estimates at all dates (including incomplete periods)
f_smoothed = result.smoothed_factors          # shape (T, r)
f_smoothed_var = result.smoothed_factors_cov  # shape (T, r, r)

# Nowcast for the last period: project onto GDP loading
gdp_loading = result.loadings[0]              # loading of GDP series (index 0)
nowcast = gdp_loading @ f_smoothed[-1]
nowcast_se = np.sqrt(gdp_loading @ f_smoothed_var[-1] @ gdp_loading)

print(f"GDP nowcast: {nowcast:.3f} +/- {1.96 * nowcast_se:.3f} (95% interval)")

# News decomposition: contribution of each new data release
news = result.news_decomposition(y_ragged, y)   # requires previous and updated dataset
print("Factor contribution by series:")
for i, contrib in enumerate(news['contributions']):
    print(f"  Series {i}: {contrib:+.4f}")
```

### 10.5 Higher-Order Factor Dynamics

```python
# VAR(2) factor dynamics: companion state has dimension r*p = 3*2 = 6
model_p2 = DynamicFactorModel(
    n_factors=3,
    factor_order=2,     # companion state dimension = 6
    idiosyncratic='diagonal',
)
result_p2 = model_p2.fit(y)

# Access companion-form system matrices
print("Companion T:", result_p2.transition_matrix.shape)  # (6, 6)
print("Design Z:", result_p2.design_matrix.shape)         # (20, 6)

# Compare information criteria
print(f"VAR(1) IC_p2: {result.ic_p2:.3f}")
print(f"VAR(2) IC_p2: {result_p2.ic_p2:.3f}")
```

---

## References

| Key | Citation |
|:----|:---------|
| Bai (2003) | Bai, J. (2003). Inferential theory for factor models of large dimensions. *Econometrica*, 71(1), 135–171. |
| Bai & Ng (2002) | Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191–221. |
| Bai & Ng (2007) | Bai, J., & Ng, S. (2007). Determining the number of primitive shocks in factor models. *Journal of Business & Economic Statistics*, 25(1), 52–60. |
| Bai & Ng (2013) | Bai, J., & Ng, S. (2013). Principal components estimation and identification of static factors. *Journal of Econometrics*, 176(1), 18–29. |
| Banbura et al. (2010) | Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. *Journal of Applied Econometrics*, 25(1), 71–92. |
| Doz et al. (2012) | Doz, C., Giannone, D., & Reichlin, L. (2012). A quasi-maximum likelihood approach for large, approximate dynamic factor models. *Review of Economics and Statistics*, 94(4), 1014–1024. |
| Forni et al. (2000) | Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The generalized dynamic-factor model: Identification and estimation. *Review of Economics and Statistics*, 82(4), 540–554. |
| Giannone et al. (2008) | Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665–676. |
| Mariano & Murasawa (2003) | Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of business cycles based on monthly and quarterly series. *Journal of Applied Econometrics*, 18(4), 427–443. |
| Onatski (2010) | Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. *Review of Economics and Statistics*, 92(4), 1004–1016. |
| Ahn & Horenstein (2013) | Ahn, S. C., & Horenstein, A. R. (2013). Eigenvalue ratio test for the number of factors. *Econometrica*, 81(3), 1203–1227. |
| Stock & Watson (2002) | Stock, J. H., & Watson, M. W. (2002). Macroeconomic forecasting using diffusion indexes. *Journal of Business & Economic Statistics*, 20(2), 147–162. |
| Chamberlain & Rothschild (1983) | Chamberlain, G., & Rothschild, M. (1983). Arbitrage, factor structure, and mean-variance analysis on large asset markets. *Econometrica*, 51(5), 1281–1304. |

---

## See Also

- [State-Space Model Theory](state-space-theory.md) — General LG-SSM notation, system matrices, stability
- [MLE Theory](mle-theory.md) — Prediction-error likelihood, score, Fisher information
- [Smoothing Theory](smoothing-theory.md) — RTS smoother derivation used in the EM E-step
- [DFM User Guide](../user-guide/advanced/dfm.md) — Practical API reference with additional examples
- [MLE User Guide](../user-guide/kalman/mle.md) — Optimisation workflow and convergence diagnostics
