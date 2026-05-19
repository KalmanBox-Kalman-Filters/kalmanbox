# Diffuse Initialization

When the state vector contains **non-stationary** components — random walks,
integrated processes, stochastic trends — there is no natural finite prior
$a_1, P_1$ for the initial state $\alpha_1$. Two strategies address this:

| Strategy | Description | Use case |
|----------|-------------|----------|
| **Exact diffuse** | Analytically tracks the $\kappa \to \infty$ limit | Production; correct likelihood |
| **Approximate diffuse** | Sets $P_0 = \kappa I$ with $\kappa$ large but finite | Prototyping; simple code paths |

`kalmanbox` implements the **exact diffuse** approach of De Jong (1991) and
Koopman (1997), as presented in Durbin & Koopman (2012, §5).

---

## The problem: infinite prior variance

Consider a Local Level model with state $\mu_t$:

$$
\mu_{t+1} = \mu_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
$$

The level $\mu_1$ is unknown. Setting $P_1 = \kappa I$ with $\kappa = 10^7$
inflates the first innovation variance $F_1 = Z P_1 Z' + H$ by $\kappa$,
distorting the log-likelihood for the first several observations.

The exact approach avoids the choice of $\kappa$ entirely.

---

## Exact diffuse initialization: theory

### Decomposition of initial covariance

Partition the initial covariance into a **stationary** part $P_*$ and a
**diffuse** part scaled by $\kappa$:

$$
P_1 = P_* + \kappa P_\infty, \qquad \kappa \to \infty
$$

- $P_\infty$ projects onto the **diffuse subspace** — the span of non-stationary
  state components.
- $P_*$ is the proper finite prior on the stationary complement.

For a purely non-stationary model such as the Local Level,
$P_\infty = I$ (scalar) and $P_* = 0$.

### Augmented recursion

The standard filter is augmented with two auxiliary covariance matrices,
$P_t^{(\infty)}$ and $P_t^{(*)}$, that satisfy separate recursions.

**Prediction step** (same as standard):

$$
a_{t|t-1} = T\, a_{t|t}, \qquad
P_{t|t-1}^{(*)} = T P_{t|t}^{(*)} T', \qquad
P_{t|t-1}^{(\infty)} = T P_{t|t}^{(\infty)} T'
$$

**Innovation and gain**:

$$
v_t = y_t - Z\, a_{t|t-1}
$$

$$
F_t^{(\infty)} = Z P_{t|t-1}^{(\infty)} Z', \qquad
F_t^{(*)} = Z P_{t|t-1}^{(*)} Z' + H
$$

**Update step** — two cases:

*Case 1: $F_t^{(\infty)} > 0$ (still diffuse)*

$$
K_t^{(\infty)} = P_{t|t-1}^{(\infty)} Z' \left(F_t^{(\infty)}\right)^{-1}
$$

$$
\begin{aligned}
a_{t|t} &= a_{t|t-1} + K_t^{(\infty)} v_t \\
P_{t|t}^{(\infty)} &= P_{t|t-1}^{(\infty)} - K_t^{(\infty)} Z P_{t|t-1}^{(\infty)} \\
P_{t|t}^{(*)} &= P_{t|t-1}^{(*)} - K_t^{(\infty)} Z P_{t|t-1}^{(*)}
             - P_{t|t-1}^{(*)} Z' K_t^{(\infty)'} \\
             &\quad + K_t^{(\infty)} F_t^{(*)} K_t^{(\infty)'}
\end{aligned}
$$

*Case 2: $F_t^{(\infty)} = 0$ (non-diffuse; standard update)*

$$
K_t = P_{t|t-1}^{(*)} Z' \left(F_t^{(*)}\right)^{-1}
$$

$$
a_{t|t} = a_{t|t-1} + K_t v_t, \qquad
P_{t|t}^{(*)} = P_{t|t-1}^{(*)} - K_t Z P_{t|t-1}^{(*)}
$$

### Transition: diffuse → non-diffuse

The recursion transitions from Case 1 to Case 2 as soon as $P_t^{(\infty)}$
reaches numerical zero. This happens after at most $d = \mathrm{rank}(P_\infty)$
steps, where $d$ is the number of non-stationary state components.

!!! note "Number of diffuse steps"
    For a Local Level model, $d = 1$ (one random-walk level).
    For a Local Linear Trend, $d = 2$ (level and slope are both non-stationary).
    For a BSM with $s$ seasonal dummies, $d = 2 + (s-1)$.

After step $d$, the algorithm proceeds identically to the standard Kalman filter.

---

## Diffuse log-likelihood

The standard prediction-error log-likelihood sums contributions from all $t$:

$$
\ell(\theta) = -\frac{n}{2}\log 2\pi
               - \frac{1}{2}\sum_{t=1}^n \left(\log |F_t| + v_t' F_t^{-1} v_t\right)
$$

During the first $d$ diffuse steps, $F_t^{(\infty)} \ne 0$, so the
log-determinant contribution is modified:

$$
\ell_{\text{diffuse}}(\theta) = -\frac{n-d}{2}\log 2\pi
  - \frac{1}{2}\sum_{t=1}^d \log \left|F_t^{(\infty)}\right|
  - \frac{1}{2}\sum_{t=d+1}^n \left(\log |F_t^{(*)}| + v_t' F_t^{(*)-1} v_t\right)
$$

!!! warning "Do not maximize the approximate log-likelihood"
    Using $P_1 = \kappa I$ gives a log-likelihood that includes spurious terms of
    order $\log \kappa$ for the first few observations. These vanish as
    $\kappa \to \infty$ but bias small-sample MLE if $\kappa$ is just "large."
    Always use exact diffuse initialization when the model contains non-stationary
    components.

---

## Approximate diffuse initialization

When you want a quick experiment or are implementing a custom model without
exact-diffuse support, `kalmanbox` accepts a large-$\kappa$ approximation:

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[0.25]])
H = np.array([[1.0]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

kf_approx = KalmanFilter(
    ssr,
    initialization="approximate_diffuse",
    kappa=1e6,          # large but finite
)
```

!!! tip "When approximate diffuse is acceptable"
    - Quick experimentation on long series ($n \gg d$) where the first few
      observations have negligible influence.
    - Custom filter implementations where you want a simple code path.
    - Do **not** use it when $n$ is small or when comparing models by likelihood.

---

## API reference

=== "Exact diffuse (default for structural models)"

    ```python
    from kalmanbox import KalmanFilter, StateSpaceRepresentation
    from kalmanbox.estimation import DiffuseInitialization

    # Option 1: string shorthand
    kf = KalmanFilter(ssr, initialization="diffuse")

    # Option 2: explicit object (lets you inspect P_infty)
    init = DiffuseInitialization.from_representation(ssr)
    print(f"Diffuse rank: {init.rank}")          # e.g. 2 for Local Linear Trend
    print(f"P_infty:\n{init.P_infty}")

    kf = KalmanFilter(ssr, initialization=init)
    out = kf.run(y)

    print(f"Diffuse steps: {out.diffuse_steps}")          # d
    print(f"Log-likelihood: {out.loglike:.4f}")            # diffuse ℓ
    print(f"Diffuse log-like: {out.loglike_diffuse:.4f}") # same thing
    ```

=== "Approximate diffuse"

    ```python
    from kalmanbox import KalmanFilter

    kf = KalmanFilter(ssr, initialization="approximate_diffuse", kappa=1e7)
    out = kf.run(y)
    ```

=== "Stationary prior (known P1)"

    ```python
    import numpy as np
    from kalmanbox import KalmanFilter
    from kalmanbox.estimation import StationaryInitialization

    # Auto-compute unconditional covariance via Lyapunov equation
    init = StationaryInitialization.from_representation(ssr)

    # Or supply manually
    P1 = np.array([[2.0]])
    init = StationaryInitialization(a1=np.zeros(1), P1=P1)

    kf = KalmanFilter(ssr, initialization=init)
    ```

---

## Practical examples

### Example 1: Local Level model with exact diffuse

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

rng = np.random.default_rng(42)
n   = 200

# Simulate a Local Level: σ_η² = 0.5, σ_ε² = 1.0
sigma_eta, sigma_eps = 0.5, 1.0
eta = rng.normal(scale=np.sqrt(sigma_eta), size=n)
eps = rng.normal(scale=np.sqrt(sigma_eps), size=n)
mu  = np.cumsum(eta)
y   = mu + eps

# State-space representation
T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[sigma_eta]])
H = np.array([[sigma_eps]])

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(y)

print(f"Diffuse steps  : {out.diffuse_steps}")   # 1
print(f"Log-likelihood : {out.loglike:.4f}")
```

### Example 2: Local Linear Trend — two diffuse components

```python
from kalmanbox.structural import LocalLinearTrend
import numpy as np

rng = np.random.default_rng(0)
n   = 300

# Simulate level + slope
slope = np.cumsum(rng.normal(scale=0.05, size=n))
level = np.cumsum(slope + rng.normal(scale=0.1, size=n))
y     = level + rng.normal(scale=1.0, size=n)

model   = LocalLinearTrend(y)
results = model.fit()        # uses exact diffuse automatically

kf_out = results.filter_output
print(f"Diffuse steps : {kf_out.diffuse_steps}")   # 2
print(f"AIC           : {results.aic:.2f}")
```

### Example 3: comparing exact vs approximate diffuse likelihood

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[1.0]])
H = np.array([[1.0]])
ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)

rng = np.random.default_rng(7)
y   = np.cumsum(rng.normal(size=50)) + rng.normal(size=50)

kf_exact   = KalmanFilter(ssr, initialization="diffuse")
kf_approx1 = KalmanFilter(ssr, initialization="approximate_diffuse", kappa=1e4)
kf_approx2 = KalmanFilter(ssr, initialization="approximate_diffuse", kappa=1e7)

ll_exact   = kf_exact.run(y).loglike
ll_approx1 = kf_approx1.run(y).loglike
ll_approx2 = kf_approx2.run(y).loglike

print(f"Exact diffuse   : {ll_exact:.4f}")
print(f"Approx κ=1e4    : {ll_approx1:.4f}  (Δ = {ll_approx1-ll_exact:+.4f})")
print(f"Approx κ=1e7    : {ll_approx2:.4f}  (Δ = {ll_approx2-ll_exact:+.4f})")
# Approx converges to exact as κ → ∞, but biased for finite κ
```

---

## Which components require diffuse initialization?

| Model component | Non-stationary? | Diffuse rank contribution |
|----------------|:---------------:|:-------------------------:|
| Local Level $\mu_t$ | ✅ | 1 |
| Local Linear Trend: level | ✅ | 1 |
| Local Linear Trend: slope | ✅ | 1 |
| BSM seasonal dummies ($s$ seasons) | ✅ | $s-1$ |
| Trigonometric seasonal | ❌ (damped) | 0 |
| Stationary AR($p$) | ❌ | 0 |
| Damped cycle ($\rho < 1$) | ❌ | 0 |
| Unit-root cycle ($\rho = 1$) | ✅ | 2 |
| Regression (time-varying TVP) | ✅ if RW | 1 per coefficient |

!!! info "Automatic detection in structural models"
    `LocalLevel`, `LocalLinearTrend`, `BSM`, and `UCM` detect non-stationary
    components automatically and configure exact diffuse initialization without
    any user action. Manual configuration (as in the examples above) is needed
    only when building custom `StateSpaceRepresentation` objects.

---

## Numerical considerations

### When $P_t^{(\infty)}$ does not collapse

In theory $P_t^{(\infty)} \to 0$ after $d$ steps. In practice, floating-point
errors can leave tiny residuals. `kalmanbox` applies a **collapse threshold**:

```python
kf = KalmanFilter(ssr, initialization="diffuse", diffuse_tol=1e-10)
```

If `P_t^{(\infty)}` has all elements below `diffuse_tol` it is set to zero and
the algorithm switches to the standard recursion.

### Ill-conditioned $F_t^{(\infty)}$

For multivariate models with some observed variables missing, $F_t^{(\infty)}$
may be rank-deficient at the first few steps. `kalmanbox` uses the
Moore–Penrose pseudoinverse in this case, consistent with Koopman (1997).

---

## Related pages

- [Missing Data](missing-data.md) — interaction with missing observations during diffuse steps
- [MLE Estimation](mle.md) — why the diffuse log-likelihood matters for parameter estimation
- [Kalman Filter](kalman-filter.md) — the standard (non-diffuse) filter recursion
- [RTS Smoother](rts-smoother.md) — smoothing through the initialization period
- [Theory: Likelihood Computation](../../theory/likelihood.md)
- [API: estimation.diffuse](../../api/estimation.md)

### References

- De Jong, P. (1991). The diffuse Kalman filter. *Annals of Statistics*, 19(2), 1073–1083.
- Koopman, S. J. (1997). Exact initial Kalman filtering and smoothing for nonstationary time series models. *Journal of the American Statistical Association*, 92(440), 1630–1638.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press. §5.
