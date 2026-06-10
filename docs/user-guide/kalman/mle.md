# Maximum Likelihood Estimation

State-space models are estimated by maximizing the **log-likelihood** computed
from the Kalman filter's prediction-error decomposition. This page covers the
full MLE workflow: the likelihood function, parameter transformations,
optimization, standard errors, and information criteria.

---

## Prediction-error decomposition

The Kalman filter produces, at each time $t$, the innovation
$v_t = y_t - Z_t a_{t|t-1}$ and its variance $F_t = Z_t P_{t|t-1} Z_t' + H_t$.
These are the ingredients of the **prediction-error decomposition** of the
Gaussian log-likelihood:

$$
\ell(\theta) = -\frac{n^*}{2}\log(2\pi)
              - \frac{1}{2}\sum_{t \in \mathcal{T}} \left(
                  \log |F_t(\theta)| + v_t(\theta)'\, F_t(\theta)^{-1}\, v_t(\theta)
                \right)
$$

where:

- $\theta$ is the vector of unknown parameters (variances, loadings, etc.)
- $\mathcal{T}$ is the set of time points with observed $y_t$
- $n^* = |\mathcal{T}|$ is the effective sample size (missing observations excluded)
- $F_t$ and $v_t$ are both functions of $\theta$ through the system matrices

!!! note "Diffuse adjustment"
    When the model has non-stationary components, the first $d$ diffuse steps
    use $F_t^{(\infty)}$ in the log-determinant. See
    [Diffuse Initialization](diffuse.md) for details.

### Concentrated likelihood (univariate models)

For a univariate model with a single variance parameter $\sigma^2$, the
likelihood can be **concentrated** by analytically profiling out $\sigma^2$:

$$
\hat{\sigma}^2 = \frac{1}{n^*}\sum_{t \in \mathcal{T}} \frac{v_t^2}{F_t / \sigma^2}
$$

This reduces the optimization dimension by one and often improves convergence.
`kalmanbox` uses the concentrated likelihood automatically for scalar models.

---

## Parameter transformations

Raw optimization over $(0, \infty)$ or $(-1, 1)$ is numerically difficult.
`kalmanbox` works in an **unconstrained** space and applies bijective maps:

| Parameter | Constraint | Transform $\psi \to \theta$ | Inverse |
|-----------|-----------|----------------------------|---------|
| Variance $\sigma^2$ | $> 0$ | $\sigma^2 = e^{\psi}$ | $\psi = \log \sigma^2$ |
| Correlation $\rho$ | $(-1, 1)$ | $\rho = \tanh(\psi)$ | $\psi = \mathrm{atanh}(\rho)$ |
| AR coefficient (stationary) | $(-1, 1)$ | $\phi = \tanh(\psi)$ | $\psi = \mathrm{atanh}(\phi)$ |
| Damping factor $\rho_c$ | $(0, 1)$ | $\rho_c = \sigma(\psi)$ | $\psi = \mathrm{logit}(\rho_c)$ |
| Factor loading $\lambda$ | unconstrained | identity | identity |

These transformations ensure that the optimizer (operating on $\psi$) never
produces an invalid model, and gradients remain well-conditioned.

### Manual parameter transformation

If you build a custom model you can register transformations explicitly:

```python
from kalmanbox.estimation import ParameterTransforms
import numpy as np

# Define your parameter vector
# theta = [log(sigma_eta^2), log(sigma_eps^2)]
transforms = ParameterTransforms(
    to_unconstrained=lambda theta: np.log(theta),
    to_constrained=lambda psi: np.exp(psi),
)
```

---

## Optimization

### Objective function

`kalmanbox` minimizes the **negative** log-likelihood using `scipy.optimize`:

```python
from scipy.optimize import minimize

def neg_loglike(psi: np.ndarray) -> float:
    theta = transforms.to_constrained(psi)
    kf    = KalmanFilter(build_ssr(theta), initialization="diffuse")
    out   = kf.run(y)
    return -out.loglike

result = minimize(neg_loglike, x0=psi0, method="L-BFGS-B")
```

The `MLEstimator` class wraps this pattern and adds gradient computation,
multiple starts, and result post-processing.

### Supported methods

| Method | Gradient? | Recommended for |
|--------|-----------|----------------|
| `"L-BFGS-B"` | Numerical (finite diff.) | Default; most models |
| `"Nelder-Mead"` | No | Discontinuous likelihoods |
| `"BFGS"` | Numerical | Small models, no bounds |
| `"Newton-CG"` | Analytical (if provided) | When exact Hessian is available |

---

## Full MLE API

### High-level: `model.fit()`

Pre-built structural models expose `fit()` which handles everything:

```python
from kalmanbox import LocalLevel, BSM
from kalmanbox.datasets import load_nile, load_airline

# ── Local Level ───────────────────────────────────────────────────────────────
nile    = load_nile()
model   = LocalLevel(nile["volume"].to_numpy())
results = model.fit(method="mle", disp=True)

print(results.summary())
```

```
                     Local Level Model Results
=============================================================
Dep. Variable:   volume     Log-Likelihood:  -632.537
No. Observations: 100       AIC:             1269.073
Df Model:          2        BIC:             1274.284
                             HQIC:            1271.184
=============================================================
           Estimate   Std.Err    z-stat    p-value
sigma2_eta  1469.01    363.15     4.044   0.0001
sigma2_eps   471.44    181.96     2.591   0.0096
=============================================================
```

### Low-level: `MLEstimator`

For custom models:

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation
from kalmanbox.estimation import MLEstimator, ParameterSpec

rng = np.random.default_rng(1)
n   = 200
y   = np.cumsum(rng.normal(scale=1.0, size=n)) + rng.normal(scale=0.5, size=n)

def build_model(params: np.ndarray) -> KalmanFilter:
    sigma_eta, sigma_eps = params
    T = np.array([[1.0]])
    Z = np.array([[1.0]])
    R = np.array([[1.0]])
    Q = np.array([[sigma_eta**2]])
    H = np.array([[sigma_eps**2]])
    ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
    return KalmanFilter(ssr, initialization="diffuse")

estimator = MLEstimator(
    model_factory=build_model,
    param_spec=ParameterSpec(
        names=["sigma_eta", "sigma_eps"],
        start_values=[1.0, 1.0],
        transforms=["log", "log"],        # both constrained to (0, ∞)
    ),
)

results = estimator.fit(y, method="L-BFGS-B", n_starts=5)
print(results.params)
print(results.std_errors)
```

---

## Standard errors

Standard errors are computed from the **Hessian** of the negative log-likelihood
at the optimum, evaluated in the unconstrained space:

$$
\widehat{\mathrm{Var}}(\hat\psi) = \mathcal{H}(\hat\psi)^{-1}, \qquad
\mathcal{H}(\hat\psi) = -\frac{\partial^2 \ell}{\partial \psi\, \partial \psi'}
$$

The delta method propagates back to the constrained space:

$$
\widehat{\mathrm{Var}}(\hat\theta) =
  J(\hat\psi)\, \mathcal{H}(\hat\psi)^{-1}\, J(\hat\psi)',
\qquad
J(\hat\psi) = \frac{\partial\, \text{to\_constrained}(\psi)}{\partial \psi}
$$

For variance parameters with the log transform, $J = \text{diag}(\hat\theta)$,
so $\widehat{\mathrm{SE}}(\hat\sigma^2) = \hat\sigma^2 \cdot \widehat{\mathrm{SE}}(\hat\psi)$.

```python
# After fitting
print(f"σ_η² = {results.params['sigma2_eta']:.4f} "
      f"± {results.std_errors['sigma2_eta']:.4f}")

# Confidence intervals (Wald, 95%)
ci = results.conf_int(alpha=0.05)
print(ci)
```

### Numerical Hessian

`kalmanbox` computes the Hessian numerically using a central finite-difference
scheme unless an analytical score is provided:

```python
results = estimator.fit(y, hessian_method="central", hessian_step=1e-4)
```

!!! tip "When standard errors look too small"
    Standard errors from the Hessian are asymptotically valid but can
    understate uncertainty for small samples. Consider bootstrap confidence
    intervals via `results.bootstrap_ci(n_boot=1000)` for $n < 50$.

---

## Convergence and multiple starts

### Single start

```python
results = model.fit(start_params=[1000.0, 500.0], disp=True)
```

### Multiple random starts (recommended)

The log-likelihood of a state-space model can be multimodal. Using multiple
starting points guards against local optima:

```python
results = model.fit(
    method="mle",
    n_starts=10,          # 10 random restarts
    random_state=42,      # reproducible
    start_params=None,    # auto-generate starts on log scale
)

print(f"Best log-likelihood : {results.loglike:.4f}")
print(f"Converged starts    : {results.n_converged}/{results.n_starts}")
```

### Diagnosing convergence

```python
# Check optimizer exit status
print(results.optimizer_result.message)       # "Optimization terminated successfully."

# Check gradient norm at optimum (should be near zero)
print(f"Gradient norm: {results.gradient_norm:.2e}")  # ideally < 1e-4

# Check Hessian is positive definite (minimum, not maximum)
eigvals = np.linalg.eigvalsh(results.hessian)
print(f"Min Hessian eigenvalue: {eigvals.min():.4f}")  # should be > 0
```

---

## Information criteria

After MLE, model selection uses penalized criteria that balance fit and
complexity.

### Definitions

Let $\hat\ell$ be the maximized log-likelihood, $k$ the number of free
parameters, and $n^*$ the effective sample size:

$$
\mathrm{AIC}  = -2\hat\ell + 2k
$$

$$
\mathrm{BIC}  = -2\hat\ell + k\,\log n^*
$$

$$
\mathrm{HQIC} = -2\hat\ell + 2k\,\log\log n^*
$$

BIC penalizes complexity more strongly than AIC for $n^* > 7$. HQIC is
intermediate and is consistent (selects the true order) like BIC but
converges more slowly.

### Usage in `kalmanbox`

```python
from kalmanbox import LocalLevel, BSM
from kalmanbox.datasets import load_airline

airline = load_airline()
y_log   = np.log(airline["passengers"].to_numpy())

# Compare three models
for ModelClass, name in [
    (LocalLevel, "Local Level"),
]:
    res = ModelClass(y_log).fit(disp=False)
    print(f"{name:<20} AIC={res.aic:.2f}  BIC={res.bic:.2f}  HQIC={res.hqic:.2f}")
```

### Model comparison table

```python
from kalmanbox import LocalLevel, BSM
from kalmanbox.structural import LocalLinearTrend
from kalmanbox.datasets import load_nile

y = load_nile()["volume"].to_numpy()

models = {
    "Local Level"          : LocalLevel(y),
    "Local Linear Trend"   : LocalLinearTrend(y),
    "BSM (monthly)"        : None,  # requires seasonal data
}

rows = []
for name, m in models.items():
    if m is None:
        continue
    r = m.fit(disp=False)
    rows.append({"Model": name, "k": r.df_model,
                 "LogLike": r.loglike, "AIC": r.aic, "BIC": r.bic})

import pandas as pd
df = pd.DataFrame(rows).sort_values("AIC")
print(df.to_string(index=False))
```

---

## Full worked example: BSM estimation

```python
import numpy as np
import pandas as pd
from kalmanbox import BSM
from kalmanbox.datasets import load_airline

# Log-transform to stabilize variance
airline = load_airline()
y_log   = np.log(airline["passengers"].to_numpy())   # n=144

# ── Fit BSM with monthly seasonality ─────────────────────────────────────────
model   = BSM(y_log, period=12, stochastic_trend=True, stochastic_seasonal=True)
results = model.fit(method="mle", n_starts=5, disp=False)

print(results.summary())
# Prints parameter table: σ²_level, σ²_slope, σ²_seasonal, σ²_irregular
# with standard errors, z-stats, and information criteria

# ── Smoothed components ───────────────────────────────────────────────────────
sm = results.smooth()

trend    = sm.components["trend"]           # (144,)
seasonal = sm.components["seasonal"]        # (144,)
irregular = y_log - trend - seasonal        # residuals

# ── Forecast 24 months ahead ─────────────────────────────────────────────────
fc = results.forecast(steps=24, alpha=0.05)

print(f"\nParameters:")
for name, val, se in zip(
    results.param_names, results.params.values(), results.std_errors.values()
):
    print(f"  {name:<20} = {val:.6f}  (se = {se:.6f})")

print(f"\nAIC  = {results.aic:.3f}")
print(f"BIC  = {results.bic:.3f}")
print(f"HQIC = {results.hqic:.3f}")
```

---

## Local Level: step-by-step MLE from scratch

This example shows the full MLE workflow using only `KalmanFilter` — no
high-level model class — to make every step explicit.

```python
import numpy as np
from scipy.optimize import minimize
from kalmanbox import KalmanFilter, StateSpaceRepresentation
from kalmanbox.datasets import load_nile

y = load_nile()["volume"].to_numpy()
n = len(y)

# ── Model factory ──────────────────────────────────────────────────────────────
def make_kf(psi: np.ndarray) -> KalmanFilter:
    """Build a Local Level KF from unconstrained psi = [log σ²_η, log σ²_ε]."""
    sigma2_eta, sigma2_eps = np.exp(psi)
    T = np.array([[1.0]])
    Z = np.array([[1.0]])
    R = np.array([[1.0]])
    Q = np.array([[sigma2_eta]])
    H = np.array([[sigma2_eps]])
    ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
    return KalmanFilter(ssr, initialization="diffuse")

# ── Negative log-likelihood ───────────────────────────────────────────────────
def neg_ll(psi: np.ndarray) -> float:
    try:
        return -make_kf(psi).run(y).loglike
    except Exception:
        return 1e10   # penalise numerical failures

# ── Optimize from several starting points ────────────────────────────────────
best_result = None
rng = np.random.default_rng(0)

for _ in range(8):
    psi0 = rng.uniform(-2, 8, size=2)   # log-scale starting values
    res  = minimize(neg_ll, psi0, method="L-BFGS-B",
                    options={"ftol": 1e-12, "gtol": 1e-8})
    if best_result is None or res.fun < best_result.fun:
        best_result = res

psi_hat   = best_result.x
theta_hat = np.exp(psi_hat)
print(f"σ²_η = {theta_hat[0]:.2f}")
print(f"σ²_ε = {theta_hat[1]:.2f}")
print(f"Log-likelihood = {-best_result.fun:.4f}")

# ── Standard errors via numerical Hessian ────────────────────────────────────
from scipy.optimize import approx_fprime

h     = 1e-4
grad  = lambda p: approx_fprime(p, neg_ll, h)
H_num = np.array([approx_fprime(psi_hat + e_i * h, grad, h)
                  for e_i in np.eye(len(psi_hat))]) / h

# Covariance in constrained space via delta method
cov_psi   = np.linalg.inv(H_num)
J         = np.diag(theta_hat)             # Jacobian of exp(psi)
cov_theta = J @ cov_psi @ J
se_theta  = np.sqrt(np.diag(cov_theta))

print(f"\n95% CI for σ²_η: ({theta_hat[0] - 1.96*se_theta[0]:.2f}, "
      f"{theta_hat[0] + 1.96*se_theta[0]:.2f})")
print(f"95% CI for σ²_ε: ({theta_hat[1] - 1.96*se_theta[1]:.2f}, "
      f"{theta_hat[1] + 1.96*se_theta[1]:.2f})")

# ── Information criteria ──────────────────────────────────────────────────────
ll_hat = -best_result.fun
k      = len(psi_hat)
aic    = -2 * ll_hat + 2 * k
bic    = -2 * ll_hat + k * np.log(n)
hqic   = -2 * ll_hat + 2 * k * np.log(np.log(n))

print(f"\nAIC  = {aic:.3f}")
print(f"BIC  = {bic:.3f}")
print(f"HQIC = {hqic:.3f}")
```

---

## Common pitfalls

!!! warning "Numerical overflow in $\log|F_t|$"
    If initial parameter values are far from the optimum, $F_t$ can be very
    large or very small, causing `log(det(F))` to overflow. The log-transform
    on variance parameters avoids this by keeping $\sigma^2$ in a bounded
    range during optimization.

!!! warning "Flat likelihood surface"
    Some models (e.g., BSM with both `stochastic_trend=False` and no
    seasonality) reduce to deterministic components. The likelihood is then
    flat in the variance dimensions. Check the Hessian eigenvalues —
    near-zero eigenvalues indicate flat directions and unreliable standard
    errors.

!!! tip "Profiling the likelihood"
    Visualize the likelihood surface along one or two parameter dimensions
    to understand the optimization landscape before running MLE:

    ```python
    psi_grid = np.linspace(-5, 5, 60)
    ll_vals  = [-neg_ll(np.array([psi, psi_hat[1]])) for psi in psi_grid]

    import matplotlib.pyplot as plt
    plt.plot(np.exp(psi_grid), ll_vals)
    plt.axvline(theta_hat[0], color="red", ls="--", label="MLE")
    plt.xlabel("σ²_η"); plt.ylabel("Log-likelihood")
    plt.title("Profile likelihood for σ²_η")
    plt.legend(); plt.show()
    ```

---

## Related pages

- [Diffuse Initialization](diffuse.md) — why the diffuse likelihood matters for MLE
- [Missing Data](missing-data.md) — how NaNs enter the log-likelihood
- [Kalman Filter](kalman-filter.md) — the forward pass that computes $v_t$ and $F_t$
- [Diagnostics: Residual Analysis](../../diagnostics/residuals.md) — checking model fit after MLE
- [Diagnostics: Information Criteria](../../diagnostics/information-criteria.md) — deeper discussion of AIC/BIC
- [Theory: Likelihood Computation](../../theory/likelihood.md) — formal derivation
- [API: estimation](../../api/estimation.md)
- [Tutorial: Nile River Local Level](../../tutorials/nile-local-level.md)

### References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter.* Cambridge University Press. §3.4.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press. §7.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464.
- Hannan, E. J. & Quinn, B. G. (1979). The determination of the order of an autoregression. *Journal of the Royal Statistical Society B*, 41(2), 190–195.
