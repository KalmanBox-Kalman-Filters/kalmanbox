# Troubleshooting

This page lists the most common runtime errors and unexpected behaviours in
kalmanbox, with their root causes and concrete fixes.

---

## "Filter diverged" / `FilterDivergenceWarning`

**What it looks like**

```
kalmanbox.exceptions.FilterDivergenceWarning: Filter diverged at t=47.
|P_47| = 1.23e+18. Consider checking Q, H, or initialisation.
```

Or the filtered state covariance `P_t` grows without bound:

```python
result.filtered_state_cov[:, 0, 0]  # grows to 1e15, 1e20, …
```

**Root causes**

1. **Q or H has wrong units** — state noise variance `Q` is orders of
   magnitude larger than the data range. Check that your matrices are in
   consistent units.
2. **T is explosive** — transition matrix eigenvalue > 1, so the state
   variance explodes forward.
3. **Z is near-zero** — the observation equation carries almost no
   information, so $P$ never contracts.
4. **Diffuse initialisation on a stationary model** — using `initialize_diffuse`
   on a stationary AR component where a finite stationary variance is known.

**Solutions**

```python
# 1. Inspect eigenvalues of T
import numpy as np
eigvals = np.linalg.eigvals(T_mat)
assert np.all(np.abs(eigvals) <= 1.0), f"T has explosive roots: {eigvals}"

# 2. Use stationary initialisation for stationary components
kf.initialize_stationary()          # computes P1 from the discrete Lyapunov equation

# 3. Switch to the Square-Root filter, which is more numerically stable
from kalmanbox import SquareRootFilter
kf_sr = SquareRootFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)

# 4. Add a small nugget to H to prevent singular observation covariance
H_mat += 1e-6 * np.eye(H_mat.shape[0])
```

---

## "Covariance matrix not positive definite"

**What it looks like**

```
numpy.linalg.LinAlgError: Matrix is not positive definite.
  at kalmanbox/filter.py:214 in _cholesky_update
```

Or a warning:

```
kalmanbox.exceptions.CovarianceWarning: P_t is not positive semi-definite
at t=103. Applying Joseph stabilization.
```

**Root causes**

1. **Accumulated floating-point rounding** in long filter runs without
   symmetry enforcement.
2. **Ill-conditioned Q or H** — very small off-diagonal elements become
   negative due to round-off.
3. **Bug in custom system matrices** — returning a non-symmetric matrix.

**Solutions**

```python
# 1. Enable Joseph-form update (default in SquareRootFilter, opt-in for KF)
kf = KalmanFilter(T=T, Z=Z, H=H, Q=Q, joseph_form=True)

# 2. Use the Square-Root filter — maintains P as its Cholesky factor S
from kalmanbox import SquareRootFilter
kf = SquareRootFilter(T=T, Z=Z, H=H, Q=Q)

# 3. Force symmetry in custom matrices
def build_system_matrices(self, params):
    Q = compute_Q(params)
    Q = (Q + Q.T) / 2          # symmetrise
    Q += 1e-10 * np.eye(Q.shape[0])   # ensure strict positive-definiteness
    return dict(..., Q=Q)

# 4. Lower numerical tolerance for covariance check
kalmanbox.config.set(cov_check_rtol=1e-6)
```

---

## "MLE did not converge" / `ConvergenceWarning`

**What it looks like**

```
kalmanbox.exceptions.ConvergenceWarning: L-BFGS-B did not converge after
1000 iterations. Current log-likelihood: -847.32. Try different starting
values or increase maxiter.
```

**Root causes**

1. **Poor starting values** — parameters start far from the optimum.
2. **Flat or multi-modal likelihood** — signal-to-noise ratio is very low,
   or the model is over-parameterised.
3. **Gradient discontinuity** — near-zero variances create numerical kinks
   in the gradient.
4. **Constraints too tight** — bounds clip the optimum.

**Solutions**

```python
# 1. Use multiple random restarts
result = model.fit(n_restarts=10, random_state=42)

# 2. Provide better starting values (e.g., from OLS or moments)
result = model.fit(start_params=np.array([0.3, 0.7]))

# 3. Increase the iteration limit
result = model.fit(options={"maxiter": 5000, "ftol": 1e-12})

# 4. Try EM for initialisation, then switch to L-BFGS-B
result_em  = model.fit(method="em", n_iter=50)
result_mle = model.fit(method="lbfgs", start_params=result_em.params)

# 5. Check for identification (reduce model complexity if needed)
print(result.hessian_eigenvalues)   # near-zero eigenvalue → under-identified param
```

---

## "NaN in filtered states"

**What it looks like**

```python
result.filtered_state
# array([[nan, nan, nan, ..., 3.41, 3.55, ...]])
```

NaNs appear in the filtered states for early time periods or throughout.

**Root causes**

1. **NaN in the input data** — `y` contains `np.nan` but
   `handle_missing="error"` is set (the default raises; use `"skip"` or
   `"interpolate"` instead).
2. **Infinite diffuse variance** — if $P_\infty$ remains infinite after the
   diffuse initialisation period, the filter cannot update. This happens
   when `Z` is not full row-rank (unidentified state).
3. **Singular H** — a zero-variance observation equation causes a division
   by zero in $F_t = Z P_t Z' + H$.
4. **NaN in system matrices** — a custom `build_system_matrices` returning
   NaN (e.g., from `log(negative)` in a transform).

**Solutions**

```python
# 1. Handle NaN observations explicitly
model = LocalLevelModel(endog=y, handle_missing="skip")
# "skip" skips the Kalman update for missing y_t; state propagates via T

# 2. Check data
import numpy as np
print(np.isnan(y).sum(), np.isinf(y).sum())

# 3. Check system matrices for NaN/inf
mats = model.build_system_matrices(model.start_params)
for k, v in mats.items():
    if np.any(~np.isfinite(v)):
        print(f"Non-finite values in {k}: {v}")

# 4. Check that Z is full row-rank
Z = mats["Z"]
rank = np.linalg.matrix_rank(Z)
print(f"Z shape {Z.shape}, rank {rank}")   # rank should equal Z.shape[0]
```

---

## "Log-likelihood is -inf"

**What it looks like**

```
FitResult.llf = -inf
```

or

```
kalmanbox.exceptions.NumericalWarning: det(F_t) <= 0 at t=12. log|F_t| = -inf.
```

**Root causes**

1. **H = 0** — zero observation noise makes $F_t = Z P_t Z'$ singular when
   $P_t \to 0$.
2. **Parameter at bound** — a variance parameter hit zero during
   optimisation, collapsing the likelihood.
3. **Mismatched dimensions** — $y$ is multivariate but $H$ is shaped for a
   scalar observation.
4. **Scale mismatch** — data range of $10^6$ with $H$ of order $10^{-3}$
   produces $F_t^{-1} \approx \infty$.

**Solutions**

```python
# 1. Add a minimum nugget to H
H_mat = np.diag(np.maximum(diag_H, 1e-8))

# 2. Set tighter lower bounds on variance parameters
result = model.fit(param_bounds=[(1e-8, None)] * len(model.start_params))

# 3. Standardise the data to mean 0, std 1 before fitting
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y_std  = scaler.fit_transform(y.reshape(-1, 1)).ravel()
result = model.fit(endog=y_std)
# Transform predictions back: result.filtered_state * scaler.scale_ + scaler.mean_

# 4. Check dimensions
print(y.shape, H_mat.shape)  # y: (T,) or (T, p); H: (p, p)
```

---

## Performance is slow

**Symptoms**: filtering $T=10\,000$, $m=5$ takes > 5 seconds;
MLE with 50 restarts takes > 10 minutes.

**Checklist and fixes**

| Root cause                          | Fix                                                                       |
|-------------------------------------|---------------------------------------------------------------------------|
| Pure-Python backend in use          | `kalmanbox.config.set(backend="numba")` or `model = MyModel(..., backend="numba")` |
| Numba not installed                 | `pip install "kalmanbox[numba]"`                                          |
| Numba not warmed up                 | Call `kalmanbox.warmup()` once at startup                                 |
| Time-varying matrices rebuilt every step | Cache them: return a 3-D array `(T, m, m)` in `build_system_matrices`, not a per-step function |
| MLE restarts with tight convergence | Use `ftol=1e-8` instead of `1e-14`; use EM for initialisation             |
| Large state dimension ($m > 20$)    | Use `InformationFilter` (sparse $P^{-1}$) or `EnsembleKalmanFilter`      |
| BLAS not linked properly            | Verify: `numpy.show_config()` should list OpenBLAS or MKL                 |

```python
# Quick check: compare backends
import time
model_py  = LocalLevelModel(endog=y, backend="numpy")
model_nb  = LocalLevelModel(endog=y, backend="numba")

t0 = time.perf_counter(); model_py.filter(); print("numpy:", time.perf_counter()-t0)
t0 = time.perf_counter(); model_nb.filter(); print("numba:", time.perf_counter()-t0)
```

See the [Performance benchmarks](../benchmarks/kalman.md) for typical timings
across state dimensions and backends.

---

## Installation fails

**Common issues and fixes**

**`pip install kalmanbox` fails with build error on NumPy/SciPy**

```bash
# Use binary wheels (avoids compilation)
pip install --only-binary=:all: kalmanbox

# Or upgrade pip first
pip install --upgrade pip
pip install kalmanbox
```

**Numba installation fails on Python 3.13**

Numba may not yet have a release for the latest Python minor version.
Use Python 3.11 or 3.12, or install kalmanbox without the optional Numba
dependency and run in pure-NumPy mode:

```bash
pip install kalmanbox          # no Numba
# kalmanbox will auto-detect absence of numba and fall back to numpy backend
```

**Conflicts with existing statsmodels/pandas**

```bash
pip install "kalmanbox>=0.4" "statsmodels>=0.14" "pandas>=2.0"
```

kalmanbox requires NumPy ≥ 1.24, SciPy ≥ 1.10, and pandas ≥ 2.0 (optional).

**ImportError on `kalmanbox._core`**

The compiled extension was not built for your platform. Re-install from
source:

```bash
pip install --no-binary kalmanbox kalmanbox
```

---

## Results differ from statsmodels

**Why this happens**

kalmanbox and statsmodels make different defaults in several areas:

| Aspect                     | kalmanbox default               | statsmodels default             |
|----------------------------|---------------------------------|---------------------------------|
| Diffuse initialisation     | Exact diffuse (Harvey 1989)     | Approximate diffuse (large κ)   |
| Convergence tolerance      | `ftol=1e-12`                    | `gtol=1e-6`                     |
| Log-likelihood convention  | Prediction-error decomposition  | Prediction-error decomposition  |
| Missing data               | Exact skip (no imputation)      | Exact skip                      |
| Covariance enforcement     | Joseph stabilisation (default)  | No (plain KF update)            |

**How to align the two**

To reproduce statsmodels output in kalmanbox, use approximate diffuse
initialisation and a matched convergence tolerance:

```python
model = LocalLevelModel(endog=y)
result = model.fit(
    initialisation="approximate_diffuse",
    initialisation_kappa=1e6,          # statsmodels default κ
    options={"gtol": 1e-6},
)
```

Conversely, to reproduce kalmanbox results in statsmodels, enable exact
diffuse initialisation (available since statsmodels 0.14):

```python
sm_model = sm.tsa.UnobservedComponents(y, level="local level")
sm_result = sm_model.fit(disp=False, low_memory=True,
                          filter_kwargs={"filter_method": "exact_initial"})
```

Small numerical differences (< 1e-8 in log-likelihood) are expected due to
floating-point ordering differences in the two implementations.
