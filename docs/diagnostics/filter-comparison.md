# Filter Comparison

When multiple filter algorithms are available for the same state-space model, choosing
the right one requires empirical evidence. This page documents the metrics, benchmarking
infrastructure, and visualisation tools kalmanbox provides for comparing filters
head-to-head on accuracy, efficiency, and numerical robustness.

---

## 1. Why compare filters?

Every filter is an approximation or specialisation of the Bayesian optimal estimator.
The standard Kalman filter is optimal only for *linear Gaussian* models. For nonlinear
or numerically sensitive models, several alternatives exist — each with distinct
accuracy/cost tradeoffs:

```mermaid
graph TD
    A[State-Space Model] --> B{Linear & Gaussian?}
    B -->|Yes| C[KalmanFilter]
    B -->|Yes, ill-conditioned| D[SquareRootFilter]
    B -->|Yes, sparse/high-dim| E[InformationFilter]
    B -->|Mildly nonlinear| F[EKF]
    B -->|Moderately nonlinear| G[UKF]
    B -->|Strongly nonlinear / non-Gaussian| H[EnKF / Particle Filter]

    C --> I[Baseline accuracy, O(n·m³)]
    D --> J[Better conditioning, same accuracy]
    E --> K[Efficient fusion of many sensors]
    F --> L[Linearisation error, fast]
    G --> M[Sigma-point accuracy, moderate cost]
    H --> N[Monte Carlo accuracy, high cost]
```

Comparing filters answers four questions:

1. **Accuracy** — does the alternative recover states as well as (or better than) the baseline?
2. **Efficiency** — is the speed-up worth the accuracy tradeoff?
3. **Stability** — does the filter remain positive-definite over long runs?
4. **Scalability** — how does performance degrade as state dimension grows?

---

## 2. Comparison metrics

### 2.1 Root Mean Squared Error (RMSE)

When a ground-truth trajectory $\alpha_t^\star$ is available (simulation studies or
datasets with known states), RMSE measures state estimation accuracy:

$$
\text{RMSE}_j = \sqrt{\frac{1}{n}\sum_{t=1}^{n} \|\hat{\alpha}_t^{(j)} - \alpha_t^\star\|_2^2}
$$

where $\hat{\alpha}_t^{(j)}$ is the filtered state estimate from filter $j$.

For multivariate states, kalmanbox reports per-component RMSE in addition to the
aggregated Euclidean norm.

!!! note "Ground truth unavailable?"
    In real data applications, $\alpha_t^\star$ is unknown. Use the smoother output
    $\hat{\alpha}_{t|n}$ from the most accurate filter (typically Square-Root) as a
    proxy, or compare relative to the model log-likelihood.

### 2.2 Log-likelihood

All Kalman-type filters compute the **prediction error decomposition** log-likelihood:

$$
\log L = -\frac{np}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^{n}\!\left(\log|F_t^{(j)}| + v_t^{(j)\top} F_t^{(j)-1} v_t^{(j)}\right)
$$

Filters that differ in log-likelihood on the same data indicate numerical discrepancies —
a well-implemented alternative should match the standard Kalman filter to floating-point
precision on linear Gaussian models.

**Interpretation**:

| $\Delta\log L = \log L_j - \log L_{\text{KF}}$ | Implication |
|---|---|
| $\approx 0$ | Filters agree numerically |
| $> 0$ (alternative better) | Standard KF suffers numerical degradation |
| $< 0$ (alternative worse) | Alternative has approximation error (EKF, EnKF) |

### 2.3 Execution time

Wall-clock time per filter run, averaged over multiple repeats:

$$
\bar{T}_j = \frac{1}{R}\sum_{r=1}^{R} T_j^{(r)}
$$

where $R \geq 10$ is the number of repeats (removes OS scheduling noise).
kalmanbox also reports the **time per time step** $\bar{T}_j / n$ for fair
comparison across series of different length.

### 2.4 Numerical stability — covariance condition number

A key robustness indicator is the condition number of the predicted covariance
$P_t$ at each time step:

$$
\kappa_t^{(j)} = \frac{\sigma_{\max}(P_t^{(j)})}{\sigma_{\min}(P_t^{(j)})}
$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values.

$$
\kappa^{(j)} = \max_{t} \kappa_t^{(j)}
$$

**Interpretation**:

| $\log_{10}\kappa$ | Stability |
|-------------------|-----------|
| $< 4$ | Excellent — well-conditioned |
| $4$–$8$ | Good — standard double precision adequate |
| $8$–$12$ | Caution — near machine precision limit |
| $> 12$ | Dangerous — use Square-Root filter |

For the Square-Root filter, $P_t$ is represented as its Cholesky factor $S_t$
such that $P_t = S_t S_t^\top$. The condition number reported is
$\kappa(S_t)^2 \approx \kappa(P_t)$, which is always at most half the
bit-precision loss of the standard covariance update.

### 2.5 Innovation consistency

Even without ground truth, filters can be compared on how well their predicted
innovations match the actual observations. The **scaled innovation squared** (SIS)
measures this per time step:

$$
s_t^{(j)} = v_t^{(j)\top} F_t^{(j)-1} v_t^{(j)}
$$

Under correct specification, $s_t^{(j)} \sim \chi^2_p$ (where $p$ is the observation
dimension). Filters that systematically over- or under-state innovation covariances
will have $\mathbb{E}[s_t] \neq p$.

---

## 3. Linear model filter comparison

### 3.1 KalmanFilter vs SquareRootFilter vs InformationFilter

For linear Gaussian models these three filters are mathematically equivalent — they
produce the same state estimates. Differences arise only in numerical precision and speed.

**When to prefer each**:

| Filter | Strengths | Weaknesses |
|--------|-----------|------------|
| `KalmanFilter` | Fast, simple, well-tested | Degrades for $\kappa(P_t) > 10^{10}$ |
| `SquareRootFilter` | Guaranteed $P_t \succ 0$, halves condition number | ~2× slower covariance update |
| `InformationFilter` | Efficient for many observations, sparse inversions | Poor for diffuse initialisation |

**Numerical degradation scenario** — a model with very different state variances
(e.g., level variance $10^6$ and cycle variance $10^{-2}$) causes $P_t$ to become
ill-conditioned. The Square-Root filter remains stable while the standard filter
loses numerical precision:

$$
P_{\text{KF}} = T P T^\top + Q \quad \text{(condition number squares per step)}
$$

$$
S_{\text{SR}} = \text{qr}\!\begin{pmatrix} S_Q \\ S_P T^\top \end{pmatrix}^\top
\quad \text{(condition number accumulates at most linearly)}
$$

### 3.2 Condition number evolution

```
Step  | KalmanFilter κ  | SquareRootFilter κ  | Threshold
------|-----------------|---------------------|----------
  10  | 1.2 × 10³      | 1.1 × 10³           |
  50  | 4.7 × 10⁷      | 2.2 × 10⁴           |
 100  | 3.1 × 10¹¹     | 4.8 × 10⁵           | ← KF near precision limit
 200  | overflows       | 2.1 × 10⁶           |
```

---

## 4. Nonlinear model filter comparison

### 4.1 EKF vs UKF vs EnKF

For nonlinear models, filter choice involves accuracy vs computation tradeoffs:

| | **EKF** | **UKF** | **EnKF** |
|---|---|---|---|
| **Approximation** | 1st-order Taylor linearisation | Unscented transform (2m+1 sigma points) | Monte Carlo ensemble ($N_e$ particles) |
| **Cost per step** | $O(m^3)$ — Jacobian + standard update | $O(m^3)$ — sigma points | $O(N_e \cdot m^2)$ |
| **Accuracy** | $O(h^2)$ error (second-order terms lost) | $O(h^4)$ error (captures 3rd-order moments) | $O(1/\sqrt{N_e})$ — converges with ensemble size |
| **Best for** | Mildly nonlinear, fast required | Moderately nonlinear, smooth $f$/$h$ | Strongly nonlinear, high-dimensional |
| **Handles non-Gaussian?** | No | Partially | Yes (with inflation) |

where $h$ is the step size or noise magnitude characterising the nonlinearity.

### 4.2 Accuracy order analysis

Let the state transition be $\alpha_{t+1} = f(\alpha_t) + \eta_t$ with Jacobian
$F_t = \partial f / \partial \alpha_t$.

**EKF linearisation error**: The Taylor expansion drops second-order terms
$\frac{1}{2}\text{tr}[\nabla^2 f_i \cdot P_t]$ from the mean propagation. For a nonlinear
system with large state uncertainty $P_t$, this bias accumulates:

$$
\mathbb{E}[\hat{\alpha}_{t+1}^{\text{EKF}}] \approx f(\hat{\alpha}_t) + O(\|P_t\|)
$$

**UKF sigma-point accuracy**: The unscented transform uses $2m+1$ deterministically
chosen sigma points that match the Gaussian mean and covariance exactly, and capture
skewness (3rd-order terms):

$$
\mathbb{E}[\hat{\alpha}_{t+1}^{\text{UKF}}] \approx f(\hat{\alpha}_t) + O(\|P_t\|^2)
$$

The UKF is second-order accurate without requiring Jacobian computation.

---

## 5. Automated benchmarking

The `filter_benchmark()` function runs a systematic comparison:

```python
from kalmanbox.diagnostics import filter_benchmark

bench = filter_benchmark(
    model,                               # model instance
    filters: list = None,                # None = auto-select by model type
    n_repeats: int = 20,                 # timing repeats
    true_states: np.ndarray = None,      # for RMSE computation
    metrics: list = ["rmse", "loglik", "time", "condition_number"],
)
print(bench.summary())
bench.to_dataframe()      # returns pd.DataFrame
```

**Auto-selected filter sets**:

- **Linear Gaussian models**: `[KalmanFilter, SquareRootFilter, InformationFilter]`
- **Nonlinear models**: `[EKF, UKF, EnKF(N=100), EnKF(N=500)]`
- **Custom**: pass `filters` explicitly

---

## 6. API reference

### `compare_filters()`

```python
from kalmanbox.diagnostics import compare_filters

ComparisonResult = compare_filters(
    model,                              # StateSpaceModel instance
    filters: list[str | Filter],        # e.g. ["kalman", "square_root", "information"]
    y: np.ndarray,                      # observed data
    true_states: np.ndarray = None,     # optional ground truth
    metrics: list[str] = None,          # None = all applicable metrics
    n_repeats: int = 10,                # timing repeats
    random_seed: int = 0,               # for EnKF reproducibility
)
```

**`ComparisonResult` attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.filters` | `list[str]` | Filter names in comparison order |
| `.rmse` | `dict[str, float]` | Per-filter RMSE (NaN if no ground truth) |
| `.loglik` | `dict[str, float]` | Per-filter log-likelihood |
| `.time_mean` | `dict[str, float]` | Mean wall-clock time (seconds) |
| `.time_std` | `dict[str, float]` | Std dev of timing |
| `.condition_number_max` | `dict[str, float]` | Max condition number over time |
| `.innovations_mean_sis` | `dict[str, float]` | Mean scaled innovation squared |
| `.states` | `dict[str, ndarray]` | Filtered states per filter, shape $(n, m)$ |
| `.covariances` | `dict[str, ndarray]` | Filtered covariances per filter, shape $(n, m, m)$ |
| `.summary()` | method | Print formatted comparison table |
| `.plot()` | method | Generate comparison figure |
| `.to_dataframe()` | method | Return `pd.DataFrame` with all metrics |

### `filter_benchmark()`

```python
from kalmanbox.diagnostics import filter_benchmark

BenchmarkResult = filter_benchmark(
    model,
    filters: list[str | Filter] = None,    # None = auto
    y: np.ndarray = None,                  # None = simulate from model
    n_sim: int = 1,                        # number of simulated datasets
    n_repeats: int = 20,                   # timing repeats per dataset
    true_states: np.ndarray = None,
    state_dims: list[int] = None,          # sweep over state dimensions
    obs_dims: list[int] = None,            # sweep over observation dimensions
    n_timesteps: list[int] = None,         # sweep over T
    random_seed: int = 0,
)
```

Supports **scaling benchmarks**: pass lists to `state_dims`, `obs_dims`, or
`n_timesteps` and the benchmark sweeps the parameter, returning a scaling curve.

### `plot_filter_comparison()`

```python
from kalmanbox.diagnostics import plot_filter_comparison

fig = plot_filter_comparison(
    comparison_result,                # ComparisonResult from compare_filters()
    plot_states: bool = True,         # overlay filtered state trajectories
    plot_covariances: bool = True,    # ±2σ bands per filter
    plot_innovations: bool = True,    # innovation time series
    dates: pd.DatetimeIndex = None,
    figsize: tuple = (16, 10),
    alpha_band: float = 0.2,          # transparency for covariance bands
)
```

Produces a multi-panel figure with:

1. **State trajectories** — overlaid $\hat{\alpha}_t^{(j)}$ with $\pm 2\sigma$ bands.
2. **RMSE over time** — cumulative RMSE $\sqrt{\frac{1}{t}\sum_{s=1}^{t}(\hat{\alpha}_s - \alpha_s^\star)^2}$.
3. **Scaled innovation squared** $s_t^{(j)}$ with $\chi^2_p$ reference lines.
4. **Condition number** $\kappa_t^{(j)}$ log-scale time series.

---

## 7. Examples

### Example 1: linear model — standard vs square-root on ill-conditioned system

```python
import numpy as np
from kalmanbox import KalmanFilter
from kalmanbox.filters import SquareRootFilter, InformationFilter
from kalmanbox.diagnostics import compare_filters, plot_filter_comparison

rng = np.random.default_rng(42)
n, m = 200, 3

# Deliberately ill-conditioned: state variances span 8 orders of magnitude
Q = np.diag([1e-4, 1.0, 1e4])    # level, trend, seasonal — very different scales
H = np.array([[1.0, 0.0, 1.0]])   # observation matrix
T = np.eye(m)
Z = H

# Simulate data
alpha = np.zeros((n+1, m))
for t in range(n):
    alpha[t+1] = T @ alpha[t] + rng.multivariate_normal(np.zeros(m), Q)
y = (Z @ alpha[1:].T).T + rng.normal(0, 1.0, (n, 1))

# Build a simple custom SSM
from kalmanbox.models import LinearGaussianSSM
model = LinearGaussianSSM(Z=Z, T=T, H=np.array([[1.0]]), Q=Q,
                           a1=np.zeros(m), P1=np.eye(m)*100)

result = compare_filters(
    model,
    filters=["kalman", "square_root", "information"],
    y=y,
    true_states=alpha[1:],
    n_repeats=20,
)
print(result.summary())
```

**Expected output**:

```
Filter Comparison Summary
=========================
Model: LinearGaussianSSM  |  n=200, m=3, p=1

Metric             KalmanFilter   SquareRoot   Information
─────────────────────────────────────────────────────────
RMSE               0.8721         0.8719       0.8720
Log-likelihood   -263.4         -263.4        -263.4
Time (ms/run)       1.23           2.41          1.87
Max κ(P)          1.3e+11        2.4e+05       n/a (info form)
Mean SIS            1.02           1.02          1.02

Notes:
  KalmanFilter: condition number 1.3e+11 near double-precision limit
  SquareRootFilter: condition number controlled at 2.4e+05  ← preferred
  InformationFilter: stable but 52% slower than KalmanFilter
```

**Interpretation**: On this ill-conditioned model, the standard Kalman filter reaches a
condition number of $1.3 \times 10^{11}$ — only two orders of magnitude below double-precision
machine epsilon ($\approx 10^{-16}$), meaning around 5 digits of precision are lost in
the covariance update. The Square-Root filter keeps $\kappa \approx 2.4 \times 10^5$
(fully safe) at a cost of $2 \times$ slower computation. All three filters agree on
RMSE and log-likelihood for this sample, but the standard KF would diverge on a longer
run or with a larger conditioning gap.

**Recommendation**: Use `SquareRootFilter` when `max_condition_number > 1e8`.

---

### Example 2: nonlinear tracking — EKF vs UKF vs EnKF

```python
import numpy as np
from kalmanbox.filters import EKF, UKF, EnsembleKalmanFilter
from kalmanbox.diagnostics import compare_filters, plot_filter_comparison
from kalmanbox.models import BearingsOnlyTracking  # 2-D nonlinear tracking model

rng = np.random.default_rng(0)

# Bearings-only tracking: strongly nonlinear observation function
# State: [x, ẋ, y, ẏ], Observation: bearing angle arctan(y/x)
model = BearingsOnlyTracking(
    process_noise=0.1,
    obs_noise=0.05,
    dt=0.1,
)

# Simulate true trajectory and noisy observations
true_states, y = model.simulate(n=300, random_seed=42)

result = compare_filters(
    model,
    filters=[
        EKF(),
        UKF(alpha=1e-3, beta=2.0, kappa=0.0),
        EnsembleKalmanFilter(N=100, inflation=1.05),
        EnsembleKalmanFilter(N=500, inflation=1.05),
    ],
    y=y,
    true_states=true_states,
    n_repeats=10,
)
print(result.summary())
plot_filter_comparison(result, plot_innovations=True)
```

**Expected output**:

```
Filter Comparison Summary
=========================
Model: BearingsOnlyTracking  |  n=300, m=4, p=1

Metric            EKF      UKF      EnKF-100   EnKF-500
────────────────────────────────────────────────────────
RMSE              1.243    0.871    0.912      0.884
Log-likelihood  -184.2   -171.3   -173.8     -172.1
Time (ms/run)     1.8      3.2      28.4       138.7
Max κ(P)        2.1e3    1.9e3    n/a        n/a
Mean SIS          1.41     1.02     1.08       1.03

Notes:
  EKF:      linearisation error — RMSE 43% higher than UKF
  UKF:      best accuracy/speed tradeoff for this model
  EnKF-100: reasonable accuracy but high variance in repeated runs
  EnKF-500: converges close to UKF at 43× higher cost
```

**Interpretation**:

- **EKF** suffers 43 % higher RMSE than UKF due to first-order linearisation error in the
  nonlinear bearing observation. The mean SIS of 1.41 (vs expected 1.0 under the $\chi^2_1$
  null) confirms that EKF innovations are over-dispersed — the filter over-estimates
  the innovation variance as a result of the Jacobian approximation.
- **UKF** achieves the best accuracy/cost tradeoff for this mild-to-moderate nonlinearity.
- **EnKF** requires $N_e = 500$ particles to approach UKF accuracy, at 43× the cost.

**Recommendation**: For smooth nonlinear models with state dimension $m \leq 20$, UKF
dominates. EKF is preferred only when Jacobians are cheap and speed is critical.

---

### Example 3: scaling benchmark — state dimension sweep

```python
import numpy as np
from kalmanbox.diagnostics import filter_benchmark
import matplotlib.pyplot as plt

# Compare KF vs Square-Root as state dimension grows
bench = filter_benchmark(
    model=None,                         # use auto-generated random SSM
    filters=["kalman", "square_root"],
    state_dims=[2, 5, 10, 20, 50, 100],
    n_timesteps=[500],
    n_repeats=10,
    random_seed=0,
)

df = bench.to_dataframe()
print(df.pivot(index="state_dim", columns="filter", values=["time_mean", "condition_number_max"]))
```

**Expected output summary**:

```
                 time_mean (ms)           max κ(P)
state_dim     KalmanFilter  SqRoot   KalmanFilter  SqRoot
     2           0.08        0.15       8.4e3       2.9e2
     5           0.19        0.34       6.1e5       7.8e3
    10           0.52        0.94       4.3e7       6.6e4
    20           2.11        3.87       2.9e9       5.2e5
    50          24.7         45.1       1.1e12      8.4e6  ← KF unsafe
   100         188.4        342.0       overflows   9.2e7
```

Condition numbers grow roughly as $O(\exp(m))$ for the standard KF and $O(\exp(m/2))$
for the Square-Root filter — consistent with the theoretical result that the SR formulation
halves the exponent of numerical error growth.

---

## 8. Interpretation guide

### When filters agree

If all compared filters yield the same log-likelihood (to within rounding) and similar
RMSE, the model is well-conditioned and any of the filters can be used. Choose the fastest.

### When filters disagree in log-likelihood

| Pattern | Likely cause | Action |
|---------|--------------|--------|
| Standard KF lower than SR | Ill-conditioned $P_t$ | Switch to `SquareRootFilter` |
| EKF lower than UKF | Linearisation error | Use `UKF` or reduce nonlinearity |
| EnKF high variance across runs | Too few particles | Increase $N_e$ or use UKF |
| All filters differ | Model misspecification | Check innovation tests |

### When filters disagree in RMSE

RMSE disagreement signals a genuine accuracy difference, not just numerical noise.
Run with multiple random seeds to confirm the difference is systematic:

```python
rng_seeds = [0, 1, 2, 3, 4]
rmse_results = {}

for seed in rng_seeds:
    true_states, y = model.simulate(n=300, random_seed=seed)
    res = compare_filters(model, filters=["ekf", "ukf"], y=y, true_states=true_states)
    for filt, val in res.rmse.items():
        rmse_results.setdefault(filt, []).append(val)

for filt, vals in rmse_results.items():
    print(f"{filt}: mean RMSE = {np.mean(vals):.3f} ± {np.std(vals):.3f}")
```

---

## Related

- [Consistency Tests](consistency.md) — NEES/NIS tests to verify filter probabilistic calibration
- [User guide: EKF](../user-guide/filters/ekf.md) — Extended Kalman Filter
- [User guide: UKF](../user-guide/filters/ukf.md) — Unscented Kalman Filter
- [User guide: Square-Root Filter](../user-guide/filters/square-root.md)
- [User guide: Ensemble KF](../user-guide/filters/ensemble.md)
- [User guide: Filter Comparison (conceptual)](../user-guide/filters/comparison.md)
- [Theory: Nonlinear Filter Theory](../theory/nonlinear-theory.md)
- [Theory: Numerical Stability](../theory/numerical-stability.md)
- [Benchmarks: Performance](../benchmarks/performance.md)
- [API: diagnostics module](../api/diagnostics.md)
