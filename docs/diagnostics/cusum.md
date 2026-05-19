# CUSUM and Structural-Break Tests

CUSUM (cumulative sum) statistics are the standard graphical and formal tools for detecting
whether the parameters of a state-space model are constant over the sample. A parameter
break — an abrupt shift in the level, trend, or variance — leaves a clear signature in the
recursive residuals even when the model does not model it explicitly.

---

## 1. Background: recursive residuals

The Kalman filter is a sequential estimator. At each time $t$ it produces the **innovation**

$$
v_t = y_t - Z_t a_t, \qquad v_t \sim \mathcal{N}(0, F_t)
$$

and the **standardised innovation**

$$
\tilde{v}_t = F_t^{-1/2} v_t \sim \mathcal{N}(0, I)
$$

Under parameter stability, the sequence $\{\tilde{v}_t\}$ is i.i.d. standard normal.
CUSUM tests accumulate these residuals to detect systematic departures from zero mean or
unit variance that would indicate a structural break.

!!! note "Recursive residuals vs OLS recursive residuals"
    In a pure regression context, recursive residuals are the OLS prediction errors as the
    estimation window expands. For state-space models the same role is played by the Kalman
    innovations, which are already conditioned only on past observations — making them the
    natural recursive residuals.

---

## 2. CUSUM statistic

### Definition

The CUSUM statistic accumulates the standardised innovations:

$$
\text{CS}_t = \sum_{s=1}^{t} \tilde{v}_s, \qquad t = 1, \ldots, n
$$

For scalar innovations, $\tilde{v}_s$ is a scalar. For multivariate models use the
**NEES (Normalised Estimation Error Squared)**:

$$
\tilde{v}_s^\top \tilde{v}_s \sim \chi^2_p
$$

and accumulate the centred NEES values instead.

### Asymptotic distribution under the null

Under parameter stability, as $n \to \infty$:

$$
\frac{1}{\sqrt{n}} \text{CS}_{\lfloor nt \rfloor} \xrightarrow{d} W(t), \quad t \in [0, 1]
$$

where $W(t)$ is a standard **Brownian motion**. The supremum $\sup_t |W(t)|$ has the
Kolmogorov distribution, giving asymptotic 5 % critical bounds:

$$
\pm c_\alpha \sqrt{n}, \qquad c_{0.05} \approx 1.358
$$

These correspond to the straight-line boundaries plotted in the CUSUM chart.

### Finite-sample boundaries

The exact boundaries at level $\alpha$ are:

$$
\text{Boundary}_t = \pm \left(c_\alpha\sqrt{n} + 2c_\alpha\frac{t}{\sqrt{n}}\right), \qquad t = 1, \ldots, n
$$

i.e. a pair of linearly expanding bands. When the CUSUM path crosses either band, the null
of parameter stability is rejected at level $\alpha$.

### Detecting mean breaks

A structural break at time $\tau$ in the **level** of the series shifts the innovations by
a constant $\delta$ for $t > \tau$:

$$
\tilde{v}_t = \delta \cdot \mathbf{1}_{t > \tau} + \varepsilon_t
$$

This produces a **linear trend** in the CUSUM path starting at $\tau$, which is detected
visually as a kink and formally when the path crosses the bands.

---

## 3. CUSUM-of-Squares statistic

### Definition

While the CUSUM detects **mean** breaks, the CUSUM-of-squares detects **variance** breaks:

$$
\text{CS}^2_t = \frac{\sum_{s=1}^{t} \tilde{v}_s^2}{\sum_{s=1}^{n} \tilde{v}_s^2}, \qquad t = 1, \ldots, n
$$

Under stability, $\text{CS}^2_t \approx t/n$ (the squared innovations each have expectation
1, so the ratio should track the time fraction linearly).

### Asymptotic distribution

$$
\frac{1}{n}\sum_{s=1}^{\lfloor nt \rfloor} \tilde{v}_s^2 \xrightarrow{d} t \quad \text{(LLN)}
$$

and the centred, scaled process converges to a **Brownian bridge** $B(t) = W(t) - tW(1)$.
The 5 % boundary for the Brownian bridge supremum gives:

$$
c_{0.05}^{SQ} \approx 1.358 / \sqrt{n}
$$

Bands are:

$$
\frac{t}{n} \pm \frac{c_\alpha}{\sqrt{n}}
$$

Crossings above the upper band indicate **increasing variance** after the breakpoint;
crossings below indicate **decreasing variance**.

---

## 4. Confidence bands

=== "Asymptotic (default)"

    Based on the Brownian-motion limit. Adequate for $n \geq 50$.

    ```python
    from kalmanbox.diagnostics import cusum

    cs = cusum(results, alpha=0.05, method="asymptotic")
    cs.plot()
    ```

=== "Bootstrap"

    For small samples ($n < 50$) use parametric bootstrap bands:

    ```python
    cs = cusum(results, alpha=0.05, method="bootstrap", n_boot=5000, seed=42)
    cs.plot()
    ```

=== "Ploberger-Krämer"

    Exact finite-sample bounds (Ploberger & Krämer 1992), recommended for very short series:

    ```python
    cs = cusum(results, alpha=0.05, method="pk")
    cs.plot()
    ```

---

## 5. Detecting breakpoints

When the CUSUM or CUSUM-SQ crosses the confidence bands, the crossing point is an estimate
of the **breakpoint location** $\hat{\tau}$:

$$
\hat{\tau} = \arg\max_t \left|\text{CS}_t\right|
$$

kalmanbox provides a formal breakpoint estimator with a confidence interval:

```python
from kalmanbox.diagnostics import detect_breakpoint

bp = detect_breakpoint(results, method="cusum")
print(f"Breakpoint at t={bp.location} ({bp.date})")
print(f"95% CI: [{bp.ci_lower}, {bp.ci_upper}]")
print(f"Bai-Perron p-value: {bp.pvalue:.4f}")
```

### Multiple breakpoints

The Bai-Perron (1998, 2003) sequential procedure tests for $k$ vs $k+1$ breaks and is
available for state-space residuals:

```python
from kalmanbox.diagnostics import multiple_breakpoints

mbp = multiple_breakpoints(results, max_breaks=5, penalty="bic")
print(mbp.summary())
# Outputs: number of breaks, locations, dates, and BIC for each specification
```

---

## 6. API reference

### `cusum()`

```python
from kalmanbox.diagnostics import cusum

CUSUMResult = cusum(
    results,                      # KalmanResults object
    alpha: float = 0.05,          # Significance level for confidence bands
    method: str = "asymptotic",   # "asymptotic" | "bootstrap" | "pk"
    n_boot: int = 2000,           # Bootstrap replications (if method="bootstrap")
    seed: int | None = None,      # RNG seed for reproducibility
    standardize: bool = True,     # Standardise innovations by F_t^{1/2}
)
```

**`CUSUMResult` attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.statistic` | `ndarray` | $\text{CS}_t$ values, shape $(n,)$ |
| `.upper_bound` | `ndarray` | Upper confidence band, shape $(n,)$ |
| `.lower_bound` | `ndarray` | Lower confidence band, shape $(n,)$ |
| `.crossings` | `list[int]` | Time indices where path exits the bands |
| `.pvalue` | `float` | $p$-value for the supremum test |
| `.reject` | `bool` | Whether to reject stability at level $\alpha$ |
| `.plot()` | method | Plot CUSUM with bands |
| `.summary()` | method | Print test results |

### `cusum_sq()`

```python
from kalmanbox.diagnostics import cusum_sq

CUSUMSQResult = cusum_sq(
    results,
    alpha: float = 0.05,
    method: str = "asymptotic",
    n_boot: int = 2000,
    seed: int | None = None,
    standardize: bool = True,
)
```

Same attributes as `CUSUMResult` plus:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.expected` | `ndarray` | Expected path $t/n$ under stability |
| `.variance_increase` | `bool` | Whether variance increased (vs decreased) |

### `plot_cusum()`

```python
from kalmanbox.diagnostics import plot_cusum

fig = plot_cusum(
    results,
    alpha: float = 0.05,
    include_sq: bool = True,   # Plot both CUSUM and CUSUM-SQ
    figsize: tuple = (12, 5),
    dates: pd.DatetimeIndex | None = None,
)
```

Produces a two-panel figure (CUSUM on left, CUSUM-SQ on right) with confidence bands and
crossing annotations.

---

## 7. Examples

### Example 1: stable model — no break detected

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import cusum, cusum_sq, plot_cusum

rng = np.random.default_rng(0)
n = 200
y = np.cumsum(rng.normal(0, 0.5, n)) + rng.normal(0, 1, n)

model = LocalLevelModel()
results = model.fit(y)

cs  = cusum(results, alpha=0.05)
csq = cusum_sq(results, alpha=0.05)

print(cs.summary())
print(csq.summary())
```

**Expected output**:

```
CUSUM Test
==========
Supremum statistic : 14.21
Critical value     : 19.20  (alpha=0.05)
p-value            : 0.213
Conclusion         : Fail to reject parameter stability [PASS]

CUSUM-of-Squares Test
=====================
Supremum deviation : 0.048
Critical value     : 0.096  (alpha=0.05)
p-value            : 0.381
Conclusion         : Fail to reject variance stability [PASS]
```

**Interpretation**: Both CUSUM and CUSUM-SQ stay well within their confidence bands. There
is no evidence of a structural break — the model parameters are stable over the sample.

---

### Example 2: detecting a mean break in UK GDP growth

```python
import numpy as np
import pandas as pd
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import cusum, detect_breakpoint, plot_cusum

# Simulate GDP growth with a break at observation 60 (early 1990s recession)
rng = np.random.default_rng(42)
n = 120
growth = np.concatenate([
    rng.normal(0.6, 0.8, 60),   # Pre-break: higher growth
    rng.normal(0.1, 0.8, 60),   # Post-break: lower growth
])

model = LocalLevelModel()
results = model.fit(growth)

# CUSUM test
cs = cusum(results, alpha=0.05)
print(cs.summary())

# Formal breakpoint estimate
bp = detect_breakpoint(results, method="cusum")
print(f"\nBreakpoint detected at t={bp.location}")
print(f"95% confidence interval: [{bp.ci_lower}, {bp.ci_upper}]")
print(f"Bai-Perron p-value: {bp.pvalue:.4f}")

# Visualise
plot_cusum(results, alpha=0.05)
```

**Expected output**:

```
CUSUM Test
==========
Supremum statistic : 38.75
Critical value     : 15.68  (alpha=0.05)
p-value            : 0.001
Conclusion         : Reject parameter stability at 5% level [BREAK DETECTED]

Breakpoint detected at t=61
95% confidence interval: [55, 67]
Bai-Perron p-value: 0.0012
```

**Interpretation**: The CUSUM statistic (38.75) far exceeds the critical value (15.68).
The CUSUM path shows a kink at around $t = 60$–$61$, exactly where the mean shifted from
0.6 to 0.1. The 95 % confidence interval [55, 67] correctly brackets the true break at
$t = 60$.

**Possible actions**:

- Split the sample and estimate separate models for each regime.
- Model the break explicitly with a **dummy variable** in the observation equation.
- Switch to a **Time-Varying Parameters** (TVP) model that allows the level to drift freely.

---

### Example 3: variance break (CUSUM-SQ)

```python
import numpy as np
from kalmanbox import LocalLevelModel
from kalmanbox.diagnostics import cusum_sq, plot_cusum

rng = np.random.default_rng(7)
n = 150
# Variance doubles after observation 75
innovations = np.concatenate([
    rng.normal(0, 1, 75),
    rng.normal(0, 2, 75),
])
y = np.cumsum(rng.normal(0, 0.3, n)) + innovations

model = LocalLevelModel()
results = model.fit(y)

csq = cusum_sq(results, alpha=0.05)
print(csq.summary())
print(f"Variance increase detected: {csq.variance_increase}")

plot_cusum(results, include_sq=True)
```

**Expected output**:

```
CUSUM-of-Squares Test
=====================
Supremum deviation : 0.187
Critical value     : 0.111  (alpha=0.05)
p-value            : 0.003
Conclusion         : Reject variance stability at 5% level [BREAK DETECTED]
Variance increase detected: True
```

**Interpretation**: The CUSUM-SQ path rises above its expected value ($t/n$) after the
midpoint, reflecting the increased innovation variance. The CUSUM (mean test) may not detect
this break because the mean did not change, illustrating why both tests should always be
run together.

---

## 8. CUSUM and CUSUM-SQ interpretation guide

| CUSUM result | CUSUM-SQ result | Most likely cause |
|--------------|-----------------|-------------------|
| Within bands | Within bands | Parameter stable |
| **Exits bands** | Within bands | Mean / level break |
| Within bands | **Exits bands** | Variance break |
| **Both exit** | **Both exit** | Full structural break (mean + variance) |

After any detected break, the recommended next steps are:

1. Identify the breakpoint date using `detect_breakpoint()`.
2. Check external events at that date (policy change, financial crisis, data revision).
3. Choose a remediation: sample split, dummy variable, TVP, or regime-switching model.

---

## Related

- [Innovation tests](innovation-tests.md)
- [Residual analysis](residuals.md)
- [Prediction error analysis](prediction-error.md)
- [User guide: TVP models](../user-guide/advanced/tvp.md)
- [Theory: identifiability](../theory/identifiability.md)
- [API: diagnostics module](../api/diagnostics.md)
