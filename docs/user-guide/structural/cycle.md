# Cycle component

The **stochastic cycle** captures medium-frequency, quasi-periodic variation that
lies between the slowly-evolving trend and the repeating seasonal pattern. It is the
key component for modelling **business cycles**, inventory cycles, and any series
whose spectrum shows a dominant frequency that is neither annual nor at very low
frequencies.

The cycle can be used as a stand-alone model (`CycleModel`) or embedded inside a
[UCM](ucm.md) alongside trend and seasonal components.

---

## Motivation

Many economic series contain oscillations of 2–10 year period that are neither
trend nor seasonality. Classical ARIMA models absorb these as MA/AR coefficients
without providing an interpretable cycle. A structural cycle component explicitly
models:

- A **frequency** $\lambda_c$ (estimated from data, or constrained to a range).
- A **damping factor** $\rho_c$ that controls how persistent the cycle is.
- A **stochastic amplitude** $\sigma_\kappa^2$ that allows the cycle to grow or
  shrink over time.

---

## Trigonometric formulation

The stochastic cycle is represented as a 2-dimensional rotating system:

$$
\begin{pmatrix} \psi_{t+1} \\ \psi_{t+1}^* \end{pmatrix}
=
\rho_c
\underbrace{\begin{pmatrix} \cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c \end{pmatrix}}_{R(\lambda_c)}
\begin{pmatrix} \psi_t \\ \psi_t^* \end{pmatrix}
+
\begin{pmatrix} \kappa_t \\ \kappa_t^* \end{pmatrix}
$$

with innovations:

$$
\begin{pmatrix} \kappa_t \\ \kappa_t^* \end{pmatrix}
\sim \mathcal{N}\!\left(\mathbf{0},\; \sigma_\kappa^2 I_2\right)
$$

Only $\psi_t$ (the first element) enters the observation equation. The auxiliary
state $\psi_t^*$ is required to make $(\psi_t, \psi_t^*)$ a proper rotation; without
it the recursion collapses to a damped cosine with no innovation path through $\sin$.

### Derivation: why a 2 × 2 system?

A single-state recursion $\psi_{t+1} = \rho \cos\lambda \cdot \psi_t + \kappa_t$
cannot sustain oscillations for long because the innovation can only push the cycle
in one direction. The rotation matrix $R(\lambda)$ rotates the 2-d state by angle
$\lambda$ each period; together with the damping $\rho$, it produces a **spiral**
in the $(\psi, \psi^*)$ plane whose projection onto the $\psi$ axis is an
exponentially decaying oscillation with frequency $\lambda$. The equal variances
$\text{Var}(\kappa_t) = \text{Var}(\kappa_t^*) = \sigma_\kappa^2$ keep the
rotational symmetry intact.

---

## Parameters

| Parameter | Symbol | Range | Interpretation |
|-----------|--------|-------|---------------|
| Frequency | $\lambda_c$ | $(0, \pi)$ | Radians per time unit; period $= 2\pi/\lambda_c$ |
| Damping factor | $\rho_c$ | $(0, 1]$ | 1 = undamped (unit root in cycle); < 1 = mean-reverting |
| Innovation variance | $\sigma_\kappa^2$ | $> 0$ | Controls amplitude variation over time |

### Period conversion

| Series frequency | Period (years) | $\lambda_c$ (rad/period) |
|-----------------|:--------------:|:------------------------:|
| Monthly, 5-yr business cycle | 60 months | $2\pi/60 \approx 0.105$ |
| Monthly, 8-yr business cycle | 96 months | $2\pi/96 \approx 0.065$ |
| Quarterly, 5-yr cycle | 20 quarters | $2\pi/20 \approx 0.314$ |
| Annual, Kuznets cycle (20 yr) | 20 years | $2\pi/20 \approx 0.314$ |

---

## Spectral density

The marginal spectral density of the stationary cycle $\psi_t$ (for $\rho_c < 1$) is:

$$
f(\omega) = \frac{\sigma_\kappa^2}{2\pi}
\left[
  \frac{1 - \rho_c^2}{1 - 2\rho_c\cos(\omega - \lambda_c) + \rho_c^2}
  +
  \frac{1 - \rho_c^2}{1 - 2\rho_c\cos(\omega + \lambda_c) + \rho_c^2}
\right]
$$

This is a symmetric bimodal spectrum with peaks at $\pm \lambda_c$.

- As $\rho_c \to 1$: the peaks sharpen (narrow-band, near-deterministic cycle).
- As $\rho_c \to 0$: the spectrum flattens (white noise, no oscillation).
- As $\sigma_\kappa^2 \to 0$: the cycle amplitude shrinks to zero (deterministic limit).

---

## Stochastic vs. deterministic cycle

| Type | Parameters fixed | State | Use when |
|------|-----------------|-------|---------|
| **Stochastic** | $\rho_c, \lambda_c, \sigma_\kappa^2$ estimated | 2-d random walk on spiral | Cycle amplitude changes over time |
| **Near-deterministic** | $\sigma_\kappa^2 \approx 0$, $\rho_c \approx 1$ | 2-d fixed spiral | Amplitude stable, frequency dominant |
| **Fixed frequency** | $\lambda_c$ fixed, $\rho_c, \sigma_\kappa^2$ estimated | 2-d with known period | Economic theory constrains the period |

```python
from kalmanbox.structural import UCM

# Stochastic cycle — frequency and amplitude both evolve
model_stoch = UCM(y, level=True, cycle=True, irregular=True)

# Fixed-period cycle — e.g., 60-month business cycle
model_fixed = UCM(y, level=True, cycle=True, cycle_period=60, irregular=True)
```

---

## Standalone usage

```python
import numpy as np
from kalmanbox import CycleModel
from kalmanbox.datasets import load_gdp_gap

gap = load_gdp_gap()
y   = gap["output_gap"].to_numpy()   # quarterly output gap

model   = CycleModel(y)
results = model.fit(method="mle", n_starts=10, disp=True)

print(results.summary())
```

```
              Cycle Model Results
=============================================
Dep. Variable:   output_gap
No. Observations: 188
Log-Likelihood:  -94.371
AIC:             -180.742
BIC:             -170.108
=============================================
          Estimate  Std.Err  z-stat  p-value
rho       0.9248    0.0187   49.45   0.0000
lambda    0.0712    0.0053   13.43   0.0000  (period ≈ 88 quarters ≈ 22 years)
sigma2_k  0.3812    0.0641    5.95   0.0000
sigma2_e  0.1229    0.0312    3.94   0.0001
=============================================
```

### Extract and plot the cycle

```python
sm    = results.smooth()
cycle = sm.states[:, 0]          # first state dimension = ψ_t

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(gap.index, y, "k-", lw=0.8, alpha=0.7, label="Output gap")
axes[0].plot(gap.index, cycle, "r-", lw=2, label="Smoothed cycle $\\hat{\\psi}_t$")
axes[0].legend()
axes[0].set_title("Output gap vs. smoothed cycle")

axes[1].plot(gap.index, sm.states[:, 1], "b-", lw=1.5, label="Auxiliary $\\psi_t^*$")
axes[1].axhline(0, color="grey", ls="--", alpha=0.4)
axes[1].set_title("Auxiliary state $\\psi_t^*$")
axes[1].legend()

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
```

---

## Multiple cycles

Some series contain cycles at two distinct frequencies simultaneously — for example,
a monthly series might carry both a **business cycle** (5–10 years) and an
**intermediate cycle** (18–24 months, e.g., Kitchin inventory cycle).

UCM supports multiple cycles by supplying a list to `cycle`:

```python
from kalmanbox.structural import UCM

model = UCM(
    y,
    level=True,
    slope=True,
    cycle=[
        dict(period_bounds=(24, 120)),   # business cycle 2–10 years
        dict(period_bounds=(12, 24)),    # Kitchin cycle 1–2 years
    ],
    irregular=True,
)
results = model.fit(n_starts=20, disp=False)

sm       = results.smooth()
cycle_bc = sm.components["cycle_0"]     # business cycle
cycle_kt = sm.components["cycle_1"]     # Kitchin cycle

print(f"Business cycle period : {results.cycle_period(0):.1f} months")
print(f"Kitchin cycle period  : {results.cycle_period(1):.1f} months")
```

Each cycle adds 3 parameters ($\rho$, $\lambda$, $\sigma_\kappa^2$) and 2 state
dimensions. With two cycles on a monthly series the state grows by 4.

!!! warning "Identification with multiple cycles"

    Two cycles with overlapping period bounds may swap labels across optimization
    restarts. Use `cycle_period_bounds` disjoint ranges and `n_starts ≥ 20` to
    guard against local optima.

---

## Cycle in UCM: trend + seasonal + cycle

The most common applied configuration is trend + seasonal + cycle, covering
economic series that have all three types of variation:

```python
from kalmanbox.structural import UCM
import numpy as np

# Monthly industrial production, not seasonally adjusted
from kalmanbox.datasets import load_dataset
y = load_dataset("industrial_production")["index"].to_numpy()

model = UCM(
    y,
    level=True,
    slope=True,
    seasonal=12,
    cycle=True,
    cycle_period_bounds=(18, 84),          # 1.5 – 7 year cycle
    irregular=True,
)
results = model.fit(n_starts=15, disp=False)
sm      = results.smooth()

# Variance decomposition
total_var = np.var(y)
comps = {k: np.var(sm.components[k]) for k in ["level", "seasonal", "cycle", "irregular"]}
print("Variance decomposition:")
for k, v in comps.items():
    print(f"  {k:12s}: {100*v/total_var:5.1f}%")
```

---

## Estimating cycle frequency

When you have economic priors on the cycle period, constrain the search:

```python
model = UCM(
    y,
    level=True,
    cycle=True,
    cycle_period_bounds=(60, 120),   # NBER-style recession: 5–10 year period
    irregular=True,
)
results = model.fit(n_starts=10)

# Posterior of cycle frequency via profile likelihood
freq_grid = np.linspace(0.05, 0.30, 200)
ll_profile = [
    UCM(y, level=True, cycle=True, cycle_freq=lam, irregular=True)
    .fit(disp=False).loglike
    for lam in freq_grid
]

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 3))
plt.plot(2*np.pi/freq_grid, ll_profile)    # plot as period
plt.xlabel("Cycle period (time units)")
plt.ylabel("Log-likelihood")
plt.title("Profile likelihood over cycle period")
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

---

## State-space representation

The cycle block of the UCM system matrices (for a single cycle) is:

### Transition block $T^{(\psi)}$

$$
T^{(\psi)} = \rho_c \begin{pmatrix} \cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c \end{pmatrix}
$$

### Observation row $Z^{(\psi)}$

$$
Z^{(\psi)} = \begin{pmatrix} 1 & 0 \end{pmatrix}
$$

Only $\psi_t$ is observed; $\psi_t^*$ is a hidden auxiliary.

### Selection matrix $R^{(\psi)}$

$$
R^{(\psi)} = I_2
$$

Both $\kappa_t$ and $\kappa_t^*$ enter the state independently.

### Disturbance covariance $Q^{(\psi)}$

$$
Q^{(\psi)} = \sigma_\kappa^2 I_2
$$

Equal variances preserve the rotational symmetry of the cycle.

---

## Initialisation

For $\rho_c < 1$ the cycle is stationary; its unconditional distribution is:

$$
\begin{pmatrix} \psi_1 \\ \psi_1^* \end{pmatrix}
\sim \mathcal{N}\!\left(\mathbf{0},\; \frac{\sigma_\kappa^2}{1-\rho_c^2} I_2\right)
$$

kalmanbox uses this **stationary initialisation** by default when `rho_cycle < 1`.
When $\rho_c = 1$ (unit root cycle), the filter falls back to a diffuse prior
on the cycle states.

---

## API reference

::: kalmanbox.models.cycle.CycleModel
    options:
      heading_level: 3
      show_source: false

---

## Related

- [UCM](ucm.md) — the natural container for cycle components alongside trend and seasonal
- [BSM](bsm.md) — no cycle; use UCM if you need one
- [ARIMA-SSM](arima-ssm.md) — AR components can approximate cycles, but without
  structural interpretation
- [Theory: identifiability](../../theory/identifiability.md) — cycle vs. seasonal
  frequency collision
- [Theory: structural models](../../theory/structural-models.md)
- [Visualization: spectral analysis](../../visualization/spectral.md)
- [API: structural models](../../api/models.md)

### References

- Harvey, A. C. (1985). Trends and cycles in macroeconomic time series. *Journal
  of Business & Economic Statistics*, 3(3), 216–227.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter.* Cambridge University Press. §2.3, §3.5.
- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press. §3.3.
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton University Press. Ch. 6.
