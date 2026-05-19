# Missing Data

State-space models handle missing observations **natively**: when $y_t$ is not
observed, the Kalman filter simply skips the update step and lets the prediction
propagate the state forward. No imputation is needed before fitting.

---

## How the Kalman filter handles `NaN`

### Standard update step (observation available)

At time $t$ when $y_t$ is observed, the standard update is:

$$
\begin{aligned}
v_t &= y_t - Z_t\, a_{t|t-1} \\
F_t &= Z_t P_{t|t-1} Z_t' + H_t \\
K_t &= P_{t|t-1} Z_t' F_t^{-1} \\
a_{t|t}   &= a_{t|t-1} + K_t\, v_t \\
P_{t|t}   &= (I - K_t Z_t)\, P_{t|t-1}
\end{aligned}
$$

### Skip update step (observation missing)

When $y_t = \texttt{NaN}$, the update step is replaced by the identity:

$$
\boxed{
v_t = 0, \quad K_t = 0, \quad
a_{t|t} = a_{t|t-1}, \quad P_{t|t} = P_{t|t-1}
}
$$

The state estimate is **unchanged** — only the prediction step runs, so
uncertainty grows by $R_t Q_t R_t'$ at each missing period:

$$
P_{t+1|t} = T_t P_{t|t} T_t' + R_t Q_t R_t' = T_t P_{t|t-1} T_t' + R_t Q_t R_t'
$$

!!! info "Log-likelihood contribution"
    The log-likelihood contribution at a missing observation is **zero**.
    This is correct: missing data are uninformative, so they should not appear
    in the likelihood.

---

## Partial observations in multivariate models

When $y_t \in \mathbb{R}^p$ with $p > 1$, some but not all components may be
missing at a given time. `kalmanbox` handles this by **subsetting** the
observation equation.

Let $\mathcal{O}_t \subseteq \{1,\ldots,p\}$ be the set of observed indices.
Define $S_t$ as the selection matrix that keeps rows in $\mathcal{O}_t$:

$$
y_t^{(\mathcal{O})} = S_t\, y_t, \quad
Z_t^{(\mathcal{O})} = S_t\, Z_t, \quad
H_t^{(\mathcal{O})} = S_t\, H_t\, S_t'
$$

The update step then uses only the observed sub-vector:

$$
\begin{aligned}
v_t^{(\mathcal{O})} &= y_t^{(\mathcal{O})} - Z_t^{(\mathcal{O})}\, a_{t|t-1} \\
F_t^{(\mathcal{O})} &= Z_t^{(\mathcal{O})} P_{t|t-1} Z_t^{(\mathcal{O})'} + H_t^{(\mathcal{O})} \\
K_t^{(\mathcal{O})} &= P_{t|t-1}\, Z_t^{(\mathcal{O})'}\, \left(F_t^{(\mathcal{O})}\right)^{-1}
\end{aligned}
$$

This is automatic — no special configuration is needed.

---

## Interpolation via the RTS smoother

The filter only uses **past** observations. At a missing time $t$, the
filtered state $a_{t|t} = a_{t|t-1}$ is just the one-step-ahead prediction.

The **RTS smoother** runs the backward pass and uses **future** observations
to produce the best linear estimate during the missing period:

$$
a_{t|n} = E[\alpha_t \mid y_{1:t-1}, y_{t+1:n}]
$$

This is the gold-standard approach to **state-space interpolation** — it
naturally accounts for both the dynamics of the system and the noise level.

```
                    ┌─ filtered (forecast)
    Past ──────┬───────────────────── Future
               │   missing gap
               └─ smoothed (interpolated) ← uses both sides
```

---

## API and code examples

### Example 1: Nile River with a synthetic gap

```python
import numpy as np
import pandas as pd
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_nile

nile = load_nile()
y    = nile["volume"].to_numpy().copy()

# Introduce a 5-year gap at positions 10–14
gap_slice = slice(10, 15)
y_gap     = y.copy()
y_gap[gap_slice] = np.nan

# Fit on the incomplete series — MLE handles NaNs automatically
model   = LocalLevel(y_gap)
results = model.fit(method="mle", disp=False)

print(f"σ_η² = {results.params['sigma2_eta']:.4f}")
print(f"σ_ε² = {results.params['sigma2_eps']:.4f}")

# Filtered and smoothed states
filter_out  = results.filter()
smoother_out = results.smooth()

# During the gap, filtered state = predicted state (no update)
a_filt   = filter_out.a_filtered[:, 0]   # (n,)
a_smooth = smoother_out.a_smoothed[:, 0] # (n,)
P_smooth = smoother_out.P_smoothed[:, 0, 0]

print(f"\nGap period filtered: {a_filt[gap_slice]}")
print(f"Gap period smoothed: {a_smooth[gap_slice]}")

# Smoothed uncertainty is smaller — uses future observations
assert np.all(P_smooth[gap_slice] <= filter_out.P_filtered[gap_slice, 0, 0])
```

### Example 2: Visualization of filtered vs smoothed through a gap

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_nile

nile     = load_nile()
y        = nile["volume"].to_numpy().copy()
years    = nile.index.to_numpy()
gap      = slice(20, 30)  # 10-year gap

y_gap = y.copy()
y_gap[gap] = np.nan

model   = LocalLevel(y_gap)
results = model.fit(disp=False)
filt    = results.filter()
sm      = results.smooth()

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(years, y, "o", ms=3, color="gray", label="Observed", alpha=0.6)
ax.axvspan(years[gap.start], years[gap.stop - 1],
           alpha=0.12, color="orange", label="Missing gap")

a_f, P_f = filt.a_filtered[:, 0], filt.P_filtered[:, 0, 0]
a_s, P_s = sm.a_smoothed[:, 0], sm.P_smoothed[:, 0, 0]

ax.plot(years, a_f, "b--", lw=1.5, label="Filtered (no future info)")
ax.plot(years, a_s, "r-",  lw=2,   label="Smoothed (interpolated)")
ax.fill_between(years, a_s - 2*np.sqrt(P_s), a_s + 2*np.sqrt(P_s),
                alpha=0.15, color="red", label="Smoothed ±2σ")

ax.set_xlabel("Year"); ax.set_ylabel("Nile flow (10⁸ m³)")
ax.set_title("Missing data: filtered vs smoothed interpolation")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("missing_data_interpolation.png", dpi=150)
```

### Example 3: Multivariate model with partial observations

```python
import numpy as np
from kalmanbox import KalmanFilter, StateSpaceRepresentation

# Bivariate Local Level: two related series with occasional gaps
n  = 150
rng = np.random.default_rng(99)

# Simulate
mu   = np.cumsum(rng.normal(scale=0.3, size=n))    # shared trend
y1   = mu + rng.normal(scale=1.0, size=n)
y2   = 0.8 * mu + rng.normal(scale=0.5, size=n)
Y    = np.column_stack([y1, y2])                    # shape (150, 2)

# Random gaps in each series independently
Y[rng.choice(n, 15, replace=False), 0] = np.nan   # gaps in y1
Y[rng.choice(n, 20, replace=False), 1] = np.nan   # gaps in y2

# State-space: common trend
T = np.array([[1.0]])
Z = np.array([[1.0], [0.8]])     # (p=2, k=1)
R = np.array([[1.0]])
Q = np.array([[0.09]])           # σ_η² = 0.3²
H = np.diag([1.0, 0.25])        # (p=2, p=2)

ssr = StateSpaceRepresentation(T=T, Z=Z, R=R, Q=Q, H=H)
kf  = KalmanFilter(ssr, initialization="diffuse")
out = kf.run(Y)

print(f"Log-likelihood: {out.loglike:.4f}")
print(f"Filtered shape: {out.a_filtered.shape}")  # (150, 1)
# kalmanbox automatically uses only the non-NaN rows of Z and H at each t
```

### Example 4: Impact of missing data on parameter estimation

```python
import numpy as np
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_nile

nile = load_nile()
y    = nile["volume"].to_numpy()

# Fit on complete data
res_full = LocalLevel(y).fit(disp=False)

# Fit on data with 30% missing at random
rng   = np.random.default_rng(42)
y_mis = y.copy()
y_mis[rng.choice(len(y), size=int(0.3 * len(y)), replace=False)] = np.nan
res_mis = LocalLevel(y_mis).fit(disp=False)

print("Parameter   Complete   30% missing")
print(f"σ_η²        {res_full.params['sigma2_eta']:.4f}     "
      f"{res_mis.params['sigma2_eta']:.4f}")
print(f"σ_ε²        {res_full.params['sigma2_eps']:.4f}     "
      f"{res_mis.params['sigma2_eps']:.4f}")
print(f"Log-like    {res_full.loglike:.2f}       {res_mis.loglike:.2f}")
# Note: loglike is not comparable (different effective sample sizes)
```

---

## Behavior during the diffuse initialization period

During the first $d$ diffuse steps, a missing observation interacts with the
diffuse recursion. `kalmanbox` handles this correctly:

- If $y_t$ is missing during a diffuse step, both $P_t^{(\infty)}$ and
  $P_t^{(*)}$ remain unchanged (prediction step only), and $t$ still counts
  toward the diffuse period.
- The diffuse period ends at step $d$ regardless of how many observations
  were actually observed.

!!! warning "Missing observations at the very start"
    If $y_1$ through $y_d$ are all missing, the model cannot distinguish the
    initial level from noise. The diffuse likelihood is still well-defined but
    the initial state will be estimated with high uncertainty.

---

## Checklist for working with missing data

- [x] Encode missing values as `np.nan` (not zeros or sentinels)
- [x] Check `np.isnan(y).sum()` before fitting — know how many gaps you have
- [x] Use the **smoother** to interpolate, not the filter
- [x] Compare smoothed uncertainty $P_{t|n}$ with filtered $P_{t|t}$ — smoother should be tighter
- [x] For forecasting beyond the sample, missing values at the end are handled the same way

---

## Related pages

- [Diffuse Initialization](diffuse.md) — interaction of missing data with the initialization period
- [MLE Estimation](mle.md) — how missing data affects the log-likelihood
- [RTS Smoother](rts-smoother.md) — the backward pass that enables interpolation
- [Kalman Filter](kalman-filter.md) — the standard forward recursion
- [Tutorial: Nile River Local Level](../../tutorials/nile-local-level.md)
- [API: KalmanFilter](../../api/filters.md)
