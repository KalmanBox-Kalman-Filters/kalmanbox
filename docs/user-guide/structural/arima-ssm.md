# ARIMA in state-space form

Any ARIMA$(p,d,q)$ or SARIMA$(p,d,q)(P,D,Q,s)$ model can be cast as a linear
Gaussian state-space model and filtered with the Kalman filter. `kalmanbox`
provides this representation as `ARIMA_SSM`, giving ARIMA models the full
state-space toolchain: **exact likelihood**, **missing data**, **diffuse
initialisation**, **RTS smoothing**, and seamless combination with structural
components.

---

## Why state-space ARIMA?

Classical ARIMA implementations (Box-Jenkins, `statsmodels.tsa.arima`) compute
either the **conditional likelihood** (conditioning on the first $\max(p,q)$
observations) or the **unconditional likelihood** via the Yule-Walker
initialisation. The state-space approach computes the **exact Gaussian likelihood**
via the Kalman prediction error decomposition:

$$
\ell(\theta) = -\frac{n}{2}\log 2\pi
              - \frac{1}{2} \sum_{t=1}^{n} \bigl(\log f_t + v_t^2 / f_t\bigr)
$$

where $v_t = y_t - Z a_t$ is the innovation and $f_t$ its variance. This
matters for:

| Feature | Classical ARIMA | ARIMA-SSM |
|---------|:--------------:|:---------:|
| Exact likelihood | ✗ (conditional) | ✅ |
| Missing observations | ✗ (requires imputation) | ✅ (Kalman skip) |
| Diffuse init for $d \geq 1$ | Approximate | ✅ (exact diffuse) |
| RTS smoother | ✗ | ✅ |
| Embed in UCM | ✗ | ✅ |
| Forecast variance | Approximate | ✅ (correct propagation) |

---

## State-space representation of ARMA$(p, q)$

For the stationary ARMA$(p,q)$ process:

$$
y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p}
    + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q},
\qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

define $r = \max(p,\, q+1)$ and the state $\alpha_t = (\alpha_{1,t}, \ldots, \alpha_{r,t})'
\in \mathbb{R}^r$. The **companion-form** representation is:

### Transition matrix $T$ ($r \times r$)

$$
T = \begin{pmatrix}
\phi_1   & 1 & 0 & \cdots & 0 \\
\phi_2   & 0 & 1 & \cdots & 0 \\
\vdots   &   &   & \ddots & \vdots \\
\phi_{r-1} & 0 & 0 & \cdots & 1 \\
\phi_r   & 0 & 0 & \cdots & 0
\end{pmatrix}
$$

where $\phi_j = 0$ for $j > p$.

### Selection vector $R$ ($r \times 1$)

$$
R = \begin{pmatrix} 1 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_{r-1} \end{pmatrix}
$$

where $\theta_j = 0$ for $j > q$.

### Observation row $Z$ ($1 \times r$)

$$
Z = \begin{pmatrix} 1 & 0 & \cdots & 0 \end{pmatrix}
$$

### State disturbance covariance $Q$ ($1 \times 1$)

$$
Q = \sigma^2
$$

The scalar disturbance $\varepsilon_t \sim \mathcal{N}(0,\sigma^2)$ enters the state
through $R$, and $H = 0$ (no additional observation noise).

### Observation equation

$$
y_t = Z \alpha_t = \alpha_{1,t}
$$

The first state element always equals the current observation.

---

## Worked example: ARMA(2, 1)

For $y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \varepsilon_t + \theta_1 \varepsilon_{t-1}$,
we have $p=2$, $q=1$, $r = \max(2, 2) = 2$:

$$
T = \begin{pmatrix} \phi_1 & 1 \\ \phi_2 & 0 \end{pmatrix},
\quad
R = \begin{pmatrix} 1 \\ \theta_1 \end{pmatrix},
\quad
Z = \begin{pmatrix} 1 & 0 \end{pmatrix}
$$

Substituting the recursion $\alpha_{1,t+1} = \phi_1 \alpha_{1,t} + \alpha_{2,t} + \varepsilon_{t+1}$
recovers the ARMA equations by construction.

---

## ARIMA$(p, d, q)$: integrating the state

For $d \geq 1$, the AR polynomial has $d$ unit roots. kalmanbox handles this by
augmenting the state with $d$ **integrated dimensions** and applying **exact
diffuse initialisation** (Koopman 1997) on those dimensions:

$$
\alpha_t^{(\text{ARIMA})} = \underbrace{(\Delta^d y_t, \, \ldots)}_{\text{ARMA part, }r\text{ dims}}
\oplus \underbrace{(y_{t-1}^{(d-1)}, \ldots, y_{t-1}^{(0)})}_{\text{integration: }d\text{ dims}}
$$

The total state dimension is $r + d$ where $r = \max(p, q+1)$.

The diffuse prior on the integrated states contributes to the prediction error
decomposition through the **modified diffuse likelihood** (see [Diffuse initialisation](../kalman/diffuse-initialization.md)).

---

## SARIMA$(p,d,q)(P,D,Q,s)$ in state space

A seasonal ARIMA model first expands the lag polynomials:

$$
\Phi(B^s)\, \phi(B)\, \Delta_s^D \Delta^d y_t = \Theta(B^s)\, \theta(B)\, \varepsilon_t
$$

where:

- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ — non-seasonal AR
- $\Phi(B^s) = 1 - \Phi_1 B^s - \cdots - \Phi_P B^{Ps}$ — seasonal AR
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ — non-seasonal MA
- $\Theta(B^s) = 1 + \Theta_1 B^s + \cdots + \Theta_Q B^{Qs}$ — seasonal MA
- $\Delta^d = (1-B)^d$, $\Delta_s^D = (1-B^s)^D$ — differencing operators

After expanding, the combined polynomial is an ARMA process with orders:

$$
p^* = p + Ps, \qquad q^* = q + Qs
$$

and $d^* = d + Ds$ unit roots (some at frequency 0, some at seasonal frequencies).
The companion-form construction proceeds as in the non-seasonal case with
$r = \max(p^*, q^*+1)$ and exact diffuse initialisation on all $d^*$ integrated
dimensions.

!!! note "State dimension for SARIMA"

    For SARIMA$(1,1,1)(1,1,1)_{12}$ (the "airline model"):
    $p^* = 1 + 12 = 13$, $q^* = 1 + 12 = 13$, $r = 14$, $d^* = 13$.
    Total state dimension = $14 + 13 = 27$. This is larger than the UCM state
    for the equivalent BSM (13 dimensions), but both models should fit equally
    well on airline-type data.

---

## Usage

### ARIMA(1, 1, 1)

```python
import numpy as np
from kalmanbox import ARIMA_SSM
from kalmanbox.datasets import load_dataset

y = load_dataset("us_gdp")["log_gdp"].to_numpy()

model   = ARIMA_SSM(y, order=(1, 1, 1))
results = model.fit(method="mle", disp=True)

print(results.summary())
```

```
         ARIMA-SSM Results (exact likelihood)
================================================
Model:       ARIMA(1, 1, 1)
Dep. Variable: log_gdp
No. Observations: 248
Log-Likelihood: 478.392
AIC: -948.784     BIC: -935.261     HQIC: -943.250
================================================
            Estimate  Std.Err  z-stat  p-value
phi_1       0.3812    0.0641   5.948   0.0000
theta_1    -0.7541    0.0513  -14.70   0.0000
sigma2      0.0004    0.0001   4.001   0.0001
================================================
Ljung-Box Q (lag=10):  11.84  (p=0.296)
Jarque-Bera:            2.12  (p=0.347)
Heteroskedasticity:     1.08  (p=0.431)
================================================
```

### SARIMA(1, 1, 1)(1, 1, 1, 12) — the airline model

```python
import numpy as np
from kalmanbox import ARIMA_SSM
from kalmanbox.datasets import load_airline

y_log = np.log(load_airline()["passengers"].to_numpy())

model   = ARIMA_SSM(y_log, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
results = model.fit(disp=True)

print(results.summary())
```

```
      ARIMA-SSM Results — SARIMA(0,1,1)(0,1,1,12)
======================================================
No. Observations: 144   Log-Likelihood:  244.703
AIC: -483.405           BIC: -474.899
======================================================
            Estimate  Std.Err  z-stat  p-value
theta_1    -0.4018    0.0864   -4.65   0.0000
Theta_1    -0.5569    0.0731   -7.62   0.0000
sigma2      0.00134   0.0002    6.71   0.0000
======================================================
```

### Forecast with confidence intervals

```python
fc = results.forecast(steps=24, alpha=0.05)

import matplotlib.pyplot as plt
import pandas as pd

hist_dates = load_airline().index
fc_dates   = pd.date_range(hist_dates[-1], periods=25, freq="MS")[1:]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hist_dates, np.exp(y_log), "k-", label="Observed")
ax.plot(fc_dates, np.exp(fc.mean + 0.5*fc.variance), "r-", lw=2,
        label="Forecast (log-normal mean)")
ax.fill_between(fc_dates, np.exp(fc.lower_95), np.exp(fc.upper_95),
                alpha=0.2, color="red", label="95% PI")
ax.set_title("SARIMA(0,1,1)(0,1,1,12) — airline passengers")
ax.legend()
plt.tight_layout()
```

---

## Missing data

The state-space form handles missing observations natively: when $y_t$ is `NaN`,
the Kalman update step is skipped (the filter propagates using only the prediction
step). No imputation is required.

```python
import numpy as np
from kalmanbox import ARIMA_SSM

rng = np.random.default_rng(0)
y   = rng.standard_normal(200)
y[40:45]  = np.nan          # 5 consecutive missing observations
y[100]    = np.nan          # single isolated missing

model   = ARIMA_SSM(y, order=(2, 0, 1))
results = model.fit()

# The filtered/smoothed series interpolates through the gaps
sm = results.smooth()
print(sm.states[38:48, 0])  # states around the gap — smoothly interpolated
```

---

## Comparison: ARIMA-SSM vs classical ARIMA

```python
import numpy as np
from kalmanbox import ARIMA_SSM
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
from kalmanbox.datasets import load_airline

y_log = np.log(load_airline()["passengers"].to_numpy())

# kalmanbox — exact likelihood
r_ssm = ARIMA_SSM(y_log, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)

# statsmodels — conditional likelihood (default in older versions)
r_sm  = SM_ARIMA(y_log, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)

print(f"kalmanbox exact LL   : {r_ssm.loglike:.4f}")
print(f"statsmodels LL       : {r_sm.llf:.4f}")
print(f"kalmanbox AIC        : {r_ssm.aic:.4f}")
print(f"statsmodels AIC      : {r_sm.aic:.4f}")
```

For short series or when $d + D > 0$, the exact likelihood can differ noticeably
from the conditional likelihood, especially in the first few observations.

---

## When to use ARIMA-SSM vs classical ARIMA

Use **ARIMA-SSM** when:

- Your series has **missing values** — the Kalman filter handles them exactly.
- You want **RTS smoothing** for a retrospective analysis of the latent ARMA process.
- You want to **combine** ARIMA dynamics with structural components (trend, seasonal,
  cycle) in a single state-space model — classical ARIMA cannot do this.
- You need the **exact likelihood** for model comparison with structural models
  (UCM, BSM) on the same scale.
- You apply ARIMA to a **non-stationary series** and want exact diffuse initialisation
  rather than an approximate unconditional variance.

Use **classical ARIMA** (e.g., `statsmodels.tsa.arima`) when:

- You want a quick benchmark on a complete series with no structural interpretation.
- You need the large set of diagnostic plots and test batteries that `statsmodels`
  provides.
- Approximate likelihood is acceptable (long series, no missing data).

---

## ARIMA-SSM as a special UCM

An ARIMA-SSM is equivalent to a UCM with only an AR component and no structural
components. You can build the same model explicitly via `UCM`:

```python
from kalmanbox.structural import UCM

# ARIMA(2, 1, 0) in UCM notation:
# - AR(2) component
# - Integration handled by level with unit root
model_ucm = UCM(y, level=True, ar=2, irregular=False)
```

For pure ARIMA use, `ARIMA_SSM` is more convenient. Use `UCM` when you want to
**add structural components** to an ARIMA backbone.

---

## SARIMA as a UCM + seasonal component

The SARIMA airline model is structurally similar to the BSM with a trigonometric
seasonal. The key difference is the MA parameterisation vs. variance parameterisation:

| Feature | SARIMA$(0,1,1)(0,1,1)_{12}$ | BSM (monthly) |
|---------|:--------------------------:|:-------------:|
| Trend | ARIMA(0,1,1) | Random-walk level + slope |
| Seasonal | Seasonal MA | Dummy/trigonometric seasonal |
| Parameters | $\theta_1, \Theta_1, \sigma^2$ (3) | $\sigma_\eta^2, \sigma_\zeta^2, \sigma_\omega^2, \sigma_\varepsilon^2$ (4) |
| Interpretability | Low | High |
| Missing data | ✅ (SSM) | ✅ |
| Decomposition | ✗ | ✅ |

On the airline data both models achieve similar AIC, but BSM additionally provides
an interpretable trend/seasonal decomposition and a direct test for stochastic
vs. deterministic seasonality.

---

## Initialisation

The initialisation strategy follows the differencing order:

| Condition | Initial prior |
|-----------|--------------|
| $d = 0$, $D = 0$ (stationary) | Unconditional variance (Yule-Walker) |
| $d + D > 0$ (unit roots) | Exact diffuse for integrated dimensions |

Diffuse dimensions use the **modified initial Kalman filter** of Durbin & Koopman
(2002), which ensures the diffuse likelihood contribution is separated from the
stationary part. See [Diffuse initialisation](../kalman/diffuse-initialization.md).

---

## API reference

::: kalmanbox.models.arima_ssm.ARIMA_SSM
    options:
      heading_level: 3
      show_source: false

---

## Related

- [UCM](ucm.md) — generalises ARIMA-SSM by adding structural components
- [BSM](bsm.md) — structural alternative to SARIMA for seasonal economic series
- [Diffuse initialisation](../kalman/diffuse-initialization.md) — exact treatment
  of unit roots
- [Missing data](../kalman/missing-data.md) — gap handling by the Kalman filter
- [Theory: state-space foundations](../../theory/state-space-theory.md)
- [Choosing a model](../../getting-started/choosing-model.md)
- [API: structural models](../../api/models.md)

### References

- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
  (2nd ed.). Oxford University Press. §4.4–4.6.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman
  Filter.* Cambridge University Press. §4.3.
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton University Press. Ch. 13.
- Koopman, S. J. (1997). Exact initial Kalman filtering and smoothing for
  non-stationary time series models. *Journal of the American Statistical Association*,
  92(440), 1630–1638.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015).
  *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley. Ch. 3–9.
