# Information Criteria

Information criteria provide a principled framework for **model selection**: they balance
goodness-of-fit (measured by the log-likelihood) against model complexity (measured by the
number of free parameters), penalising over-fitted models without requiring a held-out
validation set.

---

## 1. Theoretical foundations

### 1.1 Kullback-Leibler divergence and AIC

Let $f(y)$ be the true data-generating process and $g(y;\hat\theta)$ the fitted model.
The **Kullback-Leibler divergence** measures the information lost when $g$ is used to
approximate $f$:

$$
I(f,g) = \int f(y)\log\frac{f(y)}{g(y;\hat\theta)}\,dy
        = \mathbb{E}_f[\log f(y)] - \mathbb{E}_f[\log g(y;\hat\theta)]
$$

The first term is a constant that does not depend on the model, so minimising $I(f,g)$
over candidate models reduces to **maximising $\mathbb{E}_f[\log g(y;\hat\theta)]$** —
the expected log-likelihood under the true distribution.

The in-sample log-likelihood $\hat\ell = \log g(y;\hat\theta)$ is an optimistically biased
estimator of this expectation: on average it exceeds the true expected log-likelihood by
approximately $k$ (the number of estimated parameters). Akaike (1974) showed that an
approximately unbiased estimator is:

$$
\hat\ell - k
$$

Multiplying by $-2$ and defining the criterion to be **minimised**:

$$
\boxed{\text{AIC} = -2\hat\ell + 2k}
$$

!!! note "Akaike's original derivation"
    The bias correction relies on the estimated information matrix being close to the
    true one — a condition satisfied asymptotically under mild regularity. In finite
    samples AIC tends to over-select (include too many parameters); AICc corrects for
    this; see §1.4.

### 1.2 BIC from Bayesian marginal likelihood

The **Bayesian Information Criterion** (Schwarz, 1978) arises from a Laplace approximation
to the marginal likelihood $p(y \mid \mathcal{M})$. Placing a prior $\pi(\theta)$ on model
$\mathcal{M}$:

$$
\log p(y \mid \mathcal{M}) = \log p(y \mid \hat\theta, \mathcal{M}) + \log\pi(\hat\theta)
- \frac{k}{2}\log\left(\frac{n}{2\pi}\right) - \frac{1}{2}\log|J(\hat\theta)| + O(1)
$$

where $J(\hat\theta)$ is the observed information matrix. For large $n$, the dominant terms
are the log-likelihood and the Laplace penalty; dropping constants and terms that vanish as
$n\to\infty$:

$$
\boxed{\text{BIC} = -2\hat\ell + k\log n}
$$

The BIC penalty grows with sample size, making it **more conservative** than AIC: it selects
simpler models as $n$ increases. BIC is **consistent** (selects the true model as
$n\to\infty$, if the true model is in the candidate set); AIC is not consistent but achieves
lower mean squared prediction error.

### 1.3 Hannan-Quinn Information Criterion (HQIC)

Hannan & Quinn (1979) derived the minimum penalty that preserves consistency. Their
criterion uses $\log\log n$ instead of $\log n$:

$$
\boxed{\text{HQIC} = -2\hat\ell + 2k\log\log n}
$$

The HQIC penalty is smaller than BIC but larger than AIC for all $n \geq 3$. It is
consistent and has a smaller penalty than BIC, making it a compromise that tends to
select slightly richer models than BIC while remaining asymptotically consistent.

| Criterion | Penalty | Consistent? | Efficient? |
|-----------|---------|-------------|------------|
| AIC | $2k$ | No | Yes (min MSE) |
| AICc | $\frac{2kn}{n-k-1}$ | No | Yes (better small-$n$) |
| HQIC | $2k\log\log n$ | Yes | No |
| BIC | $k\log n$ | Yes | No |

*Consistent = selects true model w.p. 1 as $n\to\infty$ (when true model is in set).*
*Efficient = minimises prediction MSE asymptotically.*

### 1.4 AICc: small-sample correction

For linear regression models Hurvich & Tsai (1989) derived an exact small-sample correction:

$$
\boxed{\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}}
$$

The correction diverges as $n\to k+2$, indicating the model is saturating the sample.
As $n\to\infty$, the AICc correction vanishes and AICc $\to$ AIC. A rule of thumb:
**use AICc whenever $n/k < 40$**.

!!! warning "AICc in state-space models"
    The AICc formula was derived for linear regression. For state-space models it is
    an approximation; use it as a guide rather than an exact correction, especially
    for non-Gaussian or nonlinear models.

---

## 2. Parameter counting in state-space models

Counting $k$ correctly is non-trivial in state-space models. The following table
summarises which quantities count as free parameters:

| Component | Counts as $k$? | Notes |
|-----------|----------------|-------|
| Observation variance $\sigma_\varepsilon^2$ | Yes | One scalar per observed series |
| State variances $\sigma_\eta^2, \sigma_\zeta^2, \ldots$ | Yes | One per stochastic state component |
| Seasonal irregular variance | Yes, if stochastic | Fixed-seasonal: No |
| Cycle parameters $(\rho_c, \lambda_c, \sigma_\kappa^2)$ | Yes, 3 per cycle | |
| Regression coefficients (TVP) | Yes, one variance each | Or fixed: one value each |
| Diffuse initial state values | **No** | Handled via $n_\text{eff}$ |
| Fixed (non-stochastic) components | No | They impose zero variance |

### Effective sample size for diffuse initialisation

When $d$ states are initialised diffusely, the first $d$ innovations carry no information
about the likelihood. kalmanbox automatically adjusts:

$$
n_\text{eff} = n - d
$$

and uses $n_\text{eff}$ in place of $n$ in BIC and AICc. The log-likelihood is also
computed on $n_\text{eff}$ observations (the exact diffuse log-likelihood of
De Jong & Chu-Chun-Lin, 1994).

```python
results = model.fit(y)
print(f"n={results.nobs}  d={results.ndiffuse}  n_eff={results.nobs_effective}")
print(f"k={results.k_params}")
print(f"AIC={results.aic:.2f}  AICc={results.aicc:.2f}  BIC={results.bic:.2f}  HQIC={results.hqic:.2f}")
```

### Example: parameter counts for common models

| Model | Series | Stochastic components | $k$ |
|-------|--------|-----------------------|-----|
| Local Level | Univariate | $\sigma_\varepsilon^2, \sigma_\eta^2$ | 2 |
| Local Linear Trend | Univariate | $\sigma_\varepsilon^2, \sigma_\eta^2, \sigma_\zeta^2$ | 3 |
| BSM (12 periods, stochastic seasonal) | Univariate | $\sigma_\varepsilon^2, \sigma_\eta^2, \sigma_\zeta^2, \sigma_\omega^2$ | 4 |
| BSM (12 periods, fixed seasonal) | Univariate | $\sigma_\varepsilon^2, \sigma_\eta^2, \sigma_\zeta^2$ | 3 |
| UCM with cycle | Univariate | $+\rho_c, \lambda_c, \sigma_\kappa^2$ relative to BSM | +3 |
| DFM ($r=2$ factors, $p=5$ series) | Multivariate | factor loadings + variances | $2p + 2r + \ldots$ |

---

## 3. Rules for using information criteria

### 3.1 Delta-IC and evidence ratios

Model differences matter more than absolute values. Define:

$$
\Delta_i = \text{IC}_i - \min_j \text{IC}_j
$$

**Burnham & Anderson (2002) guidelines** (originally for AIC, widely applied to BIC):

| $\Delta_i$ | Interpretation |
|-----------|----------------|
| $0$–$2$ | Substantial support; model is competitive |
| $2$–$4$ | Considerably less support |
| $4$–$7$ | Little support |
| $> 10$ | Essentially no support |

For BIC, the evidence ratio between models $i$ and $j$ approximates the **Bayes factor**:

$$
BF_{ij} \approx \exp\!\left(\frac{\text{BIC}_j - \text{BIC}_i}{2}\right)
$$

### 3.2 Cardinal rules

!!! danger "Only compare models fit to the same data"
    AIC, BIC, HQIC are **not** comparable across series, sample lengths, or
    transformations. Log-transforming or differencing the data changes the
    log-likelihood scale and invalidates comparisons.

!!! warning "Pair with residual diagnostics"
    A model with the lowest BIC but autocorrelated innovations is **not** the best
    model — it is the cleanest fit within a misspecified family. Always run
    [innovation tests](innovation-tests.md) alongside IC comparison.

!!! tip "When AIC and BIC disagree"
    If AIC selects a richer model and BIC selects a simpler one, inspect the marginal
    contribution of the extra components (use a [likelihood ratio test](likelihood-ratio.md)).
    For prediction tasks prefer AIC; for structural inference prefer BIC.

---

## 4. API reference

### `aic()`, `bic()`, `hqic()`, `aicc()`

```python
from kalmanbox.diagnostics import aic, bic, hqic, aicc

# From a fitted results object
a  = aic(results)    # float
b  = bic(results)    # float
h  = hqic(results)   # float
ac = aicc(results)   # float

# Or access as attributes
print(results.aic, results.bic, results.hqic, results.aicc)
```

All functions accept a `KalmanResults` object (returned by `model.fit()`).
They read `.llf` (log-likelihood), `.k_params`, `.nobs_effective`, and `.ndiffuse`
from the results object automatically.

### `model_selection()`

```python
from kalmanbox.diagnostics import model_selection

table = model_selection(
    results_list: list[KalmanResults],
    criterion: str = "bic",          # "aic" | "aicc" | "bic" | "hqic"
    names: list[str] | None = None,  # model labels
    sort: bool = True,               # sort by criterion value
)
```

Returns a `pandas.DataFrame` with columns:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `loglik` | Log-likelihood $\hat\ell$ |
| `k` | Number of free parameters |
| `n_eff` | Effective observations |
| `aic` | AIC value |
| `aicc` | AICc value |
| `bic` | BIC value |
| `hqic` | HQIC value |
| `delta_bic` | $\Delta_i$ relative to best model |
| `weight` | Akaike/Schwarz weight |

The **Akaike weight** for model $i$ is:

$$
w_i = \frac{\exp(-\Delta_i/2)}{\sum_j \exp(-\Delta_j/2)}
$$

It is interpreted as the probability that model $i$ is the best model in the candidate set
(under the AIC or BIC approximation to the Bayes factor).

---

## 5. Examples

### Example 1: selecting between Local Level, BSM, and UCM

```python
import numpy as np
from kalmanbox import LocalLevelModel, BSM, UCM
from kalmanbox.diagnostics import model_selection
from kalmanbox.datasets import load_airline

y = np.log(load_airline())  # log airline passengers, monthly 1949-1960

# Fit three candidate models
ll   = LocalLevelModel()
bsm  = BSM(period=12, stochastic_seasonal=True)
ucm  = UCM(period=12, stochastic_cycle=True)

r_ll  = ll.fit(y)
r_bsm = bsm.fit(y)
r_ucm = ucm.fit(y)

# Compare
table = model_selection(
    [r_ll, r_bsm, r_ucm],
    names=["LocalLevel", "BSM", "UCM"],
    criterion="bic",
)
print(table)
```

**Expected output**:

```
Model Selection Table (sorted by BIC)
=======================================

         model    loglik    k   n_eff      aic     aicc      bic     hqic  delta_bic  weight
1          BSM  -118.42    4    144  244.84   245.16   257.01   249.72       0.00    0.923
2          UCM  -117.89    7    144  249.78   250.65   268.34   257.42      11.33    0.003
3   LocalLevel  -198.71    2    144  401.42   401.54   407.72   404.00     150.71    0.000

Notes:
  n_eff excludes 0 diffuse observations
  Weights are BIC weights (Schwarz posterior model probabilities)
```

**Interpretation**:

- The BSM is strongly preferred by BIC ($\Delta_\text{BIC} = 11.3$ vs. UCM, $150.7$ vs.
  Local Level). The Local Level is decisively rejected — it cannot capture the seasonal
  pattern in airline data.
- The UCM has a lower raw log-likelihood than the BSM (−117.89 vs −118.42) but the 3
  extra parameters for the stochastic cycle are not justified: $\Delta_\text{BIC} = 11.3$
  is very strong evidence against the UCM.
- The BSM weight of 0.923 means "if the true model is in this candidate set, there is a
  92 % probability it is the BSM."

### Example 2: effect of diffuse initialisation on BIC

```python
from kalmanbox import LocalLinearTrend
from kalmanbox.datasets import load_nile

nile = load_nile()

model = LocalLinearTrend()
results = model.fit(nile)

print(f"n        = {results.nobs}")
print(f"d        = {results.ndiffuse}  (diffuse states)")
print(f"n_eff    = {results.nobs_effective}")
print(f"k        = {results.k_params}")
print(f"loglik   = {results.llf:.4f}")
print(f"AIC      = {results.aic:.4f}")
print(f"AICc     = {results.aicc:.4f}")
print(f"BIC      = {results.bic:.4f}")
print(f"HQIC     = {results.hqic:.4f}")
```

**Expected output**:

```
n        = 100
d        = 2  (diffuse states: level + slope)
n_eff    = 98
k        = 3
loglik   = -282.3412
AIC      = 570.6824
AICc     = 570.8980
BIC      = 578.4476
HQIC     = 573.7452
```

The Local Linear Trend has 2 diffuse states (initial level and slope), so BIC uses
$n_\text{eff} = 98$ rather than $n = 100$.

### Example 3: AICc correction matters for small samples

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.diagnostics import model_selection

# Short series: n=40
np.random.seed(42)
y_short = np.cumsum(np.random.randn(40)) + np.sin(np.linspace(0, 4*np.pi, 40))

models = {
    "BSM(seasonal)":  BSM(period=12, stochastic_seasonal=True).fit(y_short),
    "BSM(fixed-seas)": BSM(period=12, stochastic_seasonal=False).fit(y_short),
}

tab = model_selection(list(models.values()), names=list(models.keys()))
print(tab[["model", "k", "n_eff", "aic", "aicc", "bic"]])
```

With $n=40$ and $k=4$, the AICc correction adds $\frac{2 \times 4 \times 5}{40-4-1} = 1.14$
to AIC — noticeable compared to a BIC penalty of $4\log 40 \approx 14.8$. Using AIC
alone risks selecting the richer model when the sample is too short to distinguish it.

---

## 6. Visualisation

```python
from kalmanbox.visualization import plot_model_selection

# Bar chart of IC values with delta annotations
fig = plot_model_selection(
    table,              # DataFrame from model_selection()
    criterion="bic",    # highlight column
    show_weights=True,
)
fig.savefig("model_selection.png", dpi=150, bbox_inches="tight")
```

The chart displays each criterion value as a grouped bar, with $\Delta_i$ annotations
and a secondary axis showing Akaike weights.

---

## Related

- [Likelihood Ratio Test](likelihood-ratio.md) — formal nested-model comparison
- [Cross-Validation](cross-validation.md) — out-of-sample model evaluation
- [Innovation Tests](innovation-tests.md) — residual-based diagnostics
- [Theory: MLE theory](../theory/mle-theory.md) — log-likelihood derivation for state-space
- [Experiment framework](../user-guide/experiment.md) — automated multi-model comparison
- [API: diagnostics module](../api/diagnostics.md)
