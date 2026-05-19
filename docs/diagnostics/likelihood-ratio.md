# Likelihood Ratio Test

The **likelihood ratio test** (LRT) is the standard tool for comparing two **nested**
state-space models: one with constraints imposed (the restricted model $\mathcal{M}_0$)
and one without (the unrestricted model $\mathcal{M}_1$). It provides an asymptotically
optimal test for whether the additional parameters in $\mathcal{M}_1$ significantly
improve the fit.

---

## 1. The likelihood ratio statistic

Let $\hat\ell_0$ and $\hat\ell_1$ be the maximised log-likelihoods of the restricted and
unrestricted models respectively. Because $\mathcal{M}_0 \subset \mathcal{M}_1$ we always
have $\hat\ell_1 \geq \hat\ell_0$. The test statistic is:

$$
\boxed{LR = -2\left(\hat\ell_0 - \hat\ell_1\right) = 2\left(\hat\ell_1 - \hat\ell_0\right) \geq 0}
$$

**Wilks' theorem** (1938): under the null hypothesis $H_0: \theta \in \Theta_0$, and
under standard regularity conditions,

$$
LR \xrightarrow{d} \chi^2_q \quad \text{as } n \to \infty
$$

where $q = \dim(\Theta_1) - \dim(\Theta_0)$ is the number of restrictions (equivalently,
the difference in the number of free parameters between the two models).

The $p$-value is therefore:

$$
p = P(\chi^2_q \geq LR) = 1 - F_{\chi^2_q}(LR)
$$

where $F_{\chi^2_q}$ is the $\chi^2$ CDF with $q$ degrees of freedom. A small $p$-value
($p < 0.05$) leads to rejection of $H_0$, i.e., the additional parameters are statistically
justified.

---

## 2. When the LRT applies: nested models

Two models are **nested** if one can be obtained from the other by fixing some parameters
to specific values. Common nesting structures in kalmanbox:

| $\mathcal{M}_0$ (restricted) | $\mathcal{M}_1$ (unrestricted) | Restriction | $q$ |
|-------------------------------|-------------------------------|-------------|-----|
| Local Level | Local Linear Trend | $\sigma_\zeta^2 = 0$ | 1 |
| BSM (fixed seasonal) | BSM (stochastic seasonal) | $\sigma_\omega^2 = 0$ | 1 |
| BSM without cycle | UCM (BSM + cycle) | $\sigma_\kappa^2 = 0$, $\rho_c = 0$, $\lambda_c = 0$ | 3 |
| TVP (fixed coefficients) | TVP (time-varying) | $\Sigma_Q = 0$ | $\dim(\Sigma_Q)$ |
| DFM ($r=1$) | DFM ($r=2$) | second factor loadings $= 0$ | $p$ |

!!! warning "Non-nested models"
    AIC/BIC are preferred for **non-nested** models (e.g., ARIMA vs. BSM). The LRT does
    not apply when $\mathcal{M}_0 \not\subset \mathcal{M}_1$ because the $\chi^2$
    null distribution is not valid. See §5 for alternatives.

---

## 3. Degrees of freedom

The degrees of freedom $q$ equal the number of **additional free parameters** in
$\mathcal{M}_1$ relative to $\mathcal{M}_0$. In state-space models this is typically
straightforward, but there are two subtleties:

### 3.1 Equality vs. inequality restrictions

The standard Wilks theorem assumes the constraint is an **equality** (e.g.,
$\sigma^2 = 0$). When the parameter is constrained to be **non-negative** (as all
variance components are), the restriction $\sigma^2 = 0$ is a **boundary constraint**
and the standard $\chi^2_q$ approximation is liberal (anti-conservative).

The correct null distribution in this case is a **mixture of chi-squared distributions**.
For a single boundary parameter (e.g., $H_0: \sigma^2 = 0$):

$$
LR \xrightarrow{d} \frac{1}{2}\chi^2_0 + \frac{1}{2}\chi^2_1 \quad \text{(under boundary null)}
$$

where $\chi^2_0$ is a point mass at 0. The corrected $p$-value becomes:

$$
p_\text{boundary} = \frac{1}{2}P(\chi^2_1 \geq LR) = \frac{1}{2}(1 - F_{\chi^2_1}(LR))
$$

This is precisely **half** the $p$-value from the standard $\chi^2_1$ test. For $q \geq 2$
boundary constraints, the mixture weights are binomial:

$$
LR \xrightarrow{d} \sum_{j=0}^{q} \binom{q}{j} 2^{-q} \chi^2_j
$$

kalmanbox applies the boundary correction automatically when the restricted parameter is
a variance component fixed at zero.

!!! info "Boundary problem references"
    Self & Liang (1987) derive the general theory. Stoel, Garre & Dolan (2006) give
    practical guidance for SEM/state-space applications. The correction is sometimes
    called the *chi-bar-squared* distribution.

### 3.2 Degrees of freedom table

```python
from kalmanbox.diagnostics import likelihood_ratio_test

lrt = likelihood_ratio_test(results_restricted, results_unrestricted)
print(f"LR statistic : {lrt.statistic:.4f}")
print(f"df           : {lrt.df}")
print(f"p-value      : {lrt.pvalue:.4f}")
print(f"boundary     : {lrt.boundary_corrected}")
print(f"p (boundary) : {lrt.pvalue_boundary:.4f}")
```

---

## 4. Step-by-step procedure

```
1. Specify M_0 (restricted) and M_1 (unrestricted).
2. Fit both models to the SAME data (same n, same transformations).
3. Compute LR = 2 * (ll_1 - ll_0).
4. Determine q = k_1 - k_0.
5. Check for boundary constraints on the restricted parameters.
6. Compute p-value:
   - Standard: p = 1 - chi2_cdf(LR, df=q)
   - Boundary: p = 0.5 * (1 - chi2_cdf(LR, df=q))  [for q=1 boundary]
7. Reject H_0 at level alpha if p < alpha.
```

!!! danger "Same data requirement"
    Both models **must** be estimated on the same observations with the same
    log-likelihood definition. Comparing diffuse vs. non-diffuse likelihoods, or
    models with different numbers of missing values, invalidates the test.

---

## 5. Limitations and alternatives

### 5.1 Boundary problems

As described in §3.1, variance components restricted to zero create boundary problems.
The boundary-corrected $p$-value (half the standard $\chi^2$ value for $q=1$) is
implemented in kalmanbox and reported alongside the standard $p$-value.

In practice, the boundary correction typically matters most at the 5 % level: a test
that gives $p = 0.04$ under the standard distribution gives $p = 0.02$ under the
boundary correction (i.e., it remains significant). However, near the 5 % boundary
the correction can flip the conclusion.

### 5.2 Non-nested models

For non-nested models use:

- [Information criteria](information-criteria.md) (AIC, BIC) — always applicable.
- [Cross-validation](cross-validation.md) — preferred for forecasting tasks.
- **Vuong test** (1989) — formally compares non-nested models by testing whether
  the mean difference in individual log-likelihoods is zero.
- **Clarke test** (2007) — a non-parametric version of the Vuong test, more robust
  in small samples.

### 5.3 Near-boundary maximum likelihood estimates

When $\hat\theta_0 \approx \hat\theta_1$ (the unrestricted MLE is near the boundary),
the LRT has low power. In this regime, score tests or Wald tests may be more appropriate
because they are computed at the null parameter value.

### 5.4 Misspecification

If neither model is correctly specified, the LRT still tests whether $\mathcal{M}_0$ is
closer to the data than $\mathcal{M}_1$, but the asymptotic $\chi^2$ approximation may
not hold. Always run [innovation tests](innovation-tests.md) on the winning model.

---

## 6. API reference

### `likelihood_ratio_test()`

```python
from kalmanbox.diagnostics import likelihood_ratio_test

lrt = likelihood_ratio_test(
    restricted: KalmanResults,        # M_0: restricted model
    unrestricted: KalmanResults,      # M_1: unrestricted model
    df: int | None = None,            # override df (auto-computed if None)
    boundary: bool = True,            # apply boundary correction for variance params
    alpha: float = 0.05,              # significance level for verdict
)
```

Returns a `LRTestResult` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.statistic` | `float` | LR = $2(\hat\ell_1 - \hat\ell_0)$ |
| `.df` | `int` | Degrees of freedom $q$ |
| `.pvalue` | `float` | Standard $\chi^2_q$ p-value |
| `.boundary_corrected` | `bool` | Whether boundary correction was applied |
| `.pvalue_boundary` | `float` | Boundary-corrected p-value (or same as `.pvalue`) |
| `.reject` | `bool` | True if `pvalue_boundary < alpha` |
| `.llf_restricted` | `float` | $\hat\ell_0$ |
| `.llf_unrestricted` | `float` | $\hat\ell_1$ |
| `.k_restricted` | `int` | Parameters in $\mathcal{M}_0$ |
| `.k_unrestricted` | `int` | Parameters in $\mathcal{M}_1$ |

```python
print(lrt.summary())
```

```
Likelihood Ratio Test
=====================
  Restricted model   : LocalLevelModel     (k=2,  loglik=-282.341)
  Unrestricted model : LocalLinearTrend    (k=3,  loglik=-278.014)

  LR statistic : 8.654
  Df           : 1
  p-value      : 0.003  (χ²₁)
  p-value      : 0.002  (boundary-corrected, ½×χ²₁)

  Decision (α=0.05): REJECT H₀ — slope component is significant
```

### `lr_test()` (alias)

`lr_test()` is a short alias for `likelihood_ratio_test()` with identical signature.

```python
from kalmanbox.diagnostics import lr_test

lrt = lr_test(r0, r1)
```

---

## 7. Examples

### Example 1: is a slope component significant?

Test whether adding a stochastic slope (Local Linear Trend) significantly improves
over a Local Level model on the Nile river data.

```python
from kalmanbox import LocalLevelModel, LocalLinearTrend
from kalmanbox.diagnostics import likelihood_ratio_test
from kalmanbox.datasets import load_nile

nile = load_nile()

# H_0: Local Level (sigma_zeta^2 = 0, i.e., no slope)
r0 = LocalLevelModel().fit(nile)

# H_1: Local Linear Trend (slope allowed to vary)
r1 = LocalLinearTrend().fit(nile)

lrt = likelihood_ratio_test(r0, r1)
print(lrt.summary())
```

**Expected output**:

```
Likelihood Ratio Test
=====================
  Restricted model   : LocalLevelModel       (k=2,  loglik=-298.412)
  Unrestricted model : LocalLinearTrend      (k=3,  loglik=-297.893)

  LR statistic : 1.038
  Df           : 1
  p-value      : 0.308  (χ²₁)
  p-value      : 0.154  (boundary-corrected, ½×χ²₁)

  Decision (α=0.05): FAIL TO REJECT H₀ — slope component is NOT significant
```

**Interpretation**: The slope variance $\sigma_\zeta^2$ is not significantly different
from zero for the Nile data. The Local Level is the preferred parsimonious specification.
This is consistent with the BIC evidence: $\Delta_\text{BIC}(LLT) > 2$.

---

### Example 2: is seasonality significant in a BSM?

Test whether adding a stochastic seasonal component significantly improves the fit for
monthly airline passenger data.

```python
import numpy as np
from kalmanbox import BSM
from kalmanbox.diagnostics import likelihood_ratio_test
from kalmanbox.datasets import load_airline

y = np.log(load_airline())

# H_0: BSM with fixed (deterministic) seasonal
r0 = BSM(period=12, stochastic_seasonal=False).fit(y)

# H_1: BSM with stochastic seasonal
r1 = BSM(period=12, stochastic_seasonal=True).fit(y)

lrt = likelihood_ratio_test(r0, r1)
print(lrt.summary())
```

**Expected output**:

```
Likelihood Ratio Test
=====================
  Restricted model   : BSM(fixed-seasonal)    (k=3,  loglik=-128.774)
  Unrestricted model : BSM(stochastic-seasonal)(k=4,  loglik=-118.421)

  LR statistic : 20.706
  Df           : 1
  p-value      : <0.0001  (χ²₁)
  p-value      : <0.0001  (boundary-corrected, ½×χ²₁)

  Decision (α=0.05): REJECT H₀ — stochastic seasonal is highly significant
```

**Interpretation**: $LR = 20.7$ with 1 degree of freedom gives $p < 0.0001$ even after
the boundary correction. The seasonal pattern in airline data is **not deterministic** —
its amplitude evolves over time. The stochastic BSM is preferred.

---

### Example 3: multi-parameter restriction (UCM cycle)

Test whether adding a stochastic cycle ($\rho_c, \lambda_c, \sigma_\kappa^2$) to a BSM
is justified. This involves $q = 3$ boundary parameters.

```python
from kalmanbox import BSM, UCM
from kalmanbox.diagnostics import likelihood_ratio_test
from kalmanbox.datasets import load_industrial_production

ip = load_industrial_production()

# H_0: BSM (no cycle)
r0 = BSM(period=12, stochastic_seasonal=True).fit(ip)

# H_1: UCM (BSM + stochastic cycle)
r1 = UCM(period=12, stochastic_cycle=True, stochastic_seasonal=True).fit(ip)

lrt = likelihood_ratio_test(r0, r1, df=3, boundary=True)
print(lrt.summary())
print(f"\nNote: for q=3 boundary params, use chi-bar-squared mixture distribution.")
```

**Interpretation guidance**: For $q = 3$ boundary restrictions the chi-bar-squared
mixture distribution is used. If LR falls below $\chi^2_1(0.05) \approx 3.84$, the
cycle is clearly not significant. If LR $> \chi^2_3(0.05) \approx 7.81$, it is clearly
significant. Between these values, the exact correction matters.

---

### Example 4: DFM factor count selection

```python
from kalmanbox import DFM
from kalmanbox.diagnostics import likelihood_ratio_test
from kalmanbox.datasets import load_macro_panel

Y = load_macro_panel()  # (n_obs, n_series) macro panel

# Sequential factor testing
prev_results = DFM(n_factors=1).fit(Y)
for r in range(2, 6):
    curr_results = DFM(n_factors=r).fit(Y)
    lrt = likelihood_ratio_test(prev_results, curr_results)
    status = "REJECT — add factor" if lrt.reject else "FAIL TO REJECT — stop here"
    print(f"r={r-1} vs r={r}: LR={lrt.statistic:.2f}, p={lrt.pvalue_boundary:.4f} → {status}")
    if not lrt.reject:
        break
    prev_results = curr_results
```

---

## 8. Relationship to other criteria

| Method | Applicable to | Penalises | Asymptotically optimal |
|--------|---------------|-----------|------------------------|
| LRT | Nested models | Log-likelihood difference | Neyman-Pearson (fixed $\alpha$) |
| AIC | Any | Prediction MSE | Yes (efficiency) |
| BIC | Any | Model posterior prob. | Yes (consistency) |
| Cross-CV | Any | Out-of-sample loss | Task-dependent |

For nested models where both LRT and IC are applicable:

- **LRT is preferred** when you have a specific statistical hypothesis to test
  (e.g., "is this parameter significantly non-zero?").
- **BIC is preferred** when you want a single ranking of many models without
  specifying pairwise hypotheses.
- **Both should agree** for clear cases; disagreement signals borderline evidence
  and warrants deeper investigation.

---

## Related

- [Information Criteria](information-criteria.md) — AIC, BIC, HQIC for model ranking
- [Cross-Validation](cross-validation.md) — out-of-sample model comparison
- [Innovation Tests](innovation-tests.md) — verify the winning model passes diagnostics
- [Theory: MLE theory](../theory/mle-theory.md) — log-likelihood for state-space models
- [BSM](../user-guide/structural/bsm.md) — seasonal and trend components
- [UCM](../user-guide/structural/ucm.md) — unobserved components model
- [API: diagnostics module](../api/diagnostics.md)
