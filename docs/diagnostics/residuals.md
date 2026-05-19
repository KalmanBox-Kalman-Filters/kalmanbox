# Residual analysis

The residuals from a state-space model are the **standardised
innovations**

$$
\tilde v_t = F_t^{-1/2} v_t.
$$

Under correct specification, $\{\tilde v_t\}$ should be i.i.d.
$\mathcal{N}(0, I_p)$.

## Tests shipped in `kalmanbox.diagnostics`

### Ljung–Box

Null hypothesis: no autocorrelation up to lag $h$.

$$
Q(h) = n(n+2)\sum_{k=1}^{h} \frac{\hat\rho_k^2}{n-k}\sim \chi^2_h
$$

Use $h \approx \log n$ for short series, $h \approx 2\log n$ for long.

```python
from kalmanbox.diagnostics import ljung_box

stat, pvalue = ljung_box(residuals, lags=20)
```

### Jarque–Bera (normality)

$$
JB = \tfrac{n}{6}\!\left(S^2 + \tfrac{1}{4}(K-3)^2\right) \sim \chi^2_2
$$

Sensitive to outliers — failures often indicate fat tails, not bad
model fit per se.

### ARCH-LM (heteroscedasticity)

Tests whether squared residuals exhibit autocorrelation, i.e. ARCH
effects. If positive, consider a **stochastic volatility** extension
or reparameterise $H_t$ to be time-varying.

### CUSUM and CUSUMSQ

Recursive cumulative sums of (standardised / squared) residuals. Lines
crossing the asymptotic bounds indicate a structural break.

## Visualisation

```python
from kalmanbox.visualization import plot_residual_diagnostics

plot_residual_diagnostics(results)
# 4 panels: standardised residuals, ACF, QQ-plot, histogram
```

## Multivariate case

For multi-output models, run the tests **per series** and inspect the
residual cross-correlation matrix. Strong off-diagonal entries suggest
unmodelled common factors.

## Related

- [Diagnostics: stability](stability.md)
- [Visualization: diagnostic plots](../visualization/diagnostics.md)
- [API: diagnostics](../api/diagnostics.md)
