# Bayesian Estimation API

`kalmanbox.bayesian`

This page documents the classes and functions that implement full Bayesian
inference for linear Gaussian state-space models. The workflow combines
**Forward Filtering Backward Sampling** (FFBS) to draw exact samples from
the smoothing distribution with **conjugate Gibbs steps** to update variance
and coefficient parameters, producing a complete posterior over both hidden
states and hyperparameters.

| Class / Function | Role |
|---|---|
| [`GibbsSampler`](#gibbssampler) | Main MCMC engine — alternates FFBS and conjugate parameter draws |
| [`FFBS`](#ffbs) | Draw exact state-sequence samples from the smoothing distribution |
| [`InverseGamma`](#inversegamma) | IG(α, β) conjugate prior for scalar variances |
| [`NormalPrior`](#normalprior) | N(μ, σ²) prior for regression coefficients and intercepts |
| [`HalfCauchy`](#halfcauchy) | Half-Cauchy(β) weakly-informative prior for standard deviations |
| [`InverseWishart`](#inversewishart) | IW(ν, Ψ) conjugate prior for covariance matrices |
| [Convergence diagnostics](#convergence-diagnostics) | `rhat`, `ess`, `geweke`, `trace_plot`, `posterior_predictive` |

See [User Guide: Bayesian Estimation](../user-guide/bayesian/index.md) for
conceptual background and worked examples.

---

## GibbsSampler

`kalmanbox.bayesian.GibbsSampler`

Implements a block Gibbs sampler for Bayesian inference in linear Gaussian
state-space models. At each iteration the sampler alternates between two
blocks:

1. **State block** — draw the full state trajectory
   $\alpha_{1:T}^{(s)} \sim p(\alpha_{1:T} \mid y_{1:T}, \theta^{(s-1)})$
   using the FFBS algorithm.
2. **Parameter block** — draw each variance parameter from its conjugate
   conditional posterior given the sampled states:

$$
\sigma^2 \mid \alpha^{(s)},\, y_{1:T}
\;\sim\; \mathcal{IG}\!\left(\alpha_0 + \tfrac{n}{2},\;
\beta_0 + \tfrac{1}{2}\textstyle\sum_t e_t^2\right)
$$

where $e_t$ are the residuals implied by $\alpha^{(s)}$.  After discarding
`burn_in` draws and keeping every `thinning`-th sample, the retained draws
form a Monte Carlo approximation to the joint posterior
$p(\theta, \alpha_{1:T} \mid y_{1:T})$.

!!! info "Supported prior types"

    The conjugate Gibbs update is available for `InverseGamma` (scalar
    variances), `NormalPrior` (regression coefficients), and
    `InverseWishart` (covariance matrices). Parameters with a `HalfCauchy`
    prior are updated via a Metropolis-within-Gibbs step using a
    log-normal proposal.

!!! warning "Identifiability"

    Bayesian estimation does not automatically resolve identifiability
    problems. Ensure the model is identified before placing priors — for
    example, fix the sign of a factor loading in DFM models or constrain
    the variance ratio in a local level model. Unidentified chains
    produce R-hat values well above 1.05.

### Constructor

```python
GibbsSampler(
    ss: StateSpaceRepresentation,
    priors: dict[str, Prior],
    n_iter: int = 2000,
    burn_in: int = 500,
    thinning: int = 1,
    n_chains: int = 1,
    seed: int | None = None,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ss` | `StateSpaceRepresentation` | required | The state-space model defining matrices `(Z, T, R, H, Q)`. Parameters named in `priors` must correspond to scalar or matrix entries of this representation. |
| `priors` | `dict[str, Prior]` | required | Mapping from parameter names (e.g. `"sigma2_obs"`, `"sigma2_eta"`) to `Prior` objects. Names must match the parameterisation of `ss`. |
| `n_iter` | `int` | `2000` | Total number of MCMC iterations per chain, including burn-in. |
| `burn_in` | `int` | `500` | Number of initial iterations discarded as warm-up. Must be strictly less than `n_iter`. |
| `thinning` | `int` | `1` | Retain every `k`-th post-burn-in sample. `thinning=1` keeps all draws; `thinning=5` retains 20 %. |
| `n_chains` | `int` | `1` | Number of independent chains to run. Use `n_chains >= 2` to compute R-hat convergence diagnostics. |
| `seed` | `int \| None` | `None` | Integer seed passed to `numpy.random.default_rng`. Each chain uses `seed + chain_index` for reproducibility across chains. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision for all internal computations. |

### Properties

| Property | Type | Description |
|---|---|---|
| `samples` | `dict[str, np.ndarray] \| None` | Posterior draws, keyed by parameter name. Shape `(n_draws, ...)` per parameter. `None` before `run()` is called. |
| `posterior` | `dict[str, PosteriorSummary] \| None` | Per-parameter summary statistics (mean, std, HDI 94 %). `None` before `run()`. |
| `n_draws` | `int` | Effective number of stored draws per chain: `(n_iter - burn_in) // thinning`. |
| `loglikelihood_trace` | `np.ndarray \| None` | Per-iteration log-likelihood value. Shape `(n_iter,)` for a single chain; `(n_chains, n_iter)` for multiple chains. `None` before `run()`. |

### Methods

#### `run(y)`

```python
def run(
    y: np.ndarray,
) -> GibbsResult
```

Execute the Gibbs sampler on observation sequence `y`. Each chain is
initialised at the MLE estimate (warm start) or, if MLE fails to converge,
at the prior means. Chains are run sequentially unless the process pool
backend is configured via `kalmanbox.set_backend("multiprocessing")`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y` | `np.ndarray` | Observation matrix. Shape `(n, p)` where `n` is the number of time steps and `p` the number of observed variables. `np.nan` entries are treated as missing and skipped in the likelihood computation. |

**Returns** `GibbsResult` — a named-tuple-like object with the attributes
below. The same object is also stored on `self` so that `trace_plot`,
`rhat`, and `ess` can be called without arguments after `run()`.

| Attribute | Type / Shape | Description |
|---|---|---|
| `samples` | `dict[str, np.ndarray]` | All retained draws. Shape `(n_draws,)` for scalar parameters; `(n_draws, m, m)` for matrix parameters. |
| `posterior` | `dict[str, PosteriorSummary]` | Summary with fields `mean`, `std`, `hdi_low`, `hdi_high` (94 % HDI by default), `median`. |
| `loglikelihood_trace` | `np.ndarray` | Log-likelihood at every iteration (including burn-in). Shape `(n_iter,)` or `(n_chains, n_iter)`. |
| `acceptance_rates` | `dict[str, float]` | Metropolis acceptance rates for non-conjugate parameters (e.g. HalfCauchy priors). Always 1.0 for conjugate draws. |
| `rhat` | `dict[str, float] \| None` | Gelman-Rubin split-chain R-hat. `None` when `n_chains == 1`. |
| `ess` | `dict[str, dict]` | Bulk and tail effective sample sizes. |

!!! tip "Warm-start initialisation"

    The sampler initialises each chain at the MLE estimate when possible.
    For strongly non-identified or high-dimensional models, supply
    `ss` with matrices pre-set to reasonable values so that the MLE
    starting point is in a plausible region of the posterior.

---

#### `trace_plot(param, figsize=(10, 6))`

```python
def trace_plot(
    param: str,
    figsize: tuple[float, float] = (10, 6),
) -> matplotlib.figure.Figure
```

Produce a two-panel diagnostic figure for the named parameter: the left
panel shows the iteration trace (all chains overlaid) and the right panel
shows a kernel density estimate of the posterior draws.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `param` | `str` | Parameter name matching a key in `self.samples`. |
| `figsize` | `tuple[float, float]` | Matplotlib figure size `(width, height)` in inches. |

**Returns** `matplotlib.figure.Figure`. The caller is responsible for
calling `fig.savefig(...)` or `plt.show()`.

**Raises** `RuntimeError` if called before `run()`.

---

#### `rhat()`

```python
def rhat() -> dict[str, float]
```

Compute the **Gelman-Rubin split-chain R-hat** statistic for every
parameter. Values close to 1.0 indicate convergence; values above 1.05
suggest the chains have not mixed sufficiently.

**Returns** `dict[str, float]` — R-hat value per parameter.

**Raises** `ValueError` if `n_chains < 2` (split-chain R-hat requires at
least two chains to be meaningful).

!!! warning "R-hat requirements"

    Split-chain R-hat is computed by splitting each chain in half and
    treating the halves as separate chains, giving `2 * n_chains`
    sub-chains. Reliable estimates require at least 400 post-burn-in
    draws per chain (i.e., `n_iter - burn_in >= 400`).

---

#### `ess()`

```python
def ess() -> dict[str, float]
```

Compute the **bulk effective sample size** (ESS) and **tail ESS** for
each parameter using the Vehtari et al. (2021) rank-normalised estimator.
ESS accounts for within-chain autocorrelation. Rule of thumb: bulk ESS
> 400 and tail ESS > 400 for reliable quantile estimates.

**Returns** `dict[str, float]` — dictionary with keys of the form
`"<param>.bulk"` and `"<param>.tail"` for each parameter.

---

#### `geweke(param, first=0.1, last=0.5)`

```python
def geweke(
    param: str,
    first: float = 0.1,
    last: float = 0.5,
) -> GewekeResult
```

Run the **Geweke (1992) spectral density convergence test** on a single
parameter. The test compares the mean of the first `first` fraction of the
chain with the mean of the last `last` fraction using a z-score derived
from the spectral density at frequency zero.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `param` | `str` | Parameter name. |
| `first` | `float` | Fraction of the chain used for the early-window mean. Default `0.1`. |
| `last` | `float` | Fraction of the chain used for the late-window mean. Default `0.5`. `first + last` must not exceed 1.0. |

**Returns** `GewekeResult` with fields:

| Field | Type | Description |
|---|---|---|
| `z_score` | `float` | Geweke z-score. Values outside (−2, 2) indicate non-stationarity at the 5 % level. |
| `p_value` | `float` | Two-tailed p-value for the null hypothesis of stationarity. |
| `early_mean` | `float` | Mean of the first-window draws. |
| `late_mean` | `float` | Mean of the last-window draws. |

---

### Example

```python
import numpy as np
from kalmanbox import StateSpaceRepresentation
from kalmanbox.bayesian import GibbsSampler, InverseGamma

# ── Simulate a local level model ──────────────────────────────────────────
rng = np.random.default_rng(42)
n = 200
sigma2_eta_true = 0.04   # level innovation variance
sigma2_eps_true = 0.25   # observation noise variance

level = np.cumsum(rng.normal(0, np.sqrt(sigma2_eta_true), n))
y = level + rng.normal(0, np.sqrt(sigma2_eps_true), n)

# ── Build state-space representation ──────────────────────────────────────
ss = StateSpaceRepresentation(
    Z=np.array([[1.0]]),
    T=np.array([[1.0]]),
    R=np.array([[1.0]]),
    H=np.array([[0.25]]),    # starting value; will be sampled
    Q=np.array([[0.04]]),    # starting value; will be sampled
)

# ── Weakly-informative InverseGamma priors ─────────────────────────────────
priors = {
    "sigma2_obs": InverseGamma(shape=2.0, scale=0.10),  # H
    "sigma2_eta": InverseGamma(shape=2.0, scale=0.01),  # Q
}

# ── Run the Gibbs sampler ─────────────────────────────────────────────────
sampler = GibbsSampler(
    ss=ss,
    priors=priors,
    n_iter=3000,
    burn_in=1000,
    thinning=2,
    n_chains=2,
    seed=0,
)

result = sampler.run(y[:, np.newaxis])

# ── Inspect posterior ─────────────────────────────────────────────────────
post_obs = result.posterior["sigma2_obs"]
post_eta = result.posterior["sigma2_eta"]
print(f"sigma2_obs  posterior mean: {post_obs.mean:.4f}  (true: {sigma2_eps_true})")
print(f"sigma2_eta  posterior mean: {post_eta.mean:.4f}  (true: {sigma2_eta_true})")

# R-hat convergence check
rhat_vals = result.rhat
for k, v in rhat_vals.items():
    print(f"  R-hat[{k}] = {v:.4f}  {'OK' if v < 1.05 else 'WARN'}")

# Trace plot
fig = sampler.trace_plot("sigma2_obs")
fig.savefig("trace_sigma2_obs.png", dpi=150, bbox_inches="tight")
```

---

## FFBS

`kalmanbox.bayesian.FFBS`

Standalone implementation of the **Forward Filtering Backward Sampling**
algorithm. FFBS draws an exact sample from the joint smoothing distribution:

$$
p(\alpha_{1:T} \mid y_{1:T}, \theta)
$$

in a linear Gaussian state-space model. The algorithm proceeds in two
passes:

1. **Forward pass** — run the standard Kalman filter, recording filtered
   means $a_{t|t}$ and covariances $P_{t|t}$ at every step.
2. **Backward sampling pass** — draw $\alpha_T \sim \mathcal{N}(a_{T|T}, P_{T|T})$
   and then, for $t = T{-}1, \ldots, 1$, sample:

$$
\alpha_t \mid \alpha_{t+1}, y_{1:t}
\;\sim\;
\mathcal{N}\!\left(a_{t|t} + J_t(\alpha_{t+1} - a_{t+1|t}),\;
(I - J_t T)\,P_{t|t}\right)
$$

where $J_t = P_{t|t}\,T'\,P_{t+1|t}^{-1}$ is the backward smoothing gain.

!!! info "FFBS vs RTSSmoother"

    [`RTSSmoother`](core.md#rtssmoother) returns the *mean* and *covariance*
    of the smoothing distribution. `FFBS` returns a *random draw* from
    the same distribution. Running FFBS many times and averaging the
    trajectories reproduces the RTS smoother means up to Monte Carlo
    error.

### Constructor

```python
FFBS(
    ss: StateSpaceRepresentation,
    seed: int | None = None,
    dtype: np.dtype = np.float64,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ss` | `StateSpaceRepresentation` | required | State-space model providing system matrices `(Z, T, R, H, Q)`. |
| `seed` | `int \| None` | `None` | Integer seed for the internal `numpy.random.Generator`. Set for reproducible state draws. |
| `dtype` | `np.dtype` | `np.float64` | Floating-point precision. |

### Methods

#### `sample_states(y, n_samples=1)`

```python
def sample_states(
    y: np.ndarray,
    n_samples: int = 1,
) -> np.ndarray
```

Draw `n_samples` independent state-trajectory samples from the joint
smoothing distribution $p(\alpha_{1:T} \mid y_{1:T}, \theta)$.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y` | `np.ndarray` | Observations. Shape `(n, p)`. `np.nan` for missing values. |
| `n_samples` | `int` | Number of independent trajectory draws. |

**Returns** `np.ndarray` of shape `(n_samples, n, m)` — the sampled state
trajectories. Each slice `[i, :, :]` is an independent draw.

!!! tip "Memory considerations"

    For large `n` and `m`, storing `n_samples` trajectories requires
    `n_samples × n × m × 8` bytes (float64). For `n=1000`, `m=10`,
    `n_samples=2000` this is approximately 160 MB. Use `sample_states`
    in batches if memory is constrained.

---

#### `filter_pass(y)`

```python
def filter_pass(
    y: np.ndarray,
) -> FilterResult
```

Execute the forward Kalman filter pass only, without performing the
backward sampling step. This is useful when you want to inspect the
filtered output before drawing state samples.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y` | `np.ndarray` | Observations. Shape `(n, p)`. |

**Returns** [`FilterResult`](core.md#filterresult) containing filtered
means `a_filt`, predicted means `a_pred`, covariances `P_filt` and
`P_pred`, innovations `v`, and `loglikelihood`.

---

#### `backward_sample(filter_result)`

```python
def backward_sample(
    filter_result: FilterResult,
) -> np.ndarray
```

Perform a single backward sampling pass given pre-computed filter output.
Useful when the forward pass is expensive and you want to draw multiple
state samples from the same filtering result.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `filter_result` | `FilterResult` | Output from a previous `filter_pass()` call. |

**Returns** `np.ndarray` of shape `(n, m)` — a single sampled state
trajectory.

---

### Example

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import StateSpaceRepresentation
from kalmanbox.bayesian import FFBS

# ── Local linear trend model ──────────────────────────────────────────────
rng = np.random.default_rng(7)
n = 150

# Simulate data
level = np.zeros(n + 1)
slope = np.zeros(n + 1)
slope[0] = 0.02
for t in range(n):
    slope[t+1] = slope[t] + rng.normal(0, 0.002)
    level[t+1] = level[t] + slope[t] + rng.normal(0, 0.05)
y = level[1:] + rng.normal(0, 0.3, n)

ss = StateSpaceRepresentation(
    Z=np.array([[1.0, 0.0]]),
    T=np.array([[1.0, 1.0], [0.0, 1.0]]),
    R=np.eye(2),
    H=np.array([[0.09]]),
    Q=np.diag([0.0025, 4e-6]),
)

ffbs = FFBS(ss, seed=42)

# ── Draw 200 state trajectories ───────────────────────────────────────────
draws = ffbs.sample_states(y[:, np.newaxis], n_samples=200)
# draws.shape == (200, 150, 2)

level_draws = draws[:, :, 0]   # all draws of the level component
level_mean  = level_draws.mean(axis=0)
level_lo    = np.percentile(level_draws, 3, axis=0)
level_hi    = np.percentile(level_draws, 97, axis=0)

# ── Plot posterior credible band ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(range(n), level_lo, level_hi, alpha=0.25, label="94 % credible band")
ax.plot(level_mean, color="C0", label="Posterior mean")
ax.plot(y, "k.", alpha=0.3, markersize=3, label="Observations")
ax.set_xlabel("Time"); ax.set_ylabel("Level"); ax.legend()
fig.tight_layout()
fig.savefig("ffbs_credible_band.png", dpi=150)
```

---

## Prior Classes

`kalmanbox.bayesian.priors`

All prior classes share a common interface: `sample(size)`, `log_pdf(x)`,
and `summary()`. They integrate directly with `GibbsSampler` via the
`priors` dictionary. Conjugate priors support closed-form Gibbs updates;
non-conjugate priors fall back to Metropolis-within-Gibbs.

---

### InverseGamma

`kalmanbox.bayesian.priors.InverseGamma`

The Inverse-Gamma distribution $\mathcal{IG}(\alpha, \beta)$ is the
canonical **conjugate prior** for scalar variance parameters
$\sigma^2 > 0$. Its density is:

$$
p(\sigma^2) = \frac{\beta^\alpha}{\Gamma(\alpha)}\,
(\sigma^2)^{-(\alpha+1)}\,
\exp\!\left(-\frac{\beta}{\sigma^2}\right)
$$

The conjugate Gibbs update, given $n$ squared residuals
$\{e_t^2\}$, yields:

$$
\sigma^2 \mid \text{states},\, y
\;\sim\; \mathcal{IG}\!\left(\alpha + \tfrac{n}{2},\;
\beta + \tfrac{1}{2}\sum_t e_t^2\right)
$$

!!! tip "Choosing hyperparameters"

    Set $\alpha = 2$ and $\beta = \text{(fraction of sample variance)}$
    for a weakly-informative prior. For example, if the data have unit
    variance and you expect the observation noise to account for roughly
    10 %, use `InverseGamma(shape=2.0, scale=0.1)`.

#### Constructor

```python
InverseGamma(
    shape: float,
    scale: float,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shape` | `float` | required | Shape parameter $\alpha > 0$. |
| `scale` | `float` | required | Scale parameter $\beta > 0$. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `mean` | `float` | Prior mean $\beta / (\alpha - 1)$. Defined only for $\alpha > 1$; raises `ValueError` otherwise. |
| `mode` | `float` | Prior mode $\beta / (\alpha + 1)$. Defined for all $\alpha > 0$. |
| `variance` | `float` | Prior variance $\beta^2 / [(\alpha-1)^2(\alpha-2)]$. Defined only for $\alpha > 2$. |

#### Methods

##### `sample(size=None)`

```python
def sample(size: int | tuple | None = None) -> np.ndarray | float
```

Draw i.i.d. samples from $\mathcal{IG}(\alpha, \beta)$.

**Returns** scalar `float` if `size` is `None`; otherwise `np.ndarray`
of the requested shape.

---

##### `log_pdf(x)`

```python
def log_pdf(x: float | np.ndarray) -> float | np.ndarray
```

Evaluate the log-density at `x`. Returns `−∞` for `x ≤ 0`.

---

##### `summary()`

```python
def summary() -> dict
```

Return a dictionary with keys `"distribution"`, `"shape"`, `"scale"`,
`"mean"`, `"mode"`, `"variance"`, and `"std"`.

---

#### Example

```python
from kalmanbox.bayesian.priors import InverseGamma
import numpy as np

# Weakly-informative prior: mean = 0.1, broad
prior = InverseGamma(shape=2.0, scale=0.1)
print(prior.summary())
# {'distribution': 'InverseGamma', 'shape': 2.0, 'scale': 0.1,
#  'mean': 0.1, 'mode': 0.033, 'variance': inf, 'std': inf}

draws = prior.sample(size=5000)
print(f"Sample mean: {draws.mean():.4f}  (prior mean: {prior.mean:.4f})")
```

---

### NormalPrior

`kalmanbox.bayesian.priors.NormalPrior`

The Normal distribution $\mathcal{N}(\mu_0, \sigma_0^2)$ is the conjugate
prior for regression coefficients, intercepts, and other real-valued
parameters. Given the likelihood contribution $y \sim \mathcal{N}(X\beta, \sigma^2 I)$
with $n$ observations, the posterior is:

$$
\beta \mid y, \sigma^2
\;\sim\;
\mathcal{N}\!\left(
  \bigl(\sigma_0^{-2} + \sigma^{-2} X'X\bigr)^{-1}
  \bigl(\sigma_0^{-2}\mu_0 + \sigma^{-2} X'y\bigr),\;
  \bigl(\sigma_0^{-2} + \sigma^{-2} X'X\bigr)^{-1}
\right)
$$

#### Constructor

```python
NormalPrior(
    mean: float,
    variance: float,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mean` | `float` | required | Prior mean $\mu_0$. |
| `variance` | `float` | required | Prior variance $\sigma_0^2$. Must be strictly positive. |

#### Methods

##### `sample(size=None)`

```python
def sample(size: int | tuple | None = None) -> np.ndarray | float
```

Draw i.i.d. samples from $\mathcal{N}(\mu_0, \sigma_0^2)$.

---

##### `log_pdf(x)`

```python
def log_pdf(x: float | np.ndarray) -> float | np.ndarray
```

Evaluate the Gaussian log-density at `x`.

---

##### `summary()`

```python
def summary() -> dict
```

Return a dictionary with keys `"distribution"`, `"mean"`, `"variance"`,
`"std"`, `"hdi_low"` (2.5 %), `"hdi_high"` (97.5 %).

---

#### Example

```python
from kalmanbox.bayesian.priors import NormalPrior

# Prior for a TVP regression coefficient: centred at 0, moderately diffuse
prior_beta = NormalPrior(mean=0.0, variance=1.0)
print(f"Prior std: {prior_beta.summary()['std']:.4f}")
```

---

### HalfCauchy

`kalmanbox.bayesian.priors.HalfCauchy`

The Half-Cauchy distribution $\text{HalfCauchy}(\beta)$ is a
**weakly-informative prior for standard deviations** recommended by Gelman
(2006) for hierarchical variance components. Its density on $\sigma > 0$ is:

$$
p(\sigma) = \frac{2}{\pi \beta}
\left(1 + \left(\frac{\sigma}{\beta}\right)^2\right)^{-1}
$$

The heavy tails allow the data to pull the posterior toward large values
when the evidence supports it, while still providing mild regularisation
away from zero. The mean and variance are infinite.

!!! info "Non-conjugate prior"

    `HalfCauchy` priors are updated via a log-normal Metropolis-within-Gibbs
    step. The acceptance rate is reported in `GibbsResult.acceptance_rates`.
    Acceptance rates between 0.20 and 0.60 are typical; values outside
    this range suggest tuning the step size (see `kalmanbox.bayesian.set_mwg_step`).

#### Constructor

```python
HalfCauchy(
    scale: float,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scale` | `float` | required | Scale parameter $\beta > 0$. Gelman (2006) recommends $\beta = 2.5$ for standardised predictors. Use a smaller value (e.g. 0.5) when the standard deviation is expected to be small. |

#### Methods

##### `sample(size=None)`

```python
def sample(size: int | tuple | None = None) -> np.ndarray | float
```

Draw i.i.d. samples from $\text{HalfCauchy}(\beta)$ using the inverse-CDF
method.

---

##### `log_pdf(x)`

```python
def log_pdf(x: float | np.ndarray) -> float | np.ndarray
```

Evaluate the log-density at `x`. Returns `−∞` for `x ≤ 0`.

---

##### `summary()`

```python
def summary() -> dict
```

Return a dictionary with keys `"distribution"`, `"scale"`, `"median"`
($= \beta$), `"p25"`, `"p75"`, `"p95"`.

---

#### Example

```python
from kalmanbox.bayesian.priors import HalfCauchy
import numpy as np

# Weakly-informative prior for a standard deviation
prior_sigma = HalfCauchy(scale=2.5)
draws = prior_sigma.sample(size=10_000)
print(f"Median of prior draws: {np.median(draws):.4f}  (= scale = 2.5)")
print(f"95th pct: {np.percentile(draws, 95):.4f}")
```

---

### InverseWishart

`kalmanbox.bayesian.priors.InverseWishart`

The Inverse-Wishart distribution $\mathcal{IW}(\nu, \Psi)$ is the
conjugate prior for **positive-definite covariance matrices**. It is used
as a prior for multivariate state disturbance covariances $Q$ and
observation covariance matrices $H$ in multivariate models (DFM, VAR-SSM,
multivariate UCM).

The density over a $p \times p$ positive-definite matrix $\Sigma$ is:

$$
p(\Sigma) \propto |\Sigma|^{-(\nu + p + 1)/2}
\exp\!\left(-\tfrac{1}{2}\operatorname{tr}(\Psi\,\Sigma^{-1})\right)
$$

The conjugate Gibbs update, given $n$ outer-product residual terms
$\sum_t e_t e_t'$, yields:

$$
\Sigma \mid \text{states}
\;\sim\; \mathcal{IW}\!\left(\nu + n,\; \Psi + \sum_{t=1}^n e_t e_t'\right)
$$

!!! warning "Degrees of freedom constraint"

    The mean of $\mathcal{IW}(\nu, \Psi)$ is $\Psi / (\nu - p - 1)$, which
    requires $\nu > p + 1$. For the prior to have a finite mean, set
    `df > p + 1` where `p` is the dimension of the covariance matrix.

#### Constructor

```python
InverseWishart(
    df: float,
    scale_matrix: np.ndarray,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `float` | required | Degrees of freedom $\nu > p - 1$. Higher values concentrate the prior around $\Psi / (\nu - p - 1)$. |
| `scale_matrix` | `np.ndarray` | required | Positive-definite scale matrix $\Psi$. Shape `(p, p)`. Raises `ValueError` if not positive definite. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `mean` | `np.ndarray` | Prior mean $\Psi / (\nu - p - 1)$. Shape `(p, p)`. Defined only when $\nu > p + 1$. |
| `mode` | `np.ndarray` | Prior mode $\Psi / (\nu + p + 1)$. Shape `(p, p)`. Defined for all valid $\nu$. |

#### Methods

##### `sample(size=None)`

```python
def sample(size: int | None = None) -> np.ndarray
```

Draw i.i.d. samples from $\mathcal{IW}(\nu, \Psi)$.

**Returns** `np.ndarray` of shape `(p, p)` if `size` is `None`; otherwise
shape `(size, p, p)`.

---

##### `log_pdf(X)`

```python
def log_pdf(X: np.ndarray) -> float
```

Evaluate the log-density at positive-definite matrix `X`. Shape `(p, p)`.
Returns `-∞` if `X` is not positive definite.

---

##### `summary()`

```python
def summary() -> dict
```

Return a dictionary with keys `"distribution"`, `"df"`, `"scale_matrix"`,
`"mean"`, `"mode"`, `"dim"`.

---

#### Example

```python
import numpy as np
from kalmanbox.bayesian.priors import InverseWishart

# Prior for a 3x3 state covariance matrix
Psi = np.diag([0.1, 0.05, 0.02])   # scale matrix
prior_Q = InverseWishart(df=6.0, scale_matrix=Psi)

print("Prior mean:\n", prior_Q.mean)   # Psi / (df - p - 1) = Psi / 2
print("Prior mode:\n", prior_Q.mode)   # Psi / (df + p + 1) = Psi / 10

# Draw 1000 covariance matrices
samples = prior_Q.sample(size=1000)  # shape (1000, 3, 3)
print(f"Sample mean of [0,0] entry: {samples[:, 0, 0].mean():.4f}")
```

---

## Convergence Diagnostics

`kalmanbox.bayesian`

These module-level functions operate on the raw arrays of posterior draws
and do not require a `GibbsSampler` instance. They can be applied to
output from any MCMC sampler, including third-party tools such as PyMC
or Stan, as long as the draws are provided as NumPy arrays.

---

### `trace_plot`

```python
def trace_plot(
    samples: dict[str, np.ndarray] | np.ndarray,
    param: str | None = None,
    n_chains: int = 1,
    figsize: tuple[float, float] = (10, 6),
) -> matplotlib.figure.Figure
```

Plot the iteration trace and posterior density for a single parameter.
When `samples` is a `dict`, `param` selects the key. When `samples` is an
`ndarray` of shape `(n_draws,)` or `(n_chains, n_draws)`, `param` is used
only as the axis label.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `samples` | `dict[str, np.ndarray] \| np.ndarray` | Posterior draws. |
| `param` | `str \| None` | Parameter name (dict key or label). |
| `n_chains` | `int` | Number of chains. Used to split a 1-D array into `(n_chains, n_draws // n_chains)`. |
| `figsize` | `tuple[float, float]` | Figure dimensions in inches. |

**Returns** `matplotlib.figure.Figure`.

---

### `rhat`

```python
def rhat(
    samples: dict[str, np.ndarray] | np.ndarray,
    param: str | None = None,
) -> float
```

Compute the **Gelman-Rubin split-chain R-hat** for a single parameter.
The algorithm follows Vehtari et al. (2021) and rank-normalises the draws
before computing the between-chain and within-chain variance ratio.

$$
\hat{R} = \sqrt{\frac{\widehat{\text{Var}}(\theta \mid y)}{W}}
$$

where $\widehat{\text{Var}}$ is the mixture estimate of the marginal
posterior variance and $W$ is the average within-chain variance.

**Returns** `float`. Values below 1.05 indicate convergence. Values above
1.10 indicate serious convergence problems.

!!! warning "Minimum chain requirement"

    At least 2 chains (or 1 chain that is split into 2 halves) are
    required. Pass a 2-D array of shape `(n_chains, n_draws)`.

---

### `ess`

```python
def ess(
    samples: dict[str, np.ndarray] | np.ndarray,
    param: str | None = None,
) -> dict[str, float]
```

Compute the **bulk ESS** and **tail ESS** using the Vehtari et al. (2021)
rank-normalised estimator. Bulk ESS measures how well the sampler explores
the centre of the distribution; tail ESS measures reliability of quantile
estimates.

**Returns** `dict[str, float]` with keys `"bulk"` and `"tail"`.

| Rule of thumb | Criterion |
|---|---|
| Reliable posterior mean | bulk ESS > 100 per chain |
| Reliable 5 %/95 % quantiles | tail ESS > 400 total |
| Reliable 1 %/99 % quantiles | tail ESS > 1000 total |

---

### `geweke`

```python
def geweke(
    samples: dict[str, np.ndarray] | np.ndarray,
    param: str | None = None,
    first: float = 0.1,
    last: float = 0.5,
) -> GewekeResult
```

Perform the Geweke (1992) spectral convergence test. A statistically
significant z-score (|z| > 1.96) suggests the chain has not yet converged
to stationarity by the start of the window defined by `first`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `samples` | `dict[str, np.ndarray] \| np.ndarray` | Posterior draws from a single chain. |
| `param` | `str \| None` | Key when `samples` is a dict. |
| `first` | `float` | Fraction for the early window. Default `0.1`. |
| `last` | `float` | Fraction for the late window. Default `0.5`. |

**Returns** `GewekeResult` with fields `z_score`, `p_value`, `early_mean`,
`late_mean`.

---

### `posterior_predictive`

```python
def posterior_predictive(
    gibbs_result: GibbsResult,
    y_new: np.ndarray,
    n_samples: int = 500,
) -> np.ndarray
```

Generate **posterior predictive samples** for a new observation sequence
`y_new` by combining parameter uncertainty (from the posterior draws) with
the observation model noise.

For each of `n_samples` parameter draws $\theta^{(s)}$, the function runs
the Kalman filter on `y_new` under $\theta^{(s)}$ and simulates one
realisation from the predictive distribution:

$$
\tilde{y}_t^{(s)} \sim \mathcal{N}(Z\,a_{t|t-1}^{(s)},\;
Z P_{t|t-1}^{(s)} Z' + H^{(s)})
$$

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `gibbs_result` | `GibbsResult` | Output of `GibbsSampler.run()` containing posterior draws. |
| `y_new` | `np.ndarray` | New observation sequence. Shape `(n_new, p)`. May contain `np.nan` for forecasting periods. |
| `n_samples` | `int` | Number of posterior draws to use. Subsampled uniformly from the available draws. |

**Returns** `np.ndarray` of shape `(n_samples, n_new, p)` — the posterior
predictive trajectories. Compute quantiles across axis 0 for credible
intervals.

---

## Complete Bayesian Workflow

The following example demonstrates a full end-to-end Bayesian analysis
of a quarterly GDP growth series using a **Basic Structural Model** (BSM)
with trend, seasonal, and irregular components.

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox import StateSpaceRepresentation
from kalmanbox.models import BasicStructuralModel
from kalmanbox.bayesian import (
    GibbsSampler,
    FFBS,
    InverseGamma,
    posterior_predictive,
    rhat,
    ess,
    trace_plot,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Simulate quarterly GDP-like data (trend + seasonal + noise)
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(2024)
n        = 120          # 30 years of quarterly data
n_season = 4            # quarterly seasonality

sigma2_trend  = 0.001   # level innovation
sigma2_slope  = 0.0001  # slope innovation
sigma2_seas   = 0.005   # seasonal innovation
sigma2_irreg  = 0.04    # observation noise

# Simulate trend (local linear)
level = np.zeros(n + 1)
slope = np.zeros(n + 1)
slope[0] = 0.005        # 0.5 % per quarter initial trend
for t in range(n):
    slope[t+1] = slope[t] + rng.normal(0, np.sqrt(sigma2_slope))
    level[t+1] = level[t] + slope[t] + rng.normal(0, np.sqrt(sigma2_trend))

# Simulate quarterly seasonal component (sum-zero constraint)
gamma = np.zeros(n + 1)
gamma[0] = 0.0
seasonal_pattern = np.array([0.8, -0.3, -0.9, 0.4])  # Q1 boom, Q3 trough
for t in range(n):
    gamma[t+1] = -sum(
        gamma[t - k] if t - k >= 0 else seasonal_pattern[(-k) % n_season]
        for k in range(n_season - 1)
    ) + rng.normal(0, np.sqrt(sigma2_seas))

y = level[1:] + gamma[1:] + rng.normal(0, np.sqrt(sigma2_irreg), n)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the BSM state-space representation
# ─────────────────────────────────────────────────────────────────────────────
bsm = BasicStructuralModel(period=4)
ss  = bsm.to_state_space(
    sigma2_level=0.001,   # starting values (will be estimated)
    sigma2_slope=1e-4,
    sigma2_seasonal=0.005,
    sigma2_irreg=0.04,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Define priors — weakly-informative InverseGamma on each variance
# ─────────────────────────────────────────────────────────────────────────────
priors = {
    "sigma2_irreg":    InverseGamma(shape=2.0, scale=0.02),
    "sigma2_level":    InverseGamma(shape=2.0, scale=0.001),
    "sigma2_slope":    InverseGamma(shape=2.0, scale=1e-5),
    "sigma2_seasonal": InverseGamma(shape=2.0, scale=0.005),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run the Gibbs sampler
# ─────────────────────────────────────────────────────────────────────────────
sampler = GibbsSampler(
    ss=ss,
    priors=priors,
    n_iter=4000,
    burn_in=1000,
    thinning=2,
    n_chains=4,
    seed=123,
)

result = sampler.run(y[:, np.newaxis])

# ─────────────────────────────────────────────────────────────────────────────
# 5. Check convergence
# ─────────────────────────────────────────────────────────────────────────────
print("─── Convergence diagnostics ───────────────────────────────────────")
for param in priors:
    r    = result.rhat[param]
    bulk = result.ess[f"{param}.bulk"]
    tail = result.ess[f"{param}.tail"]
    flag = "OK" if r < 1.05 else "WARN"
    print(f"  {param:<22s}  R-hat={r:.4f} {flag}   "
          f"ESS bulk={bulk:.0f}  tail={tail:.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Extract posterior means and credible intervals
# ─────────────────────────────────────────────────────────────────────────────
print("\n─── Posterior summaries ────────────────────────────────────────────")
true_vals = {
    "sigma2_irreg":    sigma2_irreg,
    "sigma2_level":    sigma2_trend,
    "sigma2_slope":    sigma2_slope,
    "sigma2_seasonal": sigma2_seas,
}
for param in priors:
    post = result.posterior[param]
    print(
        f"  {param:<22s}  mean={post.mean:.5f}  "
        f"94% HDI=[{post.hdi_low:.5f}, {post.hdi_high:.5f}]  "
        f"true={true_vals[param]:.5f}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7. Trace plots for visual diagnostics
# ─────────────────────────────────────────────────────────────────────────────
for param in ["sigma2_irreg", "sigma2_level"]:
    fig = sampler.trace_plot(param, figsize=(12, 5))
    fig.suptitle(f"Trace plot: {param}", fontsize=13)
    fig.savefig(f"trace_{param}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Posterior state draws via FFBS
# ─────────────────────────────────────────────────────────────────────────────
# Use the posterior mean parameters to construct the smoothed state
ss_post = bsm.to_state_space(
    sigma2_level    = result.posterior["sigma2_level"].mean,
    sigma2_slope    = result.posterior["sigma2_slope"].mean,
    sigma2_seasonal = result.posterior["sigma2_seasonal"].mean,
    sigma2_irreg    = result.posterior["sigma2_irreg"].mean,
)

ffbs = FFBS(ss_post, seed=0)
state_draws = ffbs.sample_states(y[:, np.newaxis], n_samples=500)
# state_draws.shape == (500, 120, state_dim)

# Extract level component (state index 0) credible band
level_draws = state_draws[:, :, 0]
level_mean  = level_draws.mean(axis=0)
level_lo    = np.percentile(level_draws,  3, axis=0)
level_hi    = np.percentile(level_draws, 97, axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(range(n), level_lo, level_hi, alpha=0.20,
                color="C0", label="94 % credible band")
ax.plot(level_mean, color="C0", linewidth=1.5, label="Posterior mean level")
ax.plot(y, "k.", alpha=0.25, markersize=3, label="Observed GDP growth")
ax.set_xlabel("Quarter")
ax.set_ylabel("Growth rate")
ax.set_title("BSM — Smoothed trend with Bayesian uncertainty")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("bsm_bayesian_trend.png", dpi=150)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Posterior predictive check — 8-quarter forecast
# ─────────────────────────────────────────────────────────────────────────────
n_fc = 8
# Append NaN rows to y so posterior_predictive handles forecasting periods
y_extended = np.full((n + n_fc, 1), np.nan)
y_extended[:n, 0] = y

pp_draws = posterior_predictive(
    gibbs_result=result,
    y_new=y_extended,
    n_samples=500,
)
# pp_draws.shape == (500, n + n_fc, 1)

fc_draws = pp_draws[:, n:, 0]          # forecast draws only
fc_mean  = fc_draws.mean(axis=0)
fc_lo    = np.percentile(fc_draws,  3, axis=0)
fc_hi    = np.percentile(fc_draws, 97, axis=0)

time_fc  = np.arange(n, n + n_fc)
fig, ax  = plt.subplots(figsize=(12, 5))
ax.plot(y, "k.", alpha=0.25, markersize=3, label="Observed")
ax.fill_between(time_fc, fc_lo, fc_hi, alpha=0.30, color="C1", label="94 % predictive")
ax.plot(time_fc, fc_mean, color="C1", linewidth=2, label="Forecast mean")
ax.axvline(n - 1, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Quarter")
ax.set_ylabel("GDP growth rate")
ax.set_title("8-quarter posterior predictive forecast")
ax.legend()
fig.tight_layout()
fig.savefig("bsm_forecast.png", dpi=150)

print("\nWorkflow complete. Figures saved to disk.")
```

!!! tip "Parallelising chains"

    For 4-chain runs on multi-core machines, set the kalmanbox backend
    before calling `run()`:

    ```python
    import kalmanbox
    kalmanbox.set_backend("multiprocessing", n_jobs=4)
    sampler.run(y[:, np.newaxis])
    ```

    Each chain is dispatched to a separate worker process. Wall-clock time
    scales roughly as `n_iter / n_cores` for chains of equal length.

---

## See Also

- [User Guide: Bayesian Estimation](../user-guide/bayesian/index.md)
- [User Guide: Gibbs Sampling](../user-guide/bayesian/gibbs.md)
- [User Guide: FFBS](../user-guide/bayesian/ffbs.md)
- [User Guide: Priors](../user-guide/bayesian/priors.md)
- [User Guide: Posterior Diagnostics](../user-guide/bayesian/posterior-diagnostics.md)
- [Theory: Bayesian State-Space Theory](../theory/bayesian-theory.md)
- [Tutorial: Bayesian Walkthrough](../tutorials/bayesian-walkthrough.md)
- [API: Core (KalmanFilter, RTSSmoother)](core.md)
- [API: Alternative Filters](filters.md)
