# Datasets API

`kalmanbox.datasets`

`kalmanbox.datasets` ships six real-world benchmark series and three simulation
generators that cover the full range of state-space modelling use cases. All
loaders return a `Dataset` named object described below. Simulation functions
return `(y, true_params)` tuples for controlled experiments where the ground
truth is known, enabling direct assessment of filter and smoother performance.

| Function | Returns | Description |
|---|---|---|
| [`load_nile()`](#load_nile) | `Dataset` | Annual Nile river discharge (1871–1970) |
| [`load_airline()`](#load_airline) | `Dataset` | Monthly airline passengers (1949–1960) |
| [`load_gas()`](#load_gas) | `Dataset` | UK quarterly gas consumption (1960–1986) |
| [`load_gdp()`](#load_gdp) | `Dataset` | US quarterly real GDP growth (1947–2023) |
| [`load_employment()`](#load_employment) | `Dataset` | US monthly employment indicators, multivariate (1948–2023) |
| [`load_inflation()`](#load_inflation) | `Dataset` | US monthly CPI inflation (1947–2023) |
| [`simulate_local_level()`](#simulate_local_level) | `tuple[np.ndarray, dict]` | Synthetic Local Level Model data |
| [`simulate_bsm()`](#simulate_bsm) | `tuple[np.ndarray, dict]` | Synthetic Basic Structural Model data |
| [`simulate_dfm()`](#simulate_dfm) | `tuple[np.ndarray, dict]` | Synthetic Dynamic Factor Model data |

See also the [Tutorials](../tutorials/index.md) for worked end-to-end examples
that use these datasets.

---

## `Dataset` namedtuple

`kalmanbox.datasets.Dataset`

```python
Dataset = collections.namedtuple(
    "Dataset",
    ["data", "target", "dates", "name", "description", "source",
     "n_obs", "n_series", "frequency"],
)
```

All loader functions return a `Dataset` instance. The fields are:

| Field | Type | Description |
|---|---|---|
| `data` | `np.ndarray` | Observation array. Shape `(T,)` for univariate or `(T, p)` for multivariate series. |
| `target` | `str` | Name of the primary target variable (or the first column for multivariate datasets). |
| `dates` | `pd.DatetimeIndex` | Datetime index aligned with the first axis of `data`. |
| `name` | `str` | Dataset name string. |
| `description` | `str` | One-paragraph description of the dataset. |
| `source` | `str` | Citation or URL for the original data source. |
| `n_obs` | `int` | Number of time periods T. |
| `n_series` | `int` | Number of observed series p (1 for univariate). |
| `frequency` | `str` | Pandas frequency string: `"A"` (annual), `"M"` (monthly), `"Q"` (quarterly). |

**Example — inspecting a dataset**

```python
from kalmanbox.datasets import load_nile

ds = load_nile()
print(f"name      : {ds.name}")
print(f"n_obs     : {ds.n_obs}")
print(f"n_series  : {ds.n_series}")
print(f"frequency : {ds.frequency}")
print(f"source    : {ds.source}")
print(f"data type : {ds.data.dtype}, shape {ds.data.shape}")
print(f"dates     : {ds.dates[0]} -> {ds.dates[-1]}")
```

---

## Real Datasets

---

### `load_nile`

```python
kalmanbox.datasets.load_nile() -> Dataset
```

Loads the annual volume of flow of the River Nile at Aswan, Egypt, measured
in 10⁸ cubic metres. This is the most widely used benchmark dataset for the
Kalman filter and the Local Level Model, appearing in Harvey (1989), Durbin &
Koopman (2012), and countless tutorials. The series spans 1871 to 1970
(100 annual observations).

**Dataset properties**

| Property | Value |
|---|---|
| Series | Annual Nile discharge |
| Frequency | Annual |
| Period | 1871–1970 |
| Observations | 100 |
| Series count | 1 (univariate) |
| Source | Cobb (1978), reproduced in Harvey (1989) |
| Units | 10⁸ m³/year |

**Parameters**

This function takes no parameters.

**Returns** `Dataset` — see the [Dataset namedtuple](#dataset-namedtuple) for
field descriptions.

!!! note "The 1898 level shift"

    The Nile series contains a well-known downward level shift around 1898,
    coinciding with the construction of the first Aswan Low Dam. This
    structural break makes the dataset a standard benchmark for CUSUM tests,
    auxiliary residual analysis, and intervention modelling. A correctly
    specified Local Level Model with diffuse initialisation will detect the
    shift via a spike in the state auxiliary residuals at that date.

**Example**

```python
from kalmanbox.datasets import load_nile
from kalmanbox import LocalLevel

ds = load_nile()
print(ds.name, ds.n_obs, ds.frequency)
# Nile River Discharge 100 A

model = LocalLevel()
result = model.fit(ds.data)
print(result.summary())
```

---

### `load_airline`

```python
kalmanbox.datasets.load_airline() -> Dataset
```

Monthly totals of international airline passengers (in thousands) from
January 1949 to December 1960. The series exhibits a strong upward trend
and multiplicative seasonality — a classic testbed for the Basic Structural
Model (BSM). Box & Jenkins (1976) used this series as their primary
illustration of seasonal ARIMA models; in the state-space literature it is
the standard example for log-additive BSM estimation.

**Dataset properties**

| Property | Value |
|---|---|
| Series | International airline passengers |
| Frequency | Monthly |
| Period | Jan 1949 – Dec 1960 |
| Observations | 144 |
| Series count | 1 (univariate) |
| Source | Box & Jenkins (1976), Series G |
| Units | Thousands of passengers |

**Parameters**

This function takes no parameters.

**Returns** `Dataset`.

!!! note "Multiplicative seasonality"

    Because the seasonal amplitude grows proportionally to the trend level,
    a log transformation is recommended before fitting a BSM or any additive
    state-space model. After taking `np.log(ds.data)` the series is well
    approximated by an additive trend-plus-seasonal decomposition.

**Example**

```python
import numpy as np
from kalmanbox.datasets import load_airline
from kalmanbox import BSM

ds = load_airline()

# Log-transform to make seasonality additive
y_log = np.log(ds.data)

model  = BSM(period=12)
result = model.fit(y_log)
print(result.summary())

# Back-transform the smoothed trend to original scale
trend_level = np.exp(result.components["trend"])
print(f"End-of-sample trend (passengers): {trend_level[-1]:.0f}k")
```

---

### `load_gas`

```python
kalmanbox.datasets.load_gas() -> Dataset
```

Quarterly UK gas consumption (in millions of therms) from Q1 1960 to Q4 1986.
The series is strongly seasonal (four-quarter cycle) with a structural level
break in the late 1970s caused by the switch from town gas to North Sea
natural gas, which substantially increased the baseline level of consumption.
Durbin & Koopman (2012) use this series as a primary illustration of UCM
estimation with a seasonal component.

**Dataset properties**

| Property | Value |
|---|---|
| Series | UK gas consumption |
| Frequency | Quarterly |
| Period | Q1 1960 – Q4 1986 |
| Observations | 108 |
| Series count | 1 (univariate) |
| Source | Durbin & Koopman (2012), Appendix G |
| Units | Millions of therms |

**Parameters**

This function takes no parameters.

**Returns** `Dataset`.

!!! note "Structural break"

    The late-1970s level shift caused by North Sea gas substitution makes this
    dataset useful for studying structural breaks alongside seasonal modelling.
    A UCM with a stochastic level will absorb the break gradually; for a
    sharper identification, consider an intervention dummy or a CUSUM test to
    pinpoint the transition quarter.

**Example**

```python
from kalmanbox.datasets import load_gas
from kalmanbox import UCM

ds = load_gas()
print(f"Loaded {ds.n_obs} quarterly observations from {ds.dates[0].year}")

model  = UCM(period=4, irregular=True, level=True, slope=False, seasonal="trigonometric")
result = model.fit(ds.data)
print(result.summary())
```

---

### `load_gdp`

```python
kalmanbox.datasets.load_gdp() -> Dataset
```

US real GDP quarterly growth rate (annualised, in percent) from Q1 1947 to
Q4 2023, sourced from the US Bureau of Economic Analysis (FRED series GDPC1).
The growth series is stationary in mean but exhibits time-varying volatility,
making it a standard dataset for trend-cycle decomposition, TVP models, and
Dynamic Factor Models of the business cycle.

**Dataset properties**

| Property | Value |
|---|---|
| Series | US real GDP growth |
| Frequency | Quarterly |
| Period | Q1 1947 – Q4 2023 |
| Observations | 308 |
| Series count | 1 (univariate) |
| Source | US Bureau of Economic Analysis, FRED series GDPC1 |
| Units | Annualised percent change |

**Parameters**

This function takes no parameters.

**Returns** `Dataset`.

**Example**

```python
import matplotlib.pyplot as plt
from kalmanbox.datasets import load_gdp
from kalmanbox import LocalLinearTrend

ds = load_gdp()

model  = LocalLinearTrend()
result = model.fit(ds.data)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ds.dates, ds.data,                   label="GDP growth",    alpha=0.6)
ax.plot(ds.dates, result.a_smooth[:, 0],     label="Trend (LLT)",   linewidth=2)
ax.set_title("US Real GDP Growth — Local Linear Trend decomposition")
ax.set_ylabel("Annualised %")
ax.legend()
plt.tight_layout()
plt.show()
```

---

### `load_employment`

```python
kalmanbox.datasets.load_employment(
    series: list[str] | None = None,
) -> Dataset
```

US monthly employment indicators panel from January 1948 to December 2023,
sourced from the Bureau of Labor Statistics via FRED. The panel contains four
series that together capture distinct dimensions of labour market conditions:
Nonfarm Payrolls (thousands), Unemployment Rate (percent), Average Hourly
Earnings (dollars), and Hours Worked (weekly hours). This dataset is widely
used for multivariate DFM estimation and TVP models of labour market dynamics.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `series` | `list[str] \| None` | `None` | Subset of columns to load. Valid names: `"payrolls"`, `"unemployment"`, `"hourly_earnings"`, `"hours_worked"`. `None` loads all four series. |

**Returns** `Dataset` with `data` of shape `(912, p)` where `p` is the number
of selected series (1–4). The `n_series` field reflects the selection.

**Dataset properties**

| Property | Value |
|---|---|
| Series | US employment indicators |
| Frequency | Monthly |
| Period | Jan 1948 – Dec 2023 |
| Observations | 912 |
| Series count | 4 (multivariate) when all series loaded |
| Source | US Bureau of Labor Statistics via FRED |
| Units | Mixed (thousands, percent, dollars, hours) |

!!! warning "Mixed units"

    The four series are measured on very different scales. Standardise or
    demean each column before passing to DFM or multivariate Kalman filter
    functions to avoid numerical dominance by the payrolls series (which is
    in hundreds of thousands).

**Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.datasets import load_employment
from kalmanbox import DynamicFactorModel

# Load all four series and standardise
ds = load_employment()
y  = (ds.data - ds.data.mean(axis=0)) / ds.data.std(axis=0)

# Fit a one-factor DFM
model  = DynamicFactorModel(n_factors=1)
result = model.fit(y)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ds.dates, result.factors[:, 0], label="Common factor (employment)")
ax.set_title("DFM — Common Employment Factor")
ax.legend()
plt.tight_layout()
plt.show()

# Load only unemployment and payrolls
ds2 = load_employment(series=["payrolls", "unemployment"])
print(f"n_series={ds2.n_series}, shape={ds2.data.shape}")
```

---

### `load_inflation`

```python
kalmanbox.datasets.load_inflation(
    measure: str = "cpi",
) -> Dataset
```

US monthly inflation rate (year-on-year percent change) derived from the CPI
or PCE deflator, from January 1947 to December 2023. Inflation persistence,
time-varying mean, and regime changes make this a natural application for
TVP models and Bayesian state-space estimation.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `measure` | `str` | `"cpi"` | Price index to use. `"cpi"` loads the Consumer Price Index (FRED: CPIAUCSL); `"pce"` loads the PCE deflator (FRED: PCEPI). |

**Returns** `Dataset` with `n_obs=924` and `frequency="M"`.

**Dataset properties**

| Property | Value |
|---|---|
| Series | US CPI/PCE inflation |
| Frequency | Monthly |
| Period | Jan 1947 – Dec 2023 |
| Observations | 924 |
| Series count | 1 (univariate) |
| Source | US Bureau of Labor Statistics / BEA via FRED |
| Units | Percent year-on-year |

**Example**

```python
import matplotlib.pyplot as plt
from kalmanbox.datasets import load_inflation
from kalmanbox import TVPModel

# CPI inflation with a time-varying AR(1) persistence parameter
ds = load_inflation(measure="cpi")

model  = TVPModel(specification="ar1")
result = model.fit(ds.data)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(ds.dates, ds.data,               label="CPI inflation")
axes[0].set_ylabel("Percent YoY")
axes[0].legend()

axes[1].plot(ds.dates, result.tvp_params[:, 0], label="AR(1) coefficient")
axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[1].set_ylabel("Coefficient")
axes[1].set_title("Time-varying inflation persistence")
axes[1].legend()
plt.tight_layout()
plt.show()
```

---

## Simulation Functions

Simulation functions generate synthetic data from a fully specified model and
return both the observed series and a dictionary of ground-truth parameters.
This enables controlled experiments: you can fit a model to the synthetic
series and directly compare the estimated parameters and state trajectories
against the known truth.

---

### `simulate_local_level`

```python
kalmanbox.datasets.simulate_local_level(
    n: int = 100,
    sigma_eps: float = 1.0,
    sigma_eta: float = 0.5,
    mu0: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]
```

Generate a synthetic time series from a Local Level Model (random walk plus
noise):

$$
\mu_{t+1} = \mu_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\, \sigma_\eta^2)
$$

$$
y_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_\varepsilon^2)
$$

The signal-to-noise ratio is $q = \sigma_\eta^2 / \sigma_\varepsilon^2$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `100` | Number of time steps T. |
| `sigma_eps` | `float` | `1.0` | Observation noise standard deviation $\sigma_\varepsilon$. |
| `sigma_eta` | `float` | `0.5` | State evolution noise standard deviation $\sigma_\eta$. |
| `mu0` | `float` | `0.0` | Initial state value $\mu_1$. |
| `seed` | `int \| None` | `None` | Random seed passed to `np.random.default_rng` for reproducibility. |

**Returns** `tuple[np.ndarray, dict]`:

| Component | Type | Description |
|---|---|---|
| `y` | `np.ndarray` shape `(n,)` | Simulated observed series. |
| `params` | `dict` | Ground-truth parameters and trajectories. Keys: `sigma_eps` (float), `sigma_eta` (float), `mu0` (float), `states` (np.ndarray shape `(n,)` — true state trajectory $\mu_1, \ldots, \mu_T$). |

**Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.datasets import simulate_local_level
from kalmanbox import LocalLevel

# Generate data with known SNR = 0.25
y, truth = simulate_local_level(n=200, sigma_eps=1.0, sigma_eta=0.5, seed=42)
true_snr  = (truth["sigma_eta"] / truth["sigma_eps"]) ** 2
print(f"True SNR  : {true_snr:.4f}")

# Fit and compare
model  = LocalLevel()
result = model.fit(y)
est_snr = result.params["signal_noise_ratio"]
print(f"Est.  SNR : {est_snr:.4f}")

# Compare true vs filtered states
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y,                       color="lightgrey",  label="Observed",        zorder=1)
ax.plot(truth["states"],         color="black",      label="True state",      linewidth=2)
ax.plot(result.a_smooth[:, 0],   color="steelblue",  label="Filtered state",  linewidth=2)
ax.set_title(f"Local Level — true vs estimated state  (SNR={true_snr:.2f})")
ax.legend()
plt.tight_layout()
plt.show()
```

---

### `simulate_bsm`

```python
kalmanbox.datasets.simulate_bsm(
    n: int = 200,
    seasonal_period: int = 12,
    sigma_level: float = 0.3,
    sigma_slope: float = 0.05,
    sigma_seasonal: float = 0.1,
    sigma_irregular: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]
```

Generate synthetic data from a Basic Structural Model (Harvey 1989) with a
local linear trend and a trigonometric seasonal component. The data-generating
process is:

$$
\mu_{t+1} = \mu_t + \nu_t + \eta_t^{(\mu)}, \qquad \eta_t^{(\mu)} \sim \mathcal{N}(0,\, \sigma_\text{level}^2)
$$

$$
\nu_{t+1} = \nu_t + \eta_t^{(\nu)}, \qquad \eta_t^{(\nu)} \sim \mathcal{N}(0,\, \sigma_\text{slope}^2)
$$

$$
y_t = \mu_t + \gamma_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_\text{irr}^2)
$$

where $\gamma_t$ is the trigonometric seasonal component with period $s$ and
disturbance variance $\sigma_\text{seasonal}^2$.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `200` | Number of observations. |
| `seasonal_period` | `int` | `12` | Seasonal period $s$ (12 for monthly, 4 for quarterly data). |
| `sigma_level` | `float` | `0.3` | Level disturbance standard deviation. |
| `sigma_slope` | `float` | `0.05` | Slope disturbance standard deviation. |
| `sigma_seasonal` | `float` | `0.1` | Seasonal disturbance standard deviation. |
| `sigma_irregular` | `float` | `0.5` | Irregular (observation noise) standard deviation. |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. |

**Returns** `tuple[np.ndarray, dict]`:

| Component | Type | Description |
|---|---|---|
| `y` | `np.ndarray` shape `(n,)` | Simulated observed series. |
| `params` | `dict` | Ground-truth parameters and component trajectories. Keys: `sigma_level`, `sigma_slope`, `sigma_seasonal`, `sigma_irregular` (all floats), `trend` (np.ndarray shape `(n,)`), `seasonal` (np.ndarray shape `(n,)`), `cycle` (`None` — not included in this DGP). |

**Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.datasets import simulate_bsm
from kalmanbox import BSM

# Simulate 10 years of monthly BSM data
y, truth = simulate_bsm(
    n=120,
    seasonal_period=12,
    sigma_level=0.3,
    sigma_slope=0.05,
    sigma_seasonal=0.1,
    sigma_irregular=0.5,
    seed=0,
)

# Fit BSM and extract components
model  = BSM(period=12)
result = model.fit(y)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(y,                             label="Observed",         color="grey",      alpha=0.7)
axes[0].plot(truth["trend"],                label="True trend",       color="black",     linewidth=2)
axes[0].plot(result.components["trend"],    label="Estimated trend",  color="steelblue", linewidth=2)
axes[0].set_title("Trend component")
axes[0].legend(fontsize=8)

axes[1].plot(truth["seasonal"],             label="True seasonal",    color="black",     linewidth=2)
axes[1].plot(result.components["seasonal"], label="Est. seasonal",    color="steelblue", linewidth=2)
axes[1].set_title("Seasonal component")
axes[1].legend(fontsize=8)

axes[2].plot(y - truth["trend"] - truth["seasonal"], label="True irregular", color="black", linewidth=1)
axes[2].set_title("Irregular component")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

---

### `simulate_dfm`

```python
kalmanbox.datasets.simulate_dfm(
    n: int = 200,
    n_series: int = 6,
    n_factors: int = 2,
    factor_var: float = 1.0,
    idiosyncratic_var: float = 0.5,
    loadings: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]
```

Generate synthetic panel data from a Dynamic Factor Model with `n_factors`
common factors driving `n_series` observed series. The data-generating process
is:

$$
f_{t+1} = \Phi\, f_t + u_t, \qquad u_t \sim \mathcal{N}(0,\, \sigma_f^2 I_r)
$$

$$
y_t = \Lambda\, f_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_\varepsilon^2 I_p)
$$

where $\Phi = 0.9\, I_r$ (stationary AR(1) factors), $\Lambda$ is the
$(p \times r)$ loading matrix, and all idiosyncratic variances are equal.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `200` | Number of time periods T. |
| `n_series` | `int` | `6` | Number of observed series $p$. |
| `n_factors` | `int` | `2` | Number of common factors $r$. Must satisfy $r < p$. |
| `factor_var` | `float` | `1.0` | Variance of each factor innovation $\sigma_f^2$. |
| `idiosyncratic_var` | `float` | `0.5` | Idiosyncratic noise variance $\sigma_\varepsilon^2$ (identical for all series). |
| `loadings` | `np.ndarray \| None` | `None` | Factor loading matrix $\Lambda$, shape `(n_series, n_factors)`. If `None`, entries are drawn independently from $\mathcal{N}(0, 1)$ and the matrix is normalised to have orthonormal columns. |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. |

**Returns** `tuple[np.ndarray, dict]`:

| Component | Type | Description |
|---|---|---|
| `y` | `np.ndarray` shape `(n, n_series)` | Simulated multivariate observed series. |
| `params` | `dict` | Ground-truth parameters and factor trajectories. Keys: `loadings` (np.ndarray shape `(n_series, n_factors)`), `factors` (np.ndarray shape `(n, n_factors)` — true factor trajectories), `factor_var` (float), `idiosyncratic_var` (float). |

!!! note "Identification"

    The DGP normalises the loading matrix so that factors and loadings are
    jointly identified up to an orthogonal rotation. When comparing estimated
    loadings to `params["loadings"]`, apply a Procrustes rotation to account
    for this ambiguity.

**Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.datasets import simulate_dfm
from kalmanbox import DynamicFactorModel

# Simulate panel: 200 periods, 6 series, 2 factors
y, truth = simulate_dfm(
    n=200,
    n_series=6,
    n_factors=2,
    factor_var=1.0,
    idiosyncratic_var=0.5,
    seed=7,
)
print(f"Observed data shape : {y.shape}")            # (200, 6)
print(f"True factor shape   : {truth['factors'].shape}")  # (200, 2)
print(f"Loading matrix      :\n{truth['loadings']}")

# Fit a 2-factor DFM
model  = DynamicFactorModel(n_factors=2)
result = model.fit(y)

# Compare estimated factors with true factors (up to sign)
corr0 = np.corrcoef(result.factors[:, 0], truth["factors"][:, 0])[0, 1]
corr1 = np.corrcoef(result.factors[:, 1], truth["factors"][:, 1])[0, 1]
print(f"Factor 0 correlation (est vs true): {corr0:.3f}")
print(f"Factor 1 correlation (est vs true): {corr1:.3f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(truth["factors"][:, i],  label=f"True factor {i+1}",      color="black",     linewidth=2)
    ax.plot(result.factors[:, i],    label=f"Estimated factor {i+1}",  color="steelblue")
    ax.set_title(f"Factor {i+1}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

---

## Full Combined Example

The example below demonstrates all loaders and all three simulation generators
in a single script. Real datasets are summarised in a metadata table; the
simulation generators are each called with a fixed seed so results are
reproducible.

```python
import numpy as np
from kalmanbox.datasets import (
    load_nile,
    load_airline,
    load_gas,
    load_gdp,
    load_employment,
    load_inflation,
    simulate_local_level,
    simulate_bsm,
    simulate_dfm,
)

# ------------------------------------------------------------------
# 1. Inspect all real datasets
# ------------------------------------------------------------------
loaders = {
    "Nile"       : load_nile,
    "Airline"    : load_airline,
    "Gas"        : load_gas,
    "GDP"        : load_gdp,
    "Employment" : load_employment,
    "Inflation"  : load_inflation,
}

print(f"{'Dataset':<14}  {'Obs':>5}  {'Series':>6}  {'Freq':>5}  {'Start':>10}  {'End':>10}")
print("-" * 62)
for label, loader in loaders.items():
    ds = loader()
    print(
        f"{label:<14}  {ds.n_obs:>5}  {ds.n_series:>6}  {ds.frequency:>5}  "
        f"{str(ds.dates[0].date()):>10}  {str(ds.dates[-1].date()):>10}"
    )

# ------------------------------------------------------------------
# 2. Simulation generators — all with fixed seeds for reproducibility
# ------------------------------------------------------------------

# Local Level Model
y_ll, params_ll = simulate_local_level(
    n=150, sigma_eps=1.0, sigma_eta=0.4, mu0=5.0, seed=1
)
print(f"\nLocalLevel sim  — shape: {y_ll.shape},  "
      f"true SNR: {(params_ll['sigma_eta']/params_ll['sigma_eps'])**2:.3f}")

# Basic Structural Model
y_bsm, params_bsm = simulate_bsm(
    n=240, seasonal_period=12,
    sigma_level=0.2, sigma_slope=0.02,
    sigma_seasonal=0.08, sigma_irregular=0.4,
    seed=2,
)
print(f"BSM sim         — shape: {y_bsm.shape},  "
      f"trend range: [{params_bsm['trend'].min():.2f}, {params_bsm['trend'].max():.2f}]")

# Dynamic Factor Model
y_dfm, params_dfm = simulate_dfm(
    n=300, n_series=8, n_factors=3,
    factor_var=1.0, idiosyncratic_var=0.3,
    seed=3,
)
print(f"DFM sim         — shape: {y_dfm.shape},  "
      f"loading matrix shape: {params_dfm['loadings'].shape}")

# ------------------------------------------------------------------
# 3. Quick fit to each simulated series to verify end-to-end
# ------------------------------------------------------------------
from kalmanbox import LocalLevel, BSM, DynamicFactorModel

result_ll  = LocalLevel().fit(y_ll)
result_bsm = BSM(period=12).fit(y_bsm)
result_dfm = DynamicFactorModel(n_factors=3).fit(y_dfm)

print(f"\nLocalLevel  loglik: {result_ll.loglikelihood:.4f}")
print(f"BSM         loglik: {result_bsm.loglikelihood:.4f}")
print(f"DFM         loglik: {result_dfm.loglikelihood:.4f}")
```

---

## See Also

- [Tutorials: Fundamentals](../tutorials/fundamentals.md)
- [Tutorials: BSM](../tutorials/bsm.md)
- [Tutorials: UCM](../tutorials/ucm.md)
- [Tutorials: DFM](../tutorials/dfm.md)
- [Tutorials: Missing Data](../tutorials/missing-data.md)
- [User Guide: Local Level](../user-guide/kalman/kalman-filter.md)
- [User Guide: Structural Models](../user-guide/structural/index.md)
- [API: Core (KalmanFilter)](core.md)
- [API: Advanced Models](advanced.md)
- [API: Diagnostics](diagnostics.md)
- [Theory: Structural Models](../theory/structural-theory.md)
