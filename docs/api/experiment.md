# Experiment API

`kalmanbox.experiment`

The Experiment module provides a structured framework for comparing multiple
state-space model configurations on a common dataset using time-series
cross-validation. The workflow is linear and composable:
**configure models** → **set data** → **run CV** → **inspect results** →
**export a report**. Every step returns `self`, so the entire pipeline can be
expressed as a fluent method chain.

| Class / Function | Role |
|---|---|
| [`Experiment`](#experiment) | Orchestrates model comparison with CV |
| [`CVStrategy`](#cvstrategy) | Enum controlling the cross-validation scheme |
| [`MetricFn`](#metricfn) | Protocol for user-supplied evaluation metrics |
| [`ExperimentResult`](#experimentresult) | Dataclass holding the results grid |
| [`rmse`](#rmse) | Root Mean Squared Error |
| [`mae`](#mae) | Mean Absolute Error |
| [`mape`](#mape) | Mean Absolute Percentage Error |
| [`log_score`](#log_score) | Gaussian log predictive score |

See [User Guide: Experiment Framework](../advanced/experiment.md) for
conceptual background, a detailed walkthrough, and CLI usage.

---

## Experiment

`kalmanbox.experiment.Experiment`

Orchestrates a multi-model comparison experiment. You register any number of
named models — each optionally accompanied by a parameter grid — attach
evaluation metrics, configure the observation series and split proportions,
then call `run()`. Under the hood, the engine iterates over every CV fold,
fits each model candidate on the training window, generates forecasts over the
held-out horizon, and evaluates all attached metrics. Results are collected in
an [`ExperimentResult`](#experimentresult) that supports ranking, visualisation,
and report export.

!!! info "Parameter grids"

    When `params` is a `list[dict]`, `add_model` performs an implicit grid
    search: each dictionary is treated as a separate parameter combination and
    registered as an independent candidate, automatically suffixed
    `model_name/0`, `model_name/1`, … in the results table. This lets you
    sweep over hyperparameters without writing explicit loop code.

!!! warning "Data leakage"

    Always call `set_data` **before** `run`. The split into training windows
    and test windows is performed internally by the CV engine based on
    `CVStrategy` and the specified `train_size` / `test_size`. Never pre-split
    the data yourself — doing so may introduce look-ahead bias or result in
    an inconsistent fold structure.

### Constructor

```python
Experiment(
    name: str,
    description: str | None = None,
    random_state: int | None = None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Short identifier for the experiment. Used as the report title and in auto-generated file names when saving results. |
| `description` | `str \| None` | `None` | Optional free-text description embedded in the `ExperimentReport`. Supports Markdown formatting. |
| `random_state` | `int \| None` | `None` | Global seed for reproducibility. Passed through to all models and CV procedures that accept a seed parameter. |

### Methods

---

#### `add_model`

```python
add_model(
    name: str,
    model_class: type,
    params: dict | list[dict] | None = None,
) -> Experiment
```

Register a model class (not an instance) for inclusion in the experiment.
The class is instantiated fresh for each CV fold so that parameter state from
one fold does not contaminate the next. Returns `self` to allow method
chaining.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Human-readable label for this model. Must be unique within the experiment. Used as a row identifier in all result DataFrames. |
| `model_class` | `type` | required | Uninstantiated class. Must implement `.fit(y, **params)` returning a `FitResult` and `.forecast(h)` returning an `np.ndarray` of length `h`. |
| `params` | `dict \| list[dict] \| None` | `None` | If `dict`, the model is instantiated once with those keyword arguments. If `list[dict]`, each entry is a separate parameter combination (grid search). If `None`, the model is instantiated with no keyword arguments. |

**Returns** `Experiment` (self).

**Example — method chaining**

```python
from kalmanbox import LocalLevel, BSM, UCM
from kalmanbox.experiment import Experiment

exp = (
    Experiment("GDP Decomposition", random_state=42)
    .add_model("local_level", LocalLevel)
    .add_model(
        "bsm_variants",
        BSM,
        params=[
            {"sigma_eps": 0.5, "sigma_eta": 0.1},
            {"sigma_eps": 1.0, "sigma_eta": 0.5},
        ],
    )
    .add_model("ucm", UCM, params={"cycle_period": 4})
)
```

After `add_model("bsm_variants", BSM, params=[...])` the experiment
registers two candidates: `bsm_variants/0` and `bsm_variants/1`.

---

#### `add_metric`

```python
add_metric(
    metric: MetricFn | str,
    name: str | None = None,
) -> Experiment
```

Attach an evaluation metric to the experiment. Built-in metrics can be
referenced by string name: `"rmse"`, `"mae"`, `"mape"`, `"log_score"`.
Custom callables must satisfy the [`MetricFn`](#metricfn) protocol.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `MetricFn \| str` | required | Callable `(y_true, y_pred) -> float` or string alias for a built-in metric. |
| `name` | `str \| None` | `None` | Column label used in the results DataFrame. When `None`, the label is inferred from the callable's `__name__` attribute, or from the string alias when a built-in name is passed. |

**Returns** `Experiment` (self).

**Example**

```python
import numpy as np

def smape(y_true, y_pred):
    """Symmetric MAPE — handles near-zero observations more gracefully."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

exp.add_metric("rmse").add_metric("mae").add_metric(smape, name="smape")
```

---

#### `set_data`

```python
set_data(
    y: np.ndarray | pd.Series,
    train_size: int | float | None = None,
    test_size: int | float | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> Experiment
```

Attach the observation series and specify the train/test split proportions
used by the CV engine. This method must be called before `run()`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y` | `np.ndarray \| pd.Series` | required | One-dimensional observation series. Shape `(T,)`. Missing values encoded as `np.nan` are propagated to every fold and handled by the model's internal missing-data logic. |
| `train_size` | `int \| float \| None` | `None` | Minimum training window length. An `int` specifies the number of observations; a `float` in `(0, 1)` specifies a proportion of `T`. For `CVStrategy.ROLLING` this is also the fixed window size. Defaults to 70 % of `T` when `None`. |
| `test_size` | `int \| float \| None` | `None` | Forecast horizon per fold. An `int` specifies the number of steps ahead; a `float` specifies a proportion of `T`. Defaults to `1` (one-step-ahead evaluation) when `None`. |
| `dates` | `pd.DatetimeIndex \| None` | `None` | Optional datetime index aligned element-wise with `y`. When provided, the fold boundaries and result DataFrames carry human-readable timestamps instead of integer positions. |

**Returns** `Experiment` (self).

**Example**

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
y = rng.normal(size=200)
dates = pd.date_range("2000-01-01", periods=200, freq="QE")

exp.set_data(y, train_size=0.75, test_size=4, dates=dates)
```

---

#### `run`

```python
run(
    cv_strategy: CVStrategy | str = CVStrategy.ROLLING,
    n_folds: int | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> ExperimentResult
```

Execute the experiment: fit each registered model on every CV fold and
evaluate all attached metrics on the held-out test window. The full grid of
`n_models × n_folds` fits is performed before any results are assembled,
so that fold-level errors for one candidate do not interrupt the others.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cv_strategy` | `CVStrategy \| str` | `CVStrategy.ROLLING` | Cross-validation scheme. Accepts `CVStrategy` enum members or the string aliases `"rolling"`, `"expanding"`, `"fixed"`. |
| `n_folds` | `int \| None` | `None` | Number of CV folds. When `None`, the maximum number of non-overlapping folds that fit in the data given `train_size` and `test_size` is used automatically. |
| `n_jobs` | `int` | `1` | Number of parallel worker processes (passed to `joblib.Parallel`). Set to `-1` to use all available cores. Parallelism is applied across models and folds simultaneously. |
| `verbose` | `bool` | `False` | Print fold-by-fold progress to stdout as each fit completes, including the elapsed time per fold. |

**Returns** [`ExperimentResult`](#experimentresult).

**Raises**

- `ValueError` — if no models have been registered, no data has been set, or
  no metrics have been attached.
- `ConvergenceError` — if every candidate model fails to converge on a given
  fold. Per-model convergence failures on individual folds are caught silently
  and recorded as `NaN` in the metrics matrix without raising.

**Example**

```python
from kalmanbox.experiment import CVStrategy

result = exp.run(
    cv_strategy=CVStrategy.EXPANDING,
    n_folds=8,
    n_jobs=-1,
    verbose=True,
)
```

---

#### `results`

```python
results() -> pd.DataFrame
```

Return the per-fold metric matrix as a tidy `DataFrame`. Columns are
`model`, `fold`, and one column per attached metric. Each row corresponds
to a single `(model, fold)` pair. Folds where the model raised an exception
appear with `NaN` for all metric columns.

**Returns** `pd.DataFrame`.

**Raises** `RuntimeError` if called before `run()`.

**Example**

```python
df = exp.results()
print(df.groupby("model")[["rmse", "mae"]].mean())
```

---

#### `best_model`

```python
best_model(
    metric: str | None = None,
    lower_is_better: bool = True,
) -> str
```

Return the name of the best-performing model, ranked by the mean metric
value averaged across all completed folds (folds with `NaN` are excluded
from the mean).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `str \| None` | `None` | Metric column to rank by. Defaults to the first metric registered via `add_metric` when `None`. |
| `lower_is_better` | `bool` | `True` | When `True`, the model with the **lowest** mean metric is returned — appropriate for RMSE, MAE, and MAPE. Set to `False` for log-score or other metrics where higher values indicate better predictive performance. |

**Returns** `str` — the model name exactly as passed to `add_model`.

**Raises** `RuntimeError` if called before `run()`.

**Example**

```python
winner = exp.best_model(metric="rmse", lower_is_better=True)
print(f"Best model: {winner}")

# For log-score (higher = better)
winner_ls = exp.best_model(metric="log_score", lower_is_better=False)
```

---

#### `report`

```python
report(
    output: str | Path | None = None,
    format: str = "html",
) -> ExperimentReport
```

Generate an `ExperimentReport` from the current results and optionally
write it to disk. The report contains the experiment metadata, CV
configuration, per-fold and summary metric tables, and (for HTML/PDF
formats) embedded visualisations of the metric distributions across folds.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output` | `str \| Path \| None` | `None` | Destination file path. When provided, the report is written to this location. The file extension in the path is ignored; the `format` parameter controls serialisation. |
| `format` | `str` | `"html"` | Export format. One of `"html"`, `"pdf"`, `"md"`, `"json"`. HTML and PDF embed charts; `"md"` and `"json"` are text-only. |

**Returns** `ExperimentReport`.

**Raises** `RuntimeError` if called before `run()`.

**Example**

```python
rep = exp.report(output="results/gdp_experiment.html", format="html")
print(rep.summary)
```

---

## CVStrategy

`kalmanbox.experiment.CVStrategy`

Enum that controls how Kalman filter training windows are constructed across
folds. The choice of strategy directly affects how much historical data each
model sees and, therefore, the variance and bias of the cross-validated metric
estimates.

| Member | Value | Description |
|---|---|---|
| `ROLLING` | `"rolling"` | Fixed-width window that slides forward by `test_size` steps each fold. The training set size is constant across folds. Appropriate when the data-generating process may be non-stationary over long horizons. |
| `EXPANDING` | `"expanding"` | Window starts at the minimum `train_size` and grows by `test_size` steps each fold. Uses all history seen so far. Appropriate when longer history is always expected to improve estimation. |
| `FIXED` | `"fixed"` | A single train/test split; no cross-validation. Equivalent to specifying `n_folds=1` with `CVStrategy.ROLLING`. Useful for rapid prototyping or final out-of-sample evaluation. |

!!! info "Fold count arithmetic"

    For a series of length $T$ with `train_size` $= L$ and `test_size` $= h$:

    - **Rolling**: at most $\lfloor (T - L) / h \rfloor$ folds.
    - **Expanding**: at most $\lfloor (T - L) / h \rfloor$ folds (same
      formula, but each fold's training window is longer).
    - **Fixed**: exactly 1 fold regardless of `n_folds`.

**Example**

```python
from kalmanbox.experiment import CVStrategy

# Use string alias or enum member — both are accepted by run()
result_roll = exp.run(cv_strategy="rolling", n_folds=10)
result_exp  = exp.run(cv_strategy=CVStrategy.EXPANDING)
```

---

## MetricFn

`kalmanbox.experiment.MetricFn`

Runtime-checkable protocol for user-supplied evaluation metrics. Any callable
that matches the signature below satisfies the protocol and can be passed
directly to `add_metric`.

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class MetricFn(Protocol):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any,
    ) -> float: ...
```

The arrays `y_true` and `y_pred` are guaranteed to be one-dimensional with no
`NaN` values when passed from the experiment engine — missing observations are
excluded at the fold level before the metric is called.

!!! tip "Custom metrics with extra arguments"

    If your metric requires additional inputs (e.g., predictive variance for a
    proper scoring rule), use `functools.partial` or a closure to bind the
    extra arguments before passing the callable to `add_metric`:

    ```python
    import functools

    def weighted_rmse(y_true, y_pred, weights):
        return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))

    w = np.linspace(0.5, 1.5, num=test_horizon)
    exp.add_metric(functools.partial(weighted_rmse, weights=w), name="wrmse")
    ```

**Checking conformance at runtime**

```python
from kalmanbox.experiment import MetricFn
import numpy as np

def my_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))  # max absolute error

print(isinstance(my_metric, MetricFn))  # True
```

---

## Built-in Metrics

`kalmanbox.experiment`

The four built-in metrics cover the most common requirements for evaluating
point forecasts and probabilistic forecasts from state-space models. All
four are importable directly from `kalmanbox.experiment` and can be referenced
by string alias in `add_metric`.

---

### `rmse`

`kalmanbox.experiment.rmse`

**Root Mean Squared Error** — the most widely used symmetric point-forecast
accuracy measure. Penalises large errors quadratically, making it sensitive
to outlier forecasts.

$$
\text{RMSE} = \sqrt{\frac{1}{H}\sum_{h=1}^{H}(y_h - \hat{y}_h)^2}
$$

```python
rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y_true` | `np.ndarray` | Realised observations. Shape `(H,)`. |
| `y_pred` | `np.ndarray` | Point forecasts. Shape `(H,)`. |

**Returns** `float`. Units match the units of `y_true`.

**Example**

```python
import numpy as np
from kalmanbox.experiment import rmse

y_true = np.array([2.1, 3.4, 2.8, 4.0])
y_pred = np.array([2.0, 3.5, 3.1, 3.8])
print(rmse(y_true, y_pred))  # ~0.212
```

---

### `mae`

`kalmanbox.experiment.mae`

**Mean Absolute Error** — a linear-penalty measure that is more robust than
RMSE to occasional large errors. Equals the median-optimal point forecast
loss under asymmetric data distributions.

$$
\text{MAE} = \frac{1}{H}\sum_{h=1}^{H}|y_h - \hat{y}_h|
$$

```python
mae(y_true: np.ndarray, y_pred: np.ndarray) -> float
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y_true` | `np.ndarray` | Realised observations. Shape `(H,)`. |
| `y_pred` | `np.ndarray` | Point forecasts. Shape `(H,)`. |

**Returns** `float`. Units match the units of `y_true`.

---

### `mape`

`kalmanbox.experiment.mape`

**Mean Absolute Percentage Error** — a scale-free accuracy measure expressed
as a percentage. Useful when comparing forecast accuracy across series
with different units or magnitudes.

$$
\text{MAPE} = \frac{100}{H}\sum_{h=1}^{H}\left|\frac{y_h - \hat{y}_h}{y_h}\right|
$$

!!! warning "Zero observations"

    `mape` raises `ZeroDivisionError` when any element of `y_true` is zero.
    For count data, bounded series near zero, or any series that can take the
    value 0, prefer `mae` or `rmse` instead. Consider the symmetric MAPE
    variant shown in the [`MetricFn`](#metricfn) example above.

```python
mape(y_true: np.ndarray, y_pred: np.ndarray) -> float
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y_true` | `np.ndarray` | Realised observations. Shape `(H,)`. Must not contain zeros. |
| `y_pred` | `np.ndarray` | Point forecasts. Shape `(H,)`. |

**Returns** `float`. Expressed as a percentage (e.g., `5.2` means 5.2 %).

---

### `log_score`

`kalmanbox.experiment.log_score`

**Gaussian log predictive score** — a proper scoring rule that rewards
calibrated probabilistic forecasts. It uses the one-step-ahead predictive
mean and variance from the Kalman filter to evaluate each forecast:

$$
\ell = -\frac{1}{H}\sum_{h=1}^{H}\log p(y_h \mid y_{1:t+h-1})
= \frac{1}{H}\sum_{h=1}^{H}
  \left[\frac{1}{2}\log(2\pi f_h) + \frac{v_h^2}{2 f_h}\right]
$$

where $v_h = y_h - \hat{y}_h$ is the one-step innovation and $f_h$ is the
predictive variance. Lower values indicate better calibrated probabilistic
forecasts (the score is a normalised negative log-likelihood).

!!! note "Predictive variance argument"

    `log_score` requires a `pred_var` keyword argument carrying the predictive
    variances $f_h$. When referenced via `add_metric("log_score")`, the
    experiment engine automatically extracts `FitResult.forecast_variance` and
    passes it as `pred_var`. If you call `log_score` manually you must supply
    this argument explicitly.

```python
log_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_var: np.ndarray,
) -> float
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `y_true` | `np.ndarray` | Realised observations. Shape `(H,)`. |
| `y_pred` | `np.ndarray` | One-step-ahead predictive means. Shape `(H,)`. |
| `pred_var` | `np.ndarray` | One-step-ahead predictive variances $f_h > 0$. Shape `(H,)`. Raises `ValueError` if any entry is non-positive. |

**Returns** `float`. Lower is better.

**Example**

```python
import numpy as np
from kalmanbox.experiment import log_score

y_true   = np.array([2.1, 3.4, 2.8, 4.0])
y_pred   = np.array([2.0, 3.5, 3.1, 3.8])
pred_var = np.array([0.4, 0.3, 0.5, 0.4])

score = log_score(y_true, y_pred, pred_var)
print(f"Log score: {score:.4f}")  # lower = better calibration
```

---

## ExperimentResult

`kalmanbox.experiment.ExperimentResult`

Immutable dataclass returned by `Experiment.run()`. All attributes are
read-only. The object is also accessible via `Experiment.results()` after
`run()` completes.

### Properties

| Property | Type | Description |
|---|---|---|
| `metrics` | `pd.DataFrame` | Tidy DataFrame with columns `model`, `fold`, and one column per metric. Each row is a `(model, fold)` pair. Folds with convergence failures contain `NaN` for all metric columns. |
| `summary` | `pd.DataFrame` | Per-model mean and standard deviation across folds for every metric. Index is the model name; columns are `<metric>_mean` and `<metric>_std`. |
| `models` | `dict[str, type]` | Registered model classes keyed by their label. Grid-search candidates appear as `name/0`, `name/1`, etc. |
| `cv_strategy` | `CVStrategy` | The cross-validation strategy used in the run. |
| `n_folds` | `int` | Actual number of folds executed (may be less than requested if the data is too short). |
| `elapsed_seconds` | `float` | Wall-clock time in seconds for the complete `run()` call. |
| `errors` | `dict[str, list[str]]` | Map of `model_name -> list[error_message]` for any folds that raised exceptions. Empty dict if all fits succeeded. |

### Methods

#### `rank(metric, lower_is_better=True)`

```python
rank(
    metric: str,
    lower_is_better: bool = True,
) -> pd.DataFrame
```

Return a summary DataFrame sorted by mean metric performance. Columns
include `mean`, `std`, `min`, `max`, and `n_folds_ok` (number of folds
that completed without error).

**Returns** `pd.DataFrame` sorted best-first.

---

#### `plot_metric(metric, figsize=(10, 5))`

```python
plot_metric(
    metric: str,
    figsize: tuple[float, float] = (10, 5),
) -> matplotlib.figure.Figure
```

Produce a box-plot of the per-fold metric distribution for all registered
models, sorted by median performance.

**Returns** `matplotlib.figure.Figure`.

---

#### `to_dict()`

```python
to_dict() -> dict
```

Serialise the result to a plain Python dictionary suitable for JSON
export. DataFrames are converted to lists of records; enums are converted
to their string values.

**Returns** `dict`.

---

## Full Example

The following end-to-end example compares three model families on the classic
Nile River annual flow series using expanding-window cross-validation.

### Python API

```python
import numpy as np
import pandas as pd
from kalmanbox import LocalLevel, BSM, UCM
from kalmanbox.experiment import (
    Experiment,
    CVStrategy,
    rmse,
    mae,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load the Nile dataset
# ─────────────────────────────────────────────────────────────────────────────
from kalmanbox.datasets import load_nile

nile = load_nile()              # pd.Series, index = year (1871–1970)
y      = nile.values            # shape (100,)
dates  = pd.DatetimeIndex(
    pd.to_datetime(nile.index.astype(str))
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Create the experiment
# ─────────────────────────────────────────────────────────────────────────────
exp = Experiment(
    name="Nile River Model Comparison",
    description=(
        "Compares LocalLevel, BSM, and UCM on the Nile annual flow series "
        "using expanding-window CV with a 5-fold evaluation."
    ),
    random_state=42,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Register models
# ─────────────────────────────────────────────────────────────────────────────
exp.add_model("local_level", LocalLevel)

# Grid search over two BSM variance configurations
exp.add_model(
    "bsm",
    BSM,
    params=[
        {"sigma_eps": 0.5, "sigma_eta": 0.1},
        {"sigma_eps": 1.0, "sigma_eta": 0.5},
    ],
)

# UCM with a stochastic annual cycle
exp.add_model("ucm", UCM, params={"cycle_period": 1, "stochastic_cycle": True})

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Attach evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────
exp.add_metric("rmse").add_metric("mae")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Attach the data
# ─────────────────────────────────────────────────────────────────────────────
exp.set_data(
    y,
    train_size=0.80,   # first 80 observations as initial training window
    test_size=1,       # one-step-ahead evaluation
    dates=dates,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Run the experiment
# ─────────────────────────────────────────────────────────────────────────────
result = exp.run(
    cv_strategy=CVStrategy.EXPANDING,
    n_folds=5,
    n_jobs=-1,
    verbose=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Inspect results
# ─────────────────────────────────────────────────────────────────────────────
print("\n─── Per-fold metrics ────────────────────────────────────────────────")
print(exp.results().to_string(index=False))

print("\n─── Model summary (mean ± std across folds) ─────────────────────────")
print(result.summary.to_string())

# Ranked by RMSE
print("\n─── Rankings by RMSE ────────────────────────────────────────────────")
print(result.rank("rmse", lower_is_better=True).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Identify the winner
# ─────────────────────────────────────────────────────────────────────────────
winner = exp.best_model(metric="rmse", lower_is_better=True)
print(f"\nBest model by RMSE: {winner}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Export the report
# ─────────────────────────────────────────────────────────────────────────────
rep = exp.report(output="nile_experiment.html", format="html")
print(f"\nReport written to: nile_experiment.html")
print(f"Elapsed time: {result.elapsed_seconds:.1f} s")
print(f"Folds with errors: {result.errors}")
```

**Expected console output (abbreviated)**

```
Fold 1/5 — local_level    fit OK (0.12 s)
Fold 1/5 — bsm/0          fit OK (0.18 s)
Fold 1/5 — bsm/1          fit OK (0.17 s)
Fold 1/5 — ucm            fit OK (0.21 s)
...

─── Model summary (mean ± std across folds) ─────────────────────────
               rmse_mean  rmse_std  mae_mean  mae_std
local_level       183.41     14.72    152.30    11.05
bsm/0             176.85     13.91    147.62    10.44
bsm/1             181.20     15.33    151.07    11.98
ucm               179.63     14.08    149.88    10.73

Best model by RMSE: bsm/0
```

---

### YAML Configuration

The same experiment can be specified as a YAML configuration file and
executed via the `kalmanbox experiment run` CLI command:

```yaml
experiment:
  name: "Nile River Comparison"
  description: >
    Compares LocalLevel, BSM, and UCM on the Nile annual flow series
    using expanding-window CV with a 5-fold evaluation.
  random_state: 42
  cv_strategy: expanding
  n_folds: 5
  n_jobs: -1

data:
  path: data/nile.csv
  column: flow
  train_size: 0.80
  test_size: 1

models:
  - name: local_level
    class: kalmanbox.LocalLevel

  - name: bsm
    class: kalmanbox.BSM
    params:
      - sigma_eps: 0.5
        sigma_eta: 0.1
      - sigma_eps: 1.0
        sigma_eta: 0.5

  - name: ucm
    class: kalmanbox.UCM
    params:
      cycle_period: 1
      stochastic_cycle: true

metrics:
  - rmse
  - mae

output:
  path: results/nile_experiment.html
  format: html
```

Run it from the terminal:

```bash
kalmanbox experiment run nile_config.yaml
```

!!! tip "CLI flags override YAML"

    Command-line flags take precedence over values in the YAML file:

    ```bash
    # Override n_folds and enable verbose output
    kalmanbox experiment run nile_config.yaml --n-folds 10 --verbose

    # Export to JSON instead of HTML
    kalmanbox experiment run nile_config.yaml --format json --output nile.json
    ```

---

## See Also

- [User Guide: Experiment Framework](../advanced/experiment.md)
- [API: Core (KalmanFilter, RTSSmoother)](core.md)
- [API: Structural Models](structural.md)
- [API: Alternative Filters](filters.md)
- [API: Diagnostics](diagnostics.md)
- [API: Reports](reports.md)
- [Tutorial: Complete Workflow](../tutorials/complete-workflow.md)
