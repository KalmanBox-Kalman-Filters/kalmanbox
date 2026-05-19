# CLI Reference

`kalmanbox` provides a full-featured command-line interface for fitting
models, generating forecasts, running diagnostics, comparing models,
creating reports, and plotting — all without writing Python code. The CLI
is installed automatically with the package and is configured either
through command-line flags or a YAML configuration file.

```bash
kalmanbox [COMMAND] [OPTIONS] [ARGS]
```

**Global options** available to every command:

| Option | Description |
|---|---|
| `--version` | Print kalmanbox version and exit. |
| `--help` | Show help message and exit. |
| `--config FILE` | Load default option values from a YAML config file (overridden by explicit flags). |
| `--quiet` | Suppress progress output. |
| `--log-level LEVEL` | Logging verbosity: `debug`, `info`, `warning`, `error`. Default `info`. |

**Command summary:**

| Command | Description |
|---|---|
| [`fit`](#kalmanbox-fit) | Estimate a state-space model from a CSV file |
| [`forecast`](#kalmanbox-forecast) | Generate point and interval forecasts |
| [`diagnose`](#kalmanbox-diagnose) | Run diagnostic tests on a fitted model |
| [`experiment`](#kalmanbox-experiment) | Run a multi-model comparison experiment |
| [`report`](#kalmanbox-report) | Generate a formatted model report |
| [`plot`](#kalmanbox-plot) | Produce diagnostic and state plots |

---

## `kalmanbox fit`

Fit a state-space model to an observed time series stored in a CSV file
and save the fitted model and results to disk.

```bash
kalmanbox fit [OPTIONS]
```

### Options

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model` | choice | Yes | — | Model type. Choices: `local-level`, `local-linear-trend`, `bsm`, `ucm`, `dfm`, `tvp`, `arima-ssm`. |
| `--data` | path | Yes | — | Path to the input CSV file. Must contain at least one numeric column. |
| `--target` | str | No | first numeric column | Column name to use as the observation series. |
| `--date-col` | str | No | auto-detect | Column containing datetime values. Auto-detected if the first column is parseable as dates. |
| `--seasonal-period` | int | No | `None` | Seasonal period *s* (required for `bsm`, `ucm`). Common values: 4 (quarterly), 12 (monthly), 7 (daily). |
| `--n-factors` | int | No | `1` | Number of common factors (for `dfm` only). |
| `--output` | path | No | `model.pkl` | Path to write the fitted model file (pickle). |
| `--format` | choice | No | `json` | Output format for the results summary: `json`, `csv`, `html`. |
| `--results` | path | No | `results.{format}` | Path to write the results summary file. |
| `--method` | choice | No | `mle` | Estimation method: `mle`, `em`, `diffuse`. |
| `--maxiter` | int | No | `200` | Maximum iterations for the optimiser. |
| `--seed` | int | No | `None` | Random seed for initialisation. |

!!! info "Model file format"
    The `--output` file is a Python pickle (`.pkl`). It stores the fitted
    model object and can be loaded back with `kalmanbox.load_model(path)`
    in Python or passed to any other CLI command via `--model`.

### Examples

```bash
# Fit a Local Level Model to Nile data
kalmanbox fit --model local-level --data nile.csv --target flow --output nile_ll.pkl

# Fit a BSM with monthly seasonality
kalmanbox fit --model bsm --data airline.csv --target passengers \
    --seasonal-period 12 --output airline_bsm.pkl --format html --results airline_results.html

# Fit a DFM with 3 factors to a multivariate panel
kalmanbox fit --model dfm --data employment.csv --n-factors 3 --output employment_dfm.pkl

# Use EM estimation with a fixed seed for reproducibility
kalmanbox fit --model local-linear-trend --data gdp.csv --target gdp_growth \
    --method em --maxiter 500 --seed 42 --output gdp_llt.pkl
```

---

## `kalmanbox forecast`

Load a previously fitted model and produce *h*-step-ahead point
forecasts with prediction intervals.

```bash
kalmanbox forecast [OPTIONS]
```

### Options

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model` | path | Yes | — | Path to a fitted model file (`.pkl`) produced by `kalmanbox fit`. |
| `--horizon` | int | Yes | — | Number of steps ahead to forecast. |
| `--confidence` | float | No | `0.95` | Prediction interval coverage level. Must be in `(0, 1)`. |
| `--output` | path | No | `forecasts.csv` | Path to write the forecast table (always CSV). |
| `--plot` | flag | No | `False` | Also generate a forecast plot and save it as `forecasts.png` in the same directory as `--output`. |
| `--new-data` | path | No | `None` | Path to new observations to append before forecasting (extends the training set without re-fitting). |

The output CSV contains the following columns:

| Column | Description |
|---|---|
| `date` | Forecast date or step index |
| `forecast` | Point forecast (posterior mean) |
| `lower_{pct}` | Lower prediction interval bound, where `{pct}` is derived from `--confidence` |
| `upper_{pct}` | Upper prediction interval bound |

!!! warning "Model stationarity"
    Forecasts for non-stationary trend models (Local Linear Trend, BSM)
    diverge rapidly. Consider whether a long horizon is meaningful for
    the fitted model class.

### Examples

```bash
# 12-step-ahead forecast with 90% prediction intervals
kalmanbox forecast --model nile_ll.pkl --horizon 12 --confidence 0.90 --output nile_forecast.csv

# Forecast with plot saved alongside the CSV
kalmanbox forecast --model airline_bsm.pkl --horizon 24 --plot --output airline_forecast.csv

# Extend training data before forecasting (no re-fitting)
kalmanbox forecast --model gdp_llt.pkl --horizon 4 \
    --new-data gdp_new.csv --output gdp_forecast.csv
```

---

## `kalmanbox diagnose`

Run a full suite of diagnostic tests on a fitted model and print a
structured summary report to stdout. Optionally write the report to a
file and generate a diagnostic plot.

```bash
kalmanbox diagnose [OPTIONS]
```

### Options

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model` | path | Yes | — | Path to a fitted model file (`.pkl`). |
| `--tests` | str | No | `all` | Comma-separated list of test groups to run. Options: `all`, `innovations`, `cusum`, `residuals`, `consistency`. |
| `--significance` | float | No | `0.05` | Significance level α for all hypothesis tests. |
| `--output` | path | No | `None` | Optional path to write the diagnostic report. Format inferred from extension: `.html`, `.md`, `.json`. |
| `--plot` | flag | No | `False` | Generate a 4-panel diagnostic plot and save alongside `--output`. |

### Test groups

| Group | Tests performed |
|---|---|
| `innovations` | Normality (Jarque-Bera), independence (Ljung-Box), heteroscedasticity tests on standardised innovations |
| `cusum` | CUSUM and CUSUM-of-squares structural stability tests |
| `residuals` | Auxiliary residuals, observation outliers, state disturbance outliers |
| `consistency` | NEES and NIS consistency tests (for simulation experiments where true states are known) |

### Examples

```bash
# Run all diagnostic test groups and print to stdout
kalmanbox diagnose --model nile_ll.pkl

# Run only innovation tests and export an HTML report with plots
kalmanbox diagnose --model nile_ll.pkl --tests innovations \
    --output diagnostics.html --plot

# Run cusum and residual tests with a stricter significance level
kalmanbox diagnose --model airline_bsm.pkl --tests cusum,residuals \
    --significance 0.01

# Full diagnostics exported to JSON
kalmanbox diagnose --model employment_dfm.pkl --tests all --output diagnostics.json
```

---

## `kalmanbox experiment`

Run a multi-model comparison experiment defined in a YAML configuration
file, optionally in parallel, and save a comparison report.

```bash
kalmanbox experiment [OPTIONS]
```

### Options

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--config` | path | Yes | — | Path to the YAML experiment configuration file. |
| `--output` | path | No | `experiment_results.html` | Path to write the experiment report. |
| `--format` | choice | No | `html` | Report format: `html`, `pdf`, `md`, `json`. |
| `--n-jobs` | int | No | `1` | Number of parallel workers. `-1` uses all available cores. |
| `--verbose` | flag | No | `False` | Print fold-by-fold progress to stdout. |
| `--seed` | int | No | `None` | Global random seed. Overrides the value in the config file. |

### YAML configuration

The `--config` file controls every aspect of the experiment. All paths
are resolved relative to the working directory at the time the command
is run.

```yaml
# experiment.yaml — Fully annotated configuration example

experiment:
  name: "GDP Trend Model Comparison"
  description: "Compare Local Level, BSM, and UCM on US GDP growth"
  random_state: 42
  cv_strategy: expanding      # rolling | expanding | fixed
  n_folds: 10
  n_jobs: -1

data:
  path: data/gdp.csv          # path to CSV relative to working directory
  column: gdp_growth          # target observation column
  date_col: date              # datetime column (optional)
  train_size: 0.7             # float [0, 1] or int (number of observations)
  test_size: 4                # forecast horizon per fold

models:
  - name: local_level
    class: kalmanbox.LocalLevel
    params: {}                # empty = use defaults
  - name: bsm_quarterly
    class: kalmanbox.BSM
    params:
      seasonal_period: 4
  - name: ucm_cycle
    class: kalmanbox.UCM
    params:
      seasonal_period: 4
      include_cycle: true
  - name: local_level_grid    # multiple param dicts = grid search
    class: kalmanbox.LocalLevel
    params:
      - sigma_eps: 0.5
        sigma_eta: 0.1
      - sigma_eps: 1.0
        sigma_eta: 0.5

metrics:
  - rmse
  - mae
  - log_score                 # requires predictive variance (built-in)

output:
  path: results/gdp_comparison.html
  format: html
```

!!! tip "Running from CI/CD"
    Pass `--seed` on the command line to override the config file value,
    ensuring deterministic results across environments without editing
    YAML files.

!!! note "Parameter grid search"
    When a model entry has a `params` list (rather than a dict), each
    element defines a separate candidate. Candidates are named
    `{model_name}_0`, `{model_name}_1`, etc. in the comparison table.

### Examples

```bash
# Run experiment from config, write HTML report
kalmanbox experiment --config experiment.yaml --output results/gdp_comparison.html

# Parallel execution with fold-by-fold progress printed
kalmanbox experiment --config experiment.yaml --n-jobs 4 --verbose

# Deterministic CI run with JSON output
kalmanbox experiment --config experiment.yaml --seed 0 \
    --output ci_results.json --format json
```

---

## `kalmanbox report`

Generate a formatted report for a single fitted model, including
parameter estimates, diagnostics, and plots.

```bash
kalmanbox report [OPTIONS]
```

### Options

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model` | path | Yes | — | Path to a fitted model file (`.pkl`). |
| `--format` | choice | No | `html` | Output format: `html`, `pdf`, `md`. |
| `--output` | path | No | `report.{format}` | Path to write the generated report. |
| `--title` | str | No | model class name | Report title shown in the header. |
| `--author` | str | No | `None` | Author name embedded in the report header. |
| `--sections` | str | No | `all` | Comma-separated list of sections to include: `summary`, `parameters`, `diagnostics`, `plots`, `forecasts`. |
| `--forecast-horizon` | int | No | `0` | Append a forecast section with this many steps ahead. `0` omits the forecast section. |
| `--theme` | choice | No | `default` | Plot theme used within the report: `default`, `dark`, `paper`. |

!!! note "PDF output"
    PDF export requires a working LaTeX installation (e.g. TeX Live or
    MiKTeX). If LaTeX is not available, use `--format html` and convert
    with a browser or `wkhtmltopdf`.

### Examples

```bash
# HTML report with all sections
kalmanbox report --model airline_bsm.pkl --format html --output airline_report.html

# PDF for publication with author, custom title, and 12-step forecast
kalmanbox report --model airline_bsm.pkl --format pdf \
    --title "Airline Passenger Model" --author "J. Smith" \
    --forecast-horizon 12 --theme paper --output airline_report.pdf

# Minimal markdown report — summary and parameters only
kalmanbox report --model nile_ll.pkl --format md \
    --sections summary,parameters --output nile_summary.md
```

---

## `kalmanbox plot`

Generate publication-quality plots from a fitted model and save them to
disk as image files.

```bash
kalmanbox plot [OPTIONS]
```

### Options

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model` | path | Yes | — | Path to a fitted model file (`.pkl`). |
| `--type` | str | No | `all` | Comma-separated plot types to generate. See [plot types](#plot-types) below. |
| `--output` | path | No | `plots/` | Directory to write plot files. Created if it does not exist. |
| `--format` | choice | No | `png` | Image format: `png`, `pdf`, `svg`. |
| `--theme` | choice | No | `default` | Visual theme: `default`, `dark`, `paper`. |
| `--dpi` | int | No | `150` | Resolution in DPI. Ignored for `svg` and `pdf` outputs. |
| `--figsize` | str | No | `10x6` | Figure size as `WxH` in inches, e.g. `12x8`. |
| `--horizon` | int | No | `0` | Steps ahead for the forecast overlay plot. `0` disables the forecast overlay. |
| `--ci` | float | No | `0.95` | Credible interval coverage for state and forecast plots. |
| `--state-idx` | str | No | `all` | Comma-separated state indices to include in the states plot, e.g. `0,1`. |

### Plot types

| Type | Output file(s) | Description |
|---|---|---|
| `states` | `filtered_states.{fmt}`, `smoothed_states.{fmt}` | Filtered and smoothed state trajectories with CI bands |
| `components` | `components.{fmt}` | Trend, seasonal, cycle, and irregular decomposition |
| `innovations` | `innovations.{fmt}` | Standardised innovations time plot |
| `diagnostics` | `diagnostic_panel.{fmt}` | 4-panel: innovations, histogram, Q-Q, ACF |
| `forecast` | `forecast.{fmt}` | Point forecast and prediction intervals |
| `factors` | `factors.{fmt}`, `loadings.{fmt}` | Factor trajectories and loading heatmap (DFM only) |
| `tvp` | `tvp_coefficients.{fmt}` | Time-varying coefficient paths (TVP only) |
| `all` | all of the above | Generate every applicable plot type |

!!! note "Model-specific plots"
    `factors` and `tvp` plots are only generated when the fitted model
    supports them (`DynamicFactorModel` and `TVPRegression` respectively).
    Requesting these types for an incompatible model raises an informative
    error and exits with code 2.

### Examples

```bash
# Generate all plots as PNG in the plots/ directory
kalmanbox plot --model airline_bsm.pkl --output plots/

# Publication-ready PDF plots with the paper theme at 300 DPI
kalmanbox plot --model airline_bsm.pkl --type components,forecast \
    --format pdf --theme paper --dpi 300 --output figs/

# 4-panel diagnostic plot only, dark theme, custom figure size
kalmanbox plot --model nile_ll.pkl --type diagnostics \
    --theme dark --figsize 12x8 --output diagnostics/

# Factor trajectory and loading heatmap for a DFM
kalmanbox plot --model employment_dfm.pkl --type factors --output dfm_plots/

# States plot restricted to state indices 0 and 1, with 90% CI bands
kalmanbox plot --model gdp_llt.pkl --type states \
    --state-idx 0,1 --ci 0.90 --output gdp_plots/
```

---

## Global YAML configuration file

Any option accepted by any command can be placed in a YAML config file
and loaded with `--config FILE`. This is useful for setting project-wide
defaults without repeating flags on every invocation.

```yaml
# kalmanbox.yaml — Example global configuration

global:
  quiet: false
  log_level: info

fit:
  method: mle
  maxiter: 500
  seed: 42

forecast:
  confidence: 0.95

diagnose:
  tests: all
  significance: 0.05
  plot: true

report:
  format: html
  theme: default

plot:
  format: png
  dpi: 150
  theme: default
  ci: 0.95
```

Load the config for any command:

```bash
kalmanbox --config kalmanbox.yaml fit --model bsm --data data.csv
```

!!! info "Option precedence"
    Explicit command-line flags always override values from the config
    file, which in turn override built-in defaults.

---

## Shell completion

Enable tab completion for your shell to auto-complete command names,
option names, and file paths:

```bash
# Bash
eval "$(kalmanbox --show-completion bash)"

# Zsh
eval "$(kalmanbox --show-completion zsh)"

# Fish
kalmanbox --show-completion fish | source
```

To make completion persistent, add the relevant line to your shell
startup file (`~/.bashrc`, `~/.zshrc`, or `~/.config/fish/config.fish`).

---

## Exit codes

All `kalmanbox` commands follow a consistent exit-code convention:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | General error (see stderr for details) |
| `2` | Invalid option or argument |
| `3` | Input file not found or unreadable |
| `4` | Model fitting failed to converge |
| `5` | Output directory not writable |
