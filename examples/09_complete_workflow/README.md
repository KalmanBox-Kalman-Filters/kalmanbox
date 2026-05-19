# Complete Workflow — Examples

End-to-end pipelines that exercise every stage of a state-space analysis with
**kalmanbox**: model specification, MLE estimation, Kalman filtering, smoothing,
forecasting, residual diagnostics, and report generation. The goal is to show
how the components introduced in FASES 1–8 fit together in practice.

## Pipeline

```
data ──▶ model spec ──▶ filter ──▶ smoother ──▶ forecast ──▶ diagnostics ──▶ report
```

Each step is independent and reproducible:

| Step          | What happens                                                           | Module                                  |
|---------------|------------------------------------------------------------------------|-----------------------------------------|
| Data          | Load CSVs from previous FASES (symlinked into `data/`).                | `kalmanbox.datasets`                    |
| Model spec    | Choose local level / BSM / UCM / DFM and fix system matrices.          | `kalmanbox.core.model`                  |
| Filter        | Run the Kalman filter to obtain `a_t`, `P_t`, `v_t`, `F_t`.            | `kalmanbox.filters.kalman`              |
| Smoother      | Backwards recursion for `α̂_t`, `V_t`, smoothed disturbances.          | `kalmanbox.filters.smoother`            |
| Forecast      | h-step-ahead forecasts with prediction intervals.                      | `kalmanbox.forecasting`                 |
| Diagnostics   | Standardised residuals, Ljung–Box, JB, CUSUM, auxiliary residuals.     | `kalmanbox.diagnostics`                 |
| Report        | Comparison tables, accuracy metrics, dashboards, LaTeX export.         | `data/report_utils.py` (this folder)    |

## Workflows

Two notebooks (built in subphases F9.2 and F9.3) walk through the pipeline end
to end:

1. **Univariate workflow** — a single time series taken through the full
   pipeline. Compares competing specifications (local level vs. local linear
   trend vs. BSM), selects by AIC/BIC, runs forecast diagnostics, and
   produces a one-page summary report.
2. **Multivariate workflow** — a panel of macro indicators handled with a
   dynamic factor model (DFM). Demonstrates nowcasting, variance
   decomposition by factor, and a multi-series report.

The remaining subphases add validation against R (KFAS, dlm, MARSS) and Stata
(`sspace`, `ucm`, `dfactor`) and a final reproducibility check that re-runs
all notebooks headless.

## Prerequisites

All previous FASES (1–8) must be completed: this folder reuses their datasets
and assumes their solution code is importable.

```bash
pip install kalmanbox jupyter matplotlib statsmodels scipy pandas
```

For cross-validation:

```bash
# R
Rscript -e 'install.packages(c("KFAS", "dlm", "MARSS"))'

# Stata 17+ with built-in sspace / ucm / dfactor
```

## Folder layout

```
examples/09_complete_workflow/
├── data/
│   ├── report_utils.py          # report-generation helpers (this folder)
│   ├── nile.csv                 # symlink → 01_local_level_trend
│   ├── airline.csv              # symlink → 01_local_level_trend
│   ├── uk_drivers.csv           # symlink → 01_local_level_trend
│   ├── brazil_gdp.csv           # symlink → 02_structural_models
│   ├── brazil_ipca.csv          # symlink → 02_structural_models
│   ├── uk_gas.csv               # symlink → 02_structural_models
│   ├── us_macro_panel.csv       # symlink → 04_dynamic_factors
│   ├── mixed_freq_macro.csv     # symlink → 04_dynamic_factors
│   ├── us_inflation_unemployment.csv  # symlink → 05_tvp
│   ├── lorenz63.csv             # symlink → 06_advanced_filters
│   ├── pendulum.csv             # symlink → 06_advanced_filters
│   ├── target_tracking.csv      # symlink → 06_advanced_filters
│   ├── airline_missing.csv      # symlink → 08_diagnostics
│   └── nile_outliers.csv        # symlink → 08_diagnostics
├── solutions/                   # reference implementations (subphases F9.2, F9.3)
├── validation/
│   ├── R/                       # KFAS / dlm / MARSS scripts
│   └── stata/                   # sspace / ucm / dfactor do-files
├── output/                      # generated reports (HTML, LaTeX, PNG)
└── README.md
```

## Running

From the repository root:

```bash
# 1. Univariate workflow notebook (FASE 9.2)
jupyter nbconvert --to notebook --execute \
    examples/09_complete_workflow/01_univariate_workflow.ipynb

# 2. Multivariate workflow notebook (FASE 9.3)
jupyter nbconvert --to notebook --execute \
    examples/09_complete_workflow/02_multivariate_workflow.ipynb

# 3. Cross-validation (FASE 9.4)
Rscript examples/09_complete_workflow/validation/R/run_all.R
stata -b do examples/09_complete_workflow/validation/stata/run_all.do
```

Reports, figures, and LaTeX tables are written to `output/`.

## Report utilities

The helper module `data/report_utils.py` exposes four entry points used by the
notebooks; each is documented in detail in its docstring.

| Function                          | Purpose                                                       |
|-----------------------------------|---------------------------------------------------------------|
| `generate_model_comparison_table` | Markdown / HTML / LaTeX table of log-likelihood, AIC, BIC, HQIC. |
| `generate_forecast_report`        | Out-of-sample RMSE / MAE / MAPE / bias (and optional MASE).   |
| `export_results_to_latex`         | Write any dict / DataFrame as a LaTeX `tabular` to disk.      |
| `create_summary_dashboard`        | Four-panel matplotlib figure (fit, residuals, ACF, distribution). |

## References

- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods*, 2nd ed. Oxford University Press.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice*, 3rd ed. OTexts.
