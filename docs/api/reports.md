# Reports API

`kalmanbox.reports`

`kalmanbox.reports` automates the generation of structured, reproducible
model reports. Three report classes cover single-model summaries, model
comparison, and experiment results. Four export formats are supported:
HTML, PDF (via LaTeX), LaTeX, and Markdown. Reports are built by
accumulating sections — text, tables, and figures — and then rendered
through a pluggable exporter.

| Class | Purpose |
|---|---|
| [`Report`](#report) | Base class; accumulates sections and delegates rendering to an exporter |
| [`ModelReport`](#modelreport) | Pre-built single-model report with diagnostics, parameters, and plots |
| [`ComparisonReport`](#comparisonreport) | Side-by-side comparison of multiple fitted models |
| [`ExperimentReport`](#experimentreport) | Report generated from an `ExperimentResult` parameter sweep |

---

## Report

`kalmanbox.reports.Report`

Base report class. Accumulates sections and delegates rendering to an
exporter. All higher-level report classes inherit from `Report` and extend
it with auto-populated sections.

### Constructor

```python
Report(
    title: str,
    author: str | None = None,
    date: str | None = None,
    description: str | None = None,
    template: str | None = None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | `str` | required | Report title shown in the header. |
| `author` | `str \| None` | `None` | Author name for the report header. |
| `date` | `str \| None` | `None` | Report date string. Defaults to today's date in ISO format (`YYYY-MM-DD`) when `None`. |
| `description` | `str \| None` | `None` | Short abstract shown beneath the title in the report header. |
| `template` | `str \| None` | `None` | Path to a custom Jinja2 template file. When `None`, the built-in template is used. |

### Properties

| Property | Type | Description |
|---|---|---|
| `sections` | `list[ReportSection]` | Ordered list of sections added to the report. |
| `n_sections` | `int` | Number of sections currently in the report. |
| `metadata` | `dict[str, str]` | Mapping of header fields: `title`, `author`, `date`, `description`. |

---

### Methods

#### `add_section(title, content, section_type="text")`

```python
def add_section(
    title: str,
    content: str | pd.DataFrame | matplotlib.figure.Figure,
    section_type: str = "text",
    level: int = 2,
) -> None
```

Append a section to the report.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | `str` | required | Section heading text. |
| `content` | `str \| pd.DataFrame \| Figure` | required | Section body. Type must match `section_type`. |
| `section_type` | `str` | `"text"` | Content type: `"text"`, `"table"`, `"figure"`, or `"code"`. |
| `level` | `int` | `2` | Heading level in the rendered output. `2` produces an H2, `3` produces an H3, and so on. |

**Returns** `None`.

---

#### `add_text(title, text, level=2)`

```python
def add_text(
    title: str,
    text: str,
    level: int = 2,
) -> None
```

Shortcut for `add_section` with `section_type="text"`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | `str` | required | Section heading text. |
| `text` | `str` | required | Body text. Markdown and HTML are both accepted and passed through to the exporter. |
| `level` | `int` | `2` | Heading level. |

**Returns** `None`.

---

#### `add_table(title, df, caption=None, level=2)`

```python
def add_table(
    title: str,
    df: pd.DataFrame,
    caption: str | None = None,
    level: int = 2,
) -> None
```

Append a section containing a `DataFrame`. In HTML output the table is
rendered as an HTML `<table>`; in LaTeX output it uses the `booktabs`
package.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | `str` | required | Section heading text. |
| `df` | `pd.DataFrame` | required | Data to tabulate. |
| `caption` | `str \| None` | `None` | Optional caption rendered beneath the table in exported formats. |
| `level` | `int` | `2` | Heading level. |

**Returns** `None`.

---

#### `add_figure(title, fig, caption=None, dpi=150, format="png", level=2)`

```python
def add_figure(
    title: str,
    fig: matplotlib.figure.Figure,
    caption: str | None = None,
    dpi: int = 150,
    format: str = "png",
    level: int = 2,
) -> None
```

Append a matplotlib `Figure` as a section. The figure is serialised at
call time; subsequent changes to `fig` have no effect.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | `str` | required | Section heading text. |
| `fig` | `matplotlib.figure.Figure` | required | The figure to embed. |
| `caption` | `str \| None` | `None` | Optional caption rendered beneath the figure. |
| `dpi` | `int` | `150` | Raster resolution for PNG output. |
| `format` | `str` | `"png"` | Serialisation format: `"png"` or `"svg"`. SVG produces vector output that scales without loss of quality. |
| `level` | `int` | `2` | Heading level. |

**Returns** `None`.

---

#### `generate() -> str`

```python
def generate() -> str
```

Render all sections to an HTML string using the active Jinja2 template.
This method is called implicitly by `export()` when `format="html"`.

**Returns** `str` — the rendered HTML document.

---

#### `export(path, format="html")`

```python
def export(
    path: str | Path,
    format: str = "html",
    dpi: int = 150,
    latex_engine: str = "pdflatex",
    open_browser: bool = False,
) -> Path
```

Export the report to a file using the specified format.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | required | Output file path. The file extension is appended automatically if not present. |
| `format` | `str` | `"html"` | Output format: `"html"`, `"pdf"`, `"latex"`, or `"markdown"`. |
| `dpi` | `int` | `150` | Figure resolution for rasterised exports (PNG figures in HTML, PDF, and Markdown). |
| `latex_engine` | `str` | `"pdflatex"` | LaTeX compiler used when `format="pdf"`. Accepted values: `"pdflatex"`, `"xelatex"`, `"lualatex"`. |
| `open_browser` | `bool` | `False` | If `True` and `format="html"`, open the generated file in the system default browser immediately after writing. |

**Returns** `Path` — absolute path to the generated file.

---

### Example

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kalmanbox.reports import Report

# Build a minimal custom report from scratch
report = Report(
    title="Nile River Analysis",
    author="G. Haase",
    description="Annual discharge at Aswan, 1871-1970.",
)

# Add a text section
report.add_text(
    "Background",
    "The Nile data records 100 observations of annual flow in 10^8 m³. "
    "A structural break is evident around 1898.",
)

# Add a table section
summary_df = pd.DataFrame({
    "Statistic": ["Mean", "Std Dev", "Min", "Max"],
    "Value":     [919.35, 169.55, 456.0, 1370.0],
})
report.add_table("Descriptive Statistics", summary_df, caption="Units: 10^8 m³/year")

# Add a figure section
fig, ax = plt.subplots(figsize=(10, 4))
rng = np.random.default_rng(0)
ax.plot(range(100), 900 + 50 * rng.standard_normal(100))
ax.set_title("Simulated series")
report.add_figure("Time Series", fig, caption="Illustrative plot.", dpi=120)

# Export to HTML
out_path = report.export("nile_analysis.html", format="html", open_browser=False)
print(f"Report written to: {out_path}")
```

---

## ModelReport

`kalmanbox.reports.ModelReport`

Pre-built report for a single fitted kalmanbox model. Automatically
populates sections: model summary, estimated parameters, information
criteria, innovation diagnostics, and component or state plots. All
sections are built lazily when `build()` or `export()` is called.

### Constructor

```python
ModelReport(
    result: FitResult,
    title: str | None = None,
    author: str | None = None,
    include_plots: bool = True,
    include_diagnostics: bool = True,
    include_components: bool = True,
    significance: float = 0.05,
    ci: float = 0.95,
    figsize: tuple = (10, 4),
    template: str | None = None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `FitResult` | required | The fitted model result returned by `.fit()`. |
| `title` | `str \| None` | `None` | Report title. Defaults to `"{ModelClass} Model Report"` when `None`. |
| `author` | `str \| None` | `None` | Author name for the report header. |
| `include_plots` | `bool` | `True` | Include filtered and smoothed state plots with forecast fan charts. |
| `include_diagnostics` | `bool` | `True` | Include innovation diagnostic tests (Ljung-Box, Jarque-Bera, ARCH-LM) and CUSUM plots. |
| `include_components` | `bool` | `True` | Include component decomposition section for structural models (trend, seasonal, cycle). Ignored for non-structural models. |
| `significance` | `float` | `0.05` | Significance level used to determine pass/fail for diagnostic tests. |
| `ci` | `float` | `0.95` | Credible interval coverage for state and forecast plots. |
| `figsize` | `tuple` | `(10, 4)` | Default figure size `(width, height)` in inches for all auto-generated plots. |
| `template` | `str \| None` | `None` | Path to a custom Jinja2 template file. When `None`, the built-in template is used. |

### Properties

| Property | Type | Description |
|---|---|---|
| `result` | `FitResult` | The underlying fit result passed at construction. |
| `sections` | `list[ReportSection]` | Ordered list of sections, populated after `build()`. |

---

### Methods

#### `build() -> ModelReport`

```python
def build() -> ModelReport
```

Populate all sections from the fit result. This method is idempotent;
calling it more than once rebuilds the sections from scratch.

**Returns** `ModelReport` — returns `self` to allow method chaining.

**Auto-generated sections (in order):**

| # | Section Title | Content |
|---|---|---|
| 1 | **Model Summary** | Model class name, number of observations, number of free parameters, log-likelihood, AIC, BIC. |
| 2 | **Parameter Estimates** | Table with columns: `name`, `estimate`, `std_error`, `t_stat`, `p_value`. |
| 3 | **Information Criteria** | Comparison table: AIC, BIC, HQIC, AICc (smaller is better). |
| 4 | **Innovation Diagnostics** | Table of test results: Ljung-Box Q-statistic, Jarque-Bera normality test, ARCH-LM heteroscedasticity test — each with statistic, p-value, and pass/fail at `significance`. |
| 5 | **State Estimates** | Plot of filtered and smoothed states with `ci`-coverage intervals. |
| 6 | **Component Decomposition** | Component plots for trend, seasonal, and cycle (structural models only; omitted when `include_components=False` or model is not structural). |
| 7 | **Forecast** | 12-step-ahead forecast fan chart with `ci`-coverage prediction intervals. |

!!! info "Selective sections"

    Set `include_plots=False` to suppress sections 5 and 7, or
    `include_diagnostics=False` to suppress section 4 and the CUSUM
    plot. The parameter table (section 2) is always included.

---

#### `export(path, format="html")`

```python
def export(
    path: str | Path,
    format: str = "html",
    dpi: int = 150,
    latex_engine: str = "pdflatex",
    open_browser: bool = False,
) -> Path
```

Build the report (if not already built) then export. Accepts the same
arguments as [`Report.export()`](#exportpath-formathtml).

**Returns** `Path` — absolute path to the generated file.

---

### Example

```python
import pandas as pd
from kalmanbox import BSM
from kalmanbox.datasets import load_dataset
from kalmanbox.reports import ModelReport

# Load airline passenger data and fit a Basic Structural Model
airline = load_dataset("airline")
y = airline["passengers"]

model = BSM(y, seasonal=12)
result = model.fit()

# One-liner: build and export
report = ModelReport(
    result,
    title="Airline BSM Report",
    author="G. Haase",
    include_diagnostics=True,
    include_components=True,
    significance=0.05,
    ci=0.95,
)

html_path = report.export("airline_bsm.html", format="html", open_browser=False)
pdf_path  = report.export("airline_bsm.pdf",  format="pdf",  latex_engine="pdflatex")

print(f"HTML report: {html_path}")
print(f"PDF  report: {pdf_path}")
```

!!! tip "Chaining build and export"

    `export()` calls `build()` automatically if the report has not been
    built yet. You can also call `build()` first to inspect `report.sections`
    before writing to disk:

    ```python
    report.build()
    print(f"{report.n_sections} sections generated")
    for s in report.sections:
        print(f"  - {s.title} ({s.section_type})")
    html_path = report.export("airline_bsm.html")
    ```

---

## ComparisonReport

`kalmanbox.reports.ComparisonReport`

Side-by-side comparison of multiple fitted models. Shows a ranking table
sorted by the chosen information criterion, Akaike weights, parameter
comparisons, per-model diagnostic summaries, pairwise likelihood ratio
tests, and optional overlay plots of filtered states.

### Constructor

```python
ComparisonReport(
    results: dict[str, FitResult],
    title: str = "Model Comparison Report",
    author: str | None = None,
    criterion: str = "aic",
    include_plots: bool = True,
    significance: float = 0.05,
    template: str | None = None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `dict[str, FitResult]` | required | Mapping of model name to fitted result. Must contain at least two entries. Model names are used as column headers in comparison tables. |
| `title` | `str` | `"Model Comparison Report"` | Report title. |
| `author` | `str \| None` | `None` | Author name for the report header. |
| `criterion` | `str` | `"aic"` | Primary ranking criterion. Accepted values: `"aic"`, `"bic"`, `"hqic"`, `"aicc"`, `"loglik"`. For `"loglik"` models are ranked in descending order (higher is better); all others rank ascending (lower is better). |
| `include_plots` | `bool` | `True` | Include an overlay plot of filtered states from all models drawn on shared axes. |
| `significance` | `float` | `0.05` | Significance level for the diagnostic pass/fail column and pairwise LRT decisions. |
| `template` | `str \| None` | `None` | Path to a custom Jinja2 template file. |

!!! warning "Mixing model types"

    `ComparisonReport` works with any mix of kalmanbox model types, but
    pairwise likelihood ratio tests are only generated for pairs where
    one model is a restricted version of the other (nested models).
    Non-nested pairs are listed in the comparison table without an LRT
    entry.

---

### Methods

#### `build() -> ComparisonReport`

```python
def build() -> ComparisonReport
```

Populate all comparison sections. Returns `self` for chaining.

**Auto-generated sections (in order):**

| # | Section Title | Content |
|---|---|---|
| 1 | **Model Rankings** | Table of all models sorted by `criterion`. Columns: model name, log-likelihood, n_params, AIC, BIC, HQIC, AICc, Δ-criterion (difference from the best model), Akaike weight. |
| 2 | **Parameter Comparison** | Parameters shared across two or more models shown side-by-side. Missing parameters for a given model are shown as `—`. |
| 3 | **Diagnostic Comparison** | Per-model pass/fail table for normality (Jarque-Bera), independence (Ljung-Box), and heteroscedasticity (ARCH-LM) at `significance`. |
| 4 | **Likelihood Ratio Tests** | Pairwise LRT table for all detected nested model pairs: test statistic, degrees of freedom, p-value, and decision at `significance`. |
| 5 | **State Overlay Plot** | Filtered state means from all models plotted on shared axes (omitted when `include_plots=False`). |

---

#### `export(path, format="html")`

```python
def export(
    path: str | Path,
    format: str = "html",
    dpi: int = 150,
    latex_engine: str = "pdflatex",
    open_browser: bool = False,
) -> Path
```

Build (if not already built) then export. Same signature as
[`Report.export()`](#exportpath-formathtml).

**Returns** `Path` — absolute path to the generated file.

---

### Example

```python
from kalmanbox import LocalLevel, BSM, UCM
from kalmanbox.datasets import load_dataset
from kalmanbox.reports import ComparisonReport

nile = load_dataset("nile")
y    = nile["volume"]

# Fit three competing models
ll_result  = LocalLevel(y).fit()
bsm_result = BSM(y).fit()
ucm_result = UCM(y, level="stochastic", seasonal=None).fit()

report = ComparisonReport(
    results={
        "LocalLevel": ll_result,
        "BSM":        bsm_result,
        "UCM":        ucm_result,
    },
    title="Nile Model Comparison",
    author="G. Haase",
    criterion="aic",
    include_plots=True,
    significance=0.05,
)

out = report.export("nile_comparison.html", format="html")
print(f"Comparison report: {out}")

# Inspect rankings programmatically before exporting
report.build()
rankings = report.sections[0]   # first section is the rankings table
print(rankings.content)         # pandas DataFrame
```

!!! info "Akaike weights"

    The Akaike weight for model $i$ is:

    $$w_i = \frac{\exp(-\Delta_i / 2)}{\sum_j \exp(-\Delta_j / 2)}$$

    where $\Delta_i = \text{AIC}_i - \min_j \text{AIC}_j$. Weights sum to
    one and can be interpreted as the probability that model $i$ is the
    best approximating model in the candidate set.

---

## ExperimentReport

`kalmanbox.reports.ExperimentReport`

Report generated from an `ExperimentResult` produced by
`kalmanbox.experiment.ExperimentRunner`. Shows parameter sweep
configurations, cross-validation scores, result rankings, hyperparameter
importance, and heatmaps for two-dimensional sweeps.

### Constructor

```python
ExperimentReport(
    experiment_result: ExperimentResult,
    title: str = "Experiment Report",
    author: str | None = None,
    include_heatmaps: bool = True,
    top_n: int = 5,
    template: str | None = None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_result` | `ExperimentResult` | required | The result object returned by `ExperimentRunner.run()`. |
| `title` | `str` | `"Experiment Report"` | Report title. |
| `author` | `str \| None` | `None` | Author name for the report header. |
| `include_heatmaps` | `bool` | `True` | Include score heatmaps when the experiment sweeps over exactly two hyperparameters. For sweeps with more than two dimensions, heatmaps are shown for the two most important hyperparameters. |
| `top_n` | `int` | `5` | Number of top-scoring configurations to highlight in the rankings section and detail tables. |
| `template` | `str \| None` | `None` | Path to a custom Jinja2 template file. |

---

### Methods

#### `build() -> ExperimentReport`

```python
def build() -> ExperimentReport
```

Populate all experiment sections. Returns `self` for chaining.

**Auto-generated sections (in order):**

| # | Section Title | Content |
|---|---|---|
| 1 | **Experiment Configuration** | Summary of the sweep grid: hyperparameter names, ranges, number of configurations, scoring metric, cross-validation strategy. |
| 2 | **Result Rankings** | Full table of all configurations ranked by score, with score mean and standard deviation across CV folds. Top `top_n` rows are highlighted. |
| 3 | **Score Distributions** | Box plots of CV scores grouped by each hyperparameter value to reveal sensitivity. |
| 4 | **Hyperparameter Importance** | Permutation importance: each hyperparameter's contribution to score variance (similar to sklearn `permutation_importance`). |
| 5 | **Best Configuration Details** | Parameter table and fit summary for the single best-scoring configuration. |
| 6 | **Score Heatmaps** | 2-D heatmap(s) of mean CV score across the two most important hyperparameters (omitted when `include_heatmaps=False`). |

---

#### `export(path, format="html")`

```python
def export(
    path: str | Path,
    format: str = "html",
    dpi: int = 150,
    latex_engine: str = "pdflatex",
    open_browser: bool = False,
) -> Path
```

Build and export. Same signature as [`Report.export()`](#exportpath-formathtml).

**Returns** `Path` — absolute path to the generated file.

---

### Example

```python
from kalmanbox.experiment import ExperimentRunner
from kalmanbox.reports import ExperimentReport

runner = ExperimentRunner(
    model_class="BSM",
    dataset="airline",
    param_grid={
        "seasonal": [4, 12],
        "level_var_init": [0.01, 0.1, 1.0],
    },
    scoring="aic",
    cv="rolling",
    n_splits=5,
)
exp_result = runner.run()

report = ExperimentReport(
    exp_result,
    title="BSM Hyperparameter Sweep — Airline",
    author="G. Haase",
    include_heatmaps=True,
    top_n=5,
)
out = report.export("bsm_experiment.html", format="html")
print(f"Experiment report: {out}")
```

---

## Export Formats

### HTML Exporter

`kalmanbox.reports.exporters.HTMLExporter`

Produces a standalone HTML file with inline CSS and base64-encoded figures.
The output is self-contained: no external assets are required to view it.

#### Constructor

```python
HTMLExporter(
    include_mathjax: bool = True,
    mathjax_cdn: str = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
    css: str | None = None,
    dark_mode: bool = False,
    toc: bool = True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_mathjax` | `bool` | `True` | Load MathJax from CDN for client-side equation rendering. When `False`, LaTeX math is passed through as raw text. |
| `mathjax_cdn` | `str` | `"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"` | MathJax CDN URL. Override to pin a specific version or use a self-hosted copy. |
| `css` | `str \| None` | `None` | Custom CSS string injected into the `<style>` block. Appended after the built-in styles so it takes precedence. |
| `dark_mode` | `bool` | `False` | Apply a dark colour scheme to the built-in CSS. |
| `toc` | `bool` | `True` | Auto-generate a table-of-contents sidebar from section headings. |

---

### LaTeX Exporter

`kalmanbox.reports.exporters.LaTeXExporter`

Generates a compilable `.tex` file. Figures are written to a `figures/`
subdirectory relative to the output path and referenced with
`\includegraphics`.

#### Constructor

```python
LaTeXExporter(
    document_class: str = "article",
    font_size: int = 11,
    paper: str = "a4paper",
    packages: list[str] | None = None,
    preamble: str | None = None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `document_class` | `str` | `"article"` | LaTeX document class: `"article"`, `"report"`, or `"standalone"`. |
| `font_size` | `int` | `11` | Base font size in points. |
| `paper` | `str` | `"a4paper"` | Paper size option passed to `\documentclass`. Common values: `"a4paper"`, `"letterpaper"`. |
| `packages` | `list[str] \| None` | `None` | Additional LaTeX packages to load beyond the defaults (`booktabs`, `graphicx`, `amsmath`, `hyperref`). Each entry is passed to `\usepackage{...}`. |
| `preamble` | `str \| None` | `None` | Custom preamble text appended after all `\usepackage` declarations, before `\begin{document}`. Useful for `\newcommand` definitions or font selection. |

!!! tip "Compiling the `.tex` file"

    After exporting with `format="latex"`, compile with:

    ```bash
    pdflatex report.tex
    # Run twice for correct cross-references and table of contents
    pdflatex report.tex
    ```

    For Unicode support or custom fonts, substitute `xelatex` or `lualatex`:

    ```bash
    xelatex report.tex
    ```

---

### Markdown Exporter

`kalmanbox.reports.exporters.MarkdownExporter`

Generates GitHub-Flavored Markdown. Figures are saved as PNG files in a
`figures/` subdirectory alongside the `.md` file and linked with
`![caption](figures/fig_N.png)`.

#### Constructor

```python
MarkdownExporter(
    math_delimiters: str = "dollars",
    figure_dir: str = "figures",
    dpi: int = 150,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `math_delimiters` | `str` | `"dollars"` | Math fence style: `"dollars"` renders inline math as `$...$` and display math as `$$...$$`, compatible with MkDocs Material and GitHub. `"brackets"` uses `\(...\)` and `\[...\]`. |
| `figure_dir` | `str` | `"figures"` | Subdirectory name for saved figure PNG files, relative to the output `.md` file location. |
| `dpi` | `int` | `150` | Resolution for saved PNG figures. |

---

### PDF Export (via LaTeX)

PDF export requires a working TeX distribution on the system `PATH`.
`export(format="pdf")` calls `LaTeXExporter` to generate a `.tex` file,
then runs `latex_engine` as a subprocess on the generated file. The
intermediate `.tex` and `figures/` directory are written to a temporary
location and cleaned up after a successful PDF build.

!!! warning "TeX distribution required"

    PDF export will raise `kalmanbox.reports.ExportError` if the chosen
    `latex_engine` (`pdflatex`, `xelatex`, or `lualatex`) is not found on
    the system `PATH`. Install [TeX Live](https://tug.org/texlive/) (Linux /
    macOS) or [MiKTeX](https://miktex.org/) (Windows) before using
    `format="pdf"`. To check availability:

    ```bash
    pdflatex --version
    ```

    Alternatively, export to `"latex"` first, inspect the `.tex` file, and
    compile manually.

---

## Templates

Reports use [Jinja2](https://jinja.palletsprojects.com/) for HTML
rendering. Pass `template="/path/to/my_template.html"` to any report
constructor to override the built-in layout while preserving all section
content.

### Template Variables

The following variables are available inside a custom template:

| Variable | Type | Description |
|---|---|---|
| `{{ title }}` | `str` | Report title. |
| `{{ author }}` | `str` | Author name (empty string if not set). |
| `{{ date }}` | `str` | Report date string. |
| `{{ description }}` | `str` | Report description / abstract. |
| `{{ metadata }}` | `dict` | All four header fields as a single dictionary. |
| `{{ sections }}` | `list` | Ordered list of `ReportSection` objects. |

### Section Object Attributes

Each element in `{{ sections }}` exposes:

| Attribute | Type | Description |
|---|---|---|
| `title` | `str` | Section heading text. |
| `content_html` | `str` | Rendered HTML content for this section. |
| `section_type` | `str` | One of `"text"`, `"table"`, `"figure"`, `"code"`. |
| `level` | `int` | Heading level (2, 3, …). |

### Custom Template Example

```python
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset
from kalmanbox.reports import ModelReport

# Minimal custom Jinja2 template stored at custom_report.html:
# ---
# <!DOCTYPE html>
# <html>
# <head><title>{{ title }}</title></head>
# <body>
#   <h1>{{ title }}</h1>
#   <p><em>{{ author }} — {{ date }}</em></p>
#   {% for section in sections %}
#     <h{{ section.level }}>{{ section.title }}</h{{ section.level }}>
#     {{ section.content_html }}
#   {% endfor %}
# </body>
# </html>
# ---

y      = load_dataset("nile")["volume"]
result = LocalLevel(y).fit()

report = ModelReport(
    result,
    title="Nile — Custom Template",
    template="/path/to/custom_report.html",
)
report.export("nile_custom.html")
```

!!! info "Template inheritance"

    Custom templates can extend the built-in base template using standard
    Jinja2 `{% extends %}` / `{% block %}` syntax. The built-in template is
    available for reference at
    `kalmanbox/reports/templates/report_base.html` in the source tree.

---

## Complete Reporting Workflow

End-to-end example: load data, fit three competing models, run a
`ComparisonReport` to identify the best model, then generate a full
`ModelReport` for the winner in both HTML and PDF.

```python
from pathlib import Path
import pandas as pd
from kalmanbox import LocalLevel, BSM, UCM
from kalmanbox.datasets import load_dataset
from kalmanbox.reports import ComparisonReport, ModelReport

# ── 1. Load airline passenger data ───────────────────────────────────────────
airline = load_dataset("airline")
y       = airline["passengers"]          # 144 monthly observations, 1949-1960

# ── 2. Fit LocalLevel, BSM, and UCM ──────────────────────────────────────────
ll_result  = LocalLevel(y).fit()

bsm_result = BSM(y, seasonal=12).fit()

ucm_result = UCM(
    y,
    level="stochastic",
    slope=True,
    seasonal=12,
    seasonal_type="trigonometric",
).fit()

results = {
    "LocalLevel": ll_result,
    "BSM":        bsm_result,
    "UCM":        ucm_result,
}

# ── 3. ComparisonReport — identify best model ─────────────────────────────────
comparison = ComparisonReport(
    results=results,
    title="Airline Model Comparison",
    author="G. Haase",
    criterion="aic",
    include_plots=True,
    significance=0.05,
)
comparison.build()

# Inspect the rankings table programmatically
rankings_df = comparison.sections[0].content      # pandas DataFrame
best_name   = rankings_df.index[0]                # name of the top-ranked model
best_result = results[best_name]

print(f"Best model: {best_name}")
print(rankings_df[["AIC", "BIC", "delta_aic", "weight"]].to_string())

comp_path = comparison.export(
    "airline_comparison.html",
    format="html",
    open_browser=False,
)
print(f"Comparison report: {comp_path.resolve()}")

# ── 4. ModelReport for the best model ────────────────────────────────────────
model_report = ModelReport(
    result=best_result,
    title=f"Airline {best_name} — Full Report",
    author="G. Haase",
    include_plots=True,
    include_diagnostics=True,
    include_components=True,
    significance=0.05,
    ci=0.95,
    figsize=(10, 4),
)
model_report.build()

html_path = model_report.export(
    f"airline_{best_name.lower()}.html",
    format="html",
    dpi=150,
    open_browser=False,
)
pdf_path = model_report.export(
    f"airline_{best_name.lower()}.pdf",
    format="pdf",
    latex_engine="pdflatex",
)

# ── 5. Print absolute paths of generated files ───────────────────────────────
print(f"\nGenerated files:")
print(f"  Comparison HTML : {comp_path.resolve()}")
print(f"  Model HTML      : {html_path.resolve()}")
print(f"  Model PDF       : {pdf_path.resolve()}")
```

!!! tip "Reproducibility"

    Pin the kalmanbox version and record it in your report description:

    ```python
    import kalmanbox
    report = ModelReport(
        result,
        description=f"Generated with kalmanbox {kalmanbox.__version__}",
    )
    ```

---

## See Also

- [User Guide: Diagnostics](../user-guide/diagnostics/index.md)
- [User Guide: Visualization](../tutorials/visualization/index.md)
- [API: Core (KalmanFilter)](core.md)
- [API: Structural Models](structural.md)
- [API: Advanced Models](advanced.md)
- [Tutorials: Complete Workflow](../tutorials/complete-workflow.md)
- [Theory: MLE](../theory/mle.md)
