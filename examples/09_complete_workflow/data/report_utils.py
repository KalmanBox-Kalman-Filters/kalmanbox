"""Report-generation utilities for kalmanbox complete-workflow examples.

The functions in this module support the end-to-end workflows in
``examples/09_complete_workflow/``:

- ``generate_model_comparison_table`` -- format an in-sample model
  comparison (log-likelihood, AIC, BIC, optional HQIC) as a Markdown,
  HTML, or LaTeX table.
- ``generate_forecast_report`` -- compute standard out-of-sample point
  forecast accuracy metrics (RMSE, MAE, MAPE, MASE, bias).
- ``export_results_to_latex`` -- render an arbitrary results dictionary
  as a LaTeX ``tabular`` environment and write it to disk.
- ``create_summary_dashboard`` -- assemble a four-panel summary figure
  (filtered/smoothed level, residuals, ACF of residuals, forecast fan)
  from a fitted model result.

All functions are pure and avoid global state so they can be reused both
inside notebooks and from headless scripts that materialise the
``output/`` directory.

References
----------
- Hyndman, R. J. and Koehler, A. B. (2006). "Another look at measures of
  forecast accuracy." *International Journal of Forecasting*, 22(4),
  679-688.
- Akaike, H. (1974). "A new look at the statistical model
  identification." *IEEE Transactions on Automatic Control*, 19(6),
  716-723.
- Schwarz, G. (1978). "Estimating the dimension of a model." *Annals of
  Statistics*, 6(2), 461-464.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


# --------------------------------------------------------------------------- #
# Model comparison                                                            #
# --------------------------------------------------------------------------- #

_COMPARISON_COLUMNS: tuple[str, ...] = (
    "n_params",
    "loglik",
    "aic",
    "bic",
    "hqic",
)


def _coerce_model_comparison_frame(
    models_dict: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a tidy DataFrame from a mapping ``name -> metrics``.

    Missing metrics are filled with ``np.nan`` and the column order is
    fixed so that downstream renderers produce stable output regardless
    of insertion order in the input mapping.
    """
    if not models_dict:
        raise ValueError("models_dict must contain at least one model.")

    rows: list[dict[str, Any]] = []
    for name, metrics in models_dict.items():
        row: dict[str, Any] = {"model": str(name)}
        for col in _COMPARISON_COLUMNS:
            value = metrics.get(col, np.nan)
            row[col] = np.nan if value is None else float(value)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df[list(_COMPARISON_COLUMNS)]


def generate_model_comparison_table(
    models_dict: Mapping[str, Mapping[str, Any]],
    fmt: str = "markdown",
    float_format: str = "{:.3f}",
    sort_by: str | None = "aic",
) -> str:
    """Render a comparison table for a collection of fitted models.

    Parameters
    ----------
    models_dict
        Mapping from model name to a dictionary of information criteria
        and the log-likelihood. Recognised keys are
        ``{"n_params", "loglik", "aic", "bic", "hqic"}``; missing keys
        are reported as ``NaN``.
    fmt
        Output format. One of ``"markdown"``, ``"html"``, ``"latex"``.
    float_format
        ``str.format`` template used to render floating-point values.
    sort_by
        Column used to sort the table in ascending order. Pass ``None``
        to preserve the order of ``models_dict``.

    Returns
    -------
    str
        The formatted table.
    """
    df = _coerce_model_comparison_frame(models_dict)

    if sort_by is not None:
        if sort_by not in df.columns:
            raise KeyError(
                f"sort_by={sort_by!r} not in {list(df.columns)}."
            )
        df = df.sort_values(sort_by, ascending=True, na_position="last")

    def formatter(x: float) -> str:
        return "" if pd.isna(x) else float_format.format(x)

    fmt_lower = fmt.lower()
    if fmt_lower == "markdown":
        floatfmt = float_format.replace("{:", "").replace("}", "")
        return df.to_markdown(floatfmt=floatfmt)
    if fmt_lower == "html":
        return df.to_html(float_format=formatter, na_rep="")
    if fmt_lower == "latex":
        return _df_to_latex_tabular(df, float_format=float_format)

    raise ValueError(
        f"Unsupported fmt={fmt!r}; expected 'markdown', 'html' or 'latex'."
    )


def _df_to_latex_tabular(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Render ``df`` as a self-contained LaTeX ``tabular`` block.

    Implemented locally to avoid the optional ``jinja2`` dependency
    that recent pandas versions impose on ``DataFrame.to_latex``.
    """
    n_cols = df.shape[1] + 1
    align = "l" + "r" * (n_cols - 1)
    header_cells = [str(df.index.name or "")] + [str(c) for c in df.columns]
    header = " & ".join(f"\\textbf{{{cell}}}" for cell in header_cells) + " \\\\"

    body_lines: list[str] = []
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for value in row.tolist():
            cells.append(_format_value(value, float_format))
        body_lines.append(" & ".join(cells) + " \\\\")

    return "\n".join(
        [
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            header,
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )


# --------------------------------------------------------------------------- #
# Forecast accuracy                                                           #
# --------------------------------------------------------------------------- #


def _as_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def generate_forecast_report(
    forecast_results: ArrayLike | Mapping[str, ArrayLike],
    actual: ArrayLike | None = None,
    seasonal_period: int | None = None,
    in_sample: ArrayLike | None = None,
) -> pd.DataFrame:
    """Compute out-of-sample forecast accuracy statistics.

    Parameters
    ----------
    forecast_results
        Either a 1-D array-like of point forecasts or a mapping
        ``{"forecast": ..., "actual": ...}``. When a mapping is given,
        the ``actual`` argument may be omitted.
    actual
        Realised values aligned with ``forecast_results``. Required when
        ``forecast_results`` is an array-like.
    seasonal_period
        If provided together with ``in_sample``, MASE is computed using
        the seasonal naive benchmark with this period.
    in_sample
        In-sample observations used to scale MASE. When ``None`` the
        MASE column is omitted.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns
        ``["n", "rmse", "mae", "mape", "bias"]`` and optionally
        ``"mase"``. MAPE is reported in percent and gracefully reports
        ``NaN`` if any actual value is zero.
    """
    if isinstance(forecast_results, Mapping):
        if "forecast" not in forecast_results:
            raise KeyError(
                "forecast_results mapping must contain key 'forecast'."
            )
        forecast = _as_1d_array(forecast_results["forecast"], "forecast")
        if actual is None:
            if "actual" not in forecast_results:
                raise KeyError(
                    "forecast_results mapping must contain key 'actual' "
                    "when the actual argument is not given."
                )
            actual_arr = _as_1d_array(forecast_results["actual"], "actual")
        else:
            actual_arr = _as_1d_array(actual, "actual")
    else:
        if actual is None:
            raise ValueError(
                "actual must be provided when forecast_results is an array."
            )
        forecast = _as_1d_array(forecast_results, "forecast")
        actual_arr = _as_1d_array(actual, "actual")

    if forecast.shape != actual_arr.shape:
        raise ValueError(
            f"forecast shape {forecast.shape} does not match "
            f"actual shape {actual_arr.shape}."
        )

    mask = ~(np.isnan(forecast) | np.isnan(actual_arr))
    if not mask.any():
        raise ValueError("No overlapping non-NaN forecast/actual pairs.")

    f = forecast[mask]
    a = actual_arr[mask]
    err = a - f

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    with np.errstate(divide="ignore", invalid="ignore"):
        if np.any(a == 0):
            mape = float("nan")
        else:
            mape = float(np.mean(np.abs(err / a)) * 100.0)

    metrics: dict[str, float] = {
        "n": float(mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "bias": bias,
    }

    if in_sample is not None:
        in_sample_arr = _as_1d_array(in_sample, "in_sample")
        period = int(seasonal_period) if seasonal_period is not None else 1
        if period < 1:
            raise ValueError("seasonal_period must be >= 1.")
        if in_sample_arr.size <= period:
            raise ValueError(
                "in_sample must have more observations than seasonal_period "
                "to compute the MASE scale."
            )
        seasonal_diff = np.abs(in_sample_arr[period:] - in_sample_arr[:-period])
        scale = float(np.mean(seasonal_diff))
        metrics["mase"] = float("nan") if scale == 0 else mae / scale

    return pd.DataFrame([metrics])


# --------------------------------------------------------------------------- #
# LaTeX export                                                                #
# --------------------------------------------------------------------------- #


def _format_value(value: Any, float_format: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return float_format.format(float(value))
    return str(value)


def export_results_to_latex(
    results_dict: Mapping[str, Any] | pd.DataFrame,
    filename: str | Path,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "{:.4f}",
    column_names: Sequence[str] | None = None,
) -> str:
    """Export ``results_dict`` to a LaTeX ``tabular`` and write to disk.

    The function accepts either a flat mapping ``{name: value}`` (which
    is rendered as a two-column ``parameter / value`` table) or a
    ``DataFrame``/mapping of mappings (rendered with one column per
    metric). The output is wrapped in a ``table`` environment when
    ``caption`` or ``label`` is supplied.

    Parameters
    ----------
    results_dict
        Results to render.
    filename
        Destination path. Parent directories are created on demand.
    caption, label
        Optional LaTeX caption / label.
    float_format
        ``str.format`` template for floats.
    column_names
        Optional explicit column names overriding the inferred ones.

    Returns
    -------
    str
        The LaTeX source written to ``filename``.
    """
    if isinstance(results_dict, pd.DataFrame):
        df = results_dict.copy()
    else:
        if not results_dict:
            raise ValueError("results_dict is empty.")
        first_value = next(iter(results_dict.values()))
        if isinstance(first_value, Mapping):
            df = pd.DataFrame(dict(results_dict)).T
        else:
            df = pd.DataFrame(
                {"parameter": list(results_dict.keys()),
                 "value": list(results_dict.values())}
            ).set_index("parameter")

    if column_names is not None:
        if len(column_names) != df.shape[1]:
            raise ValueError(
                f"column_names has length {len(column_names)} but the "
                f"table has {df.shape[1]} columns."
            )
        df.columns = list(column_names)

    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = [
            _format_value(v, float_format) for v in formatted[col]
        ]

    n_cols = formatted.shape[1] + 1
    align = "l" + "r" * (n_cols - 1)
    header = " & ".join(
        ["\\textbf{" + str(formatted.index.name or "") + "}"]
        + [f"\\textbf{{{c}}}" for c in formatted.columns]
    )

    body_lines: list[str] = []
    for idx, row in formatted.iterrows():
        cells = [str(idx)] + [str(v) for v in row.tolist()]
        body_lines.append(" & ".join(cells) + " \\\\")

    tabular = "\n".join(
        [
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            header + " \\\\",
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    if caption or label:
        block = ["\\begin{table}[ht]", "\\centering", tabular]
        if caption:
            block.append(f"\\caption{{{caption}}}")
        if label:
            block.append(f"\\label{{{label}}}")
        block.append("\\end{table}")
        latex = "\n".join(block)
    else:
        latex = tabular

    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n", encoding="utf-8")
    return latex


# --------------------------------------------------------------------------- #
# Summary dashboard                                                           #
# --------------------------------------------------------------------------- #


def _autocorr(x: np.ndarray, n_lags: int) -> np.ndarray:
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(n_lags + 1)
    out = np.empty(n_lags + 1)
    out[0] = 1.0
    for k in range(1, n_lags + 1):
        out[k] = np.dot(x[:-k], x[k:]) / denom
    return out


def create_summary_dashboard(
    model_results: Mapping[str, Any],
    figsize: tuple[float, float] = (16, 12),
    title: str | None = None,
):
    """Build a four-panel summary figure for a fitted state-space model.

    Parameters
    ----------
    model_results
        Mapping containing at minimum the key ``"observations"`` (1-D
        array of observed values). Recognised optional keys:

        - ``"filtered_level"`` / ``"smoothed_level"`` -- 1-D arrays
          aligned with ``observations``.
        - ``"residuals"`` -- standardised innovations.
        - ``"forecast"`` -- 1-D point forecast array.
        - ``"forecast_lower"`` / ``"forecast_upper"`` -- forecast bands.
        - ``"index"`` -- shared time index for in-sample data.
        - ``"forecast_index"`` -- index for the forecast horizon.
        - ``"model_name"`` -- string used in panel titles.

    figsize
        Figure size in inches.
    title
        Optional super-title.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    import matplotlib.pyplot as plt

    if "observations" not in model_results:
        raise KeyError("model_results must contain key 'observations'.")

    obs = _as_1d_array(model_results["observations"], "observations")
    n = obs.size
    index = model_results.get("index")
    if index is None:
        index = np.arange(n)
    index = np.asarray(index)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    name = model_results.get("model_name", "Model")
    if title is None:
        title = f"Summary dashboard - {name}"
    fig.suptitle(title, fontsize=14, y=1.0)

    # Panel 1: observations + filtered/smoothed level + forecast
    ax = axes[0, 0]
    ax.plot(index, obs, label="Observed", color="black", lw=1.0)
    if "filtered_level" in model_results:
        ax.plot(
            index,
            _as_1d_array(model_results["filtered_level"], "filtered_level"),
            label="Filtered",
            color="tab:blue",
            lw=1.2,
        )
    if "smoothed_level" in model_results:
        ax.plot(
            index,
            _as_1d_array(model_results["smoothed_level"], "smoothed_level"),
            label="Smoothed",
            color="tab:orange",
            lw=1.2,
        )
    if "forecast" in model_results:
        f = _as_1d_array(model_results["forecast"], "forecast")
        f_index = model_results.get("forecast_index", np.arange(n, n + f.size))
        ax.plot(f_index, f, color="tab:red", lw=1.2, label="Forecast")
        if "forecast_lower" in model_results and "forecast_upper" in model_results:
            ax.fill_between(
                f_index,
                _as_1d_array(model_results["forecast_lower"], "forecast_lower"),
                _as_1d_array(model_results["forecast_upper"], "forecast_upper"),
                color="tab:red",
                alpha=0.2,
                label="Forecast band",
            )
    ax.set_title("Observed vs. fitted")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: standardised residuals
    ax = axes[0, 1]
    if "residuals" in model_results:
        res = _as_1d_array(model_results["residuals"], "residuals")
        res_index = index[-res.size:] if res.size <= n else np.arange(res.size)
        ax.plot(res_index, res, color="tab:purple", lw=1.0)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.axhline(2.0, color="grey", lw=0.7, linestyle="--")
        ax.axhline(-2.0, color="grey", lw=0.7, linestyle="--")
        ax.set_title("Standardised residuals")
    else:
        ax.text(0.5, 0.5, "No residuals provided", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Standardised residuals")
    ax.grid(alpha=0.3)

    # Panel 3: ACF of residuals
    ax = axes[1, 0]
    if "residuals" in model_results:
        res = _as_1d_array(model_results["residuals"], "residuals")
        n_res = res.size
        n_lags = min(20, max(1, n_res // 4))
        acf = _autocorr(res, n_lags)
        lags = np.arange(n_lags + 1)
        ax.vlines(lags, 0, acf, color="tab:blue")
        ax.scatter(lags, acf, color="tab:blue", s=20)
        if n_res > 0:
            band = 1.96 / np.sqrt(n_res)
            ax.axhline(band, color="grey", lw=0.7, linestyle="--")
            ax.axhline(-band, color="grey", lw=0.7, linestyle="--")
        ax.axhline(0.0, color="black", lw=0.5)
        ax.set_title("Residual ACF")
        ax.set_xlabel("Lag")
    else:
        ax.text(0.5, 0.5, "No residuals provided", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Residual ACF")
    ax.grid(alpha=0.3)

    # Panel 4: residual histogram with normal overlay
    ax = axes[1, 1]
    if "residuals" in model_results:
        res = _as_1d_array(model_results["residuals"], "residuals")
        ax.hist(res, bins=max(10, int(np.sqrt(res.size))), density=True,
                color="tab:blue", alpha=0.6, edgecolor="white")
        grid = np.linspace(res.min(), res.max(), 200)
        sigma = float(np.std(res, ddof=1)) if res.size > 1 else 1.0
        mu = float(np.mean(res))
        if sigma > 0:
            pdf = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2.0 * np.pi)
            )
            ax.plot(grid, pdf, color="black", lw=1.2, label="Normal fit")
            ax.legend(fontsize=9)
        ax.set_title("Residual distribution")
    else:
        ax.text(0.5, 0.5, "No residuals provided", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Residual distribution")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
