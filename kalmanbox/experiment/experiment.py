"""KalmanExperiment: orchestrator for model comparison and validation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kalmanbox._logging import get_logger
from kalmanbox.experiment.comparison import ComparisonResult, ValidationResult

logger = get_logger("experiment")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "LocalLevel": ("kalmanbox.models.local_level", "LocalLevel"),
    "LocalLinearTrend": (
        "kalmanbox.models.local_linear_trend",
        "LocalLinearTrend",
    ),
    "BSM": ("kalmanbox.models.bsm", "BasicStructuralModel"),
    "UCM": ("kalmanbox.models.ucm", "UnobservedComponents"),
    "ARIMA_SSM": ("kalmanbox.models.arima_ssm", "ARIMA_SSM"),
    "TVP": ("kalmanbox.models.tvp", "TimeVaryingParameters"),
}


def _get_model_class(name: str) -> type:
    """Import and return model class by name."""
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        msg = f"Unknown model '{name}'. Available: {available}"
        raise ValueError(msg)
    module_path, class_name = _MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class KalmanExperiment:
    """Orchestrator for fitting, comparing, and validating state-space models.

    Parameters
    ----------
    y : array-like
        Time series data (1D numeric array or pandas Series).

    Examples
    --------
    >>> from kalmanbox.experiment import KalmanExperiment
    >>> from kalmanbox.datasets import load_dataset
    >>> data = load_dataset('nile')
    >>> exp = KalmanExperiment(data['volume'])
    >>> exp.fit_all_models([('LocalLevel', {})])
    >>> comparison = exp.compare_models()
    >>> print(comparison.best_model())
    """

    def __init__(self, y: NDArray[np.float64] | pd.Series) -> None:
        if isinstance(y, pd.Series):
            self._y = y.to_numpy(dtype=np.float64)
        else:
            self._y = np.asarray(y, dtype=np.float64)

        self._results: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._fitted: bool = False

    @property
    def y(self) -> NDArray[np.float64]:
        """Return the time series data."""
        return self._y

    @property
    def results(self) -> dict[str, Any]:
        """Return fitted results dict."""
        return self._results

    def fit_all_models(
        self,
        model_specs: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Fit multiple models to the data.

        Parameters
        ----------
        model_specs : list[tuple[str, dict]]
            List of (model_name, kwargs) pairs.
            model_name must be a key in the model registry.
            kwargs are passed to the model constructor.

        Examples
        --------
        >>> exp.fit_all_models([
        ...     ('LocalLevel', {}),
        ...     ('BSM', {'seasonal_period': 12}),
        ... ])
        """
        self._results.clear()
        self._models.clear()

        for model_name, kwargs in model_specs:
            logger.info("Fitting model: %s", model_name)
            try:
                model_class = _get_model_class(model_name)
                model = model_class(self._y, **kwargs)
                result = model.fit()
                self._models[model_name] = model
                self._results[model_name] = result
                logger.info(
                    "Model %s fitted: loglike=%.4f, aic=%.4f",
                    model_name,
                    result.loglike,
                    result.aic,
                )
            except Exception as e:
                logger.error("Failed to fit model %s: %s", model_name, e)
                raise

        self._fitted = True

    def compare_models(
        self,
        criteria: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare fitted models by information criteria.

        Parameters
        ----------
        criteria : list[str] or None
            List of criteria to compare. Default: ['aic', 'bic', 'loglike'].

        Returns
        -------
        ComparisonResult
            Object with ranking, best_model, to_dataframe methods.

        Raises
        ------
        RuntimeError
            If no models have been fitted yet.
        """
        if not self._fitted or not self._results:
            msg = "No models fitted. Call fit_all_models() first."
            raise RuntimeError(msg)

        if criteria is None:
            criteria = ["aic", "bic", "loglike"]

        model_names = list(self._results.keys())
        metrics: dict[str, dict[str, float]] = {}

        for name in model_names:
            result = self._results[name]
            model_metrics: dict[str, float] = {}
            for crit in criteria:
                if crit == "aic":
                    model_metrics["aic"] = float(result.aic)
                elif crit == "bic":
                    model_metrics["bic"] = float(result.bic)
                elif crit == "loglike":
                    model_metrics["loglike"] = float(result.loglike)
                else:
                    logger.warning("Unknown criterion: %s", crit)
            metrics[name] = model_metrics

        return ComparisonResult(
            model_names=model_names,
            metrics=metrics,
            criteria=criteria,
        )

    def validate_model(
        self,
        model_name: str,
        test_size: int = 24,
        horizon: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate a model with out-of-sample forecasting.

        Splits data into train/test, fits on train, forecasts test_size steps,
        and computes accuracy metrics.

        Parameters
        ----------
        model_name : str
            Name of the model to validate.
        test_size : int
            Number of observations to hold out for testing.
        horizon : int or None
            Forecast horizon. Default: test_size.
        model_kwargs : dict or None
            Additional kwargs for model constructor.

        Returns
        -------
        ValidationResult
            Object with rmse, mae, mape, coverage methods.

        Raises
        ------
        ValueError
            If test_size >= len(y).
        """
        if horizon is None:
            horizon = test_size

        n = len(self._y)
        if test_size >= n:
            msg = f"test_size ({test_size}) must be < n_obs ({n})"
            raise ValueError(msg)

        train_size = n - test_size
        y_train = self._y[:train_size]
        y_test = self._y[train_size : train_size + horizon]

        if len(y_test) < horizon:
            horizon = len(y_test)

        if model_kwargs is None:
            model_kwargs = {}

        model_class = _get_model_class(model_name)
        model = model_class(y_train, **model_kwargs)
        result = model.fit()

        fc = result.forecast(steps=horizon)

        # forecast returns 2D arrays (steps, k_endog); flatten for univariate
        fc_mean = np.asarray(fc["mean"][:horizon], dtype=np.float64).ravel()
        fc_lower = np.asarray(fc["lower"][:horizon], dtype=np.float64).ravel()
        fc_upper = np.asarray(fc["upper"][:horizon], dtype=np.float64).ravel()

        return ValidationResult(
            model_name=model_name,
            y_test=y_test,
            y_forecast=fc_mean,
            y_lower=fc_lower,
            y_upper=fc_upper,
            train_size=train_size,
            test_size=test_size,
            horizon=horizon,
        )

    def save_master_report(
        self,
        output_path: str | Path,
        theme: str = "professional",
    ) -> None:
        """Save a consolidated HTML report with all fitted model results.

        Parameters
        ----------
        output_path : str or Path
            Path for the output HTML file.
        theme : str
            Report theme. Currently only 'professional' is supported.

        Raises
        ------
        RuntimeError
            If no models have been fitted yet.
        """
        if not self._fitted or not self._results:
            msg = "No models fitted. Call fit_all_models() first."
            raise RuntimeError(msg)

        output_path = Path(output_path)

        # Build comparison table
        comparison = self.compare_models()
        comp_df = comparison.to_dataframe()

        # Build HTML
        html_parts: list[str] = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html><head>")
        html_parts.append("<meta charset='utf-8'>")
        html_parts.append("<title>KalmanBox Master Report</title>")
        html_parts.append("<style>")
        html_parts.append(
            """
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px;
                   background: #fafafa; color: #333; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db;
                 padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 10px 14px;
                     text-align: right; }
            th { background: #3498db; color: white; }
            tr:nth-child(even) { background: #f2f2f2; }
            tr:hover { background: #e8f4f8; }
            .best { background: #d5f5e3 !important; font-weight: bold; }
            .section { background: white; padding: 20px; border-radius: 8px;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                       margin: 20px 0; }
            .param-table td:first-child { text-align: left;
                                          font-family: monospace; }
        """
        )
        html_parts.append("</style></head><body>")

        html_parts.append("<h1>KalmanBox Master Report</h1>")
        html_parts.append(f"<p>Number of observations: {len(self._y)}</p>")
        html_parts.append(f"<p>Models fitted: {len(self._results)}</p>")

        # Comparison section
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Model Comparison</h2>")
        html_parts.append(comp_df.to_html(classes="param-table", float_format="%.4f"))
        best = comparison.best_model()
        html_parts.append(f"<p><strong>Best model (AIC):</strong> {best}</p>")
        html_parts.append("</div>")

        # Individual model sections
        for name, result in self._results.items():
            html_parts.append("<div class='section'>")
            html_parts.append(f"<h2>Model: {name}</h2>")
            html_parts.append("<table class='param-table'>")
            html_parts.append("<tr><th>Parameter</th><th>Value</th></tr>")
            for pname, pval in zip(result.param_names, result.params, strict=True):
                html_parts.append(f"<tr><td>{pname}</td><td>{pval:.6f}</td></tr>")
            html_parts.append("</table>")
            html_parts.append(
                f"<p>LogLike: {result.loglike:.4f} | "
                f"AIC: {result.aic:.4f} | "
                f"BIC: {result.bic:.4f}</p>"
            )
            html_parts.append("</div>")

        html_parts.append("</body></html>")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(html_parts), encoding="utf-8")
        logger.info("Master report saved to %s", output_path)
