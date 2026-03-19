"""SSM (State Space Model) data transformer for reports.

Extracts parameters, diagnostics, information criteria, and
component data from StateSpaceResults for template rendering.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any

import numpy as np


class SSMTransformer:
    """Transform StateSpaceResults into template context.

    Extracts:
    - Parameter table (name, value, SE, t-stat, p-value, significance)
    - Information criteria (AIC, BIC, HQIC, loglike)
    - Diagnostics (Ljung-Box, H-test, normality)
    - Smoothed components (arrays for plots)
    - Forecast data (if available)
    """

    def transform(self, results: Any) -> dict[str, Any]:
        """Transform results into template context dict.

        Parameters
        ----------
        results : StateSpaceResults or similar
            Fitted model results.

        Returns
        -------
        dict
            Template context with keys: model_info, parameters,
            info_criteria, diagnostics, components, forecast, metadata.
        """
        context: dict[str, Any] = {
            "model_info": self._extract_model_info(results),
            "parameters": self._extract_parameters(results),
            "info_criteria": self._extract_info_criteria(results),
            "diagnostics": self._extract_diagnostics(results),
            "components": self._extract_components(results),
            "forecast": self._extract_forecast(results),
            "metadata": self._build_metadata(results),
        }
        return context

    def _extract_model_info(self, results: Any) -> dict[str, Any]:
        """Extract basic model information."""
        info: dict[str, Any] = {}
        info["model_name"] = getattr(results, "model_name", "State Space Model")
        info["n_obs"] = int(getattr(results, "nobs", getattr(results, "n_obs", 0)))
        info["n_states"] = int(getattr(results, "n_states", getattr(results, "k_states", 0)))
        info["n_params"] = int(getattr(results, "n_params", getattr(results, "k_params", 0)))
        info["optimizer"] = str(getattr(results, "optimizer", "L-BFGS-B"))
        info["converged"] = bool(getattr(results, "converged", True))
        return info

    def _extract_parameters(self, results: Any) -> list[dict[str, Any]]:
        """Extract parameter table."""
        params: list[dict[str, Any]] = []

        param_names = getattr(results, "param_names", None)
        param_values = getattr(results, "params", None)
        param_se = getattr(results, "bse", getattr(results, "std_errors", None))
        param_pvalues = getattr(results, "pvalues", None)

        if param_names is None or param_values is None:
            return params

        param_values = np.asarray(param_values).ravel()
        n_params = len(param_names)

        if param_se is not None:
            param_se = np.asarray(param_se).ravel()
        if param_pvalues is not None:
            param_pvalues = np.asarray(param_pvalues).ravel()

        for i in range(n_params):
            entry: dict[str, Any] = {
                "name": str(param_names[i]),
                "value": float(param_values[i]),
            }
            if param_se is not None and i < len(param_se):
                se = float(param_se[i])
                entry["se"] = se
                entry["t_stat"] = float(param_values[i] / se) if se > 0 else float("nan")
            if param_pvalues is not None and i < len(param_pvalues):
                entry["p_value"] = float(param_pvalues[i])
                entry["significant"] = float(param_pvalues[i]) < 0.05
            params.append(entry)

        return params

    def _extract_info_criteria(self, results: Any) -> dict[str, float | None]:
        """Extract information criteria."""
        return {
            "loglike": _safe_float(getattr(results, "llf", getattr(results, "loglike", None))),
            "aic": _safe_float(getattr(results, "aic", None)),
            "bic": _safe_float(getattr(results, "bic", None)),
            "hqic": _safe_float(getattr(results, "hqic", None)),
        }

    def _extract_diagnostics(self, results: Any) -> dict[str, Any]:
        """Extract diagnostic test results."""
        diag: dict[str, Any] = {}

        # Ljung-Box
        lb = getattr(results, "ljung_box", None)
        if lb is not None:
            diag["ljung_box"] = lb
        elif hasattr(results, "test_serial_correlation"):
            with contextlib.suppress(Exception):
                diag["ljung_box"] = results.test_serial_correlation()

        # Heteroscedasticity
        het = getattr(results, "het_test", None)
        if het is not None:
            diag["heteroscedasticity"] = het
        elif hasattr(results, "test_heteroscedasticity"):
            with contextlib.suppress(Exception):
                diag["heteroscedasticity"] = results.test_heteroscedasticity()

        # Normality
        norm = getattr(results, "normality_test", None)
        if norm is not None:
            diag["normality"] = norm
        elif hasattr(results, "test_normality"):
            with contextlib.suppress(Exception):
                diag["normality"] = results.test_normality()

        return diag

    def _extract_components(self, results: Any) -> dict[str, Any]:
        """Extract smoothed component arrays."""
        components: dict[str, Any] = {}
        for comp_name in [
            "trend",
            "level",
            "slope",
            "seasonal",
            "cycle",
            "irregular",
        ]:
            val = getattr(results, comp_name, None)
            if val is None:
                val = getattr(results, f"smoothed_{comp_name}", None)
            if val is not None:
                components[comp_name] = np.asarray(val).tolist()
        return components

    def _extract_forecast(self, results: Any) -> dict[str, Any] | None:
        """Extract forecast data if available."""
        fc_mean = getattr(results, "forecast_mean", None)
        if fc_mean is None:
            return None
        data: dict[str, Any] = {"mean": np.asarray(fc_mean).tolist()}
        fc_se = getattr(results, "forecast_se", None)
        if fc_se is not None:
            data["se"] = np.asarray(fc_se).tolist()
        return data

    def _build_metadata(self, results: Any) -> dict[str, str]:
        """Build report metadata."""
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "kalmanbox_version": _get_version(),
            "report_type": "SSM",
        }


def _safe_float(val: Any) -> float | None:
    """Safely convert to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _get_version() -> str:
    """Get kalmanbox version string."""
    try:
        from kalmanbox.__version__ import __version__

        return __version__
    except ImportError:
        return "unknown"
