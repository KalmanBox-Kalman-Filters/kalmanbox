"""DFM (Dynamic Factor Model) data transformer for reports.

Extends SSMTransformer with factor-specific data: extracted factors,
loadings matrix, variance explained per factor, R-squared per series.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kalmanbox.reports.transformers.ssm import SSMTransformer


class DFMTransformer(SSMTransformer):
    """Transform DynamicFactorResults into template context.

    Adds to SSM context:
    - factors: extracted factor time series
    - loadings: factor loadings matrix
    - variance_explained: proportion explained by each factor
    - r_squared: R-squared per observed series
    """

    def transform(self, results: Any) -> dict[str, Any]:
        """Transform DFM results into template context.

        Parameters
        ----------
        results : DynamicFactorResults or similar
            Fitted DFM results.

        Returns
        -------
        dict
            Template context with SSM fields plus DFM-specific fields.
        """
        context = super().transform(results)
        context["metadata"]["report_type"] = "DFM"

        context["factors"] = self._extract_factors(results)
        context["loadings"] = self._extract_loadings(results)
        context["variance_explained"] = self._extract_variance_explained(results)
        context["r_squared"] = self._extract_r_squared(results)

        return context

    def _extract_factors(self, results: Any) -> list[list[float]] | None:
        """Extract factor time series."""
        for attr in ["factors", "smoothed_factors", "filtered_factors"]:
            val = getattr(results, attr, None)
            if val is not None:
                arr = np.asarray(val)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                return arr.tolist()
        return None

    def _extract_loadings(self, results: Any) -> list[list[float]] | None:
        """Extract loadings matrix."""
        for attr in ["loadings", "factor_loadings", "loading_matrix"]:
            val = getattr(results, attr, None)
            if val is not None:
                return np.asarray(val).tolist()
        return None

    def _extract_variance_explained(self, results: Any) -> list[float] | None:
        """Extract variance explained per factor."""
        val = getattr(results, "variance_explained", None)
        if val is not None:
            return np.asarray(val).ravel().tolist()

        # Compute from loadings if possible
        loadings = None
        for attr in ["loadings", "factor_loadings"]:
            loadings = getattr(results, attr, None)
            if loadings is not None:
                break
        if loadings is not None:
            loadings = np.asarray(loadings)
            total_var = np.sum(loadings**2)
            if total_var > 0:
                per_factor = np.sum(loadings**2, axis=0) / total_var
                return per_factor.tolist()
        return None

    def _extract_r_squared(self, results: Any) -> list[dict[str, Any]] | None:
        """Extract R-squared per series."""
        val = getattr(results, "r_squared", None)
        if val is None:
            val = getattr(results, "rsquared", None)
        if val is not None:
            arr = np.asarray(val).ravel()
            series_names = getattr(results, "series_names", None)
            result = []
            for i, r2 in enumerate(arr):
                name = (
                    series_names[i] if series_names and i < len(series_names) else f"Series {i + 1}"
                )
                result.append({"name": name, "r_squared": float(r2)})
            return result
        return None
