"""UCM (Unobserved Components Model) data transformer for reports.

Extends SSMTransformer with component decomposition details,
relative contributions, and per-component diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kalmanbox.reports.transformers.ssm import SSMTransformer


class UCMTransformer(SSMTransformer):
    """Transform UCM results into template context.

    Adds to SSM context:
    - component_details: per-component statistics
    - contributions: relative contribution of each component
    - component_diagnostics: per-component diagnostic info
    """

    def transform(self, results: Any) -> dict[str, Any]:
        """Transform UCM results into template context.

        Parameters
        ----------
        results : UCMResults or similar
            Fitted UCM results.

        Returns
        -------
        dict
            Template context with SSM fields plus UCM-specific fields.
        """
        context = super().transform(results)
        context["metadata"]["report_type"] = "UCM"

        context["component_details"] = self._extract_component_details(results)
        context["contributions"] = self._extract_contributions(results)
        context["component_diagnostics"] = self._extract_component_diagnostics(results)

        return context

    def _extract_component_details(self, results: Any) -> list[dict[str, Any]]:
        """Extract per-component statistics."""
        details: list[dict[str, Any]] = []
        component_names = ["trend", "slope", "seasonal", "cycle", "irregular"]

        for name in component_names:
            val = getattr(results, name, None)
            if val is None:
                val = getattr(results, f"smoothed_{name}", None)
            if val is not None:
                arr = np.asarray(val).ravel()
                entry: dict[str, Any] = {
                    "name": name.capitalize(),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                }
                var_attr = getattr(results, f"{name}_variance", None)
                if var_attr is not None:
                    entry["variance"] = float(var_attr)
                details.append(entry)

        return details

    def _extract_contributions(self, results: Any) -> list[dict[str, Any]] | None:
        """Extract relative contribution of each component to total variance."""
        contributions = getattr(results, "contributions", None)
        if contributions is not None:
            return [{"name": k, "proportion": float(v)} for k, v in contributions.items()]

        # Try to compute from component variances
        component_names = ["trend", "slope", "seasonal", "cycle", "irregular"]
        variances: dict[str, float] = {}
        for name in component_names:
            val = getattr(results, name, None)
            if val is None:
                val = getattr(results, f"smoothed_{name}", None)
            if val is not None:
                arr = np.asarray(val).ravel()
                variances[name] = float(np.nanvar(arr))

        if variances:
            total = sum(variances.values())
            if total > 0:
                return [
                    {"name": k.capitalize(), "proportion": v / total} for k, v in variances.items()
                ]
        return None

    def _extract_component_diagnostics(self, results: Any) -> list[dict[str, Any]] | None:
        """Extract per-component diagnostic information."""
        diag = getattr(results, "component_diagnostics", None)
        if diag is not None:
            return diag
        return None
