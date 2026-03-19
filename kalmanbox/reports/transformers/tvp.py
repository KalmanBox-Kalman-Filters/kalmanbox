"""TVP (Time-Varying Parameter) data transformer for reports.

Extends SSMTransformer with coefficient evolution data,
time-varying significance, and constancy tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kalmanbox.reports.transformers.ssm import SSMTransformer


class TVPTransformer(SSMTransformer):
    """Transform TVP regression results into template context.

    Adds to SSM context:
    - coefficients: time-varying coefficient paths
    - coefficient_significance: significance bands over time
    - constancy_tests: Q > 0 tests for each coefficient
    """

    def transform(self, results: Any) -> dict[str, Any]:
        """Transform TVP results into template context.

        Parameters
        ----------
        results : TVPResults or similar
            Fitted TVP regression results.

        Returns
        -------
        dict
            Template context with SSM fields plus TVP-specific fields.
        """
        context = super().transform(results)
        context["metadata"]["report_type"] = "TVP"

        context["coefficients"] = self._extract_coefficients(results)
        context["coefficient_significance"] = self._extract_significance(results)
        context["constancy_tests"] = self._extract_constancy_tests(results)

        return context

    def _extract_coefficients(self, results: Any) -> list[dict[str, Any]] | None:
        """Extract time-varying coefficient paths."""
        coefs = getattr(results, "smoothed_coefficients", None)
        if coefs is None:
            coefs = getattr(results, "coefficients", None)
        if coefs is None:
            coefs = getattr(results, "smoothed_state", None)
        if coefs is None:
            return None

        arr = np.asarray(coefs)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        _n_time, n_coefs = arr.shape
        coef_names = getattr(results, "coef_names", None)
        if coef_names is None:
            coef_names = getattr(results, "param_names", None)

        result: list[dict[str, Any]] = []
        for j in range(n_coefs):
            name = coef_names[j] if coef_names and j < len(coef_names) else f"beta_{j}"
            result.append(
                {
                    "name": str(name),
                    "values": arr[:, j].tolist(),
                    "mean": float(np.mean(arr[:, j])),
                    "std": float(np.std(arr[:, j])),
                    "min": float(np.min(arr[:, j])),
                    "max": float(np.max(arr[:, j])),
                }
            )

        return result

    def _extract_significance(self, results: Any) -> list[dict[str, Any]] | None:
        """Extract significance bands over time for each coefficient."""
        coefs = getattr(results, "smoothed_coefficients", None)
        if coefs is None:
            coefs = getattr(results, "smoothed_state", None)
        cov = getattr(results, "smoothed_state_cov", None)

        if coefs is None or cov is None:
            return None

        coefs = np.asarray(coefs)
        cov = np.asarray(cov)

        if coefs.ndim == 1:
            coefs = coefs.reshape(-1, 1)

        _n_time, n_coefs = coefs.shape
        coef_names = getattr(results, "coef_names", None)

        result: list[dict[str, Any]] = []
        for j in range(n_coefs):
            name = coef_names[j] if coef_names and j < len(coef_names) else f"beta_{j}"
            if cov.ndim == 3:
                std_j = np.sqrt(np.maximum(cov[:, j, j], 0.0))
            elif cov.ndim == 2:
                std_j = np.sqrt(np.maximum(cov[:, j], 0.0))
            else:
                continue

            # t-stats over time
            t_stats = np.where(std_j > 0, np.abs(coefs[:, j]) / std_j, 0.0)
            pct_significant = float(np.mean(t_stats > 1.96) * 100)

            result.append(
                {
                    "name": str(name),
                    "upper_95": (coefs[:, j] + 1.96 * std_j).tolist(),
                    "lower_95": (coefs[:, j] - 1.96 * std_j).tolist(),
                    "pct_significant": pct_significant,
                }
            )

        return result

    def _extract_constancy_tests(self, results: Any) -> list[dict[str, Any]] | None:
        """Extract Q > 0 constancy tests for each coefficient."""
        tests = getattr(results, "constancy_tests", None)
        if tests is not None:
            return tests

        q_diag = getattr(results, "state_cov_diagonal", None)
        if q_diag is None:
            q_diag = getattr(results, "Q_diagonal", None)

        if q_diag is not None:
            q_arr = np.asarray(q_diag).ravel()
            coef_names = getattr(results, "coef_names", None)
            result: list[dict[str, Any]] = []
            for j, q_val in enumerate(q_arr):
                name = coef_names[j] if coef_names and j < len(coef_names) else f"beta_{j}"
                result.append(
                    {
                        "name": str(name),
                        "q_value": float(q_val),
                        "is_varying": float(q_val) > 1e-10,
                    }
                )
            return result

        return None
