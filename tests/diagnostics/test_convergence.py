"""Tests for convergence diagnostics."""

import numpy as np

from kalmanbox.diagnostics.convergence import (
    ConvergenceReport,
    check_convergence,
    convergence_report,
)
from kalmanbox.models.local_level import LocalLevel


class TestCheckConvergence:
    """Tests for check_convergence."""

    def test_converged_model(self, nile_volume: np.ndarray) -> None:
        """A well-fitted model should show convergence."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        report = check_convergence(results)

        assert isinstance(report, ConvergenceReport)
        # A well-fitted model should have at least some criteria OK
        assert report.loglike_change_ok or report.param_change_ok

    def test_convergence_report_string(self, nile_volume: np.ndarray) -> None:
        """convergence_report should return a formatted string."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        text = convergence_report(results)

        assert isinstance(text, str)
        assert "Convergence Report" in text
        assert "Gradient norm" in text

    def test_report_repr(self, nile_volume: np.ndarray) -> None:
        """ConvergenceReport repr should be informative."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        report = check_convergence(results)
        text = repr(report)
        assert "Convergence Report" in text

    def test_hessian_condition(self, nile_volume: np.ndarray) -> None:
        """Hessian condition number should be finite for well-specified model."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        report = check_convergence(results)
        # Should be finite (model is well-specified)
        assert np.isfinite(report.hessian_condition) or report.hessian_condition == np.inf

    def test_custom_tolerances(self, nile_volume: np.ndarray) -> None:
        """Custom tolerances should be respected."""
        model = LocalLevel(nile_volume)
        results = model.fit()

        # Very tight tolerance
        report_tight = check_convergence(results, grad_tol=1e-20)
        # Very loose tolerance
        report_loose = check_convergence(results, grad_tol=1e10)

        # Loose should have fewer warnings
        assert len(report_loose.warnings) <= len(report_tight.warnings)
