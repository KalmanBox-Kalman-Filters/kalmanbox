"""Diagnostic tools for state-space models."""

from kalmanbox.diagnostics.convergence import (
    ConvergenceReport,
    check_convergence,
    convergence_report,
)
from kalmanbox.diagnostics.missing import (
    MissingDataHandler,
    MissingDataReport,
)
from kalmanbox.diagnostics.residuals import (
    auxiliary_residuals,
    recursive_residuals,
    standardized_residuals,
)
from kalmanbox.diagnostics.tests import (
    TestResult,
    cusum_test,
    cusumsq_test,
    heteroskedasticity_test,
    ljung_box_test,
    normality_test,
)

__all__ = [
    "ConvergenceReport",
    "MissingDataHandler",
    "MissingDataReport",
    "TestResult",
    "auxiliary_residuals",
    "check_convergence",
    "convergence_report",
    "cusum_test",
    "cusumsq_test",
    "heteroskedasticity_test",
    "ljung_box_test",
    "normality_test",
    "recursive_residuals",
    "standardized_residuals",
]
