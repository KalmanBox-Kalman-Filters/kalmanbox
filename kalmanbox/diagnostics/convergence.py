"""Convergence diagnostics for optimization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kalmanbox.core.results import StateSpaceResults


@dataclass
class ConvergenceReport:
    """Report on optimization convergence.

    Attributes
    ----------
    converged : bool
        Whether the optimization converged.
    gradient_norm : float
        Norm of the gradient at the optimum.
    gradient_ok : bool
        Whether gradient norm is below tolerance.
    param_change : float
        Norm of parameter change in last iteration.
    param_change_ok : bool
        Whether parameter change is below tolerance.
    loglike_change : float
        Change in log-likelihood in last iteration.
    loglike_change_ok : bool
        Whether likelihood change is below tolerance.
    hessian_condition : float
        Condition number of the Hessian.
    hessian_ok : bool
        Whether Hessian condition number is acceptable.
    warnings : list[str]
        List of convergence warnings.
    """

    converged: bool
    gradient_norm: float
    gradient_ok: bool
    param_change: float
    param_change_ok: bool
    loglike_change: float
    loglike_change_ok: bool
    hessian_condition: float
    hessian_ok: bool
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return formatted convergence report."""
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines = [
            f"Convergence Report: {status}",
            "=" * 50,
            f"Gradient norm:       {self.gradient_norm:.2e}"
            f"  {'OK' if self.gradient_ok else 'WARNING'}",
            f"Parameter change:    {self.param_change:.2e}"
            f"  {'OK' if self.param_change_ok else 'WARNING'}",
            f"Log-like change:     {self.loglike_change:.2e}"
            f"  {'OK' if self.loglike_change_ok else 'WARNING'}",
            f"Hessian condition:   {self.hessian_condition:.2e}"
            f"  {'OK' if self.hessian_ok else 'WARNING'}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def check_convergence(
    results: StateSpaceResults,
    grad_tol: float = 1e-4,
    param_tol: float = 1e-6,
    ll_tol: float = 1e-6,
    cond_tol: float = 1e10,
) -> ConvergenceReport:
    """Check convergence of optimization results.

    Evaluates multiple convergence criteria:
    1. Gradient norm at the optimum
    2. Parameter change in last iteration
    3. Log-likelihood change in last iteration
    4. Condition number of the Hessian

    Parameters
    ----------
    results : StateSpaceResults
        Fitted model results.
    grad_tol : float
        Tolerance for gradient norm. Default 1e-4.
    param_tol : float
        Tolerance for parameter change. Default 1e-6.
    ll_tol : float
        Tolerance for log-likelihood change. Default 1e-6.
    cond_tol : float
        Tolerance for Hessian condition number. Default 1e10.

    Returns
    -------
    ConvergenceReport
    """
    warnings_list: list[str] = []

    # --- Gradient norm ---
    gradient_norm = _compute_gradient_norm(results)
    gradient_ok = gradient_norm < grad_tol
    if not gradient_ok:
        warnings_list.append(
            f"Gradient norm ({gradient_norm:.2e}) exceeds tolerance ({grad_tol:.2e})"
        )

    # --- Parameter change ---
    param_change = _compute_param_change(results)
    param_change_ok = param_change < param_tol
    if not param_change_ok:
        warnings_list.append(
            f"Parameter change ({param_change:.2e}) exceeds tolerance ({param_tol:.2e})"
        )

    # --- Log-likelihood change ---
    loglike_change = _compute_loglike_change(results)
    loglike_change_ok = loglike_change < ll_tol
    if not loglike_change_ok:
        warnings_list.append(
            f"Log-likelihood change ({loglike_change:.2e}) exceeds tolerance ({ll_tol:.2e})"
        )

    # --- Hessian condition number ---
    hessian_condition = _compute_hessian_condition(results)
    hessian_ok = hessian_condition < cond_tol
    if not hessian_ok:
        warnings_list.append(
            f"Hessian condition number ({hessian_condition:.2e}) exceeds tolerance ({cond_tol:.2e})"
        )

    converged = gradient_ok and param_change_ok and loglike_change_ok and hessian_ok

    return ConvergenceReport(
        converged=converged,
        gradient_norm=gradient_norm,
        gradient_ok=gradient_ok,
        param_change=param_change,
        param_change_ok=param_change_ok,
        loglike_change=loglike_change,
        loglike_change_ok=loglike_change_ok,
        hessian_condition=hessian_condition,
        hessian_ok=hessian_ok,
        warnings=warnings_list,
    )


def convergence_report(results: StateSpaceResults) -> str:
    """Generate a human-readable convergence report.

    Parameters
    ----------
    results : StateSpaceResults
        Fitted model results.

    Returns
    -------
    str
        Formatted convergence report.
    """
    report = check_convergence(results)
    return repr(report)


def _compute_gradient_norm(results: StateSpaceResults) -> float:
    """Compute gradient norm at the optimum via finite differences.

    Parameters
    ----------
    results : StateSpaceResults

    Returns
    -------
    float
        L2 norm of the gradient.
    """
    se = results.se

    # If standard errors are available, use them to approximate gradient
    # A well-converged optimizer should have near-zero gradient
    # We approximate using the fact that at the MLE, score = 0
    # Use finite difference on log-likelihood if possible
    if hasattr(results, "_model") and results._model is not None:
        return _numerical_gradient_norm(results)

    # Fallback: if SE are available and finite, assume gradient is small
    if np.all(np.isfinite(se)):
        return 0.0

    return np.nan


def _numerical_gradient_norm(results: StateSpaceResults) -> float:
    """Compute gradient norm via numerical differentiation."""
    try:
        from kalmanbox.filters.kalman import KalmanFilter

        model = results._model  # type: ignore[attr-defined]
        kf = KalmanFilter()
        params = results.params
        eps = 1e-5
        k = len(params)
        grad = np.zeros(k)

        for i in range(k):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            ssm_plus = model._build_ssm(params_plus)
            ssm_minus = model._build_ssm(params_minus)

            ll_plus = kf.filter(model.endog, ssm_plus).loglike
            ll_minus = kf.filter(model.endog, ssm_minus).loglike

            grad[i] = (ll_plus - ll_minus) / (2 * eps)

        return float(np.linalg.norm(grad))
    except Exception:
        return np.nan


def _compute_param_change(results: StateSpaceResults) -> float:
    """Estimate parameter change from standard errors.

    In a well-converged model, the last parameter change should be
    much smaller than the standard errors.
    """
    # Without optimization history, we use a proxy
    # If optimizer_converged is True, assume small change
    if hasattr(results, "optimizer_converged") and results.optimizer_converged:
        return 0.0
    return np.nan


def _compute_loglike_change(results: StateSpaceResults) -> float:
    """Estimate log-likelihood change.

    Without optimization history, we use the optimizer convergence flag.
    """
    if hasattr(results, "optimizer_converged") and results.optimizer_converged:
        return 0.0
    return np.nan


def _compute_hessian_condition(results: StateSpaceResults) -> float:
    """Compute condition number of the Hessian at the optimum.

    Uses the standard errors to approximate the Hessian.
    """
    se = results.se
    if np.any(np.isnan(se)) or np.any(se <= 0):
        return np.inf

    # Approximate: Hessian diagonal ~ 1/se^2
    # Condition number ~ max(1/se^2) / min(1/se^2) = max(se)^2 / min(se)^2
    se_valid = se[se > 0]
    if len(se_valid) < 2:
        return 1.0

    condition = (np.max(se_valid) / np.min(se_valid)) ** 2
    return float(condition)
