"""Parametric bootstrap for state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from kalmanbox._logging import get_logger
from kalmanbox.simulation.simulate import simulate_ssm

if TYPE_CHECKING:
    from kalmanbox.core.model import StateSpaceModel
    from kalmanbox.core.results import StateSpaceResults

logger = get_logger("bootstrap")


@dataclass
class BootstrapResult:
    """Container for parametric bootstrap results.

    Attributes
    ----------
    params_draws : NDArray[np.float64]
        Bootstrap parameter draws, shape (n_boot, k_params).
    params_mean : NDArray[np.float64]
        Mean of bootstrap parameter draws.
    params_se : NDArray[np.float64]
        Standard error of bootstrap parameter draws.
    ci_lower : NDArray[np.float64]
        Lower confidence interval (percentile method).
    ci_upper : NDArray[np.float64]
        Upper confidence interval (percentile method).
    ci_lower_bc : NDArray[np.float64]
        Lower confidence interval (bias-corrected).
    ci_upper_bc : NDArray[np.float64]
        Upper confidence interval (bias-corrected).
    n_success : int
        Number of successful bootstrap replications.
    n_boot : int
        Total number of bootstrap replications attempted.
    param_names : list[str]
        Parameter names.
    alpha : float
        Significance level for confidence intervals.
    """

    params_draws: NDArray[np.float64]
    params_mean: NDArray[np.float64]
    params_se: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    ci_lower_bc: NDArray[np.float64]
    ci_upper_bc: NDArray[np.float64]
    n_success: int
    n_boot: int
    param_names: list[str]
    alpha: float

    def __repr__(self) -> str:
        lines = [
            "Parametric Bootstrap Results",
            "=" * 60,
            f"Successful replications: {self.n_success}/{self.n_boot}",
            f"Confidence level: {(1 - self.alpha) * 100:.0f}%",
            "",
            f"{'Parameter':<20} {'Mean':>10} {'SE':>10} {'CI Lower':>10} {'CI Upper':>10}",
            "-" * 60,
        ]
        for i, name in enumerate(self.param_names):
            lines.append(
                f"{name:<20} {self.params_mean[i]:>10.4f} "
                f"{self.params_se[i]:>10.4f} "
                f"{self.ci_lower[i]:>10.4f} {self.ci_upper[i]:>10.4f}"
            )
        return "\n".join(lines)


def parametric_bootstrap(
    model: StateSpaceModel,
    results: StateSpaceResults,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BootstrapResult:
    """Perform parametric bootstrap for a state-space model.

    The parametric bootstrap uses the estimated model as the data
    generating process (DGP):

    1. Simulate data from the estimated model
    2. Re-estimate the model on simulated data
    3. Collect parameter estimates
    4. Compute confidence intervals

    Parameters
    ----------
    model : StateSpaceModel
        The original model (will be cloned for each replication).
    results : StateSpaceResults
        Fitted model results with estimated parameters.
    n_boot : int
        Number of bootstrap replications. Default 200.
    alpha : float
        Significance level for confidence intervals. Default 0.05.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Bootstrap results with confidence intervals.
    """
    rng = np.random.default_rng(seed)
    params_hat = results.params
    ssm = results.ssm
    k_params = len(params_hat)
    nobs = results.nobs

    params_draws = np.full((n_boot, k_params), np.nan)
    n_success = 0

    for b in range(n_boot):
        boot_seed = int(rng.integers(0, 2**31))

        try:
            # 1. Simulate from estimated model
            y_boot, _ = simulate_ssm(ssm, n_periods=nobs, seed=boot_seed)

            # 2. Re-estimate model on simulated data
            # Create a new model instance with simulated data
            model_boot = model.__class__(y_boot.squeeze())

            # Set start params near the true values for faster convergence
            model_boot._start_params = params_hat.copy()

            results_boot = model_boot.fit()
            params_draws[b] = results_boot.params
            n_success += 1

        except Exception as exc:
            logger.debug("Bootstrap replication %d failed: %s", b, exc)
            continue

    if n_success < 3:
        logger.warning("Only %d/%d bootstrap replications succeeded", n_success, n_boot)
        return BootstrapResult(
            params_draws=params_draws[:n_success],
            params_mean=params_hat,
            params_se=np.full(k_params, np.nan),
            ci_lower=np.full(k_params, np.nan),
            ci_upper=np.full(k_params, np.nan),
            ci_lower_bc=np.full(k_params, np.nan),
            ci_upper_bc=np.full(k_params, np.nan),
            n_success=n_success,
            n_boot=n_boot,
            param_names=list(results.param_names),
            alpha=alpha,
        )

    # Keep only successful draws
    valid_draws = params_draws[~np.any(np.isnan(params_draws), axis=1)]

    # Bootstrap statistics
    params_mean = np.mean(valid_draws, axis=0)
    params_se = np.std(valid_draws, axis=0, ddof=1)

    # Percentile confidence intervals
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = np.percentile(valid_draws, lower_pct, axis=0)
    ci_upper = np.percentile(valid_draws, upper_pct, axis=0)

    # Bias-corrected confidence intervals (BCa simplified)
    ci_lower_bc, ci_upper_bc = _bias_corrected_ci(valid_draws, params_hat, alpha)

    return BootstrapResult(
        params_draws=valid_draws,
        params_mean=params_mean,
        params_se=params_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_lower_bc=ci_lower_bc,
        ci_upper_bc=ci_upper_bc,
        n_success=n_success,
        n_boot=n_boot,
        param_names=list(results.param_names),
        alpha=alpha,
    )


def _bias_corrected_ci(
    draws: NDArray[np.float64],
    theta_hat: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute bias-corrected confidence intervals.

    Uses the bias-corrected percentile method:
    1. Compute bias correction z0 = Phi^{-1}(prop of draws < theta_hat)
    2. Adjusted percentiles: Phi(2*z0 + z_{alpha/2}), Phi(2*z0 + z_{1-alpha/2})

    Parameters
    ----------
    draws : NDArray, shape (n_boot, k_params)
    theta_hat : NDArray, shape (k_params,)
    alpha : float

    Returns
    -------
    ci_lower, ci_upper : NDArray, each shape (k_params,)
    """
    from scipy.stats import norm

    n_boot, k_params = draws.shape
    ci_lower = np.zeros(k_params)
    ci_upper = np.zeros(k_params)

    z_alpha_lower = norm.ppf(alpha / 2.0)
    z_alpha_upper = norm.ppf(1.0 - alpha / 2.0)

    for j in range(k_params):
        # Bias correction
        prop_below = np.mean(draws[:, j] < theta_hat[j])
        prop_below = np.clip(prop_below, 1e-10, 1.0 - 1e-10)
        z0 = norm.ppf(prop_below)

        # Adjusted percentiles
        adj_lower = norm.cdf(2 * z0 + z_alpha_lower) * 100.0
        adj_upper = norm.cdf(2 * z0 + z_alpha_upper) * 100.0

        adj_lower = np.clip(adj_lower, 0.5, 99.5)
        adj_upper = np.clip(adj_upper, 0.5, 99.5)

        ci_lower[j] = np.percentile(draws[:, j], adj_lower)
        ci_upper[j] = np.percentile(draws[:, j], adj_upper)

    return ci_lower, ci_upper
