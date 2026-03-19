"""Maximum Likelihood Estimation for state-space models."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from kalmanbox.core.results import StateSpaceResults
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.rts import RTSSmoother

if TYPE_CHECKING:
    from kalmanbox.core.model import StateSpaceModel


class MLEstimator:
    """Maximum Likelihood Estimator for state-space models.

    Uses scipy.optimize.minimize to find parameters that maximize
    the log-likelihood computed via the Kalman filter prediction
    error decomposition.
    """

    def fit(
        self,
        model: StateSpaceModel,
        endog: NDArray[np.float64],
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        **kwargs: object,
    ) -> StateSpaceResults:
        """Estimate model parameters via MLE.

        Parameters
        ----------
        model : StateSpaceModel
            The model to estimate.
        endog : NDArray
            Observed data.
        method : str
            Optimization method for scipy.optimize.minimize.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------
        StateSpaceResults
        """
        kf = KalmanFilter()
        smoother = RTSSmoother()

        # Starting values in unconstrained space
        start_constrained = model.start_params
        x0 = model.untransform_params(start_constrained)

        # Number of diffuse initial observations to exclude from loglike.
        # With approximate diffuse initialization (large P1), the first
        # observation's loglike contribution is dominated by the prior
        # and should be excluded (Durbin & Koopman, 2012).
        n_diffuse = getattr(model, "_n_diffuse", 1)

        def neg_loglike(unconstrained_params: NDArray[np.float64]) -> float:
            """Negative log-likelihood (objective to minimize)."""
            try:
                constrained = model.transform_params(unconstrained_params)
                ssm = model._build_ssm(constrained)
                output = kf.filter(endog, ssm)
                return -float(np.sum(output.loglike_obs[n_diffuse:]))
            except Exception:
                return 1e10  # Return large value on failure

        # Optimize
        result = optimize.minimize(
            neg_loglike,
            x0,
            method=method,
            options={"maxiter": maxiter, "disp": False},
        )

        # Extract optimal parameters
        optimal_unconstrained = result.x
        optimal_params = model.transform_params(optimal_unconstrained)

        # Compute standard errors via numerical Hessian
        se = self.standard_errors(neg_loglike, optimal_unconstrained, model)

        # Run filter + smoother with optimal params
        ssm = model._build_ssm(optimal_params)
        filter_output = kf.filter(endog, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        # Diffuse log-likelihood (exclude initial diffuse observations)
        diffuse_loglike = float(np.sum(filter_output.loglike_obs[n_diffuse:]))

        return StateSpaceResults(
            params=optimal_params,
            param_names=model.param_names,
            se=se,
            loglike=diffuse_loglike,
            nobs=model.nobs,
            filter_output=filter_output,
            smoother_output=smoother_output,
            ssm=ssm,
            optimizer_converged=result.success,
            optimizer_message=str(result.message),
        )

    def standard_errors(
        self,
        neg_loglike_fn: Callable[[NDArray[np.float64]], float],
        unconstrained_params: NDArray[np.float64],
        model: StateSpaceModel,
    ) -> NDArray[np.float64]:
        """Compute standard errors via numerical Hessian.

        Uses the delta method to transform standard errors from
        unconstrained to constrained space.

        Parameters
        ----------
        neg_loglike_fn : callable
            Negative log-likelihood function.
        unconstrained_params : NDArray
            Optimal unconstrained parameters.
        model : StateSpaceModel
            The model (for transform_params).

        Returns
        -------
        NDArray
            Standard errors in constrained space.
        """
        k = len(unconstrained_params)
        eps = 1e-5

        # Numerical Hessian (central differences)
        hessian = np.zeros((k, k))

        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k)
                ej = np.zeros(k)
                ei[i] = eps
                ej[j] = eps

                fpp = neg_loglike_fn(unconstrained_params + ei + ej)
                fpm = neg_loglike_fn(unconstrained_params + ei - ej)
                fmp = neg_loglike_fn(unconstrained_params - ei + ej)
                fmm = neg_loglike_fn(unconstrained_params - ei - ej)

                hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
                hessian[j, i] = hessian[i, j]

        # Variance-covariance in unconstrained space
        try:
            cov_unconstrained = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            return np.full(k, np.nan)

        # Delta method: transform SE to constrained space
        # Jacobian of transform_params at optimal point
        jacobian = np.zeros((k, k))
        for i in range(k):
            ei = np.zeros(k)
            ei[i] = eps
            f_plus = model.transform_params(unconstrained_params + ei)
            f_minus = model.transform_params(unconstrained_params - ei)
            jacobian[:, i] = (f_plus - f_minus) / (2 * eps)

        cov_constrained = jacobian @ cov_unconstrained @ jacobian.T
        se = np.sqrt(np.maximum(np.diag(cov_constrained), 0.0))

        return se
