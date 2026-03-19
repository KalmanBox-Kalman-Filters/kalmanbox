"""EM algorithm for state-space model estimation (Shumway & Stoffer 1982)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from kalmanbox._logging import get_logger
from kalmanbox.core.results import StateSpaceResults
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.rts import RTSSmoother

if TYPE_CHECKING:
    from kalmanbox.core.model import StateSpaceModel

logger = get_logger("em")


class EMEstimator:
    """EM estimator for linear Gaussian state-space models.

    Uses the Expectation-Maximization algorithm:
    - E-step: Kalman smoother for smoothed states and covariances
    - M-step: Update parameters (numerical or analytical)

    Parameters
    ----------
    max_iter : int
        Maximum number of EM iterations. Default 100.
    tol : float
        Convergence tolerance for log-likelihood change. Default 1e-6.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        model: StateSpaceModel,
        endog: NDArray[np.float64],
        **kwargs: object,
    ) -> StateSpaceResults:
        """Estimate model parameters via EM algorithm.

        Parameters
        ----------
        model : StateSpaceModel
            The model to estimate.
        endog : NDArray
            Observed data.

        Returns
        -------
        StateSpaceResults
        """
        kf = KalmanFilter()
        smoother = RTSSmoother()

        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        nobs = endog.shape[0]

        # Number of diffuse initial observations to exclude (matching MLE)
        n_diffuse = getattr(model, "_n_diffuse", 1)

        def _diffuse_loglike(filt_output: object) -> float:
            return float(np.sum(filt_output.loglike_obs[n_diffuse:]))

        # Initialize with start_params
        params = model.start_params.copy()
        prev_loglike = -np.inf
        converged = False
        loglike_history: list[float] = []

        for iteration in range(self.max_iter):
            # --- E-step: run Kalman smoother ---
            ssm = model._build_ssm(params)
            filter_output = kf.filter(endog, ssm)
            smoother_output = smoother.smooth(filter_output, ssm)

            current_loglike = _diffuse_loglike(filter_output)
            loglike_history.append(current_loglike)

            logger.debug("EM iteration %d: loglike = %.4f", iteration, current_loglike)

            # Check convergence
            if iteration > 0:
                ll_change = current_loglike - prev_loglike
                if ll_change < -1e-4:
                    logger.warning(
                        "EM iteration %d: loglike decreased by %.6f",
                        iteration,
                        -ll_change,
                    )
                if abs(ll_change) < self.tol:
                    converged = True
                    break

            prev_loglike = current_loglike

            # --- M-step: update parameters ---
            params = self._m_step(model, params, endog, n_diffuse)

        # Final E-step with optimal parameters
        ssm = model._build_ssm(params)
        filter_output = kf.filter(endog, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        # Compute standard errors via numerical Hessian
        se = self._compute_se(model, params, endog, kf, n_diffuse)

        return StateSpaceResults(
            params=params,
            param_names=model.param_names,
            se=se,
            loglike=_diffuse_loglike(filter_output),
            nobs=nobs,
            filter_output=filter_output,
            smoother_output=smoother_output,
            ssm=ssm,
            optimizer_converged=converged,
            optimizer_message=f"EM converged in {len(loglike_history)} iterations"
            if converged
            else f"EM did not converge in {self.max_iter} iterations",
        )

    def _m_step(
        self,
        model: StateSpaceModel,
        current_params: NDArray[np.float64],
        endog: NDArray[np.float64],
        n_diffuse: int,
    ) -> NDArray[np.float64]:
        """M-step: update parameters using numerical approach.

        For the generic case, we use a single-step MLE optimization
        starting from current parameters (warm start). This is equivalent
        to a generalized EM (GEM) step.

        For specific models (DFM), override this method with analytical updates.
        """
        from scipy import optimize

        kf = KalmanFilter()

        unconstrained = model.untransform_params(current_params)

        def neg_loglike(x: NDArray[np.float64]) -> float:
            try:
                constrained = model.transform_params(x)
                ssm_new = model._build_ssm(constrained)
                output = kf.filter(endog, ssm_new)
                return -float(np.sum(output.loglike_obs[n_diffuse:]))
            except Exception:
                return 1e10

        # Single optimization step (warm start from current)
        result = optimize.minimize(
            neg_loglike,
            unconstrained,
            method="L-BFGS-B",
            options={"maxiter": 50},
        )

        return model.transform_params(result.x)

    def _compute_se(
        self,
        model: StateSpaceModel,
        params: NDArray[np.float64],
        endog: NDArray[np.float64],
        kf: KalmanFilter,
        n_diffuse: int = 1,
    ) -> NDArray[np.float64]:
        """Compute standard errors via numerical Hessian."""
        unconstrained = model.untransform_params(params)
        k = len(unconstrained)
        eps = 1e-5

        def neg_loglike(x: NDArray[np.float64]) -> float:
            try:
                constrained = model.transform_params(x)
                ssm = model._build_ssm(constrained)
                output = kf.filter(endog, ssm)
                return -float(np.sum(output.loglike_obs[n_diffuse:]))
            except Exception:
                return 1e10

        # Numerical Hessian
        hessian = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k)
                ej = np.zeros(k)
                ei[i] = eps
                ej[j] = eps

                fpp = neg_loglike(unconstrained + ei + ej)
                fpm = neg_loglike(unconstrained + ei - ej)
                fmp = neg_loglike(unconstrained - ei + ej)
                fmm = neg_loglike(unconstrained - ei - ej)

                hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
                hessian[j, i] = hessian[i, j]

        try:
            cov_unconstrained = np.linalg.inv(hessian)
            # Delta method
            jacobian = np.zeros((k, k))
            for i in range(k):
                ei = np.zeros(k)
                ei[i] = eps
                f_plus = model.transform_params(unconstrained + ei)
                f_minus = model.transform_params(unconstrained - ei)
                jacobian[:, i] = (f_plus - f_minus) / (2 * eps)

            cov_constrained = jacobian @ cov_unconstrained @ jacobian.T
            return np.sqrt(np.maximum(np.diag(cov_constrained), 0.0))
        except np.linalg.LinAlgError:
            return np.full(k, np.nan)


def compute_lag_one_covariance(
    smoother_gain: NDArray[np.float64],
    smoothed_cov: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute lag-one smoothed covariance P_{t,t-1|T}.

    P_{t,t-1|T} = L_{t-1} @ P_{t|T}

    Parameters
    ----------
    smoother_gain : NDArray, shape (nobs, k_states, k_states)
    smoothed_cov : NDArray, shape (nobs, k_states, k_states)

    Returns
    -------
    NDArray, shape (nobs, k_states, k_states)
        Lag-one smoothed covariances.
    """
    nobs = smoothed_cov.shape[0]
    k_states = smoothed_cov.shape[1]
    lag_one_cov = np.zeros((nobs, k_states, k_states))

    for t in range(1, nobs):
        lag_one_cov[t] = smoother_gain[t - 1] @ smoothed_cov[t]

    return lag_one_cov
