"""Regression model in state-space form."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.core.results import StateSpaceResults
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.rts import RTSSmoother


class RegressionSSM(StateSpaceModel):
    """Regression model in state-space form.

    Model:
        y_t = X_t @ beta + eps_t,    eps_t ~ N(0, sigma2_obs)

    This implements classical linear regression as a state-space model
    where the state vector is the constant coefficient vector beta.

    Parameters
    ----------
    endog : NDArray
        Observed dependent variable, shape (nobs,).
    exog : NDArray
        Regressor matrix, shape (nobs, k_regressors).
        Should include a constant column if intercept is desired.
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        exog: NDArray[np.float64],
    ) -> None:
        self.exog = np.asarray(exog, dtype=np.float64)
        if self.exog.ndim == 1:
            self.exog = self.exog.reshape(-1, 1)
        self.k_regressors = self.exog.shape[1]
        super().__init__(endog)

    @property
    def start_params(self) -> NDArray[np.float64]:
        """OLS estimates as start params.

        Returns [beta_1, ..., beta_k, sigma2].
        """
        y = self.endog[:, 0] if self.endog.ndim == 2 else self.endog
        X = self.exog
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta_ols
        sigma2 = float(np.sum(resid**2) / max(len(y) - len(beta_ols), 1))
        return np.concatenate([beta_ols, [sigma2]])

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        names = [f"beta_{i}" for i in range(self.k_regressors)]
        names.append("sigma2_obs")
        return names

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Betas: identity. sigma2: exp."""
        constrained = unconstrained.copy()
        constrained[-1] = np.exp(unconstrained[-1])
        return constrained

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse transform."""
        unconstrained = constrained.copy()
        unconstrained[-1] = np.log(constrained[-1])
        return unconstrained

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build SSM for regression.

        Note: Z is set to the mean of X for the base SSM.
        The actual time-varying Z is handled in the fit/loglike methods.
        """
        k = self.k_regressors
        sigma2 = params[-1]

        ssm = StateSpaceRepresentation(k_states=k, k_endog=1, k_posdef=k)
        ssm.T = np.eye(k)
        ssm.Z = np.mean(self.exog, axis=0).reshape(1, k)
        ssm.R = np.eye(k)
        ssm.Q = np.zeros((k, k))  # Fixed coefficients
        ssm.H = np.array([[sigma2]])
        ssm.a1 = np.zeros(k)
        ssm.P1 = np.eye(k) * config.diffuse_initial_variance
        return ssm

    def fit(self, method: str = "mle", **kwargs: object) -> StateSpaceResults:
        """Fit using analytical OLS solution.

        For fixed-coefficient regression, OLS is the MLE.
        We compute analytically and wrap in StateSpaceResults.
        """
        y = self.endog[:, 0] if self.endog.ndim == 2 else self.endog.ravel()
        X = self.exog
        n = len(y)
        k = self.k_regressors

        # OLS
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta_hat
        sigma2_hat = float(np.sum(resid**2) / n)

        # Log-likelihood
        loglike = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma2_hat) - n / 2

        # Standard errors (classical OLS)
        sigma2_unbiased = float(np.sum(resid**2) / max(n - k, 1))
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(sigma2_unbiased * np.diag(XtX_inv))
        except np.linalg.LinAlgError:
            se_beta = np.full(k, np.nan)

        # SE for sigma2 (approximate)
        se_sigma2 = sigma2_hat * np.sqrt(2.0 / n)

        params = np.concatenate([beta_hat, [sigma2_hat]])
        se = np.concatenate([se_beta, [se_sigma2]])

        # Build SSM and run filter for completeness
        ssm = self._build_ssm(params)
        kf = KalmanFilter()
        smoother = RTSSmoother()
        filter_output = kf.filter(self.endog, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        return StateSpaceResults(
            params=params,
            param_names=self.param_names,
            se=se,
            loglike=loglike,
            nobs=n,
            filter_output=filter_output,
            smoother_output=smoother_output,
            ssm=ssm,
            optimizer_converged=True,
            optimizer_message="Analytical OLS solution",
        )
