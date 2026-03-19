"""Time-Varying Parameters (TVP) regression model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.core.results import StateSpaceResults
from kalmanbox.filters.kalman import FilterOutput
from kalmanbox.smoothers.rts import RTSSmoother
from kalmanbox.utils.matrix_ops import (
    ensure_symmetric,
    log_det_via_cholesky,
    solve_via_cholesky,
)


class TimeVaryingParameters(StateSpaceModel):
    """Time-Varying Parameters regression model.

    Model:
        y_t = X_t @ beta_t + eps_t,    eps_t ~ N(0, sigma2_obs)
        beta_t = beta_{t-1} + eta_t,    eta_t ~ N(0, Q)

    The coefficients beta_t follow a random walk, allowing them to
    change smoothly over time.

    Parameters
    ----------
    endog : NDArray
        Dependent variable, shape (nobs,).
    exog : NDArray
        Regressors, shape (nobs, k). Include a column of ones for intercept.
    q_type : str
        Type of Q matrix: 'diagonal' (default) or 'scalar'.
        - 'diagonal': each coefficient evolves independently
        - 'scalar': all coefficients share the same innovation variance
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        exog: NDArray[np.float64],
        q_type: str = "diagonal",
    ) -> None:
        self.exog = np.asarray(exog, dtype=np.float64)
        if self.exog.ndim == 1:
            self.exog = self.exog.reshape(-1, 1)
        self.k_regressors = self.exog.shape[1]
        self._q_type = q_type
        super().__init__(endog)

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial guesses: sigma2_obs + Q diagonal entries."""
        var = float(np.nanvar(self.endog))
        params: list[float] = [var / 2.0]  # sigma2_obs

        if self._q_type == "diagonal":
            for _ in range(self.k_regressors):
                params.append(var / (10.0 * self.k_regressors))
        else:  # scalar
            params.append(var / 10.0)

        return np.array(params)

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        names = ["sigma2_obs"]
        if self._q_type == "diagonal":
            for i in range(self.k_regressors):
                names.append(f"sigma2_beta_{i}")
        else:
            names.append("sigma2_beta")
        return names

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """All variances: exp."""
        return np.exp(unconstrained)

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse transform: log."""
        return np.log(constrained)

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build base SSM (Z set to mean of X as placeholder)."""
        k = self.k_regressors
        sigma2_obs = params[0]

        ssm = StateSpaceRepresentation(k_states=k, k_endog=1, k_posdef=k)
        ssm.T = np.eye(k)
        ssm.Z = np.mean(self.exog, axis=0).reshape(1, k)  # placeholder
        ssm.R = np.eye(k)
        ssm.H = np.array([[sigma2_obs]])

        if self._q_type == "diagonal":
            q_diag = params[1 : 1 + k]
            ssm.Q = np.diag(q_diag)
        else:
            ssm.Q = params[1] * np.eye(k)

        ssm.a1 = np.zeros(k)
        ssm.P1 = np.eye(k) * config.diffuse_initial_variance
        return ssm

    def _build_q(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract Q matrix from params."""
        k = self.k_regressors
        if self._q_type == "diagonal":
            return np.diag(params[1 : 1 + k])
        return params[1] * np.eye(k)

    def loglike(self, params: NDArray[np.float64]) -> float:
        """Compute log-likelihood with time-varying Z.

        Overrides base class to handle Z_t = X_t at each step.
        """
        k = self.k_regressors
        sigma2_obs = params[0]
        Q = self._build_q(params)
        H = np.array([[sigma2_obs]])
        T = np.eye(k)
        R = np.eye(k)

        y = self.endog
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        nobs = y.shape[0]
        a = np.zeros(k)
        P = np.eye(k) * config.diffuse_initial_variance

        total_loglike = 0.0

        for t in range(nobs):
            Z_t = self.exog[t : t + 1, :]  # (1, k)

            # Prediction (for t > 0)
            if t > 0:
                a = T @ a
                P = T @ P @ T.T + R @ Q @ R.T
                P = ensure_symmetric(P)

            y_t = y[t]
            if np.any(np.isnan(y_t)):
                continue

            # Innovation
            v = y_t - Z_t @ a
            F = Z_t @ P @ Z_t.T + H
            F = ensure_symmetric(F)

            # Update
            K = solve_via_cholesky(F, Z_t @ P.T).T
            a = a + K @ v
            P = P - K @ Z_t @ P
            P = ensure_symmetric(P)

            # Loglike contribution
            log_det_F = log_det_via_cholesky(F)
            v_F_inv_v = float(v @ solve_via_cholesky(F, v))
            total_loglike += -0.5 * (np.log(2 * np.pi) + log_det_F + v_F_inv_v)

        return total_loglike

    def _filter_tvp(self, params: NDArray[np.float64]) -> FilterOutput:
        """Run Kalman filter with time-varying Z_t = X_t."""
        k = self.k_regressors
        sigma2_obs = params[0]
        q_mat = self._build_q(params)
        h_mat = np.array([[sigma2_obs]])
        t_mat = np.eye(k)
        r_mat = np.eye(k)

        y = self.endog
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        nobs = y.shape[0]

        filtered_state = np.zeros((nobs, k))
        filtered_cov = np.zeros((nobs, k, k))
        predicted_state = np.zeros((nobs, k))
        predicted_cov = np.zeros((nobs, k, k))
        forecast = np.zeros((nobs, 1))
        forecast_cov = np.zeros((nobs, 1, 1))
        residuals = np.zeros((nobs, 1))
        gain = np.zeros((nobs, k, 1))
        loglike_obs = np.zeros(nobs)

        a = np.zeros(k)
        p = np.eye(k) * config.diffuse_initial_variance
        loglike_total = 0.0
        nobs_effective = 0

        for t in range(nobs):
            z_t = self.exog[t : t + 1, :]  # (1, k)

            # Prediction
            if t == 0:
                a_pred = a
                p_pred = p
            else:
                a_pred = t_mat @ filtered_state[t - 1]
                p_pred = t_mat @ filtered_cov[t - 1] @ t_mat.T + r_mat @ q_mat @ r_mat.T
                p_pred = ensure_symmetric(p_pred)

            predicted_state[t] = a_pred
            predicted_cov[t] = p_pred

            forecast[t] = z_t @ a_pred

            y_t = y[t]
            if np.any(np.isnan(y_t)):
                filtered_state[t] = a_pred
                filtered_cov[t] = p_pred
                residuals[t] = np.nan
                forecast_cov[t] = np.nan
                loglike_obs[t] = 0.0
                continue

            v = y_t - z_t @ a_pred
            f = z_t @ p_pred @ z_t.T + h_mat
            f = ensure_symmetric(f)
            k_t = solve_via_cholesky(f, z_t @ p_pred.T).T

            filtered_state[t] = a_pred + k_t @ v
            filtered_cov[t] = p_pred - k_t @ z_t @ p_pred
            filtered_cov[t] = ensure_symmetric(filtered_cov[t])
            residuals[t] = v
            forecast_cov[t] = f
            gain[t] = k_t

            log_det_f = log_det_via_cholesky(f)
            v_f_inv_v = float(v @ solve_via_cholesky(f, v))
            ll_t = -0.5 * (np.log(2 * np.pi) + log_det_f + v_f_inv_v)
            loglike_obs[t] = ll_t
            loglike_total += ll_t
            nobs_effective += 1

        return FilterOutput(
            filtered_state=filtered_state,
            filtered_cov=filtered_cov,
            predicted_state=predicted_state,
            predicted_cov=predicted_cov,
            forecast=forecast,
            forecast_cov=forecast_cov,
            residuals=residuals,
            gain=gain,
            loglike_obs=loglike_obs,
            loglike=loglike_total,
            nobs_effective=nobs_effective,
        )

    def fit(self, method: str = "mle", **kwargs: object) -> StateSpaceResults:
        """Fit TVP model with time-varying Z."""
        from kalmanbox.estimation.mle import MLEstimator

        estimator = MLEstimator()
        smoother = RTSSmoother()

        x0 = self.untransform_params(self.start_params)

        def neg_loglike(unconstrained: NDArray[np.float64]) -> float:
            try:
                constrained = self.transform_params(unconstrained)
                return -self.loglike(constrained)
            except Exception:
                return 1e10

        result = optimize.minimize(
            neg_loglike,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        optimal_params = self.transform_params(result.x)

        # Standard errors
        se = estimator.standard_errors(neg_loglike, result.x, self)

        # Run custom filter with time-varying Z, then standard smoother
        ssm = self._build_ssm(optimal_params)
        filter_output = self._filter_tvp(optimal_params)
        smoother_output = smoother.smooth(filter_output, ssm)

        return StateSpaceResults(
            params=optimal_params,
            param_names=self.param_names,
            se=se,
            loglike=self.loglike(optimal_params),
            nobs=self.nobs,
            filter_output=filter_output,
            smoother_output=smoother_output,
            ssm=ssm,
            optimizer_converged=result.success,
            optimizer_message=str(result.message),
        )

    @property
    def time_varying_coefficients(self) -> None:
        """Placeholder — call fit() and use results.smoothed_state."""
        return None
