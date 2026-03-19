"""Classical Kalman filter implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.utils.matrix_ops import (
    ensure_symmetric,
    log_det_via_cholesky,
    solve_via_cholesky,
)


@dataclass
class FilterOutput:
    """Container for Kalman filter output.

    Attributes
    ----------
    filtered_state : NDArray
        Filtered state estimates a_{t|t}, shape (nobs, k_states).
    filtered_cov : NDArray
        Filtered state covariances P_{t|t}, shape (nobs, k_states, k_states).
    predicted_state : NDArray
        Predicted state estimates a_{t|t-1}, shape (nobs, k_states).
    predicted_cov : NDArray
        Predicted state covariances P_{t|t-1}, shape (nobs, k_states, k_states).
    forecast : NDArray
        Forecasted observations Z @ a_{t|t-1} + d, shape (nobs, k_endog).
    forecast_cov : NDArray
        Forecast error covariances F_t, shape (nobs, k_endog, k_endog).
    residuals : NDArray
        Prediction errors v_t = y_t - Z @ a_{t|t-1} - d, shape (nobs, k_endog).
    gain : NDArray
        Kalman gain K_t, shape (nobs, k_states, k_endog).
    loglike_obs : NDArray
        Per-observation log-likelihood contributions, shape (nobs,).
    loglike : float
        Total log-likelihood.
    nobs_effective : int
        Number of non-missing observations.
    """

    filtered_state: NDArray[np.float64]
    filtered_cov: NDArray[np.float64]
    predicted_state: NDArray[np.float64]
    predicted_cov: NDArray[np.float64]
    forecast: NDArray[np.float64]
    forecast_cov: NDArray[np.float64]
    residuals: NDArray[np.float64]
    gain: NDArray[np.float64]
    loglike_obs: NDArray[np.float64]
    loglike: float
    nobs_effective: int


class KalmanFilter:
    """Classical Kalman filter.

    Implements the prediction-update recursion for linear Gaussian
    state-space models with support for missing observations.
    """

    def filter(
        self,
        endog: NDArray[np.float64],
        ssm: StateSpaceRepresentation,
    ) -> FilterOutput:
        """Run the Kalman filter on the full time series.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).
        ssm : StateSpaceRepresentation
            State-space model specification.

        Returns
        -------
        FilterOutput
            Complete filter output.
        """
        # Ensure 2D
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        nobs = endog.shape[0]
        k_states = ssm.k_states
        k_endog = ssm.k_endog

        # Pre-allocate output arrays
        filtered_state = np.zeros((nobs, k_states))
        filtered_cov = np.zeros((nobs, k_states, k_states))
        predicted_state = np.zeros((nobs, k_states))
        predicted_cov = np.zeros((nobs, k_states, k_states))
        forecast = np.zeros((nobs, k_endog))
        forecast_cov = np.zeros((nobs, k_endog, k_endog))
        residuals = np.zeros((nobs, k_endog))
        gain = np.zeros((nobs, k_states, k_endog))
        loglike_obs = np.zeros(nobs)

        # Initialize
        a = ssm.a1.copy()
        P = ssm.P1.copy()

        loglike_total = 0.0
        nobs_effective = 0

        for t in range(nobs):
            # --- Prediction step ---
            if t == 0:
                a_pred = a
                P_pred = P
            else:
                a_pred, P_pred = self.predict_step(
                    filtered_state[t - 1],
                    filtered_cov[t - 1],
                    ssm.T,
                    ssm.R,
                    ssm.Q,
                    ssm.c,
                )

            predicted_state[t] = a_pred
            predicted_cov[t] = P_pred

            # --- Forecast ---
            y_pred = ssm.Z @ a_pred + ssm.d
            forecast[t] = y_pred

            # Check for missing data
            y_t = endog[t]
            is_missing = np.any(np.isnan(y_t))

            if is_missing:
                # No update: filtered = predicted
                filtered_state[t] = a_pred
                filtered_cov[t] = P_pred
                residuals[t] = np.nan
                forecast_cov[t] = np.nan
                loglike_obs[t] = 0.0
            else:
                # --- Update step ---
                a_filt, P_filt, v_t, F_t, K_t, ll_t = self.update_step(
                    a_pred, P_pred, y_t, ssm.Z, ssm.H, ssm.d
                )
                filtered_state[t] = a_filt
                filtered_cov[t] = P_filt
                residuals[t] = v_t
                forecast_cov[t] = F_t
                gain[t] = K_t
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

    @staticmethod
    def predict_step(
        a: NDArray[np.float64],
        P: NDArray[np.float64],
        T: NDArray[np.float64],
        R: NDArray[np.float64],
        Q: NDArray[np.float64],
        c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One prediction step.

        Parameters
        ----------
        a : filtered state a_{t|t}
        P : filtered covariance P_{t|t}
        T : transition matrix
        R : selection matrix
        Q : state disturbance covariance
        c : state intercept

        Returns
        -------
        a_pred : predicted state a_{t+1|t}
        P_pred : predicted covariance P_{t+1|t}
        """
        a_pred = T @ a + c
        P_pred = T @ P @ T.T + R @ Q @ R.T
        P_pred = ensure_symmetric(P_pred)
        return a_pred, P_pred

    @staticmethod
    def update_step(
        a_pred: NDArray[np.float64],
        P_pred: NDArray[np.float64],
        y: NDArray[np.float64],
        Z: NDArray[np.float64],
        H: NDArray[np.float64],
        d: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]:
        """One update step.

        Parameters
        ----------
        a_pred : predicted state a_{t|t-1}
        P_pred : predicted covariance P_{t|t-1}
        y : observation y_t
        Z : design matrix
        H : observation disturbance covariance
        d : observation intercept

        Returns
        -------
        a_filt : filtered state a_{t|t}
        P_filt : filtered covariance P_{t|t}
        v : prediction error v_t
        F : prediction error covariance F_t
        K : Kalman gain K_t
        loglike_t : log-likelihood contribution
        """
        # Prediction error
        v = y - Z @ a_pred - d

        # Prediction error covariance
        F = Z @ P_pred @ Z.T + H
        F = ensure_symmetric(F)

        # Kalman gain via Cholesky solve
        K = solve_via_cholesky(F, Z @ P_pred.T).T  # K = P_pred @ Z' @ F^{-1}

        # Filtered state
        a_filt = a_pred + K @ v

        # Filtered covariance
        P_filt = P_pred - K @ Z @ P_pred
        P_filt = ensure_symmetric(P_filt)

        # Log-likelihood contribution
        k_endog = Z.shape[0]
        log_det_F = log_det_via_cholesky(F)
        v_F_inv_v = float(v @ solve_via_cholesky(F, v))
        loglike_t = -0.5 * (k_endog * np.log(2.0 * np.pi) + log_det_F + v_F_inv_v)

        return a_filt, P_filt, v, F, K, loglike_t
