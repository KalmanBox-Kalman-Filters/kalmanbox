"""Information filter implementation.

Propagates the information form I = P^{-1} and i = P^{-1} @ a,
providing natural diffuse initialization and efficient updates
when k_obs >> k_states.

Reference: Anderson & Moore (1979), Optimal Filtering, Chapter 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput
from kalmanbox.utils.matrix_ops import (
    ensure_symmetric,
    log_det_via_cholesky,
    solve_via_cholesky,
)


class InformationFilter:
    """Information filter.

    Propagates the information matrix I = P^{-1} and information
    vector i = P^{-1} @ a. Supports natural diffuse initialization
    with I_0 = 0.

    On a linear Gaussian model, produces results IDENTICAL to the
    standard KalmanFilter (tolerance 1e-10).

    Parameters
    ----------
    diffuse : bool
        If True, initialize with I_0 = 0 (diffuse / no information).
        If False, initialize from ssm.P1 as I_0 = P1^{-1}.
        Default is False.
    """

    def __init__(self, diffuse: bool = False) -> None:
        self.diffuse = diffuse

    def filter(
        self,
        endog: NDArray[np.float64],
        ssm: StateSpaceRepresentation,
    ) -> FilterOutput:
        """Run the information filter on the full time series.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).
        ssm : StateSpaceRepresentation
            State-space model specification.

        Returns
        -------
        FilterOutput
            Complete filter output (same structure as standard KalmanFilter).
        """
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

        # Pre-compute H^{-1} and Z' @ H^{-1} @ Z for update
        H_inv = np.linalg.inv(ssm.H)
        ZT_Hinv = ssm.Z.T @ H_inv  # (k_states, k_endog)
        ZT_Hinv_Z = ZT_Hinv @ ssm.Z  # (k_states, k_states)

        # State noise covariance
        RQR = ssm.R @ ssm.Q @ ssm.R.T

        # Initialize information form
        if self.diffuse:
            I_curr = np.zeros((k_states, k_states))
            i_curr = np.zeros(k_states)
        else:
            I_curr = np.linalg.inv(ssm.P1)
            i_curr = I_curr @ ssm.a1

        loglike_total = 0.0
        nobs_effective = 0

        I_filt = I_curr
        i_filt = i_curr

        for t in range(nobs):
            # --- Prediction step ---
            if t == 0:
                I_pred = I_curr.copy()
                i_pred = i_curr.copy()
            else:
                I_pred, i_pred = self._predict_step(I_filt, i_filt, ssm.T, RQR, ssm.c)

            # Convert to state space for forecast / output
            P_pred, a_pred = self._info_to_state(I_pred, i_pred)

            predicted_state[t] = a_pred
            predicted_cov[t] = P_pred

            # Forecast
            y_pred = ssm.Z @ a_pred + ssm.d
            forecast[t] = y_pred

            # Check for missing data
            y_t = endog[t]
            is_missing = np.any(np.isnan(y_t))

            if is_missing:
                filtered_state[t] = a_pred
                filtered_cov[t] = P_pred
                I_filt = I_pred.copy()
                i_filt = i_pred.copy()
                residuals[t] = np.nan
                forecast_cov[t] = np.nan
                loglike_obs[t] = 0.0
            else:
                # --- Update step ---
                I_filt, i_filt = self._update_step(I_pred, i_pred, y_t, ZT_Hinv, ZT_Hinv_Z, ssm.d)

                # Convert to state space
                P_filt = np.linalg.inv(I_filt)
                P_filt = ensure_symmetric(P_filt)
                a_filt = P_filt @ i_filt

                filtered_state[t] = a_filt
                filtered_cov[t] = P_filt

                # Prediction error and forecast covariance
                v = y_t - ssm.Z @ a_pred - ssm.d
                F = ssm.Z @ P_pred @ ssm.Z.T + ssm.H
                F = ensure_symmetric(F)

                # Kalman gain
                K = solve_via_cholesky(F, ssm.Z @ P_pred.T).T

                residuals[t] = v
                forecast_cov[t] = F
                gain[t] = K

                # Log-likelihood
                log_det_F = log_det_via_cholesky(F)
                v_F_inv_v = float(v @ solve_via_cholesky(F, v))
                ll_t = -0.5 * (k_endog * np.log(2.0 * np.pi) + log_det_F + v_F_inv_v)
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
    def _predict_step(
        I_filt: NDArray[np.float64],
        i_filt: NDArray[np.float64],
        T: NDArray[np.float64],
        RQR: NDArray[np.float64],
        c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Prediction step in information form.

        Parameters
        ----------
        I_filt : information matrix I_{t|t}
        i_filt : information vector i_{t|t}
        T : transition matrix
        RQR : R @ Q @ R' (state noise covariance)
        c : state intercept

        Returns
        -------
        I_pred : predicted information matrix I_{t+1|t}
        i_pred : predicted information vector i_{t+1|t}
        """
        # Convert to state space, apply transition, convert back
        P_filt = np.linalg.inv(I_filt)
        a_filt = P_filt @ i_filt

        a_pred = T @ a_filt + c
        P_pred = T @ P_filt @ T.T + RQR
        P_pred = ensure_symmetric(P_pred)

        I_pred = np.linalg.inv(P_pred)
        I_pred = ensure_symmetric(I_pred)
        i_pred = I_pred @ a_pred

        return I_pred, i_pred

    @staticmethod
    def _update_step(
        I_pred: NDArray[np.float64],
        i_pred: NDArray[np.float64],
        y: NDArray[np.float64],
        ZT_Hinv: NDArray[np.float64],
        ZT_Hinv_Z: NDArray[np.float64],
        d: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update step in information form.

        The beauty of the information filter: update is a simple addition.

        Parameters
        ----------
        I_pred : predicted information matrix
        i_pred : predicted information vector
        y : observation
        ZT_Hinv : Z' @ H^{-1}, pre-computed
        ZT_Hinv_Z : Z' @ H^{-1} @ Z, pre-computed
        d : observation intercept

        Returns
        -------
        I_filt : filtered information matrix
        i_filt : filtered information vector
        """
        I_filt = I_pred + ZT_Hinv_Z
        i_filt = i_pred + ZT_Hinv @ (y - d)

        return I_filt, i_filt

    @staticmethod
    def _info_to_state(
        info_mat: NDArray[np.float64],
        info_vec: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert information form to state space form.

        Handles the case I = 0 (diffuse) by returning large covariance.

        Parameters
        ----------
        info_mat : information matrix
        info_vec : information vector

        Returns
        -------
        P : covariance matrix
        a : state vector
        """
        if np.max(np.abs(info_mat)) < 1e-15:
            # Diffuse: no information
            k = info_mat.shape[0]
            P = np.eye(k) * 1e7
            a = np.zeros(k)
        else:
            P = np.linalg.inv(info_mat)
            P = ensure_symmetric(P)
            a = P @ info_vec
        return P, a
