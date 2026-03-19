"""Square-root Kalman filter implementation.

Propagates the Cholesky factor S where P = S @ S.T, guaranteeing
positive-definiteness of the covariance matrix by construction.

Reference: Anderson & Moore (1979), Optimal Filtering, Chapter 6.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput


def _qr_factor(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract lower-triangular factor from QR decomposition.

    Given matrix M, computes QR of M such that M = Q @ R,
    then returns R.T with positive diagonal (lower-triangular factor S
    satisfying M.T @ M = S @ S.T).

    Parameters
    ----------
    M : NDArray, shape (p, n) with p >= n
        Matrix to factor.

    Returns
    -------
    S : NDArray, shape (n, n)
        Lower-triangular Cholesky-like factor.
    """
    _, R = np.linalg.qr(M, mode="reduced")
    # Force positive diagonal
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    R = signs[:, None] * R
    return R.T


class SquareRootKalmanFilter:
    """Square-root Kalman filter.

    Propagates the Cholesky factor S of the covariance P = S @ S.T
    using QR decompositions. This guarantees P remains positive-definite
    by construction, which is critical for ill-conditioned models.

    On a linear Gaussian model, produces results IDENTICAL to the
    standard KalmanFilter (tolerance 1e-10).
    """

    def filter(
        self,
        endog: NDArray[np.float64],
        ssm: StateSpaceRepresentation,
    ) -> FilterOutput:
        """Run the square-root Kalman filter on the full time series.

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

        # Cholesky factors of Q and H
        Q_chol = np.linalg.cholesky(ssm.Q)  # lower-triangular
        H_chol = np.linalg.cholesky(ssm.H)  # lower-triangular

        # Initialize Cholesky factor of P1
        S_filt = np.linalg.cholesky(ssm.P1)  # lower-triangular
        a = ssm.a1.copy()

        loglike_total = 0.0
        nobs_effective = 0

        for t in range(nobs):
            # --- Prediction step ---
            if t == 0:
                a_pred = a
                S_pred = S_filt
            else:
                a_pred = ssm.T @ filtered_state[t - 1] + ssm.c
                S_pred = self._predict_cholesky(S_filt, ssm.T, ssm.R, Q_chol)

            P_pred = S_pred @ S_pred.T
            predicted_state[t] = a_pred
            predicted_cov[t] = P_pred

            # --- Forecast ---
            y_pred = ssm.Z @ a_pred + ssm.d
            forecast[t] = y_pred

            # Check for missing data
            y_t = endog[t]
            is_missing = np.any(np.isnan(y_t))

            if is_missing:
                filtered_state[t] = a_pred
                filtered_cov[t] = P_pred
                S_filt = S_pred
                residuals[t] = np.nan
                forecast_cov[t] = np.nan
                loglike_obs[t] = 0.0
            else:
                # --- Update step ---
                a_filt, S_filt, v_t, S_F, K_t, ll_t = self._update_step(
                    a_pred, S_pred, y_t, ssm.Z, H_chol, ssm.d
                )
                P_filt = S_filt @ S_filt.T
                F_t = S_F @ S_F.T

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
    def _predict_cholesky(
        S_filt: NDArray[np.float64],
        T: NDArray[np.float64],
        R: NDArray[np.float64],
        Q_chol: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Prediction step via QR decomposition.

        Computes S_pred such that S_pred @ S_pred.T = T @ P_filt @ T' + R @ Q @ R'

        Parameters
        ----------
        S_filt : lower-triangular Cholesky factor of P_{t|t}
        T : transition matrix
        R : selection matrix
        Q_chol : lower-triangular Cholesky factor of Q

        Returns
        -------
        S_pred : lower-triangular Cholesky factor of P_{t+1|t}
        """
        M = np.vstack(
            [
                (T @ S_filt).T,
                (R @ Q_chol).T,
            ]
        )
        return _qr_factor(M)

    @staticmethod
    def _update_step(
        a_pred: NDArray[np.float64],
        S_pred: NDArray[np.float64],
        y: NDArray[np.float64],
        Z: NDArray[np.float64],
        H_chol: NDArray[np.float64],
        d: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]:
        """One update step using square-root formulation.

        Parameters
        ----------
        a_pred : predicted state a_{t|t-1}
        S_pred : Cholesky factor of predicted covariance
        y : observation y_t
        Z : design matrix
        H_chol : Cholesky factor of H
        d : observation intercept

        Returns
        -------
        a_filt : filtered state a_{t|t}
        S_filt : Cholesky factor of filtered covariance
        v : prediction error
        S_F : Cholesky factor of forecast error covariance
        K : Kalman gain
        loglike_t : log-likelihood contribution
        """
        k_endog = Z.shape[0]
        k_states = S_pred.shape[0]

        # Prediction error
        v = y - Z @ a_pred - d

        # Forecast error Cholesky factor via QR
        M_F = np.vstack(
            [
                (Z @ S_pred).T,
                H_chol.T,
            ]
        )
        S_F = _qr_factor(M_F)

        # Kalman gain: K = P_pred @ Z' @ F^{-1}
        P_pred_ZT = S_pred @ S_pred.T @ Z.T
        K = scipy.linalg.cho_solve((S_F, True), P_pred_ZT.T).T

        # Filtered state
        a_filt = a_pred + K @ v

        # Filtered covariance Cholesky via Joseph form QR
        # P_filt = (I-KZ) P (I-KZ)' + K H K'
        IKZ = np.eye(k_states) - K @ Z
        M_filt = np.vstack(
            [
                (IKZ @ S_pred).T,
                (K @ H_chol).T,
            ]
        )
        S_filt = _qr_factor(M_filt)

        # Log-likelihood
        log_det_F = 2.0 * np.sum(np.log(np.abs(np.diag(S_F))))
        v_Finv_v = float(v @ scipy.linalg.cho_solve((S_F, True), v))
        loglike_t = -0.5 * (k_endog * np.log(2.0 * np.pi) + log_det_F + v_Finv_v)

        return a_filt, S_filt, v, S_F, K, loglike_t
