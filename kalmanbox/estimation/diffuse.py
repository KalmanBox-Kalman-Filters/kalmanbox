"""Exact diffuse initialization for the Kalman filter (Koopman 1997)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput
from kalmanbox.utils.matrix_ops import (
    ensure_symmetric,
    log_det_via_cholesky,
    solve_via_cholesky,
)


@dataclass
class DiffuseFilterOutput(FilterOutput):
    """Extended filter output with diffuse initialization info.

    Additional Attributes
    ---------------------
    diffuse_periods : int
        Number of observations processed in diffuse mode.
    p_inf : NDArray
        Final P_inf matrix (should be all zeros after convergence).
    """

    diffuse_periods: int = 0
    p_inf: NDArray[np.float64] | None = None


class DiffuseInitialization:
    """Exact diffuse initialization handler.

    Manages the decomposition P1 = P_star + kappa * P_inf and provides
    a modified Kalman filter for the initial diffuse period.

    Parameters
    ----------
    diffuse_states : NDArray[np.bool_] | None
        Boolean mask indicating which states are diffuse.
        If None, auto-detects from P1 (states with P1[i,i] > threshold).
    threshold : float
        Threshold for auto-detecting diffuse states. Default 1e5.
    """

    def __init__(
        self,
        diffuse_states: NDArray[np.bool_] | None = None,
        threshold: float = 1e5,
    ) -> None:
        self.diffuse_states = diffuse_states
        self.threshold = threshold

    def decompose_initial(
        self, ssm: StateSpaceRepresentation
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Decompose P1 into P_star and P_inf.

        Parameters
        ----------
        ssm : StateSpaceRepresentation

        Returns
        -------
        P_star : finite part of initial covariance
        P_inf : diffuse part (identity for diffuse states)
        """
        k_states = ssm.k_states

        if self.diffuse_states is not None:
            mask = self.diffuse_states
        else:
            # Auto-detect: states with large diagonal in P1
            mask = np.diag(ssm.P1) > self.threshold

        P_inf = np.zeros((k_states, k_states))
        P_star = ssm.P1.copy()

        for i in range(k_states):
            if mask[i]:
                P_inf[i, i] = 1.0
                P_star[i, i] = 0.0
                # Zero out cross-terms with diffuse states
                for j in range(k_states):
                    if i != j:
                        P_star[i, j] = 0.0
                        P_star[j, i] = 0.0

        return P_star, P_inf

    def filter(
        self,
        endog: NDArray[np.float64],
        ssm: StateSpaceRepresentation,
    ) -> DiffuseFilterOutput:
        """Run Kalman filter with exact diffuse initialization.

        Parameters
        ----------
        endog : NDArray
            Observed data, shape (nobs,) or (nobs, k_endog).
        ssm : StateSpaceRepresentation

        Returns
        -------
        DiffuseFilterOutput
        """
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        nobs = endog.shape[0]
        k_states = ssm.k_states
        k_endog = ssm.k_endog

        # Pre-allocate
        filtered_state = np.zeros((nobs, k_states))
        filtered_cov = np.zeros((nobs, k_states, k_states))
        predicted_state = np.zeros((nobs, k_states))
        predicted_cov = np.zeros((nobs, k_states, k_states))
        forecast = np.zeros((nobs, k_endog))
        forecast_cov = np.zeros((nobs, k_endog, k_endog))
        residuals = np.zeros((nobs, k_endog))
        gain = np.zeros((nobs, k_states, k_endog))
        loglike_obs = np.zeros(nobs)

        # Decompose initial conditions
        P_star, P_inf = self.decompose_initial(ssm)
        a = ssm.a1.copy()

        loglike_total = 0.0
        nobs_effective = 0
        diffuse_periods = 0
        tol = 1e-10

        P_star_filt = P_star
        P_inf_filt = P_inf
        P_inf_pred = P_inf

        for t in range(nobs):
            # --- Prediction ---
            if t == 0:
                a_pred = a
                P_star_pred = P_star
                P_inf_pred = P_inf
            else:
                a_pred = ssm.T @ filtered_state[t - 1] + ssm.c
                P_star_pred = ssm.T @ P_star_filt @ ssm.T.T + ssm.R @ ssm.Q @ ssm.R.T
                P_inf_pred = ssm.T @ P_inf_filt @ ssm.T.T
                P_star_pred = ensure_symmetric(P_star_pred)
                P_inf_pred = ensure_symmetric(P_inf_pred)

            predicted_state[t] = a_pred
            predicted_cov[t] = P_star_pred + P_inf_pred

            # Forecast
            y_pred = ssm.Z @ a_pred + ssm.d
            forecast[t] = y_pred

            y_t = endog[t]
            is_missing = np.any(np.isnan(y_t))

            if is_missing:
                filtered_state[t] = a_pred
                filtered_cov[t] = P_star_pred + P_inf_pred
                residuals[t] = np.nan
                forecast_cov[t] = np.nan
                loglike_obs[t] = 0.0
                P_star_filt = P_star_pred
                P_inf_filt = P_inf_pred
                continue

            # Innovation
            v = y_t - ssm.Z @ a_pred - ssm.d
            residuals[t] = v

            F_inf = ssm.Z @ P_inf_pred @ ssm.Z.T
            F_star = ssm.Z @ P_star_pred @ ssm.Z.T + ssm.H

            # Check if still in diffuse mode
            if np.max(np.abs(F_inf)) > tol:
                # --- Diffuse update ---
                diffuse_periods = t + 1

                F_inf = ensure_symmetric(F_inf)
                F_star = ensure_symmetric(F_star)

                F_inf_inv = solve_via_cholesky(F_inf, np.eye(k_endog))

                K_0 = P_inf_pred @ ssm.Z.T @ F_inf_inv
                K_1 = (P_star_pred @ ssm.Z.T - K_0 @ F_star) @ F_inf_inv

                a_filt = a_pred + K_0 @ v

                L_0 = np.eye(k_states) - K_0 @ ssm.Z
                L_1 = -K_1 @ ssm.Z

                P_inf_filt = P_inf_pred @ L_0.T
                P_star_filt = P_star_pred @ L_0.T + P_inf_pred @ L_1.T

                P_inf_filt = ensure_symmetric(P_inf_filt)
                P_star_filt = ensure_symmetric(P_star_filt)

                # Clamp near-zero P_inf entries to exact zero
                P_inf_filt[np.abs(P_inf_filt) < tol] = 0.0

                # Diffuse log-likelihood (minimal info)
                log_det_F_inf = log_det_via_cholesky(F_inf)
                ll_t = -0.5 * (k_endog * np.log(2 * np.pi) + log_det_F_inf)
                loglike_obs[t] = ll_t

                filtered_state[t] = a_filt
                filtered_cov[t] = P_star_filt + P_inf_filt
                forecast_cov[t] = F_inf + F_star
                gain[t] = K_0

            else:
                # --- Normal update (diffuse period ended) ---
                F_t = F_star
                F_t = ensure_symmetric(F_t)
                forecast_cov[t] = F_t

                K_t = solve_via_cholesky(F_t, ssm.Z @ P_star_pred.T).T
                a_filt = a_pred + K_t @ v
                P_star_filt = P_star_pred - K_t @ ssm.Z @ P_star_pred
                P_star_filt = ensure_symmetric(P_star_filt)
                P_inf_filt = P_inf_pred  # should be ~zero

                # Normal log-likelihood
                log_det_F = log_det_via_cholesky(F_t)
                v_F_inv_v = float(v @ solve_via_cholesky(F_t, v))
                ll_t = -0.5 * (k_endog * np.log(2 * np.pi) + log_det_F + v_F_inv_v)
                loglike_obs[t] = ll_t
                loglike_total += ll_t
                nobs_effective += 1

                filtered_state[t] = a_filt
                filtered_cov[t] = P_star_filt
                gain[t] = K_t

        return DiffuseFilterOutput(
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
            diffuse_periods=diffuse_periods,
            p_inf=P_inf_pred if nobs > 0 else P_inf,
        )
