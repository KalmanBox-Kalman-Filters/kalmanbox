"""Disturbance smoother for diagnostics (Koopman 1993).

Computes smoothed observation disturbances (eps_hat_t) and state
disturbances (eta_hat_t) for outlier and structural break detection.

Reference: Koopman, S.J. (1993). Disturbance smoother for state space models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput, KalmanFilter
from kalmanbox.utils.matrix_ops import ensure_symmetric, solve_via_cholesky


@dataclass
class DisturbanceSmootherOutput:
    """Container for disturbance smoother output.

    Attributes
    ----------
    smoothed_obs_disturbance : NDArray
        Smoothed observation disturbances eps_hat_t, shape (nobs, k_endog).
    smoothed_state_disturbance : NDArray
        Smoothed state disturbances eta_hat_t, shape (nobs, k_posdef).
    obs_disturbance_var : NDArray
        Variance of smoothed obs disturbances, shape (nobs, k_endog, k_endog).
    state_disturbance_var : NDArray
        Variance of smoothed state disturbances, shape (nobs, k_posdef, k_posdef).
    obs_auxiliary_residual : NDArray
        Standardized obs disturbance (outlier diagnostic), shape (nobs, k_endog).
    state_auxiliary_residual : NDArray
        Standardized state disturbance (break diagnostic), shape (nobs, k_posdef).
    r : NDArray
        Backward recursion vector r_t, shape (nobs, k_states).
    N : NDArray
        Backward recursion matrix N_t, shape (nobs, k_states, k_states).
    """

    smoothed_obs_disturbance: NDArray[np.float64]
    smoothed_state_disturbance: NDArray[np.float64]
    obs_disturbance_var: NDArray[np.float64]
    state_disturbance_var: NDArray[np.float64]
    obs_auxiliary_residual: NDArray[np.float64]
    state_auxiliary_residual: NDArray[np.float64]
    r: NDArray[np.float64]
    N: NDArray[np.float64]


class DisturbanceSmoother:
    """Disturbance smoother (Koopman 1993).

    Computes smoothed disturbances and auxiliary residuals for
    diagnostic purposes: outlier detection and structural break
    detection.

    Auxiliary residuals |e_t| > 2-3 indicate possible outliers.
    Auxiliary residuals |r_t| > 2-3 indicate possible structural breaks.
    """

    def smooth(
        self,
        endog: NDArray[np.float64],
        ssm: StateSpaceRepresentation,
        filter_output: FilterOutput | None = None,
    ) -> DisturbanceSmootherOutput:
        """Run the disturbance smoother.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).
        ssm : StateSpaceRepresentation
            State-space model specification.
        filter_output : FilterOutput | None
            Pre-computed filter output. If None, runs KalmanFilter first.

        Returns
        -------
        DisturbanceSmootherOutput
            Smoothed disturbances and auxiliary residuals.
        """
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        if filter_output is None:
            kf = KalmanFilter()
            filter_output = kf.filter(endog, ssm)

        nobs = endog.shape[0]
        k_states = ssm.k_states
        k_endog = ssm.k_endog
        k_posdef = ssm.Q.shape[0]

        # Pre-allocate backward recursion arrays
        r_arr = np.zeros((nobs + 1, k_states))
        N_arr = np.zeros((nobs + 1, k_states, k_states))

        # Pre-allocate output arrays
        smoothed_obs_dist = np.zeros((nobs, k_endog))
        smoothed_state_dist = np.zeros((nobs, k_posdef))
        obs_dist_var = np.zeros((nobs, k_endog, k_endog))
        state_dist_var = np.zeros((nobs, k_posdef, k_posdef))
        obs_aux_resid = np.zeros((nobs, k_endog))
        state_aux_resid = np.zeros((nobs, k_posdef))

        # Backward recursion: t = nobs-1, ..., 0
        for t in range(nobs - 1, -1, -1):
            # Check for missing data
            y_t = endog[t]
            is_missing = np.any(np.isnan(y_t))

            if is_missing:
                # No observation: L_t = T, skip observation terms
                r_arr[t] = ssm.T.T @ r_arr[t + 1]
                N_arr[t] = ssm.T.T @ N_arr[t + 1] @ ssm.T
                N_arr[t] = ensure_symmetric(N_arr[t])

                # No disturbances for missing observations
                obs_dist_var[t] = ssm.H
                state_dist_var[t] = ssm.Q
                continue

            v_t = filter_output.residuals[t]
            F_t = filter_output.forecast_cov[t]
            P_pred_t = filter_output.predicted_cov[t]

            # F_t^{-1} @ v_t
            F_inv_v = solve_via_cholesky(F_t, v_t)

            # Kalman gain (prediction form): K_pred = T @ P @ Z' @ F^{-1}
            K_pred = ssm.T @ solve_via_cholesky(F_t, ssm.Z @ P_pred_t).T

            # L_t = T - K_pred @ Z
            L_t = ssm.T - K_pred @ ssm.Z

            # r_{t-1} = Z' @ F_t^{-1} @ v_t + L_t' @ r_t
            r_arr[t] = ssm.Z.T @ F_inv_v + L_t.T @ r_arr[t + 1]

            # N_{t-1} = Z' @ F_t^{-1} @ Z + L_t' @ N_t @ L_t
            F_inv_Z = solve_via_cholesky(F_t, ssm.Z)
            N_arr[t] = ssm.Z.T @ F_inv_Z + L_t.T @ N_arr[t + 1] @ L_t
            N_arr[t] = ensure_symmetric(N_arr[t])

            # Smoothed disturbances using filter-form gain and r_t (= r_arr[t+1])
            K_filt = filter_output.gain[t]  # P @ Z' @ F^{-1}

            # u_t = F_t^{-1} @ v_t - K_filt' @ r_t
            u_t = F_inv_v - K_filt.T @ r_arr[t + 1]

            # D_t = F_t^{-1} + K_filt' @ N_t @ K_filt
            F_inv = solve_via_cholesky(F_t, np.eye(k_endog))
            D_t = F_inv + K_filt.T @ N_arr[t + 1] @ K_filt
            D_t = ensure_symmetric(D_t)

            # Smoothed observation disturbance: eps_hat_t = H @ u_t
            eps_hat = ssm.H @ u_t
            smoothed_obs_dist[t] = eps_hat

            # Smoothed state disturbance: eta_hat_t = Q @ R' @ r_t
            eta_hat = ssm.Q @ ssm.R.T @ r_arr[t + 1]
            smoothed_state_dist[t] = eta_hat

            # Variance of smoothed observation disturbance
            var_eps = ssm.H - ssm.H @ D_t @ ssm.H
            var_eps = ensure_symmetric(var_eps)
            obs_dist_var[t] = var_eps

            # Variance of smoothed state disturbance
            var_eta = ssm.Q - ssm.Q @ ssm.R.T @ N_arr[t + 1] @ ssm.R @ ssm.Q
            var_eta = ensure_symmetric(var_eta)
            state_dist_var[t] = var_eta

            # Auxiliary residuals (standardized)
            for p in range(k_endog):
                if var_eps[p, p] > 1e-15:
                    obs_aux_resid[t, p] = eps_hat[p] / np.sqrt(abs(var_eps[p, p]))
                else:
                    obs_aux_resid[t, p] = 0.0

            for p in range(k_posdef):
                if var_eta[p, p] > 1e-15:
                    state_aux_resid[t, p] = eta_hat[p] / np.sqrt(abs(var_eta[p, p]))
                else:
                    state_aux_resid[t, p] = 0.0

        return DisturbanceSmootherOutput(
            smoothed_obs_disturbance=smoothed_obs_dist,
            smoothed_state_disturbance=smoothed_state_dist,
            obs_disturbance_var=obs_dist_var,
            state_disturbance_var=state_dist_var,
            obs_auxiliary_residual=obs_aux_resid,
            state_auxiliary_residual=state_aux_resid,
            r=r_arr[:nobs],
            N=N_arr[:nobs],
        )
