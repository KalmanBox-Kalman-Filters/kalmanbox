"""Residual diagnostics for state-space models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kalmanbox.core.results import StateSpaceResults


def standardized_residuals(
    results: StateSpaceResults,
) -> NDArray[np.float64]:
    """Compute standardized residuals e_t = v_t / sqrt(F_t).

    Under a correctly specified model, standardized residuals
    should be approximately i.i.d. N(0, 1).

    Parameters
    ----------
    results : StateSpaceResults
        Fitted model results containing filter output.

    Returns
    -------
    NDArray[np.float64]
        Standardized residuals, shape (nobs,).
    """
    v = results.filter_output.residuals  # (nobs, k_endog)
    F = results.filter_output.forecast_cov  # (nobs, k_endog, k_endog)

    nobs = v.shape[0]
    std_resid = np.full(nobs, np.nan)

    for t in range(nobs):
        if np.any(np.isnan(v[t])):
            continue
        # For univariate case: e_t = v_t / sqrt(F_t)
        if v.shape[1] == 1:
            f_t = F[t, 0, 0]
            if f_t > 0:
                std_resid[t] = v[t, 0] / np.sqrt(f_t)
        else:
            # Multivariate: use Cholesky
            try:
                L = np.linalg.cholesky(F[t])
                std_resid_vec = np.linalg.solve(L, v[t])
                std_resid[t] = np.linalg.norm(std_resid_vec)
            except np.linalg.LinAlgError:
                continue

    return std_resid


def auxiliary_residuals(
    results: StateSpaceResults,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute auxiliary residuals via the disturbance smoother.

    Auxiliary residuals detect:
    - Observation-level outliers (observation residuals)
    - State-level structural breaks (state residuals)

    Based on Koopman (1993) and Harvey & Koopman (1992).

    Parameters
    ----------
    results : StateSpaceResults
        Fitted model results containing smoother output.

    Returns
    -------
    obs_residuals : NDArray[np.float64]
        Standardized observation disturbance residuals, shape (nobs,).
        Values > 2 in absolute value suggest outliers.
    state_residuals : NDArray[np.float64]
        Standardized state disturbance residuals, shape (nobs,).
        Values > 2 in absolute value suggest structural breaks.
    """
    v = results.filter_output.residuals
    F = results.filter_output.forecast_cov
    K = results.filter_output.gain
    ssm = results.ssm

    nobs = v.shape[0]
    k_states = ssm.k_states

    Z = ssm.Z
    T = ssm.T
    H = ssm.H
    Q = ssm.Q
    R = ssm.R

    obs_residuals = np.full(nobs, np.nan)
    state_residuals = np.full(nobs, np.nan)

    r_next = np.zeros(k_states)
    N_next = np.zeros((k_states, k_states))

    for t in range(nobs - 1, -1, -1):
        if np.any(np.isnan(v[t])):
            # Missing observation: r_{t-1} = T' r_t, N_{t-1} = T' N_t T
            r_next = T.T @ r_next
            N_next = T.T @ N_next @ T
            continue

        F_t = F[t]
        v_t = v[t]
        K_t = K[t]

        # F_t inverse
        if v.shape[1] == 1:
            F_inv = np.array([[1.0 / F_t[0, 0]]]) if F_t[0, 0] > 0 else np.zeros((1, 1))
        else:
            try:
                F_inv = np.linalg.inv(F_t)
            except np.linalg.LinAlgError:
                r_next = T.T @ r_next
                N_next = T.T @ N_next @ T
                continue

        # L_t = T - K_t Z
        L_t = T - K_t @ Z

        # r_{t-1} = Z' F_t^{-1} v_t + L_t' r_t
        r_curr = Z.T @ F_inv @ v_t + L_t.T @ r_next

        # N_{t-1} = Z' F_t^{-1} Z + L_t' N_t L_t
        N_curr = Z.T @ F_inv @ Z + L_t.T @ N_next @ L_t

        # Observation disturbance (univariate simplification)
        if v.shape[1] == 1:
            h = H[0, 0]
            f_val = F_t[0, 0]
            if f_val > 0 and h > 0:
                e_hat = h * (F_inv @ v_t - K_t.T @ r_next)
                D_t = F_inv + K_t.T @ N_next @ K_t
                var_e = h - h**2 * D_t[0, 0]
                if var_e > 1e-10:
                    obs_residuals[t] = float(e_hat[0]) / np.sqrt(float(var_e))

        # State disturbance
        # eta_hat = Q R' r_t
        eta_hat = Q @ R.T @ r_next
        # Var(eta_hat) = Q - Q R' N_t R Q
        var_eta = Q - Q @ R.T @ N_next @ R @ Q
        if var_eta.shape[0] == 1:
            if var_eta[0, 0] > 1e-10:
                state_residuals[t] = float(eta_hat[0]) / np.sqrt(float(var_eta[0, 0]))
        else:
            diag_var = np.diag(var_eta)
            if np.all(diag_var > 1e-10):
                state_residuals[t] = float(np.linalg.norm(eta_hat / np.sqrt(diag_var)))

        r_next = r_curr
        N_next = N_curr

    return obs_residuals, state_residuals


def recursive_residuals(
    results: StateSpaceResults,
) -> NDArray[np.float64]:
    """Compute recursive residuals (one-step-ahead prediction errors).

    These are the basis for CUSUM and CUSUMSQ tests. They are
    computed as v_t / sqrt(F_t) after a burn-in period where the
    filter has stabilized.

    Parameters
    ----------
    results : StateSpaceResults
        Fitted model results.

    Returns
    -------
    NDArray[np.float64]
        Recursive residuals, shape (nobs,).
        First `d` observations are NaN where d = k_states (burn-in).
    """
    std_resid = standardized_residuals(results)
    k_states = results.ssm.k_states

    # Set burn-in period to NaN
    rec_resid = std_resid.copy()
    rec_resid[:k_states] = np.nan

    return rec_resid
