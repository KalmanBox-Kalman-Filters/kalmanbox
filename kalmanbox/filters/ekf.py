"""Extended Kalman Filter (EKF) for nonlinear state-space models.

Linearizes the transition and observation functions using Jacobians
at each time step, then applies the standard Kalman filter equations.

Reference: Anderson & Moore (1979), Optimal Filtering.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from kalmanbox.filters.kalman import FilterOutput
from kalmanbox.utils.matrix_ops import (
    ensure_symmetric,
    log_det_via_cholesky,
    solve_via_cholesky,
)


@runtime_checkable
class EKFModel(Protocol):
    """Protocol for nonlinear models compatible with the EKF.

    Models must provide transition/observation functions and their
    Jacobians, plus the noise covariance matrices.

    Attributes
    ----------
    k_states : int
        Number of state variables.
    k_endog : int
        Number of observed variables.
    R : NDArray
        Selection matrix, shape (k_states, k_posdef).
    Q : NDArray
        State disturbance covariance, shape (k_posdef, k_posdef).
    H : NDArray
        Observation disturbance covariance, shape (k_endog, k_endog).
    a1 : NDArray
        Initial state mean, shape (k_states,).
    P1 : NDArray
        Initial state covariance, shape (k_states, k_states).
    """

    k_states: int
    k_endog: int
    R: NDArray[np.float64]
    Q: NDArray[np.float64]
    H: NDArray[np.float64]
    a1: NDArray[np.float64]
    P1: NDArray[np.float64]

    def transition(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Nonlinear transition function f(alpha, t).

        Parameters
        ----------
        alpha : current state, shape (k_states,)
        t : time index

        Returns
        -------
        NDArray : predicted state, shape (k_states,)
        """
        ...

    def transition_jacobian(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Jacobian of transition function df/dalpha at (alpha, t).

        Parameters
        ----------
        alpha : current state, shape (k_states,)
        t : time index

        Returns
        -------
        NDArray : Jacobian matrix, shape (k_states, k_states)
        """
        ...

    def observation(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Nonlinear observation function h(alpha, t).

        Parameters
        ----------
        alpha : current state, shape (k_states,)
        t : time index

        Returns
        -------
        NDArray : predicted observation, shape (k_endog,)
        """
        ...

    def observation_jacobian(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Jacobian of observation function dh/dalpha at (alpha, t).

        Parameters
        ----------
        alpha : current state, shape (k_states,)
        t : time index

        Returns
        -------
        NDArray : Jacobian matrix, shape (k_endog, k_states)
        """
        ...


class LinearEKFModel:
    """Adapter that wraps a StateSpaceRepresentation as an EKFModel.

    This allows using the EKF on linear models, which should produce
    results identical to the standard KalmanFilter.

    Parameters
    ----------
    ssm : StateSpaceRepresentation
        Linear state-space model.
    """

    def __init__(self, ssm: object) -> None:
        self._ssm = ssm
        self.k_states: int = ssm.k_states  # type: ignore[attr-defined]
        self.k_endog: int = ssm.k_endog  # type: ignore[attr-defined]
        self.R: NDArray[np.float64] = ssm.R  # type: ignore[attr-defined]
        self.Q: NDArray[np.float64] = ssm.Q  # type: ignore[attr-defined]
        self.H: NDArray[np.float64] = ssm.H  # type: ignore[attr-defined]
        self.a1: NDArray[np.float64] = ssm.a1  # type: ignore[attr-defined]
        self.P1: NDArray[np.float64] = ssm.P1  # type: ignore[attr-defined]
        self._T: NDArray[np.float64] = ssm.T  # type: ignore[attr-defined]
        self._Z: NDArray[np.float64] = ssm.Z  # type: ignore[attr-defined]
        self._c: NDArray[np.float64] = ssm.c  # type: ignore[attr-defined]
        self._d: NDArray[np.float64] = ssm.d  # type: ignore[attr-defined]

    def transition(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Linear transition: f(alpha) = T @ alpha + c."""
        return self._T @ alpha + self._c

    def transition_jacobian(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Jacobian of linear transition is constant T."""
        return self._T

    def observation(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Linear observation: h(alpha) = Z @ alpha + d."""
        return self._Z @ alpha + self._d

    def observation_jacobian(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Jacobian of linear observation is constant Z."""
        return self._Z


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear state-space models.

    Linearizes the nonlinear transition and observation functions
    using their Jacobians at each time step, then applies the
    standard Kalman filter prediction-update recursion.

    On a linear model (using LinearEKFModel), produces results
    IDENTICAL to the standard KalmanFilter.
    """

    def filter(
        self,
        endog: NDArray[np.float64],
        model: EKFModel,
    ) -> FilterOutput:
        """Run the Extended Kalman Filter.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).
        model : EKFModel
            Nonlinear model providing transition/observation functions
            and their Jacobians.

        Returns
        -------
        FilterOutput
            Complete filter output.
        """
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        nobs = endog.shape[0]
        k_states = model.k_states
        k_endog = model.k_endog

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

        # State noise covariance
        RQR = model.R @ model.Q @ model.R.T

        # Initialize
        a = model.a1.copy()
        P = model.P1.copy()

        loglike_total = 0.0
        nobs_effective = 0

        for t in range(nobs):
            # --- Prediction step ---
            if t == 0:
                a_pred = a
                P_pred = P
            else:
                a_prev = filtered_state[t - 1]
                P_prev = filtered_cov[t - 1]

                # Nonlinear transition
                a_pred = model.transition(a_prev, t)

                # Jacobian at previous filtered state
                F_t = model.transition_jacobian(a_prev, t)

                # Predicted covariance using Jacobian
                P_pred = F_t @ P_prev @ F_t.T + RQR
                P_pred = ensure_symmetric(P_pred)

            predicted_state[t] = a_pred
            predicted_cov[t] = P_pred

            # --- Forecast ---
            # Nonlinear observation function
            y_pred = model.observation(a_pred, t)
            forecast[t] = y_pred

            # Check for missing data
            y_t = endog[t]
            is_missing = np.any(np.isnan(y_t))

            if is_missing:
                filtered_state[t] = a_pred
                filtered_cov[t] = P_pred
                residuals[t] = np.nan
                forecast_cov[t] = np.nan
                loglike_obs[t] = 0.0
            else:
                # Jacobian of observation at predicted state
                H_t = model.observation_jacobian(a_pred, t)

                # Prediction error
                v = y_t - y_pred

                # Prediction error covariance
                S = H_t @ P_pred @ H_t.T + model.H
                S = ensure_symmetric(S)

                # Kalman gain
                K = solve_via_cholesky(S, H_t @ P_pred.T).T

                # Filtered state
                a_filt = a_pred + K @ v

                # Filtered covariance
                P_filt = P_pred - K @ H_t @ P_pred
                P_filt = ensure_symmetric(P_filt)

                filtered_state[t] = a_filt
                filtered_cov[t] = P_filt
                residuals[t] = v
                forecast_cov[t] = S
                gain[t] = K

                # Log-likelihood
                log_det_S = log_det_via_cholesky(S)
                v_S_inv_v = float(v @ solve_via_cholesky(S, v))
                ll_t = -0.5 * (k_endog * np.log(2.0 * np.pi) + log_det_S + v_S_inv_v)
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
