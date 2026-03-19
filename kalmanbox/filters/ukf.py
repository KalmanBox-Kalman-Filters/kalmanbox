"""Unscented Kalman Filter (UKF) for nonlinear state-space models.

Uses sigma points to capture mean and covariance of the transformed
distribution up to 2nd order of Taylor expansion, without requiring
Jacobians.

Reference: Julier & Uhlmann (1997), A new extension of the Kalman
filter to nonlinear systems.
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
class UKFModel(Protocol):
    """Protocol for nonlinear models compatible with the UKF.

    Similar to EKFModel but does NOT require Jacobians.

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
        """Nonlinear transition function f(alpha, t)."""
        ...

    def observation(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Nonlinear observation function h(alpha, t)."""
        ...


class LinearUKFModel:
    """Adapter that wraps a StateSpaceRepresentation as a UKFModel.

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

    def observation(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Linear observation: h(alpha) = Z @ alpha + d."""
        return self._Z @ alpha + self._d


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for nonlinear state-space models.

    Uses sigma points (2n+1 for n states) to propagate mean and
    covariance through nonlinear functions without Jacobians.

    Parameters
    ----------
    alpha : float
        Spread of sigma points around the mean. Small positive value,
        typically 1e-3 to 1. Default 1e-3.
    beta : float
        Prior knowledge about distribution. beta=2 is optimal for
        Gaussian distributions. Default 2.0.
    kappa : float
        Secondary scaling parameter. Typically 0 or 3-n.
        Default 0.0.
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _compute_weights(self, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute sigma point weights.

        Parameters
        ----------
        n : int
            State dimension.

        Returns
        -------
        W_m : NDArray, shape (2*n+1,)
            Weights for mean computation.
        W_c : NDArray, shape (2*n+1,)
            Weights for covariance computation.
        """
        lam = self.alpha**2 * (n + self.kappa) - n

        W_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        W_c = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))

        W_m[0] = lam / (n + lam)
        W_c[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)

        return W_m, W_c

    def _generate_sigma_points(
        self,
        a: NDArray[np.float64],
        P: NDArray[np.float64],
        n: int,
    ) -> NDArray[np.float64]:
        """Generate 2n+1 sigma points.

        Parameters
        ----------
        a : state mean, shape (n,)
        P : state covariance, shape (n, n)
        n : state dimension

        Returns
        -------
        X : sigma points, shape (2*n+1, n)
        """
        lam = self.alpha**2 * (n + self.kappa) - n

        scaled_P = (n + lam) * P
        scaled_P = ensure_symmetric(scaled_P)

        try:
            L = np.linalg.cholesky(scaled_P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(scaled_P + 1e-10 * np.eye(n))

        X = np.zeros((2 * n + 1, n))
        X[0] = a

        for i in range(n):
            X[i + 1] = a + L[:, i]
            X[n + i + 1] = a - L[:, i]

        return X

    def filter(
        self,
        endog: NDArray[np.float64],
        model: UKFModel,
    ) -> FilterOutput:
        """Run the Unscented Kalman Filter.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).
        model : UKFModel
            Nonlinear model providing transition/observation functions.

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

        # Compute weights
        W_m, W_c = self._compute_weights(k_states)

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

                X = self._generate_sigma_points(a_prev, P_prev, k_states)

                X_pred = np.zeros_like(X)
                for j in range(2 * k_states + 1):
                    X_pred[j] = model.transition(X[j], t)

                a_pred = np.zeros(k_states)
                for j in range(2 * k_states + 1):
                    a_pred += W_m[j] * X_pred[j]

                P_pred = np.zeros((k_states, k_states))
                for j in range(2 * k_states + 1):
                    diff = X_pred[j] - a_pred
                    P_pred += W_c[j] * np.outer(diff, diff)
                P_pred += RQR
                P_pred = ensure_symmetric(P_pred)

            predicted_state[t] = a_pred
            predicted_cov[t] = P_pred

            # Check for missing data
            y_t = endog[t]
            is_missing = np.any(np.isnan(y_t))

            if is_missing:
                forecast[t] = model.observation(a_pred, t)
                filtered_state[t] = a_pred
                filtered_cov[t] = P_pred
                residuals[t] = np.nan
                forecast_cov[t] = np.nan
                loglike_obs[t] = 0.0
                continue

            # --- Update step ---
            X = self._generate_sigma_points(a_pred, P_pred, k_states)

            Y = np.zeros((2 * k_states + 1, k_endog))
            for j in range(2 * k_states + 1):
                Y[j] = model.observation(X[j], t)

            y_hat = np.zeros(k_endog)
            for j in range(2 * k_states + 1):
                y_hat += W_m[j] * Y[j]
            forecast[t] = y_hat

            S = np.zeros((k_endog, k_endog))
            for j in range(2 * k_states + 1):
                dy = Y[j] - y_hat
                S += W_c[j] * np.outer(dy, dy)
            S += model.H
            S = ensure_symmetric(S)
            forecast_cov[t] = S

            C = np.zeros((k_states, k_endog))
            for j in range(2 * k_states + 1):
                dx = X[j] - a_pred
                dy = Y[j] - y_hat
                C += W_c[j] * np.outer(dx, dy)

            K = solve_via_cholesky(S, C.T).T
            gain[t] = K

            v = y_t - y_hat
            residuals[t] = v

            a_filt = a_pred + K @ v
            P_filt = P_pred - K @ S @ K.T
            P_filt = ensure_symmetric(P_filt)

            filtered_state[t] = a_filt
            filtered_cov[t] = P_filt

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
