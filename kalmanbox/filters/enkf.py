"""Ensemble Kalman Filter (EnKF) for high-dimensional state-space models.

Maintains an ensemble of N members and estimates covariance from
sample statistics, making it tractable for systems with hundreds
or thousands of states.

Reference: Evensen, G. (2003). The Ensemble Kalman Filter: theoretical
formulation and practical implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput
from kalmanbox.utils.matrix_ops import (
    ensure_symmetric,
    log_det_via_cholesky,
    solve_via_cholesky,
)


@runtime_checkable
class EnKFModel(Protocol):
    """Protocol for models compatible with the EnKF.

    Attributes
    ----------
    k_states : int
    k_endog : int
    R : NDArray
    Q : NDArray
    H : NDArray
    a1 : NDArray
    P1 : NDArray
    """

    k_states: int
    k_endog: int
    R: NDArray[np.float64]
    Q: NDArray[np.float64]
    H: NDArray[np.float64]
    a1: NDArray[np.float64]
    P1: NDArray[np.float64]

    def transition(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Transition function (may be nonlinear)."""
        ...

    def observation(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Observation function (may be nonlinear)."""
        ...


class LinearEnKFModel:
    """Adapter that wraps a StateSpaceRepresentation for the EnKF.

    Parameters
    ----------
    ssm : StateSpaceRepresentation
        Linear state-space model.
    """

    def __init__(self, ssm: StateSpaceRepresentation) -> None:
        self._ssm = ssm
        self.k_states: int = ssm.k_states
        self.k_endog: int = ssm.k_endog
        self.R: NDArray[np.float64] = ssm.R
        self.Q: NDArray[np.float64] = ssm.Q
        self.H: NDArray[np.float64] = ssm.H
        self.a1: NDArray[np.float64] = ssm.a1
        self.P1: NDArray[np.float64] = ssm.P1
        self._T: NDArray[np.float64] = ssm.T
        self._Z: NDArray[np.float64] = ssm.Z
        self._c: NDArray[np.float64] = ssm.c
        self._d: NDArray[np.float64] = ssm.d

    def transition(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Linear transition: T @ alpha + c."""
        return self._T @ alpha + self._c

    def observation(self, alpha: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Linear observation: Z @ alpha + d."""
        return self._Z @ alpha + self._d


class EnsembleKalmanFilter:
    """Ensemble Kalman Filter for high-dimensional systems.

    Uses an ensemble of N members to approximate the state distribution,
    avoiding the need to propagate the full (k_states x k_states) covariance
    matrix.

    Parameters
    ----------
    n_ensemble : int
        Number of ensemble members. Default 100.
    inflation : float
        Covariance inflation factor. Multiply anomalies by this factor
        to compensate for ensemble undersampling. Default 1.0 (no inflation).
    random_state : int | np.random.Generator | None
        Random seed or generator for reproducibility. Default None.
    """

    def __init__(
        self,
        n_ensemble: int = 100,
        inflation: float = 1.0,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_ensemble = n_ensemble
        self.inflation = inflation

        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng(random_state)

    def filter(
        self,
        endog: NDArray[np.float64],
        model: EnKFModel,
    ) -> FilterOutput:
        """Run the Ensemble Kalman Filter.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).
        model : EnKFModel
            Model providing transition/observation functions.

        Returns
        -------
        FilterOutput
            Complete filter output. Note that covariances are estimated
            from the ensemble and may differ slightly from the exact KF
            for finite ensemble sizes.
        """
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        nobs = endog.shape[0]
        k_states = model.k_states
        k_endog = model.k_endog
        N = self.n_ensemble

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

        # Cholesky factors for noise generation
        Q_chol = np.linalg.cholesky(model.Q)
        H_chol = np.linalg.cholesky(model.H)

        # Initialize ensemble: E^{(j)} ~ N(a1, P1)
        P1_chol = np.linalg.cholesky(model.P1)
        ensemble = np.zeros((N, k_states))
        for j in range(N):
            z = self.rng.standard_normal(k_states)
            ensemble[j] = model.a1 + P1_chol @ z

        loglike_total = 0.0
        nobs_effective = 0

        for t in range(nobs):
            # --- Prediction step ---
            if t > 0:
                # Propagate each member through the transition
                ensemble_pred = np.zeros((N, k_states))
                for j in range(N):
                    # State noise for this member
                    eta = Q_chol @ self.rng.standard_normal(model.Q.shape[0])
                    ensemble_pred[j] = model.transition(ensemble[j], t) + model.R @ eta
                ensemble = ensemble_pred

            # Apply inflation
            if self.inflation != 1.0:
                ens_mean = np.mean(ensemble, axis=0)
                ensemble = ens_mean + self.inflation * (ensemble - ens_mean)

            # Ensemble statistics
            a_pred = np.mean(ensemble, axis=0)
            A = (ensemble - a_pred).T  # anomaly matrix (k_states, N)
            P_pred = (A @ A.T) / (N - 1)
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
            # Predicted observations for each ensemble member
            Y_pred = np.zeros((N, k_endog))
            for j in range(N):
                Y_pred[j] = model.observation(ensemble[j], t)

            Y_bar = np.mean(Y_pred, axis=0)
            forecast[t] = Y_bar

            # Observation anomaly matrix
            D = (Y_pred - Y_bar).T  # (k_endog, N)

            # Forecast covariance
            S = (D @ D.T) / (N - 1) + model.H
            S = ensure_symmetric(S)
            forecast_cov[t] = S

            # Cross-covariance
            C = (A @ D.T) / (N - 1)  # (k_states, k_endog)

            # Kalman gain
            K = solve_via_cholesky(S, C.T).T  # K = C @ S^{-1}
            gain[t] = K

            # Innovation
            v = y_t - Y_bar
            residuals[t] = v

            # Perturbed observations update (stochastic EnKF)
            for j in range(N):
                eps = H_chol @ self.rng.standard_normal(k_endog)
                y_perturbed = y_t + eps
                innovation_j = y_perturbed - Y_pred[j]
                ensemble[j] = ensemble[j] + K @ innovation_j

            # Filtered ensemble statistics
            a_filt = np.mean(ensemble, axis=0)
            A_filt = (ensemble - a_filt).T
            P_filt = (A_filt @ A_filt.T) / (N - 1)
            P_filt = ensure_symmetric(P_filt)

            filtered_state[t] = a_filt
            filtered_cov[t] = P_filt

            # Log-likelihood (approximate, based on ensemble statistics)
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
