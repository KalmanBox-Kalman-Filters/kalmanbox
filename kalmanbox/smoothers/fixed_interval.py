"""Fixed-interval smoother with cross-covariance for EM.

Extends the RTS (Rauch-Tung-Striebel) smoother with computation of
P_{t,t-1|T}, the cross-covariance needed for the M-step of EM.

Reference: Rauch, Tung & Striebel (1965).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput, KalmanFilter
from kalmanbox.utils.matrix_ops import ensure_symmetric, solve_via_cholesky


@dataclass
class SmootherOutput:
    """Container for smoother output.

    Attributes
    ----------
    smoothed_state : NDArray
        Smoothed state estimates a_{t|T}, shape (nobs, k_states).
    smoothed_cov : NDArray
        Smoothed state covariances P_{t|T}, shape (nobs, k_states, k_states).
    smoother_gain : NDArray
        Smoother gains L_t, shape (nobs, k_states, k_states).
    cross_cov : NDArray | None
        Cross-covariances P_{t,t-1|T}, shape (nobs, k_states, k_states).
        Only computed if compute_cross_cov=True.
    """

    smoothed_state: NDArray[np.float64]
    smoothed_cov: NDArray[np.float64]
    smoother_gain: NDArray[np.float64]
    cross_cov: NDArray[np.float64] | None = None


class FixedIntervalSmoother:
    """Fixed-interval (RTS) smoother with optional cross-covariance.

    Runs the Kalman filter forward, then applies the RTS backward
    recursion. Optionally computes the cross-covariance P_{t,t-1|T}
    needed for the EM algorithm.

    Parameters
    ----------
    compute_cross_cov : bool
        If True, compute P_{t,t-1|T} for EM. Default True.
    """

    def __init__(self, compute_cross_cov: bool = True) -> None:
        self.compute_cross_cov = compute_cross_cov

    def smooth(
        self,
        endog: NDArray[np.float64],
        ssm: StateSpaceRepresentation,
        filter_output: FilterOutput | None = None,
    ) -> SmootherOutput:
        """Run the fixed-interval smoother.

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
        SmootherOutput
            Smoothed states, covariances, and optionally cross-covariances.
        """
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        # Run forward filter if not provided
        if filter_output is None:
            kf = KalmanFilter()
            filter_output = kf.filter(endog, ssm)

        nobs = endog.shape[0]
        k_states = ssm.k_states

        # Pre-allocate
        smoothed_state = np.zeros((nobs, k_states))
        smoothed_cov = np.zeros((nobs, k_states, k_states))
        smoother_gain = np.zeros((nobs, k_states, k_states))
        cross_cov = None
        if self.compute_cross_cov:
            cross_cov = np.zeros((nobs, k_states, k_states))

        # Initialize: last smoothed = last filtered
        smoothed_state[-1] = filter_output.filtered_state[-1]
        smoothed_cov[-1] = filter_output.filtered_cov[-1]

        # Backward recursion
        for t in range(nobs - 2, -1, -1):
            P_filt_t = filter_output.filtered_cov[t]
            P_pred_tp1 = filter_output.predicted_cov[t + 1]

            # Smoother gain: L_t = P_{t|t} @ T' @ P_{t+1|t}^{-1}
            L_t = solve_via_cholesky(P_pred_tp1, ssm.T @ P_filt_t).T

            smoother_gain[t] = L_t

            # Smoothed state
            a_filt_t = filter_output.filtered_state[t]
            a_pred_tp1 = filter_output.predicted_state[t + 1]
            a_smooth_tp1 = smoothed_state[t + 1]

            smoothed_state[t] = a_filt_t + L_t @ (a_smooth_tp1 - a_pred_tp1)

            # Smoothed covariance
            P_smooth_tp1 = smoothed_cov[t + 1]
            smoothed_cov[t] = P_filt_t + L_t @ (P_smooth_tp1 - P_pred_tp1) @ L_t.T
            smoothed_cov[t] = ensure_symmetric(smoothed_cov[t])

            # Cross-covariance: P_{t+1,t|T} = P_{t+1|T} @ L_t'
            if self.compute_cross_cov and cross_cov is not None:
                cross_cov[t + 1] = smoothed_cov[t + 1] @ L_t.T

        return SmootherOutput(
            smoothed_state=smoothed_state,
            smoothed_cov=smoothed_cov,
            smoother_gain=smoother_gain,
            cross_cov=cross_cov,
        )
