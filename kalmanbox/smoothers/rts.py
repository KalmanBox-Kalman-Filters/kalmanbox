"""Rauch-Tung-Striebel fixed-interval smoother."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput
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
        Smoother gain L_t, shape (nobs, k_states, k_states).
    """

    smoothed_state: NDArray[np.float64]
    smoothed_cov: NDArray[np.float64]
    smoother_gain: NDArray[np.float64]


class RTSSmoother:
    """Rauch-Tung-Striebel fixed-interval smoother.

    Given filtered and predicted estimates from the Kalman filter,
    produces smoothed state estimates using the full sample y_{1:T}.
    """

    def smooth(
        self,
        filter_output: FilterOutput,
        ssm: StateSpaceRepresentation,
    ) -> SmootherOutput:
        """Run the RTS smoother.

        Parameters
        ----------
        filter_output : FilterOutput
            Output from KalmanFilter.filter().
        ssm : StateSpaceRepresentation
            State-space model specification.

        Returns
        -------
        SmootherOutput
            Smoothed state estimates and covariances.
        """
        nobs = filter_output.filtered_state.shape[0]
        k_states = ssm.k_states

        smoothed_state = np.zeros((nobs, k_states))
        smoothed_cov = np.zeros((nobs, k_states, k_states))
        smoother_gain = np.zeros((nobs, k_states, k_states))

        # Initialize: last smoothed = last filtered
        smoothed_state[-1] = filter_output.filtered_state[-1]
        smoothed_cov[-1] = filter_output.filtered_cov[-1]

        # Backward recursion
        for t in range(nobs - 2, -1, -1):
            P_filt = filter_output.filtered_cov[t]
            P_pred_next = filter_output.predicted_cov[t + 1]

            # Smoother gain: L_t = P_{t|t} @ T' @ P_{t+1|t}^{-1}
            # Compute via Cholesky solve: L_t = (P_{t+1|t}^{-1} @ T @ P_{t|t})'
            # which is equivalent to solving P_{t+1|t} @ X = T @ P_{t|t}
            L_t = solve_via_cholesky(P_pred_next, ssm.T @ P_filt).T

            smoother_gain[t] = L_t

            # Smoothed state
            a_diff = smoothed_state[t + 1] - filter_output.predicted_state[t + 1]
            smoothed_state[t] = filter_output.filtered_state[t] + L_t @ a_diff

            # Smoothed covariance
            P_diff = smoothed_cov[t + 1] - P_pred_next
            smoothed_cov[t] = P_filt + L_t @ P_diff @ L_t.T
            smoothed_cov[t] = ensure_symmetric(smoothed_cov[t])

        return SmootherOutput(
            smoothed_state=smoothed_state,
            smoothed_cov=smoothed_cov,
            smoother_gain=smoother_gain,
        )
