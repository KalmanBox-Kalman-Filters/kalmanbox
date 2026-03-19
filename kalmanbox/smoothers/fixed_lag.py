"""Fixed-lag smoother for online/real-time applications.

Produces E[alpha_t | y_{1:t+L}] for a fixed lag L, without needing
to wait for the end of the time series.

Reference: Anderson & Moore (1979), Optimal Filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput, KalmanFilter
from kalmanbox.utils.matrix_ops import ensure_symmetric, solve_via_cholesky


@dataclass
class FixedLagOutput:
    """Container for fixed-lag smoother output.

    Attributes
    ----------
    smoothed_state : NDArray
        Lag-smoothed state estimates a_{t|t+L}, shape (nobs, k_states).
    smoothed_cov : NDArray
        Lag-smoothed covariances P_{t|t+L}, shape (nobs, k_states, k_states).
    lag : int
        The smoothing lag L used.
    """

    smoothed_state: NDArray[np.float64]
    smoothed_cov: NDArray[np.float64]
    lag: int


class FixedLagSmoother:
    """Fixed-lag smoother for online applications.

    At each time t, produces E[alpha_{t-L} | y_{1:t}] by maintaining
    a window of smoother gains and applying backward smoothing within
    the window.

    Parameters
    ----------
    lag : int
        The smoothing lag L. Must be >= 1.
    """

    def __init__(self, lag: int) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.lag = lag

    def smooth(
        self,
        endog: NDArray[np.float64],
        ssm: StateSpaceRepresentation,
        filter_output: FilterOutput | None = None,
    ) -> FixedLagOutput:
        """Run the fixed-lag smoother.

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
        FixedLagOutput
            Lag-smoothed states and covariances.
        """
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        if filter_output is None:
            kf = KalmanFilter()
            filter_output = kf.filter(endog, ssm)

        nobs = endog.shape[0]
        k_states = ssm.k_states
        L = self.lag

        # Pre-allocate output
        smoothed_state = np.zeros((nobs, k_states))
        smoothed_cov = np.zeros((nobs, k_states, k_states))

        # Pre-compute all smoother gains
        gains = np.zeros((nobs, k_states, k_states))
        for t in range(nobs - 1):
            P_filt_t = filter_output.filtered_cov[t]
            P_pred_tp1 = filter_output.predicted_cov[t + 1]
            gains[t] = solve_via_cholesky(P_pred_tp1, ssm.T @ P_filt_t).T

        # For each time t, compute a_{t|t+L} by backward smoothing within window
        for t in range(nobs):
            # The last observation we can use is min(t + L, nobs - 1)
            end = min(t + L, nobs - 1)

            if end == t:
                # No lag smoothing possible
                smoothed_state[t] = filter_output.filtered_state[t]
                smoothed_cov[t] = filter_output.filtered_cov[t]
                continue

            # Start from filtered at `end` and smooth backward to `t`
            a_s = filter_output.filtered_state[end].copy()
            P_s = filter_output.filtered_cov[end].copy()

            for s in range(end - 1, t - 1, -1):
                L_s = gains[s]
                a_filt_s = filter_output.filtered_state[s]
                a_pred_sp1 = filter_output.predicted_state[s + 1]
                P_filt_s = filter_output.filtered_cov[s]
                P_pred_sp1 = filter_output.predicted_cov[s + 1]

                a_s_new = a_filt_s + L_s @ (a_s - a_pred_sp1)
                P_s_new = P_filt_s + L_s @ (P_s - P_pred_sp1) @ L_s.T
                P_s_new = ensure_symmetric(P_s_new)

                a_s = a_s_new
                P_s = P_s_new

            smoothed_state[t] = a_s
            smoothed_cov[t] = P_s

        return FixedLagOutput(
            smoothed_state=smoothed_state,
            smoothed_cov=smoothed_cov,
            lag=L,
        )
