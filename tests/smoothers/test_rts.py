"""Tests for RTSSmoother."""

import numpy as np

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.rts import RTSSmoother


def _build_local_level_ssm(sigma2_obs: float, sigma2_level: float) -> StateSpaceRepresentation:
    """Build a local level SSM."""
    ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
    ssm.T = np.array([[1.0]])
    ssm.Z = np.array([[1.0]])
    ssm.R = np.array([[1.0]])
    ssm.H = np.array([[sigma2_obs]])
    ssm.Q = np.array([[sigma2_level]])
    ssm.a1 = np.array([0.0])
    ssm.P1 = np.array([[1e7]])
    return ssm


class TestRTSSmoother:
    """Tests for RTSSmoother."""

    def test_smoothed_leq_filtered_variance(self, nile_volume: np.ndarray) -> None:
        """Smoothed variance <= filtered variance for all t.

        This is a fundamental property of smoothing.
        """
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        smoother = RTSSmoother()

        filter_output = kf.filter(nile_volume, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        for t in range(len(nile_volume)):
            P_filt = filter_output.filtered_cov[t, 0, 0]
            P_smooth = smoother_output.smoothed_cov[t, 0, 0]
            assert P_smooth <= P_filt + 1e-10, (
                f"t={t}: smoothed var {P_smooth:.2f} > filtered var {P_filt:.2f}"
            )

    def test_smoothed_state_smoother_than_filtered(self, nile_volume: np.ndarray) -> None:
        """Smoothed state should be less variable than filtered state."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        smoother = RTSSmoother()

        filter_output = kf.filter(nile_volume, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        # Smoothed states use more info, so changes between steps are smaller
        diff_filtered = np.diff(filter_output.filtered_state[:, 0])
        diff_smoothed = np.diff(smoother_output.smoothed_state[:, 0])
        assert np.var(diff_smoothed) <= np.var(diff_filtered)

    def test_last_smoothed_equals_last_filtered(self, nile_volume: np.ndarray) -> None:
        """At t=T, smoothed state must equal filtered state."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        smoother = RTSSmoother()

        filter_output = kf.filter(nile_volume, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        np.testing.assert_allclose(
            smoother_output.smoothed_state[-1],
            filter_output.filtered_state[-1],
        )
        np.testing.assert_allclose(
            smoother_output.smoothed_cov[-1],
            filter_output.filtered_cov[-1],
        )

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Test output array shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        smoother = RTSSmoother()

        filter_output = kf.filter(nile_volume, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        nobs = len(nile_volume)
        assert smoother_output.smoothed_state.shape == (nobs, 1)
        assert smoother_output.smoothed_cov.shape == (nobs, 1, 1)
        assert smoother_output.smoother_gain.shape == (nobs, 1, 1)

    def test_smoothed_cov_symmetric(self, nile_volume: np.ndarray) -> None:
        """Smoothed covariances must be symmetric."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        smoother = RTSSmoother()

        filter_output = kf.filter(nile_volume, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        for t in range(len(nile_volume)):
            P = smoother_output.smoothed_cov[t]
            np.testing.assert_allclose(P, P.T, atol=1e-12)

    def test_multivariate_smoother(self) -> None:
        """Test smoother with 2 states."""
        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
        ssm.T = np.array([[1.0, 1.0], [0.0, 1.0]])
        ssm.Z = np.array([[1.0, 0.0]])
        ssm.R = np.eye(2)
        ssm.H = np.array([[10.0]])
        ssm.Q = np.diag([1.0, 0.1])
        ssm.a1 = np.zeros(2)
        ssm.P1 = np.eye(2) * 1e7

        rng = np.random.default_rng(42)
        y = rng.normal(0, 5, size=50)

        kf = KalmanFilter()
        smoother = RTSSmoother()
        filter_output = kf.filter(y, ssm)
        smoother_output = smoother.smooth(filter_output, ssm)

        assert smoother_output.smoothed_state.shape == (50, 2)
        assert smoother_output.smoothed_cov.shape == (50, 2, 2)

        # Smoothed variance should be less than or equal to filtered
        for t in range(50):
            for s in range(2):
                assert smoother_output.smoothed_cov[t, s, s] <= (
                    filter_output.filtered_cov[t, s, s] + 1e-10
                )
