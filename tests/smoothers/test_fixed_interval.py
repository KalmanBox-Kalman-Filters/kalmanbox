"""Tests for FixedIntervalSmoother."""

import numpy as np
from numpy.testing import assert_allclose

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.fixed_interval import FixedIntervalSmoother


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


class TestFixedIntervalSmoother:
    """Tests for FixedIntervalSmoother."""

    def test_smoothed_state_less_variance(self, nile_volume: np.ndarray) -> None:
        """Smoothed covariance should be <= filtered covariance."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        filt = kf.filter(nile_volume, ssm)

        smoother = FixedIntervalSmoother()
        smooth_out = smoother.smooth(nile_volume, ssm, filt)

        # P_{t|T} <= P_{t|t} for all t
        for t in range(len(nile_volume) - 1):
            assert smooth_out.smoothed_cov[t, 0, 0] <= (filt.filtered_cov[t, 0, 0] + 1e-10), (
                f"Smoothed variance > filtered at t={t}"
            )

    def test_last_equals_filtered(self, nile_volume: np.ndarray) -> None:
        """Last smoothed state should equal last filtered state."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        filt = kf.filter(nile_volume, ssm)

        smoother = FixedIntervalSmoother()
        smooth_out = smoother.smooth(nile_volume, ssm, filt)

        assert_allclose(smooth_out.smoothed_state[-1], filt.filtered_state[-1], atol=1e-12)
        assert_allclose(smooth_out.smoothed_cov[-1], filt.filtered_cov[-1], atol=1e-12)

    def test_cross_covariance_computed(self, nile_volume: np.ndarray) -> None:
        """Cross-covariance should be computed when requested."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        filt = kf.filter(nile_volume, ssm)

        smoother = FixedIntervalSmoother(compute_cross_cov=True)
        smooth_out = smoother.smooth(nile_volume, ssm, filt)

        assert smooth_out.cross_cov is not None
        assert smooth_out.cross_cov.shape == (len(nile_volume), 1, 1)

    def test_cross_covariance_not_computed(self, nile_volume: np.ndarray) -> None:
        """Cross-covariance should be None when not requested."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        filt = kf.filter(nile_volume, ssm)

        smoother = FixedIntervalSmoother(compute_cross_cov=False)
        smooth_out = smoother.smooth(nile_volume, ssm, filt)

        assert smooth_out.cross_cov is None

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Output arrays must have correct shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        smoother = FixedIntervalSmoother()
        smooth_out = smoother.smooth(nile_volume, ssm)
        nobs = len(nile_volume)
        assert smooth_out.smoothed_state.shape == (nobs, 1)
        assert smooth_out.smoothed_cov.shape == (nobs, 1, 1)
        assert smooth_out.smoother_gain.shape == (nobs, 1, 1)

    def test_smoother_without_filter_output(self, nile_volume: np.ndarray) -> None:
        """Smoother should run filter internally if no filter_output given."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        smoother = FixedIntervalSmoother()
        smooth_out = smoother.smooth(nile_volume, ssm)
        assert smooth_out.smoothed_state.shape == (len(nile_volume), 1)

    def test_smoothed_cov_symmetric(self, nile_volume: np.ndarray) -> None:
        """All smoothed covariances should be symmetric."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        smoother = FixedIntervalSmoother()
        smooth_out = smoother.smooth(nile_volume, ssm)

        for t in range(len(nile_volume)):
            P = smooth_out.smoothed_cov[t]
            assert_allclose(P, P.T, atol=1e-14)

    def test_smoothed_nile_reference(self, nile_volume: np.ndarray) -> None:
        """Smoothed state for Nile should be reasonable."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        smoother = FixedIntervalSmoother()
        smooth_out = smoother.smooth(nile_volume, ssm)

        # Smoothed mean at t=0 should be pulled toward the long-run mean
        assert 800 < smooth_out.smoothed_state[0, 0] < 1200
