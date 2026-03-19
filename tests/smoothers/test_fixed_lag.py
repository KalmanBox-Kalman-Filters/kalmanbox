"""Tests for FixedLagSmoother."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.fixed_interval import FixedIntervalSmoother
from kalmanbox.smoothers.fixed_lag import FixedLagSmoother


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


class TestFixedLagSmoother:
    """Tests for FixedLagSmoother."""

    def test_invalid_lag(self) -> None:
        """Lag < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="lag must be >= 1"):
            FixedLagSmoother(lag=0)

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Output arrays must have correct shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        smoother = FixedLagSmoother(lag=5)
        output = smoother.smooth(nile_volume, ssm)
        nobs = len(nile_volume)
        assert output.smoothed_state.shape == (nobs, 1)
        assert output.smoothed_cov.shape == (nobs, 1, 1)
        assert output.lag == 5

    def test_lag_1_reduces_variance(self, nile_volume: np.ndarray) -> None:
        """Lag-1 smoothed variance should be <= filtered variance."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        filt = kf.filter(nile_volume, ssm)

        smoother = FixedLagSmoother(lag=1)
        output = smoother.smooth(nile_volume, ssm, filt)

        # For most observations, smoothed variance <= filtered
        for t in range(len(nile_volume) - 1):
            assert output.smoothed_cov[t, 0, 0] <= (filt.filtered_cov[t, 0, 0] + 1e-10)

    def test_larger_lag_closer_to_fixed_interval(self, nile_volume: np.ndarray) -> None:
        """Very large lag should approximate fixed-interval smoother."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        # Fixed interval smoother (full smoothing)
        fi_smoother = FixedIntervalSmoother()
        fi_out = fi_smoother.smooth(nile_volume, ssm)

        # Fixed lag with lag = nobs (equivalent to full smoothing)
        nobs = len(nile_volume)
        fl_smoother = FixedLagSmoother(lag=nobs)
        fl_out = fl_smoother.smooth(nile_volume, ssm)

        # Should be very close to fixed interval result
        assert_allclose(fl_out.smoothed_state, fi_out.smoothed_state, atol=1e-8)

    def test_increasing_lag_decreases_variance(self, nile_volume: np.ndarray) -> None:
        """Larger lag should produce equal or smaller variance."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        fl1 = FixedLagSmoother(lag=1)
        out1 = fl1.smooth(nile_volume, ssm)

        fl5 = FixedLagSmoother(lag=5)
        out5 = fl5.smooth(nile_volume, ssm)

        # Mean variance for lag=5 should be <= mean variance for lag=1
        mean_var_1 = np.mean(out1.smoothed_cov[10:-10, 0, 0])
        mean_var_5 = np.mean(out5.smoothed_cov[10:-10, 0, 0])
        assert mean_var_5 <= mean_var_1 + 1e-6

    def test_last_observations_equal_filtered(self, nile_volume: np.ndarray) -> None:
        """Last L observations should have less smoothing."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        kf = KalmanFilter()
        filt = kf.filter(nile_volume, ssm)

        smoother = FixedLagSmoother(lag=5)
        output = smoother.smooth(nile_volume, ssm, filt)

        # The very last observation should equal filtered
        assert_allclose(output.smoothed_state[-1], filt.filtered_state[-1], atol=1e-10)

    def test_without_filter_output(self, nile_volume: np.ndarray) -> None:
        """Smoother should run filter internally if not provided."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        smoother = FixedLagSmoother(lag=3)
        output = smoother.smooth(nile_volume, ssm)
        assert output.smoothed_state.shape == (len(nile_volume), 1)
