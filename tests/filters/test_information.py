"""Tests for InformationFilter."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.information import InformationFilter
from kalmanbox.filters.kalman import KalmanFilter


def _build_local_level_ssm(sigma2_obs: float, sigma2_level: float) -> StateSpaceRepresentation:
    """Build a local level SSM with given variances."""
    ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
    ssm.T = np.array([[1.0]])
    ssm.Z = np.array([[1.0]])
    ssm.R = np.array([[1.0]])
    ssm.H = np.array([[sigma2_obs]])
    ssm.Q = np.array([[sigma2_level]])
    ssm.a1 = np.array([0.0])
    ssm.P1 = np.array([[1e7]])
    return ssm


def _build_stationary_ar1_ssm(
    phi: float, sigma2_obs: float, sigma2_state: float
) -> StateSpaceRepresentation:
    """Build a stationary AR(1) + noise SSM."""
    ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
    ssm.T = np.array([[phi]])
    ssm.Z = np.array([[1.0]])
    ssm.R = np.array([[1.0]])
    ssm.H = np.array([[sigma2_obs]])
    ssm.Q = np.array([[sigma2_state]])
    ssm.a1 = np.array([0.0])
    ssm.P1 = np.array([[sigma2_state / (1.0 - phi**2)]])
    return ssm


class TestInformationFilter:
    """Tests for InformationFilter."""

    def test_equivalent_to_kf_local_level(self, nile_volume: np.ndarray) -> None:
        """InfoFilter must produce IDENTICAL results to KF (tol=1e-10)."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        info_f = InformationFilter(diffuse=False)
        info_out = info_f.filter(nile_volume, ssm)

        assert_allclose(
            info_out.filtered_state,
            kf_out.filtered_state,
            atol=1e-10,
            err_msg="Filtered states differ",
        )
        assert_allclose(
            info_out.filtered_cov,
            kf_out.filtered_cov,
            atol=1e-10,
            err_msg="Filtered covariances differ",
        )
        assert info_out.loglike == pytest.approx(kf_out.loglike, abs=1e-10)

    def test_equivalent_to_kf_stationary(self) -> None:
        """InfoFilter must match KF on stationary AR(1) model."""
        ssm = _build_stationary_ar1_ssm(phi=0.8, sigma2_obs=1.0, sigma2_state=0.5)
        rng = np.random.default_rng(42)
        y = rng.normal(0, 2, 100)

        kf = KalmanFilter()
        kf_out = kf.filter(y, ssm)

        info_f = InformationFilter(diffuse=False)
        info_out = info_f.filter(y, ssm)

        assert_allclose(info_out.filtered_state, kf_out.filtered_state, atol=1e-10)
        assert info_out.loglike == pytest.approx(kf_out.loglike, abs=1e-10)

    def test_diffuse_initialization(self, nile_volume: np.ndarray) -> None:
        """Diffuse init (I=0) should converge to similar results after burn-in."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        # Standard filter with large P1
        kf = KalmanFilter()
        kf_out = kf.filter(nile_volume, ssm)

        # Information filter with diffuse init
        info_f = InformationFilter(diffuse=True)
        info_out = info_f.filter(nile_volume, ssm)

        # After a few observations, filtered states should converge
        # (first few may differ due to different handling of diffuse init)
        assert_allclose(
            info_out.filtered_state[10:],
            kf_out.filtered_state[10:],
            atol=1.0,
            err_msg="Filtered states should converge after burn-in",
        )

    def test_update_is_additive(self) -> None:
        """The update step should be a simple addition of information."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        info_f = InformationFilter(diffuse=False)

        I_pred = np.array([[1.0]])
        i_pred = np.array([0.5])

        H_inv = np.linalg.inv(ssm.H)
        ZT_Hinv = ssm.Z.T @ H_inv
        ZT_Hinv_Z = ZT_Hinv @ ssm.Z

        y = np.array([1200.0])
        d = ssm.d

        I_filt, i_filt = info_f._update_step(I_pred, i_pred, y, ZT_Hinv, ZT_Hinv_Z, d)

        # I_filt = I_pred + Z' @ H^{-1} @ Z
        expected_I = I_pred + ZT_Hinv_Z
        assert_allclose(I_filt, expected_I)

        # i_filt = i_pred + Z' @ H^{-1} @ (y - d)
        expected_i = i_pred + ZT_Hinv @ (y - d)
        assert_allclose(i_filt, expected_i)

    def test_loglike_nile(self, nile_volume: np.ndarray) -> None:
        """Log-likelihood should match reference value for Nile."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        info_f = InformationFilter(diffuse=False)
        output = info_f.filter(nile_volume, ssm)
        assert output.loglike == pytest.approx(-641.59, abs=0.5)

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Output arrays must have correct shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        info_f = InformationFilter(diffuse=False)
        output = info_f.filter(nile_volume, ssm)
        nobs = len(nile_volume)
        assert output.filtered_state.shape == (nobs, 1)
        assert output.filtered_cov.shape == (nobs, 1, 1)
        assert output.predicted_state.shape == (nobs, 1)
        assert output.loglike_obs.shape == (nobs,)

    def test_missing_data(self, nile_volume: np.ndarray) -> None:
        """InfoFilter should handle missing data identically to KF."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        y_missing = nile_volume.copy()
        y_missing[10:15] = np.nan

        kf = KalmanFilter()
        kf_out = kf.filter(y_missing, ssm)

        info_f = InformationFilter(diffuse=False)
        info_out = info_f.filter(y_missing, ssm)

        assert info_out.nobs_effective == kf_out.nobs_effective
        assert_allclose(info_out.filtered_state, kf_out.filtered_state, atol=1e-10)

    def test_info_to_state_diffuse(self) -> None:
        """Converting I=0 to state space should give large P and zero a."""
        P, a = InformationFilter._info_to_state(np.zeros((2, 2)), np.zeros(2))
        assert P[0, 0] == 1e7
        assert P[1, 1] == 1e7
        assert_allclose(a, [0.0, 0.0])

    def test_nobs_effective(self, nile_volume: np.ndarray) -> None:
        """Effective observations should equal total for complete data."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        info_f = InformationFilter(diffuse=False)
        output = info_f.filter(nile_volume, ssm)
        assert output.nobs_effective == len(nile_volume)
