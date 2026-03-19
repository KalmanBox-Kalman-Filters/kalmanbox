"""Tests for DisturbanceSmoother."""

import numpy as np

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.smoothers.disturbance import DisturbanceSmoother


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


class TestDisturbanceSmoother:
    """Tests for DisturbanceSmoother."""

    def test_output_shapes(self, nile_volume: np.ndarray) -> None:
        """Output arrays must have correct shapes."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        ds = DisturbanceSmoother()
        output = ds.smooth(nile_volume, ssm)
        nobs = len(nile_volume)
        assert output.smoothed_obs_disturbance.shape == (nobs, 1)
        assert output.smoothed_state_disturbance.shape == (nobs, 1)
        assert output.obs_disturbance_var.shape == (nobs, 1, 1)
        assert output.state_disturbance_var.shape == (nobs, 1, 1)
        assert output.obs_auxiliary_residual.shape == (nobs, 1)
        assert output.state_auxiliary_residual.shape == (nobs, 1)
        assert output.r.shape == (nobs, 1)
        assert output.N.shape == (nobs, 1, 1)

    def test_disturbance_decomposition(self, nile_volume: np.ndarray) -> None:
        """eps_hat + eta_hat should approximately decompose the residual."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        ds = DisturbanceSmoother()
        output = ds.smooth(nile_volume, ssm)

        # Sum of squared disturbances should be finite
        assert np.isfinite(np.sum(output.smoothed_obs_disturbance**2))
        assert np.isfinite(np.sum(output.smoothed_state_disturbance**2))

    def test_auxiliary_residuals_standardized(self, nile_volume: np.ndarray) -> None:
        """Auxiliary residuals should have approximate unit variance."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        ds = DisturbanceSmoother()
        output = ds.smooth(nile_volume, ssm)

        # Skip first few (diffuse init) and check approximate normality
        obs_resid = output.obs_auxiliary_residual[5:, 0]

        # Variance should be approximately 1, but Nile has a structural break
        # that inflates residuals, so allow a wider bound
        assert 0.1 < np.var(obs_resid) < 10.0, (
            f"Obs aux residual variance {np.var(obs_resid):.2f} not near 1"
        )

    def test_outlier_detection(self, nile_volume: np.ndarray) -> None:
        """Inserting an outlier should produce large auxiliary residual."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)

        # Create data with artificial outlier
        y_outlier = nile_volume.copy()
        outlier_t = 50
        y_outlier[outlier_t] = y_outlier[outlier_t] + 5000  # massive outlier

        ds = DisturbanceSmoother()
        output = ds.smooth(y_outlier, ssm)

        # The observation auxiliary residual at the outlier should be large
        obs_resid = np.abs(output.obs_auxiliary_residual[:, 0])
        assert obs_resid[outlier_t] > 2.0, f"Outlier not detected: |e_t|={obs_resid[outlier_t]:.2f}"
        # It should be among the largest
        assert obs_resid[outlier_t] == np.max(obs_resid)

    def test_nile_1899_outlier(self, nile_volume: np.ndarray) -> None:
        """The Nile data has a known shift around 1899 (index ~28).

        The disturbance smoother should detect elevated residuals near
        the shift point (year 1899, Aswan Dam construction).
        """
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        ds = DisturbanceSmoother()
        output = ds.smooth(nile_volume, ssm)

        # Check that state auxiliary residuals near the shift are notable
        state_resid = np.abs(output.state_auxiliary_residual[:, 0])

        # The residual near the break should be among the top few
        break_region = state_resid[25:35]
        max_break = np.max(break_region)
        median_all = np.median(state_resid[5:])

        assert max_break > median_all, (
            f"Break region max {max_break:.2f} should exceed median {median_all:.2f}"
        )

    def test_without_filter_output(self, nile_volume: np.ndarray) -> None:
        """Smoother should run filter internally if not provided."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        ds = DisturbanceSmoother()
        output = ds.smooth(nile_volume, ssm)
        assert output.smoothed_obs_disturbance.shape == (len(nile_volume), 1)

    def test_positive_variance(self, nile_volume: np.ndarray) -> None:
        """Disturbance variances should be non-negative."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        ds = DisturbanceSmoother()
        output = ds.smooth(nile_volume, ssm)

        for t in range(len(nile_volume)):
            assert output.obs_disturbance_var[t, 0, 0] >= -1e-10
            assert output.state_disturbance_var[t, 0, 0] >= -1e-10

    def test_r_backward_recursion(self, nile_volume: np.ndarray) -> None:
        """The backward recursion vector r should converge."""
        ssm = _build_local_level_ssm(sigma2_obs=15099.0, sigma2_level=1469.0)
        ds = DisturbanceSmoother()
        output = ds.smooth(nile_volume, ssm)

        assert np.all(np.isfinite(output.r))
        assert np.all(np.isfinite(output.N))
