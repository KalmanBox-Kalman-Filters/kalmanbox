"""Tests for residual diagnostics."""

import numpy as np

from kalmanbox.diagnostics.residuals import (
    auxiliary_residuals,
    recursive_residuals,
    standardized_residuals,
)
from kalmanbox.models.local_level import LocalLevel


class TestStandardizedResiduals:
    """Tests for standardized residuals."""

    def test_standardized_mean_zero(self, nile_volume: np.ndarray) -> None:
        """Mean of standardized residuals should be ~0 (tol=0.1)."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        std_resid = standardized_residuals(results)

        valid = std_resid[~np.isnan(std_resid)]
        assert abs(np.mean(valid)) < 0.1

    def test_standardized_var_one(self, nile_volume: np.ndarray) -> None:
        """Variance of standardized residuals should be ~1 (tol=0.2)."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        std_resid = standardized_residuals(results)

        valid = std_resid[~np.isnan(std_resid)]
        assert abs(np.var(valid, ddof=1) - 1.0) < 0.2

    def test_standardized_shape(self, nile_volume: np.ndarray) -> None:
        """Standardized residuals should have shape (nobs,)."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        std_resid = standardized_residuals(results)
        assert std_resid.shape == (len(nile_volume),)

    def test_standardized_no_inf(self, nile_volume: np.ndarray) -> None:
        """Standardized residuals should not contain inf."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        std_resid = standardized_residuals(results)
        valid = std_resid[~np.isnan(std_resid)]
        assert not np.any(np.isinf(valid))


class TestAuxiliaryResiduals:
    """Tests for auxiliary residuals."""

    def test_auxiliary_detects_outlier(self, nile_volume: np.ndarray) -> None:
        """Auxiliary observation residual should detect Nile 1899 outlier.

        The year 1899 (index 28) had an unusually low flow due to the
        Aswan dam construction. The auxiliary residual should flag this.
        """
        model = LocalLevel(nile_volume)
        results = model.fit()
        obs_resid, _ = auxiliary_residuals(results)

        valid = obs_resid[~np.isnan(obs_resid)]
        if len(valid) > 28:
            # Index 28 corresponds to year 1899
            # Should have a relatively large absolute value
            outlier_val = abs(obs_resid[28])
            median_val = np.nanmedian(np.abs(obs_resid))
            # The outlier should be notably larger than typical
            assert outlier_val > median_val

    def test_auxiliary_returns_tuple(self, nile_volume: np.ndarray) -> None:
        """auxiliary_residuals should return a tuple of two arrays."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        result = auxiliary_residuals(results)
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs_r, state_r = result
        assert obs_r.shape == (len(nile_volume),)
        assert state_r.shape == (len(nile_volume),)


class TestRecursiveResiduals:
    """Tests for recursive residuals."""

    def test_recursive_burn_in(self, nile_volume: np.ndarray) -> None:
        """First k_states recursive residuals should be NaN."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        rec_resid = recursive_residuals(results)

        k_states = results.ssm.k_states
        assert np.all(np.isnan(rec_resid[:k_states]))

    def test_recursive_after_burn_in(self, nile_volume: np.ndarray) -> None:
        """Recursive residuals after burn-in should be finite."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        rec_resid = recursive_residuals(results)

        k_states = results.ssm.k_states
        valid = rec_resid[k_states:]
        valid = valid[~np.isnan(valid)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))
