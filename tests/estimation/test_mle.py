"""Tests for MLEstimator."""

import numpy as np
import pytest

from kalmanbox.models.local_level import LocalLevel


class TestMLEstimator:
    """Tests for MLE estimation."""

    def test_convergence(self, nile_volume: np.ndarray) -> None:
        """Optimizer should converge."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        assert results.optimizer_converged

    def test_se_positive(self, nile_volume: np.ndarray) -> None:
        """Standard errors should all be positive."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        assert all(se > 0 for se in results.se)

    def test_aic_bic_formula(self, nile_volume: np.ndarray) -> None:
        """AIC = -2*loglike + 2*k, BIC = -2*loglike + k*log(n)."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        k = 2
        n = 100
        assert results.aic == pytest.approx(-2 * results.loglike + 2 * k, rel=1e-10)
        assert results.bic == pytest.approx(-2 * results.loglike + k * np.log(n), rel=1e-10)

    def test_nile_loglike(self, nile_volume: np.ndarray) -> None:
        """Optimized loglike should be close to -632.54."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        # The optimized loglike should be >= -632.54 (or very close)
        assert results.loglike == pytest.approx(-632.54, abs=1.0)

    def test_nile_params(self, nile_volume: np.ndarray) -> None:
        """Estimated params should match Durbin & Koopman reference values."""
        model = LocalLevel(nile_volume)
        results = model.fit()
        sigma2_obs = results.params[0]
        sigma2_level = results.params[1]
        assert sigma2_obs == pytest.approx(15099.0, rel=0.05)
        assert sigma2_level == pytest.approx(1469.0, rel=0.05)
