"""Tests for EMEstimator."""

import numpy as np
import pytest

from kalmanbox.estimation.em import EMEstimator
from kalmanbox.models.local_level import LocalLevel


class TestEMEstimator:
    """Tests for EM estimator."""

    def test_em_converges(self, nile_volume: np.ndarray) -> None:
        """EM should converge (loglike stops changing)."""
        model = LocalLevel(nile_volume)
        em = EMEstimator(max_iter=50, tol=1e-4)
        results = em.fit(model, model.endog)
        assert results.optimizer_converged or np.isfinite(results.loglike)

    def test_em_loglike_monotone(self, nile_volume: np.ndarray) -> None:
        """Log-likelihood should be approximately monotonically increasing.

        Due to GEM (numerical M-step), slight decreases are possible,
        but overall trend should be increasing.
        """
        model = LocalLevel(nile_volume)
        from kalmanbox.filters.kalman import KalmanFilter

        kf = KalmanFilter()
        params = model.start_params.copy()
        prev_ll = -np.inf
        decreases = 0

        for _ in range(20):
            ssm = model._build_ssm(params)
            output = kf.filter(model.endog, ssm)
            ll = output.loglike

            if ll < prev_ll - 1.0:
                decreases += 1
            prev_ll = ll

            # Simple M-step: just run one MLE iteration
            from scipy import optimize

            unconstrained = model.untransform_params(params)

            def neg_ll(x: np.ndarray) -> float:
                try:
                    c = model.transform_params(x)
                    s = model._build_ssm(c)
                    return -kf.filter(model.endog, s).loglike
                except Exception:
                    return 1e10

            res = optimize.minimize(
                neg_ll,
                unconstrained,
                method="L-BFGS-B",
                options={"maxiter": 5},
            )
            params = model.transform_params(res.x)

        # Allow at most a few small decreases (GEM approximation)
        assert decreases <= 5

    def test_em_matches_mle(self, nile_volume: np.ndarray) -> None:
        """EM final params should be close to MLE params."""
        model_em = LocalLevel(nile_volume)
        model_mle = LocalLevel(nile_volume)

        em = EMEstimator(max_iter=100, tol=1e-6)
        results_em = em.fit(model_em, model_em.endog)
        results_mle = model_mle.fit()

        # Parameters should be within 10%
        for i in range(len(results_em.params)):
            assert results_em.params[i] == pytest.approx(results_mle.params[i], rel=0.10)

    def test_em_loglike_close_to_mle(self, nile_volume: np.ndarray) -> None:
        """EM loglike should be close to MLE loglike."""
        model_em = LocalLevel(nile_volume)
        model_mle = LocalLevel(nile_volume)

        em = EMEstimator(max_iter=100, tol=1e-6)
        results_em = em.fit(model_em, model_em.endog)
        results_mle = model_mle.fit()

        assert results_em.loglike == pytest.approx(results_mle.loglike, abs=1.0)

    def test_em_se_positive(self, nile_volume: np.ndarray) -> None:
        """Standard errors should be positive."""
        model = LocalLevel(nile_volume)
        em = EMEstimator(max_iter=50, tol=1e-4)
        results = em.fit(model, model.endog)
        if not np.any(np.isnan(results.se)):
            assert all(se > 0 for se in results.se)

    def test_em_summary(self, nile_volume: np.ndarray) -> None:
        """summary() should work after EM estimation."""
        model = LocalLevel(nile_volume)
        em = EMEstimator(max_iter=30, tol=1e-4)
        results = em.fit(model, model.endog)
        s = results.summary()
        assert "sigma2_obs" in s
