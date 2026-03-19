"""Tests for RegressionSSM."""

import numpy as np
import pytest

from kalmanbox.models.regression_ssm import RegressionSSM


class TestRegressionSSM:
    """Tests for RegressionSSM."""

    def _simulate_regression(
        self,
        n: int = 200,
        beta: list[float] | None = None,
        sigma2: float = 1.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate regression data: y, X, true_beta."""
        if beta is None:
            beta = [2.0, -1.5, 0.5]
        rng = np.random.default_rng(seed)
        k = len(beta)
        X = rng.normal(0, 1, (n, k))
        # Add intercept as first column
        X[:, 0] = 1.0
        true_beta = np.array(beta)
        eps = rng.normal(0, np.sqrt(sigma2), n)
        y = X @ true_beta + eps
        return y, X, true_beta

    def test_ols_recovery(self) -> None:
        """Estimated beta should match OLS exactly."""
        y, X, true_beta = self._simulate_regression(n=500)
        model = RegressionSSM(y, X)
        results = model.fit()

        # OLS reference
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(results.params[:3], beta_ols, atol=1e-10)

    def test_beta_close_to_true(self) -> None:
        """With enough data, beta should be close to true values."""
        y, X, true_beta = self._simulate_regression(n=1000, sigma2=0.5)
        model = RegressionSSM(y, X)
        results = model.fit()

        for i in range(len(true_beta)):
            assert results.params[i] == pytest.approx(true_beta[i], abs=0.2)

    def test_sigma2_recovery(self) -> None:
        """Estimated sigma2 should be close to true value."""
        true_sigma2 = 2.0
        y, X, _ = self._simulate_regression(n=500, sigma2=true_sigma2)
        model = RegressionSSM(y, X)
        results = model.fit()
        sigma2_est = results.params[-1]
        assert sigma2_est == pytest.approx(true_sigma2, rel=0.15)

    def test_se_positive(self) -> None:
        """Standard errors should be positive."""
        y, X, _ = self._simulate_regression()
        model = RegressionSSM(y, X)
        results = model.fit()
        assert all(se > 0 for se in results.se)

    def test_param_names(self) -> None:
        """Parameter names should include betas and sigma2."""
        y, X, _ = self._simulate_regression()
        model = RegressionSSM(y, X)
        assert model.param_names == ["beta_0", "beta_1", "beta_2", "sigma2_obs"]

    def test_summary(self) -> None:
        """summary() should work."""
        y, X, _ = self._simulate_regression()
        model = RegressionSSM(y, X)
        results = model.fit()
        s = results.summary()
        assert "beta_0" in s
        assert "sigma2_obs" in s

    def test_single_regressor(self) -> None:
        """Should work with a single regressor."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 1, n)
        y = 3.0 * x + rng.normal(0, 0.5, n)
        X = x.reshape(-1, 1)

        model = RegressionSSM(y, X)
        results = model.fit()
        assert results.params[0] == pytest.approx(3.0, abs=0.3)

    def test_loglike_formula(self) -> None:
        """Loglike should match the analytical formula."""
        y, X, _ = self._simulate_regression(n=100)
        model = RegressionSSM(y, X)
        results = model.fit()
        n = 100
        sigma2 = results.params[-1]
        expected_ll = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma2) - n / 2
        assert results.loglike == pytest.approx(expected_ll, rel=1e-10)

    def test_forecast(self) -> None:
        """Forecast should work (state-space based)."""
        y, X, _ = self._simulate_regression()
        model = RegressionSSM(y, X)
        results = model.fit()
        fc = results.forecast(steps=5)
        assert fc["mean"].shape[0] == 5
