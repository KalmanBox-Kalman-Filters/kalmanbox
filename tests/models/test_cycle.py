"""Tests for CycleModel."""

import numpy as np
import pytest

from kalmanbox.models.cycle import CycleModel


class TestCycleModel:
    """Tests for CycleModel."""

    def _simulate_cycle(
        self,
        n: int = 300,
        period: float = 40.0,
        rho: float = 0.95,
        sigma2_cycle: float = 1.0,
        sigma2_obs: float = 0.5,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate data from a known cycle model."""
        rng = np.random.default_rng(seed)
        lambda_c = 2 * np.pi / period

        c = np.zeros(n)
        c_star = np.zeros(n)

        for t in range(1, n):
            kappa = rng.normal(0, np.sqrt(sigma2_cycle))
            kappa_star = rng.normal(0, np.sqrt(sigma2_cycle))
            c[t] = (
                rho * np.cos(lambda_c) * c[t - 1] + rho * np.sin(lambda_c) * c_star[t - 1] + kappa
            )
            c_star[t] = (
                -rho * np.sin(lambda_c) * c[t - 1]
                + rho * np.cos(lambda_c) * c_star[t - 1]
                + kappa_star
            )

        eps = rng.normal(0, np.sqrt(sigma2_obs), n)
        return c + eps

    def test_cycle_period_recovery(self) -> None:
        """Estimated lambda_c should recover the true period."""
        true_period = 40.0
        y = self._simulate_cycle(n=400, period=true_period, rho=0.95)

        model = CycleModel(y)
        results = model.fit()

        lambda_c_est = results.params[2]
        period_est = 2 * np.pi / lambda_c_est

        # Should be within 20% of true period
        assert period_est == pytest.approx(true_period, rel=0.2)

    def test_rho_recovery(self) -> None:
        """Estimated rho should be close to true value."""
        y = self._simulate_cycle(n=400, rho=0.9)
        model = CycleModel(y)
        results = model.fit()
        rho_est = results.params[1]
        assert rho_est == pytest.approx(0.9, abs=0.15)

    def test_param_names(self) -> None:
        """Parameter names should be correct."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        model = CycleModel(y)
        assert model.param_names == ["sigma2_obs", "rho", "lambda_c", "sigma2_cycle"]

    def test_transform_roundtrip(self) -> None:
        """Transform/untransform should be inverses."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        model = CycleModel(y)
        params = np.array([1.0, 0.9, np.pi / 4, 0.5])
        unconstrained = model.untransform_params(params)
        roundtrip = model.transform_params(unconstrained)
        np.testing.assert_allclose(roundtrip, params, atol=1e-10)

    def test_convergence(self) -> None:
        """Model should converge on simulated data."""
        y = self._simulate_cycle()
        model = CycleModel(y)
        results = model.fit()
        assert results.optimizer_converged

    def test_rho_in_range(self) -> None:
        """Estimated rho should be in (0, 1)."""
        y = self._simulate_cycle()
        model = CycleModel(y)
        results = model.fit()
        assert 0 < results.params[1] < 1

    def test_lambda_in_range(self) -> None:
        """Estimated lambda_c should be in (0, pi)."""
        y = self._simulate_cycle()
        model = CycleModel(y)
        results = model.fit()
        assert 0 < results.params[2] < np.pi

    def test_forecast(self) -> None:
        """Forecast should produce oscillating pattern."""
        y = self._simulate_cycle(n=200, period=20, rho=0.98)
        model = CycleModel(y)
        results = model.fit()
        fc = results.forecast(steps=40)
        means = fc["mean"][:, 0]
        # Should show oscillation (not monotone)
        diffs = np.diff(means)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        assert sign_changes >= 2  # at least some oscillation

    def test_summary(self) -> None:
        """summary() should work."""
        y = self._simulate_cycle(n=100)
        model = CycleModel(y)
        results = model.fit()
        s = results.summary()
        assert "rho" in s
        assert "lambda_c" in s
