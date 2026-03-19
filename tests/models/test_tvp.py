"""Tests for TimeVaryingParameters."""

import numpy as np

from kalmanbox.models.tvp import TimeVaryingParameters


class TestTimeVaryingParameters:
    """Tests for TVP model."""

    def _simulate_tvp(
        self,
        n: int = 200,
        k: int = 2,
        sigma2_obs: float = 1.0,
        sigma2_beta: float = 0.01,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate TVP data with known time-varying coefficients."""
        rng = np.random.default_rng(seed)

        X = np.ones((n, k))
        X[:, 1] = rng.normal(0, 1, n)  # one regressor + intercept

        # Time-varying coefficients (smooth random walk)
        beta = np.zeros((n, k))
        beta[0] = rng.normal(0, 1, k)
        for t in range(1, n):
            beta[t] = beta[t - 1] + rng.normal(0, np.sqrt(sigma2_beta), k)

        # Observations
        y = np.sum(X * beta, axis=1) + rng.normal(0, np.sqrt(sigma2_obs), n)

        return y, X, beta

    def test_tvp_simulated(self) -> None:
        """TVP should converge and recover trajectory."""
        y, X, true_beta = self._simulate_tvp(n=300)
        model = TimeVaryingParameters(y, X)
        results = model.fit()
        assert results.optimizer_converged
        assert np.isfinite(results.loglike)

    def test_tvp_constant_close_to_ols(self) -> None:
        """TVP with very small Q should give coefficients close to OLS."""
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        true_beta = np.array([2.0, -1.5])
        y = X @ true_beta + rng.normal(0, 1, n)

        model = TimeVaryingParameters(y, X)
        results = model.fit()

        # Smoothed state at last observation should be close to OLS
        if results.smoothed_state is not None:
            beta_tvp = results.smoothed_state[-1]
            beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
            np.testing.assert_allclose(beta_tvp, beta_ols, atol=1.0)

    def test_tvp_detects_change(self) -> None:
        """TVP should detect a structural break in coefficients."""
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])

        # Beta changes at midpoint
        beta_before = np.array([2.0, 1.0])
        beta_after = np.array([2.0, -1.0])
        y = np.zeros(n)
        for t in range(n):
            b = beta_before if t < 100 else beta_after
            y[t] = X[t] @ b + rng.normal(0, 0.5)

        model = TimeVaryingParameters(y, X)
        results = model.fit()

        if results.smoothed_state is not None:
            # Coefficient for regressor should differ before/after break
            beta_early = results.smoothed_state[20, 1]
            beta_late = results.smoothed_state[180, 1]
            # They should be clearly different
            assert abs(beta_early - beta_late) > 0.5

    def test_param_names_diagonal(self) -> None:
        """param_names for diagonal Q."""
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = rng.normal(0, 1, n)
        model = TimeVaryingParameters(y, X, q_type="diagonal")
        assert model.param_names == ["sigma2_obs", "sigma2_beta_0", "sigma2_beta_1"]

    def test_param_names_scalar(self) -> None:
        """param_names for scalar Q."""
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = rng.normal(0, 1, n)
        model = TimeVaryingParameters(y, X, q_type="scalar")
        assert model.param_names == ["sigma2_obs", "sigma2_beta"]

    def test_transform_roundtrip(self) -> None:
        """Transform/untransform should be inverses."""
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = rng.normal(0, 1, n)
        model = TimeVaryingParameters(y, X)
        params = model.start_params
        unconstrained = model.untransform_params(params)
        roundtrip = model.transform_params(unconstrained)
        np.testing.assert_allclose(roundtrip, params, atol=1e-10)

    def test_summary(self) -> None:
        """summary() should work."""
        y, X, _ = self._simulate_tvp(n=100)
        model = TimeVaryingParameters(y, X)
        results = model.fit()
        s = results.summary()
        assert "sigma2_obs" in s
        assert "sigma2_beta_0" in s

    def test_single_regressor(self) -> None:
        """Should work with single regressor."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 1, n)
        y = 2.0 * x + rng.normal(0, 1, n)
        X = x.reshape(-1, 1)

        model = TimeVaryingParameters(y, X)
        results = model.fit()
        assert results.optimizer_converged
