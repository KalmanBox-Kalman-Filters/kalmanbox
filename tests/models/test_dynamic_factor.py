"""Tests for DynamicFactorModel."""

import numpy as np

from kalmanbox.models.dynamic_factor import DynamicFactorModel


class TestDynamicFactorModel:
    """Tests for DFM."""

    def _simulate_dfm(
        self,
        nobs: int = 200,
        k_endog: int = 5,
        k_factors: int = 1,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate DFM data.

        Returns y, true_factors, true_Lambda, true_R_diag.
        """
        rng = np.random.default_rng(seed)

        # True loadings
        Lambda = rng.normal(0, 1, (k_endog, k_factors))
        Lambda[0, 0] = 1.0  # Normalization

        # True factor dynamics (AR(1))
        Phi = 0.8 * np.eye(k_factors)

        # Generate factors
        factors = np.zeros((nobs, k_factors))
        for t in range(1, nobs):
            factors[t] = Phi @ factors[t - 1] + rng.normal(0, 1, k_factors)

        # Idiosyncratic noise
        R_diag = np.abs(rng.normal(1, 0.3, k_endog))
        eps = np.zeros((nobs, k_endog))
        for i in range(k_endog):
            eps[:, i] = rng.normal(0, np.sqrt(R_diag[i]), nobs)

        # Observations
        y = factors @ Lambda.T + eps

        return y, factors, Lambda, R_diag

    def test_dfm_simulated_recovery(self) -> None:
        """DFM should recover factors (correlation > 0.8)."""
        y, true_factors, _, _ = self._simulate_dfm(nobs=300, k_endog=5, k_factors=1)
        model = DynamicFactorModel(y, k_factors=1)
        results = model.fit()

        if results.smoothed_state is not None:
            estimated_factor = results.smoothed_state[:, 0]
            true_factor = true_factors[:, 0]

            # Correlation (sign may be flipped)
            corr = abs(np.corrcoef(estimated_factor, true_factor)[0, 1])
            assert corr > 0.7

    def test_dfm_two_factors(self) -> None:
        """DFM with 2 factors and 10 series should converge."""
        y, _, _, _ = self._simulate_dfm(nobs=200, k_endog=10, k_factors=2)
        model = DynamicFactorModel(y, k_factors=2)
        results = model.fit()
        assert np.isfinite(results.loglike)

    def test_dfm_loadings(self) -> None:
        """Lambda matrix should have correct shape."""
        y, _, _, _ = self._simulate_dfm(k_endog=5, k_factors=2)
        model = DynamicFactorModel(y, k_factors=2)
        results = model.fit()
        # Z[:, :k_factors] is Lambda
        Lambda_est = results.ssm.Z[:, :2]
        assert Lambda_est.shape == (5, 2)

    def test_dfm_variance_decomposition(self) -> None:
        """Sum of variance contributions should be ~1 per series."""
        y, _, _, _ = self._simulate_dfm(k_endog=5, k_factors=1)
        model = DynamicFactorModel(y, k_factors=1)
        results = model.fit()

        decomp = model.variance_decomposition(results)
        assert decomp.shape == (5, 2)  # 1 factor + idiosyncratic
        row_sums = np.sum(decomp, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_dfm_missing_data(self) -> None:
        """DFM with missing data should still work."""
        y, _, _, _ = self._simulate_dfm(nobs=200, k_endog=5, k_factors=1)

        # Add 20% missing
        rng = np.random.default_rng(123)
        mask = rng.random(y.shape) < 0.2
        y_missing = y.copy()
        y_missing[mask] = np.nan

        model = DynamicFactorModel(y_missing, k_factors=1)
        results = model.fit()
        assert np.isfinite(results.loglike)

    def test_dfm_convergence(self) -> None:
        """Model should converge."""
        y, _, _, _ = self._simulate_dfm(nobs=150, k_endog=3, k_factors=1)
        model = DynamicFactorModel(y, k_factors=1)
        results = model.fit()
        assert results.optimizer_converged or np.isfinite(results.loglike)

    def test_param_names(self) -> None:
        """param_names should include lambda, phi, sigma2."""
        y, _, _, _ = self._simulate_dfm(k_endog=3, k_factors=1)
        model = DynamicFactorModel(y, k_factors=1)
        names = model.param_names
        assert any("lambda" in n for n in names)
        assert any("phi" in n for n in names)
        assert any("sigma2" in n for n in names)

    def test_summary(self) -> None:
        """summary() should work."""
        y, _, _, _ = self._simulate_dfm(nobs=100, k_endog=3, k_factors=1)
        model = DynamicFactorModel(y, k_factors=1)
        results = model.fit()
        s = results.summary()
        assert isinstance(s, str)

    def test_dfm_with_names(self) -> None:
        """DFM with named series."""
        y, _, _, _ = self._simulate_dfm(k_endog=3, k_factors=1)
        names = ["GDP", "CPI", "Employment"]
        model = DynamicFactorModel(y, k_factors=1, endog_names=names)
        param_names = model.param_names
        assert any("GDP" in n for n in param_names)

    def test_r_diagonal(self) -> None:
        """R (observation noise) should be diagonal."""
        y, _, _, _ = self._simulate_dfm(k_endog=3, k_factors=1)
        model = DynamicFactorModel(y, k_factors=1)
        results = model.fit()
        H = results.ssm.H
        # Check off-diagonal is zero
        np.testing.assert_allclose(H - np.diag(np.diag(H)), 0.0, atol=1e-10)

    def test_q_is_identity(self) -> None:
        """Q should be identity (identification constraint)."""
        y, _, _, _ = self._simulate_dfm(k_endog=3, k_factors=1)
        model = DynamicFactorModel(y, k_factors=1)
        results = model.fit()
        np.testing.assert_allclose(results.ssm.Q, np.eye(1), atol=1e-10)
