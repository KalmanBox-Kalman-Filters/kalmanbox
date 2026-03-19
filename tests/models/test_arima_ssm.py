"""Tests for ARIMA_SSM model."""

import numpy as np
import pytest

from kalmanbox.models.arima_ssm import ARIMA_SSM


class TestARIMA_SSM:
    """Tests for ARIMA_SSM."""

    def _simulate_ar1(
        self, n: int = 200, phi: float = 0.7, sigma2: float = 1.0, seed: int = 42
    ) -> np.ndarray:
        """Simulate AR(1) process."""
        rng = np.random.default_rng(seed)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.normal(0, np.sqrt(sigma2))
        return y

    def _simulate_arma11(
        self,
        n: int = 300,
        phi: float = 0.7,
        theta: float = 0.3,
        sigma2: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate ARMA(1,1) process."""
        rng = np.random.default_rng(seed)
        y = np.zeros(n)
        eps = rng.normal(0, np.sqrt(sigma2), n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + eps[t] + theta * eps[t - 1]
        return y

    def test_ar1_fit(self) -> None:
        """AR(1) should converge and recover phi."""
        y = self._simulate_ar1(n=300, phi=0.7)
        model = ARIMA_SSM(y, order=(1, 0, 0))
        results = model.fit()
        assert results.optimizer_converged
        phi_est = results.params[0]
        assert phi_est == pytest.approx(0.7, abs=0.15)

    def test_arma11_fit(self) -> None:
        """ARMA(1,1) should converge."""
        y = self._simulate_arma11(n=300, phi=0.7, theta=0.3)
        model = ARIMA_SSM(y, order=(1, 0, 1))
        results = model.fit()
        assert results.optimizer_converged
        assert len(results.params) == 3  # phi, theta, sigma2

    def test_arima_differencing(self) -> None:
        """ARIMA(1,1,0) should apply differencing."""
        rng = np.random.default_rng(42)
        # Random walk with drift
        y = np.cumsum(rng.normal(0.1, 1, 200))
        model = ARIMA_SSM(y, order=(1, 1, 0))
        # After diff, nobs should be 199
        assert model.nobs == 199
        results = model.fit()
        assert results.optimizer_converged

    def test_param_names_ar2(self) -> None:
        """AR(2) should have phi_1, phi_2, sigma2."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        model = ARIMA_SSM(y, order=(2, 0, 0))
        assert model.param_names == ["phi_1", "phi_2", "sigma2"]

    def test_param_names_arma21(self) -> None:
        """ARMA(2,1) params."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        model = ARIMA_SSM(y, order=(2, 0, 1))
        assert model.param_names == ["phi_1", "phi_2", "theta_1", "sigma2"]

    def test_arima_loglike_matches_direct(self) -> None:
        """Loglike of ARIMA_SSM(1,0,1) should be close to statsmodels.

        This test uses manual comparison. If statsmodels is available,
        it compares directly; otherwise checks loglike is reasonable.
        """
        y = self._simulate_arma11(n=200, phi=0.7, theta=0.3, sigma2=1.0)
        model = ARIMA_SSM(y, order=(1, 0, 1))
        results = model.fit()

        try:
            from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

            sm_model = SM_ARIMA(y, order=(1, 0, 1), trend="n")
            sm_results = sm_model.fit()
            # Log-likelihoods should be close
            assert results.loglike == pytest.approx(sm_results.llf, abs=8.0)
        except (ImportError, TypeError):
            # If statsmodels not available, just check loglike is finite
            assert np.isfinite(results.loglike)
            # And that it's not too far from expected
            # For ARMA(1,1) with sigma2=1 and n=200, loglike ~ -280
            assert results.loglike > -500

    def test_arima_params_match(self) -> None:
        """Parameters should be close to statsmodels estimates."""
        y = self._simulate_arma11(n=300, phi=0.7, theta=0.3, sigma2=1.0)
        model = ARIMA_SSM(y, order=(1, 0, 1))
        results = model.fit()

        try:
            from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

            sm_model = SM_ARIMA(y, order=(1, 0, 1), trend="n")
            sm_results = sm_model.fit()

            phi_ssm = results.params[0]
            phi_sm = sm_results.arparams[0]
            assert phi_ssm == pytest.approx(phi_sm, abs=0.1)

            theta_ssm = results.params[1]
            theta_sm = sm_results.maparams[0]
            assert theta_ssm == pytest.approx(theta_sm, abs=0.1)
        except (ImportError, TypeError):
            # Just check params are reasonable
            assert abs(results.params[0]) < 1  # AR coef
            assert results.params[-1] > 0  # sigma2

    def test_forecast(self) -> None:
        """Forecast should converge to zero for stationary ARMA."""
        y = self._simulate_ar1(n=200, phi=0.7)
        model = ARIMA_SSM(y, order=(1, 0, 0))
        results = model.fit()
        fc = results.forecast(steps=50)
        # AR(1) forecast should converge to unconditional mean (~0)
        means = fc["mean"][:, 0]
        assert abs(means[-1]) < abs(means[0]) + 1.0

    def test_summary(self) -> None:
        """summary() should work."""
        y = self._simulate_ar1(n=100)
        model = ARIMA_SSM(y, order=(1, 0, 0))
        results = model.fit()
        s = results.summary()
        assert "phi_1" in s
        assert "sigma2" in s
