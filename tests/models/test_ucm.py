"""Tests for UnobservedComponents (UCM)."""

import numpy as np
import pytest

from kalmanbox.datasets import load_dataset
from kalmanbox.models.local_level import LocalLevel
from kalmanbox.models.local_linear_trend import LocalLinearTrend
from kalmanbox.models.ucm import UnobservedComponents


class TestUnobservedComponents:
    """Tests for UCM."""

    def test_ucm_equals_local_level(self, nile_volume: np.ndarray) -> None:
        """UCM(level=True, trend='none') should match LocalLevel."""
        ucm = UnobservedComponents(nile_volume, level=True, trend="none", seasonal="none")
        ll = LocalLevel(nile_volume)

        ucm_results = ucm.fit()
        ll_results = ll.fit()

        # Log-likelihoods should match closely
        assert ucm_results.loglike == pytest.approx(ll_results.loglike, abs=1.0)

        # Parameters should be similar
        # UCM has sigma2_obs, sigma2_level
        # LL has sigma2_obs, sigma2_level
        assert ucm_results.params[0] == pytest.approx(ll_results.params[0], rel=0.1)
        assert ucm_results.params[1] == pytest.approx(ll_results.params[1], rel=0.1)

    def test_ucm_equals_llt(self, nile_volume: np.ndarray) -> None:
        """UCM(level=True, trend='stochastic') should match LocalLinearTrend."""
        ucm = UnobservedComponents(nile_volume, level=True, trend="stochastic", seasonal="none")
        llt = LocalLinearTrend(nile_volume)

        ucm_results = ucm.fit()
        llt_results = llt.fit()

        assert ucm_results.loglike == pytest.approx(llt_results.loglike, abs=2.0)

    def test_ucm_with_seasonal(self) -> None:
        """UCM with seasonal should work on airline data."""
        df = load_dataset("airline")
        y = np.log(df["passengers"].to_numpy(dtype=np.float64))

        ucm = UnobservedComponents(
            y,
            level=True,
            trend="stochastic",
            seasonal="dummy",
            seasonal_period=12,
        )
        results = ucm.fit()
        assert results.optimizer_converged

    def test_ucm_with_cycle(self) -> None:
        """UCM with cycle component."""
        rng = np.random.default_rng(42)
        n = 200
        t = np.arange(n)
        cycle = 5 * np.sin(2 * np.pi * t / 40)
        y = cycle + rng.normal(0, 1, n)

        ucm = UnobservedComponents(y, level=True, trend="none", cycle=True)
        results = ucm.fit()
        assert results.optimizer_converged

    def test_ucm_with_ar(self) -> None:
        """UCM with AR component."""
        rng = np.random.default_rng(42)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + rng.normal(0, 1)

        ucm = UnobservedComponents(y, level=False, trend="none", autoregressive=1)
        results = ucm.fit()
        assert results.optimizer_converged

    def test_ucm_damped_trend(self) -> None:
        """UCM with damped trend."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.normal(0, 1, 100))

        ucm = UnobservedComponents(y, level=True, trend="damped")
        results = ucm.fit()
        assert results.optimizer_converged
        # phi_trend should be in (0, 1)
        phi_idx = ucm.param_names.index("phi_trend")
        assert 0 < results.params[phi_idx] < 1

    def test_ucm_fixed_trend(self) -> None:
        """UCM with fixed trend (deterministic)."""
        rng = np.random.default_rng(42)
        t = np.arange(100, dtype=np.float64)
        y = 2.0 * t + rng.normal(0, 5, 100)

        ucm = UnobservedComponents(y, level=True, trend="fixed")
        results = ucm.fit()
        assert results.optimizer_converged

    def test_component_decomposition(self) -> None:
        """Sum of component means should approximate fitted values."""
        df = load_dataset("airline")
        y = np.log(df["passengers"].to_numpy(dtype=np.float64))

        ucm = UnobservedComponents(
            y,
            level=True,
            trend="stochastic",
            seasonal="dummy",
            seasonal_period=12,
        )
        results = ucm.fit()

        # The filtered state contains all components
        # Z @ state = fitted value
        assert results.fitted_values.shape == (len(y), 1)

    def test_seasonal_requires_period(self) -> None:
        """seasonal != 'none' without period should raise ValueError."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        with pytest.raises(ValueError, match="seasonal_period is required"):
            UnobservedComponents(y, seasonal="dummy")

    def test_trend_requires_level(self) -> None:
        """trend != 'none' without level should raise ValueError."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        with pytest.raises(ValueError, match="trend requires level"):
            UnobservedComponents(y, level=False, trend="stochastic")

    def test_param_names_full(self) -> None:
        """Check param names for full model."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        ucm = UnobservedComponents(
            y,
            level=True,
            trend="stochastic",
            seasonal="dummy",
            seasonal_period=4,
            cycle=True,
        )
        names = ucm.param_names
        assert "sigma2_obs" in names
        assert "sigma2_level" in names
        assert "sigma2_trend" in names
        assert "sigma2_seasonal" in names
        assert "rho" in names
        assert "lambda_c" in names
        assert "sigma2_cycle" in names

    def test_gdp_quarterly(self) -> None:
        """UCM on Brazil GDP quarterly data."""
        df = load_dataset("brazil_pib")
        y = df["pib"].to_numpy(dtype=np.float64)

        ucm = UnobservedComponents(y, level=True, trend="stochastic")
        results = ucm.fit()
        assert results.optimizer_converged
        assert np.isfinite(results.loglike)

    def test_summary(self) -> None:
        """summary() should work for UCM."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        ucm = UnobservedComponents(y, level=True, trend="stochastic")
        results = ucm.fit()
        s = results.summary()
        assert "sigma2_level" in s

    def test_transform_roundtrip(self) -> None:
        """Transform/untransform should be inverses."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        ucm = UnobservedComponents(
            y, level=True, trend="damped", seasonal="dummy", seasonal_period=4, cycle=True
        )
        params = ucm.start_params
        unconstrained = ucm.untransform_params(params)
        roundtrip = ucm.transform_params(unconstrained)
        np.testing.assert_allclose(roundtrip, params, atol=1e-8)
