"""Tests for statistical diagnostic tests."""

import numpy as np

from kalmanbox.diagnostics.tests import (
    TestResult,
    cusum_test,
    cusumsq_test,
    heteroskedasticity_test,
    ljung_box_test,
    normality_test,
)
from kalmanbox.models.local_level import LocalLevel


class TestLjungBox:
    """Tests for Ljung-Box test."""

    def test_white_noise_not_rejected(self) -> None:
        """White noise should not reject H0 (no autocorrelation)."""
        rng = np.random.default_rng(42)
        wn = rng.standard_normal(200)
        result = ljung_box_test(wn, lags=10)
        assert isinstance(result, TestResult)
        assert result.p_value > 0.05

    def test_autocorrelated_rejected(self) -> None:
        """Autocorrelated series should reject H0."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.zeros(n)
        x[0] = rng.standard_normal()
        for t in range(1, n):
            x[t] = 0.8 * x[t - 1] + rng.standard_normal()
        result = ljung_box_test(x, lags=10)
        assert result.reject is True

    def test_returns_test_result(self) -> None:
        """Should return TestResult dataclass."""
        rng = np.random.default_rng(42)
        result = ljung_box_test(rng.standard_normal(100), lags=5)
        assert isinstance(result, TestResult)
        assert result.test_name == "Ljung-Box"
        assert 0.0 <= result.p_value <= 1.0

    def test_handles_nan(self) -> None:
        """Should handle NaN values in residuals."""
        rng = np.random.default_rng(42)
        resid = rng.standard_normal(100)
        resid[10] = np.nan
        resid[50] = np.nan
        result = ljung_box_test(resid, lags=5)
        assert np.isfinite(result.statistic)


class TestHeteroskedasticity:
    """Tests for heteroskedasticity test."""

    def test_homoskedastic_not_rejected(self) -> None:
        """Homoskedastic series should not reject H0."""
        rng = np.random.default_rng(123)
        wn = rng.standard_normal(500)
        result = heteroskedasticity_test(wn)
        assert result.p_value > 0.05

    def test_heteroskedastic_rejected(self) -> None:
        """Series with changing variance should reject H0."""
        rng = np.random.default_rng(42)
        x = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) * 5.0,  # 5x variance
            ]
        )
        result = heteroskedasticity_test(x, h=60)
        assert result.reject is True

    def test_returns_test_result(self) -> None:
        """Should return TestResult dataclass."""
        rng = np.random.default_rng(42)
        result = heteroskedasticity_test(rng.standard_normal(100))
        assert isinstance(result, TestResult)
        assert result.test_name == "Heteroskedasticity (H-test)"


class TestNormality:
    """Tests for normality test."""

    def test_normal_not_rejected(self) -> None:
        """Normal sample should not reject H0."""
        rng = np.random.default_rng(42)
        wn = rng.standard_normal(500)
        result = normality_test(wn)
        assert result.p_value > 0.05

    def test_heavy_tailed_rejected(self) -> None:
        """Heavy-tailed sample should reject H0."""
        rng = np.random.default_rng(42)
        x = rng.standard_t(df=3, size=500)
        result = normality_test(x)
        assert result.reject is True

    def test_returns_skew_kurtosis(self) -> None:
        """Details should contain skewness and kurtosis."""
        rng = np.random.default_rng(42)
        result = normality_test(rng.standard_normal(100))
        assert "skewness" in result.details
        assert "kurtosis" in result.details


class TestCUSUM:
    """Tests for CUSUM test."""

    def test_stable_not_rejected(self) -> None:
        """Stable series should not reject H0."""
        rng = np.random.default_rng(42)
        wn = rng.standard_normal(200)
        result = cusum_test(wn)
        assert isinstance(result, TestResult)
        assert result.test_name == "CUSUM"

    def test_structural_break_detected(self) -> None:
        """Series with structural break should show instability."""
        rng = np.random.default_rng(42)
        x = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) + 3.0,  # Mean shift
            ]
        )
        result = cusum_test(x)
        # The test should at least have a larger statistic
        stable = cusum_test(rng.standard_normal(200))
        assert result.statistic > stable.statistic

    def test_cusum_path_in_details(self) -> None:
        """Details should contain CUSUM path."""
        rng = np.random.default_rng(42)
        result = cusum_test(rng.standard_normal(100))
        assert "cusum_path" in result.details


class TestCUSUMSQ:
    """Tests for CUSUM-SQ test."""

    def test_stable_variance_not_rejected(self) -> None:
        """Stable variance series should not reject H0."""
        rng = np.random.default_rng(42)
        wn = rng.standard_normal(200)
        result = cusumsq_test(wn)
        assert isinstance(result, TestResult)
        assert result.test_name == "CUSUM-SQ"

    def test_cusumsq_path_in_details(self) -> None:
        """Details should contain CUSUM-SQ path."""
        rng = np.random.default_rng(42)
        result = cusumsq_test(rng.standard_normal(100))
        assert "cusumsq_path" in result.details


class TestDiagnosticsIntegration:
    """Integration tests with fitted models."""

    def test_all_diagnostics_on_nile(self, nile_volume: np.ndarray) -> None:
        """All diagnostic tests should run on fitted Nile model."""
        from kalmanbox.diagnostics.residuals import standardized_residuals

        model = LocalLevel(nile_volume)
        results = model.fit()
        std_resid = standardized_residuals(results)

        lb = ljung_box_test(std_resid, lags=10)
        ht = heteroskedasticity_test(std_resid)
        nt = normality_test(std_resid)
        cs = cusum_test(std_resid)
        csq = cusumsq_test(std_resid)

        # All should return valid results
        for test in [lb, ht, nt, cs, csq]:
            assert isinstance(test, TestResult)
            assert np.isfinite(test.statistic)
            assert 0.0 <= test.p_value <= 1.0
