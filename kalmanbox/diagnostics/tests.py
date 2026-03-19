"""Statistical tests for state-space model diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class TestResult:
    """Container for statistical test results.

    Attributes
    ----------
    statistic : float
        Test statistic value.
    p_value : float
        P-value of the test.
    test_name : str
        Name of the test.
    null_hypothesis : str
        Description of the null hypothesis.
    reject : bool
        Whether to reject H0 at 5% level.
    details : dict
        Additional test-specific details.
    """

    statistic: float
    p_value: float
    test_name: str
    null_hypothesis: str
    reject: bool
    details: dict

    def __repr__(self) -> str:
        """Return formatted test result string."""
        verdict = "REJECT H0" if self.reject else "FAIL TO REJECT H0"
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p-value={self.p_value:.4f} -> {verdict}"
        )


def ljung_box_test(
    residuals: NDArray[np.float64],
    lags: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    """Ljung-Box test for serial autocorrelation.

    Tests H0: residuals are independently distributed (no autocorrelation).

    Q = n(n+2) * sum_{k=1}^{h} rho_k^2 / (n - k)

    Under H0, Q ~ chi-squared(h).

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Standardized residuals (1D array).
    lags : int
        Number of lags to test. Default 10.
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    TestResult
    """
    resid = residuals[~np.isnan(residuals)]
    n = len(resid)

    if n <= lags:
        return TestResult(
            statistic=np.nan,
            p_value=np.nan,
            test_name="Ljung-Box",
            null_hypothesis="No serial autocorrelation",
            reject=False,
            details={"lags": lags, "n": n, "error": "Too few observations"},
        )

    # Compute autocorrelations
    resid_centered = resid - np.mean(resid)
    var = np.sum(resid_centered**2) / n

    acf_values = np.zeros(lags)
    for k in range(1, lags + 1):
        acf_values[k - 1] = np.sum(resid_centered[k:] * resid_centered[:-k]) / (n * var)

    # Ljung-Box statistic
    Q = n * (n + 2) * np.sum(acf_values**2 / (n - np.arange(1, lags + 1)))

    p_value = 1.0 - stats.chi2.cdf(Q, df=lags)

    return TestResult(
        statistic=float(Q),
        p_value=float(p_value),
        test_name="Ljung-Box",
        null_hypothesis="No serial autocorrelation",
        reject=bool(p_value < alpha),
        details={"lags": lags, "n": n, "acf": acf_values.tolist()},
    )


def heteroskedasticity_test(
    residuals: NDArray[np.float64],
    h: int | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Heteroskedasticity test (H-test) for residuals.

    Tests H0: the variance of residuals is constant over time.
    Compares the sum of squared residuals in the last h observations
    against the first h observations.

    H = sum(e_t^2, t=n-h+1..n) / sum(e_t^2, t=1..h)

    Under H0, H ~ F(h, h).

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Standardized residuals (1D array).
    h : int or None
        Number of observations for comparison. Default: n // 3.
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    TestResult
    """
    resid = residuals[~np.isnan(residuals)]
    n = len(resid)

    if h is None:
        h = n // 3

    if h < 2 or n < 2 * h:
        return TestResult(
            statistic=np.nan,
            p_value=np.nan,
            test_name="Heteroskedasticity (H-test)",
            null_hypothesis="Homoskedasticity",
            reject=False,
            details={"h": h, "n": n, "error": "Too few observations"},
        )

    ss_last = np.sum(resid[-h:] ** 2)
    ss_first = np.sum(resid[:h] ** 2)

    if ss_first < 1e-15:
        return TestResult(
            statistic=np.inf,
            p_value=0.0,
            test_name="Heteroskedasticity (H-test)",
            null_hypothesis="Homoskedasticity",
            reject=True,
            details={
                "h": h,
                "n": n,
                "ss_first": float(ss_first),
                "ss_last": float(ss_last),
            },
        )

    H_stat = ss_last / ss_first
    p_value = 2.0 * min(
        stats.f.cdf(H_stat, dfn=h, dfd=h),
        1.0 - stats.f.cdf(H_stat, dfn=h, dfd=h),
    )

    return TestResult(
        statistic=float(H_stat),
        p_value=float(p_value),
        test_name="Heteroskedasticity (H-test)",
        null_hypothesis="Homoskedasticity",
        reject=bool(p_value < alpha),
        details={
            "h": h,
            "n": n,
            "ss_first": float(ss_first),
            "ss_last": float(ss_last),
        },
    )


def normality_test(
    residuals: NDArray[np.float64],
    alpha: float = 0.05,
) -> TestResult:
    """Jarque-Bera normality test.

    Tests H0: residuals are normally distributed.

    JB = (n/6) * (S^2 + K^2/4)

    where S = skewness, K = excess kurtosis.
    Under H0, JB ~ chi-squared(2).

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Standardized residuals (1D array).
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    TestResult
    """
    resid = residuals[~np.isnan(residuals)]
    n = len(resid)

    if n < 8:
        return TestResult(
            statistic=np.nan,
            p_value=np.nan,
            test_name="Jarque-Bera Normality",
            null_hypothesis="Normality",
            reject=False,
            details={"n": n, "error": "Too few observations"},
        )

    skew = stats.skew(resid)
    kurt = stats.kurtosis(resid)  # excess kurtosis

    JB = (n / 6.0) * (skew**2 + kurt**2 / 4.0)
    p_value = 1.0 - stats.chi2.cdf(JB, df=2)

    return TestResult(
        statistic=float(JB),
        p_value=float(p_value),
        test_name="Jarque-Bera Normality",
        null_hypothesis="Normality",
        reject=bool(p_value < alpha),
        details={
            "n": n,
            "skewness": float(skew),
            "kurtosis": float(kurt),
        },
    )


def cusum_test(
    residuals: NDArray[np.float64],
    alpha: float = 0.05,
) -> TestResult:
    """CUSUM test for parameter stability.

    Tests H0: parameters are stable over time.

    S_r = cumsum(w_t) / sigma_w

    where w_t are the recursive residuals. The test checks whether
    the cumulative sum stays within bounds +/- c(alpha) * sqrt(n).

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Standardized residuals or recursive residuals (1D array).
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    TestResult
    """
    resid = residuals[~np.isnan(residuals)]
    n = len(resid)

    if n < 4:
        return TestResult(
            statistic=np.nan,
            p_value=np.nan,
            test_name="CUSUM",
            null_hypothesis="Parameter stability",
            reject=False,
            details={"n": n, "error": "Too few observations"},
        )

    sigma_w = np.std(resid, ddof=1)

    if sigma_w < 1e-15:
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            test_name="CUSUM",
            null_hypothesis="Parameter stability",
            reject=False,
            details={"n": n, "sigma_w": float(sigma_w)},
        )

    # Cumulative sum of standardized residuals
    cusum = np.cumsum(resid) / (sigma_w * np.sqrt(n))

    # Maximum absolute value of CUSUM
    max_cusum = float(np.max(np.abs(cusum)))

    # Critical values based on Brownian bridge distribution
    # Harvey (1989) Table 5.2 critical values
    critical_values = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}
    critical = critical_values.get(alpha, 0.948)

    # Approximate p-value via Kolmogorov-Smirnov distribution
    p_value = float(stats.kstwobign.sf(max_cusum * np.sqrt(n)))
    p_value = max(0.0, min(1.0, p_value))

    return TestResult(
        statistic=max_cusum,
        p_value=p_value,
        test_name="CUSUM",
        null_hypothesis="Parameter stability",
        reject=bool(max_cusum > critical),
        details={
            "n": n,
            "max_cusum": max_cusum,
            "critical_value": critical,
            "cusum_path": cusum.tolist(),
        },
    )


def cusumsq_test(
    residuals: NDArray[np.float64],
    alpha: float = 0.05,
) -> TestResult:
    """CUSUM of squares test for variance stability.

    Tests H0: variance of residuals is stable over time.

    S_r = cumsum(w_t^2) / sum(w_t^2)

    The test checks whether the path of S_r stays within the
    confidence bands around the 45-degree line.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Standardized residuals or recursive residuals (1D array).
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    TestResult
    """
    resid = residuals[~np.isnan(residuals)]
    n = len(resid)

    if n < 4:
        return TestResult(
            statistic=np.nan,
            p_value=np.nan,
            test_name="CUSUM-SQ",
            null_hypothesis="Variance stability",
            reject=False,
            details={"n": n, "error": "Too few observations"},
        )

    resid_sq = resid**2
    total_ss = np.sum(resid_sq)

    if total_ss < 1e-15:
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            test_name="CUSUM-SQ",
            null_hypothesis="Variance stability",
            reject=False,
            details={"n": n, "total_ss": float(total_ss)},
        )

    # Cumulative proportion
    cusumsq = np.cumsum(resid_sq) / total_ss

    # Expected path under H0: t/n
    expected = np.arange(1, n + 1) / n

    # Maximum deviation from expected
    max_deviation = float(np.max(np.abs(cusumsq - expected)))

    # Critical values (Harvey 1989, Table 5.3)
    critical_values = {
        0.01: 1.63 / np.sqrt(n),
        0.05: 1.36 / np.sqrt(n),
        0.10: 1.22 / np.sqrt(n),
    }
    critical = critical_values.get(alpha, 1.36 / np.sqrt(n))

    # Approximate p-value using KS distribution
    ks_stat = max_deviation * np.sqrt(n)
    p_value = float(stats.kstwobign.sf(ks_stat))
    p_value = max(0.0, min(1.0, p_value))

    return TestResult(
        statistic=max_deviation,
        p_value=p_value,
        test_name="CUSUM-SQ",
        null_hypothesis="Variance stability",
        reject=bool(max_deviation > critical),
        details={
            "n": n,
            "max_deviation": max_deviation,
            "critical_value": float(critical),
            "cusumsq_path": cusumsq.tolist(),
        },
    )
