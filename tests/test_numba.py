"""Tests for Numba-accelerated core loops.

Verifies that:
1. Results match pure-Python implementation exactly
2. Missing observations are handled correctly
3. Speedup is measurable when numba is available
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from kalmanbox.utils.numba_core import (
    HAS_NUMBA,
    get_backend_info,
    is_numba_available,
    kalman_filter_loop,
    rts_smoother_loop,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def local_level_ssm() -> dict[str, NDArray[np.float64]]:
    """Create a local level state-space model for testing.

    y_t = mu_t + eps_t,  eps_t ~ N(0, sigma2_obs)
    mu_t = mu_{t-1} + eta_t,  eta_t ~ N(0, sigma2_level)
    """
    sigma2_obs = 15099.0
    sigma2_level = 1469.0

    T = np.array([[1.0]])
    Z = np.array([[1.0]])
    R = np.array([[1.0]])
    H = np.array([[sigma2_obs]])
    Q = np.array([[sigma2_level]])
    a1 = np.array([0.0])
    P1 = np.array([[1e7]])

    return {"T": T, "Z": Z, "R": R, "H": H, "Q": Q, "a1": a1, "P1": P1}


@pytest.fixture
def nile_like_data() -> NDArray[np.float64]:
    """Generate Nile-like data (100 observations)."""
    rng = np.random.default_rng(42)
    n = 100
    level = np.cumsum(rng.normal(0, np.sqrt(1469), n))
    y = level + rng.normal(0, np.sqrt(15099), n)
    return y.reshape(-1, 1).astype(np.float64)


@pytest.fixture
def large_data() -> NDArray[np.float64]:
    """Generate large dataset (5000 observations) for speed test."""
    rng = np.random.default_rng(123)
    n = 5000
    level = np.cumsum(rng.normal(0, np.sqrt(1469), n))
    y = level + rng.normal(0, np.sqrt(15099), n)
    return y.reshape(-1, 1).astype(np.float64)


# ---------------------------------------------------------------------------
# Pure Python reference implementation
# ---------------------------------------------------------------------------
def _kalman_filter_pure_python(
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    Z: NDArray[np.float64],
    R: NDArray[np.float64],
    H: NDArray[np.float64],
    Q: NDArray[np.float64],
    a1: NDArray[np.float64],
    P1: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Reference pure-Python Kalman filter for comparison."""
    T_len = y.shape[0]
    k_states = a1.shape[0]
    k_obs = Z.shape[0]

    a_filtered = np.empty((T_len, k_states))
    P_filtered = np.empty((T_len, k_states, k_states))
    loglike = 0.0

    a = a1.copy()
    P = P1.copy()
    RQR = R @ Q @ R.T

    for t in range(T_len):
        # Predict
        a_pred = T @ a
        P_pred = T @ P @ T.T + RQR
        P_pred = (P_pred + P_pred.T) / 2.0

        # Check missing
        y_t = y[t]
        if np.any(np.isnan(y_t)):
            a_filtered[t] = a_pred
            P_filtered[t] = P_pred
            a = a_pred
            P = P_pred
            continue

        # Update
        v_t = y_t - Z @ a_pred
        F_t = Z @ P_pred @ Z.T + H
        F_inv = np.linalg.inv(F_t)
        K = P_pred @ Z.T @ F_inv

        a_filt = a_pred + K @ v_t
        IKZ = np.eye(k_states) - K @ Z
        P_filt = IKZ @ P_pred @ IKZ.T + K @ H @ K.T
        P_filt = (P_filt + P_filt.T) / 2.0

        a_filtered[t] = a_filt
        P_filtered[t] = P_filt

        det_F = np.linalg.det(F_t)
        log_det = np.log(det_F) if det_F > 0 else -1e10
        quad = float(v_t @ F_inv @ v_t)
        loglike += -0.5 * (k_obs * np.log(2.0 * np.pi) + log_det + quad)

        a = a_filt
        P = P_filt

    return a_filtered, P_filtered, loglike


# ---------------------------------------------------------------------------
# Tests: correctness
# ---------------------------------------------------------------------------
class TestKalmanFilterLoop:
    """Test that kalman_filter_loop produces correct results."""

    def test_output_shapes(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        ssm = local_level_ssm
        a_pred, P_pred, a_filt, P_filt, v, F, ll = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        assert a_pred.shape == (100, 1)
        assert P_pred.shape == (100, 1, 1)
        assert a_filt.shape == (100, 1)
        assert P_filt.shape == (100, 1, 1)
        assert v.shape == (100, 1)
        assert F.shape == (100, 1, 1)
        assert isinstance(ll, float)

    def test_loglike_finite(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        ssm = local_level_ssm
        *_, ll = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        assert np.isfinite(ll)
        assert ll < 0  # log-likelihood should be negative

    def test_matches_pure_python(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        ssm = local_level_ssm
        _, _, a_filt, P_filt, _, _, ll = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        a_ref, P_ref, ll_ref = _kalman_filter_pure_python(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        np.testing.assert_allclose(a_filt, a_ref, rtol=1e-10)
        np.testing.assert_allclose(P_filt, P_ref, rtol=1e-10)
        np.testing.assert_allclose(ll, ll_ref, rtol=1e-8)

    def test_missing_data(
        self,
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        """Test that NaN observations are handled (no update)."""
        ssm = local_level_ssm
        rng = np.random.default_rng(42)
        y = rng.normal(900, 100, (50, 1)).astype(np.float64)

        # Introduce missing values
        y[10, 0] = np.nan
        y[20, 0] = np.nan
        y[30, 0] = np.nan

        a_pred, _, a_filt, _, _, _, ll = kalman_filter_loop(
            y,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )

        # At missing points, filtered = predicted
        np.testing.assert_array_equal(a_filt[10], a_pred[10])
        np.testing.assert_array_equal(a_filt[20], a_pred[20])
        np.testing.assert_array_equal(a_filt[30], a_pred[30])
        assert np.isfinite(ll)

    def test_filtered_variance_decreasing(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        """Filtered variance should stabilize (converge to steady state)."""
        ssm = local_level_ssm
        _, _, _, P_filt, _, _, _ = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        # P_filt should be large initially and stabilize
        assert P_filt[0, 0, 0] > P_filt[-1, 0, 0]


# ---------------------------------------------------------------------------
# Tests: RTS smoother
# ---------------------------------------------------------------------------
class TestRTSSmootherLoop:
    """Test that rts_smoother_loop produces correct results."""

    def test_output_shapes(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        ssm = local_level_ssm
        a_pred, P_pred, a_filt, P_filt, _, _, _ = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        a_smooth, P_smooth = rts_smoother_loop(
            a_filt,
            P_filt,
            a_pred,
            P_pred,
            ssm["T"],
        )
        assert a_smooth.shape == (100, 1)
        assert P_smooth.shape == (100, 1, 1)

    def test_smoother_variance_le_filter(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        """Smoothed variance should be <= filtered variance."""
        ssm = local_level_ssm
        a_pred, P_pred, a_filt, P_filt, _, _, _ = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        _, P_smooth = rts_smoother_loop(
            a_filt,
            P_filt,
            a_pred,
            P_pred,
            ssm["T"],
        )
        # Smoothed variance <= filtered variance (except possibly last)
        for t in range(len(nile_like_data) - 1):
            assert P_smooth[t, 0, 0] <= P_filt[t, 0, 0] + 1e-10

    def test_last_smoothed_equals_filtered(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        """At T_len-1, smoothed = filtered."""
        ssm = local_level_ssm
        a_pred, P_pred, a_filt, P_filt, _, _, _ = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        a_smooth, P_smooth = rts_smoother_loop(
            a_filt,
            P_filt,
            a_pred,
            P_pred,
            ssm["T"],
        )
        np.testing.assert_array_equal(a_smooth[-1], a_filt[-1])
        np.testing.assert_array_equal(P_smooth[-1], P_filt[-1])

    def test_symmetry_of_smoothed_P(
        self,
        nile_like_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        """Smoothed covariance should be symmetric."""
        ssm = local_level_ssm
        a_pred, P_pred, a_filt, P_filt, _, _, _ = kalman_filter_loop(
            nile_like_data,
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        _, P_smooth = rts_smoother_loop(
            a_filt,
            P_filt,
            a_pred,
            P_pred,
            ssm["T"],
        )
        for t in range(len(nile_like_data)):
            np.testing.assert_allclose(P_smooth[t], P_smooth[t].T, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: backend info
# ---------------------------------------------------------------------------
class TestBackendInfo:
    """Test backend detection utilities."""

    def test_is_numba_available(self) -> None:
        result = is_numba_available()
        assert isinstance(result, bool)

    def test_get_backend_info(self) -> None:
        info = get_backend_info()
        assert "numba_available" in info
        assert "backend" in info
        assert info["backend"] in ("numba", "python")


# ---------------------------------------------------------------------------
# Tests: performance (only meaningful with numba)
# ---------------------------------------------------------------------------
class TestPerformance:
    """Performance benchmarks (informational)."""

    def test_speed_local_level(
        self,
        large_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        """Measure execution time for T=5000 local level."""
        ssm = local_level_ssm

        # Warm-up (numba JIT compile)
        kalman_filter_loop(
            large_data[:10],
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )

        # Timed run
        n_runs = 5
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            kalman_filter_loop(
                large_data,
                ssm["T"],
                ssm["Z"],
                ssm["R"],
                ssm["H"],
                ssm["Q"],
                ssm["a1"],
                ssm["P1"],
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_ms = 1000 * np.mean(times)
        print(
            f"\n  KF loop T=5000, k=1: avg={avg_ms:.2f}ms "
            f"(backend={'numba' if HAS_NUMBA else 'python'})"
        )

        # With numba: should be < 50ms for T=5000
        # Without numba: just ensure it runs
        assert avg_ms < 5000  # generous bound for CI

    def test_speed_filter_and_smoother(
        self,
        large_data: NDArray[np.float64],
        local_level_ssm: dict[str, NDArray[np.float64]],
    ) -> None:
        """Measure combined filter + smoother time."""
        ssm = local_level_ssm

        # Warm-up
        a_pred, P_pred, a_filt, P_filt, _, _, _ = kalman_filter_loop(
            large_data[:10],
            ssm["T"],
            ssm["Z"],
            ssm["R"],
            ssm["H"],
            ssm["Q"],
            ssm["a1"],
            ssm["P1"],
        )
        rts_smoother_loop(a_filt, P_filt, a_pred, P_pred, ssm["T"])

        # Timed run
        n_runs = 3
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            a_pred, P_pred, a_filt, P_filt, _, _, _ = kalman_filter_loop(
                large_data,
                ssm["T"],
                ssm["Z"],
                ssm["R"],
                ssm["H"],
                ssm["Q"],
                ssm["a1"],
                ssm["P1"],
            )
            rts_smoother_loop(a_filt, P_filt, a_pred, P_pred, ssm["T"])
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_ms = 1000 * np.mean(times)
        print(
            f"\n  KF+RTS T=5000, k=1: avg={avg_ms:.2f}ms "
            f"(backend={'numba' if HAS_NUMBA else 'python'})"
        )
        assert avg_ms < 10000  # generous bound
