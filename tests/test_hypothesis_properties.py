"""Property-based tests using hypothesis.

Tests mathematical invariants that must hold for any valid input.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


if HAS_HYPOTHESIS:

    @given(
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20, deadline=5000)
    def test_ensure_symmetric_is_symmetric(n: int) -> None:
        """ensure_symmetric should produce a symmetric matrix."""
        from kalmanbox.utils.matrix_ops import ensure_symmetric

        rng = np.random.default_rng(42)
        A = rng.normal(0, 1, (n, n))
        result = ensure_symmetric(A)
        np.testing.assert_allclose(result, result.T, atol=1e-15)

    @given(
        sigma2=st.floats(min_value=0.01, max_value=1000.0),
    )
    @settings(max_examples=20, deadline=5000)
    def test_positive_transform_roundtrip(sigma2: float) -> None:
        """positive_transform and untransform should be inverse."""
        from kalmanbox.utils.transforms import positive_transform, positive_untransform

        x = np.array([sigma2])
        unconstrained = positive_untransform(x)
        recovered = positive_transform(unconstrained)
        np.testing.assert_allclose(recovered, x, rtol=1e-10)

    @given(
        rho=st.floats(min_value=-0.99, max_value=0.99),
    )
    @settings(max_examples=20, deadline=5000)
    def test_stationary_transform_roundtrip(rho: float) -> None:
        """stationary_transform and untransform should be inverse."""
        from kalmanbox.utils.transforms import (
            stationary_transform,
            stationary_untransform,
        )

        y = np.array([rho])
        unconstrained = stationary_untransform(y)
        recovered = stationary_transform(unconstrained)
        np.testing.assert_allclose(recovered, y, rtol=1e-10)

    @given(
        n=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=10, deadline=10000)
    def test_cholesky_safe_produces_lower_triangular(n: int) -> None:
        """cholesky_safe should return a lower triangular matrix."""
        from kalmanbox.utils.matrix_ops import cholesky_safe

        rng = np.random.default_rng(42)
        A = rng.normal(0, 1, (n, n))
        P = A @ A.T + 0.01 * np.eye(n)
        L = cholesky_safe(P)
        # L should be lower triangular
        np.testing.assert_allclose(L, np.tril(L), atol=1e-15)
        # L @ L.T should recover P
        np.testing.assert_allclose(L @ L.T, P, atol=1e-10)

    @given(
        n=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=10, deadline=10000)
    def test_log_det_via_cholesky_matches_numpy(n: int) -> None:
        """log_det_via_cholesky should match np.linalg.slogdet."""
        from kalmanbox.utils.matrix_ops import log_det_via_cholesky

        rng = np.random.default_rng(42)
        A = rng.normal(0, 1, (n, n))
        P = A @ A.T + 0.1 * np.eye(n)
        log_det = log_det_via_cholesky(P)
        sign, ref_log_det = np.linalg.slogdet(P)
        assert sign > 0
        np.testing.assert_allclose(log_det, ref_log_det, rtol=1e-8)
