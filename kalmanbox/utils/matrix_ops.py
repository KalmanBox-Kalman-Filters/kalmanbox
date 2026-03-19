"""Matrix operations for numerical stability in Kalman filtering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def ensure_symmetric(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force matrix symmetry: (P + P') / 2."""
    return (mat + mat.T) / 2.0


def ensure_positive_definite(mat: NDArray[np.float64], eps: float = 1e-8) -> NDArray[np.float64]:
    """Ensure positive definiteness by adding eps * I if needed."""
    try:
        linalg.cholesky(mat, lower=True)
        return mat
    except linalg.LinAlgError:
        n = mat.shape[0]
        return mat + eps * np.eye(n)


def cholesky_safe(mat: NDArray[np.float64], eps: float = 1e-8) -> NDArray[np.float64]:
    """Cholesky decomposition with fallback for near-singular matrices."""
    mat_sym = ensure_symmetric(mat)
    try:
        return linalg.cholesky(mat_sym, lower=True)
    except linalg.LinAlgError:
        n = mat_sym.shape[0]
        mat_reg = mat_sym + eps * np.eye(n)
        return linalg.cholesky(mat_reg, lower=True)


def solve_via_cholesky(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve Ax = b via Cholesky decomposition of A."""
    low = cholesky_safe(a)
    return linalg.cho_solve((low, True), b)


def log_det_via_cholesky(a: NDArray[np.float64]) -> float:
    """Compute log|A| via Cholesky: log|A| = 2 * sum(log(diag(L)))."""
    low = cholesky_safe(a)
    return float(2.0 * np.sum(np.log(np.diag(low))))
