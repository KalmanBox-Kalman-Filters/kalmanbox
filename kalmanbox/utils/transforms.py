"""Parameter transformations for constrained optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def positive_transform(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform unconstrained -> positive via exp."""
    return np.exp(x)


def positive_untransform(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform positive -> unconstrained via log."""
    return np.log(y)


def stationary_transform(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform unconstrained -> (-1, 1) via tanh."""
    return np.tanh(x)


def stationary_untransform(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform (-1, 1) -> unconstrained via arctanh."""
    return np.arctanh(y)
