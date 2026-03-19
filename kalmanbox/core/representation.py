"""State-space representation container."""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import NDArray

from kalmanbox.utils.matrix_ops import ensure_symmetric


class StateSpaceRepresentation:
    """Container for state-space model matrices.

    State equation:   alpha_{t+1} = T @ alpha_t + R @ eta_t + c
                      eta_t ~ N(0, Q)

    Observation eq:   y_t = Z @ alpha_t + eps_t + d
                      eps_t ~ N(0, H)

    Parameters
    ----------
    k_states : int
        Number of state variables.
    k_endog : int
        Number of observed (endogenous) series.
    k_posdef : int, optional
        Dimension of the state disturbance selection. Defaults to k_states.
    """

    def __init__(
        self,
        k_states: int,
        k_endog: int,
        k_posdef: int | None = None,
    ) -> None:
        if k_states < 1:
            msg = f"k_states must be >= 1, got {k_states}"
            raise ValueError(msg)
        if k_endog < 1:
            msg = f"k_endog must be >= 1, got {k_endog}"
            raise ValueError(msg)

        self.k_states = k_states
        self.k_endog = k_endog
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        # State transition matrix
        self.T: NDArray[np.float64] = np.zeros((k_states, k_states))
        # Design (observation) matrix
        self.Z: NDArray[np.float64] = np.zeros((k_endog, k_states))
        # Selection matrix
        self.R: NDArray[np.float64] = np.zeros((k_states, self.k_posdef))
        # Observation disturbance covariance
        self.H: NDArray[np.float64] = np.zeros((k_endog, k_endog))
        # State disturbance covariance
        self.Q: NDArray[np.float64] = np.zeros((self.k_posdef, self.k_posdef))
        # State intercept
        self.c: NDArray[np.float64] = np.zeros(k_states)
        # Observation intercept
        self.d: NDArray[np.float64] = np.zeros(k_endog)
        # Initial state mean
        self.a1: NDArray[np.float64] = np.zeros(k_states)
        # Initial state covariance
        self.P1: NDArray[np.float64] = np.zeros((k_states, k_states))

    def validate(self) -> None:
        """Validate dimensions, symmetry, and positive semi-definiteness.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        # Dimension checks
        self._check_shape("T", self.T, (self.k_states, self.k_states))
        self._check_shape("Z", self.Z, (self.k_endog, self.k_states))
        self._check_shape("R", self.R, (self.k_states, self.k_posdef))
        self._check_shape("H", self.H, (self.k_endog, self.k_endog))
        self._check_shape("Q", self.Q, (self.k_posdef, self.k_posdef))
        self._check_shape("c", self.c, (self.k_states,))
        self._check_shape("d", self.d, (self.k_endog,))
        self._check_shape("a1", self.a1, (self.k_states,))
        self._check_shape("P1", self.P1, (self.k_states, self.k_states))

        # Symmetry checks
        self._check_symmetric("H", self.H)
        self._check_symmetric("Q", self.Q)
        self._check_symmetric("P1", self.P1)

        # Positive semi-definiteness checks
        self._check_psd("H", self.H)
        self._check_psd("Q", self.Q)
        self._check_psd("P1", self.P1)

    def clone(self) -> StateSpaceRepresentation:
        """Return a deep copy of this representation."""
        return copy.deepcopy(self)

    def to_companion(self) -> NDArray[np.float64]:
        """Return the companion form matrix T for stability checking.

        A system is stable if all eigenvalues of T have modulus < 1.

        Returns
        -------
        NDArray[np.float64]
            The transition matrix T (companion form).
        """
        return self.T.copy()

    def is_stable(self) -> bool:
        """Check if the system is stable (all eigenvalues of T inside unit circle)."""
        eigenvalues = np.linalg.eigvals(self.to_companion())
        return bool(np.all(np.abs(eigenvalues) < 1.0))

    @staticmethod
    def _check_shape(name: str, array: NDArray[np.float64], expected: tuple[int, ...]) -> None:
        """Validate that array has the expected shape."""
        if array.shape != expected:
            msg = f"Matrix {name} has shape {array.shape}, expected {expected}"
            raise ValueError(msg)

    @staticmethod
    def _check_symmetric(name: str, matrix: NDArray[np.float64]) -> None:
        """Validate that matrix is symmetric."""
        if matrix.ndim == 2 and not np.allclose(matrix, matrix.T):
            msg = f"Matrix {name} is not symmetric"
            raise ValueError(msg)

    @staticmethod
    def _check_psd(name: str, matrix: NDArray[np.float64]) -> None:
        """Validate that matrix is positive semi-definite."""
        if matrix.ndim == 2:
            eigenvalues = np.linalg.eigvalsh(ensure_symmetric(matrix))
            if np.any(eigenvalues < -1e-10):
                min_eig = eigenvalues.min()
                msg = f"Matrix {name} is not positive semi-definite (min eigenvalue: {min_eig:.2e})"
                raise ValueError(msg)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"StateSpaceRepresentation(k_states={self.k_states}, "
            f"k_endog={self.k_endog}, k_posdef={self.k_posdef})"
        )
