"""Numba-accelerated core loops for Kalman filtering and smoothing.

If numba is not installed, provides pure-Python fallback implementations
with identical signatures and behavior.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox._logging import get_logger

logger = get_logger("numba_core")

# ---------------------------------------------------------------------------
# Try to import numba; set flag
# ---------------------------------------------------------------------------
try:
    from numba import njit

    HAS_NUMBA = True
    logger.debug("Numba detected, JIT-compiled loops enabled")
except ImportError:
    HAS_NUMBA = False
    logger.debug("Numba not installed, using pure-Python loops")

    # Create a no-op decorator that mimics njit signature
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op decorator when numba is not available."""
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator


# ---------------------------------------------------------------------------
# Kalman filter loop (JIT-compiled if numba available)
# ---------------------------------------------------------------------------
@njit(cache=True)
def kalman_filter_loop(
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    Z: NDArray[np.float64],
    R: NDArray[np.float64],
    H: NDArray[np.float64],
    Q: NDArray[np.float64],
    a1: NDArray[np.float64],
    P1: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Core Kalman filter loop.

    Parameters
    ----------
    y : ndarray, shape (T_len, k_obs)
        Observation matrix. NaN values are treated as missing.
    T : ndarray, shape (k_states, k_states)
        State transition matrix.
    Z : ndarray, shape (k_obs, k_states)
        Observation matrix.
    R : ndarray, shape (k_states, k_posdef)
        Selection matrix.
    H : ndarray, shape (k_obs, k_obs)
        Observation noise covariance.
    Q : ndarray, shape (k_posdef, k_posdef)
        State noise covariance.
    a1 : ndarray, shape (k_states,)
        Initial state mean.
    P1 : ndarray, shape (k_states, k_states)
        Initial state covariance.

    Returns
    -------
    a_predicted : ndarray, shape (T_len, k_states)
        Predicted state means.
    P_predicted : ndarray, shape (T_len, k_states, k_states)
        Predicted state covariances.
    a_filtered : ndarray, shape (T_len, k_states)
        Filtered state means.
    P_filtered : ndarray, shape (T_len, k_states, k_states)
        Filtered state covariances.
    v : ndarray, shape (T_len, k_obs)
        Innovation (prediction error) vectors.
    F : ndarray, shape (T_len, k_obs, k_obs)
        Innovation covariance matrices.
    loglike : float
        Total log-likelihood.
    """
    T_len = y.shape[0]
    k_states = a1.shape[0]
    k_obs = Z.shape[0]

    # Allocate output arrays
    a_predicted = np.empty((T_len, k_states), dtype=np.float64)
    P_predicted = np.empty((T_len, k_states, k_states), dtype=np.float64)
    a_filtered = np.empty((T_len, k_states), dtype=np.float64)
    P_filtered = np.empty((T_len, k_states, k_states), dtype=np.float64)
    v = np.empty((T_len, k_obs), dtype=np.float64)
    F_out = np.empty((T_len, k_obs, k_obs), dtype=np.float64)

    loglike = 0.0

    # Initial state
    a = a1.copy()
    P = P1.copy()

    # Precompute R @ Q @ R'
    RQR = R @ Q @ R.T

    for t in range(T_len):
        # --- Prediction step ---
        a_pred = T @ a
        P_pred = T @ P @ T.T + RQR

        # Symmetry enforcement
        P_pred = (P_pred + P_pred.T) / 2.0

        a_predicted[t] = a_pred
        P_predicted[t] = P_pred

        # --- Check for missing observations ---
        y_t = y[t]
        all_missing = True
        for j in range(k_obs):
            if not np.isnan(y_t[j]):
                all_missing = False
                break

        if all_missing:
            # No update: filtered = predicted
            a_filtered[t] = a_pred
            P_filtered[t] = P_pred
            v[t] = np.zeros(k_obs, dtype=np.float64)
            F_out[t] = np.eye(k_obs, dtype=np.float64)
            a = a_pred
            P = P_pred
            continue

        # --- Update step ---
        # Innovation
        v_t = y_t - Z @ a_pred
        # Innovation covariance
        F_t = Z @ P_pred @ Z.T + H
        F_t = (F_t + F_t.T) / 2.0

        v[t] = v_t
        F_out[t] = F_t

        # Kalman gain: K = P_pred @ Z' @ F^{-1}
        # For univariate or small k_obs, direct inversion is fine
        F_inv = np.linalg.inv(F_t)
        K = P_pred @ Z.T @ F_inv

        # Filtered state
        a_filt = a_pred + K @ v_t
        # Joseph form for numerical stability:
        # P_filt = (I - K @ Z) @ P_pred @ (I - K @ Z)' + K @ H @ K'
        IKZ = np.eye(k_states, dtype=np.float64) - K @ Z
        P_filt = IKZ @ P_pred @ IKZ.T + K @ H @ K.T
        P_filt = (P_filt + P_filt.T) / 2.0

        a_filtered[t] = a_filt
        P_filtered[t] = P_filt

        # Log-likelihood contribution
        # log p(y_t) = -0.5 * (k_obs * log(2*pi) + log|F_t| + v_t' @ F_inv @ v_t)
        det_F = np.linalg.det(F_t)
        if det_F > 0:
            log_det = np.log(det_F)
        else:
            log_det = -1e10  # fallback for singular F

        quad = float(v_t @ F_inv @ v_t)
        loglike += -0.5 * (k_obs * np.log(2.0 * np.pi) + log_det + quad)

        # Prepare for next step
        a = a_filt
        P = P_filt

    return a_predicted, P_predicted, a_filtered, P_filtered, v, F_out, loglike


# ---------------------------------------------------------------------------
# RTS smoother loop (JIT-compiled if numba available)
# ---------------------------------------------------------------------------
@njit(cache=True)
def rts_smoother_loop(
    a_filtered: NDArray[np.float64],
    P_filtered: NDArray[np.float64],
    a_predicted: NDArray[np.float64],
    P_predicted: NDArray[np.float64],
    T: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Core RTS (Rauch-Tung-Striebel) smoother backward pass.

    Parameters
    ----------
    a_filtered : ndarray, shape (T_len, k_states)
        Filtered state means from forward pass.
    P_filtered : ndarray, shape (T_len, k_states, k_states)
        Filtered state covariances from forward pass.
    a_predicted : ndarray, shape (T_len, k_states)
        Predicted state means from forward pass.
    P_predicted : ndarray, shape (T_len, k_states, k_states)
        Predicted state covariances from forward pass.
    T : ndarray, shape (k_states, k_states)
        State transition matrix.

    Returns
    -------
    a_smoothed : ndarray, shape (T_len, k_states)
        Smoothed state means.
    P_smoothed : ndarray, shape (T_len, k_states, k_states)
        Smoothed state covariances.
    """
    T_len = a_filtered.shape[0]

    a_smoothed = np.empty_like(a_filtered)
    P_smoothed = np.empty_like(P_filtered)

    # Initialize with last filtered values
    a_smoothed[T_len - 1] = a_filtered[T_len - 1]
    P_smoothed[T_len - 1] = P_filtered[T_len - 1]

    for t in range(T_len - 2, -1, -1):
        # Smoother gain: L_t = P_filtered[t] @ T' @ P_predicted[t+1]^{-1}
        P_pred_next = P_predicted[t + 1]
        P_pred_inv = np.linalg.inv(P_pred_next)

        L = P_filtered[t] @ T.T @ P_pred_inv

        # Smoothed state
        a_smoothed[t] = a_filtered[t] + L @ (a_smoothed[t + 1] - a_predicted[t + 1])

        # Smoothed covariance
        P_diff = P_smoothed[t + 1] - P_predicted[t + 1]
        P_smoothed[t] = P_filtered[t] + L @ P_diff @ L.T
        P_smoothed[t] = (P_smoothed[t] + P_smoothed[t].T) / 2.0

    return a_smoothed, P_smoothed


# ---------------------------------------------------------------------------
# Utility: check if numba is available
# ---------------------------------------------------------------------------
def is_numba_available() -> bool:
    """Check if numba JIT compilation is available.

    Returns
    -------
    bool
        True if numba is installed and JIT is active.
    """
    return HAS_NUMBA


def get_backend_info() -> dict[str, object]:
    """Get information about the computation backend.

    Returns
    -------
    dict
        Dictionary with backend information.
    """
    info: dict[str, object] = {
        "numba_available": HAS_NUMBA,
        "backend": "numba" if HAS_NUMBA else "python",
    }
    if HAS_NUMBA:
        import numba

        info["numba_version"] = numba.__version__
    return info
