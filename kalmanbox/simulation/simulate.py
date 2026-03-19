"""Simulation of state-space models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kalmanbox.core.model import StateSpaceModel
    from kalmanbox.core.representation import StateSpaceRepresentation


def simulate_ssm(
    ssm: StateSpaceRepresentation,
    n_periods: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate observations and states from a state-space model.

    Generates synthetic data from:
        alpha_1 ~ N(a1, P1)
        alpha_{t+1} = T @ alpha_t + c + R @ eta_t,  eta_t ~ N(0, Q)
        y_t = Z @ alpha_t + d + eps_t,               eps_t ~ N(0, H)

    Parameters
    ----------
    ssm : StateSpaceRepresentation
        State-space model specification with all matrices set.
    n_periods : int
        Number of periods to simulate.
    seed : int or None
        Random seed for reproducibility. Default None.

    Returns
    -------
    y : NDArray[np.float64]
        Simulated observations, shape (n_periods, k_endog).
    states : NDArray[np.float64]
        Simulated states, shape (n_periods, k_states).
    """
    rng = np.random.default_rng(seed)

    k_states = ssm.k_states
    k_endog = ssm.k_endog
    k_posdef = ssm.Q.shape[0]

    # Pre-allocate
    states = np.zeros((n_periods, k_states))
    y = np.zeros((n_periods, k_endog))

    # Initial state
    # If P1 has very large values (diffuse), use a moderate initialization
    p1 = ssm.P1.copy()
    max_var = np.max(np.diag(p1))
    p1_sim = np.eye(k_states) * min(max_var, 1e4) if max_var > 1e6 else p1

    try:
        chol_p1 = np.linalg.cholesky(p1_sim)
        alpha = ssm.a1 + chol_p1 @ rng.standard_normal(k_states)
    except np.linalg.LinAlgError:
        alpha = ssm.a1 + np.sqrt(np.abs(np.diag(p1_sim))) * rng.standard_normal(k_states)

    # Cholesky factors for Q and H
    try:
        chol_q = np.linalg.cholesky(ssm.Q)
    except np.linalg.LinAlgError:
        chol_q = np.diag(np.sqrt(np.maximum(np.diag(ssm.Q), 0.0)))

    try:
        chol_h = np.linalg.cholesky(ssm.H)
    except np.linalg.LinAlgError:
        chol_h = np.diag(np.sqrt(np.maximum(np.diag(ssm.H), 0.0)))

    for t in range(n_periods):
        states[t] = alpha

        # Observation
        eps = chol_h @ rng.standard_normal(k_endog)
        y[t] = ssm.Z @ alpha + ssm.d + eps

        # State transition
        eta = chol_q @ rng.standard_normal(k_posdef)
        alpha = ssm.T @ alpha + ssm.c + ssm.R @ eta

    return y, states


def simulate_from_model(
    model: StateSpaceModel,
    n_periods: int,
    params: NDArray[np.float64] | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate from a state-space model with given parameters.

    Builds the SSM representation from model and parameters,
    then calls simulate_ssm.

    Parameters
    ----------
    model : StateSpaceModel
        The model to simulate from.
    n_periods : int
        Number of periods to simulate.
    params : NDArray or None
        Model parameters. If None, uses model.start_params.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    y : NDArray[np.float64]
        Simulated observations, shape (n_periods, k_endog).
    states : NDArray[np.float64]
        Simulated states, shape (n_periods, k_states).
    """
    if params is None:
        params = model.start_params

    ssm = model._build_ssm(params)
    return simulate_ssm(ssm, n_periods, seed=seed)


def simulate_missing(
    y: NDArray[np.float64],
    missing_rate: float = 0.1,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Introduce random missing values (NaN) into a time series.

    Useful for testing robustness of models to missing data.

    Parameters
    ----------
    y : NDArray[np.float64]
        Original data, shape (nobs,) or (nobs, k_endog).
    missing_rate : float
        Fraction of observations to set to NaN. Default 0.1.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Data with random NaN values introduced.

    Raises
    ------
    ValueError
        If missing_rate is not in [0, 1).
    """
    if not 0.0 <= missing_rate < 1.0:
        msg = f"missing_rate must be in [0, 1), got {missing_rate}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    y_missing = y.copy().astype(float)

    nobs = len(y) if y.ndim == 1 else y.shape[0]

    n_missing = int(nobs * missing_rate)
    if n_missing == 0:
        return y_missing

    missing_indices = rng.choice(nobs, size=n_missing, replace=False)

    if y.ndim == 1:
        y_missing[missing_indices] = np.nan
    else:
        y_missing[missing_indices, :] = np.nan

    return y_missing
