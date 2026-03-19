"""Basic Structural Model (BSM)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


class BasicStructuralModel(StateSpaceModel):
    """Basic Structural Model (Harvey & Todd 1983).

    Decomposes a time series into trend, seasonal, and irregular components.

    Model:
        y_t = mu_t + gamma_t + eps_t
        mu_t = mu_{t-1} + nu_{t-1} + eta_t        (trend level)
        nu_t = nu_{t-1} + zeta_t                    (trend slope)
        gamma_t = seasonal component                 (dummy or trigonometric)

    Parameters
    ----------
    endog : NDArray
        Observed time series data.
    seasonal_period : int
        Number of periods in a seasonal cycle (e.g., 12 for monthly, 4 for quarterly).
    seasonal : str
        Type of seasonal component: 'dummy' (default) or 'trigonometric'.
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        seasonal_period: int = 12,
        seasonal: str = "dummy",
    ) -> None:
        self.seasonal_period = seasonal_period
        self.seasonal_type = seasonal

        if seasonal not in ("dummy", "trigonometric"):
            msg = f"seasonal must be 'dummy' or 'trigonometric', got '{seasonal}'"
            raise ValueError(msg)

        if seasonal == "dummy":
            self._n_seasonal_states = seasonal_period - 1
        else:
            # Trigonometric: 2 states per harmonic, except last if s is even
            n_harmonics = seasonal_period // 2
            self._n_seasonal_states = 2 * n_harmonics
            if seasonal_period % 2 == 0:
                self._n_seasonal_states -= 1  # last harmonic has 1 state

        self._k_states = 2 + self._n_seasonal_states

        super().__init__(endog)

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial guesses: split variance 4 ways."""
        var = float(np.nanvar(self.endog))
        return np.array([var / 4.0, var / 4.0, var / 8.0, var / 4.0])

    @property
    def param_names(self) -> list[str]:
        return ["sigma2_obs", "sigma2_level", "sigma2_trend", "sigma2_seasonal"]

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.exp(unconstrained)

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.log(constrained)

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build BSM state-space representation."""
        sigma2_obs = params[0]
        sigma2_level = params[1]
        sigma2_trend = params[2]
        sigma2_seasonal = params[3]

        k_states = self._k_states
        k_posdef = 3  # level, trend, seasonal noises

        ssm = StateSpaceRepresentation(k_states=k_states, k_endog=1, k_posdef=k_posdef)

        # --- Transition matrix T ---
        transition = np.zeros((k_states, k_states))
        # Trend block
        transition[0, 0] = 1.0  # mu_t = mu_{t-1} + nu_{t-1}
        transition[0, 1] = 1.0
        transition[1, 1] = 1.0  # nu_t = nu_{t-1}

        if self.seasonal_type == "dummy":
            self._build_dummy_seasonal(transition)
        else:
            self._build_trig_seasonal(transition)

        ssm.T = transition

        # --- Design matrix Z ---
        design = np.zeros((1, k_states))
        design[0, 0] = 1.0  # level
        design[0, 2] = 1.0  # first seasonal state
        ssm.Z = design

        # --- Selection matrix R ---
        selection = np.zeros((k_states, k_posdef))
        selection[0, 0] = 1.0  # level noise
        selection[1, 1] = 1.0  # trend noise
        selection[2, 2] = 1.0  # seasonal noise
        ssm.R = selection

        # --- Covariance matrices ---
        ssm.H = np.array([[sigma2_obs]])
        ssm.Q = np.diag([sigma2_level, sigma2_trend, sigma2_seasonal])

        # --- Initial conditions (diffuse) ---
        ssm.a1 = np.zeros(k_states)
        ssm.P1 = np.eye(k_states) * config.diffuse_initial_variance

        return ssm

    def _build_dummy_seasonal(self, transition: NDArray[np.float64]) -> None:
        """Fill seasonal block of T for dummy seasonal."""
        s = self.seasonal_period
        # gamma_t = -gamma_{t-1} - ... - gamma_{t-s+1}
        for j in range(s - 1):
            transition[2, 2 + j] = -1.0
        # Shift: gamma_{t-j} = gamma_{t-j} (identity shift)
        for j in range(s - 2):
            transition[3 + j, 2 + j] = 1.0

    def _build_trig_seasonal(self, transition: NDArray[np.float64]) -> None:
        """Fill seasonal block of T for trigonometric seasonal."""
        s = self.seasonal_period
        n_harmonics = s // 2
        idx = 2  # start after trend states

        for j in range(1, n_harmonics + 1):
            lam = 2.0 * np.pi * j / s

            if j < n_harmonics or s % 2 != 0:
                # Full 2x2 rotation block
                transition[idx, idx] = np.cos(lam)
                transition[idx, idx + 1] = np.sin(lam)
                transition[idx + 1, idx] = -np.sin(lam)
                transition[idx + 1, idx + 1] = np.cos(lam)
                idx += 2
            else:
                # Last harmonic for even s: single state with cos(pi) = -1
                transition[idx, idx] = -1.0
                idx += 1
