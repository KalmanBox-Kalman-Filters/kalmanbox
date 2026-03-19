"""Local Linear Trend model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


class LocalLinearTrend(StateSpaceModel):
    """Local Linear Trend model.

    Model:
        y_t = mu_t + eps_t                      eps_t ~ N(0, sigma2_obs)
        mu_t = mu_{t-1} + nu_{t-1} + eta_t      eta_t ~ N(0, sigma2_level)
        nu_t = nu_{t-1} + zeta_t                 zeta_t ~ N(0, sigma2_trend)

    This extends the Local Level model with a stochastic slope (trend)
    component, allowing the level to have a time-varying drift.

    Parameters
    ----------
    endog : NDArray
        Observed time series data.
    """

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial parameter guesses."""
        var = float(np.nanvar(self.endog))
        return np.array([var / 3.0, var / 3.0, var / 6.0])

    @property
    def param_names(self) -> list[str]:
        return ["sigma2_obs", "sigma2_level", "sigma2_trend"]

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """exp() to ensure positive variances."""
        return np.exp(unconstrained)

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """log() inverse."""
        return np.log(constrained)

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build SSM: k_states=2 (level, slope)."""
        sigma2_obs, sigma2_level, sigma2_trend = params[0], params[1], params[2]

        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
        ssm.T = np.array([[1.0, 1.0], [0.0, 1.0]])
        ssm.Z = np.array([[1.0, 0.0]])
        ssm.R = np.eye(2)
        ssm.H = np.array([[sigma2_obs]])
        ssm.Q = np.diag([sigma2_level, sigma2_trend])
        ssm.a1 = np.zeros(2)
        ssm.P1 = np.eye(2) * config.diffuse_initial_variance
        return ssm
