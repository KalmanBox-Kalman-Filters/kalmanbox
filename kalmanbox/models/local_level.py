"""Local Level (random walk plus noise) model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


class LocalLevel(StateSpaceModel):
    """Local Level model (random walk plus noise).

    Model:
        y_t = mu_t + eps_t,      eps_t ~ N(0, sigma2_obs)
        mu_t = mu_{t-1} + eta_t,  eta_t ~ N(0, sigma2_level)

    Parameters:
        sigma2_obs   - observation noise variance
        sigma2_level - level noise variance (state innovation)

    This is the simplest structural time series model, also known
    as the random walk plus noise model.

    Parameters
    ----------
    endog : NDArray
        Observed time series data.
    """

    _n_diffuse: int = 1

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial parameter guesses: half the variance of the series each."""
        var = float(np.nanvar(self.endog))
        return np.array([var / 2.0, var / 2.0])

    @property
    def param_names(self) -> list[str]:
        return ["sigma2_obs", "sigma2_level"]

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """exp() to ensure positive variances."""
        return np.exp(unconstrained)

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """log() inverse of exp transform."""
        return np.log(constrained)

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build SSM matrices for the local level model.

        Parameters
        ----------
        params : NDArray
            [sigma2_obs, sigma2_level]
        """
        sigma2_obs = params[0]
        sigma2_level = params[1]

        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[1.0]])
        ssm.Z = np.array([[1.0]])
        ssm.R = np.array([[1.0]])
        ssm.H = np.array([[sigma2_obs]])
        ssm.Q = np.array([[sigma2_level]])
        ssm.a1 = np.array([0.0])
        ssm.P1 = np.array([[config.diffuse_initial_variance]])

        return ssm
