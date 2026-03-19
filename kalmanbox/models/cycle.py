"""Stochastic Cycle model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def _logit(p: float) -> float:
    """Inverse sigmoid."""
    return float(np.log(p / (1.0 - p)))


class CycleModel(StateSpaceModel):
    """Stochastic Cycle model (Harvey 1989).

    Model:
        y_t = c_t + eps_t
        [c_t  ]   [rho*cos(lam)  rho*sin(lam) ] [c_{t-1} ]   [kappa_t ]
        [c*_t ] = [-rho*sin(lam) rho*cos(lam) ] [c*_{t-1}] + [kappa*_t]

    Where:
        rho in (0, 1) - damping factor
        lambda_c in (0, pi) - cycle frequency
        Period = 2*pi / lambda_c

    Parameters
    ----------
    endog : NDArray
        Observed time series data.
    """

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial guesses.

        sigma2_obs: half variance
        rho: 0.9 (persistent cycle)
        lambda_c: 2*pi/40 (40-period cycle)
        sigma2_cycle: quarter variance
        """
        var = float(np.nanvar(self.endog))
        return np.array([var / 2.0, 0.9, 2.0 * np.pi / 40.0, var / 4.0])

    @property
    def param_names(self) -> list[str]:
        return ["sigma2_obs", "rho", "lambda_c", "sigma2_cycle"]

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform to constrained space.

        sigma2_obs:   exp(x)
        rho:          (tanh(x) + 1) / 2  -> (0, 1)
        lambda_c:     pi * sigmoid(x)    -> (0, pi)
        sigma2_cycle: exp(x)
        """
        constrained = np.empty_like(unconstrained)
        constrained[0] = np.exp(unconstrained[0])  # sigma2_obs > 0
        constrained[1] = (np.tanh(unconstrained[1]) + 1) / 2  # rho in (0, 1)
        constrained[2] = np.pi * _sigmoid(unconstrained[2])  # lambda_c in (0, pi)
        constrained[3] = np.exp(unconstrained[3])  # sigma2_cycle > 0
        return constrained

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform to unconstrained space."""
        unconstrained = np.empty_like(constrained)
        unconstrained[0] = np.log(constrained[0])
        unconstrained[1] = np.arctanh(2 * constrained[1] - 1)
        unconstrained[2] = _logit(constrained[2] / np.pi)
        unconstrained[3] = np.log(constrained[3])
        return unconstrained

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build cycle SSM."""
        sigma2_obs = params[0]
        rho = params[1]
        lambda_c = params[2]
        sigma2_cycle = params[3]

        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)

        cos_l = np.cos(lambda_c)
        sin_l = np.sin(lambda_c)

        ssm.T = np.array(
            [
                [rho * cos_l, rho * sin_l],
                [-rho * sin_l, rho * cos_l],
            ]
        )
        ssm.Z = np.array([[1.0, 0.0]])
        ssm.R = np.eye(2)
        ssm.H = np.array([[sigma2_obs]])
        ssm.Q = sigma2_cycle * np.eye(2)
        ssm.a1 = np.zeros(2)
        ssm.P1 = np.eye(2) * config.diffuse_initial_variance

        return ssm

    @property
    def cycle_period(self) -> float | None:
        """Return the estimated cycle period (after fit)."""
        return None
