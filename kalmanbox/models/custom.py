"""Custom state-space model interface."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


class CustomStateSpace(StateSpaceModel):
    """Convenience base class for user-defined state-space models.

    Simplifies creating custom models by:
    - Accepting k_states, k_endog, k_posdef in constructor
    - Providing create_ssm() helper
    - Providing default transform/untransform (exp/log for all params)
    - Auto-validating the SSM after build

    Subclasses must implement:
    - _build_ssm(params) -> StateSpaceRepresentation
    - start_params -> NDArray
    - param_names -> list[str]

    Optionally override:
    - transform_params / untransform_params

    Parameters
    ----------
    endog : NDArray
        Observed time series data.
    k_states : int
        Number of state variables.
    k_endog : int
        Number of observed variables. Default 1.
    k_posdef : int | None
        Dimension of state disturbance selection. Default k_states.

    Example
    -------
    ```python
    class MyModel(CustomStateSpace):
        def __init__(self, endog):
            super().__init__(endog, k_states=2, k_endog=1)

        def _build_ssm(self, params):
            ssm = self.create_ssm()
            ssm.T = np.array([[1.0, 1.0], [0.0, 1.0]])
            ssm.Z = np.array([[1.0, 0.0]])
            ssm.R = np.eye(2)
            ssm.H = np.array([[params[0]]])
            ssm.Q = np.diag([params[1], params[2]])
            ssm.P1 = np.eye(2) * 1e7
            return ssm

        @property
        def start_params(self):
            return np.array([1.0, 0.5, 0.1])

        @property
        def param_names(self):
            return ['sigma2_obs', 'sigma2_level', 'sigma2_trend']
    ```
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        k_states: int,
        k_endog: int = 1,
        k_posdef: int | None = None,
    ) -> None:
        self._custom_k_states = k_states
        self._custom_k_endog = k_endog
        self._custom_k_posdef = k_posdef if k_posdef is not None else k_states
        super().__init__(endog)

    def create_ssm(self) -> StateSpaceRepresentation:
        """Create a StateSpaceRepresentation with the configured dimensions.

        Returns a fresh SSM initialized with zeros and diffuse P1.
        """
        ssm = StateSpaceRepresentation(
            k_states=self._custom_k_states,
            k_endog=self._custom_k_endog,
            k_posdef=self._custom_k_posdef,
        )
        ssm.P1 = np.eye(self._custom_k_states) * config.diffuse_initial_variance
        return ssm

    @abstractmethod
    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build a StateSpaceRepresentation from parameters.

        Use self.create_ssm() to create the base SSM, then set the matrices.
        """
        ...

    @property
    @abstractmethod
    def start_params(self) -> NDArray[np.float64]: ...

    @property
    @abstractmethod
    def param_names(self) -> list[str]: ...

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Default transform: exp() for all parameters (assumes all positive).

        Override this for parameters with different constraints.
        """
        return np.exp(unconstrained)

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Default untransform: log() for all parameters."""
        return np.log(constrained)
