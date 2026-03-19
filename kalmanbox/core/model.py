"""Abstract base class for state-space models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.core.results import StateSpaceResults
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.rts import RTSSmoother


class StateSpaceModel(ABC):
    """Abstract base class for state-space models.

    Subclasses must implement the abstract methods to define the
    model structure and parameter mapping.

    Parameters
    ----------
    endog : NDArray
        Observed time series data, shape (nobs,) or (nobs, k_endog).
    """

    def __init__(self, endog: NDArray[np.float64]) -> None:
        self.endog = np.asarray(endog, dtype=np.float64)
        if self.endog.ndim == 1:
            self.endog = self.endog.reshape(-1, 1)
        self.nobs = self.endog.shape[0]
        self.k_endog = self.endog.shape[1]

        self._kf = KalmanFilter()
        self._smoother = RTSSmoother()

    # --- Abstract methods (subclass MUST implement) ---

    @abstractmethod
    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build a StateSpaceRepresentation from parameters.

        Parameters
        ----------
        params : NDArray
            Constrained (transformed) parameters.

        Returns
        -------
        StateSpaceRepresentation
        """
        ...

    @property
    @abstractmethod
    def start_params(self) -> NDArray[np.float64]:
        """Initial parameter guesses for optimization."""
        ...

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Parameter names."""
        ...

    @abstractmethod
    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform unconstrained parameters to constrained space."""
        ...

    @abstractmethod
    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constrained parameters to unconstrained space."""
        ...

    # --- Concrete methods ---

    def fit(self, method: str = "mle", **kwargs: object) -> StateSpaceResults:
        """Fit the model by estimating parameters.

        Parameters
        ----------
        method : str
            Estimation method. Currently only 'mle' is supported.
        **kwargs
            Additional keyword arguments passed to the estimator.

        Returns
        -------
        StateSpaceResults
        """
        # Import here to avoid circular imports
        from kalmanbox.estimation.mle import MLEstimator

        if method != "mle":
            msg = f"Unknown estimation method: {method}. Use 'mle'."
            raise ValueError(msg)

        estimator = MLEstimator()
        return estimator.fit(self, self.endog, **kwargs)

    def loglike(self, params: NDArray[np.float64]) -> float:
        """Compute log-likelihood for given parameters.

        Parameters
        ----------
        params : NDArray
            Constrained parameters.

        Returns
        -------
        float
            Log-likelihood value.
        """
        ssm = self._build_ssm(params)
        output = self._kf.filter(self.endog, ssm)
        return output.loglike

    def filter(self, params: NDArray[np.float64] | None = None) -> StateSpaceResults:
        """Run the Kalman filter with given or start parameters.

        Parameters
        ----------
        params : NDArray, optional
            Constrained parameters. If None, uses start_params.

        Returns
        -------
        StateSpaceResults
        """
        if params is None:
            params = self.start_params
        ssm = self._build_ssm(params)
        filter_output = self._kf.filter(self.endog, ssm)
        return StateSpaceResults(
            params=params,
            param_names=self.param_names,
            se=np.full_like(params, np.nan),
            loglike=filter_output.loglike,
            nobs=self.nobs,
            filter_output=filter_output,
            smoother_output=None,
            ssm=ssm,
            optimizer_converged=True,
            optimizer_message="Filter only (no estimation)",
        )

    def smooth(self, params: NDArray[np.float64] | None = None) -> StateSpaceResults:
        """Run filter + smoother with given or start parameters.

        Parameters
        ----------
        params : NDArray, optional
            Constrained parameters. If None, uses start_params.

        Returns
        -------
        StateSpaceResults
        """
        if params is None:
            params = self.start_params
        ssm = self._build_ssm(params)
        filter_output = self._kf.filter(self.endog, ssm)
        smoother_output = self._smoother.smooth(filter_output, ssm)
        return StateSpaceResults(
            params=params,
            param_names=self.param_names,
            se=np.full_like(params, np.nan),
            loglike=filter_output.loglike,
            nobs=self.nobs,
            filter_output=filter_output,
            smoother_output=smoother_output,
            ssm=ssm,
            optimizer_converged=True,
            optimizer_message="Smooth only (no estimation)",
        )

    def simulate(
        self,
        n: int,
        params: NDArray[np.float64] | None = None,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate observations and states from the model.

        Parameters
        ----------
        n : int
            Number of time steps to simulate.
        params : NDArray, optional
            Constrained parameters. If None, uses start_params.
        seed : int, optional
            Random seed.

        Returns
        -------
        y : NDArray, shape (n, k_endog)
            Simulated observations.
        states : NDArray, shape (n, k_states)
            Simulated states.
        """
        if params is None:
            params = self.start_params
        ssm = self._build_ssm(params)
        rng = np.random.default_rng(seed)

        states = np.zeros((n, ssm.k_states))
        y = np.zeros((n, ssm.k_endog))

        # Initial state
        alpha = rng.multivariate_normal(ssm.a1, ssm.P1)

        for t in range(n):
            # State equation noise
            eta = rng.multivariate_normal(np.zeros(ssm.k_posdef), ssm.Q)
            # Observation equation noise
            eps = rng.multivariate_normal(np.zeros(ssm.k_endog), ssm.H)

            if t > 0:
                alpha = ssm.T @ alpha + ssm.R @ eta + ssm.c

            states[t] = alpha
            y[t] = ssm.Z @ alpha + eps + ssm.d

        return y, states
