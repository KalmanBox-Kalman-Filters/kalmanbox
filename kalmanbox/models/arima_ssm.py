"""ARIMA model in state-space form."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


class ARIMA_SSM(StateSpaceModel):  # noqa: N801
    """ARIMA(p,d,q) model in state-space representation.

    Uses the companion form to represent ARMA(p,q) (after differencing)
    as a linear Gaussian state-space model.

    Parameters
    ----------
    endog : NDArray
        Observed time series data.
    order : tuple[int, int, int]
        (p, d, q) - AR order, differencing order, MA order.
    seasonal_order : tuple[int, int, int, int] | None
        (P, D, Q, s) - Seasonal AR, seasonal differencing, seasonal MA, period.
        If None, no seasonal component.
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._p, self._d, self._q = order

        if seasonal_order is not None:
            self._P, self._D, self._Q, self._s = seasonal_order
        else:
            self._P, self._D, self._Q, self._s = 0, 0, 0, 1

        # Apply differencing
        y = np.asarray(endog, dtype=np.float64).copy()
        # Seasonal differencing
        for _ in range(self._D):
            y = y[self._s :] - y[: -self._s]
        # Regular differencing
        for _ in range(self._d):
            y = np.diff(y)

        self._differenced_endog = y

        # Total AR and MA orders (including seasonal)
        self._total_p = self._p + self._P * self._s
        self._total_q = self._q + self._Q * self._s
        self._m = max(self._total_p, self._total_q + 1)

        super().__init__(y)

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial guesses: small AR/MA coefficients, sample variance."""
        n_ar = self._total_p
        n_ma = self._total_q
        params: list[float] = []

        # AR coefficients: small values
        for i in range(n_ar):
            if i == 0:
                params.append(0.5)
            else:
                params.append(0.0)

        # MA coefficients: small values
        for i in range(n_ma):
            if i == 0:
                params.append(0.3)
            else:
                params.append(0.0)

        # sigma2: sample variance
        params.append(float(np.nanvar(self.endog)))

        return np.array(params)

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        names: list[str] = []
        for i in range(1, self._total_p + 1):
            names.append(f"phi_{i}")
        for i in range(1, self._total_q + 1):
            names.append(f"theta_{i}")
        names.append("sigma2")
        return names

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform to constrained space.

        AR/MA coefficients: identity (no constraint enforcement)
        sigma2: exp(x)
        """
        constrained = unconstrained.copy()
        constrained[-1] = np.exp(unconstrained[-1])  # sigma2 > 0
        return constrained

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform to unconstrained space."""
        unconstrained = constrained.copy()
        unconstrained[-1] = np.log(constrained[-1])
        return unconstrained

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build companion form SSM."""
        n_ar = self._total_p
        n_ma = self._total_q
        m = self._m

        phi = params[:n_ar]
        theta = params[n_ar : n_ar + n_ma]
        sigma2 = params[-1]

        ssm = StateSpaceRepresentation(k_states=m, k_endog=1, k_posdef=1)

        # Transition matrix (companion form)
        T = np.zeros((m, m))
        for i in range(n_ar):
            T[0, i] = phi[i]
        for i in range(1, m):
            T[i, i - 1] = 1.0
        ssm.T = T

        # Design matrix
        Z = np.zeros((1, m))
        Z[0, 0] = 1.0
        for i in range(n_ma):
            if i + 1 < m:
                Z[0, i + 1] = theta[i]
        ssm.Z = Z

        # Selection matrix
        R = np.zeros((m, 1))
        R[0, 0] = 1.0
        ssm.R = R

        # Covariance matrices
        ssm.Q = np.array([[sigma2]])
        ssm.H = np.array([[0.0]])  # No observation noise in ARIMA

        # Initial conditions
        ssm.a1 = np.zeros(m)
        ssm.P1 = np.eye(m) * config.diffuse_initial_variance

        return ssm
