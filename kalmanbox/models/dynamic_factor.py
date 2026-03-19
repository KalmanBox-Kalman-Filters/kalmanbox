"""Dynamic Factor Model (DFM)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.core.results import StateSpaceResults


class DynamicFactorModel(StateSpaceModel):
    """Dynamic Factor Model.

    Extracts common latent factors from multiple observed time series.

    Model:
        y_t = Lambda @ f_t + eps_t,    eps_t ~ N(0, R)    (R diagonal)
        f_t = Phi @ f_{t-1} + eta_t,   eta_t ~ N(0, I)    (Q = I for identification)

    Parameters
    ----------
    endog : NDArray
        Observed data, shape (nobs, k_endog). Multiple series as columns.
    k_factors : int
        Number of latent factors. Default 1.
    factor_order : int
        VAR order for factor dynamics. Default 1.
    endog_names : list[str] | None
        Names of the observed series.
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        k_factors: int = 1,
        factor_order: int = 1,
        endog_names: list[str] | None = None,
    ) -> None:
        self._k_factors = k_factors
        self._factor_order = factor_order
        self._endog_names = endog_names

        # Ensure 2D
        endog_arr = np.asarray(endog, dtype=np.float64)
        if endog_arr.ndim == 1:
            endog_arr = endog_arr.reshape(-1, 1)

        self._k_endog_actual = endog_arr.shape[1]
        self._k_states_total = k_factors * factor_order

        super().__init__(endog_arr)

    def _n_lambda_params(self) -> int:
        """Number of free parameters in Lambda (with identification)."""
        k = self._k_endog_actual
        r = self._k_factors
        # Lower-triangular for first r rows: r*(r+1)/2
        # Free for remaining rows: (k-r)*r
        return r * (r + 1) // 2 + (k - r) * r

    def _n_phi_params(self) -> int:
        """Number of parameters in Phi."""
        return self._k_factors * self._k_factors

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial parameters: Lambda (flattened), Phi, R_diag."""
        params: list[float] = []

        k = self._k_endog_actual
        r = self._k_factors

        # Lambda: PCA-based initialization
        y = self.endog
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Remove NaN rows for PCA
        valid = ~np.any(np.isnan(y), axis=1)
        y_valid = y[valid]

        if y_valid.shape[0] > r:
            # Simple PCA
            y_centered = y_valid - np.mean(y_valid, axis=0)
            cov = y_centered.T @ y_centered / y_valid.shape[0]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Take top r eigenvectors
            idx = np.argsort(eigenvalues)[::-1][:r]
            lambda_init = eigenvectors[:, idx]

            # Enforce lower-triangular identification for first r rows
            for i in range(r):
                for j in range(i + 1, r):
                    lambda_init[i, j] = 0.0
                if lambda_init[i, i] < 0:
                    lambda_init[i, :] = -lambda_init[i, :]
        else:
            lambda_init = np.eye(k, r)

        # Flatten Lambda (only free params)
        for i in range(k):
            for j in range(r):
                if i < r and j > i:
                    continue  # skip upper-triangle of first r rows
                params.append(lambda_init[i, j])

        # Phi: small diagonal
        for i in range(self._k_factors):
            for j in range(self._k_factors):
                if i == j:
                    params.append(0.5)
                else:
                    params.append(0.0)

        # R_diag: sample variance / 2 for each series
        for i in range(k):
            col = y_valid[:, i] if i < y_valid.shape[1] else y_valid[:, 0]
            params.append(float(np.nanvar(col) / 2))

        return np.array(params)

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        names: list[str] = []
        k = self._k_endog_actual
        r = self._k_factors

        for i in range(k):
            for j in range(r):
                if i < r and j > i:
                    continue
                series_name = self._endog_names[i] if self._endog_names else f"y{i}"
                names.append(f"lambda_{series_name}_f{j}")

        for i in range(self._k_factors):
            for j in range(self._k_factors):
                names.append(f"phi_{i}{j}")

        for i in range(k):
            series_name = self._endog_names[i] if self._endog_names else f"y{i}"
            names.append(f"sigma2_{series_name}")

        return names

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Lambda and Phi: identity. R_diag: exp."""
        constrained = unconstrained.copy()
        # Only transform R_diag (last k_endog params)
        k = self._k_endog_actual
        constrained[-k:] = np.exp(unconstrained[-k:])
        return constrained

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constrained parameters to unconstrained space."""
        unconstrained = constrained.copy()
        k = self._k_endog_actual
        unconstrained[-k:] = np.log(constrained[-k:])
        return unconstrained

    def _parse_params(
        self, params: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Parse parameter vector into Lambda, Phi, R_diag."""
        k = self._k_endog_actual
        r = self._k_factors

        n_lambda = self._n_lambda_params()
        n_phi = self._n_phi_params()

        lambda_params = params[:n_lambda]
        phi_params = params[n_lambda : n_lambda + n_phi]
        r_diag = params[n_lambda + n_phi :]

        # Reconstruct Lambda
        lambda_mat = np.zeros((k, r))
        idx = 0
        for i in range(k):
            for j in range(r):
                if i < r and j > i:
                    lambda_mat[i, j] = 0.0
                else:
                    lambda_mat[i, j] = lambda_params[idx]
                    idx += 1

        phi_mat = phi_params.reshape(self._k_factors, self._k_factors)

        return lambda_mat, phi_mat, r_diag

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build DFM state-space representation."""
        lambda_mat, phi_mat, r_diag = self._parse_params(params)

        k = self._k_endog_actual
        r = self._k_factors
        m = self._k_states_total

        ssm = StateSpaceRepresentation(k_states=m, k_endog=k, k_posdef=r)

        # Transition: companion form for VAR(p) of factors
        t_mat = np.zeros((m, m))
        t_mat[:r, :r] = phi_mat
        if self._factor_order > 1:
            t_mat[r:, : m - r] = np.eye(m - r)
        ssm.T = t_mat

        # Design: Lambda @ [f_t, 0, ...]
        z_mat = np.zeros((k, m))
        z_mat[:, :r] = lambda_mat
        ssm.Z = z_mat

        # Selection
        r_sel = np.zeros((m, r))
        r_sel[:r, :r] = np.eye(r)
        ssm.R = r_sel

        # Covariances
        ssm.Q = np.eye(r)  # Fixed for identification
        ssm.H = np.diag(r_diag)

        # Initial conditions
        ssm.a1 = np.zeros(m)
        ssm.P1 = np.eye(m) * config.diffuse_initial_variance

        return ssm

    @property
    def factors(self) -> None:
        """Placeholder -- access via results.smoothed_state[:, :k_factors]."""
        return None

    @property
    def loadings(self) -> None:
        """Placeholder -- access via results.ssm.Z[:, :k_factors]."""
        return None

    def variance_decomposition(self, results: StateSpaceResults) -> NDArray[np.float64]:
        """Compute variance decomposition: % explained by each factor.

        Parameters
        ----------
        results : StateSpaceResults
            Fitted model results.

        Returns
        -------
        NDArray, shape (k_endog, k_factors + 1)
            Columns are [factor_1, ..., factor_r, idiosyncratic].
            Values are proportions summing to 1 per row.
        """
        lambda_mat = results.ssm.Z[:, : self._k_factors]
        r_diag = np.diag(results.ssm.H)
        k = self._k_endog_actual
        r = self._k_factors

        decomp = np.zeros((k, r + 1))

        for i in range(k):
            total_var = np.sum(lambda_mat[i] ** 2) + r_diag[i]
            for j in range(r):
                decomp[i, j] = lambda_mat[i, j] ** 2 / total_var
            decomp[i, r] = r_diag[i] / total_var

        return decomp
