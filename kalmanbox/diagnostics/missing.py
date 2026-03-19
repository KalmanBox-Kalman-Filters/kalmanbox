"""Missing data handling for state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kalmanbox.core.results import StateSpaceResults


@dataclass
class MissingDataReport:
    """Report on missing data handling.

    Attributes
    ----------
    n_total : int
        Total number of observations.
    n_missing : int
        Number of missing observations.
    missing_indices : NDArray[np.intp]
        Indices of missing observations.
    missing_rate : float
        Fraction of missing observations.
    interpolated_values : NDArray[np.float64] | None
        Interpolated values for missing observations (if strategy='interpolate').
    interpolated_variances : NDArray[np.float64] | None
        Variances of interpolated values (if strategy='interpolate').
    """

    n_total: int
    n_missing: int
    missing_indices: NDArray[np.intp]
    missing_rate: float
    interpolated_values: NDArray[np.float64] | None = None
    interpolated_variances: NDArray[np.float64] | None = None

    def __repr__(self) -> str:
        """Return formatted missing data report."""
        lines = [
            "Missing Data Report",
            "=" * 40,
            f"Total observations:   {self.n_total}",
            f"Missing observations: {self.n_missing}",
            f"Missing rate:         {self.missing_rate:.1%}",
        ]
        if self.interpolated_values is not None:
            lines.append(f"Interpolated values:  {len(self.interpolated_values)}")
        return "\n".join(lines)


class MissingDataHandler:
    """Handler for missing data in state-space models.

    Provides two strategies for dealing with missing observations:
    - 'skip': Kalman filter skips the update step for NaN observations.
      The smoother provides state estimates for missing periods.
    - 'interpolate': After smoothing, interpolates missing values using
      the conditional expectation E[y_t | y_{-t}] = Z @ a_{t|T} + d.

    Parameters
    ----------
    strategy : {'skip', 'interpolate'}
        Missing data strategy. Default 'skip'.
    """

    def __init__(self, strategy: Literal["skip", "interpolate"] = "skip") -> None:
        if strategy not in ("skip", "interpolate"):
            msg = f"Unknown strategy: {strategy!r}. Must be 'skip' or 'interpolate'."
            raise ValueError(msg)
        self.strategy = strategy

    def prepare_data(
        self,
        endog: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Identify and prepare data with missing values.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).

        Returns
        -------
        endog_prepared : NDArray[np.float64]
            Data ready for Kalman filter (NaN preserved for skip strategy).
        missing_indices : NDArray[np.intp]
            Indices of missing observations.
        """
        missing_mask = np.isnan(endog) if endog.ndim == 1 else np.any(np.isnan(endog), axis=1)
        missing_indices = np.where(missing_mask)[0].astype(np.intp)
        endog_prepared = endog.copy()

        return endog_prepared, missing_indices

    def interpolate_missing(
        self,
        endog: NDArray[np.float64],
        results: StateSpaceResults,
    ) -> MissingDataReport:
        """Interpolate missing values using smoothed states.

        Uses E[y_t | y_{-t}] = Z @ a_{t|T} + d for missing periods.
        Variance: Var(y_t | y_{-t}) = Z @ P_{t|T} @ Z' + H.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Original data with NaN values.
        results : StateSpaceResults
            Fitted model results with smoother output.

        Returns
        -------
        MissingDataReport
            Report with interpolated values and variances.
        """
        is_1d = endog.ndim == 1
        endog_2d = endog.reshape(-1, 1) if is_1d else endog

        nobs = endog_2d.shape[0]

        # Find missing indices
        missing_mask = np.any(np.isnan(endog_2d), axis=1)
        missing_indices = np.where(missing_mask)[0].astype(np.intp)
        n_missing = len(missing_indices)

        if n_missing == 0:
            return MissingDataReport(
                n_total=nobs,
                n_missing=0,
                missing_indices=missing_indices,
                missing_rate=0.0,
            )

        ssm = results.ssm
        Z = ssm.Z
        H = ssm.H
        d = ssm.d

        smoothed_state = results.smoother_output.smoothed_state
        smoothed_cov = results.smoother_output.smoothed_cov

        k_endog = Z.shape[0]
        interpolated_values = np.zeros((n_missing, k_endog))
        interpolated_variances = np.zeros((n_missing, k_endog))

        for i, t in enumerate(missing_indices):
            # E[y_t | y_{-t}] = Z @ a_{t|T} + d
            y_hat = Z @ smoothed_state[t] + d
            interpolated_values[i] = y_hat

            # Var(y_t | y_{-t}) = Z @ P_{t|T} @ Z' + H
            var_y = Z @ smoothed_cov[t] @ Z.T + H
            interpolated_variances[i] = np.diag(var_y)

        if is_1d:
            interpolated_values = interpolated_values.ravel()
            interpolated_variances = interpolated_variances.ravel()

        return MissingDataReport(
            n_total=nobs,
            n_missing=n_missing,
            missing_indices=missing_indices,
            missing_rate=n_missing / nobs,
            interpolated_values=interpolated_values,
            interpolated_variances=interpolated_variances,
        )

    def fill_missing(
        self,
        endog: NDArray[np.float64],
        results: StateSpaceResults,
    ) -> NDArray[np.float64]:
        """Return data with missing values replaced by interpolated values.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Original data with NaN values.
        results : StateSpaceResults
            Fitted model results.

        Returns
        -------
        NDArray[np.float64]
            Data with missing values filled in.
        """
        report = self.interpolate_missing(endog, results)
        endog_filled = endog.copy()

        if report.n_missing > 0 and report.interpolated_values is not None:
            for i, idx in enumerate(report.missing_indices):
                if endog.ndim == 1:
                    endog_filled[idx] = report.interpolated_values[i]
                else:
                    endog_filled[idx] = report.interpolated_values[i]

        return endog_filled
