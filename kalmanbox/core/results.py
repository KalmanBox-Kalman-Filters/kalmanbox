"""State-space model results container."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from kalmanbox.core.representation import StateSpaceRepresentation
from kalmanbox.filters.kalman import FilterOutput
from kalmanbox.smoothers.rts import SmootherOutput


class StateSpaceResults:
    """Container for state-space model results after fit/filter/smooth.

    Parameters
    ----------
    params : NDArray
        Estimated parameters.
    param_names : list[str]
        Parameter names.
    se : NDArray
        Standard errors.
    loglike : float
        Log-likelihood.
    nobs : int
        Number of observations.
    filter_output : FilterOutput
        Kalman filter output.
    smoother_output : SmootherOutput | None
        RTS smoother output, if smoothing was performed.
    ssm : StateSpaceRepresentation
        The state-space representation with final parameters.
    optimizer_converged : bool
        Whether the optimizer converged.
    optimizer_message : str
        Optimizer convergence message.
    """

    def __init__(
        self,
        params: NDArray[np.float64],
        param_names: list[str],
        se: NDArray[np.float64],
        loglike: float,
        nobs: int,
        filter_output: FilterOutput,
        smoother_output: SmootherOutput | None,
        ssm: StateSpaceRepresentation,
        optimizer_converged: bool = True,
        optimizer_message: str = "",
    ) -> None:
        self.params = params
        self.param_names = param_names
        self.se = se
        self.loglike = loglike
        self.nobs = nobs
        self.filter_output = filter_output
        self.smoother_output = smoother_output
        self.ssm = ssm
        self.optimizer_converged = optimizer_converged
        self.optimizer_message = optimizer_message

    @property
    def k_params(self) -> int:
        """Number of estimated parameters."""
        return len(self.params)

    @property
    def tvalues(self) -> NDArray[np.float64]:
        """t-statistics: params / se."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.params / self.se

    @property
    def pvalues(self) -> NDArray[np.float64]:
        """Two-sided p-values from t-distribution."""
        tvals = self.tvalues
        return np.array(
            [2.0 * (1.0 - stats.t.cdf(abs(t), df=self.nobs - self.k_params)) for t in tvals]
        )

    @property
    def aic(self) -> float:
        """Akaike Information Criterion: -2 * loglike + 2 * k."""
        return -2.0 * self.loglike + 2.0 * self.k_params

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion: -2 * loglike + k * log(n)."""
        return -2.0 * self.loglike + self.k_params * np.log(self.nobs)

    @property
    def hqic(self) -> float:
        """Hannan-Quinn Information Criterion: -2 * loglike + 2 * k * log(log(n))."""
        return -2.0 * self.loglike + 2.0 * self.k_params * np.log(np.log(self.nobs))

    @property
    def filtered_state(self) -> NDArray[np.float64]:
        """Filtered state estimates E[alpha_t | y_{1:t}]."""
        return self.filter_output.filtered_state

    @property
    def filtered_cov(self) -> NDArray[np.float64]:
        """Filtered state covariances Var[alpha_t | y_{1:t}]."""
        return self.filter_output.filtered_cov

    @property
    def smoothed_state(self) -> NDArray[np.float64] | None:
        """Smoothed state estimates E[alpha_t | y_{1:T}], or None."""
        if self.smoother_output is None:
            return None
        return self.smoother_output.smoothed_state

    @property
    def smoothed_cov(self) -> NDArray[np.float64] | None:
        """Smoothed state covariances Var[alpha_t | y_{1:T}], or None."""
        if self.smoother_output is None:
            return None
        return self.smoother_output.smoothed_cov

    @property
    def residuals(self) -> NDArray[np.float64]:
        """Prediction errors v_t."""
        return self.filter_output.residuals

    @property
    def residuals_cov(self) -> NDArray[np.float64]:
        """Prediction error covariances F_t."""
        return self.filter_output.forecast_cov

    @property
    def fitted_values(self) -> NDArray[np.float64]:
        """Fitted values: Z @ filtered_state."""
        return self.filter_output.forecast

    def forecast(
        self,
        steps: int,
        alpha: float = 0.05,
    ) -> dict[str, NDArray[np.float64]]:
        """Forecast h steps ahead with confidence intervals.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        dict
            Keys: 'mean', 'lower', 'upper', 'state_mean', 'state_cov'.
        """
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)

        # Start from last filtered state
        a_t = self.filtered_state[-1].copy()
        p_t = self.filtered_cov[-1].copy()

        forecast_mean = np.zeros((steps, self.ssm.k_endog))
        forecast_lower = np.zeros((steps, self.ssm.k_endog))
        forecast_upper = np.zeros((steps, self.ssm.k_endog))
        state_mean = np.zeros((steps, self.ssm.k_states))
        state_cov = np.zeros((steps, self.ssm.k_states, self.ssm.k_states))

        t_mat = self.ssm.T
        z_mat = self.ssm.Z
        r_mat = self.ssm.R
        q_mat = self.ssm.Q
        h_mat = self.ssm.H
        c_vec = self.ssm.c
        d_vec = self.ssm.d

        for h in range(steps):
            # Predict state
            a_t = t_mat @ a_t + c_vec
            p_t = t_mat @ p_t @ t_mat.T + r_mat @ q_mat @ r_mat.T

            state_mean[h] = a_t
            state_cov[h] = p_t

            # Forecast observation
            y_hat = z_mat @ a_t + d_vec
            f_t = z_mat @ p_t @ z_mat.T + h_mat

            forecast_mean[h] = y_hat
            se = np.sqrt(np.diag(f_t))
            forecast_lower[h] = y_hat - z_alpha * se
            forecast_upper[h] = y_hat + z_alpha * se

        return {
            "mean": forecast_mean,
            "lower": forecast_lower,
            "upper": forecast_upper,
            "state_mean": state_mean,
            "state_cov": state_cov,
        }

    def summary(self) -> str:
        """Generate a formatted summary table.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        width = 70

        lines.append("=" * width)
        lines.append("State Space Model Results".center(width))
        lines.append("=" * width)

        lines.append(f"{'Log-Likelihood:':<30} {self.loglike:>15.4f}")
        lines.append(f"{'AIC:':<30} {self.aic:>15.4f}")
        lines.append(f"{'BIC:':<30} {self.bic:>15.4f}")
        lines.append(f"{'HQIC:':<30} {self.hqic:>15.4f}")
        lines.append(f"{'Num. observations:':<30} {self.nobs:>15d}")
        lines.append(f"{'Num. parameters:':<30} {self.k_params:>15d}")
        lines.append(f"{'Converged:':<30} {str(self.optimizer_converged):>15}")

        lines.append("-" * width)
        header = (
            f"{'Parameter':<20} {'Estimate':>12} {'Std.Err':>12} {'t-value':>12} {'p-value':>12}"
        )
        lines.append(header)
        lines.append("-" * width)

        for i, name in enumerate(self.param_names):
            est = self.params[i]
            se = self.se[i]
            tv = self.tvalues[i]
            pv = self.pvalues[i]
            lines.append(f"{name:<20} {est:>12.4f} {se:>12.4f} {tv:>12.4f} {pv:>12.4f}")

        lines.append("=" * width)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert parameter results to a pandas DataFrame."""
        return pd.DataFrame(
            {
                "estimate": self.params,
                "std_error": self.se,
                "t_value": self.tvalues,
                "p_value": self.pvalues,
            },
            index=self.param_names,
        )

    def save(self, path: str | Path) -> None:
        """Save results to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> StateSpaceResults:
        """Load results from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301  # nosec B301
