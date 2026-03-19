"""Result classes for model comparison and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ComparisonResult:
    """Result of comparing multiple models.

    Parameters
    ----------
    model_names : list[str]
        Names of the compared models.
    metrics : dict[str, dict[str, float]]
        Nested dict: {model_name: {metric_name: value}}.
    criteria : list[str]
        Criteria used for comparison (e.g. ['aic', 'bic', 'loglike']).
    """

    model_names: list[str]
    metrics: dict[str, dict[str, float]]
    criteria: list[str]

    def ranking(self, criterion: str | None = None) -> list[tuple[str, float]]:
        """Return models ranked by criterion (default: first criterion).

        Parameters
        ----------
        criterion : str or None
            Criterion to rank by. If None, uses the first criterion.
            For 'aic' and 'bic', lower is better.
            For 'loglike', higher is better.

        Returns
        -------
        list[tuple[str, float]]
            List of (model_name, value) sorted best-to-worst.
        """
        if criterion is None:
            criterion = self.criteria[0]

        pairs = [
            (name, self.metrics[name][criterion])
            for name in self.model_names
            if criterion in self.metrics[name]
        ]

        # loglike: higher is better; aic/bic: lower is better
        reverse = criterion == "loglike"
        pairs.sort(key=lambda x: x[1], reverse=reverse)
        return pairs

    def best_model(self, criterion: str | None = None) -> str:
        """Return the name of the best model by criterion.

        Parameters
        ----------
        criterion : str or None
            Criterion to use. Default: first criterion.

        Returns
        -------
        str
            Name of the best model.
        """
        ranked = self.ranking(criterion)
        return ranked[0][0]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparison to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with models as rows and criteria as columns.
        """
        rows = []
        for name in self.model_names:
            row: dict[str, Any] = {"model": name}
            for crit in self.criteria:
                row[crit] = self.metrics[name].get(crit, np.nan)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def __repr__(self) -> str:
        return f"ComparisonResult(models={self.model_names}, criteria={self.criteria})"


@dataclass
class ValidationResult:
    """Result of out-of-sample model validation.

    Parameters
    ----------
    model_name : str
        Name of the validated model.
    y_test : NDArray[np.float64]
        Actual test values.
    y_forecast : NDArray[np.float64]
        Forecasted values.
    y_lower : NDArray[np.float64]
        Lower confidence interval.
    y_upper : NDArray[np.float64]
        Upper confidence interval.
    train_size : int
        Number of training observations.
    test_size : int
        Number of test observations.
    horizon : int
        Forecast horizon used.
    """

    model_name: str
    y_test: NDArray[np.float64]
    y_forecast: NDArray[np.float64]
    y_lower: NDArray[np.float64]
    y_upper: NDArray[np.float64]
    train_size: int
    test_size: int
    horizon: int

    def rmse(self) -> float:
        """Root Mean Squared Error.

        Returns
        -------
        float
            RMSE value.
        """
        return float(np.sqrt(np.mean((self.y_test - self.y_forecast) ** 2)))

    def mae(self) -> float:
        """Mean Absolute Error.

        Returns
        -------
        float
            MAE value.
        """
        return float(np.mean(np.abs(self.y_test - self.y_forecast)))

    def mape(self) -> float:
        """Mean Absolute Percentage Error.

        Returns
        -------
        float
            MAPE value as percentage.
        """
        nonzero = self.y_test != 0
        if not np.any(nonzero):
            return float("inf")
        return float(
            100.0
            * np.mean(
                np.abs((self.y_test[nonzero] - self.y_forecast[nonzero]) / self.y_test[nonzero])
            )
        )

    def coverage(self, level: float = 0.95) -> float:
        """Proportion of test values within the confidence interval.

        Parameters
        ----------
        level : float
            Nominal confidence level (for reference only, uses stored bounds).

        Returns
        -------
        float
            Coverage proportion in [0, 1].
        """
        inside = (self.y_test >= self.y_lower) & (self.y_test <= self.y_upper)
        return float(np.mean(inside))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with actual, forecast, and bounds.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: actual, forecast, lower_95, upper_95.
        """
        return pd.DataFrame(
            {
                "actual": self.y_test,
                "forecast": self.y_forecast,
                "lower_95": self.y_lower,
                "upper_95": self.y_upper,
            }
        )

    def summary(self) -> dict[str, float]:
        """Summary metrics.

        Returns
        -------
        dict[str, float]
            Dict with rmse, mae, mape, coverage.
        """
        return {
            "rmse": self.rmse(),
            "mae": self.mae(),
            "mape": self.mape(),
            "coverage": self.coverage(),
        }

    def __repr__(self) -> str:
        return (
            f"ValidationResult(model={self.model_name!r}, "
            f"rmse={self.rmse():.4f}, mae={self.mae():.4f}, "
            f"mape={self.mape():.2f}%)"
        )
