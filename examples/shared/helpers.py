"""Helper functions for kalmanbox examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_example_data(name: str, data_dir: str) -> pd.DataFrame:
    """Load example dataset by name.

    Parameters
    ----------
    name : str
        Dataset name: 'nile', 'airline', or 'uk_drivers'.
    data_dir : str
        Path to the directory containing CSV files.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with appropriate index.
    """
    path = Path(data_dir) / f"{name}.csv"
    df = pd.read_csv(path)

    if name == "nile":
        df = df.set_index("year")
    elif name == "airline":
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif name == "uk_drivers":
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    return df


def print_model_summary(
    model_name: str,
    log_likelihood: float,
    aic: float,
    bic: float,
    params_dict: dict[str, float],
) -> None:
    """Print formatted model summary table.

    Parameters
    ----------
    model_name : str
        Name of the model.
    log_likelihood : float
        Log-likelihood value.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    params_dict : dict
        Dictionary of parameter names to estimated values.
    """
    width = 50
    print("=" * width)
    print(f"  {model_name}")
    print("=" * width)
    print(f"  Log-Likelihood : {log_likelihood:>15.4f}")
    print(f"  AIC            : {aic:>15.4f}")
    print(f"  BIC            : {bic:>15.4f}")
    print("-" * width)
    print("  Parameters:")
    for name, value in params_dict.items():
        print(f"    {name:<25s} {value:>12.6f}")
    print("=" * width)


def compare_implementations(
    kalmanbox_result: np.ndarray,
    reference_result: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Compare kalmanbox vs reference implementation results.

    Parameters
    ----------
    kalmanbox_result : np.ndarray
        Results from kalmanbox.
    reference_result : np.ndarray
        Results from reference implementation (e.g., statsmodels).
    label : str
        Label for the comparison.

    Returns
    -------
    dict
        Dictionary with comparison metrics: max_abs_diff, mean_abs_diff, correlation.
    """
    kb = np.asarray(kalmanbox_result).ravel()
    ref = np.asarray(reference_result).ravel()

    max_abs_diff = float(np.max(np.abs(kb - ref)))
    mean_abs_diff = float(np.mean(np.abs(kb - ref)))
    correlation = float(np.corrcoef(kb, ref)[0, 1]) if len(kb) > 1 else 1.0

    header = f"Comparison: {label}" if label else "Comparison"
    print(f"\n  {header}")
    print(f"    Max  |diff| : {max_abs_diff:.6e}")
    print(f"    Mean |diff| : {mean_abs_diff:.6e}")
    print(f"    Correlation  : {correlation:.10f}")

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "correlation": correlation,
    }
