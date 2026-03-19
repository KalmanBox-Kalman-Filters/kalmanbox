"""KalmanBox CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kalmanbox.__version__ import __version__

# ---------------------------------------------------------------------------
# Model registry: maps CLI names to (module_path, class_name)
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "local_level": ("kalmanbox.models.local_level", "LocalLevel"),
    "local_linear_trend": ("kalmanbox.models.local_linear_trend", "LocalLinearTrend"),
    "bsm": ("kalmanbox.models.bsm", "BasicStructuralModel"),
    "ucm": ("kalmanbox.models.ucm", "UnobservedComponents"),
    "arima": ("kalmanbox.models.arima_ssm", "ARIMA_SSM"),
    "tvp": ("kalmanbox.models.tvp", "TimeVaryingParameters"),
}


def _get_model_class(name: str) -> type:  # type: ignore[type-arg]
    """Dynamically import and return a model class by CLI name."""
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        print(f"Error: unknown model '{name}'. Available: {available}", file=sys.stderr)
        sys.exit(1)

    module_path, class_name = _MODEL_REGISTRY[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def _load_data(path: str) -> pd.DataFrame:
    """Load CSV data file."""
    data_path = Path(path)
    if not data_path.exists():
        print(f"Error: data file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(data_path)  # type: ignore[no-any-return]


def _build_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build keyword arguments for model constructor from CLI args."""
    kwargs: dict[str, Any] = {}
    if hasattr(args, "seasonal_period") and args.seasonal_period is not None:
        kwargs["seasonal_period"] = args.seasonal_period
    if hasattr(args, "order") and args.order is not None:
        parts = [int(x) for x in args.order.split(",")]
        if len(parts) == 3:
            kwargs["order"] = tuple(parts)
        else:
            print("Error: --order must be p,d,q (e.g. 1,1,1)", file=sys.stderr)
            sys.exit(1)
    if hasattr(args, "exog") and args.exog is not None:
        exog_path = Path(args.exog)
        if not exog_path.exists():
            print(f"Error: exog file not found: {args.exog}", file=sys.stderr)
            sys.exit(1)
        exog_df = pd.read_csv(exog_path)
        kwargs["exog"] = exog_df.to_numpy(dtype=np.float64)
    return kwargs


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, o: Any) -> Any:
        """Encode numpy types to JSON-serializable Python types."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def estimate_command(args: argparse.Namespace) -> None:
    """Execute the 'estimate' command."""
    data = _load_data(args.data)

    # Determine the target column (first numeric column)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("Error: no numeric columns found in data", file=sys.stderr)
        sys.exit(1)

    y = data[numeric_cols[0]].to_numpy(dtype=np.float64)
    model_class = _get_model_class(args.model)
    kwargs = _build_model_kwargs(args)

    model = model_class(y, **kwargs)
    results = model.fit()

    # Build output dict
    output: dict[str, Any] = {
        "model": args.model,
        "data": args.data,
        "n_obs": len(y),
        "loglike": float(results.loglike),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "params": {
            name: float(val) for name, val in zip(results.param_names, results.params, strict=True)
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    print(f"Results saved to {output_path}")
    print(f"  Model: {args.model}")
    print(f"  LogLike: {output['loglike']:.4f}")
    print(f"  AIC: {output['aic']:.4f}")
    print(f"  BIC: {output['bic']:.4f}")


def info_command(args: argparse.Namespace) -> None:
    """Execute the 'info' command."""
    model_class = _get_model_class(args.model)

    # Create a dummy model with minimal data to inspect structure
    dummy_y = np.ones(10, dtype=np.float64)
    kwargs = _build_model_kwargs(args)

    try:
        model = model_class(dummy_y, **kwargs)
    except Exception as e:
        print(f"Error creating model: {e}", file=sys.stderr)
        sys.exit(1)

    # Build representation from start params to inspect structure
    rep = model._build_ssm(model.start_params)  # noqa: SLF001
    print(f"Model: {args.model}")
    print(f"  k_states: {rep.k_states}")
    print(f"  k_obs: {rep.k_endog}")
    print(f"  State dimensions: T={rep.T.shape}, Z={rep.Z.shape}")
    print(f"  Parameters: {model.param_names}")
    print(f"  n_params: {len(model.param_names)}")


def forecast_command(args: argparse.Namespace) -> None:
    """Execute the 'forecast' command."""
    data = _load_data(args.data)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("Error: no numeric columns found in data", file=sys.stderr)
        sys.exit(1)

    y = data[numeric_cols[0]].to_numpy(dtype=np.float64)
    model_class = _get_model_class(args.model)
    kwargs = _build_model_kwargs(args)

    model = model_class(y, **kwargs)
    results = model.fit()

    fc = results.forecast(steps=args.steps)

    # The forecast dict uses 'lower'/'upper' keys; arrays may be 2D (steps, k_endog)
    lower_key = "lower_95" if "lower_95" in fc else "lower"
    upper_key = "upper_95" if "upper_95" in fc else "upper"

    mean_arr = np.asarray(fc["mean"]).flatten()
    lower_arr = np.asarray(fc[lower_key]).flatten()
    upper_arr = np.asarray(fc[upper_key]).flatten()

    # Build forecast DataFrame
    fc_df = pd.DataFrame(
        {
            "step": list(range(1, args.steps + 1)),
            "mean": mean_arr,
            "lower_95": lower_arr,
            "upper_95": upper_arr,
        }
    )

    output_path = Path(args.output)
    fc_df.to_csv(output_path, index=False)

    print(f"Forecast saved to {output_path}")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  Forecast mean (first 5): {mean_arr[:5].tolist()}")


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------
def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add common model arguments to a subparser."""
    parser.add_argument(
        "--model",
        required=True,
        choices=list(_MODEL_REGISTRY.keys()),
        help="Model type to use",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=None,
        help="Seasonal period (required for bsm, ucm with seasonal)",
    )
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help="ARIMA order as p,d,q (e.g. 1,1,1)",
    )
    parser.add_argument(
        "--exog",
        type=str,
        default=None,
        help="Path to CSV with exogenous variables (for tvp)",
    )


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="kalmanbox",
        description="KalmanBox: State-space models and Kalman filtering for time series",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"kalmanbox {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- estimate ---
    est = subparsers.add_parser(
        "estimate",
        help="Estimate a state-space model from data",
    )
    _add_model_args(est)
    est.add_argument("--data", required=True, help="Path to CSV data file")
    est.add_argument("--output", default="results.json", help="Output JSON file path")
    est.set_defaults(func=estimate_command)

    # --- info ---
    info = subparsers.add_parser(
        "info",
        help="Show model information (dimensions, parameters)",
    )
    _add_model_args(info)
    info.set_defaults(func=info_command)

    # --- forecast ---
    fc = subparsers.add_parser(
        "forecast",
        help="Fit model and produce forecasts",
    )
    _add_model_args(fc)
    fc.add_argument("--data", required=True, help="Path to CSV data file")
    fc.add_argument("--steps", type=int, required=True, help="Number of forecast steps")
    fc.add_argument("--output", default="forecast.csv", help="Output CSV file path")
    fc.set_defaults(func=forecast_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
