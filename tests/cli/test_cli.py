"""Tests for the kalmanbox CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalmanbox.cli.main import build_parser, main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file with Nile-like data."""
    data = pd.DataFrame(
        {
            "year": list(range(1871, 1971)),
            "volume": np.random.default_rng(42).normal(900, 150, 100).tolist(),
        }
    )
    csv_path = tmp_path / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tmp_exog_csv(tmp_path: Path) -> Path:
    """Create a temporary exog CSV file."""
    rng = np.random.default_rng(99)
    data = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 100).tolist(),
            "x2": rng.normal(0, 1, 100).tolist(),
        }
    )
    csv_path = tmp_path / "exog.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------
class TestParser:
    """Tests for argument parsing."""

    def test_estimate_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "estimate",
                "--model",
                "local_level",
                "--data",
                "foo.csv",
            ]
        )
        assert args.command == "estimate"
        assert args.model == "local_level"
        assert args.data == "foo.csv"
        assert args.output == "results.json"

    def test_info_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["info", "--model", "bsm", "--seasonal-period", "12"])
        assert args.command == "info"
        assert args.model == "bsm"
        assert args.seasonal_period == 12

    def test_forecast_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "forecast",
                "--model",
                "local_level",
                "--data",
                "foo.csv",
                "--steps",
                "10",
                "--output",
                "fc.csv",
            ]
        )
        assert args.command == "forecast"
        assert args.steps == 10
        assert args.output == "fc.csv"

    def test_version(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_no_command_prints_help(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Estimate command tests
# ---------------------------------------------------------------------------
class TestEstimateCommand:
    """Tests for the estimate command."""

    def test_estimate_local_level(self, tmp_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        main(
            [
                "estimate",
                "--model",
                "local_level",
                "--data",
                str(tmp_csv),
                "--output",
                str(output),
            ]
        )
        assert output.exists()
        with open(output) as f:
            result = json.load(f)
        assert result["model"] == "local_level"
        assert "loglike" in result
        assert "aic" in result
        assert "bic" in result
        assert "params" in result
        assert result["n_obs"] == 100

    def test_estimate_missing_data_file(self) -> None:
        with pytest.raises(SystemExit):
            main(["estimate", "--model", "local_level", "--data", "nonexistent.csv"])

    def test_estimate_unknown_model(self) -> None:
        with pytest.raises(SystemExit):
            main(["estimate", "--model", "unknown_model", "--data", "foo.csv"])


# ---------------------------------------------------------------------------
# Info command tests
# ---------------------------------------------------------------------------
class TestInfoCommand:
    """Tests for the info command."""

    def test_info_local_level(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info", "--model", "local_level"])
        captured = capsys.readouterr()
        assert "k_states" in captured.out
        assert "k_obs" in captured.out
        assert "Parameters" in captured.out

    def test_info_bsm(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info", "--model", "bsm", "--seasonal-period", "12"])
        captured = capsys.readouterr()
        assert "k_states" in captured.out


# ---------------------------------------------------------------------------
# Forecast command tests
# ---------------------------------------------------------------------------
class TestForecastCommand:
    """Tests for the forecast command."""

    def test_forecast_local_level(self, tmp_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "forecast.csv"
        main(
            [
                "forecast",
                "--model",
                "local_level",
                "--data",
                str(tmp_csv),
                "--steps",
                "10",
                "--output",
                str(output),
            ]
        )
        assert output.exists()
        fc_df = pd.read_csv(output)
        assert len(fc_df) == 10
        assert "mean" in fc_df.columns
        assert "lower_95" in fc_df.columns
        assert "upper_95" in fc_df.columns
        assert fc_df["step"].tolist() == list(range(1, 11))

    def test_forecast_missing_data_file(self) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "forecast",
                    "--model",
                    "local_level",
                    "--data",
                    "nonexistent.csv",
                    "--steps",
                    "5",
                ]
            )


# ---------------------------------------------------------------------------
# Integration test via subprocess
# ---------------------------------------------------------------------------
class TestCLISubprocess:
    """Test CLI as subprocess (entry point)."""

    def test_help_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "kalmanbox.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "kalmanbox" in result.stdout.lower()

    def test_version_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "kalmanbox.cli.main", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
