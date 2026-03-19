"""Tests for KalmanExperiment and result classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalmanbox.experiment.comparison import ComparisonResult, ValidationResult
from kalmanbox.experiment.experiment import KalmanExperiment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_y() -> np.ndarray:
    """Generate a simple time series for testing."""
    rng = np.random.default_rng(42)
    n = 100
    level = np.cumsum(rng.normal(0, 1, n))
    y = level + rng.normal(0, 0.5, n)
    return y.astype(np.float64)


@pytest.fixture
def comparison_result() -> ComparisonResult:
    """Create a sample ComparisonResult."""
    return ComparisonResult(
        model_names=["ModelA", "ModelB", "ModelC"],
        metrics={
            "ModelA": {"aic": 200.0, "bic": 210.0, "loglike": -95.0},
            "ModelB": {"aic": 190.0, "bic": 205.0, "loglike": -90.0},
            "ModelC": {"aic": 195.0, "bic": 200.0, "loglike": -92.0},
        },
        criteria=["aic", "bic", "loglike"],
    )


@pytest.fixture
def validation_result() -> ValidationResult:
    """Create a sample ValidationResult."""
    rng = np.random.default_rng(42)
    n = 20
    y_test = 100 + rng.normal(0, 10, n)
    y_forecast = y_test + rng.normal(0, 5, n)
    return ValidationResult(
        model_name="TestModel",
        y_test=y_test.astype(np.float64),
        y_forecast=y_forecast.astype(np.float64),
        y_lower=(y_forecast - 20).astype(np.float64),
        y_upper=(y_forecast + 20).astype(np.float64),
        train_size=80,
        test_size=20,
        horizon=20,
    )


# ---------------------------------------------------------------------------
# ComparisonResult tests
# ---------------------------------------------------------------------------
class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_ranking_default(self, comparison_result: ComparisonResult) -> None:
        ranked = comparison_result.ranking()
        assert ranked[0][0] == "ModelB"  # lowest AIC
        assert ranked[0][1] == 190.0

    def test_ranking_by_bic(self, comparison_result: ComparisonResult) -> None:
        ranked = comparison_result.ranking("bic")
        assert ranked[0][0] == "ModelC"  # lowest BIC = 200

    def test_ranking_by_loglike(self, comparison_result: ComparisonResult) -> None:
        ranked = comparison_result.ranking("loglike")
        assert ranked[0][0] == "ModelB"  # highest loglike = -90

    def test_best_model(self, comparison_result: ComparisonResult) -> None:
        assert comparison_result.best_model() == "ModelB"
        assert comparison_result.best_model("bic") == "ModelC"

    def test_to_dataframe(self, comparison_result: ComparisonResult) -> None:
        df = comparison_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["aic", "bic", "loglike"]
        assert len(df) == 3
        assert df.loc["ModelA", "aic"] == 200.0

    def test_repr(self, comparison_result: ComparisonResult) -> None:
        r = repr(comparison_result)
        assert "ComparisonResult" in r
        assert "ModelA" in r


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------
class TestValidationResult:
    """Tests for ValidationResult."""

    def test_rmse(self, validation_result: ValidationResult) -> None:
        rmse = validation_result.rmse()
        assert isinstance(rmse, float)
        assert rmse > 0

    def test_mae(self, validation_result: ValidationResult) -> None:
        mae = validation_result.mae()
        assert isinstance(mae, float)
        assert mae > 0

    def test_mape(self, validation_result: ValidationResult) -> None:
        mape = validation_result.mape()
        assert isinstance(mape, float)
        assert mape > 0

    def test_coverage(self, validation_result: ValidationResult) -> None:
        cov = validation_result.coverage()
        assert 0 <= cov <= 1

    def test_to_dataframe(self, validation_result: ValidationResult) -> None:
        df = validation_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "actual" in df.columns
        assert "forecast" in df.columns
        assert len(df) == 20

    def test_summary(self, validation_result: ValidationResult) -> None:
        s = validation_result.summary()
        assert "rmse" in s
        assert "mae" in s
        assert "mape" in s
        assert "coverage" in s

    def test_repr(self, validation_result: ValidationResult) -> None:
        r = repr(validation_result)
        assert "ValidationResult" in r
        assert "TestModel" in r

    def test_mape_with_zeros(self) -> None:
        vr = ValidationResult(
            model_name="test",
            y_test=np.array([0.0, 0.0, 0.0]),
            y_forecast=np.array([1.0, 1.0, 1.0]),
            y_lower=np.array([-1.0, -1.0, -1.0]),
            y_upper=np.array([3.0, 3.0, 3.0]),
            train_size=10,
            test_size=3,
            horizon=3,
        )
        assert vr.mape() == float("inf")


# ---------------------------------------------------------------------------
# KalmanExperiment tests
# ---------------------------------------------------------------------------
class TestKalmanExperiment:
    """Tests for KalmanExperiment."""

    def test_init_with_array(self, sample_y: np.ndarray) -> None:
        exp = KalmanExperiment(sample_y)
        assert len(exp.y) == 100
        assert exp.y.dtype == np.float64

    def test_init_with_series(self, sample_y: np.ndarray) -> None:
        s = pd.Series(sample_y)
        exp = KalmanExperiment(s)
        assert len(exp.y) == 100

    def test_fit_all_models(self, sample_y: np.ndarray) -> None:
        exp = KalmanExperiment(sample_y)
        exp.fit_all_models([("LocalLevel", {})])
        assert "LocalLevel" in exp.results
        assert exp.results["LocalLevel"].loglike is not None

    def test_compare_models(self, sample_y: np.ndarray) -> None:
        exp = KalmanExperiment(sample_y)
        exp.fit_all_models([("LocalLevel", {})])
        comp = exp.compare_models()
        assert isinstance(comp, ComparisonResult)
        assert comp.best_model() == "LocalLevel"

    def test_compare_before_fit_raises(self, sample_y: np.ndarray) -> None:
        exp = KalmanExperiment(sample_y)
        with pytest.raises(RuntimeError, match="No models fitted"):
            exp.compare_models()

    def test_validate_model(self, sample_y: np.ndarray) -> None:
        exp = KalmanExperiment(sample_y)
        val = exp.validate_model("LocalLevel", test_size=20, horizon=10)
        assert isinstance(val, ValidationResult)
        assert val.rmse() > 0
        assert val.mae() > 0
        assert len(val.y_test) == 10
        assert len(val.y_forecast) == 10

    def test_validate_model_test_size_too_large(self, sample_y: np.ndarray) -> None:
        exp = KalmanExperiment(sample_y)
        with pytest.raises(ValueError, match="test_size"):
            exp.validate_model("LocalLevel", test_size=200)

    def test_save_master_report(self, sample_y: np.ndarray, tmp_path: Path) -> None:
        exp = KalmanExperiment(sample_y)
        exp.fit_all_models([("LocalLevel", {})])
        output = tmp_path / "report.html"
        exp.save_master_report(output)
        assert output.exists()
        content = output.read_text()
        assert "<html>" in content
        assert "LocalLevel" in content
        assert "AIC" in content.upper() or "aic" in content

    def test_save_report_before_fit_raises(self, sample_y: np.ndarray, tmp_path: Path) -> None:
        exp = KalmanExperiment(sample_y)
        with pytest.raises(RuntimeError, match="No models fitted"):
            exp.save_master_report(tmp_path / "report.html")

    def test_unknown_model_raises(self, sample_y: np.ndarray) -> None:
        exp = KalmanExperiment(sample_y)
        with pytest.raises(ValueError, match="Unknown model"):
            exp.fit_all_models([("NonExistentModel", {})])
