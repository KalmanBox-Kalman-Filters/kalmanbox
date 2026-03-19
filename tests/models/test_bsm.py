"""Tests for BasicStructuralModel."""

import numpy as np
import pytest

from kalmanbox.datasets import load_dataset
from kalmanbox.models.bsm import BasicStructuralModel


@pytest.fixture
def airline_data() -> np.ndarray:
    """Airline passengers as log-transformed numpy array."""
    df = load_dataset("airline")
    return np.log(df["passengers"].to_numpy(dtype=np.float64))


class TestBasicStructuralModel:
    """Tests for BSM."""

    def test_airline_fit(self, airline_data: np.ndarray) -> None:
        """Fit BSM on airline data (s=12) should converge."""
        model = BasicStructuralModel(airline_data, seasonal_period=12)
        results = model.fit()
        assert results.optimizer_converged
        assert len(results.params) == 4

    def test_airline_seasonal_recovery(self, airline_data: np.ndarray) -> None:
        """Smoothed states should show seasonal pattern."""
        model = BasicStructuralModel(airline_data, seasonal_period=12)
        results = model.fit()
        assert results.smoothed_state is not None
        # Seasonal component is state index 2
        seasonal = results.smoothed_state[:, 2]
        # Should show periodic pattern: check autocorrelation at lag 12
        n = len(seasonal)
        mean = np.mean(seasonal)
        var = np.var(seasonal)
        if var > 0:
            autocov = np.mean((seasonal[: n - 12] - mean) * (seasonal[12:] - mean))
            autocorr_12 = autocov / var
            # Strong positive autocorrelation at lag 12
            assert autocorr_12 > 0.5

    def test_airline_forecast_12(self, airline_data: np.ndarray) -> None:
        """12-month forecast should capture seasonality."""
        model = BasicStructuralModel(airline_data, seasonal_period=12)
        results = model.fit()
        fc = results.forecast(steps=12)
        means = fc["mean"][:, 0]
        # Forecast should not be flat (seasonality present)
        assert np.std(means) > 0.01

    def test_dummy_seasonal_k_states(self) -> None:
        """k_states should be 2 + (s-1) for dummy seasonal."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        model = BasicStructuralModel(y, seasonal_period=4, seasonal="dummy")
        assert model._k_states == 5  # 2 + 3

        model12 = BasicStructuralModel(y, seasonal_period=12, seasonal="dummy")
        assert model12._k_states == 13  # 2 + 11

    def test_trigonometric_seasonal(self) -> None:
        """Trigonometric seasonal should also work."""
        rng = np.random.default_rng(42)
        n = 200
        # Simple seasonal data
        seasonal = np.tile([1, 0, -1, 0], n // 4)
        y = seasonal + rng.normal(0, 0.3, n)

        model = BasicStructuralModel(y, seasonal_period=4, seasonal="trigonometric")
        results = model.fit()
        assert results.optimizer_converged

    def test_invalid_seasonal_type(self) -> None:
        """Invalid seasonal type should raise ValueError."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        with pytest.raises(ValueError, match="seasonal must be"):
            BasicStructuralModel(y, seasonal_period=4, seasonal="invalid")

    def test_param_names(self, airline_data: np.ndarray) -> None:
        """Parameter names should be correct."""
        model = BasicStructuralModel(airline_data, seasonal_period=12)
        assert model.param_names == [
            "sigma2_obs",
            "sigma2_level",
            "sigma2_trend",
            "sigma2_seasonal",
        ]

    def test_summary(self, airline_data: np.ndarray) -> None:
        """summary() should include all parameters."""
        model = BasicStructuralModel(airline_data, seasonal_period=12)
        results = model.fit()
        s = results.summary()
        assert "sigma2_seasonal" in s

    def test_quarterly_gas(self) -> None:
        """BSM with s=4 on UK gas data."""
        df = load_dataset("uk_gas")
        y = df["gas"].to_numpy(dtype=np.float64)
        model = BasicStructuralModel(y, seasonal_period=4)
        results = model.fit()
        assert results.optimizer_converged
