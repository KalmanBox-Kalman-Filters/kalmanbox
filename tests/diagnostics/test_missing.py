"""Tests for missing data handler."""

import numpy as np
import pytest

from kalmanbox.diagnostics.missing import MissingDataHandler, MissingDataReport
from kalmanbox.models.local_level import LocalLevel


class TestMissingDataHandler:
    """Tests for MissingDataHandler."""

    def test_prepare_data_no_missing(self, nile_volume: np.ndarray) -> None:
        """prepare_data with no NaN should return empty missing_indices."""
        handler = MissingDataHandler(strategy="skip")
        prepared, missing = handler.prepare_data(nile_volume)
        assert len(missing) == 0
        np.testing.assert_array_equal(prepared, nile_volume)

    def test_prepare_data_with_missing(self, nile_volume: np.ndarray) -> None:
        """prepare_data should identify NaN indices."""
        handler = MissingDataHandler(strategy="skip")
        y = nile_volume.copy()
        y[10] = np.nan
        y[50] = np.nan
        y[80] = np.nan

        prepared, missing = handler.prepare_data(y)
        assert len(missing) == 3
        np.testing.assert_array_equal(missing, [10, 50, 80])

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            MissingDataHandler(strategy="unknown")  # type: ignore[arg-type]

    def test_skip_strategy_preserves_nan(self, nile_volume: np.ndarray) -> None:
        """Skip strategy should preserve NaN values."""
        handler = MissingDataHandler(strategy="skip")
        y = nile_volume.copy()
        y[10] = np.nan
        prepared, _ = handler.prepare_data(y)
        assert np.isnan(prepared[10])

    def test_interpolate_missing(self, nile_volume: np.ndarray) -> None:
        """Interpolation should produce values for missing observations."""
        y = nile_volume.copy()
        y[10] = np.nan
        y[50] = np.nan

        model = LocalLevel(y)
        results = model.fit()

        handler = MissingDataHandler(strategy="interpolate")
        report = handler.interpolate_missing(y, results)

        assert isinstance(report, MissingDataReport)
        assert report.n_missing == 2
        assert report.interpolated_values is not None
        assert len(report.interpolated_values) == 2
        assert report.interpolated_variances is not None
        assert np.all(report.interpolated_variances > 0)

    def test_interpolated_values_reasonable(self, nile_volume: np.ndarray) -> None:
        """Interpolated values should be within data range."""
        y = nile_volume.copy()
        y[50] = np.nan

        model = LocalLevel(y)
        results = model.fit()

        handler = MissingDataHandler(strategy="interpolate")
        report = handler.interpolate_missing(y, results)

        y_min = np.nanmin(nile_volume)
        y_max = np.nanmax(nile_volume)
        margin = (y_max - y_min) * 0.5

        for val in report.interpolated_values:
            v = float(val) if np.ndim(val) == 0 else float(val)
            assert y_min - margin < v < y_max + margin

    def test_fill_missing(self, nile_volume: np.ndarray) -> None:
        """fill_missing should replace NaN with interpolated values."""
        y = nile_volume.copy()
        y[10] = np.nan
        y[50] = np.nan

        model = LocalLevel(y)
        results = model.fit()

        handler = MissingDataHandler(strategy="interpolate")
        y_filled = handler.fill_missing(y, results)

        assert not np.any(np.isnan(y_filled))
        assert y_filled.shape == y.shape

    def test_no_missing_report(self, nile_volume: np.ndarray) -> None:
        """Report for data without missing should have n_missing=0."""
        model = LocalLevel(nile_volume)
        results = model.fit()

        handler = MissingDataHandler(strategy="interpolate")
        report = handler.interpolate_missing(nile_volume, results)

        assert report.n_missing == 0
        assert report.missing_rate == 0.0

    def test_missing_data_report_repr(self) -> None:
        """MissingDataReport repr should be readable."""
        report = MissingDataReport(
            n_total=100,
            n_missing=5,
            missing_indices=np.array([10, 20, 30, 40, 50]),
            missing_rate=0.05,
        )
        text = repr(report)
        assert "Missing Data Report" in text
        assert "5" in text
