"""Tests for visualization export module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kalmanbox.visualization.export import (
    _fig_to_embedded_html,
    close_figure,
    export_figure,
    figure_to_bytes,
)
from kalmanbox.visualization.themes import reset_theme


@pytest.fixture
def sample_fig() -> plt.Figure:
    """Create a simple matplotlib figure for testing."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.set_title("Test Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return fig


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset theme for each test."""
    reset_theme()


class TestExportPNG:
    """Tests for PNG export."""

    def test_export_png_creates_file(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "test.png"
        result = export_figure(sample_fig, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_png_custom_dpi(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "test_hires.png"
        export_figure(sample_fig, out, dpi=300)
        assert out.exists()
        assert out.stat().st_size > 0


class TestExportSVG:
    """Tests for SVG export."""

    def test_export_svg_creates_file(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "test.svg"
        result = export_figure(sample_fig, out)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content


class TestExportPDF:
    """Tests for PDF export."""

    def test_export_pdf_creates_file(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "test.pdf"
        result = export_figure(sample_fig, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0


class TestExportHTML:
    """Tests for HTML export."""

    def test_export_html_creates_file(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "test.html"
        result = export_figure(sample_fig, out, fmt="html")
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<html" in content.lower()

    def test_fig_to_embedded_html_contains_img(self, sample_fig: plt.Figure) -> None:
        html = _fig_to_embedded_html(sample_fig)
        assert "data:image/png;base64," in html
        assert "<img" in html


class TestExportEdgeCases:
    """Tests for edge cases."""

    def test_export_unknown_format_raises(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "test.xyz"
        with pytest.raises(ValueError, match="Unsupported format"):
            export_figure(sample_fig, out)

    def test_export_no_extension_no_fmt_raises(
        self, sample_fig: plt.Figure, tmp_path: Path
    ) -> None:
        out = tmp_path / "testfile"
        with pytest.raises(ValueError, match="Cannot determine format"):
            export_figure(sample_fig, out)

    def test_export_creates_parent_dirs(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "subdir" / "deep" / "test.png"
        export_figure(sample_fig, out)
        assert out.exists()

    def test_fmt_overrides_extension(self, sample_fig: plt.Figure, tmp_path: Path) -> None:
        out = tmp_path / "test.png"
        export_figure(sample_fig, out, fmt="svg")
        assert out.exists()
        # File content is SVG even though extension is .png
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content


class TestFigureToBytes:
    """Tests for figure_to_bytes."""

    def test_returns_bytes(self, sample_fig: plt.Figure) -> None:
        data = figure_to_bytes(sample_fig)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_png_bytes_header(self, sample_fig: plt.Figure) -> None:
        data = figure_to_bytes(sample_fig, fmt="png")
        # PNG files start with 0x89504E47
        assert data[:4] == b"\x89PNG"

    def test_svg_bytes(self, sample_fig: plt.Figure) -> None:
        data = figure_to_bytes(sample_fig, fmt="svg")
        text = data.decode("utf-8")
        assert "<svg" in text


class TestCloseFigure:
    """Tests for close_figure."""

    def test_close_does_not_raise(self, sample_fig: plt.Figure) -> None:
        close_figure(sample_fig)
        # Should not raise even if called again
        close_figure(sample_fig)
