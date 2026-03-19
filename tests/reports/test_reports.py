"""Tests for the report generation system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from kalmanbox.reports.css_manager import CSSManager
from kalmanbox.reports.exporters.html import HTMLExporter
from kalmanbox.reports.exporters.latex import LaTeXExporter
from kalmanbox.reports.exporters.markdown import MarkdownExporter
from kalmanbox.reports.report_manager import ReportManager
from kalmanbox.reports.transformers.dfm import DFMTransformer
from kalmanbox.reports.transformers.ssm import SSMTransformer
from kalmanbox.reports.transformers.tvp import TVPTransformer
from kalmanbox.reports.transformers.ucm import UCMTransformer

# --- Mock Results ---


@dataclass
class MockSSMResults:
    """Mock SSM results for testing."""

    model_name: str = "Local Level"
    nobs: int = 100
    n_states: int = 1
    n_params: int = 2
    optimizer: str = "L-BFGS-B"
    converged: bool = True
    param_names: list[str] = field(default_factory=lambda: ["sigma2_obs", "sigma2_level"])
    params: NDArray[np.float64] = field(default_factory=lambda: np.array([15099.0, 1469.0]))
    bse: NDArray[np.float64] = field(default_factory=lambda: np.array([3500.0, 800.0]))
    pvalues: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0001, 0.067]))
    llf: float = -632.54
    aic: float = 1269.08
    bic: float = 1274.29
    hqic: float = 1271.18
    trend: NDArray[np.float64] = field(default_factory=lambda: np.linspace(1100, 850, 100))
    irregular: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 1, 100)
    )
    standardized_residuals: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 1, 100)
    )


@dataclass
class MockDFMResults(MockSSMResults):
    """Mock DFM results."""

    model_name: str = "Dynamic Factor Model"
    factors: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 1, (100, 2))
    )
    loadings: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).uniform(-1, 1, (5, 2))
    )
    idiosyncratic_var: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.2, 0.3, 0.15, 0.25, 0.1])
    )


@dataclass
class MockUCMResults(MockSSMResults):
    """Mock UCM results."""

    model_name: str = "Unobserved Components"
    slope: NDArray[np.float64] = field(
        default_factory=lambda: np.random.default_rng(42).normal(0, 0.5, 100)
    )
    seasonal: NDArray[np.float64] = field(
        default_factory=lambda: np.sin(np.linspace(0, 8 * np.pi, 100)) * 10
    )


@dataclass
class MockTVPResults(MockSSMResults):
    """Mock TVP results."""

    model_name: str = "TVP Regression"
    smoothed_coefficients: NDArray[np.float64] = field(
        default_factory=lambda: np.column_stack(
            [
                np.linspace(0.5, 0.8, 100),
                np.linspace(-0.2, 0.1, 100),
            ]
        )
    )
    smoothed_state_cov: NDArray[np.float64] = field(
        default_factory=lambda: np.stack([np.diag([0.01, 0.02])] * 100)
    )
    coef_names: list[str] = field(default_factory=lambda: ["beta_GDP", "beta_CPI"])


# --- Tests ---


class TestSSMTransformer:
    """Tests for SSMTransformer."""

    def test_transform_returns_dict(self) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        assert isinstance(ctx, dict)

    def test_has_all_sections(self) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        assert "model_info" in ctx
        assert "parameters" in ctx
        assert "info_criteria" in ctx
        assert "diagnostics" in ctx
        assert "components" in ctx
        assert "metadata" in ctx

    def test_parameters_extracted(self) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        params = ctx["parameters"]
        assert len(params) == 2
        assert params[0]["name"] == "sigma2_obs"
        assert abs(params[0]["value"] - 15099.0) < 0.01

    def test_info_criteria_extracted(self) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        ic = ctx["info_criteria"]
        assert ic["loglike"] is not None
        assert abs(ic["loglike"] - (-632.54)) < 0.01
        assert ic["aic"] is not None
        assert ic["bic"] is not None

    def test_metadata_has_version(self) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        assert "kalmanbox_version" in ctx["metadata"]
        assert "generated_at" in ctx["metadata"]


class TestDFMTransformer:
    """Tests for DFMTransformer."""

    def test_has_factors(self) -> None:
        t = DFMTransformer()
        ctx = t.transform(MockDFMResults())
        assert ctx["factors"] is not None

    def test_has_loadings(self) -> None:
        t = DFMTransformer()
        ctx = t.transform(MockDFMResults())
        assert ctx["loadings"] is not None

    def test_has_variance_explained(self) -> None:
        t = DFMTransformer()
        ctx = t.transform(MockDFMResults())
        assert ctx["variance_explained"] is not None

    def test_report_type_dfm(self) -> None:
        t = DFMTransformer()
        ctx = t.transform(MockDFMResults())
        assert ctx["metadata"]["report_type"] == "DFM"


class TestUCMTransformer:
    """Tests for UCMTransformer."""

    def test_has_component_details(self) -> None:
        t = UCMTransformer()
        ctx = t.transform(MockUCMResults())
        assert len(ctx["component_details"]) >= 2

    def test_has_contributions(self) -> None:
        t = UCMTransformer()
        ctx = t.transform(MockUCMResults())
        assert ctx["contributions"] is not None


class TestTVPTransformer:
    """Tests for TVPTransformer."""

    def test_has_coefficients(self) -> None:
        t = TVPTransformer()
        ctx = t.transform(MockTVPResults())
        assert ctx["coefficients"] is not None
        assert len(ctx["coefficients"]) == 2

    def test_has_significance(self) -> None:
        t = TVPTransformer()
        ctx = t.transform(MockTVPResults())
        assert ctx["coefficient_significance"] is not None

    def test_coefficient_names(self) -> None:
        t = TVPTransformer()
        ctx = t.transform(MockTVPResults())
        names = [c["name"] for c in ctx["coefficients"]]
        assert "beta_GDP" in names
        assert "beta_CPI" in names


class TestCSSManager:
    """Tests for CSSManager."""

    def test_get_css_returns_string(self) -> None:
        cm = CSSManager()
        css = cm.get_css()
        assert isinstance(css, str)
        assert len(css) > 100

    def test_css_contains_layers(self) -> None:
        cm = CSSManager()
        css = cm.get_css()
        assert "Base Layer" in css
        assert "Component Layer" in css
        assert "Theme Layer" in css

    def test_css_with_different_themes(self) -> None:
        cm = CSSManager()
        css_prof = cm.get_css("professional")
        css_acad = cm.get_css("academic")
        assert css_prof != css_acad


class TestHTMLExporter:
    """Tests for HTMLExporter."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        exp = HTMLExporter()
        out = tmp_path / "report.html"
        result = exp.export("<h1>Test</h1>", out, css="body { color: red; }")
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<h1>Test</h1>" in content
        assert "color: red" in content

    def test_export_string(self) -> None:
        exp = HTMLExporter()
        html = exp.export_string("<h1>Test</h1>")
        assert "<!DOCTYPE html>" in html
        assert "<h1>Test</h1>" in html


class TestLaTeXExporter:
    """Tests for LaTeXExporter."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        exp = LaTeXExporter()
        out = tmp_path / "report.tex"
        result = exp.export(ctx, out)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "\\documentclass" in content
        assert "\\begin{document}" in content

    def test_export_string(self) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        exp = LaTeXExporter()
        latex = exp.export_string(ctx)
        assert "\\begin{tabular}" in latex
        assert "booktabs" in latex


class TestMarkdownExporter:
    """Tests for MarkdownExporter."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        exp = MarkdownExporter()
        out = tmp_path / "report.md"
        result = exp.export(ctx, out)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "# " in content
        assert "|" in content

    def test_export_string(self) -> None:
        t = SSMTransformer()
        ctx = t.transform(MockSSMResults())
        exp = MarkdownExporter()
        md = exp.export_string(ctx)
        assert "Parameter" in md
        assert "sigma2_obs" in md


class TestReportManager:
    """Tests for ReportManager."""

    def test_generate_html(self, tmp_path: Path) -> None:
        rm = ReportManager()
        out = tmp_path / "report.html"
        result = rm.generate(MockSSMResults(), fmt="html", output=out)
        assert Path(result).exists()
        content = Path(result).read_text(encoding="utf-8")
        assert "<html" in content.lower()

    def test_generate_latex(self, tmp_path: Path) -> None:
        rm = ReportManager()
        out = tmp_path / "report.tex"
        result = rm.generate(MockSSMResults(), fmt="latex", output=out)
        assert Path(result).exists()

    def test_generate_markdown(self, tmp_path: Path) -> None:
        rm = ReportManager()
        out = tmp_path / "report.md"
        result = rm.generate(MockSSMResults(), fmt="markdown", output=out)
        assert Path(result).exists()

    def test_generate_string(self) -> None:
        rm = ReportManager()
        html = rm.generate(MockSSMResults(), fmt="html")
        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_generate_dfm(self, tmp_path: Path) -> None:
        rm = ReportManager()
        out = tmp_path / "dfm_report.html"
        result = rm.generate(MockDFMResults(), report_type="dfm", fmt="html", output=out)
        assert Path(result).exists()

    def test_generate_ucm(self, tmp_path: Path) -> None:
        rm = ReportManager()
        out = tmp_path / "ucm_report.html"
        result = rm.generate(MockUCMResults(), report_type="ucm", fmt="html", output=out)
        assert Path(result).exists()

    def test_generate_tvp(self, tmp_path: Path) -> None:
        rm = ReportManager()
        out = tmp_path / "tvp_report.html"
        result = rm.generate(MockTVPResults(), report_type="tvp", fmt="html", output=out)
        assert Path(result).exists()

    def test_unknown_type_raises(self) -> None:
        rm = ReportManager()
        with pytest.raises(ValueError, match="Unknown report_type"):
            rm.generate(MockSSMResults(), report_type="unknown")

    def test_unknown_fmt_raises(self) -> None:
        rm = ReportManager()
        with pytest.raises(ValueError, match="Unknown format"):
            rm.generate(MockSSMResults(), fmt="docx")

    def test_html_contains_params_table(self) -> None:
        rm = ReportManager()
        html = rm.generate(MockSSMResults(), fmt="html")
        assert isinstance(html, str)
        assert "sigma2_obs" in html
        assert "15099" in html

    def test_html_contains_diagnostics_section(self) -> None:
        rm = ReportManager()
        html = rm.generate(MockSSMResults(), fmt="html")
        assert isinstance(html, str)
        assert "Local Level" in html
