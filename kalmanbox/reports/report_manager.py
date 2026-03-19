"""Report manager orchestrator for kalmanbox.

Coordinates transformers, templates, CSS, and exporters to generate
complete reports from model results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kalmanbox.reports.css_manager import CSSManager
from kalmanbox.reports.exporters.html import HTMLExporter
from kalmanbox.reports.exporters.latex import LaTeXExporter
from kalmanbox.reports.exporters.markdown import MarkdownExporter
from kalmanbox.reports.template_manager import TemplateManager
from kalmanbox.reports.transformers.dfm import DFMTransformer
from kalmanbox.reports.transformers.ssm import SSMTransformer
from kalmanbox.reports.transformers.tvp import TVPTransformer
from kalmanbox.reports.transformers.ucm import UCMTransformer

# Mapping of report types to transformers
_TRANSFORMERS: dict[str, type] = {
    "ssm": SSMTransformer,
    "dfm": DFMTransformer,
    "ucm": UCMTransformer,
    "tvp": TVPTransformer,
}

# Mapping of report types to HTML template names
_TEMPLATES: dict[str, str] = {
    "ssm": "ssm_report.html",
    "dfm": "ssm_report.html",
    "ucm": "ssm_report.html",
    "tvp": "ssm_report.html",
}


class ReportManager:
    """Orchestrates report generation.

    Usage
    -----
    >>> from kalmanbox.reports import ReportManager
    >>> rm = ReportManager()
    >>> rm.generate(results, report_type='ssm', fmt='html', output='report.html')
    """

    def __init__(self, template_dir: str | Path | None = None) -> None:
        """Initialize ReportManager.

        Parameters
        ----------
        template_dir : str, Path, or None
            Custom template directory. If None, uses built-in templates.
        """
        self._css_manager = CSSManager()
        self._html_exporter = HTMLExporter()
        self._latex_exporter = LaTeXExporter()
        self._markdown_exporter = MarkdownExporter()

        try:
            self._template_manager: TemplateManager | None = TemplateManager(template_dir)
            self._has_jinja2 = True
        except ImportError:
            self._template_manager = None
            self._has_jinja2 = False

    def generate(
        self,
        results: Any,
        report_type: str = "ssm",
        fmt: str = "html",
        output: str | Path | None = None,
        theme: str | None = None,
    ) -> str | Path:
        """Generate a report from model results.

        Parameters
        ----------
        results : model results object
            Fitted model results.
        report_type : str
            Type of report: 'ssm', 'dfm', 'ucm', or 'tvp'.
        fmt : str
            Output format: 'html', 'latex', or 'markdown'.
        output : str, Path, or None
            Output file path. If None, returns the rendered string.
        theme : str or None
            Theme name for styling.

        Returns
        -------
        str or Path
            If output is None, returns the rendered string.
            Otherwise, returns the Path to the saved file.

        Raises
        ------
        ValueError
            If report_type or fmt is not recognized.
        """
        if report_type not in _TRANSFORMERS:
            available = ", ".join(sorted(_TRANSFORMERS.keys()))
            msg = f"Unknown report_type '{report_type}'. Available: {available}"
            raise ValueError(msg)

        if fmt not in ("html", "latex", "markdown"):
            msg = f"Unknown format '{fmt}'. Available: html, latex, markdown"
            raise ValueError(msg)

        # Step 1: Transform results to context
        transformer = _TRANSFORMERS[report_type]()
        context = transformer.transform(results)

        # Step 2: Generate output
        if fmt == "html":
            return self._generate_html(context, report_type, theme, output)
        elif fmt == "latex":
            return self._generate_latex(context, output)
        else:
            return self._generate_markdown(context, output)

    def _generate_html(
        self,
        context: dict[str, Any],
        report_type: str,
        theme: str | None,
        output: str | Path | None,
    ) -> str | Path:
        """Generate HTML report."""
        css = self._css_manager.get_css(theme)

        # Try Jinja2 template rendering
        if self._has_jinja2 and self._template_manager is not None:
            template_name = _TEMPLATES.get(report_type, "ssm_report.html")
            try:
                rendered_body = self._template_manager.render(template_name, context)
            except Exception:
                # Fallback to simple HTML generation
                rendered_body = self._simple_html_body(context)
        else:
            rendered_body = self._simple_html_body(context)

        if output is None:
            return self._html_exporter.export_string(rendered_body, css)
        return self._html_exporter.export(rendered_body, output, css)

    def _generate_latex(
        self,
        context: dict[str, Any],
        output: str | Path | None,
    ) -> str | Path:
        """Generate LaTeX report."""
        if output is None:
            return self._latex_exporter.export_string(context)
        return self._latex_exporter.export(context, output)

    def _generate_markdown(
        self,
        context: dict[str, Any],
        output: str | Path | None,
    ) -> str | Path:
        """Generate Markdown report."""
        if output is None:
            return self._markdown_exporter.export_string(context)
        return self._markdown_exporter.export(context, output)

    def _simple_html_body(self, context: dict[str, Any]) -> str:
        """Generate simple HTML body without Jinja2 templates."""
        parts: list[str] = []

        model_info = context.get("model_info", {})
        model_name = model_info.get("model_name", "Report")
        parts.append(f"<h1>{model_name}</h1>")

        # Model info
        parts.append('<div class="section">')
        parts.append('<h2 class="section-header">Model Summary</h2>')
        parts.append('<div class="section-content">')
        parts.append('<div class="info-grid">')
        for key in ["n_obs", "n_states", "n_params", "optimizer"]:
            val = model_info.get(key)
            if val is not None:
                label = key.replace("_", " ").title().replace("N ", "# ")
                parts.append(
                    f'<div class="info-card"><div class="label">{label}</div>'
                    f'<div class="value">{val}</div></div>'
                )
        parts.append("</div></div></div>")

        # Parameters
        parameters = context.get("parameters", [])
        if parameters:
            parts.append('<div class="section">')
            parts.append('<h2 class="section-header">Parameter Estimates</h2>')
            parts.append('<div class="section-content">')
            parts.append("<table>")
            parts.append(
                "<thead><tr><th>Parameter</th><th>Estimate</th>"
                "<th>Std. Error</th><th>t-stat</th><th>p-value</th>"
                "<th>Sig.</th></tr></thead>"
            )
            parts.append("<tbody>")
            for p in parameters:
                sig = ""
                if "p_value" in p:
                    pv = p["p_value"]
                    if pv < 0.001:
                        sig = "***"
                    elif pv < 0.01:
                        sig = "**"
                    elif pv < 0.05:
                        sig = "*"
                se = f"{p['se']:.4f}" if "se" in p else "--"
                ts = f"{p['t_stat']:.4f}" if "t_stat" in p else "--"
                pval = f"{p['p_value']:.4f}" if "p_value" in p else "--"
                parts.append(
                    f"<tr><td>{p['name']}</td>"
                    f'<td class="numeric">{p["value"]:.4f}</td>'
                    f'<td class="numeric">{se}</td>'
                    f'<td class="numeric">{ts}</td>'
                    f'<td class="numeric">{pval}</td>'
                    f'<td class="{("significant" if sig else "")}">'
                    f"{sig}</td></tr>"
                )
            parts.append("</tbody></table>")
            parts.append("</div></div>")

        # Info criteria
        ic = context.get("info_criteria", {})
        if any(v is not None for v in ic.values()):
            parts.append('<div class="section">')
            parts.append('<h2 class="section-header">Information Criteria</h2>')
            parts.append('<div class="section-content">')
            parts.append('<div class="info-grid">')
            for name, val in ic.items():
                display = name.upper() if name != "loglike" else "Log-Likelihood"
                val_str = f"{val:.4f}" if val is not None else "--"
                parts.append(
                    f'<div class="info-card">'
                    f'<div class="label">{display}</div>'
                    f'<div class="value">{val_str}</div></div>'
                )
            parts.append("</div></div></div>")

        # Footer
        metadata = context.get("metadata", {})
        version = metadata.get("kalmanbox_version", "unknown")
        generated = metadata.get("generated_at", "")
        rtype = metadata.get("report_type", "SSM")
        parts.append(
            f'<div class="report-footer">'
            f"Report type: {rtype} | Generated: {generated} | "
            f"kalmanbox v{version}</div>"
        )

        return "\n".join(parts)
