"""Markdown report exporter.

Generates Markdown documents compatible with GitHub, MkDocs, and Jupyter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MarkdownExporter:
    """Export report data as Markdown document.

    Generates clean Markdown with tables, headers, and image references.
    Compatible with GitHub Flavored Markdown, MkDocs, and Jupyter.
    """

    def export(
        self,
        context: dict[str, Any],
        output: str | Path,
    ) -> Path:
        """Export report context as Markdown document.

        Parameters
        ----------
        context : dict
            Template context from a Transformer.
        output : str or Path
            Output .md file path.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)

        md = self._render_markdown(context)
        path.write_text(md, encoding="utf-8")

        return path

    def export_string(self, context: dict[str, Any]) -> str:
        """Return Markdown as a string without saving."""
        return self._render_markdown(context)

    def _render_markdown(self, context: dict[str, Any]) -> str:
        """Render complete Markdown document from context."""
        sections: list[str] = []

        model_info = context.get("model_info", {})
        model_name = model_info.get("model_name", "State Space Model")
        sections.append(f"# {model_name} Report\n")

        sections.append(self._render_model_info(model_info))

        parameters = context.get("parameters", [])
        if parameters:
            sections.append(self._render_parameters_table(parameters))

        info_criteria = context.get("info_criteria", {})
        if any(v is not None for v in info_criteria.values()):
            sections.append(self._render_info_criteria(info_criteria))

        diagnostics = context.get("diagnostics", {})
        if diagnostics:
            sections.append(self._render_diagnostics(diagnostics))

        metadata = context.get("metadata", {})
        sections.append(self._render_footer(metadata))

        return "\n\n".join(sections)

    def _render_model_info(self, info: dict[str, Any]) -> str:
        """Render model info section."""
        lines = ["## Model Summary\n"]
        rows = []
        if info.get("model_name"):
            rows.append(("Model", info["model_name"]))
        if info.get("n_obs"):
            rows.append(("Observations", str(info["n_obs"])))
        if info.get("n_states"):
            rows.append(("States", str(info["n_states"])))
        if info.get("n_params"):
            rows.append(("Parameters", str(info["n_params"])))
        if info.get("optimizer"):
            rows.append(("Optimizer", info["optimizer"]))
        converged = info.get("converged", True)
        rows.append(("Converged", "Yes" if converged else "No"))

        lines.append("| Property | Value |")
        lines.append("|:---|:---|")
        for label, value in rows:
            lines.append(f"| {label} | {value} |")

        return "\n".join(lines)

    def _render_parameters_table(self, params: list[dict[str, Any]]) -> str:
        """Render parameter estimation table."""
        lines = ["## Parameter Estimates\n"]
        lines.append("| Parameter | Estimate | Std. Error | t-stat | p-value | Sig. |")
        lines.append("|:---|---:|---:|---:|---:|:---:|")

        for p in params:
            name = p["name"]
            val = f"{p['value']:.4f}"
            se = f"{p['se']:.4f}" if "se" in p else "--"
            tstat = f"{p['t_stat']:.4f}" if "t_stat" in p else "--"
            pval = f"{p['p_value']:.4f}" if "p_value" in p else "--"
            sig = ""
            if "p_value" in p:
                pv = p["p_value"]
                if pv < 0.001:
                    sig = "***"
                elif pv < 0.01:
                    sig = "**"
                elif pv < 0.05:
                    sig = "*"
            lines.append(f"| {name} | {val} | {se} | {tstat} | {pval} | {sig} |")

        lines.append("\n*Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05*")
        return "\n".join(lines)

    def _render_info_criteria(self, ic: dict[str, float | None]) -> str:
        """Render information criteria."""
        lines = ["## Information Criteria\n"]
        lines.append("| Criterion | Value |")
        lines.append("|:---|---:|")

        for name, val in ic.items():
            display_name = name.upper() if name != "loglike" else "Log-Likelihood"
            val_str = f"{val:.4f}" if val is not None else "--"
            lines.append(f"| {display_name} | {val_str} |")

        return "\n".join(lines)

    def _render_diagnostics(self, diag: dict[str, Any]) -> str:
        """Render diagnostics section."""
        lines = ["## Diagnostics\n"]
        for test_name, test_result in diag.items():
            display_name = test_name.replace("_", " ").title()
            lines.append(f"### {display_name}\n")
            lines.append(f"{test_result}\n")
        return "\n".join(lines)

    def _render_footer(self, metadata: dict[str, str]) -> str:
        """Render footer with metadata."""
        version = metadata.get("kalmanbox_version", "unknown")
        generated = metadata.get("generated_at", "")
        report_type = metadata.get("report_type", "SSM")
        return (
            f"---\n\n*Report type: {report_type} | Generated: {generated} | kalmanbox v{version}*"
        )
