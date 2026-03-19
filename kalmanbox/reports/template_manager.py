"""Template management for kalmanbox reports.

Loads and renders Jinja2 templates from the built-in templates directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import jinja2

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


_TEMPLATE_DIR = Path(__file__).parent / "templates"


class TemplateManager:
    """Manages Jinja2 template loading and rendering.

    Templates are loaded from the built-in templates directory
    with support for template inheritance via Jinja2.
    """

    def __init__(self, template_dir: str | Path | None = None) -> None:
        """Initialize TemplateManager.

        Parameters
        ----------
        template_dir : str, Path, or None
            Directory containing templates. If None, uses built-in directory.
        """
        if not HAS_JINJA2:
            msg = "Jinja2 is required for report generation. Install it with: pip install jinja2"
            raise ImportError(msg)

        self._template_dir = Path(template_dir) if template_dir else _TEMPLATE_DIR
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._template_dir)),
            autoescape=jinja2.select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # Register custom filters
        self._env.filters["fmt_number"] = _fmt_number
        self._env.filters["fmt_pvalue"] = _fmt_pvalue
        self._env.filters["significance"] = _significance_stars

    def load(self, template_name: str) -> jinja2.Template:
        """Load a template by name.

        Parameters
        ----------
        template_name : str
            Template name (e.g., 'ssm_report.html').

        Returns
        -------
        jinja2.Template
            The loaded template.
        """
        return self._env.get_template(template_name)

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """Load and render a template with context.

        Parameters
        ----------
        template_name : str
            Template name.
        context : dict
            Template context variables.

        Returns
        -------
        str
            Rendered template string.
        """
        template = self.load(template_name)
        return template.render(**context)

    def render_string(self, template_string: str, context: dict[str, Any]) -> str:
        """Render a template from a string.

        Parameters
        ----------
        template_string : str
            Jinja2 template string.
        context : dict
            Template context variables.

        Returns
        -------
        str
            Rendered string.
        """
        template = self._env.from_string(template_string)
        return template.render(**context)

    @property
    def template_dir(self) -> Path:
        """Return the template directory path."""
        return self._template_dir


def _fmt_number(value: Any, decimals: int = 4) -> str:
    """Format a number with specified decimal places."""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def _fmt_pvalue(value: Any) -> str:
    """Format a p-value with appropriate precision."""
    try:
        p = float(value)
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"
    except (ValueError, TypeError):
        return str(value)


def _significance_stars(p_value: Any) -> str:
    """Return significance stars for a p-value."""
    try:
        p = float(p_value)
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.10:
            return "."
        return ""
    except (ValueError, TypeError):
        return ""
