"""HTML report exporter.

Generates self-contained HTML reports with inline CSS and JavaScript.
"""

from __future__ import annotations

from pathlib import Path


class HTMLExporter:
    """Export rendered report as self-contained HTML file.

    The output is a single HTML file with all CSS and JavaScript inlined,
    making it portable and viewable in any browser without dependencies.
    """

    def export(
        self,
        rendered_html: str,
        output: str | Path,
        css: str = "",
    ) -> Path:
        """Export rendered HTML to file.

        Parameters
        ----------
        rendered_html : str
            The rendered HTML content (body).
        output : str or Path
            Output file path.
        css : str
            CSS to inject into the <style> tag.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)

        full_html = self._wrap_html(rendered_html, css)
        path.write_text(full_html, encoding="utf-8")

        return path

    def export_string(self, rendered_html: str, css: str = "") -> str:
        """Return the complete HTML as a string without saving.

        Parameters
        ----------
        rendered_html : str
            The rendered HTML content (body).
        css : str
            CSS to inject.

        Returns
        -------
        str
            Complete HTML document string.
        """
        return self._wrap_html(rendered_html, css)

    def _wrap_html(self, body: str, css: str) -> str:
        """Wrap body content in a complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>kalmanbox Report</title>
    <style>
{css}
    </style>
</head>
<body>
{body}
<script>
// Collapsible sections
document.querySelectorAll('.section-header').forEach(function(header) {{
    header.addEventListener('click', function() {{
        this.parentElement.classList.toggle('collapsed');
    }});
}});
</script>
</body>
</html>"""
