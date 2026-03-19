"""CSS management for kalmanbox reports.

Provides a 3-layer CSS system:
1. Base: layout, typography, reset
2. Component: tables, sections, cards
3. Theme: colors from active visualization theme
"""

from __future__ import annotations

from kalmanbox.visualization.themes import ThemeConfig, get_theme


class CSSManager:
    """Manages CSS generation for reports.

    Combines three layers of CSS:
    - Base: structural layout and typography
    - Component: table, section, card styling
    - Theme: dynamic colors from visualization theme
    """

    def get_css(self, theme: str | None = None) -> str:
        """Generate complete CSS string combining all layers.

        Parameters
        ----------
        theme : str or None
            Theme name for color layer. If None, uses current active theme.

        Returns
        -------
        str
            Complete CSS string.
        """
        theme_config = get_theme(theme)
        return "\n".join(
            [
                self._base_css(),
                self._component_css(),
                self._theme_css(theme_config),
            ]
        )

    def _base_css(self) -> str:
        """Base CSS: reset, layout, typography."""
        return """
/* === Base Layer === */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

h1 { font-size: 1.8rem; margin-bottom: 1rem; border-bottom: 2px solid; padding-bottom: 0.5rem; }
h2 { font-size: 1.4rem; margin-top: 2rem; margin-bottom: 0.8rem; }
h3 { font-size: 1.1rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }

p { margin-bottom: 0.8rem; }

.report-container {
    display: flex;
    gap: 2rem;
}

.report-sidebar {
    width: 220px;
    flex-shrink: 0;
    position: sticky;
    top: 1rem;
    align-self: flex-start;
    max-height: calc(100vh - 2rem);
    overflow-y: auto;
}

.report-main {
    flex: 1;
    min-width: 0;
}

.report-sidebar ul {
    list-style: none;
    padding: 0;
}

.report-sidebar li {
    margin-bottom: 0.3rem;
}

.report-sidebar a {
    text-decoration: none;
    font-size: 0.9rem;
    display: block;
    padding: 0.3rem 0.5rem;
    border-radius: 4px;
    transition: background 0.2s;
}

.report-sidebar a:hover {
    background: rgba(0,0,0,0.05);
}

.report-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid;
    font-size: 0.8rem;
    opacity: 0.7;
}

@media (max-width: 768px) {
    .report-container { flex-direction: column; }
    .report-sidebar { width: 100%; position: static; }
}
"""

    def _component_css(self) -> str:
        """Component CSS: tables, sections, cards."""
        return """
/* === Component Layer === */
.section {
    margin-bottom: 2rem;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.1);
}

.section-header {
    cursor: pointer;
    user-select: none;
}

.section-header::before {
    content: "\\25BC ";
    font-size: 0.7em;
    transition: transform 0.2s;
}

.section.collapsed .section-content {
    display: none;
}

.section.collapsed .section-header::before {
    content: "\\25B6 ";
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}

thead th {
    text-align: left;
    padding: 0.6rem 0.8rem;
    border-bottom: 2px solid;
    font-weight: 600;
}

tbody td {
    padding: 0.4rem 0.8rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}

tbody tr:hover {
    background: rgba(0,0,0,0.02);
}

.numeric {
    text-align: right;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.85rem;
}

.significant { font-weight: 600; }

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.info-card {
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid rgba(0,0,0,0.1);
}

.info-card .label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.7;
}

.info-card .value {
    font-size: 1.3rem;
    font-weight: 600;
    font-family: "SF Mono", "Fira Code", monospace;
}

.plot-container {
    margin: 1rem 0;
    text-align: center;
}

.plot-container img {
    max-width: 100%;
    border-radius: 4px;
}
"""

    def _theme_css(self, theme: ThemeConfig) -> str:
        """Theme CSS: dynamic colors from visualization theme."""
        c = theme.colors
        return f"""
/* === Theme Layer: {theme.name} === */
body {{
    color: {c.text};
    background-color: {c.background};
}}

h1 {{ color: {c.primary}; border-color: {c.primary}; }}
h2 {{ color: {c.primary}; }}

a {{ color: {c.primary}; }}
a:hover {{ color: {c.accent}; }}

.report-sidebar a {{ color: {c.text}; }}
.report-sidebar a:hover {{ background: {c.tertiary}33; }}

.section {{ background: {c.background}; }}
thead th {{ border-color: {c.primary}; color: {c.primary}; }}
tbody td {{ border-color: {c.grid}; }}

.info-card {{ background: {c.tertiary}15; border-color: {c.grid}; }}
.info-card .label {{ color: {c.secondary}; }}
.info-card .value {{ color: {c.primary}; }}

.significant {{ color: {c.accent}; }}

.report-footer {{ border-color: {c.grid}; color: {c.secondary}; }}
"""
