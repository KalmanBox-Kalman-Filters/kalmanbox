"""Export utilities for kalmanbox plots.

Supports exporting matplotlib figures to PNG, SVG, PDF, and optionally
HTML via plotly conversion.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from kalmanbox.visualization.themes import get_theme


def export_figure(
    fig: Figure,
    path: str | Path,
    fmt: str | None = None,
    dpi: int | None = None,
    transparent: bool = False,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> Path:
    """Export a matplotlib figure to file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export.
    path : str or Path
        Output file path. Format is inferred from extension if fmt is None.
    fmt : str or None
        Output format: 'png', 'svg', 'pdf', or 'html'. If None, inferred
        from file extension.
    dpi : int or None
        Resolution in dots per inch. If None, uses current theme's DPI.
    transparent : bool
        Whether to use transparent background. Default False.
    bbox_inches : str
        Bounding box setting. Default 'tight'.
    **kwargs
        Additional keyword arguments passed to fig.savefig().

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    ValueError
        If format is not supported.
    """
    path = Path(path)
    if fmt is None:
        fmt = path.suffix.lstrip(".").lower()

    if not fmt:
        msg = "Cannot determine format. Provide fmt or use a file extension."
        raise ValueError(msg)

    if dpi is None:
        theme = get_theme()
        dpi = theme.dpi

    if fmt in ("png", "svg", "pdf"):
        return _export_matplotlib(
            fig,
            path,
            fmt=fmt,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            **kwargs,
        )
    elif fmt == "html":
        return _export_html(fig, path, **kwargs)
    else:
        supported = ", ".join(["png", "svg", "pdf", "html"])
        msg = f"Unsupported format '{fmt}'. Supported: {supported}"
        raise ValueError(msg)


def _export_matplotlib(
    fig: Figure,
    path: Path,
    fmt: str,
    dpi: int,
    transparent: bool,
    bbox_inches: str,
    **kwargs: Any,
) -> Path:
    """Export using matplotlib's savefig."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path),
        format=fmt,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
        **kwargs,
    )
    return path


def _export_html(
    fig: Figure,
    path: Path,
    include_plotlyjs: bool = True,
    **kwargs: Any,
) -> Path:
    """Export figure as interactive HTML using plotly.

    Attempts to convert the matplotlib figure to a plotly figure.
    If plotly is not available, falls back to embedding a PNG in HTML.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export.
    path : Path
        Output HTML file path.
    include_plotlyjs : bool
        Whether to include plotly.js in the HTML file. Default True.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Path
        Path to the saved HTML file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.io as pio
        import plotly.tools as tls

        plotly_fig = tls.mpl_to_plotly(fig)
        html_content = pio.to_html(
            plotly_fig,
            include_plotlyjs=include_plotlyjs,
            full_html=True,
        )
        path.write_text(html_content, encoding="utf-8")
    except ImportError:
        # Fallback: embed PNG in a self-contained HTML
        html_content = _fig_to_embedded_html(fig)
        path.write_text(html_content, encoding="utf-8")

    return path


def _fig_to_embedded_html(fig: Figure) -> str:
    """Convert matplotlib figure to self-contained HTML with embedded PNG."""
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>kalmanbox plot</title>
    <style>
        body {{ margin: 0; display: flex; justify-content: center;
               align-items: center; min-height: 100vh;
               background-color: #fafafa; font-family: sans-serif; }}
        img {{ max-width: 95vw; max-height: 95vh; }}
    </style>
</head>
<body>
    <img src="data:image/png;base64,{img_b64}" alt="kalmanbox plot">
</body>
</html>"""


def figure_to_bytes(fig: Figure, fmt: str = "png", dpi: int | None = None) -> bytes:
    """Convert a matplotlib figure to bytes without saving to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.
    fmt : str
        Format: 'png', 'svg', or 'pdf'. Default 'png'.
    dpi : int or None
        Resolution. If None, uses current theme's DPI.

    Returns
    -------
    bytes
        The figure as raw bytes.
    """
    if dpi is None:
        theme = get_theme()
        dpi = theme.dpi

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data


def close_figure(fig: Figure) -> None:
    """Close a matplotlib figure to free memory."""
    plt.close(fig)
