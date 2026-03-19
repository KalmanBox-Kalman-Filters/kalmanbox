"""Visual themes for kalmanbox plots.

Provides Professional, Academic, and Presentation themes with
color palettes, font configurations, and matplotlib rcParams overrides.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for a theme."""

    primary: str
    secondary: str
    tertiary: str
    accent: str
    background: str
    text: str
    grid: str
    ci_bands: list[str] = field(default_factory=list)
    series: list[str] = field(default_factory=list)

    def get_band_colors(self, n: int = 4) -> list[str]:
        """Get n graduated band colors for fan charts."""
        if self.ci_bands:
            return self.ci_bands[:n]
        # Fallback: generate graduated alpha versions of primary
        return [self.primary] * n


@dataclass(frozen=True)
class FontConfig:
    """Font configuration for a theme."""

    family: str
    title_size: float
    label_size: float
    tick_size: float
    legend_size: float
    annotation_size: float


@dataclass(frozen=True)
class ThemeConfig:
    """Complete theme configuration."""

    name: str
    colors: ColorPalette
    fonts: FontConfig
    line_width: float = 1.5
    grid_alpha: float = 0.3
    grid_style: str = "--"
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"
    spine_visible: bool = True
    spine_color: str = "#cccccc"
    dpi: int = 150

    def to_rcparams(self) -> dict[str, Any]:
        """Convert theme to matplotlib rcParams dict."""
        return {
            "figure.facecolor": self.figure_facecolor,
            "axes.facecolor": self.axes_facecolor,
            "axes.edgecolor": self.spine_color,
            "axes.labelcolor": self.colors.text,
            "axes.labelsize": self.fonts.label_size,
            "axes.titlesize": self.fonts.title_size,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "axes.prop_cycle": mpl.cycler(color=self.colors.series),
            "grid.color": self.colors.grid,
            "grid.alpha": self.grid_alpha,
            "grid.linestyle": self.grid_style,
            "lines.linewidth": self.line_width,
            "font.family": self.fonts.family,
            "font.size": self.fonts.label_size,
            "xtick.labelsize": self.fonts.tick_size,
            "ytick.labelsize": self.fonts.tick_size,
            "legend.fontsize": self.fonts.legend_size,
            "figure.titlesize": self.fonts.title_size + 2,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "text.color": self.colors.text,
            "xtick.color": self.colors.text,
            "ytick.color": self.colors.text,
        }


# --- Theme Definitions ---

PROFESSIONAL_THEME = ThemeConfig(
    name="professional",
    colors=ColorPalette(
        primary="#2c5f8a",
        secondary="#7fa5c4",
        tertiary="#b8d4e8",
        accent="#d4592a",
        background="#ffffff",
        text="#333333",
        grid="#e0e0e0",
        ci_bands=["#2c5f8a", "#5b87ad", "#8aafcf", "#b8d4e8"],
        series=["#2c5f8a", "#d4592a", "#4a9c6d", "#8b6caf", "#c4873a", "#c44e52"],
    ),
    fonts=FontConfig(
        family="sans-serif",
        title_size=14.0,
        label_size=11.0,
        tick_size=10.0,
        legend_size=10.0,
        annotation_size=9.0,
    ),
    line_width=1.5,
    grid_alpha=0.3,
    grid_style="--",
    spine_color="#cccccc",
    dpi=150,
)

ACADEMIC_THEME = ThemeConfig(
    name="academic",
    colors=ColorPalette(
        primary="#000000",
        secondary="#555555",
        tertiary="#999999",
        accent="#cc0000",
        background="#ffffff",
        text="#000000",
        grid="#dddddd",
        ci_bands=["#666666", "#888888", "#aaaaaa", "#cccccc"],
        series=["#000000", "#cc0000", "#0066cc", "#009933", "#ff6600", "#9933cc"],
    ),
    fonts=FontConfig(
        family="serif",
        title_size=12.0,
        label_size=10.0,
        tick_size=9.0,
        legend_size=9.0,
        annotation_size=8.0,
    ),
    line_width=1.2,
    grid_alpha=0.2,
    grid_style=":",
    spine_color="#000000",
    spine_visible=True,
    dpi=300,
)

PRESENTATION_THEME = ThemeConfig(
    name="presentation",
    colors=ColorPalette(
        primary="#1a73e8",
        secondary="#34a853",
        tertiary="#fbbc04",
        accent="#ea4335",
        background="#ffffff",
        text="#202124",
        grid="#f0f0f0",
        ci_bands=["#1a73e8", "#4a90e8", "#7ab3f0", "#a8d0f8"],
        series=["#1a73e8", "#ea4335", "#34a853", "#fbbc04", "#9334e6", "#ff6d01"],
    ),
    fonts=FontConfig(
        family="sans-serif",
        title_size=18.0,
        label_size=14.0,
        tick_size=12.0,
        legend_size=12.0,
        annotation_size=11.0,
    ),
    line_width=2.5,
    grid_alpha=0.15,
    grid_style="-",
    spine_visible=False,
    spine_color="#f0f0f0",
    dpi=100,
)

# --- Theme Registry ---

_THEMES: dict[str, ThemeConfig] = {
    "professional": PROFESSIONAL_THEME,
    "academic": ACADEMIC_THEME,
    "presentation": PRESENTATION_THEME,
}

_current_theme: ThemeConfig = PROFESSIONAL_THEME


def get_theme(name: str | None = None) -> ThemeConfig:
    """Get a theme by name, or the current active theme if name is None.

    Parameters
    ----------
    name : str or None
        Theme name: 'professional', 'academic', or 'presentation'.
        If None, returns the currently active theme.

    Returns
    -------
    ThemeConfig
        The requested theme configuration.

    Raises
    ------
    ValueError
        If the theme name is not recognized.
    """
    if name is None:
        return _current_theme
    if name not in _THEMES:
        available = ", ".join(sorted(_THEMES.keys()))
        msg = f"Unknown theme '{name}'. Available: {available}"
        raise ValueError(msg)
    return _THEMES[name]


def set_theme(name: str) -> None:
    """Set the active theme and apply it to matplotlib rcParams.

    Parameters
    ----------
    name : str
        Theme name: 'professional', 'academic', or 'presentation'.
    """
    global _current_theme  # noqa: PLW0603
    theme = get_theme(name)
    _current_theme = theme
    _apply_theme(theme)


def _apply_theme(theme: ThemeConfig) -> None:
    """Apply a theme's rcParams to matplotlib."""
    plt.rcdefaults()
    rc = theme.to_rcparams()
    for key, val in rc.items():
        with contextlib.suppress(KeyError, ValueError):
            mpl.rcParams[key] = val


def register_theme(name: str, theme: ThemeConfig) -> None:
    """Register a custom theme.

    Parameters
    ----------
    name : str
        Theme name for later retrieval.
    theme : ThemeConfig
        Complete theme configuration.
    """
    _THEMES[name] = theme


def list_themes() -> list[str]:
    """List available theme names."""
    return sorted(_THEMES.keys())


def reset_theme() -> None:
    """Reset to default matplotlib settings and professional theme."""
    global _current_theme  # noqa: PLW0603
    plt.rcdefaults()
    _current_theme = PROFESSIONAL_THEME
