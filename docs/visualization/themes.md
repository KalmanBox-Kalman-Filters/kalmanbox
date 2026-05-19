# Themes

Every kalmanbox plot function inherits its visual style from a global theme. A single
call to `set_theme()` at the start of a session configures colours, fonts, line widths,
grid appearance, and figure spacing for every subsequent plot, ensuring a consistent look
across all outputs without per-function style arguments.

---

## Quick start

```python
from kalmanbox.visualization import set_theme

# Pick one of the four built-in themes
set_theme("kalmanbox_default")      # balanced, screen-friendly
set_theme("kalmanbox_dark")         # dark background for presentations
set_theme("kalmanbox_paper")        # serif fonts, publication-quality
set_theme("kalmanbox_presentation") # large fonts, high-contrast colours

# All subsequent plot calls inherit the chosen theme
from kalmanbox.visualization import plot_filtered_state, plot_components
fig1 = plot_filtered_state(result, ci=0.95)
fig2 = plot_components(result)
```

---

## Built-in themes

kalmanbox ships with four themes that cover the most common use cases for state-space
model visualisation.

### `kalmanbox_default`

The default theme is designed for day-to-day exploratory analysis on screen. It uses a
white background, a 10-colour tab palette, a light gray grid, and a sans-serif font at
a comfortable size.

| Property | Value |
|---|---|
| Background | White (`#FFFFFF`) |
| Primary palette | `tab10` |
| Font family | Sans-serif (system default) |
| Font size (base) | 12 pt |
| Grid | Thin, `#E0E0E0`, major lines only |
| Spines | All four, `#BDBDBD` |
| Line width | 1.5 pt |
| Confidence band α | 0.20 |
| Figure DPI | 100 |

```python
set_theme("kalmanbox_default")
```

**Best for:** exploratory analysis in a Jupyter notebook or IDE; quick visual checks
during model development; sharing results with colleagues who will view figures on screen.

### `kalmanbox_dark`

A dark-background theme suitable for presentations, slide decks, and dashboards with a
dark UI. All colours are chosen for high contrast against the dark background. Fonts and
line widths are slightly larger than the default for readability at a distance.

| Property | Value |
|---|---|
| Background | Near-black (`#1e1e2e`) |
| Primary palette | Bright 8-colour custom (`#7CB9FF`, `#FF8A65`, `#81C784`, `#FFD54F`, `#BA68C8`, `#4DD0E1`, `#F06292`, `#A5D6A7`) |
| Font family | Sans-serif |
| Font size (base) | 13 pt |
| Grid | Subtle, `#3a3a5c`, major only |
| Spines | Off (frameless) |
| Line width | 2.0 pt |
| Confidence band α | 0.18 |
| Figure DPI | 100 |

```python
set_theme("kalmanbox_dark")
```

**Best for:** presentations on projectors or monitors with dark themes; slide decks
produced in Beamer, PowerPoint, or Keynote with dark templates; dashboards embedded in
dark-themed web applications.

### `kalmanbox_paper`

A theme designed for figures in academic publications. It uses a serif font, no
background grid, thinner lines, and a conservative grayscale-compatible palette. Figures
produced with this theme are print-ready and comply with the style guidelines of most
econometrics and statistics journals.

| Property | Value |
|---|---|
| Background | White (`#FFFFFF`) |
| Primary palette | Grayscale-safe 6-colour sequence derived from ColorBrewer `Dark2` |
| Font family | Serif (Times New Roman or DejaVu Serif) |
| Font size (base) | 10 pt |
| Grid | None |
| Spines | Bottom and left only (L-shape) |
| Line width | 1.0 pt |
| Confidence band α | 0.15 |
| Figure DPI | 300 |

```python
set_theme("kalmanbox_paper")
```

**Best for:** journal figures; PDF reports destined for print; LaTeX documents that
use `\includegraphics`; any context where grayscale printing must remain legible.

!!! tip "LaTeX font matching"
    To match the font of a LaTeX document exactly, configure matplotlib's TeX renderer
    after calling `set_theme`:

    ```python
    import matplotlib
    set_theme("kalmanbox_paper")
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    ```

    This embeds Computer Modern math fonts in exported PDFs, matching the default
    LaTeX look. Requires a working LaTeX installation on your system.

### `kalmanbox_presentation`

A high-contrast theme optimised for live presentations. Fonts are large, lines are
thick, and the colour palette is bold so that plots read clearly from the back of a
room. Grids are minimal to reduce visual clutter when projected.

| Property | Value |
|---|---|
| Background | White (`#FFFFFF`) |
| Primary palette | Bold 8-colour sequence with maximum hue separation |
| Font family | Sans-serif |
| Font size (base) | 15 pt |
| Grid | Horizontal major lines only, `#F5F5F5` |
| Spines | Bottom and left only |
| Line width | 2.5 pt |
| Confidence band α | 0.22 |
| Figure DPI | 150 |

```python
set_theme("kalmanbox_presentation")
```

**Best for:** live talks; conference presentations; screen-shared Jupyter notebooks
during remote calls.

---

## Theme comparison

The table below summarises the four built-in themes side by side:

| Property | `kalmanbox_default` | `kalmanbox_dark` | `kalmanbox_paper` | `kalmanbox_presentation` |
|---|---|---|---|---|
| Background | White | `#1e1e2e` | White | White |
| Font | Sans-serif, 12 pt | Sans-serif, 13 pt | Serif, 10 pt | Sans-serif, 15 pt |
| Grid | Light gray | Subtle dark | None | Horizontal only |
| Spines | All four | None | L-shape | L-shape |
| Line width | 1.5 pt | 2.0 pt | 1.0 pt | 2.5 pt |
| DPI | 100 | 100 | 300 | 150 |
| Use case | Exploration | Presentations | Publications | Live talks |

---

## API reference

### `set_theme`

Apply a theme globally, affecting all subsequent plot function calls in the session.

```python
kalmanbox.visualization.set_theme(
    name: str = "kalmanbox_default",
    *,
    font_size: float | None = None,
    line_width: float | None = None,
    palette: str | list[str] | None = None,
    dpi: int | None = None,
    context: str | None = None,
    rc: dict | None = None,
) -> None
```

#### Parameters

`name` : `str`, default `"kalmanbox_default"`
:   Name of the theme to apply. One of `"kalmanbox_default"`, `"kalmanbox_dark"`,
    `"kalmanbox_paper"`, `"kalmanbox_presentation"`, or any theme registered via
    `register_theme()`.

`font_size` : `float | None`, default `None`
:   Override the theme's base font size in points. All derived font sizes (title,
    ticks, labels) scale proportionally. `None` uses the theme default.

`line_width` : `float | None`, default `None`
:   Override the default line width in points. `None` uses the theme default.

`palette` : `str | list[str] | None`, default `None`
:   Override the colour cycle. Pass a matplotlib palette name (e.g. `"Set2"`,
    `"tab20"`) or a list of hex colour strings. `None` uses the theme's default
    palette.

`dpi` : `int | None`, default `None`
:   Override the figure resolution. `None` uses the theme default.

`context` : `str | None`, default `None`
:   seaborn-style scaling context: one of `"paper"`, `"notebook"`, `"talk"`,
    `"poster"`. When provided, scales fonts and line widths independently of the
    `font_size` and `line_width` overrides. `None` uses the theme's built-in scaling.

`rc` : `dict | None`, default `None`
:   Additional `matplotlib.rcParams` key–value pairs that are applied after the theme
    settings. These take final precedence over everything else.

#### Returns

`None`. The function mutates `matplotlib.rcParams` in place.

#### Example

```python
from kalmanbox.visualization import set_theme

# Apply the paper theme with a slightly larger font
set_theme("kalmanbox_paper", font_size=11)

# Same theme but override the colour palette for a specific figure
set_theme("kalmanbox_paper", palette="Dark2")

# Apply custom rcParams on top of the dark theme
set_theme("kalmanbox_dark", rc={"axes.edgecolor": "#555577"})
```

---

### `get_theme`

Retrieve the `ThemeConfig` object for a named theme without applying it.

```python
kalmanbox.visualization.get_theme(
    name: str,
) -> ThemeConfig
```

#### Parameters

`name` : `str`
:   Name of a built-in or registered theme.

#### Returns

`ThemeConfig` — a dataclass with all theme settings. See [`ThemeConfig`](#themeconfig)
for the full field reference.

#### Example

```python
from kalmanbox.visualization import get_theme

config = get_theme("kalmanbox_paper")
print(config.font_size)     # 10
print(config.palette)       # ['#1B7837', '#762A83', ...]
print(config.grid_style)    # 'none'
```

---

### `register_theme`

Register a custom `ThemeConfig` under a name so it can be activated with `set_theme`.

```python
kalmanbox.visualization.register_theme(
    name: str,
    config: ThemeConfig,
    *,
    overwrite: bool = False,
) -> None
```

#### Parameters

`name` : `str`
:   The name under which the theme is registered. Must be unique unless
    `overwrite=True`.

`config` : `ThemeConfig`
:   The theme configuration object to register.

`overwrite` : `bool`, default `False`
:   If `True`, silently overwrite an existing theme with the same name. If `False`
    and the name already exists, a `ValueError` is raised.

#### Returns

`None`

#### Example

```python
from kalmanbox.visualization import ThemeConfig, register_theme, set_theme

my_theme = ThemeConfig(
    background_color="#F5F5F0",
    primary_palette=["#2196F3", "#E91E63", "#4CAF50", "#FF9800"],
    font_family="DejaVu Sans",
    font_size=11,
    line_width=1.8,
    grid_style="major",
    grid_color="#DEDEDE",
    spine_style="left-bottom",
    confidence_alpha=0.18,
    dpi=120,
)
register_theme("my_custom_theme", my_theme)
set_theme("my_custom_theme")
```

---

### `ThemeConfig`

A dataclass that holds all configurable properties of a kalmanbox visual theme.

```python
@dataclasses.dataclass
class kalmanbox.visualization.ThemeConfig:
    background_color: str = "#FFFFFF"
    primary_palette: list[str] = field(default_factory=lambda: _TAB10_PALETTE)
    font_family: str = "sans-serif"
    font_size: float = 12.0
    title_size: float | None = None      # defaults to font_size * 1.15
    label_size: float | None = None      # defaults to font_size
    tick_size: float | None = None       # defaults to font_size * 0.9
    legend_size: float | None = None     # defaults to font_size * 0.95
    line_width: float = 1.5
    marker_size: float = 5.0
    grid_style: str = "major"            # "major", "both", "none"
    grid_color: str = "#E0E0E0"
    grid_alpha: float = 1.0
    grid_linewidth: float = 0.5
    spine_style: str = "all"             # "all", "left-bottom", "none"
    spine_color: str = "#BDBDBD"
    confidence_alpha: float = 0.20
    figure_facecolor: str = "#FFFFFF"
    axes_facecolor: str = "#FFFFFF"
    dpi: int = 100
    rc_extra: dict = field(default_factory=dict)
```

#### Field reference

`background_color` : `str`
:   Hex colour for the figure and axes background.

`primary_palette` : `list[str]`
:   Ordered list of hex colour strings that become the default colour cycle for all
    plot elements. At least 4 colours are recommended; 8–10 is standard.

`font_family` : `str`
:   Matplotlib font family string. Common values: `"sans-serif"`, `"serif"`,
    `"monospace"`, or a specific font name like `"Helvetica"`.

`font_size` : `float`
:   Base font size in points. Titles, labels, and tick labels are derived from this
    value via their respective multipliers.

`title_size`, `label_size`, `tick_size`, `legend_size` : `float | None`
:   Override specific text element sizes in points. `None` applies the theme's
    default ratio relative to `font_size`.

`line_width` : `float`
:   Default line width for all plot lines (state trajectories, confidence band
    borders, etc.) in points.

`marker_size` : `float`
:   Default scatter marker size in points.

`grid_style` : `str`
:   Which grid lines to show. `"major"` draws major grid lines only; `"both"` draws
    major and minor; `"none"` disables the grid entirely.

`grid_color` : `str`
:   Hex colour for grid lines.

`grid_alpha` : `float`
:   Opacity of grid lines, 0–1.

`grid_linewidth` : `float`
:   Grid line width in points.

`spine_style` : `str`
:   Which axes borders (spines) to draw. `"all"` keeps all four borders (standard
    matplotlib); `"left-bottom"` keeps only the left and bottom spines (publication
    style); `"none"` removes all spines.

`spine_color` : `str`
:   Hex colour for visible spines.

`confidence_alpha` : `float`
:   Default opacity for shaded confidence / credible bands in state and coefficient
    plots. Lower values produce more transparent bands, useful when many overlapping
    bands are plotted.

`figure_facecolor` : `str`
:   Hex colour for the outer figure background (the region outside the axes).

`axes_facecolor` : `str`
:   Hex colour for the axes area background. For the dark theme this differs from
    `figure_facecolor` (which is `#1e1e2e`) by using a slightly lighter shade.

`dpi` : `int`
:   Dots per inch for figures created within this theme. Saved figures can
    independently override DPI via `fig.savefig(..., dpi=300)`.

`rc_extra` : `dict`
:   A dictionary of additional `matplotlib.rcParams` key–value pairs that are merged
    last, after all other theme settings, giving them highest precedence.

---

## Creating custom themes

### Minimal theme from scratch

The simplest way to create a custom theme is to start from `ThemeConfig` defaults and
override only the fields you want to change:

```python
from kalmanbox.visualization import ThemeConfig, register_theme, set_theme

# A warm, muted theme for internal reports
warm_report = ThemeConfig(
    background_color="#FFFDE7",      # very light yellow background
    primary_palette=[
        "#BF360C", "#E65100", "#F9A825",
        "#558B2F", "#01579B", "#4527A0",
    ],
    font_family="serif",
    font_size=11,
    line_width=1.6,
    grid_style="major",
    grid_color="#F5E6C8",
    spine_style="left-bottom",
    confidence_alpha=0.18,
    dpi=120,
)

register_theme("warm_report", warm_report)
set_theme("warm_report")
```

### Inheriting from a built-in theme

Use `get_theme()` to read an existing theme's configuration and modify it with
`dataclasses.replace()`:

```python
import dataclasses
from kalmanbox.visualization import get_theme, register_theme, set_theme

# Start from the paper theme but increase font size for a poster
poster = dataclasses.replace(
    get_theme("kalmanbox_paper"),
    font_size=16,
    line_width=2.0,
    dpi=150,
    spine_style="left-bottom",
)

register_theme("poster", poster)
set_theme("poster")
```

### Integrating with matplotlib rcParams directly

Because `ThemeConfig.rc_extra` accepts any `matplotlib.rcParams` key, you can inject
low-level matplotlib settings that kalmanbox does not expose through named fields:

```python
from kalmanbox.visualization import ThemeConfig, register_theme, set_theme

hatch_theme = ThemeConfig(
    primary_palette=["#1565C0", "#B71C1C", "#2E7D32", "#4A148C"],
    font_size=11,
    line_width=1.4,
    rc_extra={
        "hatch.linewidth": 1.2,
        "axes.prop_cycle": "cycler('color', ['#1565C0','#B71C1C','#2E7D32','#4A148C'])",
        "figure.constrained_layout.use": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#BDBDBD",
    },
)

register_theme("hatch_theme", hatch_theme)
set_theme("hatch_theme")
```

### Temporary theme overrides with context managers

To apply a theme only within a code block and restore the previous settings
afterwards, use `kalmanbox.visualization.theme_context`:

```python
from kalmanbox.visualization import set_theme, theme_context

# Apply the default theme globally
set_theme("kalmanbox_default")

# Override for a single figure without changing the global state
with theme_context("kalmanbox_paper"):
    fig = plot_filtered_state(result, ci=0.95)
    fig.savefig("paper_figure.pdf", bbox_inches="tight")

# Global theme is restored to kalmanbox_default here
fig2 = plot_components(result)   # uses kalmanbox_default
```

---

## Integration with matplotlib rcParams

`set_theme()` works by updating `matplotlib.rcParams` in place. You can inspect the
full set of parameters it writes using `get_theme()`:

```python
from kalmanbox.visualization import get_theme
import matplotlib

config = get_theme("kalmanbox_paper")

# Manually reproduce what set_theme("kalmanbox_paper") writes
matplotlib.rcParams.update({
    "figure.facecolor":      config.figure_facecolor,
    "axes.facecolor":        config.axes_facecolor,
    "axes.edgecolor":        config.spine_color,
    "axes.linewidth":        0.8,
    "axes.grid":             config.grid_style != "none",
    "axes.grid.which":       config.grid_style if config.grid_style != "none" else "major",
    "grid.color":            config.grid_color,
    "grid.alpha":            config.grid_alpha,
    "grid.linewidth":        config.grid_linewidth,
    "lines.linewidth":       config.line_width,
    "lines.markersize":      config.marker_size,
    "font.family":           config.font_family,
    "font.size":             config.font_size,
    "axes.titlesize":        config.title_size or config.font_size * 1.15,
    "axes.labelsize":        config.label_size or config.font_size,
    "xtick.labelsize":       config.tick_size or config.font_size * 0.9,
    "ytick.labelsize":       config.tick_size or config.font_size * 0.9,
    "legend.fontsize":       config.legend_size or config.font_size * 0.95,
    "figure.dpi":            config.dpi,
    **config.rc_extra,
})
```

Because themes work through `rcParams`, they are fully compatible with every matplotlib
function and library that respects `rcParams`, including seaborn (when seaborn's own
theme is not active), pandas plotting, and any custom plot code.

!!! warning "seaborn interaction"
    If you call `seaborn.set_theme()` or `seaborn.set_style()` after `set_theme()`, seaborn
    will overwrite the kalmanbox settings. Always call kalmanbox's `set_theme()` last if
    you mix both libraries.

---

## Same plot in all four themes

The following code generates an identical filtered-state plot using each built-in theme,
making it easy to compare how the same content looks in different contexts.

```python
import matplotlib.pyplot as plt
from kalmanbox import LocalLevelModel
from kalmanbox.visualization import set_theme, plot_filtered_state
import numpy as np

rng = np.random.default_rng(7)
y = np.cumsum(rng.normal(0, 1, 120)) + rng.normal(0, 2, 120)
model = LocalLevelModel()
result = model.fit(y)

themes = [
    "kalmanbox_default",
    "kalmanbox_dark",
    "kalmanbox_paper",
    "kalmanbox_presentation",
]

for theme_name in themes:
    set_theme(theme_name)
    fig = plot_filtered_state(
        result,
        ci=0.95,
        title=f"Local Level Model — theme: {theme_name}",
    )
    fig.tight_layout()
    fig.savefig(f"state_{theme_name}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())   # preserve background colour
    plt.close(fig)
```

The `facecolor=fig.get_facecolor()` argument is important for the dark theme: without
it, `savefig` applies a white background by default, washing out the dark panel.

---

## Jupyter notebook tips

### Auto-detect environment

In a Jupyter notebook the theme DPI is automatically raised to 144 so that figures
look crisp on retina displays. Override this with the `dpi` argument:

```python
set_theme("kalmanbox_default", dpi=96)   # smaller inline figures in Jupyter
```

### Inline display without saving

Jupyter renders the figure automatically when it is the last expression in a cell.
You can call plot functions directly without assigning the return value:

```python
set_theme("kalmanbox_paper", font_size=10)
plot_filtered_state(result, ci=0.95)   # displayed inline in Jupyter
```

### Persistent theme across cells

Call `set_theme` once in a top-level cell and the theme persists for all subsequent
cells in the notebook session:

```python
# Cell 1
from kalmanbox.visualization import set_theme
set_theme("kalmanbox_paper", font_size=10)

# Cell 2 — uses kalmanbox_paper automatically
plot_components(result)

# Cell 3 — uses kalmanbox_paper automatically
plot_diagnostic_panel(result)
```

---

## Theme design guidelines

When designing a custom theme for specific audience or context, the following principles
help ensure clarity and consistency:

**Use a perceptually uniform colour palette.** Palettes derived from ColorBrewer
(`Dark2`, `Set1`) or Viridis-family sequential maps are perceptually designed and
print safely in both colour and grayscale. Avoid rainbow palettes that introduce
perceptual artefacts.

**Match line width to figure size.** A 1.5 pt line that reads well at 10 × 4 inches
may be too thin at 5 × 2 inches. Thicker lines (2.0–2.5 pt) are better for small
embedded figures or projections.

**Reduce grid noise for dense plots.** Innovation plots and filter-comparison plots
draw many overlapping lines. Setting `grid_style="none"` and `spine_style="left-bottom"`
reduces visual noise in those contexts.

**Set DPI for the output medium.** `dpi=100` is sufficient for screen viewing; use
`dpi=300` for print; `dpi=150` is a safe compromise for presentations that may be
viewed both on screen and printed.

---

## Complete API quick reference

| Function / Class | Purpose |
|---|---|
| `set_theme(name, **kwargs)` | Apply a theme globally, with optional overrides |
| `get_theme(name) -> ThemeConfig` | Retrieve a theme configuration without applying it |
| `register_theme(name, config)` | Register a custom `ThemeConfig` for use with `set_theme` |
| `theme_context(name)` | Context manager: apply a theme temporarily within a `with` block |
| `ThemeConfig(...)` | Dataclass holding all theme properties |
| `ThemeConfig.rc_extra` | Dict of raw `matplotlib.rcParams` merged last |

---

## Related pages

- [Factor plots](factor-plots.md) — DFM factor visualisation functions
- [TVP plots](tvp-plots.md) — time-varying coefficient visualisation functions
- [State plots](state-plots.md) — filtered and smoothed state estimates
- [Component plots](component-plots.md) — trend, seasonal, and cycle decomposition
- [Innovation plots](innovation-plots.md) — diagnostic residual plots
- [Filter comparison plots](filter-plots.md) — NEES, covariance trace, multi-filter overlay
