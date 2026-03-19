"""Tests for visualization themes."""

from __future__ import annotations

import matplotlib
import pytest

from kalmanbox.visualization.themes import (
    ACADEMIC_THEME,
    PRESENTATION_THEME,
    PROFESSIONAL_THEME,
    ThemeConfig,
    get_theme,
    list_themes,
    register_theme,
    reset_theme,
    set_theme,
)


class TestColorPalette:
    """Tests for ColorPalette."""

    def test_palette_has_all_fields(self) -> None:
        palette = PROFESSIONAL_THEME.colors
        assert palette.primary
        assert palette.secondary
        assert palette.tertiary
        assert palette.accent
        assert palette.background
        assert palette.text
        assert palette.grid

    def test_ci_bands_returns_colors(self) -> None:
        palette = PROFESSIONAL_THEME.colors
        bands = palette.get_band_colors(4)
        assert len(bands) == 4
        assert all(isinstance(c, str) for c in bands)

    def test_series_colors_not_empty(self) -> None:
        for theme_name in list_themes():
            theme = get_theme(theme_name)
            assert len(theme.colors.series) >= 4


class TestFontConfig:
    """Tests for FontConfig."""

    def test_font_sizes_positive(self) -> None:
        for theme_name in list_themes():
            theme = get_theme(theme_name)
            assert theme.fonts.title_size > 0
            assert theme.fonts.label_size > 0
            assert theme.fonts.tick_size > 0
            assert theme.fonts.legend_size > 0


class TestThemeConfig:
    """Tests for ThemeConfig."""

    def test_to_rcparams_returns_dict(self) -> None:
        rc = PROFESSIONAL_THEME.to_rcparams()
        assert isinstance(rc, dict)
        assert "figure.dpi" in rc
        assert "axes.labelsize" in rc
        assert "font.family" in rc

    def test_all_themes_have_names(self) -> None:
        assert PROFESSIONAL_THEME.name == "professional"
        assert ACADEMIC_THEME.name == "academic"
        assert PRESENTATION_THEME.name == "presentation"

    def test_professional_uses_sans_serif(self) -> None:
        assert PROFESSIONAL_THEME.fonts.family == "sans-serif"

    def test_academic_uses_serif(self) -> None:
        assert ACADEMIC_THEME.fonts.family == "serif"

    def test_academic_higher_dpi(self) -> None:
        assert ACADEMIC_THEME.dpi >= 300


class TestThemeFunctions:
    """Tests for theme management functions."""

    def setup_method(self) -> None:
        """Reset theme before each test."""
        reset_theme()

    def test_get_theme_by_name(self) -> None:
        theme = get_theme("professional")
        assert theme.name == "professional"
        theme = get_theme("academic")
        assert theme.name == "academic"
        theme = get_theme("presentation")
        assert theme.name == "presentation"

    def test_get_theme_none_returns_current(self) -> None:
        theme = get_theme(None)
        assert theme.name == "professional"  # default

    def test_get_theme_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown theme"):
            get_theme("nonexistent")

    def test_set_theme_changes_current(self) -> None:
        set_theme("academic")
        assert get_theme().name == "academic"

    def test_set_theme_applies_rcparams(self) -> None:
        set_theme("presentation")
        assert matplotlib.rcParams["font.family"] == ["sans-serif"]

    def test_list_themes_returns_all(self) -> None:
        themes = list_themes()
        assert "professional" in themes
        assert "academic" in themes
        assert "presentation" in themes

    def test_register_custom_theme(self) -> None:
        custom = ThemeConfig(
            name="custom_test",
            colors=PROFESSIONAL_THEME.colors,
            fonts=PROFESSIONAL_THEME.fonts,
        )
        register_theme("custom_test", custom)
        assert "custom_test" in list_themes()
        retrieved = get_theme("custom_test")
        assert retrieved.name == "custom_test"

    def test_reset_theme_restores_professional(self) -> None:
        set_theme("academic")
        reset_theme()
        assert get_theme().name == "professional"
