"""Colour themes: one definition per theme for the terminal UI, the web UI and code.

    from clankit.themes import get_theme
    theme = get_theme(config.ui.theme, config)    # never raises; unknown id -> DEFAULT_THEME
    theme.bg, theme.fg_muted, theme.syntax.keyword

The renderers live in submodules so that neither UI imports the other's toolkit:
`clankit.themes.textual` (Textual themes), `clankit.themes.web` (CSS custom properties) and
`clankit.themes.syntax` (Pygments styles and ANSI colours). `clankit.themes.model` documents
what each token is for and what is guaranteed about it.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from clankit.themes import custom
from clankit.themes.model import DEFAULT_THEME, TOKENS, Syntax, Theme, build_theme
from clankit.themes.palettes import PALETTES

if TYPE_CHECKING:
    from clankit.config.schema import Config

__all__ = [
    "DEFAULT_THEME",
    "TOKENS",
    "Syntax",
    "Theme",
    "all_themes",
    "builtin_themes",
    "custom_theme_problems",
    "get_theme",
]


@functools.cache
def _builtins() -> tuple[Theme, ...]:
    return tuple(
        build_theme(theme_id, palette.name, palette.source, palette.colours)
        for theme_id, palette in PALETTES.items()
    )


def builtin_themes() -> dict[str, Theme]:
    """The themes that ship with Clankit, id -> theme, in the order they are offered."""
    return {theme.id: theme for theme in _builtins()}


def all_themes(config: Config | None = None) -> dict[str, Theme]:
    """Built-in themes followed by the usable user-defined themes from `config.themes`."""
    themes = builtin_themes()
    if config is not None:
        for theme_id, definition in config.themes.items():
            theme, _ = custom.resolve(theme_id, definition)
            if theme is not None:
                themes[theme_id] = theme
    return themes


def get_theme(theme_id: str, config: Config | None = None) -> Theme:
    """The theme with this id, or the default theme if there is none. Never raises."""
    themes = all_themes(config)
    return themes.get(theme_id) or themes[DEFAULT_THEME]


def custom_theme_problems(config: Config) -> list[str]:
    """What is wrong with the user-defined themes, one readable sentence per problem.

    Empty when every definition is clean. See `clankit.themes.custom` for what Clankit does
    with a faulty definition.
    """
    problems: list[str] = []
    for theme_id, definition in config.themes.items():
        problems.extend(custom.resolve(theme_id, definition)[1])
    return problems
