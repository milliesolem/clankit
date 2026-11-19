"""Clankit themes as Textual themes.

    class ClankitApp(App):
        def __init__(self, config):
            super().__init__()
            for theme in textual_themes(config):
                self.register_theme(theme)
            self.theme = get_theme(config.ui.theme, config).id

        def get_theme_variable_defaults(self):
            return variable_defaults()

A Textual theme has ten colour fields; everything else a stylesheet may reference comes from
its `variables`. `textual_theme` fills both:

* the fields, mapped ``background=bg``, ``surface=bg_alt``, ``panel=bg_raised``,
  ``primary=accent=accent``, ``secondary=accent_alt``, ``foreground=fg`` and the three status
  colours;
* the names in `APP_VARIABLES`, which are the tokens Textual has no field for. Every Clankit
  theme defines all of them, so a stylesheet can use ``$fg-muted`` or ``$syn-keyword`` and
  switching between Clankit themes never meets an undefined variable. Textual's stock themes
  define none of them, which is what `variable_defaults` is for;
* the variables Textual would otherwise derive on its own in ways that leave the theme:
  automatic black-or-white text, translucent cursor, hover and selection tints, a scrollbar
  track darkened toward black. They are pinned to theme tokens here, so stock widgets render
  in flat, opaque theme colours only.

What stays Textual's: the ``-lighten-N`` / ``-darken-N`` shades, the ``-muted`` tints and the
``$text-<status>`` colours, all of which it computes from the theme's own colours.

`CodeHighlight` gives Textual's code highlighter the same colours as every other renderer.

Choices a stylesheet author should know about:

* ``$text`` is `fg`, not Textual's automatic contrast colour. On an accent or status fill use
  ``$on-accent``.
* The cursor row of lists, trees and tables is the `selection` surface with `fg` text, not a
  solid accent bar; a blurred cursor and a hovered row are the `line` surface.
* Scrollbars are a `fg_subtle` thumb on a `bg` track with no separate hover track.
* The footer sits on `bg`, where its muted descriptions and accent keys are readable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.highlight import HighlightTheme
from textual.theme import Theme as TextualTheme

from clankit.themes import DEFAULT_THEME, all_themes, get_theme
from clankit.themes.model import Theme, variable_name, variables
from clankit.themes.syntax import BODY, TOKEN_COLOURS, TOKEN_FONTS

if TYPE_CHECKING:
    from clankit.config.schema import Config

#: The TCSS variables every Clankit theme defines on top of Textual's own.
APP_VARIABLES: tuple[str, ...] = (
    "fg-muted",
    "fg-subtle",
    "bg-alt",
    "bg-raised",
    "selection",
    "line",
    "accent-alt",
    "info",
    "on-accent",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "syn-keyword",
    "syn-string",
    "syn-number",
    "syn-comment",
    "syn-function",
    "syn-type",
    "syn-variable",
    "syn-operator",
    "syn-tag",
    "syn-attribute",
)


class CodeHighlight(HighlightTheme):
    """Code colours for ``textual.highlight.highlight`` and Markdown code fences.

    Textual's stock highlighter blends its colours with opacity. This one names theme variables
    (``$syn-keyword`` ...), so a single class serves every Clankit theme and follows a theme
    switch. Unlisted token types fall back to ``$text``, which Clankit themes pin to `fg`.
    """

    STYLES = {
        **{
            token: f"${variable_name(role)}"
            for token, role in TOKEN_COLOURS.items()
            if role != BODY
        },
        **{token: " ".join(fonts) for token, fonts in TOKEN_FONTS.items()},
    }


def textual_theme(theme: Theme) -> TextualTheme:
    """The Textual theme for a Clankit theme, registered under the same id."""
    return TextualTheme(
        name=theme.id,
        primary=theme.accent,
        secondary=theme.accent_alt,
        accent=theme.accent,
        foreground=theme.fg,
        background=theme.bg,
        surface=theme.bg_alt,
        panel=theme.bg_raised,
        success=theme.success,
        warning=theme.warning,
        error=theme.error,
        dark=theme.dark,
        variables={**_app_variables(theme), **_pinned(theme)},
    )


def textual_themes(config: Config | None = None) -> list[TextualTheme]:
    """Every available theme (built-in, then user-defined) as a Textual theme."""
    return [textual_theme(theme) for theme in all_themes(config).values()]


def variable_defaults() -> dict[str, str]:
    """Values for `APP_VARIABLES` when the active theme is not one of Clankit's.

    Return this from ``App.get_theme_variable_defaults``. Without it, a stylesheet that uses
    ``$fg-muted`` fails to load under Textual's stock themes, which an app starts on and which
    stay selectable unless they are unregistered. The values are the default theme's.
    """
    return _app_variables(get_theme(DEFAULT_THEME))


def _app_variables(theme: Theme) -> dict[str, str]:
    named = variables(theme)
    return {name: named[name] for name in APP_VARIABLES}


def _pinned(theme: Theme) -> dict[str, str]:
    """Textual's own variables, pinned to theme tokens (see the module docstring for why)."""
    return {
        # Textual round-trips its colour fields through HSL, which can shift a channel by one.
        # Stating them again keeps $background and friends identical to the web UI's values.
        "primary": theme.accent,
        "secondary": theme.accent_alt,
        "accent": theme.accent,
        "foreground": theme.fg,
        "background": theme.bg,
        "surface": theme.bg_alt,
        "panel": theme.bg_raised,
        "success": theme.success,
        "warning": theme.warning,
        "error": theme.error,
        # Text: by default automatic black or white at reduced opacity.
        "text": theme.fg,
        "text-muted": theme.fg_muted,
        "text-disabled": theme.fg_subtle,
        "foreground-muted": theme.fg_muted,
        "foreground-disabled": theme.fg_subtle,
        # Cursor rows and hover: by default a solid accent bar and translucent tints.
        "block-cursor-background": theme.selection,
        "block-cursor-foreground": theme.fg,
        "block-cursor-blurred-background": theme.line,
        "block-cursor-blurred-foreground": theme.fg,
        "block-hover-background": theme.line,
        # Focus and active surfaces.
        "border": theme.accent,
        "border-blurred": theme.line,
        "surface-active": theme.bg_raised,
        # Text entry and mouse selection: by default translucent accent.
        "input-cursor-background": theme.fg,
        "input-cursor-foreground": theme.bg,
        "input-selection-background": theme.selection,
        "input-selection-foreground": theme.fg,
        "screen-selection-background": theme.selection,
        # Scrollbars: by default an accent-tinted thumb on a track darkened toward black.
        "scrollbar": theme.fg_subtle,
        "scrollbar-hover": theme.fg_muted,
        "scrollbar-active": theme.accent,
        "scrollbar-background": theme.bg,
        "scrollbar-background-hover": theme.bg,
        "scrollbar-background-active": theme.bg,
        "scrollbar-corner-color": theme.bg,
        # Links: by default automatic text colour.
        "link-color": theme.accent,
        "link-background-hover": theme.accent,
        "link-color-hover": theme.on_accent,
        # Footer.
        "footer-background": theme.bg,
        "footer-foreground": theme.fg_muted,
        "footer-key-foreground": theme.accent,
        "footer-description-foreground": theme.fg_muted,
        # Buttons: label colour on a filled button is automatic by default.
        "button-foreground": theme.fg,
        "button-color-foreground": theme.on_accent,
        # Markdown headings: the sixth level is translucent by default.
        "markdown-h1-color": theme.accent,
        "markdown-h2-color": theme.accent,
        "markdown-h3-color": theme.accent,
        "markdown-h4-color": theme.fg,
        "markdown-h5-color": theme.fg,
        "markdown-h6-color": theme.fg_muted,
    }
