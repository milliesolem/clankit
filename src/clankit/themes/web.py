"""Clankit themes as CSS custom properties for the web UI.

`stylesheet` produces one block of variables per theme, selected by ``data-theme`` on the root
element, so switching themes in the browser is a single attribute change and no page ever
needs a colour literal:

    <html data-theme="dracula">
    body { background: var(--bg); color: var(--fg); }

The properties are the theme's tokens under their stylesheet names (``--bg-alt``,
``--fg-muted``, ``--syn-keyword``, ...) plus ``--on-accent``; `clankit.themes.model` says what
each is for. Every block defines the same properties and sets ``color-scheme`` so that native
controls and scrollbars follow the theme. Every value is an opaque ``#rrggbb``.
"""

from __future__ import annotations

from collections.abc import Iterable

from clankit.themes import DEFAULT_THEME, get_theme
from clankit.themes.model import Theme, variables


def css_variables(theme: Theme) -> dict[str, str]:
    """The theme as CSS custom properties: ``{"--bg": "#0d1017", "--bg-alt": ..., ...}``."""
    return {f"--{name}": colour for name, colour in variables(theme).items()}


def stylesheet(themes: Iterable[Theme], default: str = DEFAULT_THEME) -> str:
    """One stylesheet for all `themes`.

    ``:root`` carries the theme with id `default`, which is what a page shows before (or
    without) a ``data-theme`` attribute; if `themes` has no such theme, Clankit's default theme
    takes that place. Each theme, the default one included, then gets its own
    ``:root[data-theme="<id>"]`` block.
    """
    listed = list(themes)
    by_id = {theme.id: theme for theme in listed}
    blocks = [_block(":root", by_id.get(default) or get_theme(DEFAULT_THEME))]
    blocks.extend(_block(f":root[data-theme={_css_string(theme.id)}]", theme) for theme in listed)
    return "\n".join(blocks)


def _block(selector: str, theme: Theme) -> str:
    lines = [f"{selector} {{", f"  color-scheme: {'dark' if theme.dark else 'light'};"]
    lines.extend(f"  {name}: {colour};" for name, colour in css_variables(theme).items())
    lines.append("}\n")
    return "\n".join(lines)


def _css_string(text: str) -> str:
    """`text` as a quoted CSS string, safe whatever it contains.

    Theme ids from the configuration are restricted to harmless characters, but a `Theme` can
    be built by hand and its id ends up inside a selector.
    """
    unsafe = {'"', "\\"}
    escaped = (f"\\{ord(ch):x} " if ch in unsafe or not ch.isprintable() else ch for ch in text)
    return '"' + "".join(escaped) + '"'
