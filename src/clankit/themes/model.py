"""The theme model: one set of named colours that every renderer draws from.

A `Theme` is what the terminal UI, the web UI and syntax highlighting are all generated from,
so a colour is defined once and looks the same everywhere. Themes are made with `build_theme`,
which runs the legibility pass (`clankit.themes.legibility`); the values on a `Theme` are
therefore safe to use as documented below, and `Theme.canonical` remembers what the palette
said before that pass.

How to use the tokens:

* Surfaces (`bg`, `bg_alt`, `bg_raised`, `selection`, `line`) are flat fills, guaranteed to be
  tellable apart. `bg` is the main reading surface, `bg_alt` the secondary region (a sidebar),
  `bg_raised` what sits on top (inputs, popups), `selection` a selected row or text, `line` the
  cursor row, a hover or a divider band. `bg_alt` may be lighter or darker than `bg`.
* `fg` is body text and is readable on every surface. `fg_muted` (secondary text), `accent`,
  the status colours and the syntax colours are readable on `bg`; `fg_subtle` and
  `syntax.comment` are faint on purpose and are for text nobody has to read.
* `accent_alt` and the hues `red` ... `purple` are decoration. Several themes have no real blue
  or orange and reuse a neighbour, and nothing guarantees their contrast: use the status
  colours for meaning.
* `on_accent` is the text colour for anything filled with `accent` or a status colour.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

from clankit.themes.colour import BLACK, WHITE, contrast, most_readable
from clankit.themes.legibility import TEXT_CONTRAST, legible

DEFAULT_THEME = "ayu-dark"


@dataclass(frozen=True, slots=True)
class Syntax:
    """Colours for source code. `number` doubles as the colour of constants."""

    keyword: str
    string: str
    number: str
    comment: str
    function: str
    type: str
    variable: str
    operator: str
    tag: str
    attribute: str


@dataclass(frozen=True, slots=True)
class Theme:
    """A colour theme. Every colour is an opaque ``#rrggbb`` string."""

    id: str
    name: str
    dark: bool

    bg: str
    bg_alt: str
    bg_raised: str
    selection: str
    line: str

    fg: str
    fg_muted: str
    fg_subtle: str

    accent: str
    accent_alt: str

    success: str
    warning: str
    error: str
    info: str

    red: str
    orange: str
    yellow: str
    green: str
    cyan: str
    blue: str
    purple: str

    syntax: Syntax
    #: Upstream name and version the palette came from.
    source: str
    #: Token name -> colour as the palette defines it, before the legibility pass.
    canonical: Mapping[str, str] = field(compare=False, repr=False)

    @property
    def on_accent(self) -> str:
        """Text colour for accent-filled areas: `bg` or `fg`, whichever reads better.

        Falls back to black or white when neither reaches 4.5:1, which cannot happen for a
        theme made by `build_theme` (its `accent` already has that contrast against `bg`).
        """
        best = most_readable(self.accent, (self.bg, self.fg))
        if contrast(best, self.accent) >= TEXT_CONTRAST:
            return best
        return most_readable(self.accent, (BLACK, WHITE))

    @property
    def adjusted(self) -> tuple[str, ...]:
        """The tokens whose colour the legibility pass changed, in `TOKENS` order."""
        colours = self.colours()
        return tuple(token for token in TOKENS if colours[token] != self.canonical[token])

    def colours(self) -> dict[str, str]:
        """Token name -> colour for every name in `TOKENS`: the shape of `canonical`."""
        flat = {token: getattr(self, token) for token in _COLOUR_FIELDS}
        flat.update({f"syntax_{part}": getattr(self.syntax, part) for part in _SYNTAX_FIELDS})
        return flat

    def to_dict(self) -> dict[str, str | bool]:
        """The flat JSON form: ``id``, ``name``, ``dark``, then every token in `TOKENS`."""
        return {"id": self.id, "name": self.name, "dark": self.dark, **self.colours()}


_NOT_COLOURS = frozenset({"id", "name", "dark", "syntax", "source", "canonical"})
_COLOUR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Theme) if f.name not in _NOT_COLOURS)
_SYNTAX_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Syntax))

#: Every colour token, flat: the names used by `Theme.colours`, by `Theme.canonical` and by
#: user-defined themes in the configuration. Syntax colours are spelled ``syntax_keyword``.
TOKENS: tuple[str, ...] = (*_COLOUR_FIELDS, *(f"syntax_{part}" for part in _SYNTAX_FIELDS))


def build_theme(theme_id: str, name: str, source: str, colours: Mapping[str, str]) -> Theme:
    """Make a theme from a complete palette (every name in `TOKENS`), made legible.

    `dark` is read off the background: a theme is dark when white reads better on it than
    black does.
    """
    final = legible(colours)
    return Theme(
        id=theme_id,
        name=name,
        dark=most_readable(final["bg"], (BLACK, WHITE)) == WHITE,
        syntax=Syntax(**{part: final[f"syntax_{part}"] for part in _SYNTAX_FIELDS}),
        source=source,
        canonical={token: colours[token] for token in TOKENS},
        **{token: final[token] for token in _COLOUR_FIELDS},
    )


def variable_name(token: str) -> str:
    """A token's stylesheet name: hyphens for underscores, ``syntax_`` shortened to ``syn-``."""
    return token.replace("syntax_", "syn-").replace("_", "-")


def variables(theme: Theme) -> dict[str, str]:
    """The theme under its stylesheet names, shared by TCSS (``$bg-alt``) and CSS (``--bg-alt``).

    Every token under its `variable_name`, plus the derived ``on-accent`` next to the accents.
    """
    named: dict[str, str] = {}
    for token, colour in theme.colours().items():
        named[variable_name(token)] = colour
        if token == "accent_alt":
            named["on-accent"] = theme.on_accent
    return named
