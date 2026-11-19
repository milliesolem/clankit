"""User-defined themes from the configuration file.

``Config.themes`` maps a theme id to its definition: ``base`` names the built-in theme to
start from (default: the default theme), every other entry sets one token from
`clankit.themes.TOKENS` to a ``#rrggbb`` colour. Token names may be written with hyphens
(``bg-alt``), the way everything else in the configuration file is spelled.

A definition can never stop Clankit from starting. What happens to a faulty one is fixed:

* an id that a built-in theme already uses, or that contains anything but letters, digits,
  ``-``, ``_`` and ``.``: the whole definition is ignored;
* an unknown ``base``: the theme inherits from the default theme instead;
* an unknown token: that entry is ignored;
* a value that is not an opaque colour: that token keeps the base theme's value.

Each of these is reported by `resolve` as a sentence a user can act on. The overrides go
through the same legibility pass as built-in palettes, so a user colour that cannot be read is
adjusted like any other (`Theme.adjusted` says which).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping

from clankit.themes.colour import parse_hex
from clankit.themes.model import DEFAULT_THEME, TOKENS, Theme, build_theme
from clankit.themes.palettes import PALETTES

BASE_KEY = "base"

_VALID_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def resolve(theme_id: str, definition: Mapping[str, object]) -> tuple[Theme | None, list[str]]:
    """Build one user-defined theme.

    Returns the theme, or None if the definition has to be ignored, and the problems found.
    """
    label = f'theme "{theme_id}"'
    if theme_id in PALETTES:
        return None, [
            f"{label}: a built-in theme already has this id, so this definition is ignored. "
            f'To adjust the built-in, define a theme with another id and base "{theme_id}".'
        ]
    if not _VALID_ID.fullmatch(theme_id):
        return None, [
            f'{label}: a theme id may only contain letters, digits, "-", "_" and ".", '
            "so this definition is ignored."
        ]
    problems: list[str] = []
    base = _base(label, definition, problems)
    colours = {**PALETTES[base].colours, **_overrides(label, base, definition, problems)}
    source = f"user theme based on {PALETTES[base].name}"
    return build_theme(theme_id, theme_id, source, colours), problems


def _base(label: str, definition: Mapping[str, object], problems: list[str]) -> str:
    """The id of the built-in theme to inherit from; the default theme if `base` is unusable."""
    base = definition.get(BASE_KEY, DEFAULT_THEME)
    if isinstance(base, str) and base in PALETTES:
        return base
    problems.append(
        f'{label}: base "{base}" is not a built-in theme{_suggestion(str(base), PALETTES)}; '
        f'inheriting from "{DEFAULT_THEME}" instead.'
    )
    return DEFAULT_THEME


def _overrides(
    label: str, base: str, definition: Mapping[str, object], problems: list[str]
) -> dict[str, str]:
    """The valid token -> colour entries of a definition."""
    colours: dict[str, str] = {}
    for key, value in definition.items():
        if key == BASE_KEY:
            continue
        token = key.replace("-", "_")
        colour = parse_hex(value)
        if token not in TOKENS:
            problems.append(
                f'{label}: "{key}" is not a colour token{_suggestion(token, TOKENS)}; '
                "this entry is ignored."
            )
        elif colour is None:
            problems.append(
                f'{label}: {key} is set to "{value}", which is not an opaque colour; write it '
                f'as "#rrggbb". The value from "{base}" is used instead.'
            )
        else:
            colours[token] = colour
    return colours


def _suggestion(word: str, candidates: Iterable[str]) -> str:
    """A " (did you mean ...?)" hint for a misspelt name, or nothing."""
    close = difflib.get_close_matches(word, list(candidates), n=1)
    return f' (did you mean "{close[0]}"?)' if close else ""
