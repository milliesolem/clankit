"""The legibility pass: what Clankit changes in a palette before it becomes a `Theme`.

Both user interfaces separate their regions by flat surface tone alone (no borders, no
gradients, no shadows), so a theme has to deliver two things many published palettes do not:
surfaces that can be told apart, and text that can be read. `legible` enforces both and leaves
every colour that already works exactly as its author published it.

Text guarantees, measured as WCAG 2 contrast on `bg`, the surface palettes are designed for:

* the tokens in `READABLE` reach `TEXT_CONTRAST` (4.5:1, WCAG AA for body text);
* the tokens in `QUIET`, which are meant to recede, reach `QUIET_CONTRAST` (3:1);
* `fg` alone also reaches `TEXT_CONTRAST` on every other surface, so body text can sit on a
  panel, a popup, a selected row or the cursor line. (Out of reach only for a user theme whose
  surfaces lie on both sides of mid-grey; no single text colour reads on all of those.)

A colour that falls short moves toward black on a light background and toward white on a dark
one, in small steps, and stops at the first step that passes. Moving toward the end of the
scale rather than toward `fg` keeps the hue: an orange becomes a deeper orange instead of the
grey-brown it turns into when blended with grey body text.

Surface guarantee: `bg`, `bg_alt`, `bg_raised`, `selection` and `line` are pairwise at least
`SURFACE_DISTANCE` apart. A surface that collides with an earlier one in that order is derived
afresh: `fg` mixed into `bg` (the selection: `accent` mixed into `bg`), starting from the
percentage in `DERIVED_SURFACES` and rising until the tone is distinct. A derived tone can
land on a later surface, which is then derived in its turn. Nothing here assumes that `bg_alt`
is lighter or darker than `bg`; upstream themes do both.

The remaining tokens (`accent_alt` and the seven hues) are decoration and are never changed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from clankit.themes.colour import BLACK, WHITE, contrast, distance, mix, most_readable

#: Tokens used for text a user has to read.
READABLE: tuple[str, ...] = (
    "fg",
    "fg_muted",
    "accent",
    "success",
    "warning",
    "error",
    "info",
    "syntax_keyword",
    "syntax_string",
    "syntax_number",
    "syntax_function",
    "syntax_type",
    "syntax_variable",
    "syntax_operator",
    "syntax_tag",
    "syntax_attribute",
)

#: Tokens for text that is deliberately faint: hints, disabled items, code comments.
QUIET: tuple[str, ...] = ("fg_subtle", "syntax_comment")

TEXT_CONTRAST = 4.5
QUIET_CONTRAST = 3.0

#: Smallest CIE76 distance at which two surfaces count as different. Just under the classic
#: just-noticeable difference of 2.3, so the faint steps of themes such as Ayu survive while
#: near-duplicates (Dracula's `line` against its `bg_raised`, 0.8 apart) do not.
SURFACE_DISTANCE = 2.0

#: How a colliding surface is derived: the token mixed into `bg`, and the starting percentage.
DERIVED_SURFACES: dict[str, tuple[str, int]] = {
    "bg_alt": ("fg", 4),
    "bg_raised": ("fg", 8),
    "selection": ("accent", 25),
    "line": ("fg", 14),
}

#: Every surface, in the order collisions are resolved: the earlier one keeps its colour.
SURFACES: tuple[str, ...] = ("bg", *DERIVED_SURFACES)

_STEPS = 50  # a colour moves toward black or white 2% at a time


def legible(colours: Mapping[str, str]) -> dict[str, str]:
    """`colours` with the guarantees above applied; tokens that already comply are unchanged.

    `colours` must hold every theme token as ``#rrggbb``.
    """
    result = dict(colours)
    bg = result["bg"]
    # Text first: deriving surfaces mixes `fg` and `accent` into `bg`, which only yields
    # distinct tones once those two are themselves clear of the background.
    for token in READABLE:
        result[token] = readable(result[token], [bg])
    for token in QUIET:
        result[token] = readable(result[token], [bg], QUIET_CONTRAST)
    result.update(_distinct_surfaces(result))
    result["fg"] = readable(result["fg"], [result[surface] for surface in SURFACES])
    return result


def readable(colour: str, surfaces: Sequence[str], minimum: float = TEXT_CONTRAST) -> str:
    """`colour`, moved no further than needed to reach `minimum`:1 on every surface.

    The first surface decides the direction: toward white if white reads better on it than
    black, otherwise toward black. If no step satisfies every surface, that black or white is
    returned; on the first surface it is always worth at least 4.58:1.
    """
    pole = most_readable(surfaces[0], (BLACK, WHITE))
    for step in range(_STEPS):
        candidate = mix(colour, pole, step / _STEPS)
        if all(contrast(candidate, surface) >= minimum for surface in surfaces):
            return candidate
    return pole


def _distinct_surfaces(colours: Mapping[str, str]) -> dict[str, str]:
    """The five surfaces, with any that collides with an earlier one replaced."""
    bg = colours["bg"]
    kept = {"bg": bg}
    for token, (tint, start) in DERIVED_SURFACES.items():
        tone = colours[token]
        if _collides(tone, kept.values()):
            tone = _derive(bg, colours[tint], start, list(kept.values()))
        kept[token] = tone
    return kept


def _collides(tone: str, others: Iterable[str]) -> bool:
    return any(distance(tone, other) < SURFACE_DISTANCE for other in others)


def _derive(bg: str, tint: str, start: int, taken: Sequence[str]) -> str:
    """The weakest mix of `tint` into `bg`, from `start` percent up, that collides with nothing."""
    tones = (mix(bg, tint, percent / 100) for percent in range(start, 101))
    return next((tone for tone in tones if not _collides(tone, taken)), tint)
