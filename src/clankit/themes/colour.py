"""Colour maths for opaque sRGB colours written ``#rrggbb``.

Everything a theme emits is a flat, opaque colour, so this module needs no alpha channel. It
answers the three questions the legibility pass asks: can this text be read on that surface
(`contrast`, the WCAG 2 ratio), can these two surfaces be told apart (`distance`, CIE76 delta E)
and what lies between two colours (`mix`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

BLACK = "#000000"
WHITE = "#ffffff"

_HEX6 = re.compile(r"#[0-9a-f]{6}")
_HEX3 = re.compile(r"#[0-9a-f]{3}")

# D65 reference white, the white point of sRGB.
_WHITE_XYZ = (0.95047, 1.0, 1.08883)
_LAB_EPSILON = 216 / 24389
_LAB_KAPPA = 24389 / 27


def parse_hex(value: object) -> str | None:
    """Normalise ``#RGB`` or ``#RRGGBB`` in any letter case to ``#rrggbb``.

    Returns None for anything else, including colours with an alpha channel: themes separate
    regions with flat tones, so a translucent colour has no meaning here.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if _HEX6.fullmatch(text):
        return text
    if _HEX3.fullmatch(text):
        return "#" + "".join(digit * 2 for digit in text[1:])
    return None


def rgb(colour: str) -> tuple[int, int, int]:
    """The red, green and blue channels (0-255) of a ``#rrggbb`` colour."""
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def _linear(channel: int) -> float:
    """Undo the sRGB transfer curve for one 0-255 channel."""
    value = channel / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def luminance(colour: str) -> float:
    """WCAG 2 relative luminance: 0.0 for black, 1.0 for white."""
    red, green, blue = (_linear(channel) for channel in rgb(colour))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(first: str, second: str) -> float:
    """WCAG 2 contrast ratio between two colours, from 1.0 (identical) to 21.0."""
    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def mix(base: str, other: str, amount: float) -> str:
    """`base` with `amount` (0.0-1.0) of `other` blended in.

    The blend is linear in encoded sRGB, which is what a browser or editor does when it
    composites a translucent colour over an opaque one. Upstream themes define many of their
    surfaces that way, so derived tones land where a user of the theme expects them.
    """
    amount = min(max(amount, 0.0), 1.0)
    channels = (round(a + (b - a) * amount) for a, b in zip(rgb(base), rgb(other), strict=True))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _lab(colour: str) -> tuple[float, float, float]:
    """CIE L*a*b* coordinates of a colour (D65 white)."""
    red, green, blue = (_linear(channel) for channel in rgb(colour))
    xyz = (
        0.4124564 * red + 0.3575761 * green + 0.1804375 * blue,
        0.2126729 * red + 0.7151522 * green + 0.0721750 * blue,
        0.0193339 * red + 0.1191920 * green + 0.9503041 * blue,
    )
    fx, fy, fz = (
        math.cbrt(ratio) if ratio > _LAB_EPSILON else (_LAB_KAPPA * ratio + 16) / 116
        for ratio in (value / white for value, white in zip(xyz, _WHITE_XYZ, strict=True))
    )
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def distance(first: str, second: str) -> float:
    """Perceptual distance between two colours (CIE76 delta E).

    Contrast ratio only compares brightness, so it calls a red and a green of equal brightness
    identical. Surfaces are told apart by hue as well, which is what this measures. A value
    near 2.3 is the classic "just noticeable difference".
    """
    return math.dist(_lab(first), _lab(second))


def most_readable(surface: str, candidates: Iterable[str]) -> str:
    """The candidate with the highest contrast on `surface`; the first one wins a tie."""
    return max(candidates, key=lambda candidate: contrast(candidate, surface))
