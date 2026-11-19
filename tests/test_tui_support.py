"""Helpers for the terminal UI tests, and tests of the glyph lint they share.

The lint enforces the owner's visual rules on rendered screens: no rounded corners, no
box-drawing lines, no eighth-block hairlines and no emoji. The only block glyphs allowed are
the solid bars the design uses (half and full blocks).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.app import App
from textual.pilot import Pilot

from clankit.config.schema import Config
from clankit.config.store import ConfigStore
from clankit.runtime import Runtime

#: Code points above Latin-1 that have the Unicode ``Emoji`` property (emoji-data.txt,
#: Unicode 18.0), as inclusive ranges. Anything from U+1F000 up is checked separately.
EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x203C, 0x203C), (0x2049, 0x2049), (0x2122, 0x2122), (0x2139, 0x2139),
    (0x2194, 0x2199), (0x21A9, 0x21AA), (0x231A, 0x231B), (0x2328, 0x2328),
    (0x23CF, 0x23CF), (0x23E9, 0x23F3), (0x23F8, 0x23FA), (0x24C2, 0x24C2),
    (0x25AA, 0x25AB), (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE),
    (0x2600, 0x2604), (0x260E, 0x260E), (0x2611, 0x2611), (0x2614, 0x2615),
    (0x2618, 0x2618), (0x261D, 0x261D), (0x2620, 0x2620), (0x2622, 0x2623),
    (0x2626, 0x2626), (0x262A, 0x262A), (0x262E, 0x262F), (0x2638, 0x263A),
    (0x2640, 0x2640), (0x2642, 0x2642), (0x2648, 0x2653), (0x265F, 0x2660),
    (0x2663, 0x2663), (0x2665, 0x2666), (0x2668, 0x2668), (0x267B, 0x267B),
    (0x267E, 0x267F), (0x2692, 0x2697), (0x2699, 0x2699), (0x269B, 0x269C),
    (0x26A0, 0x26A1), (0x26A7, 0x26A7), (0x26AA, 0x26AB), (0x26B0, 0x26B1),
    (0x26BD, 0x26BE), (0x26C4, 0x26C5), (0x26C8, 0x26C8), (0x26CE, 0x26CF),
    (0x26D1, 0x26D1), (0x26D3, 0x26D4), (0x26E9, 0x26EA), (0x26F0, 0x26F5),
    (0x26F7, 0x26FA), (0x26FD, 0x26FD), (0x2702, 0x2702), (0x2705, 0x2705),
    (0x2708, 0x270D), (0x270F, 0x270F), (0x2712, 0x2712), (0x2714, 0x2714),
    (0x2716, 0x2716), (0x271D, 0x271D), (0x2721, 0x2721), (0x2728, 0x2728),
    (0x2733, 0x2734), (0x2744, 0x2744), (0x2747, 0x2747), (0x274C, 0x274C),
    (0x274E, 0x274E), (0x2753, 0x2755), (0x2757, 0x2757), (0x2763, 0x2764),
    (0x2795, 0x2797), (0x27A1, 0x27A1), (0x27B0, 0x27B0), (0x27BF, 0x27BF),
    (0x2934, 0x2935), (0x2B05, 0x2B07), (0x2B1B, 0x2B1C), (0x2B50, 0x2B50),
    (0x2B55, 0x2B55), (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297),
    (0x3299, 0x3299),
)  # fmt: skip

#: Block elements the design uses: half blocks for bars, the full block, the half rows.
ALLOWED_BLOCKS = frozenset("▌▐█▀▄")


def forbidden(char: str) -> bool:
    """Is `char` ruled out on screen by the visual rules?"""
    code = ord(char)
    if 0x2500 <= code <= 0x257F:  # box drawing, including the rounded corners
        return True
    if 0x2580 <= code <= 0x259F:  # block elements: only the solid bars are allowed
        return char not in ALLOWED_BLOCKS
    if code == 0xFE0F or 0x1F000 <= code <= 0x1FAFF:
        return True
    return any(low <= code <= high for low, high in EMOJI_RANGES)


def offenders(text: str) -> Counter[str]:
    return Counter(char for char in text if forbidden(char))


def screen_text(app: App) -> str:
    """What the active screen shows, as plain text (one line per terminal row)."""
    strips = app.screen._compositor.render_strips()
    return "\n".join(strip.text for strip in strips)


def assert_clean(app: App, where: str) -> None:
    found = offenders(screen_text(app))
    assert not found, f"forbidden glyphs on {where}: {dict(found)}"


async def wait_for(pilot: Pilot, predicate: Callable[[], object], within: float = 5.0) -> None:
    """Let the app run until `predicate()` is true; fail after `within` seconds."""
    deadline = time.monotonic() + within
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("the interface did not reach the expected state in time")
        await pilot.pause(0.02)
    await pilot.pause(0.05)  # let the last refresh paint


def demo_runtime(monkeypatch: pytest.MonkeyPatch, *, speed: str = "0") -> Runtime:
    """A runtime whose sessions play the scripted demo and ask before shell commands."""
    monkeypatch.setenv("CLANKIT_DEMO_SPEED", speed)
    config = Config()
    config.orchestration.method = "demo"
    config.permissions.mode = "ask"
    return Runtime(ConfigStore.in_memory(config), persist_sessions=False)


def demo_workspace(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir(exist_ok=True)
    return path


# -- tests of the lint itself -----------------------------------------------------------------


@pytest.mark.parametrize("char", [*"╭╮╰╯─│┌┐└┘├┤━┃▔▁▏▕▎▊▂", "\u26a0", "\u2714", "\u25b6", "\U0001f50e"])
def test_lint_rejects_frames_hairlines_and_emoji(char: str) -> None:
    assert forbidden(char)


@pytest.mark.parametrize("char", [*"▌█▀▄○●◐◆◇✓✗!-▾▸…•◦‣ ", "\u203a", "a", "$"])
def test_lint_accepts_the_design_glyphs(char: str) -> None:
    assert not forbidden(char)
