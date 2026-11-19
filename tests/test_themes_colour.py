"""Colour maths: hex parsing, WCAG luminance and contrast, mixing, perceptual distance."""

from __future__ import annotations

import pytest

from clankit.themes.colour import (
    BLACK,
    WHITE,
    contrast,
    distance,
    luminance,
    mix,
    most_readable,
    parse_hex,
    rgb,
)


@pytest.mark.parametrize(
    ("written", "normalised"),
    [
        ("#0d1017", "#0d1017"),
        ("#E6B450", "#e6b450"),
        ("#FFF", "#ffffff"),
        ("#1a2", "#11aa22"),
        ("  #abcdef\n", "#abcdef"),
    ],
)
def test_parse_hex_normalises_opaque_colours(written: str, normalised: str) -> None:
    assert parse_hex(written) == normalised


@pytest.mark.parametrize(
    "not_a_colour",
    [
        "",
        "blue",
        "abcdef",
        "#abcd",
        "#abcde",
        "#gggggg",
        "#11223344",
        "rgb(1, 2, 3)",
        None,
        5,
        True,
    ],
)
def test_parse_hex_rejects_everything_else(not_a_colour: object) -> None:
    assert parse_hex(not_a_colour) is None


def test_rgb_channels() -> None:
    assert rgb("#0d1017") == (13, 16, 23)


def test_luminance_matches_the_wcag_definition() -> None:
    assert luminance(BLACK) == 0.0
    assert luminance(WHITE) == pytest.approx(1.0)
    # The three coefficients of the formula, one primary at a time.
    assert luminance("#ff0000") == pytest.approx(0.2126)
    assert luminance("#00ff00") == pytest.approx(0.7152)
    assert luminance("#0000ff") == pytest.approx(0.0722)
    assert luminance("#808080") == pytest.approx(0.21586, abs=1e-5)


def test_contrast_against_known_wcag_ratios() -> None:
    assert contrast(BLACK, WHITE) == pytest.approx(21.0)
    assert contrast("#123456", "#123456") == pytest.approx(1.0)
    # The textbook pair: #767676 is the lightest grey that passes AA on white, #777777 fails.
    assert contrast("#767676", WHITE) == pytest.approx(4.54, abs=0.005)
    assert contrast("#777777", WHITE) == pytest.approx(4.48, abs=0.005)
    assert contrast("#777777", WHITE) < 4.5 <= contrast("#767676", WHITE)


def test_contrast_ignores_argument_order() -> None:
    assert contrast("#bfbdb6", "#0d1017") == contrast("#0d1017", "#bfbdb6")


@pytest.mark.parametrize(
    ("text", "background", "ratio"),
    [
        ("#bfbdb6", "#0d1017", 10.12),  # Ayu Dark 9 text on its base surface
        ("#6272a4", "#282a36", 3.03),  # Dracula comment
        ("#e6edf3", "#0d1117", 16.02),  # GitHub Dark text
        ("#657b83", "#fdf6e3", 4.13),  # Solarized Light text
        ("#f29718", "#f8f9fa", 2.16),  # Ayu Light accent
    ],
)
def test_contrast_reproduces_the_palette_research(text: str, background: str, ratio: float) -> None:
    assert round(contrast(text, background), 2) == ratio


def test_mix_endpoints_midpoint_and_clamping() -> None:
    assert mix("#102030", "#f0e0d0", 0.0) == "#102030"
    assert mix("#102030", "#f0e0d0", 1.0) == "#f0e0d0"
    assert mix(BLACK, WHITE, 0.5) == "#808080"
    assert mix(BLACK, WHITE, -3) == BLACK
    assert mix(BLACK, WHITE, 7) == WHITE


def test_mix_is_alpha_compositing_as_editors_do_it() -> None:
    # Monokai Pro defines its current line as #fcfcfa at alpha 0x0c and its selection as
    # #c1c0c0 at alpha 0x26, both over #2d2a2e; the opaque results are published values.
    assert mix("#2d2a2e", "#fcfcfa", 0x0C / 255) == "#373438"
    assert mix("#2d2a2e", "#c1c0c0", 0x26 / 255) == "#434044"


def test_distance_in_cielab() -> None:
    assert distance("#abcdef", "#abcdef") == 0.0
    assert distance(BLACK, WHITE) == pytest.approx(100.0, abs=0.01)
    # sRGB red is L*a*b* (53.24, 80.09, 67.20); from black that is the length of the vector.
    assert distance("#ff0000", BLACK) == pytest.approx(117.33, abs=0.01)
    assert distance("#ff0000", "#00ff00") == distance("#00ff00", "#ff0000")


def test_distance_sees_hue_where_contrast_does_not() -> None:
    orange, teal = "#b46400", "#008c8c"
    assert contrast(orange, teal) < 1.1
    assert distance(orange, teal) > 50


def test_most_readable_picks_the_higher_contrast_and_the_first_on_a_tie() -> None:
    assert most_readable("#0d1017", (BLACK, WHITE)) == WHITE
    assert most_readable("#fdf6e3", (BLACK, WHITE)) == BLACK
    assert most_readable("#0d1017", ("#bfbdb6", "#5a6378", "#e6b450")) == "#bfbdb6"
    # #595959 and #a6a6a6 are 7.0:1 and 2.4:1 on white; on a tie the earlier candidate wins.
    assert most_readable(WHITE, ("#a6a6a6", "#595959", "#595959")) == "#595959"
    first, second = "#10141c", "#10141c"
    assert most_readable(WHITE, (first, second)) is first
