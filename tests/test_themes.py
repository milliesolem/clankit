"""Themes: the vendored palettes, the legibility guarantees, and user-defined themes."""

from __future__ import annotations

import colorsys
import dataclasses
import itertools
import json
import re

import pytest

from clankit.config.schema import Config
from clankit.themes import (
    DEFAULT_THEME,
    TOKENS,
    Syntax,
    Theme,
    all_themes,
    builtin_themes,
    custom_theme_problems,
    get_theme,
)
from clankit.themes.colour import BLACK, WHITE, contrast, distance, mix, rgb
from clankit.themes.legibility import (
    QUIET,
    QUIET_CONTRAST,
    READABLE,
    SURFACE_DISTANCE,
    SURFACES,
    TEXT_CONTRAST,
    legible,
    readable,
)
from clankit.themes.model import variables
from clankit.themes.palettes import PALETTES

HEX = re.compile(r"#[0-9a-f]{6}")
BUILTIN_IDS = tuple(builtin_themes())
DECORATIVE = ("accent_alt", "red", "orange", "yellow", "green", "cyan", "blue", "purple")
STATUS = ("success", "warning", "error", "info")


def hue(colour: str) -> float:
    """Hue angle in degrees."""
    red, green, blue = (channel / 255 for channel in rgb(colour))
    return colorsys.rgb_to_hsv(red, green, blue)[0] * 360


# --------------------------------------------------------------------------------------------
# The built-in set
# --------------------------------------------------------------------------------------------


def test_builtin_themes_come_in_display_order() -> None:
    assert BUILTIN_IDS == (
        "ayu-dark",
        "ayu-mirage",
        "ayu-light",
        "monokai-pro",
        "monokai-pro-classic",
        "monokai-pro-machine",
        "monokai-pro-octagon",
        "monokai-pro-ristretto",
        "monokai-pro-spectrum",
        "monokai-pro-light",
        "dracula",
        "github-dark",
        "github-dark-dimmed",
        "github-light",
        "catppuccin-mocha",
        "catppuccin-latte",
        "tokyo-night",
        "gruvbox-dark",
        "nord",
        "one-dark",
        "solarized-dark",
        "solarized-light",
        "rose-pine",
        "kanagawa",
        "everforest-dark",
    )
    assert DEFAULT_THEME == BUILTIN_IDS[0] == Config().ui.theme


def test_tokens_are_the_flat_colour_names() -> None:
    assert TOKENS == (
        "bg",
        "bg_alt",
        "bg_raised",
        "selection",
        "line",
        "fg",
        "fg_muted",
        "fg_subtle",
        "accent",
        "accent_alt",
        "success",
        "warning",
        "error",
        "info",
        "red",
        "orange",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "syntax_keyword",
        "syntax_string",
        "syntax_number",
        "syntax_comment",
        "syntax_function",
        "syntax_type",
        "syntax_variable",
        "syntax_operator",
        "syntax_tag",
        "syntax_attribute",
    )


def test_every_token_has_exactly_one_legibility_role() -> None:
    roles = (*SURFACES, *READABLE, *QUIET, *DECORATIVE)
    assert sorted(roles) == sorted(TOKENS)
    assert SURFACES == ("bg", "bg_alt", "bg_raised", "selection", "line")


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_builtin_theme_is_complete(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    flat = theme.to_dict()
    assert list(flat) == ["id", "name", "dark", *TOKENS]
    assert (flat["id"], flat["name"], flat["dark"]) == (theme_id, theme.name, theme.dark)
    assert theme.name and theme.source
    for token in TOKENS:
        assert HEX.fullmatch(str(flat[token])), (token, flat[token])
        assert HEX.fullmatch(theme.canonical[token]), (token, theme.canonical[token])
    assert theme.colours() == {token: flat[token] for token in TOKENS}
    assert flat["syntax_keyword"] == theme.syntax.keyword
    assert json.loads(json.dumps(flat)) == flat


def test_palette_table_is_well_formed() -> None:
    names = [palette.name for palette in PALETTES.values()]
    assert len(set(names)) == len(names) == 25
    for theme_id, palette in PALETTES.items():
        assert re.fullmatch(r"[a-z0-9-]+", theme_id)
        assert tuple(palette.colours) == TOKENS, theme_id


def test_sources_record_the_upstream_versions() -> None:
    themes = builtin_themes()
    assert "ayu 9.1.0" in themes["ayu-dark"].source
    assert "2.0.17" in themes["monokai-pro"].source
    assert "6.3.5" in themes["github-dark"].source
    assert "1.8.0" in themes["catppuccin-mocha"].source
    assert all(theme.source == PALETTES[theme.id].source for theme in themes.values())


#: Signature colours per theme, as published upstream (palette research of 2026-11-19, with
#: its fact-check applied). A typo in the vendored table shows up here.
CANONICAL: dict[str, dict[str, str]] = {
    "ayu-dark": {
        "bg": "#10141c",
        "bg_alt": "#0d1017",
        "bg_raised": "#141821",
        "fg": "#bfbdb6",
        "fg_muted": "#5a6378",
        "accent": "#e6b450",
        "error": "#d95757",
        "syntax_keyword": "#ff8f40",
        "syntax_string": "#aad94c",
        "syntax_function": "#ffb454",
        "syntax_type": "#59c2ff",
        "syntax_number": "#d2a6ff",
        "syntax_tag": "#39bae6",
    },
    "ayu-mirage": {"bg": "#242936", "bg_alt": "#1f2430", "fg": "#cccac2", "accent": "#ffcc66"},
    "ayu-light": {"bg": "#fcfcfc", "bg_alt": "#f8f9fa", "fg": "#5c6166", "accent": "#f29718"},
    "monokai-pro": {
        "bg": "#2d2a2e",
        "bg_alt": "#221f22",
        "bg_raised": "#403e41",
        "fg": "#fcfcfa",
        "fg_muted": "#939293",
        "accent": "#ffd866",
        "red": "#ff6188",
        "orange": "#fc9867",
        "green": "#a9dc76",
        "cyan": "#78dce8",
        "purple": "#ab9df2",
        "syntax_keyword": "#ff6188",
        "syntax_string": "#ffd866",
        "syntax_comment": "#727072",
    },
    "monokai-pro-classic": {"bg": "#272822", "fg": "#fdfff1", "syntax_keyword": "#f92672"},
    "monokai-pro-machine": {"bg": "#273136", "fg": "#f2fffc", "accent": "#ffed72"},
    "monokai-pro-octagon": {"bg": "#282a3a", "fg": "#eaf2f1", "accent": "#ffd76d"},
    "monokai-pro-ristretto": {"bg": "#2c2525", "fg": "#fff1f3", "accent": "#f9cc6c"},
    "monokai-pro-spectrum": {"bg": "#222222", "fg": "#f7f1ff", "accent": "#fce566"},
    "monokai-pro-light": {"bg": "#faf4f2", "fg": "#29242a", "accent": "#e14775"},
    "dracula": {
        "bg": "#282a36",
        "bg_alt": "#21222c",
        "selection": "#44475a",
        "fg": "#f8f8f2",
        "fg_muted": "#6272a4",
        "accent": "#bd93f9",
        "accent_alt": "#ff79c6",
        "red": "#ff5555",
        "orange": "#ffb86c",
        "yellow": "#f1fa8c",
        "green": "#50fa7b",
        "cyan": "#8be9fd",
        "syntax_keyword": "#ff79c6",
        "syntax_number": "#bd93f9",
        "syntax_comment": "#6272a4",
    },
    "github-dark": {
        "bg": "#0d1117",
        "bg_alt": "#010409",
        "bg_raised": "#161b22",
        "selection": "#142744",
        "line": "#21262d",
        "fg": "#e6edf3",
        "fg_muted": "#7d8590",
        "accent": "#2f81f7",
        "success": "#3fb950",
        "error": "#f85149",
        "syntax_keyword": "#ff7b72",
        "syntax_string": "#a5d6ff",
        "syntax_function": "#d2a8ff",
        "syntax_comment": "#8b949e",
        "syntax_tag": "#7ee787",
    },
    "github-dark-dimmed": {"bg": "#22272e", "fg": "#adbac7", "selection": "#2c3e56"},
    "github-light": {"bg": "#ffffff", "fg": "#1f2328", "accent": "#0969da", "selection": "#cee1f8"},
    "catppuccin-mocha": {"bg": "#1e1e2e", "fg": "#cdd6f4", "accent": "#cba6f7"},
    "catppuccin-latte": {"bg": "#eff1f5", "fg": "#4c4f69", "accent": "#8839ef"},
    "tokyo-night": {"bg": "#1a1b26", "fg": "#a9b1d6", "accent": "#7aa2f7"},
    "gruvbox-dark": {"bg": "#282828", "fg": "#ebdbb2", "accent": "#fe8019"},
    "nord": {"bg": "#2e3440", "bg_alt": "#3b4252", "fg": "#d8dee9", "accent": "#88c0d0"},
    "one-dark": {"bg": "#282c34", "fg": "#abb2bf", "syntax_keyword": "#c678dd"},
    "solarized-dark": {"bg": "#002b36", "bg_alt": "#073642", "fg": "#839496", "accent": "#268bd2"},
    "solarized-light": {"bg": "#fdf6e3", "bg_alt": "#eee8d5", "fg": "#657b83"},
    "rose-pine": {"bg": "#191724", "fg": "#e0def4", "accent": "#ebbcba", "green": "#31748f"},
    "kanagawa": {"bg": "#1f1f28", "fg": "#dcd7ba", "accent": "#7e9cd8"},
    "everforest-dark": {"bg": "#2d353b", "fg": "#d3c6aa", "accent": "#a7c080"},
}


def test_signature_colours_are_listed_for_every_builtin() -> None:
    assert tuple(CANONICAL) == BUILTIN_IDS


@pytest.mark.parametrize(
    ("theme_id", "token", "colour"),
    [
        (theme_id, token, colour)
        for theme_id, row in CANONICAL.items()
        for token, colour in row.items()
    ],
)
def test_canonical_colour(theme_id: str, token: str, colour: str) -> None:
    assert builtin_themes()[theme_id].canonical[token] == colour
    assert PALETTES[theme_id].colours[token] == colour


def test_the_named_favourites_keep_their_signature_look() -> None:
    themes = builtin_themes()
    ayu, monokai, dracula, github = (
        themes[name] for name in ("ayu-dark", "monokai-pro", "dracula", "github-dark")
    )
    assert (ayu.bg, ayu.fg, ayu.accent) == ("#10141c", "#bfbdb6", "#e6b450")
    assert (ayu.syntax.keyword, ayu.syntax.string, ayu.syntax.function) == (
        "#ff8f40",
        "#aad94c",
        "#ffb454",
    )
    assert (monokai.bg, monokai.fg, monokai.accent) == ("#2d2a2e", "#fcfcfa", "#ffd866")
    assert (monokai.syntax.keyword, monokai.syntax.function) == ("#ff6188", "#a9dc76")
    assert (dracula.bg, dracula.fg, dracula.accent) == ("#282a36", "#f8f8f2", "#bd93f9")
    assert (dracula.selection, dracula.syntax.keyword) == ("#44475a", "#ff79c6")
    assert (github.bg, github.fg, github.accent) == ("#0d1117", "#e6edf3", "#2f81f7")
    assert github.adjusted == ()


def test_light_and_dark_follow_the_background() -> None:
    light = {theme.id for theme in builtin_themes().values() if not theme.dark}
    assert light == {
        "ayu-light",
        "monokai-pro-light",
        "github-light",
        "catppuccin-latte",
        "solarized-light",
    }


def test_theme_is_an_immutable_value() -> None:
    theme = get_theme("dracula")
    with pytest.raises(dataclasses.FrozenInstanceError):
        theme.bg = "#000000"  # type: ignore[misc]
    assert isinstance(theme.syntax, Syntax)
    assert theme == builtin_themes()["dracula"]
    assert len({theme, builtin_themes()["dracula"], get_theme("nord")}) == 2


def test_builtin_themes_returns_a_fresh_mapping() -> None:
    first = builtin_themes()
    first.clear()
    assert tuple(builtin_themes()) == BUILTIN_IDS


# --------------------------------------------------------------------------------------------
# Legibility guarantees, for every built-in theme
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_text_is_readable_on_the_background(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    colours = theme.colours()
    for token in READABLE:
        assert contrast(colours[token], theme.bg) >= TEXT_CONTRAST, token
    for token in QUIET:
        assert contrast(colours[token], theme.bg) >= QUIET_CONTRAST, token


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_body_text_is_readable_on_every_surface(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    colours = theme.colours()
    for surface in SURFACES:
        assert contrast(theme.fg, colours[surface]) >= TEXT_CONTRAST, surface


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_surfaces_can_be_told_apart(theme_id: str) -> None:
    colours = builtin_themes()[theme_id].colours()
    for first, second in itertools.combinations(SURFACES, 2):
        assert distance(colours[first], colours[second]) >= SURFACE_DISTANCE, (first, second)


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_on_accent_reads_on_accent_and_status_fills(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    assert theme.on_accent in (theme.bg, theme.fg)
    for fill in ("accent", *STATUS):
        assert contrast(theme.on_accent, theme.colours()[fill]) >= TEXT_CONTRAST, fill


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_only_what_fails_is_changed(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    colours, canonical = theme.colours(), theme.canonical
    assert theme.bg == canonical["bg"]
    for token in DECORATIVE:
        assert colours[token] == canonical[token], token
    for token in (*READABLE, *QUIET):
        minimum = QUIET_CONTRAST if token in QUIET else TEXT_CONTRAST
        passed_already = contrast(canonical[token], canonical["bg"]) >= minimum
        if passed_already and token != "fg":  # fg also answers to the other surfaces
            assert colours[token] == canonical[token], token
    assert theme.adjusted == tuple(t for t in TOKENS if colours[t] != canonical[t])


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_adjusted_colours_keep_their_hue(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    for token in theme.adjusted:
        if token in SURFACES:
            continue
        before, after = theme.canonical[token], theme.colours()[token]
        saturation = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb(before)))[1]
        if saturation > 0.2:  # hue is noise for near-greys
            drift = abs(hue(before) - hue(after))
            assert min(drift, 360 - drift) < 4, (token, before, after)


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_legibility_pass_is_stable(theme_id: str) -> None:
    once = legible(PALETTES[theme_id].colours)
    assert legible(once) == once
    assert once == builtin_themes()[theme_id].colours()


#: Which tokens the legibility pass changes in each built-in theme. Pinned so that a palette
#: edit, or a change to the pass, shows exactly what it does to the shipped themes.
ADJUSTED: dict[str, tuple[str, ...]] = {
    "ayu-dark": ("line", "fg_muted"),
    "ayu-mirage": ("fg_muted",),
    "ayu-light": (
        "bg_alt",
        "bg_raised",
        "line",
        "fg_muted",
        "fg_subtle",
        "accent",
        "success",
        "warning",
        "error",
        "info",
        "syntax_keyword",
        "syntax_string",
        "syntax_number",
        "syntax_comment",
        "syntax_function",
        "syntax_type",
        "syntax_operator",
        "syntax_tag",
        "syntax_attribute",
    ),
    "monokai-pro": ("selection", "fg_subtle", "syntax_comment"),
    "monokai-pro-classic": (
        "selection",
        "fg_subtle",
        "error",
        "syntax_keyword",
        "syntax_comment",
        "syntax_operator",
        "syntax_tag",
    ),
    "monokai-pro-machine": ("selection", "fg_muted", "fg_subtle", "syntax_comment"),
    "monokai-pro-octagon": ("selection", "fg_muted", "fg_subtle", "syntax_comment"),
    "monokai-pro-ristretto": ("selection", "fg_muted", "fg_subtle", "syntax_comment"),
    "monokai-pro-spectrum": ("selection", "fg_subtle", "syntax_comment"),
    "monokai-pro-light": (
        "line",
        "fg_subtle",
        "accent",
        "success",
        "warning",
        "error",
        "info",
        "syntax_keyword",
        "syntax_string",
        "syntax_comment",
        "syntax_function",
        "syntax_type",
        "syntax_operator",
        "syntax_tag",
        "syntax_attribute",
    ),
    "dracula": ("line", "fg_muted"),
    "github-dark": (),
    "github-dark-dimmed": ("fg_muted", "fg_subtle", "error"),
    "github-light": (),
    "catppuccin-mocha": ("line",),
    "catppuccin-latte": (
        "selection",
        "fg_muted",
        "fg_subtle",
        "success",
        "warning",
        "info",
        "syntax_string",
        "syntax_number",
        "syntax_function",
        "syntax_type",
        "syntax_operator",
        "syntax_tag",
        "syntax_attribute",
    ),
    "tokyo-night": ("line", "fg_muted", "fg_subtle", "error", "syntax_comment"),
    "gruvbox-dark": ("error", "syntax_keyword"),
    "nord": ("line", "fg_muted", "fg_subtle", "error", "syntax_number", "syntax_comment"),
    "one-dark": (
        "fg_muted",
        "fg_subtle",
        "accent",
        "syntax_comment",
        "syntax_variable",
        "syntax_tag",
    ),
    "solarized-dark": (
        "bg_raised",
        "selection",
        "line",
        "fg",
        "fg_muted",
        "fg_subtle",
        "accent",
        "error",
        "info",
        "syntax_comment",
        "syntax_function",
        "syntax_variable",
        "syntax_tag",
        "syntax_attribute",
    ),
    "solarized-light": (
        "bg_raised",
        "selection",
        "line",
        "fg",
        "fg_muted",
        "fg_subtle",
        "accent",
        "success",
        "warning",
        "error",
        "info",
        "syntax_keyword",
        "syntax_string",
        "syntax_number",
        "syntax_comment",
        "syntax_function",
        "syntax_type",
        "syntax_variable",
        "syntax_operator",
        "syntax_tag",
        "syntax_attribute",
    ),
    "rose-pine": ("success", "syntax_keyword"),
    "kanagawa": ("error", "info"),
    "everforest-dark": (),
}


def test_adjusted_tokens_snapshot() -> None:
    assert {theme.id: theme.adjusted for theme in builtin_themes().values()} == ADJUSTED


def test_adjusted_values_snapshot_for_the_default_theme_and_dracula() -> None:
    ayu, dracula = get_theme("ayu-dark"), get_theme("dracula")
    assert {token: ayu.colours()[token] for token in ayu.adjusted} == {
        "line": "#282c32",
        "fg_muted": "#787f90",
    }
    assert {token: dracula.colours()[token] for token in dracula.adjusted} == {
        "line": "#454750",
        "fg_muted": "#8591b8",
    }


# --------------------------------------------------------------------------------------------
# The legibility pass on its own
# --------------------------------------------------------------------------------------------


def palette(**overrides: str) -> dict[str, str]:
    """GitHub Dark, which needs no adjustment, with some tokens replaced."""
    return {**PALETTES["github-dark"].colours, **overrides}


def test_compliant_palette_is_returned_unchanged() -> None:
    assert legible(palette()) == palette()


def test_coinciding_surfaces_are_derived_from_fg_and_accent() -> None:
    same = "#161b22"
    colours = palette(bg_alt=same, bg_raised=same, selection=same, line=same)
    result = legible(colours)
    bg, fg, accent = colours["bg"], colours["fg"], colours["accent"]
    assert result["bg_alt"] == same  # the first of a colliding group keeps its colour
    assert result["bg_raised"] in {mix(bg, fg, percent / 100) for percent in range(8, 40)}
    assert result["line"] in {mix(bg, fg, percent / 100) for percent in range(14, 40)}
    assert result["selection"] == mix(bg, accent, 0.25)
    for first, second in itertools.combinations(SURFACES, 2):
        assert distance(result[first], result[second]) >= SURFACE_DISTANCE


def test_a_surface_equal_to_the_background_is_replaced() -> None:
    colours = palette(bg_alt=palette()["bg"])
    assert legible(colours)["bg_alt"] == mix(colours["bg"], colours["fg"], 0.04)


def test_near_duplicate_surfaces_count_as_colliding() -> None:
    # Dracula publishes #343746 and #353747 for two different surfaces.
    result = legible(PALETTES["dracula"].colours)
    assert result["bg_raised"] == "#343746"
    assert result["line"] != "#353747"


@pytest.mark.parametrize("bg_alt", ["#010409", "#1c2128"])
def test_bg_alt_may_be_darker_or_lighter_than_bg(bg_alt: str) -> None:
    assert legible(palette(bg_alt=bg_alt))["bg_alt"] == bg_alt


def test_failing_text_moves_toward_white_on_a_dark_background() -> None:
    dim_red = "#8a1c1c"
    fixed = legible(palette(error=dim_red))["error"]
    assert contrast(dim_red, palette()["bg"]) < TEXT_CONTRAST <= contrast(fixed, palette()["bg"])
    assert all(after >= before for before, after in zip(rgb(dim_red), rgb(fixed), strict=True))
    assert abs(hue(dim_red) - hue(fixed)) < 2


def test_failing_text_moves_toward_black_on_a_light_background() -> None:
    light = PALETTES["github-light"].colours
    orange = "#f29718"
    fixed = legible({**light, "accent": orange})["accent"]
    assert contrast(fixed, light["bg"]) >= TEXT_CONTRAST
    assert all(after <= before for before, after in zip(rgb(orange), rgb(fixed), strict=True))
    assert abs(hue(orange) - hue(fixed)) < 2  # still an orange, not a grey-brown


def test_adjustment_stops_at_the_first_step_that_passes() -> None:
    bg = palette()["bg"]
    fixed = readable("#4a5568", [bg])
    one_step_back = mix(fixed, "#4a5568", 0.2)
    assert contrast(fixed, bg) >= TEXT_CONTRAST > contrast(one_step_back, bg)
    assert contrast(fixed, bg) < TEXT_CONTRAST + 0.3


def test_quiet_tokens_only_need_three_to_one() -> None:
    bg = palette()["bg"]
    result = legible(palette(fg_subtle="#30363d", syntax_comment="#30363d", fg_muted="#30363d"))
    assert QUIET_CONTRAST <= contrast(result["fg_subtle"], bg) < TEXT_CONTRAST
    assert result["syntax_comment"] == result["fg_subtle"]
    assert contrast(result["fg_muted"], bg) >= TEXT_CONTRAST


def test_decorative_tokens_are_left_alone_even_when_unreadable() -> None:
    bg = palette()["bg"]
    result = legible(palette(accent_alt=bg, blue=bg))
    assert (result["accent_alt"], result["blue"]) == (bg, bg)


def test_readable_returns_a_passing_colour_untouched() -> None:
    assert readable("#e6edf3", ["#0d1117"]) == "#e6edf3"
    assert readable("#6e7681", ["#0d1117"], QUIET_CONTRAST) == "#6e7681"


@pytest.mark.parametrize("bg", [BLACK, WHITE, "#777777", "#808080", "#ff0000", "#00ff00"])
def test_any_background_gets_readable_text(bg: str) -> None:
    result = legible(palette(bg=bg, fg=bg, fg_muted=bg, accent=bg, error=bg))
    for token in ("fg", "fg_muted", "accent", "error"):
        assert contrast(result[token], bg) >= TEXT_CONTRAST, token


def test_body_text_gives_up_gracefully_when_surfaces_straddle_mid_grey() -> None:
    # No colour reads on both a black and a white surface; the pass must still terminate and
    # keep the text readable on the background.
    result = legible(palette(bg=BLACK, selection=WHITE))
    assert result["fg"] == WHITE
    assert result["selection"] == WHITE


# --------------------------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------------------------


def test_get_theme_falls_back_to_the_default() -> None:
    assert get_theme("dracula").id == "dracula"
    assert get_theme("no-such-theme").id == DEFAULT_THEME
    assert get_theme("").id == DEFAULT_THEME
    assert get_theme("no-such-theme", Config()).id == DEFAULT_THEME


def test_all_themes_without_user_themes_is_the_builtin_set(config: Config) -> None:
    assert all_themes() == all_themes(config) == builtin_themes()
    assert custom_theme_problems(config) == []


def test_variables_use_stylesheet_names() -> None:
    theme = get_theme("ayu-dark")
    named = variables(theme)
    assert named["bg-alt"] == theme.bg_alt
    assert named["fg-muted"] == theme.fg_muted
    assert named["syn-keyword"] == theme.syntax.keyword
    assert named["on-accent"] == theme.on_accent
    assert len(named) == len(TOKENS) + 1
    assert not any("_" in name for name in named)


def test_on_accent_falls_back_to_black_or_white() -> None:
    theme = get_theme("ayu-dark")
    assert theme.on_accent == theme.bg
    # A hand-built theme whose accent is too close to both bg and fg for either to read on it.
    odd = dataclasses.replace(theme, bg="#666666", fg="#999999", accent="#808080")
    assert odd.on_accent == BLACK
    assert dataclasses.replace(odd, accent="#6f6f6f").on_accent == WHITE


# --------------------------------------------------------------------------------------------
# User-defined themes
# --------------------------------------------------------------------------------------------


def test_custom_theme_inherits_from_the_default_theme(config: Config) -> None:
    config.themes = {"mine": {}}
    mine, default = get_theme("mine", config), get_theme(DEFAULT_THEME)
    assert (mine.id, mine.name) == ("mine", "mine")
    assert mine.colours() == default.colours()
    assert "Ayu Dark" in mine.source
    assert custom_theme_problems(config) == []


def test_custom_theme_overrides_tokens_of_its_base(config: Config) -> None:
    config.themes = {
        "mine": {
            "base": "dracula",
            "accent": "#50FA7B",
            "bg-alt": "#123",
            "bg_raised": "#3a3d4e",
            "syntax-keyword": "#8be9fd",
        }
    }
    mine, dracula = get_theme("mine", config), get_theme("dracula")
    assert (mine.accent, mine.bg_alt, mine.bg_raised) == ("#50fa7b", "#112233", "#3a3d4e")
    assert mine.syntax.keyword == "#8be9fd"
    assert (mine.bg, mine.fg, mine.syntax.string) == (dracula.bg, dracula.fg, dracula.syntax.string)
    assert mine.dark
    assert "Dracula" in mine.source
    assert custom_theme_problems(config) == []


def test_custom_themes_are_listed_after_the_builtins(config: Config) -> None:
    config.themes = {"zeta": {}, "alpha": {"base": "nord"}}
    assert tuple(all_themes(config)) == (*BUILTIN_IDS, "zeta", "alpha")
    assert tuple(all_themes()) == BUILTIN_IDS


def test_custom_colours_go_through_the_legibility_pass(config: Config) -> None:
    config.themes = {"dim": {"base": "github-dark", "accent": "#1b3a6b", "bg_alt": "#0d1117"}}
    dim = get_theme("dim", config)
    assert dim.canonical["accent"] == "#1b3a6b"
    assert contrast(dim.accent, dim.bg) >= TEXT_CONTRAST
    assert distance(dim.bg_alt, dim.bg) >= SURFACE_DISTANCE
    assert {"bg_alt", "accent"} <= set(dim.adjusted)
    assert custom_theme_problems(config) == []


def test_custom_theme_can_turn_a_dark_base_light(config: Config) -> None:
    config.themes = {"paper": {"base": "dracula", "bg": "#ffffff"}}
    paper = get_theme("paper", config)
    assert not paper.dark
    assert contrast(paper.fg, paper.bg) >= TEXT_CONTRAST
    for first, second in itertools.combinations(SURFACES, 2):
        assert distance(paper.colours()[first], paper.colours()[second]) >= SURFACE_DISTANCE


def test_unknown_base_falls_back_to_the_default_theme_and_is_reported(config: Config) -> None:
    config.themes = {"mine": {"base": "dracla", "accent": "#ff79c6"}}
    mine = get_theme("mine", config)
    assert mine.bg == get_theme(DEFAULT_THEME).bg
    assert mine.accent == "#ff79c6"
    assert custom_theme_problems(config) == [
        'theme "mine": base "dracla" is not a built-in theme (did you mean "dracula"?); '
        'inheriting from "ayu-dark" instead.'
    ]


def test_base_must_be_a_builtin_theme(config: Config) -> None:
    config.themes = {"first": {"base": "nord"}, "second": {"base": "first"}}
    assert get_theme("second", config).bg == get_theme(DEFAULT_THEME).bg
    (problem,) = custom_theme_problems(config)
    assert 'base "first" is not a built-in theme' in problem


def test_unknown_token_is_ignored_and_reported(config: Config) -> None:
    config.themes = {"mine": {"base": "nord", "acent": "#ffffff", "border": "#000000"}}
    assert get_theme("mine", config).colours() == get_theme("nord").colours()
    assert custom_theme_problems(config) == [
        'theme "mine": "acent" is not a colour token (did you mean "accent"?); '
        "this entry is ignored.",
        'theme "mine": "border" is not a colour token; this entry is ignored.',
    ]


@pytest.mark.parametrize("value", ["blue", "#12345", "#11223344", "", 7, None, True, ["#fff"]])
def test_bad_colour_keeps_the_base_value_and_is_reported(config: Config, value: object) -> None:
    config.themes = {"mine": {"base": "nord", "fg": value, "accent": "#ebcb8b"}}  # type: ignore[dict-item]
    mine = get_theme("mine", config)
    assert mine.fg == get_theme("nord").fg
    assert mine.accent == "#ebcb8b"
    (problem,) = custom_theme_problems(config)
    assert problem.startswith(
        f'theme "mine": fg is set to "{value}", which is not an opaque colour'
    )
    assert '"#rrggbb"' in problem and '"nord"' in problem


def test_custom_theme_cannot_replace_a_builtin(config: Config) -> None:
    config.themes = {"dracula": {"bg": "#000000"}}
    assert get_theme("dracula", config) == get_theme("dracula")
    assert tuple(all_themes(config)) == BUILTIN_IDS
    (problem,) = custom_theme_problems(config)
    assert problem.startswith('theme "dracula": a built-in theme already has this id')
    assert 'base "dracula"' in problem


@pytest.mark.parametrize("theme_id", ["my theme", 'x"]{}', "a,b", "", "-lead", "tab\tbed"])
def test_theme_id_with_unsafe_characters_is_ignored(config: Config, theme_id: str) -> None:
    config.themes = {theme_id: {"bg": "#000000"}, "fine.id_2-x": {}}
    assert tuple(all_themes(config)) == (*BUILTIN_IDS, "fine.id_2-x")
    assert get_theme(theme_id, config).id == DEFAULT_THEME
    (problem,) = custom_theme_problems(config)
    assert "may only contain letters, digits" in problem


def test_every_problem_of_every_theme_is_reported_in_file_order(config: Config) -> None:
    config.themes = {
        "one": {"base": "nope", "wat": "#fff", "fg": "red"},
        "nord": {},
        "two": {"line": "#gg0000"},
    }
    problems = custom_theme_problems(config)
    assert [problem.split(":")[0] for problem in problems] == [
        'theme "one"',
        'theme "one"',
        'theme "one"',
        'theme "nord"',
        'theme "two"',
    ]
    assert {theme.id for theme in all_themes(config).values()} >= {"one", "two"}


def test_selected_theme_resolves_through_the_config(config: Config) -> None:
    config.themes = {"mine": {"base": "monokai-pro"}}
    config.ui.theme = "mine"
    assert get_theme(config.ui.theme, config).bg == "#2d2a2e"
    config.ui.theme = "deleted-since"
    assert get_theme(config.ui.theme, config).id == DEFAULT_THEME


def test_theme_built_by_hand_reports_its_own_adjustments() -> None:
    theme = get_theme("dracula")
    assert isinstance(theme, Theme)
    assert theme.adjusted == ("line", "fg_muted")
    assert dataclasses.replace(theme, fg="#ffffff").adjusted == ("line", "fg", "fg_muted")
