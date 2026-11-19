"""Themes on the web: CSS custom properties and the all-themes stylesheet."""

from __future__ import annotations

import dataclasses
import re

import pytest

from clankit.config.schema import Config
from clankit.themes import DEFAULT_THEME, all_themes, builtin_themes, get_theme
from clankit.themes.web import css_variables, stylesheet

HEX = re.compile(r"#[0-9a-f]{6}")
BUILTIN_IDS = tuple(builtin_themes())

PROPERTIES = (
    "--bg",
    "--bg-alt",
    "--bg-raised",
    "--selection",
    "--line",
    "--fg",
    "--fg-muted",
    "--fg-subtle",
    "--accent",
    "--accent-alt",
    "--on-accent",
    "--success",
    "--warning",
    "--error",
    "--info",
    "--red",
    "--orange",
    "--yellow",
    "--green",
    "--cyan",
    "--blue",
    "--purple",
    "--syn-keyword",
    "--syn-string",
    "--syn-number",
    "--syn-comment",
    "--syn-function",
    "--syn-type",
    "--syn-variable",
    "--syn-operator",
    "--syn-tag",
    "--syn-attribute",
)


def parse(css: str) -> list[tuple[str, dict[str, str]]]:
    """Split a flat stylesheet into (selector, declarations); fail on anything left over."""
    blocks: list[tuple[str, dict[str, str]]] = []
    position = 0
    for match in re.finditer(r"\s*([^{}]+?)\s*\{([^{}]*)\}\s*", css):
        assert match.start() == position, f"unparsed text before offset {match.start()}"
        position = match.end()
        declarations: dict[str, str] = {}
        for declaration in filter(None, (part.strip() for part in match[2].split(";"))):
            name, _, value = declaration.partition(":")
            assert name.strip() not in declarations, f"{name} declared twice in {match[1]}"
            declarations[name.strip()] = value.strip()
        blocks.append((match[1], declarations))
    assert position == len(css), "unbalanced braces or trailing text"
    return blocks


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_css_variables_cover_every_token(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    properties = css_variables(theme)
    assert tuple(properties) == PROPERTIES
    assert all(HEX.fullmatch(value) for value in properties.values())
    assert properties["--bg"] == theme.bg
    assert properties["--bg-alt"] == theme.bg_alt
    assert properties["--fg-muted"] == theme.fg_muted
    assert properties["--on-accent"] == theme.on_accent
    assert properties["--syn-keyword"] == theme.syntax.keyword
    assert properties["--syn-attribute"] == theme.syntax.attribute


def test_stylesheet_has_a_root_block_and_one_block_per_theme() -> None:
    themes = builtin_themes()
    blocks = parse(stylesheet(themes.values()))
    selectors = [selector for selector, _ in blocks]
    assert selectors == [":root", *(f':root[data-theme="{theme_id}"]' for theme_id in themes)]
    for (_, declarations), theme in zip(blocks[1:], themes.values(), strict=True):
        assert tuple(declarations) == ("color-scheme", *PROPERTIES)
        assert declarations["color-scheme"] == ("dark" if theme.dark else "light")
        assert {name: declarations[name] for name in PROPERTIES} == css_variables(theme)


def test_every_block_declares_the_same_properties() -> None:
    blocks = parse(stylesheet(builtin_themes().values()))
    assert {tuple(declarations) for _, declarations in blocks} == {("color-scheme", *PROPERTIES)}
    assert len(blocks) == len(BUILTIN_IDS) + 1


def test_every_value_is_an_opaque_flat_colour() -> None:
    css = stylesheet(builtin_themes().values())
    for _, declarations in parse(css):
        scheme = declarations.pop("color-scheme")
        assert scheme in ("dark", "light")
        assert all(HEX.fullmatch(value) for value in declarations.values())
    for forbidden in ("gradient", "rgba", "hsla", "transparent", "shadow", "%"):
        assert forbidden not in css


def test_root_block_is_the_default_theme() -> None:
    themes = builtin_themes()
    root = parse(stylesheet(themes.values()))[0][1]
    assert root["--bg"] == themes[DEFAULT_THEME].bg
    assert root["color-scheme"] == "dark"


def test_root_block_follows_the_default_argument() -> None:
    themes = builtin_themes()
    blocks = dict(parse(stylesheet(themes.values(), default="github-light")))
    assert blocks[":root"] == blocks[':root[data-theme="github-light"]']
    assert blocks[":root"]["color-scheme"] == "light"
    assert blocks[":root"]["--bg"] == "#ffffff"


def test_unknown_default_falls_back_to_the_default_theme() -> None:
    nord = get_theme("nord")
    blocks = parse(stylesheet([nord], default="no-such-theme"))
    assert [selector for selector, _ in blocks] == [":root", ':root[data-theme="nord"]']
    assert blocks[0][1]["--bg"] == get_theme(DEFAULT_THEME).bg
    assert blocks[1][1]["--bg"] == nord.bg


def test_stylesheet_of_nothing_still_styles_the_page() -> None:
    blocks = parse(stylesheet([]))
    assert [selector for selector, _ in blocks] == [":root"]
    assert blocks[0][1]["--bg"] == get_theme(DEFAULT_THEME).bg


def test_user_themes_get_their_own_block(config: Config) -> None:
    config.themes = {"mine": {"base": "dracula", "accent": "#50fa7b"}}
    config.ui.theme = "mine"
    blocks = dict(parse(stylesheet(all_themes(config).values(), default=config.ui.theme)))
    assert len(blocks) == len(BUILTIN_IDS) + 2
    assert blocks[':root[data-theme="mine"]']["--accent"] == "#50fa7b"
    assert blocks[":root"] == blocks[':root[data-theme="mine"]']


def test_stylesheet_accepts_any_iterable_once() -> None:
    themes = builtin_themes()
    assert stylesheet(iter(themes.values())) == stylesheet(list(themes.values()))


def test_theme_id_is_escaped_inside_the_selector() -> None:
    hostile = dataclasses.replace(get_theme("nord"), id='a"b\\c\nd')
    css = stylesheet([hostile])
    assert ':root[data-theme="a\\22 b\\5c c\\a d"] {' in css
    assert '"b' not in css.replace('="a', "")
    assert "\nd" not in css
