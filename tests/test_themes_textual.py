"""Themes in Textual: field mapping, the app variables, and a real app switching themes."""

from __future__ import annotations

import re

import pytest
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import VerticalScroll
from textual.highlight import highlight
from textual.widgets import Button, Footer, Input, Markdown, OptionList, Static, Switch

from clankit.config.schema import Config
from clankit.themes import DEFAULT_THEME, Theme, all_themes, builtin_themes, get_theme
from clankit.themes.colour import contrast
from clankit.themes.legibility import QUIET_CONTRAST, TEXT_CONTRAST
from clankit.themes.model import variables
from clankit.themes.textual import (
    APP_VARIABLES,
    CodeHighlight,
    textual_theme,
    textual_themes,
    variable_defaults,
)

HEX = re.compile(r"#[0-9a-f]{6}")
BUILTIN_IDS = tuple(builtin_themes())
SAMPLE = 'def greet(name):\n    return "hi " + str(1)  # note\n'


def resolved(colour: Color) -> str:
    return colour.hex.lower()


class Swatches(App[None]):
    """One rule per app variable, plus the stock widgets a real screen is made of."""

    CSS = "\n".join(
        f"#v-{name} {{ color: ${name}; background: ${name}; }}" for name in APP_VARIABLES
    )

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        for theme in textual_themes(config):
            self.register_theme(theme)
        self.theme = DEFAULT_THEME

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            for name in APP_VARIABLES:
                yield Static(name, id=f"v-{name}")
            yield Static(highlight(SAMPLE, language="python", theme=CodeHighlight), id="code")
            yield Markdown("# Title\n\nText with `code` and a [link](https://example.com).\n")
        yield Input(placeholder="prompt")
        yield OptionList("one", "two")
        yield Button("Run", variant="primary")
        yield Switch()
        yield Footer()


class WithDefaults(App[None]):
    CSS = "Static { color: $fg-muted; background: $bg-alt; }"

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return variable_defaults()

    def compose(self) -> ComposeResult:
        yield Static("x")


def test_app_variables_are_the_documented_names() -> None:
    assert APP_VARIABLES == (
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


def test_textual_theme_maps_tokens_onto_textual_fields() -> None:
    theme = get_theme("dracula")
    mapped = textual_theme(theme)
    assert mapped.name == "dracula"
    assert (mapped.primary, mapped.secondary, mapped.accent) == (
        theme.accent,
        theme.accent_alt,
        theme.accent,
    )
    assert (mapped.foreground, mapped.background) == (theme.fg, theme.bg)
    assert (mapped.surface, mapped.panel) == (theme.bg_alt, theme.bg_raised)
    assert (mapped.success, mapped.warning, mapped.error) == (
        theme.success,
        theme.warning,
        theme.error,
    )
    assert mapped.dark is True
    assert textual_theme(get_theme("github-light")).dark is False


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_every_theme_defines_every_app_variable(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    defined = textual_theme(theme).variables
    named = variables(theme)
    assert {name: defined[name] for name in APP_VARIABLES} == {
        name: named[name] for name in APP_VARIABLES
    }
    assert all(HEX.fullmatch(value) for value in defined.values())
    assert set(defined.values()) <= {*theme.colours().values(), theme.on_accent}


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_no_translucent_or_automatic_colour_survives(theme_id: str) -> None:
    mapped = textual_theme(builtin_themes()[theme_id])
    generated = {**mapped.to_color_system().generate(), **mapped.variables}
    automatic = {name: value for name, value in generated.items() if "auto" in value}
    with_opacity = {name: value for name, value in generated.items() if "%" in value}
    translucent = {
        name: value
        for name, value in generated.items()
        if re.fullmatch(r"#[0-9a-fA-F]{8}", value) and not value.endswith("00")
    }
    assert (automatic, with_opacity, translucent) == ({}, {}, {})


def test_chrome_is_pinned_to_flat_theme_tones() -> None:
    theme = get_theme("ayu-dark")
    pinned = textual_theme(theme).variables
    assert pinned["text"] == pinned["foreground"] == theme.fg
    assert (pinned["text-muted"], pinned["text-disabled"]) == (theme.fg_muted, theme.fg_subtle)
    assert (pinned["block-cursor-background"], pinned["block-cursor-foreground"]) == (
        theme.selection,
        theme.fg,
    )
    assert pinned["block-hover-background"] == pinned["block-cursor-blurred-background"]
    assert pinned["block-hover-background"] == theme.line
    assert pinned["scrollbar"] == theme.fg_subtle
    # One flat track: no change on hover or drag, and the same tone in the corner.
    assert {pinned[name] for name in pinned if name.startswith("scrollbar-background")} == {
        theme.bg
    }
    assert pinned["scrollbar-corner-color"] == theme.bg
    assert (pinned["link-color"], pinned["link-color-hover"]) == (theme.accent, theme.on_accent)
    assert pinned["footer-background"] == theme.bg
    assert pinned["button-color-foreground"] == theme.on_accent
    assert pinned["input-selection-background"] == theme.selection


#: Text and the fill Textual draws it on, by variable name, as pinned by `textual_theme`.
CHROME_PAIRS = (
    ("text", "background"),
    ("text-muted", "background"),
    ("foreground-muted", "background"),
    ("block-cursor-foreground", "block-cursor-background"),
    ("block-cursor-blurred-foreground", "block-cursor-blurred-background"),
    ("foreground", "block-hover-background"),
    ("foreground", "surface-active"),
    ("input-cursor-foreground", "input-cursor-background"),
    ("input-selection-foreground", "input-selection-background"),
    ("foreground", "screen-selection-background"),
    ("link-color", "background"),
    ("link-color-hover", "link-background-hover"),
    ("footer-foreground", "footer-background"),
    ("footer-key-foreground", "footer-background"),
    ("footer-description-foreground", "footer-background"),
    ("button-foreground", "surface"),
    ("button-color-foreground", "primary"),
    ("button-color-foreground", "success"),
    ("button-color-foreground", "warning"),
    ("button-color-foreground", "error"),
    ("markdown-h1-color", "background"),
    ("markdown-h6-color", "background"),
)


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_pinned_chrome_is_readable(theme_id: str) -> None:
    pinned = textual_theme(builtin_themes()[theme_id]).variables
    for text, fill in CHROME_PAIRS:
        assert contrast(pinned[text], pinned[fill]) >= TEXT_CONTRAST, (text, fill)
    # A scrollbar thumb is a control, not text: it has to stand out from its track.
    assert contrast(pinned["scrollbar"], pinned["scrollbar-background"]) >= QUIET_CONTRAST
    assert contrast(pinned["scrollbar-hover"], pinned["scrollbar-background"]) >= TEXT_CONTRAST


def test_textual_themes_lists_builtins_then_user_themes(config: Config) -> None:
    config.themes = {"mine": {"base": "nord", "accent": "#ebcb8b"}, "dracula": {"bg": "#000000"}}
    assert [theme.name for theme in textual_themes()] == list(BUILTIN_IDS)
    themes = textual_themes(config)
    assert [theme.name for theme in themes] == [*BUILTIN_IDS, "mine"]
    assert themes[-1].primary == "#ebcb8b"
    assert themes[-1].background == get_theme("nord").bg


def test_variable_defaults_are_the_default_themes_values() -> None:
    defaults = variable_defaults()
    assert tuple(defaults) == APP_VARIABLES
    assert defaults == {
        name: textual_theme(get_theme(DEFAULT_THEME)).variables[name] for name in APP_VARIABLES
    }


def test_code_highlight_names_theme_variables_only() -> None:
    styles = set(CodeHighlight.STYLES.values())
    variables_used = {style[1:] for style in styles if style.startswith("$")}
    assert variables_used <= {*APP_VARIABLES, "success", "error"}
    assert {"syn-keyword", "syn-string", "syn-number", "syn-comment", "syn-function"} <= (
        variables_used
    )
    assert styles - {f"${name}" for name in variables_used} == {"bold", "italic", "bold italic"}
    spans = {span.style for span in highlight(SAMPLE, language="python", theme=CodeHighlight).spans}
    assert {"$syn-keyword", "$syn-string", "$syn-number", "$syn-comment", "$syn-function"} <= spans


async def test_every_theme_registers_and_resolves_every_variable() -> None:
    app = Swatches()
    assert set(BUILTIN_IDS) <= set(app.available_themes)
    async with app.run_test() as pilot:
        for theme_id, theme in builtin_themes().items():
            app.theme = theme_id
            await pilot.pause()
            assert app.current_theme.name == theme_id
            named = variables(theme)
            for name in APP_VARIABLES:
                styles = app.query_one(f"#v-{name}").styles
                assert resolved(styles.color) == named[name], (theme_id, name)
                assert resolved(styles.background) == named[name], (theme_id, name)
            assert resolved(app.screen.styles.background) == theme.bg, theme_id
            assert resolved(app.styles.color) == theme.fg, theme_id
            assert resolved(app.query_one(Input).styles.background) == theme.bg_alt, theme_id
            assert resolved(app.query_one(Footer).styles.background) == theme.bg, theme_id
            scroller = app.query_one(VerticalScroll).styles
            assert resolved(scroller.scrollbar_color) == theme.fg_subtle, theme_id
            assert resolved(scroller.scrollbar_background) == theme.bg, theme_id
            assert resolved(scroller.scrollbar_background_hover) == theme.bg, theme_id


async def test_highlighted_code_renders_in_the_themes_syntax_colours() -> None:
    app = Swatches()
    async with app.run_test() as pilot:
        for theme_id in ("ayu-dark", "github-light"):
            app.theme = theme_id
            await pilot.pause()
            theme = get_theme(theme_id)
            code = app.query_one("#code", Static)
            painted = {
                segment.text.strip(): segment.style.color.get_truecolor().hex
                for strip in code.render_lines(code.size.region)
                for segment in strip
                if segment.style is not None and segment.style.color is not None
            }
            assert painted["def"] == theme.syntax.keyword
            assert painted["greet"] == theme.syntax.function
            assert painted["1"] == theme.syntax.number
            assert painted["# note"] == theme.syntax.comment
            assert painted["(name):"] == theme.fg


async def test_user_theme_is_registered_and_switchable(config: Config) -> None:
    config.themes = {
        "mine": {"base": "github-light", "accent": "#8250df", "syntax-keyword": "#0550ae"}
    }
    mine: Theme = all_themes(config)["mine"]
    app = Swatches(config)
    async with app.run_test() as pilot:
        app.theme = "mine"
        await pilot.pause()
        assert resolved(app.screen.styles.background) == mine.bg == "#ffffff"
        assert resolved(app.query_one("#v-syn-keyword").styles.color) == "#0550ae"
        assert not app.current_theme.dark


async def test_variable_defaults_keep_stock_textual_themes_usable() -> None:
    app = WithDefaults()
    async with app.run_test() as pilot:
        await pilot.pause()
        default = get_theme(DEFAULT_THEME)
        assert app.theme not in BUILTIN_IDS  # still on a stock Textual theme
        assert resolved(app.query_one(Static).styles.color) == default.fg_muted
        app.register_theme(textual_theme(get_theme("dracula")))
        app.theme = "dracula"
        await pilot.pause()
        assert resolved(app.query_one(Static).styles.color) == get_theme("dracula").fg_muted
        app.theme = "textual-light"
        await pilot.pause()
        assert resolved(app.query_one(Static).styles.color) == default.fg_muted
