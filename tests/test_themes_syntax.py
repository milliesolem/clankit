"""Code and terminal colours: the Pygments style, its theme-independent CSS, ANSI colours."""

from __future__ import annotations

import io
import re

import pytest
from pygments import highlight
from pygments.formatters import HtmlFormatter, Terminal256Formatter
from pygments.lexers import DiffLexer, PythonLexer
from pygments.token import STANDARD_TYPES, Comment, Generic, Keyword, Name, String, Token
from rich.color import ANSI_COLOR_NAMES
from rich.console import Console
from rich.syntax import Syntax

from clankit.themes import builtin_themes, get_theme
from clankit.themes.colour import contrast, luminance, rgb
from clankit.themes.legibility import QUIET_CONTRAST, TEXT_CONTRAST
from clankit.themes.syntax import ansi_palette, pygments_css, pygments_style, terminal_theme
from clankit.themes.web import css_variables

HEX = re.compile(r"#[0-9a-f]{6}")
BUILTIN_IDS = tuple(builtin_themes())

CODE = '''import os

@cached
class Greeter(Base):
    """Says hello."""

    def greet(self, name: str = "you") -> str:
        # A comment.
        return f"hi {name}\\n" * 2 if name is not None else os.sep
'''

DIFF = """--- a/x.py
+++ b/x.py
@@ -1,2 +1,2 @@
-old = 1
+new = 2
"""


def css_rules(css: str) -> dict[str, str]:
    """Pygments class -> declarations, from the flat rules `pygments_css` writes."""
    rules: dict[str, str] = {}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        for selector in selectors.split(","):
            css_class = selector.strip().rpartition(".")[2]
            rules[css_class] = (rules.get(css_class, "") + " " + body).strip()
    return rules


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_style_renders_with_html_and_terminal_formatters(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    style = pygments_style(theme)
    html = highlight(CODE, PythonLexer(), HtmlFormatter(style=style, noclasses=True)).lower()
    assert f"color: {theme.syntax.keyword}" in html
    assert f"color: {theme.syntax.string}" in html
    assert f"background: {theme.bg}" in html
    ansi = highlight(CODE, PythonLexer(), Terminal256Formatter(style=style))
    assert "\x1b[38;5;" in ansi and "greet" in ansi


def test_style_carries_the_theme_and_nothing_else() -> None:
    theme = get_theme("ayu-dark")
    style = pygments_style(theme)
    assert style.name == "clankit-ayu-dark"
    assert style.background_color == theme.bg
    assert style.highlight_color == theme.line
    assert style.line_number_color == theme.fg_subtle
    assert style.line_number_background_color == theme.bg
    used = {f"#{entry['color']}" for _, entry in style if entry["color"]}
    assert used <= set(theme.colours().values())
    assert not any(entry["bgcolor"] or entry["border"] for _, entry in style)


def test_style_maps_token_types_to_syntax_tokens() -> None:
    theme = get_theme("dracula")
    style = pygments_style(theme)

    def colour(token: object) -> str:
        return "#" + style.style_for_token(token)["color"]

    assert colour(Keyword) == colour(Keyword.Namespace) == theme.syntax.keyword
    assert colour(String) == colour(String.Double) == theme.syntax.string
    assert colour(Token.Literal.Number.Integer) == theme.syntax.number
    assert colour(Keyword.Constant) == theme.syntax.number
    assert colour(Comment.Single) == theme.syntax.comment
    assert colour(Name.Function) == colour(Name.Decorator) == theme.syntax.function
    assert colour(Name.Class) == colour(Keyword.Type) == theme.syntax.type
    assert colour(Name.Variable.Instance) == theme.syntax.variable
    assert colour(Token.Operator) == theme.syntax.operator
    assert colour(Name.Tag) == theme.syntax.tag
    assert colour(Name.Attribute) == theme.syntax.attribute
    assert colour(Name) == colour(Token.Punctuation) == colour(Token.Error) == theme.fg
    assert colour(Generic.Inserted) == theme.success
    assert colour(Generic.Deleted) == theme.error


def test_typography_is_plain_except_for_emphasis() -> None:
    style = pygments_style(get_theme("monokai-pro"))
    for token in (Keyword, Comment, String, Name.Function, Name.Class, Generic.Subheading):
        entry = style.style_for_token(token)
        assert not (entry["bold"] or entry["italic"] or entry["underline"]), token
    assert style.style_for_token(Generic.Strong)["bold"]
    assert style.style_for_token(Generic.Emph)["italic"]
    assert style.style_for_token(Generic.Heading)["bold"]
    both = style.style_for_token(Generic.EmphStrong)
    assert both["bold"] and both["italic"]


def test_style_is_built_once_per_theme() -> None:
    assert pygments_style(get_theme("nord")) is pygments_style(get_theme("nord"))
    assert pygments_style(get_theme("nord")) is not pygments_style(get_theme("dracula"))


def test_style_works_as_a_rich_syntax_theme() -> None:
    theme = get_theme("github-dark")
    console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor", width=80)
    console.print(Syntax(CODE, "python", theme=pygments_style(theme)))
    output = console.file.getvalue()
    red, green, blue = rgb(theme.syntax.keyword)
    assert f"38;2;{red};{green};{blue}" in output
    red, green, blue = rgb(theme.bg)
    assert f"48;2;{red};{green};{blue}" in output


def test_css_references_theme_variables_and_no_colours() -> None:
    css = pygments_css()
    names = set(re.findall(r"var\(--([a-z-]+)\)", css))
    assert names
    assert all(
        name.startswith("syn-") or name.startswith("fg") or name in ("success", "error")
        for name in names
    )
    assert {f"--{name}" for name in names} <= set(css_variables(get_theme("ayu-dark")))
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css)
    assert "background" not in css
    assert css.count("{") == css.count("}") == css.count("\n")
    assert css.startswith(".highlight { color: var(--fg); }\n")


def test_css_selector_is_configurable() -> None:
    css = pygments_css("pre.code")
    assert css.startswith("pre.code { color: var(--fg); }\n")
    assert "pre.code .k," in css or "pre.code .k {" in css
    assert ".highlight" not in css


@pytest.mark.parametrize("theme_id", ["ayu-dark", "monokai-pro", "dracula", "github-light"])
def test_css_and_style_agree_on_every_token_type(theme_id: str) -> None:
    theme = get_theme(theme_id)
    style = pygments_style(theme)
    properties = css_variables(theme)
    rules = css_rules(pygments_css())
    for token, css_class in STANDARD_TYPES.items():
        if not css_class:
            continue
        entry = style.style_for_token(token)
        declarations = rules.get(css_class, "")
        variable = re.search(r"color: var\((--[a-z-]+)\)", declarations)
        from_css = properties[variable[1]] if variable else theme.fg
        assert from_css == f"#{entry['color']}", (token, declarations)
        assert ("font-weight: bold" in declarations) == entry["bold"], token
        assert ("font-style: italic" in declarations) == entry["italic"], token


def test_css_covers_the_classes_pygments_emits() -> None:
    rules = css_rules(pygments_css())
    formatter = HtmlFormatter(cssclass="highlight")
    html = highlight(CODE, PythonLexer(), formatter) + highlight(DIFF, DiffLexer(), formatter)
    emitted = set(re.findall(r'<span class="([a-z0-9]+)">', html))
    assert {"k", "kn", "nf", "nc", "nd", "s2", "sd", "mi", "c1", "gi", "gd", "gu"} <= emitted
    assert "var(--syn-keyword)" in rules["k"] and "var(--syn-keyword)" in rules["kn"]
    assert "var(--syn-function)" in rules["nf"] and "var(--syn-function)" in rules["nd"]
    assert "var(--syn-type)" in rules["nc"]
    assert "var(--syn-string)" in rules["s2"] and "var(--syn-string)" in rules["sd"]
    assert "var(--syn-number)" in rules["mi"]
    assert "var(--syn-comment)" in rules["c1"]
    assert "var(--success)" in rules["gi"] and "var(--error)" in rules["gd"]
    # Whatever has no rule of its own is plain body text.
    assert {"n", "p", "w"} & emitted and not {"n", "p", "w"} & set(rules)


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_ansi_palette_is_complete_and_readable(theme_id: str) -> None:
    theme = builtin_themes()[theme_id]
    palette = ansi_palette(theme)
    assert list(palette) == list(ANSI_COLOR_NAMES)[:16]
    assert all(HEX.fullmatch(colour) for colour in palette.values())
    hues = ("red", "green", "yellow", "blue", "magenta", "cyan")
    for name in hues:
        assert contrast(palette[name], theme.bg) >= TEXT_CONTRAST, name
        assert palette[f"bright_{name}"] == palette[name]
    for name in ("white", "bright_black", "bright_white"):
        assert contrast(palette[name], theme.bg) >= QUIET_CONTRAST, name
    neutrals = [palette[name] for name in ("black", "bright_black", "white", "bright_white")]
    assert neutrals == sorted(neutrals, key=luminance)


def test_ansi_palette_uses_the_themes_own_tones() -> None:
    dark, light = get_theme("dracula"), get_theme("github-light")
    assert ansi_palette(dark)["red"] == dark.red == "#ff5555"
    assert ansi_palette(dark)["magenta"] == dark.purple
    assert (ansi_palette(dark)["black"], ansi_palette(dark)["white"]) == (dark.bg_raised, dark.fg)
    assert ansi_palette(dark)["bright_black"] == dark.fg_subtle
    assert (ansi_palette(light)["black"], ansi_palette(light)["white"]) == (
        light.fg,
        light.fg_subtle,
    )


def test_ansi_hue_that_cannot_be_read_is_adjusted() -> None:
    kanagawa = get_theme("kanagawa")
    assert contrast(kanagawa.red, kanagawa.bg) < TEXT_CONTRAST
    assert ansi_palette(kanagawa)["red"] != kanagawa.red
    assert contrast(ansi_palette(kanagawa)["red"], kanagawa.bg) >= TEXT_CONTRAST


def test_terminal_theme_wraps_the_palette_for_rich() -> None:
    theme = get_theme("ayu-dark")
    palette = list(ansi_palette(theme).values())
    wrapped = terminal_theme(theme)
    assert tuple(wrapped.background_color) == rgb(theme.bg)
    assert tuple(wrapped.foreground_color) == rgb(theme.fg)
    assert [tuple(colour) for colour in wrapped.ansi_colors] == [rgb(colour) for colour in palette]
