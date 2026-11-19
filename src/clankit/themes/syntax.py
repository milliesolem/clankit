"""Code and terminal colours derived from a theme.

Every renderer shares one mapping from Pygments token types to theme tokens (`TOKEN_COLOURS`),
so code looks the same wherever it is shown:

* `pygments_style` builds a Pygments style class, for Rich and for any Pygments formatter;
* `pygments_css` writes the stylesheet for Pygments' HTML output in terms of the CSS variables
  from `clankit.themes.web`. It contains no colours of its own, so the web UI serves it once
  and it follows every theme;
* `clankit.themes.textual.CodeHighlight` does the same for Textual's own highlighter;
* `ansi_palette` and `terminal_theme` give the 16 ANSI colours for rendering the output of
  shell commands.

Typography is deliberately plain: no bold keywords and no italic comments. Weight and slant are
used only where the source text itself asks for emphasis (`TOKEN_FONTS`).
"""

from __future__ import annotations

import functools

from pygments.style import Style, StyleMeta
from pygments.token import (
    STANDARD_TYPES,
    Comment,
    Generic,
    Keyword,
    Literal,
    Name,
    Number,
    Operator,
    String,
    Token,
    _TokenType,
)
from rich.terminal_theme import TerminalTheme

from clankit.themes.colour import rgb
from clankit.themes.legibility import readable
from clankit.themes.model import Theme, variable_name

#: The token that colours code no rule below claims: plain body text.
BODY = "fg"

#: Which theme token colours which kind of source text. A token type that is not listed takes
#: the colour of its nearest listed ancestor, which makes `Token` the fallback for everything.
TOKEN_COLOURS: dict[_TokenType, str] = {
    Token: BODY,
    Comment: "syntax_comment",
    Comment.Preproc: "syntax_keyword",
    Comment.PreprocFile: "syntax_string",
    Keyword: "syntax_keyword",
    Keyword.Constant: "syntax_number",
    Keyword.Type: "syntax_type",
    Operator: "syntax_operator",
    Operator.Word: "syntax_keyword",
    Name.Attribute: "syntax_attribute",
    Name.Builtin: "syntax_function",
    Name.Builtin.Pseudo: "syntax_variable",
    Name.Class: "syntax_type",
    Name.Constant: "syntax_number",
    Name.Decorator: "syntax_function",
    Name.Entity: "syntax_number",
    Name.Exception: "syntax_type",
    Name.Function: "syntax_function",
    Name.Property: "syntax_attribute",
    Name.Tag: "syntax_tag",
    Name.Variable: "syntax_variable",
    Literal: "syntax_number",
    String: "syntax_string",
    String.Escape: "syntax_number",
    String.Interpol: "syntax_operator",
    String.Symbol: "syntax_number",
    Number: "syntax_number",
    # Diffs and console sessions carry meaning rather than syntax, so they use status colours.
    Generic.Inserted: "success",
    Generic.Deleted: "error",
    Generic.Error: "error",
    Generic.Traceback: "error",
    Generic.Subheading: "syntax_keyword",
    Generic.Output: "fg_muted",
    Generic.Prompt: "fg_muted",
}

#: The only token types set in bold or italic. Inherited like colours.
TOKEN_FONTS: dict[_TokenType, tuple[str, ...]] = {
    Generic.Heading: ("bold",),
    Generic.Strong: ("bold",),
    Generic.Emph: ("italic",),
    Generic.EmphStrong: ("bold", "italic"),
}

_FONT_CSS = {"bold": "font-weight: bold;", "italic": "font-style: italic;"}

_ANSI_HUES = ("red", "green", "yellow", "blue", "magenta", "cyan")


@functools.lru_cache(maxsize=64)
def pygments_style(theme: Theme) -> type[Style]:
    """A Pygments style class for `theme`, e.g. for ``rich.syntax.Syntax(..., theme=...)``.

    Its background is `bg`, the surface the syntax colours are guaranteed readable on; pass
    another surface to the renderer if code blocks sit elsewhere.
    """
    colours = theme.colours()
    styles = {token: colours[role] for token, role in TOKEN_COLOURS.items()}
    styles.update({token: " ".join(fonts) for token, fonts in TOKEN_FONTS.items()})
    attributes = {
        "__module__": __name__,
        "name": f"clankit-{theme.id}",
        "background_color": theme.bg,
        "highlight_color": theme.line,
        "line_number_color": theme.fg_subtle,
        "line_number_background_color": theme.bg,
        "line_number_special_color": theme.fg,
        "line_number_special_background_color": theme.line,
        "styles": styles,
    }
    return StyleMeta("ClankitStyle", (Style,), attributes)


def pygments_css(selector: str = ".highlight") -> str:
    """CSS for Pygments' HTML output (``HtmlFormatter(cssclass="highlight")``), for any theme.

    Colours are ``var(--syn-*)`` and ``var(--fg*)``, plus ``var(--success)`` and
    ``var(--error)`` for diffs and errors. The background is left to the page.
    """
    rules: dict[str, list[str]] = {}
    for token, css_class in STANDARD_TYPES.items():
        if not css_class:
            continue
        role = _inherited(TOKEN_COLOURS, token, BODY)
        declarations = [] if role == BODY else [f"color: var(--{variable_name(role)});"]
        declarations.extend(_FONT_CSS[font] for font in _inherited(TOKEN_FONTS, token, ()))
        if declarations:
            rules.setdefault(" ".join(declarations), []).append(f"{selector} .{css_class}")
    lines = [f"{selector} {{ color: var(--{variable_name(BODY)}); }}"]
    lines.extend(f"{', '.join(targets)} {{ {body} }}" for body, targets in rules.items())
    return "\n".join(lines) + "\n"


def ansi_palette(theme: Theme) -> dict[str, str]:
    """The 16 ANSI colours for `theme`, by Rich's names, in ANSI order (``black`` first).

    Command output is text someone has to read, so the six hues are the theme's own hue tokens
    brought up to 4.5:1 on `bg` where they fall short, and the bright variants repeat them.
    The four neutrals keep their order from darkest to lightest but are drawn from the theme's
    tones, so that neither "black" on a dark theme nor "white" on a light one disappears.
    """
    hue_tokens = {**theme.colours(), "magenta": theme.purple}
    hues = {name: readable(hue_tokens[name], [theme.bg]) for name in _ANSI_HUES}
    if theme.dark:
        black, bright_black, white, bright_white = (
            theme.bg_raised,
            theme.fg_subtle,
            theme.fg,
            theme.fg,
        )
    else:
        black, bright_black, white, bright_white = (
            theme.fg,
            theme.fg_muted,
            theme.fg_subtle,
            theme.fg_subtle,
        )
    return {
        "black": black,
        **hues,
        "white": white,
        "bright_black": bright_black,
        **{f"bright_{name}": colour for name, colour in hues.items()},
        "bright_white": bright_white,
    }


def terminal_theme(theme: Theme) -> TerminalTheme:
    """`ansi_palette` as a Rich terminal theme.

    Textual converts ANSI-coloured text with ``App.ansi_theme_dark`` / ``ansi_theme_light``;
    assigning this to both keeps command output inside the theme.
    """
    colours = [rgb(colour) for colour in ansi_palette(theme).values()]
    return TerminalTheme(rgb(theme.bg), rgb(theme.fg), colours[:8], colours[8:])


def _inherited[T](table: dict[_TokenType, T], token: _TokenType, default: T) -> T:
    """The entry for `token` or for its nearest ancestor that has one, else `default`."""
    while token not in table:
        if token.parent is None:
            return default
        token = token.parent
    return table[token]
