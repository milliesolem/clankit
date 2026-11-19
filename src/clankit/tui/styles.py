"""The house style for stock Textual widgets, and the blank scrollbar.

Textual's stock widgets draw frames out of eighth-block and box-drawing characters (``tall``
borders on inputs and lists, ``hkey`` on the command palette, ``keyline: thin`` in Markdown
tables, a ``solid`` rule). The owner's brief rules all of those out: regions are separated
by flat surface tones and space only, and the only edge glyphs allowed are solid bars on the
left edge (``outer``/``thick``) and ``blank`` placeholders that keep text from shifting when
a bar appears. This stylesheet resets every stock widget the interface uses; Clankit's own
widgets carry their styles in their ``DEFAULT_CSS``.

Surface roles, in theme variables:

* canvas ``$background`` - the transcript and settings forms;
* chrome ``$bg-alt`` - crumb row, task tree panel, settings categories, status row;
* raised ``$bg-raised`` - user turns, request panels, the composer, inputs;
* well ``$bg-alt`` inside the canvas - code, diffs and tool output;
* select ``$selection`` - the cursor row; ``$line`` - hover and a blurred cursor.
"""

from __future__ import annotations

from typing import ClassVar

from textual.scrollbar import ScrollBar, ScrollBarRender


class BlankScrollBarRender(ScrollBarRender):
    """Scrollbar thumbs drawn as solid colour only.

    The stock renderer ends the thumb with partial-block glyphs (``▁``, ``▏`` ...) at
    fractional positions, which are exactly the hairlines the design forbids.
    """

    VERTICAL_BARS: ClassVar[list[str]] = [" "] * 8
    HORIZONTAL_BARS: ClassVar[list[str]] = [" "] * 8


def install_scrollbar() -> None:
    """Make every scrollbar in the process use `BlankScrollBarRender`."""
    ScrollBar.renderer = BlankScrollBarRender


APP_CSS = """
Screen {
    background: $background;
    color: $text;
    layers: base overlay;
}

* {
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
    scrollbar-color: $selection;
    scrollbar-color-hover: $fg-subtle;
    scrollbar-color-active: $accent;
    scrollbar-background: $background;
    scrollbar-background-hover: $background;
    scrollbar-background-active: $background;
    scrollbar-corner-color: $background;
}

Input {
    border: none;
    border-left: blank;
    height: 1;
    padding: 0 1;
    background: $bg-alt;
    color: $text;
}
Input:focus {
    border: none;
    border-left: outer $accent;
    background: $bg-alt;
    background-tint: 0%;
}
Input.-invalid, Input.-invalid:focus {
    border: none;
    border-left: outer $error;
}
Input > .input--placeholder, Input > .input--suggestion {
    color: $fg-subtle;
}

TextArea, TextArea:focus {
    border: none;
    background-tint: 0%;
}

OptionList, OptionList:focus {
    border: none;
    padding: 0;
    background: transparent;
    background-tint: 0%;
}
OptionList > .option-list--option {
    padding: 0 1;
}
OptionList > .option-list--option-highlighted {
    background: $line;
    color: $text;
    text-style: none;
}
OptionList:focus > .option-list--option-highlighted {
    background: $selection;
    color: $text;
    text-style: none;
}
OptionList > .option-list--option-disabled {
    color: $fg-muted;
    text-style: bold;
}
OptionList > .option-list--option-hover {
    background: $line;
}

Button, Button:focus, Button:hover, Button.-active {
    border: none;
    height: 1;
    min-width: 6;
    padding: 0 2;
    margin: 0 2 0 0;
    background: $selection;
    color: $text;
    text-style: none;
    background-tint: 0%;
}
Button:focus {
    background: $accent;
    color: $on-accent;
    text-style: bold;
}
Button:hover {
    background: $line;
}

LoadingIndicator {
    border: none;
    color: $accent;
    background: transparent;
}

Markdown {
    padding: 0;
    margin: 0;
    color: $text;
    link-color: $text;
    link-style: underline;
    link-background-hover: $line;
    link-color-hover: $text;
}
Markdown MarkdownBlock > .code_inline {
    background: $line;
    color: $text;
}
MarkdownH1, MarkdownH2, MarkdownH3, MarkdownH4, MarkdownH5, MarkdownH6 {
    content-align: left top;
    color: $text;
    background: transparent;
    text-style: bold;
    margin: 1 0 1 0;
}
MarkdownH3, MarkdownH4, MarkdownH5, MarkdownH6 {
    color: $fg-muted;
}
MarkdownHorizontalRule {
    border: none;
    height: 1;
    padding: 0;
    margin: 0 0 1 0;
}
MarkdownBlockQuote {
    background: transparent;
    border-left: outer $fg-subtle;
    padding: 0 1;
}
MarkdownBlockQuote > BlockQuote {
    border-left: outer $fg-subtle;
}
MarkdownFence {
    background: $bg-alt;
    color: $text;
    margin: 0 0 1 0;
}
MarkdownFence > Label {
    padding: 1 2;
}
MarkdownTable {
    background: transparent;
}
MarkdownTableContent {
    keyline: none;
    grid-gutter: 0 2;
}
MarkdownTableContent > .header {
    color: $text;
    text-style: bold;
}
MarkdownTableContent > .cell, MarkdownTableContent > .header {
    padding: 0;
}
MarkdownBullet {
    color: $fg-muted;
}

CommandPalette {
    background: $background;
}
CommandPalette > Vertical {
    background: $bg-raised;
}
CommandPalette #--input {
    border: none;
    padding: 1 0;
    background: $bg-raised;
}
CommandPalette #--input.--list-visible {
    border: none;
}
CommandPalette SearchIcon {
    display: none;
}
CommandPalette #--input Label {
    margin: 0 0 0 2;
}
CommandInput, CommandInput:focus {
    border: none;
    padding: 0 2;
    background: $bg-raised;
}
CommandList, CommandList:focus, CommandList.--populating {
    border: none;
    background: $bg-raised;
}
CommandPalette LoadingIndicator {
    border: none;
}
CommandPalette > .command-palette--help-text {
    color: $fg-muted;
    text-style: none;
}
CommandPalette > .command-palette--highlight {
    color: $accent;
    text-style: bold;
}

Toast {
    background: $bg-raised;
    border-left: outer $accent;
}
Toast.-warning {
    border-left: outer $warning;
}
Toast.-error {
    border-left: outer $error;
}

Tooltip {
    background: $bg-raised;
    color: $text;
    border: none;
}

.muted {
    color: $fg-muted;
}
"""
