"""The built-in palettes, exactly as their authors publish them.

Every colour below was read from the upstream source named in the palette's `source`, by the
palette research of 2026-11-19 and its fact-check; nothing here is tuned by hand. Where an
upstream theme defines a tone with transparency, the value is that tone composited over the
theme's own background. Clankit's adjustments for legibility happen when a `Theme` is built
(see `clankit.themes.legibility`) and never touch this table, so it can be compared against
upstream line by line.

Upstream themes are written for editors, so each token is the closest official colour for its
role; the comments mark the places where that was a judgement call.

The order is the order themes are offered in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    """One upstream colour scheme: a display name, where it came from, and its colours.

    `colours` has exactly the keys in `clankit.themes.TOKENS`.
    """

    name: str
    source: str
    colours: dict[str, str]


AYU = "ayu 9.1.0 (npm package ayu), roles as in vscode-ayu 1.6.0"
MONOKAI_PRO = "Monokai Pro for VS Code 2.0.17 (monokai.theme-monokai-pro-vscode)"
DRACULA = "Dracula specification (draculatheme.com/spec), read 2026-11-19"
GITHUB = "GitHub VS Code theme 6.3.5 (primer/github-vscode-theme)"
CATPPUCCIN = "Catppuccin palette 1.8.0 and style guide (catppuccin/palette)"
TOKYO_NIGHT = "Tokyo Night VS Code theme (tokyo-night/tokyo-night-vscode-theme), read 2026-11-19"
GRUVBOX = "gruvbox.vim, medium contrast (morhetz/gruvbox), read 2026-11-19"
NORD = "Nord palette and VS Code port (nordtheme/nord), read 2026-11-19"
ONE_DARK = "Atom One Dark syntax and UI themes (atom/atom), read 2026-11-19"
SOLARIZED = "Solarized (altercation/solarized), read 2026-11-19"
ROSE_PINE = "Rosé Pine palette, main variant (rose-pine/palette), read 2026-11-19"
KANAGAWA = "kanagawa.nvim, wave variant (rebelot/kanagawa.nvim), read 2026-11-19"
EVERFOREST = "Everforest, dark medium (sainnhe/everforest), read 2026-11-19"

# fmt: off
PALETTES: dict[str, Palette] = {
    # bg is the editor surface and bg_alt the chrome of the port's default (bordered) look.
    # Surfaces are Ayu 9's; Ayu 8 used #0b0e14, #0d1017 and #0f131a.
    "ayu-dark": Palette(
        "Ayu Dark", AYU,
        {
            "bg": "#10141c", "bg_alt": "#0d1017", "bg_raised": "#141821",
            "selection": "#172e51", "line": "#161a24",
            "fg": "#bfbdb6", "fg_muted": "#5a6378", "fg_subtle": "#5a6673",
            "accent": "#e6b450", "accent_alt": "#59c2ff",
            "success": "#70bf56", "warning": "#ffb454", "error": "#d95757", "info": "#73b8ff",
            "red": "#f07178", "orange": "#ff8f40", "yellow": "#ffb454", "green": "#aad94c",
            "cyan": "#95e6cb", "blue": "#59c2ff", "purple": "#d2a6ff",
            "syntax_keyword": "#ff8f40", "syntax_string": "#aad94c", "syntax_number": "#d2a6ff",
            "syntax_comment": "#5a6673", "syntax_function": "#ffb454", "syntax_type": "#59c2ff",
            "syntax_variable": "#bfbdb6", "syntax_operator": "#f29668", "syntax_tag": "#39bae6",
            "syntax_attribute": "#ffb454",
        },
    ),
    "ayu-mirage": Palette(
        "Ayu Mirage", AYU,
        {
            "bg": "#242936", "bg_alt": "#1f2430", "bg_raised": "#282e3b",
            "selection": "#274364", "line": "#1a1f29",
            "fg": "#cccac2", "fg_muted": "#707a8c", "fg_subtle": "#6e7c8f",
            "accent": "#ffcc66", "accent_alt": "#73d0ff",
            "success": "#87d96c", "warning": "#ffcd66", "error": "#ff6666", "info": "#80bfff",
            "red": "#f28779", "orange": "#ffa659", "yellow": "#ffcd66", "green": "#d5ff80",
            "cyan": "#95e6cb", "blue": "#73d0ff", "purple": "#dfbfff",
            "syntax_keyword": "#ffa659", "syntax_string": "#d5ff80", "syntax_number": "#dfbfff",
            "syntax_comment": "#6e7c8f", "syntax_function": "#ffcd66", "syntax_type": "#73d0ff",
            "syntax_variable": "#cccac2", "syntax_operator": "#f29e74", "syntax_tag": "#5ccfe6",
            "syntax_attribute": "#ffcd66",
        },
    ),
    "ayu-light": Palette(
        "Ayu Light", AYU,
        {
            "bg": "#fcfcfc", "bg_alt": "#f8f9fa", "bg_raised": "#ffffff",
            "selection": "#d3e1f5", "line": "#eceef1",
            "fg": "#5c6166", "fg_muted": "#828e9f", "fg_subtle": "#adaeb1",
            "accent": "#f29718", "accent_alt": "#22a4e6",
            "success": "#6cbf43", "warning": "#eba400", "error": "#e65050", "info": "#478acc",
            "red": "#f07171", "orange": "#fa8532", "yellow": "#eba400", "green": "#86b300",
            "cyan": "#4cbf99", "blue": "#22a4e6", "purple": "#a37acc",
            "syntax_keyword": "#fa8532", "syntax_string": "#86b300", "syntax_number": "#a37acc",
            "syntax_comment": "#adaeb1", "syntax_function": "#eba400", "syntax_type": "#22a4e6",
            "syntax_variable": "#5c6166", "syntax_operator": "#f2a191", "syntax_tag": "#55b4d4",
            "syntax_attribute": "#eba400",
        },
    ),
    # Monokai Pro has no blue (its terminal blue is the orange accent): blue is the cyan.
    "monokai-pro": Palette(
        "Monokai Pro", MONOKAI_PRO,
        {
            "bg": "#2d2a2e", "bg_alt": "#221f22", "bg_raised": "#403e41",
            "selection": "#434044", "line": "#373438",
            "fg": "#fcfcfa", "fg_muted": "#939293", "fg_subtle": "#727072",
            "accent": "#ffd866", "accent_alt": "#78dce8",
            "success": "#a9dc76", "warning": "#fc9867", "error": "#ff6188", "info": "#78dce8",
            "red": "#ff6188", "orange": "#fc9867", "yellow": "#ffd866", "green": "#a9dc76",
            "cyan": "#78dce8", "blue": "#78dce8", "purple": "#ab9df2",
            "syntax_keyword": "#ff6188", "syntax_string": "#ffd866", "syntax_number": "#ab9df2",
            "syntax_comment": "#727072", "syntax_function": "#a9dc76", "syntax_type": "#78dce8",
            "syntax_variable": "#fcfcfa", "syntax_operator": "#ff6188", "syntax_tag": "#ff6188",
            "syntax_attribute": "#78dce8",
        },
    ),
    "monokai-pro-classic": Palette(
        "Monokai Classic", MONOKAI_PRO,
        {
            "bg": "#272822", "bg_alt": "#1d1e19", "bg_raised": "#3b3c35",
            "selection": "#3e3f38", "line": "#31322c",
            "fg": "#fdfff1", "fg_muted": "#919288", "fg_subtle": "#6e7066",
            "accent": "#e6db74", "accent_alt": "#66d9ef",
            "success": "#a6e22e", "warning": "#fd971f", "error": "#f92672", "info": "#66d9ef",
            "red": "#f92672", "orange": "#fd971f", "yellow": "#e6db74", "green": "#a6e22e",
            "cyan": "#66d9ef", "blue": "#66d9ef", "purple": "#ae81ff",
            "syntax_keyword": "#f92672", "syntax_string": "#e6db74", "syntax_number": "#ae81ff",
            "syntax_comment": "#6e7066", "syntax_function": "#a6e22e", "syntax_type": "#66d9ef",
            "syntax_variable": "#fdfff1", "syntax_operator": "#f92672", "syntax_tag": "#f92672",
            "syntax_attribute": "#66d9ef",
        },
    ),
    "monokai-pro-machine": Palette(
        "Monokai Pro (Filter Machine)", MONOKAI_PRO,
        {
            "bg": "#273136", "bg_alt": "#1d2528", "bg_raised": "#3a4449",
            "selection": "#3d474b", "line": "#313b3f",
            "fg": "#f2fffc", "fg_muted": "#8b9798", "fg_subtle": "#6b7678",
            "accent": "#ffed72", "accent_alt": "#7cd5f1",
            "success": "#a2e57b", "warning": "#ffb270", "error": "#ff6d7e", "info": "#7cd5f1",
            "red": "#ff6d7e", "orange": "#ffb270", "yellow": "#ffed72", "green": "#a2e57b",
            "cyan": "#7cd5f1", "blue": "#7cd5f1", "purple": "#baa0f8",
            "syntax_keyword": "#ff6d7e", "syntax_string": "#ffed72", "syntax_number": "#baa0f8",
            "syntax_comment": "#6b7678", "syntax_function": "#a2e57b", "syntax_type": "#7cd5f1",
            "syntax_variable": "#f2fffc", "syntax_operator": "#ff6d7e", "syntax_tag": "#ff6d7e",
            "syntax_attribute": "#7cd5f1",
        },
    ),
    "monokai-pro-octagon": Palette(
        "Monokai Pro (Filter Octagon)", MONOKAI_PRO,
        {
            "bg": "#282a3a", "bg_alt": "#1e1f2b", "bg_raised": "#3a3d4b",
            "selection": "#3d3f4e", "line": "#313343",
            "fg": "#eaf2f1", "fg_muted": "#888d94", "fg_subtle": "#696d77",
            "accent": "#ffd76d", "accent_alt": "#9cd1bb",
            "success": "#bad761", "warning": "#ff9b5e", "error": "#ff657a", "info": "#9cd1bb",
            "red": "#ff657a", "orange": "#ff9b5e", "yellow": "#ffd76d", "green": "#bad761",
            "cyan": "#9cd1bb", "blue": "#9cd1bb", "purple": "#c39ac9",
            "syntax_keyword": "#ff657a", "syntax_string": "#ffd76d", "syntax_number": "#c39ac9",
            "syntax_comment": "#696d77", "syntax_function": "#bad761", "syntax_type": "#9cd1bb",
            "syntax_variable": "#eaf2f1", "syntax_operator": "#ff657a", "syntax_tag": "#ff657a",
            "syntax_attribute": "#9cd1bb",
        },
    ),
    "monokai-pro-ristretto": Palette(
        "Monokai Pro (Filter Ristretto)", MONOKAI_PRO,
        {
            "bg": "#2c2525", "bg_alt": "#211c1c", "bg_raised": "#403838",
            "selection": "#433b3b", "line": "#362f2f",
            "fg": "#fff1f3", "fg_muted": "#948a8b", "fg_subtle": "#72696a",
            "accent": "#f9cc6c", "accent_alt": "#85dacc",
            "success": "#adda78", "warning": "#f38d70", "error": "#fd6883", "info": "#85dacc",
            "red": "#fd6883", "orange": "#f38d70", "yellow": "#f9cc6c", "green": "#adda78",
            "cyan": "#85dacc", "blue": "#85dacc", "purple": "#a8a9eb",
            "syntax_keyword": "#fd6883", "syntax_string": "#f9cc6c", "syntax_number": "#a8a9eb",
            "syntax_comment": "#72696a", "syntax_function": "#adda78", "syntax_type": "#85dacc",
            "syntax_variable": "#fff1f3", "syntax_operator": "#fd6883", "syntax_tag": "#fd6883",
            "syntax_attribute": "#85dacc",
        },
    ),
    "monokai-pro-spectrum": Palette(
        "Monokai Pro (Filter Spectrum)", MONOKAI_PRO,
        {
            "bg": "#222222", "bg_alt": "#191919", "bg_raised": "#363537",
            "selection": "#39383a", "line": "#2c2c2c",
            "fg": "#f7f1ff", "fg_muted": "#8b888f", "fg_subtle": "#69676c",
            "accent": "#fce566", "accent_alt": "#5ad4e6",
            "success": "#7bd88f", "warning": "#fd9353", "error": "#fc618d", "info": "#5ad4e6",
            "red": "#fc618d", "orange": "#fd9353", "yellow": "#fce566", "green": "#7bd88f",
            "cyan": "#5ad4e6", "blue": "#5ad4e6", "purple": "#948ae3",
            "syntax_keyword": "#fc618d", "syntax_string": "#fce566", "syntax_number": "#948ae3",
            "syntax_comment": "#69676c", "syntax_function": "#7bd88f", "syntax_type": "#5ad4e6",
            "syntax_variable": "#f7f1ff", "syntax_operator": "#fc618d", "syntax_tag": "#fc618d",
            "syntax_attribute": "#5ad4e6",
        },
    ),
    # Upstream uses the red as the light variant's accent, so accent equals error.
    "monokai-pro-light": Palette(
        "Monokai Pro Light", MONOKAI_PRO,
        {
            "bg": "#faf4f2", "bg_alt": "#ede7e5", "bg_raised": "#fefaf9",
            "selection": "#e5e0de", "line": "#f0eae9",
            "fg": "#29242a", "fg_muted": "#706b6e", "fg_subtle": "#a59fa0",
            "accent": "#e14775", "accent_alt": "#1c8ca8",
            "success": "#269d69", "warning": "#e16032", "error": "#e14775", "info": "#1c8ca8",
            "red": "#e14775", "orange": "#e16032", "yellow": "#cc7a0a", "green": "#269d69",
            "cyan": "#1c8ca8", "blue": "#1c8ca8", "purple": "#7058be",
            "syntax_keyword": "#e14775", "syntax_string": "#cc7a0a", "syntax_number": "#7058be",
            "syntax_comment": "#a59fa0", "syntax_function": "#269d69", "syntax_type": "#1c8ca8",
            "syntax_variable": "#29242a", "syntax_operator": "#e14775", "syntax_tag": "#e14775",
            "syntax_attribute": "#1c8ca8",
        },
    ),
    # Numbers are purple, as both official implementations render them; the spec prose says orange.
    # Dracula has no secondary text and no blue: fg_muted is Comment, blue is ANSI blue (Purple).
    # Status colours are the syntax palette's; the spec's UI-only 'functional' set is darker.
    "dracula": Palette(
        "Dracula", DRACULA,
        {
            "bg": "#282a36", "bg_alt": "#21222c", "bg_raised": "#343746",
            "selection": "#44475a", "line": "#353747",
            "fg": "#f8f8f2", "fg_muted": "#6272a4", "fg_subtle": "#6272a4",
            "accent": "#bd93f9", "accent_alt": "#ff79c6",
            "success": "#50fa7b", "warning": "#f1fa8c", "error": "#ff5555", "info": "#8be9fd",
            "red": "#ff5555", "orange": "#ffb86c", "yellow": "#f1fa8c", "green": "#50fa7b",
            "cyan": "#8be9fd", "blue": "#bd93f9", "purple": "#ff79c6",
            "syntax_keyword": "#ff79c6", "syntax_string": "#f1fa8c", "syntax_number": "#bd93f9",
            "syntax_comment": "#6272a4", "syntax_function": "#50fa7b", "syntax_type": "#8be9fd",
            "syntax_variable": "#f8f8f2", "syntax_operator": "#ff79c6", "syntax_tag": "#ff79c6",
            "syntax_attribute": "#50fa7b",
        },
    ),
    # The frozen VS Code theme, what editors call GitHub Dark; github.com has since drifted.
    # selection is the 20% accent tint the theme's source intends (the shipped JSON sets none).
    # line is border.muted: the theme's own current-line tint flattens onto bg_raised.
    "github-dark": Palette(
        "GitHub Dark", GITHUB,
        {
            "bg": "#0d1117", "bg_alt": "#010409", "bg_raised": "#161b22",
            "selection": "#142744", "line": "#21262d",
            "fg": "#e6edf3", "fg_muted": "#7d8590", "fg_subtle": "#6e7681",
            "accent": "#2f81f7", "accent_alt": "#f78166",
            "success": "#3fb950", "warning": "#d29922", "error": "#f85149", "info": "#2f81f7",
            "red": "#ff7b72", "orange": "#db6d28", "yellow": "#d29922", "green": "#3fb950",
            "cyan": "#39c5cf", "blue": "#58a6ff", "purple": "#bc8cff",
            "syntax_keyword": "#ff7b72", "syntax_string": "#a5d6ff", "syntax_number": "#79c0ff",
            "syntax_comment": "#8b949e", "syntax_function": "#d2a8ff", "syntax_type": "#ffa657",
            "syntax_variable": "#e6edf3", "syntax_operator": "#ff7b72", "syntax_tag": "#7ee787",
            "syntax_attribute": "#79c0ff",
        },
    ),
    "github-dark-dimmed": Palette(
        "GitHub Dark Dimmed", GITHUB,
        {
            "bg": "#22272e", "bg_alt": "#1c2128", "bg_raised": "#2d333b",
            "selection": "#2c3e56", "line": "#373e47",
            "fg": "#adbac7", "fg_muted": "#768390", "fg_subtle": "#636e7b",
            "accent": "#539bf5", "accent_alt": "#ec775c",
            "success": "#57ab5a", "warning": "#c69026", "error": "#e5534b", "info": "#539bf5",
            "red": "#f47067", "orange": "#cc6b2c", "yellow": "#c69026", "green": "#57ab5a",
            "cyan": "#39c5cf", "blue": "#539bf5", "purple": "#b083f0",
            "syntax_keyword": "#f47067", "syntax_string": "#96d0ff", "syntax_number": "#6cb6ff",
            "syntax_comment": "#768390", "syntax_function": "#dcbdfb", "syntax_type": "#f69d50",
            "syntax_variable": "#adbac7", "syntax_operator": "#f47067", "syntax_tag": "#8ddb8c",
            "syntax_attribute": "#6cb6ff",
        },
    ),
    # bg_raised is gray-1: the official overlay tone is #ffffff, the same as bg.
    "github-light": Palette(
        "GitHub Light", GITHUB,
        {
            "bg": "#ffffff", "bg_alt": "#f6f8fa", "bg_raised": "#eaeef2",
            "selection": "#cee1f8", "line": "#d8dee4",
            "fg": "#1f2328", "fg_muted": "#656d76", "fg_subtle": "#6e7781",
            "accent": "#0969da", "accent_alt": "#fd8c73",
            "success": "#1a7f37", "warning": "#9a6700", "error": "#cf222e", "info": "#0969da",
            "red": "#cf222e", "orange": "#bc4c00", "yellow": "#9a6700", "green": "#116329",
            "cyan": "#1b7c83", "blue": "#0969da", "purple": "#8250df",
            "syntax_keyword": "#cf222e", "syntax_string": "#0a3069", "syntax_number": "#0550ae",
            "syntax_comment": "#6e7781", "syntax_function": "#8250df", "syntax_type": "#953800",
            "syntax_variable": "#1f2328", "syntax_operator": "#cf222e", "syntax_tag": "#116329",
            "syntax_attribute": "#0550ae",
        },
    ),
    # Accent is Mauve, the default accent of the official ports (a user choice upstream).
    "catppuccin-mocha": Palette(
        "Catppuccin Mocha", CATPPUCCIN,
        {
            "bg": "#1e1e2e", "bg_alt": "#181825", "bg_raised": "#313244",
            "selection": "#3b3d4f", "line": "#303042",
            "fg": "#cdd6f4", "fg_muted": "#a6adc8", "fg_subtle": "#7f849c",
            "accent": "#cba6f7", "accent_alt": "#89b4fa",
            "success": "#a6e3a1", "warning": "#f9e2af", "error": "#f38ba8", "info": "#94e2d5",
            "red": "#f38ba8", "orange": "#fab387", "yellow": "#f9e2af", "green": "#a6e3a1",
            "cyan": "#94e2d5", "blue": "#89b4fa", "purple": "#cba6f7",
            "syntax_keyword": "#cba6f7", "syntax_string": "#a6e3a1", "syntax_number": "#fab387",
            "syntax_comment": "#9399b2", "syntax_function": "#89b4fa", "syntax_type": "#f9e2af",
            "syntax_variable": "#cdd6f4", "syntax_operator": "#89dceb", "syntax_tag": "#89b4fa",
            "syntax_attribute": "#f9e2af",
        },
    ),
    "catppuccin-latte": Palette(
        "Catppuccin Latte", CATPPUCCIN,
        {
            "bg": "#eff1f5", "bg_alt": "#e6e9ef", "bg_raised": "#ccd0da",
            "selection": "#cccfd8", "line": "#dfe1e7",
            "fg": "#4c4f69", "fg_muted": "#6c6f85", "fg_subtle": "#8c8fa1",
            "accent": "#8839ef", "accent_alt": "#1e66f5",
            "success": "#40a02b", "warning": "#df8e1d", "error": "#d20f39", "info": "#179299",
            "red": "#d20f39", "orange": "#fe640b", "yellow": "#df8e1d", "green": "#40a02b",
            "cyan": "#179299", "blue": "#1e66f5", "purple": "#8839ef",
            "syntax_keyword": "#8839ef", "syntax_string": "#40a02b", "syntax_number": "#fe640b",
            "syntax_comment": "#7c7f93", "syntax_function": "#1e66f5", "syntax_type": "#df8e1d",
            "syntax_variable": "#4c4f69", "syntax_operator": "#04a5e5", "syntax_tag": "#1e66f5",
            "syntax_attribute": "#df8e1d",
        },
    ),
    # Comment and type follow the README palette (#565f89, #2ac3de), not the shipped JSON.
    # Accent is the palette blue; the theme's own tab and link blues (#3d59a1, #6183bb) are dim.
    "tokyo-night": Palette(
        "Tokyo Night", TOKYO_NIGHT,
        {
            "bg": "#1a1b26", "bg_alt": "#16161e", "bg_raised": "#202330",
            "selection": "#2b2f41", "line": "#1e202e",
            "fg": "#a9b1d6", "fg_muted": "#787c99", "fg_subtle": "#565f89",
            "accent": "#7aa2f7", "accent_alt": "#bb9af7",
            "success": "#9ece6a", "warning": "#e0af68", "error": "#db4b4b", "info": "#0da0ba",
            "red": "#f7768e", "orange": "#ff9e64", "yellow": "#e0af68", "green": "#9ece6a",
            "cyan": "#7dcfff", "blue": "#7aa2f7", "purple": "#bb9af7",
            "syntax_keyword": "#bb9af7", "syntax_string": "#9ece6a", "syntax_number": "#ff9e64",
            "syntax_comment": "#565f89", "syntax_function": "#7aa2f7", "syntax_type": "#2ac3de",
            "syntax_variable": "#c0caf5", "syntax_operator": "#89ddff", "syntax_tag": "#f7768e",
            "syntax_attribute": "#bb9af7",
        },
    ),
    # Gruvbox names no accent: bright orange is used (bright yellow is the usual alternative).
    "gruvbox-dark": Palette(
        "Gruvbox Dark", GRUVBOX,
        {
            "bg": "#282828", "bg_alt": "#1d2021", "bg_raised": "#504945",
            "selection": "#665c54", "line": "#3c3836",
            "fg": "#ebdbb2", "fg_muted": "#a89984", "fg_subtle": "#928374",
            "accent": "#fe8019", "accent_alt": "#8ec07c",
            "success": "#b8bb26", "warning": "#fabd2f", "error": "#fb4934", "info": "#83a598",
            "red": "#fb4934", "orange": "#fe8019", "yellow": "#fabd2f", "green": "#b8bb26",
            "cyan": "#8ec07c", "blue": "#83a598", "purple": "#d3869b",
            "syntax_keyword": "#fb4934", "syntax_string": "#b8bb26", "syntax_number": "#d3869b",
            "syntax_comment": "#928374", "syntax_function": "#b8bb26", "syntax_type": "#fabd2f",
            "syntax_variable": "#83a598", "syntax_operator": "#ebdbb2", "syntax_tag": "#8ec07c",
            "syntax_attribute": "#8ec07c",
        },
    ),
    # fg_muted is the VS Code port's 60% nord4 over nord0: Nord's 16 colours have no muted text.
    "nord": Palette(
        "Nord", NORD,
        {
            "bg": "#2e3440", "bg_alt": "#3b4252", "bg_raised": "#434c5e",
            "selection": "#3f4758", "line": "#3b4252",
            "fg": "#d8dee9", "fg_muted": "#949aa5", "fg_subtle": "#616e88",
            "accent": "#88c0d0", "accent_alt": "#81a1c1",
            "success": "#a3be8c", "warning": "#ebcb8b", "error": "#bf616a", "info": "#88c0d0",
            "red": "#bf616a", "orange": "#d08770", "yellow": "#ebcb8b", "green": "#a3be8c",
            "cyan": "#88c0d0", "blue": "#81a1c1", "purple": "#b48ead",
            "syntax_keyword": "#81a1c1", "syntax_string": "#a3be8c", "syntax_number": "#b48ead",
            "syntax_comment": "#616e88", "syntax_function": "#88c0d0", "syntax_type": "#8fbcbb",
            "syntax_variable": "#d8dee9", "syntax_operator": "#81a1c1", "syntax_tag": "#81a1c1",
            "syntax_attribute": "#8fbcbb",
        },
    ),
    "one-dark": Palette(
        "One Dark", ONE_DARK,
        {
            "bg": "#282c34", "bg_alt": "#21252b", "bg_raised": "#353b45",
            "selection": "#3e4451", "line": "#2d323c",
            "fg": "#abb2bf", "fg_muted": "#828997", "fg_subtle": "#5c6370",
            "accent": "#528bff", "accent_alt": "#61afef",
            "success": "#73c990", "warning": "#e2c08d", "error": "#ff6347", "info": "#6494ed",
            "red": "#e06c75", "orange": "#d19a66", "yellow": "#e5c07b", "green": "#98c379",
            "cyan": "#56b6c2", "blue": "#61afef", "purple": "#c678dd",
            "syntax_keyword": "#c678dd", "syntax_string": "#98c379", "syntax_number": "#d19a66",
            "syntax_comment": "#5c6370", "syntax_function": "#61afef", "syntax_type": "#e5c07b",
            "syntax_variable": "#e06c75", "syntax_operator": "#abb2bf", "syntax_tag": "#e06c75",
            "syntax_attribute": "#d19a66",
        },
    ),
    # Solarized has one highlight tone, so four surfaces coincide; Theme building derives them.
    "solarized-dark": Palette(
        "Solarized Dark", SOLARIZED,
        {
            "bg": "#002b36", "bg_alt": "#073642", "bg_raised": "#073642",
            "selection": "#073642", "line": "#073642",
            "fg": "#839496", "fg_muted": "#586e75", "fg_subtle": "#586e75",
            "accent": "#268bd2", "accent_alt": "#2aa198",
            "success": "#859900", "warning": "#b58900", "error": "#dc322f", "info": "#268bd2",
            "red": "#dc322f", "orange": "#cb4b16", "yellow": "#b58900", "green": "#859900",
            "cyan": "#2aa198", "blue": "#268bd2", "purple": "#6c71c4",
            "syntax_keyword": "#859900", "syntax_string": "#2aa198", "syntax_number": "#2aa198",
            "syntax_comment": "#586e75", "syntax_function": "#268bd2", "syntax_type": "#b58900",
            "syntax_variable": "#268bd2", "syntax_operator": "#859900", "syntax_tag": "#268bd2",
            "syntax_attribute": "#657b83",
        },
    ),
    "solarized-light": Palette(
        "Solarized Light", SOLARIZED,
        {
            "bg": "#fdf6e3", "bg_alt": "#eee8d5", "bg_raised": "#eee8d5",
            "selection": "#eee8d5", "line": "#eee8d5",
            "fg": "#657b83", "fg_muted": "#93a1a1", "fg_subtle": "#93a1a1",
            "accent": "#268bd2", "accent_alt": "#2aa198",
            "success": "#859900", "warning": "#b58900", "error": "#dc322f", "info": "#268bd2",
            "red": "#dc322f", "orange": "#cb4b16", "yellow": "#b58900", "green": "#859900",
            "cyan": "#2aa198", "blue": "#268bd2", "purple": "#6c71c4",
            "syntax_keyword": "#859900", "syntax_string": "#2aa198", "syntax_number": "#2aa198",
            "syntax_comment": "#93a1a1", "syntax_function": "#268bd2", "syntax_type": "#b58900",
            "syntax_variable": "#268bd2", "syntax_operator": "#859900", "syntax_tag": "#268bd2",
            "syntax_attribute": "#839496",
        },
    ),
    # Comment is Muted, per the palette README; the Neovim port uses the lighter Subtle #908caa.
    "rose-pine": Palette(
        "Rosé Pine", ROSE_PINE,
        {
            "bg": "#191724", "bg_alt": "#1f1d2e", "bg_raised": "#26233a",
            "selection": "#403d52", "line": "#21202e",
            "fg": "#e0def4", "fg_muted": "#908caa", "fg_subtle": "#6e6a86",
            "accent": "#ebbcba", "accent_alt": "#c4a7e7",
            "success": "#31748f", "warning": "#f6c177", "error": "#eb6f92", "info": "#9ccfd8",
            "red": "#eb6f92", "orange": "#ebbcba", "yellow": "#f6c177", "green": "#31748f",
            "cyan": "#ebbcba", "blue": "#9ccfd8", "purple": "#c4a7e7",
            "syntax_keyword": "#31748f", "syntax_string": "#f6c177", "syntax_number": "#f6c177",
            "syntax_comment": "#6e6a86", "syntax_function": "#ebbcba", "syntax_type": "#9ccfd8",
            "syntax_variable": "#e0def4", "syntax_operator": "#908caa", "syntax_tag": "#9ccfd8",
            "syntax_attribute": "#c4a7e7",
        },
    ),
    "kanagawa": Palette(
        "Kanagawa", KANAGAWA,
        {
            "bg": "#1f1f28", "bg_alt": "#16161d", "bg_raised": "#2a2a37",
            "selection": "#223249", "line": "#363646",
            "fg": "#dcd7ba", "fg_muted": "#c8c093", "fg_subtle": "#727169",
            "accent": "#7e9cd8", "accent_alt": "#957fb8",
            "success": "#98bb6c", "warning": "#ff9e3b", "error": "#e82424", "info": "#658594",
            "red": "#c34043", "orange": "#ffa066", "yellow": "#c0a36e", "green": "#76946a",
            "cyan": "#6a9589", "blue": "#7e9cd8", "purple": "#957fb8",
            "syntax_keyword": "#957fb8", "syntax_string": "#98bb6c", "syntax_number": "#d27e99",
            "syntax_comment": "#727169", "syntax_function": "#7e9cd8", "syntax_type": "#7aa89f",
            "syntax_variable": "#dcd7ba", "syntax_operator": "#c0a36e", "syntax_tag": "#7fb4ca",
            "syntax_attribute": "#e6c384",
        },
    ),
    "everforest-dark": Palette(
        "Everforest Dark", EVERFOREST,
        {
            "bg": "#2d353b", "bg_alt": "#232a2e", "bg_raised": "#3d484d",
            "selection": "#543a48", "line": "#343f44",
            "fg": "#d3c6aa", "fg_muted": "#9da9a0", "fg_subtle": "#859289",
            "accent": "#a7c080", "accent_alt": "#e69875",
            "success": "#a7c080", "warning": "#dbbc7f", "error": "#e67e80", "info": "#7fbbb3",
            "red": "#e67e80", "orange": "#e69875", "yellow": "#dbbc7f", "green": "#a7c080",
            "cyan": "#83c092", "blue": "#7fbbb3", "purple": "#d699b6",
            "syntax_keyword": "#e67e80", "syntax_string": "#a7c080", "syntax_number": "#d699b6",
            "syntax_comment": "#859289", "syntax_function": "#a7c080", "syntax_type": "#dbbc7f",
            "syntax_variable": "#d3c6aa", "syntax_operator": "#e69875", "syntax_tag": "#e69875",
            "syntax_attribute": "#a7c080",
        },
    ),
}
# fmt: on
