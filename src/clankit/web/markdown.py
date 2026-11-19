"""Assistant prose as HTML: Markdown rendered on the server, safe to put into the page as is.

Model output is untrusted text, so the renderer is configured to produce nothing a model could
use against the page or the user: raw HTML is disabled (it comes out escaped), images are not
rendered (an image URL would make the browser fetch it, which leaks data to whoever chose the
URL), and markdown-it's link validation drops ``javascript:`` and similar schemes. Links open
in a new tab without a referrer. Table alignment becomes a class rather than a ``style``
attribute, because the page's content security policy forbids inline styles.

Code blocks are highlighted by Pygments with its standard class names; the colours come from
`clankit.themes.syntax.pygments_css`, which the web UI serves once for every theme.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.token import Token
from markdown_it.utils import EnvType, OptionsDict
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

_FORMATTER = HtmlFormatter(nowrap=True)
_ALIGNMENTS = {
    "text-align:left": "left",
    "text-align:center": "center",
    "text-align:right": "right",
}


def _highlight(code: str, language: str, _attrs: str) -> str:
    """A fenced code block as ``<pre class="highlight">``; unknown languages stay plain."""
    try:
        lexer = get_lexer_by_name(language) if language else None
    except ClassNotFound:
        lexer = None
    body = highlight(code, lexer, _FORMATTER) if lexer is not None else escapeHtml(code)
    return f'<pre class="highlight"><code>{body}</code></pre>'


def _link_open(
    self: Any, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType
) -> str:
    tokens[idx].attrSet("target", "_blank")
    tokens[idx].attrSet("rel", "noopener noreferrer")
    return self.renderToken(tokens, idx, options, env)


def _cell_open(
    self: Any, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType
) -> str:
    token = tokens[idx]
    style = token.attrGet("style")
    if isinstance(style, str):
        token.attrs.pop("style", None)
        if style in _ALIGNMENTS:
            token.attrSet("class", f"align-{_ALIGNMENTS[style]}")
    return self.renderToken(tokens, idx, options, env)


def _parser() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "highlight": _highlight})
    md.enable(["table", "strikethrough"])
    md.disable("image")
    md.add_render_rule("link_open", _link_open)
    md.add_render_rule("th_open", _cell_open)
    md.add_render_rule("td_open", _cell_open)
    return md


_MD = _parser()


@functools.lru_cache(maxsize=1024)
def render_markdown(text: str) -> str:
    """`text` (Markdown written by a model) as HTML that is safe to insert into the page.

    Cached because every browser tab that connects receives the whole transcript again.
    """
    return _MD.render(text)


def enrich_event(data: dict[str, Any]) -> dict[str, Any]:
    """Add the rendered ``html`` to an ``assistant_message`` event's wire form, in place."""
    if data.get("kind") == "assistant_message" and data.get("text"):
        data["html"] = render_markdown(data["text"])
    return data


def enrich_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Add ``html`` to every finished assistant item of a `SessionView` snapshot, in place.

    Items still streaming keep their plain text; the browser swaps in the HTML when the
    final ``assistant_message`` arrives.
    """
    for node in snapshot.get("nodes", {}).values():
        for item in node.get("items", ()):
            if item.get("kind") == "assistant" and item.get("done") and item.get("text"):
                item["html"] = render_markdown(item["text"])
    return snapshot
