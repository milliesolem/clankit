"""The web_fetch tool: read one web page or text resource.

A plain GET, with three protections around it: only http and https, a cap on how much is
downloaded and on how much is returned, and redirects followed by hand. The last matters
because permission is granted per host: a redirect to another host is put to the permission
gate before it is followed, so an allowed site cannot bounce a request somewhere else.

HTML is reduced to readable text with the standard library; a model has no use for markup,
scripts or navigation, and a typical page shrinks by an order of magnitude.
"""

from __future__ import annotations

import asyncio
import glob
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx2

from clankit import __version__
from clankit.tools.base import PermissionAsk, Tool, ToolContext, ToolOutput
from clankit.tools.files import output_budget

__all__ = ["WebFetchTool", "html_to_text", "make_transport"]

#: Seconds one request (connecting, and reading the body) may take.
FETCH_TIMEOUT = 30.0
#: Bytes of a response body that are downloaded; the rest is left on the server.
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 10

_HEADERS = {
    "User-Agent": f"Mozilla/5.0 (compatible; Clankit/{__version__})",
    "Accept": "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.5",
}
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_TEXT_TYPE_ENDINGS = (
    "+json",
    "+xml",
    "/json",
    "/xml",
    "/javascript",
    "/yaml",
    "/x-yaml",
    "/toml",
    "/x-ndjson",
)
_BINARY_FAMILIES = ("image/", "audio/", "video/", "font/")
_SNIFF_BYTES = 8192
_HTML_START = re.compile(rb"\s*<(!doctype\s+html|html[\s>])", re.IGNORECASE)


class _FetchFailed(Exception):
    """A fetch that did not produce a page. The message is the tool result."""


def make_transport() -> httpx2.AsyncBaseTransport | None:
    """The transport requests are sent through; None means httpx2's default, the network.

    A seam for tests, which replace it to serve canned responses from an
    `httpx2.MockTransport` while everything else about the client stays real.
    """
    return None


# --------------------------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------------------------


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a URL (http or https) with a GET request and return its content as text. "
        "HTML pages are converted to readable text, with links kept as `text (url)`; JSON, "
        "Markdown, source files and other text come back unchanged; binary content (images, "
        "PDFs, archives) is refused. Use it to read documentation, an issue, a changelog or "
        "an API reference whose URL you know or were given. It cannot search the web, log "
        "in, run JavaScript or send data, and it returns only the beginning of a very long "
        "page, so prefer the URL of the specific page or raw file you need."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The address to fetch, including https://."},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    read_only = True

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"fetch {tool_input['url']}"

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        url = _normalise(tool_input["url"])
        if _url_problem(url):
            return None  # nothing will be sent; run() says what is wrong with the address
        return _ask(url, f"Fetch {url}")

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        url = _normalise(tool_input["url"])
        problem = _url_problem(url)
        if problem:
            return ToolOutput.error(problem)
        try:
            page = await _fetch(url, ctx, self.name)
            return ToolOutput(_present(page, output_budget(ctx)))
        except _FetchFailed as exc:
            return ToolOutput.error(str(exc))


def _normalise(url: str) -> str:
    """Tidy an address as models write them: add a missing scheme and an empty path's slash
    (so that the rule ``https://host/*`` covers ``https://host`` too)."""
    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    return parts._replace(path="/").geturl() if parts.netloc and not parts.path else url


def _url_problem(url: str) -> str:
    """Why `url` cannot be fetched, or "" if it can."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return f"{url!r} is not a valid URL."
    if parts.scheme not in ("http", "https"):
        return (
            f"web_fetch only handles http and https URLs, not {parts.scheme}://. For a local "
            "file use read."
        )
    if not host:
        return f"{url!r} has no host name. Give a full address such as https://example.com/page."
    return ""


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2].lower()}"


def _ask(url: str, summary: str) -> PermissionAsk:
    return PermissionAsk("network", summary, url, session_rule=f"{glob.escape(_origin(url))}/*")


# --------------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Page:
    """A successful response, as far as it was downloaded."""

    url: str
    #: Media type without parameters, lower case; "" when the server sent none.
    media_type: str
    charset: str
    body: bytes
    #: The body was longer than `MAX_DOWNLOAD_BYTES`.
    cut_short: bool


async def _fetch(url: str, ctx: ToolContext, tool: str) -> _Page:
    """GET `url`, following redirects. A redirect to a host that has not been approved in
    this call is put to the permission gate first."""
    approved = {urlsplit(url).hostname}
    client = httpx2.AsyncClient(
        transport=make_transport(),
        headers=_HEADERS,
        follow_redirects=False,
        timeout=httpx2.Timeout(FETCH_TIMEOUT, connect=10.0),
    )
    async with client:
        for _ in range(MAX_REDIRECTS + 1):
            result = await _get(client, url)
            if isinstance(result, _Page):
                return result
            if _url_problem(result):
                raise _FetchFailed(f"{url} redirects to {result}, which web_fetch cannot follow.")
            if urlsplit(result).hostname not in approved:
                ask = _ask(result, f"Fetch {result} (redirected from {url})")
                decision = await ctx.permissions.check(ask, node_id=ctx.node_id, tool=tool)
                if not decision.allowed:
                    reason = f" ({decision.reason})" if decision.reason else ""
                    raise _FetchFailed(
                        f"{url} redirects to another host, {result}, and fetching that was "
                        f"not permitted{reason}. Do not retry; work without this page or "
                        "ask the user."
                    )
                approved.add(urlsplit(result).hostname)
            url = result
    raise _FetchFailed(f"Gave up after {MAX_REDIRECTS} redirects; the last one led to {url}.")


async def _get(client: httpx2.AsyncClient, url: str) -> _Page | str:
    """One request: the page, or the address the server redirects to."""
    try:
        async with asyncio.timeout(FETCH_TIMEOUT), client.stream("GET", url) as response:
            if response.has_redirect_location:
                return urljoin(url, response.headers["location"])
            if response.status_code >= 300:
                raise _FetchFailed(_status_message(url, response))
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > MAX_DOWNLOAD_BYTES:
                    break
            media_type = response.headers.get("content-type", "").partition(";")[0]
            return _Page(
                url=str(response.url),
                media_type=media_type.strip().lower(),
                charset=response.charset_encoding or "utf-8",
                body=bytes(body[:MAX_DOWNLOAD_BYTES]),
                cut_short=len(body) > MAX_DOWNLOAD_BYTES,
            )
    except (TimeoutError, httpx2.TimeoutException):
        raise _FetchFailed(
            f"{url} did not answer within {FETCH_TIMEOUT:.0f} seconds. The site may be down or "
            "slow; try once more later or use another source."
        ) from None
    except httpx2.ConnectError as exc:
        raise _FetchFailed(
            f"Could not connect to {urlsplit(url).hostname}: {exc}. Check the host name; if it "
            "is right, the site is unreachable from here."
        ) from None
    except (httpx2.HTTPError, httpx2.InvalidURL) as exc:
        raise _FetchFailed(f"Fetching {url} failed: {type(exc).__name__}: {exc}") from None


def _status_message(url: str, response: httpx2.Response) -> str:
    status = response.status_code
    if status in (401, 403):
        hint = (
            "The page needs a login or the site blocks automated clients; retrying will not help."
        )
    elif status in (404, 410):
        hint = "There is no such page. Check the address for mistakes rather than retrying it."
    elif status == 429:
        hint = "The site is rate limiting requests. Wait before asking this host again."
    elif status >= 500:
        hint = "The server failed. It may work later; otherwise use another source."
    else:
        hint = "The server did not return the page."
    answer = f"HTTP {status} {response.reason_phrase}".strip()
    return f"{url} answered {answer}. {hint}"


# --------------------------------------------------------------------------------------------
# Presenting
# --------------------------------------------------------------------------------------------


def _present(page: _Page, budget: int) -> str:
    """The page as text for the model, or `_FetchFailed` if it is not text at all."""
    kind = _classify(page.media_type, page.body)
    if kind == "binary":
        raise _FetchFailed(
            f"{page.url} is {page.media_type or 'binary data'}, not text, so web_fetch cannot "
            "show it. If you need the file itself, download it with bash (curl -LO)."
        )
    try:
        text = page.body.decode(page.charset, errors="replace")
    except LookupError:  # a charset name Python does not know
        text = page.body.decode("utf-8", errors="replace")
    if kind == "html":
        title, text = html_to_text(text, page.url)
        heading = f"Title: {title}\nURL: {page.url}" if title else f"URL: {page.url}"
        text = f"{heading}\n\n{text}"
    if not text.strip():
        return f"{page.url} returned an empty response."
    notes: list[str] = []
    if len(text) > budget:
        line_end = text.rfind("\n", budget // 2, budget)
        end = line_end if line_end > 0 else budget
        notes.append(
            f"Only the first {end:,} of {len(text):,} characters are shown. If what you need "
            "is missing, fetch a more specific URL."
        )
        text = text[:end]
    if page.cut_short:
        notes.append(f"The download was stopped at {MAX_DOWNLOAD_BYTES // 1_048_576} MB.")
    return f"{text.rstrip()}\n\n[{' '.join(notes)}]" if notes else text


def _classify(media_type: str, body: bytes) -> str:
    """ "html", "text" or "binary": by the declared type where it says, else by the bytes."""
    if media_type in _HTML_TYPES:
        return "html"
    if media_type.startswith("text/") or media_type.endswith(_TEXT_TYPE_ENDINGS):
        return "text"
    if media_type.startswith(_BINARY_FAMILIES) or not _looks_like_text(body):
        return "binary"
    return "html" if _HTML_START.match(body) else "text"


def _looks_like_text(body: bytes) -> bool:
    sample = body[:_SNIFF_BYTES]
    if b"\0" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Invalid only where the sample cuts a multi-byte character in two: still text.
        return len(body) > len(sample) and exc.start >= len(sample) - 3
    return True


# --------------------------------------------------------------------------------------------
# HTML to text
# --------------------------------------------------------------------------------------------

# fmt: off
#: Elements whose content is not part of what a page says.
_DROPPED = frozenset({
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "nav",
    "footer", "aside", "button", "select", "video", "audio",
})
#: Elements that start and end a paragraph (a blank line around them).
_PARAGRAPHS = frozenset({
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "ul", "ol", "dl", "table",
    "section", "article", "figure", "hr",
})
#: Elements that start and end a line.
_LINES = frozenset({
    "div", "li", "tr", "dt", "dd", "br", "header", "main", "summary", "details", "caption",
    "address", "figcaption", "fieldset", "form", "option",
})
# fmt: on
_HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}
_WHITESPACE = re.compile(r"\s+")
_SPACES = re.compile(r"[ \t]{2,}")
_FENCE = "```"


def html_to_text(html: str, base_url: str) -> tuple[str, str]:
    """(title, readable text) of an HTML document.

    Keeps what a reader would read: headings (as ``#`` lines), paragraphs, list items,
    table rows, preformatted blocks (fenced) and links (as ``text (url)``, made absolute
    against `base_url`). Drops scripts, styles and site navigation.
    """
    extractor = _TextExtractor(base_url)
    extractor.feed(html)
    extractor.close()
    return _WHITESPACE.sub(" ", extractor.title).strip(), _tidy("".join(extractor.out))


def _tidy(text: str) -> str:
    """Collapse runs of spaces and blank lines, leaving fenced blocks as they are."""
    lines: list[str] = []
    fenced = False
    for raw in text.split("\n"):
        if raw.strip() == _FENCE:
            fenced = not fenced
        line = raw.rstrip() if fenced else _SPACES.sub(" ", raw).strip()
        if line or fenced or (lines and lines[-1]):
            lines.append(line)
    return "\n".join(lines).strip()


class _TextExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title = ""
        self._base_url = base_url
        #: Line breaks owed before the next piece of text. Paying them late keeps runs of
        #: block elements from piling up blank lines.
        self._owed = 0
        #: The dropped element we are inside, and how many of it are open.
        self._dropping = ""
        self._depth = 0
        self._in_title = False
        self._preformatted = 0
        #: The open link's address and the index in `out` where its text starts.
        self._link: tuple[str, int] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._dropping:
            self._depth += tag == self._dropping
            return
        if tag in _DROPPED:
            self._dropping, self._depth = tag, 1
        elif tag == "title":
            self._in_title = True
        elif tag == "pre":
            self._owe(2)
            self._write(f"{_FENCE}\n")
            self._preformatted += 1
        elif tag == "br" and self._preformatted:
            self._write("\n")
        elif tag in _PARAGRAPHS or tag in _LINES:
            self._owe(2 if tag in _PARAGRAPHS else 1)
            if tag in _HEADINGS:
                self._write(_HEADINGS[tag])
            elif tag == "li":
                self._write("- ")
        elif tag in ("td", "th"):
            if not self._owed and self.out:
                self._write(" | ")
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            self._link = (href, len(self.out)) if href and not href.startswith("#") else None
        elif tag == "img":
            alt = (dict(attrs).get("alt") or "").strip()
            if alt:
                self._write(f"[image: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        if self._dropping:
            self._depth -= tag == self._dropping
            if not self._depth:
                self._dropping = ""
        elif tag == "title":
            self._in_title = False
        elif tag == "pre" and self._preformatted:
            self._preformatted -= 1
            self.out.append(_FENCE if self.out[-1].endswith("\n") else f"\n{_FENCE}")
            self._owe(2)
        elif tag in _PARAGRAPHS or tag in _LINES:
            self._owe(2 if tag in _PARAGRAPHS else 1)
        elif tag == "a" and self._link is not None:
            href, start = self._link
            self._link = None
            label = "".join(self.out[start:]).strip()
            target = urljoin(self._base_url, href)
            if label and target != label and target.startswith(("http://", "https://")):
                self.out.append(f" ({target})")

    def handle_data(self, data: str) -> None:
        if self._dropping:
            return
        if self._in_title:
            self.title += data
        elif self._preformatted:
            self._write(data)
        else:
            text = _WHITESPACE.sub(" ", data)
            if self._owed or not self.out:
                text = text.lstrip()  # at the start of a line
            if text:
                self._write(text)

    def _owe(self, breaks: int) -> None:
        self._owed = max(self._owed, breaks)

    def _write(self, text: str) -> None:
        if self._owed and self.out:
            self.out.append("\n" * self._owed)
        self._owed = 0
        self.out.append(text)
