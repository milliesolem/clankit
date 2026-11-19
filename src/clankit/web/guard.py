"""The front door: every request passes these checks before any view sees it.

The web interface can run shell commands on the user's machine, so it trusts nothing that a
web page in the same browser could forge:

* **Host allow-list.** Only the loopback names with the server's own port are accepted. A
  DNS-rebinding page reaches the server under its own host name and is refused here.
* **Authentication.** Every route except the login page needs the HttpOnly, SameSite=Strict
  cookie set by visiting ``/?token=<token>``, or ``Authorization: Bearer <token>``.
* **State changes** (any method other than GET and HEAD) must carry ``X-Clankit: 1``, which a
  cross-site form cannot send and a cross-origin script cannot send without a CORS
  preflight that this server never grants; an ``Origin`` header, when present, must be the
  server's own, and ``Sec-Fetch-Site: cross-site`` is refused outright.

Every response gets a strict content security policy (no inline script, no inline style, no
third-party anything) and headers that keep the page out of frames and its URL out of
referrers. Rejections are JSON, like every other API error.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.parse import urlsplit

from pyramid.registry import Registry
from pyramid.request import Request
from pyramid.response import Response

from clankit.web.context import WebContext

#: Host names that always mean this machine.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
#: The header every state-changing request must carry.
CSRF_HEADER = "X-Clankit"
SAFE_METHODS = frozenset({"GET", "HEAD"})
#: The login page: reachable without a cookie, it is where the cookie comes from.
LOGIN_PATH = "/"

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    )
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def error_response(status: int, message: str) -> Response:
    """A JSON error body, the shape every API failure has."""
    response = Response(
        body=json.dumps({"error": message}).encode("utf-8"),
        status=status,
        content_type="application/json",
        charset="utf-8",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def split_host(host: str) -> tuple[str, int | None]:
    """``"127.0.0.1:8420"`` -> ``("127.0.0.1", 8420)``; IPv6 hosts keep their brackets."""
    host = host.strip().lower()
    if host.startswith("["):
        name, _, rest = host.partition("]")
        name += "]"
        port_text = rest[1:] if rest.startswith(":") else ""
    else:
        name, _, port_text = host.partition(":")
    if not port_text:
        return name, None
    return name, int(port_text) if port_text.isdigit() else -1


def is_authenticated(ctx: WebContext, request: Request) -> bool:
    credentials = ctx.credentials
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        return credentials.accepts_token(authorization[len("Bearer ") :].strip())
    return credentials.accepts_cookie(request.cookies.get(credentials.cookie_name))


def check(ctx: WebContext, request: Request) -> Response | None:
    """The response that refuses `request`, or None when it may proceed."""
    name, port = split_host(request.host)
    if name not in ctx.hosts or (ctx.port is not None and port != ctx.port):
        where = f"http://127.0.0.1:{ctx.port}/" if ctx.port is not None else "the printed address"
        return error_response(400, f"Unexpected Host header; open Clankit at {where}.")
    if request.method not in SAFE_METHODS:
        refusal = _check_state_change(request)
        if refusal is not None:
            return refusal
    if request.path != LOGIN_PATH and not is_authenticated(ctx, request):
        return error_response(
            401,
            "Not logged in. Open the address printed in the terminal where Clankit runs; "
            "it contains the login token.",
        )
    return None


def _check_state_change(request: Request) -> Response | None:
    origin = request.headers.get("Origin")
    if origin is not None:
        parts = urlsplit(origin)
        if parts.scheme != "http" or parts.netloc.lower() != request.host.lower():
            return error_response(403, "Cross-origin request refused.")
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return error_response(403, "Cross-site request refused.")
    if request.headers.get(CSRF_HEADER) != "1":
        return error_response(
            403, f"Missing {CSRF_HEADER} header; requests must come from the Clankit page."
        )
    return None


def guard_tween_factory(
    handler: Callable[[Request], Response], registry: Registry
) -> Callable[[Request], Response]:
    """Pyramid tween: placed at the ingress, so it runs before routing and error views."""
    ctx: WebContext = registry["clankit"]

    def guard(request: Request) -> Response:
        response = check(ctx, request) or handler(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        else:
            # Revalidate on every load (304 when unchanged): ES modules import each other by
            # plain URL, so a cache-busting query string cannot reach them.
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    return guard
