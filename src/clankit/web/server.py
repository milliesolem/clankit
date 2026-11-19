"""Running the web interface: ``clankit web``.

`serve` starts the runtime on its own thread, opens a session, binds a loopback socket, prints
a login link and serves until Ctrl+C. Shutdown happens in the order that lets every part end
promptly: event streams first (each holds a server thread), then waitress (stopped from inside
its own loop, the only way that does not leave the loop or its workers behind), then the
runtime, which cancels any run still going.
"""

from __future__ import annotations

import contextlib
import errno
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from waitress import wasyncore
from waitress.server import create_server

from clankit.runtime import Runtime, RuntimeThread
from clankit.web.app import context_of, make_app
from clankit.web.stream import MAX_STREAMS

#: Server threads beyond those the event streams may hold, for ordinary requests.
REQUEST_THREADS = 8
#: Largest request body accepted, in bytes (a pasted file in a prompt fits easily).
MAX_BODY = 8 * 1024 * 1024
#: Addresses that listen on every interface; they say nothing about the Host to expect.
WILDCARDS = frozenset({"0.0.0.0", "::", ""})


def bind_address(host: str) -> str:
    """The address to bind for a configured host: "localhost" is pinned to 127.0.0.1, because
    it can resolve to IPv6 only, or to two sockets on different ports."""
    host = host.strip().strip("[]")
    return "127.0.0.1" if host.lower() in ("localhost", "") else host


def url_host(address: str) -> str:
    """`address` as it appears in a URL (IPv6 in brackets)."""
    return f"[{address}]" if ":" in address else address


def listen(address: str, port: int) -> socket.socket:
    """A bound socket for the server. Raises `OSError` when the port is taken."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, port))
    except OSError:
        sock.close()
        raise
    return sock


class WebServer:
    """waitress serving one app on a pre-bound socket, in a background thread."""

    def __init__(self, app: Any, sock: socket.socket) -> None:
        self._map: dict[int, Any] = {}
        self.server = create_server(
            app,
            map=self._map,
            sockets=[sock],
            threads=MAX_STREAMS + REQUEST_THREADS,
            # Lets a stream notice a closed tab within a second instead of on its next write.
            channel_request_lookahead=1,
            max_request_body_size=MAX_BODY,
            ident="clankit",
        )
        self._thread = threading.Thread(target=self.server.run, name="clankit-web", daemon=True)

    @property
    def port(self) -> int:
        return int(self.server.effective_port)

    def start(self) -> WebServer:
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop serving; True when the server thread ended within `timeout` seconds."""
        server = self.server

        def close_everything() -> None:  # runs inside the waitress loop thread
            server.task_dispatcher.shutdown(cancel_pending=True, timeout=1)
            wasyncore.close_all(self._map)

        if self._thread.is_alive():
            server.trigger.pull_trigger(close_everything)
            self._thread.join(timeout)
        return not self._thread.is_alive()


def serve(
    runtime_factory: Callable[[], Runtime],
    *,
    workspace: Path,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool | None = None,
    demo: bool = False,
) -> None:
    """Serve the web interface for `workspace` until interrupted (Ctrl+C)."""
    thread = RuntimeThread(runtime_factory).start()
    server: WebServer | None = None
    app = None
    try:
        web = thread.submit(lambda: thread.runtime.config.web)
        address = bind_address(host if host is not None else web.host)
        wanted = web.port if port is None else port
        try:
            sock = listen(address, wanted)
        except OSError as exc:
            raise SystemExit(_bind_problem(address, wanted, exc)) from None
        bound = sock.getsockname()[1]
        token = secrets.token_urlsafe(32)
        extra = () if address in WILDCARDS else (url_host(address),)
        app = make_app(thread, token=token, workspace=workspace, port=bound, hosts=extra)
        ctx = context_of(app)

        def first_session() -> None:
            session = ctx.sessions.create()
            if demo:
                session.method_override = "demo"

        thread.submit(first_session)
        server = WebServer(app, sock).start()
        shown = "127.0.0.1" if address in WILDCARDS else url_host(address)
        url = f"http://{shown}:{bound}/?token={token}"
        _announce(url, workspace, demo=demo)
        browse = web.open_browser if open_browser is None else open_browser
        if browse:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        _wait_for_interrupt()
    finally:
        if app is not None:
            context_of(app).streams.close_all()
        if server is not None:
            server.stop()
        thread.stop()


def _announce(url: str, workspace: Path, *, demo: bool) -> None:
    lines = [f"Clankit is serving {workspace}", f"  {url}"]
    if demo:
        lines.append("Playing the demo: any prompt starts the same scripted session.")
    lines.append("The link logs you in; keep it private. Press Ctrl+C to stop.")
    print("\n".join(lines), flush=True)


def _bind_problem(address: str, port: int, exc: OSError) -> str:
    if exc.errno == errno.EADDRINUSE:
        return (
            f"clankit: port {port} on {address} is already in use; "
            "pick another with --port (or web.port in the configuration)"
        )
    return f"clankit: cannot listen on {address}:{port}: {exc.strerror or exc}"


def _wait_for_interrupt() -> None:
    """Block until Ctrl+C. Sleeps in short slices so the signal is handled promptly."""
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            time.sleep(0.5)
    print("\nStopping Clankit.", file=sys.stderr, flush=True)
