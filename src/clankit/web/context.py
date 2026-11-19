"""What every request handler of one web server shares: credentials, the engine and streams.

Request handlers run in waitress worker threads while the engine lives on the runtime's event
loop in another thread. `Engine` is the only way handlers reach it: every read and write is
marshalled onto that loop with `RuntimeThread.submit` / `RuntimeThread.call`, and a loop that
does not answer in time becomes an HTTP 504 rather than a hung request.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from clankit.runtime import Runtime, RuntimeThread
    from clankit.web.sessions import SessionIndex
    from clankit.web.stream import StreamRegistry

T = TypeVar("T")

#: Seconds a request waits for the engine to answer a quick question.
ENGINE_TIMEOUT = 30.0
#: Seconds a slash command may take: /compact and /skillify call a model.
COMMAND_TIMEOUT = 600.0


class ApiError(Exception):
    """A request that cannot be served; becomes a JSON ``{"error": message}`` response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True, slots=True)
class Credentials:
    """The login token printed in the terminal and the cookie a browser gets for it.

    The cookie carries its own random value rather than the token, and its name is derived
    from the token, so two Clankit servers on one machine (cookies ignore ports) do not
    overwrite each other's login.
    """

    token: str
    cookie_name: str
    cookie_value: str

    @classmethod
    def for_token(cls, token: str) -> Credentials:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
        return cls(token, f"clankit-{digest}", secrets.token_urlsafe(32))

    def accepts_token(self, supplied: str | None) -> bool:
        return bool(supplied) and hmac.compare_digest(
            str(supplied).encode("utf-8"), self.token.encode("utf-8")
        )

    def accepts_cookie(self, supplied: str | None) -> bool:
        return bool(supplied) and hmac.compare_digest(
            str(supplied).encode("utf-8"), self.cookie_value.encode("utf-8")
        )


class Engine:
    """Runs work on the runtime's event loop and turns a stalled loop into an API error."""

    def __init__(self, thread: RuntimeThread, timeout: float = ENGINE_TIMEOUT) -> None:
        self._thread = thread
        #: Seconds a request waits for an answer unless the call says otherwise.
        self.timeout = timeout

    @property
    def runtime(self) -> Runtime:
        """The runtime, for passing into functions that `submit` runs on its loop."""
        return self._thread.runtime

    def submit(self, fn: Callable[[], T], timeout: float | None = None) -> T:
        try:
            return self._thread.submit(fn, timeout=timeout or self.timeout)
        except TimeoutError as exc:
            raise ApiError(504, "The engine did not respond in time; try again.") from exc

    def call(self, make: Callable[[], Coroutine[Any, Any, T]], timeout: float | None = None) -> T:
        try:
            return self._thread.call(make, timeout=timeout or self.timeout)
        except TimeoutError as exc:
            raise ApiError(504, "The engine did not finish in time; try again.") from exc


@dataclass(slots=True)
class WebContext:
    """Everything one server's handlers share, stored in the Pyramid registry."""

    engine: Engine
    credentials: Credentials
    workspace: Path
    sessions: SessionIndex
    streams: StreamRegistry
    #: Host names (lower case, IPv6 in brackets) a request's Host header may carry.
    hosts: frozenset[str] = field(default_factory=frozenset)
    #: The port the server listens on; None accepts any (only useful in tests).
    port: int | None = None
