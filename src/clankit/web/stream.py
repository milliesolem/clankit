"""Server-Sent Events: one session's engine events, streamed to one browser tab.

The engine publishes on its event loop; a stream is served from a waitress worker thread. The
bridge between them is a bounded queue per stream, filled by an `EventBus` listener that never
blocks: if a tab stops reading and its queue fills up, the queue is emptied and the stream
resynchronises from a fresh snapshot instead of holding up the engine.

Wire format, one JSON object per ``data:`` line:

* first ``{"kind": "snapshot", "seq": n, "session": {...}}`` with ``id: n``, or, when the
  browser reconnects with ``Last-Event-ID`` and the bus still holds everything after it, the
  missed events instead;
* then every engine event as ``Event.to_dict()`` with ``id: <event.seq>`` (assistant messages
  carry their rendered ``html``);
* ``{"kind": "session_meta", ...}`` (no id) after usage changes and finished runs, with the
  cost and context figures only the engine can compute;
* a ``: ping`` comment every `HEARTBEAT` seconds of silence.

Each open stream holds one waitress thread, so streams are capped (`MAX_STREAMS`) and the
server sizes its thread pool from that cap. A tab identifies itself with ``?tab=``; a new
stream from the same tab replaces its old one, so reconnects never pile up.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from clankit.agent.events import Event, RunFinished, UsageChanged
from clankit.web.context import ApiError
from clankit.web.markdown import enrich_event, enrich_snapshot
from clankit.web.payloads import session_meta, session_state

if TYPE_CHECKING:
    from clankit.agent.session import Session
    from clankit.web.context import WebContext

log = logging.getLogger(__name__)

#: Seconds of silence after which a comment keeps proxies and browsers from timing out.
HEARTBEAT = 15.0
#: Seconds between checks for a vanished client while nothing is being sent.
POLL = 1.0
#: Events a stream may fall behind before it is resynchronised from a snapshot.
QUEUE_LIMIT = 4000
#: Concurrent streams (browser tabs); each holds one server thread.
MAX_STREAMS = 8
#: Events written in one chunk at most.
BATCH = 500
#: The reconnect delay suggested to browsers, in milliseconds.
RETRY_MS = 2000

_CLOSE = object()
_RESYNC = object()


class Subscriber:
    """The queue between the engine and one stream.

    `push` is an `EventBus` listener: it runs on the engine's thread and must never block, so
    on overflow it discards what is queued and leaves a resync marker in its place.
    """

    def __init__(self, limit: int = QUEUE_LIMIT) -> None:
        self._queue: queue.Queue[object] = queue.Queue(maxsize=limit)
        self._lock = threading.Lock()
        self._closed = False

    def push(self, event: Event) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._replace_contents(_RESYNC)

    def close(self) -> None:
        """End the stream: the reader sees the close marker next."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._replace_contents(_CLOSE)

    def get(self, timeout: float) -> object:
        """The next item; raises `queue.Empty` after `timeout` seconds without one."""
        return self._queue.get(timeout=timeout)

    def drain(self, limit: int) -> list[object]:
        """Whatever else is queued right now, up to `limit` items, without waiting."""
        items: list[object] = []
        while len(items) < limit:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _replace_contents(self, marker: object) -> None:
        self.drain(self._queue.maxsize)
        self._queue.put_nowait(marker)


class StreamRegistry:
    """The open streams of one server: capped, one per tab, all closed at shutdown."""

    def __init__(self, limit: int = MAX_STREAMS, queue_limit: int = QUEUE_LIMIT) -> None:
        self.limit = limit
        self.queue_limit = queue_limit
        self._lock = threading.Lock()
        self._open: dict[str, Subscriber] = {}
        self._closed = False

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._open)

    def open(self, tab: str) -> Subscriber | None:
        """A subscriber for `tab`, replacing that tab's previous stream; None when full."""
        with self._lock:
            if self._closed:
                return None
            previous = self._open.pop(tab, None)
            if previous is None and len(self._open) >= self.limit:
                return None
            subscriber = self._open[tab] = Subscriber(self.queue_limit)
        if previous is not None:
            previous.close()
        return subscriber

    def release(self, tab: str, subscriber: Subscriber) -> None:
        with self._lock:
            if self._open.get(tab) is subscriber:
                del self._open[tab]

    def close_all(self) -> None:
        """Stop accepting streams and end every open one (server shutdown)."""
        with self._lock:
            self._closed = True
            subscribers = list(self._open.values())
            self._open.clear()
        for subscriber in subscribers:
            subscriber.close()


def frame(data: dict[str, Any], event_id: int | None = None) -> bytes:
    """One SSE message. JSON escapes line breaks, so the payload is always a single line."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}data: {payload}\n\n".encode("utf-8", errors="replace")


class EventStream:
    """The response body of one stream: an iterable WSGI ``app_iter`` with ``close()``.

    A class rather than a bare generator because a WSGI server calls ``close()`` even on a
    body it never iterated, and the bus listener must be removed in every case.
    """

    def __init__(
        self,
        ctx: WebContext,
        session: Session,
        tab: str,
        last_event_id: int | None,
        disconnected: Callable[[], bool],
    ) -> None:
        self._ctx = ctx
        self._session = session
        self._tab = tab
        self._disconnected = disconnected
        subscriber = ctx.streams.open(tab)
        if subscriber is None:
            raise ApiError(
                503,
                f"Too many open event streams (the limit is {ctx.streams.limit}); "
                "close other Clankit tabs.",
            )
        self._subscriber = subscriber
        self._unsubscribe: Callable[[], None] | None = session.bus.subscribe(subscriber.push)
        self._floor = 0
        self._frames: Iterator[bytes] | None = None
        try:
            self._initial = self._start(last_event_id)
        except BaseException:
            self._release()
            raise

    # -- WSGI -----------------------------------------------------------------------------------

    def __iter__(self) -> Iterator[bytes]:
        if self._frames is None:
            self._frames = self._run()
        return self._frames

    def close(self) -> None:
        """Called by the server when the response ends, however it ends."""
        self._release()
        if self._frames is not None:
            frames, self._frames = self._frames, None
            frames.close()  # type: ignore[attr-defined]

    def _release(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
            self._ctx.streams.release(self._tab, self._subscriber)

    # -- producing frames -----------------------------------------------------------------------

    def _start(self, last_event_id: int | None) -> list[bytes]:
        """The first frames: what a reconnecting tab missed, or else a snapshot.

        The listener is subscribed before this runs, so nothing published in between is lost;
        anything both replayed here and queued is dropped later by its sequence number.
        """
        bus = self._session.bus
        if last_event_id is not None and 0 <= last_event_id <= bus.last_seq:
            missed = bus.since(last_event_id)
            if missed is not None:
                self._floor = last_event_id
                return self._event_frames(missed)
        return [self._snapshot_frame()]

    def _snapshot_frame(self) -> bytes:
        ctx, session = self._ctx, self._session
        state = ctx.engine.submit(lambda: session_state(session, ctx.sessions))
        enrich_snapshot(state["snapshot"])
        self._floor = state["snapshot"]["seq"]
        data = {"kind": "snapshot", "seq": self._floor, "session": state}
        return frame(data, self._floor)

    def _meta_frame(self) -> bytes:
        ctx, session = self._ctx, self._session
        meta = ctx.engine.submit(lambda: session_meta(session, ctx.sessions))
        return frame({"kind": "session_meta", "meta": meta})

    def _event_frames(self, items: list[object]) -> list[bytes]:
        chunks: list[bytes] = []
        meta_due = False
        for item in items:
            if item is _RESYNC:
                chunks.append(self._snapshot_frame())
                meta_due = False
            elif isinstance(item, Event) and item.seq > self._floor:
                self._floor = item.seq
                chunks.append(frame(enrich_event(item.to_dict()), item.seq))
                meta_due = meta_due or isinstance(item, (UsageChanged, RunFinished))
        if meta_due:
            chunks.append(self._meta_frame())
        return chunks

    def _run(self) -> Iterator[bytes]:
        # The first chunk goes out at once: waitress sends the headers only with it.
        yield f"retry: {RETRY_MS}\n\n".encode()
        try:
            if self._initial:
                yield b"".join(self._initial)
            last_write = time.monotonic()
            while True:
                try:
                    first = self._subscriber.get(POLL)
                except queue.Empty:
                    if self._disconnected():
                        return
                    if time.monotonic() - last_write >= HEARTBEAT:
                        last_write = time.monotonic()
                        yield b": ping\n\n"
                    continue
                items = [first, *self._subscriber.drain(BATCH)]
                closing = _CLOSE in items
                chunks = self._event_frames([i for i in items if i is not _CLOSE])
                if chunks:
                    last_write = time.monotonic()
                    yield b"".join(chunks)
                if closing:
                    return
        except (ApiError, RuntimeError):
            # The engine stopped answering (shutdown, or a stalled loop): end the stream and
            # let the browser reconnect.
            log.info(
                "event stream for session %s ended: the engine is not answering", self._session.id
            )
        finally:
            self._release()
