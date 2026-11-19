"""The event stream: SSE framing, snapshots, Last-Event-ID replay, resync, caps, shutdown."""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from webob import Request

from clankit.agent.events import AssistantMessage, UsageChanged
from clankit.agent.session import Session
from clankit.llm.scripted import ScriptedProvider
from clankit.web import stream
from clankit.web.app import make_app
from clankit.web.server import WebServer, listen
from clankit.web.stream import StreamRegistry, Subscriber, frame
from test_web_support import HOST, TOKEN, WebHarness


@pytest.fixture
def web(tmp_path: Path) -> Iterator[WebHarness]:
    harness = WebHarness(tmp_path, provider=ScriptedProvider(provider_id="scripted"))
    yield harness
    harness.close()


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll for disconnects and send heartbeats quickly, so tests do not wait."""
    monkeypatch.setattr(stream, "POLL", 0.01)
    monkeypatch.setattr(stream, "HEARTBEAT", 0.05)


def parse(chunk: bytes) -> list[dict[str, Any]]:
    """SSE messages in `chunk`: {"id": int | None, "data": dict} or {"comment": str}."""
    messages = []
    for block in chunk.decode("utf-8").split("\n\n"):
        if not block:
            continue
        message: dict[str, Any] = {"id": None}
        for line in block.split("\n"):
            field, _, value = line.partition(": ")
            if field == "id":
                message["id"] = int(value)
            elif field == "data":
                message["data"] = json.loads(value)
            elif field == "":
                message["comment"] = value
            elif field == "retry":
                message["retry"] = int(value)
        messages.append(message)
    return messages


class Stream:
    """An event stream called straight through WSGI, read one chunk at a time."""

    def __init__(
        self,
        web: WebHarness,
        session_id: str,
        *,
        tab: str = "tab-1",
        headers: dict[str, str] | None = None,
        query: str = "",
        disconnected: Callable[[], bool] | None = None,
    ) -> None:
        request = Request.blank(
            f"/api/sessions/{session_id}/events?tab={tab}{query}",
            environ={"HTTP_HOST": HOST},
            headers={"Authorization": f"Bearer {TOKEN}", **(headers or {})},
        )
        if disconnected is not None:
            request.environ["waitress.client_disconnected"] = disconnected
        self.status, self.headers, self.body = request.call_application(web.app)
        self._chunks = iter(self.body)

    def next(self) -> list[dict[str, Any]]:
        return parse(next(self._chunks))

    def start(self) -> list[dict[str, Any]]:
        """Read the retry hint and the first frames."""
        assert self.next() == [{"id": None, "retry": stream.RETRY_MS}]
        return self.next()

    def close(self) -> None:
        self.body.close()


def session_of(web: WebHarness, session_id: str) -> Session:
    return web.on_loop(lambda: web.runtime.get_session(session_id))


def finished_run(web: WebHarness, provider_reply: str = "All done.") -> tuple[str, int]:
    sid = web.new_session()
    web.provider.push(provider_reply)  # type: ignore[union-attr]
    state = web.run(sid, "do it")
    return sid, state["snapshot"]["seq"]


# -- the queue between engine and stream ----------------------------------------------------------


def test_a_subscriber_that_overflows_trades_its_backlog_for_a_resync_marker() -> None:
    subscriber = Subscriber(limit=3)
    events = [AssistantMessage(node_id="n", message_id=str(i), text="x") for i in range(5)]
    for event in events:
        subscriber.push(event)  # never blocks, never raises
    # The fourth event overflowed the queue; what came after the marker is kept.
    assert subscriber.drain(10) == [stream._RESYNC, events[4]]
    subscriber.push(events[0])
    subscriber.close()
    subscriber.push(events[1])  # ignored once closed
    assert subscriber.drain(10) == [stream._CLOSE]


def test_the_registry_caps_streams_and_keeps_one_per_tab() -> None:
    registry = StreamRegistry(limit=2)
    first = registry.open("a")
    assert first is not None and registry.open("b") is not None
    assert registry.open("c") is None
    replacement = registry.open("a")
    assert replacement is not None and replacement is not first
    assert first.drain(1) == [stream._CLOSE]
    registry.release("a", first)  # a stale release does not drop the replacement
    assert registry.active == 2
    registry.close_all()
    assert registry.active == 0 and registry.open("d") is None
    assert replacement.drain(1) == [stream._CLOSE]


def test_a_frame_is_one_line_of_json_whatever_the_text() -> None:
    data = {"kind": "notice", "text": "two\nlines\r\nand ünïcode"}
    raw = frame(data, 7)
    assert raw.startswith(b"id: 7\ndata: {") and raw.endswith(b"}\n\n")
    assert raw.count(b"\n") == 3
    assert parse(raw) == [{"id": 7, "data": data}]
    assert frame({"kind": "x"}) == b'data: {"kind":"x"}\n\n'


# -- what a stream sends --------------------------------------------------------------------------


def test_a_stream_opens_with_a_snapshot_of_the_session(web: WebHarness) -> None:
    sid, seq = finished_run(web, "Done with <script>x</script>.")
    events = Stream(web, sid)
    assert events.status == "200 OK"
    headers = dict(events.headers)
    assert headers["Content-Type"] == "text/event-stream; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    (snapshot,) = events.start()
    assert snapshot["id"] == seq
    data = snapshot["data"]
    assert data["kind"] == "snapshot" and data["seq"] == seq
    assert data["session"]["meta"]["id"] == sid
    assert data["session"]["runs"][0]["prompt"] == "do it"
    items = [i for n in data["session"]["snapshot"]["nodes"].values() for i in n["items"]]
    (answer,) = [i for i in items if i["kind"] == "assistant"]
    assert "&lt;script&gt;" in answer["html"]
    events.close()
    assert web.ctx.streams.active == 0


def test_engine_events_follow_with_their_sequence_numbers(web: WebHarness) -> None:
    sid = web.new_session()
    events = Stream(web, sid)
    (snapshot,) = events.start()
    session = session_of(web, sid)
    web.on_loop(lambda: session.notice("info", "first"))
    web.on_loop(lambda: session.notice("warning", "second"))
    received = events.next()
    if len(received) == 1:
        received += events.next()
    assert [m["id"] for m in received] == [snapshot["id"] + 1, snapshot["id"] + 2]
    assert [(m["data"]["kind"], m["data"]["text"]) for m in received] == [
        ("notice", "first"),
        ("notice", "second"),
    ]
    events.close()


def test_assistant_messages_carry_their_rendered_html(web: WebHarness) -> None:
    sid = web.new_session()
    events = Stream(web, sid)
    events.start()
    session = session_of(web, sid)
    web.on_loop(
        lambda: session.emit(
            AssistantMessage(node_id="n1", message_id="m", text="# Title\n<b>x</b>")
        )
    )
    (message,) = events.next()
    assert message["data"]["html"] == "<h1>Title</h1>\n<p>&lt;b&gt;x&lt;/b&gt;</p>\n"
    events.close()


def test_usage_changes_are_followed_by_the_session_facts(web: WebHarness) -> None:
    sid = web.new_session()
    events = Stream(web, sid)
    events.start()
    session = session_of(web, sid)
    usage = {"input_tokens": 5, "output_tokens": 7}
    web.on_loop(lambda: session.emit(UsageChanged(node_id="n1", usage=usage, total=usage)))
    event, meta = events.next()
    assert event["data"]["kind"] == "usage_changed"
    assert meta["id"] is None and meta["data"]["kind"] == "session_meta"
    assert meta["data"]["meta"]["id"] == sid
    events.close()


def test_a_reconnect_replays_what_it_missed(web: WebHarness) -> None:
    sid, seq = finished_run(web)
    events = Stream(web, sid, headers={"Last-Event-ID": str(seq - 3)})
    *replayed, meta = events.start()
    assert [m["id"] for m in replayed] == [seq - 2, seq - 1, seq]
    assert replayed[-1]["data"]["kind"] == "run_finished"
    assert meta["data"]["kind"] == "session_meta"  # the run ended: fresh cost and context
    events.close()


def test_the_last_event_id_may_come_as_a_query_parameter(web: WebHarness) -> None:
    sid, seq = finished_run(web)
    events = Stream(web, sid, query=f"&last_event_id={seq}")
    assert events.next() == [{"id": None, "retry": stream.RETRY_MS}]
    session = session_of(web, sid)
    web.on_loop(lambda: session.notice("info", "new"))
    (message,) = events.next()  # nothing was missed, so no snapshot came first
    assert message["id"] == seq + 1
    events.close()


@pytest.mark.parametrize("last", ["999999", "-5", "not-a-number"])
def test_an_unusable_last_event_id_starts_from_a_snapshot(web: WebHarness, last: str) -> None:
    sid, seq = finished_run(web)
    events = Stream(web, sid, headers={"Last-Event-ID": last})
    (snapshot,) = events.start()
    assert snapshot["data"]["kind"] == "snapshot" and snapshot["id"] == seq
    events.close()


def test_a_replay_that_is_no_longer_buffered_starts_from_a_snapshot(
    web: WebHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid, _ = finished_run(web)
    monkeypatch.setattr(session_of(web, sid).bus, "since", lambda seq: None)
    events = Stream(web, sid, headers={"Last-Event-ID": "1"})
    (snapshot,) = events.start()
    assert snapshot["data"]["kind"] == "snapshot"
    events.close()


def test_a_reader_that_falls_behind_is_resynchronised(web: WebHarness) -> None:
    web.ctx.streams.queue_limit = 5
    sid = web.new_session()
    events = Stream(web, sid)
    session = session_of(web, sid)

    def flood() -> int:
        for n in range(20):
            session.notice("info", f"notice {n}")
        return session.bus.last_seq

    last = web.on_loop(flood)  # the engine is never held up by the slow reader
    events.start()
    (resync,) = events.next()
    assert resync["data"]["kind"] == "snapshot" and resync["id"] == last
    items = [i for n in resync["data"]["session"]["snapshot"]["nodes"].values() for i in n["items"]]
    assert items == []  # notices without a node have nowhere to go yet; the seq is what counts
    web.on_loop(lambda: session.notice("info", "after"))
    (after,) = events.next()
    assert after["id"] == last + 1
    events.close()


def test_silence_is_filled_with_heartbeats(web: WebHarness, fast: None) -> None:
    events = Stream(web, web.new_session())
    events.start()
    assert events.next() == [{"id": None, "comment": "ping"}]
    events.close()


def test_a_stream_ends_when_its_tab_has_gone(web: WebHarness, fast: None) -> None:
    events = Stream(web, web.new_session(), disconnected=lambda: True)
    events.start()
    with pytest.raises(StopIteration):
        events.next()
    assert web.ctx.streams.active == 0


def test_a_tab_has_one_stream_and_the_server_caps_tabs(web: WebHarness) -> None:
    web.ctx.streams.limit = 2
    sid = web.new_session()
    first = Stream(web, sid, tab="a")
    first.start()
    Stream(web, sid, tab="b").start()
    refused = web.client.get(f"/api/sessions/{sid}/events?tab=c", status=503)
    assert "Too many open event streams" in refused.json["error"]
    Stream(web, sid, tab="a").start()  # the same tab again replaces its old stream
    with pytest.raises(StopIteration):
        first.next()


def test_shutdown_ends_every_stream(web: WebHarness) -> None:
    sid = web.new_session()
    events = Stream(web, sid)
    events.start()
    web.ctx.streams.close_all()
    with pytest.raises(StopIteration):
        events.next()
    web.client.get(f"/api/sessions/{sid}/events", status=503)


def test_a_stream_for_an_unknown_session_is_404(web: WebHarness) -> None:
    web.client.get("/api/sessions/nope/events", status=404)


def test_a_stream_ends_quietly_when_the_engine_stops_answering(web: WebHarness) -> None:
    sid = web.new_session()
    events = Stream(web, sid)
    events.start()
    session = session_of(web, sid)
    web.ctx.engine.timeout = 0.01
    usage = {"output_tokens": 1}

    def publish_then_stall() -> None:
        session.emit(UsageChanged(node_id="n1", usage=usage, total=usage))
        time.sleep(0.3)  # the session_meta lookup that follows gets no answer in time

    stall = threading.Thread(target=lambda: web.on_loop(publish_then_stall))
    stall.start()
    try:
        with pytest.raises(StopIteration):
            events.next()
    finally:
        stall.join()
    assert web.ctx.streams.active == 0


# -- over a real socket ---------------------------------------------------------------------------


def test_events_arrive_incrementally_over_http(web: WebHarness) -> None:
    sock = listen("127.0.0.1", 0)
    port = sock.getsockname()[1]
    app = make_app(web.thread, token=TOKEN, workspace=web.workspace, port=port)
    server = WebServer(app, sock).start()
    sid = web.on_loop(lambda: app.registry["clankit"].sessions.create().id)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(
            "GET", f"/api/sessions/{sid}/events?tab=x", headers={"Authorization": f"Bearer {TOKEN}"}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream; charset=utf-8"

        buffered = b""

        def message() -> list[str]:
            nonlocal buffered
            while b"\n\n" not in buffered:
                buffered += response.read1(65536)
            head, buffered = buffered.split(b"\n\n", 1)
            return head.decode().split("\n")

        assert message() == [f"retry: {stream.RETRY_MS}"]
        snapshot = message()
        assert snapshot[0] == "id: 0" and '"kind":"snapshot"' in snapshot[1]
        session = session_of(web, sid)
        web.on_loop(lambda: session.notice("info", "live"))
        live = message()  # arrives while the response is still open
        assert live[0] == "id: 1" and '"text":"live"' in live[1]
    finally:
        connection.close()
        app.registry["clankit"].streams.close_all()
        assert server.stop(timeout=5)
