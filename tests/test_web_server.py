"""Serving the web UI: binding, the printed login link, browser opening and clean shutdown."""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from clankit.config.store import ConfigStore
from clankit.runtime import Runtime
from clankit.web import serve, server


def runtime_factory() -> Runtime:
    return Runtime(ConfigStore.open(), persist_sessions=False)


def fetch(port: int, path: str, headers: dict[str, str] | None = None) -> http.client.HTTPResponse:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", path, headers=headers or {})
    response = connection.getresponse()
    response.body = response.read()  # type: ignore[attr-defined]
    connection.close()
    return response


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


def test_serve_prints_a_login_link_serves_the_demo_and_stops_cleanly(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def use_the_server() -> None:
        printed = capsys.readouterr().out
        url = urlsplit(re.search(r"http://\S+", printed).group(0))  # type: ignore[union-attr]
        seen.update(printed=printed, url=url)
        login = fetch(url.port, f"/?{url.query}")
        seen["login"] = login.status
        cookie = login.getheader("Set-Cookie").split(";")[0]
        seen["state"] = json.loads(fetch(url.port, "/api/state", {"Cookie": cookie}).body)
        seen["foreign"] = fetch(
            url.port, "/api/state", {"Cookie": cookie, "Host": "evil.example"}
        ).status

    monkeypatch.setattr(server, "_wait_for_interrupt", use_the_server)
    serve(
        runtime_factory,
        workspace=workspace,
        host="localhost",
        port=0,
        open_browser=False,
        demo=True,
    )

    url = seen["url"]
    assert url.hostname == "127.0.0.1" and url.port
    assert re.fullmatch(r"token=[\w-]{40,}", url.query)
    assert str(workspace) in seen["printed"] and "Ctrl+C" in seen["printed"]
    assert "scripted session" in seen["printed"]
    assert seen["login"] == 303
    assert seen["foreign"] == 400
    session = seen["state"]["session"]
    assert session["meta"]["method"] == "demo" and session["meta"]["workspace"] == str(workspace)
    names = {thread.name for thread in threading.enumerate()}
    assert "clankit-web" not in names and "clankit-runtime" not in names
    with pytest.raises(OSError):
        fetch(url.port, "/")


def test_serve_opens_a_browser_when_the_configuration_says_so(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    monkeypatch.setattr(server, "_wait_for_interrupt", lambda: None)
    serve(runtime_factory, workspace=workspace, port=0)  # web.open-browser defaults to true
    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:")


def test_a_port_that_is_taken_is_reported_plainly(workspace: Path) -> None:
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(SystemExit) as exit_info:
            serve(runtime_factory, workspace=workspace, port=port, open_browser=False)
    finally:
        blocker.close()
    assert f"port {port} on 127.0.0.1 is already in use" in str(exit_info.value)
    assert "clankit-runtime" not in {thread.name for thread in threading.enumerate()}


@pytest.mark.parametrize(
    ("configured", "bound"),
    [
        ("localhost", "127.0.0.1"),
        ("", "127.0.0.1"),
        ("LOCALHOST", "127.0.0.1"),
        ("[::1]", "::1"),
        ("0.0.0.0", "0.0.0.0"),
    ],
)
def test_localhost_is_pinned_to_ipv4_loopback(configured: str, bound: str) -> None:
    assert server.bind_address(configured) == bound


def test_ipv6_addresses_are_bracketed_in_urls() -> None:
    assert server.url_host("::1") == "[::1]"
    assert server.url_host("127.0.0.1") == "127.0.0.1"
