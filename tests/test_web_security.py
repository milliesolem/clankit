"""The web UI's front door: login, Host and Origin checks, the custom-header rule, headers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from clankit.config import actions
from test_web_support import CSRF, HOST, PORT, TOKEN, WebHarness


@pytest.fixture
def web(tmp_path: Path) -> Iterator[WebHarness]:
    harness = WebHarness(tmp_path, login=False)
    yield harness
    harness.close()


def login(web: WebHarness) -> None:
    web.client.get(f"/?token={TOKEN}", status=303)


# -- authentication -------------------------------------------------------------------------------


def test_the_page_refuses_a_visitor_without_the_token(web: WebHarness) -> None:
    response = web.client.get("/", status=401)
    assert "printed in the terminal" in response.text
    assert "Set-Cookie" not in response.headers


@pytest.mark.parametrize(
    "path", ["/api/state", "/theme.css", "/static/app.js", "/api/sessions/x/events"]
)
def test_every_other_route_needs_the_login(web: WebHarness, path: str) -> None:
    response = web.client.get(path, status=401)
    assert response.json["error"].startswith("Not logged in")


def test_state_changes_without_login_are_refused(web: WebHarness) -> None:
    web.client.post_json("/api/sessions", {}, headers=CSRF, status=401)


def test_the_token_link_sets_a_strict_http_only_cookie_and_redirects(web: WebHarness) -> None:
    response = web.client.get(f"/?token={TOKEN}", status=303)
    assert response.headers["Location"].endswith("/")
    assert "token" not in response.headers["Location"]
    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Path=/" in cookie
    assert TOKEN not in cookie  # the cookie carries its own secret, not the token
    page = web.client.get("/")
    assert '<html lang="en" data-theme="ayu-dark">' in page.text


def test_a_wrong_token_logs_nobody_in(web: WebHarness) -> None:
    response = web.client.get("/?token=guess", status=401)
    assert "not valid" in response.text
    assert "Set-Cookie" not in response.headers
    web.client.get("/api/state", status=401)


def test_a_bearer_token_works_without_a_cookie(web: WebHarness) -> None:
    web.client.get("/api/state", headers={"Authorization": f"Bearer {TOKEN}"}, status=200)
    web.client.get("/api/state", headers={"Authorization": "Bearer nope"}, status=401)


def test_a_forged_cookie_is_refused(web: WebHarness) -> None:
    web.client.set_cookie(web.ctx.credentials.cookie_name, "forged")
    web.client.get("/api/state", status=401)


def test_the_cookie_name_differs_between_servers(tmp_path: Path) -> None:
    from clankit.web.context import Credentials

    assert Credentials.for_token("a").cookie_name != Credentials.for_token("b").cookie_name


# -- host and origin ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["evil.example", f"evil.example:{PORT}", "127.0.0.1:9999", "127.0.0.1", "10.0.0.5:8420"],
)
def test_requests_for_another_host_are_refused(web: WebHarness, host: str) -> None:
    login(web)
    response = web.client.get("/api/state", extra_environ={"HTTP_HOST": host}, status=400)
    assert "Host" in response.json["error"]


@pytest.mark.parametrize("host", [f"localhost:{PORT}", f"[::1]:{PORT}", f"LOCALHOST:{PORT}"])
def test_loopback_names_are_accepted(web: WebHarness, host: str) -> None:
    web.client.get("/api/state", extra_environ={"HTTP_HOST": host}, headers=_bearer(), status=200)


def test_state_changes_need_the_custom_header(web: WebHarness) -> None:
    login(web)
    response = web.client.post_json("/api/sessions", {}, status=403)
    assert "X-Clankit" in response.json["error"]
    web.client.post_json("/api/sessions", {}, headers={"X-Clankit": "yes"}, status=403)
    web.client.post_json("/api/sessions", {}, headers=CSRF, status=200)


@pytest.mark.parametrize(
    "origin", ["http://evil.example", "null", f"https://{HOST}", "http://127.0.0.1:9999"]
)
def test_cross_origin_state_changes_are_refused(web: WebHarness, origin: str) -> None:
    login(web)
    headers = {**CSRF, "Origin": origin}
    response = web.client.post_json("/api/sessions", {}, headers=headers, status=403)
    assert "Cross-origin" in response.json["error"]


def test_a_same_origin_state_change_passes(web: WebHarness) -> None:
    login(web)
    web.client.post_json("/api/sessions", {}, headers={**CSRF, "Origin": f"http://{HOST}"})


def test_cross_site_fetch_metadata_is_refused(web: WebHarness) -> None:
    login(web)
    headers = {**CSRF, "Sec-Fetch-Site": "cross-site"}
    web.client.post_json("/api/sessions", {}, headers=headers, status=403)


def test_any_port_is_accepted_when_the_app_is_not_told_its_port(tmp_path: Path) -> None:
    harness = WebHarness(tmp_path, port=None)
    try:
        harness.client.get("/api/state", extra_environ={"HTTP_HOST": "127.0.0.1:1234"})
        harness.client.get(
            "/api/state", extra_environ={"HTTP_HOST": "evil.example:1234"}, status=400
        )
    finally:
        harness.close()


# -- responses ------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/api/state", "/api/nothing-here", "/static/app.css"])
def test_every_response_carries_the_security_headers(web: WebHarness, path: str) -> None:
    login(web)
    response = web.client.get(path, status="*")
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_refusals_carry_the_security_headers_too(web: WebHarness) -> None:
    response = web.client.get("/api/state", status=401)
    assert "Content-Security-Policy" in response.headers
    assert response.content_type == "application/json"


def test_api_responses_are_never_cached(web: WebHarness) -> None:
    login(web)
    assert web.client.get("/api/state").headers["Cache-Control"] == "no-store"
    assert web.client.get("/static/app.js").headers["Cache-Control"] == "no-cache"


def test_unknown_routes_and_methods_answer_in_json(web: WebHarness) -> None:
    login(web)
    assert web.client.get("/api/nothing-here", status=404).json["error"]
    assert web.client.get("/static/missing.js", status=404).json["error"]
    response = web.client.delete("/api/state", headers=CSRF, status=405)
    assert response.headers["Allow"] == "GET"
    assert "DELETE" in response.json["error"]


def test_a_failure_is_logged_without_the_request_body(
    web: WebHarness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    login(web)
    secret = "sk-ant-api03-DO-NOT-LOG-ME-0000"

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(actions, "set_api_key", broken)
    with caplog.at_level(logging.DEBUG):
        response = web.client.patch_json(
            "/api/providers/anthropic", {"api_key": secret}, headers=CSRF, status=500
        )
    assert "server log" in response.json["error"]
    assert "disk on fire" not in response.text
    assert any("PATCH /api/providers/anthropic failed" in r.getMessage() for r in caplog.records)
    assert secret not in caplog.text


def _bearer() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}
