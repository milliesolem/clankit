"""The web UI's JSON API: state, sessions, run control, commands and every kind of setting."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from clankit.commands import COMMANDS
from clankit.config import fields
from clankit.llm.scripted import ScriptedProvider, call
from clankit.orchestration.demo import DemoOrchestrator
from clankit.web.markdown import render_markdown
from test_web_support import WebHarness, assistant_items, scripted_config


@pytest.fixture
def scripted() -> ScriptedProvider:
    return ScriptedProvider(provider_id="scripted", models=("test-model", "other-model"))


@pytest.fixture
def web(tmp_path: Path, scripted: ScriptedProvider) -> Iterator[WebHarness]:
    harness = WebHarness(tmp_path, provider=scripted)
    yield harness
    harness.close()


@pytest.fixture
def sid(web: WebHarness) -> str:
    return web.new_session()


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[WebHarness]:
    """A server with no provider at all, permission mode "ask", the demo at full speed."""
    monkeypatch.setenv("CLANKIT_DEMO_SPEED", "0")
    harness = WebHarness(tmp_path, config=scripted_config(mode="ask"))
    yield harness
    harness.close()


def paused_demo(web: WebHarness) -> str:
    """A demo run waiting for its permission request, so the session is busy."""
    sid = web.new_session(method="demo")
    web.post(f"/api/sessions/{sid}/prompt", {"text": "go"}, status=202)
    web.wait_for(sid, lambda s: s["snapshot"]["permissions"])
    return sid


# -- state ----------------------------------------------------------------------------------------


def test_state_has_everything_the_page_needs(web: WebHarness, sid: str) -> None:
    state = web.get(f"/api/state?session={sid}").json
    assert state["workspace"] == str(web.workspace.resolve())
    assert state["config"]["path"] == str(web.config_path)
    assert state["config"]["usable"] is True
    assert [c["name"] for c in state["commands"]] == [c.name for c in COMMANDS]
    assert [s["id"] for s in state["sections"]] == [s.id for s in fields.SECTIONS]
    theme_field = next(f for s in state["sections"] for f in s["fields"] if f["path"] == "ui.theme")
    assert theme_field["value"] == "ayu-dark"
    ayu = next(t for t in state["themes"] if t["id"] == "ayu-dark")
    assert len(ayu["swatches"]) == 6 and all(
        re.fullmatch(r"#[0-9a-f]{6}", c) for c in ayu["swatches"]
    )
    assert state["theme_css"].startswith("/theme.css?v=")
    tree = next(m for m in state["methods"] if m["name"] == "task-tree")
    depth = next(s for s in tree["settings"] if s["key"] == "max-depth")
    assert depth["value"] == depth["default"] == 10
    assert state["models"] == []  # an injected provider's models are known once discovered
    assert {"active", "pending", "diagnostics", "directory"} <= set(state["skills"])
    assert any(p["id"] == "anthropic" for p in state["providers"])
    session = state["session"]
    assert session["meta"]["id"] == sid
    assert session["meta"]["model"] == "scripted/test-model"
    assert session["snapshot"]["seq"] == 0 and session["runs"] == []
    assert any(row["id"] == sid and row["live"] for row in state["sessions"])


def test_state_without_a_session_picks_the_newest_one(web: WebHarness) -> None:
    assert web.get("/api/state").json["session"] is None
    first = web.new_session()
    second = web.new_session()
    assert web.get("/api/state").json["session"]["meta"]["id"] == second != first


def test_session_facts_can_be_fetched_alone(web: WebHarness, sid: str) -> None:
    meta = web.get(f"/api/sessions/{sid}/meta").json
    assert meta["id"] == sid and meta["model"] == "scripted/test-model"
    assert {"title", "method", "busy", "usage", "cost", "context"} <= set(meta)
    web.put("/api/settings/models.default", {"value": "scripted/other-model"})
    assert web.get(f"/api/sessions/{sid}/meta").json["model"] == "scripted/other-model"


def test_unknown_sessions_are_404(web: WebHarness) -> None:
    assert "no open session" in web.get("/api/state?session=nope", status=404).json["error"]
    web.get("/api/sessions/nope", status=404)
    web.post("/api/sessions/nope/prompt", {"text": "hi"}, status=404)


def test_a_session_of_another_folder_is_not_reachable(web: WebHarness, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    foreign = web.on_loop(lambda: web.runtime.create_session(other).id)
    web.get(f"/api/sessions/{foreign}", status=404)


# -- sessions -------------------------------------------------------------------------------------


def test_a_new_session_keeps_the_overrides_of_the_one_it_replaces(web: WebHarness) -> None:
    plain = web.new_session()
    demo = web.new_session(method="demo")
    assert web.session(web.new_session(**{"from": demo}))["meta"]["method"] == "demo"
    assert web.session(web.new_session(**{"from": plain}))["meta"]["method"] == "task-tree"
    assert web.session(web.new_session())["meta"]["method"] == "task-tree"  # newest is plain


def test_a_new_session_with_an_unknown_method_is_refused(web: WebHarness) -> None:
    assert (
        "Unknown method" in web.post("/api/sessions", {"method": "magic"}, status=400).json["error"]
    )


def test_bad_request_bodies_are_400(web: WebHarness, sid: str) -> None:
    path = f"/api/sessions/{sid}/prompt"
    headers = {"X-Clankit": "1", "Content-Type": "application/json"}
    assert "not valid JSON" in web.client.post(path, "{", headers=headers, status=400).json["error"]
    assert "JSON object" in web.client.post(path, "[1]", headers=headers, status=400).json["error"]
    assert "required" in web.post(path, {"text": "  "}, status=400).json["error"]
    assert "string" in web.post(path, {"text": 3}, status=400).json["error"]


def test_a_stored_session_is_resumed_with_its_history(tmp_path: Path) -> None:
    provider = ScriptedProvider(["First answer."], provider_id="scripted")
    first = WebHarness(tmp_path, provider=provider, persist=True)
    try:
        sid = first.new_session()
        first.run(sid, "first question")
    finally:
        first.close()

    provider = ScriptedProvider(["Second answer."], provider_id="scripted")
    again = WebHarness(tmp_path, provider=provider, persist=True)
    try:
        stored = again.get("/api/state").json["sessions"]
        assert [(s["id"], s["title"], s["live"]) for s in stored] == [
            (sid, "first question", False)
        ]
        assert again.post("/api/sessions", {"resume": sid}).json["id"] == sid
        resumed = again.session(sid)
        assert [r["prompt"] for r in resumed["runs"]] == ["first question"]
        restored_seq = resumed["snapshot"]["seq"]
        assert [i["text"] for i in assistant_items(resumed)] == ["First answer."]

        after = again.run(sid, "second question")
        assert after["snapshot"]["last_run"]["seq"] > restored_seq  # numbering continues
        assert [r["prompt"] for r in after["runs"]] == ["first question", "second question"]
        texts = [i["text"] for i in sorted(assistant_items(after), key=lambda i: i["seq"])]
        assert texts == ["First answer.", "Second answer."]
    finally:
        again.close()


def test_resuming_an_unknown_session_is_404(web: WebHarness) -> None:
    web.post("/api/sessions", {"resume": "missing"}, status=404)


# -- runs -----------------------------------------------------------------------------------------


def test_a_prompt_runs_and_the_answer_arrives_as_safe_html(
    web: WebHarness, sid: str, scripted: ScriptedProvider
) -> None:
    scripted.push("Here is <script>alert(1)</script> and **bold** text.")
    state = web.run(sid, "say something")
    assert state["snapshot"]["last_run"]["status"] == "completed"
    (item,) = assistant_items(state)
    assert "<script>" not in item["html"]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in item["html"]
    assert "<strong>bold</strong>" in item["html"]
    assert state["runs"][0]["prompt"] == "say something"
    assert state["runs"][0]["status"] == "completed"
    assert state["meta"]["title"] == "say something"


def test_tool_calls_show_up_in_the_snapshot(
    web: WebHarness, sid: str, scripted: ScriptedProvider
) -> None:
    (web.workspace / "notes.txt").write_text("hello\n")
    scripted.push(call("read", file_path="notes.txt"), "It says hello.")
    state = web.run(sid, "read the notes")
    tools = [
        i for n in state["snapshot"]["nodes"].values() for i in n["items"] if i["kind"] == "tool"
    ]
    assert tools and tools[0]["done"] and "hello" in tools[0]["output"]


def test_a_second_prompt_during_a_run_is_refused(demo: WebHarness) -> None:
    sid = paused_demo(demo)
    response = demo.post(f"/api/sessions/{sid}/prompt", {"text": "more"}, status=409)
    assert "already in progress" in response.json["error"]


def test_steering_reaches_a_running_session_only(demo: WebHarness) -> None:
    idle = demo.new_session(method="demo")
    demo.post(f"/api/sessions/{idle}/steer", {"text": "faster"}, status=409)
    busy = paused_demo(demo)
    assert demo.post(f"/api/sessions/{busy}/steer", {"text": "faster"}).json == {"delivered": True}


def test_cancel_stops_a_run(demo: WebHarness) -> None:
    sid = paused_demo(demo)
    assert demo.post(f"/api/sessions/{sid}/cancel").json == {"cancelled": True}
    state = demo.wait_for(sid, lambda s: not s["snapshot"]["running"])
    assert state["snapshot"]["last_run"]["status"] == "cancelled"
    assert state["runs"][0]["status"] == "cancelled"
    assert demo.post(f"/api/sessions/{sid}/cancel").json == {"cancelled": False}


def test_permission_answers_are_validated(demo: WebHarness) -> None:
    sid = paused_demo(demo)
    request = demo.session(sid)["snapshot"]["permissions"][0]["request_id"]
    path = f"/api/sessions/{sid}/permissions/{request}"
    assert "must be one of" in demo.post(path, {"choice": "maybe"}, status=400).json["error"]
    demo.post(f"/api/sessions/{sid}/permissions/p999", {"choice": "allow"}, status=404)
    assert demo.post(path, {"choice": "allow_session"}).json == {"resolved": True}
    assert "already answered" in demo.post(path, {"choice": "deny"}, status=404).json["error"]


def test_unknown_questions_are_404(demo: WebHarness) -> None:
    sid = demo.new_session(method="demo")
    demo.post(f"/api/sessions/{sid}/questions/q1", {"answer": "yes"}, status=404)
    demo.post(f"/api/sessions/{sid}/questions/q1", {}, status=400)


def test_undo_reports_what_was_restored(
    web: WebHarness, sid: str, scripted: ScriptedProvider
) -> None:
    assert web.post(f"/api/sessions/{sid}/undo").json == {"restored": []}
    scripted.push(call("write", file_path="new.txt", content="x\n"), "Wrote it.")
    web.run(sid, "write a file")
    assert (web.workspace / "new.txt").exists()
    assert web.post(f"/api/sessions/{sid}/undo").json == {"restored": ["new.txt"]}
    assert not (web.workspace / "new.txt").exists()


def test_undo_waits_for_the_run(demo: WebHarness) -> None:
    sid = paused_demo(demo)
    assert "Wait for the run" in demo.post(f"/api/sessions/{sid}/undo", status=409).json["error"]


# -- commands -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/help", {"view": "help"}),
        ("/settings", {"view": "settings"}),
        ("/model", {"view": "models"}),
        ("/resume", {"view": "sessions"}),
        ("/new", {"action": "new-session"}),
        ("/quit", {"action": "quit"}),
        ("/frobnicate", {"level": "error"}),
    ],
)
def test_commands_return_their_outcome(
    web: WebHarness, sid: str, line: str, expected: dict[str, str]
) -> None:
    outcome = web.post(f"/api/sessions/{sid}/command", {"line": line}).json["outcome"]
    assert {key: outcome[key] for key in expected} == expected


def test_a_command_that_changes_settings_returns_the_new_state(web: WebHarness, sid: str) -> None:
    result = web.post(f"/api/sessions/{sid}/command", {"line": "/theme dracula"}).json
    assert result["outcome"]["message"] == "Theme set to Dracula."
    assert result["state"]["config"]["theme"] == "dracula"
    assert web.saved_config().ui.theme == "dracula"
    result = web.post(f"/api/sessions/{sid}/command", {"line": "/model scripted/other-model"}).json
    assert result["meta"]["model"] == "scripted/other-model"


def test_init_returns_a_prompt_for_the_page_to_send(web: WebHarness, sid: str) -> None:
    outcome = web.post(f"/api/sessions/{sid}/command", {"line": "/init"}).json["outcome"]
    assert "AGENTS.md" in outcome["prompt"]


def test_text_that_is_not_a_command_is_refused(web: WebHarness, sid: str) -> None:
    assert (
        "slash"
        in web.post(f"/api/sessions/{sid}/command", {"line": "hello"}, status=400).json["error"]
    )
    web.post(f"/api/sessions/{sid}/command", {"line": "/usr/bin/env"}, status=400)


# -- general settings -----------------------------------------------------------------------------


def test_a_setting_is_written_to_config_kdl(web: WebHarness) -> None:
    result = web.put("/api/settings/ui.show_thinking", {"value": False}).json
    assert result["message"] == "Saved Show reasoning."
    assert result["saved"] == str(web.config_path)
    assert result["state"]["config"]["show_thinking"] is False
    assert web.saved_config().ui.show_thinking is False
    web.put("/api/settings/agent.max_turns", {"value": "40"})
    assert web.saved_config().agent.max_turns == 40
    web.put("/api/settings/models.roles.planner", {"value": "scripted/other-model"})
    assert web.saved_config().models.roles == {"planner": "scripted/other-model"}
    web.put("/api/settings/models.roles.planner", {"value": ""})
    assert web.saved_config().models.roles == {}


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("agent.max_turns", "many", "whole number"),
        ("agent.max_turns", 0, "at least 1"),
        ("agent.compact_at", 2, "at most"),
        ("permissions.mode", "yolo", "must be one of"),
        ("ui.theme", "neon", "unknown theme"),
        ("models.default", "", "required"),
        ("models.default", "no-slash", "provider/model"),
    ],
)
def test_invalid_settings_are_refused_and_not_saved(
    web: WebHarness, path: str, value: object, message: str
) -> None:
    before = web.config_path.read_text()
    response = web.put(f"/api/settings/{path}", {"value": value}, status=400)
    assert message in response.json["error"]
    assert web.config_path.read_text() == before


def test_unknown_settings_are_404(web: WebHarness) -> None:
    web.put("/api/settings/agent.nonsense", {"value": 1}, status=404)
    web.put("/api/settings/orchestration.task-tree", {"value": 1}, status=404)
    web.put("/api/settings/ui.theme", {}, status=400)


def test_a_theme_change_changes_the_stylesheet_version(web: WebHarness) -> None:
    before = web.get("/api/state").json["theme_css"]
    web.put("/api/settings/ui.theme", {"value": "github-light"})
    state = web.get("/api/state").json
    assert state["config"]["theme"] == "github-light"
    assert state["theme_css"] == before  # same themes, same stylesheet
    page = web.get("/")
    assert 'data-theme="github-light"' in page.text
    assert '<meta name="color-scheme" content="light">' in page.text


# -- orchestration --------------------------------------------------------------------------------


def test_the_method_and_its_settings_are_saved(web: WebHarness) -> None:
    web.put("/api/settings/orchestration.method", {"value": "solo"})
    web.put("/api/settings/orchestration.task-tree.max-depth", {"value": "5"})
    web.put("/api/settings/orchestration.task-tree.cross-model", {"value": False})
    config = web.saved_config()
    assert config.orchestration.method == "solo"
    assert config.orchestration.options["task-tree"] == {"max-depth": 5, "cross-model": False}
    tree = next(m for m in web.get("/api/state").json["methods"] if m["name"] == "task-tree")
    assert next(s for s in tree["settings"] if s["key"] == "max-depth")["value"] == 5


def test_method_settings_are_validated(web: WebHarness) -> None:
    assert (
        "at least"
        in web.put(
            "/api/settings/orchestration.task-tree.max-depth", {"value": -1}, status=400
        ).json["error"]
    )
    assert (
        "Unknown method"
        in web.put("/api/settings/orchestration.method", {"value": "magic"}, status=400).json[
            "error"
        ]
    )
    web.put("/api/settings/orchestration.magic.depth", {"value": 1}, status=404)
    web.put("/api/settings/orchestration.task-tree.colour", {"value": 1}, status=404)


def test_demo_speed_is_a_method_setting_too(web: WebHarness) -> None:
    (speed,) = DemoOrchestrator.settings
    web.put(f"/api/settings/orchestration.demo.{speed.key}", {"value": 250})
    assert web.saved_config().orchestration.option("demo", speed.key) == 250


# -- providers ------------------------------------------------------------------------------------


def test_an_api_key_is_saved_but_only_ever_returned_masked(web: WebHarness) -> None:
    secret = "sk-ant-api03-abcdefghijklmnop-a91f"
    result = web.patch("/api/providers/anthropic", {"api_key": secret})
    assert secret not in result.text
    row = next(p for p in result.json["state"]["providers"] if p["id"] == "anthropic")
    assert row["key_source"] == "config" and row["masked_key"] == "sk-ant-…a91f"
    assert web.saved_config().providers["anthropic"].api_key == secret
    assert secret not in web.get("/api/state").text
    web.patch("/api/providers/anthropic", {"api_key": ""})
    assert "anthropic" not in web.saved_config().providers


def test_provider_changes_are_validated(web: WebHarness) -> None:
    assert (
        "spaces"
        in web.patch("/api/providers/anthropic", {"api_key": "a b"}, status=400).json["error"]
    )
    assert (
        "http"
        in web.patch("/api/providers/anthropic", {"base_url": "ftp://x"}, status=400).json["error"]
    )
    web.patch("/api/providers/anthropic", {"enabled": "yes"}, status=400)
    web.patch("/api/providers/anthropic", {}, status=400)
    web.patch("/api/providers/nobody", {"api_key": "k"}, status=404)


def test_a_keyless_local_provider_is_switched_on(web: WebHarness) -> None:
    rows = web.get("/api/state").json["providers"]
    local = next(p for p in rows if p["key_source"] == "keyless")
    result = web.patch(f"/api/providers/{local['id']}", {"enabled": True}).json
    assert "enabled" in result["message"]
    assert web.saved_config().providers[local["id"]].enabled is True


def test_a_custom_provider_is_added_and_removed(web: WebHarness) -> None:
    body = {
        "id": "my-server",
        "base_url": "http://localhost:8000/v1",
        "api_key": "secret-key-12345",
    }
    result = web.post("/api/providers", body).json
    assert "secret-key-12345" not in str(result)
    row = next(p for p in result["state"]["providers"] if p["id"] == "my-server")
    assert row["builtin"] is False and row["masked_key"].endswith("2345")
    assert web.saved_config().providers["my-server"].base_url == "http://localhost:8000/v1"
    assert "already exists" in web.post("/api/providers", body, status=400).json["error"]
    web.delete("/api/providers/my-server")
    assert "my-server" not in web.saved_config().providers
    web.delete("/api/providers/my-server", status=404)


def test_bad_custom_providers_are_refused(web: WebHarness) -> None:
    web.post("/api/providers", {"id": "Bad Id", "base_url": "http://x"}, status=400)
    web.post("/api/providers", {"id": "ok", "base_url": "not a url"}, status=400)
    web.post("/api/providers", {"id": "ok", "base_url": "http://x", "kind": "grpc"}, status=400)
    web.post("/api/providers", {"base_url": "http://x"}, status=400)


def test_removing_a_provider_warns_about_settings_that_still_use_it(web: WebHarness) -> None:
    web.patch("/api/providers/anthropic", {"api_key": "sk-ant-api03-0000000000000"})
    web.put("/api/settings/models.default", {"value": "anthropic/claude-opus-5-5"})
    message = web.delete("/api/providers/anthropic").json["message"]
    assert "Still pointing at it: default model" in message


def test_model_discovery_asks_the_providers(web: WebHarness, scripted: ScriptedProvider) -> None:
    refs = {m["ref"] for m in web.post("/api/models/discover").json["state"]["models"]}
    assert {"scripted/test-model", "scripted/other-model"} <= refs


# -- skills ---------------------------------------------------------------------------------------

SKILL = """---
name: {name}
description: "Use when writing release notes; collects the merged changes first."
metadata:
  clankit.reason: "The agent kept forgetting the changelog."
---

# Release notes

Collect the merged pull requests, then write the notes.
"""


def stage(web: WebHarness, name: str, *, pending: bool) -> None:
    web.on_loop(
        lambda: web.runtime.skills.save(
            name, {"SKILL.md": SKILL.format(name=name)}, pending=pending
        )
    )


def test_a_proposed_skill_is_listed_with_its_reason_and_approved(web: WebHarness) -> None:
    stage(web, "release-notes", pending=True)
    (pending,) = web.get("/api/state").json["skills"]["pending"]
    assert pending["name"] == "release-notes"
    assert pending["reason"] == "The agent kept forgetting the changelog."
    result = web.post("/api/skills/release-notes/approve").json
    assert result["message"] == "Approved skill release-notes."
    assert [s["name"] for s in result["state"]["skills"]["active"]] == ["release-notes"]
    assert result["state"]["skills"]["pending"] == []


def test_a_proposal_is_discarded(web: WebHarness) -> None:
    stage(web, "release-notes", pending=True)
    web.post("/api/skills/release-notes/reject")
    assert web.get("/api/state").json["skills"]["pending"] == []
    web.post("/api/skills/release-notes/reject", status=404)


def test_an_installed_skill_is_deleted(web: WebHarness) -> None:
    stage(web, "release-notes", pending=False)
    web.delete("/api/skills/release-notes")
    assert web.get("/api/state").json["skills"]["active"] == []
    web.delete("/api/skills/release-notes", status=404)
    web.post("/api/skills/missing/approve", status=404)


def test_a_broken_proposal_cannot_be_approved(web: WebHarness) -> None:
    directory = web.on_loop(lambda: web.runtime.skills.directories[0]) / ".pending" / "broken"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("no frontmatter here\n")
    assert "cannot be approved" in web.post("/api/skills/broken/approve", status=409).json["error"]


# -- page and assets ------------------------------------------------------------------------------


def test_the_page_loads_its_assets_from_this_server_only(web: WebHarness) -> None:
    page = web.get("/").text
    assert '<script type="module" src="/static/app.js"></script>' in page
    assert "://" not in page
    assert "<script>" not in page  # no inline script: the CSP forbids it
    for asset in ("app.js", "store.js", "transcript.js", "app.css"):
        response = web.get(f"/static/{asset}")
        assert "://" not in response.text.replace("http://localhost:8000/v1", "")
    assert web.get("/static/app.js").content_type in ("text/javascript", "application/javascript")


def test_the_theme_stylesheet_has_every_theme_and_the_code_colours(web: WebHarness) -> None:
    css = web.get("/theme.css")
    assert css.content_type == "text/css"
    themes = web.get("/api/state").json["themes"]
    for theme in themes:
        assert f':root[data-theme="{theme["id"]}"]' in css.text
    assert ".highlight .k" in css.text and "var(--syn-keyword)" in css.text


def test_the_stylesheet_names_no_colours_and_draws_no_frames(web: WebHarness) -> None:
    css = web.get("/static/app.css").text
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), "colours come from /theme.css only"
    assert not re.search(r"\b(rgb|rgba|hsl|hsla)\(", css)
    assert not re.search(r"\b(linear|radial|conic)-gradient\(", css)
    assert re.findall(r"border-radius:\s*([^;]+);", css) == ["0"]
    assert re.findall(r"border:\s*([^;]+);", css) == ["0"]
    shadows = re.findall(r"box-shadow:\s*([^;]+);", css)
    assert set(shadows) - {"none"} == {
        f"inset var(--bar) 0 0 var(--{name})"
        for name in ("accent", "fg-subtle", "error", "warning", "line")
    }
    assert "text-transform" not in css and "font-weight: 600" not in css


def test_markdown_rendering_is_cached_and_safe() -> None:
    html = render_markdown("[x](javascript:alert(1)) ![img](http://tracker.example/p.png)")
    assert "javascript:" not in html.replace("[x](javascript:alert(1))", "")
    assert "<img" not in html
    assert render_markdown("same text") is render_markdown("same text")


def test_an_engine_that_does_not_answer_becomes_a_504(web: WebHarness) -> None:
    web.ctx.engine.timeout = 0.05
    busy = threading.Thread(target=lambda: web.on_loop(lambda: time.sleep(0.5)))
    busy.start()
    time.sleep(0.05)
    try:
        response = web.client.get("/api/state", status=504)
        assert "did not respond" in response.json["error"]
    finally:
        busy.join()
