"""The scripted demo, played from start to finish through the web API alone."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from test_web_support import WebHarness, assistant_items, scripted_config


@pytest.fixture
def web(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[WebHarness]:
    monkeypatch.setenv("CLANKIT_DEMO_SPEED", "0")
    harness = WebHarness(tmp_path, config=scripted_config(mode="ask"))
    yield harness
    harness.close()


def test_the_demo_runs_to_completion_through_the_api(web: WebHarness) -> None:
    state = web.get("/api/state").json
    assert state["config"]["usable"] is False  # no provider: only the demo can run

    sid = web.post("/api/sessions", {"method": "demo"}).json["id"]
    web.post(f"/api/sessions/{sid}/prompt", {"text": "Add a KDL config loader"}, status=202)

    asking = web.wait_for(sid, lambda s: s["snapshot"]["permissions"])
    (permission,) = asking["snapshot"]["permissions"]
    assert permission["category"] == "shell"
    assert permission["session_rule"] == "shell:pytest *"
    asker = asking["snapshot"]["nodes"][permission["node_id"]]
    assert asker["title"] == "Round-trip tests" and asker["model"]
    assert asking["meta"]["busy"] is True
    web.post(
        f"/api/sessions/{sid}/permissions/{permission['request_id']}", {"choice": "allow_session"}
    )

    asking = web.wait_for(sid, lambda s: s["snapshot"]["questions"])
    (question,) = asking["snapshot"]["questions"]
    assert question["options"] == ["Keep this change small", "Update the documentation too"]
    web.post(
        f"/api/sessions/{sid}/questions/{question['request_id']}",
        {"answer": "Update the documentation too"},
    )

    done = web.wait_for(sid, lambda s: not s["snapshot"]["running"] and s["snapshot"]["last_run"])
    snapshot = done["snapshot"]
    assert snapshot["last_run"]["status"] == "completed"
    assert snapshot["permissions"] == [] and snapshot["questions"] == []
    nodes = snapshot["nodes"].values()
    assert len(nodes) == 8 and {n["status"] for n in nodes} == {"done"}
    assert max(n["depth"] for n in nodes) == 2
    assert done["runs"] == [
        {
            "seq": done["runs"][0]["seq"],
            "run_id": snapshot["last_run"]["run_id"],
            "prompt": "Add a KDL config loader",
            "method": "demo",
            "status": "completed",
            "finished_seq": snapshot["last_run"]["seq"],
        }
    ]
    answer = max(assistant_items(done), key=lambda item: item["seq"])
    assert "<table>" in answer["html"] and '<pre class="highlight">' in answer["html"]
    tools = [i for n in nodes for i in n["items"] if i["kind"] == "tool"]
    assert any(t["display"] and t["display"]["type"] == "diff" for t in tools)
    assert any(t["is_error"] for t in tools)
    assert done["meta"]["usage"]["output_tokens"] > 0
    assert done["meta"]["title"] == "Add a KDL config loader"


def test_the_demo_can_be_cancelled_while_it_waits(web: WebHarness) -> None:
    sid = web.post("/api/sessions", {"method": "demo"}).json["id"]
    web.post(f"/api/sessions/{sid}/prompt", {"text": "go"}, status=202)
    web.wait_for(sid, lambda s: s["snapshot"]["permissions"])
    web.post(f"/api/sessions/{sid}/command", {"line": "/new"})  # a command does not stop it
    assert web.post(f"/api/sessions/{sid}/cancel").json == {"cancelled": True}
    done = web.wait_for(sid, lambda s: not s["snapshot"]["running"])
    assert done["snapshot"]["last_run"]["status"] == "cancelled"
    assert all(n["status"] in ("done", "cancelled") for n in done["snapshot"]["nodes"].values())
