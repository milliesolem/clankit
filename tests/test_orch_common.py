"""Shared orchestration helpers: harness checks, harness diffs, model resolution, review parsing."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.scripted import ScriptedProvider, call
from clankit.orchestration import all_orchestrators, method_names
from clankit.orchestration.common import (
    FinishTool,
    changed_paths,
    diff_stat,
    models_guide,
    resolve_model,
    run_check,
    run_diff,
    run_to_finish,
)
from clankit.orchestration.review import parse_findings


async def test_run_check_reports_success_failure_and_output(session: Session) -> None:
    node = session.new_node("check")
    ctx = session.tool_context(node)
    ok = await run_check(session, ctx, "echo hello && exit 0")
    assert ok.passed and ok.exit_code == 0 and "hello" in ok.output
    bad = await run_check(session, ctx, "echo broken >&2; exit 3")
    assert not bad.passed and bad.exit_code == 3 and "broken" in bad.output
    assert "FAILED (exit code 3)" in bad.render()


async def test_run_check_times_out_and_kills_the_process_group(
    session: Session, workspace: Path
) -> None:
    node = session.new_node("check")
    ctx = session.tool_context(node)
    marker = workspace / "survivor"
    started = time.monotonic()
    result = await run_check(session, ctx, f"(sleep 2; touch {marker}) & sleep 30", seconds=0.3)
    assert time.monotonic() - started < 5
    assert not result.passed and "Timed out" in result.output
    await asyncio.sleep(2.5)
    assert not marker.exists()


async def test_run_check_respects_permissions(session: Session, config: Config) -> None:
    config.permissions.mode = "read-only"
    node = session.new_node("check")
    result = await run_check(session, session.tool_context(node), "true")
    assert result.denied and not result.passed and "permission was denied" in result.render()


async def test_run_check_output_is_trimmed_to_its_tail(session: Session) -> None:
    node = session.new_node("check")
    result = await run_check(session, session.tool_context(node), "seq 1 20000", output_limit=200)
    assert result.output.endswith("20000\n") and "earlier output omitted" in result.output


async def test_harness_diff_and_stats_follow_the_ledger(session: Session, workspace: Path) -> None:
    (workspace / "kept.txt").write_text("one\ntwo\n")
    node = session.new_node("writer")
    session.files.begin_run()
    for name, content in (("kept.txt", "one\nTWO\nthree\n"), ("new.txt", "fresh\n")):
        path = workspace / name
        session.files.checkpoint(path)
        path.write_text(content)
        session.files.note_write(node.id, path)
    diff = run_diff(session)
    assert "--- a/kept.txt" in diff and "+TWO" in diff and "-two" in diff
    assert "--- /dev/null" in diff and "+++ b/new.txt" in diff
    stats = diff_stat(session, changed_paths(session, [node.id]))
    assert stats == ["kept.txt  +2 -1", "new.txt  +1 -0 (new)"]
    assert run_diff(session, [workspace / "new.txt"]).count("+++") == 1
    assert "[diff truncated" in run_diff(session, limit=10)


def test_model_resolution(session: Session, scripted: ScriptedProvider) -> None:
    assert str(resolve_model(session, None)) == "scripted/test-model"
    assert str(resolve_model(session, "scripted/other")) == "scripted/other"
    with pytest.raises(ValueError, match="provider/model"):
        resolve_model(session, "no-slash")
    with pytest.raises(ValueError, match="not configured"):
        resolve_model(session, "elsewhere/model")
    assert models_guide(session) == ""  # one known model: nothing to choose from


async def test_run_to_finish_reminds_then_gives_up(
    session: Session, scripted: ScriptedProvider
) -> None:
    tool = FinishTool(
        "done_tool",
        "Finish.",
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
    )
    node = session.new_node("finisher")
    agent = session.agent(node, system="s", tools=[tool])
    scripted.push("forgot", call("done_tool", x=3))
    _, payload = await run_to_finish(agent, "go", "done_tool")
    assert payload == {"x": 3}

    node2 = session.new_node("never")
    agent2 = session.agent(node2, system="s", tools=[tool])
    scripted.push("no", "still no", "never")
    _, payload2 = await run_to_finish(agent2, "go", "done_tool", reminders=2)
    assert payload2 is None and len(scripted.requests) == 5


def test_findings_without_evidence_are_dropped() -> None:
    findings = parse_findings(
        {
            "findings": [
                {
                    "severity": "major",
                    "location": "a.py:3",
                    "problem": "Off by one.",
                    "evidence": "range(n) skips n",
                },
                {
                    "severity": "critical",
                    "location": "b.py",
                    "problem": "Vague.",
                    "evidence": "trust me"[:5],
                },
                {
                    "severity": "weird",
                    "location": "c.py",
                    "problem": "Odd severity.",
                    "evidence": "$ pytest -> 1 failed",
                },
                "not a dict",
            ]
        }
    )
    assert [(f.location, f.severity) for f in findings] == [("a.py:3", "major"), ("c.py", "minor")]
    assert parse_findings(None) == []


def test_every_registered_method_loads_and_declares_valid_settings() -> None:
    loaded = {cls.name: cls for cls in all_orchestrators()}
    for name, cls in loaded.items():
        assert cls.title and cls.summary and cls.guidance, name
        keys = [s.key for s in cls.settings]
        assert len(keys) == len(set(keys)), name
        for setting in cls.settings:
            assert setting.coerce(setting.default) == setting.default, (name, setting.key)
    assert {"task-tree", "solo", "demo"} <= set(loaded) <= set(method_names())
    assert os.environ.get("CLANKIT_CONFIG_DIR")  # isolation fixture is active
