"""Saved workflows: the built-ins, the user's own, and running them end to end."""

from __future__ import annotations

from typing import Any

import pytest

from clankit import paths
from clankit.agent.session import Session
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest
from clankit.orchestration.workflow import Limits, WorkflowRun, find_saved, load_saved

BUILT_INS = ["implement", "research", "review", "understand"]


def instance(schema: dict[str, Any]) -> Any:
    """A small value that satisfies `schema` (enough for the schemas workflows use)."""
    if "enum" in schema:
        return schema["enum"][-1]
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        return {name: instance(properties[name]) for name in schema.get("required", [])}
    if kind == "array":
        return [instance(schema.get("items", {"type": "string"}))]
    if kind in ("integer", "number"):
        return schema.get("minimum", 1)
    if kind == "boolean":
        return True
    return "x"


def answer_anything(request: ChatRequest) -> Any:
    """Every agent succeeds: structured agents return a minimal valid result, others text."""
    result_tool = next((t for t in request.tools if t.name == "return_result"), None)
    if result_tool is not None:
        return call("return_result", **instance(result_tool.input_schema))
    return "Done, and checked."


@pytest.fixture
def user_dir() -> Any:
    folder = paths.config_dir() / "workflows"
    folder.mkdir(parents=True)
    return folder


def test_the_built_in_workflows_load_and_describe_themselves() -> None:
    catalog = load_saved()
    assert [w.name for w in catalog.workflows] == BUILT_INS
    assert {w.source for w in catalog.workflows} == {"built-in"}
    assert not catalog.problems
    understand = find_saved("understand")
    assert understand is not None and understand.meta.phases
    entry = understand.describe()
    assert entry.startswith("- `understand`: Map how something works")
    assert "When: Before changing unfamiliar code" in entry
    assert "Args: `question`: what to find out;" in entry
    assert find_saved("nope") is None


def test_user_workflows_are_found_checked_and_may_replace_built_ins(user_dir: Any) -> None:
    (user_dir / "mine.py").write_text(
        'meta = {"name": "mine", "description": "My own."}\nresult = 1\n'
    )
    (user_dir / "review.py").write_text(
        'meta = {"name": "review", "description": "My review."}\nresult = 2\n'
    )
    (user_dir / "broken.py").write_text(
        'meta = {"name": "broken", "description": "d"}\nimport os\nresult = 3\n'
    )
    (user_dir / "Bad Name.py").write_text('meta = {"name": "x", "description": "d"}\nresult = 4\n')
    (user_dir / "notes.txt").write_text("not a workflow")
    catalog = load_saved()
    by_name = {w.name: w for w in catalog.workflows}
    assert sorted(by_name) == ["implement", "mine", "research", "review", "understand"]
    assert by_name["mine"].source == "user"
    assert (by_name["review"].source, by_name["review"].meta.description) == ("user", "My review.")
    assert len(catalog.problems) == 2
    broken = next(p for p in catalog.problems if "broken.py" in p)
    assert "line 2: imports are not available" in broken
    assert any("Bad Name.py" in p and "file name" in p for p in catalog.problems)


@pytest.mark.parametrize(
    ("name", "args", "keys"),
    [
        ("understand", {"question": "How are sessions stored?"}, None),
        ("review", {}, {"reviewed", "files", "findings", "refuted"}),
        ("research", {"question": "Which KDL version?"}, {"report", "verified", "unverified"}),
        (
            "implement",
            {"task": "Add a flag.", "check": "true"},
            {"summary", "check", "check_passed", "review_findings"},
        ),
    ],
)
async def test_each_built_in_workflow_runs_to_completion(
    session: Session, scripted: ScriptedProvider, name: str, args: dict[str, Any], keys: Any
) -> None:
    scripted._router = answer_anything
    saved = find_saved(name)
    assert saved is not None
    root = session.new_node("Conversation", role="root")
    outcome = await WorkflowRun(session, parent=root, limits=Limits()).execute(saved.script, args)
    assert outcome.status == "completed", outcome.render()
    assert not outcome.failures and not outcome.warnings
    if keys is None:
        assert outcome.result == "Done, and checked."
    else:
        assert set(outcome.result) == keys
    assert outcome.phases == [p.title for p in saved.meta.phases]
    assert all(r.status == "done" for r in outcome.records)


async def test_implement_reports_whether_its_check_passed(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted._router = answer_anything
    saved = find_saved("implement")
    assert saved is not None
    root = session.new_node("Conversation", role="root")
    outcome = await WorkflowRun(session, parent=root, limits=Limits()).execute(
        saved.script, {"task": "Add a flag.", "check": "true"}
    )
    assert outcome.result["check_passed"] is True
    assert outcome.result["review_findings"]  # the scripted reviewers' finding was upheld
    assert any(r.label == "fix review findings" for r in outcome.records)


async def test_a_built_in_workflow_reports_a_missing_argument_by_line(
    session: Session, scripted: ScriptedProvider
) -> None:
    saved = find_saved("understand")
    assert saved is not None
    root = session.new_node("Conversation", role="root")
    outcome = await WorkflowRun(session, parent=root, limits=Limits()).execute(saved.script, {})
    assert outcome.status == "failed"
    assert "KeyError: 'question'" in outcome.error and 'args["question"]' in outcome.error
