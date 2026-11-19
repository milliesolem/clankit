"""The pattern library, exercised through scripts with scripted agents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from clankit.agent.events import Event, NodeInfo, ToolStarted
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest
from clankit.orchestration.workflow import Limits, Outcome, WorkflowRun, compile_script

Step = Any
META = 'meta = {"name": "p", "description": "Pattern test."}\n'


class Workers:
    """Answers each workflow agent by the first marker found in its task; a marker's steps
    are used in call order and the last one repeats."""

    def __init__(self) -> None:
        self.rules: list[tuple[str, list[Step]]] = []
        self.tasks: dict[str, list[str]] = {}

    def on(self, marker: str, *steps: Step) -> None:
        self.rules.append((marker, list(steps)))

    def __call__(self, request: ChatRequest) -> Step:
        task = request.messages[0].text
        for marker, steps in self.rules:
            if marker in task:
                seen = self.tasks.setdefault(marker, [])
                seen.append(task)
                return steps[min(len(seen) - 1, len(steps) - 1)]
        raise AssertionError(f"no script for a workflow agent whose task is {task[:160]!r}")


@pytest.fixture
def workers(scripted: ScriptedProvider) -> Workers:
    routed = Workers()
    scripted._router = routed
    return routed


async def run(session: Session, source: str, *, limits: Limits | None = None) -> Outcome:
    root = session.new_node("Conversation", role="root")
    return await WorkflowRun(session, parent=root, limits=limits or Limits()).execute(
        compile_script(META + source)
    )


def verdict(value: str) -> Any:
    return call("return_result", verdict=value, reason=f"checked: {value}")


def fails() -> ProviderError:
    return ProviderError("unavailable", kind="bad_request")


def workers_of(session: Session) -> list[NodeInfo]:
    return [n.info for n in session.view.nodes.values() if n.info.role == "worker"]


# --------------------------------------------------------------------------------------------
# adversarial_verify
# --------------------------------------------------------------------------------------------


async def test_claims_survive_only_with_a_majority_of_upheld_votes(
    session: Session, workers: Workers
) -> None:
    workers.on("<claim>\nA holds", verdict("upheld"))
    workers.on("<claim>\nB holds", verdict("upheld"), verdict("refuted"))
    workers.on("<claim>\nC holds", verdict("upheld"), verdict("upheld"), verdict("uncertain"))
    workers.on("<claim>\nD holds", verdict("uncertain"), verdict("uncertain"), verdict("upheld"))
    workers.on("<claim>\nE holds", verdict("upheld"), fails())
    outcome = await run(
        session,
        "result = await adversarial_verify(\n"
        "    ['A holds', 'B holds', 'C holds', 'D holds', 'E holds'], context='Python 3.12')\n",
    )
    assert outcome.result == ["A holds", "C holds"]
    assert {marker: len(tasks) for marker, tasks in workers.tasks.items()} == {
        f"<claim>\n{x} holds": 3 for x in "ABCDE"
    }
    assert outcome.logs[0] == (
        "adversarial_verify: 2 of 5 claims upheld by at least 2 of 3 skeptics"
    )
    assert outcome.logs[1].startswith("adversarial_verify dropped: B holds; D holds; E holds")
    assert outcome.logs[2].startswith("adversarial_verify: 2 skeptic(s) failed")
    task = workers.tasks["<claim>\nA holds"][0]
    assert "try to refute it" in task and "<context>\nPython 3.12\n</context>" in task


async def test_lenses_give_each_skeptic_its_own_angle(session: Session, workers: Workers) -> None:
    workers.on("<claim>\nfast", verdict("upheld"), verdict("refuted"))
    outcome = await run(
        session, "result = await adversarial_verify(['fast'], lenses=['speed', 'safety'])\n"
    )
    assert outcome.result == []  # 1 of 2 is not a majority
    angles = sorted(
        re.findall(r"from this angle: (\w+)", t)[0] for t in workers.tasks["<claim>\nfast"]
    )
    assert angles == ["safety", "speed"]


async def test_votes_default_to_the_configured_number_and_claims_keep_their_shape(
    session: Session, workers: Workers, config: Config
) -> None:
    other = ScriptedProvider(provider_id="other", router=workers)
    session.runtime.providers.register(other)
    config.models.roles["reviewer"] = "other/strict-model"
    workers.on('"id": 1', verdict("upheld"))
    outcome = await run(
        session,
        "result = await adversarial_verify([{'id': 1, 'text': 'x'}])\n",
        limits=Limits(verify_votes=1),
    )
    assert outcome.result == [{"id": 1, "text": "x"}]
    assert len(workers.tasks['"id": 1']) == 1
    assert [w.model for w in workers_of(session)] == ["other/strict-model"]


async def test_verify_handles_no_claims_and_rejects_bad_vote_counts(
    session: Session, workers: Workers
) -> None:
    empty = await run(session, "result = await adversarial_verify([])\n")
    assert empty.result == [] and not workers.tasks
    bad = await run(session, "result = await adversarial_verify(['x'], votes=0)\n")
    assert bad.status == "failed" and "votes must be a whole number from 1 to 9" in bad.error


# --------------------------------------------------------------------------------------------
# judge_panel
# --------------------------------------------------------------------------------------------


def judge(favourite: str, *, score: float = 9) -> Step:
    """A judge that scores the attempt containing `favourite` highest, by position."""

    def respond(request: ChatRequest) -> Any:
        attempts = re.findall(
            r'<attempt number="(\d+)">\n(.*?)\n</attempt>', request.messages[0].text, re.S
        )
        return call(
            "return_result",
            scores=[
                {"attempt": int(n), "score": score if favourite in text else 4, "reason": "r"}
                for n, text in attempts
            ],
        )

    return respond


async def test_the_panel_picks_the_best_attempt_and_undoes_each_judges_rotation(
    session: Session, workers: Workers
) -> None:
    for angle in ("alpha", "beta", "gamma"):
        workers.on(f"from this angle: {angle}", f"plan {angle}")
    workers.on("Several agents worked on the task", judge("plan beta"))
    outcome = await run(
        session, "result = await judge_panel('Design the cache.', ['alpha', 'beta', 'gamma'])\n"
    )
    panel = outcome.result
    assert (panel["best"], panel["index"], panel["angle"]) == ("plan beta", 1, "beta")
    assert panel["scores"] == [4.0, 9.0, 4.0]
    assert [a["result"] for a in panel["attempts"]] == ["plan alpha", "plan beta", "plan gamma"]
    firsts = {
        re.search(r'<attempt number="1">\n(.*?)\n', task).group(1)  # type: ignore[union-attr]
        for task in workers.tasks["Several agents worked on the task"]
    }
    assert firsts == {"plan alpha", "plan beta", "plan gamma"}  # every judge saw a new order
    assert (
        "<task>\nDesign the cache.\n</task>"
        in workers.tasks["Several agents worked on the task"][0]
    )


async def test_failed_attempts_are_left_out_of_judging(session: Session, workers: Workers) -> None:
    workers.on("from this angle: alpha", "plan alpha")
    workers.on("from this angle: beta", fails())
    workers.on("from this angle: gamma", "plan gamma")
    workers.on("Several agents worked on the task", judge("plan gamma"))
    outcome = await run(
        session,
        "result = await judge_panel('Design it.', ['alpha', 'beta', 'gamma'], judges=2)\n",
    )
    assert outcome.result["best"] == "plan gamma"
    assert outcome.result["scores"] == [4.0, None, 9.0]
    judged = workers.tasks["Several agents worked on the task"]
    assert len(judged) == 2 and all(t.count("<attempt ") == 2 for t in judged)


async def test_panel_edge_cases(session: Session, workers: Workers) -> None:
    workers.on("from this angle: only", "plan only")
    workers.on("from this angle: broken", fails())
    single = await run(session, "result = await judge_panel('Do it.', ['only', 'broken'])\n")
    assert single.result["best"] == "plan only" and "nothing to judge" in single.logs[0]
    assert "Several agents" not in workers.tasks

    nothing = await run(session, "result = await judge_panel('Do it.', ['broken', 'broken'])\n")
    assert nothing.result is None and nothing.logs == ["judge_panel: every attempt failed"]

    too_few = await run(session, "result = await judge_panel('Do it.', ['only'])\n")
    assert too_few.status == "failed" and "at least two angles" in too_few.error


async def test_when_every_judge_fails_the_first_attempt_is_returned(
    session: Session, workers: Workers
) -> None:
    workers.on("from this angle: a", "plan a")
    workers.on("from this angle: b", "plan b")
    workers.on("Several agents worked on the task", fails())
    outcome = await run(session, "result = await judge_panel('Do it.', ['a', 'b'])\n")
    assert outcome.result["best"] == "plan a" and outcome.result["scores"] == [None, None]
    assert any("no judge returned scores" in line for line in outcome.logs)


# --------------------------------------------------------------------------------------------
# loop_until_dry
# --------------------------------------------------------------------------------------------


def findings(*items: tuple[str, str]) -> Any:
    return call(
        "return_result",
        findings=[{"title": t, "location": where, "detail": "d"} for t, where in items],
    )


async def test_finders_run_until_rounds_stop_finding_anything_new(
    session: Session, workers: Workers
) -> None:
    workers.on(
        "search one",
        findings(("first", "a.py:1"), ("second", "a.py:2")),
        findings(("third", "b.py:9")),
        findings(),
    )
    workers.on("search two", findings(("second again", "a.py:2")), findings())
    outcome = await run(
        session, "result = await loop_until_dry(['search one', 'search two'], 'location')\n"
    )
    assert [f["title"] for f in outcome.result] == ["first", "second", "third"]
    assert len(workers.tasks["search one"]) == 4  # 2 productive rounds, then 2 dry ones
    second_round = workers.tasks["search one"][1]
    assert "Already found" in second_round and "- a.py:1\n- a.py:2" in second_round
    assert outcome.logs == [
        "loop_until_dry round 1: 2 new, 2 in total",
        "loop_until_dry round 2: 1 new, 3 in total",
        "loop_until_dry round 3: 0 new, 3 in total",
        "loop_until_dry round 4: 0 new, 3 in total",
    ]


async def test_the_round_limit_is_logged_and_keys_can_be_functions(
    session: Session, workers: Workers
) -> None:
    counter = iter(range(100))
    workers.on("search", lambda request: findings((f"Item {next(counter)}", "x")))
    outcome = await run(
        session,
        "result = await loop_until_dry(['search'], lambda f: f['title'].lower(), max_rounds=2)\n",
    )
    assert [f["title"] for f in outcome.result] == ["Item 0", "Item 1"]
    assert "stopped at the limit of 2 rounds" in outcome.logs[-1]


async def test_rounds_where_every_finder_failed_are_not_counted_as_dry(
    session: Session, workers: Workers
) -> None:
    workers.on("search", fails(), findings(("found", "a.py:1")), findings())
    outcome = await run(
        session,
        "result = await loop_until_dry(['search'], 'location', dry_rounds=1, max_rounds=5)\n",
    )
    assert [f["title"] for f in outcome.result] == ["found"]
    assert outcome.logs[0] == "loop_until_dry: every finder failed in round 1"


async def test_findings_without_a_key_are_skipped_and_counted(
    session: Session, workers: Workers
) -> None:
    workers.on("search", findings(("named", "a.py:1"), ("unnamed", "")), findings())
    outcome = await run(
        session, "result = await loop_until_dry(['search'], 'location', dry_rounds=1)\n"
    )
    assert [f["title"] for f in outcome.result] == ["named"]
    assert outcome.logs[-1] == "loop_until_dry: skipped 1 finding(s) without a usable key"


# --------------------------------------------------------------------------------------------
# completeness_critic
# --------------------------------------------------------------------------------------------


async def test_the_critic_returns_gaps_and_sees_the_report_and_task(
    session: Session, workers: Workers
) -> None:
    workers.on("Look for what is missing", call("return_result", gaps=["No tests.", "  "]))
    outcome = await run(
        session, "result = await completeness_critic({'done': ['parser']}, task='Build a parser')\n"
    )
    assert outcome.result == ["No tests."]
    task = workers.tasks["Look for what is missing"][0]
    assert '"done": [' in task and "<task>\nBuild a parser\n</task>" in task
    assert outcome.logs == ["completeness_critic: 1 gap(s) found"]


async def test_a_failed_critic_is_logged_not_mistaken_for_completeness(
    session: Session, workers: Workers
) -> None:
    workers.on("Look for what is missing", fails())
    outcome = await run(session, "result = await completeness_critic('report')\n")
    assert outcome.result == []
    assert outcome.logs == [
        "completeness_critic: the critic failed, so completeness was not checked"
    ]


# --------------------------------------------------------------------------------------------
# fix_until
# --------------------------------------------------------------------------------------------


async def test_fix_until_runs_the_check_and_an_agent_fixes_it(
    session: Session, workers: Workers, workspace: Path, events: list[Event]
) -> None:
    workers.on("This check fails", call("write", file_path="fixed.txt", content="ok"), "Fixed.")
    outcome = await run(session, "result = await fix_until('test -f fixed.txt', task='Fix it')\n")
    assert outcome.result is True and (workspace / "fixed.txt").exists()
    task = workers.tasks["This check fails"][0]
    assert "$ test -f fixed.txt" in task and "<task>\nFix it\n</task>" in task
    checks = [e for e in events if isinstance(e, ToolStarted) and e.tool == "check"]
    assert len(checks) == 2 and checks[0].node_id == next(
        n.info.id for n in session.view.nodes.values() if n.info.role == "workflow"
    )
    assert outcome.logs == ["fix_until: `test -f fixed.txt` passes after 1 fix round(s)"]


async def test_a_passing_check_needs_no_agent(session: Session, workers: Workers) -> None:
    outcome = await run(session, "result = await fix_until('true')\n")
    assert outcome.result is True and not workers.tasks


async def test_fix_until_gives_up_when_nothing_changes(session: Session, workers: Workers) -> None:
    workers.on("This check fails", "I could not find the cause.")
    outcome = await run(session, "result = await fix_until('false', max_stale=2)\n")
    assert outcome.result is False
    assert len(workers.tasks["This check fails"]) == 2
    assert "made no progress in 2 round(s) in a row" in outcome.logs[-1]


async def test_fix_until_stops_at_its_round_limit_even_while_progressing(
    session: Session, workers: Workers
) -> None:
    contents = iter(["one", "two", "three"])
    workers.on(
        "This check fails",
        lambda r: call("write", file_path="n.txt", content=next(contents)),
        "Changed it.",
        lambda r: call("write", file_path="n.txt", content=next(contents)),
        "Changed it.",
    )
    outcome = await run(
        session, "result = await fix_until('cat n.txt 2>/dev/null; exit 1', max_rounds=2)\n"
    )
    assert outcome.result is False
    assert "still fails after 2 fix round(s)" in outcome.logs[-1]


async def test_fix_until_reports_a_denied_check(
    session: Session, workers: Workers, config: Config
) -> None:
    config.permissions.mode = "read-only"
    outcome = await run(session, "result = await fix_until('true')\n")
    assert outcome.result is False
    assert outcome.logs == ["fix_until: `true` was not run: permission was denied"]
    assert not workers.tasks
