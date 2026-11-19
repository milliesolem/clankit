"""Escalate: a ladder of tiers, an objective gate, handovers, resets and the answer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from clankit.agent.events import Event, RunFinished
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, ChatResponse, Message, ModelRef, Usage
from clankit.orchestration import get_orchestrator
from clankit.orchestration.escalate import EscalateOrchestrator, Tier
from clankit.runtime import Runtime

Step = Any
Key = tuple[str, str | None, str | None]

CHECK = "grep -q fixed app.txt"
LATER = "<earlier_attempts>"
REVIEW = "Review the result against the task"
FILES_STAY = "It starts from the files as you leave them."


class Agents:
    """Routes each model call to the script of the agent that made it.

    Agents are told apart by a marker in their first user message, optionally narrowed to a
    model id and an effort (tiers of one ladder all see the user's request). Because later
    briefs quote the request near their top, the marker that occurs LATEST in the message
    wins. Each agent's script advances by one step per model call; unscripted calls fail.
    """

    def __init__(self) -> None:
        self.scripts: dict[Key, list[Step]] = {}
        self.calls: dict[Key, int] = {}
        self.requests: dict[Key, list[ChatRequest]] = {}

    def add(
        self, marker: str, *steps: Step, model: str | None = None, effort: str | None = None
    ) -> Key:
        key = (marker, model, effort)
        self.scripts[key] = list(steps)
        return key

    def __call__(self, request: ChatRequest) -> Step:
        first = request.messages[0].text
        found = [
            (first.find(key[0]), key[1] is not None, key[2] is not None, key)
            for key in self.scripts
            if key[0] in first
            and key[1] in (None, request.model)
            and key[2] in (None, request.effort)
        ]
        if not found:
            raise AssertionError(f"no script for {request.model}: {first[:200]!r}")
        key = max(found)[3]
        n = self.calls.get(key, 0)
        self.calls[key] = n + 1
        snapshot = ChatRequest(
            request.model,
            request.system,
            list(request.messages),
            list(request.tools),
            effort=request.effort,
        )
        self.requests.setdefault(key, []).append(snapshot)
        if n >= len(self.scripts[key]):
            raise AssertionError(f"agent {key!r} made more calls than scripted")
        return self.scripts[key][n]


def write(path: str, content: str) -> Any:
    return call("write", file_path=path, content=content)


def hand_over(**fields: str) -> Any:
    values = {
        "attempted": "Wrote app.txt.",
        "learned": "The check looks for the word fixed.",
        "failure": "I wrote the wrong word.",
        "files_state": "app.txt says broken.",
    }
    values.update(fields)
    return call("hand_over", **values)


def refusal(detail: str) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content=[]),
        stop_reason="refusal",
        usage=Usage(),
        detail=detail,
    )


@pytest.fixture
def config(config: Config) -> Config:
    config.orchestration.method = "escalate"
    config.orchestration.set_option("escalate", "tiers", "cheap/small, strong/big")
    config.orchestration.set_option("escalate", "check", CHECK)
    return config


@pytest.fixture
def agents(scripted: ScriptedProvider, runtime: Runtime) -> Agents:
    routed = Agents()
    scripted._router = routed
    runtime.providers.register(ScriptedProvider(provider_id="cheap", router=routed))
    runtime.providers.register(ScriptedProvider(provider_id="strong", router=routed))
    return routed


def nodes(session: Session) -> dict[str, tuple[str, str, str]]:
    """title -> (model, status, summary)"""
    return {
        n.info.title: (n.info.model, n.info.status, n.info.summary)
        for n in session.view.nodes.values()
    }


def first_message(agents: Agents, key: Key) -> str:
    return agents.requests[key][0].messages[0].text


async def test_a_tier_that_fails_the_check_hands_over_to_the_next(
    session: Session, agents: Agents, workspace: Path
) -> None:
    cheap = agents.add(
        "fix the app",
        write("app.txt", "broken\n"),
        "I fixed the app.",
        hand_over(),
        model="small",
    )
    strong = agents.add(
        LATER,
        call("read", file_path="app.txt"),
        write("app.txt", "fixed\n"),
        "Rewrote app.txt; the check passes.",
        model="big",
    )

    answer = await session.run("fix the app")

    assert (workspace / "app.txt").read_text() == "fixed\n"
    assert answer == (
        "Rewrote app.txt; the check passes.\n\n"
        f"Done by tier 2 of 2 (strong/big): `{CHECK}` passed. "
        "Not accepted before it: tier 1 of 2 (cheap/small)."
    )
    assert nodes(session) == {
        "Conversation": ("", "done", "done by tier 2 of 2 (strong/big)"),
        "Tier 1 of 2: cheap/small": (
            "cheap/small",
            "failed",
            f"not accepted: the check `{CHECK}` failed (exit code 1)",
        ),
        "Tier 2 of 2: strong/big": ("strong/big", "done", f"accepted: `{CHECK}` passed"),
    }

    request = agents.requests[cheap][2].messages[-1].text
    assert request.startswith("[Harness] Your work was not accepted: the check")
    assert "FAILED (exit code 1)" in request and FILES_STAY in request

    brief = first_message(agents, strong)
    assert brief.startswith("<user_request>\nfix the app\n</user_request>")
    assert "## Tier 1 of 2 (cheap/small)" in brief
    assert (
        "- Tried: Wrote app.txt." in brief and "- Why it failed: I wrote the wrong word." in brief
    )
    assert "- Files changed: app.txt  +1 -0 (new)" in brief and "FAILED (exit code 1)" in brief
    assert "Their changes are still on disk" in brief and "I fixed the app." not in brief

    cheap_tools = [t.name for t in agents.requests[cheap][0].tools]
    strong_tools = [t.name for t in agents.requests[strong][0].tools]
    assert "hand_over" in cheap_tools and "ask_user" in cheap_tools and "write" in cheap_tools
    assert "hand_over" not in strong_tools
    assert f"the harness runs `{CHECK}`" in agents.requests[cheap][0].system
    assert "a fresh agent on strong/big" in agents.requests[cheap][0].system

    conversation = [(i.kind, i.text) for i in session.view.conversation()]
    assert conversation == [("user", "fix the app"), ("assistant", answer)]


async def test_the_first_tier_that_passes_ends_the_ladder(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add("fix the app", write("app.txt", "fixed\n"), "Done.", model="small")
    answer = await session.run("fix the app")
    assert answer == f"Done.\n\nDone by tier 1 of 2 (cheap/small): `{CHECK}` passed."
    assert list(nodes(session)) == ["Conversation", "Tier 1 of 2: cheap/small"]
    root_statuses = [
        e.status  # type: ignore[attr-defined]
        for e in events
        if e.kind == "node_changed" and e.node_id == "n1" and e.status  # type: ignore[attr-defined]
    ]
    assert root_statuses == ["waiting", "done"]


async def test_reset_between_tiers_restores_the_files_and_undo_still_works(
    session: Session, agents: Agents, config: Config, workspace: Path
) -> None:
    config.orchestration.set_option("escalate", "reset-between-tiers", True)
    (workspace / "app.txt").write_text("original\n")
    seen: dict[str, Any] = {}

    def look_at_disk(request: ChatRequest) -> Any:
        seen["app"] = (workspace / "app.txt").read_text()
        seen["extra"] = (workspace / "extra.txt").exists()
        return call("read", file_path="app.txt")

    cheap = agents.add(
        "fix the app",
        call("read", file_path="app.txt"),
        write("app.txt", "broken\n"),
        write("extra.txt", "scratch\n"),
        "Done.",
        hand_over(),
        model="small",
    )
    strong = agents.add(LATER, look_at_disk, write("app.txt", "fixed\n"), "Fixed it.", model="big")

    await session.run("fix the app")

    assert seen == {"app": "original\n", "extra": False}
    assert (workspace / "app.txt").read_text() == "fixed\n"
    assert "restore the files you changed" in agents.requests[cheap][4].messages[-1].text
    assert "The harness restored the files they changed" in first_message(agents, strong)
    assert session.undo() == ["app.txt", "extra.txt"]
    assert (workspace / "app.txt").read_text() == "original\n"
    assert not (workspace / "extra.txt").exists()


async def test_a_single_model_climbs_reasoning_effort(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("escalate", "tiers", "")
    low = agents.add("fix the app", write("app.txt", "nope\n"), "Done.", hand_over(), effort="low")
    medium = agents.add(
        LATER,
        call("read", file_path="app.txt"),
        write("app.txt", "fixed\n"),
        "Done properly.",
        effort="medium",
    )

    answer = await session.run("fix the app")

    assert [r.effort for r in agents.requests[low]] == ["low"] * 3
    assert {r.effort for r in agents.requests[medium]} == {"medium"}
    assert answer.endswith(
        f"Done by tier 2 of 3 (scripted/test-model at medium effort): `{CHECK}` passed. "
        "Not accepted before it: tier 1 of 3 (scripted/test-model at low effort)."
    )
    assert list(nodes(session))[1:] == [
        "Tier 1 of 3: scripted/test-model at low effort",
        "Tier 2 of 3: scripted/test-model at medium effort",
    ]


async def test_each_tier_can_get_several_fresh_attempts(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("escalate", "attempts-per-tier", 2)
    first = agents.add(
        "fix the app", write("app.txt", "broken\n"), "Done.", hand_over(), model="small"
    )
    second = agents.add(
        LATER,
        call("read", file_path="app.txt"),
        write("app.txt", "fixed\n"),
        "Done.",
        model="small",
    )
    answer = await session.run("fix the app")
    assert "a fresh agent on cheap/small" in agents.requests[first][0].system
    assert "a fresh agent on strong/big" in agents.requests[second][0].system
    assert answer.endswith(
        f"Done by tier 1 of 2 (cheap/small), attempt 2: `{CHECK}` passed. "
        "Not accepted before it: tier 1 of 2 (cheap/small)."
    )
    assert list(nodes(session))[1:] == [
        "Tier 1 of 2: cheap/small",
        "Tier 1 of 2: cheap/small (attempt 2)",
    ]


def test_the_ladder_is_derived_from_the_configured_roles(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("escalate", "tiers", "")
    config.models.roles = {"utility": "cheap/small", "subagent": "strong/big"}
    escalate = EscalateOrchestrator(session)
    assert escalate.ladder() == [
        Tier(ModelRef("cheap", "small")),
        Tier(ModelRef("strong", "big")),
        Tier(ModelRef("scripted", "test-model")),
    ]
    config.models.roles = {"utility": "nowhere/model", "subagent": "scripted/test-model"}
    assert [str(t) for t in escalate.ladder()] == [
        f"scripted/test-model at {effort} effort" for effort in ("low", "medium", "high")
    ]


def test_unusable_tiers_are_skipped_with_a_notice(
    session: Session, agents: Agents, config: Config, events: list[Event]
) -> None:
    escalate = EscalateOrchestrator(session)
    config.orchestration.set_option("escalate", "tiers", "nowhere/x, no-slash, cheap/small@high")
    assert escalate.ladder() == [Tier(ModelRef("cheap", "small"), "high")]
    notice = events[-1].text  # type: ignore[attr-defined]
    assert "nowhere/x (the provider 'nowhere' is not configured)" in notice
    assert "'no-slash' is not written as provider/model" in notice

    config.orchestration.set_option("escalate", "tiers", "nowhere/x")
    assert escalate.ladder()[0] == Tier(ModelRef("scripted", "test-model"), "low")
    assert "Using the utility, subagent and session models instead." in events[-1].text  # type: ignore[attr-defined]


async def test_a_question_is_answered_by_the_first_tier_without_the_gate(
    session: Session, agents: Agents, config: Config, workspace: Path
) -> None:
    config.orchestration.set_option("escalate", "check", "touch checked; false")
    agents.add("what does the app do", "It prints hello.", model="small")
    assert await session.run("what does the app do") == "It prints hello."
    assert not (workspace / "checked").exists()
    assert nodes(session)["Tier 1 of 2: cheap/small"] == (
        "cheap/small",
        "done",
        "answered without changing files",
    )
    assert nodes(session)["Conversation"][2] == "answered by tier 1 of 2 (cheap/small)"


async def test_without_a_check_an_independent_review_is_the_gate(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("escalate", "check", "")
    finding = {
        "severity": "major",
        "location": "app.txt:1",
        "problem": "It says broken, not fixed.",
        "evidence": "app.txt line 1 reads 'broken'",
    }
    agents.add(
        REVIEW,
        call("submit_review", verdict="changes_needed", findings=[finding]),
        call("submit_review", verdict="approve", findings=[]),
    )
    cheap = agents.add(
        "fix the app", write("app.txt", "broken\n"), "Done.", hand_over(), model="small"
    )
    strong = agents.add(
        LATER, call("read", file_path="app.txt"), write("app.txt", "fixed\n"), "Done.", model="big"
    )

    answer = await session.run("fix the app")

    assert "an independent reviewer reads your changes" in agents.requests[cheap][0].system
    assert "Review findings:" in first_message(agents, strong)
    assert "It says broken, not fixed." in first_message(agents, strong)
    assert answer.endswith(
        "Done by tier 2 of 2 (strong/big): an independent review found no blocking problems. "
        "Not accepted before it: tier 1 of 2 (cheap/small)."
    )
    reviewers = [n.info for n in session.view.nodes.values() if n.info.role == "reviewer"]
    assert [session.node(r.parent_id or "").title for r in reviewers] == [
        "Tier 1 of 2: cheap/small",
        "Tier 2 of 2: strong/big",
    ]


async def test_an_attempt_that_reaches_the_turn_limit_is_escalated(
    session: Session, agents: Agents, config: Config, workspace: Path
) -> None:
    config.orchestration.set_option("escalate", "max-turns-per-tier", 5)
    (workspace / "app.txt").write_text("hello\n")
    looking = [call("grep", pattern=f"x{i}") for i in range(5)]
    cheap = agents.add("explain the app", *looking, hand_over(), model="small")
    agents.add(LATER, "It prints hello.", model="big")

    answer = await session.run("explain the app")

    assert answer == "It prints hello."
    assert len(agents.requests[cheap]) == 6
    assert "the limit of 5 model calls" in agents.requests[cheap][5].messages[-1].text
    assert nodes(session)["Tier 1 of 2: cheap/small"][1:] == (
        "failed",
        "not accepted: it reached the limit of 5 model calls without a result",
    )


async def test_an_agent_can_hand_over_without_being_checked(
    session: Session, agents: Agents, config: Config, workspace: Path
) -> None:
    config.orchestration.set_option("escalate", "check", "touch checked; grep -q fixed app.txt")
    cheap = agents.add("fix the app", hand_over(failure="This is beyond me."), model="small")
    strong = agents.add(LATER, write("app.txt", "fixed\n"), "Fixed.", model="big")
    await session.run("fix the app")
    assert len(agents.requests[cheap]) == 1
    assert "- Why it failed: This is beyond me." in first_message(agents, strong)
    assert nodes(session)["Tier 1 of 2: cheap/small"][2] == (
        "not accepted: it handed over without finishing"
    )
    assert (workspace / "checked").exists()  # only the second tier's work was checked


async def test_a_missing_handover_falls_back_to_the_harness_record(
    session: Session, agents: Agents
) -> None:
    cheap = agents.add(
        "fix the app",
        write("app.txt", "broken\n"),
        "Done.",
        "There is nothing to hand over.",
        "Still nothing.",
        model="small",
    )
    strong = agents.add(LATER, "I could not fix it either.", model="big")
    await session.run("fix the app")
    reminder = agents.requests[cheap][3].messages[-1].text
    assert "without calling `hand_over`" in reminder
    brief = first_message(agents, strong)
    assert "It wrote no handover." in brief and "FAILED (exit code 1)" in brief


async def test_a_later_tier_that_changes_nothing_is_still_checked(
    session: Session, agents: Agents
) -> None:
    agents.add("fix the app", write("app.txt", "broken\n"), "Done.", hand_over(), model="small")
    agents.add(LATER, "The app is already fine.", model="big")
    answer = await session.run("fix the app")
    assert answer.startswith("The app is already fine.\n\nNo tier's work was accepted.")
    assert nodes(session)["Tier 2 of 2: strong/big"][2] == (
        f"not accepted: the check `{CHECK}` failed (exit code 1)"
    )


async def test_when_no_tier_passes_the_last_result_is_returned(
    session: Session, agents: Agents
) -> None:
    agents.add("fix the app", write("app.txt", "broken\n"), "Done.", hand_over(), model="small")
    strong = agents.add(
        LATER, call("read", file_path="app.txt"), write("app.txt", "worse\n"), "Done.", model="big"
    )
    answer = await session.run("fix the app")
    assert answer == (
        "Done.\n\nNo tier's work was accepted. This is the result of tier 2 of 2 (strong/big), "
        f"the last, and the check `{CHECK}` failed (exit code 1). Its changes are in place; "
        "/undo reverts this run's file changes."
    )
    assert len(agents.requests[strong]) == 3  # the last tier is not asked for a handover
    assert nodes(session)["Conversation"][1:] == ("failed", "no tier's work was accepted")


async def test_a_refusal_ends_the_ladder(session: Session, agents: Agents) -> None:
    agents.add("do the thing", refusal("Not something I can help with."), model="small")
    answer = await session.run("do the thing")
    assert answer == "The model declined this request. Not something I can help with."
    assert list(nodes(session)) == ["Conversation", "Tier 1 of 2: cheap/small"]
    assert nodes(session)["Tier 1 of 2: cheap/small"][1:] == (
        "failed",
        "the model declined the request",
    )


async def test_a_model_error_fails_the_run(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add("fix the app", ProviderError("invalid API key", kind="auth"), model="small")
    assert await session.run("fix the app") == ""
    finished = events[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    assert "invalid API key" in finished.error
    _, status, summary = nodes(session)["Tier 1 of 2: cheap/small"]
    assert status == "failed" and summary.startswith("model error:")
    assert "Tier 2 of 2: strong/big" not in nodes(session)


async def test_the_run_budget_stops_the_ladder(
    runtime: Runtime, agents: Agents, config: Config, workspace: Path
) -> None:
    session = runtime.create_session(workspace, interactive=False)
    config.agent.run_budget = 8  # every scripted call produces 5 output tokens
    agents.add(
        "fix the app",
        write("app.txt", "broken\n"),
        call("read", file_path="app.txt"),
        "Done.",
        model="small",
    )
    finished: list[RunFinished] = []
    session.bus.subscribe(lambda e: finished.append(e) if isinstance(e, RunFinished) else None)
    await session.run("fix the app")
    assert finished[-1].status == "failed" and "budget" in finished[-1].error
    assert "Tier 2 of 2: strong/big" not in nodes(session)


async def test_cancelling_a_tier_cancels_the_run(session: Session, agents: Agents) -> None:
    cheap = session.runtime.providers.get("cheap")
    original = cheap.stream

    async def stalling(request: ChatRequest) -> AsyncIterator[Any]:
        await asyncio.sleep(60)
        async for event in original(request):
            yield event

    cheap.stream = stalling  # type: ignore[method-assign]
    task = session.start("fix the app")
    for _ in range(200):
        await asyncio.sleep(0.005)
        if len(session.view.nodes) == 2:
            break
    session.cancel()
    await task
    assert {status for _, status, _ in nodes(session).values()} == {"cancelled"}


async def test_steering_reaches_the_working_tier_and_every_later_one(
    session: Session, agents: Agents
) -> None:
    def steer_then_write(request: ChatRequest) -> Any:
        session.steer("Use the word fixed.")
        return write("app.txt", "broken\n")

    cheap = agents.add("fix the app", steer_then_write, "Done.", hand_over(), model="small")
    strong = agents.add(LATER, write("app.txt", "fixed\n"), "Done.", model="big")
    await session.run("fix the app")
    assert "Use the word fixed." in agents.requests[cheap][1].messages[-1].text
    assert "the user added:\nUse the word fixed." in first_message(agents, strong)


async def test_a_denied_check_accepts_the_work_unverified(
    session: Session, agents: Agents, config: Config
) -> None:
    config.permissions.deny = ["shell:grep *"]
    agents.add("fix the app", write("app.txt", "broken\n"), "Done.", model="small")
    answer = await session.run("fix the app")
    assert answer.endswith(
        "Done by tier 1 of 2 (cheap/small), unverified: the check was not run because "
        "permission was denied."
    )
    assert nodes(session)["Tier 1 of 2: cheap/small"][1] == "done"


async def test_a_later_prompt_starts_fresh_with_a_recap(session: Session, agents: Agents) -> None:
    agents.add("what does the app do", "It prints hello.", model="small")
    agents.add("and why", "Because main() says so.", model="small")
    await session.run("what does the app do")
    await session.run("and why")
    second = first_message(agents, ("and why", "small", None))
    assert second.startswith("Earlier in this session:") and "It prints hello." in second


def test_escalate_is_registered_with_bounded_settings() -> None:
    cls = get_orchestrator("escalate")
    assert cls is EscalateOrchestrator and cls.title == "Escalate"
    settings = {s.key: s for s in cls.settings}
    assert set(settings) == {
        "tiers",
        "check",
        "attempts-per-tier",
        "reset-between-tiers",
        "max-turns-per-tier",
    }
    assert settings["attempts-per-tier"].maximum == 5
    assert settings["max-turns-per-tier"].minimum == 5
    assert "weaker gate" in cls.guidance
