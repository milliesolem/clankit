"""The Review Loop orchestrator: clean-context review rounds, harness checks, reviewer models."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from clankit.agent.events import Event, NodeInfo, Notice, RunFinished, ToolFinished
from clankit.agent.session import Session
from clankit.agent.state import Item
from clankit.config.catalog import CATALOG
from clankit.config.schema import Config, ModelEntry, ProviderConfig
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, ChatResponse, Message, ModelInfo, ModelRef, Usage
from clankit.orchestration.review_loop import model_family, strongest_other_family, visible_check
from clankit.runtime import Runtime

Step = Any

PROMPT = "add a constant to f.py"
REVIEWER = "Review the result against the task"


class Agents:
    """Routes each model call to the script of the agent that made it.

    Agents are told apart by a marker in their first user message; the marker that occurs
    LATEST in the message wins, because briefs quote the user's request near their top. Each
    agent's script advances by one step per model call; unscripted calls fail loudly.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, list[Step]] = {}
        self.calls: dict[str, int] = {}
        self.requests: dict[str, list[ChatRequest]] = {}

    def add(self, marker: str, *steps: Step) -> None:
        self.scripts[marker] = list(steps)

    def __call__(self, request: ChatRequest) -> Step:
        first = request.messages[0].text
        found = [(first.find(m), m) for m in self.scripts if m in first]
        if not found:
            raise AssertionError(f"no script for an agent whose task starts: {first[:200]!r}")
        marker = max(found)[1]
        n = self.calls.get(marker, 0)
        self.calls[marker] = n + 1
        snapshot = ChatRequest(
            request.model, request.system, list(request.messages), list(request.tools)
        )
        self.requests.setdefault(marker, []).append(snapshot)
        if n >= len(self.scripts[marker]):
            raise AssertionError(f"agent {marker!r} made more calls than scripted")
        return self.scripts[marker][n]


@pytest.fixture(autouse=True)
def no_catalog_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real API keys in the environment from adding catalog models to the choice."""
    for provider in CATALOG.values():
        if provider.api_key_env:
            monkeypatch.delenv(provider.api_key_env, raising=False)


@pytest.fixture
def config(config: Config) -> Config:
    config.orchestration.method = "review-loop"
    return config


@pytest.fixture
def agents(scripted: ScriptedProvider) -> Agents:
    routed = Agents()
    scripted._router = routed
    return routed


def finding(severity: str, problem: str, evidence: str = "f.py:1 shows X = 1") -> dict[str, str]:
    return {"severity": severity, "location": "f.py:1", "problem": problem, "evidence": evidence}


def verdict(*findings: dict[str, str]) -> Step:
    return call(
        "submit_review",
        verdict="changes_needed" if findings else "approve",
        findings=list(findings),
    )


def refusal(detail: str) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content=[]),
        stop_reason="refusal",
        usage=Usage(),
        detail=detail,
    )


def nodes(session: Session) -> dict[str, NodeInfo]:
    return {view.info.title: view.info for view in session.view.nodes.values()}


def checks(session: Session) -> list[Item]:
    root = session.view.nodes[session.view.roots[-1]]
    return [item for item in root.items if item.kind == "tool" and item.tool == "check"]


def add_provider(
    session: Session, agents: Agents, provider_id: str, *models: ModelEntry
) -> ScriptedProvider:
    """A second provider whose models (and prices) are declared in the configuration."""
    config = session.config
    config.providers[provider_id] = ProviderConfig(
        id=provider_id, api_key="test-key", base_url="http://127.0.0.1:9", models=list(models)
    )
    session.runtime.providers.reconfigure(config)
    provider = ScriptedProvider(provider_id=provider_id, router=agents)
    session.runtime.providers.register(provider)
    return provider


# --------------------------------------------------------------------------------------------
# Rounds
# --------------------------------------------------------------------------------------------


async def test_blocking_findings_go_back_and_a_fresh_reviewer_looks_again(
    session: Session, agents: Agents, workspace: Path
) -> None:
    agents.add(
        PROMPT,
        call("write", file_path="f.py", content="X = 1\n"),
        "Added X to f.py; it is correct.",
        call("write", file_path="f.py", content="X = 2\n"),
        "Fixed the value; rejected nothing.",
    )
    agents.add(
        REVIEWER,
        verdict(finding("major", "X must be 2."), finding("minor", "Name is terse.")),
        verdict(),
    )

    answer = await session.run(PROMPT)

    assert answer == (
        "Fixed the value; rejected nothing.\n\n"
        "Independent review (scripted/test-model):\n"
        "- Round 1: 1 blocking finding(s) handed back for fixing; 1 minor not handed back.\n"
        "- Round 2: no problems found."
    )
    assert (workspace / "f.py").read_text() == "X = 2\n"
    tree = nodes(session)
    first, second = tree["Review, round 1"], tree["Review, round 2"]
    root = tree["Conversation"]
    assert (first.role, first.status, first.parent_id) == ("reviewer", "done", root.id)
    assert first.summary == "2 finding(s), 1 blocking."
    assert (second.status, second.summary) == ("done", "No problems found.")
    assert root.status == "done" and root.summary.startswith("Round 1: 1 blocking")

    handed = agents.requests[PROMPT][2].messages[-1].text
    assert handed.startswith("[Independent review]") and "X must be 2." in handed
    assert "Name is terse." not in handed

    review_one = agents.requests[REVIEWER][0]
    brief = review_one.messages[0].text
    assert f"<task>\n{PROMPT}" in brief and "+X = 1" in brief
    assert "it is correct" not in brief  # never the generator's own account
    assert {"write", "edit"}.isdisjoint(t.name for t in review_one.tools)
    review_two = agents.requests[REVIEWER][1].messages[0].text
    assert "An earlier review reported these findings" in review_two
    assert "X must be 2." in review_two and "+X = 2" in review_two
    assert checks(session) == []


async def test_nothing_is_reviewed_when_no_file_changed(session: Session, agents: Agents) -> None:
    agents.add("what does f.py do", "It defines X.")

    assert await session.run("what does f.py do") == "It defines X."

    assert list(nodes(session)) == ["Conversation"]
    assert nodes(session)["Conversation"].summary == "No files changed, so nothing was reviewed."


async def test_the_round_limit_leaves_the_last_fixes_unreviewed(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("review-loop", "max-rounds", 2)
    agents.add(
        PROMPT,
        call("write", file_path="f.py", content="X = 1\n"),
        "v1",
        call("write", file_path="f.py", content="X = 2\n"),
        "v2",
        call("write", file_path="f.py", content="X = 3\n"),
        "v3",
    )
    agents.add(
        REVIEWER, verdict(finding("major", "Wrong.")), verdict(finding("critical", "Still wrong."))
    )

    answer = await session.run(PROMPT)

    assert answer.startswith("v3\n\n")
    assert "- Round 2: 1 blocking finding(s) handed back for fixing." in answer
    assert answer.endswith("- The fixes for round 2 were not reviewed again (the round limit).")
    assert agents.calls[REVIEWER] == 2


async def test_minor_findings_go_back_only_when_asked_and_need_no_new_round(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("review-loop", "include-minor", True)
    agents.add(
        PROMPT,
        call("write", file_path="f.py", content="X = 1\n"),
        "Done.",
        "Renamed as suggested.",
    )
    agents.add(REVIEWER, verdict(finding("minor", "Name is terse.")))

    answer = await session.run(PROMPT)

    assert answer.endswith("- Round 1: 1 minor finding(s) handed back for fixing.")
    assert "Name is terse." in agents.requests[PROMPT][2].messages[-1].text
    assert agents.calls[REVIEWER] == 1


async def test_a_reviewer_that_never_submits_is_reported(session: Session, agents: Agents) -> None:
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")
    agents.add(REVIEWER, "Looks fine.", "Really fine.", "Fine.")

    answer = await session.run(PROMPT)

    assert answer == (
        "Done.\n\nIndependent review (scripted/test-model):\n"
        "- Round 1: the reviewer did not submit a review, so the latest changes are unreviewed."
    )
    reviewer = nodes(session)["Review, round 1"]
    assert (reviewer.status, reviewer.summary) == (
        "failed",
        "The reviewer did not submit a review.",
    )


async def test_findings_without_evidence_are_dropped(session: Session, agents: Agents) -> None:
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")
    agents.add(REVIEWER, verdict(finding("critical", "Feels wrong.", evidence="")))

    answer = await session.run(PROMPT)

    assert (
        answer
        == "Done.\n\nIndependent review (scripted/test-model):\n- Round 1: no problems found."
    )
    assert agents.calls[PROMPT] == 2


# --------------------------------------------------------------------------------------------
# Harness checks
# --------------------------------------------------------------------------------------------


async def test_the_check_is_evidence_and_a_fix_that_breaks_it_is_sent_back(
    session: Session, agents: Agents, config: Config, workspace: Path, events: list[Event]
) -> None:
    (workspace / "check.sh").write_text("grep -q good f.txt\n")
    config.orchestration.set_option("review-loop", "check", "sh check.sh")
    config.orchestration.set_option("review-loop", "max-rounds", 1)
    agents.add(
        PROMPT,
        call("write", file_path="f.txt", content="good\n"),
        "Wrote f.txt.",
        call("write", file_path="f.txt", content="bad\n"),
        "Applied the finding.",
        call("write", file_path="f.txt", content="good and better\n"),
        "Repaired the regression.",
    )
    agents.add(REVIEWER, verdict(finding("major", "Needs more.")))

    answer = await session.run(PROMPT)

    brief = agents.requests[REVIEWER][0].messages[0].text
    assert "A check the harness ran on these changes:\n$ sh check.sh\npassed" in brief
    regression = agents.requests[PROMPT][4].messages[-1].text
    assert regression.startswith("[Harness check after your fixes]")
    assert "$ sh check.sh\nFAILED (exit code 1)" in regression
    assert [c.is_error for c in checks(session)] == [False, True, False]
    assert any(isinstance(e, Notice) and "broke `sh check.sh`" in e.text for e in events)
    assert answer.startswith("Repaired the regression.")
    assert "- The fixes for round 1 were not reviewed again (the round limit)." in answer
    assert answer.endswith("- Check `sh check.sh`: passed.")


async def test_a_regression_that_stays_broken_is_flagged(
    session: Session, agents: Agents, config: Config, workspace: Path
) -> None:
    (workspace / "check.sh").write_text("grep -q good f.txt\n")
    config.orchestration.set_option("review-loop", "check", "sh check.sh")
    config.orchestration.set_option("review-loop", "max-rounds", 1)
    agents.add(
        PROMPT,
        call("write", file_path="f.txt", content="good\n"),
        "Wrote f.txt.",
        call("write", file_path="f.txt", content="bad\n"),
        "Applied the finding.",
        "I could not repair it.",
    )
    agents.add(REVIEWER, verdict(finding("major", "Needs more.")))

    answer = await session.run(PROMPT)

    assert "- The check passed before the fixes of round 1 and still fails after them." in answer
    assert answer.endswith("- Check `sh check.sh`: failed (exit code 1).")


# --------------------------------------------------------------------------------------------
# Reviewer models
# --------------------------------------------------------------------------------------------


async def test_the_reviewer_model_setting_is_used(
    session: Session, agents: Agents, config: Config
) -> None:
    other = add_provider(session, agents, "other", ModelEntry("careful-model"))
    config.orchestration.set_option("review-loop", "reviewer-model", "other/careful-model")
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")
    agents.add(REVIEWER, verdict())

    answer = await session.run(PROMPT)

    assert nodes(session)["Review, round 1"].model == "other/careful-model"
    assert [r.model for r in other.requests] == ["careful-model"]
    assert "Independent review (other/careful-model):" in answer
    assert nodes(session)["Conversation"].model == "scripted/test-model"


async def test_another_family_is_preferred_only_when_asked(
    session: Session, agents: Agents, config: Config
) -> None:
    add_provider(
        session,
        agents,
        "other",
        ModelEntry("small-model", output_price=1.0),
        ModelEntry("big-model", output_price=15.0),
        ModelEntry("free-model", free=True),
        ModelEntry("text-only", tools=False, output_price=99.0),
    )
    agents.add(
        PROMPT,
        call("write", file_path="f.py", content="X = 1\n"),
        "Done.",
        call("write", file_path="f.py", content="X = 2\n"),
        "Done again.",
    )
    agents.add(REVIEWER, verdict(), verdict())

    await session.run(PROMPT)
    assert nodes(session)["Review, round 1"].model == "scripted/test-model"

    config.orchestration.set_option("review-loop", "prefer-other-family", True)
    await session.run("change it again")
    reviewers = [n for n in session.view.nodes.values() if n.info.role == "reviewer"]
    assert reviewers[-1].info.model == "other/big-model"


async def test_a_clearly_cheaper_family_does_not_replace_the_generators_own(
    session: Session, agents: Agents, config: Config
) -> None:
    config.providers["scripted"] = ProviderConfig(
        id="scripted", api_key="k", models=[ModelEntry("test-model", output_price=20.0)]
    )
    add_provider(session, agents, "other", ModelEntry("cheap-model", output_price=2.0))
    config.orchestration.set_option("review-loop", "prefer-other-family", True)
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")
    agents.add(REVIEWER, verdict())

    await session.run(PROMPT)

    assert nodes(session)["Review, round 1"].model == "scripted/test-model"


async def test_an_unusable_reviewer_model_setting_falls_back_to_the_role(
    session: Session, agents: Agents, config: Config, events: list[Event]
) -> None:
    config.orchestration.set_option("review-loop", "reviewer-model", "nowhere/model-x")
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")
    agents.add(REVIEWER, verdict())

    await session.run(PROMPT)

    assert nodes(session)["Review, round 1"].model == "scripted/test-model"
    assert any(
        isinstance(e, Notice) and e.text.startswith("Ignoring the reviewer-model setting")
        for e in events
    )


def test_families_are_recognised_across_providers() -> None:
    assert model_family(ModelRef("openrouter", "anthropic/claude-sonnet-5-5")) == "anthropic"
    assert model_family(ModelRef("anthropic", "claude-opus-5-5")) == "anthropic"
    assert model_family(ModelRef("groq", "openai/gpt-oss-120b")) == "openai"
    assert model_family(ModelRef("ollama", "gemma4:12b")) == "google"
    assert model_family(ModelRef("custom", "house-model")) == "custom"

    models = [
        ModelInfo("anthropic", "claude-opus-5-5", output_price=20.0),
        ModelInfo("openrouter", "anthropic/claude-fable-5-2", output_price=40.0),
        ModelInfo("deepseek", "deepseek-v4-pro", output_price=3.96),
        ModelInfo("gemini", "gemini-4-hydrogen"),
    ]
    generator = ModelRef("anthropic", "claude-opus-5-5")
    assert strongest_other_family(models, generator) is None  # DeepSeek is far cheaper
    models.append(ModelInfo("openai", "gpt-6", output_price=30.0))
    assert strongest_other_family(models, generator) == ModelRef("openai", "gpt-6")
    assert strongest_other_family(models[2:4], ModelRef("x", "unpriced")) == ModelRef(
        "deepseek", "deepseek-v4-pro"
    )


def test_an_aggregators_vendor_prefix_names_the_family() -> None:
    assert model_family(ModelRef("openrouter", "openai/o3-pro")) == "openai"
    assert model_family(ModelRef("openai", "o3")) == "openai"
    assert model_family(ModelRef("openrouter", "meta-llama/llama-4-maverick")) == "meta"

    generator = ModelRef("openai", "gpt-6")
    models = [
        ModelInfo("openai", "gpt-6", output_price=30.0),
        ModelInfo("openrouter", "openai/o3-pro", output_price=80.0),
        ModelInfo("anthropic", "claude-opus-5-5", output_price=25.0),
    ]
    assert strongest_other_family(models, generator) == ModelRef("anthropic", "claude-opus-5-5")


async def test_a_check_that_cannot_start_is_not_left_running(
    session: Session, workspace: Path, events: list[Event]
) -> None:
    node = session.new_node("Conversation", role="root")
    ctx = session.tool_context(node)
    await asyncio.to_thread(workspace.rmdir)  # the project folder went away under the session

    with pytest.raises(OSError):
        await visible_check(session, ctx, "true")

    (finished,) = [e for e in events if isinstance(e, ToolFinished)]
    assert finished.is_error and finished.output.startswith("Could not run the check:")
    (item,) = checks(session)
    assert item.done and item.is_error


# --------------------------------------------------------------------------------------------
# Context, refusals, failures
# --------------------------------------------------------------------------------------------


async def test_the_reviewer_gets_earlier_requests_but_not_earlier_answers(
    session: Session, agents: Agents
) -> None:
    agents.add(
        "explain f.py",
        "The secret answer about f.py.",
        call("write", file_path="f.py", content="X = 1\n"),
        "Done.",
    )
    agents.add(REVIEWER, verdict())

    await session.run("explain f.py")
    await session.run(PROMPT)

    brief = agents.requests[REVIEWER][0].messages[0].text
    assert "Earlier requests in this session, already handled" in brief
    assert "- explain f.py" in brief and "secret answer" not in brief


@pytest.mark.parametrize("when", ["first", "fixing"])
async def test_a_refusal_ends_the_request(session: Session, agents: Agents, when: str) -> None:
    declined = refusal("Not able to help.")
    if when == "first":
        agents.add(PROMPT, declined)
    else:
        agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.", declined)
        agents.add(REVIEWER, verdict(finding("major", "Wrong.")))

    assert await session.run(PROMPT) == "The model declined this request. Not able to help."
    root = nodes(session)["Conversation"]
    assert (root.status, root.summary) == ("failed", "The model declined the request.")


async def test_a_reviewer_provider_failure_ends_the_run_as_failed(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")
    agents.add(REVIEWER, ProviderError("overloaded", kind="bad_request", provider="scripted"))

    assert await session.run(PROMPT) == ""

    finished = events[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    assert {n.status for n in nodes(session).values()} == {"failed"}


async def test_the_run_budget_stops_the_review(
    runtime: Runtime, workspace: Path, scripted: ScriptedProvider
) -> None:
    session = runtime.create_session(workspace, interactive=False)
    session.config.agent.run_budget = 10  # each scripted call generates 5 output tokens
    agents = Agents()
    scripted._router = agents
    seen: list[Event] = []
    session.bus.subscribe(seen.append)
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")
    agents.add(REVIEWER, verdict())

    assert await session.run(PROMPT) == ""

    finished = seen[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    assert "budget" in finished.error and REVIEWER not in agents.calls
    assert nodes(session)["Review, round 1"].status == "failed"


async def test_cancelling_during_review_cancels_every_node(
    session: Session, agents: Agents, scripted: ScriptedProvider
) -> None:
    original = scripted.stream

    async def stalling(request: ChatRequest):  # type: ignore[no-untyped-def]
        if REVIEWER in request.messages[0].text:
            await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    agents.add(PROMPT, call("write", file_path="f.py", content="X = 1\n"), "Done.")

    task = session.start(PROMPT)
    for _ in range(400):
        await asyncio.sleep(0.005)
        if "Review, round 1" in nodes(session):
            break
    await asyncio.sleep(0.02)
    assert session.cancel()
    assert await task == ""
    assert {n.status for n in nodes(session).values()} == {"cancelled"}
