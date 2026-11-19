"""The agent loop: model calls, tool rounds, validation, permissions, cancellation, compaction."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from clankit.agent.events import AssistantMessage, Notice, ToolFinished, ToolStarted
from clankit.agent.loop import INTERRUPTED, NOT_RUN, Agent
from clankit.agent.session import Session
from clankit.config.schema import ModelEntry, ProviderConfig
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatResponse, Message, TextBlock, ToolResultBlock, Usage
from clankit.tools.base import PermissionAsk, Tool, ToolContext, ToolOutput


class Echo(Tool):
    name = "echo"
    description = "Echo text back."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    read_only = True

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput(tool_input["text"])


class Slow(Tool):
    name = "slow"
    description = "Sleep, recording concurrency."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, *, read_only: bool) -> None:
        self.read_only = read_only
        self.active = 0
        self.peak = 0

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return ToolOutput("ok")


class Guarded(Tool):
    name = "guarded"
    description = "Needs permission."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self) -> None:
        self.ran = False

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        return PermissionAsk("shell", "Run something risky", "risky --now")

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        self.ran = True
        return ToolOutput("ran")


class Finish(Tool):
    name = "finish"
    description = "End the run."
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "additionalProperties": False,
    }

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput("recorded", finish={"value": tool_input.get("value")})


class Boom(Tool):
    name = "boom"
    description = "Always crashes."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        raise RuntimeError("kaput")


def make_agent(session: Session, *tools: Tool, **kwargs: Any) -> Agent:
    node = session.new_node("test", role="root")
    return session.agent(node, system="You are a test agent.", tools=list(tools), **kwargs)


def results_of(message: Message) -> list[ToolResultBlock]:
    return message.tool_result_blocks


async def test_plain_answer(session: Session, scripted: ScriptedProvider, events: list) -> None:
    scripted.push("hello there")
    agent = make_agent(session)
    result = await agent.run("hi")
    assert (result.text, result.stop) == ("hello there", "end_turn")
    assert [m.role for m in agent.messages] == ["user", "assistant"]
    finals = [e for e in events if isinstance(e, AssistantMessage)]
    assert [e.text for e in finals] == ["hello there"]
    assert agent.usage.output_tokens == 5 and session.usage.output_tokens == 5


async def test_tool_round_trip_and_history_shape(
    session: Session, scripted: ScriptedProvider, events: list
) -> None:
    scripted.push([call("echo", text="a"), call("echo", text="b")], "done")
    agent = make_agent(session, Echo())
    result = await agent.run("go")
    assert result.text == "done"
    # One user message answers both calls, in order.
    assert [m.role for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert [r.content for r in results_of(agent.messages[2])] == ["a", "b"]
    assert [e.tool for e in events if isinstance(e, ToolStarted)] == ["echo", "echo"]
    assert all(not e.is_error for e in events if isinstance(e, ToolFinished))
    # system and tools are identical on every request of the conversation
    assert scripted.requests[0].system == scripted.requests[1].system
    assert scripted.requests[0].tools == scripted.requests[1].tools


async def test_invalid_input_unknown_tool_and_bad_json_are_answered(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push(
        [call("echo", wrong=1), call("nope"), call("echo", _invalid_json="{oops")],
        "ok",
    )
    agent = make_agent(session, Echo())
    await agent.run("go")
    results = results_of(agent.messages[2])
    assert len(results) == 3 and all(r.is_error for r in results)
    assert "Invalid input for echo" in results[0].content
    assert "no tool named 'nope'" in results[1].content
    assert "not valid JSON" in results[2].content


async def test_tool_crash_becomes_error_result(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push(call("boom"), "recovered")
    agent = make_agent(session, Boom())
    result = await agent.run("go")
    assert result.text == "recovered"
    (res,) = results_of(agent.messages[2])
    assert res.is_error and "kaput" in res.content


async def test_read_only_calls_run_concurrently_and_others_serially(
    session: Session, scripted: ScriptedProvider
) -> None:
    reader, writer = Slow(read_only=True), Slow(read_only=False)
    reader.name, writer.name = "reader", "writer"
    scripted.push(
        [call("reader"), call("reader"), call("reader")], [call("writer"), call("writer")], "done"
    )
    agent = make_agent(session, reader, writer)
    await agent.run("go")
    assert reader.peak == 3
    assert writer.peak == 1


async def test_finishing_tool_ends_run_with_payload(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push([TextBlock("wrapping up"), call("finish", value=7)])
    agent = make_agent(session, Finish())
    result = await agent.run("go")
    assert result.stop == "finished" and result.payload == {"value": 7}
    # the finishing call is still answered, so the conversation can continue later
    assert results_of(agent.messages[-1])[0].content == "recorded"
    scripted.push("continued")
    assert (await agent.run("more")).text == "continued"


async def test_permission_denied_in_read_only_mode(
    session: Session, scripted: ScriptedProvider
) -> None:
    session.config.permissions.mode = "read-only"
    tool = Guarded()
    scripted.push(call("guarded"), "fine")
    agent = make_agent(session, tool)
    await agent.run("go")
    (res,) = results_of(agent.messages[2])
    assert res.is_error and "Permission denied" in res.content and not tool.ran


async def test_permission_asked_and_granted_for_session(
    session: Session, scripted: ScriptedProvider, events: list
) -> None:
    session.config.permissions.mode = "ask"
    tool = Guarded()
    scripted.push(call("guarded"), call("guarded"), "fine")
    agent = make_agent(session, tool)
    task = asyncio.create_task(agent.run("go"))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if session.view.permissions:
            break
    (request,) = session.view.permissions.values()
    assert request.summary == "Run something risky" and request.session_rule == "shell:risky --now"
    assert session.resolve_permission(request.request_id, "allow_session")
    result = await task
    assert result.text == "fine" and tool.ran
    # the second call was covered by the session rule: only one request was ever raised
    assert sum(1 for e in events if e.kind == "permission_requested") == 1


async def test_retryable_error_is_retried(
    session: Session, scripted: ScriptedProvider, events: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("clankit.agent.context.backoff_delay", lambda attempt: 0.0)
    scripted.push(ProviderError("overloaded", kind="overloaded"), "second time lucky")
    result = await make_agent(session).run("hi")
    assert result.text == "second time lucky"
    assert any(isinstance(e, Notice) and "retrying" in e.text for e in events)


async def test_fatal_error_propagates_and_history_stays_usable(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push(ProviderError("bad key", kind="auth"))
    agent = make_agent(session)
    with pytest.raises(ProviderError):
        await agent.run("hi")
    scripted.push("now it works")
    assert (await agent.run("again")).text == "now it works"


async def test_missing_model_error_names_the_models_the_provider_offers(
    session: Session, scripted: ScriptedProvider
) -> None:
    session.runtime.update_config(
        lambda c: c.providers.__setitem__(
            "scripted",
            ProviderConfig(
                id="scripted",
                base_url="http://localhost:8000/v1",
                keyless=True,
                models=[ModelEntry(id="qwen3-coder", name="Qwen3 Coder")],
            ),
        )
    )
    scripted.push(ProviderError("model 'nope' not found (HTTP 404)", kind="not_found"))
    with pytest.raises(ProviderError) as caught:
        await make_agent(session).run("hi")
    assert "qwen3-coder" in str(caught.value)


async def test_missing_model_error_asks_the_server_what_it_offers(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push(ProviderError("model 'nope' not found (HTTP 404)", kind="not_found"))
    with pytest.raises(ProviderError) as caught:
        await make_agent(session).run("hi")
    assert "test-model" in str(caught.value)


async def test_cancel_mid_tool_round_answers_every_call(
    session: Session, scripted: ScriptedProvider
) -> None:
    class Hang(Tool):
        name = "hang"
        description = "Never returns."
        input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

        async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
            await asyncio.sleep(60)
            return ToolOutput("never")

    scripted.push([call("echo", text="first"), call("hang"), call("hang")])
    agent = make_agent(session, Echo(), Hang())
    task = asyncio.create_task(agent.run("go"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    results = results_of(agent.messages[-1])
    assert [r.content for r in results] == ["first", INTERRUPTED, NOT_RUN]
    # and the conversation can go on
    scripted.push("resumed")
    assert (await agent.run("continue")).text == "resumed"


async def test_max_turns_stops_the_loop(session: Session, scripted: ScriptedProvider) -> None:
    scripted.push(*[call("echo", text=str(i)) for i in range(10)])
    agent = make_agent(session, Echo(), max_turns=3)
    result = await agent.run("loop forever")
    assert result.stop == "max_turns"
    assert len(scripted.requests) == 3


async def test_truncated_tool_call_is_not_run(session: Session, scripted: ScriptedProvider) -> None:
    truncated = ChatResponse(
        message=Message("assistant", [call("echo", text="partial")]),
        stop_reason="max_tokens",
        usage=Usage(1, 1),
    )
    scripted.push(truncated, "smaller steps then")
    agent = make_agent(session, Echo())
    result = await agent.run("go")
    assert result.text == "smaller steps then"
    (res,) = results_of(agent.messages[2])
    assert res.is_error and "output limit" in res.content


async def test_steering_is_delivered_with_tool_results(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push(call("echo", text="x"), "ok")
    agent = make_agent(session, Echo())
    agent.steer("also update the docs")
    await agent.run("go")
    tail = agent.messages[2].content[-1]
    assert isinstance(tail, TextBlock) and "also update the docs" in tail.text


async def test_compaction_replaces_history_with_summary(
    session: Session, scripted: ScriptedProvider, events: list
) -> None:
    session.config.agent.context_window = 1000
    big = ScriptedProvider(provider_id="scripted", usage=Usage(input_tokens=900, output_tokens=10))
    session.runtime.providers.register(big)
    big.push(
        call("echo", text="x"), "<summary>did x; next: y</summary>", "finished after compaction"
    )
    agent = make_agent(session, Echo())
    result = await agent.run("the original task")
    assert result.text == "finished after compaction"
    first = agent.messages[0].text
    assert "the original task" in first and "did x; next: y" in first
    assert len(agent.messages) == 2
    assert any(isinstance(e, Notice) and "compacted" in e.text for e in events)
    # the summary request extended the same conversation (same system and tools)
    assert (
        big.requests[1].system == big.requests[0].system
        and big.requests[1].tools == big.requests[0].tools
    )


async def test_withdrawn_partial_message_leaves_no_transcript_item(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clankit.llm.provider import Provider
    from clankit.llm.types import ResponseDone, TextDelta

    monkeypatch.setattr("clankit.agent.context.backoff_delay", lambda attempt: 0.0)

    class Flaky(Provider):
        kind = "flaky"
        calls = 0

        async def stream(self, request):  # type: ignore[no-untyped-def]
            Flaky.calls += 1
            yield TextDelta("partial ")
            if Flaky.calls == 1:
                raise ProviderError("connection reset", kind="network")
            yield ResponseDone(
                ChatResponse(Message.assistant("partial complete"), "end_turn", Usage(1, 1))
            )

    session.runtime.providers.register(Flaky("scripted"))
    agent = make_agent(session)
    result = await agent.run("hi")
    assert result.text == "partial complete"
    texts = [i.text for i in session.view.nodes[agent.node_id].items if i.kind == "assistant"]
    assert texts == ["partial complete"]


async def test_repeated_identical_calls_get_a_nudge(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push(*[call("echo", text="same") for _ in range(3)], "ok")
    agent = make_agent(session, Echo())
    await agent.run("go")
    notes = [
        m.content[-1].text
        for m in agent.messages
        if m.role == "user" and isinstance(m.content[-1], TextBlock)
    ]
    assert sum("same tool call" in n for n in notes) == 1
    assert "same tool call" in agent.messages[6].content[-1].text  # after the third round


async def test_model_calls_are_limited_per_provider(
    session: Session, scripted: ScriptedProvider
) -> None:
    session.runtime.providers._configs["scripted"] = __import__(
        "clankit.config.schema", fromlist=["ProviderConfig"]
    ).ProviderConfig(id="scripted", keyless=True, concurrency=2)
    active = peak = 0
    original = scripted.stream

    async def counting(request):  # type: ignore[no-untyped-def]
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        async for event in original(request):
            yield event

    scripted.stream = counting  # type: ignore[method-assign]
    scripted.push(*["x"] * 6)
    await asyncio.gather(*(make_agent(session).run("hi") for _ in range(6)))
    assert peak == 2


async def test_refused_turn_is_discarded(session: Session, scripted: ScriptedProvider) -> None:
    refused = ChatResponse(
        message=Message("assistant", [TextBlock("partial text that must not be kept")]),
        stop_reason="refusal",
        usage=Usage(1, 1),
        detail="declined by a safety classifier (cyber)",
    )
    scripted.push(refused, "fine now")
    agent = make_agent(session)
    result = await agent.run("hi")
    assert (result.stop, result.text, result.detail) == (
        "refusal",
        "",
        "declined by a safety classifier (cyber)",
    )
    assert [m.role for m in agent.messages] == ["user"]  # the refused turn was not appended
    assert not [i for i in session.view.nodes[agent.node_id].items if i.kind == "assistant"]
    assert (await agent.run("rephrased")).text == "fine now"
