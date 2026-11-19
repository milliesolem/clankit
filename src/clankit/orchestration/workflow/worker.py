"""One workflow agent: its role, its tools, its structured result and its token accounting.

A workflow agent is an ordinary `Agent` from the agent loop. What is particular to workflows
lives here:

* its role prompt tells it that its final message is a return value read by a program, not
  prose for a person, and that nobody can answer its questions;
* when the script asks for a schema, a finishing tool hands the result back, validated with
  `jsonschema`, with a fixed number of corrective attempts (forced tool choice is never
  used: current models reject it);
* its host counts every model call against the workflow's token budget and stops the agent
  at its next turn once the budget is spent, on top of the session's own run budget.
"""

from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jsonschema

from clankit.agent import prompts
from clankit.agent.events import Event, NodeInfo
from clankit.agent.loop import Agent, AgentResult
from clankit.config.schema import Config
from clankit.llm.provider import Provider, ProviderError
from clankit.llm.types import Effort, ModelInfo, ModelRef, ToolSpec, Usage
from clankit.orchestration.common import run_to_finish
from clankit.orchestration.workflow.errors import BudgetExhausted
from clankit.tools import READ_ONLY_TOOLS, standard_tools
from clankit.tools.base import Tool, ToolContext, ToolOutput

if TYPE_CHECKING:
    from clankit.agent.session import Session

log = logging.getLogger(__name__)

RESULT_TOOL = "return_result"
#: Calls of the result tool an agent gets before its result counts as failed.
RESULT_ATTEMPTS = 5

WORKER_ROLE = """\
# Your role

You are one agent in a workflow. A program written by another agent started you and gave \
you your task in the first message; the task did not come from the user directly, the user \
does not see your messages, and nobody can answer questions while you work. Make reasonable \
decisions within the task and state them.

Your final message is a return value that the program reads, not a reply to a person. Make \
it the complete result, in the form the task asks for, with no greeting, preamble or offer \
of further help. If you could not do the task, or only part of it, say so plainly and say \
why; never present a guess as a finding.

Other agents may be working in the same project at the same time. Change only what your \
task asks you to change."""

RESULT_NOTE = """

When you are finished, call `{tool}` once with your result. Its input must match the \
tool's schema. That call is how your work is handed back: text you write outside it is not \
read."""


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Everything one `agent()` call asked for, already validated."""

    prompt: str
    title: str
    model: ModelRef
    schema: dict[str, Any] | None = None
    phase: str | None = None
    tools: str = "all"
    write_scope: tuple[str, ...] | None = None
    effort: Effort | None = None

    @property
    def mutating(self) -> bool:
        """Could this agent change the workspace? Anything with bash or the write tools can."""
        return self.tools == "all"

    def fingerprint(self) -> list[Any]:
        """The inputs that determine what the agent does, for the resume journal."""
        scope = list(self.write_scope) if self.write_scope is not None else None
        return [self.prompt, self.schema, str(self.model), self.tools, scope, self.effort]


@dataclass(slots=True)
class WorkerOutcome:
    """What came back from one agent: a value, or the reason there is none."""

    value: Any = None
    error: str = ""


class TokenMeter:
    """Tokens a workflow's agents have used, against the workflow's budget.

    Counted: output tokens plus prompt tokens that were not read from the prompt cache
    (uncached input and cache writes). Cache reads are left out because providers bill them
    at about a tenth of the input price and, counted in full, they would dominate the total
    and make the budget track context length instead of work. Providers that report no
    caching count every prompt token.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.spent = 0

    def add(self, usage: Usage) -> None:
        self.spent += usage.input_tokens + usage.cache_write_tokens + usage.output_tokens

    def check(self) -> None:
        """Raise `BudgetExhausted` if the budget is spent. The check runs before each model
        turn, so agents already mid-turn can overshoot by that turn."""
        if self.spent >= self.total:
            raise BudgetExhausted(
                f"its agents used {self.spent:,} tokens of the {self.total:,}-token budget"
            )


class WorkerHost:
    """The session as a workflow agent sees it (an `AgentHost`).

    Everything is delegated to the session, with two additions: usage also counts against
    the workflow's `TokenMeter`, and before every model turn the meter is checked after the
    session's own run-budget check. The session stops a run that is out of budget by raising
    `ProviderError` from `before_turn`; that becomes `BudgetExhausted` here, so the workflow
    stops as a whole instead of recording every remaining agent as failed.
    """

    def __init__(self, session: Session, meter: TokenMeter) -> None:
        self._session = session
        self._meter = meter

    @property
    def config(self) -> Config:
        return self._session.config

    def provider_for(self, model: ModelRef) -> tuple[Provider, str]:
        return self._session.provider_for(model)

    def model_info(self, model: ModelRef) -> ModelInfo | None:
        return self._session.model_info(model)

    def emit(self, event: Event) -> Event:
        return self._session.emit(event)

    def model_slot(self, model: ModelRef) -> AbstractAsyncContextManager[None]:
        return self._session.model_slot(model)

    def record_usage(self, node_id: str, usage: Usage, context_tokens: int) -> None:
        self._session.record_usage(node_id, usage, context_tokens)
        self._meter.add(usage)

    async def before_turn(self, node_id: str) -> None:
        try:
            await self._session.before_turn(node_id)
        except ProviderError as exc:
            raise BudgetExhausted(str(exc)) from exc
        self._meter.check()


class ResultTool(Tool):
    """The finishing tool for an agent that must return data matching a JSON Schema.

    The model is shown the caller's schema (`spec()`), but the agent loop's own input check
    is relaxed to "any object" (`input_schema`), so every attempt reaches `run`. That lets
    the tool answer an invalid result with the exact validation errors and give up after
    `attempts` tries, instead of the loop bouncing invalid calls until its turn limit.
    """

    name = RESULT_TOOL
    description = (
        "Hand your result back to the workflow that started you. Call it once, when you are "
        "finished; the input must match this schema."
    )
    input_schema: dict[str, Any] = {"type": "object"}  # relaxed on purpose; see above

    def __init__(self, schema: dict[str, Any], *, attempts: int = RESULT_ATTEMPTS) -> None:
        self._schema = schema
        self._validator = jsonschema.Draft202012Validator(schema)
        self._limit = attempts
        self.attempts = 0
        self.value: dict[str, Any] | None = None
        self.problems = ""

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self._schema)

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return "return result"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        self.attempts += 1
        errors = sorted(self._validator.iter_errors(tool_input), key=lambda e: list(e.path))
        if not errors:
            self.value = dict(tool_input)
            return ToolOutput("Result recorded.", finish={})
        self.problems = "\n".join(
            f"- {'.'.join(str(p) for p in e.path) or 'input'}: {e.message}" for e in errors[:5]
        )
        if self.attempts >= self._limit:
            return ToolOutput(
                f"Result rejected again; that was the last of {self._limit} attempts.\n"
                f"{self.problems}",
                is_error=True,
                finish={},
            )
        return ToolOutput.error(
            f"Result rejected (attempt {self.attempts} of {self._limit}); it does not match "
            f"the schema:\n{self.problems}\nFix these and call `{self.name}` again."
        )


async def run_worker(
    session: Session, host: WorkerHost, node: NodeInfo, spec: AgentSpec
) -> WorkerOutcome:
    """Run the agent for `node` and return its value, or the reason it has none.

    Model failures and refusals become an error outcome. `WorkflowStop` and cancellation
    propagate: they end the workflow, not just this agent.
    """
    tools: list[Tool] = standard_tools(READ_ONLY_TOOLS if spec.tools == "read-only" else None)
    result_tool = ResultTool(spec.schema) if spec.schema is not None else None
    if result_tool is not None:
        tools.append(result_tool)
    scope = () if spec.tools == "read-only" else spec.write_scope
    agent = Agent(
        host=host,
        node_id=node.id,
        model=spec.model,
        system=prompts.system_prompt(session, WORKER_ROLE),
        tools=tools,
        ctx=session.tool_context(node, write_scope=scope),
        effort=spec.effort,
    )
    try:
        if result_tool is None:
            return _text_outcome(await agent.run(spec.prompt))
        result, _ = await run_to_finish(
            agent, spec.prompt + RESULT_NOTE.format(tool=RESULT_TOOL), RESULT_TOOL
        )
        return _structured_outcome(result, result_tool)
    except ProviderError as exc:
        return WorkerOutcome(error=f"the model could not be reached: {exc}")
    except Exception as exc:  # a bug in a tool or adapter must cost one agent, not the workflow
        log.exception("workflow agent %s crashed", node.id)
        return WorkerOutcome(error=f"internal error: {type(exc).__name__}: {exc}")


def _stopped(result: AgentResult) -> str:
    if result.stop == "refusal":
        return f"the model declined the task. {result.detail}".strip()
    if result.stop == "max_turns":
        return "it was stopped by the max-turns limit before it finished"
    return ""


def _text_outcome(result: AgentResult) -> WorkerOutcome:
    problem = _stopped(result)
    if problem:
        return WorkerOutcome(error=problem)
    text = result.text.strip()
    if not text:
        return WorkerOutcome(error="it finished without a final message")
    return WorkerOutcome(value=text)


def _structured_outcome(result: AgentResult, tool: ResultTool) -> WorkerOutcome:
    if tool.value is not None:
        return WorkerOutcome(value=tool.value)
    if tool.attempts >= RESULT_ATTEMPTS:
        return WorkerOutcome(
            error=f"no valid result after {tool.attempts} attempts; the last problems were:\n"
            f"{tool.problems}"
        )
    return WorkerOutcome(
        error=_stopped(result) or f"it never handed back a result with `{RESULT_TOOL}`"
    )
