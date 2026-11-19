"""Review Loop: one agent does the work, and a reviewer with a clean context checks it.

The agent that talks to the user (the generator) works exactly as in Solo. When it has
changed files, a fresh reviewer checks the harness-computed diff against the request.
Findings that block the request go back to the generator, which verifies and fixes them,
and a new reviewer looks again, up to a fixed number of rounds.

What the evidence says, and how it shapes this module:

* Authors miss many of their own errors and judges favour their own output. The reviewer
  never sees the generator's transcript or its account of the work: it gets the request, the
  diff and the harness's own check result, and every finding must carry evidence.
* Self-correction works with external feedback, not without it. An optional check command
  is run by the harness before each review and after each round of fixes; a fix that makes a
  passing check fail is put to the generator at once instead of waiting for the reviewer.
* Generate-review loops stall without a cap, and repeated LLM-only refinement can make code
  worse. Two rounds by default, and minor findings are not handed back unless asked for.
* The clean context is the well-supported lever; a reviewer from another model family helps
  less certainly. Cross-family review is therefore optional, and a clearly cheaper (weaker)
  model is never chosen over a fresh session of the generator's own: a weak second opinion
  adds nothing.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from clankit.agent import prompts
from clankit.agent.events import ToolFinished, ToolStarted
from clankit.agent.loop import PREVIEW_CHARS, Agent, AgentResult
from clankit.config.schema import Setting
from clankit.llm.types import ModelInfo, ModelRef
from clankit.orchestration.base import AskUserTool, Orchestrator
from clankit.orchestration.common import CheckResult, resolve_model, run_check, run_diff
from clankit.orchestration.review import Finding, Review, findings_message, review
from clankit.tools import standard_tools
from clankit.tools.base import ToolContext

if TYPE_CHECKING:
    from clankit.agent.session import Session

#: Substrings of model ids that identify a model family whichever provider serves it
#: (OpenRouter, for one, serves many). Checked in order; a model matching none of them
#: belongs to the vendor that prefixes its id (``openai/o3-pro``), else to its provider.
FAMILIES: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("gemini", "google"),
    ("gemma", "google"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("glm", "zhipu"),
    ("kimi", "moonshot"),
    ("minimax", "minimax"),
    ("llama", "meta"),
    ("mistral", "mistral"),
    ("codestral", "mistral"),
    ("devstral", "mistral"),
    ("grok", "xai"),
)

#: Price is the only capability signal the configuration carries. A model from another family
#: whose output price is below this share of the generator's is taken to be weaker, and a
#: fresh session of the generator's own model is used instead.
MIN_PRICE_SHARE = 0.5

#: Earlier requests of the session shown to the reviewer, so that follow-ups make sense.
EARLIER_REQUESTS = 4

REGRESSION = """\
[Harness check after your fixes]
This check passed before your last round of fixes and fails now:

{check}

Find the change that broke it and correct it, keeping the fixes that were right. Then give \
your summary for the user again."""

_check_ids = itertools.count(1)


# --------------------------------------------------------------------------------------------
# Harness checks the user can follow
# --------------------------------------------------------------------------------------------


async def visible_check(session: Session, ctx: ToolContext, command: str) -> CheckResult:
    """Run a verification command for the harness and show it on the node's transcript.

    The run appears as a "check" tool item on the node whose work it verifies, so the user can
    see what the harness ran and what it saw. No agent sees this item; agents are told about
    checks in their own messages.
    """
    call_id = f"check{next(_check_ids)}"
    session.emit(
        ToolStarted(
            node_id=ctx.node_id,
            call_id=call_id,
            tool="check",
            summary=f"$ {command}",
            input={"command": command},
        )
    )
    began = time.monotonic()
    try:
        result = await run_check(session, ctx, command)
    except BaseException as exc:
        # The item must not be left looking as if the check were still running.
        interrupted = isinstance(exc, asyncio.CancelledError)
        session.emit(
            ToolFinished(
                node_id=ctx.node_id,
                call_id=call_id,
                output="Interrupted." if interrupted else f"Could not run the check: {exc}",
                is_error=True,
                duration=time.monotonic() - began,
            )
        )
        raise
    session.emit(
        ToolFinished(
            node_id=ctx.node_id,
            call_id=call_id,
            output=result.render()[:PREVIEW_CHARS],
            is_error=not result.passed,
            duration=time.monotonic() - began,
        )
    )
    return result


def configured_model(
    session: Session, setting: str, value: str, role: str | None
) -> ModelRef | None:
    """The model a "model" setting names, or None when it is empty or cannot be used.

    A value that cannot be used (malformed, unconfigured provider, a model known not to call
    tools) is reported to the user as a warning rather than failing the run; the caller then
    falls back to the role's model.
    """
    value = value.strip()
    if not value:
        return None
    try:
        ref = resolve_model(session, value, role=role)
    except ValueError as exc:
        session.notice("warning", f"Ignoring the {setting} setting: {exc}")
        return None
    info = session.model_info(ref)
    if info is not None and info.supports_tools is False:
        session.notice(
            "warning",
            f"Ignoring the {setting} setting: {ref} cannot call tools, so it cannot run an agent.",
        )
        return None
    return ref


def check_line(check: CheckResult) -> str:
    """One sentence for the user saying what a harness check showed."""
    if check.denied:
        return f"Check `{check.command}`: not run, because permission was denied."
    if check.passed:
        return f"Check `{check.command}`: passed."
    if check.exit_code is None:
        return f"Check `{check.command}`: failed (it did not finish in time)."
    return f"Check `{check.command}`: failed (exit code {check.exit_code})."


# --------------------------------------------------------------------------------------------
# Choosing the reviewer's model
# --------------------------------------------------------------------------------------------


def model_family(ref: ModelRef) -> str:
    """The family a model belongs to: recognised from its id; else the vendor that
    aggregators put in front of the id, so OpenRouter's ``openai/o3-pro`` and OpenAI's own
    models count as one family; else its provider."""
    model = ref.model.lower()
    marked = next((family for marker, family in FAMILIES if marker in model), None)
    if marked is not None:
        return marked
    vendor, slash, _ = model.partition("/")
    return vendor if slash and vendor else ref.provider


def strongest_other_family(models: Sequence[ModelInfo], generator: ModelRef) -> ModelRef | None:
    """The most capable model from a family other than the generator's, or None.

    Capability is judged by output price, the only signal available: free and zero-priced
    models are left out, models without a known price rank last, and a candidate priced below
    `MIN_PRICE_SHARE` of the generator is rejected as weaker. Models known not to call tools
    cannot run a reviewer and are skipped.
    """
    family = model_family(generator)
    candidates = [
        m
        for m in models
        if model_family(m.ref) != family
        and m.supports_tools is not False
        and not m.free
        and m.output_price != 0
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda m: -1.0 if m.output_price is None else m.output_price)
    own = next((m for m in models if m.ref == generator), None)
    if (
        own is not None
        and own.output_price
        and best.output_price is not None
        and best.output_price < own.output_price * MIN_PRICE_SHARE
    ):
        return None
    return best.ref


# --------------------------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------------------------


class ReviewLoopOrchestrator(Orchestrator):
    name = "review-loop"
    title = "Review Loop"
    summary = "One agent does the task; a fresh reviewer checks the changes and finds what to fix."
    guidance = (
        "Worth it for changes that will be merged without a human review and for "
        "security-sensitive code: a reviewer with a clean context finds defects the author "
        "misses, and each finding must carry evidence. Costs roughly 1.3 to 2 times Solo, "
        "more with extra rounds. Not worth it for trivial edits. A reviewer from another model "
        "family is optional; the evidence for it is weaker than for the clean context."
    )
    settings = (
        Setting(
            "max-rounds",
            "int",
            2,
            "Review rounds per request. Each round reviews the changes so far and hands "
            "blocking findings back for fixing.",
            minimum=1,
            maximum=5,
        ),
        Setting(
            "reviewer-model",
            "model",
            "",
            "Model for the reviewer. Empty uses the reviewer role's model, or the session's "
            "model when no reviewer model is set.",
        ),
        Setting(
            "prefer-other-family",
            "bool",
            False,
            "When the reviewer would use the same model family as the agent doing the work, "
            "use the most capable configured model of another family instead (judged by "
            "price), unless it looks weaker. Costs that model's price.",
        ),
        Setting(
            "check",
            "string",
            "",
            "A command the harness runs before each review and after each round of fixes, "
            "e.g. 'pytest -q'. Its result goes to the reviewer as evidence, and fixes that make "
            "a passing check fail are sent straight back. Empty runs nothing.",
        ),
        Setting(
            "include-minor",
            "bool",
            False,
            "Also hand minor findings back for fixing. Off, only critical and major findings "
            "go back; minor ones stay in the review's record.",
        ),
    )

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self._root = session.new_node("Conversation", role="root")

    def _ensure_agent(self) -> Agent:
        if self.root_agent is None:
            self.root_agent = self.session.agent(
                self._root,
                system=prompts.system_prompt(self.session, prompts.USER_FACING),
                tools=[*standard_tools(), AskUserTool(self.session)],
            )
        return self.root_agent

    async def run(self, prompt: str) -> str:
        session = self.session
        agent = self._ensure_agent()
        message = prompt
        if not agent.messages and (recap := session.recap()):
            message = f"{recap}\n\nThe user now says:\n{prompt}"

        session.update_node(self._root, status="running")
        result = await agent.run(message)
        if result.stop == "refusal":
            return self._declined(result)
        if not run_diff(session):
            session.update_node(
                self._root, status="done", summary="No files changed, so nothing was reviewed."
            )
            return result.text

        reviewer = self._reviewer_model(agent.model)
        result, lines = await self._review_rounds(agent, prompt, reviewer, result)
        if result.stop == "refusal":
            return self._declined(result)
        session.update_node(self._root, status="done", summary=_one_line(" ".join(lines), 200))
        footer = "\n".join([f"Independent review ({reviewer}):", *(f"- {line}" for line in lines)])
        return f"{result.text.rstrip()}\n\n{footer}"

    # -- the loop -------------------------------------------------------------------------------

    async def _review_rounds(
        self, agent: Agent, prompt: str, reviewer: ModelRef, result: AgentResult
    ) -> tuple[AgentResult, list[str]]:
        """Review, hand findings back, re-check. `result` is the generator's latest; returns
        its latest after the rounds, and one line per observation for the user."""
        session = self.session
        rounds = self.setting("max-rounds")
        command = self.setting("check").strip()
        check = await visible_check(session, agent.ctx, command) if command else None
        lines: list[str] = []
        previous: list[Finding] = []
        for number in range(1, rounds + 1):
            session.update_node(self._root, status="waiting")
            outcome = await review(
                session,
                parent=self._root,
                task=self._task(prompt, check, previous),
                diff=run_diff(session),
                model=reviewer,
                title=f"Review, round {number}",
            )
            if not outcome.completed:
                lines.append(
                    f"Round {number}: the reviewer did not submit a review, so the latest "
                    "changes are unreviewed."
                )
                break
            handed = outcome.findings if self.setting("include-minor") else outcome.blocking
            lines.append(_round_line(number, outcome, handed))
            if not handed:
                break
            session.update_node(self._root, status="integrating")
            result = await agent.run(findings_message(Review(True, handed, outcome.node_id)))
            if result.stop == "refusal":
                return result, lines
            if command:
                result, check, regression = await self._recheck(agent, command, check, result)
                if result.stop == "refusal":
                    return result, lines
                if regression:
                    lines.append(
                        f"The check passed before the fixes of round {number} and still fails "
                        "after them."
                    )
            previous = handed
            if not outcome.blocking:
                break  # only minor findings went back; they do not need another review
            if number == rounds:
                lines.append(
                    f"The fixes for round {number} were not reviewed again (the round limit)."
                )
        if check is not None:
            lines.append(check_line(check))
        return result, lines

    async def _recheck(
        self, agent: Agent, command: str, before: CheckResult | None, result: AgentResult
    ) -> tuple[AgentResult, CheckResult, bool]:
        """Run the check after a round of fixes. When the fixes broke a check that passed,
        the generator gets one turn to repair that before anything else happens.

        Returns the generator's latest result, the latest check, and whether the check is
        still broken by the fixes."""
        session = self.session
        check = await visible_check(session, agent.ctx, command)
        if not _regressed(before, check):
            return result, check, False
        session.notice(
            "warning",
            f"The fixes broke `{command}`, which passed before them; sent back for repair.",
            self._root.id,
        )
        result = await agent.run(REGRESSION.format(check=check.render()))
        if result.stop == "refusal":
            return result, check, True
        check = await visible_check(session, agent.ctx, command)
        return result, check, _regressed(before, check)

    def _task(self, prompt: str, check: CheckResult | None, previous: list[Finding]) -> str:
        """What the reviewer is told besides the diff: the request, earlier requests for
        context, the harness's check, and the findings an earlier round handed back. Never
        the generator's own account of its work."""
        sections = [prompt.strip()]
        earlier = [turn.prompt for turn in self.session.turns[-EARLIER_REQUESTS:]]
        if earlier:
            sections.append(
                "Earlier requests in this session, already handled (for context only):\n"
                + "\n".join(f"- {_one_line(text, 300)}" for text in earlier)
            )
        if check is not None:
            sections.append(f"A check the harness ran on these changes:\n{check.render()}")
        if previous:
            sections.append(
                "An earlier review reported these findings and the author was asked to fix "
                "them. Check whether each one is resolved, and look for problems the fixes "
                "introduced:\n" + "\n".join(f.render() for f in previous)
            )
        return "\n\n".join(sections)

    # -- models ---------------------------------------------------------------------------------

    def _reviewer_model(self, generator: ModelRef) -> ModelRef:
        """The reviewer-model setting if usable; otherwise the reviewer role's model, swapped
        for another family's strongest model when prefer-other-family asks for it."""
        session = self.session
        chosen = configured_model(
            session, "reviewer-model", self.setting("reviewer-model"), "reviewer"
        )
        if chosen is not None:
            return chosen
        default = session.model_for("reviewer")
        if self.setting("prefer-other-family") and model_family(default) == model_family(generator):
            other = strongest_other_family(session.runtime.providers.known_models(), generator)
            if other is not None:
                return other
        return default

    def _declined(self, result: AgentResult) -> str:
        self.session.update_node(
            self._root, status="failed", summary="The model declined the request."
        )
        return f"The model declined this request. {result.detail}".strip()


def _regressed(before: CheckResult | None, after: CheckResult) -> bool:
    return before is not None and before.passed and not after.passed and not after.denied


def _round_line(number: int, outcome: Review, handed: list[Finding]) -> str:
    blocking = len(outcome.blocking)
    minor = len(outcome.findings) - blocking
    if not outcome.findings:
        return f"Round {number}: no problems found."
    if not handed:
        return f"Round {number}: no blocking problems; {minor} minor finding(s) not handed back."
    kept = minor if len(handed) == blocking else 0
    sent = [f"{blocking} blocking"] if blocking else []
    if minor and not kept:
        sent.append(f"{minor} minor")
    line = f"Round {number}: {' and '.join(sent)} finding(s) handed back for fixing"
    return line + (f"; {kept} minor not handed back." if kept else ".")


def _one_line(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
