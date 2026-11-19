"""The pattern library: the documented quality patterns as small, tested functions.

Scripts call these by name without the first argument, which the engine binds. They are
built only from the public primitives (`agent`, `parallel`, `log` and the harness's check
runner), so they obey the same agent cap, token budget, resume journal and progress display
as hand-written code, and they log whatever they drop.

They ship as code rather than as advice in a prompt because the agent writing the script
may be a modest model, and a verification step with an off-by-one in its majority rule is
worse than none. What each pattern is for:

* `adversarial_verify` - findings are claims until skeptics fail to refute them;
* `judge_panel` - several independent attempts, scored blind, when there is no executable
  check to pick a winner;
* `loop_until_dry` - "find all X": search in several ways until rounds stop turning up
  anything new, deduplicating against everything seen;
* `completeness_critic` - a fresh agent asks what a report is missing;
* `fix_until` - the harness, not an agent, decides when a check passes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, Protocol

from clankit.orchestration.common import CheckResult
from clankit.orchestration.workflow.account import render_value
from clankit.orchestration.workflow.errors import WorkflowError

#: Already-found identities listed in a finder's prompt; more are counted, not listed.
SEEN_IN_PROMPT = 200
#: Dropped claims named in adversarial_verify's log line; more are counted.
DROPPED_IN_LOG = 5


class WorkflowAPI(Protocol):
    """The primitives a pattern is built from (the engine's per-script frame)."""

    verify_votes: int

    def agent(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        label: str | None = None,
        phase: str | None = None,
        tools: str = "all",
        write_scope: Sequence[str] | None = None,
        effort: str | None = None,
    ) -> Awaitable[Any]: ...

    def parallel(self, items: Iterable[Any]) -> Awaitable[list[Any]]: ...

    def log(self, *parts: Any) -> None: ...

    async def check(self, command: str) -> CheckResult: ...


# --------------------------------------------------------------------------------------------
# adversarial_verify
# --------------------------------------------------------------------------------------------

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["refuted", "uncertain", "upheld"],
            "description": "upheld only if you confirmed the claim yourself.",
        },
        "reason": {"type": "string", "description": "What you checked and what it showed."},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

SKEPTIC = """\
Another agent made the claim below. Your job is to try to refute it. Check it against the \
evidence yourself: read the code or sources it refers to, run things where you can and \
where that settles the question, and look for the case that makes it false.

<claim>
{claim}
</claim>
{context}{lens}
Answer "upheld" only if you confirmed the claim yourself, "refuted" if you found it to be \
false, and "uncertain" if you could not settle it. In `reason`, say what you checked and \
what it showed."""


async def adversarial_verify(
    api: WorkflowAPI,
    claims: Iterable[Any],
    *,
    votes: int | None = None,
    lenses: Iterable[str] | None = None,
    context: str = "",
    tools: str = "read-only",
    write_scope: Sequence[str] | None = None,
    model: str | None = "reviewer",
    label: str = "verify",
) -> list[Any]:
    """The claims that a majority of independent skeptics uphold, in their original order.

    Each claim goes to `votes` agents told to refute it; "uncertain" and failed votes count
    against the claim. With `lenses`, each skeptic checks from its own angle and there is one
    vote per lens. Skeptics run on the "reviewer" model role unless `model` says otherwise.
    """
    items = list(claims)
    angles = [str(lens) for lens in lenses] if lenses is not None else []
    count = len(angles) if angles else (api.verify_votes if votes is None else votes)
    _require_count("adversarial_verify", "votes", count, 1, 9)
    if not items:
        return []
    needed = count // 2 + 1
    background = f"\n<context>\n{context.strip()}\n</context>\n" if context.strip() else ""

    async def ballot(index: int, claim: Any) -> list[Any]:
        calls = [
            api.agent(
                SKEPTIC.format(
                    claim=render_value(claim),
                    context=background,
                    lens=f"\nCheck it from this angle: {angles[k]}\n" if angles else "",
                ),
                schema=VERDICT_SCHEMA,
                label=f"{label} {index + 1}.{k + 1}",
                tools=tools,
                write_scope=write_scope,
                model=model,
            )
            for k in range(count)
        ]
        return await api.parallel(calls)

    ballots = await api.parallel([ballot(i, claim) for i, claim in enumerate(items)])
    survivors: list[Any] = []
    dropped: list[Any] = []
    failed = 0
    for claim, verdicts in zip(items, ballots, strict=True):
        verdicts = verdicts or [None] * count
        failed += sum(1 for v in verdicts if v is None)
        upheld = sum(1 for v in verdicts if v is not None and v.get("verdict") == "upheld")
        (survivors if upheld >= needed else dropped).append(claim)
    api.log(
        f"adversarial_verify: {len(survivors)} of {len(items)} claims upheld by at least "
        f"{needed} of {count} skeptics"
    )
    if dropped:
        named = "; ".join(_short(render_value(c)) for c in dropped[:DROPPED_IN_LOG])
        more = len(dropped) - DROPPED_IN_LOG
        api.log(
            f"adversarial_verify dropped: {named}" + (f" (and {more} more)" if more > 0 else "")
        )
    if failed:
        api.log(
            f"adversarial_verify: {failed} skeptic(s) failed; their votes counted against "
            "the claims"
        )
    return survivors


# --------------------------------------------------------------------------------------------
# judge_panel
# --------------------------------------------------------------------------------------------

ATTEMPT = """\
{task}

Approach the task from this angle: {angle}

Work it through from that angle and give your complete answer."""

JUDGE = """\
Several agents worked on the task below independently. Judge their answers on their merits: \
you do not know who wrote which one, and the order means nothing.

<task>
{task}
</task>

{attempts}

Score every attempt from 0 to 10 for how well it accomplishes the task: correctness first, \
then completeness, then simplicity. Check claims against the code where that settles a \
difference. Give one score per attempt, with a short reason."""


def _judge_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "attempt": {"type": "integer", "minimum": 1, "maximum": count},
                        "score": {"type": "number", "minimum": 0, "maximum": 10},
                        "reason": {"type": "string"},
                    },
                    "required": ["attempt", "score", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


async def judge_panel(
    api: WorkflowAPI,
    task: Any,
    angles: Iterable[str],
    *,
    judges: int = 3,
    tools: str = "read-only",
    write_scope: Sequence[str] | None = None,
    model: str | None = None,
    judge_model: str | None = "judge",
    label: str = "attempt",
) -> dict[str, Any] | None:
    """The best of several independent attempts, one per angle, scored by blind judges.

    Returns ``{"best", "index", "angle", "scores", "attempts"}`` (scores are mean judge
    scores, None for an attempt that failed or was not scored), or None if every attempt
    failed. Each judge sees the attempts in a different rotation, so position bias averages
    out, and attempts are labelled only by number.
    """
    task_text = render_value(task)
    angle_list = [str(a) for a in angles]
    if len(angle_list) < 2:
        raise WorkflowError("judge_panel needs at least two angles to compare")
    _require_count("judge_panel", "judges", judges, 1, 9)
    attempts = await api.parallel(
        [
            api.agent(
                ATTEMPT.format(task=task_text, angle=angle),
                label=f"{label}: {_short(angle, 50)}",
                tools=tools,
                write_scope=write_scope,
                model=model,
            )
            for angle in angle_list
        ]
    )
    candidates = [i for i, attempt in enumerate(attempts) if attempt is not None]
    if not candidates:
        api.log("judge_panel: every attempt failed")
        return None
    if len(candidates) == 1:
        api.log("judge_panel: only one attempt succeeded, so there was nothing to judge")
        return _panel(attempts, angle_list, candidates[0], {})

    orders = [
        candidates[j % len(candidates) :] + candidates[: j % len(candidates)] for j in range(judges)
    ]
    verdicts = await api.parallel(
        [
            api.agent(
                JUDGE.format(task=task_text, attempts=_render_attempts(order, attempts)),
                schema=_judge_schema(len(order)),
                label=f"judge {j + 1}",
                tools="read-only",
                model=judge_model,
            )
            for j, order in enumerate(orders)
        ]
    )
    means = _mean_scores(orders, verdicts)
    failed = sum(1 for v in verdicts if v is None)
    if failed:
        api.log(f"judge_panel: {failed} of {judges} judges failed")
    if not means:
        api.log("judge_panel: no judge returned scores; taking the first attempt that succeeded")
        return _panel(attempts, angle_list, candidates[0], {})
    best = max(means, key=lambda i: (means[i], -i))
    return _panel(attempts, angle_list, best, means)


def _render_attempts(order: list[int], attempts: list[Any]) -> str:
    return "\n\n".join(
        f'<attempt number="{position}">\n{render_value(attempts[index])}\n</attempt>'
        for position, index in enumerate(order, 1)
    )


def _mean_scores(orders: list[list[int]], verdicts: list[Any]) -> dict[int, float]:
    """Mean score per original attempt index, undoing each judge's rotation."""
    scores: dict[int, list[float]] = {}
    for order, verdict in zip(orders, verdicts, strict=True):
        if verdict is None:
            continue
        counted: set[int] = set()
        for entry in verdict["scores"]:
            position = int(entry["attempt"]) - 1
            if 0 <= position < len(order) and position not in counted:
                counted.add(position)
                scores.setdefault(order[position], []).append(float(entry["score"]))
    return {index: sum(values) / len(values) for index, values in scores.items()}


def _panel(
    attempts: list[Any], angles: list[str], best: int, means: dict[int, float]
) -> dict[str, Any]:
    scores = [round(means[i], 2) if i in means else None for i in range(len(angles))]
    return {
        "best": attempts[best],
        "index": best,
        "angle": angles[best],
        "scores": scores,
        "attempts": [
            {"angle": angles[i], "result": attempts[i], "score": scores[i]}
            for i in range(len(angles))
        ],
    }


# --------------------------------------------------------------------------------------------
# loop_until_dry
# --------------------------------------------------------------------------------------------

DEFAULT_FINDING: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The finding in a few words."},
        "location": {
            "type": "string",
            "description": "Where it can be checked: path:line, a command, or a URL.",
        },
        "detail": {"type": "string", "description": "What it is and why it matters."},
    },
    "required": ["title", "location", "detail"],
}

FINDER_NOTE = (
    "\n\nReturn every finding as an entry of `findings`. An empty list is the right answer "
    "when you find nothing new."
)


async def loop_until_dry(
    api: WorkflowAPI,
    finder_prompts: Iterable[str],
    key: str | Callable[[Any], Any],
    *,
    dry_rounds: int = 2,
    max_rounds: int = 5,
    item_schema: dict[str, Any] | None = None,
    tools: str = "read-only",
    write_scope: Sequence[str] | None = None,
    model: str | None = None,
    label: str = "find",
) -> list[Any]:
    """Deduplicated findings from repeated rounds of finders, in the order first found.

    Every round runs each finder prompt once, telling it what is already known. Rounds go on
    until `dry_rounds` in a row add nothing new, or `max_rounds` is reached (which is
    logged). `key` is a field name or a function that gives a finding's identity; findings
    are deduplicated against everything seen, not only against what survived later checks,
    or the loop would never converge.
    """
    prompts = [str(p) for p in finder_prompts]
    if not prompts:
        raise WorkflowError("loop_until_dry needs at least one finder prompt")
    identify = _identity(key)
    _require_count("loop_until_dry", "dry_rounds", dry_rounds, 1, 10)
    _require_count("loop_until_dry", "max_rounds", max_rounds, 1, 20)
    schema = {
        "type": "object",
        "properties": {"findings": {"type": "array", "items": item_schema or DEFAULT_FINDING}},
        "required": ["findings"],
    }
    seen: dict[str, Any] = {}
    unidentified = dry = rounds = 0
    while dry < dry_rounds:
        if rounds >= max_rounds:
            api.log(
                f"loop_until_dry: stopped at the limit of {max_rounds} rounds before "
                f"{dry_rounds} round(s) in a row found nothing new; there may be more"
            )
            break
        rounds += 1
        note = _seen_note(seen)
        batches = await api.parallel(
            [
                api.agent(
                    prompt + FINDER_NOTE + note,
                    schema=schema,
                    label=f"{label} {rounds}.{k + 1}",
                    tools=tools,
                    write_scope=write_scope,
                    model=model,
                )
                for k, prompt in enumerate(prompts)
            ]
        )
        if all(batch is None for batch in batches):
            api.log(f"loop_until_dry: every finder failed in round {rounds}")
            continue
        new = 0
        for batch in batches:
            for finding in (batch or {}).get("findings", []):
                ident = identify(finding)
                if not ident:
                    unidentified += 1
                elif ident not in seen:
                    seen[ident] = finding
                    new += 1
        dry = 0 if new else dry + 1
        api.log(f"loop_until_dry round {rounds}: {new} new, {len(seen)} in total")
    if unidentified:
        api.log(f"loop_until_dry: skipped {unidentified} finding(s) without a usable key")
    return list(seen.values())


def _identity(key: str | Callable[[Any], Any]) -> Callable[[Any], str]:
    if isinstance(key, str):
        field = key

        def by_field(finding: Any) -> str:
            return _as_key(finding.get(field) if isinstance(finding, dict) else None)

        return by_field
    if callable(key):
        return lambda finding: _as_key(key(finding))
    raise WorkflowError("loop_until_dry: key must be a field name or a function")


def _as_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _seen_note(seen: dict[str, Any]) -> str:
    if not seen:
        return ""
    keys = list(seen)
    listed = "\n".join(f"- {_short(k, 160)}" for k in keys[:SEEN_IN_PROMPT])
    more = len(keys) - SEEN_IN_PROMPT
    if more > 0:
        listed += f"\n- (and {more} more)"
    return f"\n\nAlready found (do not report these again; look for others):\n{listed}"


# --------------------------------------------------------------------------------------------
# completeness_critic
# --------------------------------------------------------------------------------------------

GAPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each gap as one self-contained sentence another agent can act on.",
        }
    },
    "required": ["gaps"],
    "additionalProperties": False,
}

CRITIC = """\
Below is a report{about}. Look for what is missing from it: parts of the task it does not \
cover, questions it leaves open, and claims it makes without support. Check the code where \
that helps.
{task}
<report>
{report}
</report>

List each gap as one self-contained sentence that another agent could act on. Return an \
empty list if nothing important is missing; do not list matters of style or wording."""


async def completeness_critic(
    api: WorkflowAPI,
    report: Any,
    *,
    task: str = "",
    tools: str = "read-only",
    model: str | None = "reviewer",
    label: str = "completeness critic",
) -> list[str]:
    """Gaps a fresh agent finds in `report` (empty when it finds none, or when it failed,
    which is logged)."""
    value = await api.agent(
        CRITIC.format(
            about=" written for the task that follows" if task.strip() else "",
            task=f"\n<task>\n{task.strip()}\n</task>\n" if task.strip() else "",
            report=render_value(report),
        ),
        schema=GAPS_SCHEMA,
        label=label,
        tools=tools,
        model=model,
    )
    if value is None:
        api.log("completeness_critic: the critic failed, so completeness was not checked")
        return []
    gaps = [str(gap).strip() for gap in value["gaps"] if str(gap).strip()]
    api.log(
        f"completeness_critic: {len(gaps)} gap(s) found"
        if gaps
        else "completeness_critic: no gaps found"
    )
    return gaps


# --------------------------------------------------------------------------------------------
# fix_until
# --------------------------------------------------------------------------------------------

FIXER = """\
{task}This check fails:

{check}

Find the cause and change the code so that the check passes. Do not weaken, skip or delete \
tests and do not special-case the check; if the check itself is wrong, say so instead of \
working around it. Run the check yourself before you finish, and end with what you changed \
and why."""

_DURATION = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds)\b")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


async def fix_until(
    api: WorkflowAPI,
    check_command: str,
    *,
    max_stale: int = 2,
    max_rounds: int = 5,
    task: str = "",
    write_scope: Sequence[str] | None = None,
    model: str | None = None,
    label: str = "fix",
) -> bool:
    """True once `check_command` passes when the harness runs it.

    While it fails, an agent gets the failing output and fixes the code, and the harness runs
    the command again. Gives up (False, logged) when the output has not changed for
    `max_stale` rounds in a row, after `max_rounds` rounds, or when running the command is
    not permitted.
    """
    command = str(check_command).strip()
    if not command:
        raise WorkflowError("fix_until needs a command to run")
    _require_count("fix_until", "max_stale", max_stale, 1, 10)
    _require_count("fix_until", "max_rounds", max_rounds, 1, 20)
    background = f"<task>\n{task.strip()}\n</task>\n\n" if task.strip() else ""
    check = await api.check(command)
    rounds = stale = 0
    last = _fingerprint(check)
    while not check.passed:
        if check.denied:
            api.log(f"fix_until: `{command}` was not run: permission was denied")
            return False
        if rounds >= max_rounds:
            api.log(f"fix_until: `{command}` still fails after {rounds} fix round(s); stopping")
            return False
        rounds += 1
        await api.agent(
            FIXER.format(task=background, check=check.render()),
            label=f"{label} {rounds}",
            write_scope=write_scope,
            model=model,
        )
        check = await api.check(command)
        current = _fingerprint(check)
        stale = stale + 1 if current == last else 0
        last = current
        if not check.passed and stale >= max_stale:
            api.log(
                f"fix_until: `{command}` made no progress in {stale} round(s) in a row; stopping"
            )
            return False
    api.log(f"fix_until: `{command}` passes" + (f" after {rounds} fix round(s)" if rounds else ""))
    return True


def _fingerprint(check: CheckResult) -> tuple[int | None, str]:
    """What a check run showed, minus what changes on every run (timings, addresses)."""
    return check.exit_code, _ADDRESS.sub("0x", _DURATION.sub("", check.output))


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _require_count(pattern: str, name: str, value: Any, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise WorkflowError(f"{pattern}: {name} must be a whole number from {low} to {high}")


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


#: The patterns by the names scripts use.
PATTERNS: dict[str, Callable[..., Awaitable[Any]]] = {
    "adversarial_verify": adversarial_verify,
    "judge_panel": judge_panel,
    "loop_until_dry": loop_until_dry,
    "completeness_critic": completeness_critic,
    "fix_until": fix_until,
}
