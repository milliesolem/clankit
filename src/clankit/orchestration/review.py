"""Independent review of finished work by an agent that did not do it.

Self-review misses a large share of an author's own errors, and judges favour their own
output, so the reviewer starts from a clean context: it gets the task and the diff the
harness computed, never the author's transcript or self-assessment, and it can read the
repository and run code to confirm what it suspects. Every finding must carry evidence (a
command and its output, or a cited code location); findings without evidence are dropped.

The reviewer cannot change files through the file tools (its write scope is empty). It can
run shell commands, which go through the normal permission gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from clankit.agent import prompts
from clankit.agent.events import NodeInfo
from clankit.llm.types import ModelRef
from clankit.orchestration.common import FinishTool, run_to_finish
from clankit.tools import standard_tools

if TYPE_CHECKING:
    from clankit.agent.session import Session

SEVERITIES = ("critical", "major", "minor")

REVIEWER_ROLE = """\
# Your role

You are an independent reviewer. Another agent was given the task below and made the \
changes in the diff. You did not write this code and you have not seen how it was written; \
judge only the result.

Find real defects: behaviour the task requires that is missing or wrong, bugs, broken or \
weakened tests, security problems, changes outside what the task asked for, and anything \
that will fail when run. Read the surrounding code, and run the project's tests or a quick \
script when that settles a question. Do not change any files.

Every finding needs evidence another engineer can check: a command and the output that \
shows the problem, or the exact file and line with a precise explanation. If you cannot \
support a suspicion with evidence, leave it out. Do not report style preferences or \
alternatives that are merely different. Finding nothing wrong is a perfectly good outcome.

Finish by calling `submit_review` exactly once."""

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["approve", "changes_needed"],
            "description": "approve if you found no critical or major problem.",
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "location": {"type": "string", "description": "path:line, or the command"},
                    "problem": {
                        "type": "string",
                        "description": "What is wrong and why it matters.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Command output or quoted code that demonstrates it.",
                    },
                },
                "required": ["severity", "location", "problem", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "findings"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class Finding:
    severity: str
    location: str
    problem: str
    evidence: str

    def render(self) -> str:
        return f"[{self.severity}] {self.location}: {self.problem}\n  evidence: {self.evidence}"


@dataclass(slots=True)
class Review:
    """The outcome of a review. `completed` is False when the reviewer never submitted."""

    completed: bool
    findings: list[Finding] = field(default_factory=list)
    node_id: str = ""

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity in ("critical", "major")]

    def render(self, *, include_minor: bool = True) -> str:
        chosen = self.findings if include_minor else self.blocking
        return "\n".join(f.render() for f in chosen)


def parse_findings(payload: dict[str, Any] | None) -> list[Finding]:
    """Findings from a `submit_review` payload, dropping any without real evidence."""
    findings = []
    for raw in (payload or {}).get("findings") or []:
        if not isinstance(raw, dict):
            continue
        evidence = str(raw.get("evidence", "")).strip()
        problem = str(raw.get("problem", "")).strip()
        if len(evidence) < 8 or not problem:
            continue
        severity = raw.get("severity") if raw.get("severity") in SEVERITIES else "minor"
        findings.append(Finding(severity, str(raw.get("location", "")).strip(), problem, evidence))
    return findings


async def review(
    session: Session,
    *,
    parent: NodeInfo,
    task: str,
    diff: str,
    model: ModelRef | None = None,
    title: str = "Independent review",
) -> Review:
    """Have a fresh agent review `diff` against `task`. The review appears as a child node
    of `parent` with role "reviewer"."""
    reviewer_model = model or session.model_for("reviewer")
    node = session.new_node(
        title, parent=parent, role="reviewer", model=reviewer_model, status="running"
    )
    tools = [
        *standard_tools(("read", "glob", "grep", "bash", "skill")),
        FinishTool("submit_review", "Submit your review. Call once, at the end.", REVIEW_SCHEMA),
    ]
    agent = session.agent(
        node,
        system=prompts.system_prompt(session, REVIEWER_ROLE),
        tools=tools,
        ctx=session.tool_context(node, write_scope=()),
        model=reviewer_model,
    )
    brief = (
        f"<task>\n{task.strip()}\n</task>\n\n"
        f"<diff>\n{diff.strip() or '(no file changes were recorded)'}\n</diff>\n\n"
        "Review the result against the task, then call `submit_review`."
    )
    _, payload = await run_to_finish(agent, brief, "submit_review")
    if payload is None:
        session.update_node(node, status="failed", summary="The reviewer did not submit a review.")
        return Review(completed=False, node_id=node.id)
    findings = parse_findings(payload)
    blocking = sum(1 for f in findings if f.severity in ("critical", "major"))
    summary = (
        "No problems found."
        if not findings
        else f"{len(findings)} finding(s), {blocking} blocking."
    )
    session.update_node(node, status="done", summary=summary)
    return Review(completed=True, findings=findings, node_id=node.id)


def findings_message(review_: Review) -> str:
    """The message that hands a review's findings back to the author."""
    return (
        "[Independent review]\n"
        "A reviewer who has not seen your work in progress checked the changes and reported "
        "the findings below. Reviewers can be wrong: verify each finding yourself first. Fix "
        "the ones that are real, re-run the relevant checks, and say briefly which findings "
        "you fixed and which you rejected and why.\n\n" + review_.render()
    )
