"""Skill candidates: the structured answer the extraction agent gives, and its format rules.

The agent ends its review by calling `propose_skills` with a list of candidates (usually an
empty one). Forcing a model to call a particular tool is not something every model allows,
so the tool is simply the only way to finish: the agent is told to call it, the agent loop
checks the call against the JSON schema, and `ProposeSkillsTool` then checks what a schema
cannot express. Problems the author can fix (a description that never says when the skill
does not apply, a script without ``--help``) are sent back as the tool's error so the model
can correct them. Whether a candidate is true to the session, safe, and new is judged
afterwards by `clankit.skills.forge`, where failing means being dropped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from clankit.skills import document
from clankit.skills.document import SKILL_FILE
from clankit.tools.base import Tool, ToolContext, ToolOutput

MAX_CANDIDATES = 3
MAX_SCRIPTS = 3
#: Generated instructions stay well under the format's limit: long skills measurably help
#: less than short ones.
MAX_BODY_LINES = 200

_SCRIPT_PATH_RE = re.compile(r"scripts/[A-Za-z0-9][A-Za-z0-9._-]*\.(?:py|sh)")
_PEP723_RE = re.compile(r"^# /// script[ \t]*\n(?:#.*\n)*?# ///[ \t]*$", re.MULTILINE)
_LIMITS_ITSELF_RE = re.compile(r"\b(?:not|never|except|unless|only|instead)\b|n't\b", re.IGNORECASE)

_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[a-z0-9]+(-[a-z0-9]+)*$",
            "maxLength": document.NAME_MAX,
            "description": "Short kebab-case name, e.g. 'regenerate-api-client'.",
        },
        "description": {
            "type": "string",
            "minLength": 1,
            "maxLength": document.DESCRIPTION_MAX,
            "description": (
                "What the skill does, when to use it, and when not to. Third person, no "
                "angle brackets. Agents choose skills by this text alone."
            ),
        },
        "body": {
            "type": "string",
            "minLength": 1,
            "description": "The instructions, in Markdown. Imperative, short, specific.",
        },
        "evidence_steps": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 1,
            "description": "Numbers of the steps in the record that this skill comes from.",
        },
        "prevents": {
            "type": "string",
            "minLength": 20,
            "description": (
                "The repeated work or the mistake, seen in those steps, that this skill "
                "would have prevented."
            ),
        },
        "updates": {
            "type": "string",
            "description": (
                "Only when correcting an existing generated skill: its name. The candidate "
                "then replaces that skill's instructions."
            ),
        },
        "scripts": {
            "type": "array",
            "maxItems": MAX_SCRIPTS,
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "scripts/NAME.py or scripts/NAME.sh",
                    },
                    "content": {"type": "string", "minLength": 1},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            "description": "Helper scripts to bundle, taken from what ran in the session.",
        },
    },
    "required": ["name", "description", "body", "evidence_steps", "prevents"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    description: str
    body: str
    evidence_steps: tuple[int, ...]
    prevents: str
    updates: str = ""
    #: (path inside the skill folder, source text)
    scripts: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> Candidate:
        """Build a candidate from tool input that already passed the JSON schema."""
        return cls(
            name=raw["name"],
            description=" ".join(raw["description"].split()),
            body=raw["body"].strip("\n"),
            evidence_steps=tuple(raw["evidence_steps"]),
            prevents=" ".join(raw["prevents"].split()),
            updates=raw.get("updates", "").strip(),
            scripts=tuple((s["path"], s["content"]) for s in raw.get("scripts", ())),
        )

    def format_problems(self) -> list[str]:
        """What the author has to fix before this candidate can be considered at all."""
        lines = self.body.count("\n") + 1
        too_long = f"the body is {lines} lines long; keep it under {MAX_BODY_LINES}"
        problems = [
            problem
            for problem in (
                document.name_problem(self.name),
                document.description_problem(self.description),
                too_long if lines > MAX_BODY_LINES else document.body_problem(self.body),
            )
            if problem
        ]
        if not _LIMITS_ITSELF_RE.search(self.description):
            problems.append(
                "the description never says when the skill does not apply; add that "
                "(for example 'Not for ...')"
            )
        paths = [path for path, _ in self.scripts]
        if len(set(paths)) != len(paths):
            problems.append("two scripts have the same path")
        for path, source in self.scripts:
            problems.extend(self._script_problems(path, source))
        return [f"{self.name}: {problem}" for problem in problems]

    def _script_problems(self, path: str, source: str) -> list[str]:
        if not _SCRIPT_PATH_RE.fullmatch(path):
            return [f"{path!r} is not a valid script path; use scripts/NAME.py or scripts/NAME.sh"]
        problems = []
        if path.endswith(".py") and not _PEP723_RE.search(source):
            problems.append(
                f"{path} has no PEP 723 block; start it with '# /// script', "
                "'# requires-python = \">=3.10\"', '# dependencies = []' and '# ///'"
            )
        if path.endswith(".sh") and not source.startswith("#!"):
            problems.append(f"{path} must start with a '#!' line such as '#!/usr/bin/env bash'")
        if "--help" not in source and "argparse" not in source:
            problems.append(f"{path} has no --help option")
        if path not in self.body:
            problems.append(f"the body never says how to run {path}; give the exact command")
        return problems

    def files(self, name: str, metadata: Mapping[str, str]) -> dict[str, str]:
        """The skill folder this candidate becomes when saved as `name`.

        Raises `SkillError` if the result would not be a valid SKILL.md.
        """
        files = {SKILL_FILE: document.render(name, self.description, self.body, metadata)}
        files.update(self.scripts)
        return files


class ProposeSkillsTool(Tool):
    """The extraction agent's only way to finish: report its candidates, or that it has none."""

    name = "propose_skills"
    description = (
        "Record the outcome of your review and end it. Call this exactly once. Pass an "
        "empty list when nothing in the record is worth saving, which is the usual outcome. "
        "If the call is rejected, fix what the message says and call it again."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "maxItems": MAX_CANDIDATES,
                "items": _CANDIDATE_SCHEMA,
                "description": "Zero to three skills learned from this session.",
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }
    #: It only reports; nothing is written until the proposal has passed every gate.
    read_only = True

    def summarize(self, tool_input: dict[str, Any]) -> str:
        names = [candidate["name"] for candidate in tool_input["candidates"]]
        return f"propose skills: {', '.join(names)}" if names else "propose no skills"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        candidates = tool_input["candidates"]
        problems = [p for raw in candidates for p in Candidate.from_payload(raw).format_problems()]
        if problems:
            listed = "\n".join(f"- {problem}" for problem in problems)
            return ToolOutput.error(
                "Nothing was recorded. Fix these problems and call propose_skills again, "
                f"or leave out a candidate that is not worth the trouble:\n{listed}"
            )
        return ToolOutput(
            f"Recorded {len(candidates)} candidate(s).", finish={"candidates": candidates}
        )
