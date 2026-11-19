"""The `skill` tool: how an agent loads a skill's instructions.

Agents see only the catalog (name and description of each skill) in their system prompt.
Calling this tool is the second step of progressive disclosure: it returns the skill's full
instructions, where the skill lives, and which files come with it, so the agent can run a
bundled script through its ordinary shell tool (and that tool's permission checks).

The input schema is deliberately static. The set of installed skills changes while agents
are running (proposals get approved, files are edited), but an agent's tool list must stay
byte-identical for the whole conversation, so names are checked when the tool runs rather
than enumerated in the schema.
"""

from __future__ import annotations

import difflib
from typing import Any

from clankit.skills.library import Skill, SkillLibrary
from clankit.tools.base import Tool, ToolContext, ToolOutput

#: Characters of a skill's instructions returned by one call. A few published skills are far
#: longer than the format recommends; past this the agent is pointed at the file instead.
BODY_LIMIT = 24_000
#: Bundled files listed by one call.
FILE_LIMIT = 20
#: Skill names offered when the requested one does not exist.
SUGGESTION_LIMIT = 40

NO_LIBRARY = (
    "Skills are not available in this session (no skill library is attached). Carry on without one."
)


class SkillTool(Tool):
    name = "skill"
    description = (
        "Load a skill: instructions (and sometimes scripts) saved for a recurring kind of "
        "work. Your instructions list the available skills as `- name: description`. When "
        "a task matches a description, call this with that name before you start, then "
        "follow what it returns. The result gives the skill's instructions, its directory, "
        "and its bundled files; run bundled scripts by absolute path with your shell tool."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill's name, exactly as listed.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    read_only = True

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"skill {tool_input.get('name', '')}".rstrip()

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        library: SkillLibrary | None = ctx.skills
        if library is None:
            return ToolOutput.error(NO_LIBRARY)
        name = tool_input["name"].strip()
        skill = library.get(name)
        if skill is None:
            return ToolOutput.error(_unknown(name, library))
        refusal = _refusal(skill)
        if refusal:
            return ToolOutput.error(refusal)
        try:
            return ToolOutput(_activation(skill))
        except OSError as exc:
            return ToolOutput.error(
                f"The skill {name!r} could not be read ({exc}). Carry on without it and "
                "mention the problem in your answer."
            )


def _refusal(skill: Skill) -> str:
    """Why an agent may not load `skill` on its own, or "" when it may."""
    if skill.disable_model_invocation:
        return (
            f"The skill {skill.name!r} can only be started by the user (it sets "
            "disable-model-invocation). Do not work around this; if it is needed, say so "
            "in your answer."
        )
    if not skill.trusted:
        return (
            f"The skill {skill.name!r} comes from this project's files, and the user has "
            "not trusted this project's skills. Carry on without it; do not read its "
            "files as a substitute."
        )
    return ""


def _unknown(name: str, library: SkillLibrary) -> str:
    if any(proposal.name == name for proposal in library.pending()):
        return (
            f"The skill {name!r} is a proposal waiting for the user's approval, so it "
            "cannot be loaded yet. Carry on without it."
        )
    names = [skill.name for skill in library.skills if skill.model_invocable]
    if not names:
        return f"There is no skill named {name!r}; no skills are installed. Carry on without one."
    close = difflib.get_close_matches(name, names, n=len(names), cutoff=0.5)
    ordered = close + [n for n in names if n not in close]
    listed = ", ".join(ordered[:SUGGESTION_LIMIT])
    more = len(ordered) - SUGGESTION_LIMIT
    if more > 0:
        listed += f" (and {more} more)"
    return f"There is no skill named {name!r}. Available skills, closest match first: {listed}."


def _activation(skill: Skill) -> str:
    """What the agent reads: instructions first, then where the skill's files are."""
    parts = [
        f'<skill_content name="{skill.name}">',
        _clipped(skill),
        "</skill_content>",
        "",
        f"Skill directory: {skill.path}",
        "Paths in this skill are relative to that directory. Use absolute paths when you "
        "read its files or run its scripts.",
    ]
    files = skill.files()
    if files:
        parts.append("Bundled files:")
        parts.extend(f"- {relative}" for relative in files[:FILE_LIMIT])
        if len(files) > FILE_LIMIT:
            parts.append(f"({len(files) - FILE_LIMIT} more files not listed)")
    return "\n".join(parts)


def _clipped(skill: Skill) -> str:
    text = skill.instructions().strip("\n")
    if len(text) <= BODY_LIMIT:
        return text
    kept = text[:BODY_LIMIT].rsplit("\n", 1)[0]
    return (
        f"{kept}\n\n[These instructions continue for another {len(text) - len(kept):,} "
        f"characters. Read the rest of {skill.file} when you need it.]"
    )
