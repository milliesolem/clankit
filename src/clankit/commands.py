"""Slash commands, shared by the terminal UI and the web UI.

Both interfaces accept the same ``/command [argument]`` lines. This module owns the list of
commands (so help and completion agree everywhere) and everything a command does that is
not purely visual. The outcome tells the UI what remains for it to do: show a message,
open one of its views, submit a prompt, or perform a UI-level action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from clankit.config.schema import PERMISSION_MODES, ConfigError
from clankit.llm.types import EFFORT_LEVELS, ModelRef

if TYPE_CHECKING:
    from clankit.agent.session import Session
    from clankit.runtime import Runtime


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    summary: str
    argument: str = ""


@dataclass(slots=True)
class Outcome:
    """What the UI should do after a command ran.

    `message` (with `level` "info" | "warning" | "error") is shown to the user. `view` names
    a view to open: "help", "settings", "models", "methods", "themes", "skills", "sessions".
    `prompt` is a prompt to submit as a normal run. `action` is "new-session" or "quit".
    """

    message: str = ""
    level: str = "info"
    view: str = ""
    prompt: str = ""
    action: str = ""


COMMANDS: tuple[Command, ...] = (
    Command("help", "Show commands and keys"),
    Command("model", "Choose the model, or set it directly", "[provider/model]"),
    Command("method", "Choose how work is orchestrated", "[name]"),
    Command("effort", "Set reasoning effort", "[low|medium|high|xhigh|max|default]"),
    Command("theme", "Choose the colour theme", "[id]"),
    Command(
        "permissions", "Set what agents may do without asking", "[ask|accept-edits|auto|read-only]"
    ),
    Command("settings", "Open settings"),
    Command("skills", "Browse, approve and remove skills"),
    Command("skillify", "Distil this session's work into new skills", "[what to focus on]"),
    Command("init", "Write an AGENTS.md describing this project for agents"),
    Command("undo", "Revert the file changes of the last run"),
    Command("compact", "Summarise the conversation to free up context"),
    Command("context", "Show how full the context window is"),
    Command("cost", "Show token usage and estimated cost"),
    Command("new", "Start a new session"),
    Command("resume", "Reopen an earlier session"),
    Command("quit", "Leave Clankit"),
)

INIT_PROMPT = """\
Create an AGENTS.md file at the root of this project (or improve the existing one). It is \
read by coding agents at the start of every session, so it should contain only what an \
agent cannot quickly work out for itself and would otherwise get wrong:

- the exact commands to install, build, run, lint, type-check and test (including how to \
run a single test);
- the layout of the code at a level that helps navigation, and where new code of each kind \
belongs;
- conventions that differ from the defaults of the language or framework;
- anything surprising: generated files that must not be edited, required environment, \
steps that are slow or need care.

Investigate the repository first (build files, CI configuration, existing documentation, \
a sample of the code and tests) and verify the commands you list by running the cheap ones. \
Keep it short and factual: no generic advice, no restating what is obvious from the tree."""


def parse(text: str) -> tuple[str, str] | None:
    """Split ``/name argument`` into (name, argument); None if `text` is not a command."""
    stripped = text.strip()
    if not stripped.startswith("/") or stripped.startswith("//") or len(stripped) < 2:
        return None
    name, _, argument = stripped[1:].partition(" ")
    if not name or "/" in name:  # a path such as /usr/bin is not a command
        return None
    return name.lower(), argument.strip()


def find(name: str) -> Command | None:
    return next((c for c in COMMANDS if c.name == name), None)


def complete(prefix: str) -> list[Command]:
    """Commands whose name starts with `prefix` (which may include the leading slash)."""
    prefix = prefix.lstrip("/").lower()
    return [c for c in COMMANDS if c.name.startswith(prefix)]


def _tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def usage_summary(session: Session) -> str:
    """Token usage of the session with an estimated cost when prices are known."""
    usage = session.usage
    parts = [
        f"{_tokens(usage.prompt_tokens)} tokens in",
        f"{_tokens(usage.output_tokens)} out",
    ]
    if usage.cache_read_tokens:
        parts.append(f"{_tokens(usage.cache_read_tokens)} of the input read from cache")
    cost = session.cost()
    if cost is not None:
        parts.append(f"about ${cost:.2f}")
    return ", ".join(parts)


async def execute(runtime: Runtime, session: Session, name: str, argument: str = "") -> Outcome:
    """Run a command. Unknown commands and bad arguments come back as error outcomes."""
    command = find(name)
    if command is None:
        close = complete(name[:2])
        hint = f" Did you mean /{close[0].name}?" if close else " Type /help for the list."
        return Outcome(f"Unknown command /{name}.{hint}", level="error")

    try:
        if name == "help":
            return Outcome(view="help")
        if name == "settings":
            return Outcome(view="settings")
        if name == "skills":
            return Outcome(view="skills")
        if name == "resume":
            return Outcome(view="sessions")
        if name == "new":
            return Outcome(action="new-session")
        if name == "quit":
            return Outcome(action="quit")
        if name == "init":
            return Outcome(prompt=INIT_PROMPT)
        if name == "model":
            return _set_model(runtime, session, argument)
        if name == "method":
            return _set_method(runtime, session, argument)
        if name == "theme":
            return _set_theme(runtime, argument)
        if name == "effort":
            return _set_effort(runtime, argument)
        if name == "permissions":
            return _set_permissions(runtime, argument)
        if name == "undo":
            restored = session.undo()
            if not restored:
                return Outcome("Nothing to undo: no files were changed through the file tools.")
            return Outcome(
                f"Reverted {len(restored)} file(s). Effects of shell commands are not reverted."
            )
        if name == "compact":
            done = await session.compact()
            return Outcome("Conversation summarised." if done else "Nothing to compact yet.")
        if name == "context":
            usage = session.context_usage()
            if usage is None:
                return Outcome("No conversation yet.")
            used, window = usage
            return Outcome(
                f"Context: {_tokens(used)} of {_tokens(window)} tokens ({used * 100 // max(window, 1)}%)."
            )
        if name == "cost":
            return Outcome(f"This session: {usage_summary(session)}.")
        if name == "skillify":
            from clankit.skills import forge

            skills = await forge.generate_from_session(session, focus=argument)
            if not skills:
                return Outcome()  # the generator has already explained why, as a notice
            names = ", ".join(s.name for s in skills)
            staged = any(s.pending for s in skills)
            return Outcome(
                f"Proposed skill(s): {names}. Review them with /skills."
                if staged
                else f"Saved skill(s): {names}."
            )
    except (ConfigError, RuntimeError, ValueError) as exc:
        return Outcome(str(exc), level="error")
    return Outcome(f"/{name} is not available here.", level="error")


def _set_model(runtime: Runtime, session: Session, argument: str) -> Outcome:
    if not argument:
        return Outcome(view="models")
    ref = ModelRef.parse(argument)
    if ref.provider not in runtime.providers.provider_ids():
        usable = ", ".join(runtime.providers.provider_ids()) or "none yet"
        return Outcome(
            f"The provider {ref.provider!r} is not set up (usable providers: {usable}). "
            "Add its API key in /settings.",
            level="error",
        )
    runtime.update_config(lambda c: setattr(c.models, "default", str(ref)))
    session.model_override = None
    known = runtime.providers.model_info(ref) is not None
    note = "" if known else " (not a model Clankit knows; it will be tried as written)"
    return Outcome(f"Model set to {ref}{note}. It applies to new conversations and subagents.")


def _set_method(runtime: Runtime, session: Session, argument: str) -> Outcome:
    from clankit.orchestration import get_orchestrator, method_names

    if not argument:
        return Outcome(view="methods")
    if argument not in method_names():
        return Outcome(
            f"Unknown method {argument!r}. Available: {', '.join(method_names())}.", level="error"
        )
    runtime.update_config(lambda c: setattr(c.orchestration, "method", argument))
    session.method_override = None
    return Outcome(f"Orchestration method set to {get_orchestrator(argument).title}.")


def _set_theme(runtime: Runtime, argument: str) -> Outcome:
    from clankit.themes import all_themes

    if not argument:
        return Outcome(view="themes")
    themes = all_themes(runtime.config)
    if argument not in themes:
        return Outcome(
            f"Unknown theme {argument!r}. Available: {', '.join(themes)}.", level="error"
        )
    runtime.update_config(lambda c: setattr(c.ui, "theme", argument))
    return Outcome(f"Theme set to {themes[argument].name}.")


def _set_effort(runtime: Runtime, argument: str) -> Outcome:
    current = runtime.config.agent.effort or "default"
    if not argument:
        return Outcome(
            f"Reasoning effort is {current}. Options: {', '.join(EFFORT_LEVELS)}, default."
        )
    value = "" if argument == "default" else argument
    if value and value not in EFFORT_LEVELS:
        return Outcome(
            f"Effort must be one of: {', '.join(EFFORT_LEVELS)}, default.", level="error"
        )
    runtime.update_config(lambda c: setattr(c.agent, "effort", value))
    return Outcome(
        f"Reasoning effort set to {argument}. It applies to new conversations and subagents."
    )


def _set_permissions(runtime: Runtime, argument: str) -> Outcome:
    if not argument:
        return Outcome(
            f"Permission mode is {runtime.config.permissions.mode}. Options: {', '.join(PERMISSION_MODES)}."
        )
    if argument not in PERMISSION_MODES:
        return Outcome(f"Mode must be one of: {', '.join(PERMISSION_MODES)}.", level="error")
    runtime.update_config(lambda c: setattr(c.permissions, "mode", argument))
    return Outcome(f"Permission mode set to {argument}.")
