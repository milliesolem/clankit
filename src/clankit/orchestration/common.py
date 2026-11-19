"""Building blocks shared by orchestration methods.

Nothing here decides *how* work is organised; these are the parts every method needs:
a finishing tool for structured results, a way to insist that an agent actually produces
one, harness-side verification commands, harness-computed diffs, and help for agents that
choose a model for someone else's work.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import os
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clankit.agent.loop import Agent, AgentResult
from clankit.llm.types import ModelRef
from clankit.tools.base import PermissionAsk, Tool, ToolContext, ToolOutput

if TYPE_CHECKING:
    from clankit.agent.session import Session


# --------------------------------------------------------------------------------------------
# Structured results
# --------------------------------------------------------------------------------------------


class FinishTool(Tool):
    """A control tool that ends the agent's run and returns its input as the result.

    This is how agents hand back structured data on every provider: the tool's schema is the
    result schema, the agent loop validates the call against it (and sends validation errors
    back to the model), and `AgentResult.payload` carries the accepted input. Forced tool
    choice is deliberately not used; current models reject it. Use `run_to_finish` to insist.

    Schema advice: keep it flat and small, mark what is required, and never name a field
    "reasoning", "thinking" or "trace" - safety classifiers treat requests to write out
    reasoning as an extraction attempt and refuse.
    """

    def __init__(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        *,
        reply: str = "Recorded.",
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = schema
        self._reply = reply

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return self.name

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput(self._reply, finish=dict(tool_input))


async def run_to_finish(
    agent: Agent,
    prompt: str,
    tool_name: str,
    *,
    reminders: int = 2,
) -> tuple[AgentResult, dict[str, Any] | None]:
    """Run `agent` on `prompt` and make sure it ends by calling the finishing tool `tool_name`.

    Models sometimes just stop talking instead of calling the tool. Each time that happens
    the agent is reminded, up to `reminders` times. Returns the last result and the payload,
    which is None if the agent never called the tool (or was stopped by a limit or refusal).
    """
    result = await agent.run(prompt)
    for _ in range(reminders):
        if result.stop != "end_turn":
            break
        result = await agent.run(
            f"You ended your turn without calling `{tool_name}`. Your work is not handed back "
            f"until you do. If you are finished, call `{tool_name}` now with your result; if you "
            "are not, continue the work and call it when you are."
        )
    return result, result.payload if result.stop == "finished" else None


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def models_guide(session: Session, *, limit: int = 24) -> str:
    """A list of the models an agent may assign to other agents, with the notes from the
    configuration and catalog, for inclusion in a role prompt. "" if only one is known."""
    models = session.runtime.providers.known_models()
    if len(models) < 2:
        return ""
    lines = []
    for info in models[:limit]:
        facts = []
        if info.free:
            facts.append("free")
        if info.supports_tools is False:
            facts.append("no tool calling: cannot be used for agents")
        note = info.note.strip()
        detail = "; ".join([*facts, note] if note else facts)
        lines.append(f"- `{info.ref}`" + (f": {detail}" if detail else ""))
    if len(models) > limit:
        lines.append(f"- ({len(models) - limit} more models are configured)")
    return "\n".join(lines)


def resolve_model(
    session: Session, requested: str | None, *, role: str | None = "subagent"
) -> ModelRef:
    """Turn an agent's model choice into a usable `ModelRef`.

    None or "" means "no preference": the model configured for `role` is used. Raises
    ValueError, with a message meant for the model that made the choice, when the reference
    is malformed or its provider cannot be used.
    """
    if not requested or not requested.strip():
        return session.model_for(role)
    try:
        ref = ModelRef.parse(requested)
    except ValueError as exc:
        raise ValueError(f"{exc}. Omit the model to use the default.") from exc
    usable = session.runtime.providers.provider_ids()
    if ref.provider not in usable:
        known = ", ".join(str(m.ref) for m in session.runtime.providers.known_models()[:12])
        raise ValueError(
            f"the provider {ref.provider!r} is not configured, so {requested!r} cannot be used. "
            f"Omit the model to use the default, or choose one of: {known or 'none'}."
        )
    return ref


# --------------------------------------------------------------------------------------------
# Diffs computed by the harness
# --------------------------------------------------------------------------------------------


def _decode(data: bytes | None) -> list[str] | None:
    if data is None:
        return []
    try:
        return data.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return None


def run_diff(session: Session, paths: Sequence[Path] | None = None, *, limit: int = 60_000) -> str:
    """Unified diff of what the file tools changed during the current run.

    Computed by the harness from pre-run snapshots, not reported by any agent, so it can be
    handed to a reviewer as evidence. `paths` restricts it to some files. Changes made by
    shell commands are not included. The diff is cut at about `limit` characters.
    """
    originals = session.files.originals()
    wanted = set(paths) if paths is not None else None
    chunks: list[str] = []
    size = 0
    for path in sorted(originals):
        if wanted is not None and path not in wanted:
            continue
        try:
            rel = path.relative_to(session.workspace).as_posix()
        except ValueError:
            rel = str(path)
        before = _decode(originals[path])
        try:
            after = _decode(path.read_bytes()) if path.exists() else []
        except OSError:
            after = None
        if before is None or after is None:
            chunk = f"Binary file {rel} changed\n"
        elif before == after:
            continue
        else:
            chunk = "".join(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"a/{rel}" if originals[path] is not None else "/dev/null",
                    tofile=f"b/{rel}" if path.exists() else "/dev/null",
                )
            )
            if chunk and not chunk.endswith("\n"):
                chunk += "\n"
        if size + len(chunk) > limit:
            chunks.append(f"[diff truncated: {rel} and any later files are not shown]\n")
            break
        chunks.append(chunk)
        size += len(chunk)
    return "".join(chunks)


def subtree_ids(session: Session, node_id: str) -> list[str]:
    """`node_id` and all its descendants, depth first."""
    ids = [node_id]
    for child in session.children_of(node_id):
        ids.extend(subtree_ids(session, child.id))
    return ids


def changed_paths(session: Session, node_ids: Sequence[str]) -> list[Path]:
    """Absolute paths the given nodes' agents changed through the file tools."""
    seen: list[Path] = []
    for node_id in node_ids:
        for path in session.files.changed_by(node_id):
            if path not in seen:
                seen.append(path)
    return seen


def diff_stat(session: Session, paths: Sequence[Path]) -> list[str]:
    """One line per changed file, ``path  +added -removed`` (``new``/``deleted`` noted)."""
    originals = session.files.originals()
    lines = []
    for path in paths:
        try:
            rel = path.relative_to(session.workspace).as_posix()
        except ValueError:
            rel = str(path)
        before = _decode(originals.get(path)) if path in originals else None
        try:
            after = _decode(path.read_bytes()) if path.exists() else []
        except OSError:
            after = None
        if before is None or after is None:
            lines.append(f"{rel}  (binary or unreadable)")
            continue
        added = removed = 0
        for line in difflib.unified_diff(before, after, n=0):
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        state = (
            " (new)" if originals.get(path) is None else (" (deleted)" if not path.exists() else "")
        )
        lines.append(f"{rel}  +{added} -{removed}{state}")
    return lines


def changed_files(session: Session, node_ids: Sequence[str]) -> list[str]:
    """Workspace-relative paths the given nodes' agents changed through the file tools."""
    seen: list[str] = []
    for node_id in node_ids:
        for path in session.files.changed_by(node_id):
            try:
                rel = path.relative_to(session.workspace).as_posix()
            except ValueError:
                rel = str(path)
            if rel not in seen:
                seen.append(rel)
    return seen


# --------------------------------------------------------------------------------------------
# Verification commands
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class CheckResult:
    """The outcome of a verification command run by the harness (not by an agent)."""

    command: str
    passed: bool
    exit_code: int | None
    output: str
    denied: bool = False

    def render(self, *, tail: int = 3000) -> str:
        if self.denied:
            return f"$ {self.command}\n(not run: permission was denied)"
        status = "passed" if self.passed else f"FAILED (exit code {self.exit_code})"
        output = self.output if len(self.output) <= tail else "[...]\n" + self.output[-tail:]
        return f"$ {self.command}\n{status}\n{output}".rstrip()


async def run_check(
    session: Session,
    ctx: ToolContext,
    command: str,
    *,
    seconds: float | None = None,
    output_limit: int = 20_000,
) -> CheckResult:
    """Run a verification command in the workspace on behalf of the harness.

    The command still goes through the permission gate as a shell action, attributed to the
    node whose work it checks. Output is stdout and stderr interleaved, trimmed to its tail.
    The whole process group is killed after `seconds` (default: the configured shell
    timeout) or on cancellation.
    """
    ask = PermissionAsk(
        "shell",
        f"Run check: {command}",
        command,
        detail=command,
        session_rule=command,
    )
    decision = await ctx.permissions.check(ask, node_id=ctx.node_id, tool="check")
    if not decision.allowed:
        return CheckResult(command, passed=False, exit_code=None, output="", denied=True)

    limit = seconds or float(session.config.agent.shell_timeout)
    env = {**os.environ, **ctx.env, "PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"}
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=session.workspace,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    try:
        raw, _ = await asyncio.wait_for(process.communicate(), timeout=limit)
    except (TimeoutError, asyncio.CancelledError) as exc:
        _kill_group(process)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
        if isinstance(exc, asyncio.CancelledError):
            raise
        return CheckResult(
            command, passed=False, exit_code=None, output=f"Timed out after {limit:.0f}s."
        )
    text = raw.decode("utf-8", errors="replace")
    if len(text) > output_limit:
        text = "[... earlier output omitted ...]\n" + text[-output_limit:]
    return CheckResult(
        command, passed=process.returncode == 0, exit_code=process.returncode, output=text
    )


def _kill_group(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
