"""The bash tool: run a shell command in the workspace.

Three things here need more care than "call subprocess":

* **Containment.** The command runs in its own session and process group with stdin closed.
  When it times out or the user interrupts, the whole group is stopped (SIGTERM, a short
  grace period, SIGKILL), so no test runner or dev server outlives the call that started it.
  On Windows the command runs in WSL bash, Git Bash or cmd.exe - whichever is installed - in
  a console group of its own, stopped with CTRL_BREAK and then terminated.
* **Bounded output.** Only the beginning and the end of what the command prints are kept,
  however much it prints.
* **Approval.** Every command asks, except a small set that can be shown, from its text alone,
  to do nothing but read inside the workspace (`is_inert`). That analysis fails closed: any
  shell syntax it does not fully understand makes a command "not inert".
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from clankit.tools.base import PermissionAsk, Tool, ToolContext, ToolOutput
from clankit.tools.files import inside_workspace, looks_secret, output_budget

__all__ = ["BashTool", "is_inert", "session_rule"]

#: Seconds a command may run when neither the call nor the configuration says otherwise.
DEFAULT_TIMEOUT = 120
#: The longest timeout a call can ask for, in seconds.
MAX_TIMEOUT = 600
#: Seconds between asking a command's processes to stop (SIGTERM) and killing them.
TERM_GRACE = 2.0
#: Seconds to keep collecting output after the shell has exited. Only matters when the
#: command left a background process holding its output open.
DRAIN_GRACE = 0.5

#: Stops common tools from paging, prompting or opening an editor. Nothing is there to answer.
_UNATTENDED = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "MANPAGER": "cat",
    "SYSTEMD_PAGER": "",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_EDITOR": "true",
    "DEBIAN_FRONTEND": "noninteractive",
    "PIP_NO_INPUT": "1",
}


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command with bash in the workspace directory and return what it printed "
        "(stdout and stderr together) and its exit code. Use it to run tests, builds, "
        "linters, git and other programs. Do not use it for what a dedicated tool does: "
        "read files with read (not cat, head or tail), search with grep and glob (not grep, "
        "rg, find or ls -R), and change files with edit and write (not sed, tee or "
        "redirection). The dedicated tools are tracked and undoable, respect other agents' "
        "work, and need fewer approvals. Commands must run unattended: stdin is closed, so "
        "anything that prompts, pages or opens an editor fails; pass --yes, -m and the like. "
        "Every call starts a new shell in the workspace directory, so cd and variables do "
        "not carry over; chain steps with && instead. On Windows the shell is WSL bash when "
        "it is installed, else Git Bash, else cmd.exe. A command that runs past its timeout "
        "is killed together with everything it started. Long output is shortened to its "
        "beginning and end; redirect to a file and search that if you need all of it. A "
        "process meant to keep running after the call must be started in the background "
        "with its output redirected (cmd > log 2>&1 &)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command line to run."},
            "timeout": {
                "type": "integer",
                "description": (
                    f"Seconds to allow, at most {MAX_TIMEOUT}. Leave out for the configured "
                    f"default (normally {DEFAULT_TIMEOUT}); raise it for slow builds and "
                    "test suites."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "What the command does, in a few plain words. Shown to the user who "
                    "has to approve it."
                ),
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"$ {_one_line(tool_input['command'], 120)}"

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        command = tool_input["command"].strip()
        if not command or is_inert(command, ctx):
            return None
        description = (tool_input.get("description") or "").strip()
        return PermissionAsk(
            "shell",
            f"Run {_one_line(command, 80)}",
            command,
            detail=f"{command}\n\n{description}" if description else command,
            session_rule=session_rule(command),
        )

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        command = tool_input["command"]
        if not command.strip():
            return ToolOutput.error("command is empty; there is nothing to run.")
        timeout = _timeout(tool_input.get("timeout"), ctx)
        capture = _Capture(output_budget(ctx) - 400)  # room for the status line and notes
        try:
            code, timed_out = await _execute(command, ctx, timeout, capture)
        except OSError as exc:
            return ToolOutput.error(f"The command could not be started: {exc.strerror or exc}.")
        return _report(capture.text().rstrip("\n"), code, timed_out, timeout)


def _one_line(command: str, width: int) -> str:
    first, _, rest = command.strip().partition("\n")
    if len(first) > width:
        return f"{first[: width - 4]} ..."
    return f"{first} ..." if rest.strip() else first


def _timeout(requested: Any, ctx: ToolContext) -> int:
    configured = ctx.config.agent.shell_timeout if ctx.config is not None else DEFAULT_TIMEOUT
    seconds = requested if requested and requested > 0 else configured
    return max(1, min(int(seconds), MAX_TIMEOUT))


def _report(output: str, code: int, timed_out: bool, timeout: int) -> ToolOutput:
    """The result as the model reads it: what happened first, then what was printed."""
    if timed_out:
        return ToolOutput.error(
            f"Timed out after {timeout} s: the command and everything it started were "
            f"killed. If it needs longer, pass a larger timeout (up to {MAX_TIMEOUT} s); if "
            "it was waiting for input, run it in a way that needs none.\n\n"
            f"Output before the timeout:\n{output or '(none)'}"
        )
    if code == 0:
        return ToolOutput(output or "(The command succeeded and printed nothing.)")
    status = f"Exit code {code}" if code > 0 else f"Killed by {_signal_name(-code)}"
    return ToolOutput(f"{status}\n\n{output or '(The command printed nothing.)'}")


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


# --------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------


class _Capture:
    """The beginning and the end of a byte stream, within a fixed budget.

    The start of a command's output says what ran and usually holds the first error; the end
    holds the summary. What falls in between is counted and dropped.
    """

    def __init__(self, budget: int) -> None:
        self._head_room = budget // 2
        self._tail_room = budget - self._head_room
        self._head = bytearray()
        self._tail = bytearray()
        self._dropped = 0

    def feed(self, chunk: bytes) -> None:
        room = self._head_room - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        self._tail += chunk
        excess = len(self._tail) - self._tail_room
        if excess > 0:
            del self._tail[:excess]
            self._dropped += excess

    def text(self) -> str:
        if not self._dropped:
            return (self._head + self._tail).decode(errors="replace")
        head, tail = self._head.decode(errors="replace"), self._tail.decode(errors="replace")
        return (
            f"{head}\n\n[... {self._dropped:,} bytes of output omitted here. To see all of "
            f"it, redirect the output to a file and search that with grep ...]\n\n{tail}"
        )


class _OutputPipe:
    """The pipe a command writes to.

    We own the read end instead of letting asyncio manage it, so that the call can close it
    for certain when it is over: a background process the command left behind may hold the
    write end open indefinitely, and nothing should keep reading on its behalf.
    """

    def __init__(self, capture: _Capture) -> None:
        self._capture = capture
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._read_fd, False)
        self._loop = asyncio.get_running_loop()
        self._ended = asyncio.Event()
        self._loop.add_reader(self._read_fd, self._on_readable)

    @property
    def write_fd(self) -> int:
        return self._write_fd

    def _on_readable(self) -> None:
        try:
            chunk = os.read(self._read_fd, 65536)
        except BlockingIOError:
            return
        if chunk:
            self._capture.feed(chunk)
        else:  # every writer has closed its end
            self._loop.remove_reader(self._read_fd)
            self._ended.set()

    def release_write_end(self) -> None:
        """Close our copy of the write end once the command has inherited it; until then
        the read end can never see end-of-file."""
        os.close(self._write_fd)

    async def drain(self, patience: float) -> None:
        """Wait until all output has arrived, but no longer than `patience` seconds."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._ended.wait(), patience)

    def close(self) -> None:
        self._loop.remove_reader(self._read_fd)
        os.close(self._read_fd)


def _shell() -> str:
    if os.access("/bin/bash", os.X_OK):
        return "/bin/bash"
    return shutil.which("bash") or shutil.which("sh") or "/bin/sh"


def _environment(ctx: ToolContext) -> dict[str, str]:
    return {**os.environ, **_UNATTENDED, **ctx.env}


async def _execute(
    command: str, ctx: ToolContext, time_limit: int, capture: _Capture
) -> tuple[int, bool]:
    """Run `command` to its end or for `time_limit` seconds, feeding what it prints to
    `capture`.

    Returns the exit code and whether the time limit struck. Whenever the command does not end
    by itself (timeout, cancellation), its process group is gone before this returns or
    raises.
    """
    if sys.platform == "win32":
        return await _execute_windows(command, ctx, time_limit, capture)
    pipe = _OutputPipe(capture)
    try:
        try:
            process = await asyncio.create_subprocess_exec(
                _shell(),
                "-c",
                command,
                cwd=ctx.workspace,
                env=_environment(ctx),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=pipe.write_fd,
                stderr=pipe.write_fd,
                # A session of its own: one process group to signal, and no controlling
                # terminal for a child to grab (which would fight a terminal UI for input).
                start_new_session=True,
            )
        finally:
            pipe.release_write_end()
        try:
            timed_out = not await _exits_within(process, time_limit)
            if timed_out:
                await _stop_group(process)
            await pipe.drain(DRAIN_GRACE)
        except BaseException:
            await _stop_group(process)
            raise
        assert process.returncode is not None
        return process.returncode, timed_out
    finally:
        pipe.close()


async def _execute_windows(
    command: str, ctx: ToolContext, time_limit: int, capture: _Capture
) -> tuple[int, bool]:
    """`_execute` for Windows: the same contract, through a native console process group.

    asyncio cannot watch file descriptors for readability on Windows (`add_reader` raises
    `NotImplementedError` for pipes), so the output is read from the subprocess stream by a
    pump task instead. The group is stopped with CTRL_BREAK rather than signals.
    """
    process = await asyncio.create_subprocess_exec(
        *_windows_command(command),
        cwd=ctx.workspace,
        env=_environment(ctx),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # A console group of its own: CTRL_BREAK stops the command and everything it started.
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    assert process.stdout is not None
    pumping = asyncio.create_task(_pump(process.stdout, capture))
    try:
        timed_out = not await _exits_within(process, time_limit)
        if timed_out:
            await _stop_windows(process)
        await _drain(pumping, DRAIN_GRACE)
    except BaseException:
        await _stop_windows(process)
        raise
    assert process.returncode is not None
    return process.returncode, timed_out


def _windows_command(command: str) -> list[str]:
    """The command as a program and arguments: WSL or Git Bash when either is installed,
    cmd.exe otherwise. (On Windows, WSL puts its own ``bash.exe`` on the PATH.)"""
    bash = shutil.which("bash")
    if bash:
        return [bash, "-c", command]
    # `getenv` is case-insensitive on Windows, where the variable is "ComSpec".
    comspec = os.getenv("ComSpec", "") or shutil.which("cmd") or "cmd.exe"  # noqa: SIM112
    return [comspec, "/d", "/c", command]


async def _pump(stream: asyncio.StreamReader, capture: _Capture) -> None:
    """Read one subprocess stream to its end, feeding `capture`."""
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        capture.feed(chunk)


async def _drain(pumping: asyncio.Task[None], patience: float) -> None:
    """Wait for the pump to finish, but no longer than `patience` seconds: a background
    process the command left behind may hold the stream open indefinitely."""
    try:
        await asyncio.wait_for(asyncio.shield(pumping), patience)
    except TimeoutError:
        pumping.cancel()


async def _stop_windows(process: asyncio.subprocess.Process) -> None:
    """End the command's console group: CTRL_BREAK, time to clean up, then terminate."""
    if process.returncode is not None:
        return
    _ctrl_break(process)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TERM_GRACE
    try:
        while process.returncode is None and loop.time() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.02)
    finally:
        if process.returncode is None:
            process.kill()
    await process.wait()


def _ctrl_break(process: asyncio.subprocess.Process) -> None:
    """CTRL_BREAK reaches every console process in the command's group, like a group signal."""
    event = getattr(signal, "CTRL_BREAK_EVENT", None)
    if event is None:
        return
    with contextlib.suppress(OSError, ProcessLookupError):
        os.kill(process.pid, event)


async def _exits_within(process: asyncio.subprocess.Process, seconds: float) -> bool:
    try:
        await asyncio.wait_for(process.wait(), seconds)
    except TimeoutError:
        return False
    return True


async def _stop_group(process: asyncio.subprocess.Process) -> None:
    """End every process in the command's group: SIGTERM, time to clean up, then SIGKILL.

    The shell was started as a session leader, so its pid is the group id, and everything it
    started is in the group unless it deliberately left. The SIGKILL is sent even if this
    task is cancelled again while waiting.
    """
    group = process.pid
    _signal_group(group, signal.SIGTERM)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TERM_GRACE
    try:
        # Nothing signals that a process group has emptied; it can only be polled.
        while _signal_group(group, 0) and loop.time() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.02)
    finally:
        _signal_group(group, signal.SIGKILL)
    await process.wait()


def _signal_group(group: int, signum: int) -> bool:
    """Send `signum` to the process group (0 only probes). False if the group is gone."""
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, but holds a process we may not signal
        return True
    return True


# --------------------------------------------------------------------------------------------
# Which commands need no approval
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Word:
    """One word of a command line after the shell's quote removal."""

    text: str
    #: Contains an unquoted ``*``, ``?`` or ``[``: the shell may expand it to file names
    #: that this analysis never sees.
    globbing: bool


#: Outside quotes these chain commands, redirect, substitute, expand or escape. A command
#: containing any of them is beyond what is analysed here.
_LIVE_SYNTAX = frozenset(";&|<>()$`\\{}\n\r")
_LIVE_IN_DOUBLE_QUOTES = frozenset("$`\\")


def _split_words(command: str) -> list[_Word] | None:
    """Split a simple command into words as the shell would, or return None if it contains
    any syntax that makes the shell do more than run one program with literal arguments."""
    words: list[_Word] = []
    text: list[str] = []
    in_word = globbing = False
    quote = ""
    for char in command:
        if quote:
            if char == quote:
                quote = ""
            elif quote == '"' and char in _LIVE_IN_DOUBLE_QUOTES:
                return None
            else:
                text.append(char)
        elif char in "'\"":
            quote, in_word = char, True
        elif char in " \t":
            if in_word:
                words.append(_Word("".join(text), globbing))
                text, in_word, globbing = [], False, False
        elif char in _LIVE_SYNTAX:
            return None
        else:
            text.append(char)
            in_word = True
            globbing = globbing or char in "*?["
    if quote:
        return None
    if in_word:
        words.append(_Word("".join(text), globbing))
    return words


def is_inert(command: str, ctx: ToolContext) -> bool:
    """Can `command` be shown to do nothing but read inside the workspace?

    True only for one allow-listed program with literal arguments, no shell syntax, no option
    that writes, executes or follows links out of the tree, and no operand that is outside
    the workspace or looks like a secret. Everything else, including anything not understood,
    is False and goes to the permission gate.
    """
    words = _split_words(command)
    if not words or words[0].globbing:
        return False
    check = _INERT_PROGRAMS.get(words[0].text)
    return check is not None and check(words[1:], ctx)


def _harmless(args: Sequence[_Word], ctx: ToolContext) -> bool:
    return True


def _lists_workspace(args: Sequence[_Word], ctx: ToolContext) -> bool:
    """`ls`: names only, so wildcards are fine, but still only inside the workspace."""
    return all(
        arg.text.startswith("-") or inside_workspace(ctx, ctx.resolve(arg.text)) for arg in args
    )


def _reads_workspace(args: Sequence[_Word], ctx: ToolContext) -> bool:
    return all(_plain_operand(arg, ctx) for arg in args)


def _plain_operand(arg: _Word, ctx: ToolContext) -> bool:
    """An argument of a program that prints file contents: an option, or something that, if
    it is a path, is inside the workspace and not a secret. (Patterns, numbers and revisions
    pass as harmless relative "paths"; a wildcard does not, since its expansion is unknown.)"""
    if arg.globbing:
        return False
    text = arg.text
    if text.startswith("-"):
        _, glued, value = text.partition("=")
        if not glued:
            return "/" not in text and "~" not in text  # a path glued to a short option
        text = value
    if not text:
        return True
    path = ctx.resolve(text)
    return inside_workspace(ctx, path) and not looks_secret(path)


def _uses(args: Sequence[_Word], short: str, *long: str) -> bool:
    """Is one of these options present? `short` lists single letters; long options count in
    any abbreviated form, because GNU tools and git accept unambiguous prefixes."""
    for arg in args:
        option = arg.text.partition("=")[0]
        if option.startswith("--"):
            if len(option) > 2 and any(name.startswith(option) for name in long):
                return True
        elif option.startswith("-") and any(letter in option for letter in short):
            return True
    return False


def _grep_reads_workspace(args: Sequence[_Word], ctx: ToolContext) -> bool:
    # -R follows symbolic links, which can lead out of the workspace.
    return _reads_workspace(args, ctx) and not _uses(args, "R", "--dereference-recursive")


def _rg_reads_workspace(args: Sequence[_Word], ctx: ToolContext) -> bool:
    # --pre and --hostname-bin run a program; -L follows symbolic links.
    return _reads_workspace(args, ctx) and not _uses(
        args, "L", "--follow", "--pre", "--pre-glob", "--hostname-bin"
    )


# fmt: off
_GIT_BRANCH_LISTING = frozenset({
    "-a", "-r", "-v", "-vv", "-l", "--all", "--remotes", "--verbose", "--list",
    "--show-current", "--color", "--no-color",
})
# fmt: on


def _git_reads_workspace(args: Sequence[_Word], ctx: ToolContext) -> bool:
    """`git status|diff|log|show|branch` in their reading forms.

    Not inert: options placed before the subcommand other than ``--no-pager`` (``-c`` can
    make git run anything), ``--output`` (writes a file), ``--ext-diff`` (runs a program),
    ``--no-index`` (reads arbitrary files), and any ``git branch`` that is not a plain
    listing.
    """
    if args and args[0].text == "--no-pager":
        args = args[1:]
    if not args or any(arg.globbing for arg in args):
        return False
    subcommand, rest = args[0].text, args[1:]
    if subcommand == "branch":
        return all(arg.text in _GIT_BRANCH_LISTING for arg in rest)
    if subcommand not in ("status", "diff", "log", "show"):
        return False
    return _reads_workspace(rest, ctx) and not _uses(
        rest, "", "--output", "--ext-diff", "--no-index"
    )


_INERT_PROGRAMS: dict[str, Callable[[Sequence[_Word], ToolContext], bool]] = {
    "pwd": _harmless,
    "echo": _harmless,
    "which": _harmless,
    "ls": _lists_workspace,
    "cat": _reads_workspace,
    "head": _reads_workspace,
    "tail": _reads_workspace,
    "wc": _reads_workspace,
    "grep": _grep_reads_workspace,
    "rg": _rg_reads_workspace,
    "git": _git_reads_workspace,
}


# --------------------------------------------------------------------------------------------
# What "allow for this session" should remember
# --------------------------------------------------------------------------------------------

# fmt: off
#: Programs whose first argument names what they do: the rule keeps it (``git status *``).
_SUBCOMMAND_PROGRAMS = frozenset({
    "git", "npm", "pnpm", "yarn", "bun", "cargo", "go", "uv", "pip", "pip3", "poetry", "pdm",
    "hatch", "pipenv", "conda", "docker", "kubectl", "helm", "gh", "make", "apt", "apt-get",
    "brew", "dotnet", "gradle", "mvn", "terraform", "systemctl", "rustup",
})
#: Subcommands that run whatever follows: the rule keeps the program they run as well.
_RUNNER_SUBCOMMANDS = frozenset({
    "uv run", "poetry run", "pdm run", "hatch run", "pipenv run", "conda run", "npm exec",
    "pnpm exec", "pnpm dlx", "yarn dlx", "docker run", "docker exec", "kubectl exec",
})
#: Programs that run a script or package named by their first argument.
_SCRIPT_RUNNERS = frozenset({
    "python", "node", "ruby", "perl", "php", "bash", "sh", "zsh", "deno", "npx", "uvx",
    "bunx", "pipx",
})
#: Prefixes that merely modify how the rest runs: the rule looks through them.
_TRANSPARENT = frozenset({"sudo", "doas", "time", "nice", "nohup"})
#: Programs for which no pattern is offered, only the exact command: they run arbitrary
#: other programs, or do damage that a broad rule would wave through.
_NEVER_GENERALISED = frozenset({
    "env", "xargs", "eval", "exec", "command", "timeout", "watch", "ssh", "rm", "rmdir",
    "dd", "mkfs", "shred", "truncate", "chmod", "chown", "kill", "pkill", "killall",
    "shutdown", "reboot", "curl", "wget",
})
# fmt: on
_PYTHON = re.compile(r"python[0-9.]*")


def session_rule(command: str) -> str:
    """The pattern to remember if the user allows `command` for the rest of the session.

    ``git status -s`` gives ``git status *``, ``pytest -q`` gives ``pytest *`` and
    ``uv run pytest -q`` gives ``uv run pytest *``. A command with shell syntax, with no
    arguments beyond what identifies it, or led by a program too powerful to generalise
    gives exactly itself, with wildcard characters escaped.
    """
    command = command.strip()
    exact = glob.escape(command)
    words = _split_words(command)
    if not words:
        return exact
    names = [word.text for word in words]
    depth = _identifying_words(names)
    prefix = " ".join(names[:depth])
    # The rule is matched against the command as written, so the prefix has to be a literal
    # prefix of it (it is not when its words were quoted or spaced unusually).
    if depth == 0 or len(names) <= depth or not command.startswith(f"{prefix} "):
        return exact
    return f"{glob.escape(prefix)} *"


def _identifying_words(names: Sequence[str]) -> int:
    """How many leading words say what a command does; 0 if no prefix is safe to generalise."""
    if not names or names[0].startswith("-") or "=" in names[0]:
        return 0
    program = os.path.basename(names[0])
    if _PYTHON.fullmatch(program):
        program = "python"
    if program in _NEVER_GENERALISED:
        return 0
    if program in _TRANSPARENT:
        return _extended(1, names)
    if " ".join((program, *names[1:2])) in _RUNNER_SUBCOMMANDS:
        return _extended(2, names)
    named_second = len(names) > 1 and not names[1].startswith("-")
    if program in _SCRIPT_RUNNERS:
        if len(names) > 2 and names[1] == "-m":
            return 3
        return 2 if named_second else 0
    if program in _SUBCOMMAND_PROGRAMS:
        return 2 if named_second else 0
    return 1


def _extended(depth: int, names: Sequence[str]) -> int:
    inner = _identifying_words(names[depth:])
    return depth + inner if inner else 0
