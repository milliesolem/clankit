"""The evidence record: what counts as a lesson, and what is shown to the extraction model."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from clankit.agent.events import (
    EventBus,
    NodeAdded,
    NodeInfo,
    RunFinished,
    ToolFinished,
    ToolStarted,
    UserMessage,
)
from clankit.agent.state import SessionView
from clankit.skills.evidence import collect


class History:
    """Builds a `SessionView` the way a session does: by publishing events and folding them."""

    def __init__(self) -> None:
        self.view = SessionView()
        self.bus = EventBus()
        self.bus.subscribe(self.view.apply)
        self._ids = itertools.count(1)
        self.node("n1", role="root")

    def node(self, node_id: str, *, role: str = "leaf", parent: str | None = None) -> None:
        info = NodeInfo(
            id=node_id, parent_id=parent, depth=0 if parent is None else 1, title=role, role=role
        )
        self.bus.publish(NodeAdded(node=info))

    def prompt(self, text: str, node: str = "n1") -> None:
        self.bus.publish(UserMessage(node_id=node, text=text))

    def tool(
        self,
        tool: str,
        tool_input: dict[str, Any],
        output: str = "",
        *,
        failed: bool = False,
        node: str = "n1",
        diff: str = "",
        finished: bool = True,
        summary: str = "",
    ) -> None:
        call_id = f"c{next(self._ids)}"
        self.bus.publish(
            ToolStarted(
                node_id=node, call_id=call_id, tool=tool, summary=summary or tool, input=tool_input
            )
        )
        if finished:
            display = {"type": "diff", "path": diff, "diff": "+x"} if diff else None
            self.bus.publish(
                ToolFinished(
                    node_id=node, call_id=call_id, output=output, is_error=failed, display=display
                )
            )

    def bash(
        self, command: str, output: str = "", *, failed: bool = False, node: str = "n1"
    ) -> None:
        # As the shell tool does: a non-zero exit is an ordinary result that says so.
        result = f"Exit code 1\n\n{output}" if failed else output
        self.tool("bash", {"command": command}, result, node=node)

    def write(self, path: str, content: str = "", node: str = "n1") -> None:
        self.tool("write_file", {"path": path, "content": content}, "written", diff=path, node=node)


def test_a_session_without_struggle_or_reuse_is_not_worth_distilling(workspace: Path) -> None:
    history = History()
    history.prompt("tidy the readme")
    history.tool("read_file", {"path": "README.md"}, "body")
    history.write("README.md", "new text")
    history.bash("git status", "clean")
    evidence = collect(history.view, workspace)

    assert [(s.number, s.tool) for s in evidence.steps] == [
        (1, "read_file"),
        (2, "write_file"),
        (3, "bash"),
    ]
    assert evidence.signals == frozenset() and not evidence.worth_distilling
    assert evidence.prompts == ("tidy the readme",)


def test_a_failed_command_that_was_made_to_work_is_a_recovery(workspace: Path) -> None:
    history = History()
    history.bash("pytest -q", "ModuleNotFoundError", failed=True)
    history.tool("read_file", {"path": "pyproject.toml"})
    history.bash("uv run pytest -q", "12 passed")
    evidence = collect(history.view, workspace)

    assert evidence.recoveries == ((1, 3),)
    assert evidence.signals == {1, 3} and evidence.worth_distilling
    assert evidence.steps[0].output == "Exit code 1\n\nModuleNotFoundError"
    assert evidence.steps[2].output == ""


def test_commands_fail_by_exit_status_signal_or_tool_error(workspace: Path) -> None:
    history = History()
    history.tool("bash", {"command": "make a"}, "Exit code 2\n\nmake: *** [a] Error 1")
    history.tool(
        "bash", {"command": "make b"}, "Killed by SIGKILL\n\n(The command printed nothing.)"
    )
    history.tool("bash", {"command": "make c"}, "Timed out after 120 s", failed=True)
    history.tool("bash", {"command": "make d"}, "built\nExit code 3 was returned by a child")
    history.tool("read", {"file_path": "ci.log"}, "Exit code 1\n\nfrom an old log")
    evidence = collect(history.view, workspace)
    assert [step.failed for step in evidence.steps] == [True, True, True, False, False]


def test_failures_that_are_not_struggles_do_not_count(workspace: Path) -> None:
    history = History()
    history.node("n2", parent="n1")
    history.tool("read_file", {"path": "missing.txt"}, "not found", failed=True)
    history.tool("read_file", {"path": "present.txt"}, "body")  # a failed lookup is not a lesson
    history.bash("make deploy", "denied", failed=True)  # never overcome
    history.bash("make test", "ok", node="n2")  # another agent's success is not this one's fix
    evidence = collect(history.view, workspace)
    assert evidence.recoveries == () and not evidence.worth_distilling


def test_a_failed_edit_recovers_only_on_the_same_file(workspace: Path) -> None:
    history = History()
    history.tool("edit", {"file_path": "a.py"}, "stale", failed=True)
    history.tool("edit", {"file_path": "b.py"}, "ok", diff="b.py")
    history.tool("edit", {"file_path": "a.py"}, "ok", diff="a.py")
    evidence = collect(history.view, workspace)
    assert evidence.recoveries == ((1, 3),)


def test_paths_are_compared_however_the_agent_spelled_them(workspace: Path) -> None:
    history = History()
    history.tool("edit", {"file_path": str(workspace / "src" / "app.py")}, "stale", failed=True)
    history.tool("edit", {"file_path": "./src/../src/app.py"}, "ok", diff="src/app.py")
    history.tool("edit", {"file_path": "/etc/hosts"}, "denied", failed=True)
    evidence = collect(history.view, workspace)
    assert [step.path for step in evidence.steps] == ["src/app.py", "src/app.py", "/etc/hosts"]
    assert evidence.recoveries == ((1, 2),)


def test_commands_run_three_times_are_repeated_work(workspace: Path) -> None:
    history = History()
    for _ in range(2):
        history.bash("uv run ruff check")
    for _ in range(3):
        history.bash("uv  run   pytest -q")  # spacing does not make a command different
    evidence = collect(history.view, workspace)
    assert evidence.repeated == (("uv run pytest -q", (3, 4, 5)),)
    assert evidence.signals == {3, 4, 5}


def test_scripts_count_once_written_and_run_and_show_their_current_source(workspace: Path) -> None:
    history = History()
    history.write("tools/seed.py", "print('first draft')\n")
    history.write("tools/never_run.py", "print('unused')\n")
    history.write("notes.md", "not a script\n")
    history.bash("python tools/seed.py", "boom", failed=True)
    history.tool("edit", {"path": "tools/seed.py"}, "ok", diff="tools/seed.py")
    history.bash("python tools/seed.py dev.db", "seeded")
    (workspace / "tools").mkdir()
    (workspace / "tools" / "seed.py").write_text("print('fixed version')\n")
    evidence = collect(history.view, workspace)

    (script,) = evidence.scripts
    assert (script.path, script.written, script.runs) == ("tools/seed.py", 1, (6,))
    assert script.source == "print('fixed version')\n"  # as edited, not as first written
    assert {1, 6} <= evidence.signals


def test_deleted_or_outside_scripts_fall_back_to_what_was_written(
    workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere.sh"
    outside.write_text("echo current contents outside the workspace\n")
    history = History()
    history.write("scratch.py", "print('scratch')\n")
    history.bash("python scratch.py")
    history.write(str(outside), "echo as written\n")
    history.bash(f"bash {outside}")
    evidence = collect(history.view, workspace)
    assert [s.source for s in evidence.scripts] == ["print('scratch')\n", "echo as written\n"]


def test_unfinished_calls_the_generators_own_work_and_old_events_are_left_out(
    workspace: Path,
) -> None:
    history = History()
    history.prompt("first request")
    history.bash("old command")
    history.bus.publish(RunFinished(run_id="r1", status="completed", answer="first answer"))
    watermark = history.view.last_seq
    history.prompt("second request")
    history.bash("new command")
    history.tool("bash", {"command": "sleep 60"}, finished=False)
    history.node("n9", role="skills", parent="n1")
    history.prompt("a digest", node="n9")
    history.bash("generator command", node="n9")

    whole = collect(history.view, workspace)
    assert [s.command for s in whole.steps] == ["old command", "new command"]
    assert whole.prompts == ("first request", "second request") and whole.outcome == "first answer"
    recent = collect(history.view, workspace, since=watermark)
    assert [(s.number, s.command) for s in recent.steps] == [(1, "new command")]
    assert recent.prompts == ("second request",) and recent.outcome == ""


def test_commands_are_recognised_by_input_or_by_summary(workspace: Path) -> None:
    history = History()
    history.tool("shell", {"cmd": "make"}, summary="$ make build")
    history.tool("run", {"command": "make test"}, summary="run tests")
    evidence = collect(history.view, workspace)
    assert [s.command for s in evidence.steps] == ["make build", "make test"]


def test_grounding_requires_real_steps_and_optionally_a_signal(workspace: Path) -> None:
    history = History()
    history.bash("pytest", "boom", failed=True)
    history.bash("uv run pytest", "ok")
    history.tool("read_file", {"path": "README.md"})
    evidence = collect(history.view, workspace)

    assert (
        evidence.grounding_problem([9, 10], need_signal=False) == "it cites no step of this session"
    )
    assert evidence.grounding_problem([3], need_signal=False) == ""
    assert "no failure that was overcome" in evidence.grounding_problem([3], need_signal=True)
    assert evidence.grounding_problem([3, 2, 99], need_signal=True) == ""
    assert evidence.cites_execution([1]) and not evidence.cites_execution([3, 99])


def test_record_stays_within_budget_and_keeps_the_steps_that_matter(workspace: Path) -> None:
    history = History()
    for index in range(400):
        history.bash(f"echo routine command number {index} " + "x" * 60)
    history.bash("pytest -q", "E   AssertionError: " + "y" * 5000, failed=True)
    history.bash("uv run pytest -q", "ok")
    text = collect(history.view, workspace).render()

    assert "Steps (402 tool calls; " in text and "entries omitted to save space" in text
    assert len(text) < 14_000
    assert "[401] FAILED  $ pytest -q" in text and "[402] ok  $ uv run pytest -q" in text
    assert "[1] ok  $ echo routine command number 0" in text  # the rest is filled in order
    assert "[400] ok" not in text
    assert text.count("y") < 700  # failure output is clipped


def test_long_inputs_are_clipped_to_one_line(workspace: Path) -> None:
    history = History()
    history.prompt("please\n" + "very long request " * 200)
    history.bash("printf 'a\\nb'\n" + "&& echo more " * 100)
    evidence = collect(history.view, workspace)
    assert len(evidence.prompts[0]) <= 800 and evidence.prompts[0].endswith("[...]")
    assert "\n" not in evidence.steps[0].command and len(evidence.steps[0].command) <= 300
