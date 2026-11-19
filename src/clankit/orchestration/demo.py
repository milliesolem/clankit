"""Demo: a scripted run that exercises the whole interface without calling any model.

It plays a believable task-tree session - streaming prose, reasoning, tool calls with
diffs, nested subtasks on different models, a failed check that gets fixed, a permission
request, a question to the user, usage numbers - through the session's normal event path.
Useful for trying the interface before any API key is configured, and it gives UI tests
and screenshots a deterministic session that touches every kind of event.

Nothing here touches the file system or the network. `speed` scales the pauses (0 plays
instantly); set it with the method's ``speed`` setting, or ``CLANKIT_DEMO_SPEED``.
"""

from __future__ import annotations

import asyncio
import itertools
import os

from clankit.agent.events import (
    AgentTextDelta,
    AgentThinkingDelta,
    AssistantMessage,
    NodeInfo,
    ToolFinished,
    ToolStarted,
    UserMessage,
)
from clankit.config.schema import Setting
from clankit.llm.types import Usage
from clankit.orchestration.base import Orchestrator
from clankit.tools.base import PermissionAsk

_ids = itertools.count(1)

DIFF = """\
--- a/src/config/writer.py
+++ b/src/config/writer.py
@@ -41,5 +41,7 @@ def write(doc: Document, path: Path) -> None:
     text = render(doc)
     tmp = path.with_suffix(".tmp")
-    path.write_text(text)
+    tmp.write_text(text)
+    os.replace(tmp, path)
     log.debug("wrote %s", path)
"""

ANSWER = """\
The loader is in place and keeps comments when settings are written back.

## What changed

- `src/config/parser.py` parses KDL into a tree that records the source span of every node.
- `src/config/writer.py` applies each setting change as a minimal text edit, then replaces \
the file atomically.
- `src/config/settings.py` routes the settings screen through the new writer.

```python
def write(doc: Document, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(render(doc))
    os.replace(tmp, path)
```

## Verification

| Check | Result |
|---|---|
| `pytest tests/config -q` | 42 passed |
| round trip of the sample files | byte-identical |

One thing is left open: the documentation page was not updated, because you chose to keep \
this change small."""


class DemoOrchestrator(Orchestrator):
    name = "demo"
    title = "Demo"
    summary = "Plays a scripted session so you can try the interface without a model."
    guidance = "Calls no model and changes no files. Any prompt starts the same scripted run."
    needs_model = False
    settings = (
        Setting(
            "speed",
            "int",
            100,
            "Playback speed in percent; 0 plays instantly.",
            minimum=0,
            maximum=1000,
        ),
    )

    def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
        super().__init__(session)
        self._root = session.new_node(
            "Conversation", role="root", model="anthropic/claude-opus-5-5"
        )

    # -- pacing ---------------------------------------------------------------------------------

    def _speed(self) -> float:
        override = os.environ.get("CLANKIT_DEMO_SPEED")
        try:
            percent = float(override) if override is not None else float(self.setting("speed"))
        except ValueError:
            percent = 100.0
        return max(percent, 0.0) / 100.0

    async def _pause(self, seconds: float) -> None:
        speed = self._speed()
        await asyncio.sleep(seconds / speed if speed > 0 else 0)

    # -- transcript helpers ---------------------------------------------------------------------

    async def _say(self, node: NodeInfo, text: str, *, thinking: str = "") -> None:
        emit = self.session.emit
        message_id = f"demo{next(_ids)}"
        for piece in _chunks(thinking, 24):
            emit(AgentThinkingDelta(node_id=node.id, message_id=message_id, text=piece))
            await self._pause(0.03)
        for piece in _chunks(text, 12):
            emit(AgentTextDelta(node_id=node.id, message_id=message_id, text=piece))
            await self._pause(0.02)
        emit(AssistantMessage(node_id=node.id, message_id=message_id, text=text, thinking=thinking))

    async def _tool(
        self,
        node: NodeInfo,
        tool: str,
        summary: str,
        output: str,
        *,
        seconds: float = 0.4,
        is_error: bool = False,
        display: dict | None = None,
        tool_input: dict | None = None,
    ) -> None:
        call_id = f"demo-call{next(_ids)}"
        self.session.emit(
            ToolStarted(
                node_id=node.id, call_id=call_id, tool=tool, summary=summary, input=tool_input or {}
            )
        )
        await self._pause(seconds)
        self.session.emit(
            ToolFinished(
                node_id=node.id,
                call_id=call_id,
                output=output,
                is_error=is_error,
                duration=seconds,
                display=display,
            )
        )

    def _spend(self, node: NodeInfo, prompt: int, output: int) -> None:
        self.session.record_usage(
            node.id,
            Usage(
                input_tokens=prompt // 5,
                output_tokens=output,
                cache_read_tokens=prompt - prompt // 5,
            ),
            context_tokens=prompt + output,
        )

    # -- the script -----------------------------------------------------------------------------

    async def run(self, prompt: str) -> str:
        s = self.session
        root = self._root
        s.update_node(root, status="running")
        await self._say(
            root,
            "I'll look at how settings are read and written today before deciding how to split this.",
            thinking="The request has three parts: parsing, writing without losing comments, and "
            "wiring the settings screen. The writer is the risky part.",
        )
        await self._tool(root, "read", "read src/config/loader.py", "212 lines", seconds=0.2)
        await self._tool(
            root, "grep", "grep 'write_text' in src/", "4 matches in 3 files", seconds=0.2
        )
        self._spend(root, 18_400, 620)
        await self._say(
            root,
            "The loader splits cleanly into a parser, a comment-preserving writer and the settings "
            "wiring. The writer gets its own subtree and a round-trip test.",
        )

        s.update_node(root, status="waiting")
        parser = s.new_node(
            "Parser with source spans",
            parent=root,
            model="deepseek/deepseek-v4-pro",
            brief="Parse KDL into a tree that records the source span of every node.",
        )
        writer = s.new_node(
            "Writer that preserves comments",
            parent=root,
            role="branch",
            model="anthropic/claude-sonnet-5-5",
            brief="Apply setting changes as minimal text edits; replace the file atomically.",
        )
        wiring = s.new_node(
            "Settings wiring",
            parent=root,
            model="anthropic/claude-sonnet-5-5",
            brief="Route the settings screen through the new writer.",
        )
        docs = s.new_node(
            "Documentation",
            parent=root,
            model="anthropic/claude-haiku-5-5",
            brief="Describe the config format.",
        )

        await asyncio.gather(self._parser(parser), self._writer(writer))
        await self._leaf(wiring, "edit src/config/settings.py", 9_800, 540)

        s.update_node(root, status="integrating")
        await self._tool(
            root, "bash", "$ pytest tests/config -q", "42 passed in 1.31s", seconds=0.8
        )
        self._spend(root, 31_000, 480)
        answer = await s.ask_user(
            root.id,
            "The documentation page still describes the old format. Update it as part of this change?",
            ["Keep this change small", "Update the documentation too"],
        )
        if answer.lower().startswith("update"):
            await self._leaf(docs, "edit docs/configuration.md", 4_100, 380)
        else:
            s.update_node(
                docs, status="cancelled", summary="Left for a follow-up, as the user chose."
            )
        await self._say(root, ANSWER)
        self._spend(root, 33_500, 910)
        s.update_node(root, status="done")
        return ANSWER

    async def _leaf(self, node: NodeInfo, edit: str, prompt: int, output: int) -> None:
        s = self.session
        s.update_node(node, status="running")
        s.emit(UserMessage(node_id=node.id, text=node.brief))
        await self._tool(node, "read", edit.replace("edit", "read"), "148 lines", seconds=0.2)
        await self._tool(
            node,
            "edit",
            edit,
            "Edited.",
            seconds=0.3,
            display={"type": "diff", "path": edit.split(" ", 1)[1], "diff": DIFF},
        )
        await self._say(node, "Done; the change is covered by the existing tests.")
        self._spend(node, prompt, output)
        s.update_node(node, status="done", summary="Implemented and verified.")

    async def _parser(self, node: NodeInfo) -> None:
        s = self.session
        s.update_node(node, status="running")
        s.emit(UserMessage(node_id=node.id, text=node.brief))
        await self._tool(
            node, "write", "write src/config/parser.py", "Created (184 lines).", seconds=0.5
        )
        await self._tool(
            node,
            "bash",
            "$ pytest tests/config/test_parser.py -q",
            "17 passed in 0.42s",
            seconds=0.7,
        )
        await self._say(
            node, "The parser is done: 17 tests pass, including the official sample files."
        )
        self._spend(node, 9_200, 1_640)
        s.update_node(node, status="done", summary="Parser with spans; 17 tests pass.")

    async def _writer(self, node: NodeInfo) -> None:
        s = self.session
        s.update_node(node, status="running")
        s.emit(UserMessage(node_id=node.id, text=node.brief))
        await self._say(
            node, "Two independent pieces plus tests; the tests need both, so they go last."
        )
        self._spend(node, 6_300, 410)
        s.update_node(node, status="waiting")
        formatter = s.new_node(
            "Round-trip formatter",
            parent=node,
            model="anthropic/claude-haiku-5-5",
            brief="Render a changed value in place without touching its neighbours.",
        )
        replace = s.new_node(
            "Atomic file replace",
            parent=node,
            model="deepseek/deepseek-v4-pro",
            brief="Write through a temporary file and os.replace.",
        )
        tests = s.new_node(
            "Round-trip tests",
            parent=node,
            model="deepseek/deepseek-v4-pro",
            brief="Property tests: edit, re-read, compare; comments must survive.",
        )

        await asyncio.gather(
            self._leaf(formatter, "edit src/config/render.py", 3_100, 720), self._replace(replace)
        )
        await self._tests(tests)

        s.update_node(node, status="integrating")
        await self._tool(
            node, "bash", "$ pytest tests/config -q", "42 passed in 1.29s", seconds=0.6
        )
        await self._say(
            node, "Both pieces fit together; the round trip is byte-identical on the samples."
        )
        self._spend(node, 12_400, 380)
        s.update_node(node, status="done", summary="Writer complete; 42 tests pass.")

    async def _replace(self, node: NodeInfo) -> None:
        s = self.session
        s.update_node(node, status="running")
        s.emit(UserMessage(node_id=node.id, text=node.brief))
        await self._tool(
            node,
            "edit",
            "edit src/config/writer.py",
            "Edited.",
            seconds=0.3,
            display={"type": "diff", "path": "src/config/writer.py", "diff": DIFF},
        )
        await self._tool(
            node,
            "bash",
            "$ pytest tests/config/test_writer.py -q",
            "1 failed, 5 passed\nFAILED test_replace_keeps_mode - assert 0o644 == 0o600",
            seconds=0.6,
            is_error=True,
        )
        await self._say(
            node, "The replacement loses the file mode. Copying it across before the swap."
        )
        await self._tool(node, "edit", "edit src/config/writer.py", "Edited.", seconds=0.3)
        await self._tool(
            node,
            "bash",
            "$ pytest tests/config/test_writer.py -q",
            "6 passed in 0.18s",
            seconds=0.5,
        )
        self._spend(node, 2_200, 610)
        s.update_node(
            node, status="done", summary="Atomic replace keeps the file mode; 6 tests pass."
        )

    async def _tests(self, node: NodeInfo) -> None:
        s = self.session
        s.update_node(node, status="running")
        s.emit(UserMessage(node_id=node.id, text=node.brief))
        await self._tool(
            node,
            "write",
            "write tests/config/test_roundtrip.py",
            "Created (96 lines).",
            seconds=0.4,
        )
        command = "pytest tests/config/test_roundtrip.py -q --hypothesis-seed=0"
        call_id = f"demo-call{next(_ids)}"
        s.emit(
            ToolStarted(
                node_id=node.id,
                call_id=call_id,
                tool="bash",
                summary=f"$ {command}",
                input={"command": command},
            )
        )
        decision = await s.permissions.check(
            PermissionAsk(
                "shell", f"Run: {command}", command, detail=command, session_rule="pytest *"
            ),
            node_id=node.id,
            tool="bash",
        )
        if decision.allowed:
            await self._pause(0.8)
            s.emit(
                ToolFinished(
                    node_id=node.id, call_id=call_id, output="19 passed in 2.07s", duration=2.07
                )
            )
            summary = "19 round-trip tests pass."
        else:
            s.emit(
                ToolFinished(
                    node_id=node.id,
                    call_id=call_id,
                    is_error=True,
                    output=f"Permission denied: {decision.reason}",
                )
            )
            summary = "Tests written but not run: permission was denied."
        await self._say(node, summary)
        self._spend(node, 400, 520)
        s.update_node(node, status="done", summary=summary)


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
