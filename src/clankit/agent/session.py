"""A session: one user, one workspace, one conversation, and the task tree it grows.

A `Session` is the only object a user interface holds. It

* runs prompts through the selected orchestration method (`start` / `run` / `cancel`),
* publishes everything that happens on `bus` and keeps `view` (the folded state) current,
* carries answers back to the engine (`resolve_permission`, `answer_question`, `steer`),
* and gives orchestrators their building blocks (`new_node`, `update_node`, `agent`).

Everything here must be called from the event loop the session was created in. The web UI,
whose request handlers run in other threads, goes through `clankit.runtime.RuntimeThread`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clankit.agent.events import (
    TERMINAL_STATUSES,
    Event,
    EventBus,
    NodeAdded,
    NodeChanged,
    NodeInfo,
    NodeStatus,
    Notice,
    PermissionChoice,
    QuestionAnswered,
    QuestionAsked,
    RunFinished,
    RunStarted,
    UsageChanged,
)
from clankit.agent.loop import Agent
from clankit.agent.permissions import PermissionManager
from clankit.agent.state import SessionView
from clankit.agent.store import SessionLog
from clankit.config.schema import Config
from clankit.llm.provider import Provider, ProviderError
from clankit.llm.types import Effort, ModelInfo, ModelRef, Usage
from clankit.tools.base import FileLedger, Tool, ToolContext

if TYPE_CHECKING:
    from clankit.orchestration.base import Orchestrator
    from clankit.runtime import Runtime

log = logging.getLogger(__name__)

NO_USER = (
    "(Nobody is available to answer: this is a non-interactive run. Use your best judgment "
    "and state the assumption you made in your final answer.)"
)


@dataclass(slots=True)
class Turn:
    """One completed exchange: what the user asked and what they were told."""

    prompt: str
    answer: str
    method: str
    status: str


class Session:
    def __init__(
        self,
        runtime: Runtime,
        workspace: Path,
        *,
        session_id: str | None = None,
        interactive: bool = True,
        log_to: SessionLog | None = None,
    ) -> None:
        self.runtime = runtime
        self.workspace = workspace.resolve()
        self.id = session_id or uuid.uuid4().hex[:12]
        self.interactive = interactive

        self.bus = EventBus()
        self.view = SessionView()
        self.bus.subscribe(self.view.apply)
        self._log = log_to
        if log_to is not None:
            self.bus.subscribe(log_to.append)

        self.files = FileLedger()
        self.permissions = PermissionManager(
            policy=lambda: self.config.permissions, emit=self.emit, interactive=interactive
        )
        self.usage = Usage()
        self.turns: list[Turn] = []

        #: Orchestration method for the next run. None follows the configuration.
        self.method_override: str | None = None
        #: Model for the next run's root agent. None follows the configuration.
        self.model_override: ModelRef | None = None

        self._nodes: dict[str, NodeInfo] = {}
        self._node_usage: dict[str, Usage] = {}
        self._node_ids = itertools.count(1)
        self._orchestrators: dict[str, Orchestrator] = {}
        self._active: Orchestrator | None = None
        self._task: asyncio.Task[Any] | None = None
        self._running = False
        self._cancel_requested = False
        self._questions: dict[str, asyncio.Future[str]] = {}
        self._question_ids = itertools.count(1)
        self._slots: dict[str, tuple[int, asyncio.Semaphore]] = {}
        self._run_output = 0
        self._budget_extensions = 0
        self._budget_lock = asyncio.Lock()

    # -- configuration --------------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self.runtime.config

    @property
    def method(self) -> str:
        return self.method_override or self.config.orchestration.method

    @property
    def model(self) -> ModelRef:
        """The model the root agent uses. Raises `ProviderError` if no provider is usable."""
        if self.model_override is not None:
            return self.model_override
        ref = self.runtime.providers.default_model()
        if ref is None:
            raise ProviderError(
                "No model is available yet. Add an API key for a provider in the settings "
                "(or in ~/.config/clankit/config.kdl).",
                kind="auth",
            )
        return ref

    def model_for(self, role: str | None) -> ModelRef:
        """The model configured for a role ("subagent", "planner", "reviewer", "judge",
        "utility"), falling back to the session's model."""
        configured = self.config.models.roles.get(role or "", "")
        if configured:
            try:
                ref = ModelRef.parse(configured)
            except ValueError:
                ref = None
            if ref is not None and ref.provider in self.runtime.providers.provider_ids():
                return ref
        return self.model

    # -- AgentHost ------------------------------------------------------------------------------

    def provider_for(self, model: ModelRef) -> tuple[Provider, str]:
        return self.runtime.providers.resolve(model)

    def model_info(self, model: ModelRef) -> ModelInfo | None:
        return self.runtime.providers.model_info(model)

    def emit(self, event: Event) -> Event:
        return self.bus.publish(event)

    def model_slot(self, model: ModelRef) -> asyncio.Semaphore:
        """Limits model calls in flight per provider (free tiers tolerate only one or two)."""
        slot = self._slots.get(model.provider)
        size = self.runtime.providers.concurrency(model.provider)
        if slot is None or slot[0] != size:
            slot = self._slots[model.provider] = (size, asyncio.Semaphore(size))
        return slot[1]

    async def before_turn(self, node_id: str) -> None:
        """Pause the run when it has spent its output-token budget and ask whether to go on."""
        budget = self.config.agent.run_budget
        if budget <= 0 or self._run_output < self._budget_limit(budget):
            return
        async with self._budget_lock:
            limit = self._budget_limit(budget)
            if self._run_output < limit:
                return  # another agent already got the budget extended
            if self.interactive:
                answer = await self.ask_user(
                    node_id,
                    f"This run has generated {self._run_output:,} output tokens, past its budget "
                    f"of {limit:,}. Keep going?",
                    ["Continue", "Stop"],
                )
                if answer.strip().lower().startswith(("c", "y")):
                    self._budget_extensions += 1
                    return
            raise ProviderError(
                f"stopped: the run reached its budget of {limit:,} output tokens "
                "(agent.run-budget in the configuration)",
                kind="bad_request",
            )

    def _budget_limit(self, budget: int) -> int:
        return budget * (1 + self._budget_extensions)

    def record_usage(self, node_id: str, usage: Usage, context_tokens: int) -> None:
        self._run_output += usage.output_tokens
        self.usage.add(usage)
        node_usage = self._node_usage.setdefault(node_id, Usage())
        node_usage.add(usage)
        self.emit(
            UsageChanged(
                node_id=node_id,
                usage=node_usage.to_dict(),
                total=self.usage.to_dict(),
                context_tokens=context_tokens,
            )
        )

    # -- task tree ------------------------------------------------------------------------------

    def new_node(
        self,
        title: str,
        *,
        parent: NodeInfo | None = None,
        role: str = "leaf",
        model: ModelRef | str | None = None,
        brief: str = "",
        status: NodeStatus = "pending",
    ) -> NodeInfo:
        """Add a node to the task tree and announce it."""
        info = NodeInfo(
            id=f"n{next(self._node_ids)}",
            parent_id=parent.id if parent else None,
            depth=parent.depth + 1 if parent else 0,
            title=title,
            role=role,
            model=str(model) if model else "",
            status=status,
            brief=brief,
        )
        self._nodes[info.id] = info
        self.emit(NodeAdded(node=dataclasses.replace(info)))
        return info

    def update_node(
        self,
        node: NodeInfo | str,
        *,
        status: NodeStatus | None = None,
        title: str | None = None,
        role: str | None = None,
        model: ModelRef | str | None = None,
        summary: str | None = None,
    ) -> None:
        """Change a node and announce the change. Unchanged values publish nothing."""
        info = self._nodes[node if isinstance(node, str) else node.id]
        changes: dict[str, Any] = {}
        for key, value in (
            ("status", status),
            ("title", title),
            ("role", role),
            ("summary", summary),
        ):
            if value is not None and getattr(info, key) != value:
                setattr(info, key, value)
                changes[key] = value
        if model is not None and info.model != str(model):
            info.model = str(model)
            changes["model"] = info.model
        if changes:
            self.emit(NodeChanged(node_id=info.id, **changes))

    def node(self, node_id: str) -> NodeInfo:
        return self._nodes[node_id]

    def children_of(self, node_id: str) -> list[NodeInfo]:
        return [n for n in self._nodes.values() if n.parent_id == node_id]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # -- agents ---------------------------------------------------------------------------------

    def tool_context(
        self,
        node: NodeInfo,
        *,
        write_scope: Sequence[str] | None = None,
        parent: ToolContext | None = None,
    ) -> ToolContext:
        """A tool context for `node`'s agent.

        With `parent` given, the new context inherits that agent's write restrictions and adds
        `write_scope` on top; a child can never write more than its parent.
        """
        # None means unrestricted; an empty scope means the agent may not write at all.
        scope = tuple(write_scope) if write_scope is not None else None
        if parent is not None:
            return parent.for_child(node.id, scope)
        return ToolContext(
            workspace=self.workspace,
            node_id=node.id,
            files=self.files,
            permissions=self.permissions,
            emit=self.emit,
            write_scope=scope,
            config=self.config,
            skills=self.runtime.skills,
        )

    def agent(
        self,
        node: NodeInfo,
        *,
        system: str,
        tools: Sequence[Tool],
        ctx: ToolContext | None = None,
        model: ModelRef | None = None,
        effort: Effort | None = None,
        max_turns: int | None = None,
    ) -> Agent:
        """Create the agent that works on `node`. The node's `model` is kept in step."""
        ref = model or (ModelRef.parse(node.model) if node.model else self.model)
        self.update_node(node, model=ref)
        return Agent(
            host=self,
            node_id=node.id,
            model=ref,
            system=system,
            tools=tools,
            ctx=ctx or self.tool_context(node),
            effort=effort,
            max_turns=max_turns,
        )

    # -- talking to the user --------------------------------------------------------------------

    async def ask_user(self, node_id: str, question: str, options: Sequence[str] = ()) -> str:
        """Put a question to the user and wait for the answer."""
        if not self.interactive:
            return NO_USER
        request_id = f"q{next(self._question_ids)}"
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._questions[request_id] = future
        self.emit(
            QuestionAsked(
                request_id=request_id, node_id=node_id, question=question, options=list(options)
            )
        )
        try:
            return await future
        finally:
            if self._questions.pop(request_id, None) is not None and not future.done():
                self.emit(QuestionAnswered(request_id=request_id, answer=""))

    def answer_question(self, request_id: str, answer: str) -> bool:
        future = self._questions.pop(request_id, None)
        if future is None or future.done():
            return False
        future.set_result(answer)
        self.emit(QuestionAnswered(request_id=request_id, answer=answer))
        return True

    def resolve_permission(self, request_id: str, choice: PermissionChoice) -> bool:
        return self.permissions.resolve(request_id, choice)

    def notice(self, level: str, text: str, node_id: str = "") -> None:
        self.emit(Notice(level=level, text=text, node_id=node_id))  # type: ignore[arg-type]

    # -- runs -----------------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._running

    def start(self, prompt: str) -> asyncio.Task[str]:
        """Begin a run in the background. Progress arrives as events; raises if one is active."""
        if self._running:
            raise RuntimeError("a run is already in progress")
        self._running = True
        task = asyncio.get_running_loop().create_task(self._run(prompt))
        self._task = task
        return task

    async def run(self, prompt: str) -> str:
        """Run a prompt to completion and return the answer ("" if cancelled or failed)."""
        if self._running:
            raise RuntimeError("a run is already in progress")
        self._running = True
        self._task = asyncio.current_task()
        return await self._run(prompt)

    def cancel(self) -> bool:
        """Interrupt the active run. Returns False if nothing is running."""
        if not self._running or self._task is None or self._task.done():
            return False
        self._cancel_requested = True
        self._task.cancel()
        return True

    def steer(self, text: str) -> bool:
        """Send a message to the working root agent without interrupting it."""
        if not self._running or self._active is None:
            return False
        self._active.steer(text)
        return True

    def recap(self, limit: int = 6, chars: int = 1500) -> str:
        """Earlier turns of this session as text, for an agent that starts without them."""
        if not self.turns:
            return ""
        lines = ["Earlier in this session:"]
        for turn in self.turns[-limit:]:
            answer = turn.answer if len(turn.answer) <= chars else turn.answer[:chars] + " [...]"
            lines.append(
                f"\nThe user asked:\n{turn.prompt}\n\nThe answer given ({turn.status}):\n{answer or '(none)'}"
            )
        return "\n".join(lines)

    async def _run(self, prompt: str) -> str:
        from clankit.orchestration import get_orchestrator

        run_id = uuid.uuid4().hex[:8]
        method = self.method
        status = "failed"
        answer = ""
        error = ""
        self._cancel_requested = False
        self._run_output = 0
        self._budget_extensions = 0
        self.files.begin_run()
        try:
            orchestrator_cls = get_orchestrator(method)
            model = str(self.model) if orchestrator_cls.needs_model else ""
            self.emit(RunStarted(run_id=run_id, prompt=prompt, method=method, model=model))
            try:
                orchestrator = self._orchestrators.get(method)
                if orchestrator is None:
                    orchestrator = self._orchestrators[method] = orchestrator_cls(self)
                self._active = orchestrator
                answer = await orchestrator.run(prompt)
                status = "completed"
            except asyncio.CancelledError:
                status = "cancelled"
                if not self._cancel_requested:
                    raise
            except ProviderError as exc:
                error = str(exc)
            except Exception as exc:
                log.exception("run %s failed", run_id)
                error = f"{type(exc).__name__}: {exc}"
            finally:
                self._close_open_nodes("cancelled" if status == "cancelled" else "failed")
                if error:
                    self.emit(Notice(level="error", text=error))
                self.emit(RunFinished(run_id=run_id, status=status, answer=answer, error=error))  # type: ignore[arg-type]
                self.turns.append(Turn(prompt=prompt, answer=answer, method=method, status=status))
        except Exception as exc:
            # Failed before the run could start (no usable model, unknown or broken method).
            if not isinstance(exc, (ProviderError, KeyError)):
                log.exception("run %s could not start", run_id)
            error = str(exc).strip("'\"") or type(exc).__name__
            self.emit(RunStarted(run_id=run_id, prompt=prompt, method=method, model=""))
            self.emit(Notice(level="error", text=error))
            self.emit(RunFinished(run_id=run_id, status="failed", error=error))
        finally:
            self._active = None
            self._running = False
            self._task = None

        if status == "completed" and self._active_needs_model(method):
            for hook in list(self.runtime.after_run):
                try:
                    await hook(self, self.turns[-1])
                except Exception:
                    log.exception("after-run hook failed")
        return answer

    async def compact(self) -> bool:
        """Summarise the root conversation of the current method into a fresh one.

        Returns False when there is nothing to compact. Raises RuntimeError during a run.
        """
        if self._running:
            raise RuntimeError("cannot compact while a run is in progress")
        orchestrator = self._orchestrators.get(self.method)
        agent = orchestrator.root_agent if orchestrator is not None else None
        return await agent.compact() if agent is not None else False

    def context_usage(self) -> tuple[int, int] | None:
        """(tokens in the root conversation's context, the model's context window), if known."""
        orchestrator = self._orchestrators.get(self.method)
        agent = orchestrator.root_agent if orchestrator is not None else None
        if agent is None:
            return None
        info = self.model_info(agent.model)
        window = (
            info.context_window
            if info and info.context_window
            else self.config.agent.context_window
        )
        return agent.context_tokens, window

    def cost(self) -> float | None:
        """Estimated cost of the session so far in US dollars, or None when the price of any
        model that was used is unknown. An estimate: providers bill by their own rules."""
        total = 0.0
        for node_id, usage in self._node_usage.items():
            info = None
            model = self._nodes[node_id].model if node_id in self._nodes else ""
            if model:
                try:
                    info = self.model_info(ModelRef.parse(model))
                except ValueError:
                    info = None
            cost = info.cost(usage) if info is not None else None
            if cost is None:
                return None
            total += cost
        return total

    def undo(self) -> list[str]:
        """Revert the file changes of the most recent run that changed files.

        Only changes made through the file tools are reverted, not the effects of shell
        commands. Returns workspace-relative paths; raises RuntimeError while a run is active.
        """
        if self._running:
            raise RuntimeError("cannot undo while a run is in progress")
        restored = []
        for path in self.files.undo():
            try:
                restored.append(path.relative_to(self.workspace).as_posix())
            except ValueError:
                restored.append(str(path))
        if restored:
            self.notice("info", f"Reverted {len(restored)} file(s): {', '.join(sorted(restored))}")
        return sorted(restored)

    def _active_needs_model(self, method: str) -> bool:
        """After-run hooks distil model work; a scripted, model-free method has none."""
        orchestrator = self._orchestrators.get(method)
        return orchestrator is None or orchestrator.needs_model

    def _close_open_nodes(self, status: NodeStatus) -> None:
        """After a run ends, no node may be left looking active."""
        for info in self._nodes.values():
            if info.status not in TERMINAL_STATUSES:
                self.update_node(info, status=status)

    # -- persistence ----------------------------------------------------------------------------

    def restore(self, events: Sequence[Event]) -> None:
        """Rebuild the visible state of a stored session (transcripts, tree, past turns).

        Agents do not come back to life: orchestrators start fresh and get `recap()`.
        """
        prompt = ""
        method = ""
        highest = 0
        for event in events:
            self.view.apply(event)
            if isinstance(event, RunStarted):
                prompt, method = event.prompt, event.method
            elif isinstance(event, RunFinished):
                self.turns.append(
                    Turn(prompt=prompt, answer=event.answer, method=method, status=event.status)
                )
            elif isinstance(event, NodeAdded):
                self._nodes[event.node.id] = dataclasses.replace(event.node)
                if event.node.id[1:].isdigit():
                    highest = max(highest, int(event.node.id[1:]))
            elif isinstance(event, UsageChanged):
                self.usage = Usage.from_dict(event.total)
        for node_id, view in self.view.nodes.items():
            self._nodes[node_id] = dataclasses.replace(view.info)
            if view.info.status not in TERMINAL_STATUSES:
                view.info.status = "cancelled"
                self._nodes[node_id].status = "cancelled"
        self.view.running = False
        self.view.permissions.clear()
        self.view.questions.clear()
        self._node_ids = itertools.count(highest + 1)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
