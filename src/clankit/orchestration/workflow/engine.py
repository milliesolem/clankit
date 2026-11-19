"""Running workflow scripts: the primitives scripts call, and the bookkeeping behind them.

`WorkflowRun.execute(script, args)` runs a compiled `Script` in-process, in the session's
event loop, with a namespace holding exactly the workflow API:

* ``agent(prompt, *, schema, model, label, phase, tools, write_scope, effort)`` starts one
  agent and gives back its final text, or a dict matching `schema`, or None if it failed;
* ``parallel(items)`` is a barrier over awaitables and zero-argument callables whose failed
  slots become None and which never raises for a failed slot;
* ``pipeline(items, *stages)`` moves each item through the stages on its own, with no
  barrier between stages; a stage that fails or returns None ends that item;
* ``phase(title)`` and ``log(message)`` drive the progress display; ``args``, ``budget``,
  ``workflow(name, args)`` and the pattern functions complete the set.

Everything the script starts shows up on the session's task tree: a "workflow" node under
the agent that ran it, "phase" nodes under that, and a "worker" node per agent under its
phase. Limits are enforced here, not left to the script: concurrent agents (on top of the
per-provider slots the session already has), agents per workflow, items per call, loop
iterations, and the token budget, which stops the workflow cleanly with what finished.

Each awaitable call is identified by its structural position in the script
(``root/pipeline:0/item:3/stage:1/agent:0``), assigned when the call is made rather than
when it happens to start, so the identity is the same on every run of the same script.
That is what the resume journal keys on.

Async primitives return a `Deferred`: nothing starts until it is awaited, an un-awaited
call is reported in the account instead of vanishing, and argument mistakes are raised at
the call, on the script's own line.
"""

from __future__ import annotations

import asyncio
import builtins
import contextvars
import copy
import functools
import inspect
import itertools
import json
import types
from collections.abc import Awaitable, Callable, Generator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jsonschema

from clankit.agent.events import NodeInfo, NodeStatus, ToolFinished, ToolStarted
from clankit.config.schema import MODEL_ROLES
from clankit.llm.types import EFFORT_LEVELS, ModelRef
from clankit.orchestration.common import CheckResult, resolve_model, run_check
from clankit.orchestration.workflow.account import (
    AgentRecord,
    Outcome,
    OutcomeStatus,
    render_value,
)
from clankit.orchestration.workflow.errors import AgentLimitReached, WorkflowError, WorkflowStop
from clankit.orchestration.workflow.journal import JournalRun
from clankit.orchestration.workflow.patterns import PATTERNS
from clankit.orchestration.workflow.saved import find_saved, saved_names
from clankit.orchestration.workflow.script import TICK, Script
from clankit.orchestration.workflow.worker import AgentSpec, TokenMeter, WorkerHost, run_worker

if TYPE_CHECKING:
    from clankit.agent.session import Session

#: Items one parallel() or pipeline() call may take. More is an error, not a silent cap.
MAX_ITEMS = 4096
#: Loop iterations one workflow may run; a runaway loop would freeze the shared event loop.
MAX_STEPS = 5_000_000
#: Largest range() a script may create.
MAX_RANGE = 1_000_000
#: Characters of an agent's prompt kept as its node's brief.
BRIEF_CHARS = 4000

TOOL_SETS = ("all", "read-only")
ROLE_NAMES = tuple(role for role, _ in MODEL_ROLES)

#: Built-ins a script may use. Everything else (open, getattr, __import__, ...) is absent.
_BUILTIN_NAMES: tuple[str, ...] = (
    "abs", "all", "any", "bool", "callable", "chr", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "isinstance", "iter", "len", "list", "map", "max",
    "min", "next", "ord", "pow", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "ArithmeticError", "AssertionError", "AttributeError", "Exception",
    "IndexError", "KeyError", "LookupError", "NameError", "NotImplementedError",
    "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)  # fmt: skip
_BUILTINS: dict[str, Any] = {name: getattr(builtins, name) for name in _BUILTIN_NAMES}
_check_ids = itertools.count(1)

_JSON = types.SimpleNamespace(
    dumps=functools.partial(json.dumps, ensure_ascii=False, default=str), loads=json.loads
)


@dataclass(frozen=True, slots=True)
class Limits:
    """What one workflow (including any workflow it runs) may use."""

    max_concurrent: int = 6
    max_agents: int = 60
    #: Tokens, counted as `TokenMeter` documents.
    budget: int = 1_500_000
    #: Default skeptics per claim for adversarial_verify.
    verify_votes: int = 3
    #: Announce a workflow expected to start more agents than this; 0 never announces.
    announce_above: int = 25


# --------------------------------------------------------------------------------------------
# Call identity and deferred calls
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Scope:
    """A position in the script's call structure, with counters for the calls made in it."""

    path: str
    counters: dict[str, int] = field(default_factory=dict)

    def next(self, kind: str) -> str:
        number = self.counters.get(kind, 0)
        self.counters[kind] = number + 1
        return f"{self.path}/{kind}:{number}"


_SCOPE: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "clankit_workflow_scope", default=None
)


class Deferred:
    """An awaitable workflow call that starts only when it is awaited.

    Until then it sits in its run's `unawaited` table, so a call the script forgot to await
    is reported (with its line) instead of disappearing.
    """

    __slots__ = ("_factory", "_run", "line", "what")

    def __init__(
        self, run: WorkflowRun, what: str, line: int, factory: Callable[[], Awaitable[Any]]
    ) -> None:
        self._run = run
        self._factory: Callable[[], Awaitable[Any]] | None = factory
        self.what = what
        self.line = line
        run.unawaited[self] = None

    def __await__(self) -> Generator[Any, None, Any]:
        if self._factory is None:
            raise WorkflowError(
                f"the {self.what} call from line {self.line} was awaited twice; keep its "
                "result in a variable instead"
            )
        factory, self._factory = self._factory, None
        self._run.unawaited.pop(self, None)
        return factory().__await__()

    def __repr__(self) -> str:
        state = ", never awaited" if self._factory is not None else ""
        return f"<{self.what} from line {self.line}{state}>"


def _raise_if_cancelling() -> None:
    """Refuse to start new work in a task that is being cancelled (for example from a
    script's ``finally`` block while the run is being stopped)."""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


class WorkflowRun:
    """One workflow execution and everything a workflow nested in it shares: limits, the
    token meter, the concurrency slots, the agent count, the journal and the account."""

    def __init__(
        self,
        session: Session,
        *,
        parent: NodeInfo,
        limits: Limits,
        journal: JournalRun | None = None,
    ) -> None:
        self.session = session
        self.parent = parent
        self.limits = limits
        self.journal = journal
        self.meter = TokenMeter(limits.budget)
        self.host = WorkerHost(session, self.meter)
        self.slots = asyncio.Semaphore(limits.max_concurrent)
        self.records: list[AgentRecord] = []
        self.failures: list[str] = []
        self.empties: list[str] = []
        self.warnings: list[str] = []
        self.logs: list[str] = []
        self.unawaited: dict[Deferred, None] = {}
        self.started = 0
        self._steps = 0
        self._announced = False
        self._name = ""

    async def execute(
        self, script: Script, args: Any = None, *, estimated_agents: int | None = None
    ) -> Outcome:
        """Run `script` to the end and account for it.

        Script errors and limits end in an `Outcome` (status "failed" or "stopped"); only
        cancellation propagates, after every node this run created has been closed.
        """
        session = self.session
        self._name = script.meta.name
        estimate = max(estimated_agents or 0, script.estimated_agents)
        if 0 < self.limits.announce_above < estimate:
            self._announce(f"about {estimate} agents")
        node = session.new_node(
            script.meta.name,
            parent=self.parent,
            role="workflow",
            status="running",
            brief=script.meta.description,
        )
        frame = _Frame(self, script, node, nested=False, path="root")
        status: OutcomeStatus = "completed"
        error = ""
        result: Any = None
        try:
            result = await frame.execute(args)
        except WorkflowStop as stop:
            status, error = "stopped", str(stop)
            result = frame.namespace.get("result")
        except asyncio.CancelledError:
            frame.close("cancelled")
            session.update_node(node, status="cancelled", summary="Cancelled.")
            raise
        except Exception as exc:
            status, error = "failed", describe_error(exc, script)
        finally:
            if self.journal is not None:
                self.journal.commit()
        if status == "completed":
            self._report_unawaited()
        frame.close("done" if status == "completed" else "failed")
        outcome = Outcome(
            name=script.meta.name,
            status=status,
            result=result,
            error=error,
            node_id=node.id,
            tokens=self.meter.spent,
            budget=self.meter.total,
            records=self.records,
            phases=frame.phase_titles(),
            failures=self.failures,
            empties=self.empties,
            warnings=self._warnings(),
            logs=self.logs,
        )
        session.update_node(
            node,
            status="done" if status == "completed" else "failed",
            summary=outcome.tally() + ("" if status == "completed" else f"; {status}"),
        )
        return outcome

    # -- shared bookkeeping ---------------------------------------------------------------

    def reserve_agent(self) -> None:
        """Count one more started agent; past the cap the whole workflow stops."""
        if self.started >= self.limits.max_agents:
            raise AgentLimitReached(f"it tried to start more than {self.limits.max_agents} agents")
        self.started += 1
        if 0 < self.limits.announce_above < self.started and not self._announced:
            self._announce(f"{self.started} agents started so far")

    def tick(self) -> bool:
        """Called on every loop iteration of a script (the compiler inserts the call)."""
        self._steps += 1
        if self._steps > MAX_STEPS:
            raise WorkflowError(
                f"the script ran more than {MAX_STEPS:,} loop iterations; workflow scripts "
                "should orchestrate agents, not compute"
            )
        return True

    def _announce(self, size: str) -> None:
        self._announced = True
        self.session.notice(
            "info",
            f'Large workflow "{self._name}": {size}, at most {self.limits.max_concurrent} at a '
            f"time, within a budget of {self.limits.budget:,} tokens.",
            self.parent.id,
        )

    def _report_unawaited(self) -> None:
        """Calls a script that ran to its end made but never awaited. (When a run stops or
        fails part way, work it never reached is expected and not reported.)"""
        for deferred in self.unawaited:
            self.warnings.append(
                f"line {deferred.line}: {deferred.what} was never awaited, so it never ran"
            )
        self.unawaited.clear()

    def discard(self, items: list[Any]) -> None:
        """Drop calls that will never run: close coroutines (so Python does not warn about
        them) and forget deferred calls (so they are not reported as un-awaited)."""
        for item in items:
            if inspect.iscoroutine(item):
                item.close()
            elif isinstance(item, Deferred):
                self.unawaited.pop(item, None)

    def _warnings(self) -> list[str]:
        if self.journal is not None and self.journal.requested and not self.journal.available:
            return [*self.warnings, "resume was requested, but no earlier run left results"]
        return self.warnings


# --------------------------------------------------------------------------------------------
# One script being executed (the top-level workflow, or one it runs)
# --------------------------------------------------------------------------------------------


class _Frame:
    """The namespace and state of one executing script. Its public methods are the workflow
    API (they are what a script's names are bound to) and the API the patterns use."""

    def __init__(
        self, run: WorkflowRun, script: Script, node: NodeInfo, *, nested: bool, path: str
    ) -> None:
        self.run = run
        self.script = script
        self.node = node
        self.nested = nested
        self.verify_votes = run.limits.verify_votes
        self.namespace: dict[str, Any] = {}
        self._scope = _Scope(path)
        self._phases: dict[str, NodeInfo] = {}
        self._phase_records: dict[str, list[AgentRecord]] = {}
        self._current: str | None = None

    # -- running ----------------------------------------------------------------------------

    async def execute(self, args: Any) -> Any:
        self.namespace = self._namespace(args)
        token = _SCOPE.set(self._scope)
        try:
            # The code was checked and compiled by compile_script; see its module docstring.
            pending = eval(self.script.code, self.namespace)
            if inspect.iscoroutine(pending):
                await pending
        finally:
            _SCOPE.reset(token)
        if "result" not in self.namespace:
            self.run.warnings.append(
                f'workflow "{self.script.meta.name}" ended without assigning `result`'
            )
        return self.namespace.get("result")

    def _namespace(self, args: Any) -> dict[str, Any]:
        builtins = dict(_BUILTINS)
        builtins.update(range=_capped_range, print=self._print, **{TICK: self.run.tick})
        namespace: dict[str, Any] = {
            "__builtins__": builtins,
            "__name__": "workflow",
            "agent": self.agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "phase": self.phase,
            "log": self.log,
            "workflow": self.workflow,
            "args": copy.deepcopy(args) if args is not None else {},
            "budget": Budget(self.run.meter),
            "json": _JSON,
            "WorkflowError": WorkflowError,
        }
        for name, function in PATTERNS.items():
            namespace[name] = self._bind_pattern(name, function)
        return namespace

    def close(self, status: NodeStatus) -> None:
        """Settle the phase nodes when the script has ended (`status`: done/failed/cancelled)."""
        session = self.run.session
        for title, node in self._phases.items():
            records = self._phase_records.get(title, [])
            final: NodeStatus = "done"
            if status == "cancelled" or any(r.status == "cancelled" for r in records):
                final = "cancelled"
            elif status == "failed" and title == self._current:
                final = "failed"
            session.update_node(node, status=final, summary=_phase_summary(records))

    def phase_titles(self) -> list[str]:
        return list(self._phases)

    # -- the workflow API -------------------------------------------------------------------

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
    ) -> Deferred:
        """Start one agent when awaited; its final text, a dict matching `schema`, or None."""
        spec = self._spec(prompt, schema, model, label, phase, tools, write_scope, effort)
        path = self._current_scope().next("agent")
        line = self._line()
        return Deferred(self.run, "agent(...)", line, lambda: self._run_agent(spec, path, line))

    def parallel(self, items: Iterable[Any]) -> Deferred:
        """Run awaitables (or zero-argument callables) together; a failed slot becomes None."""
        jobs = self._materialise(items, "parallel")
        for item in jobs:
            if not (inspect.isawaitable(item) or callable(item)):
                self.run.discard(jobs)
                raise WorkflowError(
                    "parallel() takes awaitable calls such as agent(...), or functions without "
                    f"arguments that return one; got {type(item).__name__}"
                )
        base = self._current_scope().next("parallel")
        line = self._line()

        def slot(index: int, item: Any) -> Callable[[], Awaitable[Any]]:
            async def run_slot() -> Any:
                _SCOPE.set(_Scope(f"{base}/slot:{index}"))
                value = item if inspect.isawaitable(item) else item()
                return await value if inspect.isawaitable(value) else value

            return run_slot

        return Deferred(
            self.run,
            "parallel(...)",
            line,
            lambda: self._gather("parallel", line, [slot(i, x) for i, x in enumerate(jobs)]),
        )

    def pipeline(self, items: Iterable[Any], *stages: Callable[..., Any]) -> Deferred:
        """Move each item through `stages` on its own; `stage(previous, item, index)`."""
        values = self._materialise(items, "pipeline")
        if not stages:
            raise WorkflowError("pipeline() needs at least one stage after the items")
        for stage in stages:
            if not callable(stage):
                raise WorkflowError(
                    "each pipeline stage must be a function called as stage(previous, item, "
                    f"index); got {type(stage).__name__}"
                )
        base = self._current_scope().next("pipeline")
        line = self._line()

        def flow(index: int, item: Any) -> Callable[[], Awaitable[Any]]:
            async def run_item() -> Any:
                previous = item
                for number, stage in enumerate(stages):
                    _SCOPE.set(_Scope(f"{base}/item:{index}/stage:{number}"))
                    value = stage(previous, item, index)
                    if inspect.isawaitable(value):
                        value = await value
                    if value is None:
                        return None
                    previous = value
                return previous

            return run_item

        return Deferred(
            self.run,
            "pipeline(...)",
            line,
            lambda: self._gather("pipeline", line, [flow(i, x) for i, x in enumerate(values)]),
        )

    def phase(self, title: str) -> None:
        """Make `title` the phase that agents started from now on belong to."""
        title = _text(title, "phase title")
        previous, self._current = self._current, title
        self._phase_node(title)
        self._refresh_phase(title)
        if previous is not None and previous != title:
            self._refresh_phase(previous)

    def log(self, *parts: Any) -> None:
        """Show the user a progress line (a notice on the workflow's node) and keep it for
        the account."""
        text = " ".join(_as_text(part) for part in parts).strip()
        if len(text) > 1000:
            text = text[:1000] + f" [{len(text) - 1000:,} more characters not shown]"
        self.run.logs.append(text)
        self.run.session.notice("info", f"{self.script.meta.name}: {text}", self.node.id)

    def _print(self, *parts: Any, sep: str = " ", **_options: Any) -> None:
        """`print` in a script logs; `end`, `file` and `flush` have no meaning there."""
        self.log(sep.join(_as_text(part) for part in parts))

    def workflow(self, name: str, args: Any = None) -> Deferred:
        """Run the saved workflow `name` inside this one, sharing its limits and budget."""
        if self.nested:
            raise WorkflowError(
                "workflow() is available one level deep only: a workflow started by another "
                "workflow cannot start a third"
            )
        saved = find_saved(_text(name, "workflow name"))
        if saved is None:
            raise WorkflowError(
                f"there is no saved workflow named {name!r}; available: "
                + (", ".join(saved_names()) or "none")
            )
        path = self._current_scope().next(f"workflow:{saved.name}")
        line = self._line()
        return Deferred(
            self.run,
            f"workflow({name!r})",
            line,
            lambda: self._run_nested(saved.script, args, path),
        )

    async def check(self, command: str) -> CheckResult:
        """Run a check command for the harness, shown as a tool call on the workflow node."""
        session = self.run.session
        call_id = f"{self.node.id}-check{next(_check_ids)}"
        session.emit(
            ToolStarted(
                node_id=self.node.id,
                call_id=call_id,
                tool="check",
                summary=f"$ {command}",
                input={"command": command},
            )
        )
        result = await run_check(session, session.tool_context(self.node), command)
        session.emit(
            ToolFinished(
                node_id=self.node.id,
                call_id=call_id,
                output=result.render(tail=4000),
                is_error=not result.passed,
            )
        )
        return result

    # -- agents -----------------------------------------------------------------------------

    def _spec(
        self,
        prompt: Any,
        schema: Any,
        model: Any,
        label: Any,
        phase: Any,
        tools: Any,
        write_scope: Any,
        effort: Any,
    ) -> AgentSpec:
        text = _text(prompt, "agent prompt")
        if tools not in TOOL_SETS:
            raise WorkflowError(f'agent(): tools must be "all" or "read-only", not {tools!r}')
        if effort is not None and effort not in EFFORT_LEVELS:
            raise WorkflowError(f"agent(): effort must be one of {', '.join(EFFORT_LEVELS)}")
        title = _text(label, "agent label") if label is not None else _title_from(text)
        return AgentSpec(
            prompt=text,
            title=title[:80],
            model=self._model(model),
            schema=_check_schema(schema) if schema is not None else None,
            phase=_text(phase, "phase") if phase is not None else None,
            tools=tools,
            write_scope=_check_scope(write_scope),
            effort=effort,
        )

    def _model(self, requested: Any) -> ModelRef:
        session = self.run.session
        if requested is not None and not isinstance(requested, str):
            raise WorkflowError("agent(): model must be a string")
        if requested in ROLE_NAMES:
            return session.model_for(requested)
        try:
            ref = resolve_model(session, requested, role="subagent")
        except ValueError as exc:
            raise WorkflowError(
                f"agent(): {exc} Roles work too: {', '.join(ROLE_NAMES)}."
            ) from None
        info = session.model_info(ref)
        if info is not None and info.supports_tools is False:
            raise WorkflowError(f"agent(): {ref} cannot call tools, so it cannot run an agent")
        return ref

    async def _run_agent(self, spec: AgentSpec, path: str, line: int) -> Any:
        run, session = self.run, self.run.session
        _raise_if_cancelling()
        title = spec.phase or self._current
        parent = self._phase_node(title) if title else self.node
        key = JournalRun.key(path, spec.fingerprint())
        if run.journal is not None:
            hit, value = run.journal.lookup(key, mutating=spec.mutating)
            if hit:
                return self._reuse(spec, parent, title, line, value)
        run.reserve_agent()
        run.meter.check()
        node = session.new_node(
            spec.title, parent=parent, role="worker", model=spec.model, brief=_brief(spec.prompt)
        )
        record = self._record(node, spec.title, title, line)
        self._refresh_phase(title)
        try:
            async with run.slots:
                run.meter.check()
                record.status = "running"
                session.update_node(node, status="running")
                outcome = await run_worker(session, run.host, node, spec)
        except (WorkflowStop, asyncio.CancelledError) as exc:
            record.status = "cancelled"
            reason = f"Stopped: {exc}." if isinstance(exc, WorkflowStop) else "Cancelled."
            session.update_node(node, status="cancelled", summary=reason)
            self._refresh_phase(title)
            raise
        if outcome.error:
            record.status, record.error = "failed", outcome.error
            session.update_node(node, status="failed", summary=_one_line(outcome.error))
            run.failures.append(f'line {line}: agent "{spec.title}" ({node.id}): {outcome.error}')
            self._refresh_phase(title)
            return None
        record.status, record.value = "done", outcome.value
        session.update_node(node, status="done", summary=_one_line(render_value(outcome.value)))
        if run.journal is not None:
            run.journal.record(key, outcome.value)
        self._refresh_phase(title)
        return outcome.value

    def _reuse(
        self, spec: AgentSpec, parent: NodeInfo, phase: str | None, line: int, value: Any
    ) -> Any:
        node = self.run.session.new_node(
            spec.title,
            parent=parent,
            role="worker",
            model=spec.model,
            brief=_brief(spec.prompt),
            status="done",
        )
        self.run.session.update_node(node, summary="Reused the result of the previous run.")
        record = self._record(node, spec.title, phase, line)
        record.status, record.value = "reused", value
        self._refresh_phase(phase)
        return value

    def _record(self, node: NodeInfo, label: str, phase: str | None, line: int) -> AgentRecord:
        record = AgentRecord(node.id, label, phase or "", line)
        self.run.records.append(record)
        if phase:
            self._phase_records.setdefault(phase, []).append(record)
        return record

    # -- nesting ----------------------------------------------------------------------------

    async def _run_nested(self, script: Script, args: Any, path: str) -> Any:
        session = self.run.session
        parent = self._phase_node(self._current) if self._current else self.node
        node = session.new_node(
            script.meta.name,
            parent=parent,
            role="workflow",
            status="running",
            brief=script.meta.description,
        )
        frame = _Frame(self.run, script, node, nested=True, path=path)
        try:
            value = await frame.execute(args)
        except (WorkflowStop, asyncio.CancelledError):
            frame.close("cancelled")
            session.update_node(node, status="cancelled")
            raise
        except Exception as exc:
            frame.close("failed")
            session.update_node(node, status="failed", summary=_one_line(str(exc)))
            raise WorkflowError(
                f'the workflow "{script.meta.name}" failed: {describe_error(exc, script)}'
            ) from None
        frame.close("done")
        session.update_node(node, status="done")
        return value

    # -- fan-out ----------------------------------------------------------------------------

    async def _gather(
        self, kind: str, line: int, jobs: list[Callable[[], Awaitable[Any]]]
    ) -> list[Any]:
        """Run every job as its own task; failures become None and are accounted for.

        A `WorkflowStop` in any job cancels the others and propagates, and so does the
        cancellation of the caller. No task outlives this call.
        """
        tasks = [asyncio.ensure_future(job()) for job in jobs]
        results: list[Any] = [None] * len(tasks)
        index = {task: i for i, task in enumerate(tasks)}
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                stop: WorkflowStop | None = None
                for task in done:
                    if task.cancelled():
                        continue
                    error = task.exception()
                    if isinstance(error, WorkflowStop):
                        stop = stop or error
                    elif error is not None:
                        self.run.failures.append(
                            f"line {line}: {kind} {'slot' if kind == 'parallel' else 'item'} "
                            f"{index[task]}: {describe_error(error, self.script)}"
                        )
                    else:
                        results[index[task]] = task.result()
                if stop is not None:
                    raise stop
        finally:
            unfinished = [t for t in tasks if not t.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
        empty = sum(1 for value in results if value is None)
        if empty:
            unit = "slots" if kind == "parallel" else "items"
            self.run.empties.append(
                f"line {line}: {kind} of {len(results)} {unit}: {empty} ended without a result"
            )
        return results

    # -- phases -----------------------------------------------------------------------------

    def _phase_node(self, title: str) -> NodeInfo:
        node = self._phases.get(title)
        if node is None:
            detail = next((p.detail for p in self.script.meta.phases if p.title == title), "")
            node = self.run.session.new_node(
                title, parent=self.node, role="phase", status="running", brief=detail
            )
            self._phases[title] = node
        return node

    def _refresh_phase(self, title: str | None) -> None:
        """A phase is running while it is current or has unfinished agents, else done."""
        if title is None or title not in self._phases:
            return
        records = self._phase_records.get(title, [])
        active = title == self._current or any(r.status in ("pending", "running") for r in records)
        self.run.session.update_node(
            self._phases[title],
            status="running" if active else "done",
            summary=_phase_summary(records),
        )

    # -- helpers ----------------------------------------------------------------------------

    def _current_scope(self) -> _Scope:
        return _SCOPE.get() or self._scope

    def _materialise(self, items: Iterable[Any], kind: str) -> list[Any]:
        if isinstance(items, (str, bytes, dict)) or not isinstance(items, Iterable):
            raise WorkflowError(f"{kind}() takes a list of items, not {type(items).__name__}")
        values = list(items)
        if len(values) > MAX_ITEMS:
            self.run.discard(values)
            raise WorkflowError(
                f"{kind}() got {len(values):,} items; the limit is {MAX_ITEMS:,} per call. "
                "Split the work into batches, or narrow it first (and log what was left out)."
            )
        return values

    def _line(self) -> int:
        """The line of the script that made the current call (0 if it cannot be found)."""
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_filename == self.script.filename:
                    return frame.f_lineno or 0
                frame = frame.f_back
            return 0
        finally:
            del frame

    def _bind_pattern(
        self, name: str, function: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Deferred]:
        @functools.wraps(function)
        def call(*args: Any, **kwargs: Any) -> Deferred:
            return Deferred(
                self.run, f"{name}(...)", self._line(), lambda: function(self, *args, **kwargs)
            )

        return call


class Budget:
    """The workflow's token budget as scripts see it: ``total``, ``spent()``, ``remaining()``."""

    __slots__ = ("_meter",)

    def __init__(self, meter: TokenMeter) -> None:
        self._meter = meter

    @property
    def total(self) -> int:
        return self._meter.total

    def spent(self) -> int:
        return self._meter.spent

    def remaining(self) -> int:
        return max(0, self._meter.total - self._meter.spent)

    def __repr__(self) -> str:
        return f"budget(total={self.total}, spent={self.spent()})"


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def describe_error(error: BaseException, script: Script) -> str:
    """An exception raised while `script` ran, located at the script line that caused it."""
    text = str(error) if isinstance(error, WorkflowError) else f"{type(error).__name__}: {error}"
    line = 0
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename == script.filename:
            line = traceback.tb_lineno
        traceback = traceback.tb_next
    if not line:
        return text
    source = script.line(line)
    return f"line {line}: {text}" + (f"\n    {source}" if source else "")


def _as_text(value: Any) -> str:
    """A value for a log line: strings as they are, anything else as compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=repr)


def _capped_range(*bounds: int) -> range:
    span = range(*bounds)
    if len(span) > MAX_RANGE:
        raise WorkflowError(f"range() of {len(span):,} numbers is too large for a workflow script")
    return span


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{what} must be a non-empty string")
    return value.strip()


def _title_from(prompt: str) -> str:
    first = prompt.strip().splitlines()[0]
    return first if len(first) <= 60 else first[:59] + "…"


def _brief(prompt: str) -> str:
    return prompt if len(prompt) <= BRIEF_CHARS else prompt[:BRIEF_CHARS] + "…"


def _one_line(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _phase_summary(records: list[AgentRecord]) -> str:
    failed = sum(1 for r in records if r.status == "failed")
    noun = "agent" if len(records) == 1 else "agents"
    return f"{len(records)} {noun}" + (f", {failed} failed" if failed else "")


def _check_schema(schema: Any) -> dict[str, Any]:
    """Refuse schemas that no result could satisfy, before any agent is started."""
    if not isinstance(schema, dict):
        raise WorkflowError("agent(): schema must be a dict holding a JSON Schema")
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        json.dumps(schema)
    except jsonschema.SchemaError as exc:
        raise WorkflowError(f"agent(): schema is not a valid JSON Schema: {exc.message}") from None
    except (TypeError, ValueError) as exc:
        raise WorkflowError(f"agent(): schema must be plain JSON data: {exc}") from None
    if schema.get("type") != "object":
        raise WorkflowError(
            'agent(): the schema\'s root must be {"type": "object", ...}; wrap a list or '
            "value in a property"
        )
    properties = schema.get("properties", {})
    missing = [name for name in schema.get("required", []) if name not in properties]
    if missing:
        raise WorkflowError(
            f"agent(): schema requires {', '.join(missing)} but does not define "
            f"{'it' if len(missing) == 1 else 'them'} under properties"
        )
    return schema


def _check_scope(scope: Any) -> tuple[str, ...] | None:
    if scope is None:
        return None
    if isinstance(scope, str) or not isinstance(scope, Sequence):
        raise WorkflowError("agent(): write_scope must be a list of paths or globs")
    cleaned: list[str] = []
    for entry in scope:
        if not isinstance(entry, str):
            raise WorkflowError("agent(): write_scope entries must be strings")
        text = entry.strip()
        while text.startswith("./"):
            text = text[2:]
        if text.startswith(("/", "~")) or ".." in text.split("/"):
            raise WorkflowError(
                f"agent(): write_scope entries must be paths inside the project, got {entry!r}"
            )
        if text:
            cleaned.append(text)
    return tuple(cleaned)
