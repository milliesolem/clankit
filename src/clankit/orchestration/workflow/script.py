"""Workflow scripts: checking and compiling them before anything runs.

A workflow script is Python with top-level ``await``. Its first statement is a literal
``meta`` dict, and it hands its answer back by assigning ``result``. The script only
orchestrates: it has no imports, no clock, no randomness and no file or network access of
its own; the agents it starts do the real work.

These restrictions are determinism hygiene, not a security boundary. Scripts run in-process
under CPython, and the agents they start already have a shell. What the checks buy is
reproducibility (a script that never reads the clock or draws random numbers takes the same
path when it runs again, which is what lets a resumed run reuse finished agents) and early,
precise feedback: every problem is reported with its line number before a single agent
starts, so the agent that wrote the script can fix all of them in one go.

One Python pitfall gets special treatment. A function created in a loop that refers to the
loop variable sees that variable's *last* value when it finally runs, so
``parallel([lambda: agent(f"vote {k}") for k in range(3)])`` would send "vote 2" three
times. Such closures are rejected with a pointer to the two safe forms: pass the calls
themselves (``parallel([agent(f"vote {k}") for k in range(3)])``) or bind the variable as a
default argument (``lambda k=k: ...``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from types import CodeType
from typing import Any

#: Name of the loop-iteration counter the compiler inserts into every loop. It starts with
#: two underscores, so no script can refer to it.
TICK = "__clankit_tick__"

#: Calls that return something to await; a bare statement calling one is always a mistake.
ASYNC_CALLS: frozenset[str] = frozenset(
    {
        "agent",
        "parallel",
        "pipeline",
        "workflow",
        "adversarial_verify",
        "judge_panel",
        "loop_until_dry",
        "completeness_critic",
        "fix_until",
    }
)

META_KEYS: tuple[str, ...] = ("name", "description", "when_to_use", "phases", "args")
MAX_PROBLEMS = 20

_DETERMINISM = (
    "a workflow must take the same path when it runs again, so it cannot read the clock or "
    "draw random numbers; pass such values in `args`"
)
_SYSTEM = "the script itself cannot touch files, processes or the network; ask an agent to"
_REFLECTION = "it would let the script reach outside the workflow API"

#: Names a script may not use, with the reason given to the agent that wrote it.
BANNED_NAMES: dict[str, str] = {
    "time": _DETERMINISM,
    "datetime": _DETERMINISM,
    "random": _DETERMINISM,
    "uuid": _DETERMINISM,
    "secrets": _DETERMINISM,
    "os": _SYSTEM,
    "sys": _SYSTEM,
    "subprocess": _SYSTEM,
    "pathlib": _SYSTEM,
    "shutil": _SYSTEM,
    "socket": _SYSTEM,
    "open": _SYSTEM,
    "input": "nobody can answer questions while a workflow runs",
    "asyncio": "use `parallel([...])` or `pipeline(...)` to run things at the same time",
    "eval": _REFLECTION,
    "exec": _REFLECTION,
    "compile": _REFLECTION,
    "globals": _REFLECTION,
    "locals": _REFLECTION,
    "vars": _REFLECTION,
    "getattr": _REFLECTION,
    "setattr": _REFLECTION,
    "delattr": _REFLECTION,
    "breakpoint": _REFLECTION,
    "exit": "assign `result` and let the script end",
    "quit": "assign `result` and let the script end",
}

#: Calls that use a ``key`` function before they return, so a lambda passed to them cannot
#: outlive the loop iteration it was made in.
IMMEDIATE_KEY_CALLS: frozenset[str] = frozenset({"sorted", "min", "max", "sort"})

#: Exception names an ``except`` clause may not catch: they carry cancellation and limits.
UNCATCHABLE: frozenset[str] = frozenset(
    {"BaseException", "CancelledError", "GeneratorExit", "KeyboardInterrupt", "SystemExit"}
)

IMPORTS = (
    "imports are not available in workflow scripts; agent, parallel, pipeline, phase, log, "
    "args, budget, workflow, the pattern functions and `json` are already defined"
)
LATE_BINDING = (
    "this {kind} uses the loop variable `{name}`, but Python looks the variable up only when "
    "the {kind} runs, by which time the loop has moved on and every copy sees the last "
    "value. Pass the calls themselves (`parallel([agent(...) for {name} in ...])`) or bind "
    "the current value as a default argument (`lambda {name}={name}: ...`)"
)


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a script, located by line."""

    line: int
    message: str
    source: str = ""

    def render(self) -> str:
        text = f"line {self.line}: {self.message}"
        return f"{text}\n    {self.source}" if self.source else text


class ScriptError(Exception):
    """A script was rejected before it ran. `problems` are ordered by line."""

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = tuple(problems)
        super().__init__("\n".join(p.render() for p in self.problems))


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    title: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Meta:
    """The literal ``meta`` block: what the workflow is called, what it does, its phases.

    `args` documents the arguments a saved workflow expects, as (name, description) pairs.
    """

    name: str
    description: str
    when_to_use: str = ""
    phases: tuple[PhaseSpec, ...] = ()
    args: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Script:
    """A checked, compiled workflow script, ready to run any number of times."""

    meta: Meta
    source: str
    filename: str
    code: CodeType
    #: Agents the script starts at the very least, counting calls in loops over literals.
    estimated_agents: int

    def line(self, number: int) -> str:
        """Source text of line `number` (1-based), stripped; "" if out of range."""
        lines = self.source.splitlines()
        return lines[number - 1].strip() if 0 < number <= len(lines) else ""


def compile_script(source: str, *, filename: str = "<workflow>") -> Script:
    """Check `source` and compile it. Raises `ScriptError` listing every problem found."""
    lines = source.splitlines()

    def problem(line: int, message: str) -> Problem:
        text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        return Problem(line, message, text)

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ScriptError([problem(exc.lineno or 1, f"syntax error: {exc.msg}")]) from None

    meta, found = _read_meta(tree)
    checker = _Checker(meta)
    checker.visit(tree)
    found.extend(checker.problems)
    if not _assigns_result(tree):
        last = (tree.body[-1].end_lineno or 1) if tree.body else 1
        found.append(
            (last, "the script never assigns `result`; assign the value to hand back to it")
        )
    if found or meta is None:
        unique = sorted(set(found))[:MAX_PROBLEMS]
        raise ScriptError([problem(line, message) for line, message in unique])

    _Ticker().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        code = compile(
            tree, filename, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True
        )
    except SyntaxError as exc:
        message = exc.msg
        if "'return' outside function" in message:
            message += "; assign the value to `result` instead"
        raise ScriptError([problem(exc.lineno or 1, message)]) from None
    return Script(meta, source, filename, code, _estimate_agents(tree))


# --------------------------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------------------------

Found = list[tuple[int, str]]


def _read_meta(tree: ast.Module) -> tuple[Meta | None, Found]:
    first = tree.body[0] if tree.body else None
    line = first.lineno if first is not None else 1
    if not (
        isinstance(first, ast.Assign)
        and len(first.targets) == 1
        and isinstance(first.targets[0], ast.Name)
        and first.targets[0].id == "meta"
    ):
        return None, [
            (line, 'the first statement must be `meta = {"name": ..., "description": ...}`')
        ]
    try:
        value = ast.literal_eval(first.value)
    except ValueError:
        return None, [
            (
                line,
                "`meta` must be a plain literal (strings, numbers, lists and dicts; no "
                "variables, calls or f-strings)",
            )
        ]
    return _meta_from(value, line)


def _meta_from(value: Any, line: int) -> tuple[Meta | None, Found]:
    if not isinstance(value, dict):
        return None, [(line, "`meta` must be a dict")]
    found: Found = []
    unknown = sorted(str(k) for k in value if k not in META_KEYS)
    if unknown:
        found.append(
            (line, f"unknown meta key(s) {', '.join(unknown)}; allowed: {', '.join(META_KEYS)}")
        )
    name = _meta_text(value, "name", line, found, required=True, limit=60)
    description = _meta_text(value, "description", line, found, required=True, limit=600)
    when = _meta_text(value, "when_to_use", line, found, required=False, limit=600)
    phases = _meta_phases(value.get("phases", []), line, found)
    args = _meta_args(value.get("args", {}), line, found)
    if found:
        return None, found
    return Meta(name, description, when, phases, args), []


def _meta_text(
    value: dict[Any, Any], key: str, line: int, found: Found, *, required: bool, limit: int
) -> str:
    text = value.get(key, "")
    if not isinstance(text, str) or (required and not text.strip()):
        found.append((line, f'meta["{key}"] must be a non-empty string'))
        return ""
    text = " ".join(text.split())
    if len(text) > limit:
        found.append((line, f'meta["{key}"] is too long (at most {limit} characters)'))
    return text


def _meta_phases(raw: Any, line: int, found: Found) -> tuple[PhaseSpec, ...]:
    if not isinstance(raw, list):
        found.append((line, 'meta["phases"] must be a list of titles'))
        return ()
    phases: list[PhaseSpec] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            phases.append(PhaseSpec(entry.strip()))
        elif isinstance(entry, dict) and isinstance(entry.get("title"), str):
            detail = entry.get("detail", "")
            phases.append(
                PhaseSpec(entry["title"].strip(), detail if isinstance(detail, str) else "")
            )
        else:
            found.append((line, 'each phase must be a title, or {"title": ..., "detail": ...}'))
    titles = [p.title for p in phases]
    if len(set(titles)) != len(titles):
        found.append((line, "phase titles must be unique"))
    return tuple(phases)


def _meta_args(raw: Any, line: int, found: Found) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
    ):
        found.append((line, 'meta["args"] must map argument names to descriptions'))
        return ()
    return tuple(raw.items())


# --------------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------------


def _called_name(node: ast.Call) -> str:
    return node.func.id if isinstance(node.func, ast.Name) else ""


def _bound_names(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


class _Checker(ast.NodeVisitor):
    """Collects every problem in one pass; the script is rejected if there are any."""

    def __init__(self, meta: Meta | None) -> None:
        self.problems: Found = []
        self._phases = {p.title for p in meta.phases} if meta and meta.phases else None
        self._loops: list[frozenset[str]] = []
        #: Lambdas that are consumed on the spot (key functions of sorted, min, max, sort).
        self._immediate: set[int] = set()

    def _add(self, node: ast.AST, message: str) -> None:
        self.problems.append((getattr(node, "lineno", 1), message))

    # -- forbidden constructs --------------------------------------------------------------

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        self._add(node, IMPORTS)

    visit_ImportFrom = visit_Import

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node, "classes are not available in workflow scripts; use dicts and functions")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self._add(node, f"`{node.id}`: names starting with two underscores are not available")
        elif node.id in BANNED_NAMES:
            self._add(node, f"`{node.id}` is not available: {BANNED_NAMES[node.id]}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self._add(
                node, f"`.{node.attr}`: attributes starting with an underscore are not available"
            )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._add(
                node,
                "a bare `except:` would also catch the signals that stop a workflow; "
                "write `except Exception:`",
            )
        else:
            caught = {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
            for name in sorted(caught & UNCATCHABLE):
                self._add(
                    node,
                    f"`{name}` carries cancellation and limits and must not be caught; "
                    "catch `Exception` instead",
                )
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call) and _called_name(node.value) in ASYNC_CALLS:
            name = _called_name(node.value)
            self._add(
                node,
                f"`{name}(...)` is not awaited, so it would never run; write `await {name}(...)`",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else _called_name(node)
        if called in IMMEDIATE_KEY_CALLS:
            arguments = [*node.args, *(k.value for k in node.keywords if k.arg == "key")]
            self._immediate.update(id(a) for a in arguments if isinstance(a, ast.Lambda))
        if self._phases is not None:
            title = _literal_phase(node)
            if title is not None and title not in self._phases:
                declared = ", ".join(f'"{t}"' for t in sorted(self._phases))
                self._add(node, f'phase "{title}" is not in meta["phases"] ({declared})')
        self.generic_visit(node)

    # -- late binding ----------------------------------------------------------------------

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        self._loops.append(frozenset(_bound_names(node.target)))
        for statement in node.body:
            self.visit(statement)
        self._loops.pop()
        for statement in node.orelse:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def _comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        parts: list[ast.expr],
    ) -> None:
        names: set[str] = set()
        for index, generator in enumerate(node.generators):
            self.visit(generator.target)
            # The first iterable is evaluated outside the comprehension, the others inside.
            self._loops.append(frozenset(names if index else ()))
            self.visit(generator.iter)
            self._loops.pop()
            names |= _bound_names(generator.target)
            self._loops.append(frozenset(names))
            for condition in generator.ifs:
                self.visit(condition)
            self._loops.pop()
        self._loops.append(frozenset(names))
        for part in parts:
            self.visit(part)
        self._loops.pop()

    def visit_ListComp(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp) -> None:
        self._comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node, [node.key, node.value])

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if id(node) not in self._immediate:
            self._closure(node, node.args, [node.body], "lambda")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._closure(node, node.args, node.body, "function")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _closure(self, node: ast.AST, arguments: ast.arguments, body: list[Any], kind: str) -> None:
        active = set().union(*self._loops) if self._loops else set()
        if not active:
            return
        params = {a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
        params |= {a.arg for a in (arguments.vararg, arguments.kwarg) if a is not None}
        loads: set[str] = set()
        stores: set[str] = set()
        for part in body:
            for sub in ast.walk(part):
                if isinstance(sub, ast.Name):
                    (loads if isinstance(sub.ctx, ast.Load) else stores).add(sub.id)
        captured = sorted((loads - stores - params) & active)
        if captured:
            self._add(node, LATE_BINDING.format(kind=kind, name=captured[0]))


def _literal_phase(node: ast.Call) -> str | None:
    """The phase title a call names with a string literal, if it names one."""
    name = _called_name(node)
    value: ast.expr | None = None
    if name == "phase" and node.args:
        value = node.args[0]
    elif name == "agent":
        value = next((k.value for k in node.keywords if k.arg == "phase"), None)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _assigns_result(tree: ast.Module) -> bool:
    """Does the module body (outside function definitions) assign `result`?"""
    if any(isinstance(n, ast.Global) and "result" in n.names for n in ast.walk(tree)):
        return True
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Name) and node.id == "result" and isinstance(node.ctx, ast.Store):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


# --------------------------------------------------------------------------------------------
# compilation helpers
# --------------------------------------------------------------------------------------------


def _tick() -> ast.Call:
    return ast.Call(func=ast.Name(id=TICK, ctx=ast.Load()), args=[], keywords=[])


class _Ticker(ast.NodeTransformer):
    """Counts loop iterations, so a runaway loop ends with an error instead of freezing the
    event loop that every agent and both user interfaces share."""

    def _loop(self, node: ast.For | ast.AsyncFor | ast.While) -> ast.AST:
        self.generic_visit(node)
        node.body.insert(0, ast.copy_location(ast.Expr(_tick()), node.body[0]))
        return node

    visit_For = visit_AsyncFor = visit_While = _loop

    def visit_comprehension(self, node: ast.comprehension) -> ast.AST:
        self.generic_visit(node)
        node.ifs.append(ast.copy_location(_tick(), node.target))
        return node


def _literal_length(node: ast.expr) -> int:
    """How many items a loop over `node` visits, when that is evident; otherwise 1."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return max(len(node.elts), 1)
    if (
        isinstance(node, ast.Call)
        and _called_name(node) == "range"
        and node.args
        and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in node.args)
    ):
        try:
            return max(len(range(*(a.value for a in node.args))), 1)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return 1
    return 1


def _estimate_agents(node: ast.AST, factor: int = 1) -> int:
    """Lower-bound estimate of the agents a script starts: every `agent()` call site,
    multiplied by the length of each enclosing loop over a literal or `range(<number>)`."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        inner = factor * _literal_length(node.iter)
        return (
            _estimate_agents(node.iter, factor)
            + sum(_estimate_agents(s, inner) for s in node.body)
            + sum(_estimate_agents(s, factor) for s in node.orelse)
        )
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        total = 0
        multiplier = factor
        for generator in node.generators:
            total += _estimate_agents(generator.iter, multiplier)
            multiplier *= _literal_length(generator.iter)
        parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        return total + sum(_estimate_agents(p, multiplier) for p in parts)
    own = factor if isinstance(node, ast.Call) and _called_name(node) == "agent" else 0
    return own + sum(_estimate_agents(child, factor) for child in ast.iter_child_nodes(node))
