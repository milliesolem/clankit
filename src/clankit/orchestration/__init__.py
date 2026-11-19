"""Orchestration methods and their registry.

Built-in methods are imported on first lookup, so importing this package stays cheap and free
of import cycles. Third parties (and tests) can add a method with `register()`.
"""

from __future__ import annotations

import importlib
import logging

from clankit.orchestration.base import Orchestrator

#: name -> "module:Class" for the built-in methods, in the order pickers show them.
BUILTIN: dict[str, str] = {
    "task-tree": "clankit.orchestration.task_tree:TaskTreeOrchestrator",
    "solo": "clankit.orchestration.solo:SoloOrchestrator",
    "review-loop": "clankit.orchestration.review_loop:ReviewLoopOrchestrator",
    "test-loop": "clankit.orchestration.test_loop:TestLoopOrchestrator",
    "architect-editor": "clankit.orchestration.architect:ArchitectEditorOrchestrator",
    "escalate": "clankit.orchestration.escalate:EscalateOrchestrator",
    "best-of-n": "clankit.orchestration.best_of_n:BestOfNOrchestrator",
    "ultra": "clankit.orchestration.ultra:UltraOrchestrator",
    "demo": "clankit.orchestration.demo:DemoOrchestrator",
}

_registered: dict[str, type[Orchestrator]] = {}
log = logging.getLogger(__name__)


def register(cls: type[Orchestrator]) -> type[Orchestrator]:
    """Add an orchestration method. Usable as a class decorator."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a name")
    _registered[cls.name] = cls
    return cls


def get_orchestrator(name: str) -> type[Orchestrator]:
    if name in _registered:
        return _registered[name]
    target = BUILTIN.get(name)
    if target is None:
        raise KeyError(
            f"unknown orchestration method {name!r} (available: {', '.join(method_names())})"
        )
    module_name, _, class_name = target.partition(":")
    cls: type[Orchestrator] = getattr(importlib.import_module(module_name), class_name)
    _registered[name] = cls
    return cls


def method_names() -> list[str]:
    names = list(BUILTIN)
    names.extend(n for n in _registered if n not in BUILTIN)
    return names


def all_orchestrators() -> list[type[Orchestrator]]:
    """Every method that can be loaded, built-ins first.

    A method whose module fails to import is skipped (and logged) rather than taking the
    pickers and settings screens down with it.
    """
    found: list[type[Orchestrator]] = []
    for name in method_names():
        try:
            found.append(get_orchestrator(name))
        except (ImportError, AttributeError):
            log.exception("orchestration method %r could not be loaded", name)
    return found
