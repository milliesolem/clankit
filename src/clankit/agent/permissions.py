"""Deciding whether an agent may do something, and asking the user when policy says to.

Decision order for every request:

1. a matching **deny** rule refuses it;
2. mode ``read-only`` refuses anything that is not a read;
3. a matching **allow** rule (from the config, or granted earlier in this session) permits it;
4. mode ``auto`` permits it; mode ``accept-edits`` permits edits inside the workspace;
5. otherwise the user is asked, via a `PermissionRequested` event.

Rules are written ``category:pattern`` with shell-style wildcards and are matched against the
request's target. Shell commands get extra care: a compound command (``a && b``, ``a | b``,
``a; b``) is allowed by rules only if every part is, refused if any part is denied, and a
command using substitution (backticks or ``$(...)``) is never auto-allowed by a rule.
"""

from __future__ import annotations

import asyncio
import fnmatch
import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass

from clankit.agent.events import Event, PermissionChoice, PermissionRequested, PermissionResolved
from clankit.config.schema import PermissionsConfig
from clankit.tools.base import PermissionAsk, PermissionDecision

_ids = itertools.count(1)
_SEGMENT_SPLIT = re.compile(r"\|\||&&|[;|&\n]")
_SUBSTITUTION = re.compile(r"`|\$\(|<\(|>\(")


def split_rule(rule: str) -> tuple[str, str]:
    category, sep, pattern = rule.partition(":")
    if not sep:
        return "shell", rule.strip()  # a bare rule is a shell command pattern
    return category.strip(), pattern.strip()


def shell_segments(command: str) -> list[str]:
    """The simple commands a compound shell command is made of (a conservative split)."""
    return [part.strip() for part in _SEGMENT_SPLIT.split(command) if part.strip()]


def rule_matches(rule: str, category: str, target: str) -> bool:
    rule_category, pattern = split_rule(rule)
    if rule_category != category or not pattern:
        return False
    return fnmatch.fnmatchcase(target, pattern)


def any_rule_matches(rules: list[str], category: str, target: str, *, every_segment: bool) -> bool:
    """Match `target` against `rules`.

    For shell commands, `every_segment=True` demands that each part of a compound command
    matches (used for allow rules) while False is satisfied by any part (used for deny rules).
    """
    if category != "shell":
        return any(rule_matches(r, category, target) for r in rules)
    segments = shell_segments(target) or [target]
    hits = [any(rule_matches(r, "shell", seg) for r in rules) for seg in segments]
    if every_segment:
        return all(hits) and not _SUBSTITUTION.search(target)
    return any(hits) or any(rule_matches(r, "shell", target) for r in rules)


@dataclass(slots=True)
class _Pending:
    ask: PermissionAsk
    future: asyncio.Future[PermissionChoice]


class PermissionManager:
    """A session's permission gate. Implements `clankit.tools.base.PermissionGate`."""

    def __init__(
        self,
        *,
        policy: Callable[[], PermissionsConfig],
        emit: Callable[[Event], Event],
        interactive: bool = True,
    ) -> None:
        self._policy = policy  # a callable so config changes take effect immediately
        self._emit = emit
        self.interactive = interactive
        self.session_rules: list[str] = []
        self._pending: dict[str, _Pending] = {}

    # -- PermissionGate -------------------------------------------------------------------------

    async def check(self, ask: PermissionAsk, *, node_id: str, tool: str) -> PermissionDecision:
        decision = self._decide(ask)
        if decision is not None:
            return decision
        if not self.interactive:
            return PermissionDecision(
                False,
                "nobody is available to approve this (non-interactive run); "
                "it needs an allow rule or a more permissive permission mode",
            )

        request_id = f"p{next(_ids)}"
        future: asyncio.Future[PermissionChoice] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = _Pending(ask, future)
        rule = self._session_rule(ask)
        self._emit(
            PermissionRequested(
                request_id=request_id,
                node_id=node_id,
                tool=tool,
                category=ask.category,
                summary=ask.summary,
                detail=ask.detail,
                session_rule=rule,
            )
        )
        try:
            choice = await future
        except asyncio.CancelledError:
            if self._pending.pop(request_id, None) is not None:
                self._emit(PermissionResolved(request_id=request_id, choice="deny"))
            raise
        if choice == "deny":
            return PermissionDecision(False, "the user declined")
        return PermissionDecision(True)

    # -- answers --------------------------------------------------------------------------------

    def resolve(self, request_id: str, choice: PermissionChoice) -> bool:
        """Answer a pending request. Returns False if it is unknown or already answered."""
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return False
        if choice == "allow_session":
            self.session_rules.append(self._session_rule(pending.ask))
        if not pending.future.done():
            pending.future.set_result(choice)
        self._emit(PermissionResolved(request_id=request_id, choice=choice))
        if choice == "allow_session":
            # Other requests that the new rule now covers need no answer of their own.
            for other_id, other in list(self._pending.items()):
                decision = self._decide(other.ask)
                if decision is not None and decision.allowed:
                    self.resolve(other_id, "allow")
        return True

    @property
    def pending_ids(self) -> list[str]:
        return list(self._pending)

    # -- policy ---------------------------------------------------------------------------------

    def _decide(self, ask: PermissionAsk) -> PermissionDecision | None:
        """Decide from policy alone; None means the user has to be asked."""
        policy = self._policy()
        if any_rule_matches(policy.deny, ask.category, ask.target, every_segment=False):
            return PermissionDecision(False, "blocked by a deny rule in the configuration")
        if policy.mode == "read-only" and ask.category != "read":
            return PermissionDecision(False, "the permission mode is read-only")
        allow = [*policy.allow, *self.session_rules]
        if any_rule_matches(allow, ask.category, ask.target, every_segment=True):
            return PermissionDecision(True)
        if policy.mode == "auto":
            return PermissionDecision(True)
        inside_workspace = not ask.target.startswith(("/", "~", ".."))
        if policy.mode == "accept-edits" and ask.category == "edit" and inside_workspace:
            return PermissionDecision(True)
        return None

    @staticmethod
    def _session_rule(ask: PermissionAsk) -> str:
        return f"{ask.category}:{ask.session_rule or ask.target}"
