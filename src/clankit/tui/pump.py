"""Batching a session's events into the UI at a bounded rate.

A streaming model produces dozens of deltas per second, and the session's `EventBus` calls
its listeners synchronously in the publishing task. Repainting per event would spend the
event loop on layout instead of on the agents, so the pump only queues events and hands
them over as one batch at most every `interval` seconds. The UI then re-reads the folded
`SessionView` once per batch; the events in the batch only say what changed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from clankit.agent.events import Event, EventBus

Deliver = Callable[[list[Event]], None]


class EventPump:
    """Collects events from one session's bus and delivers them in batches.

    `deliver` is called on the event loop the pump was attached on, at most once every
    `interval` seconds and never with an empty batch. Attaching to another bus detaches from
    the previous one and drops anything still queued from it.
    """

    def __init__(self, deliver: Deliver, interval: float = 1 / 20) -> None:
        self._deliver = deliver
        self.interval = interval
        self._pending: list[Event] = []
        self._handle: asyncio.TimerHandle | None = None
        self._unsubscribe: Callable[[], None] | None = None

    @property
    def idle(self) -> bool:
        """True when nothing is queued for delivery."""
        return not self._pending and self._handle is None

    def attach(self, bus: EventBus) -> None:
        self.detach()
        self._unsubscribe = bus.subscribe(self._receive)

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._pending.clear()

    def flush(self) -> None:
        """Deliver whatever is queued now, without waiting for the interval."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if self._pending:
            batch, self._pending = self._pending, []
            self._deliver(batch)

    def _receive(self, event: Event) -> None:
        self._pending.append(event)
        if self._handle is None:
            loop = asyncio.get_running_loop()
            self._handle = loop.call_later(self.interval, self._fire)

    def _fire(self) -> None:
        self._handle = None
        self.flush()
