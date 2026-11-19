"""The terminal user interface.

    from clankit.tui import run_tui
    run_tui(runtime, session)

`run_tui` runs the Textual application until the user quits, closes the runtime, and then
prints where the session is stored and how to pick it up again, so that information
survives in the normal terminal after the full-screen interface is gone.
"""

from __future__ import annotations

from clankit.agent.session import Session
from clankit.runtime import Runtime
from clankit.tui.app import ClankitApp

__all__ = ["ClankitApp", "exit_summary", "run_tui"]


def exit_summary(session: Session) -> str:
    """The plain lines printed after the interface closes."""
    turns = len(session.turns)
    count = f"{turns} turn" + ("" if turns == 1 else "s")
    return f"Session {session.id} ({count}).\nResume it with: clankit --resume {session.id}"


def run_tui(runtime: Runtime, session: Session) -> None:
    """Run the terminal interface for `session` until the user quits."""
    app = ClankitApp(runtime, session)
    app.run()
    print(exit_summary(app.session))
