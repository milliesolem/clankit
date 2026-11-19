"""The web interface: a Pyramid app on waitress with a no-build vanilla JavaScript front end.

``clankit web`` calls `serve`; tests build the WSGI app directly with `make_app`. The browser
talks to a small JSON API and receives the session's engine events as Server-Sent Events; the
page folds them with the same rules as `clankit.agent.state.SessionView`.
"""

from clankit.web.app import make_app
from clankit.web.server import serve

__all__ = ["make_app", "serve"]
