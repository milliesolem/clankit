"""The WSGI application: routes, the guard tween, error views and static files.

API routes (all JSON; state-changing ones need the ``X-Clankit: 1`` header)::

    GET    /api/state[?session=ID]               everything the page needs, plus one session
    POST   /api/sessions                          new session, or {"resume": ID}
    GET    /api/sessions/{sid}                    a session: meta, snapshot, runs
    GET    /api/sessions/{sid}/meta               just its facts (model, usage, cost, context)
    GET    /api/sessions/{sid}/events             Server-Sent Events (see clankit.web.stream)
    POST   /api/sessions/{sid}/prompt             {"text"}            start a run (202)
    POST   /api/sessions/{sid}/steer              {"text"}            message the working agent
    POST   /api/sessions/{sid}/cancel                                 stop the run
    POST   /api/sessions/{sid}/permissions/{rid}  {"choice"}          allow | allow_session | deny
    POST   /api/sessions/{sid}/questions/{rid}    {"answer"}
    POST   /api/sessions/{sid}/command            {"line": "/name argument"}
    POST   /api/sessions/{sid}/undo
    PUT    /api/settings/{path}                   {"value"}           any single setting
    POST   /api/providers                         add a custom provider
    PATCH  /api/providers/{pid}                   {"api_key"?, "base_url"?, "enabled"?}
    DELETE /api/providers/{pid}
    POST   /api/models/discover                   ask providers for their model lists
    POST   /api/skills/{name}/approve | /reject   a proposed skill
    DELETE /api/skills/{name}                     an installed skill

Pages: ``/`` (the shell, and the login with ``?token=``), ``/theme.css``, ``/static/...``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pyramid.config import Configurator
from pyramid.router import Router
from pyramid.tweens import INGRESS

from clankit.config.schema import ConfigError
from clankit.runtime import RuntimeThread
from clankit.web import views
from clankit.web.context import ApiError, Credentials, Engine, WebContext
from clankit.web.guard import LOOPBACK_HOSTS
from clankit.web.sessions import SessionIndex
from clankit.web.stream import StreamRegistry

#: (route name, pattern, [(method, view), ...])
ROUTES: tuple[tuple[str, str, tuple[tuple[str, Callable[..., Any]], ...]], ...] = (
    ("index", "/", (("GET", views.index),)),
    ("theme_css", "/theme.css", (("GET", views.theme_css),)),
    ("state", "/api/state", (("GET", views.state),)),
    ("sessions", "/api/sessions", (("POST", views.create_session),)),
    ("session", "/api/sessions/{sid}", (("GET", views.get_session),)),
    ("session_meta", "/api/sessions/{sid}/meta", (("GET", views.get_session_meta),)),
    ("events", "/api/sessions/{sid}/events", (("GET", views.events),)),
    ("prompt", "/api/sessions/{sid}/prompt", (("POST", views.prompt),)),
    ("steer", "/api/sessions/{sid}/steer", (("POST", views.steer),)),
    ("cancel", "/api/sessions/{sid}/cancel", (("POST", views.cancel),)),
    ("permission", "/api/sessions/{sid}/permissions/{rid}", (("POST", views.permission),)),
    ("question", "/api/sessions/{sid}/questions/{rid}", (("POST", views.question),)),
    ("command", "/api/sessions/{sid}/command", (("POST", views.command),)),
    ("undo", "/api/sessions/{sid}/undo", (("POST", views.undo),)),
    ("setting", "/api/settings/{path}", (("PUT", views.set_setting),)),
    ("providers", "/api/providers", (("POST", views.add_provider),)),
    (
        "provider",
        "/api/providers/{pid}",
        (("PATCH", views.update_provider), ("DELETE", views.remove_provider)),
    ),
    ("discover", "/api/models/discover", (("POST", views.discover_models),)),
    ("skill_approve", "/api/skills/{name}/approve", (("POST", views.approve_skill),)),
    ("skill_reject", "/api/skills/{name}/reject", (("POST", views.reject_skill),)),
    ("skill", "/api/skills/{name}", (("DELETE", views.delete_skill),)),
)

#: Views that build their own response instead of returning JSON for the renderer.
_RAW_VIEWS = frozenset({views.index, views.theme_css, views.events})


def make_app(
    thread: RuntimeThread,
    *,
    token: str,
    workspace: Path,
    port: int | None = None,
    hosts: Iterable[str] = (),
) -> Router:
    """The web interface for sessions in `workspace`, served from a started `thread`.

    `token` is the login token (see `clankit.web.guard`). `port` is the port the server
    listens on; requests whose Host header names another port are refused (None, for tests,
    accepts any). `hosts` adds host names to the loopback ones a Host header may carry, for a
    server deliberately bound to another address.
    """
    engine = Engine(thread)
    resolved = workspace.resolve()
    ctx = WebContext(
        engine=engine,
        credentials=Credentials.for_token(token),
        workspace=resolved,
        sessions=SessionIndex(lambda: engine.runtime, resolved),
        streams=StreamRegistry(),
        hosts=frozenset({*LOOPBACK_HOSTS, *(h.lower() for h in hosts)}),
        port=port,
    )
    with Configurator(settings={"pyramid.reload_templates": False}) as config:
        config.include("pyramid_jinja2")
        config.registry["clankit"] = ctx
        config.add_tween("clankit.web.guard.guard_tween_factory", under=INGRESS)
        config.add_static_view("static", "clankit.web:static", cache_max_age=None)
        config.registry["clankit.methods"] = {
            name: frozenset(method for method, _ in handlers) for name, _, handlers in ROUTES
        }
        for name, pattern, handlers in ROUTES:
            config.add_route(name, pattern)
            for method, view in handlers:
                renderer = None if view in _RAW_VIEWS else "json"
                config.add_view(view, route_name=name, request_method=method, renderer=renderer)
        config.add_exception_view(views.api_error, context=ApiError)
        config.add_exception_view(views.config_error, context=ConfigError)
        config.add_notfound_view(views.not_found)
        config.add_exception_view(views.unexpected_error, context=Exception)
        return config.make_wsgi_app()


def context_of(app: Router) -> WebContext:
    """The shared state behind an application made by `make_app`."""
    return app.registry["clankit"]
