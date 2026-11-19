"""The HTTP API and the page shell.

Handlers run in waitress threads and reach the engine only through ``ctx.engine`` (see
`clankit.web.context`): each one hands a small function to the runtime's event loop and turns
the result into JSON. Failures become ``{"error": "..."}`` with a fitting status: 400 for
input the user can correct, 404 for things that do not exist (or no longer do), 409 for
actions that conflict with what the engine is doing, 504 when the engine does not answer.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from pyramid.httpexceptions import HTTPSeeOther
from pyramid.renderers import render_to_response
from pyramid.request import Request
from pyramid.response import Response

from clankit.agent.session import Session
from clankit.commands import execute, parse
from clankit.config import actions, fields
from clankit.config.catalog import effective_providers
from clankit.config.schema import PROVIDER_KINDS, Config, ConfigError
from clankit.llm.types import ModelRef
from clankit.orchestration import get_orchestrator, method_names
from clankit.skills.library import SkillError
from clankit.themes import all_themes, get_theme
from clankit.themes.syntax import pygments_css
from clankit.themes.web import stylesheet
from clankit.web.context import COMMAND_TIMEOUT, ApiError, WebContext
from clankit.web.guard import LOGIN_PATH, error_response, is_authenticated
from clankit.web.markdown import enrich_snapshot
from clankit.web.payloads import app_state, session_meta, session_state
from clankit.web.stream import EventStream

log = logging.getLogger(__name__)

PERMISSION_CHOICES = ("allow", "allow_session", "deny")
#: Seconds model discovery may take; it asks every usable provider over the network.
DISCOVERY_TIMEOUT = 60.0

JSON = dict[str, Any]


def _ctx(request: Request) -> WebContext:
    return request.registry["clankit"]


# -- request bodies -----------------------------------------------------------------------------


def _body(request: Request) -> JSON:
    if not request.body:
        return {}
    try:
        data = request.json_body
    except ValueError as exc:
        raise ApiError(400, "The request body is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ApiError(400, "The request body must be a JSON object.")
    return data


def _text(body: JSON, key: str, *, required: bool = True) -> str:
    value = body.get(key, "")
    if not isinstance(value, str):
        raise ApiError(400, f"{key!r} must be a string.")
    if required and not value.strip():
        raise ApiError(400, f"{key!r} is required.")
    return value


def _flag(body: JSON, key: str) -> bool:
    value = body.get(key)
    if not isinstance(value, bool):
        raise ApiError(400, f"{key!r} must be true or false.")
    return value


# -- the page -----------------------------------------------------------------------------------


def _plain(status: int, text: str) -> Response:
    return Response(text + "\n", status=status, content_type="text/plain", charset="utf-8")


def index(request: Request) -> Response:
    """The page shell; with ``?token=`` it is the login that sets the session cookie."""
    ctx = _ctx(request)
    token = request.GET.get("token")
    if token is not None:
        if not ctx.credentials.accepts_token(token):
            return _plain(
                401,
                "This login link is not valid for the running Clankit server. Open the address "
                "printed in the terminal where it runs; a restarted server prints a new one.",
            )
        response = HTTPSeeOther(location=LOGIN_PATH)
        response.set_cookie(
            ctx.credentials.cookie_name,
            ctx.credentials.cookie_value,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response
    if not is_authenticated(ctx, request):
        return _plain(
            401,
            "Not logged in. Open the address printed in the terminal where Clankit runs; "
            "it contains the login token.",
        )

    def page() -> JSON:
        config = ctx.engine.runtime.config
        theme = get_theme(config.ui.theme, config)
        return {"theme": theme.id, "dark": theme.dark}

    values = ctx.engine.submit(page)
    values["workspace"] = ctx.workspace.name or str(ctx.workspace)
    return render_to_response("clankit.web:templates/index.jinja2", values, request=request)


def theme_css(request: Request) -> Response:
    """Every theme as CSS variables, plus the code colours that follow them."""
    ctx = _ctx(request)

    def build() -> str:
        config = ctx.engine.runtime.config
        return stylesheet(all_themes(config).values(), get_theme(config.ui.theme, config).id)

    css = ctx.engine.submit(build) + "\n" + pygments_css(".highlight")
    return Response(css, content_type="text/css", charset="utf-8")


# -- state --------------------------------------------------------------------------------------


def state(request: Request) -> JSON:
    """Application state, plus the requested (or most recent) session with its snapshot."""
    ctx = _ctx(request)
    session_id = request.GET.get("session", "")

    def build() -> JSON:
        session = ctx.sessions.get(session_id) if session_id else ctx.sessions.current()
        data = app_state(ctx.engine.runtime, ctx.sessions)
        data["session"] = session_state(session, ctx.sessions) if session is not None else None
        return data

    data = ctx.engine.submit(build)
    if data["session"] is not None:
        enrich_snapshot(data["session"]["snapshot"])
    return data


def get_session(request: Request) -> JSON:
    ctx = _ctx(request)
    session_id = request.matchdict["sid"]
    data = ctx.engine.submit(lambda: session_state(ctx.sessions.get(session_id), ctx.sessions))
    enrich_snapshot(data["snapshot"])
    return data


def get_session_meta(request: Request) -> JSON:
    """Just the facts the status row shows; cheap enough to ask after every settings change."""
    ctx = _ctx(request)
    session_id = request.matchdict["sid"]
    return ctx.engine.submit(lambda: session_meta(ctx.sessions.get(session_id), ctx.sessions))


def create_session(request: Request) -> JSON:
    """Start a new session (``{"method"?, "from"?}``) or reopen one (``{"resume": id}``)."""
    ctx = _ctx(request)
    body = _body(request)
    resume = _text(body, "resume", required=False)
    method = _text(body, "method", required=False) or None
    origin = _text(body, "from", required=False)

    def open_session() -> JSON:
        index = ctx.sessions
        if resume:
            session = index.resume(resume)
        else:
            like = index.get(origin) if origin else index.current()
            session = index.create(method=method, like=like)
        return {"id": session.id, "state": app_state(ctx.engine.runtime, index)}

    return ctx.engine.submit(open_session)


def events(request: Request) -> Response:
    """The session's event stream (Server-Sent Events)."""
    ctx = _ctx(request)
    session = ctx.engine.submit(lambda: ctx.sessions.get(request.matchdict["sid"]))
    stream = EventStream(
        ctx,
        session,
        tab=request.GET.get("tab", "")[:64] or "default",
        last_event_id=_last_event_id(request),
        disconnected=request.environ.get("waitress.client_disconnected", lambda: False),
    )
    response = Response(content_type="text/event-stream", charset="utf-8")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Accel-Buffering"] = "no"
    response.app_iter = stream
    return response


def _last_event_id(request: Request) -> int | None:
    raw = request.headers.get("Last-Event-ID") or request.GET.get("last_event_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# -- session actions ----------------------------------------------------------------------------


def _on_session(request: Request, action: Any) -> Any:
    """Run ``action(session)`` on the engine loop for the session named in the URL."""
    ctx = _ctx(request)
    session_id = request.matchdict["sid"]
    return ctx.engine.submit(lambda: action(ctx.sessions.get(session_id)))


def prompt(request: Request) -> JSON:
    text = _text(_body(request), "text")

    def start(session: Session) -> JSON:
        if session.busy:
            raise ApiError(
                409, "A run is already in progress. Send a steering message or cancel it first."
            )
        session.start(text)
        return {"started": True}

    result = _on_session(request, start)
    request.response.status_int = 202
    return result


def steer(request: Request) -> JSON:
    text = _text(_body(request), "text")

    def deliver(session: Session) -> JSON:
        if not session.steer(text):
            raise ApiError(409, "No run is in progress; send this as a prompt instead.")
        return {"delivered": True}

    return _on_session(request, deliver)


def cancel(request: Request) -> JSON:
    return _on_session(request, lambda session: {"cancelled": session.cancel()})


def permission(request: Request) -> JSON:
    choice = _text(_body(request), "choice")
    if choice not in PERMISSION_CHOICES:
        raise ApiError(400, f"'choice' must be one of: {', '.join(PERMISSION_CHOICES)}.")
    request_id = request.matchdict["rid"]

    def resolve(session: Session) -> JSON:
        if not session.resolve_permission(request_id, choice):  # type: ignore[arg-type]
            raise ApiError(404, "This permission request was already answered or has expired.")
        return {"resolved": True}

    return _on_session(request, resolve)


def question(request: Request) -> JSON:
    answer = _text(_body(request), "answer")
    request_id = request.matchdict["rid"]

    def reply(session: Session) -> JSON:
        if not session.answer_question(request_id, answer):
            raise ApiError(404, "This question was already answered or has expired.")
        return {"answered": True}

    return _on_session(request, reply)


def command(request: Request) -> JSON:
    """Run a slash command; the outcome tells the page what to do next."""
    ctx = _ctx(request)
    line = _text(_body(request), "line")
    parsed = parse(line)
    if parsed is None:
        raise ApiError(400, "A command starts with a slash, for example /help.")
    name, argument = parsed
    session_id = request.matchdict["sid"]

    async def run() -> JSON:
        session = ctx.sessions.get(session_id)
        outcome = await execute(ctx.engine.runtime, session, name, argument)
        return {
            "outcome": asdict(outcome),
            "meta": session_meta(session, ctx.sessions),
            "state": app_state(ctx.engine.runtime, ctx.sessions),
        }

    return ctx.engine.call(run, timeout=COMMAND_TIMEOUT)


def undo(request: Request) -> JSON:
    def revert(session: Session) -> JSON:
        try:
            return {"restored": session.undo()}
        except RuntimeError as exc:
            raise ApiError(409, "Wait for the run to finish before undoing its changes.") from exc

    return _on_session(request, revert)


# -- settings -----------------------------------------------------------------------------------


def _save(request: Request, mutate: Any, message: str) -> JSON:
    """Apply `mutate` to the configuration (written to config.kdl) and report the new state."""
    ctx = _ctx(request)

    def apply() -> JSON:
        runtime = ctx.engine.runtime
        runtime.update_config(mutate)
        path = runtime.store.path
        return {
            "message": message,
            "saved": str(path) if path is not None else "",
            "state": app_state(runtime, ctx.sessions),
        }

    return ctx.engine.submit(apply)


def set_setting(request: Request) -> JSON:
    """One value: a general field, ``orchestration.method``, or ``orchestration.<method>.<key>``."""
    path = request.matchdict["path"]
    body = _body(request)
    if "value" not in body:
        raise ApiError(400, "'value' is required.")
    value = body["value"]
    label = _setting_label(path)
    return _save(request, lambda config: _apply_setting(config, path, value), f"Saved {label}.")


def _setting_label(path: str) -> str:
    field = fields.find_field(path)
    if field is not None:
        return field.label
    if path == "orchestration.method":
        return "Orchestration method"
    _, method, key = [*path.split(".", 2), "", ""][:3]
    if path.startswith("orchestration.") and method and key:
        return f"{key} for {method}"
    raise ApiError(404, f"There is no setting {path!r}.")


def _apply_setting(config: Config, path: str, value: Any) -> None:
    if fields.find_field(path) is not None:
        fields.set_value(config, path, value)
        return
    if path == "orchestration.method":
        if value not in method_names():
            raise ConfigError(f"Unknown method {value!r}. Available: {', '.join(method_names())}.")
        config.orchestration.method = value
        return
    _, method, key = path.split(".", 2)
    try:
        orchestrator = get_orchestrator(method)
    except KeyError as exc:
        raise ApiError(404, f"There is no orchestration method {method!r}.") from exc
    spec = next((s for s in orchestrator.settings if s.key == key), None)
    if spec is None:
        raise ApiError(404, f"{orchestrator.title or method} has no setting {key!r}.")
    cleaned = spec.coerce(value)
    if spec.kind == "model" and cleaned:
        try:
            ModelRef.parse(cleaned)
        except ValueError as exc:
            raise ConfigError(f"{key}: {exc}") from exc
    config.orchestration.set_option(method, key, cleaned)


def _known_provider(config: Config, provider_id: str) -> None:
    if provider_id not in effective_providers(config):
        raise ApiError(404, f"There is no provider {provider_id!r}.")


def add_provider(request: Request) -> JSON:
    body = _body(request)
    provider_id = _text(body, "id")
    base_url = _text(body, "base_url")
    kind = _text(body, "kind", required=False) or "openai-compatible"
    if kind not in PROVIDER_KINDS:
        raise ApiError(400, f"'kind' must be one of: {', '.join(PROVIDER_KINDS)}.")
    api_key = _text(body, "api_key", required=False)
    name = _text(body, "name", required=False)
    keyless = _flag(body, "keyless") if "keyless" in body else not api_key

    def mutate(config: Config) -> None:
        actions.add_custom_provider(
            config,
            provider_id,
            base_url=base_url,
            kind=kind,
            api_key=api_key,
            keyless=keyless,
            name=name,
        )

    return _save(request, mutate, f"Added provider {provider_id.strip().lower()}.")


def update_provider(request: Request) -> JSON:
    """Change a provider's key, base URL or enabled flag (any subset)."""
    provider_id = request.matchdict["pid"]
    body = _body(request)
    changes: list[str] = []
    api_key = base_url = None
    enabled: bool | None = None
    if "api_key" in body:
        api_key = _text(body, "api_key", required=False)
        changes.append("API key removed" if not api_key.strip() else "API key saved")
    if "base_url" in body:
        base_url = _text(body, "base_url", required=False)
        changes.append("base URL saved")
    if "enabled" in body:
        enabled = _flag(body, "enabled")
        changes.append("enabled" if enabled else "disabled")
    if not changes:
        raise ApiError(400, "Nothing to change: send 'api_key', 'base_url' or 'enabled'.")

    def mutate(config: Config) -> None:
        _known_provider(config, provider_id)
        if api_key is not None:
            actions.set_api_key(config, provider_id, api_key)
        if base_url is not None:
            actions.set_base_url(config, provider_id, base_url)
        if enabled is not None:
            actions.set_enabled(config, provider_id, enabled)

    message = f"{provider_id}: {', '.join(changes)}."
    return _save(request, mutate, message[0].upper() + message[1:])


def remove_provider(request: Request) -> JSON:
    provider_id = request.matchdict["pid"]
    ctx = _ctx(request)
    still_used = ctx.engine.submit(
        lambda: actions.models_using(ctx.engine.runtime.config, provider_id)
    )

    def mutate(config: Config) -> None:
        if provider_id not in config.providers:
            raise ApiError(404, f"{provider_id} has no settings of its own to remove.")
        actions.remove_provider(config, provider_id)

    message = f"Removed the settings of {provider_id}."
    if still_used:
        message += f" Still pointing at it: {', '.join(still_used)}."
    return _save(request, mutate, message)


def discover_models(request: Request) -> JSON:
    """Ask every usable provider which models it offers (network), then report the state."""
    ctx = _ctx(request)

    async def discover() -> JSON:
        runtime = ctx.engine.runtime
        await runtime.providers.discover()
        return {"state": app_state(runtime, ctx.sessions)}

    return ctx.engine.call(discover, timeout=DISCOVERY_TIMEOUT)


# -- skills -------------------------------------------------------------------------------------


def _skill_action(request: Request, act: Any, message: str) -> JSON:
    ctx = _ctx(request)
    name = request.matchdict["name"]

    def run() -> JSON:
        runtime = ctx.engine.runtime
        try:
            act(runtime.skills, name)
        except KeyError as exc:
            raise ApiError(404, f"There is no skill {name!r} here.") from exc
        except SkillError as exc:
            raise ApiError(409, str(exc)) from exc
        return {"message": message.format(name=name), "state": app_state(runtime, ctx.sessions)}

    return ctx.engine.submit(run)


def approve_skill(request: Request) -> JSON:
    return _skill_action(request, lambda lib, name: lib.approve(name), "Approved skill {name}.")


def reject_skill(request: Request) -> JSON:
    return _skill_action(request, lambda lib, name: lib.reject(name), "Discarded proposal {name}.")


def delete_skill(request: Request) -> JSON:
    return _skill_action(request, lambda lib, name: lib.delete(name), "Deleted skill {name}.")


# -- errors -------------------------------------------------------------------------------------


def api_error(exc: ApiError, request: Request) -> Response:
    return error_response(exc.status, exc.message)


def config_error(exc: ConfigError, request: Request) -> Response:
    return error_response(400, str(exc))


def not_found(request: Request) -> Response:
    route = request.matched_route
    allowed = request.registry.get("clankit.methods", {}).get(route.name) if route else None
    if allowed and request.method not in allowed:
        response = error_response(405, f"{request.method} is not supported here.")
        response.headers["Allow"] = ", ".join(sorted(allowed))
        return response
    return error_response(404, "There is nothing at this address.")


def unexpected_error(exc: Exception, request: Request) -> Response:
    # The traceback goes to the log; the request body never does (it may hold an API key).
    log.error("%s %s failed", request.method, request.path, exc_info=exc)
    return error_response(500, "Something went wrong in Clankit; the server log has details.")
