"""Shared harness for the web UI tests (``test_web_*.py``); it holds no tests itself.

`WebHarness` runs a real `RuntimeThread` (its own event loop, as under ``clankit web``) with a
scripted provider, builds the WSGI app with `make_app`, and drives it with WebTest. Settings
are written to a real config.kdl in a temporary directory so tests can check what lands in
the file.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from webtest import TestApp

from clankit.config.schema import Config
from clankit.config.store import ConfigStore
from clankit.llm.scripted import ScriptedProvider
from clankit.runtime import Runtime, RuntimeThread
from clankit.web import make_app
from clankit.web.app import context_of
from clankit.web.context import WebContext

TOKEN = "test-token-0123456789"
PORT = 8420
HOST = f"127.0.0.1:{PORT}"
CSRF = {"X-Clankit": "1"}


def scripted_config(*, mode: str = "auto") -> Config:
    config = Config()
    config.models.default = "scripted/test-model"
    config.permissions.mode = mode
    return config


class WebHarness:
    """The web app on a live runtime thread, with a logged-in WebTest client."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        config: Config | None = None,
        provider: ScriptedProvider | None = None,
        persist: bool = False,
        login: bool = True,
        port: int | None = PORT,
    ) -> None:
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir(exist_ok=True)
        self.config_path = tmp_path / "config" / "config.kdl"
        self.provider = provider
        self._config = config if config is not None else scripted_config()
        self._persist = persist
        self.thread = RuntimeThread(self._make_runtime).start()
        self.app = make_app(self.thread, token=TOKEN, workspace=self.workspace, port=port)
        self.client = TestApp(self.app, extra_environ={"HTTP_HOST": HOST})
        if login:
            self.client.get(f"/?token={TOKEN}", status=303)

    def _make_runtime(self) -> Runtime:
        store = ConfigStore.open(self.config_path)
        config = self._config
        store.update(lambda c: _copy_into(c, config))
        runtime = Runtime(store, persist_sessions=self._persist)
        if self.provider is not None:
            runtime.providers.register(self.provider)
        return runtime

    # -- access ---------------------------------------------------------------------------------

    @property
    def ctx(self) -> WebContext:
        return context_of(self.app)

    @property
    def runtime(self) -> Runtime:
        return self.thread.runtime

    def on_loop(self, fn: Callable[[], Any]) -> Any:
        return self.thread.submit(fn)

    def saved_config(self) -> Config:
        """The configuration as a fresh read of config.kdl sees it."""
        return ConfigStore.open(self.config_path).config

    # -- requests -------------------------------------------------------------------------------

    def get(self, path: str, status: int | str = 200) -> Any:
        return self.client.get(path, status=status)

    def post(self, path: str, body: dict[str, Any] | None = None, status: int | str = 200) -> Any:
        return self.client.post_json(path, body or {}, headers=CSRF, status=status)

    def put(self, path: str, body: dict[str, Any], status: int | str = 200) -> Any:
        return self.client.put_json(path, body, headers=CSRF, status=status)

    def patch(self, path: str, body: dict[str, Any], status: int | str = 200) -> Any:
        return self.client.patch_json(path, body, headers=CSRF, status=status)

    def delete(self, path: str, status: int | str = 200) -> Any:
        return self.client.delete(path, headers=CSRF, status=status)

    # -- sessions -------------------------------------------------------------------------------

    def new_session(self, **body: Any) -> str:
        return self.post("/api/sessions", body).json["id"]

    def session(self, session_id: str) -> dict[str, Any]:
        return self.get(f"/api/sessions/{session_id}").json

    def wait_for(
        self, session_id: str, predicate: Callable[[dict[str, Any]], Any], timeout: float = 10.0
    ) -> dict[str, Any]:
        """Poll the session over the API until `predicate(session)` holds."""
        deadline = time.monotonic() + timeout
        while True:
            state = self.session(session_id)
            if predicate(state):
                return state
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"session never reached the expected state: {state['snapshot']}"
                )
            time.sleep(0.01)

    def run(self, session_id: str, text: str) -> dict[str, Any]:
        """Submit a prompt and wait for the run to finish."""
        self.post(f"/api/sessions/{session_id}/prompt", {"text": text}, status=202)
        return self.wait_for(
            session_id,
            lambda s: not s["snapshot"]["running"] and s["snapshot"]["last_run"] is not None,
        )

    def close(self) -> None:
        self.ctx.streams.close_all()
        self.thread.stop()


def _copy_into(target: Config, source: Config) -> None:
    fresh = source.copy()
    for name in Config.__dataclass_fields__:
        setattr(target, name, getattr(fresh, name))


def assistant_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for node in state["snapshot"]["nodes"].values()
        for item in node["items"]
        if item["kind"] == "assistant"
    ]
