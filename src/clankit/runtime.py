"""The application runtime: configuration, providers, skills and sessions in one place.

Both user interfaces and the headless command create exactly one `Runtime`. The terminal UI
uses it directly on its own event loop. The web UI, whose request handlers run in server
threads, wraps it in a `RuntimeThread`, which owns an event loop in a background thread and
marshals calls onto it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from clankit import paths
from clankit.agent.session import Session, Turn
from clankit.agent.store import SessionStore
from clankit.config.schema import Config
from clankit.config.store import ConfigStore
from clankit.llm.registry import ProviderRegistry

if TYPE_CHECKING:
    from clankit.skills.library import SkillLibrary

log = logging.getLogger(__name__)

T = TypeVar("T")
AfterRun = Callable[[Session, Turn], Awaitable[None]]


class Runtime:
    def __init__(
        self,
        store: ConfigStore | None = None,
        *,
        session_store: SessionStore | None = None,
        persist_sessions: bool = True,
    ) -> None:
        self.store = store or ConfigStore.open()
        self.providers = ProviderRegistry(self.store.config)
        self.session_store = session_store or SessionStore()
        self.persist_sessions = persist_sessions
        self.sessions: dict[str, Session] = {}
        #: Coroutines called after every completed run, e.g. skill generation.
        self.after_run: list[AfterRun] = []
        self._skills: SkillLibrary | None = None
        self._unsubscribe = self.store.subscribe(self._config_changed)

    # -- configuration --------------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self.store.config

    def update_config(self, mutate: Callable[[Config], None]) -> Config:
        """Change the configuration; it is saved to config.kdl and takes effect immediately."""
        return self.store.update(mutate)

    def _config_changed(self, config: Config) -> None:
        self.providers.reconfigure(config)
        if self._skills is not None:
            self._skills.set_directories(self.skill_directories())

    # -- skills ---------------------------------------------------------------------------------

    def skill_directories(self) -> list[Path]:
        configured = self.config.skills.directory
        return [Path(configured).expanduser() if configured else paths.skills_dir()]

    @property
    def skills(self) -> SkillLibrary:
        if self._skills is None:
            from clankit.skills.library import SkillLibrary

            self._skills = SkillLibrary(self.skill_directories())
        return self._skills

    # -- sessions -------------------------------------------------------------------------------

    def create_session(self, workspace: Path | None = None, *, interactive: bool = True) -> Session:
        workspace = (workspace or Path.cwd()).resolve()
        session_id = uuid.uuid4().hex[:12]
        session = Session(
            self,
            workspace,
            session_id=session_id,
            interactive=interactive,
            log_to=self.session_store.open(session_id, workspace)
            if self.persist_sessions
            else None,
        )
        self.sessions[session.id] = session
        return session

    def resume_session(self, session_id: str, *, interactive: bool = True) -> Session:
        """Reopen a stored session: its transcript and tree are restored, new events append."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        header, events = self.session_store.load(session_id)
        workspace = Path(header.get("workspace") or Path.cwd())
        session = Session(
            self,
            workspace,
            session_id=session_id,
            interactive=interactive,
            log_to=self.session_store.open(session_id, workspace)
            if self.persist_sessions
            else None,
        )
        session.restore(events)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    # -- lifecycle ------------------------------------------------------------------------------

    async def aclose(self) -> None:
        for session in self.sessions.values():
            session.cancel()
            session.close()
        self._unsubscribe()
        await self.providers.aclose()


class RuntimeThread:
    """A `Runtime` living on its own event loop in a background thread.

    Used by the web server. `call(coro_fn)` runs a coroutine on the loop and blocks the calling
    thread for its result; `submit(fn)` runs a plain function on the loop the same way. Never
    touch `runtime` or its sessions from another thread except through these two methods
    (subscribing to a session's `EventBus` is the one exception: it is thread-safe).
    """

    def __init__(self, factory: Callable[[], Runtime] = Runtime) -> None:
        self._factory = factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.runtime: Runtime = None  # type: ignore[assignment]  # set once the thread is up
        self._thread = threading.Thread(target=self._main, name="clankit-runtime", daemon=True)

    def start(self) -> RuntimeThread:
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error
        return self

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            self.runtime = self._factory()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(self.runtime.aclose())
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()

    def call(self, make: Callable[[], Coroutine[Any, Any, T]], timeout: float | None = 30.0) -> T:
        """Run ``make()`` (a coroutine factory) on the runtime loop; block for the result."""
        assert self._loop is not None, "RuntimeThread not started"

        async def runner() -> T:
            return await make()

        future = asyncio.run_coroutine_threadsafe(runner(), self._loop)
        return future.result(timeout)

    def submit(self, fn: Callable[[], T], timeout: float | None = 30.0) -> T:
        """Run a plain function on the runtime loop; block for the result."""
        assert self._loop is not None, "RuntimeThread not started"
        result: concurrent.futures.Future[T] = concurrent.futures.Future()

        def run() -> None:
            try:
                result.set_result(fn())
            except BaseException as exc:
                result.set_exception(exc)

        self._loop.call_soon_threadsafe(run)
        return result.result(timeout)

    def stop(self) -> None:
        if self._loop is not None and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
