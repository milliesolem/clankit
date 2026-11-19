"""Shared fixtures.

Every test runs with CLANKIT_CONFIG_DIR and CLANKIT_STATE_DIR pointed into a temp directory,
so no test can read or write the real ~/.config/clankit.

The usual way to test engine behaviour without a network:

    def test_something(session, scripted):
        scripted.push(call("read_file", path="a.txt"), "done")
        answer = await session.run("do it")

`scripted` is a `ScriptedProvider` registered as provider "scripted"; the `runtime` fixture's
default model is "scripted/test-model" and its permission mode is "auto".
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from clankit.agent.events import Event
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.config.store import ConfigStore
from clankit.llm.scripted import ScriptedProvider
from clankit.runtime import Runtime


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLANKIT_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("CLANKIT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CLANKIT_NO_UPDATE", "1")
    for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.models.default = "scripted/test-model"
    cfg.permissions.mode = "auto"
    return cfg


@pytest.fixture
def scripted() -> ScriptedProvider:
    return ScriptedProvider(provider_id="scripted")


@pytest.fixture
def runtime(config: Config, scripted: ScriptedProvider) -> Iterator[Runtime]:
    rt = Runtime(ConfigStore.in_memory(config), persist_sessions=False)
    rt.providers.register(scripted)
    yield rt


@pytest.fixture
def session(runtime: Runtime, workspace: Path) -> Session:
    return runtime.create_session(workspace)


@pytest.fixture
def events(session: Session) -> list[Event]:
    """Every event the session publishes, in order."""
    seen: list[Event] = []
    session.bus.subscribe(seen.append)
    return seen
