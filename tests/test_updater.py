"""The quiet self-updater: versions, cadence, dev-tree and kill-switch guards, installs."""

from __future__ import annotations

import time

import pytest

from clankit import __version__, paths, updater


class _FakeThread:
    """Stands in for threading.Thread: records its arguments, never runs anything."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def start(self) -> None:
        pass


def _spawned(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    spawned: list[dict] = []
    monkeypatch.setattr(
        updater, "Thread", lambda **kw: spawned.append(kw) or _FakeThread(**kw)
    )
    return spawned


def test_version_keys_order_naturally() -> None:
    assert updater._version_key("0.1.6") < updater._version_key("0.1.10")
    assert updater._version_key("0.1.10") < updater._version_key("0.2.0")
    assert updater._version_key("1.0.0rc1") < updater._version_key("1.0.0")


def test_maybe_update_stays_quiet_in_a_dev_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = _spawned(monkeypatch)
    monkeypatch.setattr(updater, "_is_dev_install", lambda: True)
    updater.maybe_update()
    assert spawned == []


def test_maybe_update_stays_quiet_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = _spawned(monkeypatch)
    monkeypatch.setattr(updater, "_is_dev_install", lambda: False)
    monkeypatch.setenv(updater.KILL_SWITCH, "1")
    updater.maybe_update()
    assert spawned == []


def test_maybe_update_starts_a_daemon_thread_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = _spawned(monkeypatch)
    monkeypatch.setattr(updater, "_is_dev_install", lambda: False)
    monkeypatch.delenv(updater.KILL_SWITCH, raising=False)
    updater.maybe_update()
    assert len(spawned) == 1
    assert spawned[0]["daemon"] is True and spawned[0]["name"] == "clankit-update"


def test_run_installs_a_newer_release_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pip_calls: list[str] = []
    monkeypatch.setattr(updater, "_enabled", lambda: True)
    monkeypatch.setattr(updater, "_latest_version", lambda: "9.9.9")

    def fake_pip(latest: str) -> bool:
        pip_calls.append(latest)
        return True

    monkeypatch.setattr(updater, "_run_pip", fake_pip)
    updater._run()
    assert pip_calls == ["9.9.9"]
    assert "clankit: installed version 9.9.9" in capsys.readouterr().err
    assert (paths.state_dir() / "update-check.txt").exists()


def test_run_does_nothing_when_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[str] = []
    monkeypatch.setattr(updater, "_enabled", lambda: True)
    monkeypatch.setattr(updater, "_latest_version", lambda: __version__)
    monkeypatch.setattr(updater, "_install", lambda latest: installed.append(latest) or True)
    updater._run()
    assert installed == []


def test_run_checks_at_most_once_a_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stamp = paths.state_dir() / "update-check.txt"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(str(time.time()))
    checked: list[bool] = []
    monkeypatch.setattr(updater, "_enabled", lambda: True)
    monkeypatch.setattr(
        updater, "_latest_version", lambda: checked.append(True) or __version__
    )
    updater._run()
    assert checked == []


def test_run_swallows_every_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updater, "_enabled", lambda: (_ for _ in ()).throw(OSError("boom")))
    updater._run()  # must not raise


def test_run_skips_when_disabled_in_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[bool] = []
    monkeypatch.setattr(updater, "_enabled", lambda: False)
    monkeypatch.setattr(
        updater, "_latest_version", lambda: checked.append(True) or __version__
    )
    updater._run()
    assert checked == []


def test_dev_install_detection() -> None:
    assert updater._is_dev_install()  # the tests run from the source tree
