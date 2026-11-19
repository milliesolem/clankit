"""Quiet self-updates: check PyPI at most once a day and install the newest release.

The check runs on a background thread from the command line entry point, so starting up never
waits on the network. Only a regular installation is updated - a checkout, an editable
install or a `uv sync` tree belongs to the person working on it. Everything here is
best-effort: no failure of checking, comparing or installing may break a run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from threading import Thread

from clankit import __version__, paths

#: The JSON API needs no credentials and answers faster than the index.
PYPI_JSON = "https://pypi.org/pypi/clankit/json"
#: Smallest pause between two checks, in seconds.
CHECK_INTERVAL = 24 * 3600
#: Seconds a check may take.
TIMEOUT = 5.0
#: Seconds an install may take.
INSTALL_TIMEOUT = 120.0

#: Set (by tests) to keep the updater from ever running.
KILL_SWITCH = "CLANKIT_NO_UPDATE"


def maybe_update() -> None:
    """Check for a newer release and install it, quietly, on a background thread."""
    if os.environ.get(KILL_SWITCH) or _is_dev_install():
        return
    Thread(target=_run, name="clankit-update", daemon=True).start()


def _run() -> None:
    try:
        if not _enabled() or not _due():
            return
        _mark_checked()
        latest = _latest_version()
        if latest is None:
            return
        if _version_key(latest) <= _version_key(__version__):
            return
        if _install(latest):
            print(
                f"clankit: installed version {latest} (takes effect next launch)",
                file=sys.stderr,
            )
    except Exception:
        pass  # the update must never break a run


def _enabled() -> bool:
    """The configuration's `updates.check`; on by default when the file cannot be read."""
    from clankit.config.store import ConfigStore

    return ConfigStore.open().config.updates.check


def _due() -> bool:
    """Has the check interval passed since the last check? (Never checked: yes.)"""
    try:
        stamp = float(_stamp_file().read_text())
        return time.time() - stamp >= CHECK_INTERVAL
    except (OSError, ValueError):
        return True


def _mark_checked() -> None:
    try:
        stamp = _stamp_file()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(time.time()))
    except OSError:
        pass


def _stamp_file() -> Path:
    return paths.state_dir() / "update-check.txt"


def _latest_version() -> str | None:
    """The newest released version, or None when PyPI cannot be asked."""
    request = urllib.request.Request(
        PYPI_JSON, headers={"User-Agent": f"clankit/{__version__}"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = json.load(response)
    version = payload.get("info", {}).get("version")
    return str(version) if version else None


def _install(latest: str) -> bool:
    """Upgrade this installation to `latest`. True when an installer reported success."""
    return _run_pip(latest) or _run_uv()


def _run_pip(latest: str) -> bool:
    """`pip install --upgrade --no-deps` from this interpreter - the usual case.

    `--no-deps` replaces only clankit itself, so a self-update is quick and cannot drag
    other packages along.
    """
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "--upgrade",
        "--no-deps",
        f"clankit=={latest}",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=INSTALL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _run_uv() -> bool:
    """uv-managed tool environments usually lack pip; `uv tool upgrade` covers them."""
    prefix = Path(sys.prefix).parts
    if "uv" not in prefix or "tools" not in prefix:
        return False
    uv = shutil.which("uv")
    if not uv:
        return False
    try:
        result = subprocess.run(
            [uv, "tool", "upgrade", "clankit"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=INSTALL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _is_dev_install() -> bool:
    """True when clankit runs from its own source tree rather than an installed copy."""
    root = Path(__file__).resolve()
    for _ in range(4):  # clankit/updater.py -> the repository root is a few levels up
        root = root.parent
        if (root / "pyproject.toml").is_file() and (root / "src" / "clankit").is_dir():
            return True
    return False


def _version_key(version: str) -> tuple[int, ...]:
    """A version as a comparable tuple. Handles the plain ``0.1.7`` style clankit uses; a
    pre-release segment (``1.0rc1``) counts as smaller than the release."""
    try:
        from packaging.version import Version
    except ImportError:
        return tuple(int(d) for d in re.findall(r"\d+", version))
    parsed = Version(version)
    return (parsed.major, parsed.minor, parsed.micro, 0 if parsed.is_prerelease else 1)
