"""Where Clankit keeps things on disk.

Configuration and skills live under ``~/.config/clankit`` (honouring ``XDG_CONFIG_HOME``);
session logs live under ``~/.local/state/clankit`` (honouring ``XDG_STATE_HOME``).
``CLANKIT_CONFIG_DIR`` and ``CLANKIT_STATE_DIR`` override both, which tests rely on.
"""

from __future__ import annotations

import os
from pathlib import Path

APP = "clankit"


def config_dir() -> Path:
    override = os.environ.get("CLANKIT_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / APP


def config_file() -> Path:
    return config_dir() / "config.kdl"


def skills_dir() -> Path:
    return config_dir() / "skills"


def state_dir() -> Path:
    override = os.environ.get("CLANKIT_STATE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or "~/.local/state"
    return Path(base).expanduser() / APP


def sessions_dir() -> Path:
    return state_dir() / "sessions"
