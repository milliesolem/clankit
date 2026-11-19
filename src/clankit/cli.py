"""Command line entry point.

clankit                      terminal UI in the current directory
clankit web                  web UI on http://127.0.0.1:8420
clankit run "prompt"         run one prompt without a UI and print the answer
clankit sessions             list stored sessions for this directory
clankit skills               list installed skills
clankit config               show where the configuration lives and what is set up
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from clankit import __version__, paths
from clankit.config.schema import PERMISSION_MODES, ConfigError


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-C", "--cwd", metavar="DIR", help="work in DIR instead of the current directory"
    )
    common.add_argument("-m", "--model", metavar="PROVIDER/MODEL", help="model for this session")
    common.add_argument("--method", metavar="NAME", help="orchestration method for this session")
    common.add_argument(
        "--permission-mode", choices=PERMISSION_MODES, help="permission mode for this session"
    )
    common.add_argument(
        "-c",
        "--continue",
        dest="resume_last",
        action="store_true",
        help="continue the most recent session in this directory",
    )
    common.add_argument("-r", "--resume", metavar="ID", help="resume a stored session by id")
    common.add_argument(
        "--demo", action="store_true", help="play a scripted session (no model, no API key needed)"
    )

    parser = argparse.ArgumentParser(
        prog="clankit",
        description="A coding agent that grows a task tree. Terminal and web UIs, any model.",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"clankit {__version__}")
    sub = parser.add_subparsers(dest="command")

    web = sub.add_parser("web", parents=[common], help="start the web UI")
    web.add_argument("--host", help="address to bind (default from config: 127.0.0.1)")
    web.add_argument("--port", type=int, help="port to bind (default from config: 8420)")
    web.add_argument("--no-browser", action="store_true", help="do not open a browser window")

    run = sub.add_parser("run", parents=[common], help="run one prompt without a UI")
    run.add_argument(
        "prompt", nargs="?", help="the prompt; read from standard input when omitted or '-'"
    )
    run.add_argument("--json", action="store_true", help="print every event as a JSON line")
    run.add_argument("-q", "--quiet", action="store_true", help="print only the final answer")

    sub.add_parser("sessions", parents=[common], help="list stored sessions for this directory")
    sub.add_parser("skills", help="list installed skills")
    sub.add_parser("config", help="show configuration status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "config":
            return _config()
        if args.command == "skills":
            return _skills()
        if args.command == "sessions":
            return _sessions(args)
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "web":
            return _web(args)
        return _tui(args)
    except ConfigError as exc:
        print(f"clankit: configuration error in {paths.config_file()}:\n  {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


# -- shared -----------------------------------------------------------------------------------


def _workspace(args: argparse.Namespace) -> Path:
    workspace = Path(args.cwd).expanduser() if getattr(args, "cwd", None) else Path.cwd()
    if not workspace.is_dir():
        raise SystemExit(f"clankit: not a directory: {workspace}")
    return workspace.resolve()


def _runtime() -> Any:
    from clankit.runtime import Runtime
    from clankit.skills import forge

    runtime = Runtime()
    forge.install_hook(runtime)
    return runtime


def _open_session(runtime: Any, args: argparse.Namespace, *, interactive: bool) -> Any:
    """Create or resume the session the command line asks for and apply per-run overrides."""
    from clankit.llm.types import ModelRef

    workspace = _workspace(args)
    session_id = args.resume
    if not session_id and args.resume_last:
        stored = runtime.session_store.list(workspace)
        if not stored:
            raise SystemExit(f"clankit: no stored session to continue in {workspace}")
        session_id = stored[0].id
    if session_id:
        try:
            session = runtime.resume_session(session_id, interactive=interactive)
        except FileNotFoundError:
            raise SystemExit(f"clankit: no stored session {session_id!r}") from None
    else:
        session = runtime.create_session(workspace, interactive=interactive)

    if args.demo:
        session.method_override = "demo"
    elif args.method:
        from clankit.orchestration import method_names

        if args.method not in method_names():
            raise SystemExit(
                f"clankit: unknown method {args.method!r} (available: {', '.join(method_names())})"
            )
        session.method_override = args.method
    if args.model:
        try:
            session.model_override = ModelRef.parse(args.model)
        except ValueError as exc:
            raise SystemExit(f"clankit: {exc}") from None
    if args.permission_mode:
        # A session-only override: the configuration file is not touched.
        runtime.config.permissions.mode = args.permission_mode
    return session


# -- commands ---------------------------------------------------------------------------------


def _tui(args: argparse.Namespace) -> int:
    from clankit.tui import run_tui

    runtime = _runtime()
    session = _open_session(runtime, args, interactive=True)
    run_tui(runtime, session)
    return 0


def _web(args: argparse.Namespace) -> int:
    from clankit.web import serve

    workspace = _workspace(args)
    serve(
        _runtime,
        workspace=workspace,
        host=args.host,
        port=args.port,
        open_browser=False if args.no_browser else None,
        demo=args.demo,
    )
    return 0


async def _run(args: argparse.Namespace) -> int:
    from clankit.headless import run_headless

    prompt = args.prompt
    if prompt is None or prompt == "-":
        prompt = sys.stdin.read()
    if not prompt.strip() and not args.demo:
        raise SystemExit("clankit run: no prompt given")
    runtime = _runtime()
    session = _open_session(runtime, args, interactive=False)
    try:
        return await run_headless(session, prompt, as_json=args.json, quiet=args.quiet)
    finally:
        await runtime.aclose()


def _sessions(args: argparse.Namespace) -> int:
    import datetime

    from clankit.agent.store import SessionStore

    stored = SessionStore().list(_workspace(args))
    if not stored:
        print("No stored sessions for this directory.")
        return 0
    for meta in stored:
        when = datetime.datetime.fromtimestamp(meta.updated).strftime("%Y-%m-%d %H:%M")
        print(f"{meta.id}  {when}  {meta.turns:>3} turns  {meta.title}")
    print("\nResume one with: clankit --resume <id>   (or clankit --continue for the newest)")
    return 0


def _skills() -> int:
    runtime = _runtime()
    library = runtime.skills
    for skill in library.skills:
        print(f"{skill.name}\n    {skill.description}\n    {skill.path}")
    pending = library.pending()
    if pending:
        print(
            f"\n{len(pending)} proposed skill(s) await review: "
            + ", ".join(s.name for s in pending)
        )
    if not library.skills and not pending:
        print(f"No skills installed. Skills live in {library.directories[0]}")
    for problem in library.diagnostics:
        print(f"warning: {problem}", file=sys.stderr)
    return 0


def _config() -> int:
    from clankit.config.catalog import effective_providers, key_source
    from clankit.config.store import ConfigStore

    store = ConfigStore.open()
    config = store.config
    print(f"Configuration file: {store.path}")
    print(f"Skills directory:   {paths.skills_dir()}")
    print(f"Session logs:       {paths.sessions_dir()}")
    print(f"\nDefault model:        {config.models.default}")
    print(f"Orchestration method: {config.orchestration.method}")
    print(f"Permission mode:      {config.permissions.mode}")
    print(f"Theme:                {config.ui.theme}")
    print("\nProviders:")
    for provider_id, provider in effective_providers(config).items():
        source = key_source(config, provider_id)
        status = {
            "config": "key in config file",
            "env": f"key from ${provider.api_key_env}",
            "keyless": "no key needed" + ("" if provider.enabled else " (not enabled)"),
            "none": "not set up",
        }[source]
        print(f"  {provider_id:<14} {status}")
    for warning in store.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
