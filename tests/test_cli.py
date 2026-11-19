"""The command line: headless runs, listings and argument handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clankit import cli, paths


@pytest.fixture(autouse=True)
def fast_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLANKIT_DEMO_SPEED", "0")


def test_run_demo_prints_the_answer_and_progress(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["run", "--demo", "--permission-mode", "auto", "-C", str(workspace), "anything"]
    )
    out, err = capsys.readouterr()
    assert code == 0
    assert "keeps comments" in out
    assert "+ Parser with source spans" in err and "= Writer that preserves comments: ok" in err
    assert paths.config_file().exists()  # first run writes the starter configuration


def test_run_json_streams_events(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["run", "--demo", "--json", "--permission-mode", "auto", "-C", str(workspace), "x"]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0
    assert lines[0]["kind"] == "run_started" and lines[-1]["kind"] == "run_finished"
    assert lines[-1]["status"] == "completed"


def test_non_interactive_permission_refusal_is_reported(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["run", "--demo", "-q", "--permission-mode", "ask", "-C", str(workspace), "x"])
    out, err = capsys.readouterr()
    assert code == 0  # the demo carries on; the refused command shows up in its summary
    assert "keeps comments" in out and err == ""


def test_run_without_any_provider_fails_clearly(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["run", "--method", "solo", "-C", str(workspace), "hello"])
    _, err = capsys.readouterr()
    assert code == 1 and "API key" in err


def test_sessions_can_be_listed_and_continued(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["sessions", "-C", str(workspace)]) == 0
    assert "No stored sessions" in capsys.readouterr().out
    cli.main(
        ["run", "--demo", "-q", "--permission-mode", "auto", "-C", str(workspace), "first prompt"]
    )
    capsys.readouterr()
    assert cli.main(["sessions", "-C", str(workspace)]) == 0
    listing = capsys.readouterr().out
    assert "first prompt" in listing and "1 turns" in listing
    code = cli.main(
        [
            "run",
            "--demo",
            "-q",
            "--continue",
            "--permission-mode",
            "auto",
            "-C",
            str(workspace),
            "again",
        ]
    )
    assert code == 0
    capsys.readouterr()
    cli.main(["sessions", "-C", str(workspace)])
    assert "2 turns" in capsys.readouterr().out


def test_bad_arguments_exit_with_a_message(workspace: Path) -> None:
    with pytest.raises(SystemExit, match="unknown method"):
        cli.main(["run", "--method", "nope", "-C", str(workspace), "x"])
    with pytest.raises(SystemExit, match="provider/model"):
        cli.main(["run", "--model", "noslash", "-C", str(workspace), "x"])
    with pytest.raises(SystemExit, match="no stored session"):
        cli.main(["run", "--resume", "deadbeef", "-C", str(workspace), "x"])
    with pytest.raises(SystemExit, match="not a directory"):
        cli.main(["run", "-C", str(workspace / "missing"), "x"])


def test_config_and_skills_listings(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert str(paths.config_file()) in out and "anthropic" in out and "not set up" in out
    assert cli.main(["skills"]) == 0
    assert "No skills installed" in capsys.readouterr().out


def test_malformed_config_is_reported_with_its_location(capsys: pytest.CaptureFixture[str]) -> None:
    paths.config_dir().mkdir(parents=True, exist_ok=True)
    paths.config_file().write_text(
        'model "anthropic/claude-opus-5-5"\nui {\n    show-thinking true\n}\n'
    )
    assert cli.main(["config"]) == 2
    err = capsys.readouterr().err
    assert "configuration error" in err and "#true" in err
