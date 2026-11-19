"""The terminal UI's pure parts: formats, the transcript model, the pump, row layout."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.cells import cell_len

from clankit.agent.events import (
    AssistantMessage,
    Event,
    EventBus,
    NodeAdded,
    NodeInfo,
    PermissionRequested,
    PermissionResolved,
    QuestionAnswered,
    QuestionAsked,
    RunFinished,
    RunStarted,
    SkillSaved,
    UserMessage,
)
from clankit.agent.state import Item, SessionView
from clankit.config.schema import Config, ConfigError, Setting
from clankit.tui.entries import Entry, History, node_entries, root_entries
from clankit.tui.format import (
    ellipsize,
    ellipsize_left,
    human_duration,
    human_elapsed,
    human_tokens,
    model_alias,
    pad,
    wrap_cells,
)
from clankit.tui.pump import EventPump
from clankit.tui.requests import looks_like_diff, node_path, pending_requests
from clankit.tui.settings import display_path, set_method, setting_saver
from clankit.tui.tasktree import MIN_LABEL, Row, indent_for, render_row
from clankit.tui.transcript import Block, Context, TextBlock, clean_markdown, diff_stat


@pytest.mark.parametrize(
    ("count", "text"),
    [(0, "-"), (830, "830"), (41_234, "41.2k"), (412_345, "412k"), (1_250_000, "1.25M")],
)
def test_human_tokens_fit_five_cells(count: int, text: str) -> None:
    assert human_tokens(count) == text and len(text) <= 5


@pytest.mark.parametrize(("seconds", "text"), [(4, "0:04"), (751, "12:31"), (3725, "1h02")])
def test_human_elapsed(seconds: float, text: str) -> None:
    assert human_elapsed(seconds) == text


def test_human_duration_scales_its_unit() -> None:
    assert human_duration(0.4) == "0.4s"
    assert human_duration(12.4) == "12s"
    assert human_duration(75) == "1:15"


def test_model_alias_drops_provider_and_vendor_prefix() -> None:
    assert model_alias("anthropic/claude-sonnet-5-5") == "sonnet-5-5"
    assert model_alias("openrouter/qwen/qwen3-coder:free") == "qwen3-coder:free"
    assert model_alias("deepseek/deepseek-v4-pro") == "deepseek-v4-pro"
    assert model_alias("") == ""


def test_ellipsizing_respects_cell_widths() -> None:
    assert ellipsize("short", 10) == "short"
    assert ellipsize("a long sentence", 8) == "a long…"
    assert cell_len(ellipsize("日本語のテキスト", 7)) <= 7
    assert ellipsize_left("/very/long/path/to/file.py", 12) == "…/to/file.py"
    assert pad("ab", 4) == "ab  " and pad("ab", 4, right=True) == "  ab"
    assert wrap_cells("abcdefgh", 3) == ["abc", "def", "gh"] and wrap_cells("", 3) == [""]


# -- transcript model -------------------------------------------------------------------------


def fold(events: list[Event]) -> tuple[SessionView, History]:
    view, history = SessionView(), History()
    for seq, event in enumerate(events, start=1):
        event.seq = seq
        event.ts = 1000.0 + seq
        view.apply(event)
        history.observe(event)
    return view, history


def root_node() -> NodeAdded:
    return NodeAdded(node=NodeInfo(id="n1", parent_id=None, depth=0, title="Conversation"))


def test_root_entries_show_the_prompt_once_and_mark_the_run_end() -> None:
    view, history = fold(
        [
            RunStarted(run_id="r1", prompt="fix the bug", method="solo", model="m"),
            root_node(),
            UserMessage(
                node_id="n1", text="Earlier in this session: ...\nThe user now says:\nfix the bug"
            ),
            AssistantMessage(node_id="n1", message_id="a", text="Fixed.", thinking="hmm"),
            UserMessage(node_id="n1", text="[review findings] one more thing"),
            RunFinished(run_id="r1", status="completed", answer="Fixed."),
        ]
    )
    kinds = [entry.kind for entry in root_entries(view, history, show_thinking=True)]
    assert kinds == ["prompt", "thinking", "assistant", "message", "run-end"]
    kinds = [entry.kind for entry in root_entries(view, history, show_thinking=False)]
    assert "thinking" not in kinds


def test_restored_history_without_runs_shows_user_items_as_prompts() -> None:
    view, _ = fold([root_node(), UserMessage(node_id="n1", text="hello")])
    entries = root_entries(view, History(), show_thinking=True)
    assert [e.kind for e in entries] == ["prompt"]


def test_answers_and_decisions_become_quiet_records() -> None:
    _, history = fold(
        [
            root_node(),
            PermissionRequested(
                request_id="p1", node_id="n1", tool="bash", category="shell", summary="$ ls"
            ),
            PermissionResolved(request_id="p1", choice="deny"),
            QuestionAsked(request_id="q1", node_id="n1", question="Which one?", options=["a"]),
            QuestionAnswered(request_id="q1", answer="a"),
            SkillSaved(name="tidy", path="/x", action="proposed", description="Tidy up."),
        ]
    )
    records = [(e.kind, e.text, e.detail, e.level) for e in history.local]
    assert records[0] == ("resolution", "denied", "$ ls", "error")
    assert records[1][:3] == ("resolution", "answered", "a")
    assert history.local[1].context == "Which one?"
    assert records[2] == ("skill", "tidy", "Tidy up.", "proposed")


def test_local_entries_sort_after_what_came_before() -> None:
    view, history = fold(
        [RunStarted(run_id="r1", prompt="go", method="demo", model=""), root_node()]
    )
    note = history.add("message", "Model set.")
    entries = root_entries(view, history, show_thinking=True)
    assert entries[-1].local is note
    assert history.current_run() is not None


def test_node_entries_keep_the_agents_order() -> None:
    view, _ = fold(
        [
            root_node(),
            NodeAdded(node=NodeInfo(id="n2", parent_id="n1", depth=1, title="Child")),
            UserMessage(node_id="n2", text="brief"),
            AssistantMessage(node_id="n2", message_id="m", text="done"),
        ]
    )
    assert [e.kind for e in node_entries(view, "n2", show_thinking=True)] == [
        "message",
        "assistant",
    ]
    assert node_entries(view, "missing", show_thinking=True) == []
    assert node_path(view, "n2") == "Conversation / Child"


def test_pending_requests_are_oldest_first() -> None:
    view, _ = fold(
        [
            root_node(),
            QuestionAsked(request_id="q1", node_id="n1", question="?"),
            PermissionRequested(
                request_id="p1", node_id="n1", tool="bash", category="shell", summary="$ ls"
            ),
        ]
    )
    assert [r.request_id for r in pending_requests(view)] == ["q1", "p1"]


def test_diff_helpers() -> None:
    diff = "--- a/x\n+++ b/x\n@@ -1 +1,2 @@\n-old\n+new\n+more\n"
    assert diff_stat(diff) == (2, 1)
    assert looks_like_diff(diff) and not looks_like_diff("rm -rf build")
    assert clean_markdown("see ![plot](p.png)") == "see [plot](p.png)"


# -- transcript blocks ------------------------------------------------------------------------


async def test_plain_block_syncs_and_shows_nothing() -> None:
    entry = Entry(key="k", kind="unknown", ts=0.0, seq=0.0)
    block = Block(entry)
    assert block.signature(entry, Context(now=0.0)) is None
    await block.sync(entry, Context(now=0.0))  # must not raise


async def test_text_block_shows_the_entry_text_by_default() -> None:
    item = Item(id="i", kind="notice", seq=1, ts=0.0, text="hello")
    entry = Entry(key="k", kind="unknown", ts=0.0, seq=0.0, item=item)
    block = TextBlock(entry)
    await block.sync(entry, Context(now=0.0))
    assert block.render().plain == "hello"
    item.text = "updated"
    await block.sync(entry, Context(now=0.0))
    assert block.render().plain == "updated"


# -- pump -------------------------------------------------------------------------------------


async def test_pump_batches_events_at_a_bounded_rate() -> None:
    batches: list[list[Event]] = []
    pump = EventPump(batches.append, interval=0.02)
    bus = EventBus()
    pump.attach(bus)
    for _ in range(50):
        bus.publish(UserMessage(node_id="n", text="x"))
    assert batches == [] and not pump.idle
    await asyncio.sleep(0.05)
    assert len(batches) == 1 and len(batches[0]) == 50 and pump.idle

    bus.publish(UserMessage(node_id="n", text="y"))
    pump.detach()
    await asyncio.sleep(0.05)
    assert len(batches) == 1, "a detached pump delivers nothing more"
    bus.publish(UserMessage(node_id="n", text="z"))
    assert pump.idle


async def test_pump_flush_delivers_immediately() -> None:
    batches: list[list[Event]] = []
    pump = EventPump(batches.append, interval=10)
    bus = EventBus()
    pump.attach(bus)
    bus.publish(UserMessage(node_id="n", text="x"))
    pump.flush()
    assert len(batches) == 1 and pump.idle


# -- tree rows --------------------------------------------------------------------------------


def make_row(depth: int, title: str = "A task with a long descriptive title") -> Row:
    return Row(
        node_id="n",
        depth=depth,
        title=title,
        state="running",
        model="sonnet-5-5",
        model_inherited=True,
        tokens=41_200,
        elapsed="3:12",
        caret="",
        hidden=0,
    )


@pytest.mark.parametrize("width", [40, 44, 54, 62])
@pytest.mark.parametrize("depth", [0, 1, 5, 10])
def test_rows_fit_their_width_and_keep_labels_readable(width: int, depth: int) -> None:
    content = render_row(make_row(depth), width)
    assert content.cell_length == width
    label = content.plain[content.plain.index("A task") :].split("…")[0]
    assert len(label) >= MIN_LABEL - 1


def test_indentation_is_capped_at_depth() -> None:
    assert indent_for(0, 30) == 0
    assert indent_for(1, 30) == 2
    assert indent_for(4, 30) == 8
    assert indent_for(6, 30) == 10
    assert indent_for(10, 22) == 22 - MIN_LABEL


def test_collapsed_rows_show_how_much_they_hide() -> None:
    row = Row("n", 1, "Writer", "done", "haiku-5-5", False, 27_000, "1:02", "▸", 3)
    text = render_row(row, 54).plain
    assert "Writer (+3)" in text and "haiku-5-5" in text and "27.0k" in text and "1:02" in text


# -- settings helpers -------------------------------------------------------------------------


def test_display_path_uses_the_home_shorthand() -> None:
    assert display_path(Path.home() / ".config" / "clankit" / "config.kdl") == (
        "~/.config/clankit/config.kdl"
    )
    assert display_path(Path("/etc/clankit.kdl")) == "/etc/clankit.kdl"
    assert "no configuration file" in display_path(None)


def test_orchestration_savers_validate() -> None:
    config = Config()
    set_method(config, "solo")
    assert config.orchestration.method == "solo"
    with pytest.raises(ConfigError):
        set_method(config, "no-such-method")
    save = setting_saver("solo", Setting("reviewer", "model", "", "Model for review."))
    with pytest.raises(ConfigError):
        save(config, "not a reference")
    save(config, "anthropic/claude-haiku-5-5")
    assert config.orchestration.option("solo", "reviewer") == "anthropic/claude-haiku-5-5"
