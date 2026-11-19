"""Context-window housekeeping for the agent loop: estimates, truncation, compaction text."""

from __future__ import annotations

import random
import re

from clankit.llm.types import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock

MAX_RETRIES = 4
CHARS_PER_TOKEN = 3.5

#: Asked of the model when its context is nearly full. The closing sentence matters: the
#: request still carries the agent's tools, and without it models sometimes call one.
SUMMARY_PROMPT = """\
Your context window is nearly full, so this conversation will continue in a fresh one that \
starts from a summary. Write that summary now, inside <summary></summary> tags.

Include what is needed to continue without redoing work or being told the constraints again. \
Be sure to preserve:
1. The task exactly as given, including every acceptance criterion and constraint.
2. Exactly where things stand: what is finished, what is in progress, what remains.
3. Every file created or changed, with its path and what changed in it.
4. Decisions made and approaches tried, including ones that failed and why.
5. Problems encountered and how they were resolved, or that they are still open.
6. Details that would be hard to reconstruct: names, signatures, commands and their results, \
error messages, numbers - quoted exactly.
7. What you were about to do next.

Be complete on these points even at the cost of length; keep everything else brief. \
Do not call any tools while writing this summary; respond with text only."""

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: about 2s, 4s, 8s, 16s."""
    return min(60.0, 2.0**attempt) + random.uniform(0, 1)


def estimate_tokens(system: str, messages: list[Message]) -> int:
    """Rough token count for when a provider reports no usage."""
    chars = len(system)
    for message in messages:
        for block in message.content:
            if isinstance(block, (TextBlock, ThinkingBlock)):
                chars += len(block.text)
            elif isinstance(block, ToolUseBlock):
                chars += len(block.name) + len(str(block.input))
            elif isinstance(block, ToolResultBlock):
                chars += len(block.content)
    return int(chars / CHARS_PER_TOKEN)


def elide_middle(text: str, limit: int) -> str:
    """Keep the head and tail of `text` within `limit` characters, noting what was cut."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    cut = len(text) - head - tail
    return (
        f"{text[:head]}\n\n[... {cut:,} characters omitted. Narrow the request "
        f"(a line range, a more specific pattern, a filter) to see them ...]\n\n{text[-tail:]}"
    )


def render_transcript(messages: list[Message], *, budget: int) -> str:
    """Plain-text rendering of a conversation, shortened to about `budget` characters.

    Individual tool results are clipped first; if the whole is still too long the middle of
    the conversation is dropped, keeping the opening (the task) and the most recent work.
    """
    parts: list[str] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(f"[{message.role}]\n{block.text}")
            elif isinstance(block, ToolUseBlock):
                parts.append(
                    f"[assistant called {block.name}]\n{elide_middle(str(block.input), 2000)}"
                )
            elif isinstance(block, ToolResultBlock):
                label = "tool error" if block.is_error else "tool result"
                parts.append(f"[{label}]\n{elide_middle(block.content, 2000)}")
    text = "\n\n".join(parts)
    if len(text) <= budget:
        return text
    head = int(budget * 0.25)
    tail = budget - head
    return f"{text[:head]}\n\n[... earlier work omitted ...]\n\n{text[-tail:]}"


def extract_summary(text: str) -> str:
    match = _SUMMARY_RE.search(text)
    return (match.group(1) if match else text).strip()


def continuation_message(task: str, summary: str) -> str:
    """The first message of the fresh conversation that follows a compaction."""
    return (
        "You are continuing a task. The earlier part of this conversation was replaced by the "
        "summary below because the context window filled up; files on disk are as you left them.\n\n"
        f"<original_task>\n{task}\n</original_task>\n\n"
        f"<summary_of_work_so_far>\n{summary}\n</summary_of_work_so_far>\n\n"
        "Continue from where you left off. Re-read any file before editing it."
    )
