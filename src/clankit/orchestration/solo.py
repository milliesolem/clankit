"""Solo: one agent with every tool, talking to the user.

The baseline every other method has to beat. On current frontier models a single agent with
a linear history matches or beats multi-agent arrangements on most coding tasks that fit in
one context window, at a fraction of the cost; reach for something else when the work is
larger than one context, splits into genuinely independent parts, or needs independent
verification.
"""

from __future__ import annotations

from clankit.agent import prompts
from clankit.agent.loop import Agent
from clankit.config.schema import Setting
from clankit.orchestration.base import AskUserTool, Orchestrator
from clankit.orchestration.common import run_diff
from clankit.tools import standard_tools

SELF_REVIEW = """\
[Harness check before your answer is shown to the user]
These are the changes made to files during this task:

<diff>
{diff}
</diff>

Check your own work against the request once more: is every requested behaviour there, did \
anything unrequested slip in (stray debug output, scratch files, unrelated edits, changed or \
weakened tests), and does the verification you ran actually cover the change? If something \
is off, fix it now. Then give your final summary for the user."""


class SoloOrchestrator(Orchestrator):
    name = "solo"
    title = "Solo"
    summary = "One agent does the whole task in a single conversation."
    guidance = (
        "The fastest and cheapest method, and on strong models usually the most reliable for "
        "anything that fits in one context window. It has no independent check on its own "
        "work beyond a self-review of the final diff."
    )
    settings = (
        Setting(
            "self-review",
            "bool",
            True,
            "After a task that changed files, show the agent its own diff and have it check "
            "the work once before answering.",
        ),
    )

    def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
        super().__init__(session)
        self._root = session.new_node("Conversation", role="root")

    def _ensure_agent(self) -> Agent:
        if self.root_agent is None:
            self.root_agent = self.session.agent(
                self._root,
                system=prompts.system_prompt(self.session, prompts.USER_FACING),
                tools=[*standard_tools(), AskUserTool(self.session)],
            )
        return self.root_agent

    async def run(self, prompt: str) -> str:
        session = self.session
        fresh = self.root_agent is None
        agent = self._ensure_agent()
        if fresh and (recap := session.recap()):
            prompt = f"{recap}\n\nThe user now says:\n{prompt}"

        session.update_node(self._root, status="running")
        result = await agent.run(prompt)
        if result.stop == "end_turn" and self.setting("self-review"):
            diff = run_diff(session)
            if diff:
                result = await agent.run(SELF_REVIEW.format(diff=diff))
        if result.stop == "refusal":
            session.update_node(self._root, status="failed")
            return f"The model declined this request. {result.detail}".strip()
        session.update_node(self._root, status="done")
        return result.text
