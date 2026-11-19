"""The orchestration interface.

An orchestrator decides how a user prompt turns into agent work: one agent, a recursive tree
of agents, competing candidates, a planner and an executor, and so on. It is built from the
session's primitives:

* ``session.new_node(...)`` / ``session.update_node(...)`` put work on the visible task tree;
* ``session.agent(node, system=..., tools=...)`` creates the agent that works on a node;
* ``await agent.run(prompt)`` runs it and returns an `AgentResult`;
* ``session.model_for(role)`` resolves the model configured for a role;
* ``self.setting(key)`` reads the method's own user-adjustable settings.

One orchestrator instance serves a session for as long as that method stays selected, so it
may keep state (typically its root agent, and with it the conversation) between prompts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from clankit.config.schema import ConfigError, Setting
from clankit.tools.base import Tool, ToolContext, ToolOutput

if TYPE_CHECKING:
    from clankit.agent.loop import Agent
    from clankit.agent.session import Session


class Orchestrator(ABC):
    #: Identifier used in the configuration and on the command line, e.g. "task-tree".
    name: ClassVar[str] = ""
    #: Display name, e.g. "Task Tree".
    title: ClassVar[str] = ""
    #: One sentence for pickers: what the method does.
    summary: ClassVar[str] = ""
    #: A short paragraph: when it helps, when it does not, what it costs.
    guidance: ClassVar[str] = ""
    #: False for methods that never call a model, so they run with no provider configured.
    needs_model: ClassVar[bool] = True
    #: User-adjustable settings, stored under ``orchestration { <name> { ... } }``.
    settings: ClassVar[tuple[Setting, ...]] = ()

    def __init__(self, session: Session) -> None:
        self.session = session
        #: The agent that talks to the user, once it exists. Methods that keep a persistent
        #: conversation set this; steering and manual compaction act on it.
        self.root_agent: Agent | None = None

    def setting(self, key: str) -> Any:
        """Current value of one of this method's settings (validated, default if unset)."""
        spec = next((s for s in self.settings if s.key == key), None)
        if spec is None:
            raise KeyError(f"{self.name} has no setting {key!r}")
        raw = self.session.config.orchestration.option(self.name, key)
        try:
            return spec.coerce(raw)
        except ConfigError:
            return spec.default

    @abstractmethod
    async def run(self, prompt: str) -> str:
        """Handle one user prompt and return the answer to show the user.

        Raise `ProviderError` for model failures that end the run. Cancellation arrives as
        `asyncio.CancelledError` and must propagate; the session then marks every unfinished
        node as cancelled.
        """
        raise NotImplementedError

    def steer(self, text: str) -> None:
        """Deliver a mid-run message from the user to the agent that talks to the user."""
        if self.root_agent is not None:
            self.root_agent.steer(text)


class AskUserTool(Tool):
    """Lets the agent that talks to the user ask a question and wait for the answer."""

    name = "ask_user"
    description = (
        "Ask the user a question and wait for the answer. Use this only for a decision that "
        "is genuinely the user's to make and that you cannot settle from the request, the "
        "code, or a sensible default; otherwise proceed and state your assumption. Offer "
        "2-4 concrete options when you can, with your recommendation first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question, self-contained."},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Suggested answers. The user may also answer freely.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }
    read_only = False

    def __init__(self, session: Session) -> None:
        self._session = session

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"ask: {tool_input.get('question', '')[:80]}"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        answer = await self._session.ask_user(
            ctx.node_id, tool_input["question"], tool_input.get("options") or ()
        )
        return ToolOutput(answer or "(The user gave no answer; use your best judgment.)")
