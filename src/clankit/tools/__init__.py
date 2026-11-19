"""The standard tool set.

`standard_tools()` is what orchestration methods hand to an agent. The order is fixed and
the instances are fresh per call: an agent's tool list is part of its cached prompt prefix
and must never vary between requests.
"""

from __future__ import annotations

from clankit.tools.base import Tool

#: Tools that never change anything; safe for planners, reviewers and judges.
READ_ONLY_TOOLS: tuple[str, ...] = ("read", "glob", "grep", "web_fetch", "skill")
#: Everything a building agent gets.
ALL_TOOLS: tuple[str, ...] = ("read", "write", "edit", "glob", "grep", "bash", "web_fetch", "skill")


def standard_tools(names: tuple[str, ...] | None = None) -> list[Tool]:
    """Fresh instances of the built-in tools, in canonical order.

    `names` selects a subset (unknown names raise KeyError); None means all of them.
    """
    from clankit.skills.tool import SkillTool
    from clankit.tools.files import EditTool, ReadTool, WriteTool
    from clankit.tools.search import GlobTool, GrepTool
    from clankit.tools.shell import BashTool
    from clankit.tools.web import WebFetchTool

    factories: dict[str, type[Tool]] = {
        "read": ReadTool,
        "write": WriteTool,
        "edit": EditTool,
        "glob": GlobTool,
        "grep": GrepTool,
        "bash": BashTool,
        "web_fetch": WebFetchTool,
        "skill": SkillTool,
    }
    if names is not None:
        unknown = [n for n in names if n not in factories]
        if unknown:
            raise KeyError(f"unknown tool(s): {', '.join(unknown)}")
    wanted = set(names) if names is not None else set(factories)
    return [factory() for name, factory in factories.items() if name in wanted]


def read_only_tools() -> list[Tool]:
    return standard_tools(READ_ONLY_TOOLS)
