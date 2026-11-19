"""Workflows: short scripts that fan work out to many agents, check what comes back and
combine it.

An agent writes a workflow script (Python with top-level ``await``); the harness checks it,
runs it in-process and hands back the script's ``result`` with an account of what ran. The
pieces:

* `script` - checking and compiling scripts, with line-numbered errors before anything runs;
* `engine` - the primitives scripts call (agent, parallel, pipeline, phase, log, budget,
  workflow) and the limits, node tree and account behind them;
* `worker` - one workflow agent: its role, tools, structured result and token accounting;
* `patterns` - the quality patterns (adversarial verification, judge panels, loop-until-dry
  discovery, completeness critics, fix loops) as tested functions scripts call by name;
* `journal` - reusing finished agents when an edited script runs again;
* `saved` - the built-in workflows and the user's own, by name.

`clankit.orchestration.ultra` is the orchestration method built on this package.
"""

from clankit.orchestration.workflow.account import AgentRecord, Outcome
from clankit.orchestration.workflow.engine import MAX_ITEMS, Limits, WorkflowRun
from clankit.orchestration.workflow.errors import (
    AgentLimitReached,
    BudgetExhausted,
    WorkflowError,
    WorkflowStop,
)
from clankit.orchestration.workflow.journal import Journal, JournalRun
from clankit.orchestration.workflow.saved import (
    Catalog,
    SavedWorkflow,
    find_saved,
    load_saved,
    user_workflow_dir,
)
from clankit.orchestration.workflow.script import Meta, Problem, Script, ScriptError, compile_script

__all__ = [
    "MAX_ITEMS",
    "AgentLimitReached",
    "AgentRecord",
    "BudgetExhausted",
    "Catalog",
    "Journal",
    "JournalRun",
    "Limits",
    "Meta",
    "Outcome",
    "Problem",
    "SavedWorkflow",
    "Script",
    "ScriptError",
    "WorkflowError",
    "WorkflowRun",
    "WorkflowStop",
    "compile_script",
    "find_saved",
    "load_saved",
    "user_workflow_dir",
]
