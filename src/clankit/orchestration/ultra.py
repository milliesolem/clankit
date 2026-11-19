"""Ultra: the agent that talks to the user writes workflow scripts that fan work out.

In the spirit of Claude Code's Ultracode. For every substantive task (or, with activation
"auto", only when breadth or independent verification clearly helps), the user-facing agent
writes a short, deterministic Python script and runs it with the `workflow` tool: the script
starts many agents, has their findings checked by skeptics, and combines what survives.
Where the Task Tree lets each model decide its recursion turn by turn, here a script holds
the loop, so the fan-out, the verification and the synthesis happen exactly as written, and
only the script's result enters the conversation. The scripting runtime, the primitives,
the pattern library and the saved workflows live in `clankit.orchestration.workflow`.

The documented failure of the original is cost: fleets of agents the user never saw
spending millions of tokens. So every workflow is bounded in code, not by advice:

* a token budget per workflow (setting ``workflow-budget``), checked before every agent
  turn, which stops the workflow and hands back what finished;
* a cap on agents per workflow (``max-agents``) and on agents running at once
  (``max-concurrent``, on top of each provider's own concurrency limit);
* a notice to the user before a workflow expected to be large (``large-workflow``);
* the session's run budget, which applies to workflow agents like any other.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from clankit.agent import prompts
from clankit.agent.loop import Agent
from clankit.config.schema import MODEL_ROLES, Setting
from clankit.orchestration.base import AskUserTool, Orchestrator
from clankit.orchestration.common import models_guide
from clankit.orchestration.workflow import (
    MAX_ITEMS,
    Catalog,
    Journal,
    Limits,
    Script,
    ScriptError,
    WorkflowRun,
    compile_script,
    find_saved,
    load_saved,
)
from clankit.tools import standard_tools
from clankit.tools.base import Tool, ToolContext, ToolOutput

if TYPE_CHECKING:
    from clankit.agent.session import Session

# --------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------

ACTIVATION = {
    "always": """\
Plan a workflow for every substantive task: investigating how something works, reviewing, \
researching, designing, or making a change. One request often becomes several workflows in \
a row (understand, then change, then verify), which keeps you in the loop between them. \
Work directly, without a workflow, only for conversation, questions you can answer from \
what you already know, and trivial mechanical edits.""",
    "auto": """\
Use a workflow when it clearly helps: the work splits into many independent pieces \
(reviewing many files, checking many findings, comparing several approaches), its findings \
need independent verification before anyone relies on them, or it needs more breadth than \
one conversation can hold. Otherwise do the work yourself: one agent working step by step \
is cheaper, and on most tasks just as good.""",
}

ROLE = """\
# Working with workflows

You can run workflows with the `workflow` tool: short Python scripts that fan work out to \
other agents, have their findings checked independently, and combine the results. The \
harness runs the script; intermediate results stay in its variables and only its `result` \
comes back to you, so a workflow can cover far more ground than one conversation.

## When to use one

{activation}

## Writing a script

A script is plain Python with top-level `await`. Its first statement is a literal `meta` \
dict, and it hands back its answer by assigning `result`:

```python
meta = {{
    "name": "find-unsafe-sql",
    "description": "Find SQL built from user input and confirm each case.",
    "phases": ["Find", "Verify"],
}}
schema = {{
    "type": "object",
    "properties": {{"issues": {{"type": "array", "items": {{"type": "string"}}}}}},
    "required": ["issues"],
}}
phase("Find")
reports = await parallel([
    agent(f"Read {{path}} and list every SQL query built from user input, as path:line "
          "and one sentence each.", schema=schema, label=f"scan {{path}}", tools="read-only")
    for path in args["files"]
])
issues = [issue for report in reports if report for issue in report["issues"]]
log(f"{{len(issues)}} candidate issues in {{len(args['files'])}} files")
phase("Verify")
result = await adversarial_verify(issues, context="The project uses psycopg.")
```

Defined for you (`import` is not available):
- `await agent(prompt, *, schema=None, model=None, label=None, phase=None, tools="all", \
write_scope=None, effort=None)` runs one agent and returns its final text or, with \
`schema` (a JSON Schema whose root is an object), a dict that matches it; None if the agent \
failed. Every agent starts with an empty context, so put all it needs in the prompt. \
`tools="read-only"` gives read, glob, grep, web_fetch and skill; "all" adds write, edit and \
bash. `write_scope` lists the files or globs it may change (`[]`: it may run commands but \
change no files). `model` is "provider/model" or a role: {roles}. `effort` is low, medium, \
high, xhigh or max.
- `await parallel([...])` runs agent calls (or other awaitables, or functions without \
arguments) at the same time and waits for all of them; a slot that fails becomes None. Use \
it when the next step needs every result.
- `await pipeline(items, stage1, stage2, ...)` sends each item through the stages on its \
own, with no waiting between stages: each stage is called as `stage(previous, item, index)` \
and the first gets the item as `previous`. A stage that fails or returns None ends that \
item, whose slot becomes None. Prefer it to parallel when items do not need each other, and \
pass `phase=` to the agents in each stage.
- `phase(title)` starts a progress group (titles must match `meta["phases"]` when it lists \
them); `log(message)` shows the user a progress line.
- `args` is the object you passed to the tool; `budget.total`, `budget.spent()` and \
`budget.remaining()` count this workflow's tokens.
- `await workflow(name, args)` runs a saved workflow inside this one (one level deep).
- `json.dumps`, `json.loads` and the usual built-ins (len, sorted, min, max, range, ...).

Limits: {max_concurrent} agents run at a time; one workflow may start {max_agents} agents \
and use {budget:,} tokens (output plus prompt tokens not read from the cache); a parallel \
or pipeline call takes at most {max_items:,} items. When a limit is reached the workflow \
stops and you get what finished. The script itself cannot use the clock, randomness, files \
or the network; agents do that work. Nobody can answer questions while a workflow runs: \
ask the user first, or run workflows one after another.

## Patterns

Ready-made, tested functions; prefer them to writing the same logic by hand.
- `await adversarial_verify(claims, votes={votes}, lenses=None, context="")` returns the \
claims that survive: each goes to `votes` agents told to refute it, and survives only if a \
majority uphold it (uncertain counts against it). `lenses` (a list of angles such as \
"correctness", "security", "does it reproduce") gives each skeptic its own angle instead. \
Use it on findings before you report them.
- `await judge_panel(task, angles, judges=3)` returns `{{"best", "index", "angle", \
"scores", "attempts"}}`: one attempt per angle, scored by judges who do not know who wrote \
which. Use it for design questions with several reasonable answers.
- `await loop_until_dry(finder_prompts, key, dry_rounds=2)` returns deduplicated findings: \
every prompt searches in its own way, round after round, until `dry_rounds` rounds in a row \
bring nothing new. `key` is a field name or a function that identifies a finding. Use it \
for "find all ..." tasks.
- `await completeness_critic(report, task="")` returns the gaps a fresh agent sees in a \
report; follow them up in another round.
- `await fix_until(check_command, max_stale=2)` returns True once the command passes: the \
harness runs it, and an agent fixes the failures until it passes or stops making progress.

## Saved workflows

Run one with the tool's `name` and `args`, or from a script with `workflow(name, args)`:
{saved}

## Doing it well

- Scale to the request: a quick check needs a few agents and one vote; "audit everything" \
justifies wide fan-out and several votes. Tell the tool how many agents you expect with \
`estimated_agents`.
- Agents that run at the same time must not change the same files: give writers separate \
`write_scope`s, or let a single agent write.
- What agents return are claims, not facts. Verify findings with adversarial_verify and \
changes with fix_until or a check command before you report them.
- No silent caps: whenever the script drops, truncates or samples something, `log()` what \
was left out.
- When a workflow fails, fix the script and run it again with `resume: true`: agents whose \
call and inputs are unchanged return their earlier results instead of running again.
- When the workflow is done, read what came back, check what matters, and give the user \
the answer."""

MODELS = """\
## Models for workflow agents

Agents run on {default} unless you choose another with `model`. Put strong models on work \
that needs judgment and cheaper ones on well-specified, mechanical work; checking findings \
on a different model than the one that produced them catches more. Available models:
{guide}"""

FAILED_HINT = (
    "\n\nFix the script and call `workflow` again with `resume: true`: agents whose call and "
    "inputs are unchanged return their earlier results instead of running again."
)
STOPPED_HINT = (
    "\n\nA limit stopped the workflow. Use what finished, or run a narrower workflow (fewer "
    "items or votes). The user can raise the limits in the Ultra settings (max-agents, "
    "workflow-budget)."
)

_META_NAME = re.compile(r"""["']name["']\s*:\s*["']([^"'\n]{1,60})["']""")


# --------------------------------------------------------------------------------------------
# The workflow tool
# --------------------------------------------------------------------------------------------


class WorkflowTool(Tool):
    """Runs a workflow script, or a saved workflow by name, and returns its result."""

    name = "workflow"
    read_only = False
    description = (
        "Run a workflow: a short Python script that fans work out to other agents, checks "
        "their findings and combines the results, or a saved workflow by name. Returns the "
        "script's `result` and an account of what ran (agents, failures, tokens). The script "
        "API is described under 'Working with workflows' in your instructions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "A workflow script you wrote. Give this or `name`.",
            },
            "name": {
                "type": "string",
                "description": "The name of a saved workflow to run instead of a script.",
            },
            "args": {
                "type": "object",
                "description": "A JSON object the script reads as `args`.",
            },
            "estimated_agents": {
                "type": "integer",
                "minimum": 1,
                "description": "How many agents you expect the workflow to start.",
            },
            "resume": {
                "type": "boolean",
                "description": "Reuse the results of agents from the previous run of this "
                "workflow whose call and inputs are unchanged. Use it after fixing a script "
                "that failed part way, not after changing files those agents read.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, orchestrator: UltraOrchestrator) -> None:
        self._orchestrator = orchestrator

    def summarize(self, tool_input: dict[str, Any]) -> str:
        name = tool_input.get("name")
        if not name:
            found = _META_NAME.search(str(tool_input.get("script") or ""))
            name = found.group(1) if found else "script"
        return f"workflow: {name}"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return await self._orchestrator.run_workflow(tool_input)


# --------------------------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------------------------


class UltraOrchestrator(Orchestrator):
    name = "ultra"
    title = "Ultra"
    summary = (
        "Writes short scripts that fan work out to many agents, verify their findings and "
        "combine the results."
    )
    guidance = (
        "For broad or high-stakes work: reviews, audits, research, and changes that must be "
        "verified. Each substantive task becomes one or more workflows of many agents, so it "
        "costs several times what Solo does; a token budget and an agent cap per workflow "
        "keep that bounded. Set activation to auto to keep simple tasks cheap."
    )
    settings = (
        Setting(
            "activation",
            "choice",
            "always",
            "always: plan a workflow for every substantive task. auto: only when fan-out, "
            "independent verification or breadth clearly helps.",
            choices=("always", "auto"),
        ),
        Setting(
            "max-concurrent",
            "int",
            6,
            "Workflow agents that may run at the same time.",
            minimum=1,
            maximum=64,
        ),
        Setting(
            "max-agents",
            "int",
            60,
            "Agents one workflow may start; past this it stops and returns what finished.",
            minimum=1,
            maximum=1000,
        ),
        Setting(
            "workflow-budget",
            "int",
            1_500_000,
            "Tokens one workflow's agents may use (output plus prompt tokens not read from the "
            "cache); past this it stops and returns what finished.",
            minimum=10_000,
            maximum=100_000_000,
        ),
        Setting(
            "verify-votes",
            "int",
            3,
            "Skeptics per finding when a workflow verifies findings adversarially.",
            minimum=1,
            maximum=9,
        ),
        Setting(
            "large-workflow",
            "int",
            25,
            "Tell the user before a workflow that is expected to start more agents than this.",
            minimum=1,
            maximum=1000,
        ),
    )

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self._root = session.new_node("Conversation", role="root")
        #: Results of finished workflow agents, for `resume`. Lives as long as the method.
        self.journal = Journal()

    def limits(self) -> Limits:
        """The current limits for a workflow, from this method's settings."""
        return Limits(
            max_concurrent=self.setting("max-concurrent"),
            max_agents=self.setting("max-agents"),
            budget=self.setting("workflow-budget"),
            verify_votes=self.setting("verify-votes"),
            announce_above=self.setting("large-workflow"),
        )

    # -- the conversation -----------------------------------------------------------------

    def _ensure_agent(self) -> Agent:
        if self.root_agent is None:
            session = self.session
            catalog = load_saved()
            for problem in catalog.problems:
                session.notice("warning", f"A saved workflow could not be loaded: {problem}")
            role = f"{prompts.USER_FACING}\n\n{self.role_text(catalog)}"
            self.root_agent = session.agent(
                self._root,
                system=prompts.system_prompt(session, role),
                tools=[*standard_tools(), AskUserTool(session), WorkflowTool(self)],
            )
        return self.root_agent

    def role_text(self, catalog: Catalog) -> str:
        """The workflow section of the user-facing agent's system prompt."""
        limits = self.limits()
        saved = "\n".join(w.describe() for w in catalog.workflows) or "(none)"
        text = ROLE.format(
            activation=ACTIVATION[self.setting("activation")],
            roles=", ".join(f'"{role}"' for role, _ in MODEL_ROLES),
            max_concurrent=limits.max_concurrent,
            max_agents=limits.max_agents,
            budget=limits.budget,
            max_items=MAX_ITEMS,
            votes=limits.verify_votes,
            saved=saved,
        )
        guide = models_guide(self.session)
        if guide:
            text += "\n\n" + MODELS.format(default=self.session.model_for("subagent"), guide=guide)
        return text

    async def run(self, prompt: str) -> str:
        session = self.session
        fresh = self.root_agent is None
        agent = self._ensure_agent()
        if fresh and (recap := session.recap()):
            prompt = f"{recap}\n\nThe user now says:\n{prompt}"
        session.update_node(self._root, status="running")
        result = await agent.run(prompt)
        if result.stop == "refusal":
            session.update_node(self._root, status="failed")
            return f"The model declined this request. {result.detail}".strip()
        session.update_node(self._root, status="done")
        return result.text

    # -- workflows --------------------------------------------------------------------------

    async def run_workflow(self, tool_input: dict[str, Any]) -> ToolOutput:
        """Check and run the workflow a `workflow` tool call asks for; report the outcome."""
        session = self.session
        script = self._script_for(tool_input)
        if isinstance(script, ToolOutput):
            return script

        limits = self.limits()
        estimate = tool_input.get("estimated_agents")
        if isinstance(estimate, int) and estimate > limits.max_agents:
            return ToolOutput.error(
                f"This workflow would start about {estimate} agents, but one workflow may "
                f"start at most {limits.max_agents} (setting max-agents). Narrow the fan-out "
                "(fewer items or votes), split the work into several workflows, or ask the "
                "user to raise the limit."
            )
        run = WorkflowRun(
            session,
            parent=self._root,
            limits=limits,
            journal=self.journal.begin(script.meta.name, resume=bool(tool_input.get("resume"))),
        )
        session.update_node(self._root, status="waiting")
        try:
            outcome = await run.execute(
                script,
                tool_input.get("args"),
                estimated_agents=estimate if isinstance(estimate, int) else None,
            )
        finally:
            session.update_node(self._root, status="running")
        text = outcome.render(limit=max(session.config.agent.tool_output_limit - 1000, 4000))
        if outcome.status == "failed":
            text += FAILED_HINT
        elif outcome.status == "stopped":
            text += STOPPED_HINT
        return ToolOutput(text, is_error=outcome.status != "completed")

    @staticmethod
    def _script_for(tool_input: dict[str, Any]) -> Script | ToolOutput:
        """The checked script a tool call names or contains, or the error to answer with."""
        source = str(tool_input.get("script") or "").strip()
        name = str(tool_input.get("name") or "").strip()
        if bool(source) == bool(name):
            return ToolOutput.error(
                "Give either `script` (a workflow you wrote) or `name` (a saved workflow), "
                "not both and not neither."
            )
        if name:
            saved = find_saved(name)
            if saved is None:
                names = ", ".join(w.name for w in load_saved().workflows) or "none"
                return ToolOutput.error(
                    f"There is no saved workflow named {name!r}. Saved: {names}."
                )
            return saved.script
        try:
            return compile_script(source)
        except ScriptError as exc:
            count = len(exc.problems)
            return ToolOutput.error(
                f"The script was not run: {count} problem{'' if count == 1 else 's'} found."
                f"\n\n{exc}\n\nFix every problem listed and call `workflow` again."
            )
