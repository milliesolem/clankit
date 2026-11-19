"""Exceptions shared by the workflow engine, its primitives and the pattern library.

Two families, with deliberately different bases:

* `WorkflowError` is an ordinary `Exception`: a script used the API wrongly (bad arguments,
  too many items, nesting too deep). It is raised into the script at the offending call, so
  the script can catch it like any other error, and otherwise it ends the workflow with a
  line-numbered message for the agent that wrote the script.
* `WorkflowStop` is a `BaseException`: a limit was reached (token budget, agent cap). It must
  end the whole workflow, so a script's ``except Exception`` cannot swallow it, and
  `parallel` and `pipeline` let it through instead of turning it into a None slot.
"""

from __future__ import annotations


class WorkflowError(Exception):
    """A script called a workflow primitive in a way that cannot work."""


class WorkflowStop(BaseException):
    """A limit was reached and the workflow must stop; what finished is still reported."""


class BudgetExhausted(WorkflowStop):
    """The workflow (or the run it belongs to) has used up its token budget."""


class AgentLimitReached(WorkflowStop):
    """The workflow tried to start more agents than it is allowed to."""
