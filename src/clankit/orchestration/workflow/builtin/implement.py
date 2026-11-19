# A saved workflow script. clankit.orchestration.workflow runs it with top-level await and
# defines agent, parallel, phase, log, args, json and the pattern functions for it.
# ruff: noqa: F704, F821
meta = {
    "name": "implement",
    "description": (
        "Make a change and prove it: a planner studies the code and writes a plan, a single "
        "agent builds it, the harness runs the check command until it passes, and "
        "independent reviewers look for defects, which are verified and then fixed."
    ),
    "when_to_use": "For a well-defined change, ideally one with a command that proves it works.",
    "phases": ["Plan", "Build", "Check", "Review"],
    "args": {
        "task": "the change to make, with its acceptance criteria",
        "check": "optional shell command that must pass, such as the project's test command",
    },
}

task = args["task"]
check = args.get("check")
plan_schema = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "string"}},
        "files": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["steps", "files"],
}
review_schema = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "path:line"},
                    "problem": {"type": "string"},
                    "evidence": {"type": "string", "description": "Quoted code or command output."},
                },
                "required": ["location", "problem", "evidence"],
            },
        }
    },
    "required": ["findings"],
}
lenses = ["correctness and completeness against the task", "edge cases, failure handling and tests"]

phase("Plan")
plan = await agent(
    f"Plan this change: {task}\n\nRead the relevant code first. List the steps, the files "
    "that will change and the risks. Do not change anything.",
    schema=plan_schema,
    label="plan",
    tools="read-only",
    model="planner",
)
if plan is None:
    log("The planner failed, so the builder works without a plan.")

phase("Build")
brief = f"Make this change: {task}\n\n"
if plan is not None:
    brief += (
        "An agent that studied the code proposed this plan:\n"
        f"{json.dumps(plan, indent=2)}\n\nFollow it where it holds up; where it does not, do "
        "what the task needs and say why.\n\n"
    )
if check:
    brief += f"The change is accepted when this command passes: `{check}`.\n\n"
brief += "Verify your work by running the relevant tests, then say what you changed and how you checked it."
build = await agent(brief, label="build")

phase("Check")
passed = None
if check:
    passed = await fix_until(check, task=task)
else:
    log("No check command was given, so the harness did not verify the build.")

phase("Review")
reviews = await parallel(
    [
        agent(
            f"Review the uncommitted changes (git diff HEAD) against this task: {task}\n\n"
            f"Look for problems with {lens}. Report only defects you can show with evidence; "
            "finding nothing is a fine outcome.",
            schema=review_schema,
            label=f"review: {lens}",
            tools="all",
            write_scope=[],
            model="reviewer",
        )
        for lens in lenses
    ]
)
issues = [finding for review in reviews if review for finding in review["findings"]]
confirmed = await adversarial_verify(
    issues, context=f"The task: {task}\nThe changes: git diff HEAD", tools="all", write_scope=[]
)
if confirmed:
    await agent(
        f"Independent reviewers confirmed these defects in the change made for this task: "
        f"{task}\n\n{json.dumps(confirmed, indent=2)}\n\nFix each one, run the relevant tests, "
        "and say what you changed.",
        label="fix review findings",
    )
    if check:
        passed = await fix_until(check, task=task)

result = {
    "summary": build,
    "check": check,
    "check_passed": passed,
    "review_findings": confirmed,
}
