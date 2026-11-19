# A saved workflow script. clankit.orchestration.workflow runs it with top-level await and
# defines agent, parallel, phase, log, args, json and the pattern functions for it.
# ruff: noqa: F704, F821
meta = {
    "name": "understand",
    "description": (
        "Map how something works in this codebase: explorers read it from different angles "
        "at the same time, a fresh agent looks for gaps that are then followed up, and the "
        "findings are merged into one explanation with file references."
    ),
    "when_to_use": "Before changing unfamiliar code, or to answer a question that spans many files.",
    "phases": ["Explore", "Check", "Explain"],
    "args": {
        "question": "what to find out",
        "angles": "optional list of angles to explore from (three general ones by default)",
    },
}

FOLLOW_UPS = 4

question = args["question"]
angles = args.get("angles") or [
    "the entry points and the main flow of control",
    "the data structures and how data moves between components",
    "error handling, edge cases, and the tests that pin the behaviour down",
]
findings_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "where": {"type": "string", "description": "path:line"},
                },
                "required": ["fact", "where"],
            },
        },
    },
    "required": ["summary", "facts"],
}


def explore(focus):
    return (
        f"Investigate this question about the codebase: {question}\n\n"
        f"Focus on {focus}. Read the code rather than guessing, cite every fact with "
        "path:line, and report only what you saw in the files."
    )


phase("Explore")
reports = await parallel(
    [
        agent(
            explore(angle),
            schema=findings_schema,
            label=f"explore: {angle}",
            tools="read-only",
        )
        for angle in angles
    ]
)
found = [report for report in reports if report]
if len(found) < len(reports):
    log(f"{len(reports) - len(found)} of {len(reports)} explorers came back without findings")

phase("Check")
gaps = await completeness_critic(json.dumps(found, indent=2), task=question)
if len(gaps) > FOLLOW_UPS:
    log(
        f"Not followed up ({len(gaps) - FOLLOW_UPS} of {len(gaps)} gaps): "
        + "; ".join(gaps[FOLLOW_UPS:])
    )
if gaps:
    extra = await parallel(
        [
            agent(
                explore(f"this open point: {gap}"),
                schema=findings_schema,
                label=f"follow up: {gap[:50]}",
                tools="read-only",
            )
            for gap in gaps[:FOLLOW_UPS]
        ]
    )
    found += [report for report in extra if report]

phase("Explain")
explanation = await agent(
    f"Answer this question about the codebase: {question}\n\n"
    "Other agents explored the code and reported these findings:\n"
    f"<findings>\n{json.dumps(found, indent=2)}\n</findings>\n\n"
    "Write a clear explanation that answers the question. Check each fact you rely on "
    "against the code, cite path:line, and say where the findings disagree or leave "
    "something open.",
    label="explain",
    tools="read-only",
)
if explanation is None:
    log("The explanation could not be written; returning the explorers' findings instead.")
result = explanation if explanation is not None else {"findings": found}
