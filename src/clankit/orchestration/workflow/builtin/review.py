# A saved workflow script. clankit.orchestration.workflow runs it with top-level await and
# defines agent, parallel, phase, log, args, json and the pattern functions for it.
# ruff: noqa: F704, F821
meta = {
    "name": "review",
    "description": (
        "Review code changes for real defects: finders look in different ways until a round "
        "turns up nothing new, every finding is checked by independent skeptics, and only "
        "the ones that survive are reported, most severe first."
    ),
    "when_to_use": "Before reporting a change as done, or when asked to review a diff, a branch or some files.",
    "phases": ["Scope", "Find", "Verify"],
    "args": {
        "target": "optional: what to review (the uncommitted changes, git diff HEAD, by default)",
        "focus": "optional list of angles to review from (four general ones by default)",
    },
}

target = args.get("target") or "the uncommitted changes in the working tree (git diff HEAD)"
lenses = args.get("focus") or [
    "correctness: wrong logic or conditions, off-by-one errors, broken invariants",
    "failure handling and edge cases: empty or unusual input, errors, concurrency",
    "tests: behaviour the change needs that no test covers, tests weakened or skipped",
    "security and data safety: injection, leaked secrets, unsafe file or shell use",
]
scope_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "What the change does, in a few sentences."},
        "files": {"type": "array", "items": {"type": "string"}},
        "diff_command": {
            "type": "string",
            "description": "A command that shows exactly the changes under review.",
        },
    },
    "required": ["summary", "files", "diff_command"],
}
finding_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "path:line"},
        "problem": {"type": "string", "description": "What is wrong and why it matters."},
        "evidence": {"type": "string", "description": "Quoted code or command output."},
        "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
    },
    "required": ["location", "problem", "evidence", "severity"],
}
severity_order = {"critical": 0, "major": 1, "minor": 2}

phase("Scope")
scope = await agent(
    f"Work out exactly what is to be reviewed: {target}. Use git (status, diff, log) and read "
    "files as needed; do not change anything.",
    schema=scope_schema,
    label="scope",
    tools="all",
    write_scope=[],
)

if scope is None:
    log("Could not determine what to review, so nothing was reviewed.")
    result = {"error": "Could not determine what to review; nothing was reviewed.", "findings": []}
else:
    context = (
        f"The change under review: {scope['summary']}\n"
        f"Changed files: {', '.join(scope['files']) or 'none listed'}\n"
        f"See the change with: {scope['diff_command']}"
    )
    phase("Find")
    findings = await loop_until_dry(
        [
            f"Review this change for {lens}.\n\n{context}\n\nReport only defects you can point "
            "to in the code, with evidence. Finding nothing is a fine outcome."
            for lens in lenses
        ],
        "location",
        item_schema=finding_schema,
        dry_rounds=1,
        tools="all",
        write_scope=[],
    )
    phase("Verify")
    confirmed = await adversarial_verify(findings, context=context, tools="all", write_scope=[])
    confirmed = sorted(confirmed, key=lambda finding: severity_order.get(finding["severity"], 3))
    result = {
        "reviewed": scope["summary"],
        "files": scope["files"],
        "findings": confirmed,
        "refuted": len(findings) - len(confirmed),
    }
