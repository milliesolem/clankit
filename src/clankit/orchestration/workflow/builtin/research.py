# A saved workflow script. clankit.orchestration.workflow runs it with top-level await and
# defines agent, parallel, phase, log, args, json and the pattern functions for it.
# ruff: noqa: F704, F821
meta = {
    "name": "research",
    "description": (
        "Answer a question with evidence: searchers look in the code, the project's history "
        "and documentation, and the web, round after round until nothing new turns up; "
        "skeptics check every claim; the report cites its sources and keeps verified and "
        "unverified claims apart."
    ),
    "when_to_use": (
        "For questions whose answer must be right and sourced: how a library behaves, why "
        "the code is the way it is, what the options are."
    ),
    "phases": ["Search", "Verify", "Report"],
    "args": {
        "question": "what to find out",
        "sources": "optional list of places to search (the code, its history and docs, and the web by default)",
    },
}

question = args["question"]
sources = args.get("sources") or [
    "the code in this repository",
    "the project's history (git log and blame), its documentation and its configuration",
    "authoritative documentation on the web (fetch official documentation with web_fetch)",
]
claim_schema = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "source": {"type": "string", "description": "path:line, a commit, or a URL"},
        "support": {"type": "string", "description": "A quote or command output that supports it."},
    },
    "required": ["claim", "source", "support"],
}

phase("Search")
claims = await loop_until_dry(
    [
        f"Research this question: {question}\n\nSearch {source}. Return claims that help "
        "answer it, each with its exact source and a quote or output that supports it."
        for source in sources
    ],
    "claim",
    item_schema=claim_schema,
    tools="all",
    write_scope=[],
)

phase("Verify")
verified = await adversarial_verify(claims, context=f"The question being researched: {question}")
unverified = [claim for claim in claims if claim not in verified]

phase("Report")
report = await agent(
    f"Write a report that answers this question: {question}\n\n"
    "Claims that independent checks confirmed:\n"
    f"<verified>\n{json.dumps(verified, indent=2)}\n</verified>\n\n"
    "Claims that could not be confirmed:\n"
    f"<unverified>\n{json.dumps(unverified, indent=2)}\n</unverified>\n\n"
    "Build the answer on the verified claims and cite their sources. Mention an unverified "
    "claim only where it matters, and label it as unverified. Say plainly what remains unknown.",
    label="report",
    tools="read-only",
)
if report is None:
    log("The report could not be written; returning the claims instead.")
result = {"report": report, "verified": verified, "unverified": unverified}
