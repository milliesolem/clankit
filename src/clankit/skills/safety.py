"""Static checks on the files of a generated skill, before anything is saved.

A skill is text that a future agent will trust and scripts it will run, written by a model
from a transcript that may contain hostile content. Research on skill ecosystems and on
experience-to-skill pipelines agrees on what to look for (commands that destroy data,
remote code execution, credential harvesting and upload, text that tries to redirect the
agent, hidden content) and on the limits of asking another model to judge: so these checks
are deterministic, and a hit drops the candidate rather than asking for a rewrite, which
would only teach the author to evade the pattern.

The checks are conservative: they flag shapes, not intent, and a skill that says "never run
sudo" trips the same rule as one that runs it. A false alarm costs one proposal; the user can
still write that skill by hand.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass

from clankit.skills.redaction import Redactor

_SHELL_CHECK_TIMEOUT = 10

_I = re.IGNORECASE
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, flags))
    for label, pattern, flags in (
        # -- destroying data ---------------------------------------------------------------
        (
            "deletes a root, home or wildcard path recursively",
            r"\brm\s+(?:-\S+\s+)*-(?:[a-zA-Z]*[rR][a-zA-Z]*|-recursive)\s+(?:-\S+\s+)*"
            r"""["']?(?:/|~|\$\{?HOME\}?|\*|\.\.?)(?:/\*?)?["']?(?=\s|[;&|)]|$)""",
            re.MULTILINE,
        ),
        (
            "deletes a root or home directory tree",
            r"\brmtree\(\s*(?:Path\.home\(\)|(?:os\.path\.)?expanduser\(\s*[\"']~[\"']\s*\)"
            r"|[\"'](?:/|~)[\"'])",
            0,
        ),
        (
            "formats or overwrites a disk",
            r"\bmkfs(?:\.\w+)?\b|\bdd\b[^\n]*\bof=/dev/|>\s*/dev/(?:sd|nvme|hd|disk)",
            0,
        ),
        ("contains a fork bomb", r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", 0),
        (
            "makes files world-writable recursively",
            r"\bchmod\s+(?:-\S+\s+)*-R\s+(?:-\S+\s+)*0?777\b",
            0,
        ),
        (
            "discards uncommitted work or rewrites published history",
            r"\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[a-zA-Z]*f|"
            r"\bgit\s+push\b[^\n]*(?:--force(?!-with-lease)|\s-f\b)",
            0,
        ),
        ("runs commands as root", r"(?:^|[\s;&|(`])sudo\s+\S", re.MULTILINE),
        # -- running code from elsewhere ---------------------------------------------------
        (
            "pipes a download into an interpreter",
            r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:(?:ba|z|da|k)?sh|python[0-9.]*|perl|ruby|node)\b"
            r"|\b(?:ba|z)?sh\s+<\(\s*(?:curl|wget)\b|\beval\s+[\"']?\$\(\s*(?:curl|wget)\b",
            0,
        ),
        (
            "decodes and executes hidden code",
            r"\bbase64\s+(?:-d|--decode)\b[^\n]*\|\s*(?:ba|z)?sh\b|\b(?:exec|eval)\s*\([^\n]*b64decode",
            0,
        ),
        # -- sending data away -------------------------------------------------------------
        (
            "uploads a file or variable with curl or wget",
            r"\bcurl\b[^\n]*\s(?:-d|--data(?:-[a-z]+)?|-F|--form|-T|--upload-file)[\s=]+[\"']?[^\n\"']*[@$]"
            r"|\bwget\b[^\n]*--post-(?:file|data)",
            0,
        ),
        ("opens a raw network connection", r"/dev/tcp/|\bnc(?:at)?\b[^\n]*\s\d{2,5}\b[^\n]*<", 0),
        (
            "reads stored credentials",
            r"\.ssh/|\bid_(?:rsa|ed25519|ecdsa)\b|\.aws/credentials|\.netrc\b|\.git-credentials\b"
            r"|\.gnupg\b|/etc/shadow\b|clankit/config\.kdl",
            0,
        ),
        # -- redirecting the agent ---------------------------------------------------------
        (
            "tells the agent to ignore its instructions",
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+|your\s+)*"
            r"(?:previous|prior|above|earlier|system|other)\s+(?:instructions?|prompts?|rules|messages)"
            r"|\bnew\s+system\s+prompt\b|\byou\s+are\s+now\s+(?:a|an|in)\b",
            _I,
        ),
        (
            "tells the agent to get around safety or permission checks",
            r"\b(?:override|bypass|circumvent|disable|skip)\s+(?:the\s+|any\s+|all\s+)?"
            r"(?:system\s+prompt|safety|permissions?|approvals?|guardrails?|sandbox)"
            r"|\b(?:always|automatically)\s+(?:approve|allow|accept)\s+(?:all|every|any)\b"
            r"|\b(?:don'?t|do\s+not|never)\s+ask\s+(?:the\s+user\s+)?(?:again|for\s+(?:permission|confirmation|approval))",
            _I,
        ),
        (
            "tells the agent to hide things from the user",
            r"\b(?:do\s+not|don'?t|never)\s+(?:tell|inform|notify|alert|mention\s+(?:this\s+|it\s+)?to)\s+the\s+user"
            r"|\bwithout\s+(?:asking|telling|informing|notifying)\s+the\s+user",
            _I,
        ),
        (
            "asks for secrets or the system prompt to be revealed",
            r"\b(?:reveal|print|output|send|post|upload|exfiltrate)\s+(?:the\s+|your\s+|all\s+)*"
            r"(?:system\s+prompt|api[\s_-]?keys?|credentials|secrets|passwords?)",
            _I,
        ),
        # -- hiding content ----------------------------------------------------------------
        ("contains an HTML comment (text a reader would not see)", r"<!--", 0),
        ("contains invisible or direction-changing characters", "[​-‏‪-‮⁠-⁤﻿]", 0),
    )
)

_HARVEST_RE = re.compile(
    r"\bprintenv\b|\benv\s*\||\bos\.environ\b(?!\s*(?:\.get\s*\(|\[))|\bset\s*\|"
)
_NETWORK_RE = re.compile(
    r"\brequests\.|\burllib\.request\b|\burlopen\b|\bhttp\.client\b|\bhttpx\b|\bsocket\.|"
    r"\bcurl\b|\bwget\b|\bnc\b"
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One reason not to save a generated file."""

    path: str
    problem: str

    def __str__(self) -> str:
        return f"{self.path} {self.problem}"


def scan(path: str, text: str, redactor: Redactor) -> list[Finding]:
    """Everything about the generated file `path` that makes it unsafe to keep."""
    findings = [Finding(path, label) for label, pattern in _RULES if pattern.search(text)]
    if _HARVEST_RE.search(text) and _NETWORK_RE.search(text):
        findings.append(Finding(path, "collects the whole environment and uses the network"))
    if redactor.finds_secret(text):
        findings.append(Finding(path, "contains what looks like a credential"))
    return findings


def syntax_problem(path: str, source: str) -> str:
    """Why the script at `path` would not even start, or "" when it parses.

    Python is compiled (never executed); shell scripts are parsed with ``bash -n`` when bash
    is installed and accepted unchecked otherwise.
    """
    if path.endswith(".py"):
        return _python_problem(path, source)
    if path.endswith(".sh"):
        return _shell_problem(source)
    return ""


def _python_problem(path: str, source: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(source, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        line = f" (line {exc.lineno})" if isinstance(exc, SyntaxError) and exc.lineno else ""
        return f"is not valid Python{line}: {getattr(exc, 'msg', None) or exc}"
    return ""


def _shell_problem(source: str) -> str:
    bash = shutil.which("bash")
    if bash is None:
        return ""
    try:
        result = subprocess.run(
            [bash, "-n"],
            input=source,
            capture_output=True,
            text=True,
            timeout=_SHELL_CHECK_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode == 0:
        return ""
    detail = " ".join(result.stderr.split())[:300]
    return f"is not valid shell: {detail}"
