"""The configuration file written on first run.

It has to make sense to someone who has never seen KDL, so it explains the format in a few
lines and then shows every setting where it belongs. Nothing in it is switched on: each
setting is written with "/-" in front, which keeps the file equal to the built-in defaults
(a default that changes in a later version reaches everyone who never touched the line) while
showing exactly what to write. The sections themselves are real, empty blocks, so that a
setting changed from the settings screens lands next to the comment that explains it.

Values come from the configuration classes and the provider catalog rather than being typed
out here, so the examples cannot drift from the real defaults.
"""

from __future__ import annotations

import textwrap
from string import Template

from clankit.config.catalog import CATALOG
from clankit.config.kdlsyntax import format_name, format_value
from clankit.config.schema import (
    MODEL_ROLES,
    AgentConfig,
    ModelsConfig,
    OrchestrationConfig,
    PermissionsConfig,
    SkillsConfig,
    UIConfig,
    UpdatesConfig,
    WebConfig,
)
from clankit.llm.types import EFFORT_LEVELS

#: First line of every file Clankit writes. Editors and other tools use it to tell KDL v2 from
#: v1; to a parser it is a switched-off node.
MARKER = "/- kdl-version 2"

_TEMPLATE = Template(
    """\
$marker
// Clankit configuration, in KDL (https://kdl.dev). A setting is a name and a value; groups
// of settings sit in { braces }. Text goes in "double quotes", switches are #true or #false.
// "//" starts a comment, and "/-" switches off the setting or group after it.
//
// Everything is optional: what is missing or switched off keeps its default. The
// switched-off lines below show the defaults and some examples; delete the "/-" to use one.
// Clankit's settings screens edit this file as well, and leave your comments and layout
// alone. (The first line tells editors which version of KDL this is.)

// The model agents use, written "provider/model".
/- model $model

// Models for particular jobs. A role that is not listed uses `model`.
roles {
$roles
}

// Where models come from. A provider Clankit knows needs only an API key, written here or
// kept in an environment variable. It knows:
$known
providers {
$providers
    // Any other server that speaks the OpenAI chat API can be described in full.
    /- my-server {
        kind "openai-compatible"
        name "My server"
        base-url "http://localhost:11434/v1"
        keyless #true
        model "qwen3-coder" {
            name "Qwen3 Coder"
            context-window 64000
            free #true
            note "Good at mechanical edits."
        }
    }
}

// How a prompt turns into agent work. "task-tree" breaks it into subtasks, and those into
// smaller ones, until each is simple enough for a single agent.
orchestration {
    /- method $method
    // Each method keeps its own settings in a group named after it.
    /- task-tree {
        max-depth 10
    }
}

// How agents work.
agent {
    // How hard models that support it should think: $efforts.
    /- effort "high"
    // Model calls one agent may make for one task before it is stopped.
    /- max-turns $max_turns
    // Context window, in tokens, assumed for models whose real one is unknown.
    /- context-window $context_window
    // Fraction of the context window at which an agent summarises its work and starts fresh.
    /- compact-at $compact_at
    // Characters of one tool result passed to the model; longer results lose their middle.
    /- tool-output-limit $tool_output_limit
    // Seconds a shell command may run.
    /- shell-timeout $shell_timeout
    // Output tokens one prompt may use across all its agents before Clankit asks whether to
    // continue. 0 switches the check off.
    /- run-budget $run_budget
}

// What agents may do without asking.
permissions {
    // "ask" confirms edits, shell commands and network access. "accept-edits" stops asking
    // about edits inside the workspace. "auto" asks nothing. "read-only" refuses every change.
    /- mode $mode
    // Rules make exceptions, one per line, written "category:pattern" with * as a wildcard.
    // A deny rule beats an allow rule.
    /- allow "shell:git status*"
    /- allow "edit:docs/**"
    /- deny "shell:sudo *"
}

// Skills are folders with a SKILL.md and helper scripts that agents pick up when relevant.
skills {
    // Where they live. Without this line: the "skills" folder next to this file.
    /- directory "~/my-skills"
    // After a run, turn what was learned into new skills: "off" never does, "ask" proposes
    // them and waits for you, "auto" saves them without asking.
    /- generate $generate
}

// How Clankit looks, in the terminal and in the browser.
ui {
    // A built-in theme, or one of your own from `themes` below.
    /- theme $theme
    // Show the models' reasoning in the transcript.
    /- show-thinking $show_thinking
    // Keep the task tree panel open.
    /- show-tree $show_tree
}

// The web interface.
web {
    // Keep this address unless other machines should reach the interface.
    /- host $host
    /- port $port
    // Open a browser window when the web interface starts.
    /- open-browser $open_browser
}

// Keeping current.
updates {
    // Check PyPI at most once a day; a regular pip installation is upgraded quietly.
    /- check $check
}

// Your own colour themes. `base` starts from an existing theme and the other lines replace
// single colours, written "#rrggbb".
themes {
    /- my-theme {
        base $theme
        accent "#c08040"
    }
}
"""
)


def starter_document() -> str:
    """The commented file for a fresh installation. It reads back as exactly `Config()`."""
    agent, web, ui = AgentConfig(), WebConfig(), UIConfig()
    known = ", ".join(format_name(provider_id) for provider_id in CATALOG) + "."
    return _TEMPLATE.substitute(
        marker=MARKER,
        model=format_value(ModelsConfig().default),
        roles="\n".join(_role_lines()),
        known=textwrap.fill(known, 92, initial_indent="// ", subsequent_indent="// "),
        providers="\n".join(_provider_lines()),
        method=format_value(OrchestrationConfig().method),
        efforts=_listed([format_value(level) for level in EFFORT_LEVELS]),
        max_turns=format_value(agent.max_turns),
        context_window=format_value(agent.context_window),
        compact_at=format_value(agent.compact_at),
        tool_output_limit=format_value(agent.tool_output_limit),
        shell_timeout=format_value(agent.shell_timeout),
        run_budget=format_value(agent.run_budget),
        mode=format_value(PermissionsConfig().mode),
        generate=format_value(SkillsConfig().generate),
        theme=format_value(ui.theme),
        show_thinking=format_value(ui.show_thinking),
        show_tree=format_value(ui.show_tree),
        host=format_value(web.host),
        port=format_value(web.port),
        open_browser=format_value(web.open_browser),
        check=format_value(UpdatesConfig().check),
    )


def _listed(items: list[str]) -> str:
    """``a, b or c``."""
    return " or ".join(filter(None, [", ".join(items[:-1]), *items[-1:]]))


def _role_lines() -> list[str]:
    """One switched-off example per role. The first shows a role handed to another provider."""
    default = ModelsConfig().default
    other = next(
        (
            f"{provider.id}/{provider.default_model}"
            for provider in CATALOG.values()
            if provider.default_model and not default.startswith(f"{provider.id}/")
        ),
        default,
    )
    lines = []
    for index, (role, purpose) in enumerate(MODEL_ROLES):
        example = format_value(default if index else other)
        lines += [f"    // {purpose}.", f"    /- {format_name(role)} {example}"]
    return lines


def _provider_lines() -> list[str]:
    """Switched-off entries for catalog providers: the first two that take an API key, one
    with the key in the file and one reading it from the environment, and the first that runs
    on the user's machine."""
    keyed = [provider for provider in CATALOG.values() if not provider.keyless][:2]
    local = [provider for provider in CATALOG.values() if provider.keyless][:1]
    lines = []
    for index, provider in enumerate(keyed):
        if index and provider.api_key_env:
            setting = f"api-key-env {format_value(provider.api_key_env)}"
        else:
            setting = 'api-key "paste your key here"'
        lines += [f"    /- {format_name(provider.id)} {{", f"        {setting}", "    }"]
    for provider in local:
        lines += [
            f"    // {provider.name} runs on your own machine and needs no key: naming it is enough.",
            f"    /- {format_name(provider.id)}",
        ]
    return lines
