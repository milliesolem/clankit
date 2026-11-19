# Clankit

A coding agent that grows a **task tree**. Clankit is a Python (≥ 3.12) agent with a terminal
UI and a web UI, built so you can point it at any model — Anthropic, DeepSeek, OpenAI-compatible
servers, and a range of free and local models.

**Highlights**

- **Recursive Task Tree** (the default orchestration): agents split work into subtasks, each
  running on its own model if you want, while parents review, integrate and verify.
- **Eight more orchestration strategies** — Solo, Review Loop, Test Loop, Architect-Editor,
  Escalate, Best of N, Ultra, and a model-free Demo — with per-method, user-adjustable settings.
- **Cross-model delegation**: any subtask or helper role can run on a different provider/model.
- **Two UIs**: a Textual terminal UI and a Pyramid web UI, both driving the same engine.
- **Skills**: reusable `SKILL.md` instructions with bundled scripts, plus automatic skill
  generation from the work a session actually did.
- **KDL configuration** in `~/.config/clankit/config.kdl`, with a permission system, session
  resume, `/undo`, token/cost tracking and 25 built-in colour themes.

---

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone <this-repo> clankit
cd clankit
uv sync            # creates .venv and installs the `clankit` command
uv run clankit --version
```

`uv sync` installs the `clankit` entry point into the project virtualenv, so you can run
`uv run clankit` from anywhere in the repo. To install it as a standalone tool, use
`uv tool install .` from the repo root.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'zope.deprecation'` when starting `clankit web`**:
  your Python mixes Debian's `python3-zope.interface` (in the system site-packages) with a
  pip-installed `zope.deprecation` (in `~/.local`), and the two namespace styles cannot see
  each other. Either install the matching system package
  (`sudo apt install python3-zope.deprecation`), or uninstall `python3-zope.interface`
  so pip owns the whole `zope` namespace.

- **`unexpected TypeError: process() takes no keyword arguments` when prompting**:
  an old `brotli` binding (e.g. Debian's `python3-brotli` 1.1) is on the path and
  `httpx2`'s brotli decoder calls it with a keyword argument it does not know. Clankit
  now only advertises `gzip`/`deflate`, so upgrade to clankit 0.1.0 with this fix, or
  install a newer brotli (`pip install -U brotli`).

## Getting started

On first run Clankit writes a commented starter file at `~/.config/clankit/config.kdl`
(honouring `XDG_CONFIG_HOME`). Every line in it is switched off with `/-`, so the file reads
back as the built-in defaults while showing you exactly what to write.

The one thing you need is a model. Either set an API key in your environment:

```sh
export ANTHROPIC_API_KEY=sk-ant-...     # or DEEPSEEK_API_KEY, OPENROUTER_API_KEY, etc.
```

or put it in `config.kdl` (see [Configuration](#configuration)). Then:

```sh
clankit                # open the terminal UI in the current directory
clankit config         # show where config lives and which providers are set up
clankit --demo         # play a scripted session — no model or key needed
```

## CLI

```
clankit                          terminal UI in the current directory
clankit web                      web UI on http://127.0.0.1:8420
clankit run "prompt"             run one prompt with no UI and print the answer
clankit sessions                 list stored sessions for this directory
clankit skills                   list installed skills
clankit config                   show configuration status
```

Flags shared by the UI/`run` commands:

| Flag | Meaning |
|---|---|
| `-C, --cwd DIR` | work in `DIR` instead of the current directory |
| `-m, --model PROVIDER/MODEL` | model for this session |
| `--method NAME` | orchestration method for this session |
| `--permission-mode MODE` | `ask` / `accept-edits` / `auto` / `read-only` |
| `-c, --continue` | continue the most recent session in this directory |
| `-r, --resume ID` | resume a stored session by id |
| `--demo` | play a scripted session (no model, no API key) |

`clankit web` also takes `--host`, `--port` and `--no-browser`. `clankit run` takes `--json`
(one JSON object per event) and `-q/--quiet` (print only the final answer), and reads the
prompt from stdin when it is omitted or `-`.

Both UIs share the same `/slash` commands: `/help`, `/model [provider/model]`, `/method
[name]`, `/effort [low|medium|high|xhigh|max|default]`, `/theme [id]`, `/permissions [mode]`,
`/settings`, `/skills`, `/skillify`, `/init`, `/undo`, `/compact`, `/context`, `/cost`, `/new`,
`/resume`, `/quit`.

## Model providers

Clankit ships with a catalog of known providers. A known provider needs only an API key; a
custom one needs `kind` and `base-url` as well.

| Provider | Key (env var) | Notes |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | Claude models; the default provider |
| DeepSeek | `DEEPSEEK_API_KEY` | OpenAI-compatible, very cheap |
| OpenRouter | `OPENROUTER_API_KEY` | one key for hundreds of models, some free |
| Google Gemini | `GEMINI_API_KEY` | free tier for the listed models |
| Groq | `GROQ_API_KEY` | fast open-weight models; free plan |
| NVIDIA API Catalog | `NVIDIA_API_KEY` | free for development |
| Z.ai GLM | `ZAI_API_KEY` | Flash models are free |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` | small daily free allocation |
| Cerebras | `CEREBRAS_API_KEY` | paid (new accounts get a credit) |
| Ollama / LM Studio / llama.cpp / vLLM | — (keyless) | run on your own machine |

**Free models.** OpenRouter serves models whose ids end in `:free` (discovered at runtime,
since the lineup rotates); Gemini, Groq, NVIDIA and Z.ai have free tiers; and the local
providers cost nothing but your own hardware. Local providers are *keyless*: because Clankit
cannot know a local server is running, you switch one on by naming it in the config.

**Custom providers.** Any server that speaks the OpenAI chat API works as a provider:

```kdl
providers {
    my-server {
        kind "openai-compatible"
        base-url "http://localhost:11434/v1"
        keyless #true
    }
}
```

(To reach OpenAI itself you'd add `base-url "https://api.openai.com/v1"` and an
`api-key`/`api-key-env`; it is not a built-in catalog entry.)

## Configuration

Config lives at `~/.config/clankit/config.kdl` (KDL v2). Everything is optional: what is
missing or switched off (`/-`) keeps its default, and Clankit's settings screens edit the file
in place, leaving your comments and layout alone. A realistic file:

```kdl
/- kdl-version 2

model "anthropic/claude-opus-5-5"

roles {
    planner "anthropic/claude-opus-5-5"
    reviewer "deepseek/deepseek-v4-pro"
}

providers {
    anthropic {
        api-key-env "ANTHROPIC_API_KEY"
    }
    deepseek {
        api-key-env "DEEPSEEK_API_KEY"
    }
    openrouter {
        api-key "sk-or-your-key"
    }
    ollama
}

orchestration {
    method "task-tree"
    task-tree {
        max-depth 8
        max-parallel 4
        decomposition "adaptive"
    }
}

agent {
    effort "high"
    max-turns 200
    shell-timeout 120
}

permissions {
    mode "ask"
    allow "shell:git status*"
    allow "edit:docs/**"
}

skills {
    generate "ask"
}

ui {
    theme "dracula"
}

web {
    host "127.0.0.1"
    port 8420
}
```

Key reference (all defaults shown):

| Section | Keys (default) |
|---|---|
| top level | `model "anthropic/claude-opus-5-5"` — the default model, `provider/model` |
| `roles { }` | `subagent`, `planner`, `reviewer`, `judge`, `utility` — each a `provider/model`; unlisted roles use `model` |
| `providers { }` | per provider: `kind`, `name`, `api-key`, `api-key-env`, `base-url`, `keyless #false`, `enabled #true`, `concurrency`, plus `model "id" { name, context-window, max-output, tools, free, note, input-price, output-price }` |
| `orchestration { }` | `method "task-tree"`; each method's options in a block named after it |
| `agent { }` | `effort ""`, `max-turns 200`, `context-window 128000`, `compact-at 0.8`, `tool-output-limit 30000`, `shell-timeout 120`, `run-budget 1000000` |
| `permissions { }` | `mode "ask"`, `allow`, `deny` (rules are `category:pattern`) |
| `skills { }` | `directory ""`, `generate "ask"` (`off` / `ask` / `auto`) |
| `ui { }` | `theme "ayu-dark"`, `show-thinking #true`, `show-tree #true` |
| `web { }` | `host "127.0.0.1"`, `port 8420`, `open-browser #true` |
| `themes { }` | user-defined themes: `base <built-in>` plus token → `#rrggbb` |

The five `roles` let an orchestrator put different models on different jobs (e.g. a strong
planner, a cheap utility model for summaries and skill distillation).

## Orchestration methods

`method` picks the orchestrator; pick one with `--method`, `/method`, or `orchestration.method`.

| Method | What it does | Use when |
|---|---|---|
| `task-tree` | Agents split work into subtasks recursively; each subtask owns files, is accepted by commands the harness runs, and parents review and integrate. | Large tasks with genuinely independent parts. Small tasks cost about the same as Solo. |
| `solo` | One agent does the whole task in a single conversation, with a self-review of the final diff. | Anything that fits one context window; the fastest and cheapest baseline. |
| `review-loop` | One agent does the work, then a fresh reviewer with a clean context checks the diff and hands blocking findings back, in rounds. | Changes that will be merged without human review, security-sensitive code. |
| `test-loop` | A strong agent writes failing tests first; another implements until the harness sees them pass (with an optional debugging agent). | Bug fixes and well-specified features in projects with a runnable test command. |
| `architect-editor` | A strong "architect" explores and writes a plan; a cheaper "editor" makes the changes and can consult it. | Weak, free or local models doing work beyond them. |
| `escalate` | Tries a cheap model first and only climbs to stronger ones when the work fails a gate. | Mixed workloads where most requests are easy; saves money. |
| `best-of-n` | N agents build the change independently in isolated git worktrees; the harness tests them and a selector applies the best. | Hard, valuable changes where a wrong result is expensive. |
| `ultra` | The agent writes short workflow scripts that fan work out to many agents, verify findings adversarially and combine results. | Broad or high-stakes work: reviews, audits, research. |
| `demo` | Plays a scripted task-tree session through the normal event path. Calls no model, changes no files. | Trying the interface before any API key is configured. |

Per-method settings (stored under `orchestration { <method> { ... } }`):

- **task-tree** — `max-depth` 10, `max-nodes` 50, `max-children` 6, `max-parallel` 4,
  `decomposition` `adaptive|eager|minimal`, `cross-model` true, `fix-attempts` 1,
  `redispatch` true, `review` `off|final|branches`.
- **solo** — `self-review` true.
- **review-loop** — `max-rounds` 2, `reviewer-model` "", `prefer-other-family` false,
  `check` "", `include-minor` false.
- **test-loop** — `test-command` "", `test-paths` (default globs), `max-iterations` 5,
  `require-failing-first` true, `debugger-after` 2, `author-model` "".
- **architect-editor** — `architect-model` "", `editor-model` "", `max-consults` 3,
  `final-check` false.
- **escalate** — `tiers` "", `check` "", `attempts-per-tier` 1, `reset-between-tiers` false,
  `max-turns-per-tier` 60.
- **best-of-n** — `n` 3, `check` "", `write-tests` true, `selector-votes` 3, `approach-hints` "".
- **ultra** — `activation` `always|auto`, `max-concurrent` 6, `max-agents` 60,
  `workflow-budget` 1_500_000, `verify-votes` 3, `large-workflow` 25.
- **demo** — `speed` 100 (percent; 0 plays instantly).

The `model`-typed settings (`reviewer-model`, `author-model`, `architect-model`,
`editor-model`) take a `provider/model` value; empty means "use the role's model".

## Skills

A **skill** is a folder containing a `SKILL.md` file: YAML frontmatter between two `---` lines
(`name` and `description` are required) followed by Markdown instructions, plus any bundled
scripts and reference files. Optional frontmatter fields include `when_to_use`, `argument-hint`,
`disable-model-invocation`, `user-invocable` and `metadata`.

```markdown
---
name: release-notes
description: Generate release notes for this repository from merged PRs. Use when preparing a release.
---

Run `scripts/release.py --since <tag>` from the repo root, then edit the output into
`CHANGELOG.md`. Keep the "### Security" section even when empty.
```

Skills live in `~/.config/clankit/skills` (your library), and can also be read from
`<project>/.clankit/skills` and `<project>/.agents/skills` (project skills are listed but not
loaded until you trust the project). Agents see only each skill's name and description in
their system prompt; when a task matches, they call the `skill` tool to load the full
instructions ("progressive disclosure"). Bundled scripts are run with the ordinary shell tool,
so they pass the same permission checks as any command; `$ARGUMENTS` and
`${CLANKIT_SKILL_DIR}` are substituted when a skill is loaded.

**Automatic generation.** After each completed run, Clankit can distil what the session
learned into new skills (`skills.generate`: `off` / `ask` / `auto`, default `ask`). The
pipeline collects evidence from executed work, redacts secrets, has an agent on the cheap
*utility* model propose candidates, and passes them through deterministic gates (must cite
session steps, pass a safety scan and a syntax check, and not duplicate an existing skill).
Proposals wait in `~/.config/clankit/skills/.pending` for your approval; hand-written skills
are never rewritten. You can also trigger it explicitly with `/skillify`.

## Themes

Clankit ships 25 built-in themes, each one colour theme for the terminal, the web UI and
syntax highlighting: **Ayu Dark** (default), Ayu Mirage, Ayu Light, **Monokai Pro** (and its
Classic/Machine/Octagon/Ristretto/Spectrum/Light filters), **Dracula**, GitHub Dark/Dimmed/
Light, Catppuccin Mocha/Latte, Tokyo Night, Gruvbox Dark, Nord, One Dark, Solarized Dark/
Light, Rosé Pine, Kanagawa and Everforest Dark.

Switch with `/theme dracula`, set `ui { theme "dracula" }`, or pick from the settings screen
(F2). You can define your own in the `themes { }` section, starting from a built-in and
overriding single colours:

```kdl
themes {
    my-theme {
        base "ayu-dark"
        accent "#c08040"
    }
}
```

## UIs

**Terminal UI** — `clankit` (add `--demo` to explore without a model). Type to send a prompt,
`/` for commands, `f1` for keys, `f2` for settings, `f3` to review skills; `ctrl+t` toggles the
task tree, `ctrl+y` answers a pending permission/question, `esc` stops a run.

**Web UI** — `clankit web` serves the same interface at `http://127.0.0.1:8420` and prints a
login link containing a random token (`/?token=...`); keep it private. Use `--host`/`--port` to
bind elsewhere (only do so if other machines should reach it) and `--no-browser` to stop it
opening a window.

## Permissions

Four modes govern what agents may do without asking: `ask` (confirm edits, shell commands and
network access), `accept-edits` (stop asking about edits inside the workspace), `auto` (ask
nothing) and `read-only` (refuse anything that changes state). `allow`/`deny` rules refine the
mode, written `category:pattern` with shell wildcards (e.g. `shell:git status*`,
`edit:docs/**`); a deny rule always wins. Rules and the mode can also be set per session with
`--permission-mode` or `/permissions`.

## Development

```sh
uv run pytest          # test suite (asyncio_mode=auto)
uv run ruff check      # lint (and `uv run ruff format` to format)
```

The project targets Python 3.12; tests live in `tests/` and lint rules in `pyproject.toml`.
