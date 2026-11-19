
Make a coding agent named `Clankit`, written in python with uv with unit tests written in pytest, and pyramid and textual for UIs.

## Model support

Should *at least* support Anthropic's models, DeepSeek and freely available models. Configurable by adding API key.

## UI and visual design

It should have both a terminal-ui and a web-ui. Visual design should be minimalist but tasteful, but support different color themes; I really like Ayu Dark -- should also have other favourites like Monokai Pro, Dracula and Github Dark.

Avoid sloppy anti-patterns: no rounder corners, 1px borders, gradients, overdone typgraphy.

## Configuration

All configuration, including API keys, lives in a well-formated and human-readable KDL Document Language file under `~/.config/clankit/config.kdl`. Configuration may be done from the UI, which updates this file.

## Skills and tooling

Should have support for skills in the form of a directory under `~/.config/clankit/skills` with SKILL.md files and tooling scripts, similar to how most coding agents do it. There should also be a feature to automatically generated and supply this skill bank with new skills and tools based on work done in the current session.

## Orchistration

Clankit's primary orchitration method is constructing a "Task Tree" with a user-configurable depth-limit; default is 10. The user prompt should be broken down into subtasks, then those subtasks would recursively be broken down into subtasks until an agent is satisfied with the simplicity of the task to stop deploying subagents and become a leaf agent. The leaf agents do the building, the intermediate agents look over its subagents' work, does QA and conflict resolution, then glues it all together into a working result of its task, and the root agent reports back to the user and takes input.

Orchistration methods should be configurable, research other methods and add the best ones you find. Something like Claude Code's Ultracode might also be a nice addition. Focus on what has empirically been shown to produce the best results. Publicly available research papers on vibecoding, AI and computational linguistics might give good insight to what works and what doesn't.

Clankit should also have cross-model support: an agent can opt to deploy subagents using a different model than itself.

