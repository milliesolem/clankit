"""The settings screen: categories on the chrome surface, a two-column form on the canvas.

Every change is written to config.kdl at once, through `Runtime.update_config`, and the
status row confirms it with the file's path; a value the configuration refuses leaves the
file alone and shows the reason under the field. The forms are built from declarations
that both interfaces share: `clankit.config.fields.SECTIONS` for general settings, the
provider helpers in `clankit.config.actions` (which never hand out a full API key), each
orchestration method's `Setting` list, and the skill library.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.widget import Widget
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from clankit.config import actions, fields
from clankit.config.schema import PROVIDER_KINDS, Config, ConfigError, Setting
from clankit.llm.types import ModelRef
from clankit.orchestration import all_orchestrators, get_orchestrator, method_names
from clankit.skills.document import SkillError
from clankit.skills.library import Skill
from clankit.themes import all_themes
from clankit.tui.editors import (
    ActionButton,
    Edited,
    FieldRow,
    Note,
    Pressed,
    SectionTitle,
    Segments,
    TextEditor,
    choice,
    toggle,
)
from clankit.tui.format import STATE_GLYPHS, STATE_STYLES, ellipsize, pad
from clankit.tui.pickers import theme_prompt
from clankit.tui.screens import ViewScreen

Saver = Callable[[Config, Any], None]

CATEGORIES: tuple[tuple[str, str], ...] = (
    ("providers", "Providers"),
    ("models", "Models"),
    ("orchestration", "Orchestration"),
    ("agents", "Agents"),
    ("permissions", "Permissions"),
    ("appearance", "Appearance"),
    ("skills", "Skills"),
    ("web", "Web interface"),
)

_FREE_TAGS = {"partly": "free tier", "yes": "free", "local": "local"}
_KEY_SOURCES = {
    "config": "key in config",
    "env": "key from environment",
    "keyless": "no key needed",
    "none": "not set up",
}


def display_path(path: Path | None) -> str:
    """A configuration path as the status row shows it, with the home directory as ``~``."""
    if path is None:
        return "this run only (no configuration file)"
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def set_method(config: Config, name: Any) -> None:
    if name not in method_names():
        raise ConfigError(f"unknown orchestration method {name!r}")
    config.orchestration.method = str(name)


def setting_saver(method: str, spec: Setting) -> Saver:
    """Store one orchestration setting after validating it as its declaration says."""

    def save(config: Config, value: Any) -> None:
        if spec.kind == "model" and str(value).strip():
            try:
                ModelRef.parse(str(value))
            except ValueError as exc:
                raise ConfigError(f"{spec.key}: {exc}") from exc
        config.orchestration.set_option(method, spec.key, spec.coerce(value))

    return save


class ThemeList(OptionList):
    """Every theme with its swatches; the highlighted one is previewed live.

    Enter keeps the highlighted theme, esc (or leaving the list) puts the saved one back.
    """

    DEFAULT_CSS = """
    ThemeList {
        height: auto;
        max-height: 16;
        border-left: blank;
    }
    ThemeList:focus {
        border-left: outer $accent;
    }
    """

    def __init__(self, config: Config) -> None:
        themes = all_themes(config)
        super().__init__(
            *(
                Option(theme_prompt(theme, config.ui.theme), id=theme.id)
                for theme in themes.values()
            ),
            compact=True,
        )
        self.saved = config.ui.theme
        ids = list(themes)
        self._initial = ids.index(self.saved) if self.saved in ids else 0

    def on_mount(self) -> None:
        self.highlighted = self._initial

    @property
    def previewing(self) -> bool:
        return self.app.theme != self.saved

    @on(OptionList.OptionHighlighted)
    def _preview(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        if self.has_focus and event.option.id in self.app.available_themes:
            self.app.theme = event.option.id

    @on(OptionList.OptionSelected)
    def _keep(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id is not None:
            self.post_message(Edited("ui.theme", event.option.id))

    def revert(self) -> None:
        """Put the saved theme back, and the highlight on it."""
        if self.saved in self.app.available_themes:
            self.app.theme = self.saved
        ids = [option.id for option in self._options]
        if self.saved in ids:
            self.highlighted = ids.index(self.saved)

    def on_blur(self) -> None:
        if self.previewing:
            self.revert()


class SettingsScreen(ViewScreen):
    """All settings, by category."""

    DEFAULT_CSS = """
    SettingsScreen > #settings-body {
        height: 1fr;
    }
    SettingsScreen #categories-column {
        width: 24;
        height: 1fr;
        background: $bg-alt;
        padding: 1 0;
    }
    SettingsScreen #categories {
        height: auto;
        background: $bg-alt;
    }
    SettingsScreen #categories > .option-list--option {
        padding: 0 2;
    }
    SettingsScreen #form {
        width: 1fr;
        height: 1fr;
        padding: 0 4 1 4;
        background: $background;
    }
    SettingsScreen #form > * {
        max-width: 110;
    }
    SettingsScreen .detail {
        height: auto;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $bg-raised;
    }
    SettingsScreen .buttons {
        height: auto;
        margin: 0 0 1 27;
    }
    SettingsScreen .list {
        height: auto;
        max-height: 16;
        margin: 0 0 1 0;
        border-left: blank;
    }
    SettingsScreen .list:focus {
        border-left: outer $accent;
    }
    SettingsScreen .well {
        background: $bg-alt;
        padding: 1 2;
        height: auto;
        margin: 1 0 0 0;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "close", "Back", show=False)]

    title_text = "settings"
    title_keys: ClassVar[list[tuple[str, str]]] = [
        ("tab", "next field"),
        ("enter", "save"),
        ("esc", "back"),
    ]

    def __init__(self, category: str | None = None) -> None:
        super().__init__()
        self.category = category
        self.categories = OptionList(
            *(Option(title, id=cid) for cid, title in CATEGORIES), id="categories", compact=True
        )
        self.form = VerticalScroll(id="form")
        self._savers: dict[str, Saver] = {}
        self._rows: dict[str, FieldRow] = {}
        self._provider: str | None = None
        self._skill: str | None = None
        self._method: str | None = None
        self._new_provider: dict[str, Any] = {}
        self._confirm_delete: str | None = None

    @property
    def runtime(self):  # type: ignore[no-untyped-def]
        return self.clankit.runtime

    def compose(self) -> ComposeResult:
        yield self.title_bar
        with Horizontal(id="settings-body"):
            with Vertical(id="categories-column"):
                yield self.categories
            yield self.form
        yield self.status_bar

    async def on_mount(self) -> None:
        if self.category is None:
            no_model = self.runtime.providers.default_model() is None
            self.category = "providers" if no_model else "models"
        ids = [cid for cid, _ in CATEGORIES]
        self.categories.highlighted = ids.index(self.category) if self.category in ids else 0
        await self.show_category(self.category)
        self.categories.focus()
        self.set_status(
            Content.styled(f"configuration  {display_path(self.runtime.store.path)}", "$fg-muted")
        )

    @on(OptionList.OptionHighlighted, "#categories")
    async def _category(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id != self.category:
            await self.show_category(event.option.id)

    @on(OptionList.OptionSelected, "#categories")
    def _enter_form(self) -> None:
        self.focus_next()

    async def show_category(self, category: str, *, focus: str | None = None) -> None:
        """Build the form of `category` from the current configuration."""
        self.category = category
        self._savers.clear()
        self._rows.clear()
        builders: dict[str, Callable[[], list[Widget]]] = {
            "providers": self._providers,
            "orchestration": self._orchestration,
            "appearance": self._appearance,
            "skills": self._skills,
        }
        widgets = builders.get(category, lambda: self._section(category))()
        scroll_y = self.form.scroll_y if focus else 0
        with self.app.batch_update():
            await self.form.remove_children()
            await self.form.mount_all(widgets)
        self.form.scroll_to(y=scroll_y, animate=False)
        if focus is not None:
            target = next(
                (w for w in self.form.query("*") if getattr(w, "key", None) == focus), None
            )
            if target is not None and target.focusable:
                target.focus()

    # -- building blocks ------------------------------------------------------------------------

    def _row(
        self, key: str, label: str, control: Widget, help_text: str, saver: Saver | None
    ) -> FieldRow:
        row = FieldRow(key, label, control, help_text)
        self._rows[key] = row
        if saver is not None:
            self._savers[key] = saver
        return row

    def _model_suggestions(self) -> list[str]:
        return [str(info.ref) for info in self.runtime.providers.known_models()]

    def _field_control(self, field: fields.Field, config: Config) -> Widget:
        value = fields.get_value(config, field.path)
        if field.kind == "bool":
            return toggle(field.path, bool(value))
        if field.kind == "choice":
            return choice(field.path, field.choices, str(value), field.empty)
        if field.kind == "model":
            return TextEditor(
                field.path,
                value_text(value),
                placeholder=field.empty or "provider/model",
                suggestions=self._model_suggestions(),
            )
        return TextEditor(field.path, value_text(value), placeholder=field.empty)

    def _section(self, section_id: str) -> list[Widget]:
        section = next((s for s in fields.SECTIONS if s.id == section_id), None)
        if section is None:
            return [Note("Nothing to set here.")]
        config = self.runtime.config
        widgets: list[Widget] = [SectionTitle(section.title)]
        for field in section.fields:
            if field.kind == "theme":
                continue
            widgets.append(
                self._row(
                    field.path,
                    field.label,
                    self._field_control(field, config),
                    field.help,
                    lambda c, v, path=field.path: fields.set_value(c, path, v),
                )
            )
        return widgets

    # -- providers ------------------------------------------------------------------------------

    def _providers(self) -> list[Widget]:
        rows = actions.provider_rows(self.runtime.config)
        if self._provider not in {r.id for r in rows}:
            self._provider = next((r.id for r in rows if r.usable), rows[0].id if rows else None)
        provider_list = OptionList(
            *(Option(self._provider_line(row), id=row.id) for row in rows),
            id="provider-list",
            classes="list",
            compact=True,
        )
        ids = [row.id for row in rows]
        if self._provider in ids:
            provider_list.highlighted = ids.index(self._provider)
        widgets: list[Widget] = [
            SectionTitle("providers"),
            Note(
                "Keys are saved in the configuration file, which only you can read. A key in "
                "the provider's environment variable works too and is never written anywhere."
            ),
            provider_list,
            Vertical(*self._provider_detail(), id="provider-detail", classes="detail"),
        ]
        widgets.extend(self._add_provider_form())
        return widgets

    @staticmethod
    def _provider_line(row: actions.ProviderRow) -> Content:
        if row.usable:
            glyph = (STATE_GLYPHS["done"], STATE_STYLES["done"])
        elif row.url_problem:
            glyph = (STATE_GLYPHS["failed"], STATE_STYLES["failed"])
        elif row.key_source == "keyless" and not row.enabled:
            glyph = (STATE_GLYPHS["cancelled"], STATE_STYLES["cancelled"])
        else:
            glyph = (STATE_GLYPHS["pending"], STATE_STYLES["pending"])
        status = _KEY_SOURCES.get(row.key_source, row.key_source)
        if row.url_problem:
            status = row.url_problem
        elif row.key_source == "keyless" and not row.enabled:
            status = "off"
        return Content.assemble(
            glyph,
            "  ",
            (pad(row.name, 24), "$text"),
            (pad(_FREE_TAGS.get(row.free, ""), 11), "$fg-muted"),
            (pad(status, 22), "$fg-muted"),
            (row.masked_key, "$fg-muted"),
        )

    def _provider_detail(self) -> list[Widget]:
        rows = {row.id: row for row in actions.provider_rows(self.runtime.config)}
        row = rows.get(self._provider or "")
        if row is None:
            return [Note("Choose a provider above.")]
        pid = row.id
        tag = _FREE_TAGS.get(row.free, "")
        heading = Content.assemble(
            (row.name, "bold $text"),
            "   ",
            (row.id, "$fg-muted"),
            "   ",
            (row.kind, "$fg-muted"),
            ("   " + tag if tag else "", "$fg-muted"),
        )
        widgets: list[Widget] = [Static(heading, classes="detail-heading")]
        if row.url_problem:
            widgets.append(Note(f"Not usable yet: {row.url_problem}."))
        if row.note:
            widgets.append(Note(row.note))
        if row.signup_url:
            widgets.append(Note(f"Get a key: {row.signup_url}"))
        if row.key_source != "keyless":
            placeholder = (
                f"{row.masked_key}   type a new key to replace it"
                if row.masked_key
                else "paste a key and press enter"
            )
            env = f" When none is set here, ${row.api_key_env} is used." if row.api_key_env else ""
            widgets.append(
                self._row(
                    f"provider:{pid}:key",
                    "API key",
                    TextEditor(
                        f"provider:{pid}:key",
                        "",
                        placeholder=placeholder,
                        password=True,
                        keep=False,
                    ),
                    f"Saved to the configuration file on enter.{env}",
                    lambda c, v: actions.set_api_key(c, pid, str(v)),
                )
            )
        else:
            widgets.append(
                self._row(
                    f"provider:{pid}:enabled",
                    "Enabled",
                    toggle(f"provider:{pid}:enabled", row.enabled),
                    "Switch on once the local server is running.",
                    lambda c, v: actions.set_enabled(c, pid, bool(v)),
                )
            )
        user_entry = self.runtime.config.providers.get(pid)
        own_url = user_entry.base_url if (user_entry and row.builtin) else row.base_url
        widgets.append(
            self._row(
                f"provider:{pid}:base-url",
                "Base URL",
                TextEditor(
                    f"provider:{pid}:base-url",
                    own_url or "",
                    placeholder=row.base_url or "the provider's own address",
                ),
                "Leave empty for the provider's usual address." if row.builtin else "",
                lambda c, v: actions.set_base_url(c, pid, str(v)),
            )
        )
        buttons: list[Widget] = []
        if row.key_source == "config":
            buttons.append(ActionButton(f"provider:{pid}:remove-key", "Remove key"))
        if not row.builtin:
            buttons.append(ActionButton(f"provider:{pid}:remove", "Remove provider", danger=True))
        if buttons:
            widgets.append(Horizontal(*buttons, classes="buttons"))
        used = actions.models_using(self.runtime.config, pid)
        if used and not row.usable:
            widgets.append(
                Note(f"Used by: {', '.join(used)}. Those settings need a usable provider.")
            )
        return widgets

    def _add_provider_form(self) -> list[Widget]:
        state = self._new_provider
        kinds = PROVIDER_KINDS
        return [
            SectionTitle("add a provider"),
            Note(
                "Any server that speaks the OpenAI or Anthropic API, such as a self-hosted model."
            ),
            self._row(
                "new:id",
                "Id",
                TextEditor("new:id", state.get("id", ""), placeholder="my-server"),
                "Lowercase letters, digits and hyphens.",
                None,
            ),
            self._row(
                "new:base-url",
                "Base URL",
                TextEditor(
                    "new:base-url",
                    state.get("base-url", ""),
                    placeholder="http://localhost:8000/v1",
                ),
                "",
                None,
            ),
            self._row(
                "new:kind",
                "Protocol",
                choice("new:kind", kinds, state.get("kind", kinds[-1])),
                "",
                None,
            ),
            self._row(
                "new:key",
                "API key",
                TextEditor("new:key", "", placeholder="optional", password=True),
                "",
                None,
            ),
            self._row(
                "new:keyless",
                "Needs no key",
                toggle("new:keyless", bool(state.get("keyless", False))),
                "",
                None,
            ),
            Horizontal(ActionButton("new:add", "Add provider"), classes="buttons"),
        ]

    async def _add_provider(self) -> None:
        state = self._new_provider
        row = self._rows.get("new:id")
        try:
            self.runtime.update_config(
                lambda c: actions.add_custom_provider(
                    c,
                    str(state.get("id", "")),
                    base_url=str(state.get("base-url", "")),
                    kind=str(state.get("kind", PROVIDER_KINDS[-1])),
                    api_key=str(state.get("key", "")),
                    keyless=bool(state.get("keyless", False)),
                )
            )
        except ConfigError as exc:
            if row is not None:
                row.show_error(str(exc))
            self._failed(str(exc))
            return
        self._provider = str(state.get("id", "")).strip().lower()
        self._new_provider = {}
        self._saved()
        await self.show_category("providers")

    async def _refresh_providers(self, *, focus: str) -> None:
        """Repaint the provider list in place and rebuild the open provider's details."""
        provider_list = self._find("#provider-list")
        if not isinstance(provider_list, OptionList):
            return  # the form changed under us; the next build starts from fresh state
        for row in actions.provider_rows(self.runtime.config):
            provider_list.replace_option_prompt(row.id, self._provider_line(row))
        await self._rebuild("#provider-detail", self._provider_detail)
        target = next((w for w in self.form.query("*") if getattr(w, "key", None) == focus), None)
        if target is not None and target.focusable:
            target.focus()

    @on(OptionList.OptionHighlighted, "#provider-list")
    async def _provider_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id != self._provider:
            self._provider = event.option.id
            await self._rebuild("#provider-detail", self._provider_detail)

    def _find(self, selector: str) -> Widget | None:
        """The first widget matching `selector`, or None when the form was rebuilt since."""
        return next(iter(self.form.query(selector)), None)

    async def _rebuild(self, selector: str, build: Callable[[], list[Widget]]) -> None:
        container = self._find(selector)
        if container is None:
            return  # a category switch rebuilt the form while this rebuild was queued
        for key in [k for k, row in self._rows.items() if row in container.query(FieldRow)]:
            self._rows.pop(key, None)
            self._savers.pop(key, None)
        with self.app.batch_update():
            await container.remove_children()
            await container.mount_all(build())

    # -- orchestration --------------------------------------------------------------------------

    def _orchestration(self) -> list[Widget]:
        config = self.runtime.config
        current = config.orchestration.method
        methods = all_orchestrators()
        method_list = OptionList(
            *(
                Option(
                    Content.assemble(
                        (pad(m.title, 20), "bold $text" if m.name == current else "$text"),
                        (pad("default" if m.name == current else "", 9), "$fg-muted"),
                        (ellipsize(m.summary, 72), "$fg-muted"),
                    ),
                    id=m.name,
                )
                for m in methods
            ),
            id="method-list",
            classes="list",
            compact=True,
        )
        names = [m.name for m in methods]
        self._method = self._method if self._method in names else current
        if self._method in names:
            method_list.highlighted = names.index(self._method)
        widgets: list[Widget] = [
            SectionTitle("method"),
            Note(
                "How a prompt becomes agent work. Enter makes the highlighted method the default."
            ),
            method_list,
            Static(self._guidance(self._method), id="method-guidance", classes="detail"),
        ]
        override = self.clankit.session.method_override
        if override and override != current:
            widgets.append(
                Note(
                    f"This session uses {override}, chosen on the command line; the default applies to new sessions."
                )
            )
        try:
            selected = get_orchestrator(current)
        except KeyError:
            return widgets
        widgets.append(SectionTitle(f"{selected.title} settings"))
        if not selected.settings:
            widgets.append(Note("This method has no settings."))
        for spec in selected.settings:
            raw = config.orchestration.option(selected.name, spec.key, spec.default)
            key = f"method:{selected.name}:{spec.key}"
            widgets.append(
                self._row(
                    key,
                    spec.key.replace("-", " ").capitalize(),
                    self._setting_control(key, spec, raw),
                    spec.help,
                    setting_saver(selected.name, spec),
                )
            )
        return widgets

    def _setting_control(self, key: str, spec: Setting, value: Any) -> Widget:
        if spec.kind == "bool":
            return toggle(key, bool(value))
        if spec.kind == "choice":
            return choice(key, spec.choices, str(value))
        if spec.kind == "model":
            return TextEditor(
                key,
                value_text(value),
                placeholder="role default",
                suggestions=self._model_suggestions(),
            )
        return TextEditor(key, value_text(value))

    @staticmethod
    def _guidance(name: str | None) -> Content:
        try:
            method = get_orchestrator(name or "")
        except KeyError:
            return Content("")
        return Content.assemble(
            (method.title, "bold"), "\n", Content(method.guidance or method.summary)
        )

    @on(OptionList.OptionHighlighted, "#method-list")
    def _method_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._method = event.option.id
        guidance = self._find("#method-guidance")
        if isinstance(guidance, Static):
            guidance.update(self._guidance(self._method))

    @on(OptionList.OptionSelected, "#method-list")
    async def _method_selected(self, event: OptionList.OptionSelected) -> None:
        name = event.option.id
        if name is None or name == self.runtime.config.orchestration.method:
            return
        try:
            self.runtime.update_config(lambda c: set_method(c, name))
        except ConfigError as exc:
            self._failed(str(exc))
            return
        self._saved()
        await self.show_category("orchestration")
        method_list = self._find("#method-list")
        if method_list is not None:
            method_list.focus()

    # -- appearance -----------------------------------------------------------------------------

    def _appearance(self) -> list[Widget]:
        config = self.runtime.config
        widgets: list[Widget] = [
            SectionTitle("theme"),
            Note(
                "Moving through the list previews a theme. Enter keeps it; esc puts the saved one back."
            ),
            ThemeList(config).add_class("list"),
        ]
        self._savers["ui.theme"] = lambda c, v: fields.set_value(c, "ui.theme", v)
        widgets.extend(self._section("appearance")[1:])
        return widgets

    # -- skills ---------------------------------------------------------------------------------

    def _skills(self) -> list[Widget]:
        library = self.runtime.skills
        library.refresh()
        widgets = self._section("skills")
        pending = library.pending()
        installed = library.skills
        everything = [*pending, *installed]
        names = [self._skill_id(s) for s in everything]
        if self._skill not in names:
            self._skill = names[0] if names else None
        widgets.append(SectionTitle("library"))
        widgets.append(Note(f"Skills live in {library.directories[0]}."))
        if not everything:
            widgets.append(
                Note(
                    "No skills yet. Clankit proposes skills from finished work, or you can add SKILL.md folders yourself."
                )
            )
        else:
            skill_list = OptionList(
                *(Option(self._skill_line(s), id=self._skill_id(s)) for s in everything),
                id="skill-list",
                classes="list",
                compact=True,
            )
            if self._skill in names:
                skill_list.highlighted = names.index(self._skill)
            widgets.append(skill_list)
            widgets.append(Vertical(*self._skill_detail(), id="skill-detail", classes="detail"))
        for problem in library.diagnostics:
            widgets.append(Note(f"! {problem}"))
        return widgets

    @staticmethod
    def _skill_id(skill: Skill) -> str:
        return f"{'pending' if skill.pending else 'skill'}:{skill.name}"

    @staticmethod
    def _skill_line(skill: Skill) -> Content:
        if skill.pending:
            reason = skill.metadata.get("clankit.reason", "") or skill.description
            return Content.assemble(
                ("! ", "$warning"),
                (pad(skill.name, 28), "$text"),
                (pad("proposed", 10), "$fg-muted"),
                (ellipsize(reason, 70), "$fg-muted"),
            )
        return Content.assemble(
            ("  ", ""),
            (pad(skill.name, 28), "$text"),
            (pad(skill.source, 10), "$fg-muted"),
            (ellipsize(skill.description, 70), "$fg-muted"),
        )

    def _current_skill(self) -> Skill | None:
        if self._skill is None:
            return None
        kind, _, name = self._skill.partition(":")
        library = self.runtime.skills
        if kind == "pending":
            return next((s for s in library.pending() if s.name == name), None)
        return library.get(name)

    def _skill_detail(self) -> list[Widget]:
        skill = self._current_skill()
        if skill is None:
            return [Note("Choose a skill above.")]
        widgets: list[Widget] = [
            Static(
                Content.assemble((skill.name, "bold"), "   ", (skill.source, "$fg-muted")),
                classes="detail-heading",
            ),
            Static(Content(skill.description)),
        ]
        reason = skill.metadata.get("clankit.reason", "")
        if skill.pending and reason:
            widgets.append(Static(Content.assemble(("Why: ", "$fg-muted"), Content(reason))))
        widgets.append(Note(str(skill.path)))
        for warning in skill.warnings:
            widgets.append(Note(f"! {warning}"))
        try:
            body = skill.body()
        except OSError as exc:
            body = f"(cannot read SKILL.md: {exc})"
        lines = body.strip().splitlines()
        preview = "\n".join(lines[:40]) + (
            f"\n... {len(lines) - 40} more lines" if len(lines) > 40 else ""
        )
        widgets.append(Static(Content(preview), classes="well"))
        if skill.pending:
            buttons = [
                ActionButton(f"skill:approve:{skill.name}", "Approve"),
                ActionButton(f"skill:reject:{skill.name}", "Reject", danger=True),
            ]
        elif skill.source == "user":
            label = "Press again to delete" if self._confirm_delete == skill.name else "Delete"
            buttons = [ActionButton(f"skill:delete:{skill.name}", label, danger=True)]
        else:
            buttons = []
        if buttons:
            widgets.append(Horizontal(*buttons, classes="buttons"))
        return widgets

    @on(OptionList.OptionHighlighted, "#skill-list")
    async def _skill_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id != self._skill:
            self._skill = event.option.id
            self._confirm_delete = None
            await self._rebuild("#skill-detail", self._skill_detail)

    async def _skill_action(self, action: str, name: str) -> None:
        library = self.runtime.skills
        try:
            if action == "approve":
                library.approve(name)
                self._skill = f"skill:{name}"
                message = f"Approved {name}; agents can use it now."
            elif action == "reject":
                library.reject(name)
                message = f"Rejected the proposal {name}."
            else:
                if self._confirm_delete != name:
                    self._confirm_delete = name
                    await self._rebuild("#skill-detail", self._skill_detail)
                    button = self._find("#skill-detail ActionButton")
                    if button is not None:
                        button.focus()
                    return
                library.delete(name)
                self._confirm_delete = None
                message = f"Deleted {name}."
        except (KeyError, SkillError, OSError) as exc:
            self._failed(str(exc).strip("'\""))
            return
        self.set_status(Content.styled(message, "$text"))
        await self.show_category("skills")

    # -- saving ---------------------------------------------------------------------------------

    @on(Edited)
    async def _edited(self, message: Edited) -> None:
        message.stop()
        if message.key.startswith("new:"):
            self._new_provider[message.key.removeprefix("new:")] = message.value
            return
        saver = self._savers.get(message.key)
        if saver is None:
            return
        row = self._rows.get(message.key)
        try:
            self.runtime.update_config(lambda c: saver(c, message.value))
        except ConfigError as exc:
            if row is not None:
                row.show_error(str(exc))
            self._failed(str(exc))
            return
        if row is not None:
            row.clear_error()
            if isinstance(row.control, TextEditor):
                row.control.accepted()
        if message.key == "ui.theme":
            for theme_list in self.form.query(ThemeList):
                theme_list.saved = str(message.value)
        self._saved()
        if message.key.startswith("provider:"):
            await self._refresh_providers(focus=message.key)

    @on(Pressed)
    async def _pressed(self, message: Pressed) -> None:
        message.stop()
        key = message.key
        if key == "new:add":
            await self._add_provider()
            return
        if key.startswith("skill:"):
            _, action, name = key.split(":", 2)
            await self._skill_action(action, name)
            return
        if key.startswith("provider:"):
            _, pid, action = key.split(":", 2)
            mutate: Callable[[Config], None] = (
                (lambda c: actions.set_api_key(c, pid, ""))
                if action == "remove-key"
                else (lambda c: actions.remove_provider(c, pid))
            )
            try:
                self.runtime.update_config(mutate)
            except ConfigError as exc:
                self._failed(str(exc))
                return
            self._saved()
            if action == "remove":
                await self.show_category("providers")
            else:
                await self._refresh_providers(focus=f"provider:{pid}:key")

    def _saved(self) -> None:
        when = time.strftime("%H:%M:%S")
        self.set_status(
            Content.assemble(
                ("saved  ", "$text"),
                (display_path(self.runtime.store.path), "$fg-muted"),
                f"  {when}",
            )
        )

    def _failed(self, text: str) -> None:
        self.set_status(Content.assemble(("not saved  ", "bold $text"), Content(text)))

    # -- leaving --------------------------------------------------------------------------------

    def action_close(self) -> None:
        theme_list = next(iter(self.form.query(ThemeList)), None)
        if theme_list is not None and theme_list.previewing:
            theme_list.revert()
            if theme_list.has_focus:
                return
        self.dismiss()


__all__ = ["CATEGORIES", "Segments", "SettingsScreen", "ThemeList", "display_path"]
