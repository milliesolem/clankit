// The settings view: every setting the configuration file holds, grouped as the server
// declares them (clankit.config.fields.SECTIONS), plus providers and keys, the orchestration
// method with its own settings, the theme list with live preview, and the skill library.
// Every change is saved at once; the status row confirms the write to config.kdl.

import { api } from "./api.js";
import { button, clear, glyph, h } from "./dom.js";
import { swatch } from "./popup.js";

const SOURCE_WORDS = {
  config: "key in config file",
  env: "key from the environment",
  keyless: "no key needed",
  none: "not set up",
};

function fieldId(key) {
  return `set-${key.replace(/[^\w-]/g, "-")}`;
}

export class SettingsView {
  constructor(app) {
    this.app = app;
    this.el = document.getElementById("settings");
    this.editing = ""; // provider whose key entry is open
    this.addingProvider = false;
    this.confirming = ""; // skill awaiting a second click to delete
    this.errors = new Map(); // setting key -> message from the last failed save
    this.body = null;
  }

  // -- structure -------------------------------------------------------------------------------

  sections() {
    const state = this.app.state;
    const byId = Object.fromEntries(state.sections.map((s) => [s.id, s]));
    return [
      { id: "providers", title: "Providers", render: () => this.providers() },
      byId.models && { id: "models", title: byId.models.title, render: () => this.fields(byId.models) },
      { id: "orchestration", title: "Orchestration", render: () => this.orchestration() },
      byId.agents && { id: "agents", title: byId.agents.title, render: () => this.fields(byId.agents) },
      byId.permissions && { id: "permissions", title: byId.permissions.title, render: () => this.fields(byId.permissions) },
      byId.appearance && { id: "appearance", title: byId.appearance.title, render: () => this.fields(byId.appearance) },
      byId.skills && { id: "skills", title: byId.skills.title, render: () => [this.fields(byId.skills), this.library()] },
      byId.web && {
        id: "web",
        title: byId.web.title,
        render: () => [
          this.fields(byId.web),
          h("p", { class: "help", text: "Changes here take effect the next time the web interface starts." }),
        ],
      },
    ].filter(Boolean);
  }

  render(scrollTo = "") {
    if (!this.app.state) return;
    const focusKey = document.activeElement?.closest?.("#settings [data-key]")?.dataset.key;
    const scroll = this.body?.scrollTop ?? 0;
    clear(this.el);
    const sections = this.sections();
    const nav = h(
      "nav",
      { class: "settings-nav", "aria-label": "Settings sections" },
      h(
        "ul",
        {},
        sections.map((s) =>
          h("li", {}, h("a", { href: `#settings/${s.id}`, dataset: { section: s.id }, text: s.title.toLowerCase() })),
        ),
      ),
    );
    const content = h("div", { class: "settings-content" }, h("h1", { class: "title", text: "Settings" }));
    const warnings = this.app.state.config.warnings;
    if (warnings.length) {
      content.append(
        h("div", { class: "warnings" }, warnings.map((w) => h("p", { class: "notice" }, glyph("input"), h("span", { text: w })))),
      );
    }
    for (const section of sections) {
      content.append(
        h("section", { class: "settings-section", id: `section-${section.id}`, "aria-labelledby": `heading-${section.id}` },
          h("h2", { id: `heading-${section.id}`, text: section.title }),
          section.render(),
        ),
      );
    }
    this.body = h("div", { class: "settings-body" }, content);
    this.body.addEventListener("scroll", () => this.spy(), { passive: true });
    this.el.append(nav, this.body);
    nav.addEventListener("click", (event) => {
      const link = event.target.closest("a[data-section]");
      if (!link) return;
      event.preventDefault();
      this.scrollTo(link.dataset.section);
    });
    if (scrollTo) this.scrollTo(scrollTo);
    else this.body.scrollTop = scroll;
    if (focusKey) this.el.querySelector(`[data-key="${CSS.escape(focusKey)}"]`)?.focus({ preventScroll: true });
    this.spy();
  }

  scrollTo(id) {
    const target = this.el.querySelector(`#section-${id}`);
    if (!target || !this.body) return;
    this.body.scrollTop = target.offsetTop - 24;
    this.spy(id);
  }

  spy(forced = "") {
    if (!this.body) return;
    let current = forced;
    if (!current) {
      for (const section of this.el.querySelectorAll(".settings-section")) {
        if (section.offsetTop - 48 <= this.body.scrollTop) current = section.id.replace("section-", "");
      }
    }
    current ||= "providers";
    for (const link of this.el.querySelectorAll(".settings-nav a")) {
      link.classList.toggle("active", link.dataset.section === current);
      if (link.dataset.section === current) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    }
  }

  // -- saving ----------------------------------------------------------------------------------

  async save(path, value) {
    try {
      const result = await api.put(`/api/settings/${encodeURIComponent(path)}`, { value });
      this.errors.delete(path);
      this.app.applyState(result.state);
      this.app.status.saved(result.message, result.saved);
      this.app.refreshMeta();
    } catch (error) {
      this.errors.set(path, error.message);
      this.app.status.flash(error.message, "error");
    }
    this.render();
  }

  async act(request) {
    try {
      const result = await request();
      this.app.applyState(result.state);
      this.app.status.saved(result.message, result.saved);
      this.app.refreshMeta();
      return true;
    } catch (error) {
      this.app.status.flash(error.message, "error");
      return false;
    } finally {
      this.render();
    }
  }

  // -- generic rows ----------------------------------------------------------------------------

  row(key, label, control, help) {
    const error = this.errors.get(key);
    return h(
      "div",
      { class: "field" },
      h("label", { class: "field-label", for: fieldId(key), text: label }),
      h(
        "div",
        { class: "field-value" },
        control,
        error ? h("p", { class: "field-error" }, glyph("failed"), h("span", { text: error })) : null,
        help ? h("p", { class: "help", text: help }) : null,
      ),
    );
  }

  fields(section) {
    return h("div", { class: "fields" }, section.fields.map((field) => this.field(field)));
  }

  field(field) {
    const key = field.path;
    const save = (value) => this.save(key, value);
    let control;
    switch (field.kind) {
      case "bool":
        control = this.segments(key, [["on", true], ["off", false]], field.value, save);
        break;
      case "choice":
        control = this.segments(key, field.choices.map((c) => [c || field.empty || "default", c]), field.value, save);
        break;
      case "model":
        control = this.modelControl(key, field.value, field.empty, save, field.path !== "models.default");
        break;
      case "theme":
        control = this.themeList(field.value);
        break;
      default:
        control = this.textInput(key, field.value, field.empty, save, field.kind);
    }
    return this.row(key, field.label, control, field.help);
  }

  textInput(key, value, empty, save, kind = "string") {
    const shown = value === null || value === undefined ? "" : String(value);
    const input = h("input", {
      type: "text",
      id: fieldId(key),
      value: shown,
      placeholder: empty || "",
      inputmode: kind === "int" ? "numeric" : kind === "float" ? "decimal" : null,
      class: kind === "int" || kind === "float" ? "text number" : "text",
      spellcheck: "false",
      dataset: { key },
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") input.blur();
      if (event.key === "Escape") {
        event.stopPropagation();
        input.value = shown;
        input.blur();
      }
    });
    input.addEventListener("change", () => {
      if (input.value !== shown) save(input.value);
    });
    return input;
  }

  segments(key, options, value, save) {
    const group = h("div", { class: "segments", role: "radiogroup", id: fieldId(key) });
    options.forEach(([label, option]) => {
      const checked = option === value;
      const seg = h("button", {
        type: "button",
        role: "radio",
        class: "segment",
        "aria-checked": String(checked),
        text: label,
        dataset: { key: `${key}=${option}` },
        onclick: () => {
          if (!checked) save(option);
        },
      });
      group.append(seg);
    });
    group.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const items = [...group.children];
      const index = items.indexOf(document.activeElement);
      const next = items[(index + (event.key === "ArrowRight" ? 1 : -1) + items.length) % items.length];
      next?.focus();
      event.preventDefault();
    });
    return group;
  }

  modelControl(key, value, empty, save, clearable) {
    const trigger = h("button", {
      type: "button",
      class: "picker-button",
      id: fieldId(key),
      dataset: { key },
      text: value || empty || "choose a model",
    });
    trigger.classList.toggle("unset", !value);
    trigger.addEventListener("click", () =>
      this.app.pickModel({
        anchor: trigger,
        current: value,
        clearLabel: clearable ? `Use the ${empty || "default model"}` : "",
        onPick: (ref) => save(ref),
      }),
    );
    return trigger;
  }

  // -- appearance ------------------------------------------------------------------------------

  themeList(value) {
    const app = this.app;
    const list = h("div", { class: "theme-list", role: "radiogroup", id: fieldId("ui.theme"), "aria-label": "Theme" });
    const themes = app.state.themes;
    const rows = themes.map((theme) => {
      const checked = theme.id === value;
      return h(
        "button",
        {
          type: "button",
          role: "radio",
          class: "theme-row",
          "aria-checked": String(checked),
          tabindex: checked ? "0" : "-1",
          dataset: { key: `theme=${theme.id}`, theme: theme.id },
          onclick: () => {
            if (!checked) this.save("ui.theme", theme.id);
          },
          onfocus: () => app.previewTheme(theme.id),
        },
        h("span", { class: "swatches" }, theme.swatches.map(swatch)),
        h("span", { class: "theme-name", text: theme.name }),
        h("span", { class: "theme-id", text: theme.id }),
      );
    });
    list.append(...rows);
    list.addEventListener("keydown", (event) => {
      const moves = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };
      if (!(event.key in moves)) return;
      event.preventDefault();
      const index = rows.indexOf(document.activeElement);
      const next = rows[(index + moves[event.key] + rows.length) % rows.length];
      rows.forEach((row) => (row.tabIndex = row === next ? 0 : -1));
      next.focus();
    });
    list.addEventListener("focusout", (event) => {
      if (!list.contains(event.relatedTarget)) app.previewTheme(null);
    });
    return list;
  }

  // -- providers -------------------------------------------------------------------------------

  providers() {
    const rows = this.app.state.providers;
    const wrap = h(
      "div",
      { class: "providers" },
      h("p", {
        class: "help",
        text: "Keys are stored in config.kdl, which only your user can read. They are never shown again in full.",
      }),
      rows.map((row) => this.provider(row)),
    );
    wrap.append(this.addingProvider ? this.addProviderForm() : h("div", { class: "actions" },
      button("add a custom provider", { onclick: () => ((this.addingProvider = true), this.render()) }),
    ));
    return wrap;
  }

  provider(row) {
    const path = `/api/providers/${encodeURIComponent(row.id)}`;
    const patch = (body) => this.act(() => api.patch(path, body));
    let status;
    if (row.key_source === "config") status = `${SOURCE_WORDS.config}  ${row.masked_key}`;
    else if (row.key_source === "env") status = `key from $${row.api_key_env}  ${row.masked_key}`;
    else if (row.key_source === "keyless") status = row.enabled ? "local, switched on" : "local, switched off";
    else status = SOURCE_WORDS.none;
    const actions = h("div", { class: "row-actions" });
    if (row.key_source === "keyless") {
      actions.append(
        button(row.enabled ? "switch off" : "switch on", {
          onclick: () => patch({ enabled: !row.enabled }),
          cls: "small",
        }),
      );
    } else {
      actions.append(
        button(row.key_source === "config" ? "replace key" : "add key", {
          cls: "small",
          onclick: () => {
            this.editing = row.id;
            this.render();
            this.el.querySelector(`#${fieldId(`key-${row.id}`)}`)?.focus();
          },
        }),
      );
      if (row.key_source === "config") actions.append(button("remove key", { cls: "small", onclick: () => patch({ api_key: "" }) }));
    }
    if (!row.builtin) {
      actions.append(button("remove", { cls: "small", onclick: () => this.act(() => api.del(path)) }));
    }
    const notes = [row.note, row.base_url && !row.builtin ? row.base_url : ""].filter(Boolean).join("  ");
    return h(
      "div",
      { class: "provider" },
      h(
        "div",
        { class: "provider-line" },
        glyph(row.usable ? "done" : "pending"),
        h("span", { class: "provider-name", text: row.name }),
        h("span", { class: "provider-status", text: status }),
        actions,
      ),
      notes || row.signup_url
        ? h(
            "p",
            { class: "help" },
            notes,
            row.signup_url && row.key_source === "none"
              ? [notes ? "  " : "", h("a", { href: row.signup_url, target: "_blank", rel: "noopener noreferrer", text: "get a key" })]
              : null,
          )
        : null,
      this.editing === row.id ? this.keyEntry(row, patch) : null,
      row.key_source === "keyless" ? this.baseUrlEntry(row, patch) : null,
    );
  }

  keyEntry(row, patch) {
    const input = h("input", {
      type: "password",
      class: "text",
      id: fieldId(`key-${row.id}`),
      placeholder: `${row.name} API key`,
      autocomplete: "off",
      spellcheck: "false",
      "aria-label": `${row.name} API key`,
    });
    const submit = async () => {
      if (!input.value.trim()) return;
      const key = input.value;
      input.value = "";
      this.editing = "";
      await patch({ api_key: key });
    };
    const cancel = () => {
      this.editing = "";
      this.render();
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") submit();
      if (event.key === "Escape") {
        event.stopPropagation();
        cancel();
      }
    });
    return h("div", { class: "key-entry" }, input, button("save", { primary: true, onclick: submit }), button("cancel", { onclick: cancel }));
  }

  baseUrlEntry(row, patch) {
    const key = `base-${row.id}`;
    return h(
      "div",
      { class: "key-entry" },
      this.textInput(key, row.base_url, "default address", (value) => patch({ base_url: value })),
    );
  }

  addProviderForm() {
    const input = (name, label, type = "text") =>
      h("input", { type, class: "text", name, placeholder: label, "aria-label": label, spellcheck: "false", autocomplete: "off" });
    const id = input("id", "id, e.g. my-server");
    const url = input("base_url", "base URL, e.g. http://localhost:8000/v1");
    const key = input("api_key", "API key (leave empty if none is needed)", "password");
    let kind = "openai-compatible";
    const kinds = h("div", { class: "segments", role: "radiogroup", "aria-label": "Protocol" });
    for (const option of ["openai-compatible", "anthropic"]) {
      const seg = h("button", { type: "button", role: "radio", class: "segment", "aria-checked": String(option === kind), text: option });
      seg.addEventListener("click", () => {
        kind = option;
        for (const other of kinds.children) other.setAttribute("aria-checked", String(other === seg));
      });
      kinds.append(seg);
    }
    const submit = async () => {
      const ok = await this.act(() =>
        api.post("/api/providers", { id: id.value, base_url: url.value, kind, api_key: key.value }),
      );
      if (ok) this.addingProvider = false;
      this.render();
    };
    return h(
      "div",
      { class: "add-provider" },
      h("h3", { text: "Custom provider" }),
      id,
      url,
      kinds,
      key,
      h(
        "div",
        { class: "actions" },
        button("add provider", { primary: true, onclick: submit }),
        button("cancel", { onclick: () => ((this.addingProvider = false), this.render()) }),
      ),
    );
  }

  // -- orchestration ---------------------------------------------------------------------------

  orchestration() {
    const state = this.app.state;
    const current = state.config.method;
    const methods = state.methods;
    const list = h(
      "div",
      { class: "method-list", role: "radiogroup", "aria-label": "Orchestration method" },
      methods.map((method) =>
        h(
          "button",
          {
            type: "button",
            role: "radio",
            class: "method-row",
            "aria-checked": String(method.name === current),
            dataset: { key: `method=${method.name}` },
            onclick: () => method.name !== current && this.save("orchestration.method", method.name),
          },
          h("span", { class: "method-title", text: method.title }),
          h("span", { class: "method-summary", text: method.summary }),
        ),
      ),
    );
    const chosen = methods.find((m) => m.name === current);
    const parts = [list];
    const session = this.app.store.meta;
    if (session && session.method !== current) {
      parts.push(
        h("p", {
          class: "help",
          text: `The current session uses ${session.method_title} (chosen for that session only).`,
        }),
      );
    }
    if (chosen) {
      if (chosen.guidance) parts.push(h("p", { class: "help guidance", text: chosen.guidance }));
      if (chosen.settings.length) {
        parts.push(
          h("h3", { text: `${chosen.title} settings` }),
          h("div", { class: "fields" }, chosen.settings.map((s) => this.methodSetting(chosen, s))),
        );
      }
    }
    return parts;
  }

  methodSetting(method, setting) {
    const key = `orchestration.${method.name}.${setting.key}`;
    const save = (value) => this.save(key, value);
    let control;
    if (setting.kind === "bool") control = this.segments(key, [["on", true], ["off", false]], setting.value, save);
    else if (setting.kind === "choice") control = this.segments(key, setting.choices.map((c) => [c, c]), setting.value, save);
    else if (setting.kind === "model") control = this.modelControl(key, setting.value, "role default", save, true);
    else control = this.textInput(key, setting.value, String(setting.default ?? ""), save, setting.kind);
    const label = setting.key.replace(/-/g, " ");
    return this.row(key, label.charAt(0).toUpperCase() + label.slice(1), control, setting.help);
  }

  // -- skills ----------------------------------------------------------------------------------

  library() {
    const skills = this.app.state.skills;
    const parts = [];
    const path = (name, action) => `/api/skills/${encodeURIComponent(name)}${action ? `/${action}` : ""}`;
    if (skills.pending.length) {
      parts.push(h("h3", { text: "Proposed skills" }));
      for (const skill of skills.pending) {
        parts.push(
          h(
            "div",
            { class: "skill" },
            h("div", { class: "skill-line" }, glyph("input"), h("span", { class: "skill-name", text: skill.name })),
            h("p", { class: "skill-description", text: skill.description }),
            skill.reason ? h("p", { class: "help", text: `Why it was proposed: ${skill.reason}` }) : null,
            h(
              "div",
              { class: "actions" },
              button("approve", { primary: true, onclick: () => this.act(() => api.post(path(skill.name, "approve"))) }),
              button("discard", { onclick: () => this.act(() => api.post(path(skill.name, "reject"))) }),
            ),
          ),
        );
      }
    }
    parts.push(h("h3", { text: "Installed skills" }));
    if (!skills.active.length) {
      parts.push(h("p", { class: "help", text: `No skills installed yet. They live in ${skills.directory}.` }));
    }
    for (const skill of skills.active) {
      const confirming = this.confirming === skill.name;
      parts.push(
        h(
          "div",
          { class: "skill" },
          h(
            "div",
            { class: "skill-line" },
            glyph("done"),
            h("span", { class: "skill-name", text: skill.name }),
            h("span", { class: "skill-source", text: skill.source }),
            skill.source === "user"
              ? h(
                  "div",
                  { class: "row-actions" },
                  button(confirming ? "confirm delete" : "delete", {
                    cls: `small${confirming ? " danger" : ""}`,
                    onclick: () => {
                      if (!confirming) {
                        this.confirming = skill.name;
                        this.render();
                        return;
                      }
                      this.confirming = "";
                      this.act(() => api.del(path(skill.name)));
                    },
                  }),
                )
              : null,
          ),
          h("p", { class: "skill-description", text: skill.description }),
        ),
      );
    }
    if (skills.diagnostics.length) {
      parts.push(h("h3", { text: "Problems" }));
      for (const problem of skills.diagnostics) parts.push(h("p", { class: "notice" }, glyph("input"), h("span", { text: problem })));
    }
    return h("div", { class: "library" }, parts);
  }
}
