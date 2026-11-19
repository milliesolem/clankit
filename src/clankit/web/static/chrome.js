// The two chrome rows: the crumb at the top (where you are, how to get elsewhere) and the
// status row at the bottom (model, tokens, cost, context, pending requests, method, theme,
// and the confirmation of whatever was just done).

import { clear, glyph, h } from "./dom.js";
import { money, plural, timeOfDay, tokens, usageTotal } from "./format.js";

function linkButton(text, onclick, title) {
  return h("button", { type: "button", class: "link", text, onclick, title });
}

export class Crumb {
  constructor(app) {
    this.app = app;
    this.el = document.getElementById("crumb");
  }

  render() {
    const app = this.app;
    clear(this.el);
    const path = h("nav", { class: "crumb-path", "aria-label": "Location" });
    if (app.view === "settings") {
      path.append(h("span", { class: "here", text: "settings" }));
    } else {
      // The root of the path leads back to the conversation; the rest to their own transcripts.
      const nodes = app.focus ? app.store.path(app.focus) : [];
      const here = (text) => h("span", { class: "here", text, "aria-current": "location" });
      const sep = () => h("span", { class: "sep", "aria-hidden": "true", text: "/" });
      if (!nodes.length) path.append(here(app.store.meta?.title || "New session"));
      nodes.forEach((node, index) => {
        if (index) path.append(sep());
        if (index === 0) path.append(linkButton(node.title, () => app.focusNode(""), "Back to the conversation (Esc)"));
        else if (index === nodes.length - 1) path.append(here(node.title));
        else path.append(linkButton(node.title, () => app.focusNode(node.id)));
      });
      if (nodes.length === 1) path.append(sep(), here("root agent"));
    }
    const actions = h("div", { class: "crumb-actions" });
    if (app.view === "settings") {
      actions.append(linkButton("back to session", () => app.closeSettings(), "Esc"));
    } else {
      actions.append(
        linkButton("new", () => app.newSession(), "Start a new session (/new)"),
        linkButton("sessions", () => app.openView("sessions"), "Reopen an earlier session (/resume)"),
        linkButton("settings", () => app.openSettings(), "Settings (/settings)"),
      );
    }
    this.el.append(path, actions);
  }
}

export class StatusRow {
  constructor(app) {
    this.app = app;
    this.el = document.getElementById("status");
    this.message = h("span", { class: "message", role: "status", "aria-live": "polite" });
    this.timer = 0;
    this.connection = "";
  }

  flash(text, level = "info") {
    clear(this.message);
    if (level === "error") this.message.append(glyph("failed"), " ");
    else if (level === "warning") this.message.append(glyph("input"), " ");
    else if (level === "saved") this.message.append(glyph("done"), " ");
    this.message.append(text);
    this.message.title = text;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => clear(this.message), level === "error" ? 20000 : 10000);
  }

  saved(text, path) {
    this.flash(path ? `${text}  ${path}  ${timeOfDay()}` : text, "saved");
  }

  setConnection(text) {
    this.connection = text;
    this.render();
  }

  render() {
    const app = this.app;
    const store = app.store;
    const meta = store.meta;
    const items = [];
    const model = (store.running && store.run?.model) || meta?.model;
    if (model || !meta?.demo) {
      items.push(linkButton(model || "no model", () => app.openView("models"), "Choose the model (/model)"));
    }
    const total = usageTotal(store.total);
    if (total) items.push(h("span", { text: `${tokens(total)} tokens` }));
    if (meta?.cost) items.push(h("span", { text: money(meta.cost), title: "Estimated cost of this session" }));
    if (meta?.context?.window) {
      const percent = Math.round((meta.context.used * 100) / meta.context.window);
      items.push(h("span", { text: `ctx ${percent}%`, title: "How full the conversation's context window is" }));
    }
    const permissions = store.permissions.size;
    const questions = store.questions.size;
    if (permissions || questions) {
      const words = [];
      if (permissions) words.push(plural(permissions, "permission"));
      if (questions) words.push(plural(questions, "question"));
      items.push(h("span", { class: "pending" }, glyph("input"), ` ${words.join(", ")}`));
    }
    if (meta?.method) {
      items.push(linkButton(meta.method, () => app.openView("methods"), "Choose the orchestration method (/method)"));
    }
    if (this.connection) items.push(h("span", { class: "pending" }, glyph("input"), ` ${this.connection}`));
    clear(this.el);
    this.el.append(
      h("div", { class: "facts" }, items),
      this.message,
      h(
        "div",
        { class: "facts end" },
        linkButton(app.state?.config?.theme ?? "", () => app.openView("themes"), "Choose the theme (/theme)"),
        linkButton("? keys", () => app.openView("help"), "Keys and commands (?)"),
      ),
    );
  }
}
