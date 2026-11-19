// Requests that wait for the user, shown inline above the composer (never as a modal):
// permission requests name the node and model asking and show the exact command or diff;
// questions offer their suggested answers. Their keys act only while the composer is empty,
// so typing a message can never approve anything by accident.

import { api, sessionPath } from "./api.js";
import { button, clear, glyph, h } from "./dom.js";
import { looksLikeDiff } from "./format.js";
import { renderDiff } from "./transcript.js";

export class Panels {
  constructor(app) {
    this.app = app;
    this.el = document.getElementById("panels");
    this.busy = false;
  }

  permission() {
    return this.app.store.permissions.values().next().value ?? null;
  }

  question() {
    return this.app.store.questions.values().next().value ?? null;
  }

  render(changes) {
    if (!changes.all && !changes.panels) return;
    clear(this.el);
    const permission = this.permission();
    const question = this.question();
    if (permission) this.el.append(this.permissionPanel(permission));
    if (question) this.el.append(this.questionPanel(question));
  }

  where(nodeId) {
    const nodes = this.app.store.path(nodeId);
    const model = nodes.length ? nodes[nodes.length - 1].model : "";
    return [
      h("span", { class: "ask-path", text: nodes.map((n) => n.title).join(" / ") || "the root agent" }),
      model ? h("span", { class: "ask-model", text: model }) : null,
    ];
  }

  permissionPanel(request) {
    const count = this.app.store.permissions.size;
    const rule = request.session_rule.includes(":")
      ? request.session_rule.slice(request.session_rule.indexOf(":") + 1)
      : request.session_rule;
    // The detail is shown only when it adds something: a diff, or text the summary leaves out.
    const detail = request.detail && !request.summary.includes(request.detail) ? request.detail : "";
    const id = `ask-${request.request_id}`;
    return h(
      "section",
      { class: "ask permission", role: "group", "aria-labelledby": id },
      h(
        "div",
        { class: "ask-head" },
        h("strong", { id }, glyph("input"), " Permission"),
        count > 1 ? h("span", { text: `1 of ${count}` }) : null,
        this.where(request.node_id),
      ),
      h(
        "div",
        { class: "ask-what" },
        h("span", { class: "ask-category", text: request.category }),
        h("span", { class: "ask-summary", text: request.summary }),
      ),
      detail ? (looksLikeDiff(detail) ? renderDiff(detail) : h("pre", { class: "ask-detail", text: detail })) : null,
      h(
        "div",
        { class: "actions" },
        button("allow once", { key: "y", primary: true, onclick: () => this.resolve(request, "allow") }),
        rule
          ? button(`allow ${rule} for this session`, { key: "a", onclick: () => this.resolve(request, "allow_session") })
          : null,
        button("deny", { key: "n", onclick: () => this.resolve(request, "deny") }),
      ),
    );
  }

  questionPanel(question) {
    const id = `ask-${question.request_id}`;
    return h(
      "section",
      { class: "ask question", role: "group", "aria-labelledby": id },
      h("div", { class: "ask-head" }, h("strong", { id }, glyph("input"), " Question"), this.where(question.node_id)),
      h("p", { class: "ask-question", text: question.question }),
      question.options.length
        ? h(
            "div",
            { class: "actions" },
            question.options.map((option, index) =>
              button(option, { key: index < 9 ? String(index + 1) : "", onclick: () => this.answer(question, option) }),
            ),
          )
        : null,
      h("p", { class: "ask-hint", text: "Or type an answer in the box below and press Enter." }),
    );
  }

  async resolve(request, choice) {
    await this.send(sessionPath(this.app.sessionId, `/permissions/${encodeURIComponent(request.request_id)}`), {
      choice,
    });
  }

  async answer(question, text) {
    await this.send(sessionPath(this.app.sessionId, `/questions/${encodeURIComponent(question.request_id)}`), {
      answer: text,
    });
  }

  async send(path, body) {
    if (this.busy) return;
    this.busy = true;
    try {
      await api.post(path, body);
    } catch (error) {
      this.app.status.flash(error.message, "error");
    } finally {
      this.busy = false;
    }
  }

  // Keys for the panels; the caller checks that the composer is empty. True when handled.
  key(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return false;
    const permission = this.permission();
    if (permission) {
      const choice = { y: "allow", a: permission.session_rule ? "allow_session" : "", n: "deny" }[event.key];
      if (choice) {
        this.resolve(permission, choice);
        return true;
      }
    }
    const question = this.question();
    if (question && /^[1-9]$/.test(event.key)) {
      const option = question.options[Number(event.key) - 1];
      if (option !== undefined) {
        this.answer(question, option);
        return true;
      }
    }
    return false;
  }
}
