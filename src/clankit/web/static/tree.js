// The task tree: an indented outline with fixed columns (status, caret, title, model,
// tokens, elapsed), the same row grammar and glyphs as the terminal UI.
//
// Keyboard follows the WAI-ARIA tree pattern with a roving tabindex: Up/Down move, Right
// opens or steps into a node, Left closes or steps out, Home/End jump, * opens siblings,
// Enter shows the node's transcript, Escape returns to the conversation.

import { STATUS_WORDS, clear, glyph, h } from "./dom.js";
import { alias, clock, plural, tokens, usageTotal } from "./format.js";

const MAX_INDENT = 6; // deeper rows stop indenting; the crumb and aria-level keep the depth
const ACTIVE = new Set(["running", "waiting", "integrating"]);
const SETTLED = new Set(["done", "cancelled"]);

export class TaskTree {
  constructor(app) {
    this.app = app;
    this.panel = document.getElementById("tree");
    this.rollup = document.getElementById("rollup");
    this.toggled = new Map(); // node id -> expanded, where the user overrode the default
    this.cursor = ""; // node id holding the roving tabindex
    this.rows = new Map(); // node id -> {li, time}
    this.panel.addEventListener("keydown", (e) => this.onKey(e));
    this.rollup.addEventListener("click", () => this.app.toggleTreeView());
  }

  // -- state -----------------------------------------------------------------------------------

  expanded(node) {
    if (this.toggled.has(node.id)) return this.toggled.get(node.id);
    if (!node.children.length) return false;
    // Finished subtrees fold away; a path to anything active, failed or waiting stays open.
    if (!node.parent_id) return true;
    return !(SETTLED.has(node.status) && !this.hasTrouble(node));
  }

  hasTrouble(node) {
    const store = this.app.store;
    return node.children.some((id) => {
      const child = store.node(id);
      if (!child) return false;
      const status = store.displayStatus(child);
      return status === "failed" || status === "input" || ACTIVE.has(status) || this.hasTrouble(child);
    });
  }

  visible() {
    const store = this.app.store;
    const out = [];
    const walk = (ids, depth) => {
      for (const id of ids) {
        const node = store.node(id);
        if (!node) continue;
        out.push([depth, node]);
        if (this.expanded(node)) walk(node.children, depth + 1);
      }
    };
    walk(store.roots, 0);
    return out;
  }

  // -- rendering -------------------------------------------------------------------------------

  render(changes) {
    if (!changes.all && !changes.tree && !changes.panels) return;
    const store = this.app.store;
    const hadFocus = this.panel.contains(document.activeElement);
    clear(this.panel);
    this.rows.clear();
    this.renderRollup();
    if (!store.roots.length) {
      this.panel.append(
        h("div", { class: "tree-head" }, h("span", { text: "tasks" })),
        h("p", { class: "tree-empty", text: "The task tree grows here once a run starts." }),
      );
      return;
    }
    this.panel.append(this.header());
    const visible = this.visible();
    if (!visible.some(([, node]) => node.id === this.cursor)) this.cursor = this.app.focus || visible[0][1].id;
    const list = h("ul", { role: "tree", "aria-label": "Task tree", class: "tree-list" });
    const groups = new Map();
    for (const [depth, node] of visible) {
      const li = this.row(depth, node);
      const parentGroup = node.parent_id ? groups.get(node.parent_id) : null;
      (parentGroup ?? list).append(li);
      if (node.children.length && this.expanded(node)) {
        const group = h("ul", { role: "group" });
        li.append(group);
        groups.set(node.id, group);
      }
    }
    this.panel.append(list);
    if (hadFocus) this.rows.get(this.cursor)?.li.focus({ preventScroll: true });
  }

  header() {
    const store = this.app.store;
    const counts = store.counts();
    const total = store.nodes.size;
    const active = (counts.running ?? 0) + (counts.waiting ?? 0) + (counts.integrating ?? 0);
    const tally = [
      ["done", counts.done],
      ["running", active],
      ["input", counts.input],
      ["failed", counts.failed],
    ]
      .filter(([, n]) => n)
      .map(([status, n]) => h("span", { class: "tally" }, glyph(status), String(n)));
    const limit = this.app.depthLimit();
    const depth = store.maxDepth();
    return h(
      "div",
      { class: "tree-head" },
      h("span", { class: "strong", text: plural(total, "task") }),
      ...tally,
      h("span", { class: "spacer" }),
      h("span", { text: limit ? `depth ${depth}/${limit}` : `depth ${depth}` }),
    );
  }

  row(depth, node) {
    const store = this.app.store;
    const status = store.displayStatus(node);
    const open = this.expanded(node);
    const parent = node.parent_id ? store.node(node.parent_id) : null;
    const crossModel = Boolean(node.model && (!parent || parent.model !== node.model));
    const hidden = node.children.length && !open ? store.descendants(node.id) : 0;
    const time = h("span", { class: "num time", text: this.elapsed(node) });
    const label = h(
      "span",
      { class: "label" },
      h("span", { class: "title-text", text: node.title }),
      hidden ? h("span", { class: "rollup-count", text: `(+${hidden})` }) : null,
    );
    const row = h(
      "div",
      { class: `row${SETTLED.has(status) ? " settled" : ""}`, vars: { "--d": String(Math.min(depth, MAX_INDENT)) } },
      glyph(status, ACTIVE.has(status) ? "live" : ""),
      h("span", { class: "caret", "aria-hidden": "true", text: node.children.length ? (open ? "▾" : "▸") : "" }),
      label,
      h("span", { class: `model${crossModel ? " differs" : ""}`, text: alias(node.model) || "-", title: node.model || "" }),
      h("span", { class: "num", text: node.usage && usageTotal(node.usage) ? tokens(usageTotal(node.usage)) : "-" }),
      time,
    );
    const li = h(
      "li",
      {
        role: "treeitem",
        "aria-level": String(depth + 1),
        "aria-selected": node.id === (this.app.focus || "") ? "true" : "false",
        "aria-expanded": node.children.length ? String(open) : null,
        "aria-label": `${node.title}, ${STATUS_WORDS[status]}${node.model ? `, ${node.model}` : ""}`,
        tabindex: node.id === this.cursor ? "0" : "-1",
        dataset: { id: node.id },
        title: node.summary || node.brief || node.title,
      },
      row,
    );
    row.addEventListener("click", (e) => {
      e.stopPropagation();
      this.cursor = node.id;
      if (e.target.classList.contains("caret")) this.toggle(node, !open);
      else this.app.focusNode(node.id);
    });
    this.rows.set(node.id, { li, time, node });
    return li;
  }

  elapsed(node) {
    if (!node.started) return "-";
    const end = node.finished || Date.now() / 1000;
    return clock(end - node.started);
  }

  tick() {
    for (const { time, node } of this.rows.values()) {
      if (node.started && !node.finished) time.textContent = this.elapsed(node);
    }
  }

  renderRollup() {
    const store = this.app.store;
    const counts = store.counts();
    clear(this.rollup);
    if (!store.nodes.size) {
      this.rollup.hidden = true;
      return;
    }
    this.rollup.hidden = false;
    const active = (counts.running ?? 0) + (counts.waiting ?? 0) + (counts.integrating ?? 0);
    this.rollup.append(
      h("span", { class: "strong", text: plural(store.nodes.size, "task") }),
      ...[
        ["done", counts.done],
        ["running", active],
        ["input", counts.input],
        ["failed", counts.failed],
      ]
        .filter(([, n]) => n)
        .map(([status, n]) => h("span", { class: "tally" }, glyph(status), String(n))),
      h("span", { class: "spacer" }),
      h("span", { class: "link", text: this.app.treeOpen ? "hide tree" : "show tree" }),
    );
  }

  // -- keyboard --------------------------------------------------------------------------------

  toggle(node, open) {
    this.toggled.set(node.id, open);
    this.render({ tree: true });
  }

  move(id) {
    this.cursor = id;
    for (const [rowId, { li }] of this.rows) li.tabIndex = rowId === id ? 0 : -1;
    this.rows.get(id)?.li.focus();
  }

  onKey(event) {
    const visible = this.visible();
    const index = visible.findIndex(([, node]) => node.id === this.cursor);
    if (index < 0) return;
    const node = visible[index][1];
    const store = this.app.store;
    const keys = {
      ArrowDown: () => index + 1 < visible.length && this.move(visible[index + 1][1].id),
      ArrowUp: () => index > 0 && this.move(visible[index - 1][1].id),
      Home: () => this.move(visible[0][1].id),
      End: () => this.move(visible[visible.length - 1][1].id),
      ArrowRight: () => {
        if (!node.children.length) return;
        if (!this.expanded(node)) this.toggle(node, true);
        else this.move(node.children[0]);
      },
      ArrowLeft: () => {
        if (node.children.length && this.expanded(node)) this.toggle(node, false);
        else if (node.parent_id) this.move(node.parent_id);
      },
      "*": () => {
        const siblings = node.parent_id ? store.node(node.parent_id).children : store.roots;
        for (const id of siblings) if (store.node(id)?.children.length) this.toggled.set(id, true);
        this.render({ tree: true });
      },
      Enter: () => this.app.focusNode(node.id),
      " ": () => this.app.focusNode(node.id),
      Escape: () => this.app.focusNode(""),
    };
    const action = keys[event.key];
    if (!action) return;
    event.preventDefault();
    event.stopPropagation();
    action();
  }
}

