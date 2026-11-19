// The transcript: the conversation (every run's prompt and what the root agents did), or the
// transcript of one node of the task tree.
//
// Rendering is incremental. Each entry keeps its element; streamed text is appended to a text
// node; finished items are patched in place; the list is only reordered when its shape
// changes. Scrolling follows the bottom unless the user has scrolled up, in which case a
// "N new" button counts what arrived meanwhile.

import { button, clear, glyph, h } from "./dom.js";
import { clock, diffStat, duration, firstLine, looksLikeDiff, plural } from "./format.js";
import { itemKey } from "./store.js";

const NEAR_BOTTOM = 24; // px from the bottom that still counts as "at the bottom"

export function renderDiff(text) {
  const lines = String(text || "").replace(/\n$/, "").split("\n");
  const code = h("code", { class: "diff-lines" });
  for (const line of lines) {
    let cls = "ctx";
    if (line.startsWith("+++") || line.startsWith("---")) cls = "file";
    else if (line.startsWith("@@")) cls = "hunk";
    else if (line.startsWith("+")) cls = "add";
    else if (line.startsWith("-")) cls = "del";
    code.append(h("span", { class: cls, text: line || " " }));
  }
  return h("pre", { class: "diff" }, code);
}

function toolArgument(item) {
  const summary = item.summary || "";
  return summary.startsWith(`${item.tool} `) ? summary.slice(item.tool.length + 1) : summary;
}

function toolMeta(item, now) {
  if (!item.done) return `running ${clock(now - item.ts)}`;
  const parts = [];
  if (item.display?.type === "diff") {
    const { added, removed } = diffStat(item.display.diff);
    parts.push(`+${added} -${removed}`);
  } else if (item.output) {
    parts.push(firstLine(item.output, 48));
  }
  if (item.duration) parts.push(duration(item.duration));
  return parts.join("  ");
}

function toolStatus(item) {
  if (!item.done) return "running";
  return item.is_error ? "failed" : "done";
}

export class Transcript {
  constructor(app) {
    this.app = app;
    this.scroller = document.getElementById("scroller");
    this.log = document.getElementById("log");
    this.jump = document.getElementById("jump");
    this.focus = ""; // "" is the conversation; otherwise a node id
    this.entries = [];
    this.els = new Map(); // entry key -> element
    this.live = new Map(); // key of a running tool row -> [element, item]
    this.pinned = true;
    this.unseen = 0;
    this.scroller.addEventListener("scroll", () => this.onScroll(), { passive: true });
    this.jump.addEventListener("click", () => this.toBottom());
    // The viewport or the content can change size without new entries (fonts arriving, the
    // dock growing, a block expanding): a transcript that follows the bottom keeps doing so.
    const follow = new ResizeObserver(() => {
      if (this.pinned) this.scroller.scrollTop = this.scroller.scrollHeight;
    });
    follow.observe(this.scroller);
    follow.observe(this.log);
  }

  setFocus(nodeId) {
    this.focus = nodeId || "";
    this.rebuild();
    this.toBottom();
  }

  onScroll() {
    const s = this.scroller;
    this.pinned = s.scrollHeight - s.scrollTop - s.clientHeight <= NEAR_BOTTOM;
    if (this.pinned) this.setUnseen(0);
  }

  toBottom() {
    this.scroller.scrollTop = this.scroller.scrollHeight;
    this.pinned = true;
    this.setUnseen(0);
  }

  setUnseen(count) {
    this.unseen = count;
    this.jump.hidden = count === 0;
    this.jump.textContent = count ? `${plural(count, "new block")}  jump to bottom` : "";
  }

  // -- rendering -------------------------------------------------------------------------------

  visibleEntries() {
    const store = this.app.store;
    const entries = this.focus ? store.nodeTranscript(this.focus) : store.conversation();
    const showThinking = this.app.state?.config?.show_thinking ?? true;
    return showThinking ? entries : entries.filter((e) => e.item.kind !== "thinking");
  }

  shownNodes() {
    return this.focus ? [this.focus] : this.app.store.roots;
  }

  render(changes) {
    const pinned = this.pinned;
    const shapeChanged =
      changes.all || changes.runs || this.shownNodes().some((id) => changes.nodes.has(id));
    let added = 0;
    if (changes.all) {
      this.rebuild();
    } else {
      for (const [key, text] of changes.appends) {
        const el = this.els.get(key);
        if (el?._stream) el._stream.appendData(text);
      }
      for (const key of changes.items) {
        const entry = this.entries.find((e) => this.keyOf(e) === key);
        if (entry) this.patch(entry);
      }
      if (shapeChanged) added = this.reconcile();
    }
    if (pinned) this.scroller.scrollTop = this.scroller.scrollHeight;
    else if (added) this.setUnseen(this.unseen + added);
  }

  keyOf(entry) {
    return itemKey(entry.nodeId, entry.item.id);
  }

  rebuild() {
    clear(this.log);
    this.els.clear();
    this.live.clear();
    this.entries = [];
    this.reconcile();
  }

  // Bring the DOM in line with the visible entries; returns how many blocks were added.
  reconcile() {
    const entries = this.visibleEntries();
    const keep = new Set(entries.map((e) => this.keyOf(e)));
    for (const [key, el] of this.els) {
      if (!keep.has(key)) {
        el.remove();
        this.els.delete(key);
        this.live.delete(key);
      }
    }
    let added = 0;
    let cursor = this.log.firstChild;
    for (const entry of entries) {
      const key = this.keyOf(entry);
      let el = this.els.get(key);
      if (!el) {
        el = this.build(entry);
        this.els.set(key, el);
        added += 1;
      }
      if (el === cursor) cursor = cursor.nextSibling;
      else this.log.insertBefore(el, cursor);
    }
    while (cursor) {
      const next = cursor.nextSibling;
      cursor.remove();
      cursor = next;
    }
    this.entries = entries;
    if (!entries.length) this.log.append(this.welcome());
    return added;
  }

  // -- blocks ----------------------------------------------------------------------------------

  build(entry) {
    const { item } = entry;
    switch (item.kind) {
      case "prompt":
        return h("div", { class: "block turn", text: item.text });
      case "user": {
        // A subtask's first message is the brief its parent wrote, not the user's words.
        const brief = Boolean(this.app.store.node(entry.nodeId)?.parent_id);
        return h("div", { class: `block turn${brief ? " brief" : ""}`, text: item.text });
      }
      case "assistant":
        return this.patchProse(h("div", { class: "block prose" }), item);
      case "thinking":
        return this.buildThinking(item);
      case "tool":
        return this.buildTool(entry);
      case "end":
        return h(
          "div",
          { class: "block notice" },
          glyph(item.status),
          h("span", { text: item.status === "cancelled" ? "Run cancelled." : "Run failed." }),
        );
      default:
        return h(
          "div",
          { class: `block notice level-${item.level || "info"}` },
          item.level === "error" ? glyph("failed") : item.level === "warning" ? glyph("input") : h("span"),
          h("span", { class: "notice-text", text: item.text }),
        );
    }
  }

  patch(entry) {
    const el = this.els.get(this.keyOf(entry));
    if (!el) return;
    const { item } = entry;
    if (item.kind === "assistant") this.patchProse(el, item);
    else if (item.kind === "thinking") this.patchThinking(el, item);
    else if (item.kind === "tool") this.patchTool(el, entry);
  }

  patchProse(el, item) {
    clear(el);
    el._stream = null;
    el.classList.toggle("streaming", !item.done);
    if (item.done && item.html) {
      el.innerHTML = item.html; // rendered and sanitised by the server
    } else {
      el._stream = document.createTextNode(item.text);
      el.append(el._stream);
    }
    return el;
  }

  buildThinking(item) {
    const text = document.createTextNode(item.text);
    const el = h(
      "details",
      { class: "block thinking" },
      h("summary", {}, h("span", { class: "caret", "aria-hidden": "true" }), h("span", { class: "label" })),
      h("div", { class: "thinking-text" }, text),
    );
    el._stream = text;
    this.patchThinking(el, item);
    return el;
  }

  patchThinking(el, item) {
    el.querySelector(".label").textContent = item.done ? "Reasoning" : "Reasoning…";
    el.classList.toggle("streaming", !item.done);
    if (item.done) el._stream.data = item.text;
  }

  buildTool(entry) {
    const { item } = entry;
    const meta = h("span", { class: "meta" });
    const summary = h(
      "summary",
      { class: "tool-row" },
      h("span", { class: "glyph-slot" }),
      h("span", { class: "verb", text: item.tool }),
      h("span", { class: "arg", text: toolArgument(item), title: item.summary }),
      meta,
    );
    const el = h("details", { class: "block tool" }, summary);
    el._meta = meta;
    el._entry = entry;
    el.addEventListener("toggle", () => {
      if (el.open) this.fillTool(el);
    });
    this.patchTool(el, entry);
    return el;
  }

  patchTool(el, entry) {
    const { item } = entry;
    const key = this.keyOf(entry);
    el._entry = entry;
    const status = toolStatus(item);
    el.classList.toggle("failed", status === "failed");
    el.querySelector(".glyph-slot").replaceChildren(glyph(status));
    el._meta.textContent = toolMeta(item, Date.now() / 1000);
    if (item.done) this.live.delete(key);
    else this.live.set(key, [el, item]);
    if (el.open) this.fillTool(el);
  }

  fillTool(el) {
    const { item } = el._entry;
    el.querySelector(".tool-body")?.remove();
    const body = h("div", { class: "tool-body" });
    if (!item.done) body.append(h("p", { class: "quiet", text: "Still running." }));
    else if (item.display?.type === "diff") body.append(renderDiff(item.display.diff));
    else if (!item.output) body.append(h("p", { class: "quiet", text: "No output." }));
    else if (looksLikeDiff(item.output)) body.append(renderDiff(item.output));
    else body.append(h("pre", { class: "output", text: item.output }));
    el.append(body);
  }

  // Once a second: running tools show how long they have been at it.
  tick(now) {
    for (const [el, item] of this.live.values()) el._meta.textContent = toolMeta(item, now);
  }

  welcome() {
    const app = this.app;
    const state = app.state;
    if (this.focus) return h("p", { class: "block quiet", text: "Nothing has happened on this task yet." });
    const parts = [
      h("h1", { class: "title", text: "Clankit" }),
      h("p", { class: "lede", text: `Working in ${state?.workspace ?? "this folder"}.` }),
    ];
    const demo = app.store.meta?.demo;
    if (state && !state.config.usable && !demo) {
      parts.push(
        h("p", {
          class: "lede",
          text:
            "No model is set up yet. Add an API key for a provider to start working, or play " +
            "the scripted demo, which needs no key and changes no files.",
        }),
        h(
          "div",
          { class: "actions" },
          button("Open settings", { primary: true, onclick: () => app.openSettings("providers") }),
          button("Play the demo", { onclick: () => app.newSession({ method: "demo" }) }),
        ),
      );
    } else {
      parts.push(
        h("p", {
          class: "lede",
          text: demo
            ? "This session plays a scripted demo: send any prompt to watch a task tree grow."
            : "Describe a change or ask a question below. Type / for commands.",
        }),
      );
    }
    return h("div", { class: "welcome" }, parts);
  }
}
