// Small DOM helpers. Text always goes in as text nodes; the only HTML ever inserted is the
// Markdown the server renders and sanitises (see clankit/web/markdown.py).

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "vars") for (const [name, v] of Object.entries(value)) el.style.setProperty(name, v);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.firstChild.remove();
  return el;
}

// The status vocabulary shared with the terminal UI: shape carries the meaning, colour repeats it.
export const GLYPHS = {
  pending: "○",
  running: "●",
  waiting: "◐",
  integrating: "◆",
  done: "✓",
  failed: "✗",
  cancelled: "-",
  input: "!",
};

export const STATUS_WORDS = {
  pending: "waiting to start",
  running: "working",
  waiting: "waiting for subtasks",
  integrating: "integrating subtasks",
  done: "done",
  failed: "failed",
  cancelled: "cancelled",
  input: "needs your input",
};

export function glyph(status, extra = "") {
  return h("span", {
    class: `glyph g-${status}${extra ? ` ${extra}` : ""}`,
    "aria-hidden": "true",
    text: GLYPHS[status] || " ",
  });
}

// A button whose label may start with the key that triggers it.
export function button(label, { key = "", primary = false, onclick, title, cls = "" } = {}) {
  return h(
    "button",
    { type: "button", class: `btn${primary ? " primary" : ""}${cls ? ` ${cls}` : ""}`, onclick, title },
    key ? h("kbd", { text: key }) : null,
    label,
  );
}

export function isTyping(target) {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}
