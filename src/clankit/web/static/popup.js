// A filterable list in a popup: the picker for models, methods, themes and sessions, and the
// slash-command menu's bigger sibling. One at a time; it sits on the chrome surface next to
// what opened it, and keeps keyboard focus until it closes.
//
//   openPicker({ title, items, anchor, onPick, onPreview, onCancel, custom })
//
// `items` are {id, label, detail?, note?, group?, current?}. `onPreview` runs as the
// highlight moves (live theme preview); `onCancel` when the picker closes without a choice.
// `custom(text)` may return an extra item for free text (a model nobody listed).

import { clear, h } from "./dom.js";

const popup = () => document.getElementById("popup");
let active = null;

export function closePopup(cancelled = true) {
  if (!active) return;
  const current = active;
  active = null;
  popup().hidden = true;
  clear(popup());
  document.removeEventListener("mousedown", current.outside, true);
  if (cancelled) current.onCancel?.();
  current.restore?.focus?.({ preventScroll: true });
}

export function popupOpen() {
  return active !== null;
}

function place(el, anchor) {
  const rect = anchor.getBoundingClientRect();
  const width = Math.min(Math.max(rect.width, 420), window.innerWidth - 32);
  const left = Math.min(Math.max(16, rect.left), window.innerWidth - width - 16);
  el.style.setProperty("--popup-left", `${left}px`);
  el.style.setProperty("--popup-width", `${width}px`);
  const below = rect.top < window.innerHeight / 2;
  el.classList.toggle("below", below);
  if (below) {
    el.style.setProperty("--popup-top", `${rect.bottom + 4}px`);
    el.style.setProperty("--popup-max", `${window.innerHeight - rect.bottom - 24}px`);
  } else {
    el.style.setProperty("--popup-bottom", `${window.innerHeight - rect.top + 4}px`);
    el.style.setProperty("--popup-max", `${rect.top - 24}px`);
  }
}

function open(anchor, content, onCancel) {
  closePopup();
  const el = popup();
  clear(el);
  el.append(...content);
  el.hidden = false;
  place(el, anchor);
  const outside = (event) => {
    if (!el.contains(event.target)) closePopup();
  };
  document.addEventListener("mousedown", outside, true);
  active = { onCancel, outside, restore: document.activeElement };
  return el;
}

export function openPicker({ title, items, anchor, onPick, onPreview, onCancel, custom, empty }) {
  const input = h("input", {
    type: "text",
    class: "popup-filter",
    "aria-label": `Filter ${title.toLowerCase()}`,
    placeholder: "Type to filter",
    spellcheck: "false",
  });
  const list = h("ul", { class: "popup-list", role: "listbox", id: "popup-list", "aria-label": title });
  let shown = [];
  let index = 0;

  const highlight = (next, preview = true) => {
    if (!shown.length) return;
    index = (next + shown.length) % shown.length;
    list.querySelectorAll("[role=option]").forEach((el, i) => el.setAttribute("aria-selected", String(i === index)));
    const el = list.querySelector(`[data-index="${index}"]`);
    el?.scrollIntoView({ block: "nearest" });
    input.setAttribute("aria-activedescendant", el?.id ?? "");
    if (preview) onPreview?.(shown[index]);
  };

  const choose = (item) => {
    closePopup(false);
    onPick(item);
  };

  const render = () => {
    const query = input.value.trim().toLowerCase();
    shown = items.filter((item) =>
      `${item.label} ${item.detail ?? ""} ${item.group ?? ""}`.toLowerCase().includes(query),
    );
    const extra = query && custom ? custom(input.value.trim()) : null;
    if (extra && !shown.some((item) => item.id === extra.id)) shown.push(extra);
    clear(list);
    let group = null;
    shown.forEach((item, i) => {
      if (item.group && item.group !== group) {
        group = item.group;
        list.append(h("li", { class: "popup-group", role: "presentation", text: group }));
      }
      list.append(
        h(
          "li",
          {
            role: "option",
            id: `popup-option-${i}`,
            "aria-selected": "false",
            dataset: { index: String(i) },
            class: item.current ? "current" : "",
            onclick: () => choose(item),
          },
          h(
            "span",
            { class: "popup-label" },
            item.label,
            item.current ? h("span", { class: "popup-note", text: "current" }) : null,
          ),
          item.swatches ? h("span", { class: "swatches" }, item.swatches.map(swatch)) : null,
          item.detail ? h("span", { class: "popup-detail", text: item.detail }) : null,
        ),
      );
    });
    if (!shown.length) list.append(h("li", { class: "popup-empty", text: empty || "Nothing matches." }));
    const currentIndex = shown.findIndex((item) => item.current);
    highlight(query || currentIndex < 0 ? 0 : currentIndex, Boolean(query));
  };

  input.addEventListener("input", render);
  input.addEventListener("keydown", (event) => {
    const moves = { ArrowDown: 1, ArrowUp: -1, PageDown: 8, PageUp: -8 };
    if (event.key in moves) {
      event.preventDefault();
      highlight(Math.max(0, Math.min(shown.length - 1, index + moves[event.key])));
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (shown[index]) choose(shown[index]);
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closePopup();
    }
  });

  open(
    anchor,
    [
      h(
        "div",
        { class: "popup-head" },
        h("strong", { text: title }),
        h("span", { class: "popup-keys", text: "enter choose   esc close" }),
      ),
      input,
      list,
    ],
    onCancel,
  );
  render();
  input.focus();
}

export function swatch(colour) {
  return h("span", { class: "swatch", vars: { "--swatch": colour } });
}

// Keys and commands, for "?" and /help.
export function openHelp(anchor, commands, onCommand) {
  const keys = [
    ["enter", "send; while a run works, steer it; with a question open, answer it"],
    ["shift+enter", "new line"],
    ["/", "commands"],
    ["esc", "cancel the run, close a menu, or go back"],
    ["y  a  n", "allow once, allow for the session, deny (composer empty)"],
    ["1-9", "pick a suggested answer (composer empty)"],
    ["tree: arrows", "move; right and left open and close"],
    ["tree: enter", "show that task's transcript; esc returns"],
    ["?", "this help"],
  ];
  const el = open(
    anchor,
    [
      h(
        "div",
        { class: "popup-head" },
        h("strong", { text: "Keys and commands" }),
        h("span", { class: "popup-keys", text: "esc close" }),
      ),
      h(
        "div",
        { class: "popup-help", tabindex: "-1" },
        h("table", { class: "keys" }, keys.map(([key, what]) => h("tr", {}, h("th", { text: key }), h("td", { text: what })))),
        h(
          "table",
          { class: "keys" },
          commands.map((c) =>
            h(
              "tr",
              {},
              h("th", {}, h("button", { type: "button", class: "link", text: `/${c.name}`, onclick: () => (closePopup(false), onCommand(c)) })),
              h("td", { text: c.argument ? `${c.summary}  ${c.argument}` : c.summary }),
            ),
          ),
        ),
      ),
    ],
    null,
  );
  const body = el.querySelector(".popup-help");
  body.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      closePopup();
    }
  });
  body.focus();
}
