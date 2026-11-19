// The composer: Enter sends (a prompt, a steering message while a run works, or the answer
// to an open question), Shift+Enter starts a new line, "/" opens the command menu, Escape
// closes the menu or cancels the run.

import { clear, h } from "./dom.js";

const MAX_ROWS = 10;

// Mirrors clankit.commands.parse: "/name argument", but not "//" and not a path like /usr/bin.
export function parseCommand(text) {
  const stripped = text.trim();
  if (!stripped.startsWith("/") || stripped.startsWith("//") || stripped.length < 2) return null;
  const [name, ...rest] = stripped.slice(1).split(" ");
  if (!name || name.includes("/")) return null;
  return { name: name.toLowerCase(), argument: rest.join(" ").trim() };
}

export class Composer {
  constructor(app) {
    this.app = app;
    this.form = document.getElementById("composer");
    this.input = document.getElementById("input");
    this.menu = document.getElementById("menu");
    this.hint = document.getElementById("hint");
    this.matches = [];
    this.selected = 0;
    this.sending = false;
    this.input.addEventListener("keydown", (e) => this.onKey(e));
    this.input.addEventListener("input", () => this.onInput());
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.send();
    });
    this.form.addEventListener("click", () => this.input.focus());
  }

  get empty() {
    return this.input.value.trim() === "";
  }

  focus(text) {
    if (text !== undefined) {
      this.input.value = text;
      this.onInput();
    }
    this.input.focus();
  }

  // -- input -----------------------------------------------------------------------------------

  onInput() {
    this.autosize();
    this.updateMenu();
  }

  autosize() {
    const input = this.input;
    input.style.height = "auto";
    const line = parseFloat(getComputedStyle(input).lineHeight) || 20;
    input.style.height = `${Math.min(input.scrollHeight, line * MAX_ROWS)}px`;
  }

  onKey(event) {
    if (event.isComposing) return;
    if (!this.menu.hidden && this.menuKey(event)) return;
    if (event.key === "Enter" && !event.shiftKey && !event.altKey) {
      event.preventDefault();
      this.send();
    } else if (event.key === "Escape") {
      event.preventDefault();
      this.app.escape();
    } else if (this.empty && this.app.panels.key(event)) {
      event.preventDefault();
    }
  }

  // -- the command menu ------------------------------------------------------------------------

  updateMenu() {
    const match = /^\/([\w-]*)$/.exec(this.input.value);
    const commands = this.app.state?.commands ?? [];
    this.matches = match ? commands.filter((c) => c.name.startsWith(match[1].toLowerCase())) : [];
    if (!this.matches.length) {
      this.closeMenu();
      return;
    }
    this.selected = Math.min(this.selected, this.matches.length - 1);
    clear(this.menu);
    this.matches.forEach((command, index) => {
      this.menu.append(
        h(
          "div",
          {
            class: "menu-item",
            role: "option",
            id: `menu-${command.name}`,
            "aria-selected": String(index === this.selected),
            onmousedown: (e) => {
              e.preventDefault();
              this.selected = index;
              this.runSelected();
            },
          },
          h("span", { class: "menu-name", text: `/${command.name}` }),
          h("span", { class: "menu-arg", text: command.argument }),
          h("span", { class: "menu-summary", text: command.summary }),
        ),
      );
    });
    this.menu.hidden = false;
    this.input.setAttribute("aria-activedescendant", `menu-${this.matches[this.selected].name}`);
    this.menu.querySelector("[aria-selected=true]")?.scrollIntoView({ block: "nearest" });
  }

  closeMenu() {
    this.menu.hidden = true;
    this.selected = 0;
    this.input.removeAttribute("aria-activedescendant");
    clear(this.menu);
  }

  menuKey(event) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      this.selected = (this.selected + step + this.matches.length) % this.matches.length;
      this.updateMenu();
      return true;
    }
    if (event.key === "Tab") {
      event.preventDefault();
      this.input.value = `/${this.matches[this.selected].name} `;
      this.closeMenu();
      this.autosize();
      return true;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      this.runSelected();
      return true;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      this.closeMenu();
      return true;
    }
    return false;
  }

  runSelected() {
    const command = this.matches[this.selected];
    if (!command) return;
    this.input.value = `/${command.name}`;
    this.closeMenu();
    this.send();
  }

  // -- sending ---------------------------------------------------------------------------------

  async send() {
    const text = this.input.value;
    if (!text.trim() || this.sending) return;
    this.sending = true;
    try {
      const sent = await this.app.submit(text);
      if (sent) {
        this.input.value = "";
        this.onInput();
      }
    } finally {
      this.sending = false;
    }
  }

  render() {
    const store = this.app.store;
    let placeholder = "Describe a change or ask a question";
    let keys = ["enter send", "shift+enter new line", "/ commands"];
    if (store.questions.size) {
      placeholder = "Type an answer to the question above";
      keys = ["enter answer", "shift+enter new line", "esc cancel the run"];
    } else if (store.running) {
      placeholder = "Steer the working agent: it reads this at its next step";
      keys = ["enter steer", "shift+enter new line", "esc cancel the run"];
    }
    this.input.placeholder = placeholder;
    clear(this.hint);
    for (const key of keys) this.hint.append(h("span", { text: key }));
  }
}
