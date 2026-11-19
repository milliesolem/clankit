// Clankit's web interface. One page, one EventSource per tab, no framework.
//
// Data flow: the server sends a snapshot of the session, then every engine event. Events are
// folded into the Store as they arrive; the views read the Store's change record once per
// animation frame and patch only what changed.

import { ApiError, api, sessionPath } from "./api.js";
import { Crumb, StatusRow } from "./chrome.js";
import { Composer, parseCommand } from "./composer.js";
import { isTyping } from "./dom.js";
import { plural, when } from "./format.js";
import { Panels } from "./panels.js";
import { closePopup, openHelp, openPicker, popupOpen } from "./popup.js";
import { SettingsView } from "./settings.js";
import { Store } from "./store.js";
import { Transcript } from "./transcript.js";
import { TaskTree } from "./tree.js";

const SESSION_KEY = "clankit.session";
const DISCOVER = "discover-models"; // the model picker's "ask the providers" entry

// Names this page's event stream, so a reconnect replaces its old stream on the server. New
// for every page load: a duplicated tab must not share it, or the two would evict each other.
function tabId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function remember(key, value) {
  try {
    sessionStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable (private windows); the page works without it.
  }
}

function recall(key) {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

class App {
  constructor() {
    this.root = document.getElementById("app");
    this.store = new Store();
    this.state = null;
    this.sessionId = "";
    this.focus = ""; // node whose transcript is shown; "" is the conversation
    this.view = "main";
    this.treeOpen = false; // the tree as a full view (narrow windows, or a hidden panel)
    this.tab = tabId();
    this.narrow = window.matchMedia("(max-width: 959px)");
    this.narrow.addEventListener("change", () => this.schedule(true));
    this.source = null;
    this.frame = 0;
    this.previewing = null;
    this.transcript = new Transcript(this);
    this.tree = new TaskTree(this);
    this.panels = new Panels(this);
    this.composer = new Composer(this);
    this.crumb = new Crumb(this);
    this.status = new StatusRow(this);
    this.settings = new SettingsView(this);
    document.addEventListener("keydown", (e) => this.onKey(e));
    window.addEventListener("hashchange", () => this.route());
    setInterval(() => this.tick(), 1000);
  }

  // -- start-up --------------------------------------------------------------------------------

  async boot() {
    let state;
    try {
      state = await this.loadState(recall(SESSION_KEY));
    } catch (error) {
      this.fail(error);
      return;
    }
    this.applyState(state);
    if (state.session) this.open(state.session);
    else await this.newSession();
    this.route();
    if (this.view === "main") this.composer.focus();
  }

  async loadState(sessionId) {
    try {
      return await api.get(`/api/state${sessionId ? `?session=${encodeURIComponent(sessionId)}` : ""}`);
    } catch (error) {
      if (sessionId && error instanceof ApiError && error.status === 404) return api.get("/api/state");
      throw error;
    }
  }

  fail(error) {
    const text =
      error instanceof ApiError && error.status === 401
        ? "This page is no longer logged in: the Clankit server was restarted or stopped. Open the new address it printed in its terminal."
        : error.message;
    this.status.flash(text, "error");
    this.status.setConnection("disconnected");
  }

  applyState(state) {
    const previous = this.state;
    this.state = state;
    if (!this.previewing) this.applyTheme(state.config.theme);
    const link = document.getElementById("theme-css");
    if (link && link.getAttribute("href") !== state.theme_css) link.setAttribute("href", state.theme_css);
    const thinkingChanged = previous && previous.config.show_thinking !== state.config.show_thinking;
    if (thinkingChanged || !this.transcript.entries.length) this.transcript.rebuild();
    this.applyLayout();
    this.schedule(true);
  }

  applyTheme(id) {
    document.documentElement.dataset.theme = id;
    const theme = this.state?.themes.find((t) => t.id === id);
    if (theme) document.querySelector('meta[name="color-scheme"]').content = theme.dark ? "dark" : "light";
  }

  // Show a theme without saving it (theme lists); null puts the saved one back.
  previewTheme(id) {
    this.previewing = id;
    this.applyTheme(id ?? this.state.config.theme);
  }

  // The tree is a side panel in wide windows; otherwise a rollup line opens it as a full view.
  applyLayout() {
    const compact = this.narrow.matches || this.state?.config.show_tree === false;
    this.root.dataset.view = this.view;
    this.settings.el.hidden = this.view !== "settings";
    this.root.dataset.layout = compact ? "compact" : "wide";
    this.root.dataset.treeOpen = String(compact && this.treeOpen);
  }

  // -- sessions --------------------------------------------------------------------------------

  open(session) {
    this.sessionId = session.meta.id;
    remember(SESSION_KEY, this.sessionId);
    this.focus = "";
    this.store.reset(session);
    this.transcript.setFocus("");
    this.connect(session.snapshot.seq);
    this.schedule(true);
  }

  connect(seq) {
    this.source?.close();
    const query = new URLSearchParams({ tab: this.tab });
    if (seq !== undefined) query.set("last_event_id", String(seq));
    const source = new EventSource(`${sessionPath(this.sessionId, "/events")}?${query}`);
    this.source = source;
    source.onmessage = (message) => {
      if (source === this.source) this.receive(JSON.parse(message.data));
    };
    source.onopen = () => this.status.setConnection("");
    source.onerror = () => {
      if (source !== this.source) return;
      if (source.readyState === EventSource.CLOSED) this.lostStream();
      else this.status.setConnection("reconnecting");
    };
  }

  async lostStream() {
    this.status.setConnection("disconnected");
    try {
      const state = await this.loadState(this.sessionId);
      this.applyState(state);
      if (state.session) this.open(state.session);
    } catch (error) {
      this.fail(error);
    }
  }

  receive(data) {
    if (data.kind === "snapshot") {
      this.store.reset(data.session); // marks everything for redrawing
      if (this.focus && !this.store.node(this.focus)) this.focus = this.transcript.focus = "";
    } else if (data.kind === "session_meta") {
      this.store.setMeta(data.meta);
    } else {
      this.store.apply(data);
      if (data.kind === "skill_saved") this.skillSaved(data);
    }
    this.schedule();
  }

  async skillSaved(event) {
    const verb = { proposed: "Proposed", created: "Saved", updated: "Updated" }[event.action] ?? "Saved";
    const review = event.action === "proposed" ? " Review it in settings." : "";
    this.status.flash(`${verb} skill ${event.name}.${review}`);
    await this.refresh();
  }

  async refresh() {
    try {
      const state = await api.get(`/api/state?session=${encodeURIComponent(this.sessionId)}`);
      this.applyState(state);
      if (state.session) this.store.setMeta(state.session.meta);
      if (this.view === "settings") this.settings.render();
    } catch (error) {
      this.status.flash(error.message, "error");
    }
  }

  // After a settings change the session's model (and so its cost and context) may differ.
  async refreshMeta() {
    try {
      this.store.setMeta(await api.get(sessionPath(this.sessionId, "/meta")));
      this.schedule(true);
    } catch (error) {
      this.status.flash(error.message, "error");
    }
  }

  async newSession(body = {}) {
    try {
      const result = await api.post("/api/sessions", { from: this.sessionId || undefined, ...body });
      this.applyState(result.state);
      const session = await api.get(sessionPath(result.id));
      this.closeSettings();
      this.open(session);
      this.composer.focus();
    } catch (error) {
      this.status.flash(error.message, "error");
    }
  }

  async switchSession(id) {
    if (id === this.sessionId) return;
    try {
      const result = await api.post("/api/sessions", { resume: id });
      this.applyState(result.state);
      this.open(await api.get(sessionPath(result.id)));
      this.status.flash("Session reopened. Agents start fresh and are told what happened before.");
    } catch (error) {
      this.status.flash(error.message, "error");
    }
  }

  // -- actions ---------------------------------------------------------------------------------

  // The composer's Enter: a command, an answer, a steering message or a prompt. True if sent.
  async submit(text) {
    const command = parseCommand(text);
    try {
      if (command) return await this.runCommand(text);
      const question = this.panels.question();
      if (question) {
        await this.panels.answer(question, text.trim());
      } else if (this.store.running) {
        await api.post(sessionPath(this.sessionId, "/steer"), { text });
        this.status.flash("Sent to the working agent; it reads it at its next step.");
      } else {
        await this.prompt(text);
      }
      return true;
    } catch (error) {
      this.status.flash(error.message, "error");
      return false;
    }
  }

  async prompt(text) {
    this.focusNode("");
    this.transcript.toBottom();
    await api.post(sessionPath(this.sessionId, "/prompt"), { text });
  }

  async runCommand(line) {
    const result = await api.post(sessionPath(this.sessionId, "/command"), { line });
    this.applyState(result.state);
    this.store.setMeta(result.meta);
    const outcome = result.outcome;
    if (outcome.message) this.status.flash(outcome.message, outcome.level);
    if (outcome.prompt) await this.prompt(outcome.prompt);
    if (outcome.view) this.openView(outcome.view);
    if (outcome.action === "new-session") await this.newSession();
    if (outcome.action === "quit") {
      this.status.flash("The web interface runs until you stop it with Ctrl+C in its terminal.");
    }
    return true;
  }

  async cancel() {
    try {
      const result = await api.post(sessionPath(this.sessionId, "/cancel"));
      if (result.cancelled) this.status.flash("Cancelling the run.");
    } catch (error) {
      this.status.flash(error.message, "error");
    }
  }

  // Escape outside a text field (and in an empty composer): step back one level.
  escape() {
    if (popupOpen()) closePopup();
    else if (this.view === "settings") this.closeSettings();
    else if (this.store.running && this.composer.empty) this.cancel();
    else if (this.treeOpen) this.toggleTreeView(false);
    else if (this.focus) this.focusNode("");
  }

  focusNode(id) {
    if (this.store.node(id) === null && id) return;
    this.focus = id;
    this.transcript.setFocus(id);
    if (this.treeOpen && id) this.toggleTreeView(false);
    this.schedule(true);
  }

  toggleTreeView(open = !this.treeOpen) {
    this.treeOpen = open;
    this.applyLayout();
    this.schedule(true);
    if (open) requestAnimationFrame(() => this.tree.panel.querySelector('[tabindex="0"]')?.focus());
  }

  depthLimit() {
    const method = this.state?.methods.find((m) => m.name === this.store.meta?.method);
    return method?.settings.find((s) => s.key === "max-depth")?.value ?? null;
  }

  // -- views -----------------------------------------------------------------------------------

  route() {
    const hash = location.hash.replace(/^#/, "");
    const [view, section] = hash.split("/");
    this.view = view === "settings" ? "settings" : "main";
    this.applyLayout();
    if (this.view === "settings") {
      this.settings.render(section || "");
      this.settings.el.querySelector(".settings-nav a")?.focus({ preventScroll: true });
    }
    this.schedule(true);
  }

  openSettings(section = "") {
    const hash = `#settings${section ? `/${section}` : ""}`;
    if (location.hash === hash) this.route();
    else location.hash = hash;
  }

  closeSettings() {
    if (this.view !== "settings") return;
    history.replaceState(null, "", location.pathname);
    this.route();
    this.composer.focus();
  }

  // Open what a command's outcome names: a view, or one of the pickers.
  openView(view) {
    const anchor = document.getElementById(this.view === "main" ? "composer" : "status");
    if (view === "settings") this.openSettings();
    else if (view === "skills") this.openSettings("skills");
    else if (view === "help") {
      openHelp(anchor, this.state.commands, (c) => this.composer.focus(`/${c.name}${c.argument ? " " : ""}`));
    } else if (view === "models") {
      this.pickModel({ anchor, current: this.store.meta?.model, onPick: (ref) => this.command(`/model ${ref}`) });
    } else if (view === "methods") this.pickMethod(anchor);
    else if (view === "themes") this.pickTheme(anchor);
    else if (view === "sessions") this.pickSession(anchor);
  }

  // A command chosen from a picker; failures go to the status row.
  async command(line) {
    try {
      await this.runCommand(line);
    } catch (error) {
      this.status.flash(error.message, "error");
    }
  }

  pickModel({ anchor, current, onPick, clearLabel = "" }) {
    const models = this.state.models.map((m) => ({
      id: m.ref,
      label: m.ref,
      group: m.provider,
      current: m.ref === current,
      detail: [
        m.name !== m.id ? m.name : "",
        m.context_window ? `${Math.round(m.context_window / 1000)}k context` : "",
        m.free ? "free" : m.input_price !== null ? `$${m.input_price}/$${m.output_price} per M` : "",
        m.note,
      ]
        .filter(Boolean)
        .join("  "),
    }));
    if (clearLabel) models.unshift({ id: "", label: clearLabel, current: !current });
    const usable = this.state.config.usable;
    const discover = { id: DISCOVER, label: "Ask the providers which models they offer now" };
    if (usable) models.push(discover);
    openPicker({
      title: "Model",
      items: models,
      anchor,
      empty: usable
        ? "No known model matches; type provider/model to use one by name."
        : "No provider is set up yet: add an API key in settings.",
      custom: (text) => (text.includes("/") ? { id: text, label: text, detail: "use this model by name" } : null),
      onPick: async (item) => {
        if (item.id !== DISCOVER) {
          onPick(item.id);
          return;
        }
        this.status.flash("Asking the providers for their model lists…");
        try {
          this.applyState((await api.post("/api/models/discover")).state);
          this.status.flash(`${plural(this.state.models.length, "model")} known.`);
          this.pickModel({ anchor, current, onPick, clearLabel });
        } catch (error) {
          this.status.flash(error.message, "error");
        }
      },
    });
  }

  pickMethod(anchor) {
    const current = this.store.meta?.method;
    openPicker({
      title: "Orchestration method",
      anchor,
      items: this.state.methods.map((m) => ({ id: m.name, label: m.title, detail: m.summary, current: m.name === current })),
      onPick: (item) => this.command(`/method ${item.id}`),
    });
  }

  pickTheme(anchor) {
    openPicker({
      title: "Theme",
      anchor,
      items: this.state.themes.map((t) => ({
        id: t.id,
        label: t.name,
        swatches: t.swatches,
        current: t.id === this.state.config.theme,
      })),
      onPreview: (item) => this.previewTheme(item.id),
      onCancel: () => this.previewTheme(null),
      onPick: async (item) => {
        await this.command(`/theme ${item.id}`);
        this.previewTheme(null);
      },
    });
  }

  pickSession(anchor) {
    openPicker({
      title: "Sessions in this folder",
      anchor,
      empty: "No other sessions yet.",
      items: this.state.sessions.map((s) => ({
        id: s.id,
        label: s.title,
        current: s.id === this.sessionId,
        detail: [when(s.updated), s.turns ? plural(s.turns, "turn") : "nothing asked yet", s.live ? "open" : ""]
          .filter(Boolean)
          .join("   "),
      })),
      onPick: (item) => this.switchSession(item.id),
    });
  }

  // -- keys ------------------------------------------------------------------------------------

  onKey(event) {
    if (popupOpen() || event.defaultPrevented) return;
    const typing = isTyping(event.target);
    if (!typing && event.key === "?") {
      event.preventDefault();
      this.openView("help");
    } else if (!typing && event.key === "/" && this.view === "main") {
      event.preventDefault();
      this.composer.focus("/");
    } else if (event.key === "Escape" && (!typing || event.target.id === "input")) {
      event.preventDefault();
      this.escape();
    } else if (!typing && this.view === "main" && this.panels.key(event)) {
      event.preventDefault();
    }
  }

  // -- drawing ---------------------------------------------------------------------------------

  // Draw on the next animation frame. `chrome` also redraws the tree, panels and rows, for
  // changes that come from outside the event stream (settings, focus, layout).
  schedule(chrome = false) {
    if (chrome) Object.assign(this.store.changes, { tree: true, panels: true, status: true });
    if (!this.frame) this.frame = requestAnimationFrame(() => this.flush());
  }

  flush() {
    this.frame = 0;
    const changes = this.store.take();
    this.applyLayout();
    // Everything that changes the height of the dock first, the transcript (which settles
    // its scroll position) last.
    this.panels.render(changes);
    if (changes.all || changes.status || changes.runs || changes.panels) {
      this.composer.render();
      this.status.render();
      this.crumb.render();
      document.title = `${this.store.meta?.title ?? "Clankit"} - Clankit`;
    } else if (changes.tree) {
      this.crumb.render();
    }
    this.tree.render(changes);
    this.transcript.render(changes);
  }

  tick() {
    const now = Date.now() / 1000;
    this.transcript.tick(now);
    this.tree.tick();
  }
}

new App().boot();
