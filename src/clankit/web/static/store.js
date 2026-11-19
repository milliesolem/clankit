// The session as the page knows it: a snapshot from the server, then every event folded in
// with the same rules as clankit.agent.state.SessionView, so both stay identical.
//
// Folding only mutates data and records what changed; views read `take()` once per animation
// frame and patch the DOM for exactly those changes.

const TERMINAL = new Set(["done", "failed", "cancelled"]);

function freshChanges(all = false) {
  return {
    all, // everything must be redrawn (a snapshot arrived)
    tree: all, // nodes were added or changed
    nodes: new Set(), // ids of nodes whose item list changed shape
    items: new Set(), // "node/item" keys of items whose content changed
    appends: new Map(), // "node/item" -> text streamed onto the end of that item
    panels: all, // permission requests or questions changed
    status: all, // usage, run state or session facts changed
    runs: all, // a run started or finished
  };
}

export function itemKey(nodeId, itemId) {
  return `${nodeId}/${itemId}`;
}

export class Store {
  constructor() {
    this.reset(null);
  }

  reset(session) {
    const snap = session?.snapshot ?? {};
    this.meta = session?.meta ?? null;
    this.seq = snap.seq ?? 0;
    this.running = Boolean(snap.running);
    this.run = snap.run ?? null;
    this.lastRun = snap.last_run ?? null;
    this.roots = [...(snap.roots ?? [])];
    this.nodes = new Map();
    for (const [id, node] of Object.entries(snap.nodes ?? {})) {
      this.nodes.set(id, { ...node, children: [...node.children], items: node.items.map((i) => ({ ...i })) });
    }
    this.total = { ...(snap.total_usage ?? {}) };
    this.permissions = new Map((snap.permissions ?? []).map((p) => [p.request_id, p]));
    this.questions = new Map((snap.questions ?? []).map((q) => [q.request_id, q]));
    this.runs = (session?.runs ?? []).map((r) => ({ ...r }));
    this.changes = freshChanges(true);
  }

  take() {
    const changes = this.changes;
    this.changes = freshChanges(false);
    return changes;
  }

  // -- folding ---------------------------------------------------------------------------------

  apply(event) {
    if (event.seq) {
      if (event.seq <= this.seq) return; // already folded (a replay overlapping the snapshot)
      this.seq = event.seq;
    }
    const handler = FOLD[event.kind];
    if (handler) handler(this, event);
  }

  setMeta(meta) {
    this.meta = meta;
    this.changes.status = true;
  }

  appendItem(nodeId, item) {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    node.items.push(item);
    this.changes.nodes.add(nodeId);
  }

  findItem(nodeId, itemId) {
    const node = this.nodes.get(nodeId);
    if (!node) return null;
    for (let i = node.items.length - 1; i >= 0; i -= 1) if (node.items[i].id === itemId) return node.items[i];
    return null;
  }

  extend(nodeId, itemId, kind, event) {
    const item = this.findItem(nodeId, itemId);
    if (!item) {
      this.appendItem(nodeId, { id: itemId, kind, seq: event.seq, ts: event.ts, text: event.text, done: false });
      return;
    }
    item.text += event.text;
    const key = itemKey(nodeId, itemId);
    this.changes.appends.set(key, (this.changes.appends.get(key) ?? "") + event.text);
  }

  finalise(nodeId, itemId, kind, text, event, html) {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    const item = this.findItem(nodeId, itemId);
    if (!text) {
      if (item) {
        node.items.splice(node.items.indexOf(item), 1);
        this.changes.nodes.add(nodeId);
      }
      return;
    }
    if (!item) {
      this.appendItem(nodeId, { id: itemId, kind, seq: event.seq, ts: event.ts, text, done: true, html });
      return;
    }
    Object.assign(item, { text, done: true, seq: event.seq, ts: event.ts, html });
    this.changes.items.add(itemKey(nodeId, itemId));
    this.changes.nodes.add(nodeId); // its position by sequence number may have moved
  }

  // -- reading ---------------------------------------------------------------------------------

  node(id) {
    return this.nodes.get(id) ?? null;
  }

  // Depth-first (depth, node) pairs, like SessionView.walk.
  walk(ids = this.roots, depth = 0, out = []) {
    for (const id of ids) {
      const node = this.nodes.get(id);
      if (!node) continue;
      out.push([depth, node]);
      this.walk(node.children, depth + 1, out);
    }
    return out;
  }

  path(id) {
    const out = [];
    for (let node = this.nodes.get(id); node; node = this.nodes.get(node.parent_id)) out.unshift(node);
    return out;
  }

  descendants(id) {
    const node = this.nodes.get(id);
    if (!node) return 0;
    return node.children.reduce((sum, child) => sum + 1 + this.descendants(child), 0);
  }

  needsInput(nodeId) {
    for (const p of this.permissions.values()) if (p.node_id === nodeId) return true;
    for (const q of this.questions.values()) if (q.node_id === nodeId) return true;
    return false;
  }

  // The status a tree row shows: a pending question or permission outranks the node's own.
  displayStatus(node) {
    return this.needsInput(node.id) ? "input" : node.status;
  }

  // The top-level transcript as the user sees it: their prompts, then what the root agents
  // said and did. A root agent's own user messages within a known run are left out: they are
  // the prompt again, or text the orchestrator wrote (a recap, review findings).
  conversation() {
    const entries = [];
    for (const run of this.runs) {
      entries.push({ nodeId: "", item: { id: `run-${run.run_id}`, kind: "prompt", seq: run.seq, text: run.prompt } });
      if (run.status === "failed" || run.status === "cancelled") {
        entries.push({
          nodeId: "",
          item: { id: `end-${run.run_id}`, kind: "end", seq: run.finished_seq, status: run.status },
        });
      }
    }
    const firstRun = this.runs.length ? this.runs[0].seq : Infinity;
    for (const rootId of this.roots) {
      const root = this.nodes.get(rootId);
      if (!root) continue;
      for (const item of root.items) {
        if (item.kind === "user" && item.seq > firstRun) continue;
        entries.push({ nodeId: rootId, item });
      }
    }
    entries.sort((a, b) => a.item.seq - b.item.seq);
    return entries;
  }

  nodeTranscript(nodeId) {
    const node = this.nodes.get(nodeId);
    return node ? node.items.map((item) => ({ nodeId, item })) : [];
  }

  maxDepth() {
    let depth = 0;
    for (const node of this.nodes.values()) depth = Math.max(depth, node.depth);
    return depth;
  }

  counts() {
    const counts = {};
    for (const node of this.nodes.values()) {
      const status = this.displayStatus(node);
      counts[status] = (counts[status] ?? 0) + 1;
    }
    return counts;
  }
}

const FOLD = {
  run_started(store, e) {
    store.running = true;
    store.run = e;
    store.runs.push({ seq: e.seq, run_id: e.run_id, prompt: e.prompt, method: e.method, status: "", finished_seq: 0 });
    Object.assign(store.changes, { status: true, runs: true, panels: true });
  },
  run_finished(store, e) {
    store.running = false;
    store.lastRun = e;
    store.permissions.clear();
    store.questions.clear();
    const run = [...store.runs].reverse().find((r) => r.run_id === e.run_id);
    if (run) Object.assign(run, { status: e.status, finished_seq: e.seq });
    Object.assign(store.changes, { status: true, runs: true, panels: true, tree: true });
  },
  node_added(store, e) {
    const info = e.node;
    store.nodes.set(info.id, { ...info, children: [], items: [], usage: {}, context_tokens: 0, started: 0, finished: 0 });
    const parent = info.parent_id ? store.nodes.get(info.parent_id) : null;
    if (parent) parent.children.push(info.id);
    else store.roots.push(info.id);
    store.changes.tree = true;
  },
  node_changed(store, e) {
    const node = store.nodes.get(e.node_id);
    if (!node) return;
    if (e.status !== null && e.status !== undefined) {
      node.status = e.status;
      if (e.status === "running" && !node.started) node.started = e.ts;
      node.finished = TERMINAL.has(e.status) ? e.ts : 0;
    }
    for (const field of ["title", "role", "model", "summary"]) {
      if (e[field] !== null && e[field] !== undefined) node[field] = e[field];
    }
    store.changes.tree = true;
    store.changes.status = true;
  },
  user_message(store, e) {
    store.appendItem(e.node_id, { id: `u${e.seq}`, kind: "user", seq: e.seq, ts: e.ts, text: e.text, done: true });
  },
  thinking_delta(store, e) {
    store.extend(e.node_id, `${e.message_id}:thinking`, "thinking", e);
  },
  text_delta(store, e) {
    store.extend(e.node_id, `${e.message_id}:text`, "assistant", e);
  },
  assistant_message(store, e) {
    store.finalise(e.node_id, `${e.message_id}:thinking`, "thinking", e.thinking, e);
    store.finalise(e.node_id, `${e.message_id}:text`, "assistant", e.text, e, e.html);
  },
  tool_started(store, e) {
    store.appendItem(e.node_id, {
      id: e.call_id,
      kind: "tool",
      seq: e.seq,
      ts: e.ts,
      text: "",
      done: false,
      tool: e.tool,
      summary: e.summary,
      input: e.input,
      output: "",
      is_error: false,
      duration: 0,
      display: null,
    });
  },
  tool_finished(store, e) {
    const item = store.findItem(e.node_id, e.call_id);
    if (!item) return;
    Object.assign(item, { done: true, output: e.output, is_error: e.is_error, duration: e.duration, display: e.display });
    store.changes.items.add(itemKey(e.node_id, e.call_id));
  },
  usage_changed(store, e) {
    const node = store.nodes.get(e.node_id);
    if (node) {
      node.usage = { ...e.usage };
      node.context_tokens = e.context_tokens;
    }
    store.total = { ...e.total };
    store.changes.tree = true;
    store.changes.status = true;
  },
  permission_requested(store, e) {
    store.permissions.set(e.request_id, e);
    Object.assign(store.changes, { panels: true, tree: true, status: true });
  },
  permission_resolved(store, e) {
    store.permissions.delete(e.request_id);
    Object.assign(store.changes, { panels: true, tree: true, status: true });
  },
  question_asked(store, e) {
    store.questions.set(e.request_id, e);
    Object.assign(store.changes, { panels: true, tree: true, status: true });
  },
  question_answered(store, e) {
    store.questions.delete(e.request_id);
    Object.assign(store.changes, { panels: true, tree: true, status: true });
  },
  notice(store, e) {
    const target = store.nodes.has(e.node_id) ? e.node_id : store.roots[store.roots.length - 1];
    if (!target) return;
    store.appendItem(target, { id: `n${e.seq}`, kind: "notice", seq: e.seq, ts: e.ts, text: e.text, level: e.level, done: true });
  },
};
