// Number, time and text formatting shared by every view. Columns rely on these widths.

export function tokens(count) {
  const n = Number(count) || 0;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function clock(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const seconds = String(s % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${seconds}` : `${minutes}:${seconds}`;
}

export function duration(seconds) {
  const s = Number(seconds) || 0;
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 60) return `${Math.round(s)}s`;
  return clock(s);
}

export function money(dollars) {
  if (dollars === null || dollars === undefined) return "";
  return dollars > 0 && dollars < 0.01 ? "<$0.01" : `$${dollars.toFixed(2)}`;
}

export function usageTotal(usage) {
  const u = usage || {};
  return (
    (u.input_tokens || 0) + (u.output_tokens || 0) + (u.cache_read_tokens || 0) + (u.cache_write_tokens || 0)
  );
}

// "anthropic/claude-sonnet-5-5" -> "sonnet-5-5": the part that tells models apart in a column.
export function alias(ref) {
  if (!ref) return "";
  const model = ref.includes("/") ? ref.slice(ref.indexOf("/") + 1) : ref;
  return model.replace(/^claude-/, "");
}

export function when(timestamp) {
  const date = new Date(timestamp * 1000);
  const now = new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (date.toDateString() === now.toDateString()) return `today ${time}`;
  const day = date.toLocaleDateString([], { day: "numeric", month: "short" });
  return `${day} ${time}`;
}

export function timeOfDay(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function firstLine(text, max = 80) {
  const line = String(text || "").split("\n").find((l) => l.trim()) || "";
  const trimmed = line.trim();
  return trimmed.length > max ? `${trimmed.slice(0, max - 1)}…` : trimmed;
}

export function plural(count, word, many = `${word}s`) {
  return `${count} ${count === 1 ? word : many}`;
}

export function diffStat(diff) {
  let added = 0;
  let removed = 0;
  for (const line of String(diff || "").split("\n")) {
    if (line.startsWith("+") && !line.startsWith("+++")) added += 1;
    else if (line.startsWith("-") && !line.startsWith("---")) removed += 1;
  }
  return { added, removed };
}

export function looksLikeDiff(text) {
  return /^(--- |\+\+\+ |@@ )/m.test(String(text || ""));
}
