// The landing page: install commands, and live numbers and recent models from the registry.
"use strict";

const { el, ago, short, command, logLine, loadIndex, registryUrl } = window.chungus;
const $ = (id) => document.getElementById(id);

// One install command per release target, with tabs to switch and the visitor's platform
// picked by default.
const version = "v0.1.0";
const platforms = [
  { id: "linux-x64", label: "Linux x86_64", target: "x86_64-unknown-linux-gnu" },
  { id: "linux-arm", label: "Linux arm64", target: "aarch64-unknown-linux-gnu" },
  { id: "mac-arm", label: "macOS Apple silicon", target: "aarch64-apple-darwin" },
  { id: "mac-x64", label: "macOS Intel", target: "x86_64-apple-darwin" },
];
const installCmd = (t) => `v=${version} t=${t}
curl -L https://github.com/buwunny/chungus/releases/download/$v/chungus-$v-$t.tar.gz | tar xz
sudo mv chungus-$v-$t/chungus /usr/local/bin/ && chungus --version`;

// A best guess from what the browser tells us. Browsers hide the CPU on macOS, so a Mac
// counts as Apple silicon, the common case; the tabs are there when it's wrong.
async function detectPlatform() {
  const ua = navigator.userAgent;
  let arch = "";
  try {
    arch = (await navigator.userAgentData?.getHighEntropyValues(["architecture"]))?.architecture || "";
  } catch {}
  if (/Mac/.test(ua)) return arch === "x86" ? "mac-x64" : "mac-arm";
  if (/Windows/.test(ua)) return "windows";
  if (arch === "arm" || /aarch64|arm64|armv8/i.test(ua)) return "linux-arm";
  return "linux-x64";
}

function installer() {
  const panel = el("div", { class: "install-panel" });
  const note = el("p", { class: "muted install-note" });
  let touched = false;
  const tabs = platforms.map((p) => {
    const b = el("button", { type: "button", class: "tab", "aria-pressed": "false", text: p.label });
    b.addEventListener("click", () => {
      touched = true;
      pick(p.id);
    });
    return b;
  });
  function pick(id, detected) {
    platforms.forEach((p, i) => tabs[i].setAttribute("aria-pressed", String(p.id === id)));
    panel.replaceChildren(command(installCmd(platforms.find((p) => p.id === id).target)));
    note.textContent = detected === "windows"
      ? "There's no Windows build yet. Use WSL with Ubuntu 24.04 or newer, and the Linux command."
      : "";
  }
  pick("linux-x64");
  detectPlatform().then((id) => touched || pick(id === "windows" ? "linux-x64" : id, id));
  return el("div", { class: "installer" },
    el("div", { class: "tabs", role: "group", "aria-label": "Platform" }, ...tabs),
    note,
    panel,
  );
}

const step = (title, note, body) =>
  el("li", { class: "step" },
    el("h3", { text: title }),
    note ? el("p", { class: "muted", text: note }) : null,
    typeof body === "string" ? command(body) : body,
  );

$("start").append(
  el("ol", { class: "steps" },
    step("Install", "Linux needs glibc 2.39 or newer. Or build from source with cargo.", installer()),
    step("Find a model and download it",
      "Every chunk is checked against its hash, and the publisher's signature is required. On Linux, chungus mount instead starts loading before the download finishes.",
      `export CHUNGUS_REGISTRY=${registryUrl}\nchungus search llama\nchungus fetch acme/tiny-llama --swarm -o tiny-llama/`),
    step("Share it back", "A node seeds everything in its store, even from behind NAT.", "chungus node"),
    step("Publish your own", "Weights as safetensors or GGUF. Pickle-based files are refused.",
      "chungus keygen\nchungus pack path/to/model          # prints: root <hash>\nchungus publish <hash> --name you/my-model --description \"What it is\"\nchungus node"),
  ),
  el("p", { class: "muted" },
    "chungus is alpha (v0.1.0). Formats are versioned, but expect rough edges. ",
    el("a", { href: "https://github.com/buwunny/chungus#quickstart", text: "Full quickstart" }),
  ),
);

// Bytes to download per model, chungus vs Xet, from bench/results/2026-09-28-arch.md.
const receipts = [
  { name: "TinyLlama 1.1B Chat", kind: "BF16", chungus: 1610.7, xet: 1902.1 },
  { name: "Qwen3 0.6B", kind: "BF16", chungus: 878.7, xet: 1062.4 },
  { name: "Qwen2.5 0.5B Instruct", kind: "BF16", chungus: 728.0, xet: 865.4 },
  { name: "Qwen3 0.6B FP8", kind: "FP8", chungus: 598.4, xet: 732.7 },
  { name: "Qwen2.5 0.5B GGUF Q8", kind: "GGUF Q8_0", chungus: 504.9, xet: 570.9 },
  { name: "Qwen2.5 0.5B GGUF Q4", kind: "GGUF Q4_K_M", chungus: 478.8, xet: 485.6 },
  { name: "GPT-2", kind: "F32", chungus: 440.4, xet: 480.2 },
  { name: "Pythia 160M", kind: "F16", chungus: 293.4, xet: 326.8 },
  { name: "SmolLM2 135M Instruct", kind: "BF16", chungus: 199.1, xet: 271.5 },
  { name: "MiniLM L6", kind: "F32", chungus: 79.1, xet: 85.7 },
];

const mbFmt = (n) => `${n.toLocaleString("en-US", { maximumFractionDigits: 0 })} MB`;
const saved = (r) => `−${((100 * (r.xet - r.chungus)) / r.xet).toFixed(1)}%`;

function drawReceipts() {
  const chart = $("receipts-chart");
  if (!chart) return;
  const max = Math.max(...receipts.map((r) => r.xet));
  const bar = (who, v) => {
    const b = el("span", { class: `bar ${who}`, title: `${who}: ${mbFmt(v)}` });
    b.style.width = `${(100 * v) / max}%`;
    return b;
  };
  for (const r of receipts) {
    chart.append(
      el("div", { class: "bar-row" },
        el("div", { class: "bar-label" },
          el("span", { class: "bar-name", text: r.name }),
          el("span", { class: "meta", text: `${r.kind} · ${mbFmt(r.chungus)} vs ${mbFmt(r.xet)}` }),
        ),
        el("div", { class: "bar-pair", "aria-hidden": "true" }, bar("chungus", r.chungus), bar("xet", r.xet)),
        el("div", { class: "bar-diff", text: saved(r) }),
      ),
    );
  }
  $("receipts-table").append(
    el("thead", {}, el("tr", {},
      ...["Model", "Kind", "chungus", "Xet", "chungus vs Xet"].map((t) => el("th", { text: t })))),
    el("tbody", {}, ...receipts.map((r) => el("tr", {},
      el("td", { text: r.name }),
      el("td", { text: r.kind }),
      el("td", { text: mbFmt(r.chungus) }),
      el("td", { text: mbFmt(r.xet) }),
      el("td", { text: saved(r) }),
    ))),
  );
}
drawReceipts();

// Click the bunny on the pile and it gets bigger, up to a proper big chungus, then
// starts over.
const buddy = document.querySelector(".pile .buddy");
if (buddy) {
  const sizes = [1.5, 1.75, 2, 2.3, 2.6];
  let i = 0;
  buddy.addEventListener("click", () => {
    i = (i + 1) % sizes.length;
    const s = sizes[i];
    // Keep its feet on the top chunk and centered on it as it grows.
    buddy.style.transform = `translate(${150 - 32 * s}px, ${141.25 - 61.5 * s}px) scale(${s})`;
  });
}

async function main() {
  let models, head;
  try {
    ({ models, head } = await loadIndex());
  } catch (e) {
    $("recent").replaceChildren(el("p", { class: "muted", text: "Couldn't reach the registry right now." }));
    console.error(e);
    return;
  }
  const n = models.length;
  $("stats").textContent = head
    ? `${n} model${n === 1 ? "" : "s"} published · ${head.size} signed log entr${head.size === 1 ? "y" : "ies"}`
    : `${n} model${n === 1 ? "" : "s"} published`;
  $("log").textContent = logLine(head);
  const recent = $("recent");
  recent.replaceChildren();
  if (!n) recent.append(el("p", { class: "muted", text: "Nothing yet. Be the first to publish." }));
  for (const m of models.slice(0, 5)) {
    const latest = m.revs[0];
    recent.append(
      el("a", { class: "hit", href: "search.html#/" + m.name },
        el("div", { class: "name", text: m.name }),
        latest.description ? el("div", { class: "desc", text: latest.description }) : null,
        el("div", { class: "meta", text: `updated ${ago(latest.time)} · by ${short(latest.publisher)}` }),
      ),
    );
  }
}

main();
