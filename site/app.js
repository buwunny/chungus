// The chungus search site: loads the registry's index of published models once and
// searches it in the browser, ranking results the same way `chungus search` does.
"use strict";

const { el, ago, bytes, short, command, logLine, loadIndex, getJSON, registryUrl } = window.chungus;
const $ = (id) => document.getElementById(id);

let models = []; // one per name: { name, revs: [hit...] } with revs newest first
let head = null;

function words(s) {
  return s.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
}

// Every word of the query must start some word of the name or the description; name
// matches count double. Ties go to the newest.
function search(query) {
  const terms = words(query);
  if (!terms.length) return models;
  const hits = [];
  for (const m of models) {
    const latest = m.revs[0];
    const nameWords = words(m.name);
    const descWords = words(latest.description);
    let score = 0;
    const all = terms.every((t) => {
      const inName = nameWords.some((w) => w.startsWith(t));
      const inDesc = descWords.some((w) => w.startsWith(t));
      score += 2 * inName + inDesc;
      return inName || inDesc;
    });
    if (all) hits.push([score, m]);
  }
  hits.sort((a, b) => b[0] - a[0] || b[1].revs[0].time - a[1].revs[0].time);
  return hits.map((h) => h[1]);
}

function renderList(query) {
  const found = search(query);
  const list = $("list");
  list.replaceChildren();
  const n = models.length;
  $("status").textContent = query.trim()
    ? `${found.length} of ${n} model${n === 1 ? "" : "s"} match`
    : n
      ? `${n} model${n === 1 ? "" : "s"}, newest first`
      : "No models have been published yet.";
  for (const m of found.slice(0, 200)) {
    const latest = m.revs[0];
    list.append(
      el("a", { class: "hit", href: "#/" + m.name },
        el("div", { class: "name", text: m.name }),
        latest.description ? el("div", { class: "desc", text: latest.description }) : null,
        el("div", {
          class: "meta",
          text: [
            latest.size != null ? bytes(latest.size) : null,
            `${m.revs.length} rev${m.revs.length === 1 ? "" : "s"}`,
            `updated ${ago(latest.time)}`,
            `by ${short(latest.publisher)}`,
          ]
            .filter(Boolean)
            .join(" · "),
        }),
      ),
    );
  }
  list.hidden = false;
  $("model").hidden = true;
}

function renderModel(name) {
  const m = models.find((x) => x.name === name);
  const view = $("model");
  view.replaceChildren();
  $("list").hidden = true;
  view.hidden = false;
  if (!m) {
    $("status").textContent = "";
    view.append(el("h1", { text: name }), el("p", { text: "This name isn't published, or it was blocked." }));
    return;
  }
  const latest = m.revs[0];
  const ref = `${m.name}@${latest.rev}`;
  const setup = `export CHUNGUS_REGISTRY=${registryUrl}`;
  $("status").textContent = "";
  // append() would turn a null (no description, not gated) into the text "null".
  view.append(
    ...[
      el("h1", { text: m.name }),
      latest.description ? el("p", { text: latest.description }) : null,
      el("p", { class: "meta", text: `Published by ${latest.publisher}` }),
      el("div", { id: "summary" }),
      latest.gated
        ? el("p", {
            text: `Gated: accept the license at huggingface.co/${latest.gated}, then set HF_TOKEN to your Hugging Face token. Only your token's access is checked; it goes to this registry and Hugging Face, never to peers.`,
          })
        : null,
      el("h2", { text: "Download" }),
      command(`${setup}${latest.gated ? "\nexport HF_TOKEN=hf_..." : ""}\nchungus fetch ${ref} --swarm -o ${m.name.split("/")[1]}/`),
      el("p", { class: "muted", text: "Or mount it and start loading before the download finishes:" }),
      command(`chungus mount ${ref} ${m.name.split("/")[1]}/ --swarm`),
      el("p", {
        class: "muted",
        text: "chungus checks every chunk against its hash and refuses a model its publisher hasn't signed.",
      }),
    ].filter(Boolean),
  );
  const rows = m.revs.map((r) =>
    el("tr", {},
      el("td", {}, el("code", { text: r.rev })),
      el("td", {}, el("code", { text: r.root, title: r.root })),
      el("td", { text: ago(r.time), title: new Date(r.time * 1000).toISOString() }),
    ),
  );
  view.append(
    el("div", { id: "files" }),
    el("h2", { text: "Revisions" }),
    el("div", { class: "table-wrap" },
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", { text: "Rev" }), el("th", { text: "Manifest root" }), el("th", { text: "Published" }))),
        el("tbody", {}, ...rows),
      ),
    ),
  );
  renderSummary(m.name, latest.root);
}

// 3821079552 -> "3.8B", the way model cards write parameter counts.
function count(x) {
  for (const [div, unit] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]]) {
    if (x >= div) {
      const v = x / div;
      return `${v < 100 ? +v.toFixed(1) : Math.round(v)}${unit}`;
    }
  }
  return String(x);
}

// Stats and files of the latest rev, from the registry's summary of its manifest. The
// index snapshot deployed with the site has no summaries, so this can come back empty.
async function renderSummary(name, root) {
  let s;
  try {
    s = await getJSON(`${registryUrl}/v1/summary/${root}`);
  } catch {
    return;
  }
  const box = document.getElementById("summary");
  const filesBox = document.getElementById("files");
  // The reader may have moved on to another model while this loaded.
  if (!box || !filesBox || decodeURIComponent(location.hash.slice(2)) !== name) return;
  const n = (x) => x.toLocaleString();
  const saved = s.size - s.unique_bytes;
  const shards = s.files.filter((f) => /\.(safetensors|gguf)$/i.test(f.path)).length;
  const dtypes = Object.keys(s.dtypes || {});
  const tiles = [
    s.params != null ? ["Parameters", count(s.params), `${n(s.params)} exactly`] : null,
    ["Size", bytes(s.size), `${bytes(s.weights)} of weights`],
    [
      "Format",
      s.formats.join(", ") || "none",
      [dtypes.join(", "), `${n(shards)} weight file${shards === 1 ? "" : "s"}`].filter(Boolean).join(" · "),
    ],
    ["Files", n(s.files.length), `${bytes(s.size - s.weights)} besides weights`],
    ["Chunks", n(s.unique_chunks), saved > 0 ? `${bytes(saved)} repeated within the model` : "all distinct"],
  ];
  const files = [...s.files].sort((a, b) => b.size - a.size);
  box.append(
    el("div", { class: "stats" },
      ...tiles.filter(Boolean).map(([label, value, note]) =>
        el("div", {},
          el("div", { class: "label", text: label }),
          // Words like "safetensors" shrink rather than break mid-word in a narrow tile.
          el("div", { class: /\d/.test(value) ? "value" : "value word", text: value }),
          el("div", { class: "note", text: note }),
        ),
      ),
    ),
  );
  filesBox.append(
    el("h2", { text: "Files" }),
    el("div", { class: "table-wrap" },
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", { text: "File" }), el("th", { class: "num", text: "Size" }))),
        el("tbody", {},
          ...files.map((f) =>
            el("tr", {},
              el("td", {}, el("code", { text: f.path })),
              el("td", { class: "num", text: bytes(f.size), title: `${n(f.size)} bytes` }),
            ),
          ),
        ),
      ),
    ),
  );
}

function route() {
  const hash = decodeURIComponent(location.hash.slice(1));
  if (hash.startsWith("/") && hash.length > 1) {
    renderModel(hash.slice(1));
  } else {
    renderList($("q").value);
  }
}

async function main() {
  const params = new URLSearchParams(location.search);
  $("q").value = params.get("q") || "";
  try {
    ({ models, head } = await loadIndex());
  } catch (e) {
    $("status").textContent = "Couldn't reach the registry. Try again in a minute.";
    console.error(e);
    return;
  }
  $("log").textContent = logLine(head);
  $("q").addEventListener("input", () => {
    const q = $("q").value;
    const url = new URL(location.href);
    if (q) url.searchParams.set("q", q);
    else url.searchParams.delete("q");
    url.hash = "";
    history.replaceState(null, "", url);
    renderList(q);
  });
  $("search").addEventListener("submit", (e) => e.preventDefault());
  window.addEventListener("hashchange", route);
  route();
}

main();
