// The node leaderboard: registered nodes ranked on what others confirm (anchors' probes
// and downloaders' receipts), and each node's daily history at #/<peer id>.
"use strict";

const { el, ago, bytes, short, getJSON, registryUrl } = window.chungus;
const $ = (id) => document.getElementById(id);

const METRICS = [
  ["bytes", "Bytes served"],
  ["downloads", "Downloads served"],
  ["uptime", "Uptime"],
  ["models", "Models held"],
];
const WINDOWS = [7, 30, 90];

function state() {
  const p = new URLSearchParams(location.search);
  const metric = METRICS.some(([m]) => m === p.get("metric")) ? p.get("metric") : "bytes";
  const days = WINDOWS.includes(+p.get("days")) ? +p.get("days") : 30;
  return { metric, days };
}

function setState(next) {
  const url = new URL(location.href);
  for (const [k, v] of Object.entries(next)) url.searchParams.set(k, v);
  url.hash = "";
  history.pushState(null, "", url);
  route();
}

const uptime = (u) => (u == null ? "–" : `${(u * 100).toFixed(u < 0.999 ? 1 : 0)}%`);
const nodeName = (n) => n.name ?? "(name hidden)";

async function renderBoard() {
  const { metric, days } = state();
  const view = $("board");
  $("node").hidden = true;
  let rows, totals;
  try {
    [rows, totals] = await Promise.all([
      getJSON(`${registryUrl}/v1/leaderboard?metric=${metric}&days=${days}`),
      getJSON(`${registryUrl}/v1/downloads`).catch(() => null),
    ]);
  } catch (e) {
    $("status").textContent = "Couldn't reach the registry. Try again in a minute.";
    console.error(e);
    return;
  }
  $("status").textContent = "";
  view.replaceChildren();
  view.append(el("h1", { text: "Node leaderboard" }));
  if (totals) {
    view.append(
      el("div", { class: "stats" },
        el("div", {},
          el("div", { class: "label", text: "Served by the community" }),
          el("div", { class: "value", text: bytes(totals.bytes_served) }),
          el("div", { class: "note", text: "credited by downloaders' receipts" }),
        ),
        el("div", {},
          el("div", { class: "label", text: "Downloads" }),
          el("div", { class: "value", text: totals.total.toLocaleString() }),
          el("div", { class: "note", text: "one per model, network and day" }),
        ),
        el("div", {},
          el("div", { class: "label", text: "Listed nodes" }),
          el("div", { class: "value", text: rows.length.toLocaleString() }),
          el("div", { class: "note", text: "opted in with --leaderboard" }),
        ),
      ),
    );
  }
  const windows = el("p", { class: "meta" }, "Last ");
  WINDOWS.forEach((d, i) => {
    if (i) windows.append(" · ");
    windows.append(
      d === days
        ? el("strong", { text: `${d} days` })
        : el("a", { href: `?metric=${metric}&days=${d}`, "data-days": d, text: `${d} days` }),
    );
  });
  windows.addEventListener("click", (e) => {
    const d = e.target.dataset?.days;
    if (d) {
      e.preventDefault();
      setState({ days: d });
    }
  });
  view.append(windows);
  if (!rows.length) {
    view.append(
      el("p", {
        text: "No nodes are listed yet. Run a node with --leaderboard \"<your name>\" to join.",
      }),
    );
    view.hidden = false;
    return;
  }
  const head = el("tr", {}, el("th", { class: "num", text: "#" }), el("th", { text: "Node" }));
  for (const [m, label] of METRICS) {
    const th = el("th", { class: "num" });
    if (m === metric) th.append(el("strong", { text: `${label} ↓` }));
    else {
      const a = el("a", { href: `?metric=${m}&days=${days}`, text: label });
      a.addEventListener("click", (e) => {
        e.preventDefault();
        setState({ metric: m });
      });
      th.append(a);
    }
    head.append(th);
  }
  const body = rows.map((r, i) =>
    el("tr", {},
      el("td", { class: "num", text: String(i + 1) }),
      el("td", {},
        el("a", { href: `#/${r.peer}`, text: nodeName(r) }),
        el("div", { class: "meta", text: short(r.peer), title: r.peer }),
      ),
      el("td", { class: "num", text: bytes(r.bytes), title: `${r.bytes.toLocaleString()} bytes` }),
      el("td", { class: "num", text: r.downloads.toLocaleString() }),
      el("td", { class: "num", text: uptime(r.uptime) }),
      el("td", { class: "num", text: r.models.toLocaleString() }),
    ),
  );
  view.append(
    el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body))),
    el("p", {
      class: "muted",
      text: "Uptime and models held come from anchors asking each node for a random chunk every few minutes. Bytes and downloads come from receipts, capped at one model's worth per downloader.",
    }),
  );
  view.hidden = false;
}

async function renderNode(peer) {
  const view = $("node");
  $("board").hidden = true;
  let h;
  try {
    h = await getJSON(`${registryUrl}/v1/nodes/${encodeURIComponent(peer)}`);
  } catch (e) {
    $("status").textContent = "";
    view.replaceChildren(el("h1", { text: short(peer) }), el("p", { text: "This node isn't registered." }));
    view.hidden = false;
    return;
  }
  $("status").textContent = "";
  const sum = h.days.reduce(
    (s, d) => ({ bytes: s.bytes + d.bytes, downloads: s.downloads + d.downloads, probes: s.probes + d.probes, answered: s.answered + d.answered }),
    { bytes: 0, downloads: 0, probes: 0, answered: 0 },
  );
  view.replaceChildren(
    el("p", {}, el("a", { href: location.pathname + location.search, text: "← Leaderboard" })),
    el("h1", { text: nodeName(h) }),
    el("p", { class: "meta", text: `Peer ${h.peer} · registered ${ago(h.registered)} · last seen ${ago(h.last_seen)}` }),
    el("div", { class: "stats" },
      el("div", {},
        el("div", { class: "label", text: "Bytes served" }),
        el("div", { class: "value", text: bytes(sum.bytes) }),
        el("div", { class: "note", text: `over ${h.days.length} day${h.days.length === 1 ? "" : "s"}` }),
      ),
      el("div", {},
        el("div", { class: "label", text: "Downloads served" }),
        el("div", { class: "value", text: sum.downloads.toLocaleString() }),
        el("div", { class: "note", text: "distinct networks per day" }),
      ),
      el("div", {},
        el("div", { class: "label", text: "Uptime" }),
        el("div", { class: "value", text: uptime(sum.probes ? sum.answered / sum.probes : null) }),
        el("div", { class: "note", text: `${sum.answered} of ${sum.probes} probes answered` }),
      ),
      el("div", {},
        el("div", { class: "label", text: "Models held" }),
        el("div", { class: "value", text: h.held.length.toLocaleString() }),
        el("div", { class: "note", text: "verified in the last 7 days" }),
      ),
    ),
    el("h2", { text: "By day" }),
    el("div", { class: "table-wrap" },
      el("table", {},
        el("thead", {},
          el("tr", {},
            el("th", { text: "Day (UTC)" }),
            el("th", { class: "num", text: "Bytes" }),
            el("th", { class: "num", text: "Downloads" }),
            el("th", { class: "num", text: "Probes answered" }),
            el("th", { text: "Totals" }),
          ),
        ),
        el("tbody", {},
          ...h.days.map((d) =>
            el("tr", {},
              el("td", { text: d.day }),
              el("td", { class: "num", text: bytes(d.bytes) }),
              el("td", { class: "num", text: d.downloads.toLocaleString() }),
              el("td", { class: "num", text: d.probes ? `${d.answered} / ${d.probes}` : "–" }),
              el("td", { text: d.signed ? "signed" : "still counting" }),
            ),
          ),
        ),
      ),
    ),
  );
  view.hidden = false;
}

function route() {
  const hash = decodeURIComponent(location.hash.slice(1));
  if (hash.startsWith("/") && hash.length > 1) renderNode(hash.slice(1));
  else renderBoard();
}

window.addEventListener("hashchange", route);
window.addEventListener("popstate", route);
route();
