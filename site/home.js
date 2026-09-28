// The landing page: install commands, and live numbers and recent models from the registry.
"use strict";

const { el, ago, short, command, logLine, loadIndex, registryUrl } = window.chungus;
const $ = (id) => document.getElementById(id);

$("start").append(
  command("cargo install --locked --git https://github.com/buwunny/chungus"),
  el("p", { class: "muted", text: "Then find a model here and fetch it from the swarm:" }),
  command(`export CHUNGUS_REGISTRY=${registryUrl}\nchungus search llama\nchungus fetch acme/tiny-llama --swarm -o tiny-llama/`),
);

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
