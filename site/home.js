// The landing page: install commands, and live numbers and recent models from the registry.
"use strict";

const { el, ago, short, command, logLine, loadIndex, registryUrl } = window.chungus;
const $ = (id) => document.getElementById(id);

const install = `v=v0.1.0
case "$(uname -sm)" in
  "Linux x86_64")  t=x86_64-unknown-linux-gnu ;;
  "Linux aarch64") t=aarch64-unknown-linux-gnu ;;
  "Darwin arm64")  t=aarch64-apple-darwin ;;
  "Darwin x86_64") t=x86_64-apple-darwin ;;
esac
curl -L https://github.com/buwunny/chungus/releases/download/$v/chungus-$v-$t.tar.gz | tar xz
sudo mv chungus-$v-$t/chungus /usr/local/bin/ && chungus --version`;

const step = (title, note, cmd) =>
  el("li", { class: "step" },
    el("h3", { text: title }),
    note ? el("p", { class: "muted", text: note }) : null,
    command(cmd),
  );

$("start").append(
  el("ol", { class: "steps" },
    step("Install", "Linux (x86_64, aarch64; glibc 2.39+) or macOS. Or build from source with cargo.", install),
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
