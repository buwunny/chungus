# chungus

A peer-to-peer network for distributing AI models, their runtimes and Docker AI images. Think of it as a decentralized Hugging Face with its own take on Xet-style storage.

This repository currently holds **milestones 1 to 4**: the storage format, a benchmark tool, sharing models between machines on a LAN, a drop-in Hugging Face cache, signed models, sharing over the internet, and a registry for names, search and blocklists.

> **Alpha.** chungus is at v0.1.0: formats are versioned and old data keeps working (see [docs/formats.md](docs/formats.md)), but expect rough edges and don't rely on it for anything production-critical yet.

## Quickstart

**1. Install** (Linux x86_64/aarch64 with glibc 2.39+, or macOS; see [Install](#install) for other options):

```sh
v=v0.1.0
case "$(uname -sm)" in
  "Linux x86_64")  t=x86_64-unknown-linux-gnu ;;
  "Linux aarch64") t=aarch64-unknown-linux-gnu ;;
  "Darwin arm64")  t=aarch64-apple-darwin ;;
  "Darwin x86_64") t=x86_64-apple-darwin ;;
esac
curl -L https://github.com/buwunny/chungus/releases/download/$v/chungus-$v-$t.tar.gz | tar xz
sudo mv chungus-$v-$t/chungus /usr/local/bin/ && chungus --version
```

**2. Find a model and download it** from the swarm. Every chunk is checked against its hash, and the publisher's signature is required:

```sh
export CHUNGUS_REGISTRY=https://<the registry's address>
chungus search llama
chungus fetch acme/tiny-llama --swarm -o tiny-llama/
```

On Linux, `chungus mount acme/tiny-llama tiny-llama/ --swarm` instead makes the files appear at once and downloads them as they're read (needs `fuse3`). Gated models also need `HF_TOKEN` (see [Gated models](#gated-and-licensed-models)).

**3. Share it back.** A node seeds everything in its store, reachable even from behind NAT:

```sh
chungus node
```

**4. Publish your own** (safetensors or GGUF weights):

```sh
chungus keygen                                   # once: your signing key in ~/.chungus/key
chungus pack path/to/model                       # prints: root <hash>
chungus publish <hash> --name you/my-model --description "What it is"
chungus node                                     # so others can download it
```

Already use Hugging Face tools? `chungus hub` plus `export HF_ENDPOINT=http://localhost:8080` puts a shared cache under them without changing any code ([details](#drop-in-hugging-face-cache)).

## The pipeline

```
model files ─► segments ─► FastCDC chunks ─► BLAKE3 ─► float transform ─► zstd ─► chunk store
              (per tensor)   (~64 KiB)       (raw bytes)  (exponent split)
```

1. **Segments.** Safetensors files are split at tensor boundaries using the file's header, so a chunk never spans two tensors. Other files are one segment.
2. **Content-defined chunking.** FastCDC cuts each segment into chunks of 16–256 KiB (64 KiB average). Cut points inside float tensors are rounded to whole elements.
3. **Hashing.** Each chunk is addressed by the BLAKE3 hash of its *raw* bytes. Identical chunks are stored once, across files and across models.
4. **Float transform.** For BF16 and F32 tensors, each element is rearranged into an exponent byte and sign+mantissa bytes, then grouped into planes. Exponents are low-entropy and compress well. This is lossless: unpacking gives back the exact bits. The split and unshuffle run SIMD kernels (SSSE3 on x86-64, NEON on ARM) at 4 to 9 GB/s per core, faster than zstd decodes; `cargo run --release --example transform_speed` measures them.
5. **zstd.** Each chunk is compressed on its own so any chunk can be read independently. The encoder keeps whichever of {stored, zstd, transform + zstd} is smallest.

## Install

Each release has prebuilt binaries for Linux (x86_64, aarch64; glibc 2.39 or newer, e.g. Ubuntu 24.04, Debian 13) and macOS (Apple silicon, Intel) on the [releases page](https://github.com/buwunny/chungus/releases):

```sh
# Pick your platform: x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu,
# aarch64-apple-darwin or x86_64-apple-darwin
v=v0.1.0 t=aarch64-apple-darwin
curl -LO https://github.com/buwunny/chungus/releases/download/$v/chungus-$v-$t.tar.gz
tar xzf chungus-$v-$t.tar.gz && sudo mv chungus-$v-$t/chungus /usr/local/bin/
```

On macOS, a downloaded binary is quarantined; `xattr -d com.apple.quarantine /usr/local/bin/chungus` lets it run. Or build from source with `cargo build --release` (the binary is `target/release/chungus`). The node container is published as `ghcr.io/buwunny/chungus` for linux/amd64 and arm64, from every commit to main.

## Usage

```sh
cargo build --release

# Pack a model directory into a shared chunk store and write its manifest
./target/release/chungus pack path/to/model --store .chungus/store -o model.manifest.json

# Rebuild it (every chunk and every file is verified against its hash)
./target/release/chungus unpack model.manifest.json --store .chungus/store -o restored/

# Measure compression and dedup without writing anything
./target/release/chungus bench path/to/base-model path/to/fine-tune
```

`uv run bench/run.py` runs the benchmark suite on real models from Hugging Face: storage saved per dtype, dedup between versions, and download time direct vs through chungus. See [docs/benchmarks.md](docs/benchmarks.md).

## Sharing on a LAN

One machine packs a model and serves its store. It advertises itself over mDNS, so others on the same network find it without configuration.

```sh
# Machine A
./target/release/chungus pack path/to/model          # prints: root <hash>
./target/release/chungus serve                       # port 7447, advertised on the LAN

# Machine B
./target/release/chungus fetch <hash> -o model/      # finds A, downloads, verifies, unpacks
```

`fetch` downloads only the chunks it doesn't already have, so a second model that shares chunks with the first transfers less, and an interrupted fetch picks up where it stopped. Every chunk is checked against its BLAKE3 hash on arrival. A peer that sends bad data is skipped and the chunk is taken from the next peer. With several peers, chunks are spread across them. `--peer http://host:7447` adds a peer by hand (for networks that block multicast), and `--origin URL` names a server to use only when no peer has a chunk. `chungus list` shows the models in a store.

`serve` exposes the store read-only over plain HTTP to anyone who can reach the port. Run it only on networks you trust; encrypted, authenticated transport comes with the internet milestone.

## Drop-in Hugging Face cache

`chungus hub` speaks the part of the Hugging Face Hub API that `huggingface_hub`, `transformers`, vLLM and similar tools use. Point them at it and they work unchanged:

```sh
./target/release/chungus hub                          # http://localhost:8080
export HF_ENDPOINT=http://localhost:8080
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B')"
```

For each file, the hub serves it from its store if it has it. Otherwise it pulls the chunks from other hubs on the LAN (found over mDNS), and otherwise downloads the file from huggingface.co, streaming it to you while it packs it into the store. The next machine in the office gets it from the LAN.

- **Tokens** (`HF_TOKEN`) are forwarded only to huggingface.co, never to peers.
- **Files that can run code** (pickle-based `.bin`, `.pt`, `.ckpt` and the like) are passed straight through from huggingface.co and never cached or shared with peers. See [Safe files only](#safe-files-only).
- **Gated models** are served only to requests whose token huggingface.co accepts for that file. When huggingface.co can't be reached, a gated file is served only to a token that was accepted earlier in this run.
- **Peer copies are checked.** While huggingface.co is reachable, a file assembled from LAN peers must match its SHA-256 (or git blob hash). The last bytes are held back until it does, so a bad copy never arrives complete, and that file is then fetched from huggingface.co instead. With `--offline`, peers are trusted, as with `serve`.
- Tree listings drop Xet hashes, so clients download through the hub instead of going around it.

`--offline` never contacts huggingface.co, `--upstream URL` points at a different Hub-compatible server, and `--peer URL` adds a hub by hand.

No model handy? Generate a synthetic BF16 file:

```sh
cargo run --release --example synth -- synthetic.safetensors 64   # 64M parameters
```

## Signed models

A publisher signs a model's manifest root with an ed25519 key. The root commits to every file and chunk, so one signature covers the whole model, and anyone can check it without trusting the peer that served it.

```sh
# Publisher: create a key once (stored in ~/.chungus/key), then sign what you pack
chungus keygen
chungus sign <root>

# Anyone: see who signed a model, or require a particular key
chungus verify <root>
chungus verify <root> --trust chungus1<publisher key>

# Only download if a trusted key signed it; nothing is fetched otherwise
chungus fetch <root> --trust chungus1<publisher key>
```

Signatures travel with the manifest: `fetch` collects them from every peer and keeps only the valid ones, so a peer can't forge or swap one.

## Sharing over the internet

Nodes form a swarm with [libp2p](https://libp2p.io). Each node announces the models in its store on a Kademlia DHT; a fetcher asks the DHT who has a model and downloads chunks from all of them at once, verifying every chunk as it does on a LAN. Connections are encrypted (Noise over TCP, or QUIC).

```sh
# Share your store with the swarm, even from behind NAT
chungus node

# Anywhere: fetch a model from whoever has it
chungus fetch <root> --swarm -o model/
```

With no flags, both join through the project's public node (`/ip4/40.160.91.185/tcp/4001/p2p/12D3KooWRaVx8DKtusbdVeThtaFxqR7C8jgSvz6fArBwh52SCAeR`, also on QUIC), and `node` also uses it as its relay, so a node behind NAT is reachable without any setup. `--bootstrap` and `--relay` (or `CHUNGUS_BOOTSTRAP` and `CHUNGUS_RELAY`) replace the defaults, and `--no-default-bootstrap` leaves the public node out entirely, for a private swarm. A node started with `--public`, `--relay-server`, `--external` or `--download-only` asks for no relay. To run a swarm of your own:

```sh
# A public server (a VPS, say) that others join through, and relays for nodes behind NAT
chungus node --store relay --public --relay-server --no-default-bootstrap
#   prints: address /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW...

# At home, behind NAT: share your store, reachable through the relay
chungus node --bootstrap /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW... \
             --relay     /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW...
```

Models are announced at two levels. A node with the whole model announces its root; every node also announces each 64 MB block of a model it holds in full. A fetcher asks the DHT for both, so peers that are still downloading a model already serve the parts they have, and the swarm grows during a flash crowd instead of waiting for complete copies. Announcing blocks rather than individual chunks keeps the DHT small: a 140 GB model is about 2,200 blocks but 2 million chunks.

A node behind NAT is reached through its relay, and the two ends then try to hole-punch a direct connection (DCUtR). A node's identity lives in `node.key` in its store. `node` announces models added to the store while it runs within a minute.

`node` logs what it connects to, so you can tell it joined: `connected to bootstrap <peer>`, `joined the DHT`, and with `--relay`, `relay reservation accepted by <peer>` followed by a `/p2p-circuit` address that others can reach you at. `could not reach bootstrap <peer>: <error>` means the bootstrap node's port is closed or the address is wrong. The public node logs a line for each peer that connects to it.

### Limits and attack resistance

A node is someone's desktop, so it protects its owner. `--max-upload <MB/s>` caps upload bandwidth (on `serve` too), `--max-connections`, `--max-requests-per-peer` and `--max-uploads` bound how many peers and requests it serves at once (a peer over its share is told to come back later), and `--download-only` fetches through the swarm without serving or announcing anything. A one-off `chungus fetch` is always download-only.

Two DHT attacks matter most here. In a **Sybil** attack someone runs thousands of fake nodes; in an **eclipse** attack those nodes surround a model's key, or a victim's routing table, so lookups only ever reach the attacker, who can then hide a model (data is still verified by hash, so it can't be forged). Three defenses are on by default:

- **Disjoint lookup paths.** Each DHT lookup follows several independent paths (S/Kademlia), so one poisoned path doesn't hide a model.
- **Subnet caps.** At most `--max-peers-per-subnet` (default 2) routing-table entries come from one IPv4 /24 or IPv6 /48, so a single host or small cloud block can't fill a routing table cheaply.
- **Anchor nodes.** The registry operator signs a list of anchor nodes (`chungus anchors <addr>... --key operator.key`). Nodes started with `--anchors-from <registry>` keep them in their routing table, exempt from the limits above, and ask them directly for every model alongside the DHT. `chungus fetch --swarm` asks them too, alongside the public node.

```sh
# The operator publishes the anchor list
chungus anchors /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW... --key .chungus/registry/operator.key

# Nodes and fetchers use it
chungus node --anchors-from https://chungus.example.com --operator chungus1<operator key> --max-upload 20
chungus fetch acme/tiny-llama@v1 --swarm -o model/
```

### Running a public node

Every swarm needs a few public nodes for others to join through and to relay for peers behind NAT. The repository has a container for one, with limits suited to a small VPS: 20 MB/s upload, 400 connections, and at most 32 relayed connections of 1 GB and 10 minutes each. On a server with a public IP and Docker:

```sh
git clone https://github.com/buwunny/chungus && cd chungus/deploy
cp node.env.example node.env   # set CHUNGUS_EXTERNAL to the server's IP, adjust the limits
docker compose up -d
docker compose logs node       # the addresses to share, ending in /p2p/<peer id>
```

Open TCP and UDP port 4001 in the provider's firewall. The node runs as a non-root user in a read-only container with no shell, restarts on failure, and keeps its store and identity key in the `chungus-data` volume, so its peer id survives upgrades (`docker compose pull && docker compose up -d`; `up -d --build` builds from the checkout instead of pulling). Every `chungus node` flag can be set in `node.env` as a `CHUNGUS_*` variable: set `CHUNGUS_BLOCKLIST` and `CHUNGUS_OPERATOR` to follow a registry's blocklist, and `CHUNGUS_BOOTSTRAP` to join a swarm other than the project's. A node never bootstraps through itself, so the project's own public node runs with the same defaults.

The relay only carries connections between two chungus peers that both asked for it, and only until they hole-punch a direct one. The node seeds nothing until you pin a model into its store:

```sh
docker compose exec node chungus fetch acme/tiny-llama@v1 --swarm --store /data/store
```

## Lazy loading

`chungus mount` makes a model's files appear at once, before any weights have downloaded. Reads fetch the chunks they need on demand, verifying each against its hash, while a prefetcher fills in the rest in the order a loader wants it: config and tokenizer files first, then every safetensors header, then tensors layer by layer across all shards (embeddings, `layers.0`, `layers.1`, ..., then the output head). Each read also moves the chunks right after it to the front of the queue. Loaders that `mmap` safetensors, as transformers and vLLM do, work unchanged, so inference can start before the download finishes.

```sh
mkdir llama
chungus mount acme/tiny-llama@v1 llama/          # or a root; --swarm or --bootstrap for the internet
python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('llama')"
```

Everything fetched lands in the store, so once prefetching finishes the model is complete and is shared like any other. `--prefetch 0` fetches only what is read. Mounting uses FUSE and works on Linux only so far (install `fuse3`). On macOS, `chungus fetch <model> -o <dir>` downloads the whole model instead; mounting there would mean linking against macFUSE, which every macOS user would then need installed.

Lazy reads need a manifest whose root commits to each file's chunk list (format v2, what `pack` writes now). Older v1 manifests only committed to whole-file hashes, which can't be checked until a file is complete, so they can still be fetched but not mounted; re-pack to upgrade.

## The registry: names, search and the blocklist

Peers move bytes; a registry gives models names. `acme/tiny-llama@v1` points at a manifest root, signed by the publisher's key. The first key to publish under an org owns it, and only its owners (see `chungus grant`) can publish there after that. Every change goes into an append-only, hash-chained log whose head the registry signs, so anyone can download the log and check that no name was rewritten.

```sh
# Run a registry (it creates an operator key in its data directory)
chungus registry --data .chungus/registry
export CHUNGUS_REGISTRY=https://chungus.example.com

# Publish a packed model under a name, then find and fetch it anywhere
chungus publish <root> --name acme/tiny-llama@v1 --description "A tiny Llama for tests"
chungus search tiny llama
chungus fetch acme/tiny-llama@v1 -o model/     # requires the publisher's signature

# Check the whole log against the registry's signed head
chungus audit --operator chungus1<operator key>
```

`publish` sends the manifest along with the signed statement. The registry checks that it hashes to the published root and lists no unsafe files, keeps it (`GET /v1/manifests/<root>` shows what a name contains), and only lists models whose manifest it checked in search and the index. Models published before this check need publishing again to be listed. It also sends the header of each safetensors file, which the registry checks against the file's leading chunks (packing gives a header chunks of its own, and the signed root covers their hashes), so it can count parameters without trusting the publisher. `GET /v1/summary/<root>` returns a model's size, files, chunk counts and parameters per dtype. Publishing a model again adds the count to one published before this.

The registry operator can block a model's root or a single chunk hash (`chungus block <hash> --key operator.key`), so re-packing a banned model with a small change is still caught by its chunks. Nodes that follow the blocklist (`--blocklist <registry url>` on `serve`, `hub` and `node`) delete blocked data, stop announcing it and refuse to serve or store it. Blocks are log entries too, so they are public and auditable.

### A public registry and website

`site/` is a static website for a registry: a landing page with install commands, live numbers and recently published models, and a model search (`search.html`) that downloads the list of published models (`GET /v1/index`) and searches it in the browser. Each model's page shows its parameters, size, format and files, its revisions, and the commands to fetch or mount it. The registry allows cross-origin reads, so the site can live anywhere, but it only talks to a registry over HTTPS.

To run a public registry on the same server as the public node, with the website on the same domain, HTTPS only (Caddy gets the certificate, redirects HTTP to HTTPS and sends HSTS):

```sh
cd chungus/deploy
cp .env.example .env           # set CHUNGUS_DOMAIN, with its DNS pointing at the server
docker compose --profile registry up -d
docker compose logs registry   # the operator key: nodes pin it with --operator
```

Open TCP ports 80 and 443 (and UDP 443 for HTTP/3). The registry's log and keys live in the `registry-data` volume.

### Keeping the operator key offline

The operator key signs the log head, the blocklist and the anchor list, and every node pins it, so whoever steals it can block models and point nodes at their own anchors. It doesn't need to live on the server. The registry has two keys: the **root key** (`operator.key`), which nodes pin, and an **online key** (`online.key`), which the root key delegates to for a limited time. With the root key offline, a stolen online key works only until its delegation expires or you revoke it, and it can never delegate to itself or to another key.

```sh
# 1. Start the registry once; it creates online.key and prints it
docker compose logs registry            # "... delegate to online key chungus1..."

# 2. Copy the root key to your own machine, and keep a backup of it somewhere offline
docker compose cp registry:/data/registry/operator.key ./chungus-root.key

# 3. From your machine: let the online key act as the operator for 90 days
chungus delegate chungus1<online key> --key ./chungus-root.key --registry https://chungus.example.com

# 4. Remove the root key from the server and restart (the container has no shell, so
#    borrow one that mounts its volume)
docker run --rm --volumes-from "$(docker compose ps -q registry)" alpine rm /data/registry/operator.key
docker compose restart registry
```

The registry keeps `operator.pub` so it still knows which key nodes pin. Renew before the 90 days are up by running step 3 again (the registry's log warns two weeks ahead); block, unblock and anchors commands work with either key. If the server may be compromised, `chungus delegate --revoke --key ./chungus-root.key` ends the delegation at once, and to rotate the online key, delete `online.key`, restart, and delegate to the new one. Operator commands run inside the container, e.g. `docker compose exec registry chungus block <hash> --key /data/registry/operator.key`.

The site can also be published to GitHub Pages: set the repository variable `CHUNGUS_REGISTRY_URL` to the registry's HTTPS address (the workflow refuses anything else), set Settings > Pages > Source to "GitHub Actions" and tick "Enforce HTTPS". The Pages workflow points the page at the live registry and bundles a snapshot of its index, refreshed every six hours, for when the registry is down.

### The node leaderboard

Nodes can opt in to a public leaderboard, ranked only on numbers someone other than the node confirms. A node's own counters never rank, since a modified node can report anything.

```sh
# List this node, under a display name, on the registry it follows
chungus node --blocklist https://chungus.example.com --leaderboard "bunny burrow"
```

- **Uptime and models held** come from probes. Every few minutes each anchor node (run with `--probe`) asks each listed node for a random chunk of a model it claims, from a separate download-only identity so the node can't tell probes from downloads, and reports the result signed with its node key. The registry counts reports only from the anchors in its signed list.
- **Bytes and downloads served** come from receipts. `fetch` and `mount` over the swarm bind their name lookup to their one-run peer id, then sign a receipt for what each peer sent and hand it to that peer, which submits it. The registry credits a receipt only if its signer made a lookup the registry counted (one per model, client network and UTC day), and credits at most the model's unique bytes per downloader. `--no-receipts` turns them off.

`GET /v1/leaderboard?metric=bytes|downloads|uptime|models&days=30` ranks the listed nodes, `GET /v1/nodes/<peer id>` is one node's daily history, and `GET /v1/downloads` has the swarm-wide totals. Once a day can get no more receipts, the registry signs its totals (`GET /v1/totals/<YYYY-MM-DD>`), so they can't be quietly revised. The site's `leaderboard.html` shows it all. Receipts carry only a one-run peer id, the registry stores client networks only as hashes under a key it forgets every day, and raw receipts are deleted after a week. The operator can hide an abusive name with `chungus hide-node <peer id> --key <operator key>`; its numbers stay.

Behind a reverse proxy the registry needs `--behind-proxy` to see each client's network (deploy/ sets it); never set it on a registry clients can reach directly.

## Safe files only

A chunk's hash proves a file arrived intact, not that it is safe to load, and pickle-based weights can run arbitrary code the moment PyTorch, joblib or NumPy opens them. So chungus carries weights only as **safetensors** or **GGUF**, plus the small files around them (configs, tokenizers, READMEs). Files ending in `.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.joblib`, `.dill`, `.pd`, `.npy`, `.npz`, `.h5`, `.hdf5` or `.keras` are kept out everywhere:

- `pack` skips them and says which it skipped (convert them to safetensors first).
- Stores refuse manifests that list them, so `fetch`, `unpack` and `mount` won't write them, and nodes neither announce nor serve older manifests that do.
- The registry won't publish them.
- `chungus hub` passes them through from huggingface.co without caching them or taking them from peers.

## Gated and licensed models

Some models, such as Llama, are gated on Hugging Face: you must accept a license before downloading. A swarm would otherwise let anyone re-share them without the gate, so gated models are served only to people whose own Hugging Face token passes it.

- A publisher marks a model as gated when publishing it: `chungus publish <root> --name acme/llama-3.2-1b --gated-by meta-llama/Llama-3.2-1B`. The registry operator can gate any model after the fact (`chungus gate <root> meta-llama/Llama-3.2-1B --key operator.key`, or with no repo to lift it). Only the operator can lift a gate.
- Nodes that follow a registry (`--blocklist <registry>`) learn which models are gated, and serve their chunks only to peers that present an **access ticket**: a statement signed by the registry's operator key that one peer id may download one repo's models, valid for a day.
- To get a ticket, `fetch` and `mount` send the user's Hugging Face token (from `HF_TOKEN` or `huggingface-cli login`) to the registry, which asks Hugging Face whether that token may download the repo and, if so, signs a ticket for the fetch's peer id. The token is only used for that check: it is never stored, logged, or sent to peers. Peers only ever see the ticket, and a ticket is useless to any other peer id.
- Without a token, or with one that hasn't accepted the license, the fetch stops and says which license to accept.

This keeps the network itself from spreading gated models. It can't stop someone who already downloaded a model from re-packing it and sharing it without the gate; that is what the blocklist is for.

## What to expect

On synthetic BF16 weights (normal distribution, 64M parameters), `bench` reports zstd alone at 78% of the original size and the full pipeline at 73%. Real models usually compress somewhat better than synthetic ones. Published results (ZipNN, DFloat11) put BF16 near 67–70% of original size. Models already quantized to 4 bits barely compress. Dedup savings depend on how much two models actually share: re-uploads and format copies dedup almost completely, and full fine-tunes dedup very little.

## Manifest and store layout

A store holds `chunks/<hh>/<hash>` blobs, `manifests/<root>.json`, their signatures in `manifests/<root>.sigs.json`, and `meta/hub/...` records of cached Hub repos. A manifest lists every file, its size and BLAKE3 hash, and the ordered chunks that rebuild it. Its `root` hash commits to all of that, including each chunk's hash and length, and is the value a publisher signs.

Every format is versioned: manifests name theirs in `format`, a store records its layout in `VERSION`, each chunk blob starts with a version byte, and the swarm protocols carry a version in their ids. A newer chungus keeps reading older data, and an older one refuses newer data with a message to upgrade rather than misreading it. [docs/formats.md](docs/formats.md) lists every version and the rules for changing one; `chungus --version` shows what a binary speaks.

## Roadmap

| Milestone | Scope |
|---|---|
| **M1** (done) | Storage format, `pack` / `unpack` / `bench` |
| **M2** (done) | Share models between machines on a LAN (mDNS discovery, verified transfer, origin fallback) |
| **M3** (done) | Local cache that speaks the Hugging Face Hub API, so existing tools work via `HF_ENDPOINT` |
| **M4** (done) | Signed models, internet swarm over libp2p with 64 MB block announcements, registry with a transparency log, search and blocklist |
| Node limits (done) | Upload caps, connection and request limits, download-only mode, disjoint DHT lookups, subnet caps, registry-signed anchor nodes |
| Lazy loading (done) | `chungus mount`: read models before they finish downloading, with layer-order prefetch |
| Later | OCI images, dedicated nodes, voting, optional GPU-side decode straight into VRAM (in addition to the CPU SIMD path) |

The earlier OCI registry proxy design is kept in [docs/archive/oci-proxy-design.md](docs/archive/oci-proxy-design.md) for the Docker image work.
