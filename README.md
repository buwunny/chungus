# chungus

A peer-to-peer network for distributing AI models, their runtimes and Docker AI images. Think of it as a decentralized Hugging Face with its own take on Xet-style storage.

This repository currently holds **milestones 1 to 4**: the storage format, a benchmark tool, sharing models between machines on a LAN, a drop-in Hugging Face cache, signed models, sharing over the internet, and a registry for names, search and blocklists.

## The pipeline

```
model files ─► segments ─► FastCDC chunks ─► BLAKE3 ─► float transform ─► zstd ─► chunk store
              (per tensor)   (~64 KiB)       (raw bytes)  (exponent split)
```

1. **Segments.** Safetensors files are split at tensor boundaries using the file's header, so a chunk never spans two tensors. Other files are one segment.
2. **Content-defined chunking.** FastCDC cuts each segment into chunks of 16–256 KiB (64 KiB average). Cut points inside float tensors are rounded to whole elements.
3. **Hashing.** Each chunk is addressed by the BLAKE3 hash of its *raw* bytes. Identical chunks are stored once, across files and across models.
4. **Float transform.** For BF16 and F32 tensors, each element is rearranged into an exponent byte and sign+mantissa bytes, then grouped into planes. Exponents are low-entropy and compress well. This is lossless: unpacking gives back the exact bits.
5. **zstd.** Each chunk is compressed on its own so any chunk can be read independently. The encoder keeps whichever of {stored, zstd, transform + zstd} is smallest.

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
# A public server (a VPS, say) that others join through, and relays for nodes behind NAT
chungus node --store relay --public --relay-server
#   prints: address /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW...

# At home, behind NAT: share your store, reachable through the relay
chungus node --bootstrap /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW... \
             --relay     /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW...

# Anywhere: fetch a model from whoever has it
chungus fetch <root> --bootstrap /ip4/203.0.113.7/tcp/4001/p2p/12D3KooW... -o model/
```

Models are announced at two levels. A node with the whole model announces its root; every node also announces each 64 MB block of a model it holds in full. A fetcher asks the DHT for both, so peers that are still downloading a model already serve the parts they have, and the swarm grows during a flash crowd instead of waiting for complete copies. Announcing blocks rather than individual chunks keeps the DHT small: a 140 GB model is about 2,200 blocks but 2 million chunks.

A node behind NAT is reached through its relay, and the two ends then try to hole-punch a direct connection (DCUtR). A node's identity lives in `node.key` in its store. `node` announces models added to the store while it runs within a minute. There are no public bootstrap nodes yet, so someone has to run the first one.

## The registry: names, search and the blocklist

Peers move bytes; a registry gives models names. `acme/tiny-llama@v1` points at a manifest root, signed by the publisher's key. The first key to publish under an org owns it, and only its owners (see `chungus grant`) can publish there after that. Every change goes into an append-only, hash-chained log whose head the registry signs, so anyone can download the log and check that no name was rewritten.

```sh
# Run a registry (it creates an operator key in its data directory)
chungus registry --data .chungus/registry
export CHUNGUS_REGISTRY=http://registry.example:7450

# Publish a packed model under a name, then find and fetch it anywhere
chungus publish <root> --name acme/tiny-llama@v1 --description "A tiny Llama for tests"
chungus search tiny llama
chungus fetch acme/tiny-llama@v1 -o model/     # requires the publisher's signature

# Check the whole log against the registry's signed head
chungus audit --operator chungus1<operator key>
```

The registry operator can block a model's root or a single chunk hash (`chungus block <hash> --key operator.key`), so re-packing a banned model with a small change is still caught by its chunks. Nodes that follow the blocklist (`--blocklist <registry url>` on `serve`, `hub` and `node`) delete blocked data, stop announcing it and refuse to serve or store it. Blocks are log entries too, so they are public and auditable.

## What to expect

On synthetic BF16 weights (normal distribution, 64M parameters), `bench` reports zstd alone at 78% of the original size and the full pipeline at 73%. Real models usually compress somewhat better than synthetic ones. Published results (ZipNN, DFloat11) put BF16 near 67–70% of original size. Models already quantized to 4 bits barely compress. Dedup savings depend on how much two models actually share: re-uploads and format copies dedup almost completely, and full fine-tunes dedup very little.

## Manifest and store layout

A store holds `chunks/<hh>/<hash>` blobs, `manifests/<root>.json`, their signatures in `manifests/<root>.sigs.json`, and `meta/hub/...` records of cached Hub repos. A manifest lists every file, its size and BLAKE3 hash, and the ordered chunks that rebuild it. Its `root` hash commits to all of that and is the value a publisher signs.

## Roadmap

| Milestone | Scope |
|---|---|
| **M1** (done) | Storage format, `pack` / `unpack` / `bench` |
| **M2** (done) | Share models between machines on a LAN (mDNS discovery, verified transfer, origin fallback) |
| **M3** (done) | Local cache that speaks the Hugging Face Hub API, so existing tools work via `HF_ENDPOINT` |
| **M4** (done) | Signed models, internet swarm over libp2p with 64 MB block announcements, registry with a transparency log, search and blocklist |
| Later | OCI images, lazy layer loading, dedicated nodes, voting, GPU-side decode |

The earlier OCI registry proxy design is kept in [docs/archive/oci-proxy-design.md](docs/archive/oci-proxy-design.md) for the Docker image work.
