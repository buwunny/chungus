# chungus

A peer-to-peer network for distributing AI models, their runtimes and Docker AI images. Think of it as a decentralized Hugging Face with its own take on Xet-style storage.

This repository currently holds **milestones 1 to 3**: the storage format, a benchmark tool, sharing models between machines on a LAN, and a drop-in Hugging Face cache.

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

## What to expect

On synthetic BF16 weights (normal distribution, 64M parameters), `bench` reports zstd alone at 78% of the original size and the full pipeline at 73%. Real models usually compress somewhat better than synthetic ones. Published results (ZipNN, DFloat11) put BF16 near 67–70% of original size. Models already quantized to 4 bits barely compress. Dedup savings depend on how much two models actually share: re-uploads and format copies dedup almost completely, and full fine-tunes dedup very little.

## Manifest and store layout

A store holds `chunks/<hh>/<hash>` blobs, `manifests/<root>.json`, and `meta/hub/...` records of cached Hub repos. A manifest lists every file, its size and BLAKE3 hash, and the ordered chunks that rebuild it. Its `root` hash commits to all of that and is the value an author will sign in a later milestone.

## Roadmap

| Milestone | Scope |
|---|---|
| **M1** (done) | Storage format, `pack` / `unpack` / `bench` |
| **M2** (done) | Share models between machines on a LAN (mDNS discovery, verified transfer, origin fallback) |
| **M3** (done) | Local cache that speaks the Hugging Face Hub API, so existing tools work via `HF_ENDPOINT` |
| M4 | Internet swarm, signed publishing, registry and search |
| Later | OCI images, lazy layer loading, dedicated nodes, voting, GPU-side decode |

The earlier OCI registry proxy design is kept in [docs/archive/oci-proxy-design.md](docs/archive/oci-proxy-design.md) for the Docker image work.
