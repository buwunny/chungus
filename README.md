# chungus

A peer-to-peer network for distributing AI models, their runtimes and Docker AI images. Think of it as a decentralized Hugging Face with its own take on Xet-style storage.

This repository currently holds **milestone 1**: the storage format and a benchmark tool. Networking comes next.

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

No model handy? Generate a synthetic BF16 file:

```sh
cargo run --release --example synth -- synthetic.safetensors 64   # 64M parameters
```

## What to expect

On synthetic BF16 weights (normal distribution, 64M parameters), `bench` reports zstd alone at 78% of the original size and the full pipeline at 73%. Real models usually compress somewhat better than synthetic ones. Published results (ZipNN, DFloat11) put BF16 near 67–70% of original size. Models already quantized to 4 bits barely compress. Dedup savings depend on how much two models actually share: re-uploads and format copies dedup almost completely, and full fine-tunes dedup very little.

## Manifest

A manifest lists every file, its size and BLAKE3 hash, and the ordered chunks that rebuild it. Its `root` hash commits to all of that and is the value an author will sign in a later milestone.

## Roadmap

| Milestone | Scope |
|---|---|
| **M1** (this) | Storage format, `pack` / `unpack` / `bench` |
| M2 | Share models between machines on a LAN (mDNS discovery, verified transfer, origin fallback) |
| M3 | Local cache that speaks the Hugging Face Hub API, so existing tools work via `HF_ENDPOINT` |
| M4 | Internet swarm, signed publishing, registry and search |
| Later | OCI images, lazy layer loading, dedicated nodes, voting, GPU-side decode |

The earlier OCI registry proxy design is kept in [docs/archive/oci-proxy-design.md](docs/archive/oci-proxy-design.md) for the Docker image work.
