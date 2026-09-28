# chungus

A byte-level deduplicating front-proxy and P2P mesh for OCI registries. chungus cuts WAN egress and pull times for standard container images and OCI-packaged AI models. It needs no image conversion and no changes to how images are built.

When a 10 GB layer changes slightly, chungus transfers only the changed chunks instead of the whole layer.

## How it works

- **Content-defined chunking:** Layers are decompressed and split with FastCDC. Only chunks the node doesn't already have cross the WAN.
- **Metadata Recipes:** For each layer, the server stores the ordered chunk hashes, the tar and compression metadata, and an encoder fingerprint. Nodes use the recipe to rebuild the layer exactly.
- **Front-proxy topology:** chungus sits in front of your existing registry of record (ECR, Harbor, Docker Hub, GHCR). The upstream registry still controls authorization: chungus relays its bearer-token challenges and checks every request against it.
- **Cold pulls:** The first time an image is pulled, the original blob streams straight through with no added latency. Chunking and verification happen in the background.

## Node delivery modes

| Mode | Setup | How layers are delivered |
|---|---|---|
| **Mirror** (default) | containerd `hosts.toml` mirror only | Recompresses chunks byte-for-byte to match the compressed digest. Speed is limited by single-core gzip. |
| **Snapshotter** | containerd remote snapshotter DaemonSet | Rebuilds the uncompressed tar and verifies it against `diff_id`. No recompression, so speed is limited only by network and disk. |

In Mirror Mode, an **Adaptive Path Planner** chooses per layer between chunk reconstruction and a native blob pull, using measured bandwidth and recompression rate. Mirror Mode is therefore never meaningfully slower than a native pull. Set `optimize_for = "egress"` to always deduplicate.

## Where it helps

- Edge and remote sites on constrained links (≤ 500 Mbps)
- Cross-region and cross-cloud pulls where egress fees dominate
- Fast datacenter links, when using Snapshotter Mode
- Many nodes pulling the same update over a shared link (P2P)

## Components

- **`chungusd`:** A single daemon that runs as `--mode proxy` on nodes or `--mode server` next to the upstream registry. It contains the chunking engine, the planner, the cache, and libp2p networking. Pass `--snapshotter` to enable the snapshotter.
- **`chungus-cli`:** Admin tool for health checks, dedup metrics, GC, and index inspection.

All networking runs on rust-libp2p: QUIC, with TCP fallback. P2P chunk requests use short-lived JWTs bound to the requester's PeerId.

## Key risk

OCI digests require byte-exact reproduction of compressed layers, and different gzip implementations produce different bytes. chungus includes a Rust port of Go's `compress/flate` and verifies every layer at ingest. If reproduction fails, that layer is stored unchunked. Snapshotter Mode verifies against uncompressed digests, so this risk does not affect it.

## Roadmap

| Phase | Scope |
|---|---|
| 0 | Offline spikes: digest reproduction (≥ 85% of bytes) and dedup ratio (≥ 70% savings) |
| 1 | MVP: read-only Mirror Mode pull-through cache, Adaptive Path Planner, CPU budget |
| 2 | Snapshotter Mode |
| 3 | P2P swarming and push path (CI uses a 1-line image retag to a local proxy) |
| 4 | Lazy pulling |

See [idea.md](idea.md) for the full design.
