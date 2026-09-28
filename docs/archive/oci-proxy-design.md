# chungus (chungus.io)

A byte-level data-deduplication front-proxy and P2P mesh designed to drastically reduce WAN egress and deployment times for standard OCI container images and OCI-packaged AI models.

## Target Environments

In **Mirror Mode**, chungus trades node CPU (sequential gzip recompression, ~30–100 MB/s per layer) for WAN bytes, so recompression alone would only win on time where **WAN bandwidth is lower than recompression throughput**. Two mechanisms extend the win to fast links:

* The **Adaptive Path Planner** picks the faster path per layer, so Mirror Mode is never meaningfully slower than a native pull, even on fast links.
* **Snapshotter Mode** skips recompression completely, removing the CPU ceiling so pulls are limited by the network or disk.

Primary environments:

* Edge sites, retail/factory/telco nodes, and remote clusters on constrained links (≤ 500 Mbps): wins on time and egress in either mode.
* Cross-region and cross-cloud pulls where egress fees dominate: wins on cost in either mode.
* Fast datacenter links (≥ 1 Gbps): wins on time in Snapshotter Mode. In Mirror Mode the planner falls back to native-speed pulls unless the policy is set to optimize for egress.
* Congested shared links where many nodes pull the same updates (P2P phase).

## Dual-Path Specification

### Push Path (Ingest)

1. CI Runner / Developer Machine pushes standard compressed layers (gzipped or zstd tarballs) to the local client proxy via the Image Retagging Pattern.
2. The client proxy intercepts the stream, decompresses the layers to achieve raw byte visibility, tracks tar headers and filesystem alignments, and executes Content-Defined Chunking (CDC).
3. The client proxy filters hashes against its local index, Zstd-compresses unique chunks, and transmits them over the WAN via rust-libp2p.
4. The remote server proxy saves the raw chunks and registers the Metadata Recipe (the ordered manifest of chunk hashes, exact original tarball serialization metadata, compression parameters, encoder fingerprinting, and expected output digests).

### Pull Path (Deploy)

1. The Target Node Runtime (containerd / Docker) requests an image from the Node Proxy.
2. The Node Proxy forwards the request and the runtime's upstream credentials to chungus-server, which validates authorization against the upstream registry (see *Upstream Authorization Passthrough*) and returns the small Metadata Recipe.
3. The Node Proxy diffs the recipe against its local chunk index, then queries the local P2P swarm over the LAN for missing chunk hashes.
4. Chunks are swarmed out-of-order from authenticated local peers, falling back to the remote server proxy for missing blocks.
5. The Node Proxy assembles chunks in order and feeds them into a single sequential encoder using the recipe's saved encoder fingerprint, **streaming** the recompressed output to the runtime as it is produced. Chunk fetching, recompression, and the runtime's own download/unpack pipeline overlap rather than running back-to-back.

### Cold Pulls (No Recipe Yet)

When a requested blob has never been seen by chungus-server, blocking the first pull on download → decompress → chunk → verify would make it slower than native. Instead:

1. chungus-server streams the **original compressed blob** straight through to the Node Proxy (and on to the runtime) with no added latency.
2. In parallel, the server chunks, fingerprints, and verifies the blob asynchronously and registers its Metadata Recipe once verification passes.
3. The Node Proxy independently runs the same deterministic CDC over the blob it just served and populates its local chunk index. No recipe round-trip is needed for the node to benefit from v1.0's chunks when v1.1 arrives.

## Node Delivery Modes

The Node Proxy can hand layers to the runtime in two ways. Both consume the same Metadata Recipes and chunk store; they differ only in the final step.

### Mirror Mode (Zero-Install Default)

containerd talks to the Node Proxy as a registry mirror via `hosts.toml`. Because containerd verifies the **compressed** blob digest from the manifest, the Node Proxy must recompress chunks byte-for-byte (see Pull Path step 5). This works with no runtime changes, but its speed is bounded by single-threaded recompression per layer.

### Snapshotter Mode (Fast Path)

chungusd also exposes a containerd **remote snapshotter** (the same plugin mechanism used by stargz, SOCI, and Nydus), deployed as a DaemonSet.

* The OCI image config records `rootfs.diff_ids`, the SHA-256 of each layer's **uncompressed** tar. The snapshotter rebuilds the uncompressed tar from chunks, verifies it against the `diff_id`, and unpacks it directly into the snapshot.
* **No gzip recompression happens at all.** Pull speed becomes bounded by the network (for missing chunks) and local disk, not by a single CPU core.
* **Digest reproduction leaves the pull path.** The gzip-matching risk (*Deep Technical Risks, §1*) then only affects Mirror Mode clients and upstream pushes.
* Works on 100% standard, unconverted OCI images, preserving the advantage over Nydus and eStargz.

Trade-offs:

* Pulls are no longer "just edit `hosts.toml`": a snapshotter plugin must be installed and containerd configured to use it.
* Compressed blobs are not present in the node's content store, so `ctr export` and re-pushing from the node are unavailable for those images (the same limitation other remote snapshotters have).

### Future: Lazy Pulling

With the snapshotter in place, chungus can start containers before the full image arrives and fetch chunks on first file access, as stargz and SOCI do. Most images read only a small fraction of their files at startup, so this can beat native pulls even on cold, fast-link pulls.

## Adaptive Path Planner

In Mirror Mode, the planner decides **per layer** whether to use deduplicated reconstruction or to fetch the original compressed blob directly (from the chungus-server cache or upstream):

```
native_time  = compressed_size / wan_bandwidth
chungus_time = max(missing_bytes / wan_bandwidth,
                   uncompressed_size / recompress_rate)
```

* If `native_time < chungus_time`, the Node Proxy streams the original compressed blob through unchanged. Otherwise it reconstructs from chunks.
* `wan_bandwidth` and `recompress_rate` are **measured**, not configured: rolling throughput per upstream/server, and observed encoder throughput under the current core budget.
* `missing_bytes` comes from diffing the Metadata Recipe against the local chunk index, before any data is fetched.
* Layers that fell back to unchunked storage at ingest always take the native path.
* In Snapshotter Mode the recompression term disappears, so the planner chooses deduplicated reconstruction whenever `missing_bytes < compressed_size`.

A policy knob resolves conflicts between speed and egress:

```toml
[planner]
optimize_for = "latency"   # "latency": pick the faster path per layer
                           # "egress":  always dedup, accept slower pulls to save bytes
```

The result: Mirror Mode is never meaningfully slower than native, and still saves bytes wherever doing so is also faster.

## Node Resource Budget

Recompression CPU usage is configurable so chungus never starves co-located workloads:

```toml
[recompression]
max_workers = 2            # layers recompressed concurrently (1 core each)
nice = 10                  # yield CPU to workloads on the node
cgroup_cpu_quota = "150%"  # optional hard cap via cgroup v2 cpu.max
```

* On Kubernetes, the simplest control is `resources.limits.cpu` on the chungusd DaemonSet. chungusd reads its own cgroup quota at startup and derives `max_workers` from it, so the two settings can never disagree.
* Extra cores only help when several layers are recompressed at once. **A single large layer is still bound to one core** regardless of budget. The budget caps chungus's impact on the node; it does not speed up one big layer.
* The planner uses the budget-adjusted `recompress_rate`, so a tighter budget automatically shifts more layers onto the native path instead of slowing pulls down.
* Snapshotter Mode has low CPU cost (fast zstd chunk decompression plus SHA-256 verification, both multi-GB/s), so the budget mainly matters in Mirror Mode.

## The Integration & Friction Reality Check

* **Pulls (Mirror Mode):** 100% transparent. In containerd, this utilizes native `hosts.toml` mirrors, allowing seamless routing without altering Kubernetes manifests or host configurations.
* **Pulls (Snapshotter Mode):** Requires installing the chungus snapshotter DaemonSet and pointing containerd at it. Kubernetes manifests are still unchanged.
* **Pushes:** Not transparent. Docker/OCI push mirroring cannot be automated natively without intrusive TLS interception.
* **The Compromise:** CI/CD pipelines must use the Image Retagging Pattern. Build scripts require a 1-line change to tag and push images to a local address (e.g., `localhost:5000/my-app:v1`) instead of the upstream cloud registry.

## Topology & Pipeline Synchronicity

chungus operates as an OCI Registry Front-Proxy, acting as an optimization gateway in front of an existing registry of record (AWS ECR, Harbor, Docker Hub, GHCR). To maximize network efficiency during full-blob uploads, chungus-server must be deployed in the same cloud region as the upstream registry.

### Upstream Push Validation & Synchronous Manifest Commits

An OCI push consists of multiple independent blob uploads followed by a final manifest PUT. To avoid breaking CI pipelines, deployment tools, and downstream vulnerability scanners:

* **Holding the Manifest PUT:** chungus-server streams individual unique chunks to its store in real time, but it holds the acknowledgment of the final manifest PUT to the CI client until it has fully reassembled the image and successfully written it upstream to the registry of record.
* **Failure Handling:** If the upstream registry rejects the manifest PUT or times out, the push fails hard back to the CI pipeline. The underlying unique chunks remain safely indexed in the chungus-server cache, ensuring that a subsequent retry finishes instantly without re-uploading data across the WAN.

## Unified Networking & Performance

* **The Stack:** Standardized on rust-libp2p for all networking layers, entirely dropping standalone gRPC-over-QUIC to avoid ecosystem immaturity.
* **Capabilities:** Utilizes rust-libp2p native QUIC for high-performance multiplexing, request/response protocols for chunk requests, and its built-in TCP fallback mechanism for environments blocking UDP traffic.

## P2P Access & Security Model

### Upstream Authorization Passthrough

To prevent the pull-through cache from bypassing private registry authorization, **chungus-server** (not the Node Proxy) is the single enforcement point and never serves a cached manifest blindly.

* **Bearer Challenge Relay:** Registries such as Docker Hub and GHCR use the token-auth challenge flow. When containerd hits the Node Proxy without a token, chungus relays the upstream registry's `WWW-Authenticate` challenge (realm, service, scope) back to containerd unchanged. containerd obtains a scoped bearer token directly from the upstream auth server, and chungus forwards that token on subsequent requests. chungus never mints or stores upstream credentials itself.
* **Per-Request Validation:** Before returning a manifest, recipe, or P2P token, chungus-server issues a cheap manifest `HEAD` to the upstream registry using the caller's forwarded token.
* The server only returns data and issues P2P tokens if the upstream registry returns HTTP 200.

### Peer-Bound JWT Scheme

To prevent token replay attacks within a shared mesh network, P2P access tokens are explicitly cryptographically bound to the requester:

1. During a pull request, chungus-server issues a JWT containing the parent Image Manifest/Digest, an expiration timestamp, and the requester's unique libp2p PeerId.
2. When requesting chunks from a local peer, the pulling node presents this token over an authenticated rust-libp2p connection.
3. The serving peer verifies the signature using the server's public key, checks that the connection's authenticated PeerId matches the one encoded in the JWT, confirms the token hasn't expired, and cross-references its local copy of the Metadata Recipe to ensure the requested chunk hash belongs to that manifest.
4. **Token Renewal:** For large ML model pulls that exceed the token's lifetime, the pulling node proxy executes a background refresh handshake with chungus-server (re-running upstream validation) to obtain an extended lease.
5. **Key Rotation:** The central server rotates its asymmetric signing keys periodically, distributing updated public keys to nodes over active libp2p control channels.

## Deep Technical Risks & Mitigations

### 1. Digest Reconstruction Nondeterminism (The Primary Risk)

OCI requires exact byte-matching for image digests (`sha256:abcd...`). Because different gzip implementations (Go's `compress/gzip` or `klauspost/compress` across varying versions, zlib, pigz, zlib-ng) yield different byte streams at identical compression levels, reconstructing a layer byte-for-byte is highly complex.

```
                      ┌──► Recompression Matches Original Digest? ──► YES: Store Chunks & Recipe
[ Chunking Stream ] ──┤
                      └──► Recompression Fails or Fails to Match ───► NO: Fall back to unchunked blob
```

* **Verification at Ingest (Safety Fallback):** Wherever chunking occurs, the system immediately runs a recompression test. If the resulting output stream fails to match the original layer digest exactly, the engine aborts deduplication for that layer and falls back to storing the compressed blob completely unchunked.
* **Encoder Fingerprinting:** The Metadata Recipe explicitly records the detected compressor ecosystem variant. The open-core layer incorporates a faithful Rust port of Go's `compress/flate` family to maximize match rates, as the majority of upstream layers are built via Go tooling (BuildKit, crane, ko).
* **Format Awareness:** Core natively recognizes and parses zstd, estargz, and zstd:chunked formatted layers.
* **Scope Reduction via Snapshotter Mode:** Snapshotter Mode verifies layers against the uncompressed `diff_id` and never needs the compressed bytes, so a failed reproduction does not block deduplication for snapshotter nodes. For those nodes, chunks and recipes are kept even when the gzip match fails. Only Mirror Mode clients and upstream pushes fall back to the unchunked blob.

### 2. Sequential Recompression vs. Deployment Velocity

To guarantee a byte-for-byte digest match with Go-gzipped layers, a single layer must be processed sequentially by one encoder instance; pigz-style parallelism changes the output. A single 5 GB ML layer is therefore limited to roughly 30–100 MB/s of recompression.

Worked example: a 5 GB layer with a small change.

| Path | 1 Gbps WAN | 100 Mbps WAN |
|---|---|---|
| Native containerd pull | ~40 s | ~400 s |
| Mirror Mode, always dedup (small WAN delta + recompression) | ~50–170 s | ~50–170 s |
| Mirror Mode, Adaptive Path Planner | ~40 s (native path chosen) | ~50–170 s (dedup path chosen) |
| Snapshotter Mode (small WAN delta, no recompression) | Seconds (network/disk bound) | Seconds (network/disk bound) |

Unassisted recompression loses on fast links. Mitigations, in order of impact:

* **Snapshotter Mode:** Eliminates recompression entirely (see *Node Delivery Modes*). This is the real fix and the reason it is prioritized as Phase 2.
* **Adaptive Path Planner:** Ensures Mirror Mode picks the native path whenever recompression would be slower (see *Adaptive Path Planner*).
* **Streaming overlap:** Recompression output is streamed to the runtime as it is produced, overlapping with chunk fetching and with containerd's own decompress/unpack, so costs are not additive.
* **Cross-layer parallelism:** chungusd recompresses multiple discrete layers concurrently.
* **Once per node per layer version:** containerd retains compressed blobs in its content store, so repeat pulls of the same version skip recompression. This does **not** help the first pull of a new version (the primary optimization target), and it holds only if `discard_unpacked_layers` is disabled and kubelet image GC has not evicted the blob. Both must be verified on target clusters.
* **Research item: encoder checkpoints.** During ingest, record encoder state resume points in the Metadata Recipe so that segments of a single layer can be recompressed in parallel. The existing verify-at-ingest step confirms that checkpointed reconstruction still reproduces the digest before a recipe is accepted. This is the only known path to intra-layer parallelism in Mirror Mode and is unproven. Its priority drops once Snapshotter Mode ships.

### 3. State Management & Garbage Collection Racing

Ingest and deletion processes introduce severe race condition risks where data can be swept prematurely.

* **Server Epoch-Based GC:** To prevent a race where ingestion identifies a chunk as pre-existing but a background GC cycle purges it before the corresponding Metadata Recipe commits, the server utilizes a generational grace period protocol. Newly uploaded chunks and chunks referenced by active, in-flight push sessions are marked with an active epoch identifier that exempts them from mark-and-sweep passes.
* **Node LRU Protection:** On local edge nodes, the LRU cache eviction engine is explicitly locked against active assembly pipelines. Chunks that belong to an image currently being pulled, assembled, or actively served to a neighboring peer in the swarm are pinned and cannot be swept.

### 4. Tag Caching and Revalidation

* **Manifest Digests:** Cached strictly by cryptographic hash; since digests are immutable, they are never revalidated.
* **Mutable Tags:** To handle mutable targets (e.g., `:latest`), the proxy issues a cheap HTTP `HEAD` request upstream for every incoming tag request to fetch the authoritative digest mapping. This avoids stale image errors, and Docker Hub does not count `HEAD` requests against pull rate limits.

## Market Comparison & Positioning

| Feature | chungus | Spegel | Dragonfly | Nydus | eStargz / zstd:chunked |
|---|---|---|---|---|---|
| Granularity | Byte/chunk level (CDC) | Layer level | Layer, split into pieces for P2P | File/chunk level | File/chunk level |
| Cross-version dedup | Yes (arbitrary byte CDC) | No | No | Yes (within Nydus format) | Partial (file-level) |
| Consumer runtime change | None in Mirror Mode; optional snapshotter for speed | None | None (P2P daemon only) | Required (Nydus snapshotter) | None for standard pulls; snapshotter for lazy-pull benefits |
| Producer build change | None | None | None | Required (conversion, e.g. nydusify) | Required (custom build/compression step) |
| Optimized targets | Container layers & OCI models | Container layers & OCI artifacts | Container layers & AI models | Container layers | Container layers |

* **The Practical Advantage:** Unlike Spegel and Dragonfly, if a 10 GB layer updates slightly, chungus transfers only the impacted regions plus adjacent boundary chunks rather than re-transferring the entire layer. Unlike Nydus or eStargz, it delivers sub-layer efficiency on 100% unaltered standard OCI images without modifying upstream developer build steps, while unifying container layers and OCI-packaged AI weights under a single optimization mesh.

## Distributed Binaries & Tiering

* **chungusd (The Node Daemon):** A unified binary that runs on both client machines and server instances, configured via runtime flags (`--mode proxy` or `--mode server`). It houses the chunking engine, Adaptive Path Planner, local cache managers, and libp2p swarming code. In proxy mode it serves the registry mirror endpoint and, when enabled (`--snapshotter`), the containerd remote snapshotter socket.
* **chungus-cli:** An administrative utility used exclusively for checking cluster health, viewing deduplication metrics, triggering concurrent garbage collection, and inspecting storage backend indices.

### Open Core (chungus-core)

* OCI protocol compliance and OCI-packaged AI weight parsing.
* Local P2P swarming mechanics via rust-libp2p.
* Fragmented chunk assembly and in-memory virtual mapping layers.
* Go-compatible gzip/flate recompression engine with unchunked safety fallback.
* Mirror Mode and Snapshotter Mode, the Adaptive Path Planner, and node resource budgets.
* Epoch-gated concurrent background garbage collection.
* Basic authentication and pass-through credentials.

### Enterprise Offering (chungus-plus)

* Centralized hosted management dashboard.
* Legacy/non-standard transport protocols (e.g., raw S3 bucket syncing, SFTP hooks).
* Multi-region cross-datacenter server replication and centralized peer trackers.
* High-availability server clustering and load-balancing logic.
* Zero Trust RBAC, SSO, audit logging, and compliance reporting.

## Phase 0: Offline Feasibility Spikes

Both make-or-break risks can be tested in roughly a week with no networking code. Proxy work does not begin until both pass.

1. **Digest Reproduction Spike:** Download every layer of the benchmark images, decompress, recompress with Go's own `compress/gzip` (a small Go program; the Rust port comes later), and compare digests. Report the reproduction rate by bytes, broken down by image source. **Gate:** ≥ 85% of bytes reproduce, weighted toward third-party-built images.
2. **Dedup Ratio Spike:** Run FastCDC (Rust `fastcdc` crate) over the decompressed tars of successive tags and compute the bytes a node would be missing versus the layers native containerd would re-download. **Gate:** ≥ 70% reduction versus the warm native baseline.

If either gate fails, the design is revisited before any proxy code is written.

## Phase 1 MVP Roadmap

The MVP scope is locked to a read-only, pull-through cache in **Mirror Mode** for containerd, including the Adaptive Path Planner and node resource budget. Snapshotter Mode, P2P swarming, and push-path modifications are deferred. In this phase, chungus-server handles chunking, indexing, and recompression verification asynchronously after streaming cold blobs through (see *Cold Pulls*), and the Node Proxy builds its own chunk index locally.

```
[ containerd ] ──► [ Node Proxy ] ──► (WAN) ──► [ chungus-server (Ingest/Verify/Chunk) ] ──► [ GHCR / Docker Hub ]
```

### Static Evaluation Benchmarks

The verification suite utilizes fixed, successive releases pinned strictly by digest to ensure reproducibility.

1. **Go-Built Tooling:** `golang:1.22-alpine` successive patch releases (third-party built).
2. **Large Machine Learning Stack:** Five consecutive patch releases of `pytorch/pytorch`, pinned to a specific CUDA/cuDNN variant (third-party built).
3. **Python / Pip Application:** Successive application tags built on `python:3.11-slim` with varying pip dependency updates (self-built).

**Bias note:** Self-built images use our own BuildKit version and will trivially match a Go-flate port. The digest-reproduction metric is therefore reported separately for third-party-built (1, 2) and self-built (3) images, and the success gate applies to the third-party set.

### Network Conditions

Every benchmark runs over a throttled WAN link (e.g., `tc netem`) at three profiles: **100 Mbps**, **500 Mbps**, and **1 Gbps**, and at two node CPU budgets (**1 core** and **2 cores**). Each combination is run with `optimize_for = "latency"` and `optimize_for = "egress"`, to locate the crossover point where recompression stops winning on time and to confirm the planner correctly switches paths there.

### Dual-Scenario Cache Configuration

* **Scenario A (Unbounded Cache):** Storage is sized to hold all benchmark components to measure best-case deduplication.
* **Scenario B (Constrained Cache):** Storage is limited to 1.5× the size of the largest single target image to actively exercise LRU eviction and pinning under churn.

### Success Metrics Matrix

| Metric | Evaluation Context | Success Threshold |
|---|---|---|
| WAN Data Reduction | Versus a warm native pull (v1.1 pulled directly after v1.0 on a clean native node), `optimize_for = "egress"` | ≥ 70% bytes saved over the WAN link |
| Pull Wall-Clock Time | End-to-end container readiness, `optimize_for = "latency"`, all link speeds | Within 5% of native containerd pulls, or faster |
| Pull Wall-Clock Time | Links ≤ 500 Mbps, `optimize_for = "latency"` | Faster than native containerd pulls |
| Planner Accuracy | Per-layer path choice versus the measured-faster path | ≥ 90% of layers take the faster path |
| Node CPU Overhead | Cores consumed by chungusd during a pull | Never exceeds the configured budget |
| Digest-Reproduction Rate | By total bytes, third-party-built images | ≥ 85% of bytes deduplicated without fallback |
| Cold-Pull Latency | First-ever pull of an image through chungus | ≤ 5% slower than native (pass-through path) |

## Post-MVP Roadmap

| Phase | Scope | Why this order |
|---|---|---|
| 0 | Offline feasibility spikes (digest reproduction, dedup ratio) | Validates both make-or-break risks before writing proxy code |
| 1 | Mirror Mode pull-through cache + Adaptive Path Planner + resource budget | Zero-install adoption; proves WAN savings and "never slower than native" |
| 2 | Snapshotter Mode | Removes the recompression CPU ceiling and takes digest reproduction off the pull path; makes chungus faster on fast links |
| 3 | P2P swarming and the push path | Multiplies savings when many nodes pull the same update; optimizes CI uploads |
| 4 | Lazy pulling | Starts containers before full download; beats native even on cold, fast-link pulls |

### Phase 2 Success Metrics (Snapshotter Mode)

| Metric | Evaluation Context | Success Threshold |
|---|---|---|
| Pull Wall-Clock Time | 1 Gbps link, successive tags | Faster than native containerd pulls |
| Integrity | Every unpacked layer | 100% verified against `rootfs.diff_ids` |
| Node CPU Overhead | Cores consumed during a pull | ≤ 0.5 cores sustained |
