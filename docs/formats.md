# Formats and versions

Everything chungus writes to disk or sends to another machine carries a version, so a
later release can change a format without stranding anyone's data or splitting the
network. `chungus --version` lists the versions a binary speaks.

| What | Current | Where the version lives | Also reads |
|---|---|---|---|
| Manifest | `chungus/manifest/v2` | `format` field of every manifest | `chungus/manifest/v1` |
| Store layout | 1 | `<store>/VERSION` | stores made before the file existed (treated as 1) |
| Chunk blob | 1 | first byte of every blob in `chunks/` | |
| Block id | `chungus/block/v1` | domain prefix of the block hash | |
| Swarm requests | `/chungus/1` | libp2p protocol id | |
| DHT | `/chungus/kad/1` | libp2p protocol id | |
| Peer HTTP API (`serve`, `hub`) | `/v1/...` | URL prefix | |
| Registry API | `/v1/...` | URL prefix | |
| Registry statements and head | `chungus/registry-statement/v1`, `chungus/registry-head/v1` | signature domain prefix | |
| Signatures | `chungus1<hex>` keys, ed25519 over the manifest root | key prefix | |

## Weight files

Safetensors and GGUF files are split per tensor before chunking, so chunk edges fall on
tensor edges and a tensor two files share chunks the same way in both. A GGUF file's
header (key/values, including the tokenizer, and tensor info) is one segment, each tensor
is another, and padding between tensors makes small raw segments. F32, F16, BF16 and F64
tensors get the same float transforms they get in safetensors; quantized types are
stored as raw bytes. A GGUF file chungus can't parse (a version other than 2 or 3, or
anything malformed) is chunked whole as one raw segment, so it still packs and serves
unchanged. Segmenting changes where chunks are cut, so a GGUF file packed by an older
chungus has a different manifest root, but no format changes: the dtypes are the ones
manifests already carry.

## Rules for changing a format

- **Never change what an existing version means.** A change gets a new version; the old
  one keeps its exact bytes and hashes. Manifest v2 is the example: v1 roots still verify
  and fetch, and only new packs get v2 roots.
- **Readers keep reading old versions** for as long as data in them may exist. Drop one
  only in a release that says so, with a way to upgrade (for manifests, re-packing).
- **Readers refuse newer versions with a clear message.** A store, manifest or blob from a
  newer chungus is rejected with "upgrade chungus" rather than misread.
- **Stores migrate in place.** Bumping the store version means adding a migration from
  the previous one in `store::check_version`, which runs when a store is opened.
- **Wire changes add a protocol, they don't edit one.** A new request type goes in
  `/chungus/2`, and nodes offer both `/chungus/1` and `/chungus/2` until old nodes are
  gone; libp2p picks the newest protocol both sides speak. Nodes advertise
  `chungus/<release>` as their identify agent, and a node logs peers that run chungus
  but share no protocol with it.
- **Block ids never change within a swarm.** Every node must agree on `BLOCK_BYTES` and
  the block id domain, or peers can't find each other's blocks. Changing either is a new
  DHT protocol id.

Until v1.0, releases are alpha: the rules above still hold, but an old format may be
dropped sooner, always with a release note and an upgrade path.
