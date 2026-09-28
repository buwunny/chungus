# Fuzzing

Everything chungus reads from other machines gets fuzzed: manifests, swarm requests and
responses, chunk blobs (decoded before their hash can be checked), safetensors headers,
and registry logs.

```sh
cargo install cargo-fuzz          # once; fuzzing needs a nightly toolchain
cargo +nightly fuzz list
cargo +nightly fuzz run manifest -- -max_total_time=300
```

Targets: `manifest`, `peer_message`, `blob`, `safetensors`, `registry_log`. CI runs each
for a minute on every pull request. A crash is saved under `fuzz/artifacts/<target>/`;
`cargo +nightly fuzz run <target> <file>` replays it. Turn each fix into a unit test next
to the code it fixes, as `store::tests::decode_refuses_oversized_lengths` was.
