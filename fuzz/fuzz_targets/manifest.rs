//! Manifests arrive from peers, registries and files on disk.
#![no_main]

use chungus::manifest::Manifest;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(m) = serde_json::from_slice::<Manifest>(data) {
        let _ = m.verify_root();
        let _ = m.commits_to_chunks();
        let _ = m.blocks(1 << 16);
    }
});
