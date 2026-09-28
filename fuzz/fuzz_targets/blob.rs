//! Chunk blobs arrive from peers before their hash is checked, so decoding must survive
//! anything. The first four bytes are the raw length the manifest claims.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    if let Ok(raw) = chungus::store::decode(&data[4..], len) {
        assert_eq!(raw.len(), len);
    }
});
