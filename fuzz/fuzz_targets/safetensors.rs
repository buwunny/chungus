//! Safetensors headers are parsed when packing and when mounting.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(Some(segs)) = chungus::safetensors::segments(data) {
        for s in &segs {
            assert!(s.start <= s.end && s.end <= data.len() as u64);
        }
        let _ = chungus::chunk::chunk(data, &segs);
    }
});
