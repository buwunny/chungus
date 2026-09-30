//! GGUF headers are parsed when packing (including by `chungus hub`, on files fetched
//! from Hugging Face) and when publishing.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(Some(segs)) = chungus::gguf::segments(data) {
        let mut at = 0;
        for s in &segs {
            assert_eq!(s.start, at);
            assert!(s.start <= s.end && s.end <= data.len() as u64);
            at = s.end;
        }
        assert_eq!(at, data.len() as u64);
        let _ = chungus::chunk::chunk(data, &segs);
    }
    if let Ok(Some(n)) = chungus::gguf::header_len(data) {
        let _ = chungus::gguf::params(&data[..n]);
    }
});
