//! Registry log pages, replayed by every node that follows a registry and every auditor.
#![no_main]

use chungus::registry::{Entry, Log};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(entries) = serde_json::from_slice::<Vec<Entry>>(data) {
        let operator = "chungus1".to_string() + &"00".repeat(32);
        if let Ok(log) = Log::replay(operator, entries) {
            let _ = log.index();
            let _ = log.search("a b", 10);
        }
    }
});
