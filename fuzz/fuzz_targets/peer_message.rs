//! Requests and responses from swarm peers, decoded as libp2p's CBOR codec does.
#![no_main]

use chungus::p2p::{Request, Response};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(req) = cbor4ii::serde::from_slice::<Request>(data) {
        let bytes = cbor4ii::serde::to_vec(Vec::new(), &req).unwrap();
        cbor4ii::serde::from_slice::<Request>(&bytes).unwrap();
    }
    if let Ok(resp) = cbor4ii::serde::from_slice::<Response>(data) {
        let bytes = cbor4ii::serde::to_vec(Vec::new(), &resp).unwrap();
        cbor4ii::serde::from_slice::<Response>(&bytes).unwrap();
    }
});
