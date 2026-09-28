//! Throughput of the byte-plane and exponent transforms, in GB/s of raw data.
//!
//! `cargo run --release --example transform_speed`

use chungus::transform::{self, FloatKind};
use std::hint::black_box;
use std::time::Instant;

const CHUNK: usize = 64 << 10;
const TOTAL: usize = 256 << 20;

fn rate(name: &str, data: &[u8], f: impl Fn(&[u8]) -> Vec<u8>) {
    // Warm up, then time passes over the data one chunk at a time, as the store does.
    for c in data.chunks(CHUNK).take(64) {
        black_box(f(c));
    }
    let start = Instant::now();
    for c in data.chunks(CHUNK) {
        black_box(f(black_box(c)));
    }
    let secs = start.elapsed().as_secs_f64();
    println!("{name:<22} {:>6.2} GB/s", data.len() as f64 / secs / 1e9);
}

fn main() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let data: Vec<u8> = (0..TOTAL)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    let bf16 = |d: &[u8]| transform::split_exponent(d, FloatKind::Bf16);
    let f32 = |d: &[u8]| transform::split_exponent(d, FloatKind::F32);
    let (bf16_planes, f32_planes, p2, p4): (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) = (
        data.chunks(CHUNK).flat_map(bf16).collect(),
        data.chunks(CHUNK).flat_map(f32).collect(),
        data.chunks(CHUNK)
            .flat_map(|c| transform::split(c, 2))
            .collect(),
        data.chunks(CHUNK)
            .flat_map(|c| transform::split(c, 4))
            .collect(),
    );
    rate("copy (ceiling)", &data, |c| c.to_vec());
    // For scale: decoding a chunk also means decompressing it, at the store's zstd level.
    let packed: Vec<Vec<u8>> = bf16_planes
        .chunks(CHUNK)
        .map(|c| zstd::bulk::compress(c, 3).unwrap())
        .collect();
    let start = Instant::now();
    for p in &packed {
        black_box(zstd::bulk::decompress(black_box(p), CHUNK).unwrap());
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{:<22} {:>6.2} GB/s",
        "zstd decompress",
        TOTAL as f64 / secs / 1e9
    );
    rate("split width 2", &data, |c| transform::split(c, 2));
    rate("join width 2", &p2, |c| transform::join(c, 2));
    rate("split width 4", &data, |c| transform::split(c, 4));
    rate("join width 4", &p4, |c| transform::join(c, 4));
    rate("split exponent bf16", &data, bf16);
    rate("join exponent bf16", &bf16_planes, |c| {
        transform::join_exponent(c, FloatKind::Bf16)
    });
    rate("split exponent f32", &data, f32);
    rate("join exponent f32", &f32_planes, |c| {
        transform::join_exponent(c, FloatKind::F32)
    });
}
