//! Write a synthetic BF16 safetensors file for trying `chungus bench` without
//! downloading a model: `cargo run --release --example synth -- out.safetensors 64`
//! (size in millions of parameters).

use std::fs;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: synth <out.safetensors> [millions of params]");
    let millions: usize = args.next().map_or(16, |s| s.parse().expect("number"));
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    let per_tensor = 1_000_000;
    let mut header = serde_json::Map::new();
    let mut data = Vec::with_capacity(millions * per_tensor * 2);
    for t in 0..millions {
        let start = data.len();
        for _ in 0..per_tensor {
            // Box-Muller normal, std 0.02 like a typical initialised weight matrix.
            let (u1, u2) = (next().max(1e-300), next());
            let x = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos() * 0.02;
            data.extend_from_slice(&(((x as f32).to_bits() >> 16) as u16).to_le_bytes());
        }
        header.insert(
            format!("layers.{t}.weight"),
            serde_json::json!({"dtype": "BF16", "shape": [per_tensor], "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    fs::write(&path, out).unwrap();
    println!("wrote {path}: {millions}M BF16 parameters");
}
