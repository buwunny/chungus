use std::fs;
use std::path::Path;

use chungus::store::Store;

/// Deterministic xorshift so the test needs no extra dependencies.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Roughly normal: sum of uniforms (Irwin-Hall), scaled like a weight matrix.
    fn weight(&mut self) -> f32 {
        let s: f64 = (0..4)
            .map(|_| (self.next() >> 11) as f64 / (1u64 << 53) as f64)
            .sum();
        ((s - 2.0) * 0.03) as f32
    }
}

fn bf16(x: f32) -> [u8; 2] {
    ((x.to_bits() >> 16) as u16).to_le_bytes()
}

/// Write a safetensors file with BF16 tensors of the given element counts.
fn write_model(path: &Path, tensors: &[(&str, usize)], seed: u64) {
    let mut rng = Rng(seed);
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, n) in tensors {
        let start = data.len();
        for _ in 0..*n {
            data.extend_from_slice(&bf16(rng.weight()));
        }
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "BF16", "shape": [n], "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    fs::write(path, out).unwrap();
}

#[test]
fn pack_unpack_is_bit_identical_and_dedups() {
    let dir = std::env::temp_dir().join(format!("chungus-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let (base, tuned, out) = (dir.join("base"), dir.join("tuned"), dir.join("out"));
    fs::create_dir_all(&base).unwrap();
    fs::create_dir_all(&tuned).unwrap();

    let layers = [("a.weight", 700_000), ("b.weight", 300_001), ("c.bias", 17)];
    write_model(&base.join("model.safetensors"), &layers, 7);
    fs::write(base.join("config.json"), br#"{"hidden_size": 64}"#).unwrap();
    // "Fine-tune": same first tensor, different later tensors.
    fs::copy(
        base.join("model.safetensors"),
        tuned.join("model.safetensors"),
    )
    .unwrap();
    let mut bytes = fs::read(tuned.join("model.safetensors")).unwrap();
    let tail = bytes.len() - 300_000;
    for b in &mut bytes[tail..] {
        *b = b.wrapping_add(1);
    }
    fs::write(tuned.join("model.safetensors"), bytes).unwrap();

    let store = Store::open(&dir.join("store")).unwrap();
    let (m1, s1) = chungus::pack(&base, &store).unwrap();
    let (m2, s2) = chungus::pack(&tuned, &store).unwrap();

    // Float transform should beat raw size on BF16 weights.
    assert!(s1.new_stored_bytes < s1.new_raw_bytes * 85 / 100, "{s1:?}");
    // The second pack should reuse the unchanged tensor's chunks.
    assert!(s2.new_raw_bytes < s2.raw_bytes / 2, "{s2:?}");

    for (m, src) in [(&m1, &base), (&m2, &tuned)] {
        let _ = fs::remove_dir_all(&out);
        chungus::unpack(m, &store, &out).unwrap();
        for f in &m.files {
            assert_eq!(
                fs::read(out.join(&f.path)).unwrap(),
                fs::read(src.join(&f.path)).unwrap()
            );
        }
    }

    // A corrupted blob must be rejected, not silently written out.
    let victim = &m1
        .files
        .iter()
        .find(|f| f.path == "model.safetensors")
        .unwrap()
        .chunks[0];
    let blob_path = dir
        .join("store")
        .join("chunks")
        .join(&victim.hash[..2])
        .join(&victim.hash);
    fs::write(&blob_path, [1u8, 0, 1, 0xde, 0xad]).unwrap();
    let _ = fs::remove_dir_all(&out);
    assert!(chungus::unpack(&m1, &store, &out).is_err());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bench_counts_each_input_against_the_ones_before() {
    let dir = std::env::temp_dir().join(format!("chungus-bench-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let (base, tuned) = (dir.join("base"), dir.join("tuned"));
    fs::create_dir_all(&base).unwrap();
    fs::create_dir_all(&tuned).unwrap();
    let layers = [("a.weight", 700_000), ("b.weight", 300_000)];
    write_model(&base.join("model.safetensors"), &layers, 7);
    // Same first tensor, retrained second one.
    let mut bytes = fs::read(base.join("model.safetensors")).unwrap();
    let tail = bytes.len() - 600_000;
    for b in &mut bytes[tail..] {
        *b = b.wrapping_add(1);
    }
    fs::write(tuned.join("model.safetensors"), bytes).unwrap();

    let r = chungus::bench(&[base.clone(), tuned.clone()]).unwrap();
    assert_eq!(r.inputs.len(), 2);
    let (a, b) = (&r.inputs[0], &r.inputs[1]);
    assert_eq!(a.new_raw_bytes, a.raw_bytes);
    // Only the changed tensor (plus the chunk the change starts in) is new.
    assert!(
        b.new_raw_bytes > 500_000 && b.new_raw_bytes < 800_000,
        "{b:?}"
    );
    assert_eq!(r.unique_raw_bytes, a.new_raw_bytes + b.new_raw_bytes);
    assert_eq!(r.dedup_bytes, a.new_stored_bytes + b.new_stored_bytes);
    let bf16 = &r.by_dtype["bf16"];
    assert_eq!(bf16.raw_bytes, 2 * 2_000_000);
    assert!(bf16.encoded_bytes < bf16.zstd_bytes, "{bf16:?}");

    // The same model twice is entirely deduplicated.
    let r = chungus::bench(&[base.clone(), base.clone()]).unwrap();
    assert_eq!(r.inputs[1].new_raw_bytes, 0);

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pack_skips_pickles_and_stores_refuse_them() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), b"{}").unwrap();
    std::fs::write(model.join("pytorch_model.bin"), b"\x80\x04pickle").unwrap();
    let store = chungus::store::Store::open(&tmp.path().join("store")).unwrap();
    let (m, stats) = chungus::pack(&model, &store).unwrap();
    let paths: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, ["config.json"]);
    assert_eq!(stats.skipped.len(), 1);

    // A directory of nothing but pickles has nothing to pack.
    std::fs::remove_file(model.join("config.json")).unwrap();
    assert!(chungus::pack(&model, &store).is_err());

    // A hand-made manifest listing a pickle is refused by the store and by unpack.
    let bad = chungus::manifest::Manifest::new(vec![chungus::manifest::FileEntry {
        path: "weights.pt".into(),
        ..m.files[0].clone()
    }]);
    assert!(store.put_manifest(&bad).is_err());
    assert!(chungus::unpack(&bad, &store, &tmp.path().join("out")).is_err());
}
