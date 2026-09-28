//! Reading a model before it has finished downloading, directly and through a mount.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use chungus::lazy::{ChunkSource, Lazy};
use chungus::manifest::Manifest;
use chungus::net;
use chungus::store::Store;

/// A safetensors file whose tensors are stored in name order ("layers.10" before
/// "layers.2"), as real files are, with an embedding, 12 layers and a head.
fn write_model(dir: &Path) -> Vec<String> {
    fs::create_dir_all(dir).unwrap();
    let mut names = vec![
        "lm_head.weight".to_string(),
        "model.embed_tokens.weight".into(),
    ];
    names.extend((0..12).map(|i| format!("model.layers.{i}.mlp.weight")));
    names.sort();
    let mut state = 99u64;
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for name in &names {
        let start = data.len();
        // 64k BF16 values each, with realistic exponents so chunks compress.
        for _ in 0..65_536 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1;
            data.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": [65_536], "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    fs::write(dir.join("model.safetensors"), file).unwrap();
    fs::write(dir.join("config.json"), br#"{"num_hidden_layers": 12}"#).unwrap();
    names
}

/// A seed serving a packed model over HTTP, and an empty store to read it into.
async fn setup(tmp: &Path) -> (Manifest, Arc<Store>, Arc<dyn ChunkSource>) {
    let model = tmp.join("model");
    write_model(&model);
    let seed = Arc::new(Store::open(&tmp.join("seed")).unwrap());
    let (m, _) = chungus::pack(&model, &seed).unwrap();
    seed.put_manifest(&m).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = net::url_for(listener.local_addr().unwrap());
    tokio::spawn(net::serve_on(listener, seed));
    let local = Arc::new(Store::open(&tmp.join("local")).unwrap());
    let client = net::client().unwrap();
    let peers = vec![url];
    let m = net::prepare(&client, &m.root, &local, &peers, &[], &[])
        .await
        .unwrap();
    let source = Arc::new(net::HttpSource {
        client,
        peers,
        origin: vec![],
    });
    (m, local, source)
}

#[tokio::test(flavor = "multi_thread")]
async fn reads_fetch_only_what_they_need_then_prefetch_fills_in() {
    let tmp = tempfile::tempdir().unwrap();
    let (m, local, source) = setup(tmp.path()).await;
    let original = fs::read(tmp.path().join("model/model.safetensors")).unwrap();
    let file = m
        .files
        .iter()
        .position(|f| f.path == "model.safetensors")
        .unwrap();
    let lazy = Lazy::new(local.clone(), m, source).unwrap();

    // Reads at scattered offsets, including ones spanning chunk boundaries and the end.
    for (offset, len) in [
        (0, 8),
        (1_000_000, 300_000),
        (1_500_000, 70_000),
        (original.len() as u64 - 10, 100),
    ] {
        let got = lazy.read(file, offset, len).await.unwrap();
        let end = (offset as usize + len).min(original.len());
        assert_eq!(got, original[offset as usize..end], "read at {offset}");
    }
    let local_bytes = lazy.stats.local_bytes.load(Ordering::Relaxed);
    assert!(lazy.stats.on_demand.load(Ordering::Relaxed) > 0);
    assert!(
        local_bytes < lazy.total_bytes() / 2,
        "fetched {local_bytes} bytes for a few reads"
    );

    lazy.prefetch(8).await.unwrap();
    assert_eq!(
        lazy.stats.local_bytes.load(Ordering::Relaxed),
        lazy.total_bytes()
    );
    assert!(lazy.stats.prefetched.load(Ordering::Relaxed) > 0);
    // Everything is in the store now, so the model unpacks with full verification.
    let out = tmp.path().join("out");
    chungus::unpack(lazy.manifest(), &local, &out).unwrap();
    assert_eq!(fs::read(out.join("model.safetensors")).unwrap(), original);
}

#[tokio::test(flavor = "multi_thread")]
async fn prefetch_goes_in_layer_order() {
    let tmp = tempfile::tempdir().unwrap();
    let (m, local, source) = setup(tmp.path()).await;
    let original = fs::read(tmp.path().join("model/model.safetensors")).unwrap();
    let file = m
        .files
        .iter()
        .position(|f| f.path == "model.safetensors")
        .unwrap();
    let chunks = m.files[file].chunks.clone();
    let lazy = Lazy::new(local, m, source).unwrap();
    let order = lazy.prefetch_order().await.unwrap();

    // Where each tensor starts, from the file's own header.
    let header_len = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&original[8..8 + header_len]).unwrap();
    let chunk_at = |offset: u64| {
        let mut at = 0u64;
        for c in &chunks {
            if offset < at + c.len as u64 {
                return c.hash.clone();
            }
            at += c.len as u64;
        }
        unreachable!()
    };
    let position = |name: &str| {
        let start = header[name]["data_offsets"][0].as_u64().unwrap() + 8 + header_len as u64;
        let hash = chunk_at(start);
        order.iter().position(|c| c.hash == hash).unwrap()
    };
    // config.json first, the embedding before any layer, layers numerically, head last.
    assert_eq!(order[0].len as usize, br#"{"num_hidden_layers": 12}"#.len());
    let mut last = position("model.embed_tokens.weight");
    for i in 0..12 {
        let p = position(&format!("model.layers.{i}.mlp.weight"));
        assert!(p > last, "layer {i} at {p}, before {last}");
        last = p;
    }
    assert!(position("lm_head.weight") > last);
    // Every unique chunk exactly once.
    let unique: std::collections::HashSet<_> = order.iter().map(|c| &c.hash).collect();
    assert_eq!(unique.len(), order.len());
}

#[tokio::test(flavor = "multi_thread")]
async fn chunk_lists_are_committed_to_by_the_root() {
    let tmp = tempfile::tempdir().unwrap();
    let (m, local, source) = setup(tmp.path()).await;
    // Swapping two chunks keeps every file hash but changes the chunk list: v2 catches it.
    let json = serde_json::to_vec(&m).unwrap();
    let mut forged: Manifest = serde_json::from_slice(&json).unwrap();
    let chunks = &mut forged
        .files
        .iter_mut()
        .max_by_key(|f| f.chunks.len())
        .unwrap()
        .chunks;
    chunks.swap(1, 2);
    assert!(!forged.verify_root());
    assert!(Lazy::new(local.clone(), forged, source.clone()).is_err());

    // A v1 manifest can't prove its chunk lists, so it can't be read lazily.
    let mut v1: Manifest = serde_json::from_slice(&json).unwrap();
    v1.format = chungus::manifest::FORMAT_V1.into();
    let mut h = blake3::Hasher::new();
    for f in &v1.files {
        h.update(f.path.as_bytes());
        h.update(&[0]);
        h.update(&f.size.to_le_bytes());
        h.update(f.hash.as_bytes());
    }
    v1.root = h.finalize().to_hex().to_string();
    assert!(v1.verify_root());
    let Err(err) = Lazy::new(local, v1, source) else {
        panic!("read a v1 manifest lazily");
    };
    assert!(err.to_string().contains("older format"), "{err}");
}

#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread")]
async fn a_mounted_model_reads_like_the_original() {
    let tmp = tempfile::tempdir().unwrap();
    let (m, local, source) = setup(tmp.path()).await;
    let lazy = Lazy::new(local, m, source).unwrap();
    let dir = tmp.path().join("mnt");
    fs::create_dir(&dir).unwrap();
    let mounted = chungus::mount::mount(lazy.clone(), &dir, tokio::runtime::Handle::current())
        .expect("FUSE mount (needs /dev/fuse and fusermount3)");

    let dir2 = dir.clone();
    let model = tmp.path().join("model");
    tokio::task::spawn_blocking(move || {
        let mut names: Vec<_> = fs::read_dir(&dir2)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["config.json", "model.safetensors"]);
        for name in names {
            assert_eq!(
                fs::read(dir2.join(&name)).unwrap(),
                fs::read(model.join(&name)).unwrap()
            );
        }
        assert!(fs::read(dir2.join("missing")).is_err());
    })
    .await
    .unwrap();
    assert!(lazy.stats.on_demand.load(Ordering::Relaxed) > 0);
    mounted.unmount().unwrap();
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
}
