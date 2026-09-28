//! Two (and three) stores syncing over HTTP on localhost.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use chungus::net;
use chungus::store::Store;

/// A small safetensors-shaped file with enough data for dozens of chunks.
fn write_model(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let mut state = 12345u64;
    let data: Vec<u8> = (0..3_000_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    let header = format!(
        r#"{{"w":{{"dtype":"BF16","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        data.len() / 2,
        data.len()
    );
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    fs::write(dir.join("model.safetensors"), file).unwrap();
    fs::write(dir.join("config.json"), b"{}").unwrap();
}

async fn spawn_server(store: Arc<Store>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = net::url_for(listener.local_addr().unwrap());
    tokio::spawn(net::serve_on(listener, store));
    url
}

fn pack_into(model: &Path, store_dir: &Path) -> (Arc<Store>, String) {
    let store = Store::open(store_dir).unwrap();
    let (m, _) = chungus::pack(model, &store).unwrap();
    store.put_manifest(&m).unwrap();
    (Arc::new(store), m.root)
}

#[tokio::test(flavor = "multi_thread")]
async fn fetch_from_peer_is_verified_and_resumes() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let (seed, root) = pack_into(&model, &tmp.path().join("a"));
    let peer = spawn_server(seed).await;

    let local = Arc::new(Store::open(&tmp.path().join("b")).unwrap());
    let (m, stats) = net::fetch(&root, local.clone(), std::slice::from_ref(&peer), None, &[])
        .await
        .unwrap();
    assert_eq!(stats.already_local, 0);
    assert_eq!(stats.bytes_by_source.len(), 1);

    let out = tmp.path().join("out");
    chungus::unpack(&m, &local, &out).unwrap();
    for f in ["model.safetensors", "config.json"] {
        assert_eq!(
            fs::read(out.join(f)).unwrap(),
            fs::read(model.join(f)).unwrap()
        );
    }

    // A second fetch finds everything already local and transfers nothing.
    let (_, again) = net::fetch(&root, local, &[peer], None, &[]).await.unwrap();
    assert_eq!(again.already_local, again.chunks);
    assert!(again.bytes_by_source.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_peer_is_rejected_and_origin_fills_in() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let (honest, root) = pack_into(&model, &tmp.path().join("honest"));

    // The liar has the manifest, but every chunk it serves is corrupted.
    let (liar, _) = pack_into(&model, &tmp.path().join("liar"));
    for entry in walk(&tmp.path().join("liar").join("chunks")) {
        let mut blob = fs::read(&entry).unwrap();
        let last = blob.len() - 1;
        blob[last] ^= 0xff;
        fs::write(&entry, blob).unwrap();
    }
    // An empty peer that has nothing at all.
    let empty = Arc::new(Store::open(&tmp.path().join("empty")).unwrap());

    let liar = spawn_server(liar).await;
    let empty = spawn_server(empty).await;
    let origin = spawn_server(honest).await;

    let local = Arc::new(Store::open(&tmp.path().join("local")).unwrap());
    let (m, stats) = net::fetch(&root, local.clone(), &[liar, empty], Some(&origin), &[])
        .await
        .unwrap();
    assert_eq!(stats.rejected, stats.chunks);
    assert_eq!(stats.bytes_by_source.keys().collect::<Vec<_>>(), [&origin]);
    chungus::unpack(&m, &local, &tmp.path().join("out")).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn fetch_fails_cleanly_when_nobody_has_the_model() {
    let tmp = tempfile::tempdir().unwrap();
    let empty = spawn_server(Arc::new(Store::open(&tmp.path().join("e")).unwrap())).await;
    let local = Arc::new(Store::open(&tmp.path().join("l")).unwrap());
    let root = "0".repeat(64);
    assert!(net::fetch(&root, local, &[empty], None, &[]).await.is_err());
}

fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for e in fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn trusted_fetch_needs_a_signature_from_a_trusted_key() {
    use chungus::sign;
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let (seed, root) = pack_into(&model, &tmp.path().join("seed"));
    let author = sign::generate_key(&tmp.path().join("author.key")).unwrap();
    let stranger = sign::generate_key(&tmp.path().join("stranger.key")).unwrap();
    seed.add_signatures(&root, &[sign::sign(&author, &root)])
        .unwrap();
    // A forged signature slipped into the peer's file is dropped by the fetcher.
    let forged = sign::Signature {
        key: sign::public_key_string(&stranger.verifying_key()),
        sig: sign::sign(&author, &root).sig,
    };
    let sigs_file = tmp
        .path()
        .join("seed/manifests")
        .join(format!("{root}.sigs.json"));
    let mut sigs = seed.signatures(&root).unwrap();
    sigs.push(forged);
    fs::write(&sigs_file, serde_json::to_vec(&sigs).unwrap()).unwrap();
    let peer = spawn_server(seed).await;

    // Trusting only the stranger: refused before any chunk is downloaded.
    let wary = Arc::new(Store::open(&tmp.path().join("wary")).unwrap());
    let Err(err) = net::fetch(
        &root,
        wary.clone(),
        std::slice::from_ref(&peer),
        None,
        &[stranger.verifying_key()],
    )
    .await
    else {
        panic!("fetch without a trusted signature succeeded");
    };
    assert!(err.to_string().contains("no trusted key"), "{err}");
    assert!(
        !tmp.path()
            .join("wary/chunks")
            .read_dir()
            .unwrap()
            .any(|e| { e.unwrap().path().read_dir().unwrap().next().is_some() })
    );

    // Trusting the author: downloads, and keeps only the valid signature.
    let local = Arc::new(Store::open(&tmp.path().join("local")).unwrap());
    let (m, _) = net::fetch(
        &root,
        local.clone(),
        &[peer],
        None,
        &[author.verifying_key()],
    )
    .await
    .unwrap();
    chungus::unpack(&m, &local, &tmp.path().join("out")).unwrap();
    let kept = local.signatures(&root).unwrap();
    assert_eq!(kept.len(), 1);
    assert_eq!(
        kept[0].key,
        sign::public_key_string(&author.verifying_key())
    );
}
