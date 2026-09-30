//! A registry over HTTP: publishing, resolving, searching and auditing the log.

use std::collections::BTreeMap;
use std::sync::Arc;

use chungus::registry::{self, Claim, Client, Registry, Statement};
use chungus::sign;

async fn spawn(reg: Arc<Registry>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, registry::router(reg)).await });
    url
}

fn publish(k: &ed25519_dalek::SigningKey, name: &str, root: &str, desc: &str) -> Statement {
    Statement::new(
        k,
        Claim::Publish {
            name: name.into(),
            rev: "main".into(),
            root: root.into(),
            description: desc.into(),
            gated: None,
        },
    )
}

/// A manifest listing one empty file called `path`: its root and JSON.
fn manifest(path: &str) -> (String, Vec<u8>) {
    let m = chungus::manifest::Manifest::new(vec![chungus::manifest::FileEntry {
        path: path.into(),
        size: 0,
        hash: blake3::hash(b"").to_hex().to_string(),
        chunks: vec![],
    }]);
    (m.root.clone(), serde_json::to_vec(&m).unwrap())
}

#[tokio::test(flavor = "multi_thread")]
async fn publish_resolve_search_audit() {
    let tmp = tempfile::tempdir().unwrap();
    let reg = Arc::new(Registry::open(&tmp.path().join("reg")).unwrap());
    let operator = reg.operator();
    let url = spawn(reg).await;
    let client = Client::new(&url).unwrap();
    let alice = sign::generate_key(&tmp.path().join("alice")).unwrap();
    let mallory = sign::generate_key(&tmp.path().join("mallory")).unwrap();
    let (root, bytes) = manifest("model.safetensors");

    client
        .publish(
            &publish(&alice, "acme/tiny-llama", &root, "A tiny Llama for tests"),
            &bytes,
        )
        .await
        .unwrap();
    // Someone else can't take the name.
    let (other, other_bytes) = manifest("other.safetensors");
    let err = client
        .publish(
            &publish(&mallory, "acme/tiny-llama", &other, ""),
            &other_bytes,
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not an owner"), "{err}");
    // A statement altered in transit fails its signature check.
    let mut forged = publish(&alice, "acme/other", &root, "");
    forged.time += 1;
    assert!(client.publish(&forged, &bytes).await.is_err());
    // The manifest must be the one the statement names.
    let st = publish(&alice, "acme/other", &root, "");
    assert!(client.publish(&st, &other_bytes).await.is_err());
    // Publishing takes the manifest, not a bare statement.
    assert!(client.submit(&st).await.is_err());
    // Pickles are refused, even from an org's owner.
    let (pickle, pickle_bytes) = manifest("pytorch_model.bin");
    let err = client
        .publish(&publish(&alice, "acme/pickled", &pickle, ""), &pickle_bytes)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("can run code"), "{err}");
    // Anyone can see what a published model contains.
    let listed = reqwest::get(format!("{url}/v1/manifests/{root}"))
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(listed.as_ref(), bytes.as_slice());

    let entry = client.resolve("acme/tiny-llama", "main").await.unwrap();
    assert_eq!(
        entry.statement.signature.key,
        sign::public_key_string(&alice.verifying_key())
    );
    assert!(client.resolve("acme/tiny-llama", "v2").await.is_err());

    let hits = client.search("llama").await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].root, root);
    assert!(client.search("mistral").await.unwrap().is_empty());
    let index = client.index().await.unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(index[0].name, "acme/tiny-llama");
    // Web pages on other origins, like the search site, may read the registry.
    let resp = reqwest::get(format!("{url}/v1/index")).await.unwrap();
    assert_eq!(resp.headers()["access-control-allow-origin"], "*");

    let (log, head) = client.audit(Some(&operator)).await.unwrap();
    assert_eq!((log.entries.len(), head.size), (1, 1));
    // Pinning the wrong operator key fails.
    let wrong = sign::public_key_string(&mallory.verifying_key());
    assert!(client.audit(Some(&wrong)).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn nodes_enforce_the_blocklist() {
    use chungus::store::Store;
    let tmp = tempfile::tempdir().unwrap();
    let reg = Arc::new(Registry::open(&tmp.path().join("reg")).unwrap());
    let operator = reg.operator();
    let op_key = reg.operator_key().unwrap().clone();
    let url = spawn(reg).await;
    let client = Client::new(&url).unwrap();

    // A node with a model in its store, following the registry's blocklist.
    let model = tmp.path().join("model.safetensors");
    std::fs::write(&model, vec![42u8; 300_000]).unwrap();
    let seed = Arc::new(Store::open(&tmp.path().join("seed")).unwrap());
    let (m, _) = chungus::pack(&model, &seed).unwrap();
    seed.put_manifest(&m).unwrap();
    let mut follower = registry::Follower::new(&url, Some(operator)).unwrap();
    assert_eq!(follower.sync(&seed).await.unwrap(), 0);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let peer = chungus::net::url_for(listener.local_addr().unwrap());
    tokio::spawn(chungus::net::serve_on(listener, seed.clone()));

    // Only the operator's statement is accepted.
    let block = Claim::Block {
        hash: m.root.clone(),
        reason: "test".into(),
    };
    let alice = sign::generate_key(&tmp.path().join("alice")).unwrap();
    assert!(
        client
            .submit(&Statement::new(&alice, block.clone()))
            .await
            .is_err()
    );
    client
        .submit(&Statement::new(&op_key, block))
        .await
        .unwrap();

    // After syncing, the node deletes the manifest, stops listing it and won't serve it.
    assert_eq!(follower.sync(&seed).await.unwrap(), 1);
    assert!(seed.manifests().unwrap().is_empty());
    let local = Arc::new(Store::open(&tmp.path().join("local")).unwrap());
    let mut f2 = registry::Follower::new(&url, None).unwrap();
    f2.sync(&local).await.unwrap();
    assert!(
        chungus::net::fetch(&m.root, local.clone(), &[peer], None, &[])
            .await
            .is_err()
    );
    // Nor can anyone publish the blocked root.
    let publish = publish(&alice, "alice/banned", &m.root, "");
    let bytes = serde_json::to_vec(&m).unwrap();
    let err = client.publish(&publish, &bytes).await.unwrap_err();
    assert!(err.to_string().contains("blocked"), "{err}");

    // Blocking a single chunk stops any model that contains it.
    let chunk = m.files[0].chunks[0].hash.clone();
    let mut set = std::collections::HashSet::new();
    set.insert(chunk.clone());
    let other = Arc::new(Store::open(&tmp.path().join("other")).unwrap());
    other.set_blocked(set);
    assert!(chungus::pack(&model, &other).is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn only_the_operator_sets_anchors() {
    let tmp = tempfile::tempdir().unwrap();
    let reg = Arc::new(Registry::open(&tmp.path().join("reg")).unwrap());
    let op_key = reg.operator_key().unwrap().clone();
    let operator = reg.operator();
    let url = spawn(reg).await;
    let client = Client::new(&url).unwrap();
    let mallory = sign::generate_key(&tmp.path().join("mallory")).unwrap();
    assert!(client.anchors(None).await.unwrap().is_empty());

    let peer = libp2p::identity::Keypair::generate_ed25519()
        .public()
        .to_peer_id();
    let anchor = format!("/ip4/203.0.113.7/tcp/4001/p2p/{peer}");
    let set = |k: &ed25519_dalek::SigningKey, addrs: Vec<String>| {
        Statement::new(k, Claim::Anchors { addrs })
    };
    let err = client
        .submit(&set(&mallory, vec![anchor.clone()]))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("only the registry operator"),
        "{err}"
    );
    // Anchors must name their peer.
    assert!(
        client
            .submit(&set(&op_key, vec!["/ip4/203.0.113.7/tcp/4001".into()]))
            .await
            .is_err()
    );

    client
        .submit(&set(&op_key, vec![anchor.clone()]))
        .await
        .unwrap();
    let got = client.anchors(Some(&operator)).await.unwrap();
    assert_eq!(got, vec![anchor.parse::<libp2p::Multiaddr>().unwrap()]);
    // Pinning a different operator key rejects the list.
    let other = sign::public_key_string(&mallory.verifying_key());
    assert!(client.anchors(Some(&other)).await.is_err());
    // A new list replaces the old one, and the log still audits.
    client.submit(&set(&op_key, vec![])).await.unwrap();
    assert!(client.anchors(None).await.unwrap().is_empty());
    client.audit(Some(&operator)).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn followers_accept_a_delegated_online_key() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("reg");
    let operator = Registry::open(&data).unwrap().operator();
    let root_path = tmp.path().join("root.key");
    std::fs::rename(data.join("operator.key"), &root_path).unwrap();
    let root = sign::load_key(&root_path).unwrap();
    let reg = Arc::new(Registry::open(&data).unwrap());
    let online = reg.online_public();
    let url = spawn(reg).await;
    let client = Client::new(&url).unwrap();
    let store = chungus::store::Store::open(&tmp.path().join("store")).unwrap();

    // Before delegating, a follower pinned to the root refuses the online key's head.
    let mut follower = registry::Follower::new(&url, Some(operator.clone())).unwrap();
    assert!(follower.sync(&store).await.is_err());
    assert!(client.audit(Some(&operator)).await.is_err());

    client
        .submit(&Statement::new(
            &root,
            Claim::Delegate {
                key: online,
                expires: registry::now() + 3600,
            },
        ))
        .await
        .unwrap();
    let mut follower = registry::Follower::new(&url, Some(operator.clone())).unwrap();
    follower.sync(&store).await.unwrap();
    client.audit(Some(&operator)).await.unwrap();
    // Unpinned clients learn the root from the head, and check it against the log.
    client.audit(None).await.unwrap();

    // Anchors set by the online key are accepted as the operator's.
    let online_key = sign::load_key(&data.join("online.key")).unwrap();
    let anchor =
        "/ip4/203.0.113.7/tcp/4001/p2p/12D3KooWRaVx8DKtusbdVeThtaFxqR7C8jgSvz6fArBwh52SCAeR";
    client
        .submit(&Statement::new(
            &online_key,
            Claim::Anchors {
                addrs: vec![anchor.into()],
            },
        ))
        .await
        .unwrap();
    assert_eq!(client.anchors(Some(&operator)).await.unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_site_gets_a_summary_of_each_model() {
    use chungus::manifest::{ChunkRef, FileEntry, Manifest};
    use chungus::registry::{Summary, SummaryFile};
    use chungus::segment::Dtype;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("reg");
    let url = spawn(Arc::new(Registry::open(&dir).unwrap())).await;
    let client = Client::new(&url).unwrap();
    let alice = sign::generate_key(&tmp.path().join("alice")).unwrap();

    let chunk = |s: &str, len| ChunkRef {
        hash: blake3::hash(s.as_bytes()).to_hex().to_string(),
        len,
        dtype: Dtype::Raw,
    };
    let file = |path: &str, chunks: Vec<ChunkRef>| FileEntry {
        path: path.into(),
        size: chunks.iter().map(|c| u64::from(c.len)).sum(),
        hash: blake3::hash(path.as_bytes()).to_hex().to_string(),
        chunks,
    };
    // A repeated chunk counts once towards what a download transfers.
    let m = Manifest::new(vec![
        file("config.json", vec![chunk("config", 100)]),
        file(
            "model.safetensors",
            vec![chunk("a", 1000), chunk("b", 1000), chunk("a", 1000)],
        ),
    ]);
    let bytes = serde_json::to_vec(&m).unwrap();
    client
        .publish(&publish(&alice, "acme/model", &m.root, ""), &bytes)
        .await
        .unwrap();

    let want = Summary {
        size: 3100,
        weights: 3000,
        formats: vec!["safetensors".into()],
        chunks: 4,
        unique_chunks: 3,
        unique_bytes: 2100,
        // No headers were sent, so there's no count.
        params: None,
        dtypes: Default::default(),
        files: vec![
            SummaryFile {
                path: "config.json".into(),
                size: 100,
            },
            SummaryFile {
                path: "model.safetensors".into(),
                size: 3000,
            },
        ],
    };
    let get = |path: String| async move {
        serde_json::from_slice::<serde_json::Value>(
            &reqwest::get(path)
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .bytes()
                .await
                .unwrap(),
        )
        .unwrap()
    };
    let got: Summary =
        serde_json::from_value(get(format!("{url}/v1/summary/{}", m.root)).await).unwrap();
    assert_eq!(got, want);
    let index = get(format!("{url}/v1/index")).await;
    assert_eq!(index[0]["size"], 3100);
    let missing = reqwest::get(format!("{url}/v1/summary/{}", "0".repeat(64)))
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);

    // A restarted registry rebuilds the summaries from the manifests on disk.
    let reopened = Registry::open(&dir).unwrap();
    assert_eq!(*reopened.summary(&m.root).unwrap(), want);
}

/// Writes a safetensors file with tensors of the given dtype and shape, zero-filled.
fn safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[u64])]) {
    let mut header = serde_json::Map::new();
    let mut at = 0u64;
    for (name, dtype, shape) in tensors {
        let width = if *dtype == "F32" { 4 } else { 2 };
        let len = shape.iter().product::<u64>() * width;
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + len]}),
        );
        at += len;
    }
    let json = serde_json::to_vec(&header).unwrap();
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend(json);
    file.resize(file.len() + at as usize, 0);
    std::fs::write(path, file).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn parameters_are_counted_from_checked_headers() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("reg");
    let reg = Arc::new(Registry::open(&dir).unwrap());
    let url = spawn(reg.clone()).await;
    let client = Client::new(&url).unwrap();
    let alice = sign::generate_key(&tmp.path().join("alice")).unwrap();

    // Two shards, so the count covers both.
    let model = tmp.path().join("model");
    std::fs::create_dir(&model).unwrap();
    safetensors(
        &model.join("model-1.safetensors"),
        &[("embed", "BF16", &[1000, 64]), ("norm", "F32", &[64])],
    );
    safetensors(
        &model.join("model-2.safetensors"),
        &[("head", "BF16", &[64, 1000])],
    );
    std::fs::write(model.join("config.json"), "{}").unwrap();
    let store = chungus::store::Store::open(&tmp.path().join("store")).unwrap();
    let (m, _) = chungus::pack(&model, &store).unwrap();
    let bytes = serde_json::to_vec(&m).unwrap();
    let headers = chungus::safetensors_headers(&m, &store).unwrap();
    assert_eq!(headers.len(), 2);

    // A header that doesn't match the file's chunks is refused, with the publish.
    let mut forged = headers.clone();
    let h = forged.get_mut("model-2.safetensors").unwrap();
    *h = h.replace("[64,1000]", "[64,9000]");
    let err = client
        .publish_with_headers(
            &publish(&alice, "acme/model", &m.root, ""),
            &bytes,
            forged,
            BTreeMap::new(),
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("doesn't match its chunks"),
        "{err}"
    );

    client
        .publish_with_headers(
            &publish(&alice, "acme/model", &m.root, ""),
            &bytes,
            headers,
            BTreeMap::new(),
        )
        .await
        .unwrap();
    let s = reg.summary(&m.root).unwrap();
    assert_eq!(s.params, Some(128_064));
    assert_eq!(
        s.dtypes,
        std::collections::BTreeMap::from([("BF16".into(), 128_000), ("F32".into(), 64)])
    );

    // The headers are kept, so a restarted registry still has the count, and so does a
    // republish that sends none.
    client
        .publish(&publish(&alice, "acme/model", &m.root, "again"), &bytes)
        .await
        .unwrap();
    assert_eq!(reg.summary(&m.root).unwrap().params, Some(128_064));
    drop(reg);
    let reopened = Registry::open(&dir).unwrap();
    assert_eq!(reopened.summary(&m.root).unwrap().params, Some(128_064));
}

/// A GGUF file with a tokenizer-like array, a Q8_0 matrix and an F32 vector.
fn gguf(path: &std::path::Path, quant: u32, rows: u64) {
    use chungus::gguf::testing::{Tensor, Value, write};
    // Q8_0 and Q4_0 blocks: 32 elements in 34 or 18 bytes.
    let block = if quant == 8 { 34 } else { 18 };
    let tokens = (0..2000).map(|i| Value::Str(format!("tok{i}"))).collect();
    let file = write(
        vec![("tokenizer.ggml.tokens", Value::Array(8, tokens))],
        &[
            Tensor {
                name: "token_embd.weight".into(),
                dims: vec![64, rows],
                ty: quant,
                data: vec![quant as u8; (rows * 2 * block) as usize],
            },
            Tensor {
                name: "norm.weight".into(),
                dims: vec![64],
                ty: 0,
                data: vec![1; 256],
            },
        ],
        32,
    );
    std::fs::write(path, file).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn gguf_parameters_count_one_quantization() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("reg");
    let reg = Arc::new(Registry::open(&dir).unwrap());
    let url = spawn(reg.clone()).await;
    let client = Client::new(&url).unwrap();
    let alice = sign::generate_key(&tmp.path().join("alice")).unwrap();
    let store = chungus::store::Store::open(&tmp.path().join("store")).unwrap();

    // One quantization: its count, by ggml type.
    let one = tmp.path().join("one");
    std::fs::create_dir(&one).unwrap();
    gguf(&one.join("model-q8_0.gguf"), 8, 1000);
    let (m, _) = chungus::pack(&one, &store).unwrap();
    let bytes = serde_json::to_vec(&m).unwrap();
    let headers = chungus::gguf_headers(&m, &store).unwrap();
    assert_eq!(headers.len(), 1);

    // A header that doesn't match the file's chunks is refused, with the publish.
    let mut forged = headers.clone();
    let h = forged.get_mut("model-q8_0.gguf").unwrap();
    let at = h.len() - 200;
    h.replace_range(at..at + 2, if &h[at..at + 2] == "00" { "01" } else { "00" });
    let err = client
        .publish_with_headers(
            &publish(&alice, "acme/gguf", &m.root, ""),
            &bytes,
            BTreeMap::new(),
            forged,
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("doesn't match its chunks"),
        "{err}"
    );

    client
        .publish_with_headers(
            &publish(&alice, "acme/gguf", &m.root, ""),
            &bytes,
            BTreeMap::new(),
            headers,
        )
        .await
        .unwrap();
    let s = reg.summary(&m.root).unwrap();
    assert_eq!(s.params, Some(64_064));
    assert_eq!(
        s.dtypes,
        BTreeMap::from([("F32".into(), 64), ("Q8_0".into(), 64_000)])
    );

    // Two quantizations of one model: one count, not the sum, and no types.
    let two = tmp.path().join("two");
    std::fs::create_dir(&two).unwrap();
    gguf(&two.join("model-q8_0.gguf"), 8, 1000);
    gguf(&two.join("model-q4_0.gguf"), 2, 1000);
    let (m2, _) = chungus::pack(&two, &store).unwrap();
    let headers = chungus::gguf_headers(&m2, &store).unwrap();
    assert_eq!(headers.len(), 2);
    client
        .publish_with_headers(
            &publish(&alice, "acme/gguf2", &m2.root, ""),
            &serde_json::to_vec(&m2).unwrap(),
            BTreeMap::new(),
            headers,
        )
        .await
        .unwrap();
    let s = reg.summary(&m2.root).unwrap();
    assert_eq!(s.params, Some(64_064));
    assert!(s.dtypes.is_empty());

    // Quantizations that disagree (not one model after all): no count.
    let odd = tmp.path().join("odd");
    std::fs::create_dir(&odd).unwrap();
    gguf(&odd.join("a.gguf"), 8, 1000);
    gguf(&odd.join("b.gguf"), 8, 2000);
    let (m3, _) = chungus::pack(&odd, &store).unwrap();
    let headers = chungus::gguf_headers(&m3, &store).unwrap();
    client
        .publish_with_headers(
            &publish(&alice, "acme/gguf3", &m3.root, ""),
            &serde_json::to_vec(&m3).unwrap(),
            BTreeMap::new(),
            headers,
        )
        .await
        .unwrap();
    assert_eq!(reg.summary(&m3.root).unwrap().params, None);

    // The counts are kept, so a restarted registry still has them.
    drop(reg);
    let reopened = Registry::open(&dir).unwrap();
    assert_eq!(reopened.summary(&m.root).unwrap().params, Some(64_064));
    assert_eq!(reopened.summary(&m2.root).unwrap().params, Some(64_064));
}
