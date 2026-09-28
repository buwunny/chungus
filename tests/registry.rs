//! A registry over HTTP: publishing, resolving, searching and auditing the log.

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
    let op_key = reg.operator_key().clone();
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
    let op_key = reg.operator_key().clone();
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
