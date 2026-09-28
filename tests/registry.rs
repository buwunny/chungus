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

#[tokio::test(flavor = "multi_thread")]
async fn publish_resolve_search_audit() {
    let tmp = tempfile::tempdir().unwrap();
    let reg = Arc::new(Registry::open(&tmp.path().join("reg")).unwrap());
    let operator = reg.operator();
    let url = spawn(reg).await;
    let client = Client::new(&url).unwrap();
    let alice = sign::generate_key(&tmp.path().join("alice")).unwrap();
    let mallory = sign::generate_key(&tmp.path().join("mallory")).unwrap();
    let root = "ab".repeat(32);

    client
        .submit(&publish(
            &alice,
            "acme/tiny-llama",
            &root,
            "A tiny Llama for tests",
        ))
        .await
        .unwrap();
    // Someone else can't take the name.
    let err = client
        .submit(&publish(&mallory, "acme/tiny-llama", &"cd".repeat(32), ""))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not an owner"), "{err}");
    // A statement altered in transit fails its signature check.
    let mut forged = publish(&alice, "acme/other", &root, "");
    forged.time += 1;
    assert!(client.submit(&forged).await.is_err());

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
    let model = tmp.path().join("model.bin");
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
    assert!(client.submit(&publish).await.is_err());

    // Blocking a single chunk stops any model that contains it.
    let chunk = m.files[0].chunks[0].hash.clone();
    let mut set = std::collections::HashSet::new();
    set.insert(chunk.clone());
    let other = Arc::new(Store::open(&tmp.path().join("other")).unwrap());
    other.set_blocked(set);
    assert!(chungus::pack(&model, &other).is_err());
}
