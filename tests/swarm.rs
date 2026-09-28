//! Nodes finding each other over the DHT and sharing models on localhost.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chungus::limits::Limits;
use chungus::p2p::{self, Config, Node};
use chungus::sign;
use chungus::store::Store;
use libp2p::Multiaddr;
use libp2p::identity::Keypair;

fn write_model(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let mut state = 777u64;
    let data: Vec<u8> = (0..2_000_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    fs::write(dir.join("weights.safetensors"), data).unwrap();
    fs::write(dir.join("config.json"), b"{\"layers\": 2}").unwrap();
}

fn tcp() -> Vec<Multiaddr> {
    vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()]
}

async fn node(store: &Arc<Store>, cfg: Config) -> (Node, Vec<Multiaddr>) {
    let node = Node::start(store.clone(), Keypair::generate_ed25519(), cfg)
        .await
        .unwrap();
    let addrs = node.addresses(Duration::from_secs(5)).await.unwrap();
    (node, addrs)
}

fn store(dir: &Path) -> Arc<Store> {
    Arc::new(Store::open(dir).unwrap())
}

/// Poll until someone provides `root`, since announcements take a moment to land.
async fn wait_for_providers(node: &Node, root: &str) {
    for _ in 0..50 {
        if !node.providers(root).await.unwrap().is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("nobody announced {root}");
}

#[tokio::test(flavor = "multi_thread")]
async fn fetch_through_the_dht() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();
    let author = sign::generate_key(&tmp.path().join("author")).unwrap();
    seed_store
        .add_signatures(&m.root, &[sign::sign(&author, &m.root)])
        .unwrap();

    // A bootstrap node with nothing in it, a seed that has the model, and a fetcher.
    let (_boot, boot_addrs) = node(
        &store(&tmp.path().join("boot")),
        Config {
            listen: tcp(),
            public: true,
            ..Default::default()
        },
    )
    .await;
    let join = Config {
        listen: tcp(),
        public: true,
        bootstrap: boot_addrs,
        ..Default::default()
    };
    let (seed, _) = node(&seed_store, join.clone()).await;
    seed.announce().unwrap();
    let local = store(&tmp.path().join("local"));
    let (fetcher, _) = node(&local, join).await;
    wait_for_providers(&fetcher, &m.root).await;

    let (got, stats) = p2p::fetch(
        &fetcher,
        &m.root,
        local.clone(),
        &[author.verifying_key()],
        None,
    )
    .await
    .unwrap();
    assert_eq!(stats.already_local, 0);
    assert_eq!(stats.bytes_by_source.len(), 1);
    assert_eq!(local.signatures(&m.root).unwrap().len(), 1);
    let out = tmp.path().join("out");
    chungus::unpack(&got, &local, &out).unwrap();
    for f in ["weights.safetensors", "config.json"] {
        assert_eq!(
            fs::read(out.join(f)).unwrap(),
            fs::read(model.join(f)).unwrap()
        );
    }

    // Nobody provides a model that doesn't exist.
    let missing = "0".repeat(64);
    let Err(err) = p2p::fetch(&fetcher, &missing, local, &[], None).await else {
        panic!("fetched a model nobody has");
    };
    assert!(err.to_string().contains("nobody"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn fetch_from_a_node_reachable_only_through_a_relay() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();

    let (_relay, relay_addrs) = node(
        &store(&tmp.path().join("relay")),
        Config {
            listen: tcp(),
            public: true,
            relay_server: true,
            ..Default::default()
        },
    )
    .await;
    // The seed has no direct listen address at all: only the relayed one.
    let (seed, seed_addrs) = node(
        &seed_store,
        Config {
            bootstrap: relay_addrs.clone(),
            relays: relay_addrs.clone(),
            ..Default::default()
        },
    )
    .await;
    assert!(
        seed_addrs
            .iter()
            .all(|a| a.to_string().contains("p2p-circuit")),
        "{seed_addrs:?}"
    );
    seed.announce().unwrap();

    let local = store(&tmp.path().join("local"));
    let (fetcher, _) = node(
        &local,
        Config {
            listen: tcp(),
            bootstrap: relay_addrs,
            ..Default::default()
        },
    )
    .await;
    wait_for_providers(&fetcher, &m.root).await;
    let (got, _) = p2p::fetch(&fetcher, &m.root, local.clone(), &[], None)
        .await
        .unwrap();
    chungus::unpack(&got, &local, &tmp.path().join("out")).unwrap();
}

/// A peer behind NAT is only reachable through its relay, and a fetcher that learns of
/// it from a provider lookup gets a bare peer id, or at best a private address. Requests
/// then try reaching it through the nodes this one joined through.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_reaches_a_nat_peer_through_a_known_relay() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();

    let (_relay, relay_addrs) = node(
        &store(&tmp.path().join("relay")),
        Config {
            listen: tcp(),
            public: true,
            relay_server: true,
            ..Default::default()
        },
    )
    .await;
    // Behind NAT: no direct listen address, only the relayed one.
    let (seed, _) = node(
        &seed_store,
        Config {
            bootstrap: relay_addrs.clone(),
            relays: relay_addrs.clone(),
            ..Default::default()
        },
    )
    .await;

    // The fetcher knows the relay but runs no DHT lookup, so it has no address for the
    // seed at all: only its peer id, as a provider lookup leaves it.
    let (fetcher, _) = node(
        &store(&tmp.path().join("local")),
        Config {
            relays: relay_addrs,
            limits: Limits {
                download_only: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let resp = fetcher
        .request(seed.peer_id, p2p::Request::Manifest(m.root.clone()))
        .await
        .unwrap();
    assert!(matches!(resp, p2p::Response::Manifest(Some(_))), "{resp:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_node_with_part_of_a_model_serves_its_blocks() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let block_bytes = 256 << 10;
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();

    // The partial node has the manifest and only the first half of the blocks.
    let partial_store = store(&tmp.path().join("partial"));
    partial_store.put_manifest(&m).unwrap();
    let blocks = m.blocks(block_bytes);
    assert!(blocks.len() >= 4, "{} blocks", blocks.len());
    for b in &blocks[..blocks.len() / 2] {
        for c in &b.chunks {
            partial_store
                .put(&c.hash, &seed_store.get(&c.hash).unwrap())
                .unwrap();
        }
    }

    let (_boot, boot_addrs) = node(
        &store(&tmp.path().join("boot")),
        Config {
            listen: tcp(),
            public: true,
            block_bytes,
            ..Default::default()
        },
    )
    .await;
    let join = Config {
        listen: tcp(),
        public: true,
        bootstrap: boot_addrs,
        block_bytes,
        ..Default::default()
    };
    let (seed, _) = node(&seed_store, join.clone()).await;
    let (partial, _) = node(&partial_store, join.clone()).await;
    seed.announce().unwrap();
    partial.announce().unwrap();
    let local = store(&tmp.path().join("local"));
    let (fetcher, _) = node(&local, join).await;
    wait_for_providers(&fetcher, &m.root).await;
    for _ in 0..50 {
        if fetcher.holders(&m.root).await.unwrap().len() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Only the seed claims the whole model; both hold the manifest.
    assert_eq!(fetcher.providers(&m.root).await.unwrap(), [seed.peer_id]);
    assert_eq!(fetcher.holders(&m.root).await.unwrap().len(), 2);

    let (got, stats) = p2p::fetch(&fetcher, &m.root, local.clone(), &[], None)
        .await
        .unwrap();
    assert_eq!(stats.rejected, 0);
    assert!(
        stats
            .bytes_by_source
            .contains_key(&partial.peer_id.to_string()),
        "{:?}",
        stats.bytes_by_source
    );
    chungus::unpack(&got, &local, &tmp.path().join("out")).unwrap();
}

#[test]
fn blocks_cover_every_unique_chunk_once() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let s = store(&tmp.path().join("s"));
    let (m, _) = chungus::pack(&model, &s).unwrap();
    let blocks = m.blocks(256 << 10);
    let mut seen = std::collections::HashSet::new();
    for b in &blocks {
        for c in &b.chunks {
            assert!(seen.insert(c.hash.clone()));
        }
    }
    let unique: std::collections::HashSet<_> = m
        .files
        .iter()
        .flat_map(|f| &f.chunks)
        .map(|c| &c.hash)
        .collect();
    assert_eq!(seen.len(), unique.len());
    // Block ids are stable.
    let again = m.blocks(256 << 10);
    assert!(blocks.iter().zip(&again).all(|(a, b)| a.id == b.id));
}

#[tokio::test(flavor = "multi_thread")]
async fn anchors_serve_under_tight_limits() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();

    // The seed never announces, serves one request per peer at a time and uploads at
    // 4 MB/s. The fetcher knows it only as an anchor, with no bootstrap node.
    let (_seed, seed_addrs) = node(
        &seed_store,
        Config {
            listen: tcp(),
            public: true,
            limits: Limits {
                upload_bytes_per_sec: Some(4_000_000),
                max_requests_per_peer: 1,
                max_uploads: 2,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let local = store(&tmp.path().join("local"));
    let (fetcher, _) = node(
        &local,
        Config {
            listen: tcp(),
            anchors: seed_addrs,
            limits: Limits {
                download_only: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let (got, stats) = p2p::fetch(&fetcher, &m.root, local.clone(), &[], None)
        .await
        .unwrap();
    assert_eq!(stats.bytes_by_source.len(), 1);
    let out = tmp.path().join("out");
    chungus::unpack(&got, &local, &out).unwrap();
    assert_eq!(
        fs::read(out.join("weights.safetensors")).unwrap(),
        fs::read(model.join("weights.safetensors")).unwrap()
    );

    // A download-only node holds the model but serves none of it.
    let (_leech, leech_addrs) = node(
        &local,
        Config {
            listen: tcp(),
            public: true,
            limits: Limits {
                download_only: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let other = store(&tmp.path().join("other"));
    let (asker, _) = node(
        &other,
        Config {
            listen: tcp(),
            anchors: leech_addrs,
            ..Default::default()
        },
    )
    .await;
    assert!(p2p::fetch(&asker, &m.root, other, &[], None).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_reads_lazily_from_the_swarm() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();
    let (seed, seed_addrs) = node(
        &seed_store,
        Config {
            listen: tcp(),
            public: true,
            ..Default::default()
        },
    )
    .await;
    seed.announce().unwrap();
    let local = store(&tmp.path().join("local"));
    let (fetcher, _) = node(
        &local,
        Config {
            listen: tcp(),
            bootstrap: seed_addrs,
            ..Default::default()
        },
    )
    .await;
    wait_for_providers(&fetcher, &m.root).await;

    let (manifest, sources) = p2p::prepare(&fetcher, &m.root, &local, &[], None)
        .await
        .unwrap();
    let file = manifest
        .files
        .iter()
        .position(|f| f.path == "weights.safetensors")
        .unwrap();
    let lazy = chungus::lazy::Lazy::new(local.clone(), manifest, Arc::new(sources)).unwrap();
    let original = fs::read(model.join("weights.safetensors")).unwrap();
    let got = lazy.read(file, 1_234_567, 100_000).await.unwrap();
    assert_eq!(got, original[1_234_567..1_334_567]);
    lazy.prefetch(8).await.unwrap();
    assert_eq!(
        lazy.stats
            .local_bytes
            .load(std::sync::atomic::Ordering::Relaxed),
        lazy.total_bytes()
    );
}

/// A fake huggingface.co whose `auth-check` accepts only the token "good".
async fn fake_hf() -> String {
    use axum::http::{HeaderMap, StatusCode};
    let app = axum::Router::new().fallback(|headers: HeaderMap| async move {
        let auth = headers.get("authorization").and_then(|v| v.to_str().ok());
        if auth == Some("Bearer good") {
            StatusCode::OK
        } else {
            StatusCode::FORBIDDEN
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await });
    url
}

#[tokio::test(flavor = "multi_thread")]
async fn gated_models_need_a_ticket() {
    use chungus::registry::{self, Claim, Registry, Statement};
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model");
    write_model(&model);
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();

    // A registry that checks access against the fake Hub, with the model published gated.
    let reg = Arc::new(
        Registry::open(&tmp.path().join("reg"))
            .unwrap()
            .with_hf(&fake_hf().await),
    );
    let alice = sign::generate_key(&tmp.path().join("alice")).unwrap();
    reg.submit(Statement::new(
        &alice,
        Claim::Publish {
            name: "acme/llama".into(),
            rev: "main".into(),
            root: m.root.clone(),
            description: String::new(),
            gated: Some("meta-llama/Llama-3.2-1B".into()),
        },
    ))
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let reg_url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, registry::router(reg)).await });

    // The seed follows the registry, so it knows the model is gated.
    registry::Follower::new(&reg_url, None)
        .unwrap()
        .sync(&seed_store)
        .await
        .unwrap();
    let (seed, seed_addrs) = node(
        &seed_store,
        Config {
            listen: tcp(),
            public: true,
            ..Default::default()
        },
    )
    .await;
    seed.announce().unwrap();
    let join = Config {
        listen: tcp(),
        public: true,
        bootstrap: seed_addrs,
        ..Default::default()
    };
    let access = |token: &str| p2p::Access {
        registry: reg_url.clone(),
        hf_token: token.into(),
    };

    // No token: refused, and told why.
    let local = store(&tmp.path().join("none"));
    let (fetcher, _) = node(&local, join.clone()).await;
    wait_for_providers(&fetcher, &m.root).await;
    let err = p2p::fetch(&fetcher, &m.root, local, &[], None)
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains("gated by https://huggingface.co/meta-llama/Llama-3.2-1B"),
        "{err}"
    );

    // A token the Hub rejects: refused.
    let local = store(&tmp.path().join("bad"));
    let (fetcher, _) = node(&local, join.clone()).await;
    wait_for_providers(&fetcher, &m.root).await;
    let err = p2p::fetch(&fetcher, &m.root, local, &[], Some(access("bad")))
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("doesn't have access"), "{err}");

    // An accepted token: the registry issues a ticket, the seed accepts it, and the model
    // downloads intact.
    let local = store(&tmp.path().join("good"));
    let (fetcher, _) = node(&local, join).await;
    wait_for_providers(&fetcher, &m.root).await;
    let (got, _) = p2p::fetch(&fetcher, &m.root, local.clone(), &[], Some(access("good")))
        .await
        .unwrap();
    chungus::unpack(&got, &local, &tmp.path().join("out")).unwrap();
}
