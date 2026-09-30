//! The leaderboard end to end: a registered node serves a download, the downloader's
//! receipt reaches the registry through it and is credited against the counted lookup,
//! and an anchor's probe confirms the node is up and holds the model.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chungus::downloads::Totals;
use chungus::leaderboard::{Prober, Registration, Row};
use chungus::limits::Limits;
use chungus::p2p::{self, Config, Node};
use chungus::registry::{self, Claim, Client, Registry, Statement};
use chungus::sign;
use chungus::store::Store;
use libp2p::Multiaddr;
use libp2p::identity::Keypair;

fn write_model(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let mut state = 4242u64;
    let data: Vec<u8> = (0..1_000_000)
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

async fn start(store: &Arc<Store>, key: Keypair, cfg: Config) -> (Node, Vec<Multiaddr>) {
    let node = Node::start(store.clone(), key, cfg).await.unwrap();
    let addrs = node.addresses(Duration::from_secs(5)).await.unwrap();
    (node, addrs)
}

fn store(dir: &Path) -> Arc<Store> {
    Arc::new(Store::open(dir).unwrap())
}

async fn get<T: serde::de::DeserializeOwned>(url: &str) -> T {
    let body = reqwest::get(url).await.unwrap().bytes().await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn receipts_and_probes_rank_a_node() {
    let tmp = tempfile::tempdir().unwrap();
    let reg = Arc::new(Registry::open(&tmp.path().join("reg")).unwrap());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(registry::serve(
        listener,
        reg.clone(),
        std::future::pending(),
    ));
    let client = Client::new(&url).unwrap();

    // A model, packed into the seed's store and published.
    let model = tmp.path().join("model");
    write_model(&model);
    let seed_store = store(&tmp.path().join("seed"));
    let (m, _) = chungus::pack(&model, &seed_store).unwrap();
    seed_store.put_manifest(&m).unwrap();
    let author = sign::generate_key(&tmp.path().join("author")).unwrap();
    seed_store
        .add_signatures(&m.root, &[sign::sign(&author, &m.root)])
        .unwrap();
    let st = Statement::new(
        &author,
        Claim::Publish {
            name: "acme/tiny".into(),
            rev: "main".into(),
            root: m.root.clone(),
            description: String::new(),
            gated: None,
        },
    );
    client
        .publish(&st, &seed_store.get_manifest_bytes(&m.root).unwrap())
        .await
        .unwrap();

    // A bootstrap node, and a seed that takes receipts and is on the leaderboard.
    let (_boot, boot_addrs) = start(
        &store(&tmp.path().join("boot")),
        Keypair::generate_ed25519(),
        Config {
            listen: tcp(),
            public: true,
            ..Default::default()
        },
    )
    .await;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let seed_key = Keypair::generate_ed25519();
    let (seed, _) = start(
        &seed_store,
        seed_key.clone(),
        Config {
            listen: tcp(),
            public: true,
            bootstrap: boot_addrs.clone(),
            receipts: Some(tx),
            ..Default::default()
        },
    )
    .await;
    seed.announce().unwrap();
    client
        .register(&Registration::new(
            &seed_key,
            "bunny burrow",
            vec![m.root.clone()],
        ))
        .await
        .unwrap();

    // A download: the lookup is counted and bound to the fetch's one-run key.
    let fetch_key = Keypair::generate_ed25519();
    let fetch_peer = fetch_key.public().to_peer_id().to_string();
    client
        .resolve_download("acme/tiny", "main", Some(&fetch_peer))
        .await
        .unwrap();
    let local = store(&tmp.path().join("local"));
    let (fetcher, _) = start(
        &local,
        fetch_key,
        Config {
            listen: tcp(),
            bootstrap: boot_addrs.clone(),
            limits: Limits {
                download_only: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    for _ in 0..50 {
        if !fetcher.providers(&m.root).await.unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let (_, stats) = p2p::fetch(&fetcher, &m.root, local.clone(), &[], None)
        .await
        .unwrap();
    let served = p2p::served_by(&stats);
    assert_eq!(served.len(), 1);
    assert_eq!(p2p::send_receipts(&fetcher, &m.root, &served).await, 1);

    // The seed passes the receipt on; the registry credits it.
    let receipt = rx.recv().await.unwrap();
    assert_eq!(receipt.server, seed.peer_id.to_string());
    assert_eq!(receipt.fetcher, fetch_peer);
    assert_eq!(
        client
            .receipts(std::slice::from_ref(&receipt))
            .await
            .unwrap(),
        1
    );
    // A second lookup from the same network isn't counted, so a second one-run peer id
    // gets no credit for the same model.
    let again = Keypair::generate_ed25519();
    client
        .resolve_download(
            "acme/tiny",
            "main",
            Some(&again.public().to_peer_id().to_string()),
        )
        .await
        .unwrap();
    let copy = chungus::leaderboard::Receipt::new(&again, &m.root, &seed.peer_id, 1000, 1);
    assert_eq!(client.receipts(&[copy]).await.unwrap(), 0);

    // Credit is capped at the model's unique bytes, which the blobs of incompressible
    // test data slightly exceed.
    let cap = reg.summary(&m.root).unwrap().unique_bytes;
    let credited = receipt.bytes.min(cap);
    assert!(credited > cap * 9 / 10);
    let rows: Vec<Row> = get(&format!("{url}/v1/leaderboard?metric=bytes")).await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].name.as_deref(), Some("bunny burrow"));
    assert_eq!(rows[0].bytes, credited);
    assert_eq!(rows[0].downloads, 1);
    assert_eq!(rows[0].uptime, None);
    let d: Totals = get(&format!("{url}/v1/downloads")).await;
    assert_eq!((d.total, d.bytes_served), (1, credited));

    // Only an anchor's probes count. Make one an anchor, then probe from a separate
    // download-only identity.
    let anchor = Keypair::generate_ed25519();
    let prober_node = |key| async {
        start(
            &store(&tmp.path().join("prober")),
            key,
            Config {
                listen: tcp(),
                bootstrap: boot_addrs.clone(),
                limits: Limits {
                    download_only: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .0
    };
    let mut stranger = Prober::new(
        url.clone(),
        Keypair::generate_ed25519(),
        prober_node(Keypair::generate_ed25519()).await,
    );
    assert!(stranger.round().await.is_err());
    let addr = format!("/ip4/127.0.0.1/tcp/1/p2p/{}", anchor.public().to_peer_id());
    reg.submit(Statement::new(
        reg.operator_key().unwrap(),
        Claim::Anchors { addrs: vec![addr] },
    ))
    .unwrap();
    let mut prober = Prober::new(
        url.clone(),
        anchor,
        prober_node(Keypair::generate_ed25519()).await,
    );
    assert_eq!(prober.round().await.unwrap(), (1, 1));
    let rows: Vec<Row> = get(&format!("{url}/v1/leaderboard?metric=uptime")).await;
    assert_eq!((rows[0].uptime, rows[0].models), (Some(1.0), 1));

    let h: chungus::leaderboard::History = get(&format!("{url}/v1/nodes/{}", seed.peer_id)).await;
    assert_eq!(h.held, std::slice::from_ref(&m.root));
    assert_eq!(h.days[0].totals.bytes, credited);
    assert_eq!(h.days[0].totals.probes, 1);

    // State survives a restart.
    reg.flush().unwrap();
    drop(reg);
    let reopened = Registry::open(&tmp.path().join("reg")).unwrap();
    let rows =
        reopened.board(|b| b.leaderboard(chungus::leaderboard::Metric::Bytes, 30, registry::now()));
    assert_eq!(rows[0].bytes, credited);
}
