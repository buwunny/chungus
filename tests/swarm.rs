//! Nodes finding each other over the DHT and sharing models on localhost.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

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
    fs::write(dir.join("weights.bin"), data).unwrap();
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

    let (got, stats) = p2p::fetch(&fetcher, &m.root, local.clone(), &[author.verifying_key()])
        .await
        .unwrap();
    assert_eq!(stats.already_local, 0);
    assert_eq!(stats.bytes_by_source.len(), 1);
    assert_eq!(local.signatures(&m.root).unwrap().len(), 1);
    let out = tmp.path().join("out");
    chungus::unpack(&got, &local, &out).unwrap();
    for f in ["weights.bin", "config.json"] {
        assert_eq!(
            fs::read(out.join(f)).unwrap(),
            fs::read(model.join(f)).unwrap()
        );
    }

    // Nobody provides a model that doesn't exist.
    let missing = "0".repeat(64);
    let Err(err) = p2p::fetch(&fetcher, &missing, local, &[]).await else {
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
    let (got, _) = p2p::fetch(&fetcher, &m.root, local.clone(), &[])
        .await
        .unwrap();
    chungus::unpack(&got, &local, &tmp.path().join("out")).unwrap();
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

    let (got, stats) = p2p::fetch(&fetcher, &m.root, local.clone(), &[])
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
