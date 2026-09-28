use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use libp2p::Multiaddr;

use chungus::hub;
use chungus::manifest::Manifest;
use chungus::net::{self, FetchStats};
use chungus::p2p;
use chungus::sign;
use chungus::store::{self, Store};

#[derive(Parser)]
#[command(version, about = "Chunk, compress, deduplicate and share model files")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

const DEFAULT_STORE: &str = ".chungus/store";

#[derive(Subcommand)]
enum Cmd {
    /// Pack a model file or directory into a chunk store. Prints the manifest root.
    Pack {
        input: PathBuf,
        /// Chunk store directory (shared across models, so repeats dedup).
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Also write the manifest JSON here.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Rebuild the files of a manifest (a root hash or a manifest file), verifying every chunk.
    Unpack {
        manifest: String,
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Output directory.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// List the models in a store.
    List {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
    },
    /// Report compression and dedup ratios for one or more models, without writing.
    Bench {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Share this store with peers on the LAN (read-only HTTP, advertised over mDNS).
    Serve {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        #[arg(long, default_value_t = net::DEFAULT_PORT)]
        port: u16,
        /// Don't advertise over mDNS; peers must name this node with --peer.
        #[arg(long)]
        no_mdns: bool,
    },
    /// Run a local Hugging Face cache. Point tools at it with HF_ENDPOINT=http://localhost:8080.
    Hub {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        #[arg(long, default_value_t = hub::DEFAULT_PORT)]
        port: u16,
        /// Where to get files no peer has.
        #[arg(long, default_value = hub::DEFAULT_UPSTREAM)]
        upstream: String,
        /// Never contact the upstream; serve only what this hub and its peers have.
        #[arg(long)]
        offline: bool,
        /// A peer to use in addition to discovered ones, e.g. http://192.168.1.20:8080.
        #[arg(long)]
        peer: Vec<String>,
        /// Don't advertise or discover peers over mDNS.
        #[arg(long)]
        no_mdns: bool,
    },
    /// Download a model by manifest root from LAN peers, falling back to an origin.
    Fetch {
        root: String,
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Also unpack the files into this directory.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// A peer to use in addition to discovered ones, e.g. http://192.168.1.20:7447.
        #[arg(long)]
        peer: Vec<String>,
        /// A server to use only when no peer has a chunk.
        #[arg(long)]
        origin: Option<String>,
        /// Skip mDNS discovery and use only --peer and --origin.
        #[arg(long)]
        no_mdns: bool,
        /// How long to listen for peers on the LAN, in seconds.
        #[arg(long, default_value_t = 2.0)]
        discover_secs: f64,
        /// Only download if this public key (chungus1...) has signed the manifest. Repeatable.
        #[arg(long)]
        trust: Vec<String>,
        /// Fetch over the internet swarm instead of the LAN, joining through this node
        /// (a multiaddr ending in /p2p/<peer id>). Repeatable.
        #[arg(long)]
        bootstrap: Vec<Multiaddr>,
    },
    /// Join the internet swarm: announce this store's models on the DHT and serve them.
    Node {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Address to listen on. Repeatable. Default: TCP and QUIC on port 4001.
        #[arg(long)]
        listen: Vec<Multiaddr>,
        /// A node to join through (a multiaddr ending in /p2p/<peer id>). Repeatable.
        #[arg(long)]
        bootstrap: Vec<Multiaddr>,
        /// A relay to be reachable through when behind NAT (ending in /p2p/<peer id>).
        #[arg(long)]
        relay: Vec<Multiaddr>,
        /// An address others can reach this node at, e.g. a public IP with a forwarded port.
        #[arg(long)]
        external: Vec<Multiaddr>,
        /// This node is directly reachable on its listen addresses (a public server).
        #[arg(long)]
        public: bool,
        /// Relay connections for nodes behind NAT. Use with --public.
        #[arg(long)]
        relay_server: bool,
    },
    /// Create a signing key and print its public key.
    Keygen {
        /// Where to write the key (default: ~/.chungus/key).
        #[arg(long)]
        key: Option<PathBuf>,
    },
    /// Sign a model in the store, so others can check it came from you.
    Sign {
        root: String,
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Signing key (default: ~/.chungus/key).
        #[arg(long)]
        key: Option<PathBuf>,
    },
    /// Show who has signed a model, or check that a given key has.
    Verify {
        root: String,
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Fail unless this public key (chungus1...) has signed it. Repeatable.
        #[arg(long)]
        trust: Vec<String>,
    },
}

fn key_path(key: Option<PathBuf>) -> Result<PathBuf> {
    match key {
        Some(k) => Ok(k),
        None => Ok(
            PathBuf::from(std::env::var_os("HOME").context("HOME is not set; pass --key")?)
                .join(".chungus/key"),
        ),
    }
}

fn report_fetch(s: &FetchStats) {
    let received: u64 = s.bytes_by_source.values().sum();
    println!(
        "{} chunks: {} already local, {} fetched ({:.1} MB) in {:.1}s",
        s.chunks,
        s.already_local,
        s.chunks - s.already_local,
        mb(received),
        s.secs
    );
    for (source, bytes) in &s.bytes_by_source {
        println!("  {source}: {:.1} MB", mb(*bytes));
    }
    if s.rejected > 0 {
        println!("  rejected {} bad chunk(s) and refetched them", s.rejected);
    }
}

fn parse_keys(keys: &[String]) -> Result<Vec<ed25519_dalek::VerifyingKey>> {
    keys.iter().map(|k| sign::parse_public_key(k)).collect()
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

/// A manifest argument is either a root hash in the store or a path to a manifest file.
fn load_manifest(arg: &str, store: &Store) -> Result<Manifest> {
    if store::is_hash(arg) && !Path::new(arg).exists() {
        return store.get_manifest(arg);
    }
    Ok(serde_json::from_slice(
        &fs::read(arg).with_context(|| format!("read {arg}"))?,
    )?)
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Pack {
            input,
            store,
            output,
        } => {
            let store = Store::open(&store)?;
            let (manifest, s) = chungus::pack(&input, &store)?;
            store.put_manifest(&manifest)?;
            if let Some(output) = output {
                fs::write(&output, serde_json::to_vec_pretty(&manifest)?)
                    .with_context(|| format!("write {}", output.display()))?;
            }
            println!("packed {:.1} MB in {} chunks", mb(s.raw_bytes), s.chunks);
            println!(
                "new: {} chunks, {:.1} MB raw -> {:.1} MB stored ({:.1}%)",
                s.new_chunks,
                mb(s.new_raw_bytes),
                mb(s.new_stored_bytes),
                pct(s.new_stored_bytes, s.new_raw_bytes)
            );
            println!("root {}", manifest.root);
        }
        Cmd::Unpack {
            manifest,
            store,
            output,
        } => {
            let store = Store::open(&store)?;
            let m = load_manifest(&manifest, &store)?;
            chungus::unpack(&m, &store, &output)?;
            println!("unpacked {} files, all chunks verified", m.files.len());
        }
        Cmd::List { store } => {
            let store = Store::open(&store)?;
            for root in store.manifests()? {
                let m = store.get_manifest(&root)?;
                let size: u64 = m.files.iter().map(|f| f.size).sum();
                let names: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
                println!("{root}  {:>10.1} MB  {}", mb(size), names.join(", "));
            }
        }
        Cmd::Bench { inputs } => {
            let r = chungus::bench(&inputs)?;
            let row = |name: &str, bytes: u64| {
                println!(
                    "{name:<28} {:>12.1} MB  {:>6.1}%",
                    mb(bytes),
                    pct(bytes, r.raw_bytes)
                );
            };
            row("raw", r.raw_bytes);
            row("zstd only", r.zstd_bytes);
            row("float transform + zstd", r.encoded_bytes);
            row("+ dedup (full pipeline)", r.dedup_bytes);
            println!(
                "chunks: {} total, {} unique ({:.1}% of raw bytes unique)",
                r.chunks,
                r.unique_chunks,
                pct(r.unique_raw_bytes, r.raw_bytes)
            );
            println!(
                "encode {:.0} MB/s, decode {:.0} MB/s (raw bytes, all cores)",
                mb(r.raw_bytes) / r.encode_secs.max(1e-9),
                mb(r.raw_bytes) / r.decode_secs.max(1e-9)
            );
        }
        Cmd::Serve {
            store,
            port,
            no_mdns,
        } => {
            let store = Arc::new(Store::open(&store)?);
            let addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("bind {addr}"))?;
            let _mdns = if no_mdns {
                None
            } else {
                Some(net::advertise(port, &net::node_id())?)
            };
            println!(
                "serving {} models on port {port}{}",
                store.manifests()?.len(),
                if no_mdns {
                    ""
                } else {
                    ", advertised on the LAN"
                }
            );
            net::serve_on(listener, store).await?;
        }
        Cmd::Hub {
            store,
            port,
            upstream,
            offline,
            peer,
            no_mdns,
        } => {
            let store = Arc::new(Store::open(&store)?);
            let upstream = (!offline).then_some(upstream);
            let hub = Arc::new(hub::Hub::new(store, upstream.clone(), peer.clone())?);
            let addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("bind {addr}"))?;
            let id = net::node_id();
            let _mdns = if no_mdns {
                None
            } else {
                // Keep looking for peers in the background; hubs come and go.
                let (hub, own_id) = (hub.clone(), id.clone());
                tokio::spawn(async move {
                    loop {
                        let found = {
                            let id = own_id.clone();
                            tokio::task::spawn_blocking(move || {
                                net::discover(Duration::from_secs(2), Some(&id))
                            })
                            .await
                        };
                        if let Ok(Ok(found)) = found {
                            let mut peers = peer.clone();
                            peers.extend(found);
                            peers.sort();
                            peers.dedup();
                            hub.set_peers(peers);
                        }
                        tokio::time::sleep(Duration::from_secs(30)).await;
                    }
                });
                Some(net::advertise(port, &id)?)
            };
            println!(
                "Hugging Face cache on http://localhost:{port} (upstream: {})",
                upstream.as_deref().unwrap_or("none, offline")
            );
            println!("use it with: export HF_ENDPOINT=http://localhost:{port}");
            axum::serve(listener, hub::router(hub)).await?;
        }
        Cmd::Fetch {
            root,
            store,
            output,
            peer,
            origin,
            no_mdns,
            discover_secs,
            trust,
            bootstrap,
        } => {
            let trust = parse_keys(&trust)?;
            let store = Arc::new(Store::open(&store)?);
            let (manifest, s) = if bootstrap.is_empty() {
                let mut peers = peer;
                if !no_mdns {
                    let wait = Duration::from_secs_f64(discover_secs);
                    let found =
                        tokio::task::spawn_blocking(move || net::discover(wait, None)).await??;
                    println!("found {} peer(s) on the LAN", found.len());
                    peers.extend(found);
                }
                peers.sort();
                peers.dedup();
                net::fetch(&root, store.clone(), &peers, origin.as_deref(), &trust).await?
            } else {
                let key = p2p::load_or_create_identity(&store.dir().join("node.key"))?;
                let config = p2p::Config {
                    listen: vec![
                        "/ip4/0.0.0.0/tcp/0".parse()?,
                        "/ip4/0.0.0.0/udp/0/quic-v1".parse()?,
                    ],
                    bootstrap,
                    ..Default::default()
                };
                let node = p2p::Node::start(store.clone(), key, config).await?;
                p2p::fetch(&node, &root, store.clone(), &trust).await?
            };
            report_fetch(&s);
            if let Some(output) = output {
                tokio::task::spawn_blocking(move || chungus::unpack(&manifest, &store, &output))
                    .await??;
                println!("unpacked, all chunks verified");
            }
        }
        Cmd::Node {
            store,
            listen,
            bootstrap,
            relay,
            external,
            public,
            relay_server,
        } => {
            let store = Arc::new(Store::open(&store)?);
            let key = p2p::load_or_create_identity(&store.dir().join("node.key"))?;
            let listen = if listen.is_empty() {
                vec![
                    format!("/ip4/0.0.0.0/tcp/{}", p2p::DEFAULT_PORT).parse()?,
                    format!("/ip4/0.0.0.0/udp/{}/quic-v1", p2p::DEFAULT_PORT).parse()?,
                ]
            } else {
                listen
            };
            let config = p2p::Config {
                listen,
                bootstrap,
                relays: relay,
                external,
                public,
                relay_server,
            };
            let node = p2p::Node::start(store.clone(), key, config).await?;
            println!("peer id {}", node.peer_id);
            println!("sharing {} models", store.manifests()?.len());
            // Relayed addresses appear once a reservation is made, so keep reporting new ones.
            let mut shown = std::collections::HashSet::new();
            loop {
                for a in node.addresses(Duration::ZERO).await? {
                    if shown.insert(a.clone()) {
                        println!("address {a}");
                    }
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
        Cmd::Keygen { key } => {
            let path = key_path(key)?;
            let k = sign::generate_key(&path)?;
            println!("wrote {}", path.display());
            println!("public key {}", sign::public_key_string(&k.verifying_key()));
        }
        Cmd::Sign { root, store, key } => {
            let store = Store::open(&store)?;
            store.get_manifest(&root)?;
            let k = sign::load_key(&key_path(key)?)?;
            let s = sign::sign(&k, &root);
            store.add_signatures(&root, std::slice::from_ref(&s))?;
            println!("signed {root} as {}", s.key);
        }
        Cmd::Verify { root, store, trust } => {
            let trust = parse_keys(&trust)?;
            let store = Store::open(&store)?;
            let m = store.get_manifest(&root)?;
            if !m.verify_root() {
                anyhow::bail!("manifest {root} does not match its root hash");
            }
            let sigs = store.signatures(&root)?;
            for s in sigs.iter().filter(|s| sign::verify(s, &root)) {
                println!("signed by {}", s.key);
            }
            if sigs.is_empty() {
                println!("no signatures");
            }
            if !trust.is_empty() && !sign::trusted_by(&sigs, &root, &trust) {
                anyhow::bail!("no trusted key has signed {root}");
            }
        }
    }
    Ok(())
}
