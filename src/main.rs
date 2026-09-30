use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::fs;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use libp2p::Multiaddr;

use chungus::hub;
use chungus::limits::{Limits, RateLimiter, RelayLimits};
use chungus::manifest::{self, Manifest};
use chungus::net::{self, FetchStats};
use chungus::p2p;
use chungus::progress;
use chungus::registry::{self, Claim, Statement};
use chungus::sign;
use chungus::store::{self, Store};

/// `chungus --version` also names every format it speaks (see docs/formats.md). A test
/// keeps this in step with the constants.
const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\nmanifest: chungus/manifest/v2 (also reads v1)",
    "\nstore: v1, chunk blobs: v1",
    "\nwire: /chungus/1, /chungus/kad/1, HTTP /v1",
);

#[derive(Parser)]
#[command(version, long_version = LONG_VERSION, about = "Chunk, compress, deduplicate and share model files")]
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
    /// With several inputs, each is deduplicated against the ones before it.
    Bench {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Print the full report as JSON.
        #[arg(long)]
        json: bool,
        /// Chunk every file whole, without splitting safetensors and GGUF per tensor.
        #[arg(long)]
        whole_files: bool,
    },
    /// Share this store with peers on the LAN (read-only HTTP, advertised over mDNS).
    Serve {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Follow this registry's blocklist: refuse to hold or serve blocked models and
        /// chunks, and delete any already stored.
        #[arg(long)]
        blocklist: Option<String>,
        /// The registry operator's public key, to pin when following its blocklist.
        #[arg(long)]
        operator: Option<String>,
        #[arg(long, default_value_t = net::DEFAULT_PORT)]
        port: u16,
        /// Don't advertise over mDNS; peers must name this node with --peer.
        #[arg(long)]
        no_mdns: bool,
        /// Upload cap in MB/s, across all peers.
        #[arg(long)]
        max_upload: Option<f64>,
    },
    /// Run a local Hugging Face cache. Point tools at it with HF_ENDPOINT=http://localhost:8080.
    Hub {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Follow this registry's blocklist: refuse to hold or serve blocked models and
        /// chunks, and delete any already stored.
        #[arg(long)]
        blocklist: Option<String>,
        /// The registry operator's public key, to pin when following its blocklist.
        #[arg(long)]
        operator: Option<String>,
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
    /// Download a model by manifest root or registry name from peers.
    Fetch {
        /// A manifest root, or a registry name (org/model[@rev]). A name is resolved in the
        /// registry, and the download then requires its publisher's signature.
        root: String,
        /// Also unpack the files into this directory.
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[command(flatten)]
        from: FromArgs,
    },
    /// Mount a model as a read-only directory that works before the download finishes:
    /// reads fetch what they need, and the rest is prefetched in layer order. Linux only.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Mount {
        /// A manifest root, or a registry name (org/model[@rev]).
        model: String,
        /// An empty directory to mount on.
        dir: PathBuf,
        /// Chunks to prefetch at once; 0 fetches only what is read.
        #[arg(long, default_value_t = 16)]
        prefetch: usize,
        #[command(flatten)]
        from: FromArgs,
    },
    /// Join the internet swarm: announce this store's models on the DHT and serve them.
    Node {
        #[arg(long, env = "CHUNGUS_STORE", default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Follow this registry's blocklist and gates: refuse to hold or serve blocked
        /// models and chunks, delete any already stored, and serve gated models only to
        /// peers with an access ticket.
        #[arg(long, env = "CHUNGUS_BLOCKLIST")]
        blocklist: Option<String>,
        /// The registry operator's public key, to pin when following its blocklist.
        #[arg(long, env = "CHUNGUS_OPERATOR")]
        operator: Option<String>,
        /// Address to listen on. Repeatable. Default: TCP and QUIC on port 4001.
        #[arg(long, env = "CHUNGUS_LISTEN", value_delimiter = ',')]
        listen: Vec<Multiaddr>,
        /// A node to join through (a multiaddr ending in /p2p/<peer id>). Repeatable.
        /// Default: the project's public node.
        #[arg(long, env = "CHUNGUS_BOOTSTRAP", value_delimiter = ',')]
        bootstrap: Vec<Multiaddr>,
        /// A relay to be reachable through when behind NAT (ending in /p2p/<peer id>).
        /// Default: the project's public node, unless this node is --public, a
        /// --relay-server, has an --external address or is --download-only.
        #[arg(long, env = "CHUNGUS_RELAY", value_delimiter = ',')]
        relay: Vec<Multiaddr>,
        /// Don't join through or relay via the project's public node; use only
        /// --bootstrap, --relay and anchors (for a private swarm).
        #[arg(long, env = "CHUNGUS_NO_DEFAULT_BOOTSTRAP")]
        no_default_bootstrap: bool,
        /// An address others can reach this node at, e.g. a public IP with a forwarded port.
        #[arg(long, env = "CHUNGUS_EXTERNAL", value_delimiter = ',')]
        external: Vec<Multiaddr>,
        /// This node is directly reachable on its listen addresses (a public server).
        #[arg(long, env = "CHUNGUS_PUBLIC")]
        public: bool,
        /// Relay connections for nodes behind NAT. Use with --public.
        #[arg(long, env = "CHUNGUS_RELAY_SERVER")]
        relay_server: bool,
        /// A node to trust as a starting point: always kept in the routing table and asked
        /// directly for every model alongside the DHT. Repeatable.
        #[arg(long, env = "CHUNGUS_ANCHOR", value_delimiter = ',')]
        anchor: Vec<Multiaddr>,
        /// Use this registry's signed list of anchor nodes (checked against --operator).
        #[arg(long, env = "CHUNGUS_ANCHORS_FROM")]
        anchors_from: Option<String>,
        /// Upload cap in MB/s, across all peers.
        #[arg(long, env = "CHUNGUS_MAX_UPLOAD")]
        max_upload: Option<f64>,
        /// Open connections, in and out.
        #[arg(long, env = "CHUNGUS_MAX_CONNECTIONS", default_value_t = Limits::default().max_connections)]
        max_connections: u32,
        /// Requests one peer may have served at once; more are told to come back later.
        #[arg(long, env = "CHUNGUS_MAX_REQUESTS_PER_PEER", default_value_t = Limits::default().max_requests_per_peer)]
        max_requests_per_peer: usize,
        /// Requests served at once, across all peers.
        #[arg(long, env = "CHUNGUS_MAX_UPLOADS", default_value_t = Limits::default().max_uploads)]
        max_uploads: usize,
        /// Routing-table entries from one IPv4 /24 or IPv6 /48, so no one network can
        /// crowd out the rest.
        #[arg(long, env = "CHUNGUS_MAX_PEERS_PER_SUBNET", default_value_t = Limits::default().max_peers_per_subnet)]
        max_peers_per_subnet: usize,
        /// Download through the swarm but never serve or announce anything.
        #[arg(long, env = "CHUNGUS_DOWNLOAD_ONLY")]
        download_only: bool,
        /// With --relay-server: relayed connections open at once.
        #[arg(long, env = "CHUNGUS_RELAY_MAX_CIRCUITS", default_value_t = RelayLimits::default().max_circuits)]
        relay_max_circuits: usize,
        /// With --relay-server: MB one relayed connection may carry before it is closed.
        #[arg(long, env = "CHUNGUS_RELAY_CIRCUIT_MB", default_value_t = RelayLimits::default().circuit_bytes / 1_000_000)]
        relay_circuit_mb: u64,
        /// With --relay-server: seconds one relayed connection may stay open.
        #[arg(long, env = "CHUNGUS_RELAY_CIRCUIT_SECS", default_value_t = RelayLimits::default().circuit_duration.as_secs())]
        relay_circuit_secs: u64,
        /// With --relay-server: peers that may be reachable through this relay at once.
        #[arg(long, env = "CHUNGUS_RELAY_MAX_RESERVATIONS", default_value_t = RelayLimits::default().max_reservations)]
        relay_max_reservations: usize,
        /// List this node on the registry's leaderboard under this display name. It
        /// registers every 24 hours and is ranked only on what others can confirm:
        /// anchors' probes and downloaders' receipts.
        #[arg(long, env = "CHUNGUS_LEADERBOARD")]
        leaderboard: Option<String>,
        /// The registry to register with and to submit downloaders' receipts to.
        /// Default: the --blocklist registry. Without either, receipts are refused.
        #[arg(long, env = "CHUNGUS_REGISTRY")]
        registry: Option<String>,
        /// Probe the registry's listed nodes for the leaderboard, from a separate
        /// download-only identity. Only counts for the registry's anchor nodes.
        #[arg(long, env = "CHUNGUS_PROBE")]
        probe: bool,
    },
    /// Run a registry: model names, a signed append-only log of every change, and search.
    Registry {
        /// Where the log and the operator key live.
        #[arg(
            long,
            env = "CHUNGUS_REGISTRY_DATA",
            default_value = ".chungus/registry"
        )]
        data: PathBuf,
        #[arg(long, env = "CHUNGUS_REGISTRY_PORT", default_value_t = registry::DEFAULT_PORT)]
        port: u16,
        /// The registry is reachable only through a reverse proxy (like deploy/'s Caddy),
        /// so take each client's address, for counting downloads, from the last
        /// X-Forwarded-For entry. Never set this on a registry clients can reach directly.
        #[arg(long, env = "CHUNGUS_BEHIND_PROXY")]
        behind_proxy: bool,
    },
    /// Give a model in the store a name (org/model[@rev]) in the registry, signed by you.
    Publish {
        root: String,
        /// The name, e.g. acme/tiny-llama or acme/tiny-llama@v1 (the rev defaults to main).
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "")]
        description: String,
        /// The Hugging Face repo (org/name) whose license gate covers this model, e.g.
        /// meta-llama/Llama-3.2-1B. Peers then need their own accepted token to download
        /// it. Required for anything that is gated on Hugging Face.
        #[arg(long)]
        gated_by: Option<String>,
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Signing key (default: ~/.chungus/key).
        #[arg(long)]
        key: Option<PathBuf>,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Look up which manifest root a name points at, and who published it.
    Resolve {
        name: String,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Search the registry by name and description.
    Search {
        #[arg(required = true)]
        query: Vec<String>,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Let another key publish under an org you own (or, with --revoke, stop it).
    Grant {
        org: String,
        /// The public key (chungus1...) to add or remove.
        public_key: String,
        #[arg(long)]
        revoke: bool,
        #[arg(long)]
        key: Option<PathBuf>,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Download the registry's whole log and check every entry and the signed head.
    Audit {
        /// The registry operator's public key, to check the head against.
        #[arg(long)]
        operator: Option<String>,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Show the registry's anchor nodes, or replace them (operator only, with --key).
    Anchors {
        /// The new list: multiaddrs ending in /p2p/<peer id>. Empty with --key clears it.
        addrs: Vec<Multiaddr>,
        /// The registry's operator key (operator.key in its data directory).
        #[arg(long)]
        key: Option<PathBuf>,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Add a hash (a model's root, or a chunk) to the registry's blocklist. Operator only.
    Block {
        hash: String,
        #[arg(long, default_value = "")]
        reason: String,
        /// Remove it from the blocklist instead.
        #[arg(long)]
        unblock: bool,
        /// The registry's operator key (operator.key in its data directory).
        #[arg(long)]
        key: PathBuf,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Let a registry's online key act as the operator for a while, so the root key can
    /// stay offline (operator root key only). Run it again before it expires to renew, or
    /// with --revoke if the online key may be stolen.
    ///
    /// Moving an existing registry's root key offline: start the upgraded registry once
    /// (it prints its online key), delegate to that key from here, then move
    /// operator.key off the server and restart it.
    Delegate {
        /// The registry's online key (chungus1...), printed when the registry starts.
        online_key: Option<String>,
        /// How many days the delegation lasts.
        #[arg(long, default_value_t = registry::DEFAULT_DELEGATION_DAYS)]
        days: u64,
        /// End the current delegation now.
        #[arg(long, conflicts_with = "online_key")]
        revoke: bool,
        /// The operator's root key.
        #[arg(long)]
        key: PathBuf,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Put a model behind a Hugging Face repo's gate (or lift it with no repo). Operator
    /// only.
    Gate {
        root: String,
        /// The Hugging Face repo, e.g. meta-llama/Llama-3.2-1B. Omit to lift the gate.
        repo: Option<String>,
        /// The operator's root key, or the registry's online key (online.key) while
        /// delegated.
        #[arg(long)]
        key: PathBuf,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
    },
    /// Hide a node's display name on the leaderboard, keeping its numbers (or show it
    /// again with --unhide). Operator only.
    HideNode {
        /// The node's peer id.
        peer: String,
        #[arg(long)]
        unhide: bool,
        /// The operator's root key, or the registry's online key (online.key) while
        /// delegated.
        #[arg(long)]
        key: PathBuf,
        #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
        registry: String,
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

fn mb_per_sec(mb: f64) -> Result<u64> {
    if !(mb > 0.0 && mb.is_finite()) {
        bail!("an upload cap must be a positive number of MB/s");
    }
    Ok((mb * 1e6) as u64)
}

fn follow_blocklist(
    store: &Arc<Store>,
    url: Option<String>,
    operator: Option<String>,
) -> Result<()> {
    if let Some(url) = url {
        let follower = registry::Follower::new(&url, operator)?;
        registry::follow(store.clone(), follower, Duration::from_secs(60));
        println!("following the blocklist and gates of {url}");
    }
    Ok(())
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

/// The user's Hugging Face token, found where `huggingface_hub` looks: `HF_TOKEN`, then
/// the file `huggingface-cli login` writes.
fn hf_token() -> Option<String> {
    let non_empty = |s: String| {
        let s = s.trim().to_string();
        (!s.is_empty()).then_some(s)
    };
    if let Some(t) = std::env::var("HF_TOKEN").ok().and_then(non_empty) {
        return Some(t);
    }
    let path = match (
        std::env::var_os("HF_TOKEN_PATH"),
        std::env::var_os("HF_HOME"),
    ) {
        (Some(p), _) => PathBuf::from(p),
        (None, Some(home)) => PathBuf::from(home).join("token"),
        (None, None) => PathBuf::from(std::env::var_os("HOME")?).join(".cache/huggingface/token"),
    };
    fs::read_to_string(path).ok().and_then(non_empty)
}

/// Where to get a model from, shared by `fetch` and `mount`.
#[derive(clap::Args)]
struct FromArgs {
    #[arg(long, env = "CHUNGUS_REGISTRY", default_value = registry::DEFAULT_URL)]
    registry: String,
    #[arg(long, default_value = DEFAULT_STORE)]
    store: PathBuf,
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
    #[arg(long, env = "CHUNGUS_BOOTSTRAP", value_delimiter = ',')]
    bootstrap: Vec<Multiaddr>,
    /// Fetch over the internet swarm, joining through the project's public node (or
    /// --bootstrap) and the registry's signed anchor nodes.
    #[arg(long)]
    swarm: bool,
    /// With --swarm, don't join through the project's public node.
    #[arg(long, env = "CHUNGUS_NO_DEFAULT_BOOTSTRAP")]
    no_default_bootstrap: bool,
    /// Don't sign receipts for the peers that served this download. Receipts credit
    /// them on the registry's leaderboard and carry nothing but this download's one-run
    /// peer id.
    #[arg(long, env = "CHUNGUS_NO_RECEIPTS")]
    no_receipts: bool,
}

impl FromArgs {
    fn over_swarm(&self) -> bool {
        self.swarm || !self.bootstrap.is_empty()
    }

    /// What to show peers of a gated model: a ticket from the registry, got with the
    /// user's Hugging Face token (used only if a peer says a model is gated).
    fn access(&self) -> Option<p2p::Access> {
        hf_token().map(|hf_token| p2p::Access {
            registry: self.registry.clone(),
            hf_token,
        })
    }

    /// The one-run identity for a swarm download, made before the name lookup so the
    /// lookup can be bound to it (see [`FromArgs::resolve`]).
    fn swarm_key(&self) -> Option<libp2p::identity::Keypair> {
        self.over_swarm()
            .then(libp2p::identity::Keypair::generate_ed25519)
    }

    /// The manifest root for `model` (a root or a registry name) and the keys that must
    /// have signed it. A name trusts its publisher, and syncs the registry's blocklist.
    /// The lookup counts as a download; unless `--no-receipts`, it is bound to `key`, the
    /// swarm download's identity, so the receipts it signs are credited. Returns whether
    /// it was.
    async fn resolve(
        &self,
        model: &str,
        store: &Store,
        key: Option<&libp2p::identity::Keypair>,
    ) -> Result<(String, Vec<ed25519_dalek::VerifyingKey>, bool)> {
        let mut trust = parse_keys(&self.trust)?;
        if store::is_hash(model) {
            return Ok((model.to_string(), trust, false));
        }
        let (name, rev) = registry::parse_ref(model)?;
        let peer = key
            .filter(|_| !self.no_receipts)
            .map(|k| k.public().to_peer_id().to_string());
        let entry = registry::Client::new(&self.registry)?
            .resolve_download(&name, &rev, peer.as_deref())
            .await?;
        let Claim::Publish { root, .. } = entry.statement.claim else {
            unreachable!("resolve returns publishes")
        };
        println!(
            "{name}@{rev} is {root}, published by {}",
            entry.statement.signature.key
        );
        trust.push(sign::parse_public_key(&entry.statement.signature.key)?);
        registry::Follower::new(&self.registry, None)?
            .sync(store)
            .await?;
        Ok((root, trust, peer.is_some()))
    }

    /// `--peer`s plus, unless `--no-mdns`, peers found on the LAN.
    async fn lan_peers(&self) -> Result<Vec<String>> {
        let mut peers = self.peer.clone();
        if !self.no_mdns {
            let wait = Duration::from_secs_f64(self.discover_secs);
            let found = tokio::task::spawn_blocking(move || net::discover(wait, None)).await??;
            println!("found {} peer(s) on the LAN", found.len());
            peers.extend(found);
        }
        peers.sort();
        peers.dedup();
        Ok(peers)
    }

    #[cfg(target_os = "linux")]
    fn origin(&self) -> Vec<String> {
        self.origin.iter().cloned().collect()
    }

    /// A download-only swarm node joined through `--bootstrap` and, with `--swarm`, the
    /// project's public node and the registry's anchors.
    async fn swarm_node(
        &self,
        store: &Arc<Store>,
        key: libp2p::identity::Keypair,
    ) -> Result<p2p::Node> {
        let mut bootstrap = self.bootstrap.clone();
        let mut anchors = Vec::new();
        if self.swarm {
            if bootstrap.is_empty() && !self.no_default_bootstrap {
                bootstrap = p2p::default_bootstrap();
            }
            // The public node is enough to join, so a registry that can't be reached
            // only costs the anchors.
            match registry::Client::new(&self.registry)?.anchors(None).await {
                Ok(found) => anchors = found,
                Err(e) if !bootstrap.is_empty() => {
                    eprintln!("no anchor nodes from {}: {e:#}", self.registry)
                }
                Err(e) => return Err(e),
            }
            if anchors.is_empty() && bootstrap.is_empty() {
                bail!(
                    "{} lists no anchor nodes; join with --bootstrap",
                    self.registry
                );
            }
        }
        // `key` is a fresh identity: a one-off download needs none of its own, and reusing
        // the store's would clash with a node serving the same store (and skip it as
        // "self").
        let config = p2p::Config {
            listen: vec![
                "/ip4/0.0.0.0/tcp/0".parse()?,
                "/ip4/0.0.0.0/udp/0/quic-v1".parse()?,
            ],
            bootstrap,
            anchors,
            // A one-off download leaves before it could usefully serve anyone.
            limits: Limits {
                download_only: true,
                ..Default::default()
            },
            ..Default::default()
        };
        p2p::Node::start(store.clone(), key, config).await
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
    manifest::parse(&fs::read(arg).with_context(|| format!("read {arg}"))?)
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
            for why in &s.skipped {
                eprintln!("skipped {why}");
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
        Cmd::Bench {
            inputs,
            json,
            whole_files,
        } => {
            let r = chungus::bench(&inputs, whole_files)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r)?);
                return Ok(());
            }
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
            if r.by_dtype.len() > 1 {
                println!("by dtype (zstd only / transform + zstd, % of raw):");
                for (dtype, d) in &r.by_dtype {
                    println!(
                        "  {dtype:<8} {:>12.1} MB  {:>6.1}%  {:>6.1}%",
                        mb(d.raw_bytes),
                        pct(d.zstd_bytes, d.raw_bytes),
                        pct(d.encoded_bytes, d.raw_bytes)
                    );
                }
            }
            if r.inputs.len() > 1 {
                println!("per input, deduplicated against the inputs before it:");
                for i in &r.inputs {
                    println!(
                        "  {:>10.1} MB raw, {:>5.1}% new, {:>10.1} MB to fetch  {}",
                        mb(i.raw_bytes),
                        pct(i.new_raw_bytes, i.raw_bytes),
                        mb(i.new_stored_bytes),
                        i.path
                    );
                }
            }
            println!(
                "MB/s of raw bytes: chunk {:.0} (one core), hash + encode {:.0}, decode {:.0} (all cores)",
                mb(r.raw_bytes) / r.chunk_secs.max(1e-9),
                mb(r.raw_bytes) / r.encode_secs.max(1e-9),
                mb(r.raw_bytes) / r.decode_secs.max(1e-9)
            );
        }
        Cmd::Serve {
            store,
            blocklist,
            operator,
            port,
            no_mdns,
            max_upload,
        } => {
            let store = Arc::new(Store::open(&store)?);
            follow_blocklist(&store, blocklist, operator)?;
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
            let mut router = net::router(store);
            if let Some(mbps) = max_upload {
                router = net::rate_limited(router, Arc::new(RateLimiter::new(mb_per_sec(mbps)?)));
            }
            axum::serve(listener, router).await?;
        }
        Cmd::Hub {
            store,
            blocklist,
            operator,
            port,
            upstream,
            offline,
            peer,
            no_mdns,
        } => {
            let store = Arc::new(Store::open(&store)?);
            follow_blocklist(&store, blocklist, operator)?;
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
        Cmd::Fetch { root, output, from } => {
            let store = Arc::new(Store::open(&from.store)?);
            let key = from.swarm_key();
            let (root, trust, bound) = from.resolve(&root, &store, key.as_ref()).await?;
            let bar = progress::Bar::new();
            let label = format!("fetching {}", &root[..12.min(root.len())]);
            let result = if let Some(key) = key {
                let node = from.swarm_node(&store, key).await?;
                let display = progress::show(label, bar.progress());
                let fetch = p2p::fetch(&node, &root, store.clone(), &trust, from.access());
                let r = progress::track(bar, fetch).await;
                progress::done(display, &r).await;
                if bound && let Ok((_, s)) = &r {
                    p2p::send_receipts(&node, &root, &p2p::served_by(s)).await;
                }
                r
            } else {
                let peers = from.lan_peers().await?;
                let display = progress::show(label, bar.progress());
                let origin = from.origin.as_deref();
                let fetch = net::fetch(&root, store.clone(), &peers, origin, &trust);
                let r = progress::track(bar, fetch).await;
                progress::done(display, &r).await;
                r
            };
            let (manifest, s) = result?;
            report_fetch(&s);
            if let Some(output) = output {
                tokio::task::spawn_blocking(move || chungus::unpack(&manifest, &store, &output))
                    .await??;
                println!("unpacked, all chunks verified");
            }
        }
        #[cfg(target_os = "linux")]
        Cmd::Mount {
            model,
            dir,
            prefetch,
            from,
        } => {
            let store = Arc::new(Store::open(&from.store)?);
            let key = from.swarm_key();
            let (root, trust, bound) = from.resolve(&model, &store, key.as_ref()).await?;
            // The swarm's sources, when receipts are to be sent for what they serve.
            let mut receipts: Option<Arc<p2p::ModelSources>> = None;
            let (manifest, source): (Manifest, Arc<dyn chungus::lazy::ChunkSource>) =
                if let Some(key) = key {
                    let node = from.swarm_node(&store, key).await?;
                    let (m, s) = p2p::prepare(&node, &root, &store, &trust, from.access()).await?;
                    let s = Arc::new(s);
                    if bound {
                        receipts = Some(s.clone());
                    }
                    (m, s)
                } else {
                    let client = net::client()?;
                    let peers = from.lan_peers().await?;
                    let origin = from.origin();
                    let m = net::prepare(&client, &root, &store, &peers, &origin, &trust).await?;
                    (
                        m,
                        Arc::new(net::HttpSource {
                            client,
                            peers,
                            origin,
                        }),
                    )
                };
            let lazy = chungus::lazy::Lazy::new(store, manifest, source)?;
            let mounted =
                chungus::mount::mount(lazy.clone(), &dir, tokio::runtime::Handle::current())?;
            println!(
                "mounted {root} at {} ({:.1} MB, {:.1} MB already local); Ctrl-C to unmount",
                dir.display(),
                mb(lazy.total_bytes()),
                mb(lazy
                    .stats
                    .local_bytes
                    .load(std::sync::atomic::Ordering::Relaxed)),
            );
            let prefetching = {
                let lazy = lazy.clone();
                async move {
                    if prefetch == 0 {
                        return std::future::pending().await;
                    }
                    let started = std::time::Instant::now();
                    let display = progress::show("prefetching", {
                        let lazy = lazy.clone();
                        move || {
                            let local = lazy
                                .stats
                                .local_bytes
                                .load(std::sync::atomic::Ordering::Relaxed);
                            (local, lazy.total_bytes())
                        }
                    });
                    let ticker = async {
                        loop {
                            tokio::time::sleep(Duration::from_secs(5)).await;
                            if display.is_some() {
                                continue;
                            }
                            let local = lazy
                                .stats
                                .local_bytes
                                .load(std::sync::atomic::Ordering::Relaxed);
                            println!(
                                "  {:.0}% local ({:.1} of {:.1} MB)",
                                pct(local, lazy.total_bytes()),
                                mb(local),
                                mb(lazy.total_bytes())
                            );
                        }
                    };
                    let r = tokio::select! {
                        r = lazy.prefetch(prefetch) => r,
                        _ = ticker => unreachable!(),
                    };
                    progress::done(display, &r).await;
                    match r {
                        Ok(()) => println!(
                            "prefetch done in {:.1}s: the whole model is local",
                            started.elapsed().as_secs_f64()
                        ),
                        Err(e) => eprintln!("prefetch stopped: {e:#}"),
                    }
                    std::future::pending::<()>().await
                }
            };
            // Receipts every few minutes while prefetching, since a mount can run for days.
            let every_few_minutes = {
                let receipts = receipts.clone();
                async move {
                    let Some(sources) = receipts else {
                        return std::future::pending().await;
                    };
                    loop {
                        tokio::time::sleep(Duration::from_secs(300)).await;
                        sources.send_receipts().await;
                    }
                }
            };
            tokio::select! {
                _ = prefetching => {}
                _ = every_few_minutes => {}
                r = tokio::signal::ctrl_c() => r?,
            }
            mounted.unmount()?;
            println!("unmounted");
            if let Some(sources) = receipts {
                sources.send_receipts().await;
            }
        }
        #[cfg(not(target_os = "linux"))]
        Cmd::Mount { .. } => bail!(
            "mounting only works on Linux so far; download the whole model with \
             `chungus fetch <model> -o <dir>` instead"
        ),
        Cmd::Node {
            store,
            blocklist,
            operator,
            listen,
            bootstrap,
            relay,
            no_default_bootstrap,
            external,
            public,
            relay_server,
            mut anchor,
            anchors_from,
            max_upload,
            max_connections,
            max_requests_per_peer,
            max_uploads,
            max_peers_per_subnet,
            download_only,
            relay_max_circuits,
            relay_circuit_mb,
            relay_circuit_secs,
            relay_max_reservations,
            leaderboard,
            registry,
            probe,
        } => {
            let store = Arc::new(Store::open(&store)?);
            let registry = registry.or_else(|| blocklist.clone());
            if let Some(url) = anchors_from {
                let found = registry::Client::new(&url)?
                    .anchors(operator.as_deref())
                    .await?;
                println!("{} anchor node(s) from {url}", found.len());
                anchor.extend(found);
            }
            follow_blocklist(&store, blocklist, operator)?;
            let key = p2p::load_or_create_identity(&store.dir().join("node.key"))?;
            if let Some(name) = &leaderboard
                && !chungus::leaderboard::valid_display_name(name)
            {
                bail!(
                    "--leaderboard takes a display name of 1 to {} printable characters",
                    chungus::leaderboard::MAX_NAME
                );
            }
            if (leaderboard.is_some() || probe) && registry.is_none() {
                bail!("--leaderboard and --probe need a --registry (or --blocklist)");
            }
            // Receipts downloaders hand this node go to the registry, and are kept as
            // proof of the node's work.
            let receipts = registry.clone().map(|url| {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                chungus::leaderboard::forward_receipts(url, rx, store.dir().join("receipts.json"));
                tx
            });
            let listen = if listen.is_empty() {
                vec![
                    format!("/ip4/0.0.0.0/tcp/{}", p2p::DEFAULT_PORT).parse()?,
                    format!("/ip4/0.0.0.0/udp/{}/quic-v1", p2p::DEFAULT_PORT).parse()?,
                ]
            } else {
                listen
            };
            let defaults = !no_default_bootstrap;
            let bootstrap = if bootstrap.is_empty() && defaults {
                p2p::default_bootstrap()
            } else {
                bootstrap
            };
            // A node that others can reach directly, or that never serves, needs no relay.
            let reachable = public || relay_server || !external.is_empty() || download_only;
            let relay = if relay.is_empty() && defaults && !reachable {
                p2p::default_relays()
            } else {
                relay
            };
            // Probes go out from a separate download-only identity, so a node can't tell
            // them from any other download and answer only those.
            let prober = if probe {
                let config = p2p::Config {
                    listen: vec![
                        "/ip4/0.0.0.0/tcp/0".parse()?,
                        "/ip4/0.0.0.0/udp/0/quic-v1".parse()?,
                    ],
                    bootstrap: bootstrap.clone(),
                    anchors: anchor.clone(),
                    limits: Limits {
                        download_only: true,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let fresh = libp2p::identity::Keypair::generate_ed25519();
                Some(p2p::Node::start(store.clone(), fresh, config).await?)
            } else {
                None
            };
            let config = p2p::Config {
                listen,
                bootstrap,
                relays: relay,
                external,
                public,
                relay_server,
                anchors: anchor,
                receipts,
                limits: Limits {
                    upload_bytes_per_sec: max_upload.map(mb_per_sec).transpose()?,
                    max_connections,
                    max_requests_per_peer,
                    max_uploads,
                    max_peers_per_subnet,
                    download_only,
                },
                relay: RelayLimits {
                    max_circuits: relay_max_circuits,
                    circuit_bytes: relay_circuit_mb.saturating_mul(1_000_000),
                    circuit_duration: Duration::from_secs(relay_circuit_secs),
                    max_reservations: relay_max_reservations,
                    ..Default::default()
                },
                log: true,
                ..Default::default()
            };
            let node = p2p::Node::start(store.clone(), key.clone(), config).await?;
            println!("peer id {}", node.peer_id);
            if let (Some(name), Some(url)) = (leaderboard, &registry) {
                chungus::leaderboard::register_loop(url.clone(), key.clone(), name, store.clone());
            }
            if let (Some(prober), Some(url)) = (prober, &registry) {
                println!("probing {url}'s listed nodes as {}", prober.peer_id);
                chungus::leaderboard::probe_loop(url.clone(), key, prober);
            }
            if download_only {
                println!("download-only: serving and announcing nothing");
            } else {
                println!("sharing {} models", store.manifests()?.len());
            }
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
        Cmd::Registry {
            data,
            port,
            behind_proxy,
        } => {
            let reg = Arc::new(registry::Registry::open(&data)?.with_behind_proxy(behind_proxy));
            let addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("bind {addr}"))?;
            println!("registry on http://localhost:{port}");
            println!("operator key {}", reg.operator());
            println!("{} entries in the log", reg.head().size);
            let online = reg.online_public();
            match reg.with_log(|log| log.delegate().map(|(k, e)| (k == online, e))) {
                Some((true, expires)) => {
                    let days = expires.saturating_sub(registry::now()) / 86400;
                    println!("signing with online key {online}, delegated for {days} more day(s)");
                    if days < 14 {
                        println!("renew it soon: chungus delegate {online} --key <root key>");
                    }
                }
                _ if reg.has_root_key() => println!(
                    "signing with the root key; to keep it offline, delegate to online key \
                     {online} (see `chungus delegate --help`)"
                ),
                _ => println!(
                    "online key {online} is not delegated, so clients will reject this \
                     registry's head. From the machine with the root key, run:\n  \
                     chungus delegate {online} --key <root key> --registry <this registry>"
                ),
            }
            registry::serve(listener, reg).await?;
        }
        Cmd::Publish {
            root,
            name,
            description,
            gated_by,
            store,
            key,
            registry,
        } => {
            let (name, rev) = registry::parse_ref(&name)?;
            let store = Store::open(&store)?;
            let manifest = store
                .get_manifest_bytes(&root)
                .context("publish a model that is in your store (see `chungus list`)")?;
            let k = sign::load_key(&key_path(key)?)?;
            // Sign the manifest too, so peers can prove it came from the name's owner.
            store.add_signatures(&root, &[sign::sign(&k, &root)])?;
            let st = Statement::new(
                &k,
                Claim::Publish {
                    name: name.clone(),
                    rev: rev.clone(),
                    root: root.clone(),
                    description,
                    gated: gated_by,
                },
            );
            let m = chungus::manifest::parse(&manifest)?;
            let headers = chungus::safetensors_headers(&m, &store)?;
            let gguf_headers = chungus::gguf_headers(&m, &store)?;
            let entry = registry::Client::new(&registry)?
                .publish_with_headers(&st, &manifest, headers, gguf_headers)
                .await?;
            println!("published {name}@{rev} -> {root} (log entry {})", entry.seq);
        }
        Cmd::Resolve { name, registry } => {
            let (name, rev) = registry::parse_ref(&name)?;
            let entry = registry::Client::new(&registry)?
                .resolve(&name, &rev)
                .await?;
            if let Claim::Publish {
                root, description, ..
            } = &entry.statement.claim
            {
                println!("root {root}");
                println!("publisher {}", entry.statement.signature.key);
                println!("log entry {}", entry.seq);
                if !description.is_empty() {
                    println!("{description}");
                }
            }
        }
        Cmd::Search { query, registry } => {
            let hits = registry::Client::new(&registry)?
                .search(&query.join(" "))
                .await?;
            if hits.is_empty() {
                println!("no matches");
            }
            for h in hits {
                println!("{}@{}  {}", h.name, h.rev, h.root);
                if !h.description.is_empty() {
                    println!("    {}", h.description);
                }
            }
        }
        Cmd::Grant {
            org,
            public_key,
            revoke,
            key,
            registry,
        } => {
            sign::parse_public_key(&public_key)?;
            let k = sign::load_key(&key_path(key)?)?;
            let claim = if revoke {
                Claim::Revoke {
                    org: org.clone(),
                    key: public_key.clone(),
                }
            } else {
                Claim::Grant {
                    org: org.clone(),
                    key: public_key.clone(),
                }
            };
            let entry = registry::Client::new(&registry)?
                .submit(&Statement::new(&k, claim))
                .await?;
            let verb = if revoke { "revoked" } else { "granted" };
            println!("{verb} {public_key} on {org} (log entry {})", entry.seq);
        }
        Cmd::Audit { operator, registry } => {
            let (log, head) = registry::Client::new(&registry)?
                .audit(operator.as_deref())
                .await?;
            println!(
                "{} entries verified; head signed by {}",
                log.entries.len(),
                head.signature.key
            );
            if operator.is_none() {
                println!(
                    "pass --operator {} to pin this registry",
                    head.signature.key
                );
            }
        }
        Cmd::Delegate {
            online_key,
            days,
            revoke,
            key,
            registry,
        } => {
            let k = sign::load_key(&key)?;
            let claim = match (online_key, revoke) {
                (_, true) => Claim::Delegate {
                    key: String::new(),
                    expires: 0,
                },
                (Some(online), false) => Claim::Delegate {
                    key: online,
                    expires: registry::now() + days * 86400,
                },
                (None, false) => bail!("name the registry's online key, or pass --revoke"),
            };
            let entry = registry::Client::new(&registry)?
                .submit(&Statement::new(&k, claim.clone()))
                .await?;
            match claim {
                Claim::Delegate { key, .. } if !key.is_empty() => {
                    println!(
                        "{key} acts as the operator for {days} day(s) (log entry {})",
                        entry.seq
                    )
                }
                _ => println!("revoked the online key (log entry {})", entry.seq),
            }
        }
        Cmd::Block {
            hash,
            reason,
            unblock,
            key,
            registry,
        } => {
            let k = sign::load_key(&key)?;
            let claim = if unblock {
                Claim::Unblock { hash: hash.clone() }
            } else {
                Claim::Block {
                    hash: hash.clone(),
                    reason,
                }
            };
            let entry = registry::Client::new(&registry)?
                .submit(&Statement::new(&k, claim))
                .await?;
            let verb = if unblock { "unblocked" } else { "blocked" };
            println!("{verb} {hash} (log entry {})", entry.seq);
        }
        Cmd::Gate {
            root,
            repo,
            key,
            registry,
        } => {
            let k = sign::load_key(&key)?;
            let claim = Claim::Gate {
                root: root.clone(),
                repo: repo.clone().unwrap_or_default(),
            };
            let entry = registry::Client::new(&registry)?
                .submit(&Statement::new(&k, claim))
                .await?;
            match repo {
                Some(r) => println!("{root} is gated by {r} (log entry {})", entry.seq),
                None => println!("lifted the gate on {root} (log entry {})", entry.seq),
            }
        }
        Cmd::Anchors {
            addrs,
            key,
            registry,
        } => {
            let client = registry::Client::new(&registry)?;
            if let Some(key) = key {
                let k = sign::load_key(&key)?;
                let claim = Claim::Anchors {
                    addrs: addrs.iter().map(|a| a.to_string()).collect(),
                };
                let entry = client.submit(&Statement::new(&k, claim)).await?;
                println!("set {} anchor(s) (log entry {})", addrs.len(), entry.seq);
            } else if !addrs.is_empty() {
                bail!("setting the anchors needs the operator's --key");
            } else {
                for a in client.anchors(None).await? {
                    println!("{a}");
                }
            }
        }
        Cmd::HideNode {
            peer,
            unhide,
            key,
            registry,
        } => {
            let k = sign::load_key(&key)?;
            registry::Client::new(&registry)?
                .hide(&chungus::leaderboard::Hide::new(&k, &peer, !unhide))
                .await?;
            let verb = if unhide { "showed" } else { "hid" };
            println!("{verb} {peer}'s name on the leaderboard");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_matches_formats() {
        assert!(LONG_VERSION.contains(manifest::FORMAT));
        assert!(LONG_VERSION.contains(&format!("store: v{}", store::STORE_VERSION)));
        assert!(LONG_VERSION.contains(&format!("chunk blobs: v{}", store::BLOB_VERSION)));
        assert!(LONG_VERSION.contains(p2p::PROTOCOL.as_ref()));
        assert!(LONG_VERSION.contains(p2p::KAD_PROTOCOL.as_ref()));
    }
}
