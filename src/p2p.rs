//! Sharing models over the internet with libp2p.
//!
//! Every node joins a Kademlia DHT and announces itself as a provider of each model
//! (manifest root) in its store. To fetch a model, a node asks the DHT who provides it,
//! then requests the manifest, its signatures and the chunks from those providers over
//! `/chungus/1`, spreading chunks across them and verifying each one against its hash.
//!
//! Models the registry marks as gated (see [`crate::registry::AccessTicket`]) are only
//! served to peers that first present a ticket for the gate's Hugging Face repo. A node
//! learns which models are gated, and whose tickets to accept, by following a registry.
//!
//! Connections are encrypted (Noise over TCP, TLS inside QUIC). A node behind NAT listens
//! through a relay (circuit relay v2) so others can reach it, and DCUtR then tries to
//! upgrade relayed connections to direct ones by hole punching. Any public node can offer
//! to be a relay with `--relay-server`.

use anyhow::{Context, Result, anyhow, bail};
use axum::body::Bytes;
use futures::StreamExt;
use libp2p::identity::Keypair;
use libp2p::kad::{self, store::MemoryStore};
use libp2p::multiaddr::Protocol;
use libp2p::request_response::{self, OutboundRequestId, ProtocolSupport, ResponseChannel};
use libp2p::swarm::{NetworkBehaviour, SwarmEvent, behaviour::toggle::Toggle};
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, Swarm, connection_limits, dcutr, identify, noise, ping,
    relay, tcp, yamux,
};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

use crate::leaderboard::Receipt;
use crate::limits::{Limits, RateLimiter, SubnetCaps};
use crate::manifest::{BLOCK_BYTES, Manifest};
use crate::net::{self, FetchStats};
use crate::registry::{self, AccessTicket};
use crate::sign::{self, Signature};
use crate::store::{self, Store};

pub const PROTOCOL: StreamProtocol = StreamProtocol::new("/chungus/1");
pub const KAD_PROTOCOL: StreamProtocol = StreamProtocol::new("/chungus/kad/1");
pub const DEFAULT_PORT: u16 = 4001;
/// The project's public node. Nodes and swarm fetches join through it unless told
/// otherwise, and nodes behind NAT use it as their relay.
const DEFAULT_NODE: &str = "12D3KooWRaVx8DKtusbdVeThtaFxqR7C8jgSvz6fArBwh52SCAeR";
const DEFAULT_NODE_ADDRS: &[&str] = &[
    "/ip4/40.160.91.185/tcp/4001",
    "/ip4/40.160.91.185/udp/4001/quic-v1",
];
/// Largest response a peer may send: a manifest of a very large model, or one chunk.
const MAX_RESPONSE: u64 = 256 << 20;
/// How often a node looks for newly added models to announce.
const RESCAN: Duration = Duration::from_secs(60);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Request {
    Manifest(String),
    Signatures(String),
    Chunk(String),
    /// Present an access ticket, so later chunk requests for its repo are served. Added
    /// after the first release; older nodes don't gate anything and fail the request.
    Access(AccessTicket),
    /// Hand the node a downloader's receipt for what it served, to submit to the
    /// registry (see [`crate::leaderboard`]). Older nodes fail the request.
    Receipt(Receipt),
    /// Which manifest root indexes this Ollama manifest digest (`sha256:<hex>`). Added
    /// after the first release; older nodes fail the request.
    Ollama(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Response {
    Manifest(Option<ByteBuf>),
    Signatures(Vec<Signature>),
    Chunk(Option<ByteBuf>),
    /// The node is serving as much as its limits allow; try another peer.
    Busy,
    /// The chunk belongs to a model behind this Hugging Face repo's gate: present an
    /// access ticket first.
    Gated(String),
    /// Whether an access ticket was accepted.
    Granted(bool),
    /// Whether a receipt was taken.
    Receipt(bool),
    /// The manifest root for an Ollama manifest digest, if this node has one.
    Root(Option<String>),
}

#[derive(NetworkBehaviour)]
struct Behaviour {
    kad: kad::Behaviour<MemoryStore>,
    identify: identify::Behaviour,
    ping: ping::Behaviour,
    rr: request_response::cbor::Behaviour<Request, Response>,
    relay_client: relay::client::Behaviour,
    relay: Toggle<relay::Behaviour>,
    dcutr: dcutr::Behaviour,
    limits: connection_limits::Behaviour,
}

/// How a node joins the network.
#[derive(Default, Clone)]
pub struct Config {
    /// Addresses to listen on, e.g. `/ip4/0.0.0.0/tcp/4001`.
    pub listen: Vec<Multiaddr>,
    /// Known nodes to join through, each ending in `/p2p/<peer id>`.
    pub bootstrap: Vec<Multiaddr>,
    /// Relays to listen through when this node can't be reached directly.
    pub relays: Vec<Multiaddr>,
    /// Addresses others can reach this node at (e.g. a public IP with a forwarded port).
    pub external: Vec<Multiaddr>,
    /// Treat every listen address as reachable. For public servers and tests.
    pub public: bool,
    /// Relay traffic for nodes behind NAT.
    pub relay_server: bool,
    /// Raw bytes per announced block. Leave at 0 for [`BLOCK_BYTES`]; every node in a
    /// swarm must use the same value, so change it only in tests.
    pub block_bytes: u64,
    /// Nodes to trust as starting points, usually the registry's signed list: always kept
    /// in the routing table, exempt from connection and subnet limits, and asked directly
    /// during every fetch alongside the DHT.
    pub anchors: Vec<Multiaddr>,
    pub limits: Limits,
    /// Caps on relaying, when `relay_server` is set.
    pub relay: crate::limits::RelayLimits,
    /// Print connections, bootstrap progress and relay reservations as they happen.
    pub log: bool,
    /// Where to send receipts downloaders hand this node. Without it they are refused.
    pub receipts: Option<mpsc::UnboundedSender<Receipt>>,
    /// Serve counters as Prometheus text at `http://<addr>/metrics`.
    pub metrics: Option<SocketAddr>,
}

/// What a node has been doing since it started. Counters only grow; gauges are the
/// current value.
#[derive(Default)]
pub struct Metrics {
    /// Bytes sent to peers in responses.
    pub bytes_served: AtomicU64,
    /// Requests peers have sent us.
    pub requests: AtomicU64,
    /// Requests answered "busy" because of the upload limits.
    pub busy: AtomicU64,
    /// Peers connected right now (gauge).
    pub peers: AtomicU64,
    /// Connections this node is relaying right now, as a relay server (gauge).
    pub relay_circuits: AtomicU64,
    /// Connections this node has agreed to relay.
    pub relay_circuits_accepted: AtomicU64,
}

impl Metrics {
    /// The Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (name, kind, help, v) in [
            (
                "bytes_served_total",
                "counter",
                "Bytes sent to peers in responses.",
                &self.bytes_served,
            ),
            (
                "requests_total",
                "counter",
                "Requests received from peers.",
                &self.requests,
            ),
            (
                "busy_total",
                "counter",
                "Requests answered busy because of upload limits.",
                &self.busy,
            ),
            ("peers", "gauge", "Peers connected now.", &self.peers),
            (
                "relay_circuits",
                "gauge",
                "Connections being relayed now.",
                &self.relay_circuits,
            ),
            (
                "relay_circuits_total",
                "counter",
                "Connections accepted for relaying.",
                &self.relay_circuits_accepted,
            ),
        ] {
            let v = v.load(Ordering::Relaxed);
            out.push_str(&format!(
                "# HELP chungus_{name} {help}\n# TYPE chungus_{name} {kind}\nchungus_{name} {v}\n"
            ));
        }
        out
    }
}

fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

/// Serve `metrics` at `http://<addr>/metrics` until the process exits.
async fn serve_metrics(addr: SocketAddr, metrics: Arc<Metrics>) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind metrics on {addr}"))?;
    let app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || async move {
            (
                [(
                    axum::http::header::CONTENT_TYPE,
                    "text/plain; version=0.0.4; charset=utf-8",
                )],
                metrics.render(),
            )
        }),
    );
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(())
}

enum Command {
    Providers(String, oneshot::Sender<Vec<PeerId>>),
    Request(PeerId, Request, oneshot::Sender<Result<Response>>),
    Announce,
    Addresses(oneshot::Sender<Vec<Multiaddr>>),
    Find(PeerId, oneshot::Sender<()>),
}

/// A running node. Cheap to clone; the swarm runs in a background task until every
/// handle is dropped.
#[derive(Clone)]
pub struct Node {
    tx: mpsc::UnboundedSender<Command>,
    pub peer_id: PeerId,
    block_bytes: u64,
    anchors: Vec<PeerId>,
    key: Keypair,
    pub metrics: Arc<Metrics>,
}

/// Load this node's identity from `path`, creating it on first use.
pub fn load_or_create_identity(path: &Path) -> Result<Keypair> {
    if let Ok(bytes) = fs::read(path) {
        return Keypair::from_protobuf_encoding(&bytes).context("node identity file is corrupt");
    }
    let key = Keypair::generate_ed25519();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    use std::io::Write;
    opts.open(path)
        .with_context(|| format!("create {}", path.display()))?
        .write_all(&key.to_protobuf_encoding()?)?;
    Ok(key)
}

fn peer_of(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|p| match p {
        Protocol::P2p(id) => Some(id),
        _ => None,
    })
}

/// Where to join the swarm when no bootstrap node is given: the project's public node,
/// over TCP and QUIC.
pub fn default_bootstrap() -> Vec<Multiaddr> {
    DEFAULT_NODE_ADDRS
        .iter()
        .map(|a| {
            format!("{a}/p2p/{DEFAULT_NODE}")
                .parse()
                .expect("valid multiaddr")
        })
        .collect()
}

/// The relay to use behind NAT when none is given: the project's public node, over TCP.
pub fn default_relays() -> Vec<Multiaddr> {
    default_bootstrap().into_iter().take(1).collect()
}

/// `addrs` without those of `me`, so a node never bootstraps or relays through itself
/// (the public node runs with the same defaults as everyone else).
fn without_self(addrs: &mut Vec<Multiaddr>, me: PeerId) {
    addrs.retain(|a| peer_of(a) != Some(me));
}

/// DHT keys. A model's root is announced by nodes that hold all of it, `manifest/<root>`
/// by any node with its manifest, and each block id by nodes that hold that block.
fn dht_key(key: &str) -> kad::RecordKey {
    kad::RecordKey::new(&key)
}

fn holder_key(root: &str) -> String {
    format!("manifest/{root}")
}

/// DHT key announced by nodes that hold a whole Ollama model with manifest `digest`.
fn ollama_key(digest: &str) -> String {
    format!("ollama/{digest}")
}

/// What this node knows it holds of one model.
struct Held {
    blocks: Vec<String>,
    /// Blocks known to be complete.
    done: HashSet<String>,
}

impl Node {
    /// Start a node serving `store`. It announces every model already in the store and
    /// any added later.
    pub async fn start(store: Arc<Store>, key: Keypair, mut cfg: Config) -> Result<Node> {
        let peer_id = key.public().to_peer_id();
        let signing_key = key.clone();
        without_self(&mut cfg.bootstrap, peer_id);
        without_self(&mut cfg.relays, peer_id);
        without_self(&mut cfg.anchors, peer_id);
        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_quic()
            .with_dns()?
            .with_relay_client(noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key, relay_client| {
                let id = key.public().to_peer_id();
                let mut kad_cfg = kad::Config::new(KAD_PROTOCOL);
                kad_cfg.set_query_timeout(Duration::from_secs(30));
                // Look up keys along several independent paths (S/Kademlia), so peers that
                // capture one part of the network can't hide a model from us.
                kad_cfg.disjoint_query_paths(true);
                // We decide who enters the routing table (see `route`), to cap how many
                // entries one subnet can take.
                kad_cfg.set_kbucket_inserts(kad::BucketInserts::Manual);
                let mut kad = kad::Behaviour::with_config(id, MemoryStore::new(id), kad_cfg);
                if cfg.limits.download_only {
                    kad.set_mode(Some(kad::Mode::Client));
                } else if cfg.public || cfg.relay_server || !cfg.external.is_empty() {
                    kad.set_mode(Some(kad::Mode::Server));
                }
                let support = if cfg.limits.download_only {
                    ProtocolSupport::Outbound
                } else {
                    ProtocolSupport::Full
                };
                let relay = cfg.relay_server.then(|| {
                    relay::Behaviour::new(
                        id,
                        relay::Config {
                            max_circuits: cfg.relay.max_circuits,
                            max_circuits_per_peer: cfg.relay.max_circuits_per_peer,
                            max_circuit_duration: cfg.relay.circuit_duration,
                            max_circuit_bytes: cfg.relay.circuit_bytes,
                            max_reservations: cfg.relay.max_reservations,
                            ..Default::default()
                        },
                    )
                });
                Behaviour {
                    kad,
                    identify: identify::Behaviour::new(
                        identify::Config::new("/chungus/1.0.0".into(), key.public())
                            .with_agent_version(format!("chungus/{}", env!("CARGO_PKG_VERSION"))),
                    ),
                    ping: ping::Behaviour::default(),
                    rr: request_response::cbor::Behaviour::with_codec(
                        request_response::cbor::codec::Codec::default()
                            .set_request_size_maximum(4096)
                            .set_response_size_maximum(MAX_RESPONSE),
                        [(PROTOCOL, support)],
                        request_response::Config::default()
                            .with_request_timeout(Duration::from_secs(60)),
                    ),
                    relay_client,
                    relay: relay.into(),
                    dcutr: dcutr::Behaviour::new(id),
                    limits: connection_limits::Behaviour::new(
                        connection_limits::ConnectionLimits::default()
                            .with_max_established(Some(cfg.limits.max_connections))
                            .with_max_established_per_peer(Some(4))
                            .with_max_pending_incoming(Some(64)),
                    ),
                }
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        for addr in &cfg.listen {
            swarm
                .listen_on(addr.clone())
                .with_context(|| format!("listen on {addr}"))?;
        }
        for addr in &cfg.external {
            swarm.add_external_address(addr.clone());
        }
        // Connect to each relay first and ask for a reservation once connected (see
        // `on_event`); asking before the connection exists loses the reservation when it
        // races with other dials to the same peer.
        let mut relays = HashMap::new();
        for addr in &cfg.relays {
            let peer = peer_of(addr)
                .with_context(|| format!("relay address {addr} must end in /p2p/<peer id>"))?;
            swarm.behaviour_mut().kad.add_address(&peer, addr.clone());
            swarm.dial(addr.clone())?;
            relays.insert(peer, (addr.clone(), None));
        }
        for addr in &cfg.bootstrap {
            let peer = peer_of(addr)
                .with_context(|| format!("bootstrap address {addr} must end in /p2p/<peer id>"))?;
            swarm.behaviour_mut().kad.add_address(&peer, addr.clone());
            if !relays.contains_key(&peer) {
                swarm.dial(addr.clone())?;
            }
        }
        let mut anchors = Vec::new();
        let mut known: HashMap<PeerId, &'static str> = HashMap::new();
        for addr in &cfg.bootstrap {
            known.extend(peer_of(addr).map(|p| (p, "bootstrap")));
        }
        for addr in &cfg.relays {
            known.extend(peer_of(addr).map(|p| (p, "relay")));
        }
        for addr in &cfg.anchors {
            let peer = peer_of(addr)
                .with_context(|| format!("anchor address {addr} must end in /p2p/<peer id>"))?;
            swarm.behaviour_mut().kad.add_address(&peer, addr.clone());
            swarm.behaviour_mut().limits.bypass_peer_id(&peer);
            known.entry(peer).or_insert("anchor");
            if !relays.contains_key(&peer) {
                swarm.dial(addr.clone())?;
            }
            anchors.push(peer);
        }
        if !cfg.bootstrap.is_empty() || !anchors.is_empty() {
            let _ = swarm.behaviour_mut().kad.bootstrap();
        }

        let block_bytes = if cfg.block_bytes == 0 {
            BLOCK_BYTES
        } else {
            cfg.block_bytes
        };
        let mut via: Vec<Multiaddr> = Vec::new();
        for addr in cfg.relays.iter().chain(&cfg.bootstrap).chain(&cfg.anchors) {
            let relayed = addr.iter().any(|p| matches!(p, Protocol::P2pCircuit));
            if !relayed && !via.contains(addr) {
                via.push(addr.clone());
            }
        }
        let metrics = Arc::new(Metrics::default());
        if let Some(addr) = cfg.metrics {
            serve_metrics(addr, metrics.clone()).await?;
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let runner = Runner {
            swarm,
            store,
            public: cfg.public,
            rx,
            addrs: HashMap::new(),
            via,
            providers: HashMap::new(),
            requests: HashMap::new(),
            provided: HashSet::new(),
            held: HashMap::new(),
            block_bytes,
            relays,
            caps: SubnetCaps::new(cfg.limits.max_peers_per_subnet),
            limiter: cfg
                .limits
                .upload_bytes_per_sec
                .map(|r| Arc::new(RateLimiter::new(r))),
            serving: HashMap::new(),
            anchor_set: anchors.iter().copied().collect(),
            limits: cfg.limits.clone(),
            log: cfg.log,
            known,
            observed: HashSet::new(),
            refused: HashSet::new(),
            mismatched: HashSet::new(),
            joined: false,
            granted: HashMap::new(),
            closest: HashMap::new(),
            receipts: cfg.receipts.clone(),
            metrics: metrics.clone(),
        };
        tokio::spawn(runner.run());
        Ok(Node {
            tx,
            peer_id,
            block_bytes,
            anchors,
            key: signing_key,
            metrics,
        })
    }

    fn send(&self, cmd: Command) -> Result<()> {
        self.tx
            .send(cmd)
            .map_err(|_| anyhow!("the node has stopped"))
    }

    /// Peers that hold all of model `root`, not counting this node.
    pub async fn providers(&self, root: &str) -> Result<Vec<PeerId>> {
        self.providers_of(root).await
    }

    /// Peers that hold at least the manifest of `root`.
    pub async fn holders(&self, root: &str) -> Result<Vec<PeerId>> {
        self.providers_of(&holder_key(root)).await
    }

    /// Peers that hold the whole Ollama model with manifest `digest` (`sha256:<hex>`).
    pub async fn ollama_providers(&self, digest: &str) -> Result<Vec<PeerId>> {
        self.providers_of(&ollama_key(digest)).await
    }

    /// Peers that announce DHT key `key`, not counting this node.
    async fn providers_of(&self, key: &str) -> Result<Vec<PeerId>> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Providers(key.to_string(), tx))?;
        Ok(rx.await?)
    }

    pub async fn request(&self, peer: PeerId, req: Request) -> Result<Response> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Request(peer, req, tx))?;
        rx.await?
    }

    /// This node's identity, which signs its receipts.
    pub fn key(&self) -> &Keypair {
        &self.key
    }

    /// Look `peer` up in the DHT unless we already know how to reach it, so a request
    /// to it can find an address.
    pub async fn find(&self, peer: PeerId) {
        let (tx, rx) = oneshot::channel();
        if self.send(Command::Find(peer, tx)).is_ok() {
            let _ = rx.await;
        }
    }

    /// Announce what the store holds now instead of at the next rescan.
    pub fn announce(&self) -> Result<()> {
        self.send(Command::Announce)
    }

    /// Addresses others can dial this node at, each ending in `/p2p/<peer id>`. Waits up
    /// to `wait` for the node to have at least one.
    pub async fn addresses(&self, wait: Duration) -> Result<Vec<Multiaddr>> {
        let deadline = Instant::now() + wait;
        loop {
            let (tx, rx) = oneshot::channel();
            self.send(Command::Addresses(tx))?;
            let addrs = rx.await?;
            if !addrs.is_empty() || Instant::now() >= deadline {
                return Ok(addrs);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

struct Runner {
    swarm: Swarm<Behaviour>,
    store: Arc<Store>,
    public: bool,
    rx: mpsc::UnboundedReceiver<Command>,
    /// Addresses learned for peers, used when a request needs a new connection.
    addrs: HashMap<PeerId, HashSet<Multiaddr>>,
    /// Bootstrap, relay and anchor addresses: possible relays to reach a peer through.
    via: Vec<Multiaddr>,
    providers: HashMap<kad::QueryId, (HashSet<PeerId>, oneshot::Sender<Vec<PeerId>>)>,
    requests: HashMap<OutboundRequestId, oneshot::Sender<Result<Response>>>,
    /// DHT keys this node currently announces.
    provided: HashSet<String>,
    held: HashMap<String, Held>,
    block_bytes: u64,
    /// Relays to listen through: their address, and the listener while we have one.
    relays: HashMap<PeerId, (Multiaddr, Option<libp2p::core::transport::ListenerId>)>,
    caps: SubnetCaps,
    limiter: Option<Arc<RateLimiter>>,
    /// Requests being served, per peer.
    serving: HashMap<PeerId, usize>,
    anchor_set: HashSet<PeerId>,
    limits: Limits,
    log: bool,
    /// Peers the operator named, and what as ("bootstrap", "relay"), for clearer logs.
    known: HashMap<PeerId, &'static str>,
    /// Named peers that have told us which address they see us at.
    observed: HashSet<PeerId>,
    /// Models in the store that list unsafe files, so are never announced.
    refused: HashSet<String>,
    /// chungus peers on an incompatible protocol version, reported once each.
    mismatched: HashSet<PeerId>,
    /// Whether a DHT bootstrap has succeeded yet.
    joined: bool,
    /// Gated repos each peer has shown a valid ticket for, and when each ticket expires.
    granted: HashMap<PeerId, HashMap<String, u64>>,
    /// DHT lookups of a peer's address, from [`Node::find`].
    closest: HashMap<kad::QueryId, oneshot::Sender<()>>,
    receipts: Option<mpsc::UnboundedSender<Receipt>>,
    metrics: Arc<Metrics>,
}

type Reply = (PeerId, ResponseChannel<Response>, Response);

impl Runner {
    async fn run(mut self) {
        let (resp_tx, mut resp_rx) = mpsc::unbounded_channel::<Reply>();
        let mut rescan = tokio::time::interval(RESCAN);
        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => self.on_event(event, &resp_tx),
                cmd = self.rx.recv() => match cmd {
                    Some(cmd) => self.on_command(cmd),
                    None => return,
                },
                Some((peer, channel, resp)) = resp_rx.recv() => {
                    let size = resp.size() as u64;
                    if self.swarm.behaviour_mut().rr.send_response(channel, resp).is_ok() {
                        add(&self.metrics.bytes_served, size);
                    }
                    if let Some(n) = self.serving.get_mut(&peer) {
                        *n -= 1;
                        if *n == 0 {
                            self.serving.remove(&peer);
                        }
                    }
                }
                _ = rescan.tick() => {
                    self.announce();
                    self.reconnect_relays();
                }
            }
        }
    }

    /// Bring the DHT announcements in line with the store: every model's manifest, every
    /// complete block, and the root of every complete model. Kademlia republishes the
    /// records on its own; this adds new ones and withdraws those that no longer hold.
    fn announce(&mut self) {
        if self.limits.download_only {
            return;
        }
        let Ok(roots) = self.store.manifests() else {
            return;
        };
        let mut want = HashSet::new();
        for root in &roots {
            if self.refused.contains(root) {
                continue;
            }
            let held = match self.held.get_mut(root) {
                Some(h) => h,
                None => {
                    let Ok(m) = self.store.get_manifest(root) else {
                        continue;
                    };
                    // Stores packed before chungus refused pickles may still hold some.
                    if crate::safety::check_manifest(&m).is_err() {
                        self.refused.insert(root.clone());
                        continue;
                    }
                    let blocks = m.blocks(self.block_bytes);
                    self.held.entry(root.clone()).or_insert(Held {
                        blocks: blocks.iter().map(|b| b.id.clone()).collect(),
                        done: HashSet::new(),
                    })
                }
            };
            if held.done.len() < held.blocks.len() {
                // Only models still downloading need their chunks checked again.
                if let Ok(m) = self.store.get_manifest(root) {
                    for b in m.blocks(self.block_bytes) {
                        if !held.done.contains(&b.id)
                            && b.chunks.iter().all(|c| self.store.contains(&c.hash))
                        {
                            held.done.insert(b.id);
                        }
                    }
                }
            }
            want.insert(holder_key(root));
            want.extend(held.done.iter().cloned());
            if held.done.len() == held.blocks.len() {
                want.insert(root.clone());
            }
        }
        self.held.retain(|r, _| roots.contains(r));
        // Ollama models held in full, by manifest digest, so pulls can find them.
        for (digest, root) in crate::ollama::digest_records(&self.store) {
            if want.contains(&root) {
                want.insert(ollama_key(&digest));
            }
        }

        let stale: Vec<String> = self.provided.difference(&want).cloned().collect();
        for key in stale {
            self.swarm
                .behaviour_mut()
                .kad
                .stop_providing(&dht_key(&key));
            self.provided.remove(&key);
        }
        for key in want {
            if !self.provided.contains(&key)
                && self
                    .swarm
                    .behaviour_mut()
                    .kad
                    .start_providing(dht_key(&key))
                    .is_ok()
            {
                self.provided.insert(key);
            }
        }
    }

    /// Redial relays we have lost, so this node stays reachable.
    fn reconnect_relays(&mut self) {
        let lost: Vec<Multiaddr> = self
            .relays
            .iter()
            .filter(|(peer, (_, listener))| listener.is_none() && !self.swarm.is_connected(peer))
            .map(|(_, (addr, _))| addr.clone())
            .collect();
        for addr in lost {
            let _ = self.swarm.dial(addr);
        }
    }

    fn on_command(&mut self, cmd: Command) {
        match cmd {
            Command::Providers(key, tx) => {
                let id = self.swarm.behaviour_mut().kad.get_providers(dht_key(&key));
                self.providers.insert(id, (HashSet::new(), tx));
            }
            Command::Request(peer, req, tx) => {
                let mut addrs: Vec<Multiaddr> = self
                    .addrs
                    .get(&peer)
                    .map(|a| a.iter().cloned().collect())
                    .unwrap_or_default();
                // A peer behind NAT is reachable only through a relay, and what we know of
                // it may be a private address or nothing at all (provider lookups return
                // bare peer ids). So also try reaching it through each node we joined
                // through; one that isn't its relay just refuses.
                if !self.swarm.is_connected(&peer) {
                    addrs.extend(
                        self.via
                            .iter()
                            .filter(|a| peer_of(a) != Some(peer))
                            .map(|a| {
                                a.clone()
                                    .with(Protocol::P2pCircuit)
                                    .with(Protocol::P2p(peer))
                            }),
                    );
                }
                let id = self
                    .swarm
                    .behaviour_mut()
                    .rr
                    .send_request_with_addresses(&peer, req, addrs);
                self.requests.insert(id, tx);
            }
            Command::Announce => self.announce(),
            Command::Find(peer, tx) => {
                if self.swarm.is_connected(&peer) || self.addrs.contains_key(&peer) {
                    let _ = tx.send(());
                } else {
                    let id = self.swarm.behaviour_mut().kad.get_closest_peers(peer);
                    self.closest.insert(id, tx);
                }
            }
            Command::Addresses(tx) => {
                let me = *self.swarm.local_peer_id();
                let mut addrs: Vec<Multiaddr> = self
                    .swarm
                    .external_addresses()
                    .chain(self.swarm.listeners())
                    .map(|a| {
                        if a.iter().last() == Some(Protocol::P2p(me)) {
                            a.clone()
                        } else {
                            a.clone().with(Protocol::P2p(me))
                        }
                    })
                    .collect();
                addrs.sort();
                addrs.dedup();
                let _ = tx.send(addrs);
            }
        }
    }

    fn say(&self, msg: impl std::fmt::Display) {
        if self.log {
            println!("{msg}");
        }
    }

    /// How to name `peer` in logs: "bootstrap 12D3…" for peers the operator named.
    fn who(&self, peer: &PeerId) -> String {
        match self.known.get(peer) {
            Some(role) => format!("{role} {peer}"),
            None => format!("peer {peer}"),
        }
    }

    fn peers(&self) -> usize {
        let n = self.swarm.connected_peers().count();
        self.metrics.peers.store(n as u64, Ordering::Relaxed);
        n
    }

    fn learn(&mut self, peer: PeerId, addr: Multiaddr) {
        self.addrs.entry(peer).or_default().insert(addr);
    }

    /// Offer a DHT peer to the routing table, subject to the subnet caps. Anchors always
    /// get in.
    fn route(&mut self, peer: PeerId, addr: Multiaddr) {
        if self.anchor_set.contains(&peer) || self.caps.admit(peer, &addr) {
            self.swarm.behaviour_mut().kad.add_address(&peer, addr);
        }
    }

    fn on_event(
        &mut self,
        event: SwarmEvent<BehaviourEvent>,
        resp_tx: &mpsc::UnboundedSender<Reply>,
    ) {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                let relayed = address.iter().any(|p| matches!(p, Protocol::P2pCircuit));
                // A relayed address is reachable by construction; a direct one only if
                // the operator says so.
                if relayed || self.public {
                    self.swarm.add_external_address(address);
                }
            }
            SwarmEvent::ConnectionEstablished {
                peer_id,
                endpoint,
                num_established,
                ..
            } => {
                if num_established.get() == 1 {
                    self.peers();
                    let how = if endpoint.is_relayed() {
                        "through a relay"
                    } else if endpoint.is_dialer() {
                        "outbound"
                    } else {
                        "inbound"
                    };
                    let addr = endpoint.get_remote_address();
                    self.say(format_args!(
                        "connected to {} ({how}, {addr}); {} peer(s) connected",
                        self.who(&peer_id),
                        self.peers()
                    ));
                }
                if !endpoint.is_relayed()
                    && let Some((addr, listener @ None)) = self.relays.get_mut(&peer_id)
                {
                    *listener = self
                        .swarm
                        .listen_on(addr.clone().with(Protocol::P2pCircuit))
                        .ok();
                    if self.log {
                        println!("asking relay {peer_id} for a reservation");
                    }
                }
            }
            SwarmEvent::ConnectionClosed {
                peer_id,
                num_established: 0,
                cause,
                ..
            } => {
                self.peers();
                let why = cause.map(|c| format!(": {c}")).unwrap_or_default();
                self.say(format_args!(
                    "disconnected from {}{why}; {} peer(s) connected",
                    self.who(&peer_id),
                    self.peers()
                ));
            }
            SwarmEvent::OutgoingConnectionError {
                peer_id: Some(peer),
                error,
                ..
            } if self.known.contains_key(&peer) => {
                // Dial errors nest across several lines; keep the log to one.
                let error = error.to_string().split_whitespace().collect::<Vec<_>>().join(" ");
                self.say(format_args!("could not reach {}: {error}", self.who(&peer)));
            }
            SwarmEvent::ListenerClosed {
                listener_id,
                reason,
                ..
            } => {
                // Lost a relay reservation: ask again when next connected, and reconnect.
                let mut lost = Vec::new();
                for (peer, (addr, listener)) in self.relays.iter_mut() {
                    if *listener == Some(listener_id) {
                        *listener = None;
                        let _ = self.swarm.dial(addr.clone());
                        lost.push(*peer);
                    }
                }
                for peer in lost {
                    let why = reason.as_ref().err().map(|e| format!(": {e}")).unwrap_or_default();
                    self.say(format_args!("lost the reservation on relay {peer}{why}; retrying"));
                }
            }
            SwarmEvent::Behaviour(BehaviourEvent::RelayClient(ev)) => match ev {
                relay::client::Event::ReservationReqAccepted {
                    relay_peer_id,
                    renewal: false,
                    ..
                } => self.say(format_args!(
                    "relay reservation accepted by {relay_peer_id}; others can now reach this node through it"
                )),
                relay::client::Event::InboundCircuitEstablished { src_peer_id, .. } => self.say(
                    format_args!("peer {src_peer_id} connected to us through a relay"),
                ),
                _ => {}
            },
            SwarmEvent::Behaviour(BehaviourEvent::Relay(ev)) => match ev {
                relay::Event::ReservationReqAccepted {
                    src_peer_id,
                    renewed: false,
                } => self.say(format_args!("relaying for {src_peer_id} (reservation accepted)")),
                relay::Event::ReservationReqDenied {
                    src_peer_id,
                    status,
                } => self.say(format_args!(
                    "refused a relay reservation from {src_peer_id}: {status:?}"
                )),
                relay::Event::ReservationTimedOut { src_peer_id } => {
                    self.say(format_args!("relay reservation for {src_peer_id} expired"))
                }
                relay::Event::CircuitReqAccepted {
                    src_peer_id,
                    dst_peer_id,
                } => {
                    add(&self.metrics.relay_circuits, 1);
                    add(&self.metrics.relay_circuits_accepted, 1);
                    self.say(format_args!(
                        "relaying a connection from {src_peer_id} to {dst_peer_id}"
                    ))
                }
                relay::Event::CircuitClosed { .. } => {
                    let _ = self.metrics.relay_circuits.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |n| n.checked_sub(1),
                    );
                }
                relay::Event::CircuitReqDenied {
                    src_peer_id,
                    dst_peer_id,
                    status,
                } => self.say(format_args!(
                    "refused to relay {src_peer_id} to {dst_peer_id}: {status:?}"
                )),
                _ => {}
            },
            SwarmEvent::Behaviour(BehaviourEvent::Dcutr(dcutr::Event {
                remote_peer_id,
                result,
            })) => match result {
                Ok(_) => self.say(format_args!(
                    "hole punch to {remote_peer_id} worked; now connected directly"
                )),
                Err(e) => self.say(format_args!("hole punch to {remote_peer_id} failed: {e}")),
            },
            SwarmEvent::Behaviour(BehaviourEvent::Identify(identify::Event::Received {
                peer_id,
                info,
                ..
            })) => {
                // What a named peer sees tells the operator whether they're behind NAT.
                if self.known.contains_key(&peer_id) && self.observed.insert(peer_id) {
                    self.say(format_args!(
                        "{} sees this node at {}",
                        self.who(&peer_id),
                        info.observed_addr
                    ));
                }
                let dht = info.protocols.contains(&KAD_PROTOCOL);
                // A peer that speaks none of our protocols is usually a chungus from after
                // a wire change; say so once rather than failing requests silently.
                if info.agent_version.starts_with("chungus/")
                    && !info.protocols.contains(&PROTOCOL)
                    && self.mismatched.insert(peer_id)
                {
                    self.say(format_args!(
                        "{peer_id} runs {} and speaks no protocol this node does; one of you \
                         needs to upgrade",
                        info.agent_version
                    ));
                }
                for addr in info.listen_addrs {
                    if dht {
                        self.route(peer_id, addr.clone());
                    }
                    self.learn(peer_id, addr);
                }
            }
            SwarmEvent::Behaviour(BehaviourEvent::Kad(ev)) => match ev {
                kad::Event::RoutablePeer { peer, address }
                | kad::Event::PendingRoutablePeer { peer, address } => {
                    self.learn(peer, address.clone());
                    self.route(peer, address);
                }
                kad::Event::RoutingUpdated {
                    old_peer: Some(old),
                    ..
                } => self.caps.remove(&old),
                kad::Event::OutboundQueryProgressed {
                    result: kad::QueryResult::Bootstrap(res),
                    step,
                    ..
                } => {
                    // Kademlia bootstraps again every few minutes; report the first join,
                    // and failures until then.
                    if step.last && !self.joined {
                        // The bootstrap query succeeds on the seed addresses alone, even
                        // when none of them answer, so also require a live connection.
                        self.joined = res.is_ok() && self.peers() > 0;
                        let table: usize = self
                            .swarm
                            .behaviour_mut()
                            .kad
                            .kbuckets()
                            .map(|b| b.num_entries())
                            .sum();
                        match res {
                            Ok(_) if !self.joined => self.say(
                                "DHT bootstrap found no reachable peers yet; still trying",
                            ),
                            Ok(_) => self.say(format_args!(
                                "joined the DHT: {table} peer(s) in the routing table"
                            )),
                            Err(e) => self.say(format_args!(
                                "DHT bootstrap did not finish ({e:?}); {table} peer(s) in the routing table"
                            )),
                        }
                    }
                }
                kad::Event::OutboundQueryProgressed {
                    id, result, step, ..
                } => {
                    if let kad::QueryResult::GetClosestPeers(Ok(ok)) = &result {
                        for p in &ok.peers {
                            for a in &p.addrs {
                                self.learn(p.peer_id, a.clone());
                            }
                        }
                    }
                    if step.last
                        && let Some(tx) = self.closest.remove(&id)
                    {
                        let _ = tx.send(());
                    }
                    if let kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
                        providers,
                        ..
                    })) = result
                        && let Some((found, _)) = self.providers.get_mut(&id)
                    {
                        found.extend(providers);
                    }
                    if step.last
                        && let Some((found, tx)) = self.providers.remove(&id)
                    {
                        let me = *self.swarm.local_peer_id();
                        let _ = tx.send(found.into_iter().filter(|p| *p != me).collect());
                    }
                }
                _ => {}
            },
            SwarmEvent::Behaviour(BehaviourEvent::Rr(ev)) => match ev {
                request_response::Event::Message { peer, message, .. } => match message {
                    request_response::Message::Request {
                        request, channel, ..
                    } => {
                        add(&self.metrics.requests, 1);
                        let mine = self.serving.get(&peer).copied().unwrap_or(0);
                        let total: usize = self.serving.values().sum();
                        *self.serving.entry(peer).or_default() += 1;
                        if mine >= self.limits.max_requests_per_peer
                            || total >= self.limits.max_uploads
                        {
                            add(&self.metrics.busy, 1);
                            let _ = resp_tx.send((peer, channel, Response::Busy));
                            return;
                        }
                        if let Request::Access(ticket) = &request {
                            let ok = self.store.ticket_issuers().iter().any(|op| {
                                ticket.valid_for(op, &peer.to_string())
                            });
                            if ok {
                                self.granted
                                    .entry(peer)
                                    .or_default()
                                    .insert(ticket.repo.clone(), ticket.expires);
                            }
                            let _ = resp_tx.send((peer, channel, Response::Granted(ok)));
                            return;
                        }
                        if let Request::Receipt(r) = request {
                            // Only the downloader itself may hand over its receipt, and
                            // only for what this node served.
                            let ok = r.fetcher == peer.to_string()
                                && r.server == self.swarm.local_peer_id().to_string()
                                && r.verify()
                                && self.receipts.as_ref().is_some_and(|tx| tx.send(r).is_ok());
                            let _ = resp_tx.send((peer, channel, Response::Receipt(ok)));
                            return;
                        }
                        let now = registry::now();
                        let granted: HashSet<String> = self
                            .granted
                            .get(&peer)
                            .map(|g| {
                                g.iter()
                                    .filter(|(_, exp)| **exp > now)
                                    .map(|(r, _)| r.clone())
                                    .collect()
                            })
                            .unwrap_or_default();
                        let (store, resp_tx) = (self.store.clone(), resp_tx.clone());
                        let limiter = self.limiter.clone();
                        tokio::spawn(async move {
                            let resp =
                                tokio::task::spawn_blocking(move || answer(&store, request, &granted))
                                .await
                                .unwrap_or(Response::Busy);
                            if let Some(lim) = limiter {
                                lim.take(resp.size() as u64).await;
                            }
                            let _ = resp_tx.send((peer, channel, resp));
                        });
                    }
                    request_response::Message::Response {
                        request_id,
                        response,
                    } => {
                        if let Some(tx) = self.requests.remove(&request_id) {
                            let _ = tx.send(Ok(response));
                        }
                    }
                },
                request_response::Event::OutboundFailure {
                    request_id, error, ..
                } => {
                    if let Some(tx) = self.requests.remove(&request_id) {
                        let _ = tx.send(Err(anyhow!("request failed: {error}")));
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
}

impl Response {
    /// Roughly how many bytes this response puts on the wire.
    fn size(&self) -> usize {
        match self {
            Response::Manifest(Some(b)) | Response::Chunk(Some(b)) => b.len(),
            Response::Signatures(s) => s.len() * 200,
            _ => 16,
        }
    }
}

/// Answer a peer's request from the store. Anything missing or malformed is `None`.
/// `granted` holds the gated repos the peer has a valid ticket for.
fn answer(store: &Store, req: Request, granted: &HashSet<String>) -> Response {
    match req {
        Request::Manifest(root) => {
            Response::Manifest(store.get_safe_manifest_bytes(&root).ok().map(ByteBuf::from))
        }
        Request::Signatures(root) => {
            Response::Signatures(store.signatures(&root).unwrap_or_default())
        }
        Request::Chunk(hash) => {
            let gates = store.gates_of(&hash);
            if !gates.is_empty() && !gates.iter().any(|g| granted.contains(g)) {
                return Response::Gated(gates[0].clone());
            }
            Response::Chunk(
                store::is_hash(&hash)
                    .then(|| store.get(&hash).ok())
                    .flatten()
                    .map(ByteBuf::from),
            )
        }
        Request::Access(_) => Response::Granted(false),
        Request::Receipt(_) => Response::Receipt(false),
        Request::Ollama(digest) => Response::Root(crate::ollama::root_for_digest(store, &digest)),
    }
}

/// Chunk requests a fetch keeps in flight to one peer.
const PER_PEER_REQUESTS: usize = 8;
/// How long a fetch keeps retrying a peer that answers "busy".
const BUSY_PATIENCE: Duration = Duration::from_secs(60);

/// Download model `root` from whoever has it on the DHT: nodes with the whole model, and
/// nodes with some of its blocks (typically ones still downloading it themselves). As with
/// a LAN fetch, the manifest must hash to `root` and every chunk to its hash, and when
/// `trust` is non-empty one of those keys must have signed the manifest before any chunk
/// is fetched.
pub async fn fetch(
    node: &Node,
    root: &str,
    store: Arc<Store>,
    trust: &[ed25519_dalek::VerifyingKey],
    access: Option<Access>,
) -> Result<(Manifest, FetchStats)> {
    let started = Instant::now();
    let (manifest, sources) = prepare(node, root, &store, trust, access).await?;
    let mut stats = net::fetch_chunks_from(
        &store,
        manifest.files.iter().flat_map(|f| &f.chunks),
        |hash| sources.sources(hash),
        |peer, hash| sources.get(peer, hash),
    )
    .await
    .map_err(|e| sources.explain(e))?;
    stats.secs = started.elapsed().as_secs_f64();
    Ok((manifest, stats))
}

/// Get ready to download `root` from the swarm: fetch and check its manifest and
/// signatures, then look up who holds each block the store still lacks.
pub async fn prepare(
    node: &Node,
    root: &str,
    store: &Arc<Store>,
    trust: &[ed25519_dalek::VerifyingKey],
    access: Option<Access>,
) -> Result<(Manifest, ModelSources)> {
    if !store::is_hash(root) {
        bail!("{root:?} is not a manifest root hash");
    }
    let (complete, holders) = tokio::try_join!(node.providers(root), node.holders(root))?;
    // Nodes with the whole model first; anyone with the manifest can supply it.
    let mut peers = complete.clone();
    for &p in holders.iter().chain(&node.anchors) {
        if !peers.contains(&p) {
            peers.push(p);
        }
    }

    let manifest = match store.get_manifest(root) {
        Ok(m) => m,
        Err(_) => {
            if peers.is_empty() {
                bail!("nobody on the network is sharing {root}");
            }
            // Two rounds: a peer behind NAT may still be connecting through its relay.
            let mut found = None;
            let mut why = Vec::new();
            for round in 0..2 {
                if round > 0 {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
                why.clear();
                for &p in &peers {
                    let problem = match node.request(p, Request::Manifest(root.to_string())).await {
                        Ok(Response::Manifest(Some(bytes))) => {
                            match crate::manifest::parse(&bytes) {
                                Ok(m) if m.root == root && m.verify_root() => {
                                    found = Some(m);
                                    break;
                                }
                                Ok(_) => "sent an invalid manifest".to_string(),
                                Err(e) => format!("{e:#}"),
                            }
                        }
                        Ok(Response::Manifest(None)) => "doesn't have it".to_string(),
                        Ok(Response::Busy) => "busy".to_string(),
                        Ok(_) => "sent an unexpected answer".to_string(),
                        Err(e) => format!("{e:#}"),
                    };
                    why.push(format!("{p}: {problem}"));
                }
                if found.is_some() {
                    break;
                }
            }
            match found {
                Some(m) => m,
                None => bail!(
                    "no peer sent a valid manifest for {root}:\n  {}",
                    why.join("\n  ")
                ),
            }
        }
    };
    store.put_manifest(&manifest)?;

    let mut sigs = Vec::new();
    for &p in &peers {
        if let Ok(Response::Signatures(s)) =
            node.request(p, Request::Signatures(root.to_string())).await
        {
            sigs.extend(s);
        }
    }
    store.add_signatures(root, &sigs)?;
    if !trust.is_empty() && !sign::trusted_by(&store.signatures(root)?, root, trust) {
        bail!("no trusted key has signed {root}; refusing to download it");
    }

    // Look up who holds each block we still need. Blocks we already have need no lookup.
    let blocks = manifest.blocks(node.block_bytes);
    let needed: Vec<usize> = (0..blocks.len())
        .filter(|&i| blocks[i].chunks.iter().any(|c| !store.contains(&c.hash)))
        .collect();
    let any_needed = !needed.is_empty();
    let found: Vec<(usize, Vec<PeerId>)> = futures::stream::iter(needed)
        .map(|i| {
            let key = blocks[i].id.clone();
            async move { (i, node.providers_of(&key).await.unwrap_or_default()) }
        })
        .buffer_unordered(16)
        .collect()
        .await;
    let mut block_peers: Vec<Vec<PeerId>> = vec![Vec::new(); blocks.len()];
    for (i, p) in found {
        block_peers[i] = p;
    }
    let block_of: HashMap<String, usize> = blocks
        .iter()
        .enumerate()
        .flat_map(|(i, b)| b.chunks.iter().map(move |c| (c.hash.clone(), i)))
        .collect();
    if any_needed
        && complete.is_empty()
        && node.anchors.is_empty()
        && block_peers.iter().all(Vec::is_empty)
    {
        bail!("nobody on the network has the chunks of {root}");
    }
    let sources = ModelSources {
        node: node.clone(),
        root: root.to_string(),
        complete,
        block_of,
        block_peers,
        slots: Default::default(),
        access,
        tickets: Default::default(),
        presented: Default::default(),
        gated_by: Default::default(),
        served: Default::default(),
    };
    Ok((manifest, sources))
}

/// How to get past a gate: the registry that issues access tickets, and the user's own
/// Hugging Face token to prove access with. The token goes only to the registry, which
/// checks it with Hugging Face; peers only ever see the ticket.
#[derive(Clone)]
pub struct Access {
    pub registry: String,
    pub hf_token: String,
}

/// Who to ask for each chunk of one model, from [`prepare`].
pub struct ModelSources {
    node: Node,
    root: String,
    complete: Vec<PeerId>,
    block_of: HashMap<String, usize>,
    block_peers: Vec<Vec<PeerId>>,
    /// Stay under a peer's default request limit, so a well-behaved fetch rarely hears
    /// "busy" even when one peer is its only source.
    slots: std::sync::Mutex<HashMap<PeerId, Arc<tokio::sync::Semaphore>>>,
    access: Option<Access>,
    /// Tickets by repo; None once the registry has refused one.
    tickets: tokio::sync::Mutex<HashMap<String, Option<AccessTicket>>>,
    /// (peer, repo) pairs a ticket has been accepted for.
    presented: std::sync::Mutex<HashSet<(PeerId, String)>>,
    /// Why gated chunks couldn't be had, for the error message.
    gated_by: std::sync::Mutex<Option<String>>,
    /// Bytes and chunks each peer has sent that checked out, for receipts.
    served: std::sync::Mutex<HashMap<PeerId, (u64, u64)>>,
}

impl ModelSources {
    /// Peers to ask for `hash`, best first: holders of the chunk's block, then holders of
    /// the whole model, then anchors (they may have it even if the DHT is being kept from
    /// us).
    pub fn sources(&self, hash: &str) -> Vec<PeerId> {
        let mut order = self
            .block_of
            .get(hash)
            .map(|&i| net::rotated(&self.block_peers[i], hash))
            .unwrap_or_default();
        for p in net::rotated(&self.complete, hash)
            .into_iter()
            .chain(self.node.anchors.iter().copied())
        {
            if !order.contains(&p) {
                order.push(p);
            }
        }
        order
    }

    /// Bytes and chunks each peer has sent so far that checked out.
    pub fn served(&self) -> Vec<(PeerId, u64, u64)> {
        let served = self.served.lock().unwrap();
        served.iter().map(|(p, (b, c))| (*p, *b, *c)).collect()
    }

    /// Hand each peer a receipt for what it has sent so far (see [`send_receipts`]).
    pub async fn send_receipts(&self) -> usize {
        send_receipts(&self.node, &self.root, &self.served()).await
    }

    /// Turn a failed download into a clearer error when gates were in the way.
    pub fn explain(&self, e: anyhow::Error) -> anyhow::Error {
        match self.gated_by.lock().unwrap().clone() {
            Some(why) => anyhow!("{e:#}\n{why}"),
            None => e,
        }
    }

    /// A ticket for `repo`, asking the registry the first time.
    async fn ticket(&self, repo: &str) -> Option<AccessTicket> {
        let mut tickets = self.tickets.lock().await;
        if let Some(t) = tickets.get(repo) {
            return t.clone();
        }
        let got = match &self.access {
            None => Err(anyhow!(
                "set HF_TOKEN to a Hugging Face token that has accepted its license"
            )),
            Some(a) => match registry::Client::new(&a.registry) {
                Ok(c) => {
                    c.access(&self.root, &self.node.peer_id.to_string(), &a.hf_token)
                        .await
                }
                Err(e) => Err(e),
            },
        };
        let t = match got {
            Ok(t) if t.repo == repo => Some(t),
            Ok(t) => {
                self.note_gate(
                    repo,
                    &format!("the registry gates it by {} instead", t.repo),
                );
                None
            }
            Err(e) => {
                self.note_gate(repo, &format!("{e:#}"));
                None
            }
        };
        tickets.insert(repo.to_string(), t.clone());
        t
    }

    fn note_gate(&self, repo: &str, why: &str) {
        *self.gated_by.lock().unwrap() = Some(format!(
            "this model is gated by https://huggingface.co/{repo}: {why}"
        ));
    }

    /// Show `peer` our ticket for `repo`. True if it accepted.
    async fn present(&self, peer: PeerId, repo: &str) -> bool {
        if self
            .presented
            .lock()
            .unwrap()
            .contains(&(peer, repo.to_string()))
        {
            return true;
        }
        let Some(t) = self.ticket(repo).await else {
            return false;
        };
        let ok = matches!(
            self.node.request(peer, Request::Access(t)).await,
            Ok(Response::Granted(true))
        );
        if ok {
            self.presented
                .lock()
                .unwrap()
                .insert((peer, repo.to_string()));
        }
        ok
    }

    /// Ask `peer` for chunk `hash`, showing it a ticket first if the chunk is gated.
    /// Unverified.
    pub async fn get(&self, peer: PeerId, hash: String) -> Option<Bytes> {
        match self.get_once(peer, hash.clone()).await {
            Err(repo) if self.present(peer, &repo).await => {
                self.get_once(peer, hash).await.ok().flatten()
            }
            other => other.ok().flatten(),
        }
    }

    /// Ask `peer` for chunk `hash` once (retrying while it's busy). Err(repo) if the
    /// chunk is gated and we haven't shown this peer a ticket.
    fn get_once(
        &self,
        peer: PeerId,
        hash: String,
    ) -> impl Future<Output = Result<Option<Bytes>, String>> + use<> {
        let node = self.node.clone();
        let slot = self
            .slots
            .lock()
            .unwrap()
            .entry(peer)
            .or_insert_with(|| Arc::new(tokio::sync::Semaphore::new(PER_PEER_REQUESTS)))
            .clone();
        async move {
            let _permit = slot.acquire_owned().await.map_err(|_| String::new())?;
            // A busy peer is at its upload limits but alive; keep asking for a while
            // rather than moving on, since it may be the only one with this chunk.
            let give_up = tokio::time::Instant::now() + BUSY_PATIENCE;
            let mut attempt = 0u32;
            loop {
                match node.request(peer, Request::Chunk(hash.clone())).await {
                    Ok(Response::Chunk(Some(b))) => return Ok(Some(Bytes::from(b.into_vec()))),
                    Ok(Response::Gated(repo)) => return Err(repo),
                    Ok(Response::Busy) if tokio::time::Instant::now() < give_up => {
                        // Spread retries out so waiting requests don't all return at once.
                        let jitter = u64::from(hash.as_bytes()[attempt as usize % 64]) % 50;
                        let wait = (50 << attempt.min(4)) + jitter;
                        tokio::time::sleep(Duration::from_millis(wait)).await;
                        attempt += 1;
                    }
                    _ => return Ok(None),
                }
            }
        }
    }
}

impl crate::lazy::ChunkSource for ModelSources {
    fn fetch<'a>(
        &'a self,
        store: &'a Arc<Store>,
        hash: &'a str,
        len: usize,
    ) -> futures::future::BoxFuture<'a, bool> {
        Box::pin(async move {
            for peer in self.sources(hash) {
                if let Some(blob) = self.get(peer, hash.to_string()).await
                    && let Some(bytes) = net::verify_and_store(store, hash, len, blob).await
                {
                    let mut served = self.served.lock().unwrap();
                    let s = served.entry(peer).or_default();
                    s.0 += bytes;
                    s.1 += 1;
                    return true;
                }
            }
            false
        })
    }
}

/// Sign a receipt for each of `served` (peer, bytes, chunks) with this node's key and
/// hand it to that peer, which submits it to the registry. Best effort: returns how many
/// peers took theirs. Receipts are cumulative, so sending again later replaces them.
pub async fn send_receipts(node: &Node, root: &str, served: &[(PeerId, u64, u64)]) -> usize {
    let sends = served
        .iter()
        .filter(|(_, bytes, _)| *bytes > 0)
        .map(|&(peer, bytes, chunks)| {
            let r = Receipt::new(&node.key, root, &peer, bytes, chunks);
            async move {
                let resp = tokio::time::timeout(
                    Duration::from_secs(10),
                    node.request(peer, Request::Receipt(r)),
                )
                .await;
                matches!(resp, Ok(Ok(Response::Receipt(true))))
            }
        });
    futures::future::join_all(sends)
        .await
        .into_iter()
        .filter(|ok| *ok)
        .count()
}

/// What each peer sent in a finished fetch, from its stats, for [`send_receipts`].
pub fn served_by(stats: &FetchStats) -> Vec<(PeerId, u64, u64)> {
    stats
        .bytes_by_source
        .iter()
        .filter_map(|(source, bytes)| {
            let chunks = stats.chunks_by_source.get(source).copied().unwrap_or(0);
            Some((source.parse().ok()?, *bytes, chunks))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_nodes_parse_and_skip_self() {
        let boot = default_bootstrap();
        assert_eq!(boot.len(), 2);
        let me: PeerId = DEFAULT_NODE.parse().unwrap();
        assert!(boot.iter().all(|a| peer_of(a) == Some(me)));
        assert_eq!(default_relays().len(), 1);

        let mut addrs = boot.clone();
        without_self(&mut addrs, PeerId::random());
        assert_eq!(addrs, boot);
        without_self(&mut addrs, me);
        assert!(addrs.is_empty());
    }
}
