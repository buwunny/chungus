//! Sharing models over the internet with libp2p.
//!
//! Every node joins a Kademlia DHT and announces itself as a provider of each model
//! (manifest root) in its store. To fetch a model, a node asks the DHT who provides it,
//! then requests the manifest, its signatures and the chunks from those providers over
//! `/chungus/1`, spreading chunks across them and verifying each one against its hash.
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
    Multiaddr, PeerId, StreamProtocol, Swarm, dcutr, identify, noise, ping, relay, tcp, yamux,
};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

use crate::manifest::Manifest;
use crate::net::{self, FetchStats};
use crate::sign::{self, Signature};
use crate::store::{self, Store};

pub const PROTOCOL: StreamProtocol = StreamProtocol::new("/chungus/1");
pub const KAD_PROTOCOL: StreamProtocol = StreamProtocol::new("/chungus/kad/1");
pub const DEFAULT_PORT: u16 = 4001;
/// Largest response a peer may send: a manifest of a very large model, or one chunk.
const MAX_RESPONSE: u64 = 256 << 20;
/// How often a node looks for newly added models to announce.
const RESCAN: Duration = Duration::from_secs(60);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Request {
    Manifest(String),
    Signatures(String),
    Chunk(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Response {
    Manifest(Option<ByteBuf>),
    Signatures(Vec<Signature>),
    Chunk(Option<ByteBuf>),
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
}

enum Command {
    Providers(String, oneshot::Sender<Vec<PeerId>>),
    Request(PeerId, Request, oneshot::Sender<Result<Response>>),
    Provide(String),
    Addresses(oneshot::Sender<Vec<Multiaddr>>),
}

/// A running node. Cheap to clone; the swarm runs in a background task until every
/// handle is dropped.
#[derive(Clone)]
pub struct Node {
    tx: mpsc::UnboundedSender<Command>,
    pub peer_id: PeerId,
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

fn root_key(root: &str) -> kad::RecordKey {
    kad::RecordKey::new(&root)
}

impl Node {
    /// Start a node serving `store`. It announces every model already in the store and
    /// any added later.
    pub async fn start(store: Arc<Store>, key: Keypair, cfg: Config) -> Result<Node> {
        let peer_id = key.public().to_peer_id();
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
                let mut kad = kad::Behaviour::with_config(id, MemoryStore::new(id), kad_cfg);
                if cfg.public || cfg.relay_server || !cfg.external.is_empty() {
                    kad.set_mode(Some(kad::Mode::Server));
                }
                let relay = cfg.relay_server.then(|| {
                    relay::Behaviour::new(
                        id,
                        relay::Config {
                            // Model downloads are long and large; the defaults suit only
                            // hole-punch coordination.
                            max_circuit_duration: Duration::from_secs(3600),
                            max_circuit_bytes: 64 << 30,
                            ..Default::default()
                        },
                    )
                });
                Behaviour {
                    kad,
                    identify: identify::Behaviour::new(identify::Config::new(
                        "/chungus/1.0.0".into(),
                        key.public(),
                    )),
                    ping: ping::Behaviour::default(),
                    rr: request_response::cbor::Behaviour::with_codec(
                        request_response::cbor::codec::Codec::default()
                            .set_request_size_maximum(4096)
                            .set_response_size_maximum(MAX_RESPONSE),
                        [(PROTOCOL, ProtocolSupport::Full)],
                        request_response::Config::default()
                            .with_request_timeout(Duration::from_secs(60)),
                    ),
                    relay_client,
                    relay: relay.into(),
                    dcutr: dcutr::Behaviour::new(id),
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
        if !cfg.bootstrap.is_empty() {
            let _ = swarm.behaviour_mut().kad.bootstrap();
        }

        let (tx, rx) = mpsc::unbounded_channel();
        let runner = Runner {
            swarm,
            store,
            public: cfg.public,
            rx,
            addrs: HashMap::new(),
            providers: HashMap::new(),
            requests: HashMap::new(),
            provided: HashSet::new(),
            relays,
        };
        tokio::spawn(runner.run());
        Ok(Node { tx, peer_id })
    }

    fn send(&self, cmd: Command) -> Result<()> {
        self.tx
            .send(cmd)
            .map_err(|_| anyhow!("the node has stopped"))
    }

    /// Peers that announce `root` on the DHT, not counting this node.
    pub async fn providers(&self, root: &str) -> Result<Vec<PeerId>> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Providers(root.to_string(), tx))?;
        Ok(rx.await?)
    }

    pub async fn request(&self, peer: PeerId, req: Request) -> Result<Response> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Request(peer, req, tx))?;
        rx.await?
    }

    /// Announce `root` now instead of at the next store rescan.
    pub fn provide(&self, root: &str) -> Result<()> {
        self.send(Command::Provide(root.to_string()))
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
    providers: HashMap<kad::QueryId, (HashSet<PeerId>, oneshot::Sender<Vec<PeerId>>)>,
    requests: HashMap<OutboundRequestId, oneshot::Sender<Result<Response>>>,
    provided: HashSet<String>,
    /// Relays to listen through: their address, and the listener while we have one.
    relays: HashMap<PeerId, (Multiaddr, Option<libp2p::core::transport::ListenerId>)>,
}

impl Runner {
    async fn run(mut self) {
        let (resp_tx, mut resp_rx) =
            mpsc::unbounded_channel::<(ResponseChannel<Response>, Response)>();
        let mut rescan = tokio::time::interval(RESCAN);
        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => self.on_event(event, &resp_tx),
                cmd = self.rx.recv() => match cmd {
                    Some(cmd) => self.on_command(cmd),
                    None => return,
                },
                Some((channel, resp)) = resp_rx.recv() => {
                    let _ = self.swarm.behaviour_mut().rr.send_response(channel, resp);
                }
                _ = rescan.tick() => {
                    self.announce_new();
                    self.reconnect_relays();
                }
            }
        }
    }

    /// Announce every model in the store that hasn't been announced yet. Kademlia
    /// republishes the records it already has on its own.
    fn announce_new(&mut self) {
        let Ok(roots) = self.store.manifests() else {
            return;
        };
        for root in roots {
            if !self.provided.contains(&root) {
                self.provide(root);
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

    fn provide(&mut self, root: String) {
        if self
            .swarm
            .behaviour_mut()
            .kad
            .start_providing(root_key(&root))
            .is_ok()
        {
            self.provided.insert(root);
        }
    }

    fn on_command(&mut self, cmd: Command) {
        match cmd {
            Command::Providers(root, tx) => {
                let id = self
                    .swarm
                    .behaviour_mut()
                    .kad
                    .get_providers(root_key(&root));
                self.providers.insert(id, (HashSet::new(), tx));
            }
            Command::Request(peer, req, tx) => {
                let addrs = self
                    .addrs
                    .get(&peer)
                    .map(|a| a.iter().cloned().collect())
                    .unwrap_or_default();
                let id = self
                    .swarm
                    .behaviour_mut()
                    .rr
                    .send_request_with_addresses(&peer, req, addrs);
                self.requests.insert(id, tx);
            }
            Command::Provide(root) => self.provide(root),
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

    fn learn(&mut self, peer: PeerId, addr: Multiaddr) {
        self.addrs.entry(peer).or_default().insert(addr);
    }

    fn on_event(
        &mut self,
        event: SwarmEvent<BehaviourEvent>,
        resp_tx: &mpsc::UnboundedSender<(ResponseChannel<Response>, Response)>,
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
                peer_id, endpoint, ..
            } if !endpoint.is_relayed() => {
                if let Some((addr, listener @ None)) = self.relays.get_mut(&peer_id) {
                    *listener = self
                        .swarm
                        .listen_on(addr.clone().with(Protocol::P2pCircuit))
                        .ok();
                }
            }
            SwarmEvent::ListenerClosed { listener_id, .. } => {
                // Lost a relay reservation: ask again when next connected, and reconnect.
                for (addr, listener) in self.relays.values_mut() {
                    if *listener == Some(listener_id) {
                        *listener = None;
                        let _ = self.swarm.dial(addr.clone());
                    }
                }
            }
            SwarmEvent::Behaviour(BehaviourEvent::Identify(identify::Event::Received {
                peer_id,
                info,
                ..
            })) => {
                let dht = info.protocols.contains(&KAD_PROTOCOL);
                for addr in info.listen_addrs {
                    if dht {
                        self.swarm
                            .behaviour_mut()
                            .kad
                            .add_address(&peer_id, addr.clone());
                    }
                    self.learn(peer_id, addr);
                }
            }
            SwarmEvent::Behaviour(BehaviourEvent::Kad(ev)) => match ev {
                kad::Event::RoutablePeer { peer, address }
                | kad::Event::PendingRoutablePeer { peer, address } => self.learn(peer, address),
                kad::Event::OutboundQueryProgressed {
                    id, result, step, ..
                } => {
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
                request_response::Event::Message { message, .. } => match message {
                    request_response::Message::Request {
                        request, channel, ..
                    } => {
                        let (store, resp_tx) = (self.store.clone(), resp_tx.clone());
                        tokio::task::spawn_blocking(move || {
                            let _ = resp_tx.send((channel, answer(&store, request)));
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

/// Answer a peer's request from the store. Anything missing or malformed is `None`.
fn answer(store: &Store, req: Request) -> Response {
    match req {
        Request::Manifest(root) => Response::Manifest(
            store::is_hash(&root)
                .then(|| store.get_manifest_bytes(&root).ok())
                .flatten()
                .map(ByteBuf::from),
        ),
        Request::Signatures(root) => {
            Response::Signatures(store.signatures(&root).unwrap_or_default())
        }
        Request::Chunk(hash) => Response::Chunk(
            store::is_hash(&hash)
                .then(|| store.get(&hash).ok())
                .flatten()
                .map(ByteBuf::from),
        ),
    }
}

/// Download model `root` from whoever provides it on the DHT. As with a LAN fetch, the
/// manifest must hash to `root` and every chunk to its hash, and when `trust` is
/// non-empty one of those keys must have signed the manifest before any chunk is fetched.
pub async fn fetch(
    node: &Node,
    root: &str,
    store: Arc<Store>,
    trust: &[ed25519_dalek::VerifyingKey],
) -> Result<(Manifest, FetchStats)> {
    if !store::is_hash(root) {
        bail!("{root:?} is not a manifest root hash");
    }
    let started = Instant::now();
    let providers = node.providers(root).await?;

    let manifest = match store.get_manifest(root) {
        Ok(m) => m,
        Err(_) => {
            let mut found = None;
            for &p in &providers {
                if let Ok(Response::Manifest(Some(bytes))) =
                    node.request(p, Request::Manifest(root.to_string())).await
                    && let Ok(m) = serde_json::from_slice::<Manifest>(&bytes)
                    && m.root == root
                    && m.verify_root()
                {
                    found = Some(m);
                    break;
                }
            }
            match found {
                Some(m) => m,
                None if providers.is_empty() => bail!("nobody on the network is sharing {root}"),
                None => bail!("no provider sent a valid manifest for {root}"),
            }
        }
    };
    store.put_manifest(&manifest)?;

    let mut sigs = Vec::new();
    for &p in &providers {
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

    let mut stats = net::fetch_chunks_with(
        &store,
        manifest.files.iter().flat_map(|f| &f.chunks),
        &providers,
        &[],
        |peer, hash| {
            let node = node.clone();
            async move {
                match node.request(peer, Request::Chunk(hash)).await {
                    Ok(Response::Chunk(Some(b))) => Some(Bytes::from(b.into_vec())),
                    _ => None,
                }
            }
        },
    )
    .await?;
    stats.secs = started.elapsed().as_secs_f64();
    Ok((manifest, stats))
}
