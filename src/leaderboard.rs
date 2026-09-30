//! The verified node leaderboard: an opt-in ranking of nodes on numbers someone other
//! than the node can confirm.
//!
//! A node's own counters never rank, since a modified node can report anything. Instead:
//!
//! - **Probes.** Anchor nodes connect to each registered node on a schedule and ask for a
//!   random chunk of a model it claims to hold, then report the result to the registry,
//!   signed with their node key. This gives uptime and models held.
//! - **Receipts.** At the end of a fetch the downloader signs a [`Receipt`] for what each
//!   peer sent it, with the one-run key its lookup was bound to, and hands it to that
//!   peer, which submits it. This gives bytes and downloads served.
//! - **Counted lookups.** A receipt is credited only if its signer made a name lookup
//!   the registry counted (at most one per model, client network and UTC day), so every
//!   credited downloader costs a real network. Credit per downloader is capped at the
//!   model's `unique_bytes`.
//!
//! The registry keeps registrations in `nodes.json`, download counts in `downloads.json`
//! and the rest in `ledger.json`, and signs each day's totals once no more receipts can
//! arrive for it (`days/<date>.json`), so published numbers can't be quietly revised.
//! Client networks are only ever stored as hashes under a key that changes every UTC day
//! and is never written down.

use anyhow::{Context, Result, anyhow, bail};
use axum::Router;
use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use ed25519_dalek::SigningKey;
use futures::StreamExt;
use libp2p::PeerId;
use libp2p::identity::{Keypair, PublicKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::p2p::{self, Request, Response as P2pResponse};
use crate::registry::{self, AccessTicket, Registry, now};
use crate::sign::{self, Signature};
use crate::store::{self, Store};

const REGISTRATION_DOMAIN: &[u8] = b"chungus/node-registration/v1\0";
const RECEIPT_DOMAIN: &[u8] = b"chungus/receipt/v1\0";
const PROBE_DOMAIN: &[u8] = b"chungus/probe-report/v1\0";
const ANCHOR_ACCESS_DOMAIN: &[u8] = b"chungus/anchor-access/v1\0";
const HIDE_DOMAIN: &[u8] = b"chungus/hide-node/v1\0";
const TOTALS_DOMAIN: &[u8] = b"chungus/daily-totals/v1\0";

const DAY: u64 = 86400;
/// How far a signed message's time may be from the registry's clock.
const MAX_SKEW: u64 = 600;
/// A registered node silent this long is delisted.
pub const DELIST_SECS: u64 = 7 * DAY;
/// How often a node re-registers.
pub const REGISTER_EVERY: Duration = Duration::from_secs(DAY);
/// How long after a counted lookup its receipts are credited.
const LOOKUP_SECS: u64 = DAY;
/// Raw receipts are kept this long, for disputes.
const RECEIPT_DAYS: u64 = 7;
/// A model counts as held while it has a verified probe this recent.
const HELD_SECS: u64 = 7 * DAY;
/// Probe counts and daily totals are kept this long.
const HISTORY_DAYS: u64 = 90;
/// Uptime and the other metrics default to this window.
pub const DEFAULT_DAYS: u64 = 30;
pub const MAX_NAME: usize = 40;
/// Models one registration may claim. Keeps registrations under the proxy's body limit.
pub const MAX_MODELS: usize = 500;
/// Receipts per submission.
pub const MAX_RECEIPTS: usize = 100;
/// Probe results per report.
pub const MAX_RESULTS: usize = 500;
/// Registrations one client network may post per hour.
const REGISTRATIONS_PER_HOUR: u32 = 10;
/// Probe results from one anchor about one node closer together than this are dropped.
const MIN_PROBE_GAP: u64 = 60;
/// How often an anchor probes each registered node, before jitter.
pub const PROBE_EVERY: Duration = Duration::from_secs(300);

fn day_of(t: u64) -> u64 {
    t / DAY
}

/// `YYYY-MM-DD` for a day number (days since 1970-01-01).
pub fn date(day: u64) -> String {
    // Howard Hinnant's civil_from_days.
    let z = day as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The day number for `YYYY-MM-DD`.
pub fn parse_date(s: &str) -> Option<u64> {
    let mut parts = s.splitn(3, '-').map(|p| p.parse::<i64>().ok());
    let (y, m, d) = (parts.next()??, parts.next()??, parts.next()??);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // days_from_civil
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let day = era * 146_097 + doe - 719_468;
    let day = u64::try_from(day).ok()?;
    (date(day) == s).then_some(day)
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// The public key a peer id names. Every chungus key is ed25519, whose peer ids embed
/// the key itself.
pub fn public_key(peer: &PeerId) -> Option<PublicKey> {
    let mh: &libp2p::multihash::Multihash<64> = peer.as_ref();
    if mh.code() != 0 {
        return None;
    }
    PublicKey::try_decode_protobuf(mh.digest()).ok()
}

fn sign_as(key: &Keypair, domain: &[u8], body: &impl Serialize) -> String {
    let msg = [domain, &serde_json::to_vec(body).expect("serializes")].concat();
    sign::to_hex(&key.sign(&msg).expect("ed25519 keys can sign"))
}

/// Whether `sig` is `peer`'s signature over `body` under `domain`.
fn signed_by(peer: &str, domain: &[u8], body: &impl Serialize, sig: &str) -> bool {
    let Ok(peer) = peer.parse::<PeerId>() else {
        return false;
    };
    let (Some(key), Some(sig)) = (public_key(&peer), unhex(sig)) else {
        return false;
    };
    let msg = [domain, &serde_json::to_vec(body).expect("serializes")].concat();
    key.verify(&msg, &sig)
}

fn valid_peer(s: &str) -> bool {
    s.parse::<PeerId>().is_ok()
}

/// A display name: printable, trimmed, at most [`MAX_NAME`] characters.
pub fn valid_display_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().count() <= MAX_NAME
        && name.trim() == name
        && !name.chars().any(char::is_control)
}

// ---------- signed messages ----------

/// A node's request to be listed: its peer id, a display name, and the models it claims
/// to hold, signed with its node key.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Registration {
    pub peer: String,
    pub name: String,
    pub models: Vec<String>,
    /// Unix seconds.
    pub time: u64,
    pub sig: String,
}

impl Registration {
    pub fn new(key: &Keypair, name: &str, models: Vec<String>) -> Registration {
        let (peer, time) = (key.public().to_peer_id().to_string(), now());
        let sig = sign_as(key, REGISTRATION_DOMAIN, &(&peer, name, &models, time));
        Registration {
            peer,
            name: name.into(),
            models,
            time,
            sig,
        }
    }

    pub fn verify(&self) -> bool {
        signed_by(
            &self.peer,
            REGISTRATION_DOMAIN,
            &(&self.peer, &self.name, &self.models, self.time),
            &self.sig,
        )
    }
}

/// A downloader's signed statement of what one peer sent it for one model. `bytes` is
/// what arrived on the wire (compressed). Signed with the one-run key the downloader's
/// lookup was bound to, so the registry can match it to a counted lookup. Receipts are
/// cumulative: a later one for the same fetcher, server and model replaces an earlier one.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    pub root: String,
    pub server: String,
    pub bytes: u64,
    pub chunks: u64,
    pub fetcher: String,
    /// Unix seconds.
    pub time: u64,
    pub sig: String,
}

impl Receipt {
    pub fn new(key: &Keypair, root: &str, server: &PeerId, bytes: u64, chunks: u64) -> Receipt {
        Receipt::at(key, root, server, bytes, chunks, now())
    }

    /// A receipt signed as of `time`.
    pub fn at(
        key: &Keypair,
        root: &str,
        server: &PeerId,
        bytes: u64,
        chunks: u64,
        time: u64,
    ) -> Receipt {
        let (server, fetcher) = (server.to_string(), key.public().to_peer_id().to_string());
        let sig = sign_as(
            key,
            RECEIPT_DOMAIN,
            &(root, &server, bytes, chunks, &fetcher, time),
        );
        Receipt {
            root: root.into(),
            server,
            bytes,
            chunks,
            fetcher,
            time,
            sig,
        }
    }

    pub fn verify(&self) -> bool {
        signed_by(
            &self.fetcher,
            RECEIPT_DOMAIN,
            &(
                &self.root,
                &self.server,
                self.bytes,
                self.chunks,
                &self.fetcher,
                self.time,
            ),
            &self.sig,
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Sent the probed chunk, and it checked out: up, and holds the model.
    Held,
    /// Answered but sent no chunk (busy, or doesn't have it): up, proves nothing more.
    Up,
    /// No answer, or a wrong chunk.
    Down,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ProbeResult {
    pub node: String,
    pub root: String,
    pub outcome: Outcome,
}

/// An anchor's signed batch of probe results.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProbeReport {
    pub anchor: String,
    pub results: Vec<ProbeResult>,
    pub time: u64,
    pub sig: String,
}

impl ProbeReport {
    pub fn new(key: &Keypair, results: Vec<ProbeResult>) -> ProbeReport {
        let (anchor, time) = (key.public().to_peer_id().to_string(), now());
        let sig = sign_as(key, PROBE_DOMAIN, &(&anchor, &results, time));
        ProbeReport {
            anchor,
            results,
            time,
            sig,
        }
    }

    pub fn verify(&self) -> bool {
        signed_by(
            &self.anchor,
            PROBE_DOMAIN,
            &(&self.anchor, &self.results, self.time),
            &self.sig,
        )
    }
}

/// An anchor's request for an access ticket for its probing node (`prober`), so it can
/// probe nodes that hold gated models.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnchorAccess {
    pub anchor: String,
    pub prober: String,
    pub root: String,
    pub time: u64,
    pub sig: String,
}

impl AnchorAccess {
    pub fn new(key: &Keypair, prober: &PeerId, root: &str) -> AnchorAccess {
        let (anchor, prober, time) = (
            key.public().to_peer_id().to_string(),
            prober.to_string(),
            now(),
        );
        let sig = sign_as(key, ANCHOR_ACCESS_DOMAIN, &(&anchor, &prober, root, time));
        AnchorAccess {
            anchor,
            prober,
            root: root.into(),
            time,
            sig,
        }
    }

    pub fn verify(&self) -> bool {
        signed_by(
            &self.anchor,
            ANCHOR_ACCESS_DOMAIN,
            &(&self.anchor, &self.prober, &self.root, self.time),
            &self.sig,
        )
    }
}

/// The operator hiding (or showing again) a node's display name. Its stats stay.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hide {
    pub peer: String,
    pub hidden: bool,
    pub time: u64,
    pub signature: Signature,
}

impl Hide {
    pub fn new(key: &SigningKey, peer: &str, hidden: bool) -> Hide {
        let time = now();
        let msg = hide_message(peer, hidden, time);
        Hide {
            peer: peer.into(),
            hidden,
            time,
            signature: sign::sign_message(key, &msg),
        }
    }
}

fn hide_message(peer: &str, hidden: bool, time: u64) -> Vec<u8> {
    [HIDE_DOMAIN, format!("{peer}\0{hidden}\0{time}").as_bytes()].concat()
}

/// One node's numbers for one day.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeDay {
    /// Credited bytes served.
    pub bytes: u64,
    /// Distinct credited downloader networks served.
    pub downloads: u64,
    /// Probes, and how many were answered.
    pub probes: u64,
    pub answered: u64,
}

/// A day's totals, signed by the registry operator once no more receipts can arrive for
/// that day. Only registered nodes are named; `bytes_served` covers every node.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DayTotals {
    pub day: String,
    pub nodes: BTreeMap<String, NodeDay>,
    pub bytes_served: u64,
    /// Downloads the registry counted that day.
    pub downloads: u64,
    pub signature: Signature,
}

fn totals_message(
    day: &str,
    nodes: &BTreeMap<String, NodeDay>,
    bytes_served: u64,
    downloads: u64,
) -> Vec<u8> {
    let body = serde_json::to_vec(&(day, nodes, bytes_served, downloads)).expect("serializes");
    [TOTALS_DOMAIN, &body].concat()
}

impl DayTotals {
    pub fn verify(&self) -> bool {
        sign::verify_message(
            &self.signature,
            &totals_message(&self.day, &self.nodes, self.bytes_served, self.downloads),
        )
    }
}

// ---------- the registry's side ----------

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NodeRecord {
    name: String,
    models: Vec<String>,
    registered: u64,
    last_seen: u64,
    #[serde(default)]
    hidden: bool,
}

/// A counted lookup: what a one-run peer id looked up, from which (hashed) network, and
/// the most its receipts can be credited for.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Lookup {
    network: String,
    time: u64,
    cap: u64,
}

/// A receipt that matched a counted lookup.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Credit {
    receipt: Receipt,
    network: String,
    day: u64,
    cap: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct Probes {
    probes: u64,
    answered: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct Ledger {
    /// "<peer> <root>" -> lookup.
    lookups: BTreeMap<String, Lookup>,
    /// "<fetcher> <server> <root>" -> the latest credited receipt.
    credits: BTreeMap<String, Credit>,
    /// node -> day -> probe counts, for days not yet signed.
    probes: BTreeMap<String, BTreeMap<u64, Probes>>,
    /// node -> root -> when a chunk of it last checked out.
    held: BTreeMap<String, BTreeMap<String, u64>>,
    /// Downloads counted per day, for days not yet signed.
    counted: BTreeMap<u64, u64>,
}

/// A registered node, as the leaderboard and probers see it.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Listing {
    pub peer: String,
    /// None when the operator has hidden it.
    pub name: Option<String>,
    pub models: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Row {
    pub peer: String,
    pub name: Option<String>,
    pub bytes: u64,
    pub downloads: u64,
    /// Answered probes over probes, when there were any.
    pub uptime: Option<f64>,
    pub models: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct History {
    pub peer: String,
    pub name: Option<String>,
    pub registered: u64,
    pub last_seen: u64,
    /// Models with a verified probe in the last 7 days.
    pub held: Vec<String>,
    pub days: Vec<HistoryDay>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HistoryDay {
    pub day: String,
    #[serde(flatten)]
    pub totals: NodeDay,
    /// Whether the registry has signed this day's totals.
    pub signed: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Downloads {
    /// Counted downloads by manifest root.
    pub models: BTreeMap<String, u64>,
    pub downloads: u64,
    /// Credited bytes served, by every node, registered or not.
    pub bytes_served: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Bytes,
    Downloads,
    Uptime,
    Models,
}

/// The registry's leaderboard state.
pub struct Board {
    dir: PathBuf,
    nodes: BTreeMap<String, NodeRecord>,
    downloads: BTreeMap<String, u64>,
    ledger: Ledger,
    days: BTreeMap<u64, DayTotals>,
    /// The day the network key is for, and the key. Never written down.
    key_day: u64,
    day_key: [u8; 32],
    /// (root, network hash) counted today.
    counted_today: HashSet<(String, String)>,
    /// When each (anchor, node) pair last reported.
    last_probe: HashMap<(String, String), u64>,
    /// Registrations per network this hour.
    posts: HashMap<String, (u64, u32)>,
    dirty: bool,
}

fn read_json<T: serde::de::DeserializeOwned + Default>(path: &FsPath) -> Result<T> {
    match fs::read(path) {
        Ok(b) => serde_json::from_slice(&b).with_context(|| format!("{}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
    }
}

fn write_json(path: &FsPath, value: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(value)?)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// The client network an address belongs to: its IPv4 /24 or IPv6 /48.
pub fn network_of(ip: IpAddr) -> String {
    match ip {
        IpAddr::V6(v6) if v6.to_ipv4_mapped().is_some() => {
            network_of(IpAddr::V4(v6.to_ipv4_mapped().unwrap()))
        }
        IpAddr::V4(v4) => {
            let o = v4.octets();
            format!("{}.{}.{}.0/24", o[0], o[1], o[2])
        }
        IpAddr::V6(v6) => {
            let s = v6.segments();
            format!("{:x}:{:x}:{:x}::/48", s[0], s[1], s[2])
        }
    }
}

impl Board {
    pub fn open(dir: &FsPath) -> Result<Board> {
        let days_dir = dir.join("days");
        fs::create_dir_all(&days_dir)?;
        let mut days = BTreeMap::new();
        for e in fs::read_dir(&days_dir)? {
            let path = e?.path();
            let Some(day) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(parse_date)
            else {
                continue;
            };
            if path.extension().is_some_and(|x| x == "json") {
                let t: DayTotals = serde_json::from_slice(&fs::read(&path)?)
                    .with_context(|| format!("{}", path.display()))?;
                days.insert(day, t);
            }
        }
        let mut day_key = [0u8; 32];
        getrandom::fill(&mut day_key).map_err(|e| anyhow!("no randomness: {e}"))?;
        Ok(Board {
            nodes: read_json(&dir.join("nodes.json"))?,
            downloads: read_json(&dir.join("downloads.json"))?,
            ledger: read_json(&dir.join("ledger.json"))?,
            days,
            dir: dir.to_path_buf(),
            key_day: day_of(now()),
            day_key,
            counted_today: HashSet::new(),
            last_probe: HashMap::new(),
            posts: HashMap::new(),
            dirty: false,
        })
    }

    /// Write whatever changed since the last save.
    pub fn save(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        write_json(&self.dir.join("nodes.json"), &self.nodes)?;
        write_json(&self.dir.join("downloads.json"), &self.downloads)?;
        write_json(&self.dir.join("ledger.json"), &self.ledger)?;
        self.dirty = false;
        Ok(())
    }

    /// `network` hashed under today's key. A new day gets a new random key, and the old
    /// one is forgotten, so hashes from different days can't be linked.
    fn network_hash(&mut self, t: u64, network: &str) -> String {
        if day_of(t) != self.key_day {
            self.key_day = day_of(t);
            getrandom::fill(&mut self.day_key).expect("randomness");
            self.counted_today.clear();
        }
        blake3::keyed_hash(&self.day_key, network.as_bytes())
            .to_hex()
            .to_string()
    }

    /// Count a download lookup of `root` from `network`, at most once per network and
    /// day. When counted and `peer` names the lookup's one-run peer id, that peer's
    /// receipts for `root` can be credited, up to `cap` bytes. Returns whether it counted.
    pub fn count_lookup(
        &mut self,
        t: u64,
        root: &str,
        network: &str,
        peer: Option<&str>,
        cap: u64,
    ) -> bool {
        let hash = self.network_hash(t, network);
        if !self.counted_today.insert((root.to_string(), hash.clone())) {
            return false;
        }
        *self.downloads.entry(root.to_string()).or_default() += 1;
        *self.ledger.counted.entry(day_of(t)).or_default() += 1;
        // A peer id binds once: a later lookup naming someone else's peer id can't take
        // over its receipts.
        if let Some(peer) = peer.filter(|p| valid_peer(p)) {
            self.ledger
                .lookups
                .entry(format!("{peer} {root}"))
                .or_insert(Lookup {
                    network: hash,
                    time: t,
                    cap,
                });
        }
        self.dirty = true;
        true
    }

    /// Register or refresh a node. `network` is the poster's network, for rate limiting.
    pub fn register(&mut self, r: Registration, network: Option<&str>, t: u64) -> Result<()> {
        if !r.verify() {
            bail!("bad signature");
        }
        if r.time.abs_diff(t) > MAX_SKEW {
            bail!("registration time is too far from the registry's clock");
        }
        if !valid_display_name(&r.name) {
            bail!("the display name must be 1 to {MAX_NAME} printable characters");
        }
        if r.models.len() > MAX_MODELS || !r.models.iter().all(|m| store::is_hash(m)) {
            bail!("claim at most {MAX_MODELS} models, each a manifest root");
        }
        if let Some(net) = network {
            let hour = t / 3600;
            if self.posts.len() > 100_000 {
                self.posts.retain(|_, (h, _)| *h == hour);
            }
            let (h, n) = self.posts.entry(net.to_string()).or_insert((hour, 0));
            if *h != hour {
                (*h, *n) = (hour, 0);
            }
            if *n >= REGISTRATIONS_PER_HOUR {
                bail!("too many registrations from this network; try again later");
            }
            *n += 1;
        }
        let mut models = r.models;
        models.sort();
        models.dedup();
        let rec = self.nodes.entry(r.peer).or_insert(NodeRecord {
            name: String::new(),
            models: Vec::new(),
            registered: t,
            last_seen: t,
            hidden: false,
        });
        rec.name = r.name;
        rec.models = models;
        rec.last_seen = t;
        self.dirty = true;
        Ok(())
    }

    pub fn hide(&mut self, peer: &str, hidden: bool) -> Result<()> {
        let rec = self.nodes.get_mut(peer).context("no such node")?;
        rec.hidden = hidden;
        self.dirty = true;
        Ok(())
    }

    fn active(&self, t: u64) -> impl Iterator<Item = (&String, &NodeRecord)> {
        self.nodes
            .iter()
            .filter(move |(_, r)| t.saturating_sub(r.last_seen) < DELIST_SECS)
    }

    /// Registered nodes that haven't gone silent.
    pub fn listings(&self, t: u64) -> Vec<Listing> {
        self.active(t)
            .map(|(peer, r)| Listing {
                peer: peer.clone(),
                name: (!r.hidden).then(|| r.name.clone()),
                models: r.models.clone(),
            })
            .collect()
    }

    /// Take an anchor's probe results. `anchors` are the anchors' peer ids. Returns how
    /// many results were counted.
    pub fn probes(&mut self, rep: ProbeReport, anchors: &HashSet<String>, t: u64) -> Result<usize> {
        if !anchors.contains(&rep.anchor) {
            bail!("{} is not one of the registry's anchors", rep.anchor);
        }
        if !rep.verify() {
            bail!("bad signature");
        }
        if rep.time.abs_diff(t) > MAX_SKEW {
            bail!("report time is too far from the registry's clock");
        }
        if rep.results.len() > MAX_RESULTS {
            bail!("at most {MAX_RESULTS} results per report");
        }
        let mut counted = 0;
        for r in rep.results {
            let Some(rec) = self.nodes.get(&r.node) else {
                continue;
            };
            let held = r.outcome == Outcome::Held;
            if held && !rec.models.contains(&r.root) {
                continue;
            }
            let last = self
                .last_probe
                .entry((rep.anchor.clone(), r.node.clone()))
                .or_insert(0);
            if t.saturating_sub(*last) < MIN_PROBE_GAP {
                continue;
            }
            *last = t;
            let p = self
                .ledger
                .probes
                .entry(r.node.clone())
                .or_default()
                .entry(day_of(t))
                .or_default();
            p.probes += 1;
            if r.outcome != Outcome::Down {
                p.answered += 1;
            }
            if held {
                self.ledger
                    .held
                    .entry(r.node)
                    .or_default()
                    .insert(r.root, t);
            }
            counted += 1;
        }
        self.dirty = true;
        Ok(counted)
    }

    /// Check receipts and keep those that match a counted lookup. Returns how many were
    /// taken.
    pub fn receipts(&mut self, receipts: Vec<Receipt>, t: u64) -> Result<usize> {
        if receipts.len() > MAX_RECEIPTS {
            bail!("at most {MAX_RECEIPTS} receipts per request");
        }
        let mut taken = 0;
        for r in receipts {
            // Cheapest checks first, so a flood costs little.
            let Some(lookup) = self
                .ledger
                .lookups
                .get(&format!("{} {}", r.fetcher, r.root))
            else {
                continue;
            };
            if t.saturating_sub(lookup.time) > LOOKUP_SECS
                || r.time > t + MAX_SKEW
                || r.time + MAX_SKEW < lookup.time
                || r.server == r.fetcher
                || !valid_peer(&r.server)
                || lookup.cap == 0
                || !r.verify()
            {
                continue;
            }
            let key = format!("{} {} {}", r.fetcher, r.server, r.root);
            if self
                .ledger
                .credits
                .get(&key)
                .is_some_and(|c| c.receipt.time >= r.time)
            {
                continue;
            }
            let credit = Credit {
                network: lookup.network.clone(),
                day: day_of(lookup.time),
                cap: lookup.cap,
                receipt: r,
            };
            self.ledger.credits.insert(key, credit);
            taken += 1;
        }
        if taken > 0 {
            self.dirty = true;
        }
        Ok(taken)
    }

    /// Every node's numbers for `day`, from its signed totals or else from the ledger.
    /// Includes unregistered nodes when computed from the ledger.
    fn day(&self, day: u64) -> BTreeMap<String, NodeDay> {
        if let Some(t) = self.days.get(&day) {
            return t.nodes.clone();
        }
        let mut out: BTreeMap<String, NodeDay> = BTreeMap::new();
        // Each downloader's total claim, to scale it down to the cap.
        let mut claimed: HashMap<(&str, &str), u64> = HashMap::new();
        for c in self.ledger.credits.values().filter(|c| c.day == day) {
            *claimed
                .entry((c.receipt.fetcher.as_str(), c.receipt.root.as_str()))
                .or_default() += c.receipt.bytes;
        }
        let mut networks: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for c in self.ledger.credits.values().filter(|c| c.day == day) {
            let total = claimed[&(c.receipt.fetcher.as_str(), c.receipt.root.as_str())];
            let bytes = if total > c.cap {
                (u128::from(c.receipt.bytes) * u128::from(c.cap) / u128::from(total)) as u64
            } else {
                c.receipt.bytes
            };
            if bytes == 0 {
                continue;
            }
            out.entry(c.receipt.server.clone()).or_default().bytes += bytes;
            networks
                .entry(c.receipt.server.as_str())
                .or_default()
                .insert(c.network.as_str());
        }
        for (server, nets) in networks {
            out.get_mut(server).unwrap().downloads = nets.len() as u64;
        }
        for (node, days) in &self.ledger.probes {
            if let Some(p) = days.get(&day) {
                let d = out.entry(node.clone()).or_default();
                d.probes = p.probes;
                d.answered = p.answered;
            }
        }
        out
    }

    fn held(&self, peer: &str, t: u64) -> Vec<String> {
        self.ledger
            .held
            .get(peer)
            .map(|h| {
                h.iter()
                    .filter(|(_, at)| t.saturating_sub(**at) < HELD_SECS)
                    .map(|(r, _)| r.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Registered nodes ranked by `metric` over the last `days` days.
    pub fn leaderboard(&self, metric: Metric, days: u64, t: u64) -> Vec<Row> {
        let today = day_of(t);
        let range: Vec<BTreeMap<String, NodeDay>> = (0..days.clamp(1, HISTORY_DAYS))
            .filter_map(|i| today.checked_sub(i))
            .map(|d| self.day(d))
            .collect();
        let mut rows: Vec<Row> = self
            .active(t)
            .map(|(peer, rec)| {
                let mut sum = NodeDay::default();
                for d in range.iter().filter_map(|d| d.get(peer)) {
                    sum.bytes += d.bytes;
                    sum.downloads += d.downloads;
                    sum.probes += d.probes;
                    sum.answered += d.answered;
                }
                Row {
                    peer: peer.clone(),
                    name: (!rec.hidden).then(|| rec.name.clone()),
                    bytes: sum.bytes,
                    downloads: sum.downloads,
                    uptime: (sum.probes > 0).then(|| sum.answered as f64 / sum.probes as f64),
                    models: self.held(peer, t).len() as u64,
                }
            })
            .collect();
        let key = |r: &Row| -> (f64, u64) {
            match metric {
                Metric::Bytes => (r.bytes as f64, r.downloads),
                Metric::Downloads => (r.downloads as f64, r.bytes),
                Metric::Uptime => (r.uptime.unwrap_or(-1.0), r.models),
                Metric::Models => (r.models as f64, r.bytes),
            }
        };
        rows.sort_by(|a, b| {
            key(b)
                .partial_cmp(&key(a))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.peer.cmp(&b.peer))
        });
        rows
    }

    /// One registered node's daily history, newest first.
    pub fn history(&self, peer: &str, t: u64) -> Option<History> {
        let rec = self.nodes.get(peer)?;
        let today = day_of(t);
        let first = day_of(rec.registered);
        let days = (0..HISTORY_DAYS)
            .filter_map(|i| today.checked_sub(i))
            .take_while(|d| *d >= first)
            .map(|d| HistoryDay {
                day: date(d),
                totals: self.day(d).remove(peer).unwrap_or_default(),
                signed: self.days.contains_key(&d),
            })
            .collect();
        Some(History {
            peer: peer.into(),
            name: (!rec.hidden).then(|| rec.name.clone()),
            registered: rec.registered,
            last_seen: rec.last_seen,
            held: self.held(peer, t),
            days,
        })
    }

    pub fn downloads(&self, t: u64) -> Downloads {
        let today = day_of(t);
        let signed: u64 = self.days.values().map(|d| d.bytes_served).sum();
        let live: u64 = self
            .ledger
            .credits
            .values()
            .map(|c| c.day)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|d| !self.days.contains_key(d) && *d <= today)
            .map(|d| self.day(d).values().map(|n| n.bytes).sum::<u64>())
            .sum();
        Downloads {
            downloads: self.downloads.values().sum(),
            models: self.downloads.clone(),
            bytes_served: signed + live,
        }
    }

    pub fn totals(&self, day: u64) -> Option<&DayTotals> {
        self.days.get(&day)
    }

    pub fn signed_days(&self) -> Vec<String> {
        self.days.keys().map(|d| date(*d)).collect()
    }

    /// Sign every day that no more receipts can arrive for (its lookups have expired), and
    /// drop what is no longer needed: expired lookups, receipts older than a week, and
    /// probe counts of signed days. Without `key` (an offline root key that hasn't
    /// delegated) days stay unsigned until one is available.
    pub fn finalize(&mut self, t: u64, key: Option<&SigningKey>) -> Result<()> {
        let today = day_of(t);
        if let Some(key) = key {
            let mut pending: BTreeSet<u64> = self.ledger.credits.values().map(|c| c.day).collect();
            pending.extend(self.ledger.probes.values().flat_map(|d| d.keys().copied()));
            pending.extend(self.ledger.counted.keys().copied());
            for day in pending {
                // Lookups on `day` are credited until a day later, and receipts may take
                // a little while to be forwarded.
                if day + 2 > today || self.days.contains_key(&day) {
                    continue;
                }
                let all = self.day(day);
                let bytes_served = all.values().map(|n| n.bytes).sum();
                let nodes: BTreeMap<String, NodeDay> = all
                    .into_iter()
                    .filter(|(peer, _)| self.nodes.contains_key(peer))
                    .collect();
                let (d, downloads) = (
                    date(day),
                    self.ledger.counted.get(&day).copied().unwrap_or(0),
                );
                let signature =
                    sign::sign_message(key, &totals_message(&d, &nodes, bytes_served, downloads));
                let totals = DayTotals {
                    day: d.clone(),
                    nodes,
                    bytes_served,
                    downloads,
                    signature,
                };
                write_json(&self.dir.join("days").join(format!("{d}.json")), &totals)?;
                self.days.insert(day, totals);
                self.dirty = true;
            }
        }
        let before = (
            self.ledger.lookups.len(),
            self.ledger.credits.len(),
            self.ledger.held.values().map(BTreeMap::len).sum::<usize>(),
        );
        self.ledger
            .lookups
            .retain(|_, l| t.saturating_sub(l.time) <= LOOKUP_SECS + DAY);
        let days = &self.days;
        self.ledger
            .credits
            .retain(|_, c| !(days.contains_key(&c.day) && c.day + RECEIPT_DAYS < today));
        for d in self.ledger.probes.values_mut() {
            d.retain(|day, _| !days.contains_key(day));
        }
        self.ledger.probes.retain(|_, d| !d.is_empty());
        self.ledger.counted.retain(|day, _| !days.contains_key(day));
        for h in self.ledger.held.values_mut() {
            h.retain(|_, at| t.saturating_sub(*at) < HELD_SECS);
        }
        self.ledger.held.retain(|_, h| !h.is_empty());
        let old = today.saturating_sub(HISTORY_DAYS);
        self.days.retain(|d, _| *d >= old);
        // Nodes silent for longer than any history shown are forgotten.
        let nodes = self.nodes.len();
        self.nodes
            .retain(|_, r| t.saturating_sub(r.last_seen) < HISTORY_DAYS * DAY);
        let last_probe = &mut self.last_probe;
        last_probe.retain(|_, at| t.saturating_sub(*at) < DAY);
        if nodes != self.nodes.len() {
            self.dirty = true;
        }
        let after = (
            self.ledger.lookups.len(),
            self.ledger.credits.len(),
            self.ledger.held.values().map(BTreeMap::len).sum::<usize>(),
        );
        if before != after {
            self.dirty = true;
        }
        Ok(())
    }
}

// ---------- HTTP ----------

fn err(code: StatusCode, e: impl std::fmt::Display) -> Response {
    (code, e.to_string()).into_response()
}

/// The client's IP address: the connection's, or behind a reverse proxy the last
/// `X-Forwarded-For` entry (the one the proxy itself added).
pub(crate) fn client_ip(
    reg: &Registry,
    headers: &HeaderMap,
    conn: Option<&ConnectInfo<SocketAddr>>,
) -> Option<IpAddr> {
    if reg.behind_proxy() {
        headers
            .get_all("x-forwarded-for")
            .iter()
            .next_back()?
            .to_str()
            .ok()?
            .rsplit(',')
            .next()?
            .trim()
            .parse()
            .ok()
    } else {
        conn.map(|c| c.0.ip())
    }
}

type Conn = Option<axum::Extension<ConnectInfo<SocketAddr>>>;

/// The leaderboard's routes, merged into [`registry::router`].
pub(crate) fn routes() -> Router<Arc<Registry>> {
    Router::new()
        .route("/v1/nodes", post(register).get(nodes))
        .route("/v1/nodes/{peer}", get(node))
        .route("/v1/nodes/hide", post(hide))
        .route("/v1/receipts", post(receipts))
        .route("/v1/probes", post(probes))
        .route("/v1/anchor-access", post(anchor_access))
        .route("/v1/leaderboard", get(leaderboard))
        .route("/v1/downloads", get(downloads))
        .route("/v1/totals", get(signed_days))
        .route("/v1/totals/{day}", get(totals))
}

async fn register(
    State(reg): State<Arc<Registry>>,
    headers: HeaderMap,
    conn: Conn,
    axum::Json(r): axum::Json<Registration>,
) -> Response {
    let network = client_ip(&reg, &headers, conn.as_ref().map(|c| &c.0)).map(network_of);
    match reg.board(|b| b.register(r, network.as_deref(), now())) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err(StatusCode::FORBIDDEN, format!("{e:#}")),
    }
}

async fn nodes(State(reg): State<Arc<Registry>>) -> Response {
    axum::Json(reg.board(|b| b.listings(now()))).into_response()
}

async fn node(State(reg): State<Arc<Registry>>, Path(peer): Path<String>) -> Response {
    match reg.board(|b| b.history(&peer, now())) {
        Some(h) => axum::Json(h).into_response(),
        None => err(StatusCode::NOT_FOUND, format!("{peer} is not registered")),
    }
}

async fn hide(State(reg): State<Arc<Registry>>, axum::Json(h): axum::Json<Hide>) -> Response {
    let t = now();
    let ok = h.time.abs_diff(t) <= MAX_SKEW
        && sign::verify_message(&h.signature, &hide_message(&h.peer, h.hidden, h.time))
        && reg.with_log(|log| log.acts_as_operator(&h.signature.key, h.time));
    if !ok {
        return err(
            StatusCode::FORBIDDEN,
            "only the registry operator can hide a node",
        );
    }
    match reg.board(|b| b.hide(&h.peer, h.hidden)) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e),
    }
}

async fn receipts(
    State(reg): State<Arc<Registry>>,
    axum::Json(rs): axum::Json<Vec<Receipt>>,
) -> Response {
    // Signature checks are the costly part; keep them off the async workers.
    let r = tokio::task::spawn_blocking(move || reg.board(|b| b.receipts(rs, now()))).await;
    match r {
        Ok(Ok(n)) => axum::Json(serde_json::json!({ "credited": n })).into_response(),
        Ok(Err(e)) => err(StatusCode::BAD_REQUEST, e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn probes(
    State(reg): State<Arc<Registry>>,
    axum::Json(rep): axum::Json<ProbeReport>,
) -> Response {
    let anchors = reg.anchor_peers();
    match reg.board(|b| b.probes(rep, &anchors, now())) {
        Ok(n) => axum::Json(serde_json::json!({ "counted": n })).into_response(),
        Err(e) => err(StatusCode::FORBIDDEN, format!("{e:#}")),
    }
}

async fn anchor_access(
    State(reg): State<Arc<Registry>>,
    axum::Json(req): axum::Json<AnchorAccess>,
) -> Response {
    let bad = |e: &str| err(StatusCode::FORBIDDEN, e);
    if !reg.anchor_peers().contains(&req.anchor) {
        return bad("only the registry's anchors can ask for probe tickets");
    }
    if !valid_peer(&req.prober) || req.time.abs_diff(now()) > MAX_SKEW || !req.verify() {
        return bad("bad request signature");
    }
    let Some(repo) = reg.with_log(|log| log.gate(&req.root).map(str::to_string)) else {
        return err(StatusCode::BAD_REQUEST, "that model isn't gated");
    };
    let Some(key) = reg.operator_key() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "this registry's online key isn't delegated to, so it can't issue tickets",
        );
    };
    axum::Json(AccessTicket::new(
        key,
        &repo,
        &req.prober,
        now() + registry::TICKET_SECS,
    ))
    .into_response()
}

#[derive(Deserialize)]
struct BoardQuery {
    metric: Option<Metric>,
    days: Option<u64>,
}

async fn leaderboard(State(reg): State<Arc<Registry>>, Query(q): Query<BoardQuery>) -> Response {
    let metric = q.metric.unwrap_or(Metric::Bytes);
    let days = q.days.unwrap_or(DEFAULT_DAYS);
    axum::Json(reg.board(|b| b.leaderboard(metric, days, now()))).into_response()
}

async fn downloads(State(reg): State<Arc<Registry>>) -> Response {
    axum::Json(reg.board(|b| b.downloads(now()))).into_response()
}

async fn signed_days(State(reg): State<Arc<Registry>>) -> Response {
    axum::Json(reg.board(|b| b.signed_days())).into_response()
}

async fn totals(State(reg): State<Arc<Registry>>, Path(day): Path<String>) -> Response {
    match parse_date(&day).and_then(|d| reg.board(|b| b.totals(d).cloned())) {
        Some(t) => axum::Json(t).into_response(),
        None => err(StatusCode::NOT_FOUND, format!("no signed totals for {day}")),
    }
}

// ---------- the node's side ----------

/// Re-register `name` with the registry every [`REGISTER_EVERY`], claiming the models
/// in `store`.
pub fn register_loop(
    registry: String,
    key: Keypair,
    name: String,
    store: Arc<Store>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let models: Vec<String> = store
                .manifests()
                .unwrap_or_default()
                .into_iter()
                .take(MAX_MODELS)
                .collect();
            let n = models.len();
            let r = Registration::new(&key, &name, models);
            let wait = match registry::Client::new(&registry) {
                Ok(c) => match c.register(&r).await {
                    Ok(()) => {
                        println!("leaderboard: registered as {name:?} with {n} model(s)");
                        REGISTER_EVERY
                    }
                    Err(e) => {
                        eprintln!("leaderboard: registering failed: {e:#}");
                        Duration::from_secs(600)
                    }
                },
                Err(e) => {
                    eprintln!("leaderboard: {e:#}");
                    Duration::from_secs(600)
                }
            };
            tokio::time::sleep(wait).await;
        }
    })
}

/// Keep receipts downloaders hand this node, as proof of its work (the latest per
/// downloader and model, for a week, in `proof`), and submit them to the registry.
pub fn forward_receipts(
    registry: String,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Receipt>,
    proof: PathBuf,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut kept: BTreeMap<String, Receipt> = read_json(&proof).unwrap_or_default();
        let mut queue: Vec<Receipt> = Vec::new();
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        loop {
            tokio::select! {
                r = rx.recv() => match r {
                    Some(r) => {
                        let key = format!("{} {}", r.fetcher, r.root);
                        if kept.get(&key).is_none_or(|k| k.time < r.time) {
                            kept.insert(key, r.clone());
                            queue.push(r);
                        }
                    }
                    None => return,
                },
                _ = tick.tick() => {
                    if queue.is_empty() {
                        continue;
                    }
                    let t = now();
                    kept.retain(|_, r| t.saturating_sub(r.time) < RECEIPT_DAYS * DAY);
                    if let Err(e) = write_json(&proof, &kept) {
                        eprintln!("receipts: can't save {}: {e:#}", proof.display());
                    }
                    let Ok(client) = registry::Client::new(&registry) else {
                        continue;
                    };
                    let mut failed = Vec::new();
                    for batch in queue.chunks(MAX_RECEIPTS) {
                        if let Err(e) = client.receipts(batch).await {
                            eprintln!("receipts: submitting to {registry} failed: {e:#}");
                            failed.extend_from_slice(batch);
                        }
                    }
                    // Keep what failed for the next round, but not forever.
                    failed.retain(|r| t.saturating_sub(r.time) < LOOKUP_SECS);
                    queue = failed;
                }
            }
        }
    })
}

/// Probe every registered node every [`PROBE_EVERY`] (with jitter) from `prober`, a
/// download-only node with its own identity, and report the results signed as `anchor`.
pub fn probe_loop(
    registry: String,
    anchor: Keypair,
    prober: p2p::Node,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut prober = Prober::new(registry, anchor, prober);
        loop {
            match prober.round().await {
                Ok((n, counted)) if n > 0 => {
                    println!("probes: {counted} of {n} result(s) counted")
                }
                Ok(_) => {}
                Err(e) => eprintln!("probes: {e:#}"),
            }
            // Up to a fifth either way, so nodes can't learn when the next probe comes.
            let base = PROBE_EVERY.as_millis() as u64;
            let jitter = random_below(base * 2 / 5);
            tokio::time::sleep(Duration::from_millis(base * 4 / 5 + jitter)).await;
        }
    })
}

fn random_below(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut b = [0u8; 8];
    getrandom::fill(&mut b).expect("randomness");
    u64::from_le_bytes(b) % n
}

/// Probes nodes for one anchor, caching manifests and gate tickets.
pub struct Prober {
    registry: String,
    anchor: Keypair,
    node: p2p::Node,
    manifests: HashMap<String, Arc<crate::manifest::Manifest>>,
    /// Roots the registry has no manifest for, so they can't be probed.
    unknown: HashSet<String>,
    tickets: HashMap<String, AccessTicket>,
}

impl Prober {
    pub fn new(registry: String, anchor: Keypair, node: p2p::Node) -> Prober {
        Prober {
            registry,
            anchor,
            node,
            manifests: HashMap::new(),
            unknown: HashSet::new(),
            tickets: HashMap::new(),
        }
    }

    /// Probe every registered node once and report. Returns (results, counted).
    pub async fn round(&mut self) -> Result<(usize, usize)> {
        let client = registry::Client::new(&self.registry)?;
        let nodes = client.nodes().await?;
        let mut jobs = Vec::new();
        for n in nodes {
            let Ok(peer) = n.peer.parse::<PeerId>() else {
                continue;
            };
            if peer == self.node.peer_id || peer == self.anchor.public().to_peer_id() {
                continue;
            }
            let models: Vec<&String> = n
                .models
                .iter()
                .filter(|m| !self.unknown.contains(*m))
                .collect();
            if models.is_empty() {
                continue;
            }
            let root = models[random_below(models.len() as u64) as usize].clone();
            let manifest = match self.manifests.get(&root) {
                Some(m) => m.clone(),
                None => match client.manifest(&root).await {
                    Ok(m) => {
                        let m = Arc::new(m);
                        self.manifests.insert(root.clone(), m.clone());
                        m
                    }
                    Err(_) => {
                        self.unknown.insert(root);
                        continue;
                    }
                },
            };
            jobs.push((peer, root, manifest));
        }
        let this = &*self;
        let results: Vec<(ProbeResult, Option<(String, AccessTicket)>)> =
            futures::stream::iter(jobs)
                .map(|(peer, root, m)| async move {
                    let (outcome, ticket) = this.probe(peer, &root, &m).await;
                    let node = peer.to_string();
                    (
                        ProbeResult {
                            node,
                            root,
                            outcome,
                        },
                        ticket,
                    )
                })
                .buffer_unordered(16)
                .collect()
                .await;
        let mut report = Vec::new();
        for (r, ticket) in results {
            if let Some((repo, t)) = ticket {
                self.tickets.insert(repo, t);
            }
            report.push(r);
        }
        if report.is_empty() {
            return Ok((0, 0));
        }
        let n = report.len();
        let mut counted = 0;
        for batch in report.chunks(MAX_RESULTS) {
            counted += client
                .probes(&ProbeReport::new(&self.anchor, batch.to_vec()))
                .await?;
        }
        Ok((n, counted))
    }

    /// Ask `peer` for a random chunk of `m`. Also returns a new gate ticket, if one had to
    /// be fetched.
    async fn probe(
        &self,
        peer: PeerId,
        root: &str,
        m: &crate::manifest::Manifest,
    ) -> (Outcome, Option<(String, AccessTicket)>) {
        let chunks: Vec<&crate::manifest::ChunkRef> =
            m.files.iter().flat_map(|f| &f.chunks).collect();
        if chunks.is_empty() {
            return (Outcome::Up, None);
        }
        let c = chunks[random_below(chunks.len() as u64) as usize];
        self.node.find(peer).await;
        let ask = || self.node.request(peer, Request::Chunk(c.hash.clone()));
        let mut new_ticket = None;
        let mut resp = ask().await;
        if let Ok(P2pResponse::Gated(repo)) = &resp {
            let ticket = match self.tickets.get(repo) {
                Some(t) if t.expires > now() + 60 => Some(t.clone()),
                _ => {
                    let req = AnchorAccess::new(&self.anchor, &self.node.peer_id, root);
                    let got = match registry::Client::new(&self.registry) {
                        Ok(c) => c.anchor_access(&req).await.ok(),
                        Err(_) => None,
                    };
                    if let Some(t) = &got {
                        new_ticket = Some((repo.clone(), t.clone()));
                    }
                    got
                }
            };
            // Without a ticket it can't get past the gate: it answered, which is all
            // this shows.
            let Some(t) = ticket else {
                return (Outcome::Up, new_ticket);
            };
            if !matches!(
                self.node.request(peer, Request::Access(t)).await,
                Ok(P2pResponse::Granted(true))
            ) {
                return (Outcome::Up, new_ticket);
            }
            resp = ask().await;
        }
        let outcome = match resp {
            Ok(P2pResponse::Chunk(Some(blob))) => {
                let ok = store::decode(&blob, c.len as usize)
                    .is_ok_and(|raw| blake3::hash(&raw).to_hex().as_str() == c.hash);
                if ok { Outcome::Held } else { Outcome::Down }
            }
            Ok(P2pResponse::Chunk(None) | P2pResponse::Busy | P2pResponse::Gated(_)) => Outcome::Up,
            _ => Outcome::Down,
        };
        (outcome, new_ticket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(i: u8) -> String {
        format!("{i:02x}").repeat(32)
    }

    #[test]
    fn dates_round_trip() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(20361), "2025-09-30");
        for d in [0, 59, 60, 365, 11016, 20361, 30000] {
            assert_eq!(parse_date(&date(d)), Some(d));
        }
        assert_eq!(parse_date("2025-02-30"), None);
        assert_eq!(parse_date("nonsense"), None);
    }

    #[test]
    fn networks() {
        assert_eq!(network_of("203.0.113.9".parse().unwrap()), "203.0.113.0/24");
        assert_eq!(
            network_of("::ffff:203.0.113.9".parse().unwrap()),
            "203.0.113.0/24"
        );
        assert_eq!(
            network_of("2001:db8:1234:5678::1".parse().unwrap()),
            "2001:db8:1234::/48"
        );
    }

    #[test]
    fn signed_messages_bind_their_signer() {
        let k = Keypair::generate_ed25519();
        let server = PeerId::random();
        let r = Receipt::new(&k, &root(1), &server, 100, 2);
        assert!(r.verify());
        let mut forged = r.clone();
        forged.bytes = 1000;
        assert!(!forged.verify());
        // Claiming to be another fetcher fails, since the peer id names the key.
        let other = Keypair::generate_ed25519();
        let mut stolen = r.clone();
        stolen.fetcher = other.public().to_peer_id().to_string();
        assert!(!stolen.verify());

        let reg = Registration::new(&k, "bunny hq", vec![root(1)]);
        assert!(reg.verify());
        let mut renamed = reg.clone();
        renamed.name = "someone else".into();
        assert!(!renamed.verify());
    }

    #[test]
    fn receipts_need_a_counted_lookup_and_are_capped() {
        let dir = tempfile::tempdir().unwrap();
        let mut b = Board::open(dir.path()).unwrap();
        let t = 100 * DAY + 3600;
        let (a, s1, s2) = (
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
        );
        let (s1p, s2p) = (s1.public().to_peer_id(), s2.public().to_peer_id());
        b.register(Registration::new(&s1, "one", vec![root(1)]), None, now())
            .unwrap();
        let fetcher = Keypair::generate_ed25519();
        let fp = fetcher.public().to_peer_id().to_string();

        // Without a lookup, nothing is credited.
        let r1 = Receipt::at(&fetcher, &root(1), &s1p, 600, 3, t);
        assert_eq!(b.receipts(vec![r1.clone()], t).unwrap(), 0);

        assert!(b.count_lookup(t, &root(1), "198.51.100.0/24", Some(&fp), 1000));
        // The same network on the same day isn't counted again, nor bound.
        let again = Keypair::generate_ed25519()
            .public()
            .to_peer_id()
            .to_string();
        assert!(!b.count_lookup(t, &root(1), "198.51.100.0/24", Some(&again), 1000));

        let r2 = Receipt::at(&fetcher, &root(1), &s2p, 900, 4, t);
        assert_eq!(b.receipts(vec![r1.clone(), r2, r1.clone()], t).unwrap(), 2);
        // A receipt signed by someone else for this fetcher isn't taken.
        let mut fake = Receipt::at(&a, &root(1), &s1p, 10_000, 3, t);
        fake.fetcher = fp.clone();
        assert_eq!(b.receipts(vec![fake], t).unwrap(), 0);

        // 1500 bytes claimed against a cap of 1000: scaled down proportionally.
        let day = b.day(day_of(t));
        assert_eq!(day[&s1p.to_string()].bytes, 400);
        assert_eq!(day[&s2p.to_string()].bytes, 600);
        assert_eq!(day[&s1p.to_string()].downloads, 1);

        // Receipts come too late once the lookup is over a day old.
        let late = Receipt::at(&fetcher, &root(1), &s1p, 800, 4, t + DAY + 60);
        assert_eq!(b.receipts(vec![late], t + DAY + 60).unwrap(), 0);

        // Only the registered node is listed, but both count toward the swarm's total.
        let rows = b.leaderboard(Metric::Bytes, 30, t);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bytes, 400);
        assert_eq!(b.downloads(t).bytes_served, 1000);

        // Two days on, the day is signed and frozen.
        let op = sign::generate_key(&dir.path().join("op.key")).unwrap();
        b.finalize(t + 2 * DAY, Some(&op)).unwrap();
        let totals = b.totals(day_of(t)).unwrap().clone();
        assert!(totals.verify());
        assert_eq!(totals.bytes_served, 1000);
        assert_eq!(totals.downloads, 1);
        assert_eq!(totals.nodes.len(), 1);
        b.save().unwrap();
        let reopened = Board::open(dir.path()).unwrap();
        assert_eq!(reopened.day(day_of(t))[&s1p.to_string()].bytes, 400);
        assert_eq!(reopened.downloads(t).downloads, 1);
    }

    #[test]
    fn probes_come_only_from_anchors() {
        let dir = tempfile::tempdir().unwrap();
        let mut b = Board::open(dir.path()).unwrap();
        let t = now();
        let (node, anchor, stranger) = (
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
        );
        let np = node.public().to_peer_id().to_string();
        b.register(Registration::new(&node, "node", vec![root(1)]), None, t)
            .unwrap();
        let anchors: HashSet<String> = [anchor.public().to_peer_id().to_string()].into();
        let result = |root: String, outcome| ProbeResult {
            node: np.clone(),
            root,
            outcome,
        };
        let rep = ProbeReport::new(&stranger, vec![result(root(1), Outcome::Held)]);
        assert!(b.probes(rep, &anchors, t).is_err());
        // A model the node doesn't claim doesn't count as held.
        let rep = ProbeReport::new(&anchor, vec![result(root(2), Outcome::Held)]);
        assert_eq!(b.probes(rep, &anchors, t).unwrap(), 0);
        let rep = ProbeReport::new(&anchor, vec![result(root(1), Outcome::Held)]);
        assert_eq!(b.probes(rep, &anchors, t).unwrap(), 1);
        // Too soon after the last one from the same anchor.
        let rep = ProbeReport::new(&anchor, vec![result(root(1), Outcome::Down)]);
        assert_eq!(b.probes(rep, &anchors, t + 1).unwrap(), 0);
        let rep = ProbeReport::new(&anchor, vec![result(root(1), Outcome::Down)]);
        assert_eq!(b.probes(rep, &anchors, t + 120).unwrap(), 1);
        let rows = b.leaderboard(Metric::Uptime, 30, t + 120);
        assert_eq!(rows[0].uptime, Some(0.5));
        assert_eq!(rows[0].models, 1);
        // Hiding the name keeps the numbers.
        b.hide(&np, true).unwrap();
        let rows = b.leaderboard(Metric::Models, 30, t + 120);
        assert_eq!((rows[0].name.clone(), rows[0].models), (None, 1));
        // A node silent for a week is delisted.
        assert!(b.leaderboard(Metric::Bytes, 30, t + DELIST_SECS).is_empty());
    }

    #[test]
    fn registrations_are_checked_and_rate_limited() {
        let dir = tempfile::tempdir().unwrap();
        let mut b = Board::open(dir.path()).unwrap();
        let t = now();
        let k = Keypair::generate_ed25519();
        assert!(
            b.register(Registration::new(&k, "", vec![]), None, t)
                .is_err()
        );
        assert!(
            b.register(Registration::new(&k, "a\nb", vec![]), None, t)
                .is_err()
        );
        assert!(
            b.register(Registration::new(&k, "ok", vec!["nope".into()]), None, t)
                .is_err()
        );
        for _ in 0..REGISTRATIONS_PER_HOUR {
            b.register(Registration::new(&k, "ok", vec![]), Some("net"), t)
                .unwrap();
        }
        assert!(
            b.register(Registration::new(&k, "ok", vec![]), Some("net"), t)
                .is_err()
        );
        assert_eq!(b.listings(t).len(), 1);
    }
}
