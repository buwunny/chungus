//! The registry: names for models, kept in a public, append-only log.
//!
//! Peers move bytes; the registry only answers "which manifest root is `org/model@rev`,
//! and who published it". Every change is a statement signed by its author's key:
//!
//! - `publish` points `org/model@rev` at a manifest root. The first key to publish under
//!   an org owns it; after that only the org's owners can publish there.
//! - `grant` and `revoke` add and remove an org's owner keys.
//! - `block` and `unblock` (registry operator only) maintain the blocklist that nodes
//!   enforce.
//! - `delegate` (operator root key only) lets an online key act as the operator until a
//!   given time, so the root key can stay offline. See [`Registry::open`].
//! - `gate` (registry operator only) puts a model behind a Hugging Face repo's gate, as
//!   a publisher can when publishing. See [`AccessTicket`].
//!
//! The registry appends each accepted statement to a hash-chained log and signs the head.
//! Anyone can download the log and replay it with [`Log::replay`] to check that every
//! entry is signed by a key that was allowed to make it, so the registry can't quietly
//! rewrite a name or history without it showing.

use anyhow::{Context, Result, anyhow, bail};
use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::sign::{self, Signature};
use crate::store;

pub const DEFAULT_PORT: u16 = 7450;
pub const DEFAULT_URL: &str = "http://localhost:7450";
const STATEMENT_DOMAIN: &[u8] = b"chungus/registry-statement/v1\0";
const HEAD_DOMAIN: &[u8] = b"chungus/registry-head/v1\0";
const TICKET_DOMAIN: &[u8] = b"chungus/access-ticket/v1\0";
/// How long an access ticket lasts. Hugging Face doesn't tell anyone when access is
/// revoked, so this bounds how long a revoked user keeps downloading.
pub const TICKET_SECS: u64 = 24 * 3600;
/// How far a statement's time may be from the registry's clock when it is submitted.
const MAX_SKEW_SECS: u64 = 600;
const MAX_DESCRIPTION: usize = 2000;
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Claim {
    Publish {
        name: String,
        rev: String,
        root: String,
        #[serde(default)]
        description: String,
        /// A Hugging Face repo (`org/name`) whose gate this model is behind, e.g. Llama
        /// weights. Nodes following the registry only serve its chunks to peers holding
        /// an [`AccessTicket`], which requires a Hugging Face token the repo accepts.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gated: Option<String>,
    },
    Grant {
        org: String,
        key: String,
    },
    Revoke {
        org: String,
        key: String,
    },
    Block {
        hash: String,
        #[serde(default)]
        reason: String,
    },
    Unblock {
        hash: String,
    },
    /// Put `root` behind the gate of Hugging Face repo `repo` (operator only), for models
    /// published without saying so. An empty `repo` lifts the gate.
    Gate {
        root: String,
        #[serde(default)]
        repo: String,
    },
    /// The operator's list of anchor nodes (multiaddrs ending in `/p2p/<peer id>`), which
    /// replaces any earlier list. Nodes ask anchors alongside the DHT, so an attacker who
    /// fills the DHT around a model still can't hide it.
    Anchors {
        addrs: Vec<String>,
    },
    /// Let `key` act as the operator (sign heads, blocks and anchors) until `expires`
    /// (Unix seconds), replacing any earlier delegation. Only the operator's root key may
    /// make this statement, so a stolen online key can't extend itself. An empty `key`
    /// revokes the delegation.
    Delegate {
        key: String,
        #[serde(default)]
        expires: u64,
    },
}

/// A claim, when it was made, and the signature of the key that made it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub claim: Claim,
    /// Unix seconds.
    pub time: u64,
    pub signature: Signature,
}

/// One entry of the log. `prev` is the hash of the entry before it, so changing any
/// entry changes every hash after it.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Entry {
    pub seq: u64,
    pub prev: String,
    pub statement: Statement,
}

/// The registry's signed summary of the log: its length and the hash of its last entry.
/// It is signed by the operator's root key, or by the online key the log delegates to.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Head {
    pub size: u64,
    pub hash: String,
    pub time: u64,
    pub signature: Signature,
    /// The operator's root key. Informational: clients that pin a key check against the
    /// pin, and everyone checks the signature against the replayed log.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub operator: String,
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn statement_message(claim: &Claim, time: u64, key: &str) -> Vec<u8> {
    let body = serde_json::to_vec(&(claim, time, key)).expect("claims serialize");
    [STATEMENT_DOMAIN, &body].concat()
}

fn head_message(size: u64, hash: &str, time: u64) -> Vec<u8> {
    [HEAD_DOMAIN, format!("{size}\0{hash}\0{time}").as_bytes()].concat()
}

impl Statement {
    pub fn new(key: &SigningKey, claim: Claim) -> Statement {
        let time = now();
        let pk = sign::public_key_string(&key.verifying_key());
        let signature = sign::sign_message(key, &statement_message(&claim, time, &pk));
        Statement {
            claim,
            time,
            signature,
        }
    }

    pub fn verify(&self) -> bool {
        sign::verify_message(
            &self.signature,
            &statement_message(&self.claim, self.time, &self.signature.key),
        )
    }

    pub fn key(&self) -> &str {
        &self.signature.key
    }
}

/// The registry operator's permission for one peer to download models behind one Hugging
/// Face repo's gate, for a day. The registry issues it after checking the peer's own
/// Hugging Face token against the repo; nodes check it before serving gated chunks.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct AccessTicket {
    pub repo: String,
    /// The libp2p peer id the ticket was issued to. Nodes only honour it from that peer.
    pub peer: String,
    /// Unix seconds.
    pub expires: u64,
    pub signature: Signature,
}

fn ticket_message(repo: &str, peer: &str, expires: u64) -> Vec<u8> {
    [
        TICKET_DOMAIN,
        format!("{repo}\0{peer}\0{expires}").as_bytes(),
    ]
    .concat()
}

impl AccessTicket {
    pub fn new(key: &SigningKey, repo: &str, peer: &str, expires: u64) -> AccessTicket {
        AccessTicket {
            repo: repo.into(),
            peer: peer.into(),
            expires,
            signature: sign::sign_message(key, &ticket_message(repo, peer, expires)),
        }
    }

    /// Whether `operator` issued this ticket to `peer` and it hasn't expired.
    pub fn valid_for(&self, operator: &str, peer: &str) -> bool {
        self.signature.key == operator
            && self.peer == peer
            && self.expires > now()
            && sign::verify_message(
                &self.signature,
                &ticket_message(&self.repo, &self.peer, self.expires),
            )
    }
}

impl Entry {
    pub fn hash(&self) -> String {
        blake3::hash(&serde_json::to_vec(self).expect("entries serialize"))
            .to_hex()
            .to_string()
    }
}

impl Head {
    pub fn verify(&self) -> bool {
        sign::verify_message(
            &self.signature,
            &head_message(self.size, &self.hash, self.time),
        )
    }
}

fn valid_part(s: &str, max: usize) -> bool {
    !s.is_empty()
        && s.len() <= max
        && !s.starts_with('.')
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// A Hugging Face repo id, `org/name`.
pub fn valid_hf_repo(repo: &str) -> bool {
    matches!(repo.split_once('/'), Some((org, name)) if valid_part(org, 96) && valid_part(name, 96))
}

/// `org/model`, each part of letters, digits, `-`, `_` and `.`.
pub fn valid_name(name: &str) -> bool {
    matches!(name.split_once('/'), Some((org, model)) if valid_part(org, 64) && valid_part(model, 96))
}

pub fn valid_rev(rev: &str) -> bool {
    valid_part(rev, 64)
}

/// Split `org/model@rev` into name and rev; the rev defaults to `main`.
pub fn parse_ref(s: &str) -> Result<(String, String)> {
    let (name, rev) = s.split_once('@').unwrap_or((s, "main"));
    if !valid_name(name) || !valid_rev(rev) {
        bail!("{s:?} is not a model name like org/model or org/model@rev");
    }
    Ok((name.to_string(), rev.to_string()))
}

const MAX_ANCHORS: usize = 64;

/// An anchor must be a multiaddr that names its peer, so nodes know whom they reach.
fn valid_anchor(addr: &str) -> Result<libp2p::Multiaddr> {
    let a: libp2p::Multiaddr = addr
        .parse()
        .with_context(|| format!("{addr} is not a multiaddr"))?;
    if !matches!(a.iter().last(), Some(libp2p::multiaddr::Protocol::P2p(_))) {
        bail!("{addr} must end in /p2p/<peer id>");
    }
    Ok(a)
}

fn org_of(name: &str) -> &str {
    name.split_once('/').map(|(o, _)| o).unwrap_or(name)
}

/// The state that replaying the log produces.
#[derive(Default)]
pub struct Log {
    /// The key allowed to block and unblock, and to sign heads.
    pub operator: String,
    pub entries: Vec<Entry>,
    head_hash: String,
    owners: HashMap<String, Vec<String>>,
    /// (name, rev) -> index of the publish entry that currently defines it.
    names: BTreeMap<(String, String), usize>,
    blocked: HashSet<String>,
    /// Manifest root -> the Hugging Face repo whose gate it is behind.
    gated: HashMap<String, String>,
    /// Index of the latest anchors entry.
    anchors: Option<usize>,
    /// The online key currently acting for the operator, and when that ends.
    delegate: Option<(String, u64)>,
    seen: HashSet<String>,
}

impl Log {
    pub fn new(operator: String) -> Log {
        Log {
            operator,
            head_hash: GENESIS.into(),
            ..Default::default()
        }
    }

    /// Check a statement against the log so far, without adding it.
    pub fn check(&self, st: &Statement) -> Result<()> {
        if !st.verify() {
            bail!("bad signature");
        }
        if self.seen.contains(&st.signature.sig) {
            bail!("this statement is already in the log");
        }
        let key = st.key();
        match &st.claim {
            Claim::Publish {
                name,
                rev,
                root,
                description,
                gated,
            } => {
                if gated.as_deref().is_some_and(|r| !valid_hf_repo(r)) {
                    bail!("gated must be a Hugging Face repo like org/name");
                }
                if !valid_name(name) || !valid_rev(rev) {
                    bail!("invalid name or rev");
                }
                if !store::is_hash(root) {
                    bail!("invalid manifest root");
                }
                if description.len() > MAX_DESCRIPTION {
                    bail!("description is longer than {MAX_DESCRIPTION} bytes");
                }
                if self.blocked.contains(root) {
                    bail!("{root} is blocked");
                }
                if let Some(owners) = self.owners.get(org_of(name))
                    && !owners.iter().any(|o| o == key)
                {
                    bail!("{key} is not an owner of {}", org_of(name));
                }
            }
            Claim::Grant { org, key: new } => {
                sign::parse_public_key(new)?;
                let owners = self.owners.get(org).context("no such org")?;
                if !owners.iter().any(|o| o == key) {
                    bail!("{key} is not an owner of {org}");
                }
                if owners.contains(new) {
                    bail!("{new} already owns {org}");
                }
            }
            Claim::Revoke { org, key: old } => {
                let owners = self.owners.get(org).context("no such org")?;
                if !owners.iter().any(|o| o == key) {
                    bail!("{key} is not an owner of {org}");
                }
                if !owners.contains(old) {
                    bail!("{old} does not own {org}");
                }
                if owners.len() == 1 {
                    bail!("can't revoke an org's last owner");
                }
            }
            Claim::Block { hash, .. } | Claim::Unblock { hash } => {
                if !self.acts_as_operator(key, st.time) {
                    bail!("only the registry operator can change the blocklist");
                }
                if !store::is_hash(hash) {
                    bail!("invalid hash");
                }
            }
            Claim::Gate { root, repo } => {
                if !self.acts_as_operator(key, st.time) {
                    bail!("only the registry operator can gate a model");
                }
                if !store::is_hash(root) {
                    bail!("invalid manifest root");
                }
                if !repo.is_empty() && !valid_hf_repo(repo) {
                    bail!("{repo:?} is not a Hugging Face repo like org/name");
                }
            }
            Claim::Delegate {
                key: online,
                expires,
            } => {
                if key != self.operator {
                    bail!("only the operator's root key can delegate");
                }
                if !online.is_empty() {
                    sign::parse_public_key(online)?;
                    if *expires <= st.time {
                        bail!("the delegation has already expired");
                    }
                    if online == &self.operator {
                        bail!("the root key can't delegate to itself");
                    }
                }
            }
            Claim::Anchors { addrs } => {
                if !self.acts_as_operator(key, st.time) {
                    bail!("only the registry operator can set the anchors");
                }
                if addrs.len() > MAX_ANCHORS {
                    bail!("at most {MAX_ANCHORS} anchors");
                }
                for a in addrs {
                    valid_anchor(a)?;
                }
            }
        }
        Ok(())
    }

    /// Check and append a statement. Returns the new entry.
    pub fn append(&mut self, st: Statement) -> Result<Entry> {
        self.check(&st)?;
        let entry = Entry {
            seq: self.entries.len() as u64,
            prev: self.head_hash.clone(),
            statement: st,
        };
        self.apply(entry.clone());
        Ok(entry)
    }

    fn apply(&mut self, entry: Entry) {
        let st = &entry.statement;
        let key = st.key().to_string();
        let idx = self.entries.len();
        match &st.claim {
            Claim::Publish {
                name,
                rev,
                root,
                gated,
                ..
            } => {
                // A publisher can add a gate but not lift one: only the operator can.
                if let Some(repo) = gated {
                    self.gated.insert(root.clone(), repo.clone());
                }
                self.owners
                    .entry(org_of(name).to_string())
                    .or_insert_with(|| vec![key.clone()]);
                self.names.insert((name.clone(), rev.clone()), idx);
            }
            Claim::Grant { org, key: new } => {
                self.owners.get_mut(org).unwrap().push(new.clone());
            }
            Claim::Revoke { org, key: old } => {
                self.owners.get_mut(org).unwrap().retain(|k| k != old);
            }
            Claim::Block { hash, .. } => {
                self.blocked.insert(hash.clone());
            }
            Claim::Unblock { hash } => {
                self.blocked.remove(hash);
            }
            Claim::Gate { root, repo } => {
                if repo.is_empty() {
                    self.gated.remove(root);
                } else {
                    self.gated.insert(root.clone(), repo.clone());
                }
            }
            Claim::Anchors { .. } => {
                self.anchors = Some(idx);
            }
            Claim::Delegate { key, expires } => {
                self.delegate = (!key.is_empty()).then(|| (key.clone(), *expires));
            }
        }
        self.seen.insert(st.signature.sig.clone());
        self.head_hash = entry.hash();
        self.entries.push(entry);
    }

    /// Rebuild the state from a downloaded log, checking every link and every statement.
    /// This is what an auditor runs.
    pub fn replay(operator: String, entries: Vec<Entry>) -> Result<Log> {
        let mut log = Log::new(operator);
        log.extend(entries)?;
        Ok(log)
    }

    /// Add entries that continue this log, checking each as [`Log::replay`] does.
    pub fn extend(&mut self, entries: Vec<Entry>) -> Result<()> {
        for e in entries {
            if e.seq != self.entries.len() as u64 {
                bail!("entry {} is out of order", e.seq);
            }
            if e.prev != self.head_hash {
                bail!("entry {} does not follow the one before it", e.seq);
            }
            self.check(&e.statement)
                .with_context(|| format!("entry {} should not have been accepted", e.seq))?;
            self.apply(e);
        }
        Ok(())
    }

    pub fn head_hash(&self) -> &str {
        &self.head_hash
    }

    pub fn resolve(&self, name: &str, rev: &str) -> Option<&Entry> {
        self.names
            .get(&(name.to_string(), rev.to_string()))
            .map(|&i| &self.entries[i])
    }

    pub fn owners(&self, org: &str) -> &[String] {
        self.owners.get(org).map(Vec::as_slice).unwrap_or_default()
    }

    pub fn is_blocked(&self, hash: &str) -> bool {
        self.blocked.contains(hash)
    }

    pub fn blocked(&self) -> impl Iterator<Item = &String> {
        self.blocked.iter()
    }

    /// Whether `key` may act as the operator at `time`: the root key always, the delegated
    /// online key until its delegation expires.
    pub fn acts_as_operator(&self, key: &str, time: u64) -> bool {
        key == self.operator
            || self
                .delegate
                .as_ref()
                .is_some_and(|(k, exp)| k == key && time < *exp)
    }

    /// The delegated online key and when its delegation expires.
    pub fn delegate(&self) -> Option<(&str, u64)> {
        self.delegate.as_ref().map(|(k, e)| (k.as_str(), *e))
    }

    /// Whether `head` is signed by a key the log allows to sign it.
    pub fn signs_head(&self, head: &Head) -> bool {
        head.verify() && self.acts_as_operator(&head.signature.key, head.time)
    }

    /// The Hugging Face repo whose gate `root` is behind, if any.
    pub fn gate(&self, root: &str) -> Option<&str> {
        self.gated.get(root).map(String::as_str)
    }

    /// Every gated root and its repo.
    pub fn gates(&self) -> &HashMap<String, String> {
        &self.gated
    }

    /// The keys whose access tickets nodes accept, each with when it stops being accepted:
    /// the root key, and the delegated online key until its delegation ends.
    pub fn ticket_issuers(&self) -> Vec<(String, u64)> {
        let mut keys = vec![(self.operator.clone(), u64::MAX)];
        keys.extend(self.delegate.clone());
        keys
    }

    /// The entry holding the current anchor list, if the operator has set one.
    pub fn anchors(&self) -> Option<&Entry> {
        self.anchors.map(|i| &self.entries[i])
    }

    /// Models whose name or description matches `query`, best first. Every word of the
    /// query must match the start of some word of the model's name or description.
    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        let terms = words(query);
        let mut hits: Vec<(usize, &Entry)> = Vec::new();
        for ((name, _), &i) in &self.names {
            let entry = &self.entries[i];
            let Claim::Publish {
                description, root, ..
            } = &entry.statement.claim
            else {
                continue;
            };
            if self.blocked.contains(root) {
                continue;
            }
            let name_words = words(name);
            let desc_words = words(description);
            let mut score = 0;
            let all = terms.iter().all(|t| {
                let in_name = name_words.iter().any(|w| w.starts_with(t.as_str()));
                let in_desc = desc_words.iter().any(|w| w.starts_with(t.as_str()));
                score += 2 * in_name as usize + in_desc as usize;
                in_name || in_desc
            });
            if all {
                hits.push((score, entry));
            }
        }
        hits.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(b.1.statement.time.cmp(&a.1.statement.time))
        });
        hits.into_iter()
            .take(limit)
            .map(|(_, e)| Hit::from(e, self))
            .collect()
    }
}

impl Log {
    /// Every published name that isn't blocked, newest first: what a search page needs to
    /// list and search models on its own.
    pub fn index(&self) -> Vec<Hit> {
        let mut hits: Vec<&Entry> = self
            .names
            .values()
            .map(|&i| &self.entries[i])
            .filter(|e| match &e.statement.claim {
                Claim::Publish { root, .. } => !self.blocked.contains(root),
                _ => false,
            })
            .collect();
        hits.sort_by_key(|e| std::cmp::Reverse(e.statement.time));
        hits.into_iter().map(|e| Hit::from(e, self)).collect()
    }
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// A search result.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hit {
    pub name: String,
    pub rev: String,
    pub root: String,
    pub description: String,
    pub publisher: String,
    pub time: u64,
    /// The Hugging Face repo whose gate the model is behind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gated: Option<String>,
    /// Total size of the model's files in bytes, when the registry holds its manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

impl Hit {
    fn from(e: &Entry, log: &Log) -> Hit {
        let Claim::Publish {
            name,
            rev,
            root,
            description,
            ..
        } = &e.statement.claim
        else {
            unreachable!("search only returns publishes")
        };
        Hit {
            name: name.clone(),
            rev: rev.clone(),
            root: root.clone(),
            description: description.clone(),
            publisher: e.statement.key().to_string(),
            time: e.statement.time,
            gated: log.gate(root).map(str::to_string),
            size: None,
        }
    }
}

/// What a model's manifest says about it, for the site's model page.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Summary {
    /// Total size of every file, in bytes.
    pub size: u64,
    /// Bytes in weight files (safetensors and GGUF).
    pub weights: u64,
    /// Weight formats present, e.g. `["safetensors"]`.
    pub formats: Vec<String>,
    /// Chunks across all files, and how many of them are distinct.
    pub chunks: u64,
    pub unique_chunks: u64,
    /// Bytes of the distinct chunks: what a download with an empty store transfers
    /// before compression.
    pub unique_bytes: u64,
    /// Parameters across the safetensors files, when the publisher sent every file's
    /// header (see [`Registry::publish`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<u64>,
    /// Those parameters by dtype, e.g. `{"BF16": 3821079552}`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dtypes: BTreeMap<String, u64>,
    pub files: Vec<SummaryFile>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SummaryFile {
    pub path: String,
    pub size: u64,
}

impl Summary {
    /// Summarize `m`, counting parameters from `headers` (checked with [`check_headers`]).
    pub fn of(m: &crate::manifest::Manifest, headers: &BTreeMap<String, String>) -> Summary {
        let mut seen = HashSet::new();
        let (mut chunks, mut unique_bytes, mut weights) = (0, 0, 0);
        let mut formats = std::collections::BTreeSet::new();
        for f in &m.files {
            let ext = f.path.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
            if let Some(ext @ ("safetensors" | "gguf")) = ext.as_deref() {
                weights += f.size;
                formats.insert(ext.to_string());
            }
            for c in &f.chunks {
                chunks += 1;
                if seen.insert(c.hash.as_str()) {
                    unique_bytes += u64::from(c.len);
                }
            }
        }
        let mut dtypes = BTreeMap::new();
        let mut counted = 0;
        for f in m.files.iter().filter(|f| f.path.ends_with(".safetensors")) {
            let Some(p) = headers
                .get(&f.path)
                .and_then(|h| crate::safetensors::params(h.as_bytes()).ok())
            else {
                continue;
            };
            counted += 1;
            for (dtype, n) in p {
                *dtypes.entry(dtype).or_default() += n;
            }
        }
        // A count missing some shards would be wrong, so show none instead.
        let shards = m
            .files
            .iter()
            .filter(|f| f.path.ends_with(".safetensors"))
            .count();
        if counted < shards || shards == 0 {
            dtypes.clear();
        }
        Summary {
            params: (!dtypes.is_empty()).then(|| dtypes.values().sum()),
            dtypes,
            size: m.files.iter().map(|f| f.size).sum(),
            weights,
            formats: formats.into_iter().collect(),
            chunks,
            unique_chunks: seen.len() as u64,
            unique_bytes,
            files: m
                .files
                .iter()
                .map(|f| SummaryFile {
                    path: f.path.clone(),
                    size: f.size,
                })
                .collect(),
        }
    }
}

/// Check that each of `headers` (safetensors header JSON by path) is exactly the start
/// of that file in `m`: prefixed with its length, it must split into the file's leading
/// chunks and hash to them. Packing puts a header in chunks of its own, so this holds for
/// any file `pack` wrote, and the signed root vouches for the result.
pub fn check_headers(
    m: &crate::manifest::Manifest,
    headers: &BTreeMap<String, String>,
) -> Result<()> {
    for (path, json) in headers {
        let f = m
            .files
            .iter()
            .find(|f| &f.path == path && path.ends_with(".safetensors"))
            .with_context(|| format!("{path} isn't a safetensors file of this model"))?;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        let mut at = 0;
        for c in &f.chunks {
            if at == bytes.len() {
                break;
            }
            let end = at + c.len as usize;
            let ok = bytes
                .get(at..end)
                .is_some_and(|b| blake3::hash(b).to_hex().as_str() == c.hash);
            if !ok {
                bail!("the header sent for {path} doesn't match its chunks");
            }
            at = end;
        }
        if at != bytes.len() {
            bail!("the header sent for {path} is longer than the file");
        }
    }
    Ok(())
}

// ---------- server ----------

/// A registry backed by a directory: `log.jsonl` (one entry per line), the operator's
/// public root key in `operator.pub`, signing keys (see [`Registry::open`]), and
/// `manifests/<root>.json` for every model published through it.
pub struct Registry {
    log: Mutex<Log>,
    operator: String,
    /// The operator's root key, when it is kept here rather than offline.
    root: Option<SigningKey>,
    /// The online key, which signs once the log delegates to it.
    online: SigningKey,
    file: Mutex<fs::File>,
    manifests: PathBuf,
    /// `<root>.json` for each model whose publisher sent its safetensors headers.
    headers: PathBuf,
    /// Roots whose manifest the registry holds and has checked (see [`crate::safety`]),
    /// with a summary of each. Only these are listed in search and the index.
    checked: RwLock<HashMap<String, Arc<Summary>>>,
    /// Where access to gated repos is checked: huggingface.co, or a fake in tests.
    hf: String,
    http: reqwest::Client,
}

pub const DEFAULT_HF: &str = "https://huggingface.co";

/// A request for an [`AccessTicket`]: the model, the peer that will download it, and the
/// requester's own Hugging Face token, which the registry only sends to Hugging Face.
#[derive(Serialize, Deserialize)]
pub struct AccessRequest {
    pub root: String,
    pub peer: String,
    pub token: String,
}

/// How long `chungus delegate` delegates for by default. Renew before it runs out.
pub const DEFAULT_DELEGATION_DAYS: u64 = 90;

/// Largest manifest the registry accepts: about a 1 TB model.
pub const MAX_MANIFEST: usize = 256 << 20;

impl Registry {
    /// Open the registry in `dir`, creating it if new.
    ///
    /// The operator is identified by a root key, which nodes pin. For a new registry the
    /// root key is created in `operator.key`, and the registry works as before. To keep
    /// the root key offline, move `operator.key` to another machine: the registry then
    /// signs with `online.key` (created here on first use), once the root key delegates
    /// to it with `chungus delegate`. A stolen online key can act as the operator only
    /// until its delegation expires or is revoked, and can never delegate.
    pub fn open(dir: &FsPath) -> Result<Registry> {
        fs::create_dir_all(dir)?;
        let key_path = dir.join("operator.key");
        let pub_path = dir.join("operator.pub");
        let root = if key_path.exists() {
            Some(sign::load_key(&key_path)?)
        } else if pub_path.exists() {
            None
        } else {
            Some(sign::generate_key(&key_path)?)
        };
        let operator = match &root {
            Some(k) => {
                let op = sign::public_key_string(&k.verifying_key());
                fs::write(&pub_path, format!("{op}\n"))?;
                op
            }
            None => {
                let op = fs::read_to_string(&pub_path)?.trim().to_string();
                sign::parse_public_key(&op)
                    .with_context(|| format!("{} is not a public key", pub_path.display()))?;
                op
            }
        };
        let online_path = dir.join("online.key");
        let online = if online_path.exists() {
            sign::load_key(&online_path)?
        } else {
            sign::generate_key(&online_path)?
        };
        let path: PathBuf = dir.join("log.jsonl");
        let mut entries = Vec::new();
        if let Ok(text) = fs::read_to_string(&path) {
            for (i, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                entries.push(
                    serde_json::from_str(line)
                        .with_context(|| format!("{} line {}", path.display(), i + 1))?,
                );
            }
        }
        let log =
            Log::replay(operator.clone(), entries).context("the registry's own log is invalid")?;
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        let manifests = dir.join("manifests");
        fs::create_dir_all(&manifests)?;
        let headers = dir.join("headers");
        fs::create_dir_all(&headers)?;
        let mut checked = HashMap::new();
        for e in fs::read_dir(&manifests)? {
            let e = e?;
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(root) = name.strip_suffix(".json")
                && store::is_hash(root)
            {
                let m = crate::manifest::parse(&fs::read(e.path())?)
                    .with_context(|| format!("{}", e.path().display()))?;
                let h = match fs::read(headers.join(&name)) {
                    Ok(b) => serde_json::from_slice(&b).unwrap_or_default(),
                    Err(_) => BTreeMap::new(),
                };
                checked.insert(root.to_string(), Arc::new(Summary::of(&m, &h)));
            }
        }
        Ok(Registry {
            log: Mutex::new(log),
            operator,
            root,
            online,
            file: Mutex::new(file),
            manifests,
            headers,
            checked: RwLock::new(checked),
            hf: DEFAULT_HF.into(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()?,
        })
    }

    /// The key that acts as the operator now: the online key while the log delegates to
    /// it, otherwise the root key if it is here. None means an offline root key has not
    /// delegated to (or has revoked) this registry's online key.
    pub fn operator_key(&self) -> Option<&SigningKey> {
        let online = self.online_public();
        if self.with_log(|log| log.acts_as_operator(&online, now())) {
            return Some(&self.online);
        }
        self.root.as_ref()
    }

    /// Check gated access against `url` instead of huggingface.co.
    pub fn with_hf(mut self, url: &str) -> Registry {
        self.hf = url.trim_end_matches('/').to_string();
        self
    }

    /// Issue a ticket for `req.peer` to download `req.root`, if Hugging Face says the
    /// requester's token may access the repo the model is gated by. The token is used
    /// for that one check and never stored or logged.
    pub async fn access(&self, req: &AccessRequest) -> Result<AccessTicket, (StatusCode, String)> {
        let bad = |s: &str| (StatusCode::BAD_REQUEST, s.to_string());
        if req.peer.parse::<libp2p::PeerId>().is_err() {
            return Err(bad("peer must be a libp2p peer id"));
        }
        let Some(repo) = self.with_log(|log| log.gate(&req.root).map(str::to_string)) else {
            return Err(bad("that model isn't gated"));
        };
        // Hugging Face's own check for "may this token download from this repo".
        let resp = self
            .http
            .get(format!("{}/api/models/{repo}/auth-check", self.hf))
            .bearer_auth(&req.token)
            .send()
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    format!("can't reach Hugging Face: {e}"),
                )
            })?;
        match resp.status().as_u16() {
            200..=299 => {}
            401 | 403 | 404 => {
                return Err((
                    StatusCode::FORBIDDEN,
                    format!(
                        "your Hugging Face token doesn't have access to {repo}; accept its \
                         license at https://huggingface.co/{repo} and try again"
                    ),
                ));
            }
            s => {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    format!("Hugging Face answered {s} when checking access to {repo}"),
                ));
            }
        }
        let Some(key) = self.operator_key() else {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "this registry's online key isn't delegated to, so it can't issue tickets".into(),
            ));
        };
        Ok(AccessTicket::new(
            key,
            &repo,
            &req.peer,
            now() + TICKET_SECS,
        ))
    }

    /// The operator's root public key, the one nodes pin.
    pub fn operator(&self) -> String {
        self.operator.clone()
    }

    /// The online key's public half, for `chungus delegate`.
    pub fn online_public(&self) -> String {
        sign::public_key_string(&self.online.verifying_key())
    }

    /// Whether the root key is on this machine.
    pub fn has_root_key(&self) -> bool {
        self.root.is_some()
    }

    /// Accept a statement: check it, write it to disk, then add it to the log.
    pub fn submit(&self, st: Statement) -> Result<Entry> {
        let skew = now().abs_diff(st.time);
        if skew > MAX_SKEW_SECS {
            bail!("statement time is {skew}s away from the registry's clock");
        }
        let mut log = self.log.lock().unwrap();
        log.check(&st)?;
        let entry = Entry {
            seq: log.entries.len() as u64,
            prev: log.head_hash().to_string(),
            statement: st,
        };
        let mut line = serde_json::to_vec(&entry)?;
        line.push(b'\n');
        let mut file = self.file.lock().unwrap();
        file.write_all(&line)?;
        file.sync_data()?;
        log.apply(entry.clone());
        Ok(entry)
    }

    /// Publish a model: the statement must name `manifest`'s root, the manifest must
    /// verify, and it must list only files chungus carries (no pickles). The registry keeps
    /// the manifest, so anyone can see what a name contains before fetching it.
    /// Publish `st` with the manifest it names, and optionally the header JSON of each
    /// safetensors file (by path), from which the model page counts parameters.
    pub fn publish(
        &self,
        st: Statement,
        manifest: &[u8],
        headers: &BTreeMap<String, String>,
    ) -> Result<Entry> {
        let Claim::Publish { root, .. } = &st.claim else {
            bail!("not a publish statement");
        };
        let m = crate::manifest::parse(manifest)?;
        if &m.root != root || !m.verify_root() {
            bail!("the manifest doesn't match the published root {root}");
        }
        crate::safety::check_manifest(&m)?;
        check_headers(&m, headers)?;
        // Check the statement before writing anything, so strangers can't fill the disk.
        self.log.lock().unwrap().check(&st)?;
        let path = self.manifests.join(format!("{root}.json"));
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, manifest)?;
        fs::rename(&tmp, &path)?;
        // Republishing a model without headers keeps the ones sent before.
        if !headers.is_empty() {
            let path = self.headers.join(format!("{root}.json"));
            let tmp = path.with_extension("tmp");
            fs::write(&tmp, serde_json::to_vec(headers)?)?;
            fs::rename(&tmp, &path)?;
        }
        let entry = self.submit(st)?;
        let headers = match fs::read(self.headers.join(format!("{}.json", m.root))) {
            Ok(b) => serde_json::from_slice(&b).unwrap_or_default(),
            Err(_) => BTreeMap::new(),
        };
        let summary = Arc::new(Summary::of(&m, &headers));
        self.checked.write().unwrap().insert(m.root, summary);
        Ok(entry)
    }

    /// The manifest published under `root`, if this registry holds it.
    pub fn manifest(&self, root: &str) -> Option<Vec<u8>> {
        if !store::is_hash(root) {
            return None;
        }
        fs::read(self.manifests.join(format!("{root}.json"))).ok()
    }

    /// Whether search and the index list `root`: only models whose manifest was checked
    /// when they were published.
    pub fn is_listed(&self, root: &str) -> bool {
        self.checked.read().unwrap().contains_key(root)
    }

    /// The summary of a listed model's manifest.
    pub fn summary(&self, root: &str) -> Option<Arc<Summary>> {
        self.checked.read().unwrap().get(root).cloned()
    }

    fn listed(&self, hits: Vec<Hit>) -> Vec<Hit> {
        let checked = self.checked.read().unwrap();
        hits.into_iter()
            .filter_map(|mut h| {
                h.size = Some(checked.get(&h.root)?.size);
                Some(h)
            })
            .collect()
    }

    /// The signed head. Signed by the online key if delegated, else the root key; with
    /// neither, the online key signs anyway and clients reject it until it is delegated.
    pub fn head(&self) -> Head {
        let key = self.operator_key().unwrap_or(&self.online).clone();
        let log = self.log.lock().unwrap();
        let (size, hash, time) = (log.entries.len() as u64, log.head_hash().to_string(), now());
        Head {
            signature: sign::sign_message(&key, &head_message(size, &hash, time)),
            size,
            hash,
            time,
            operator: self.operator.clone(),
        }
    }

    pub fn with_log<T>(&self, f: impl FnOnce(&Log) -> T) -> T {
        f(&self.log.lock().unwrap())
    }
}

fn err(code: StatusCode, e: impl std::fmt::Display) -> Response {
    (code, e.to_string()).into_response()
}

/// `POST /v1/statements`, `POST /v1/publish`, `GET /v1/manifests/{root}`, `GET /v1/head`, `GET /v1/log?from=&limit=`,
/// `GET /v1/resolve/{org}/{model}/{rev}`, `GET /v1/search?q=`, `GET /v1/index`,
/// `GET /v1/owners/{org}`, `GET /v1/anchors` and `POST /v1/access`.
///
/// Everything it serves is public and signed, so any web page may read it: responses
/// allow every origin. Browsers can't submit statements, since there is no CORS
/// preflight answer for the JSON `POST`.
pub fn router(reg: Arc<Registry>) -> Router {
    Router::new()
        .route("/v1/statements", post(submit))
        .route(
            "/v1/publish",
            post(publish).layer(axum::extract::DefaultBodyLimit::max(
                MAX_MANIFEST + (1 << 16),
            )),
        )
        .route("/v1/manifests/{root}", get(manifest))
        .route("/v1/summary/{root}", get(summary))
        .route("/v1/head", get(head))
        .route("/v1/log", get(log_entries))
        .route("/v1/resolve/{org}/{model}/{rev}", get(resolve))
        .route("/v1/search", get(search))
        .route("/v1/index", get(index))
        .route("/v1/owners/{org}", get(owners))
        .route("/v1/anchors", get(anchors))
        .route("/v1/access", post(access))
        .layer(axum::middleware::map_response(allow_any_origin))
        .with_state(reg)
}

async fn allow_any_origin(mut resp: Response) -> Response {
    resp.headers_mut().insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
        axum::http::HeaderValue::from_static("*"),
    );
    resp
}

async fn submit(
    State(reg): State<Arc<Registry>>,
    axum::Json(st): axum::Json<Statement>,
) -> Response {
    if matches!(st.claim, Claim::Publish { .. }) {
        return err(
            StatusCode::BAD_REQUEST,
            "publish through /v1/publish, with the manifest (upgrade chungus)",
        );
    }
    match tokio::task::spawn_blocking(move || reg.submit(st)).await {
        Ok(Ok(entry)) => axum::Json(entry).into_response(),
        Ok(Err(e)) => err(StatusCode::FORBIDDEN, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// A publish statement and the manifest it names.
#[derive(Serialize, Deserialize)]
pub struct Publication {
    pub statement: Statement,
    /// The manifest's JSON, exactly as packed.
    pub manifest: String,
    /// Header JSON of each safetensors file, by path. Optional; checked against the
    /// manifest's chunks.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

async fn publish(
    State(reg): State<Arc<Registry>>,
    axum::Json(p): axum::Json<Publication>,
) -> Response {
    match tokio::task::spawn_blocking(move || {
        reg.publish(p.statement, p.manifest.as_bytes(), &p.headers)
    })
    .await
    {
        Ok(Ok(entry)) => axum::Json(entry).into_response(),
        Ok(Err(e)) => err(StatusCode::FORBIDDEN, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn manifest(State(reg): State<Arc<Registry>>, Path(root): Path<String>) -> Response {
    match reg.manifest(&root) {
        Some(bytes) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        None => err(StatusCode::NOT_FOUND, format!("no manifest for {root}")),
    }
}

async fn summary(State(reg): State<Arc<Registry>>, Path(root): Path<String>) -> Response {
    match reg.summary(&root) {
        Some(s) => axum::Json(&*s).into_response(),
        None => err(StatusCode::NOT_FOUND, format!("no manifest for {root}")),
    }
}

async fn access(
    State(reg): State<Arc<Registry>>,
    axum::Json(req): axum::Json<AccessRequest>,
) -> Response {
    match reg.access(&req).await {
        Ok(ticket) => axum::Json(ticket).into_response(),
        Err((code, e)) => err(code, e),
    }
}

async fn head(State(reg): State<Arc<Registry>>) -> Response {
    axum::Json(reg.head()).into_response()
}

#[derive(Deserialize)]
struct Page {
    #[serde(default)]
    from: u64,
    limit: Option<usize>,
}

async fn log_entries(State(reg): State<Arc<Registry>>, Query(p): Query<Page>) -> Response {
    let limit = p.limit.unwrap_or(1000).min(1000);
    let page: Vec<Entry> = reg.with_log(|log| {
        log.entries
            .iter()
            .skip(p.from as usize)
            .take(limit)
            .cloned()
            .collect()
    });
    axum::Json(page).into_response()
}

async fn resolve(
    State(reg): State<Arc<Registry>>,
    Path((org, model, rev)): Path<(String, String, String)>,
) -> Response {
    let name = format!("{org}/{model}");
    match reg.with_log(|log| log.resolve(&name, &rev).cloned()) {
        Some(entry) => axum::Json(entry).into_response(),
        None => err(
            StatusCode::NOT_FOUND,
            format!("{name}@{rev} is not published"),
        ),
    }
}

#[derive(Deserialize)]
struct SearchQuery {
    #[serde(default)]
    q: String,
}

async fn search(State(reg): State<Arc<Registry>>, Query(q): Query<SearchQuery>) -> Response {
    // Search a little past the limit, since unlisted models are dropped afterwards.
    let hits = reg.with_log(|log| log.search(&q.q, 200));
    axum::Json(reg.listed(hits).into_iter().take(50).collect::<Vec<_>>()).into_response()
}

async fn index(State(reg): State<Arc<Registry>>) -> Response {
    axum::Json(reg.listed(reg.with_log(|log| log.index()))).into_response()
}

async fn owners(State(reg): State<Arc<Registry>>, Path(org): Path<String>) -> Response {
    axum::Json(reg.with_log(|log| log.owners(&org).to_vec())).into_response()
}

async fn anchors(State(reg): State<Arc<Registry>>) -> Response {
    match reg.with_log(|log| log.anchors().cloned()) {
        Some(e) => axum::Json(e).into_response(),
        None => err(StatusCode::NOT_FOUND, "no anchors set"),
    }
}

/// The root key a head names, for clients that haven't pinned one. Older registries
/// don't name it, and sign with the root key itself.
fn head_operator(head: &Head) -> String {
    if head.operator.is_empty() {
        head.signature.key.clone()
    } else {
        head.operator.clone()
    }
}

// ---------- client ----------

pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base: &str) -> Result<Client> {
        Ok(Client {
            base: base.trim_end_matches('/').to_string(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()?,
        })
    }

    async fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let resp = self
            .http
            .get(format!("{}{path}", self.base))
            .send()
            .await
            .with_context(|| format!("reach the registry at {}", self.base))?;
        let status = resp.status();
        let body = resp.bytes().await?;
        if !status.is_success() {
            bail!("{}", String::from_utf8_lossy(&body));
        }
        Ok(serde_json::from_slice(&body)?)
    }

    pub async fn submit(&self, st: &Statement) -> Result<Entry> {
        let resp = self
            .http
            .post(format!("{}/v1/statements", self.base))
            .header("content-type", "application/json")
            .body(serde_json::to_vec(st)?)
            .send()
            .await
            .with_context(|| format!("reach the registry at {}", self.base))?;
        let status = resp.status();
        let body = resp.bytes().await?;
        if !status.is_success() {
            bail!(
                "registry refused ({status}): {}",
                String::from_utf8_lossy(&body)
            );
        }
        Ok(serde_json::from_slice(&body)?)
    }

    /// Ask for a ticket to download gated model `root` as `peer`, proving access with
    /// the user's Hugging Face `token`.
    pub async fn access(&self, root: &str, peer: &str, token: &str) -> Result<AccessTicket> {
        let body = AccessRequest {
            root: root.into(),
            peer: peer.into(),
            token: token.into(),
        };
        let resp = self
            .http
            .post(format!("{}/v1/access", self.base))
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&body)?)
            .send()
            .await
            .with_context(|| format!("reach the registry at {}", self.base))?;
        let status = resp.status();
        let body = resp.bytes().await?;
        if !status.is_success() {
            bail!("{}", String::from_utf8_lossy(&body));
        }
        Ok(serde_json::from_slice(&body)?)
    }

    /// Publish a model with its manifest (see [`Registry::publish`]).
    pub async fn publish(&self, st: &Statement, manifest: &[u8]) -> Result<Entry> {
        self.publish_with_headers(st, manifest, BTreeMap::new())
            .await
    }

    /// [`Client::publish`], also sending safetensors headers (see
    /// [`crate::safetensors_headers`]) so the registry can count parameters.
    pub async fn publish_with_headers(
        &self,
        st: &Statement,
        manifest: &[u8],
        headers: BTreeMap<String, String>,
    ) -> Result<Entry> {
        let body = Publication {
            statement: st.clone(),
            manifest: String::from_utf8(manifest.to_vec()).context("manifest isn't UTF-8")?,
            headers,
        };
        let resp = self
            .http
            .post(format!("{}/v1/publish", self.base))
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&body)?)
            .timeout(std::time::Duration::from_secs(600))
            .send()
            .await
            .with_context(|| format!("reach the registry at {}", self.base))?;
        let status = resp.status();
        let body = resp.bytes().await?;
        if !status.is_success() {
            bail!(
                "registry refused ({status}): {}",
                String::from_utf8_lossy(&body)
            );
        }
        Ok(serde_json::from_slice(&body)?)
    }

    pub async fn head(&self) -> Result<Head> {
        let head: Head = self.get("/v1/head").await?;
        if !head.verify() {
            bail!("the registry's head signature is invalid");
        }
        Ok(head)
    }

    /// The entry that currently defines `name@rev`, with its signature checked.
    pub async fn resolve(&self, name: &str, rev: &str) -> Result<Entry> {
        let entry: Entry = self.get(&format!("/v1/resolve/{name}/{rev}")).await?;
        match &entry.statement.claim {
            Claim::Publish {
                name: n,
                rev: r,
                root,
                ..
            } if n == name && r == rev && store::is_hash(root) && entry.statement.verify() => {
                Ok(entry)
            }
            _ => Err(anyhow!("the registry sent a bad entry for {name}@{rev}")),
        }
    }

    pub async fn index(&self) -> Result<Vec<Hit>> {
        self.get("/v1/index").await
    }

    pub async fn search(&self, query: &str) -> Result<Vec<Hit>> {
        let url =
            reqwest::Url::parse_with_params(&format!("{}/v1/search", self.base), [("q", query)])?;
        let resp = self.http.get(url).send().await?;
        if !resp.status().is_success() {
            bail!("search failed: {}", resp.status());
        }
        Ok(serde_json::from_slice(&resp.bytes().await?)?)
    }

    /// The operator's anchor nodes, checked against the operator's signature. `operator`
    /// pins the registry's key; without it, the key that signs the registry's head is
    /// trusted. Empty if none are set.
    pub async fn anchors(&self, operator: Option<&str>) -> Result<Vec<libp2p::Multiaddr>> {
        let operator = match operator {
            Some(op) => op.to_string(),
            None => head_operator(&self.head().await?),
        };
        let resp = self
            .http
            .get(format!("{}/v1/anchors", self.base))
            .send()
            .await
            .with_context(|| format!("reach the registry at {}", self.base))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(Vec::new());
        }
        if !resp.status().is_success() {
            bail!("anchors failed: {}", resp.status());
        }
        let entry: Entry = serde_json::from_slice(&resp.bytes().await?)?;
        let st = &entry.statement;
        let Claim::Anchors { addrs } = &st.claim else {
            bail!("the registry sent a bad anchors entry");
        };
        // Signed by the root key, or by an online key the log delegated to at the time.
        let allowed = st.key() == operator
            || self
                .audit(Some(&operator))
                .await
                .is_ok_and(|(log, _)| log.entries.iter().any(|e| e.statement == *st));
        if !st.verify() || !allowed {
            bail!("the anchor list is not signed by the registry operator {operator}");
        }
        addrs.iter().map(|a| valid_anchor(a)).collect()
    }

    /// Entries from `from` onwards.
    pub async fn entries(&self, from: u64) -> Result<Vec<Entry>> {
        let mut out = Vec::new();
        loop {
            let page: Vec<Entry> = self
                .get(&format!("/v1/log?from={}", from + out.len() as u64))
                .await?;
            if page.is_empty() {
                return Ok(out);
            }
            out.extend(page);
        }
    }

    /// Download the whole log, replay it, and check it against the signed head. `operator`
    /// pins the registry's key; without it, the key that signed the head is trusted.
    pub async fn audit(&self, operator: Option<&str>) -> Result<(Log, Head)> {
        let head = self.head().await?;
        let op = operator
            .map(str::to_string)
            .unwrap_or_else(|| head_operator(&head));
        let entries = self.entries(0).await?;
        let log = Log::replay(op.clone(), entries)?;
        if !log.signs_head(&head) {
            bail!(
                "the registry's head is signed by {}, which is neither {op} nor a key it \
                 currently delegates to",
                head.signature.key
            );
        }
        // The log may have grown since we read the head; the head must match a prefix.
        let size = head.size as usize;
        if size > log.entries.len() {
            bail!(
                "the head claims {size} entries but the log has {}",
                log.entries.len()
            );
        }
        let at_head = if size == 0 {
            GENESIS.to_string()
        } else {
            log.entries[size - 1].hash()
        };
        if at_head != head.hash {
            bail!("the log does not match the registry's signed head");
        }
        Ok((log, head))
    }
}

/// Keeps a store's blocklist and gates in step with a registry's log.
pub struct Follower {
    client: Client,
    operator: Option<String>,
    log: Option<Log>,
}

impl Follower {
    /// `operator` pins the registry's key; without it, the key that signs the registry's
    /// head the first time is trusted from then on.
    pub fn new(url: &str, operator: Option<String>) -> Result<Follower> {
        if let Some(op) = &operator {
            sign::parse_public_key(op)?;
        }
        Ok(Follower {
            client: Client::new(url)?,
            operator,
            log: None,
        })
    }

    /// Fetch new log entries, check them, and apply the blocklist to `store`. Returns how
    /// many blocked chunks and manifests were deleted from disk.
    pub async fn sync(&mut self, store: &store::Store) -> Result<usize> {
        let head = self.client.head().await?;
        if self.log.is_none() {
            let operator = self
                .operator
                .clone()
                .unwrap_or_else(|| head_operator(&head));
            self.log = Some(Log::new(operator));
        }
        let log = self.log.as_mut().unwrap();
        let new = self.client.entries(log.entries.len() as u64).await?;
        log.extend(new)?;
        if !log.signs_head(&head) {
            bail!(
                "the registry's head is signed by {}, which is neither the operator {} nor a \
                 key it currently delegates to",
                head.signature.key,
                log.operator
            );
        }
        store.set_gates(log.ticket_issuers(), log.gates().clone());
        Ok(store.set_blocked(log.blocked().cloned().collect()))
    }

    pub fn log(&self) -> Option<&Log> {
        self.log.as_ref()
    }
}

/// Follow a registry's blocklist in the background, checking every `every`.
pub fn follow(
    store: Arc<store::Store>,
    mut follower: Follower,
    every: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match follower.sync(&store).await {
                Ok(0) => {}
                Ok(n) => eprintln!("blocklist: deleted {n} blocked item(s) from the store"),
                Err(e) => eprintln!("blocklist: {e:#}"),
            }
            tokio::time::sleep(every).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(dir: &FsPath, name: &str) -> SigningKey {
        sign::generate_key(&dir.join(name)).unwrap()
    }

    fn publish(k: &SigningKey, name: &str, rev: &str, root: &str) -> Statement {
        Statement::new(
            k,
            Claim::Publish {
                name: name.into(),
                rev: rev.into(),
                root: root.into(),
                description: "a small test model".into(),
                gated: None,
            },
        )
    }

    #[test]
    fn root_key_offline_with_a_delegated_online_key() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("reg");
        let operator = Registry::open(&data).unwrap().operator();
        // Take the root key off the server.
        let root_path = dir.path().join("root.key");
        fs::rename(data.join("operator.key"), &root_path).unwrap();
        let root = sign::load_key(&root_path).unwrap();
        let reg = Registry::open(&data).unwrap();
        assert_eq!(reg.operator(), operator);
        assert!(reg.operator_key().is_none());
        let online = reg.online_public();

        // Until delegated, the online key's head isn't accepted, nor its blocks.
        let block = Claim::Block {
            hash: "aa".repeat(32),
            reason: String::new(),
        };
        let head = reg.head();
        assert!(!reg.with_log(|log| log.signs_head(&head)));
        assert!(
            reg.submit(Statement::new(&reg.online, block.clone()))
                .is_err()
        );

        // The root delegates; the online key can't delegate, even to itself.
        let delegate = |k: &SigningKey, key: &str, expires: u64| {
            Statement::new(
                k,
                Claim::Delegate {
                    key: key.into(),
                    expires,
                },
            )
        };
        assert!(
            reg.submit(delegate(&reg.online, &online, now() + 3600))
                .is_err()
        );
        reg.submit(delegate(&root, &online, now() + 3600)).unwrap();
        let head = reg.head();
        assert_eq!(head.signature.key, online);
        assert!(reg.with_log(|log| log.signs_head(&head)));
        reg.submit(Statement::new(reg.operator_key().unwrap(), block))
            .unwrap();
        // Tickets the online key issues are accepted while the delegation lasts.
        let peer = libp2p::PeerId::random().to_string();
        let t = AccessTicket::new(reg.operator_key().unwrap(), "x/y", &peer, now() + 60);
        let issuers = reg.with_log(|log| log.ticket_issuers());
        assert!(
            issuers
                .iter()
                .any(|(k, until)| t.valid_for(k, &peer) && now() < *until)
        );
        // Delegations end when they expire, and the root can revoke one early.
        reg.with_log(|log| {
            assert!(log.acts_as_operator(&online, now()));
            assert!(!log.acts_as_operator(&online, now() + 3601));
        });
        reg.submit(delegate(&root, "", 0)).unwrap();
        assert!(reg.operator_key().is_none());

        // The whole log still replays for an auditor who pinned the root key.
        let entries = reg.with_log(|log| log.entries.clone());
        Log::replay(operator, entries).unwrap();
    }

    #[test]
    fn gates_and_tickets() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::open(&dir.path().join("reg")).unwrap();
        let alice = key(dir.path(), "alice");
        let (r1, r2) = ("aa".repeat(32), "bb".repeat(32));

        // Statements without a gate sign the same bytes as before gates existed.
        let st = publish(&alice, "acme/tiny", "main", &r1);
        assert!(!serde_json::to_string(&st).unwrap().contains("gated"));

        // A publisher can gate their own model; only the operator can gate others'.
        let mut gated = publish(&alice, "acme/llama", "main", &r2);
        if let Claim::Publish { gated: g, .. } = &mut gated.claim {
            *g = Some("meta-llama/Llama-3.2-1B".into());
        }
        gated = Statement::new(&alice, gated.claim);
        reg.submit(gated).unwrap();
        let gate = |root: &str, repo: &str| Claim::Gate {
            root: root.into(),
            repo: repo.into(),
        };
        assert!(
            reg.submit(Statement::new(&alice, gate(&r1, "x/y")))
                .is_err()
        );
        reg.submit(Statement::new(
            reg.operator_key().unwrap(),
            gate(&r1, "x/y"),
        ))
        .unwrap();
        reg.with_log(|log| {
            assert_eq!(log.gate(&r1), Some("x/y"));
            assert_eq!(log.gate(&r2), Some("meta-llama/Llama-3.2-1B"));
        });
        // Re-publishing without a gate doesn't lift it; the operator can.
        reg.submit(publish(&alice, "acme/llama", "v2", &r2))
            .unwrap();
        assert!(reg.with_log(|log| log.gate(&r2).is_some()));
        reg.submit(Statement::new(reg.operator_key().unwrap(), gate(&r2, "")))
            .unwrap();
        assert!(reg.with_log(|log| log.gate(&r2).is_none()));

        // Tickets are bound to the operator, the peer and the clock.
        let peer = libp2p::PeerId::random().to_string();
        let t = AccessTicket::new(reg.operator_key().unwrap(), "x/y", &peer, now() + 60);
        assert!(t.valid_for(&reg.operator(), &peer));
        assert!(!t.valid_for(&reg.operator(), &libp2p::PeerId::random().to_string()));
        assert!(!t.valid_for(&sign::public_key_string(&alice.verifying_key()), &peer));
        let old = AccessTicket::new(reg.operator_key().unwrap(), "x/y", &peer, now() - 1);
        assert!(!old.valid_for(&reg.operator(), &peer));
        let mut forged = t.clone();
        forged.repo = "other/repo".into();
        assert!(!forged.valid_for(&reg.operator(), &peer));
    }

    #[test]
    fn names_are_owned_and_the_log_replays() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::open(dir.path()).unwrap();
        let (alice, bob) = (key(dir.path(), "a"), key(dir.path(), "b"));
        let bob_pk = sign::public_key_string(&bob.verifying_key());
        let r1 = "11".repeat(32);
        let r2 = "22".repeat(32);

        reg.submit(publish(&alice, "acme/tiny", "main", &r1))
            .unwrap();
        // Bob can't publish under acme until Alice grants him.
        assert!(reg.submit(publish(&bob, "acme/tiny", "main", &r2)).is_err());
        reg.submit(Statement::new(
            &alice,
            Claim::Grant {
                org: "acme".into(),
                key: bob_pk.clone(),
            },
        ))
        .unwrap();
        reg.submit(publish(&bob, "acme/tiny", "main", &r2)).unwrap();
        // The same statement can't be replayed.
        let st = publish(&bob, "bob/x", "v1", &r1);
        reg.submit(st.clone()).unwrap();
        assert!(reg.submit(st).is_err());
        // Only the operator can block.
        let block = Claim::Block {
            hash: r1.clone(),
            reason: "test".into(),
        };
        assert!(reg.submit(Statement::new(&alice, block.clone())).is_err());
        reg.submit(Statement::new(reg.operator_key().unwrap(), block))
            .unwrap();
        assert!(reg.submit(publish(&bob, "bob/y", "main", &r1)).is_err());

        reg.with_log(|log| {
            let e = log.resolve("acme/tiny", "main").unwrap();
            assert!(matches!(&e.statement.claim, Claim::Publish { root, .. } if *root == r2));
            assert_eq!(log.owners("acme").len(), 2);
            // bob/x points at a blocked root, so search hides it.
            let names: Vec<_> = log.search("tiny", 10).into_iter().map(|h| h.name).collect();
            assert_eq!(names, ["acme/tiny"]);
            assert!(log.search("bob", 10).is_empty());
            assert_eq!(log.search("small model", 10).len(), 1);
        });

        // Reopening replays the file; a tampered file is refused.
        drop(reg);
        let reg = Registry::open(dir.path()).unwrap();
        let head = reg.head();
        assert!(head.verify());
        assert_eq!(head.size, 5);
        drop(reg);
        let path = dir.path().join("log.jsonl");
        let text = fs::read_to_string(&path).unwrap();
        fs::write(&path, text.replace(&r2, &"33".repeat(32))).unwrap();
        assert!(Registry::open(dir.path()).is_err());
    }

    #[test]
    fn refs_parse() {
        assert_eq!(
            parse_ref("org/model").unwrap(),
            ("org/model".into(), "main".into())
        );
        assert_eq!(
            parse_ref("org/model@v1.0").unwrap(),
            ("org/model".into(), "v1.0".into())
        );
        for bad in [
            "model", "org/", "/m", "org/m/x", "org/../x", "org/m@", "o g/m",
        ] {
            assert!(parse_ref(bad).is_err(), "{bad}");
        }
    }
}
