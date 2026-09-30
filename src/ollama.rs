//! Ollama models from LAN peers first, and from the registry only when nobody nearby has
//! them.
//!
//! `chungus ollama` sits on Ollama's default port (11434) in front of a stock Ollama moved
//! to 11433. It forwards the whole API unchanged and steps in on one call, `POST
//! /api/pull`: before forwarding it, it fills the model's blobs into `$OLLAMA_MODELS` from
//! this machine's store and LAN peers, and from the registry for anything they lack.
//! Ollama then does its normal pull, finds every blob already there and just writes the
//! manifest. `chungus ollama pull` does the same fill without a daemon.
//!
//! Ollama's blob files are the only copy of the weights. The store keeps a manifest for
//! each Ollama manifest and *links* its blobs (see [`crate::store::Link`]), so the
//! machine can seed them to peers at the cost of about 0.15% extra disk.
//!
//! Trust:
//! - Online, the registry's manifest (a few KB, fetched on every pull) pins every blob by
//!   sha256. A blob written into `$OLLAMA_MODELS` always matches its digest, whoever sent
//!   the bytes; a peer whose copy doesn't match isn't asked for that blob again.
//! - Offline, a tag is resolved from this store, or from LAN peers when two of them agree
//!   on its digest (or one does, with `--trust-peers`).
//! - Only models the registry serves without a login are fetched, so nothing private is
//!   ever offered to peers.

use anyhow::{Context, Result, anyhow, bail};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::{StreamExt, stream};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::manifest::{ChunkRef, FileEntry, Manifest};
use crate::net;
use crate::p2p;
use crate::registry;
use crate::sign;
use crate::store::{self, Link, Store};

/// Where the shim listens: Ollama's own default, so clients need no settings.
pub const DEFAULT_LISTEN: &str = "127.0.0.1:11434";
/// Where the real Ollama listens once moved aside (`OLLAMA_HOST=127.0.0.1:11433`).
pub const DEFAULT_BACKEND: &str = "http://127.0.0.1:11433";
pub const DEFAULT_HOST: &str = "registry.ollama.ai";
/// Names under this host (`chungus.io/<org>/<model>:<tag>`) are models published to the
/// chungus registry, resolved there and fetched from peers only.
pub const REGISTRY_HOST: &str = "chungus.io";
/// How blobs are chunked when indexed. Bumped when chunking changes (2: GGUF files are
/// cut per tensor), so `import` re-indexes models indexed the old way.
pub const INDEX_VERSION: u32 = 2;
pub const DEFAULT_UPSTREAM: &str = "https://registry.ollama.ai";
const MANIFEST_TYPE: &str = "application/vnd.docker.distribution.manifest.v2+json";
/// Blobs smaller than this (templates, params, licenses) are copied into the store like
/// any other file; bigger ones are linked where Ollama keeps them.
const LINK_MIN: u64 = 1 << 20;
/// Blobs at least this big download from the registry in parallel ranges.
const PARALLEL_MIN: u64 = 64 << 20;
const PART_SIZE: u64 = 100 << 20;
const MAX_PARTS: u64 = 16;
/// Chunk requests to peers in flight at once.
const CONCURRENCY: usize = 32;
/// Gaps up to this size between missing chunks are fetched from the registry anyway, so
/// a partly shared blob takes a few range requests rather than one per chunk.
const RANGE_GAP: u64 = 1 << 20;

// ---------- names ----------

/// A model name as Ollama reads it: `[host/][namespace/]model[:tag]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name {
    pub host: String,
    pub namespace: String,
    pub model: String,
    pub tag: String,
}

impl Name {
    /// `llama3.2` is `registry.ollama.ai/library/llama3.2:latest`, as it is in Ollama.
    pub fn parse(s: &str) -> Result<Name> {
        let s = s
            .trim()
            .trim_start_matches("https://")
            .trim_start_matches("http://");
        if s.contains('@') {
            bail!("{s}: pulling by digest isn't supported; name a tag");
        }
        let (path, tag) = match s.rsplit_once(':') {
            Some((p, t)) if !t.contains('/') => (p, t),
            _ => (s, "latest"),
        };
        let parts: Vec<&str> = path.split('/').collect();
        let (host, namespace, model) = match parts.as_slice() {
            [m] => (DEFAULT_HOST, "library", *m),
            [n, m] => (DEFAULT_HOST, *n, *m),
            [h, n, m] => (*h, *n, *m),
            _ => bail!("{s}: expected [host/][namespace/]model[:tag]"),
        };
        let ok = |p: &str, extra: &[u8]| {
            !p.is_empty()
                && p != "."
                && p != ".."
                && p.len() <= 128
                && p.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b) || extra.contains(&b))
        };
        if !ok(host, b":") || !ok(namespace, b"") || !ok(model, b"") || !ok(tag, b"") {
            bail!("{s} is not a valid model name");
        }
        Ok(Name {
            host: host.into(),
            namespace: namespace.into(),
            model: model.into(),
            tag: tag.into(),
        })
    }

    /// `host/namespace/model`, as used in registry URLs after `/v2/`.
    fn repo(&self) -> String {
        format!("{}/{}", self.namespace, self.model)
    }

    /// Where Ollama keeps this tag's manifest.
    pub fn manifest_path(&self, models: &Path) -> PathBuf {
        models
            .join("manifests")
            .join(&self.host)
            .join(&self.namespace)
            .join(&self.model)
            .join(&self.tag)
    }

    /// Store metadata key for this tag's [`TagRecord`].
    fn meta_key(&self) -> String {
        format!(
            "ollama/tags/{}/{}/{}/{}.json",
            self.host.replace(':', "%3A"),
            self.namespace,
            self.model,
            self.tag
        )
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.host == DEFAULT_HOST && self.namespace == "library" {
            write!(f, "{}:{}", self.model, self.tag)
        } else if self.host == DEFAULT_HOST {
            write!(f, "{}/{}:{}", self.namespace, self.model, self.tag)
        } else {
            write!(
                f,
                "{}/{}/{}:{}",
                self.host, self.namespace, self.model, self.tag
            )
        }
    }
}

// ---------- Ollama's manifests ----------

#[derive(Serialize, Deserialize, Clone)]
pub struct OllamaManifest {
    #[serde(rename = "schemaVersion", default)]
    pub schema_version: u32,
    #[serde(rename = "mediaType", default)]
    pub media_type: String,
    pub config: Layer,
    pub layers: Vec<Layer>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Layer {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
}

impl OllamaManifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let m: OllamaManifest = serde_json::from_slice(bytes).context("not an Ollama manifest")?;
        for l in m.blobs() {
            hex_of(&l.digest)?;
        }
        Ok(m)
    }

    /// The config and every layer, each digest once.
    pub fn blobs(&self) -> Vec<&Layer> {
        let mut seen = HashSet::new();
        std::iter::once(&self.config)
            .chain(&self.layers)
            .filter(|l| seen.insert(l.digest.as_str()))
            .collect()
    }
}

impl Layer {
    /// This blob's path inside a chungus manifest. Weights get `.gguf`, so they go through
    /// the GGUF segmenter where there is one.
    fn store_path(&self) -> String {
        let hex = self.digest.trim_start_matches("sha256:");
        let gguf = [".model", ".projector", ".adapter", ".draft"]
            .iter()
            .any(|s| self.media_type.ends_with(s));
        format!("blobs/sha256-{hex}{}", if gguf { ".gguf" } else { "" })
    }

    fn file_name(&self) -> String {
        self.digest.replace(':', "-")
    }
}

/// The 64 hex digits of a `sha256:<hex>` digest.
fn hex_of(digest: &str) -> Result<&str> {
    match digest.strip_prefix("sha256:") {
        Some(h) if h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) => {
            Ok(h)
        }
        _ => bail!("bad digest {digest:?}"),
    }
}

fn sha256_digest(bytes: &[u8]) -> String {
    format!("sha256:{}", to_hex(&sha2::Sha256::digest(bytes)))
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Path in every chungus manifest of an Ollama model for Ollama's own manifest, verbatim.
const MANIFEST_FILE: &str = "ollama/manifest.json";

/// `ollama/tags/<host>/<ns>/<model>/<tag>.json`: which manifest a tag pointed to when it
/// was last pulled or imported.
#[derive(Serialize, Deserialize, Clone)]
pub struct TagRecord {
    pub manifest_digest: String,
    pub root: String,
    pub checked_at: u64,
    /// The [`INDEX_VERSION`] `root` was made with.
    #[serde(default = "first_index")]
    pub index: u32,
}

fn first_index() -> u32 {
    1
}

/// Store metadata key mapping an Ollama manifest digest to its chungus manifest root.
/// Peers answer `GET /v1/ollama/<digest>` from it.
pub fn digest_key(digest: &str) -> Result<String> {
    Ok(format!("ollama/digests/sha256-{}", hex_of(digest)?))
}

/// The manifest root indexing Ollama manifest `digest`, if the store has that manifest.
pub fn root_for_digest(store: &Store, digest: &str) -> Option<String> {
    let root = String::from_utf8(store.get_meta(&digest_key(digest).ok()?).ok()?).ok()?;
    store.get_manifest(root.trim()).ok().map(|m| m.root)
}

/// Every (Ollama manifest digest, manifest root) pair the store has indexed.
pub fn digest_records(store: &Store) -> Vec<(String, String)> {
    store
        .list_meta("ollama/digests")
        .into_iter()
        .filter_map(|name| {
            let digest = name.replacen('-', ":", 1);
            hex_of(&digest).ok()?;
            let root = store.get_meta(&format!("ollama/digests/{name}")).ok()?;
            let root = String::from_utf8(root).ok()?.trim().to_string();
            store::is_hash(&root).then_some((digest, root))
        })
        .collect()
}

/// Where a fill gets chunks: this store, LAN peers, then swarm peers holding the model.
struct Sources {
    lan: Vec<String>,
    swarm: Option<p2p::ModelSources>,
}

/// What a fill will write: Ollama's manifest, and what is known about its blobs.
struct Plan {
    manifest_bytes: Vec<u8>,
    /// The registry has the blobs, and Ollama can finish the pull itself.
    online: bool,
    /// A chungus manifest for this Ollama manifest, giving each blob's chunks.
    known: Option<Manifest>,
    sources: Sources,
}

// ---------- progress ----------

/// What a fill is doing, for progress bars.
#[derive(Default)]
pub struct Progress {
    blobs: Mutex<Vec<(String, u64, Arc<AtomicU64>)>>,
}

impl Progress {
    fn start(&self, digest: &str, total: u64) -> Arc<AtomicU64> {
        let done = Arc::new(AtomicU64::new(0));
        self.blobs
            .lock()
            .unwrap()
            .push((digest.to_string(), total, done.clone()));
        done
    }

    /// (digest, total, completed) for each blob started so far.
    pub fn snapshot(&self) -> Vec<(String, u64, u64)> {
        self.blobs
            .lock()
            .unwrap()
            .iter()
            .map(|(d, t, c)| (d.clone(), *t, c.load(Ordering::Relaxed).min(*t)))
            .collect()
    }
}

#[derive(Default, Debug)]
pub struct FillStats {
    pub blobs: usize,
    /// Blobs already in `$OLLAMA_MODELS`.
    pub present: usize,
    /// Bytes copied from chunks this machine already had (another tag of the model, say).
    pub local_bytes: u64,
    pub peer_bytes: u64,
    pub upstream_bytes: u64,
}

/// A filled model: every blob is in `$OLLAMA_MODELS` and indexed in the store.
pub struct Filled {
    pub manifest_bytes: Vec<u8>,
    pub digest: String,
    pub root: String,
    /// Whether the registry was reached, so forwarding the pull to Ollama will work.
    pub online: bool,
    pub stats: FillStats,
}

// ---------- the filler ----------

pub struct Ollama {
    pub store: Arc<Store>,
    pub models: PathBuf,
    upstream: Option<String>,
    peers: RwLock<Vec<String>>,
    trust_peers: bool,
    client: reqwest::Client,
    peer_client: reqwest::Client,
    /// Blob digests whose copy assembled from peers failed its sha256 check.
    distrust: Mutex<HashSet<String>>,
    /// One fill at a time, so two pulls of the same model don't write the same blob.
    fill_lock: tokio::sync::Mutex<()>,
    /// A swarm node to find models on beyond the LAN.
    swarm: RwLock<Option<p2p::Node>>,
    /// The chungus registry, for `chungus.io/...` names.
    registry_url: RwLock<String>,
}

impl Ollama {
    /// `upstream` replaces `https://registry.ollama.ai` (for tests and mirrors); None means
    /// offline.
    pub fn new(
        store: Arc<Store>,
        models: PathBuf,
        upstream: Option<String>,
        peers: Vec<String>,
        trust_peers: bool,
    ) -> Result<Self> {
        Ok(Ollama {
            store,
            models,
            upstream: upstream.map(|u| u.trim_end_matches('/').to_string()),
            peers: RwLock::new(peers),
            trust_peers,
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .read_timeout(Duration::from_secs(60))
                .build()?,
            peer_client: net::client()?,
            distrust: Default::default(),
            fill_lock: Default::default(),
            swarm: Default::default(),
            registry_url: RwLock::new(registry::DEFAULT_URL.to_string()),
        })
    }

    /// Also look for models on the internet swarm, through `node`.
    pub fn set_swarm(&self, node: p2p::Node) {
        *self.swarm.write().unwrap() = Some(node);
    }

    /// The chungus registry that `chungus.io/...` names resolve in.
    pub fn set_registry(&self, url: &str) {
        *self.registry_url.write().unwrap() = url.trim_end_matches('/').to_string();
    }

    fn swarm(&self) -> Option<p2p::Node> {
        self.swarm.read().unwrap().clone()
    }

    pub fn set_peers(&self, peers: Vec<String>) {
        *self.peers.write().unwrap() = peers;
    }

    fn peers(&self) -> Vec<String> {
        self.peers.read().unwrap().clone()
    }

    pub fn is_offline(&self) -> bool {
        self.upstream.is_none()
    }

    /// Base URL of the registry serving `name`, or None when offline.
    fn registry(&self, name: &Name) -> Option<String> {
        let up = self.upstream.as_ref()?;
        Some(if name.host == DEFAULT_HOST {
            up.clone()
        } else {
            format!("https://{}", name.host)
        })
    }

    // ----- registry -----

    /// GET from the registry, answering a bearer challenge anonymously if one comes back.
    async fn registry_get(
        &self,
        url: &str,
        token: &mut Option<String>,
        accept: Option<&str>,
    ) -> reqwest::Result<reqwest::Response> {
        let send = |token: &Option<String>| {
            let mut req = self.client.get(url);
            if let Some(a) = accept {
                req = req.header(header::ACCEPT, a);
            }
            if let Some(t) = token {
                req = req.bearer_auth(t);
            }
            req.send()
        };
        let resp = send(token).await?;
        if resp.status() != reqwest::StatusCode::UNAUTHORIZED || token.is_some() {
            return Ok(resp);
        }
        let Some(challenge) = resp
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_challenge)
        else {
            return Ok(resp);
        };
        #[derive(Deserialize)]
        struct Token {
            #[serde(alias = "access_token")]
            token: String,
        }
        let query: Vec<String> = challenge
            .1
            .iter()
            .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
            .collect();
        let sep = if challenge.0.contains('?') { '&' } else { '?' };
        let got = self
            .client
            .get(format!("{}{sep}{}", challenge.0, query.join("&")))
            .send()
            .await
            .ok()
            .filter(|r| r.status().is_success());
        if let Some(r) = got
            && let Ok(bytes) = r.bytes().await
            && let Ok(t) = serde_json::from_slice::<Token>(&bytes)
        {
            *token = Some(t.token);
            return send(token).await;
        }
        Ok(resp)
    }

    /// The registry's manifest for `name`. `Ok(None)` means the registry couldn't be
    /// reached; a missing or private model is an error.
    async fn upstream_manifest(
        &self,
        name: &Name,
        token: &mut Option<String>,
    ) -> Result<Option<Vec<u8>>> {
        let Some(base) = self.registry(name) else {
            return Ok(None);
        };
        let url = format!("{base}/v2/{}/manifests/{}", name.repo(), name.tag);
        let resp = match self.registry_get(&url, token, Some(MANIFEST_TYPE)).await {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };
        match resp.status().as_u16() {
            200 => {}
            404 => bail!("{name} not found on {}", name.host),
            401 | 403 => bail!(
                "{name} needs a login; chungus only shares public models, so pull it with \
                 Ollama directly"
            ),
            s if s >= 500 => return Ok(None),
            s => bail!("{} answered {s} for {name}", name.host),
        }
        let expected = resp
            .headers()
            .get("docker-content-digest")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let bytes = resp.bytes().await?.to_vec();
        if let Some(d) = expected
            && d != sha256_digest(&bytes)
        {
            bail!("the manifest for {name} doesn't match its digest {d}");
        }
        Ok(Some(bytes))
    }

    // ----- resolving a tag -----

    /// What to fill for `name`: its manifest, and whoever has its chunks.
    async fn resolve(&self, name: &Name, token: &mut Option<String>) -> Result<Plan> {
        if name.host == REGISTRY_HOST {
            return self.resolve_published(name).await;
        }
        let (manifest_bytes, online) = self.resolve_tag(name, token).await?;
        let (known, swarm) = self.find_chungus(&sha256_digest(&manifest_bytes)).await;
        Ok(Plan {
            manifest_bytes,
            online,
            known,
            sources: Sources {
                lan: self.peers(),
                swarm,
            },
        })
    }

    /// A model published to the chungus registry as `org/model@tag`: its manifest must be
    /// signed by the publisher, and its blobs come from peers and the swarm only.
    async fn resolve_published(&self, name: &Name) -> Result<Plan> {
        if !registry::valid_rev(&name.tag) {
            bail!("{name}: chungus registry tags are at most 64 letters, digits, '-', '_' or '.'");
        }
        let reg = self.registry_url.read().unwrap().clone();
        let entry = registry::Client::new(&reg)?
            .resolve(&name.repo(), &name.tag)
            .await
            .with_context(|| format!("resolve {name} in {reg}"))?;
        let registry::Claim::Publish { root, .. } = entry.statement.claim else {
            bail!("{name} is not a published model");
        };
        let trust = [sign::parse_public_key(&entry.statement.signature.key)?];
        let lan = self.peers();
        let signed = |store: &Store| {
            store.get_manifest(&root).is_ok()
                && sign::trusted_by(&store.signatures(&root).unwrap_or_default(), &root, &trust)
        };
        let mut swarm = None;
        if !signed(&self.store) {
            let from_lan = !lan.is_empty()
                && net::prepare(&self.peer_client, &root, &self.store, &lan, &[], &trust)
                    .await
                    .is_ok();
            if !from_lan {
                let node = self
                    .swarm()
                    .with_context(|| format!("no LAN peer has {name}; try --swarm"))?;
                let (_, src) = p2p::prepare(&node, &root, &self.store, &trust, None).await?;
                swarm = Some(src);
            }
        }
        let m = self.store.get_manifest(&root)?;
        let sources = Sources { lan, swarm };
        let manifest_bytes = self.read_small(&m, MANIFEST_FILE, &sources).await?;
        OllamaManifest::parse(&manifest_bytes)?;
        Ok(Plan {
            manifest_bytes,
            online: false,
            known: Some(m),
            sources,
        })
    }

    /// The manifest `name` points to: from the registry, else this store, else peers.
    async fn resolve_tag(
        &self,
        name: &Name,
        token: &mut Option<String>,
    ) -> Result<(Vec<u8>, bool)> {
        if let Some(bytes) = self.upstream_manifest(name, token).await? {
            return Ok((bytes, true));
        }
        // Offline: this store's own record first.
        if let Some(rec) = self
            .store
            .get_meta(&name.meta_key())
            .ok()
            .and_then(|b| serde_json::from_slice::<TagRecord>(&b).ok())
            && let Ok(bytes) = self.manifest_file(&rec.root, &rec.manifest_digest).await
        {
            return Ok((bytes, false));
        }
        // Then peers, when enough of them agree.
        let mut votes: BTreeMap<String, Vec<TagRecord>> = BTreeMap::new();
        for peer in self.peers() {
            let url = format!("{peer}/v1/meta/{}", name.meta_key());
            if let Ok(r) = self.peer_client.get(url).send().await
                && r.status().is_success()
                && let Ok(bytes) = r.bytes().await
                && let Ok(rec) = serde_json::from_slice::<TagRecord>(&bytes)
            {
                votes
                    .entry(rec.manifest_digest.clone())
                    .or_default()
                    .push(rec);
            }
        }
        let need = if self.trust_peers { 1 } else { 2 };
        let best = votes.into_values().max_by_key(|v| v.len());
        match best {
            Some(recs) if recs.len() >= need => {
                for rec in &recs {
                    if let Ok(bytes) = self.manifest_file(&rec.root, &rec.manifest_digest).await {
                        return Ok((bytes, false));
                    }
                }
                bail!("peers agree on {name}, but none could send its manifest")
            }
            Some(_) => bail!(
                "offline and unverified: only one peer knows {name} (pass --trust-peers to \
                 accept that)"
            ),
            None => bail!("offline, and no peer knows {name}"),
        }
    }

    /// Ollama's manifest from chungus manifest `root`, checked against `digest`.
    async fn manifest_file(&self, root: &str, digest: &str) -> Result<Vec<u8>> {
        let m = self.chungus_manifest(root).await?;
        let sources = Sources {
            lan: self.peers(),
            swarm: None,
        };
        let bytes = self.read_small(&m, MANIFEST_FILE, &sources).await?;
        if sha256_digest(&bytes) != digest {
            bail!("manifest in {root} doesn't match {digest}");
        }
        Ok(bytes)
    }

    /// A small file of `m`, kept in the store's own chunks.
    async fn read_small(&self, m: &Manifest, path: &str, sources: &Sources) -> Result<Vec<u8>> {
        let f = m
            .files
            .iter()
            .find(|f| f.path == path)
            .with_context(|| format!("{} has no {path}; not an Ollama model", m.root))?;
        if f.size > LINK_MIN {
            bail!("{path} is too big");
        }
        let mut bytes = Vec::with_capacity(f.size as usize);
        for c in &f.chunks {
            let (blob, raw, local) = self
                .chunk(&c.hash, c.len as usize, sources)
                .await
                .with_context(|| format!("no peer has chunk {} of {path}", c.hash))?;
            if !local {
                self.store.put(&c.hash, &blob)?;
            }
            bytes.extend(raw);
        }
        if blake3::hash(&bytes).to_hex().as_str() != f.hash {
            bail!("{path} failed verification");
        }
        Ok(bytes)
    }

    /// Chunk `hash` (`len` raw bytes): its blob, its verified raw bytes, and whether it
    /// came from this store. A source that sends bad data is skipped for the next.
    async fn chunk(
        &self,
        hash: &str,
        len: usize,
        sources: &Sources,
    ) -> Option<(Bytes, Vec<u8>, bool)> {
        let check = |blob: Bytes| {
            let hash = hash.to_string();
            async move {
                tokio::task::spawn_blocking(move || {
                    let raw = store::decode(&blob, len).ok()?;
                    (blake3::hash(&raw).to_hex().as_str() == hash).then_some((blob, raw))
                })
                .await
                .ok()
                .flatten()
            }
        };
        if self.store.contains(hash) {
            let (store, h) = (self.store.clone(), hash.to_string());
            if let Ok(Ok(b)) = tokio::task::spawn_blocking(move || store.get(&h)).await
                && let Some((blob, raw)) = check(Bytes::from(b)).await
            {
                return Some((blob, raw, true));
            }
        }
        for peer in net::rotated(&sources.lan, hash) {
            if let Some(b) = net::download_chunk(&self.peer_client, &peer, hash).await
                && let Some((blob, raw)) = check(b).await
            {
                return Some((blob, raw, false));
            }
        }
        if let Some(src) = &sources.swarm {
            for peer in src.sources(hash) {
                if let Some(b) = src.get(peer, hash.to_string()).await
                    && let Some((blob, raw)) = check(b).await
                {
                    return Some((blob, raw, false));
                }
            }
        }
        None
    }

    /// Chungus manifest `root`, from the store or a peer.
    async fn chungus_manifest(&self, root: &str) -> Result<Manifest> {
        if let Ok(m) = self.store.get_manifest(root) {
            return Ok(m);
        }
        let peers = self.peers();
        if peers.is_empty() {
            bail!("no manifest {root}");
        }
        let store = self.store.clone();
        net::prepare(&self.peer_client, root, &store, &peers, &[], &[]).await
    }

    /// The chungus manifest this store, a LAN peer or a swarm peer has for Ollama
    /// manifest `digest`, and the swarm peers holding it when it came from the swarm.
    async fn find_chungus(&self, digest: &str) -> (Option<Manifest>, Option<p2p::ModelSources>) {
        if let Some(root) = root_for_digest(&self.store, digest)
            && let Ok(m) = self.store.get_manifest(&root)
        {
            return (Some(m), None);
        }
        for peer in self.peers() {
            let url = format!("{peer}/v1/ollama/{}", digest.replace(':', "-"));
            let Ok(r) = self.peer_client.get(url).send().await else {
                continue;
            };
            if !r.status().is_success() {
                continue;
            }
            let Ok(root) = r.text().await else { continue };
            if let Ok(m) = self.chungus_manifest(root.trim()).await {
                return (Some(m), None);
            }
        }
        let Some(node) = self.swarm() else {
            return (None, None);
        };
        for peer in node.ollama_providers(digest).await.unwrap_or_default() {
            if let Ok(p2p::Response::Root(Some(root))) = node
                .request(peer, p2p::Request::Ollama(digest.to_string()))
                .await
                && let Ok((m, src)) = p2p::prepare(&node, &root, &self.store, &[], None).await
            {
                return (Some(m), Some(src));
            }
        }
        (None, None)
    }

    /// Swarm peers holding the chunks of `root`, when this node is on the swarm.
    async fn swarm_sources(&self, root: &str) -> Option<p2p::ModelSources> {
        let node = self.swarm()?;
        p2p::prepare(&node, root, &self.store, &[], None)
            .await
            .ok()
            .map(|(_, src)| src)
    }

    // ----- filling -----

    /// Put every blob of `name` into `$OLLAMA_MODELS`, and index the model in the store.
    pub async fn fill(&self, name: &Name, progress: &Progress) -> Result<Filled> {
        let _one = self.fill_lock.lock().await;
        let mut token = None;
        let Plan {
            manifest_bytes,
            online,
            known,
            mut sources,
        } = self.resolve(name, &mut token).await?;
        let digest = sha256_digest(&manifest_bytes);
        let om = OllamaManifest::parse(&manifest_bytes)?;
        let blobs_dir = self.models.join("blobs");
        fs::create_dir_all(&blobs_dir)
            .with_context(|| format!("create {}", blobs_dir.display()))?;

        // A chungus manifest for this exact Ollama manifest, from anyone, tells us each
        // blob's chunks. Its files are only used where path and size match; the bytes
        // are checked by sha256 anyway.
        // Blobs this store already links, unchanged and indexed the current way, needn't
        // be read again.
        let current = self
            .store
            .get_meta(&name.meta_key())
            .ok()
            .and_then(|b| serde_json::from_slice::<TagRecord>(&b).ok())
            .is_some_and(|r| r.manifest_digest == digest && r.index == INDEX_VERSION);
        let linked: Vec<Link> = known
            .as_ref()
            .filter(|_| current)
            .map(|m| self.store.links(&m.root))
            .unwrap_or_default();
        // A model known from here or the LAN may still have chunks only the swarm has.
        if sources.swarm.is_none()
            && let Some(m) = &known
            && om.blobs().iter().any(|l| {
                !fs::metadata(blobs_dir.join(l.file_name())).is_ok_and(|md| md.len() == l.size)
            })
        {
            sources.swarm = self.swarm_sources(&m.root).await;
        }
        let known: HashMap<&str, &FileEntry> = known
            .as_ref()
            .map(|m| m.files.iter().map(|f| (f.path.as_str(), f)).collect())
            .unwrap_or_default();

        let mut stats = FillStats::default();
        let mut entries: HashMap<String, FileEntry> = HashMap::new();
        for layer in om.blobs() {
            stats.blobs += 1;
            let target = blobs_dir.join(layer.file_name());
            let done = progress.start(&layer.digest, layer.size);
            if fs::metadata(&target).is_ok_and(|md| md.len() == layer.size) {
                done.store(layer.size, Ordering::Relaxed);
                stats.present += 1;
                let path = layer.store_path();
                if let Some(entry) = known.get(path.as_str())
                    && linked
                        .iter()
                        .any(|l| l.path == path && l.target == target && l.is_current())
                {
                    entries.insert(layer.digest.clone(), (*entry).clone());
                }
                continue;
            }
            let partial = blobs_dir.join(format!("{}-chungus-partial", layer.file_name()));
            let blob_url = self
                .registry(name)
                .filter(|_| online)
                .map(|b| format!("{b}/v2/{}/blobs/{}", name.repo(), layer.digest));
            let entry = known
                .get(layer.store_path().as_str())
                .filter(|f| f.size == layer.size)
                .filter(|_| !self.distrust.lock().unwrap().contains(&layer.digest));
            let mut ok = false;
            if let Some(entry) = entry {
                done.store(0, Ordering::Relaxed);
                match self
                    .assemble(
                        entry,
                        &partial,
                        blob_url.as_deref(),
                        &token,
                        &sources,
                        &done,
                        &mut stats,
                    )
                    .await
                {
                    // Indexed here rather than taken from the peer: the same read checks
                    // the blob, and its chunks come out the way this version cuts them.
                    Ok(()) => match index_blob(&partial, &layer.store_path()).await? {
                        (fresh, sha) if sha == layer.digest => {
                            entries.insert(layer.digest.clone(), fresh);
                            ok = true;
                        }
                        _ => {
                            eprintln!("{}: copy from peers failed its sha256 check", layer.digest);
                            self.distrust.lock().unwrap().insert(layer.digest.clone());
                        }
                    },
                    Err(e) if blob_url.is_some() => {
                        eprintln!("{}: {e:#}; downloading it instead", layer.digest)
                    }
                    Err(e) => {
                        let _ = fs::remove_file(&partial);
                        return Err(e);
                    }
                }
            }
            if !ok {
                let Some(url) = &blob_url else {
                    let _ = fs::remove_file(&partial);
                    bail!(
                        "offline, and no peer has {} of {name}",
                        layer.digest.trim_start_matches("sha256:")
                    );
                };
                done.store(0, Ordering::Relaxed);
                self.download(url, &token, layer.size, &partial, &done)
                    .await
                    .with_context(|| format!("download {}", layer.digest))?;
                stats.upstream_bytes += layer.size;
                // Index it now: the same read checks it before Ollama can see it.
                let (entry, sha) = index_blob(&partial, &layer.store_path()).await?;
                if sha != layer.digest {
                    let _ = fs::remove_file(&partial);
                    bail!("download of {} doesn't match its digest", layer.digest);
                }
                entries.insert(layer.digest.clone(), entry);
            }
            done.store(layer.size, Ordering::Relaxed);
            fs::rename(&partial, &target)?;
        }
        let root = self.record(name, &manifest_bytes, &om, entries).await?;
        Ok(Filled {
            manifest_bytes,
            digest,
            root,
            online,
            stats,
        })
    }

    /// Write Ollama's manifest for `name`, as `ollama pull` would once every blob is there.
    pub fn write_tag(&self, name: &Name, manifest_bytes: &[u8]) -> Result<()> {
        let path = name.manifest_path(&self.models);
        let dir = path.parent().unwrap();
        fs::create_dir_all(dir)?;
        let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
        tmp.write_all(manifest_bytes)?;
        // Ollama's own files are world-readable; a temp file is private by default.
        let _ = tmp
            .as_file()
            .set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o644));
        tmp.persist(&path).map_err(|e| e.error)?;
        Ok(())
    }

    /// Rebuild a blob in `partial` from `entry`'s chunks: from this store, then peers, then
    /// ranges of the registry's copy for whatever nobody had.
    #[allow(clippy::too_many_arguments)]
    async fn assemble(
        &self,
        entry: &FileEntry,
        partial: &Path,
        blob_url: Option<&str>,
        token: &Option<String>,
        sources: &Sources,
        done: &AtomicU64,
        stats: &mut FillStats,
    ) -> Result<()> {
        let file = Arc::new(create_sized(partial, entry.size)?);
        let mut at: HashMap<&str, (u32, Vec<u64>)> = HashMap::new();
        let mut order = Vec::new();
        let mut offset = 0u64;
        for c in &entry.chunks {
            let slot = at.entry(&c.hash).or_insert_with(|| {
                order.push(c.hash.clone());
                (c.len, Vec::new())
            });
            slot.1.push(offset);
            offset += c.len as u64;
        }
        let at = &at;
        let results = stream::iter(order)
            .map(|hash: String| {
                let (len, offsets) = at[hash.as_str()].clone();
                let bytes = len as u64 * offsets.len() as u64;
                let file = file.clone();
                async move {
                    let (blob, raw, local) = self.chunk(&hash, len as usize, sources).await?;
                    let wire = blob.len() as u64;
                    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                        for o in &offsets {
                            file.write_all_at(&raw, *o)?;
                        }
                        Ok(())
                    })
                    .await
                    .ok()?
                    .ok()?;
                    done.fetch_add(bytes, Ordering::Relaxed);
                    Some((hash, local, wire, bytes))
                }
            })
            .buffer_unordered(CONCURRENCY)
            .collect::<Vec<_>>()
            .await;
        let mut have = HashSet::new();
        for (hash, local, wire, bytes) in results.into_iter().flatten() {
            if local {
                stats.local_bytes += bytes;
            } else {
                stats.peer_bytes += wire;
            }
            have.insert(hash);
        }
        // Whatever no one had comes from the registry, in as few ranges as is sensible.
        let mut missing: Vec<(u64, u64)> = Vec::new();
        let mut offset = 0u64;
        for c in &entry.chunks {
            let end = offset + c.len as u64;
            if !have.contains(&c.hash) {
                match missing.last_mut() {
                    Some(last) if offset - last.1 <= RANGE_GAP => last.1 = end,
                    _ => missing.push((offset, end)),
                }
            }
            offset = end;
        }
        if missing.is_empty() {
            return Ok(());
        }
        let Some(url) = blob_url else {
            bail!("{} chunk ranges are on no peer", missing.len());
        };
        let bytes: u64 = missing.iter().map(|(a, b)| b - a).sum();
        stream::iter(missing)
            .map(|(start, end)| {
                let file = file.clone();
                async move { self.fetch_range(url, token, start, end, &file, done).await }
            })
            .buffer_unordered(4)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<()>>()?;
        stats.upstream_bytes += bytes;
        Ok(())
    }

    /// Download a whole blob into `partial`, in parallel ranges when it is big.
    async fn download(
        &self,
        url: &str,
        token: &Option<String>,
        size: u64,
        partial: &Path,
        done: &AtomicU64,
    ) -> Result<()> {
        let file = Arc::new(create_sized(partial, size)?);
        if size < PARALLEL_MIN {
            return self.fetch_range(url, token, 0, size, &file, done).await;
        }
        let parts = size.div_ceil(PART_SIZE).clamp(2, MAX_PARTS);
        let part = size.div_ceil(parts);
        stream::iter((0..parts).map(|i| (i * part, ((i + 1) * part).min(size))))
            .map(|(start, end)| {
                let file = file.clone();
                async move { self.fetch_range(url, token, start, end, &file, done).await }
            })
            .buffer_unordered(parts as usize)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect()
    }

    /// Write bytes `start..end` of the blob at `url` into `file` at the same offsets.
    async fn fetch_range(
        &self,
        url: &str,
        token: &Option<String>,
        start: u64,
        end: u64,
        file: &Arc<fs::File>,
        done: &AtomicU64,
    ) -> Result<()> {
        if start >= end {
            return Ok(());
        }
        let mut req = self
            .client
            .get(url)
            .timeout(Duration::from_secs(24 * 3600))
            .header(header::RANGE, format!("bytes={start}-{}", end - 1));
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await?;
        let status = resp.status().as_u16();
        // A server that ignores ranges sends the whole blob; take the part we want.
        let mut pos = match status {
            206 => start,
            200 => 0,
            s => bail!("registry answered {s}"),
        };
        let mut body = resp.bytes_stream();
        while pos < end
            && let Some(piece) = body.next().await
        {
            let piece = piece?;
            let lo = start.saturating_sub(pos).min(piece.len() as u64) as usize;
            let hi = (end - pos).min(piece.len() as u64) as usize;
            if lo < hi {
                let at = pos + lo as u64;
                let (file, slice) = (file.clone(), piece.slice(lo..hi));
                tokio::task::spawn_blocking(move || file.write_all_at(&slice, at)).await??;
                done.fetch_add((hi - lo) as u64, Ordering::Relaxed);
            }
            pos += piece.len() as u64;
        }
        if pos < end {
            bail!(
                "registry sent {} of {} bytes",
                pos.saturating_sub(start),
                end - start
            );
        }
        Ok(())
    }

    // ----- indexing -----

    /// Index a model whose blobs are all in `$OLLAMA_MODELS`: its chungus manifest goes in
    /// the store with big blobs linked, and the tag and digest records point at it.
    /// `known` has entries (by blob digest) already checked against their digest; other
    /// blobs are read once, and must match their digest to be indexed.
    async fn record(
        &self,
        name: &Name,
        manifest_bytes: &[u8],
        om: &OllamaManifest,
        known: HashMap<String, FileEntry>,
    ) -> Result<String> {
        let blobs_dir = self.models.join("blobs");
        let mut files = Vec::new();
        let mut links = Vec::new();
        {
            let store = self.store.clone();
            let bytes = manifest_bytes.to_vec();
            files.push(
                tokio::task::spawn_blocking(move || store_bytes(&store, MANIFEST_FILE, &bytes))
                    .await??,
            );
        }
        for layer in om.blobs() {
            let target = blobs_dir.join(layer.file_name());
            let path = layer.store_path();
            let entry = match known.get(&layer.digest) {
                Some(e) => FileEntry {
                    path: path.clone(),
                    ..e.clone()
                },
                None => {
                    let (t, p, d) = (target.clone(), path.clone(), layer.digest.clone());
                    let (entry, sha) =
                        tokio::task::spawn_blocking(move || index_file(&t, &p)).await??;
                    if sha != d {
                        bail!("{} doesn't match its digest", target.display());
                    }
                    entry
                }
            };
            if layer.size < LINK_MIN {
                let (store, t, p) = (self.store.clone(), target.clone(), path.clone());
                let data = fs::read(&t)?;
                let stored =
                    tokio::task::spawn_blocking(move || store_bytes(&store, &p, &data)).await??;
                if stored.hash != entry.hash {
                    bail!("{} changed while being indexed", t.display());
                }
                files.push(stored);
            } else {
                links.push(Link::new(&path, &target)?);
                files.push(entry);
            }
        }
        let m = Manifest::new(files);
        self.store.put_manifest(&m)?;
        self.store.add_links(&m.root, links)?;
        let digest = sha256_digest(manifest_bytes);
        self.store
            .put_meta(&digest_key(&digest)?, m.root.as_bytes())?;
        let rec = TagRecord {
            manifest_digest: digest,
            root: m.root.clone(),
            checked_at: crate::registry::now(),
            index: INDEX_VERSION,
        };
        self.store
            .put_meta(&name.meta_key(), &serde_json::to_vec(&rec)?)?;
        Ok(m.root)
    }

    /// Index models already in `$OLLAMA_MODELS` so this machine can seed them. With no
    /// names, every model. Returns each model indexed and its root.
    pub async fn import(&self, names: &[Name]) -> Result<Vec<(Name, Result<String>)>> {
        let mut out = Vec::new();
        for (name, path) in self.local_tags()? {
            if !names.is_empty() && !names.contains(&name) {
                continue;
            }
            let result = self.import_one(&name, &path).await;
            out.push((name, result));
        }
        for n in names {
            if !out.iter().any(|(m, _)| m == n) {
                out.push((n.clone(), Err(anyhow!("not in {}", self.models.display()))));
            }
        }
        Ok(out)
    }

    async fn import_one(&self, name: &Name, path: &Path) -> Result<String> {
        let _one = self.fill_lock.lock().await;
        let bytes = fs::read(path)?;
        let om = OllamaManifest::parse(&bytes)?;
        let digest = sha256_digest(&bytes);
        // Already indexed the current way, and every linked blob unchanged: nothing to
        // read. Indexed by an older chungus: read again, since the chunks now differ.
        let current = self
            .store
            .get_meta(&name.meta_key())
            .ok()
            .and_then(|b| serde_json::from_slice::<TagRecord>(&b).ok())
            .filter(|r| r.manifest_digest == digest && r.index == INDEX_VERSION);
        if let Some(rec) = current
            && let Ok(m) = self.store.get_manifest(&rec.root)
        {
            let linked = self.store.links(&rec.root).len();
            let big = om.blobs().iter().filter(|l| l.size >= LINK_MIN).count();
            if linked == big
                && m.files
                    .iter()
                    .all(|f| f.chunks.iter().all(|c| self.store.contains(&c.hash)))
            {
                return Ok(rec.root);
            }
        }
        for l in om.blobs() {
            let p = self.models.join("blobs").join(l.file_name());
            if !fs::metadata(&p).is_ok_and(|md| md.len() == l.size) {
                bail!("blob {} is missing or incomplete", l.digest);
            }
        }
        self.record(name, &bytes, &om, HashMap::new()).await
    }

    /// Index `name` for publishing and return its root. Refused for models ollama.com
    /// serves: they are pinned by ollama.com's digests already, and a copy in the chungus
    /// registry would only let someone pass it off under another name.
    pub async fn publishable(&self, name: &Name) -> Result<String> {
        if name.host == REGISTRY_HOST {
            bail!("{name} is already in the chungus registry");
        }
        let path = name.manifest_path(&self.models);
        let bytes = fs::read(&path).with_context(|| format!("Ollama has no {name}"))?;
        let mut token = None;
        match self.upstream_manifest(name, &mut token).await {
            Ok(Some(up)) if up == bytes => {
                bail!("{name} is ollama.com's own model; publish models you made (ollama create)")
            }
            Ok(None) if self.upstream.is_some() => bail!(
                "can't reach {} to check {name} isn't one of its models",
                name.host
            ),
            _ => {}
        }
        self.import_one(name, &path).await
    }

    /// Every tag Ollama has a manifest for.
    fn local_tags(&self) -> Result<Vec<(Name, PathBuf)>> {
        let base = self.models.join("manifests");
        let mut out = Vec::new();
        let Ok(hosts) = fs::read_dir(&base) else {
            return Ok(out);
        };
        let dirs = |p: &Path| -> Vec<(String, PathBuf)> {
            fs::read_dir(p)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
                .collect()
        };
        for host in hosts.flatten() {
            for (ns, ns_path) in dirs(&host.path()) {
                for (model, m_path) in dirs(&ns_path) {
                    for (tag, t_path) in dirs(&m_path) {
                        let full =
                            format!("{}/{ns}/{model}:{tag}", host.file_name().to_string_lossy());
                        if t_path.is_file()
                            && let Ok(name) = Name::parse(&full)
                        {
                            out.push((name, t_path));
                        }
                    }
                }
            }
        }
        out.sort_by_key(|(n, _)| n.to_string());
        Ok(out)
    }
}

/// Parse `Bearer realm="...",service="...",scope="..."` into the realm and its query.
fn parse_challenge(h: &str) -> Option<(String, Vec<(String, String)>)> {
    let rest = h.strip_prefix("Bearer ")?;
    let mut realm = None;
    let mut query = Vec::new();
    for part in rest.split(',') {
        let (k, v) = part.trim().split_once('=')?;
        let v = v.trim_matches('"').to_string();
        if k == "realm" {
            realm = Some(v);
        } else {
            query.push((k.to_string(), v));
        }
    }
    Some((realm?, query))
}

fn url_encode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

/// Create (or truncate) `path` at `size` bytes, readable by Ollama.
fn create_sized(path: &Path, size: u64) -> Result<fs::File> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.set_len(size)?;
    Ok(file)
}

/// [`index_file`] off the async runtime.
async fn index_blob(file: &Path, path: &str) -> Result<(FileEntry, String)> {
    let (file, path) = (file.to_path_buf(), path.to_string());
    tokio::task::spawn_blocking(move || index_file(&file, &path)).await?
}

/// Chunk a file where it lies, without storing anything: its manifest entry (as `path`)
/// and its sha256 digest, from one read.
fn index_file(file: &Path, path: &str) -> Result<(FileEntry, String)> {
    let f = fs::File::open(file).with_context(|| format!("open {}", file.display()))?;
    let len = f.metadata()?.len();
    let map;
    let data: &[u8] = if len == 0 {
        &[]
    } else {
        // Safety: read-only mapping of a file only chungus writes to, and Ollama doesn't
        // change blobs once written.
        map = unsafe { memmap2::Mmap::map(&f) }?;
        &map
    };
    let spans = crate::chunk::chunk(data, &crate::file_segments(Path::new(path), data)?);
    let (chunks, (sha, whole)) = rayon::join(
        || {
            spans
                .par_iter()
                .map(|s| ChunkRef {
                    hash: blake3::hash(&data[s.start..s.end]).to_hex().to_string(),
                    len: (s.end - s.start) as u32,
                    dtype: s.dtype,
                })
                .collect::<Vec<_>>()
        },
        || {
            rayon::join(
                || format!("sha256:{}", to_hex(&sha2::Sha256::digest(data))),
                || {
                    let mut h = blake3::Hasher::new();
                    h.update(data);
                    h.finalize().to_hex().to_string()
                },
            )
        },
    );
    Ok((
        FileEntry {
            path: path.to_string(),
            size: data.len() as u64,
            hash: whole,
            chunks,
        },
        sha,
    ))
}

/// Chunk `data` into the store's own `chunks/` as file `path`.
fn store_bytes(store: &Store, path: &str, data: &[u8]) -> Result<FileEntry> {
    let spans = crate::chunk::chunk(data, &crate::file_segments(Path::new(path), data)?);
    let mut chunks = Vec::new();
    for s in spans {
        let raw = &data[s.start..s.end];
        let hash = blake3::hash(raw).to_hex().to_string();
        if !store.contains(&hash) {
            store.put(&hash, &store::encode(raw, s.dtype)?)?;
        }
        chunks.push(ChunkRef {
            hash,
            len: raw.len() as u32,
            dtype: s.dtype,
        });
    }
    Ok(FileEntry {
        path: path.to_string(),
        size: data.len() as u64,
        hash: blake3::hash(data).to_hex().to_string(),
        chunks,
    })
}

/// Where Ollama keeps models: `$OLLAMA_MODELS`, else `~/.ollama/models`, else the Linux
/// service's `/usr/share/ollama/.ollama/models`.
pub fn default_models() -> PathBuf {
    if let Some(p) = std::env::var_os("OLLAMA_MODELS").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".ollama/models"));
    let service = PathBuf::from("/usr/share/ollama/.ollama/models");
    match home {
        Some(h) if h.exists() || !service.exists() => h,
        _ => service,
    }
}

/// Fail early, with the fix, if chungus can't write Ollama's blob directory.
pub fn check_access(models: &Path) -> Result<()> {
    let blobs = models.join("blobs");
    let probe =
        fs::create_dir_all(&blobs).and_then(|_| tempfile::NamedTempFile::new_in(&blobs).map(drop));
    if let Err(e) = probe {
        bail!(
            "can't write {} ({e}). Run chungus as the user that owns Ollama's models (the \
             `ollama` user for the Linux service; see deploy/ollama), or pass --models",
            blobs.display()
        );
    }
    Ok(())
}

// ---------- the shim ----------

struct Shim {
    ollama: Arc<Ollama>,
    backend: String,
    /// No timeout: generations and pulls stream for as long as they take.
    client: reqwest::Client,
}

/// Ollama's API, forwarded to `backend`, with pulls pre-filled from peers first.
pub fn router(ollama: Arc<Ollama>, backend: &str) -> Result<Router> {
    let shim = Arc::new(Shim {
        ollama,
        backend: backend.trim_end_matches('/').to_string(),
        client: reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(5))
            .build()?,
    });
    Ok(Router::new()
        .route("/api/pull", axum::routing::post(pull))
        .fallback(forward)
        .with_state(shim))
}

/// Forward a request to the real Ollama unchanged, streaming both ways.
async fn forward(State(shim): State<Arc<Shim>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let body = reqwest::Body::wrap_stream(body.into_data_stream());
    send_upstream(&shim, parts.method, path, &parts.headers, body).await
}

async fn send_upstream(
    shim: &Shim,
    method: axum::http::Method,
    path: &str,
    headers: &HeaderMap,
    body: reqwest::Body,
) -> Response {
    let mut out = HeaderMap::new();
    for (k, v) in headers {
        if !is_hop_header(k.as_str()) && k != header::HOST && k != header::CONTENT_LENGTH {
            out.append(k, v.clone());
        }
    }
    let resp = shim
        .client
        .request(method, format!("{}{path}", shim.backend))
        .headers(out)
        .body(body)
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!(
                    "chungus can't reach Ollama at {} ({e}); start it with \
                     OLLAMA_HOST={} ollama serve",
                    shim.backend,
                    shim.backend.trim_start_matches("http://")
                ),
            )
                .into_response();
        }
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut h = HeaderMap::new();
    for (k, v) in resp.headers() {
        if !is_hop_header(k.as_str())
            && let Ok(v) = HeaderValue::from_bytes(v.as_bytes())
        {
            h.append(k.clone(), v);
        }
    }
    let body = Body::from_stream(resp.bytes_stream());
    (status, h, body).into_response()
}

fn is_hop_header(name: &str) -> bool {
    matches!(
        name,
        "connection" | "keep-alive" | "transfer-encoding" | "upgrade" | "te" | "trailer"
    )
}

#[derive(Deserialize)]
struct PullRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    stream: Option<bool>,
}

/// `POST /api/pull`: fill the blobs, then let Ollama finish the pull (or finish it here
/// when the registry can't be reached, since Ollama would fail).
async fn pull(State(shim): State<Arc<Shim>>, headers: HeaderMap, body: Bytes) -> Response {
    let req: PullRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(_) => return forward_bytes(&shim, &headers, body).await,
    };
    let raw = if req.model.is_empty() {
        &req.name
    } else {
        &req.model
    };
    let Ok(name) = Name::parse(raw) else {
        return forward_bytes(&shim, &headers, body).await;
    };
    let streaming = req.stream.unwrap_or(true);
    // Ollama can't finish these pulls itself, so errors are answered here.
    let final_here = shim.ollama.is_offline() || name.host == REGISTRY_HOST;
    let progress = Arc::new(Progress::default());

    if !streaming {
        let filled = shim.ollama.fill(&name, &progress).await;
        return match filled {
            Ok(f) if !f.online => match shim.ollama.write_tag(&name, &f.manifest_bytes) {
                Ok(()) => axum::Json(serde_json::json!({"status": "success"})).into_response(),
                Err(e) => error_json(&e),
            },
            Err(e) if final_here => error_json(&e),
            Err(e) => {
                eprintln!("pull {name}: {e:#}; leaving it to Ollama");
                forward_bytes(&shim, &headers, body).await
            }
            Ok(_) => forward_bytes(&shim, &headers, body).await,
        };
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Bytes>(64);
    tokio::spawn(async move {
        let line = |v: serde_json::Value| Bytes::from(format!("{v}\n"));
        let _ = tx
            .send(line(serde_json::json!({"status": "pulling manifest"})))
            .await;
        let fill = {
            let (ollama, name, progress) = (shim.ollama.clone(), name.clone(), progress.clone());
            tokio::spawn(async move { ollama.fill(&name, &progress).await })
        };
        tokio::pin!(fill);
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        let result = loop {
            tokio::select! {
                r = &mut fill => break r.map_err(anyhow::Error::from).and_then(|r| r),
                _ = tick.tick() => {
                    for (digest, total, completed) in progress.snapshot() {
                        let short = &digest.trim_start_matches("sha256:")[..12];
                        let _ = tx.send(line(serde_json::json!({
                            "status": format!("pulling {short}"),
                            "digest": digest, "total": total, "completed": completed,
                        }))).await;
                    }
                }
            }
        };
        match result {
            Ok(f) if !f.online => {
                let done = [
                    serde_json::json!({"status": "verifying sha256 digest"}),
                    serde_json::json!({"status": "writing manifest"}),
                ];
                for v in done {
                    let _ = tx.send(line(v)).await;
                }
                let last = match shim.ollama.write_tag(&name, &f.manifest_bytes) {
                    Ok(()) => serde_json::json!({"status": "success"}),
                    Err(e) => serde_json::json!({"error": format!("{e:#}")}),
                };
                let _ = tx.send(line(last)).await;
            }
            Err(e) if final_here => {
                let _ = tx
                    .send(line(serde_json::json!({"error": format!("{e:#}")})))
                    .await;
            }
            other => {
                if let Err(e) = other {
                    eprintln!("pull {name}: {e:#}; leaving it to Ollama");
                }
                // Ollama finds the blobs in place and only writes the manifest.
                let resp = send_upstream(
                    &shim,
                    axum::http::Method::POST,
                    "/api/pull",
                    &headers,
                    reqwest::Body::from(body.clone()),
                )
                .await;
                let mut s = resp.into_body().into_data_stream();
                while let Some(Ok(b)) = s.next().await {
                    if tx.send(b).await.is_err() {
                        break;
                    }
                }
            }
        }
    });
    let body =
        Body::from_stream(stream::poll_fn(move |cx| rx.poll_recv(cx)).map(Ok::<_, std::io::Error>));
    ([(header::CONTENT_TYPE, "application/x-ndjson")], body).into_response()
}

async fn forward_bytes(shim: &Shim, headers: &HeaderMap, body: Bytes) -> Response {
    send_upstream(
        shim,
        axum::http::Method::POST,
        "/api/pull",
        headers,
        reqwest::Body::from(body),
    )
    .await
}

fn error_json(e: &anyhow::Error) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        axum::Json(serde_json::json!({"error": format!("{e:#}")})),
    )
        .into_response()
}

// ---------- a registry for `ollama pull --insecure` ----------

/// Registry routes (`/v2/...`) serving this machine's Ollama models, for machines that
/// can't run the shim: `ollama pull --insecure <this host>:11435/library/llama3.2`. A
/// model this machine lacks is filled first (from peers, the swarm, then the registry),
/// so it works as a pull-through cache for the LAN.
pub fn registry_router(ollama: Arc<Ollama>) -> Router {
    use axum::routing::get;
    Router::new()
        .route("/v2/", get(|| async { axum::Json(serde_json::json!({})) }))
        .route("/v2/{ns}/{model}/manifests/{tag}", get(registry_manifest))
        .route("/v2/{ns}/{model}/blobs/{digest}", get(registry_blob))
        .with_state(ollama)
}

async fn registry_manifest(
    State(o): State<Arc<Ollama>>,
    axum::extract::Path((ns, model, tag)): axum::extract::Path<(String, String, String)>,
) -> Response {
    let Ok(name) = Name::parse(&format!("{ns}/{model}:{tag}")) else {
        return registry_error(StatusCode::BAD_REQUEST, "NAME_INVALID", "bad name");
    };
    let path = name.manifest_path(&o.models);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(_) => {
            let filled = match o.fill(&name, &Progress::default()).await {
                Ok(f) => f,
                Err(e) => {
                    return registry_error(
                        StatusCode::NOT_FOUND,
                        "MANIFEST_UNKNOWN",
                        &format!("{e:#}"),
                    );
                }
            };
            if let Err(e) = o.write_tag(&name, &filled.manifest_bytes) {
                eprintln!("{name}: couldn't write Ollama's manifest: {e:#}");
            }
            filled.manifest_bytes
        }
    };
    (
        [
            (header::CONTENT_TYPE, MANIFEST_TYPE.to_string()),
            (
                header::HeaderName::from_static("docker-content-digest"),
                sha256_digest(&bytes),
            ),
        ],
        bytes,
    )
        .into_response()
}

async fn registry_blob(
    State(o): State<Arc<Ollama>>,
    axum::extract::Path((_, _, digest)): axum::extract::Path<(String, String, String)>,
    headers: HeaderMap,
) -> Response {
    if hex_of(&digest).is_err() {
        return registry_error(StatusCode::BAD_REQUEST, "DIGEST_INVALID", "bad digest");
    }
    let path = o.models.join("blobs").join(digest.replace(':', "-"));
    let Ok(file) = fs::File::open(&path) else {
        return registry_error(StatusCode::NOT_FOUND, "BLOB_UNKNOWN", "no such blob");
    };
    let size = match file.metadata() {
        Ok(m) => m.len(),
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .map(|r| parse_range(r, size));
    let (status, start, end) = match range {
        None => (StatusCode::OK, 0, size),
        Some(Some((a, b))) => (StatusCode::PARTIAL_CONTENT, a, b),
        Some(None) => {
            return (
                StatusCode::RANGE_NOT_SATISFIABLE,
                [(header::CONTENT_RANGE, format!("bytes */{size}"))],
            )
                .into_response();
        }
    };
    let file = Arc::new(file);
    let body = stream::unfold(start, move |pos| {
        let file = file.clone();
        async move {
            if pos >= end {
                return None;
            }
            let n = (end - pos).min(1 << 20) as usize;
            let read = tokio::task::spawn_blocking(move || {
                let mut buf = vec![0u8; n];
                file.read_exact_at(&mut buf, pos).map(|_| buf)
            })
            .await
            .map_err(std::io::Error::other)
            .and_then(|r| r);
            Some((read.map(Bytes::from), pos + n as u64))
        }
    });
    let mut h = HeaderMap::new();
    h.insert(header::CONTENT_LENGTH, HeaderValue::from(end - start));
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    if let Ok(v) = HeaderValue::from_str(&digest) {
        h.insert("docker-content-digest", v);
    }
    if status == StatusCode::PARTIAL_CONTENT
        && let Ok(v) = HeaderValue::from_str(&format!("bytes {start}-{}/{size}", end - 1))
    {
        h.insert(header::CONTENT_RANGE, v);
    }
    (status, h, Body::from_stream(body)).into_response()
}

/// An error in the registry API's JSON shape.
fn registry_error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        axum::Json(serde_json::json!({"errors": [{"code": code, "message": message}]})),
    )
        .into_response()
}

/// `bytes=a-b`, `bytes=a-` or `bytes=-n` as a half-open range within `size`.
fn parse_range(h: &str, size: u64) -> Option<(u64, u64)> {
    let (a, b) = h.strip_prefix("bytes=")?.split_once('-')?;
    let (start, end) = match (a.trim(), b.trim()) {
        ("", n) => (size.saturating_sub(n.parse().ok()?), size),
        (a, "") => (a.parse().ok()?, size),
        (a, b) => (
            a.parse().ok()?,
            b.parse::<u64>().ok()?.saturating_add(1).min(size),
        ),
    };
    (start < end).then_some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_like_ollama() {
        let n = Name::parse("llama3.2").unwrap();
        assert_eq!(
            (
                n.host.as_str(),
                n.namespace.as_str(),
                n.model.as_str(),
                n.tag.as_str()
            ),
            (DEFAULT_HOST, "library", "llama3.2", "latest")
        );
        assert_eq!(n.to_string(), "llama3.2:latest");
        let n = Name::parse("llama3.2:3b").unwrap();
        assert_eq!(n.tag, "3b");
        let n = Name::parse("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M").unwrap();
        assert_eq!(
            (
                n.host.as_str(),
                n.namespace.as_str(),
                n.model.as_str(),
                n.tag.as_str()
            ),
            ("hf.co", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Q4_K_M")
        );
        let n = Name::parse("localhost:5000/me/model").unwrap();
        assert_eq!(
            (n.host.as_str(), n.tag.as_str()),
            ("localhost:5000", "latest")
        );
        assert_eq!(
            n.meta_key(),
            "ollama/tags/localhost%3A5000/me/model/latest.json"
        );
        assert!(store::is_meta_key(&n.meta_key()));
        assert_eq!(
            Name::parse("llama3.2")
                .unwrap()
                .manifest_path(Path::new("/m")),
            Path::new("/m/manifests/registry.ollama.ai/library/llama3.2/latest")
        );
        for bad in ["", "a/b/c/d", "../x", "x:../y", "model@sha256:00", "a b"] {
            assert!(Name::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn layer_paths() {
        let l = |t: &str| Layer {
            media_type: format!("application/vnd.ollama.image.{t}"),
            digest: format!("sha256:{}", "ab".repeat(32)),
            size: 1,
        };
        assert!(l("model").store_path().ends_with(".gguf"));
        assert!(l("projector").store_path().ends_with(".gguf"));
        assert!(!l("template").store_path().ends_with(".gguf"));
        assert!(crate::safety::is_allowed(&l("template").store_path()));
        assert_eq!(
            l("model").file_name(),
            format!("sha256-{}", "ab".repeat(32))
        );
    }

    #[test]
    fn parses_challenges() {
        let (realm, q) = parse_challenge(
            r#"Bearer realm="https://auth.example/token",service="registry",scope="repository:library/x:pull""#,
        )
        .unwrap();
        assert_eq!(realm, "https://auth.example/token");
        assert_eq!(q[0], ("service".into(), "registry".into()));
    }
}
