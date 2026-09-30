//! Sharing chunk stores between machines on a LAN.
//!
//! A node runs `serve`, which exposes its store read-only over HTTP and advertises itself
//! with mDNS as `_chungus._tcp.local.`. Another node runs `fetch` with a manifest root:
//! it finds peers, downloads only the chunks it doesn't already have, verifies each one
//! against its BLAKE3 hash, and falls back to the next peer (and finally to an origin
//! server) when a peer is missing a chunk or sends bad data.
//!
//! Chunks travel in their stored, compressed form, so the LAN carries the compressed size.

use anyhow::{Context, Result, anyhow, bail};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures::{StreamExt, stream};
use std::collections::{BTreeMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::manifest::{ChunkRef, Manifest};
use crate::sign::{self, Signature};
use crate::store::{self, Store};
use ed25519_dalek::VerifyingKey;

pub const SERVICE: &str = "_chungus._tcp.local.";
pub const DEFAULT_PORT: u16 = 7447;
/// Chunk requests in flight at once during a fetch.
const CONCURRENCY: usize = 32;

// ---------- server ----------

/// HTTP routes for serving a store: `GET /v1/manifests`, `GET /v1/manifests/{root}`,
/// `GET /v1/signatures/{root}`, `GET /v1/chunks/{hash}` and `GET /v1/meta/{key}`.
pub fn router(store: Arc<Store>) -> Router {
    Router::new()
        .route("/v1/manifests", get(list_manifests))
        .route("/v1/manifests/{root}", get(get_manifest))
        .route("/v1/signatures/{root}", get(get_signatures))
        .route("/v1/chunks/{hash}", get(get_chunk))
        .route("/v1/meta/{*key}", get(get_meta))
        .with_state(store)
}

async fn list_manifests(State(store): State<Arc<Store>>) -> Response {
    match store.manifests() {
        Ok(roots) => axum::Json(roots).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn get_manifest(State(store): State<Arc<Store>>, Path(root): Path<String>) -> Response {
    let root = root.trim_end_matches(".json").to_string();
    if !store::is_hash(&root) {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match tokio::task::spawn_blocking(move || store.get_safe_manifest_bytes(&root)).await {
        Ok(Ok(bytes)) => ([(header::CONTENT_TYPE, "application/json")], bytes).into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn get_signatures(State(store): State<Arc<Store>>, Path(root): Path<String>) -> Response {
    if !store::is_hash(&root) {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match tokio::task::spawn_blocking(move || store.signatures(&root)).await {
        Ok(Ok(sigs)) => axum::Json(sigs).into_response(),
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn get_chunk(State(store): State<Arc<Store>>, Path(hash): Path<String>) -> Response {
    if !store::is_hash(&hash) {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match tokio::task::spawn_blocking(move || store.get(&hash)).await {
        Ok(Ok(blob)) => (
            [(header::CONTENT_TYPE, "application/octet-stream")],
            Bytes::from(blob),
        )
            .into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn get_meta(State(store): State<Arc<Store>>, Path(key): Path<String>) -> Response {
    if !store::is_meta_key(&key) {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match tokio::task::spawn_blocking(move || store.get_meta(&key)).await {
        Ok(Ok(bytes)) => ([(header::CONTENT_TYPE, "application/json")], bytes).into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Serve `store` on an already-bound listener until the process exits.
pub async fn serve_on(listener: tokio::net::TcpListener, store: Arc<Store>) -> Result<()> {
    axum::serve(listener, router(store)).await?;
    Ok(())
}

/// Hold each response until `limiter` allows its bytes, capping what a router uploads.
pub fn rate_limited(router: Router, limiter: Arc<crate::limits::RateLimiter>) -> Router {
    router.layer(axum::middleware::from_fn(
        move |req: axum::extract::Request, next: axum::middleware::Next| {
            let limiter = limiter.clone();
            async move {
                let resp = next.run(req).await;
                let bytes = axum::body::HttpBody::size_hint(resp.body()).lower();
                limiter.take(bytes).await;
                resp
            }
        },
    ))
}

/// Advertise this node on the LAN under instance `id`. Keep the returned daemon alive for
/// as long as the advert should last.
pub fn advertise(port: u16, id: &str) -> Result<mdns_sd::ServiceDaemon> {
    let daemon = mdns_sd::ServiceDaemon::new().context("start mDNS")?;
    let host = hostname();
    let instance = format!("{host}-{port}");
    let info = mdns_sd::ServiceInfo::new(
        SERVICE,
        &instance,
        &format!("{host}.local."),
        "",
        port,
        &[("v", "1"), ("id", id)][..],
    )?
    .enable_addr_auto();
    daemon.register(info).context("register mDNS service")?;
    Ok(daemon)
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "chungus".into())
        .replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "-")
}

// ---------- discovery ----------

/// A random id for this process, so a node can recognise (and skip) its own advert.
pub fn node_id() -> String {
    let seed = format!("{:?}{}", Instant::now(), std::process::id());
    blake3::hash(seed.as_bytes()).to_hex()[..16].to_string()
}

/// Browse mDNS for `wait` and return the base URL of every peer found, except the one
/// advertising `exclude_id`.
pub fn discover(wait: Duration, exclude_id: Option<&str>) -> Result<Vec<String>> {
    let daemon = mdns_sd::ServiceDaemon::new().context("start mDNS")?;
    let rx = daemon.browse(SERVICE)?;
    let deadline = Instant::now() + wait;
    let mut found = BTreeMap::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(mdns_sd::ServiceEvent::ServiceResolved(svc))
                if exclude_id.is_none()
                    || svc.txt_properties.get_property_val_str("id") != exclude_id =>
            {
                // Prefer IPv4: link-local IPv6 needs a scope id that URLs handle badly.
                let mut addrs: Vec<IpAddr> = svc.addresses.iter().map(|a| a.to_ip_addr()).collect();
                addrs.sort_by_key(|a| !a.is_ipv4());
                if let Some(ip) = addrs.first() {
                    found.insert(
                        svc.fullname.clone(),
                        url_for(SocketAddr::new(*ip, svc.port)),
                    );
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let _ = daemon.shutdown();
    Ok(found.into_values().collect())
}

pub fn url_for(addr: SocketAddr) -> String {
    format!("http://{addr}")
}

// ---------- client ----------

#[derive(Default, Debug)]
pub struct FetchStats {
    pub chunks: usize,
    pub already_local: usize,
    /// Compressed bytes received from each source that served at least one chunk.
    pub bytes_by_source: BTreeMap<String, u64>,
    /// Chunks received from each of those sources.
    pub chunks_by_source: BTreeMap<String, u64>,
    /// Chunks a source sent that failed verification.
    pub rejected: usize,
    pub secs: f64,
}

/// Download manifest `root` and every chunk it needs that `store` lacks.
///
/// `peers` are tried first, with chunks spread across them; `origin`, if given, is the
/// last resort for each chunk. Chunks already in the store are skipped, so an interrupted
/// fetch resumes where it stopped.
///
/// Signatures on the manifest are collected from every source and kept if valid. When
/// `trust` is non-empty, the fetch stops before downloading any chunk unless one of those
/// keys has signed the manifest.
pub async fn fetch(
    root: &str,
    store: Arc<Store>,
    peers: &[String],
    origin: Option<&str>,
    trust: &[VerifyingKey],
) -> Result<(Manifest, FetchStats)> {
    let started = Instant::now();
    let client = client()?;
    let origin: Vec<String> = origin.map(str::to_string).into_iter().collect();
    let manifest = prepare(&client, root, &store, peers, &origin, trust).await?;
    let mut stats = fetch_chunks(
        &client,
        &store,
        manifest.files.iter().flat_map(|f| &f.chunks),
        peers,
        &origin,
    )
    .await?;
    stats.secs = started.elapsed().as_secs_f64();
    Ok((manifest, stats))
}

/// Get the manifest for `root` (from the store, or else from `peers` and `origin`) and
/// every source's signatures for it, and check `trust` as [`fetch`] does.
pub async fn prepare(
    client: &reqwest::Client,
    root: &str,
    store: &Arc<Store>,
    peers: &[String],
    origin: &[String],
    trust: &[VerifyingKey],
) -> Result<Manifest> {
    if !store::is_hash(root) {
        bail!("{root:?} is not a manifest root hash");
    }
    let all: Vec<&String> = peers.iter().chain(origin).collect();
    if all.is_empty() {
        bail!("no peers found and no origin given");
    }
    let manifest = match store.get_manifest(root) {
        Ok(m) => m,
        Err(_) => fetch_manifest(client, root, &all).await?,
    };
    store.put_manifest(&manifest)?;
    let sigs = fetch_signatures(client, root, &all).await;
    store.add_signatures(root, &sigs)?;
    if !trust.is_empty() && !sign::trusted_by(&store.signatures(root)?, root, trust) {
        bail!("no trusted key has signed {root}; refusing to download it");
    }
    Ok(manifest)
}

/// LAN peers (spread by hash) and then an origin, as a source for [`crate::lazy`].
pub struct HttpSource {
    pub client: reqwest::Client,
    pub peers: Vec<String>,
    pub origin: Vec<String>,
}

impl crate::lazy::ChunkSource for HttpSource {
    fn fetch<'a>(
        &'a self,
        store: &'a Arc<Store>,
        hash: &'a str,
        len: usize,
    ) -> futures::future::BoxFuture<'a, bool> {
        Box::pin(async move {
            let mut sources = rotated(&self.peers, hash);
            sources.extend(self.origin.iter().cloned());
            fetch_one(&self.client, store, hash, len, &sources).await
        })
    }
}

pub fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(60))
        .build()?)
}

/// Download every chunk in `chunks` that `store` lacks, trying `peers` (spread by hash)
/// and then `origin`. Each chunk is verified against its hash before it is stored.
pub async fn fetch_chunks<'a>(
    client: &reqwest::Client,
    store: &Arc<Store>,
    chunks: impl IntoIterator<Item = &'a ChunkRef>,
    peers: &[String],
    origin: &[String],
) -> Result<FetchStats> {
    fetch_chunks_with(store, chunks, peers, origin, |source, hash| {
        let client = client.clone();
        async move {
            let resp = client
                .get(format!("{source}/v1/chunks/{hash}"))
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            resp.bytes().await.ok()
        }
    })
    .await
}

/// Download every chunk in `chunks` that `store` lacks with `get(source, hash)`, trying
/// `peers` (spread by hash) and then `fallback`. Each chunk is verified against its hash
/// before it is stored, so a source can't make us keep bad data.
pub(crate) async fn fetch_chunks_with<'a, S, F, Fut>(
    store: &Arc<Store>,
    chunks: impl IntoIterator<Item = &'a ChunkRef>,
    peers: &[S],
    fallback: &[S],
    get: F,
) -> Result<FetchStats>
where
    S: Clone + std::fmt::Display,
    F: Fn(S, String) -> Fut,
    Fut: std::future::Future<Output = Option<Bytes>>,
{
    let sources = |hash: &str| -> Vec<S> {
        // Spread load: each chunk starts at a different peer, chosen by its hash.
        let mut order = rotated(peers, hash);
        order.extend(fallback.iter().cloned());
        order
    };
    fetch_chunks_from(store, chunks, sources, get).await
}

/// `items`, starting at a position picked by `hash`, so requests spread evenly.
pub(crate) fn rotated<S: Clone>(items: &[S], hash: &str) -> Vec<S> {
    if items.is_empty() {
        return Vec::new();
    }
    let start = usize::from_str_radix(&hash[..4], 16).unwrap_or(0) % items.len();
    items[start..]
        .iter()
        .chain(&items[..start])
        .cloned()
        .collect()
}

/// Download every chunk in `chunks` that `store` lacks, trying `sources(hash)` in order
/// with `get(source, hash)`. Each chunk is verified before it is stored.
pub(crate) async fn fetch_chunks_from<'a, S, Src, F, Fut>(
    store: &Arc<Store>,
    chunks: impl IntoIterator<Item = &'a ChunkRef>,
    sources: Src,
    get: F,
) -> Result<FetchStats>
where
    S: Clone + std::fmt::Display,
    Src: Fn(&str) -> Vec<S>,
    F: Fn(S, String) -> Fut,
    Fut: std::future::Future<Output = Option<Bytes>>,
{
    // Unique chunks, in order.
    let mut wanted: Vec<(String, usize)> = Vec::new();
    let mut seen = HashSet::new();
    for c in chunks {
        if seen.insert(c.hash.clone()) {
            wanted.push((c.hash.clone(), c.len as usize));
        }
    }
    if let Some((h, _)) = wanted.iter().find(|(h, _)| store.is_blocked(h)) {
        bail!("this model contains chunk {h}, which is on the blocklist");
    }
    let mut stats = FetchStats {
        chunks: wanted.len(),
        ..Default::default()
    };
    let missing: Vec<_> = wanted
        .into_iter()
        .filter(|(h, _)| !store.contains(h))
        .collect();
    stats.already_local = stats.chunks - missing.len();

    let (get, sources) = (&get, &sources);
    let results = stream::iter(missing)
        .map(|(hash, len)| {
            let store = store.clone();
            async move {
                let mut rejected = 0;
                for source in sources(&hash) {
                    let Some(blob) = get(source.clone(), hash.clone()).await else {
                        continue;
                    };
                    match verify_and_store(&store, &hash, len, blob).await {
                        Some(bytes) => return Ok((source.to_string(), bytes, rejected)),
                        None => rejected += 1,
                    }
                }
                Err(anyhow!("chunk {hash} is not available from any source"))
            }
        })
        .buffer_unordered(CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

    for r in results {
        let (source, bytes, rejected) = r?;
        *stats.chunks_by_source.entry(source.clone()).or_default() += 1;
        *stats.bytes_by_source.entry(source).or_default() += bytes;
        stats.rejected += rejected;
    }
    Ok(stats)
}

/// Store `blob` as chunk `hash` if it decodes to `len` bytes with that hash. Returns the
/// blob's size, or None if it was rejected.
pub(crate) async fn verify_and_store(
    store: &Arc<Store>,
    hash: &str,
    len: usize,
    blob: Bytes,
) -> Option<u64> {
    let (store, hash) = (store.clone(), hash.to_string());
    tokio::task::spawn_blocking(move || -> Result<u64> {
        let raw = store::decode(&blob, len)?;
        if blake3::hash(&raw).to_hex().as_str() != hash {
            bail!("hash mismatch");
        }
        store.put(&hash, &blob)?;
        Ok(blob.len() as u64)
    })
    .await
    .ok()?
    .ok()
}

/// Every signature any source has for `root`. Unverified; the store keeps only valid ones.
async fn fetch_signatures(
    client: &reqwest::Client,
    root: &str,
    sources: &[&String],
) -> Vec<Signature> {
    let mut out = Vec::new();
    for source in sources {
        let Ok(resp) = client
            .get(format!("{source}/v1/signatures/{root}"))
            .send()
            .await
        else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        if let Ok(bytes) = resp.bytes().await
            && let Ok(sigs) = serde_json::from_slice::<Vec<Signature>>(&bytes)
        {
            out.extend(sigs);
        }
    }
    out
}

async fn fetch_manifest(
    client: &reqwest::Client,
    root: &str,
    sources: &[&String],
) -> Result<Manifest> {
    for source in sources {
        let Ok(resp) = client
            .get(format!("{source}/v1/manifests/{root}"))
            .send()
            .await
        else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(bytes) = resp.bytes().await else {
            continue;
        };
        // A manifest is only trusted if it hashes to the root we asked for.
        if let Ok(m) = crate::manifest::parse(&bytes)
            && m.root == root
            && m.verify_root()
        {
            return Ok(m);
        }
    }
    bail!("no source has a valid manifest for {root}")
}

/// Fetch one chunk from the first of `sources` that has a valid copy. Returns false if
/// none did.
pub async fn fetch_one(
    client: &reqwest::Client,
    store: &Arc<Store>,
    hash: &str,
    len: usize,
    sources: &[String],
) -> bool {
    for source in sources {
        if try_chunk(client, store, source, hash, len).await {
            return true;
        }
    }
    false
}

async fn try_chunk(
    client: &reqwest::Client,
    store: &Arc<Store>,
    source: &str,
    hash: &str,
    len: usize,
) -> bool {
    let resp = match client
        .get(format!("{source}/v1/chunks/{hash}"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        _ => return false,
    };
    let Ok(blob) = resp.bytes().await else {
        return false;
    };
    verify_and_store(store, hash, len, blob).await.is_some()
}
