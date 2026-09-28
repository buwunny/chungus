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

use crate::manifest::Manifest;
use crate::store::{self, Store};

pub const SERVICE: &str = "_chungus._tcp.local.";
pub const DEFAULT_PORT: u16 = 7447;
/// Chunk requests in flight at once during a fetch.
const CONCURRENCY: usize = 32;

// ---------- server ----------

/// HTTP routes for serving a store:
/// `GET /v1/manifests`, `GET /v1/manifests/{root}`, `GET /v1/chunks/{hash}`.
pub fn router(store: Arc<Store>) -> Router {
    Router::new()
        .route("/v1/manifests", get(list_manifests))
        .route("/v1/manifests/{root}", get(get_manifest))
        .route("/v1/chunks/{hash}", get(get_chunk))
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
    match tokio::task::spawn_blocking(move || store.get_manifest_bytes(&root)).await {
        Ok(Ok(bytes)) => ([(header::CONTENT_TYPE, "application/json")], bytes).into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
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

/// Serve `store` on an already-bound listener until the process exits.
pub async fn serve_on(listener: tokio::net::TcpListener, store: Arc<Store>) -> Result<()> {
    axum::serve(listener, router(store)).await?;
    Ok(())
}

/// Advertise this node on the LAN. Keep the returned daemon alive for as long as the
/// advert should last.
pub fn advertise(port: u16) -> Result<mdns_sd::ServiceDaemon> {
    let daemon = mdns_sd::ServiceDaemon::new().context("start mDNS")?;
    let host = hostname();
    let instance = format!("{host}-{port}");
    let info = mdns_sd::ServiceInfo::new(
        SERVICE,
        &instance,
        &format!("{host}.local."),
        "",
        port,
        &[("v", "1")][..],
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

/// Browse mDNS for `wait` and return the base URL of every peer found.
pub fn discover(wait: Duration) -> Result<Vec<String>> {
    let daemon = mdns_sd::ServiceDaemon::new().context("start mDNS")?;
    let rx = daemon.browse(SERVICE)?;
    let deadline = Instant::now() + wait;
    let mut found = BTreeMap::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(mdns_sd::ServiceEvent::ServiceResolved(svc)) => {
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
    /// Chunks a source sent that failed verification.
    pub rejected: usize,
    pub secs: f64,
}

enum Attempt {
    Stored(u64),
    Missing,
    Rejected,
}

/// Download manifest `root` and every chunk it needs that `store` lacks.
///
/// `peers` are tried first, with chunks spread across them; `origin`, if given, is the
/// last resort for each chunk. Chunks already in the store are skipped, so an interrupted
/// fetch resumes where it stopped.
pub async fn fetch(
    root: &str,
    store: Arc<Store>,
    peers: &[String],
    origin: Option<&str>,
) -> Result<(Manifest, FetchStats)> {
    if !store::is_hash(root) {
        bail!("{root:?} is not a manifest root hash");
    }
    let started = Instant::now();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(60))
        .build()?;
    let origin: Vec<String> = origin.map(str::to_string).into_iter().collect();
    let all: Vec<&String> = peers.iter().chain(&origin).collect();
    if all.is_empty() {
        bail!("no peers found and no origin given");
    }

    let manifest = match store.get_manifest(root) {
        Ok(m) => m,
        Err(_) => fetch_manifest(&client, root, &all).await?,
    };
    store.put_manifest(&manifest)?;

    // Unique chunks, in manifest order.
    let mut wanted: Vec<(String, usize)> = Vec::new();
    let mut seen = HashSet::new();
    for c in manifest.files.iter().flat_map(|f| &f.chunks) {
        if seen.insert(c.hash.clone()) {
            wanted.push((c.hash.clone(), c.len as usize));
        }
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

    let results = stream::iter(missing)
        .map(|(hash, len)| {
            let (client, store) = (client.clone(), store.clone());
            let (peers, origin) = (peers, &origin);
            async move {
                // Spread load: each chunk starts at a different peer, chosen by its hash.
                let start = if peers.is_empty() {
                    0
                } else {
                    usize::from_str_radix(&hash[..4], 16).unwrap() % peers.len()
                };
                let order = peers[start..]
                    .iter()
                    .chain(&peers[..start])
                    .chain(origin.iter());
                let mut rejected = 0;
                for source in order {
                    match try_chunk(&client, &store, source, &hash, len).await {
                        Attempt::Stored(bytes) => return Ok((source.clone(), bytes, rejected)),
                        Attempt::Rejected => rejected += 1,
                        Attempt::Missing => {}
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
        *stats.bytes_by_source.entry(source).or_default() += bytes;
        stats.rejected += rejected;
    }
    stats.secs = started.elapsed().as_secs_f64();
    Ok((manifest, stats))
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
        if let Ok(m) = serde_json::from_slice::<Manifest>(&bytes)
            && m.root == root
            && m.verify_root()
        {
            return Ok(m);
        }
    }
    bail!("no source has a valid manifest for {root}")
}

async fn try_chunk(
    client: &reqwest::Client,
    store: &Arc<Store>,
    source: &str,
    hash: &str,
    len: usize,
) -> Attempt {
    let resp = match client
        .get(format!("{source}/v1/chunks/{hash}"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        _ => return Attempt::Missing,
    };
    let blob = match resp.bytes().await {
        Ok(b) => b,
        Err(_) => return Attempt::Missing,
    };
    let (store, hash) = (store.clone(), hash.to_string());
    let verified = tokio::task::spawn_blocking(move || -> Result<u64> {
        let raw = store::decode(&blob, len)?;
        if blake3::hash(&raw).to_hex().as_str() != hash {
            bail!("hash mismatch");
        }
        store.put(&hash, &blob)?;
        Ok(blob.len() as u64)
    })
    .await;
    match verified {
        Ok(Ok(bytes)) => Attempt::Stored(bytes),
        _ => Attempt::Rejected,
    }
}
