//! A local cache that speaks the part of the Hugging Face Hub API that `huggingface_hub`,
//! `transformers` and friends use, so existing tools work unchanged with
//! `HF_ENDPOINT=http://localhost:8080`.
//!
//! For each file it is asked for, the hub serves it from the local store if it has it,
//! otherwise pulls its chunks from LAN peers, otherwise downloads it from huggingface.co
//! and packs it. Every hub also serves the peer API from [`crate::net`], so hubs on the
//! same network fill each other's caches.
//!
//! Trust rules:
//! - The user's `Authorization` header is only ever forwarded to the upstream, never to
//!   peers.
//! - Files of gated models are only served to requests whose token the upstream accepts
//!   for that file. With no reachable upstream, gated files are refused.
//! - While the upstream is reachable, a file assembled from peer chunks must match the
//!   upstream's SHA-256 (LFS) or git blob hash, or it is discarded and downloaded instead.
//!   Offline, LAN peers are trusted for metadata, as with `serve`.

use anyhow::{Context, Result, anyhow, bail};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::Digest as _;
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

use crate::manifest::{ChunkRef, FileEntry};
use crate::net;
use crate::store::{self, Store};

pub const DEFAULT_PORT: u16 = 8080;
pub const DEFAULT_UPSTREAM: &str = "https://huggingface.co";

/// A cached file of a Hub repo: the upstream's ETag (so clients see the same cache keys
/// they would get from huggingface.co) and the chunks that rebuild it.
#[derive(Serialize, Deserialize, Clone)]
pub struct HubFile {
    pub etag: String,
    pub entry: FileEntry,
    /// True once the content was checked against the upstream's content-hash ETag.
    #[serde(default)]
    pub verified: bool,
}

type Files = BTreeMap<String, HubFile>;

/// An error that becomes an HTTP response.
pub struct HubError(StatusCode, String);

impl IntoResponse for HubError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

impl From<anyhow::Error> for HubError {
    fn from(e: anyhow::Error) -> Self {
        HubError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

type HubResult<T> = std::result::Result<T, HubError>;

fn err<T>(status: StatusCode, msg: impl Into<String>) -> HubResult<T> {
    Err(HubError(status, msg.into()))
}

pub struct Hub {
    store: Arc<Store>,
    upstream: Option<String>,
    peers: RwLock<Vec<String>>,
    /// Follows redirects; used for API calls and downloads.
    client: reqwest::Client,
    /// Doesn't follow redirects, so HEAD responses keep the Hub's X-Linked-* headers.
    head_client: reqwest::Client,
    record_lock: Mutex<()>,
    /// blake3(token, repo) pairs the upstream has accepted for a gated repo.
    allowed: Mutex<HashSet<String>>,
    /// Files currently being downloaded into the store.
    downloading: Mutex<HashSet<String>>,
    /// Files whose peer copy failed the upstream hash check; fetched upstream from now on.
    distrust_peers: Mutex<HashSet<String>>,
}

impl Hub {
    pub fn new(store: Arc<Store>, upstream: Option<String>, peers: Vec<String>) -> Result<Self> {
        let upstream = upstream.map(|u| u.trim_end_matches('/').to_string());
        Ok(Hub {
            store,
            upstream,
            peers: RwLock::new(peers),
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .read_timeout(Duration::from_secs(60))
                .build()?,
            head_client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            record_lock: Mutex::new(()),
            allowed: Default::default(),
            downloading: Default::default(),
            distrust_peers: Default::default(),
        })
    }

    pub fn set_peers(&self, peers: Vec<String>) {
        *self.peers.write().unwrap() = peers;
    }

    fn peers(&self) -> Vec<String> {
        self.peers.read().unwrap().clone()
    }

    // ----- metadata records -----

    fn put_meta_json(&self, key: &str, v: &impl Serialize) -> Result<()> {
        self.store.put_meta(key, &serde_json::to_vec(v)?)
    }

    /// A metadata record from the local store, or else from the first peer that has it
    /// (which is then cached locally).
    async fn meta(&self, key: &str) -> Option<Vec<u8>> {
        if let Ok(bytes) = self.store.get_meta(key) {
            return Some(bytes);
        }
        let bytes = self.peer_meta(key).await?;
        let _ = self.store.put_meta(key, &bytes);
        Some(bytes)
    }

    async fn peer_meta(&self, key: &str) -> Option<Vec<u8>> {
        for peer in self.peers() {
            let Ok(resp) = self
                .head_client
                .get(format!("{peer}/v1/meta/{key}"))
                .send()
                .await
            else {
                continue;
            };
            if resp.status().is_success()
                && let Ok(bytes) = resp.bytes().await
            {
                return Some(bytes.to_vec());
            }
        }
        None
    }

    fn local_files(&self, rk: &str, commit: &str) -> Files {
        self.store
            .get_meta(&format!("hub/{rk}/{commit}/files.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    fn save_file(&self, rk: &str, commit: &str, path: &str, file: &HubFile) -> Result<()> {
        let _guard = self.record_lock.lock().unwrap();
        let mut files = self.local_files(rk, commit);
        files.insert(path.to_string(), file.clone());
        self.put_meta_json(&format!("hub/{rk}/{commit}/files.json"), &files)
    }

    // ----- upstream helpers -----

    /// Map an upstream response status to what the client should see: auth and not-found
    /// errors pass through unchanged, anything else is a bad gateway.
    async fn upstream_error(resp: reqwest::Response) -> HubError {
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let body = resp.text().await.unwrap_or_default();
        if matches!(status.as_u16(), 401 | 403 | 404) {
            HubError(status, body)
        } else {
            HubError(
                StatusCode::BAD_GATEWAY,
                format!("upstream returned {status}: {body}"),
            )
        }
    }

    // ----- revisions -----

    /// Resolve `rev` of `repo` to a commit, and return the repo's model info at that commit.
    /// Asks the upstream first; falls back to the local cache, then to peers.
    async fn revision(
        &self,
        repo: &str,
        rev: &str,
        auth: &HeaderMap,
    ) -> HubResult<(String, Value)> {
        let rk = repo_key(repo);
        if let Some(up) = &self.upstream {
            let url = format!("{up}/api/models/{repo}/revision/{}", encode(rev));
            match self.client.get(url).headers(auth.clone()).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes().await.map_err(|e| anyhow!(e))?;
                    let info: Value = serde_json::from_slice(&bytes).map_err(|e| anyhow!(e))?;
                    let commit = info["sha"].as_str().unwrap_or_default().to_string();
                    if !is_commit(&commit) {
                        return err(StatusCode::BAD_GATEWAY, "upstream returned no commit sha");
                    }
                    self.put_meta_json(&format!("hub/{rk}/{commit}/info.json"), &info)?;
                    if rev != commit {
                        self.store.put_meta(
                            &format!("hub/{rk}/refs/{}", encode(rev)),
                            commit.as_bytes(),
                        )?;
                    }
                    return Ok((commit, info));
                }
                Ok(resp) if matches!(resp.status().as_u16(), 401 | 403 | 404) => {
                    return Err(Self::upstream_error(resp).await);
                }
                // Unreachable or failing upstream: fall back to what we and our peers know.
                _ => {}
            }
        }
        let commit = if is_commit(rev) {
            rev.to_string()
        } else {
            let key = format!("hub/{rk}/refs/{}", encode(rev));
            match self
                .meta(&key)
                .await
                .and_then(|b| String::from_utf8(b).ok())
            {
                Some(c) if is_commit(c.trim()) => c.trim().to_string(),
                _ => return err(StatusCode::NOT_FOUND, format!("{repo}@{rev} is not cached")),
            }
        };
        match self.meta(&format!("hub/{rk}/{commit}/info.json")).await {
            Some(bytes) => Ok((
                commit,
                serde_json::from_slice(&bytes).map_err(|e| anyhow!(e))?,
            )),
            None => err(
                StatusCode::NOT_FOUND,
                format!("{repo}@{commit} is not cached"),
            ),
        }
    }

    // ----- trees -----

    async fn tree(
        &self,
        repo: &str,
        rev: &str,
        subpath: &str,
        recursive: bool,
        auth: &HeaderMap,
    ) -> HubResult<Vec<Value>> {
        let (commit, _) = self.revision(repo, rev, auth).await?;
        let rk = repo_key(repo);
        let full_key = format!("hub/{rk}/{commit}/tree.json");
        if let Some(up) = &self.upstream {
            let mut url = format!("{up}/api/models/{repo}/tree/{commit}");
            if !subpath.is_empty() {
                url += &format!("/{}", encode_path(subpath));
            }
            url += if recursive {
                "?recursive=true"
            } else {
                "?recursive=false"
            };
            match self.list_pages(&url, auth).await {
                Ok(mut entries) => {
                    // Without Xet hashes the client downloads through us instead of going
                    // straight to Hugging Face's Xet storage.
                    for e in &mut entries {
                        if let Some(obj) = e.as_object_mut() {
                            obj.remove("xetHash");
                        }
                    }
                    if subpath.is_empty() && recursive {
                        self.put_meta_json(&full_key, &entries)?;
                    }
                    return Ok(entries);
                }
                Err(Some(e)) => return Err(e),
                Err(None) => {}
            }
        }
        // Offline: answer from the cached full listing.
        let Some(bytes) = self.meta(&full_key).await else {
            return err(
                StatusCode::NOT_FOUND,
                format!("tree of {repo}@{commit} is not cached"),
            );
        };
        let all: Vec<Value> = serde_json::from_slice(&bytes).map_err(|e| anyhow!(e))?;
        let prefix = if subpath.is_empty() {
            String::new()
        } else {
            format!("{subpath}/")
        };
        Ok(all
            .into_iter()
            .filter(|e| {
                let p = e["path"].as_str().unwrap_or_default();
                p.strip_prefix(&prefix)
                    .is_some_and(|rest| !rest.is_empty() && (recursive || !rest.contains('/')))
            })
            .collect())
    }

    /// GET a paginated upstream listing. `Err(None)` means the upstream was unreachable.
    async fn list_pages(
        &self,
        url: &str,
        auth: &HeaderMap,
    ) -> std::result::Result<Vec<Value>, Option<HubError>> {
        let mut out = Vec::new();
        let mut next = Some(url.to_string());
        while let Some(url) = next.take() {
            let resp = self
                .client
                .get(&url)
                .headers(auth.clone())
                .send()
                .await
                .map_err(|_| None)?;
            if !resp.status().is_success() {
                return Err(Some(Self::upstream_error(resp).await));
            }
            next = resp
                .headers()
                .get(header::LINK)
                .and_then(|v| v.to_str().ok())
                .and_then(next_link);
            let bytes = resp.bytes().await.map_err(|_| None)?;
            let page: Vec<Value> = serde_json::from_slice(&bytes).map_err(|_| None)?;
            out.extend(page);
        }
        Ok(out)
    }

    // ----- files -----

    /// The record for `path` if this hub has it locally or a peer does. A peer's record
    /// is never marked verified.
    async fn find_file(&self, rk: &str, commit: &str, path: &str) -> Option<HubFile> {
        if let Some(f) = self.local_files(rk, commit).remove(path) {
            return Some(f);
        }
        let key = format!("{rk}/{commit}/{path}");
        if self.distrust_peers.lock().unwrap().contains(&key) {
            return None;
        }
        let bytes = self
            .peer_meta(&format!("hub/{rk}/{commit}/files.json"))
            .await?;
        let mut f = serde_json::from_slice::<Files>(&bytes).ok()?.remove(path)?;
        f.verified = false;
        (f.entry.path == path).then_some(f)
    }

    fn is_complete(&self, f: &HubFile) -> bool {
        f.entry.chunks.iter().all(|c| self.store.contains(&c.hash))
    }

    /// HEAD the file upstream. `Ok(None)` means the upstream is unreachable or there is
    /// none; auth and not-found errors are returned as errors.
    async fn upstream_head(
        &self,
        repo: &str,
        commit: &str,
        path: &str,
        auth: &HeaderMap,
    ) -> HubResult<Option<UpstreamHead>> {
        let Some(up) = self.upstream.as_deref() else {
            return Ok(None);
        };
        let url = format!("{up}/{repo}/resolve/{commit}/{}", encode_path(path));
        let resp = match self
            .head_client
            .head(&url)
            .headers(auth.clone())
            .send()
            .await
        {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };
        let status = resp.status().as_u16();
        if !(200..400).contains(&status) {
            if matches!(status, 401 | 403 | 404) {
                return Err(Self::upstream_error(resp).await);
            }
            return Ok(None);
        }
        let h = resp.headers();
        let get = |name: &str| h.get(name).and_then(|v| v.to_str().ok());
        let etag = get("x-linked-etag")
            .or(get("etag"))
            .map(normalize_etag)
            .unwrap_or_default();
        let size = get("x-linked-size")
            .or(if status < 300 {
                get("content-length")
            } else {
                None
            })
            .and_then(|s| s.parse().ok());
        Ok(Some(UpstreamHead { url, etag, size }))
    }

    /// With no upstream to ask, a gated file is only served to a token the upstream has
    /// already accepted for this repo.
    fn check_gate_offline(&self, repo: &str, auth: &HeaderMap) -> HubResult<()> {
        if !auth.contains_key(header::AUTHORIZATION) {
            return err(
                StatusCode::UNAUTHORIZED,
                format!("{repo} is gated: send your Hugging Face token"),
            );
        }
        if self.allowed.lock().unwrap().contains(&gate_key(repo, auth)) {
            return Ok(());
        }
        err(
            StatusCode::FORBIDDEN,
            format!("{repo} is gated and huggingface.co can't be reached to check your access"),
        )
    }

    /// Hash a complete local file against the upstream's ETag and record the result.
    async fn verify_now(
        &self,
        rk: &str,
        commit: &str,
        f: &mut HubFile,
        etag: &str,
    ) -> Result<bool> {
        let (store, entry, e) = (self.store.clone(), f.entry.clone(), etag.to_string());
        let ok = tokio::task::spawn_blocking(move || -> Result<bool> {
            let mut h = ContentHasher::new(&e, entry.size).context("not a content hash")?;
            for c in &entry.chunks {
                h.update(&store::decode(&store.get(&c.hash)?, c.len as usize)?);
            }
            Ok(h.finish() == e)
        })
        .await??;
        if ok {
            f.etag = etag.to_string();
            f.verified = true;
            self.save_file(rk, commit, &f.entry.path, f)?;
        }
        Ok(ok)
    }

    /// Get one chunk's raw bytes, fetching it from peers if it isn't local.
    async fn chunk_bytes(self: &Arc<Self>, c: &ChunkRef) -> Result<Bytes> {
        if !self.store.contains(&c.hash) {
            let peers = self.peers();
            if !net::fetch_one(&self.client, &self.store, &c.hash, c.len as usize, &peers).await {
                bail!("chunk {} is not available from any peer", c.hash);
            }
        }
        let (store, c) = (self.store.clone(), c.clone());
        tokio::task::spawn_blocking(move || -> Result<Bytes> {
            let raw = store::decode(&store.get(&c.hash)?, c.len as usize)?;
            if blake3::hash(&raw).to_hex().as_str() != c.hash {
                bail!("chunk {} failed verification", c.hash);
            }
            Ok(Bytes::from(raw))
        })
        .await?
    }

    fn claim_download(&self, key: &str) -> bool {
        self.downloading.lock().unwrap().insert(key.to_string())
    }

    fn release_download(&self, key: &str) {
        self.downloading.lock().unwrap().remove(key);
    }
}

/// Everything known about a requested file before its body is sent.
struct FileRequest {
    repo: String,
    rk: String,
    commit: String,
    path: String,
    auth: HeaderMap,
}

struct UpstreamHead {
    url: String,
    etag: String,
    size: Option<u64>,
}

// ----- HTTP -----

/// The Hub API routes plus the peer API, on one server.
pub fn router(hub: Arc<Hub>) -> Router {
    net::router(hub.store.clone()).merge(Router::new().fallback(handle).with_state(hub))
}

async fn handle(
    State(hub): State<Arc<Hub>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    if method != Method::GET && method != Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let mut auth = HeaderMap::new();
    if let Some(v) = headers.get(header::AUTHORIZATION) {
        auth.insert(header::AUTHORIZATION, v.clone());
    }
    let result = match parse_route(uri.path(), uri.query().unwrap_or("")) {
        Some(Route::Info { repo, rev }) => hub
            .revision(&repo, &rev, &auth)
            .await
            .map(|(_, info)| axum::Json(info).into_response()),
        Some(Route::Tree {
            repo,
            rev,
            path,
            recursive,
        }) => hub
            .tree(&repo, &rev, &path, recursive, &auth)
            .await
            .map(|t| axum::Json(t).into_response()),
        Some(Route::Resolve { repo, rev, path }) => {
            serve_file(
                &hub,
                &repo,
                &rev,
                &path,
                &auth,
                &headers,
                method == Method::HEAD,
            )
            .await
        }
        None => Err(HubError(StatusCode::NOT_FOUND, "unknown route".into())),
    };
    result.unwrap_or_else(|e| e.into_response())
}

async fn serve_file(
    hub: &Arc<Hub>,
    repo: &str,
    rev: &str,
    path: &str,
    auth: &HeaderMap,
    req: &HeaderMap,
    head_only: bool,
) -> HubResult<Response> {
    let (commit, info) = hub.revision(repo, rev, auth).await?;
    let gated = !matches!(
        info.get("gated"),
        None | Some(Value::Null) | Some(Value::Bool(false))
    );
    let fr = FileRequest {
        repo: repo.to_string(),
        rk: repo_key(repo),
        commit,
        path: path.to_string(),
        auth: auth.clone(),
    };

    // The upstream's view of the file. For gated repos this is also the access check,
    // made with the requester's own token.
    let head = hub.upstream_head(repo, &fr.commit, path, auth).await?;
    match (&head, gated) {
        (Some(_), true) => {
            hub.allowed.lock().unwrap().insert(gate_key(repo, auth));
        }
        (None, true) => hub.check_gate_offline(repo, auth)?,
        _ => {}
    }

    // Files that can run code when loaded never enter the store or come from peers.
    if !crate::safety::is_allowed(path) {
        return passthrough(hub, &fr, head, req, head_only).await;
    }

    let found = hub.find_file(&fr.rk, &fr.commit, path).await;
    // A record is usable if it matches what the upstream says the file is now.
    let found = match (found, &head) {
        (Some(f), Some(h))
            if h.size.is_none_or(|s| s == f.entry.size)
                && (f.etag == h.etag || !is_content_hash(&h.etag)) =>
        {
            Some(f)
        }
        (Some(f), None) => Some(f),
        _ => None,
    };
    let (etag, size) = match (&found, &head) {
        (Some(f), _) => (f.etag.clone(), f.entry.size),
        (None, Some(h)) if h.size.is_some() => (h.etag.clone(), h.size.unwrap()),
        (None, Some(h)) => (h.etag.clone(), 0),
        (None, None) => {
            return err(
                StatusCode::NOT_FOUND,
                format!("{repo}@{}/{path} is not cached", fr.commit),
            );
        }
    };

    let range = match req.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        None => None,
        Some(r) => match parse_range(r, size) {
            Some(r) => Some(r),
            None => {
                return Ok((
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{size}"))],
                )
                    .into_response());
            }
        },
    };
    let mut headers = file_headers(&fr.commit, &etag, size, range);
    if found.is_none() && head.as_ref().is_some_and(|h| h.size.is_none()) {
        // Size unknown until the upstream sends it: let the body define the length.
        headers.1.remove(header::CONTENT_LENGTH);
        headers.1.remove("x-linked-size");
    }
    if head_only {
        return Ok((headers.0, headers.1, Body::empty()).into_response());
    }

    // Must the bytes be checked against the upstream's hash before we vouch for them?
    let check = head
        .as_ref()
        .filter(|h| is_content_hash(&h.etag))
        .map(|h| h.etag.clone());

    if let Some(mut f) = found {
        if range.is_some() || hub.is_complete(&f) {
            // Ranges can't be checked piecewise, so pull the whole file first.
            if !hub.is_complete(&f) {
                net::fetch_chunks(&hub.client, &hub.store, &f.entry.chunks, &hub.peers(), &[])
                    .await?;
            }
            if let Some(etag) = &check
                && !f.verified
                && !hub.verify_now(&fr.rk, &fr.commit, &mut f, etag).await?
            {
                return err(
                    StatusCode::BAD_GATEWAY,
                    format!("cached copy of {path} doesn't match huggingface.co"),
                );
            }
            let (start, end) = range.unwrap_or((0, size));
            let body = Body::from_stream(chunk_stream(hub.clone(), f.entry.chunks, start, end));
            return Ok((headers.0, headers.1, body).into_response());
        }
        // Stream from peers while fetching, checking the hash before the last byte goes out.
        let body = Body::from_stream(verified_stream(hub.clone(), fr, f, check));
        return Ok((headers.0, headers.1, body).into_response());
    }

    let head = head.expect("no record and no upstream was handled above");
    if let Some((start, end)) = range {
        // Not cached yet: pass the range straight through.
        return proxy_range(hub, &fr, &head, start, end, headers).await;
    }
    tee_download(hub.clone(), fr, head, headers).await
}

/// Status and headers for a file response, matching what huggingface.co sends.
fn file_headers(
    commit: &str,
    etag: &str,
    size: u64,
    range: Option<(u64, u64)>,
) -> (StatusCode, HeaderMap) {
    let mut h = HeaderMap::new();
    let (start, end) = range.unwrap_or((0, size));
    let status = if range.is_some() {
        h.insert(
            header::CONTENT_RANGE,
            hv(&format!("bytes {start}-{}/{size}", end.saturating_sub(1))),
        );
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    let quoted = format!("\"{etag}\"");
    h.insert(header::CONTENT_LENGTH, hv(&(end - start).to_string()));
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    h.insert(header::ETAG, hv(&quoted));
    h.insert("x-linked-etag", hv(&quoted));
    h.insert("x-linked-size", hv(&size.to_string()));
    h.insert("x-repo-commit", hv(commit));
    (status, h)
}

/// Stream bytes `start..end` of a complete local file.
fn chunk_stream(
    hub: Arc<Hub>,
    chunks: Vec<ChunkRef>,
    start: u64,
    end: u64,
) -> impl Stream<Item = std::io::Result<Bytes>> {
    let mut offset = 0u64;
    let mut parts = Vec::new();
    for c in chunks {
        let (lo, hi) = (offset, offset + c.len as u64);
        offset = hi;
        if hi > start && lo < end {
            let skip = start.saturating_sub(lo) as usize;
            let take = (end.min(hi) - lo) as usize;
            parts.push((c, skip, take));
        }
    }
    stream::iter(parts)
        .map(move |(c, skip, take)| {
            let hub = hub.clone();
            async move {
                hub.chunk_bytes(&c)
                    .await
                    .map(|b| b.slice(skip..take))
                    .map_err(std::io::Error::other)
            }
        })
        .buffered(8)
}

/// Stream a whole file whose chunks may still be on peers. Chunks are fetched ahead of
/// the reader. When `check` is set, the last chunk is held back until the whole file's
/// hash matches it, so a client never receives a complete file that doesn't match
/// huggingface.co. On success the record is saved locally.
fn verified_stream(
    hub: Arc<Hub>,
    fr: FileRequest,
    file: HubFile,
    check: Option<String>,
) -> impl Stream<Item = std::io::Result<Bytes>> {
    let hasher = check
        .as_deref()
        .and_then(|e| ContentHasher::new(e, file.entry.size));
    let h2 = hub.clone();
    let chunks = stream::iter(file.entry.chunks.clone())
        .map(move |c| {
            let hub = h2.clone();
            async move { hub.chunk_bytes(&c).await }
        })
        .buffered(16)
        .boxed();
    struct St {
        chunks: futures::stream::BoxStream<'static, Result<Bytes>>,
        hasher: Option<ContentHasher>,
        pending: Option<Bytes>,
        done: bool,
    }
    let st = St {
        chunks,
        hasher,
        pending: None,
        done: false,
    };
    stream::unfold(
        (st, hub, fr, file, check),
        |(mut st, hub, fr, mut file, check)| async move {
            if st.done {
                return None;
            }
            let item = match st.chunks.next().await {
                Some(Ok(bytes)) => {
                    if let Some(h) = &mut st.hasher {
                        h.update(&bytes);
                    }
                    Ok(st.pending.replace(bytes).unwrap_or_default())
                }
                Some(Err(e)) => {
                    st.done = true;
                    Err(std::io::Error::other(e))
                }
                None => {
                    st.done = true;
                    let ok = match (st.hasher.take(), &check) {
                        (Some(h), Some(expected)) => h.finish() == *expected,
                        _ => true,
                    };
                    if ok {
                        file.verified = check.is_some();
                        let _ = hub.save_file(&fr.rk, &fr.commit, &fr.path, &file);
                        Ok(st.pending.take().unwrap_or_default())
                    } else {
                        // Don't trust peers for this file again; the next request
                        // downloads it from the upstream.
                        let key = format!("{}/{}/{}", fr.rk, fr.commit, fr.path);
                        hub.distrust_peers.lock().unwrap().insert(key);
                        Err(std::io::Error::other(format!(
                            "{} from peers doesn't match huggingface.co",
                            fr.path
                        )))
                    }
                }
            };
            Some((item, (st, hub, fr, file, check)))
        },
    )
}

/// Pass a range request for an uncached file straight to the upstream.
async fn proxy_range(
    hub: &Hub,
    fr: &FileRequest,
    head: &UpstreamHead,
    start: u64,
    end: u64,
    headers: (StatusCode, HeaderMap),
) -> HubResult<Response> {
    let resp = hub
        .client
        .get(&head.url)
        .headers(fr.auth.clone())
        .header(header::RANGE, format!("bytes={start}-{}", end - 1))
        .send()
        .await
        .map_err(|e| HubError(StatusCode::BAD_GATEWAY, e.to_string()))?;
    if resp.status().as_u16() != 206 {
        return Err(Hub::upstream_error(resp).await);
    }
    let body = Body::from_stream(
        resp.bytes_stream()
            .map(|r| r.map_err(std::io::Error::other)),
    );
    Ok((headers.0, headers.1, body).into_response())
}

/// Serve a file straight from the upstream without caching it, for files chungus won't
/// carry (see [`crate::safety`]). Clients that ask for them still work while
/// huggingface.co is reachable; nothing is stored or offered to peers.
async fn passthrough(
    hub: &Hub,
    fr: &FileRequest,
    head: Option<UpstreamHead>,
    req: &HeaderMap,
    head_only: bool,
) -> HubResult<Response> {
    let Some(head) = head else {
        return err(
            StatusCode::BAD_GATEWAY,
            format!(
                "{} can run code when loaded, so chungus doesn't cache it, and huggingface.co \
                 can't be reached",
                fr.path
            ),
        );
    };
    if head_only {
        let (status, mut h) = file_headers(&fr.commit, &head.etag, head.size.unwrap_or(0), None);
        if head.size.is_none() {
            h.remove(header::CONTENT_LENGTH);
            h.remove("x-linked-size");
        }
        return Ok((status, h, Body::empty()).into_response());
    }
    let mut get = hub
        .client
        .get(&head.url)
        .headers(fr.auth.clone())
        .timeout(Duration::from_secs(24 * 3600));
    if let Some(range) = req.get(header::RANGE) {
        get = get.header(header::RANGE, range.as_bytes());
    }
    let resp = get
        .send()
        .await
        .map_err(|e| HubError(StatusCode::BAD_GATEWAY, e.to_string()))?;
    if !resp.status().is_success() {
        return Err(Hub::upstream_error(resp).await);
    }
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::OK);
    let (_, mut h) = file_headers(&fr.commit, &head.etag, head.size.unwrap_or(0), None);
    h.remove(header::CONTENT_LENGTH);
    for name in [header::CONTENT_LENGTH, header::CONTENT_RANGE] {
        if let Some(v) = resp
            .headers()
            .get(name.as_str())
            .and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok())
        {
            h.insert(name, v);
        }
    }
    let body = Body::from_stream(
        resp.bytes_stream()
            .map(|r| r.map_err(std::io::Error::other)),
    );
    Ok((status, h, body).into_response())
}

/// Download an uncached file from the upstream, streaming it to the client while writing
/// it to disk. When the download finishes and matches the upstream's hash, it's packed
/// into the store. The download continues if the client goes away.
async fn tee_download(
    hub: Arc<Hub>,
    fr: FileRequest,
    head: UpstreamHead,
    headers: (StatusCode, HeaderMap),
) -> HubResult<Response> {
    let resp = hub
        .client
        .get(&head.url)
        .headers(fr.auth.clone())
        .timeout(Duration::from_secs(24 * 3600))
        .send()
        .await
        .map_err(|e| HubError(StatusCode::BAD_GATEWAY, e.to_string()))?;
    if !resp.status().is_success() {
        return Err(Hub::upstream_error(resp).await);
    }
    let key = format!("{}/{}/{}", fr.rk, fr.commit, fr.path);
    if !hub.claim_download(&key) {
        // Someone else is already caching this file: just pass it through.
        let body = Body::from_stream(
            resp.bytes_stream()
                .map(|r| r.map_err(std::io::Error::other)),
        );
        return Ok((headers.0, headers.1, body).into_response());
    }
    let (tx, mut rx) = tokio::sync::mpsc::channel::<std::io::Result<Bytes>>(32);
    tokio::spawn(async move {
        let result = download_to_store(&hub, &fr, &head, resp, tx.clone()).await;
        if let Err(e) = &result {
            eprintln!("caching {key} failed: {e:#}");
            let _ = tx.send(Err(std::io::Error::other(format!("{e:#}")))).await;
        }
        hub.release_download(&key);
    });
    let body = Body::from_stream(stream::poll_fn(move |cx| rx.poll_recv(cx)));
    Ok((headers.0, headers.1, body).into_response())
}

async fn download_to_store(
    hub: &Hub,
    fr: &FileRequest,
    head: &UpstreamHead,
    resp: reqwest::Response,
    tx: tokio::sync::mpsc::Sender<std::io::Result<Bytes>>,
) -> Result<()> {
    let dir = tempfile::tempdir_in(hub.store.tmp_dir()?)?;
    let name = fr.path.rsplit('/').next().unwrap_or("file");
    let tmp = dir.path().join(name);
    let mut out = tokio::fs::File::create(&tmp).await?;
    let mut hasher = ContentHasher::new(&head.etag, head.size.unwrap_or(0));
    let mut body = resp.bytes_stream();
    let mut client_gone = false;
    while let Some(piece) = body.next().await {
        let piece = piece?;
        out.write_all(&piece).await?;
        if let Some(h) = &mut hasher {
            h.update(&piece);
        }
        if !client_gone && tx.send(Ok(piece)).await.is_err() {
            client_gone = true;
        }
    }
    out.flush().await?;
    drop(out);
    if let Some(h) = hasher
        && h.finish() != head.etag
    {
        bail!(
            "download of {}/{} doesn't match upstream hash {}",
            fr.repo,
            fr.path,
            head.etag
        );
    }
    let (store, path, etag) = (hub.store.clone(), fr.path.clone(), head.etag.clone());
    let file = tokio::task::spawn_blocking(move || -> Result<HubFile> {
        let (manifest, _) = crate::pack(&tmp, &store)?;
        let mut entry = manifest
            .files
            .into_iter()
            .next()
            .context("empty download")?;
        entry.path = path;
        let verified = is_content_hash(&etag);
        let etag = if etag.is_empty() {
            entry.hash.clone()
        } else {
            etag
        };
        Ok(HubFile {
            etag,
            entry,
            verified,
        })
    })
    .await??;
    hub.save_file(&fr.rk, &fr.commit, &fr.path, &file)?;
    Ok(())
}

fn hv(s: &str) -> HeaderValue {
    HeaderValue::from_str(s).unwrap_or_else(|_| HeaderValue::from_static(""))
}

#[derive(Debug, PartialEq)]
enum Route {
    Info {
        repo: String,
        rev: String,
    },
    Tree {
        repo: String,
        rev: String,
        path: String,
        recursive: bool,
    },
    Resolve {
        repo: String,
        rev: String,
        path: String,
    },
}

/// Parse the Hub URL shapes clients use. Revisions arrive percent-encoded (so
/// `refs/pr/1` is one segment), and file paths keep their slashes.
fn parse_route(path: &str, query: &str) -> Option<Route> {
    if let Some(rest) = path.strip_prefix("/api/models/") {
        if let Some((repo, rev)) = rest.split_once("/revision/") {
            let (repo, rev) = (decode(repo)?, decode(rev)?);
            return (is_repo(&repo) && is_rev(&rev)).then_some(Route::Info { repo, rev });
        }
        if let Some((repo, rest)) = rest.split_once("/tree/") {
            let (rev, sub) = rest.split_once('/').unwrap_or((rest, ""));
            let (repo, rev, sub) = (
                decode(repo)?,
                decode(rev)?,
                decode(sub.trim_end_matches('/'))?,
            );
            let recursive = query
                .split('&')
                .any(|kv| kv == "recursive=true" || kv == "recursive=True");
            return (is_repo(&repo) && is_rev(&rev) && (sub.is_empty() || is_file_path(&sub)))
                .then_some(Route::Tree {
                    repo,
                    rev,
                    path: sub,
                    recursive,
                });
        }
        return None;
    }
    let (repo, rest) = path.trim_start_matches('/').split_once("/resolve/")?;
    let (rev, file) = rest.split_once('/')?;
    let (repo, rev, file) = (decode(repo)?, decode(rev)?, decode(file)?);
    (is_repo(&repo) && is_rev(&rev) && is_file_path(&file)).then_some(Route::Resolve {
        repo,
        rev,
        path: file,
    })
}

fn is_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// `name` or `org/name`.
fn is_repo(s: &str) -> bool {
    let parts: Vec<&str> = s.split('/').collect();
    parts.len() <= 2 && parts.iter().all(|p| is_segment(p))
}

fn is_rev(s: &str) -> bool {
    !s.is_empty() && s.split('/').all(is_segment)
}

fn is_commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A repo-relative file path: no empty, `.` or `..` segments, no control characters.
fn is_file_path(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('/')
        && s.split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
        && !s.chars().any(|c| c.is_control() || c == '\\')
}

/// Store key for a repo id: `org/name` becomes `org--name`.
fn repo_key(repo: &str) -> String {
    repo.replace('/', "--")
}

fn decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Percent-encode everything but unreserved characters (so `/` becomes `%2F`).
fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

fn encode_path(s: &str) -> String {
    s.split('/').map(encode).collect::<Vec<_>>().join("/")
}

fn normalize_etag(s: &str) -> String {
    s.trim_start_matches("W/").trim_matches('"').to_string()
}

/// `<url>; rel="next"` from a Link header.
fn next_link(link: &str) -> Option<String> {
    link.split(',').find_map(|part| {
        let (url, params) = part.split_once(';')?;
        params.contains("rel=\"next\"").then(|| {
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

/// `bytes=a-b`, `bytes=a-` or `bytes=-n`, as a half-open `start..end`.
fn parse_range(h: &str, size: u64) -> Option<(u64, u64)> {
    let spec = h.strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let (start, end) = match (a.trim(), b.trim()) {
        ("", n) => {
            let n: u64 = n.parse().ok()?;
            (size.saturating_sub(n), size)
        }
        (a, "") => (a.parse().ok()?, size),
        (a, b) => (
            a.parse().ok()?,
            b.parse::<u64>().ok()?.saturating_add(1).min(size),
        ),
    };
    (start < end && start < size).then_some((start, end))
}

fn gate_key(repo: &str, auth: &HeaderMap) -> String {
    let token = auth
        .get(header::AUTHORIZATION)
        .map(|v| v.as_bytes())
        .unwrap_or_default();
    let mut h = blake3::Hasher::new();
    h.update(token);
    h.update(b"\0");
    h.update(repo.as_bytes());
    h.finalize().to_hex().to_string()
}

/// ETags that are content hashes: SHA-256 (LFS files) or a git blob id (small files).
fn is_content_hash(etag: &str) -> bool {
    (etag.len() == 64 || etag.len() == 40) && etag.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Hashes content the way a Hub ETag was made: SHA-256 for 64 hex chars (LFS files),
/// git blob SHA-1 (`blob <len>\0<content>`) for 40 (small files).
enum ContentHasher {
    Sha256(sha2::Sha256),
    Git(sha1::Sha1),
}

impl ContentHasher {
    fn new(etag: &str, size: u64) -> Option<Self> {
        if !is_content_hash(etag) {
            return None;
        }
        Some(if etag.len() == 64 {
            ContentHasher::Sha256(sha2::Sha256::new())
        } else {
            let mut s = sha1::Sha1::new();
            s.update(format!("blob {size}\0").as_bytes());
            ContentHasher::Git(s)
        })
    }

    fn update(&mut self, bytes: &[u8]) {
        match self {
            ContentHasher::Sha256(s) => s.update(bytes),
            ContentHasher::Git(s) => s.update(bytes),
        }
    }

    fn finish(self) -> String {
        let digest: Vec<u8> = match self {
            ContentHasher::Sha256(s) => s.finalize().to_vec(),
            ContentHasher::Git(s) => s.finalize().to_vec(),
        };
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hub_routes() {
        assert_eq!(
            parse_route("/api/models/org/model/revision/main", ""),
            Some(Route::Info {
                repo: "org/model".into(),
                rev: "main".into()
            })
        );
        assert_eq!(
            parse_route("/api/models/gpt2/revision/refs%2Fpr%2F1", ""),
            Some(Route::Info {
                repo: "gpt2".into(),
                rev: "refs/pr/1".into()
            })
        );
        assert_eq!(
            parse_route("/org/model/resolve/main/sub%20dir/model.safetensors", ""),
            Some(Route::Resolve {
                repo: "org/model".into(),
                rev: "main".into(),
                path: "sub dir/model.safetensors".into()
            })
        );
        assert_eq!(
            parse_route(
                "/api/models/org/model/tree/main",
                "recursive=true&expand=false"
            ),
            Some(Route::Tree {
                repo: "org/model".into(),
                rev: "main".into(),
                path: "".into(),
                recursive: true
            })
        );
        assert_eq!(
            parse_route("/org/model/resolve/main/../../etc/passwd", ""),
            None
        );
        assert_eq!(parse_route("/a/b/c/resolve/main/x", ""), None);
    }

    #[test]
    fn parses_ranges() {
        assert_eq!(parse_range("bytes=0-9", 100), Some((0, 10)));
        assert_eq!(parse_range("bytes=90-", 100), Some((90, 100)));
        assert_eq!(parse_range("bytes=-5", 100), Some((95, 100)));
        assert_eq!(parse_range("bytes=50-500", 100), Some((50, 100)));
        assert_eq!(parse_range("bytes=100-", 100), None);
        assert_eq!(parse_range("bytes=0-1,5-6", 100), None);
    }

    #[test]
    fn finds_next_page() {
        let link =
            r#"<https://hf.co/api/x?cursor=abc>; rel="next", <https://hf.co/api/x>; rel="first""#;
        assert_eq!(
            next_link(link).as_deref(),
            Some("https://hf.co/api/x?cursor=abc")
        );
    }
}
