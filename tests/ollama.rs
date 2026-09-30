//! Ollama pulls against a fake registry, LAN peers and a fake Ollama.

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use sha2::Digest as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use chungus::net;
use chungus::ollama::{Name, Ollama, Progress};
use chungus::store::Store;

fn random(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect()
}

fn digest(b: &[u8]) -> String {
    let h = sha2::Sha256::digest(b);
    format!(
        "sha256:{}",
        h.iter().map(|x| format!("{x:02x}")).collect::<String>()
    )
}

struct Model {
    manifest: Vec<u8>,
    blobs: Vec<Vec<u8>>,
}

fn model(weights: Vec<u8>) -> Model {
    let config = br#"{"model_format":"gguf","model_family":"llama"}"#.to_vec();
    let template = b"{{ .Prompt }}".to_vec();
    let layer = |t: &str, b: &[u8]| serde_json::json!({"mediaType": format!("application/vnd.ollama.image.{t}"), "digest": digest(b), "size": b.len()});
    let manifest = serde_json::to_vec(&serde_json::json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": digest(&config), "size": config.len()},
        "layers": [layer("model", &weights), layer("template", &template)],
    }))
    .unwrap();
    Model {
        manifest,
        blobs: vec![config, weights, template],
    }
}

/// registry.ollama.ai: `library/tiny:latest` and `library/tiny:v2`, a second quant that
/// shares most of its bytes. Manifests need an (anonymous) bearer token, as registries do.
struct Registry {
    latest: Model,
    v2: Model,
    /// Blob bytes served.
    served: AtomicU64,
    down: AtomicBool,
}

impl Registry {
    fn new() -> Arc<Self> {
        let weights = random(3_000_000, 1);
        let mut other = weights.clone();
        other[1_500_000..1_600_000].copy_from_slice(&random(100_000, 2));
        Arc::new(Registry {
            latest: model(weights),
            v2: model(other),
            served: AtomicU64::new(0),
            down: AtomicBool::new(false),
        })
    }

    fn blob(&self, d: &str) -> Option<&Vec<u8>> {
        self.latest
            .blobs
            .iter()
            .chain(&self.v2.blobs)
            .find(|b| digest(b) == d)
    }
}

async fn registry(State(r): State<Arc<Registry>>, uri: Uri, headers: HeaderMap) -> Response {
    if r.down.load(Ordering::SeqCst) {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let path = uri.path();
    if path == "/token" {
        return axum::Json(serde_json::json!({"token": "anon"})).into_response();
    }
    if let Some(tag) = path.strip_prefix("/v2/library/tiny/manifests/") {
        if headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            != Some("Bearer anon")
        {
            let host = headers[header::HOST].to_str().unwrap();
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header(
                    header::WWW_AUTHENTICATE,
                    format!(
                        r#"Bearer realm="http://{host}/token",service="registry",scope="repository:library/tiny:pull""#
                    ),
                )
                .body(Body::empty())
                .unwrap();
        }
        let m = match tag {
            "latest" => &r.latest,
            "v2" => &r.v2,
            _ => return StatusCode::NOT_FOUND.into_response(),
        };
        return (
            [("docker-content-digest", digest(&m.manifest))],
            m.manifest.clone(),
        )
            .into_response();
    }
    if let Some(d) = path.strip_prefix("/v2/library/tiny/blobs/") {
        // Blobs live on a CDN, behind a redirect.
        return Response::builder()
            .status(StatusCode::TEMPORARY_REDIRECT)
            .header(header::LOCATION, format!("/cdn/{d}"))
            .body(Body::empty())
            .unwrap();
    }
    if let Some(d) = path.strip_prefix("/cdn/") {
        let Some(b) = r.blob(d) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let (start, end) = match headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
            Some(range) => {
                let (a, z) = range.trim_start_matches("bytes=").split_once('-').unwrap();
                (a.parse::<usize>().unwrap(), z.parse::<usize>().unwrap() + 1)
            }
            None => (0, b.len()),
        };
        r.served.fetch_add((end - start) as u64, Ordering::SeqCst);
        let status = if headers.contains_key(header::RANGE) {
            StatusCode::PARTIAL_CONTENT
        } else {
            StatusCode::OK
        };
        return (status, b[start..end].to_vec()).into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = net::url_for(listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await });
    url
}

struct Machine {
    dir: PathBuf,
    store: Arc<Store>,
    ollama: Arc<Ollama>,
}

impl Machine {
    fn new(root: &Path, name: &str, upstream: Option<&str>, peers: &[String]) -> Self {
        Self::with(root, name, upstream, peers, false)
    }

    fn with(
        root: &Path,
        name: &str,
        upstream: Option<&str>,
        peers: &[String],
        trust_peers: bool,
    ) -> Self {
        let dir = root.join(name);
        let store = Arc::new(Store::open(&dir.join("store")).unwrap());
        let ollama = Arc::new(
            Ollama::new(
                store.clone(),
                dir.join("models"),
                upstream.map(str::to_string),
                peers.to_vec(),
                trust_peers,
            )
            .unwrap(),
        );
        Machine { dir, store, ollama }
    }

    async fn pull(&self, name: &str) -> anyhow::Result<chungus::ollama::Filled> {
        let name = Name::parse(name).unwrap();
        let f = self.ollama.fill(&name, &Progress::default()).await?;
        self.ollama.write_tag(&name, &f.manifest_bytes)?;
        Ok(f)
    }

    async fn serve(&self) -> String {
        spawn(net::router(self.store.clone())).await
    }

    fn has(&self, m: &Model, tag: &str) {
        let models = self.dir.join("models");
        for b in &m.blobs {
            let path = models.join("blobs").join(digest(b).replace(':', "-"));
            assert_eq!(&std::fs::read(&path).unwrap(), b, "{}", path.display());
        }
        let tag = models.join(format!("manifests/registry.ollama.ai/library/tiny/{tag}"));
        assert_eq!(std::fs::read(tag).unwrap(), m.manifest);
    }

    /// Bytes in the store's own chunk directory.
    fn chunk_bytes(&self) -> u64 {
        fn walk(p: &Path) -> u64 {
            std::fs::read_dir(p)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| {
                    let md = e.metadata().unwrap();
                    if md.is_dir() {
                        walk(&e.path())
                    } else {
                        md.len()
                    }
                })
                .sum()
        }
        walk(&self.dir.join("store/chunks"))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn pulls_from_peers_and_falls_back_to_the_registry() {
    let tmp = tempfile::tempdir().unwrap();
    let reg = Registry::new();
    let up = spawn(Router::new().fallback(registry).with_state(reg.clone())).await;

    // A pulls from the registry. The weights stay only in Ollama's directory.
    let a = Machine::new(tmp.path(), "a", Some(&up), &[]);
    let f = a.pull("tiny").await.unwrap();
    a.has(&reg.latest, "latest");
    assert_eq!(
        f.stats.upstream_bytes,
        reg.latest.blobs.iter().map(|b| b.len() as u64).sum::<u64>()
    );
    assert!(a.chunk_bytes() < 30_000, "{}", a.chunk_bytes());
    assert_eq!(a.store.links(&f.root).len(), 1);
    let a_url = a.serve().await;

    // B pulls the same model from A; only the manifest comes from the registry.
    reg.served.store(0, Ordering::SeqCst);
    let b = Machine::new(tmp.path(), "b", Some(&up), std::slice::from_ref(&a_url));
    let fb = b.pull("tiny").await.unwrap();
    b.has(&reg.latest, "latest");
    assert_eq!(reg.served.load(Ordering::SeqCst), 0);
    assert!(fb.stats.peer_bytes > 3_000_000, "{:?}", fb.stats);
    assert_eq!(fb.root, f.root);

    // A pulls a second quant from the registry; B then gets only its new chunks from A,
    // and copies the rest from its own copy of the first.
    a.pull("tiny:v2").await.unwrap();
    let fb2 = b.pull("tiny:v2").await.unwrap();
    b.has(&reg.v2, "v2");
    assert!(fb2.stats.local_bytes > 2_500_000, "{:?}", fb2.stats);
    assert!(fb2.stats.peer_bytes < 1_000_000, "{:?}", fb2.stats);

    // A peer that sends bad chunks: every one is refused, and the missing ranges come
    // from the registry.
    let evil = spawn(Router::new().fallback(evil_peer).with_state(a_url.clone())).await;
    reg.served.store(0, Ordering::SeqCst);
    let c = Machine::new(tmp.path(), "c", Some(&up), &[evil]);
    let fc = c.pull("tiny").await.unwrap();
    c.has(&reg.latest, "latest");
    assert_eq!(fc.stats.peer_bytes, 0);
    assert!(reg.served.load(Ordering::SeqCst) >= 3_000_000);

    // `ollama rm` of both tags on A: its blobs are gone, so its links are dropped and D
    // falls back to the registry.
    for b in reg.latest.blobs.iter().chain(&reg.v2.blobs) {
        let _ = std::fs::remove_file(a.dir.join("models/blobs").join(digest(b).replace(':', "-")));
    }
    let d = Machine::new(tmp.path(), "d", Some(&up), std::slice::from_ref(&a_url));
    let fd = d.pull("tiny").await.unwrap();
    d.has(&reg.latest, "latest");
    assert!(fd.stats.upstream_bytes >= 3_000_000, "{:?}", fd.stats);
    a.store.gc_links().unwrap();
    assert!(a.store.links(&f.root).is_empty());

    // Import: B's models, copied to a fresh machine, index to the same root with nothing
    // copied into the store.
    let e = Machine::new(tmp.path(), "e", None, &[]);
    copy_dir(&b.dir.join("models"), &e.dir.join("models"));
    let imported = e.ollama.import(&[]).await.unwrap();
    assert_eq!(imported.len(), 2);
    let root = imported
        .iter()
        .find(|(n, _)| n.tag == "latest")
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_eq!(root, &f.root);
    assert!(e.chunk_bytes() < 30_000);
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_pulls_need_agreeing_peers() {
    let tmp = tempfile::tempdir().unwrap();
    let reg = Registry::new();
    let up = spawn(Router::new().fallback(registry).with_state(reg.clone())).await;
    let a = Machine::new(tmp.path(), "a", Some(&up), &[]);
    a.pull("tiny").await.unwrap();
    let a_url = a.serve().await;

    // With the registry down, one peer's word isn't enough...
    reg.down.store(true, Ordering::SeqCst);
    let b = Machine::new(tmp.path(), "b", Some(&up), std::slice::from_ref(&a_url));
    let e = b.pull("tiny").await.err().unwrap().to_string();
    assert!(e.contains("offline and unverified"), "{e}");

    // ...unless the user says so.
    let b = Machine::with(tmp.path(), "b", None, std::slice::from_ref(&a_url), true);
    b.pull("tiny").await.unwrap();
    b.has(&reg.latest, "latest");
    let b_url = b.serve().await;

    // Two agreeing peers are.
    let c = Machine::new(tmp.path(), "c", None, &[a_url, b_url]);
    let f = c.pull("tiny").await.unwrap();
    c.has(&reg.latest, "latest");
    assert!(!f.online);
}

/// Forwards to a real peer, but corrupts every chunk.
async fn evil_peer(State(real): State<String>, uri: Uri) -> Response {
    let resp = reqwest::get(format!("{real}{}", uri.path())).await.unwrap();
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap();
    let mut body = resp.bytes().await.unwrap().to_vec();
    if uri.path().starts_with("/v1/chunks/") && body.len() > 3 {
        // A Stored blob of the right length and the wrong bytes.
        let n = chungus::store::decode(&body, 1 << 18)
            .map(|r| r.len())
            .unwrap_or(body.len() - 3);
        body = vec![1, 0, 0];
        body.extend(std::iter::repeat_n(7u8, n));
    }
    (status, body).into_response()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), dest).unwrap();
        }
    }
}

// ---------- the shim ----------

/// A stand-in for Ollama: `POST /api/pull` succeeds only if every blob is already in
/// place (so the shim must have filled them), and `GET /api/tags` answers.
struct FakeOllama {
    models: PathBuf,
    manifest: Vec<u8>,
    pulls: AtomicU64,
}

async fn fake_ollama(State(o): State<Arc<FakeOllama>>, uri: Uri, body: Bytes) -> Response {
    match uri.path() {
        "/api/tags" => axum::Json(serde_json::json!({"models": []})).into_response(),
        "/api/pull" => {
            o.pulls.fetch_add(1, Ordering::SeqCst);
            let req: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(req["model"], "tiny");
            let m: serde_json::Value = serde_json::from_slice(&o.manifest).unwrap();
            let mut digests = vec![m["config"]["digest"].as_str().unwrap().to_string()];
            for l in m["layers"].as_array().unwrap() {
                digests.push(l["digest"].as_str().unwrap().to_string());
            }
            let missing = digests
                .iter()
                .any(|d| !o.models.join("blobs").join(d.replace(':', "-")).exists());
            let last = if missing {
                r#"{"error":"would have downloaded"}"#
            } else {
                r#"{"status":"success"}"#
            };
            format!("{{\"status\":\"pulling manifest\"}}\n{last}\n").into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn shim_fills_blobs_then_lets_ollama_finish() {
    let tmp = tempfile::tempdir().unwrap();
    let reg = Registry::new();
    let up = spawn(Router::new().fallback(registry).with_state(reg.clone())).await;
    let a = Machine::new(tmp.path(), "a", Some(&up), &[]);
    a.pull("tiny").await.unwrap();
    let a_url = a.serve().await;

    let b = Machine::new(tmp.path(), "b", Some(&up), std::slice::from_ref(&a_url));
    let fake = Arc::new(FakeOllama {
        models: b.dir.join("models"),
        manifest: reg.latest.manifest.clone(),
        pulls: AtomicU64::new(0),
    });
    let backend = spawn(Router::new().fallback(fake_ollama).with_state(fake.clone())).await;
    let shim = spawn(chungus::ollama::router(b.ollama.clone(), &backend).unwrap()).await;

    // Other calls pass straight through.
    let tags = reqwest::get(format!("{shim}/api/tags")).await.unwrap();
    assert!(tags.status().is_success());
    assert_eq!(tags.text().await.unwrap(), r#"{"models":[]}"#);

    reg.served.store(0, Ordering::SeqCst);
    let resp = reqwest::Client::new()
        .post(format!("{shim}/api/pull"))
        .body(r#"{"model":"tiny"}"#)
        .send()
        .await
        .unwrap();
    let text = resp.text().await.unwrap();
    let last = text.lines().last().unwrap();
    assert_eq!(last, r#"{"status":"success"}"#, "{text}");
    assert_eq!(fake.pulls.load(Ordering::SeqCst), 1);
    assert_eq!(reg.served.load(Ordering::SeqCst), 0);

    // Offline, the shim writes the manifest itself instead of asking Ollama.
    let c = Machine::with(tmp.path(), "c", None, std::slice::from_ref(&a_url), true);
    let shim = spawn(chungus::ollama::router(c.ollama.clone(), &backend).unwrap()).await;
    let resp = reqwest::Client::new()
        .post(format!("{shim}/api/pull"))
        .body(r#"{"model":"tiny","stream":false}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.text().await.unwrap(), r#"{"status":"success"}"#);
    c.has(&reg.latest, "latest");
    assert_eq!(fake.pulls.load(Ordering::SeqCst), 1);
}
