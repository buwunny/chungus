//! The Hugging Face cache against a fake huggingface.co.

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use sha1::Digest as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use chungus::hub::{self, Hub};
use chungus::store::Store;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const TOKEN: &str = "Bearer good";

struct Fake {
    weights: Vec<u8>,
    config: Vec<u8>,
    /// Downloads of file bodies served.
    downloads: AtomicUsize,
    /// When set, every request fails with 503, as if huggingface.co were down.
    down: AtomicBool,
}

impl Fake {
    fn new() -> Self {
        let mut state = 99u64;
        let weights = (0..2_500_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 56) as u8
            })
            .collect();
        Fake {
            weights,
            config: br#"{"hidden_size": 64}"#.to_vec(),
            downloads: AtomicUsize::new(0),
            down: AtomicBool::new(false),
        }
    }

    fn file(&self, name: &str) -> Option<(&[u8], String)> {
        match name {
            "model.safetensors" => Some((&self.weights, hex(&sha2::Sha256::digest(&self.weights)))),
            "config.json" => {
                let mut h = sha1::Sha1::new();
                h.update(format!("blob {}\0", self.config.len()).as_bytes());
                h.update(&self.config);
                Some((&self.config, hex(&h.finalize())))
            }
            _ => None,
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Mimics the parts of huggingface.co the hub talks to. `org/gated` requires TOKEN for
/// file access, like a gated model.
async fn fake_hf(
    State(f): State<Arc<Fake>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    if f.down.load(Ordering::SeqCst) {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let path = uri.path();
    for repo in ["org/model", "org/gated"] {
        let gated = repo == "org/gated";
        if path.starts_with(&format!("/api/models/{repo}/revision/")) {
            let info = serde_json::json!({
                "id": repo, "sha": COMMIT, "gated": if gated { serde_json::json!("manual") } else { serde_json::json!(false) },
                "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors"}],
            });
            return axum::Json(info).into_response();
        }
        if path.starts_with(&format!("/api/models/{repo}/tree/")) {
            let tree: Vec<_> = ["config.json", "model.safetensors"]
                .iter()
                .map(|name| {
                    let (data, etag) = f.file(name).unwrap();
                    serde_json::json!({"type": "file", "path": name, "size": data.len(), "oid": "x", "xetHash": "abc"})
                        .as_object()
                        .cloned()
                        .map(|mut o| {
                            if etag.len() == 64 {
                                o.insert("lfs".into(), serde_json::json!({"oid": etag, "size": data.len(), "pointerSize": 134}));
                            }
                            serde_json::Value::Object(o)
                        })
                        .unwrap()
                })
                .collect();
            return axum::Json(tree).into_response();
        }
        if let Some(rest) = path.strip_prefix(&format!("/{repo}/resolve/")) {
            let name = rest.split_once('/').unwrap().1;
            let authed = headers
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                == Some(TOKEN);
            if gated && !authed {
                return (StatusCode::UNAUTHORIZED, "gated").into_response();
            }
            let Some((data, etag)) = f.file(name) else {
                return StatusCode::NOT_FOUND.into_response();
            };
            // LFS files redirect to a CDN, like huggingface.co does.
            if etag.len() == 64 {
                return Response::builder()
                    .status(StatusCode::FOUND)
                    .header(header::LOCATION, format!("/cdn/{name}"))
                    .header("x-linked-etag", format!("\"{etag}\""))
                    .header("x-linked-size", data.len().to_string())
                    .header("x-repo-commit", COMMIT)
                    .body(Body::empty())
                    .unwrap();
            }
            if method == Method::GET {
                f.downloads.fetch_add(1, Ordering::SeqCst);
            }
            return Response::builder()
                .header(header::ETAG, format!("\"{etag}\""))
                .header("x-repo-commit", COMMIT)
                .body(Body::from(data.to_vec()))
                .unwrap();
        }
    }
    if let Some(name) = path.strip_prefix("/cdn/") {
        let (data, _) = f.file(name).unwrap();
        if method == Method::GET {
            f.downloads.fetch_add(1, Ordering::SeqCst);
        }
        if let Some(r) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
            let (a, b) = r.strip_prefix("bytes=").unwrap().split_once('-').unwrap();
            let (a, b): (usize, usize) = (a.parse().unwrap(), b.parse().unwrap());
            return Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes {a}-{b}/{}", data.len()),
                )
                .body(Body::from(data[a..=b].to_vec()))
                .unwrap();
        }
        return Body::from(data.to_vec()).into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await });
    url
}

async fn spawn_fake() -> (Arc<Fake>, String) {
    let fake = Arc::new(Fake::new());
    let url = spawn(Router::new().fallback(fake_hf).with_state(fake.clone())).await;
    (fake, url)
}

async fn spawn_hub(dir: &std::path::Path, upstream: Option<String>, peers: Vec<String>) -> String {
    let store = Arc::new(Store::open(dir).unwrap());
    spawn(hub::router(Arc::new(
        Hub::new(store, upstream, peers).unwrap(),
    )))
    .await
}

async fn get(url: &str, auth: Option<&str>) -> reqwest::Response {
    let mut req = reqwest::Client::new().get(url);
    if let Some(a) = auth {
        req = req.header(header::AUTHORIZATION, a);
    }
    req.send().await.unwrap()
}

/// Downloads are packed in the background after the response ends; wait for that.
async fn wait_cached(hub: &str, repo: &str, file: &str) {
    for _ in 0..100 {
        let key = format!(
            "{hub}/v1/meta/hub/{}/{COMMIT}/files.json",
            repo.replace('/', "--")
        );
        if let Ok(r) = reqwest::get(&key).await
            && r.status().is_success()
            && r.text().await.unwrap().contains(file)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("{file} was never cached");
}

#[tokio::test(flavor = "multi_thread")]
async fn caches_from_upstream_and_serves_ranges() {
    let tmp = tempfile::tempdir().unwrap();
    let (fake, up) = spawn_fake().await;
    let hub = spawn_hub(&tmp.path().join("a"), Some(up), vec![]).await;

    let head = reqwest::Client::new()
        .head(format!("{hub}/org/model/resolve/main/model.safetensors"))
        .send()
        .await
        .unwrap();
    assert_eq!(head.status(), 200);
    assert_eq!(head.headers()["x-repo-commit"], COMMIT);
    let (_, etag) = fake.file("model.safetensors").unwrap();
    assert_eq!(
        head.headers()["x-linked-etag"],
        format!("\"{etag}\"").as_str()
    );
    assert_eq!(
        head.headers()["x-linked-size"],
        fake.weights.len().to_string().as_str()
    );
    assert_eq!(
        fake.downloads.load(Ordering::SeqCst),
        0,
        "HEAD must not download"
    );

    for name in ["model.safetensors", "config.json"] {
        let body = get(&format!("{hub}/org/model/resolve/main/{name}"), None).await;
        assert_eq!(body.status(), 200);
        assert_eq!(body.bytes().await.unwrap(), fake.file(name).unwrap().0);
        wait_cached(&hub, "org/model", name).await;
    }
    assert_eq!(fake.downloads.load(Ordering::SeqCst), 2);

    // Served from the cache now, including ranges.
    let again = get(
        &format!("{hub}/org/model/resolve/{COMMIT}/model.safetensors"),
        None,
    )
    .await;
    assert_eq!(again.bytes().await.unwrap(), fake.weights);
    let part = reqwest::Client::new()
        .get(format!("{hub}/org/model/resolve/main/model.safetensors"))
        .header(header::RANGE, "bytes=100000-100099")
        .send()
        .await
        .unwrap();
    assert_eq!(part.status(), 206);
    assert_eq!(part.bytes().await.unwrap(), fake.weights[100000..100100]);
    assert_eq!(fake.downloads.load(Ordering::SeqCst), 2);

    // Tree listings pass through without Xet hashes.
    let tree = get(
        &format!("{hub}/api/models/org/model/tree/main?recursive=true"),
        None,
    )
    .await;
    let tree = tree.text().await.unwrap();
    assert!(tree.contains("model.safetensors") && !tree.contains("xetHash"));
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_hub_fills_from_a_peer() {
    let tmp = tempfile::tempdir().unwrap();
    let (fake, up) = spawn_fake().await;
    let online = spawn_hub(&tmp.path().join("a"), Some(up), vec![]).await;
    let url = format!("{online}/org/model/resolve/main/model.safetensors");
    get(&url, None).await.bytes().await.unwrap();
    wait_cached(&online, "org/model", "model.safetensors").await;
    get(
        &format!("{online}/api/models/org/model/tree/main?recursive=true"),
        None,
    )
    .await;

    let offline = spawn_hub(&tmp.path().join("b"), None, vec![online]).await;
    let info = get(
        &format!("{offline}/api/models/org/model/revision/main"),
        None,
    )
    .await;
    assert_eq!(info.status(), 200);
    let body = get(
        &format!("{offline}/org/model/resolve/main/model.safetensors"),
        None,
    )
    .await;
    assert_eq!(body.status(), 200);
    assert_eq!(body.bytes().await.unwrap(), fake.weights);
    let tree = get(
        &format!("{offline}/api/models/org/model/tree/main?recursive=true"),
        None,
    )
    .await;
    assert!(tree.text().await.unwrap().contains("model.safetensors"));
    assert_eq!(fake.downloads.load(Ordering::SeqCst), 1);

    let missing = get(
        &format!("{offline}/org/model/resolve/main/config.json"),
        None,
    )
    .await;
    assert_eq!(missing.status(), 404);
}

#[tokio::test(flavor = "multi_thread")]
async fn peer_copy_that_does_not_match_upstream_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (fake, up) = spawn_fake().await;

    // A liar peer claiming org/model's weights are some other bytes.
    let liar_store = Arc::new(Store::open(&tmp.path().join("liar")).unwrap());
    let fake_file = tmp.path().join("model.safetensors");
    std::fs::write(&fake_file, vec![7u8; fake.weights.len()]).unwrap();
    let (m, _) = chungus::pack(&fake_file, &liar_store).unwrap();
    let (_, real_etag) = fake.file("model.safetensors").unwrap();
    let record = serde_json::json!({
        "model.safetensors": {"etag": real_etag, "entry": m.files[0], "verified": true}
    });
    liar_store
        .put_meta(
            &format!("hub/org--model/{COMMIT}/files.json"),
            record.to_string().as_bytes(),
        )
        .unwrap();
    let liar = spawn(chungus::net::router(liar_store)).await;

    let hub = spawn_hub(&tmp.path().join("h"), Some(up), vec![liar]).await;
    let url = format!("{hub}/org/model/resolve/main/model.safetensors");
    // First try: the peer's copy is streamed but cut off before the end.
    let first = get(&url, None).await.bytes().await;
    assert!(first.is_err() || first.unwrap() != fake.weights);
    // After that the hub goes to the upstream.
    let second = get(&url, None).await.bytes().await.unwrap();
    assert_eq!(second, fake.weights);
}

#[tokio::test(flavor = "multi_thread")]
async fn gated_files_need_an_accepted_token() {
    let tmp = tempfile::tempdir().unwrap();
    let (fake, up) = spawn_fake().await;
    let hub = spawn_hub(&tmp.path().join("a"), Some(up), vec![]).await;
    let url = format!("{hub}/org/gated/resolve/main/config.json");

    assert_eq!(get(&url, None).await.status(), 401);
    let ok = get(&url, Some(TOKEN)).await;
    assert_eq!(ok.status(), 200);
    assert_eq!(ok.bytes().await.unwrap(), fake.config);
    wait_cached(&hub, "org/gated", "config.json").await;
    assert_eq!(get(&url, Some("Bearer evil")).await.status(), 401);

    // With huggingface.co down, the cache serves only tokens it saw accepted.
    fake.down.store(true, Ordering::SeqCst);
    let cached = get(&url, Some(TOKEN)).await;
    assert_eq!(cached.status(), 200);
    assert_eq!(cached.bytes().await.unwrap(), fake.config);
    assert_eq!(get(&url, Some("Bearer evil")).await.status(), 403);
    assert_eq!(get(&url, None).await.status(), 401);
}
