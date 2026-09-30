//! Content-addressed chunk store on local disk.
//!
//! Each chunk is stored once, at `<root>/<hh>/<hash>`, where the hash is BLAKE3 of the
//! chunk's *raw* bytes. Addressing by raw content means the codec can change without
//! breaking any hash, and a reader verifies exactly the bytes it will use.
//!
//! Blob layout: `[version=1][codec][param][payload...]`, where `param` is the element
//! width for `PlaneZstd` and the float kind for `ExponentZstd`.
//!
//! A store can also hold *linked* files: files kept elsewhere on disk (an Ollama blob,
//! say) that are byte for byte one of a manifest's files. Their chunks are read from the
//! file itself instead of being copied into `chunks/`, checked by size, mtime and BLAKE3
//! on every read, and handed out as `Stored` blobs. A link whose file changed or vanished
//! is dropped.

use anyhow::{Context, Result, bail};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::manifest::Manifest;
use crate::segment::Dtype;
use crate::sign::{self, Signature};
use crate::transform::{self, FloatKind};

/// Version byte at the start of every chunk blob.
pub const BLOB_VERSION: u8 = 1;
/// Version of the on-disk layout, kept in `<store>/VERSION`. Stores made before the file
/// existed are version 1.
pub const STORE_VERSION: u32 = 1;
pub const ZSTD_LEVEL: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    /// Payload is the raw chunk, uncompressed (used when compression doesn't help).
    Stored = 0,
    /// Payload is zstd(chunk).
    Zstd = 1,
    /// Payload is zstd(byte_plane_split(chunk, width)).
    PlaneZstd = 2,
    /// Payload is zstd(split_exponent(chunk, kind)).
    ExponentZstd = 3,
}

impl Codec {
    fn from_u8(b: u8) -> Result<Self> {
        Ok(match b {
            0 => Codec::Stored,
            1 => Codec::Zstd,
            2 => Codec::PlaneZstd,
            3 => Codec::ExponentZstd,
            _ => bail!("unknown codec {b}"),
        })
    }
}

/// Encode a chunk. Float chunks also try the byte-plane and exponent-split transforms;
/// whichever encoding is smallest wins, falling back to storing the bytes as-is.
pub fn encode(raw: &[u8], dtype: Dtype) -> Result<Vec<u8>> {
    let width = dtype.width();
    let mut best = (Codec::Stored, 0u8, raw.to_vec());
    let plain = zstd::bulk::compress(raw, ZSTD_LEVEL)?;
    if plain.len() < best.2.len() {
        best = (Codec::Zstd, 0, plain);
    }
    if let Some(kind) = dtype.float_kind() {
        let split = zstd::bulk::compress(&transform::split_exponent(raw, kind), ZSTD_LEVEL)?;
        if split.len() < best.2.len() {
            best = (Codec::ExponentZstd, kind as u8, split);
        }
    } else if width > 1 {
        let planes = zstd::bulk::compress(&transform::split(raw, width), ZSTD_LEVEL)?;
        if planes.len() < best.2.len() {
            best = (Codec::PlaneZstd, width as u8, planes);
        }
    }
    let mut blob = Vec::with_capacity(3 + best.2.len());
    blob.extend_from_slice(&[BLOB_VERSION, best.0 as u8, best.1]);
    blob.extend_from_slice(&best.2);
    Ok(blob)
}

/// Decode a blob back to the raw chunk. `len` is the expected raw length.
pub fn decode(blob: &[u8], len: usize) -> Result<Vec<u8>> {
    // `len` comes from a manifest, which anyone can write; don't let it size a buffer.
    if len > crate::chunk::MAX_SIZE {
        bail!(
            "chunk length {len} is over the {} byte maximum",
            crate::chunk::MAX_SIZE
        );
    }
    if blob.len() < 3 {
        bail!("bad blob header");
    }
    if blob[0] != BLOB_VERSION {
        if blob[0] > BLOB_VERSION {
            bail!(
                "chunk encoded by a newer chungus (blob v{}); upgrade chungus",
                blob[0]
            );
        }
        bail!("bad blob header");
    }
    let (codec, param, payload) = (Codec::from_u8(blob[1])?, blob[2], &blob[3..]);
    let raw = match codec {
        Codec::Stored => payload.to_vec(),
        Codec::Zstd => zstd::bulk::decompress(payload, len)?,
        Codec::PlaneZstd => transform::join(&zstd::bulk::decompress(payload, len)?, param as usize),
        Codec::ExponentZstd => {
            let kind = match param {
                0 => FloatKind::Bf16,
                1 => FloatKind::F32,
                _ => bail!("unknown float kind {param}"),
            };
            transform::join_exponent(&zstd::bulk::decompress(payload, len)?, kind)
        }
    };
    if raw.len() != len {
        bail!("decoded {} bytes, expected {len}", raw.len());
    }
    Ok(raw)
}

/// True if `s` is a lowercase hex BLAKE3 digest. Checked before any hash from the
/// network or a manifest is turned into a path.
pub fn is_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// On-disk layout: `chunks/<hh>/<hash>` for blobs, `manifests/<root>.json` for manifests.
pub struct Store {
    root: PathBuf,
    /// Manifest roots and chunk hashes this store refuses to hold or hand out.
    blocked: RwLock<HashSet<String>>,
    gates: RwLock<Gates>,
    links: RwLock<Links>,
    opened: Instant,
}

/// One manifest file that lives outside the store, in `target`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Link {
    /// The file's path in the manifest.
    pub path: String,
    /// Where the file is on disk.
    pub target: PathBuf,
    pub size: u64,
    /// Modification time when linked, in nanoseconds since the Unix epoch.
    pub mtime_ns: u64,
}

impl Link {
    /// A link to `target` as it is on disk now.
    pub fn new(path: &str, target: &Path) -> Result<Self> {
        let md = fs::metadata(target).with_context(|| format!("stat {}", target.display()))?;
        Ok(Link {
            path: path.to_string(),
            target: target.to_path_buf(),
            size: md.len(),
            mtime_ns: mtime_ns(&md),
        })
    }

    /// Whether the file on disk still looks like the one that was linked.
    pub fn is_current(&self) -> bool {
        fs::metadata(&self.target)
            .is_ok_and(|md| md.is_file() && md.len() == self.size && mtime_ns(&md) == self.mtime_ns)
    }
}

fn mtime_ns(md: &fs::Metadata) -> u64 {
    md.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos() as u64)
}

/// `links/<root>.json`: the files of manifest `root` that are linked rather than stored.
#[derive(Serialize, Deserialize, Default)]
struct LinkRecord {
    files: Vec<Link>,
}

/// Where each linked chunk can be read, built from the link records when the store opens.
#[derive(Default)]
struct Links {
    targets: Vec<Target>,
    /// Chunk hash -> (index into `targets`, offset, length). One location per chunk.
    chunks: HashMap<[u8; 32], (usize, u64, u32)>,
    /// Modification time of `links/` when loaded, so links another process adds (a
    /// `chungus ollama` next to a `chungus node` on the same store) are picked up.
    dir_mtime: Option<std::time::SystemTime>,
}

struct Target {
    root: String,
    link: Link,
    /// When the file was last seen unchanged, in ms since the store opened (0 = never), so
    /// `contains` doesn't stat a file for every chunk it's asked about.
    checked_ms: AtomicU64,
}

/// How long a linked file that looked unchanged is trusted by `contains` before it is
/// checked again. Reads always check.
const LINK_RECHECK_MS: u64 = 10_000;

/// Models behind a Hugging Face repo's gate, from a registry this store follows.
#[derive(Default)]
struct Gates {
    /// The registry keys that sign access tickets, and when each stops being accepted.
    issuers: Vec<(String, u64)>,
    /// Manifest root -> Hugging Face repo.
    roots: HashMap<String, String>,
    /// Chunk hash -> repos of the gated models in this store that contain it.
    chunks: HashMap<String, Vec<String>>,
}

impl Store {
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("chunks"))
            .with_context(|| format!("create store {}", root.display()))?;
        fs::create_dir_all(root.join("manifests"))?;
        check_version(root)?;
        let store = Store {
            root: root.to_path_buf(),
            blocked: Default::default(),
            gates: Default::default(),
            links: Default::default(),
            opened: Instant::now(),
        };
        store.reload_links()?;
        Ok(store)
    }

    /// Replace the blocklist, and delete any blocked chunks and manifests already on disk.
    /// Returns how many were deleted.
    pub fn set_blocked(&self, hashes: HashSet<String>) -> usize {
        let mut removed = 0;
        let mut unlinked = false;
        for h in hashes.iter().filter(|h| is_hash(h)) {
            for path in [self.path(h), self.manifest_path(h)] {
                if fs::remove_file(path).is_ok() {
                    removed += 1;
                }
            }
            // A linked file isn't the store's to delete; forget it instead.
            unlinked |= fs::remove_file(self.link_path(h)).is_ok();
        }
        *self.blocked.write().unwrap() = hashes;
        if unlinked {
            let _ = self.reload_links();
        }
        removed
    }

    /// Replace the gated models (root -> Hugging Face repo) and the registry keys whose
    /// access tickets open them (with when each stops being accepted).
    pub fn set_gates(&self, issuers: Vec<(String, u64)>, roots: HashMap<String, String>) {
        let mut chunks: HashMap<String, Vec<String>> = HashMap::new();
        for (root, repo) in &roots {
            if let Ok(m) = self.get_manifest(root) {
                add_gated_chunks(&mut chunks, &m, repo);
            }
        }
        *self.gates.write().unwrap() = Gates {
            issuers,
            roots,
            chunks,
        };
    }

    /// The Hugging Face repos whose gates cover chunk `hash`; empty if it's free to share.
    pub fn gates_of(&self, hash: &str) -> Vec<String> {
        self.gates
            .read()
            .unwrap()
            .chunks
            .get(hash)
            .cloned()
            .unwrap_or_default()
    }

    /// The registry keys whose access tickets are accepted now; none until a registry has
    /// been followed.
    pub fn ticket_issuers(&self) -> Vec<String> {
        let now = crate::registry::now();
        let g = self.gates.read().unwrap();
        g.issuers
            .iter()
            .filter(|(_, until)| now < *until)
            .map(|(k, _)| k.clone())
            .collect()
    }

    pub fn is_blocked(&self, hash: &str) -> bool {
        self.blocked.read().unwrap().contains(hash)
    }

    fn check_allowed(&self, hash: &str) -> Result<()> {
        if self.is_blocked(hash) {
            bail!("{hash} is on the blocklist");
        }
        Ok(())
    }

    /// The store's directory.
    pub fn dir(&self) -> &Path {
        &self.root
    }

    fn path(&self, hash: &str) -> PathBuf {
        self.root.join("chunks").join(&hash[..2]).join(hash)
    }

    pub fn contains(&self, hash: &str) -> bool {
        is_hash(hash)
            && !self.is_blocked(hash)
            && (self.path(hash).exists() || self.has_linked(hash))
    }

    /// Write a blob unless it's already present. Returns true if it was new.
    pub fn put(&self, hash: &str, blob: &[u8]) -> Result<bool> {
        if !is_hash(hash) {
            bail!("invalid chunk hash {hash:?}");
        }
        self.check_allowed(hash)?;
        let path = self.path(hash);
        if path.exists() {
            return Ok(false);
        }
        // Several threads may store the same chunk at once (a file that repeats itself);
        // exactly one of them wins, and the rest see it already there.
        let dir = path.parent().unwrap();
        fs::create_dir_all(dir)?;
        let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
        tmp.write_all(blob)?;
        match tmp.persist_noclobber(&path) {
            Ok(_) => Ok(true),
            Err(e) if path.exists() => {
                drop(e);
                Ok(false)
            }
            Err(e) => Err(e.error.into()),
        }
    }

    pub fn get(&self, hash: &str) -> Result<Vec<u8>> {
        if !is_hash(hash) {
            bail!("invalid chunk hash {hash:?}");
        }
        self.check_allowed(hash)?;
        match fs::read(self.path(hash)) {
            Ok(blob) => Ok(blob),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match self.read_linked(hash) {
                Some(r) => r,
                None => Err(e).with_context(|| format!("missing chunk {hash}")),
            },
            Err(e) => Err(e).with_context(|| format!("read chunk {hash}")),
        }
    }

    fn manifest_path(&self, root: &str) -> PathBuf {
        self.root.join("manifests").join(format!("{root}.json"))
    }

    pub fn put_manifest(&self, m: &Manifest) -> Result<()> {
        if !is_hash(&m.root) || !m.verify_root() {
            bail!("refusing to store a manifest whose root doesn't verify");
        }
        crate::safety::check_manifest(m)?;
        self.check_allowed(&m.root)?;
        write_atomic(&self.manifest_path(&m.root), &serde_json::to_vec_pretty(m)?)?;
        let mut gates = self.gates.write().unwrap();
        if let Some(repo) = gates.roots.get(&m.root).cloned() {
            add_gated_chunks(&mut gates.chunks, m, &repo);
        }
        Ok(())
    }

    pub fn get_manifest_bytes(&self, root: &str) -> Result<Vec<u8>> {
        if !is_hash(root) {
            bail!("invalid manifest root {root:?}");
        }
        self.check_allowed(root)?;
        fs::read(self.manifest_path(root)).with_context(|| format!("no manifest {root}"))
    }

    /// A manifest's bytes, to hand to someone else: refused if it lists unsafe files (it
    /// may predate the check in [`Store::put_manifest`]).
    pub fn get_safe_manifest_bytes(&self, root: &str) -> Result<Vec<u8>> {
        let bytes = self.get_manifest_bytes(root)?;
        crate::safety::check_manifest(&crate::manifest::parse(&bytes)?)?;
        Ok(bytes)
    }

    pub fn get_manifest(&self, root: &str) -> Result<Manifest> {
        crate::manifest::parse(&self.get_manifest_bytes(root)?)
    }

    fn signatures_path(&self, root: &str) -> PathBuf {
        self.root
            .join("manifests")
            .join(format!("{root}.sigs.json"))
    }

    /// Signatures on manifest `root` held by this store.
    pub fn signatures(&self, root: &str) -> Result<Vec<Signature>> {
        if !is_hash(root) {
            bail!("invalid manifest root {root:?}");
        }
        match fs::read(self.signatures_path(root)) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e.into()),
        }
    }

    /// Add signatures to manifest `root`, keeping only valid ones and one per key.
    /// Returns how many were new.
    pub fn add_signatures(&self, root: &str, new: &[Signature]) -> Result<usize> {
        let mut all = self.signatures(root)?;
        let before = all.len();
        for s in new {
            if sign::verify(s, root) && !all.iter().any(|a| a.key == s.key) {
                all.push(s.clone());
            }
        }
        if all.len() > before {
            write_atomic(
                &self.signatures_path(root),
                &serde_json::to_vec_pretty(&all)?,
            )?;
        }
        Ok(all.len() - before)
    }

    /// Roots of every manifest in the store, sorted.
    pub fn manifests(&self) -> Result<Vec<String>> {
        let mut roots = Vec::new();
        for entry in fs::read_dir(self.root.join("manifests"))? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if let Some(root) = name.strip_suffix(".json")
                && is_hash(root)
                && !self.is_blocked(root)
            {
                roots.push(root.to_string());
            }
        }
        roots.sort();
        Ok(roots)
    }
}

fn add_gated_chunks(chunks: &mut HashMap<String, Vec<String>>, m: &Manifest, repo: &str) {
    for c in m.files.iter().flat_map(|f| &f.chunks) {
        let repos = chunks.entry(c.hash.clone()).or_default();
        if !repos.iter().any(|r| r == repo) {
            repos.push(repo.to_string());
        }
    }
}

/// True if `key` is a safe relative metadata path: segments of `[A-Za-z0-9._%-]`, no
/// `.` or `..` segments.
pub fn is_meta_key(key: &str) -> bool {
    !key.is_empty()
        && key.split('/').all(|seg| {
            !seg.is_empty()
                && seg != "."
                && seg != ".."
                && seg
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'%' | b'-'))
        })
}

impl Store {
    fn link_path(&self, root: &str) -> PathBuf {
        self.root.join("links").join(format!("{root}.json"))
    }

    /// Record that the files `links` name hold manifest `root`'s files of the same path,
    /// replacing earlier links for those paths. The manifest must already be in the store,
    /// and each file must have the size the manifest gives it. Nothing is read or copied:
    /// the caller has already checked the contents (by packing them, say).
    pub fn add_links(&self, root: &str, links: Vec<Link>) -> Result<()> {
        let m = self.get_manifest(root)?;
        for l in &links {
            let Some(f) = m.files.iter().find(|f| f.path == l.path) else {
                bail!("manifest {root} has no file {}", l.path);
            };
            if f.size != l.size {
                bail!("{} is {} bytes, not {}", l.target.display(), l.size, f.size);
            }
        }
        let mut record = self.link_record(root);
        record
            .files
            .retain(|old| !links.iter().any(|l| l.path == old.path));
        record.files.extend(links);
        write_atomic(&self.link_path(root), &serde_json::to_vec_pretty(&record)?)?;
        self.reload_links()
    }

    /// The linked files of manifest `root`.
    pub fn links(&self, root: &str) -> Vec<Link> {
        self.link_record(root).files
    }

    fn link_record(&self, root: &str) -> LinkRecord {
        if !is_hash(root) {
            return LinkRecord::default();
        }
        fs::read(self.link_path(root))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// Drop every link whose file is gone or changed, and every link record whose manifest
    /// is gone. Returns how many files were unlinked.
    pub fn gc_links(&self) -> Result<usize> {
        let dir = self.root.join("links");
        let mut dropped = 0;
        let Ok(entries) = fs::read_dir(&dir) else {
            return Ok(0);
        };
        for entry in entries {
            let path = entry?.path();
            let Some(root) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".json"))
                .filter(|r| is_hash(r))
                .map(str::to_string)
            else {
                continue;
            };
            let record = self.link_record(&root);
            let before = record.files.len();
            if !self.manifest_path(&root).exists() {
                fs::remove_file(&path)?;
                dropped += before;
                continue;
            }
            let files: Vec<Link> = record.files.into_iter().filter(Link::is_current).collect();
            if files.len() < before {
                dropped += before - files.len();
                if files.is_empty() {
                    fs::remove_file(&path)?;
                } else {
                    write_atomic(&path, &serde_json::to_vec_pretty(&LinkRecord { files })?)?;
                }
            }
        }
        self.reload_links()?;
        Ok(dropped)
    }

    /// Reload the links if another process changed them since they were loaded.
    fn refresh_links(&self) {
        let now = fs::metadata(self.root.join("links"))
            .and_then(|m| m.modified())
            .ok();
        if now.is_some() && now != self.links.read().unwrap().dir_mtime {
            let _ = self.reload_links();
        }
    }

    /// Rebuild the in-memory chunk locations from the link records on disk.
    fn reload_links(&self) -> Result<()> {
        let mut links = Links {
            dir_mtime: fs::metadata(self.root.join("links"))
                .and_then(|m| m.modified())
                .ok(),
            ..Default::default()
        };
        if let Ok(entries) = fs::read_dir(self.root.join("links")) {
            for entry in entries {
                let name = entry?.file_name().to_string_lossy().into_owned();
                let Some(root) = name.strip_suffix(".json").filter(|r| is_hash(r)) else {
                    continue;
                };
                if self.is_blocked(root) {
                    continue;
                }
                let Ok(m) = self.get_manifest(root) else {
                    continue;
                };
                for link in self.link_record(root).files {
                    let Some(f) = m.files.iter().find(|f| f.path == link.path) else {
                        continue;
                    };
                    if f.size != link.size {
                        continue;
                    }
                    let t = links.targets.len();
                    let mut offset = 0u64;
                    for c in &f.chunks {
                        if let Some(key) = hash_bytes(&c.hash) {
                            links.chunks.entry(key).or_insert((t, offset, c.len));
                        }
                        offset += c.len as u64;
                    }
                    links.targets.push(Target {
                        root: root.to_string(),
                        link,
                        checked_ms: AtomicU64::new(0),
                    });
                }
            }
        }
        *self.links.write().unwrap() = links;
        Ok(())
    }

    fn now_ms(&self) -> u64 {
        self.opened.elapsed().as_millis() as u64 + 1
    }

    fn has_linked(&self, hash: &str) -> bool {
        let Some(key) = hash_bytes(hash) else {
            return false;
        };
        if !self.links.read().unwrap().chunks.contains_key(&key) {
            self.refresh_links();
        }
        let target = {
            let links = self.links.read().unwrap();
            let Some(&(t, _, _)) = links.chunks.get(&key) else {
                return false;
            };
            let target = &links.targets[t];
            let checked = target.checked_ms.load(Ordering::Relaxed);
            if checked != 0 && self.now_ms() - checked < LINK_RECHECK_MS {
                return true;
            }
            if target.link.is_current() {
                target.checked_ms.store(self.now_ms(), Ordering::Relaxed);
                return true;
            }
            (target.root.clone(), target.link.path.clone())
        };
        self.drop_link(&target.0, &target.1);
        // Another linked file may hold the same chunk.
        let links = self.links.read().unwrap();
        links.chunks.contains_key(&key)
    }

    /// A linked chunk as a `Stored` blob, or None if no linked file holds it.
    fn read_linked(&self, hash: &str) -> Option<Result<Vec<u8>>> {
        let key = hash_bytes(hash)?;
        if !self.links.read().unwrap().chunks.contains_key(&key) {
            self.refresh_links();
        }
        // A file that fails its checks is unlinked, and the next file holding the chunk
        // (if any) is tried.
        for _ in 0..8 {
            let (root, link, offset, len) = {
                let links = self.links.read().unwrap();
                let &(t, offset, len) = links.chunks.get(&key)?;
                let target = &links.targets[t];
                (target.root.clone(), target.link.clone(), offset, len)
            };
            match read_at(&link, offset, len as usize) {
                Ok(raw) if blake3::hash(&raw).as_bytes() == &key => {
                    let mut blob = Vec::with_capacity(3 + raw.len());
                    blob.extend_from_slice(&[BLOB_VERSION, Codec::Stored as u8, 0]);
                    blob.extend_from_slice(&raw);
                    return Some(Ok(blob));
                }
                _ => self.drop_link(&root, &link.path),
            }
        }
        Some(Err(anyhow::anyhow!(
            "no linked copy of chunk {hash} is intact"
        )))
    }

    /// Forget that manifest `root`'s file `path` is linked.
    fn drop_link(&self, root: &str, path: &str) {
        let mut record = self.link_record(root);
        record.files.retain(|l| l.path != path);
        let file = self.link_path(root);
        let _ = if record.files.is_empty() {
            fs::remove_file(&file).map_err(anyhow::Error::from)
        } else {
            serde_json::to_vec_pretty(&record)
                .map_err(anyhow::Error::from)
                .and_then(|b| write_atomic(&file, &b))
        };
        let _ = self.reload_links();
    }
}

fn hash_bytes(hash: &str) -> Option<[u8; 32]> {
    if !is_hash(hash) {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hash[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// `len` bytes at `offset` of a linked file, if the file still looks as it was linked.
fn read_at(link: &Link, offset: u64, len: usize) -> Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let file = fs::File::open(&link.target)?;
    let md = file.metadata()?;
    if md.len() != link.size || mtime_ns(&md) != link.mtime_ns {
        bail!("{} changed since it was linked", link.target.display());
    }
    let mut buf = vec![0u8; len];
    file.read_exact_at(&mut buf, offset)?;
    Ok(buf)
}

impl Store {
    fn meta_path(&self, key: &str) -> Result<PathBuf> {
        if !is_meta_key(key) {
            bail!("invalid metadata key {key:?}");
        }
        Ok(self.root.join("meta").join(key))
    }

    /// Small metadata records (Hub model info, file lists), keyed by relative path.
    pub fn put_meta(&self, key: &str, bytes: &[u8]) -> Result<()> {
        write_atomic(&self.meta_path(key)?, bytes)
    }

    /// Names of the metadata records directly under `dir` (a key prefix, without a
    /// trailing `/`).
    pub fn list_meta(&self, dir: &str) -> Vec<String> {
        let Ok(path) = self.meta_path(dir) else {
            return Vec::new();
        };
        let mut out: Vec<String> = fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| is_meta_key(n))
            .collect();
        out.sort();
        out
    }

    pub fn get_meta(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.meta_path(key)?;
        fs::read(&path).with_context(|| format!("no metadata {key}"))
    }

    /// Scratch directory inside the store, on the same filesystem as the chunks.
    pub fn tmp_dir(&self) -> Result<PathBuf> {
        let dir = self.root.join("tmp");
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }
}

/// Refuse a store laid out by a newer chungus, and stamp older or new stores with the
/// current version. A future layout change bumps [`STORE_VERSION`] and migrates here.
fn check_version(root: &Path) -> Result<()> {
    let path = root.join("VERSION");
    match fs::read_to_string(&path) {
        Ok(s) => {
            let v: u32 = s
                .trim()
                .parse()
                .with_context(|| format!("{} is not a store version", path.display()))?;
            if v > STORE_VERSION {
                bail!(
                    "store {} was written by a newer chungus (store v{v}, this one reads up to \
                     v{STORE_VERSION}); upgrade chungus",
                    root.display()
                );
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            write_atomic(&path, format!("{STORE_VERSION}\n").as_bytes())
        }
        Err(e) => Err(e.into()),
    }
}

/// Write via a temp file and rename, so a crash never leaves a truncated file.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    let dir = path.parent().unwrap();
    fs::create_dir_all(dir)?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(bytes)?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_refuses_oversized_lengths() {
        // Found by fuzzing: a claimed length of ~4 GB was allocated before decompressing.
        let blob = encode(b"hello", Dtype::Raw).unwrap();
        assert!(decode(&blob, 4_096_293_148).is_err());
        assert_eq!(decode(&blob, 5).unwrap(), b"hello");
    }

    #[test]
    fn store_version_is_stamped_and_checked() {
        let dir = tempfile::tempdir().unwrap();
        Store::open(dir.path()).unwrap();
        let v = fs::read_to_string(dir.path().join("VERSION")).unwrap();
        assert_eq!(v.trim(), STORE_VERSION.to_string());
        // Reopening a current store is fine; a newer one is refused with a clear message.
        Store::open(dir.path()).unwrap();
        fs::write(dir.path().join("VERSION"), "99\n").unwrap();
        let e = Store::open(dir.path()).err().unwrap().to_string();
        assert!(e.contains("newer chungus"), "{e}");
    }

    #[test]
    fn linked_files_serve_chunks_until_they_change() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("store")).unwrap();
        let mut state = 7u64;
        let data: Vec<u8> = (0..400_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 56) as u8
            })
            .collect();
        let file = dir.path().join("blob");
        fs::write(&file, &data).unwrap();
        let spans = crate::chunk::chunk(&data, &crate::file_segments(&file, &data).unwrap());
        let chunks: Vec<_> = spans
            .iter()
            .map(|s| crate::manifest::ChunkRef {
                hash: blake3::hash(&data[s.start..s.end]).to_hex().to_string(),
                len: (s.end - s.start) as u32,
                dtype: Dtype::Raw,
            })
            .collect();
        let m = Manifest::new(vec![crate::manifest::FileEntry {
            path: "blob".into(),
            size: data.len() as u64,
            hash: blake3::hash(&data).to_hex().to_string(),
            chunks: chunks.clone(),
        }]);
        store.put_manifest(&m).unwrap();
        assert!(!store.contains(&chunks[1].hash));
        // Another process with the same store open.
        let other = Store::open(&dir.path().join("store")).unwrap();
        assert!(!other.contains(&chunks[0].hash));
        store
            .add_links(&m.root, vec![Link::new("blob", &file).unwrap()])
            .unwrap();
        // Nothing was copied, and every chunk reads back from the linked file.
        assert_eq!(
            fs::read_dir(dir.path().join("store/chunks"))
                .unwrap()
                .count(),
            0
        );
        for c in &chunks {
            assert!(store.contains(&c.hash));
            let raw = decode(&store.get(&c.hash).unwrap(), c.len as usize).unwrap();
            assert_eq!(blake3::hash(&raw).to_hex().as_str(), c.hash);
        }
        // Survives reopening, and another open store picks up links added later.
        assert!(other.contains(&chunks[0].hash));
        let store = Store::open(&dir.path().join("store")).unwrap();
        assert!(store.get(&chunks[0].hash).is_ok());
        assert_eq!(store.links(&m.root).len(), 1);

        // Edited in place (same size): the read fails its check and the link is dropped.
        let mut edited = data.clone();
        edited[10] ^= 1;
        fs::write(&file, &edited).unwrap();
        assert!(store.get(&chunks[0].hash).is_err());
        assert!(!store.contains(&chunks[1].hash));
        assert!(store.links(&m.root).is_empty());

        // Deleted: gc sweeps the link.
        fs::write(&file, &data).unwrap();
        store
            .add_links(&m.root, vec![Link::new("blob", &file).unwrap()])
            .unwrap();
        fs::remove_file(&file).unwrap();
        assert_eq!(store.gc_links().unwrap(), 1);
        assert!(!store.contains(&chunks[0].hash));
    }

    #[test]
    fn newer_blob_is_named() {
        let mut blob = encode(b"hello", Dtype::Raw).unwrap();
        assert_eq!(decode(&blob, 5).unwrap(), b"hello");
        blob[0] = BLOB_VERSION + 1;
        assert!(decode(&blob, 5).unwrap_err().to_string().contains("newer"));
    }
}
