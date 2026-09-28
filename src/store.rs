//! Content-addressed chunk store on local disk.
//!
//! Each chunk is stored once, at `<root>/<hh>/<hash>`, where the hash is BLAKE3 of the
//! chunk's *raw* bytes. Addressing by raw content means the codec can change without
//! breaking any hash, and a reader verifies exactly the bytes it will use.
//!
//! Blob layout: `[version=1][codec][param][payload...]`, where `param` is the element
//! width for `PlaneZstd` and the float kind for `ExponentZstd`.

use anyhow::{Context, Result, bail};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

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
}

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
        Ok(Store {
            root: root.to_path_buf(),
            blocked: Default::default(),
            gates: Default::default(),
        })
    }

    /// Replace the blocklist, and delete any blocked chunks and manifests already on disk.
    /// Returns how many were deleted.
    pub fn set_blocked(&self, hashes: HashSet<String>) -> usize {
        let mut removed = 0;
        for h in hashes.iter().filter(|h| is_hash(h)) {
            for path in [self.path(h), self.manifest_path(h)] {
                if fs::remove_file(path).is_ok() {
                    removed += 1;
                }
            }
        }
        *self.blocked.write().unwrap() = hashes;
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
        is_hash(hash) && !self.is_blocked(hash) && self.path(hash).exists()
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
        fs::read(self.path(hash)).with_context(|| format!("missing chunk {hash}"))
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
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
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
    fn newer_blob_is_named() {
        let mut blob = encode(b"hello", Dtype::Raw).unwrap();
        assert_eq!(decode(&blob, 5).unwrap(), b"hello");
        blob[0] = BLOB_VERSION + 1;
        assert!(decode(&blob, 5).unwrap_err().to_string().contains("newer"));
    }
}
