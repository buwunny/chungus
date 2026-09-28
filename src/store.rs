//! Content-addressed chunk store on local disk.
//!
//! Each chunk is stored once, at `<root>/<hh>/<hash>`, where the hash is BLAKE3 of the
//! chunk's *raw* bytes. Addressing by raw content means the codec can change without
//! breaking any hash, and a reader verifies exactly the bytes it will use.
//!
//! Blob layout: `[version=1][codec][param][payload...]`, where `param` is the element
//! width for `PlaneZstd` and the float kind for `ExponentZstd`.

use anyhow::{Context, Result, bail};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::manifest::Manifest;
use crate::segment::Dtype;
use crate::sign::{self, Signature};
use crate::transform::{self, FloatKind};

const VERSION: u8 = 1;
const ZSTD_LEVEL: i32 = 3;

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
    blob.extend_from_slice(&[VERSION, best.0 as u8, best.1]);
    blob.extend_from_slice(&best.2);
    Ok(blob)
}

/// Decode a blob back to the raw chunk. `len` is the expected raw length.
pub fn decode(blob: &[u8], len: usize) -> Result<Vec<u8>> {
    if blob.len() < 3 || blob[0] != VERSION {
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
}

impl Store {
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("chunks"))
            .with_context(|| format!("create store {}", root.display()))?;
        fs::create_dir_all(root.join("manifests"))?;
        Ok(Store {
            root: root.to_path_buf(),
        })
    }

    fn path(&self, hash: &str) -> PathBuf {
        self.root.join("chunks").join(&hash[..2]).join(hash)
    }

    pub fn contains(&self, hash: &str) -> bool {
        is_hash(hash) && self.path(hash).exists()
    }

    /// Write a blob unless it's already present. Returns true if it was new.
    pub fn put(&self, hash: &str, blob: &[u8]) -> Result<bool> {
        if !is_hash(hash) {
            bail!("invalid chunk hash {hash:?}");
        }
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
        fs::read(self.path(hash)).with_context(|| format!("missing chunk {hash}"))
    }

    fn manifest_path(&self, root: &str) -> PathBuf {
        self.root.join("manifests").join(format!("{root}.json"))
    }

    pub fn put_manifest(&self, m: &Manifest) -> Result<()> {
        if !is_hash(&m.root) || !m.verify_root() {
            bail!("refusing to store a manifest whose root doesn't verify");
        }
        write_atomic(&self.manifest_path(&m.root), &serde_json::to_vec_pretty(m)?)
    }

    pub fn get_manifest_bytes(&self, root: &str) -> Result<Vec<u8>> {
        if !is_hash(root) {
            bail!("invalid manifest root {root:?}");
        }
        fs::read(self.manifest_path(root)).with_context(|| format!("no manifest {root}"))
    }

    pub fn get_manifest(&self, root: &str) -> Result<Manifest> {
        Ok(serde_json::from_slice(&self.get_manifest_bytes(root)?)?)
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
            {
                roots.push(root.to_string());
            }
        }
        roots.sort();
        Ok(roots)
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
