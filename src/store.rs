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

use crate::segment::Dtype;
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

pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root).with_context(|| format!("create store {}", root.display()))?;
        Ok(Store {
            root: root.to_path_buf(),
        })
    }

    fn path(&self, hash: &str) -> PathBuf {
        self.root.join(&hash[..2]).join(hash)
    }

    pub fn contains(&self, hash: &str) -> bool {
        self.path(hash).exists()
    }

    /// Write a blob unless it's already present. Returns true if it was new.
    /// Writes go to a temp file first so a crash never leaves a truncated blob.
    pub fn put(&self, hash: &str, blob: &[u8]) -> Result<bool> {
        let path = self.path(hash);
        if path.exists() {
            return Ok(false);
        }
        fs::create_dir_all(path.parent().unwrap())?;
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        fs::File::create(&tmp)?.write_all(blob)?;
        fs::rename(&tmp, &path)?;
        Ok(true)
    }

    pub fn get(&self, hash: &str) -> Result<Vec<u8>> {
        fs::read(self.path(hash)).with_context(|| format!("missing chunk {hash}"))
    }
}
