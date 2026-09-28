//! A manifest lists every file in a model and the chunks that rebuild it.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::segment::Dtype;

/// The current format. Its root also commits to every file's chunk list, so a single
/// chunk can be trusted before the rest of its file arrives (see [`crate::lazy`]).
pub const FORMAT: &str = "chungus/manifest/v2";
/// The first format, whose root covers only whole-file hashes. Still read and verified,
/// but its files can only be trusted once complete.
pub const FORMAT_V1: &str = "chungus/manifest/v1";
/// Every format this version reads, newest first. See docs/formats.md for the policy.
pub const KNOWN_FORMATS: &[&str] = &[FORMAT, FORMAT_V1];
/// Raw bytes per block, the unit announced on the DHT. Every node in a swarm must agree on
/// it, since block ids depend on it.
pub const BLOCK_BYTES: u64 = 64 << 20;

/// A run of a model's unique chunks, about [`BLOCK_BYTES`] of raw data. Nodes announce the
/// blocks they hold in full, so a peer that has only part of a model can still serve it.
pub struct Block {
    pub id: String,
    pub chunks: Vec<ChunkRef>,
}

#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub files: Vec<FileEntry>,
    /// BLAKE3 over every file's path, size and hash, in order. This is the one value an
    /// author signs; everything else is checked against it transitively.
    pub root: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    /// BLAKE3 of the whole file.
    pub hash: String,
    pub chunks: Vec<ChunkRef>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChunkRef {
    pub hash: String,
    pub len: u32,
    pub dtype: Dtype,
}

impl Manifest {
    pub fn new(files: Vec<FileEntry>) -> Self {
        let root = compute_root(FORMAT, &files).expect("current format");
        Manifest {
            format: FORMAT.into(),
            files,
            root,
        }
    }

    pub fn verify_root(&self) -> bool {
        compute_root(&self.format, &self.files).is_some_and(|r| r == self.root)
    }

    /// Whether the root commits to each chunk, so chunks can be used as they arrive.
    pub fn commits_to_chunks(&self) -> bool {
        self.format == FORMAT
    }

    /// The model's unique chunks in order of first use, cut into blocks of at least `size`
    /// raw bytes (the last may be smaller). A block's id hashes its chunk hashes, so it is
    /// the same in every manifest that contains the same run of chunks.
    pub fn blocks(&self, size: u64) -> Vec<Block> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        let mut chunks = Vec::new();
        let mut bytes = 0u64;
        let finish = |chunks: Vec<ChunkRef>, out: &mut Vec<Block>| {
            let mut h = blake3::Hasher::new();
            h.update(b"chungus/block/v1\0");
            for c in &chunks {
                h.update(c.hash.as_bytes());
            }
            out.push(Block {
                id: h.finalize().to_hex().to_string(),
                chunks,
            });
        };
        for c in self.files.iter().flat_map(|f| &f.chunks) {
            if !seen.insert(&c.hash) {
                continue;
            }
            bytes += c.len as u64;
            chunks.push(c.clone());
            if bytes >= size {
                finish(std::mem::take(&mut chunks), &mut out);
                bytes = 0;
            }
        }
        if !chunks.is_empty() {
            finish(chunks, &mut out);
        }
        out
    }
}

/// Parse a manifest, saying plainly when it was written by a newer chungus rather than
/// failing on a field or root this version doesn't understand.
pub fn parse(bytes: &[u8]) -> Result<Manifest> {
    #[derive(Deserialize)]
    struct Head {
        format: String,
    }
    let head: Head = serde_json::from_slice(bytes).context("not a chungus manifest")?;
    if !KNOWN_FORMATS.contains(&head.format.as_str()) {
        if head.format.starts_with("chungus/manifest/") {
            bail!(
                "manifest format {} is newer than this chungus reads ({}); upgrade chungus",
                head.format,
                KNOWN_FORMATS.join(", ")
            );
        }
        bail!("not a chungus manifest (format {:?})", head.format);
    }
    serde_json::from_slice(bytes).context("malformed manifest")
}

/// The root of `files` in `format`, or None for a format this version doesn't know.
fn compute_root(format: &str, files: &[FileEntry]) -> Option<String> {
    let chunks = match format {
        FORMAT => true,
        FORMAT_V1 => false,
        _ => return None,
    };
    let mut h = blake3::Hasher::new();
    if chunks {
        h.update(b"chungus/manifest/v2\0");
    }
    for f in files {
        h.update(f.path.as_bytes());
        h.update(&[0]);
        h.update(&f.size.to_le_bytes());
        h.update(f.hash.as_bytes());
        // v1 relied on the file hash to cover its chunks, which holds only once the
        // whole file is downloaded. v2 commits to each chunk directly.
        if chunks {
            h.update(&(f.chunks.len() as u64).to_le_bytes());
            for c in &f.chunks {
                h.update(c.hash.as_bytes());
                h.update(&c.len.to_le_bytes());
            }
        }
    }
    Some(h.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_names_newer_formats() {
        let m = Manifest::new(Vec::new());
        let bytes = serde_json::to_vec(&m).unwrap();
        assert_eq!(parse(&bytes).unwrap().root, m.root);

        let newer = String::from_utf8(bytes)
            .unwrap()
            .replace(FORMAT, "chungus/manifest/v9");
        let e = parse(newer.as_bytes()).err().unwrap().to_string();
        assert!(e.contains("newer") && e.contains("upgrade"), "{e}");

        let other = br#"{"format":"something/else","files":[],"root":""}"#;
        assert!(
            parse(other)
                .err()
                .unwrap()
                .to_string()
                .contains("not a chungus manifest")
        );
    }
}
