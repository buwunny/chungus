//! A manifest lists every file in a model and the chunks that rebuild it.

use serde::{Deserialize, Serialize};

use crate::segment::Dtype;

pub const FORMAT: &str = "chungus/manifest/v1";
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
        let root = compute_root(&files);
        Manifest {
            format: FORMAT.into(),
            files,
            root,
        }
    }

    pub fn verify_root(&self) -> bool {
        compute_root(&self.files) == self.root
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

fn compute_root(files: &[FileEntry]) -> String {
    let mut h = blake3::Hasher::new();
    for f in files {
        // Each chunk hash is covered by the file hash, so the root commits to them too.
        h.update(f.path.as_bytes());
        h.update(&[0]);
        h.update(&f.size.to_le_bytes());
        h.update(f.hash.as_bytes());
    }
    h.finalize().to_hex().to_string()
}
