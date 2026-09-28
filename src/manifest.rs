//! A manifest lists every file in a model and the chunks that rebuild it.

use serde::{Deserialize, Serialize};

use crate::segment::Dtype;

pub const FORMAT: &str = "chungus/manifest/v1";

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
