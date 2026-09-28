use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;

use super::{Descriptor, ImageConfig, Index, Manifest};

/// An OCI image layout directory on disk.
pub struct Layout {
    root: PathBuf,
}

impl Layout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `sha256:abcd…` → `<root>/blobs/sha256/abcd…`
    pub fn blob_path(&self, digest: &str) -> io::Result<PathBuf> {
        let (algo, hex) = digest.split_once(':').ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad digest: {digest}"))
        })?;
        Ok(self.root.join("blobs").join(algo).join(hex))
    }

    pub fn index(&self) -> io::Result<Index> {
        read_json(&self.root.join("index.json"))
    }

    pub fn manifest(&self, desc: &Descriptor) -> io::Result<Manifest> {
        let path = self.blob_path(&desc.digest)?;
        read_json(&path)
    }

    pub fn config(&self, desc: &Descriptor) -> io::Result<ImageConfig> {
        let path = self.blob_path(&desc.digest)?;
        read_json(&path)
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> io::Result<T> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}
