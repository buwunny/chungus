//! Hashing engine: computes the OCI digests of a gzip-compressed layer.
//!
//! For each layer we need two digests:
//! * `compressed` — SHA-256 of the blob as stored in the registry (the manifest digest).
//! * `diff_id`    — SHA-256 of the uncompressed tar (`rootfs.diff_ids` in the image config).

mod digest;
mod reader;

pub use digest::Sha256Digest;
pub use reader::HashingReader;

use std::io::{self, Read};

use flate2::read::GzDecoder;

/// Digests and sizes for a single layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerDigests {
    pub compressed: Sha256Digest,
    pub compressed_size: u64,
    pub diff_id: Sha256Digest,
    pub uncompressed_size: u64,
}

/// Streams a gzip layer once, hashing both the compressed and uncompressed bytes.
///
/// Blocking: call from `tokio::task::spawn_blocking` in async contexts.
pub fn digest_gzip_layer<R: Read>(layer: R) -> io::Result<LayerDigests> {
    let compressed = HashingReader::new(layer);
    let mut uncompressed = HashingReader::new(GzDecoder::new(compressed));

    io::copy(&mut uncompressed, &mut io::sink())?;

    let (gz, diff_id, uncompressed_size) = uncompressed.finish();
    let (_, compressed, compressed_size) = gz.into_inner().finish();

    Ok(LayerDigests {
        compressed,
        compressed_size,
        diff_id,
        uncompressed_size,
    })
}
