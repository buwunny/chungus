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

use flate2::read::MultiGzDecoder;

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
    let mut uncompressed = HashingReader::new(MultiGzDecoder::new(compressed));

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tar() -> Vec<u8> {
        let data = b"hello chungus\n";
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);

        let mut builder = tar::Builder::new(Vec::new());
        builder
            .append_data(&mut header, "hello.txt", &data[..])
            .unwrap();
        builder.into_inner().unwrap()
    }

    fn gzip_tar(tar_bytes: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(tar_bytes).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn compare_digests_simple() {
        use sha2::{Digest, Sha256};

        let tar_bytes = make_tar();
        let gz = gzip_tar(&tar_bytes);

        let expected_tar: [u8; 32] = Sha256::digest(&tar_bytes).into();
        let expected_gz: [u8; 32] = Sha256::digest(&gz).into();

        let tar_digests = digest_gzip_layer(&gz[..]).unwrap();
        assert_eq!(tar_digests.compressed.as_bytes(), &expected_gz);
        assert_eq!(tar_digests.diff_id.as_bytes(), &expected_tar);
        assert_eq!(tar_digests.compressed_size, gz.len() as u64);
        assert_eq!(tar_digests.uncompressed_size, tar_bytes.len() as u64);
    }

    #[test]
    fn compare_digests_multi() {
        use sha2::{Digest, Sha256};
        // make a tar
        let tar_bytes = make_tar();

        // split it in half, gzip each half, and concatenate the gzipped halves
        let (tar_first_half, tar_second_half) = tar_bytes.split_at(tar_bytes.len() / 2);
        let gz_first_half = gzip_tar(tar_first_half);
        let gz_second_half = gzip_tar(tar_second_half);
        let gz_concat = [gz_first_half.as_slice(), gz_second_half.as_slice()].concat();

        let expected_tar: [u8; 32] = Sha256::digest(&tar_bytes).into();
        let expected_gz: [u8; 32] = Sha256::digest(&gz_concat).into();

        // digest the concatenated gzipped tar
        let d = digest_gzip_layer(&gz_concat[..]).unwrap();
        // test that the digests and sizes match what we expect
        assert_eq!(d.diff_id.as_bytes(), &expected_tar);
        assert_eq!(d.uncompressed_size, tar_bytes.len() as u64);
        assert_eq!(d.compressed.as_bytes(), &expected_gz);
        assert_eq!(d.compressed_size, gz_concat.len() as u64);
    }
}
