use std::io::{self, Read};

use sha2::{Digest, Sha256};

use super::Sha256Digest;

/// A `Read` adapter that SHA-256 hashes and counts every byte passing through it.
pub struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
    bytes: u64,
}

impl<R: Read> HashingReader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            bytes: 0,
        }
    }

    /// Consumes the reader, returning the inner reader, digest, and byte count.
    pub fn finish(self) -> (R, Sha256Digest, u64) {
        let digest = Sha256Digest(self.hasher.finalize().into());
        (self.inner, digest, self.bytes)
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
}
