//! Publisher signatures on manifests.
//!
//! A publisher signs a manifest's root hash with an ed25519 key. Because the root commits
//! to every file hash, and every file hash to its bytes, one signature covers the whole
//! model. A manifest can carry signatures from several keys (an author, a mirror, an
//! organisation), and anyone can check them without trusting the peer that served them.
//!
//! Keys are shown as `chungus1` followed by 64 hex characters.

use anyhow::{Context, Result, bail};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const DOMAIN: &[u8] = b"chungus/manifest-signature/v1\0";
const KEY_PREFIX: &str = "chungus1";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// The signer's public key, `chungus1<hex>`.
    pub key: String,
    /// ed25519 signature over the domain tag and the manifest root, hex.
    pub sig: String,
}

fn message(root: &str) -> Vec<u8> {
    [DOMAIN, root.as_bytes()].concat()
}

fn from_hex<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

fn to_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn public_key_string(key: &VerifyingKey) -> String {
    format!("{KEY_PREFIX}{}", to_hex(key.as_bytes()))
}

pub fn parse_public_key(s: &str) -> Result<VerifyingKey> {
    let hex = s
        .trim()
        .strip_prefix(KEY_PREFIX)
        .with_context(|| format!("{s:?} is not a chungus public key"))?;
    let bytes = from_hex::<32>(hex).context("public key must be 64 hex characters")?;
    Ok(VerifyingKey::from_bytes(&bytes)?)
}

/// Create a new signing key at `path` (readable only by the owner) and return it.
pub fn generate_key(path: &Path) -> Result<SigningKey> {
    if path.exists() {
        bail!(
            "{} already exists; refusing to overwrite a key",
            path.display()
        );
    }
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).map_err(|e| anyhow::anyhow!("no randomness: {e}"))?;
    let key = SigningKey::from_bytes(&seed);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    use std::io::Write;
    opts.open(path)
        .with_context(|| format!("create {}", path.display()))?
        .write_all(to_hex(&seed).as_bytes())?;
    Ok(key)
}

pub fn load_key(path: &Path) -> Result<SigningKey> {
    let text = fs::read_to_string(path).with_context(|| {
        format!(
            "read signing key {} (create one with `chungus keygen`)",
            path.display()
        )
    })?;
    let seed = from_hex::<32>(text.trim()).context("signing key file is corrupt")?;
    Ok(SigningKey::from_bytes(&seed))
}

pub fn sign(key: &SigningKey, root: &str) -> Signature {
    Signature {
        key: public_key_string(&key.verifying_key()),
        sig: to_hex(&key.sign(&message(root)).to_bytes()),
    }
}

/// True if `s` is a valid signature of `root` by the key it names.
pub fn verify(s: &Signature, root: &str) -> bool {
    let Ok(key) = parse_public_key(&s.key) else {
        return false;
    };
    let Some(bytes) = from_hex::<64>(&s.sig) else {
        return false;
    };
    let sig = ed25519_dalek::Signature::from_bytes(&bytes);
    key.verify_strict(&message(root), &sig).is_ok()
}

/// True if any of `sigs` is a valid signature of `root` by one of `trusted` keys.
pub fn trusted_by(sigs: &[Signature], root: &str, trusted: &[VerifyingKey]) -> bool {
    sigs.iter()
        .any(|s| parse_public_key(&s.key).is_ok_and(|k| trusted.contains(&k)) && verify(s, root))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_and_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let key = generate_key(&dir.path().join("key")).unwrap();
        let root = "ab".repeat(32);
        let s = sign(&key, &root);
        assert!(verify(&s, &root));
        assert!(!verify(&s, &"cd".repeat(32)));
        let other = generate_key(&dir.path().join("other")).unwrap();
        assert!(trusted_by(
            std::slice::from_ref(&s),
            &root,
            &[key.verifying_key()]
        ));
        assert!(!trusted_by(
            std::slice::from_ref(&s),
            &root,
            &[other.verifying_key()]
        ));
        // A signature re-labelled with another key doesn't verify.
        let forged = Signature {
            key: public_key_string(&other.verifying_key()),
            ..s
        };
        assert!(!verify(&forged, &root));
        // Round trips through its string form and its key file.
        let pk = public_key_string(&key.verifying_key());
        assert_eq!(parse_public_key(&pk).unwrap(), key.verifying_key());
        assert_eq!(
            load_key(&dir.path().join("key")).unwrap().to_bytes(),
            key.to_bytes()
        );
        assert!(generate_key(&dir.path().join("key")).is_err());
    }
}
