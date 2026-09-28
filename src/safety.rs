//! Which files chungus is willing to carry.
//!
//! Chunk hashes prove a file arrived intact, not that it is safe to load. Pickle-based
//! weights (`.bin`, `.pt`, `.ckpt`, ...) and a few other formats can run arbitrary code
//! when a framework loads them, so chungus keeps them out of the network entirely:
//! `pack` skips them, stores refuse manifests that list them, nodes don't serve them, the
//! registry won't publish them, and the Hugging Face cache passes them through from
//! huggingface.co without caching or sharing them. Weights travel as safetensors or GGUF.
//! Small metadata files (configs, tokenizers, READMEs) are fine.

use anyhow::{Result, bail};

use crate::manifest::Manifest;

/// Extensions of formats that can execute code on load, lowercase.
const UNSAFE: &[&str] = &[
    // pickle, and the formats built on it (PyTorch, joblib, dill, pandas)
    "bin", "pt", "pth", "ckpt", "pkl", "pickle", "joblib", "dill", "pd",
    // NumPy arrays may hold pickled objects
    "npy", "npz", // Keras and HDF5 models can carry Lambda layers (Python bytecode)
    "h5", "hdf5", "keras",
];

/// Why the file at `path` (as named in a manifest or repo) is refused, or None if it's
/// allowed.
pub fn refusal(path: &str) -> Option<String> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = name.rsplit_once('.')?.1.to_ascii_lowercase();
    UNSAFE.contains(&ext.as_str()).then(|| {
        format!(
            "{path}: .{ext} files can run code when loaded; chungus only carries weights as \
             safetensors or GGUF"
        )
    })
}

pub fn is_allowed(path: &str) -> bool {
    refusal(path).is_none()
}

/// Refuse a manifest that lists any unsafe file.
pub fn check_manifest(m: &Manifest) -> Result<()> {
    if let Some(why) = m.files.iter().find_map(|f| refusal(&f.path)) {
        bail!("refusing manifest {}: {why}", m.root);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pickles_are_refused_and_weights_allowed() {
        for bad in [
            "pytorch_model.bin",
            "sub/dir/model-00001-of-00002.BIN",
            "weights.pt",
            "model.ckpt",
            "optimizer.pth",
            "tf_model.h5",
            "x.npy",
        ] {
            assert!(!is_allowed(bad), "{bad}");
        }
        for good in [
            "model.safetensors",
            "model.Q4_K_M.gguf",
            "config.json",
            "tokenizer.model",
            "README.md",
            "LICENSE",
            ".gitattributes",
        ] {
            assert!(is_allowed(good), "{good}");
        }
    }
}
