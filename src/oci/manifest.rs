use std::collections::HashMap;

use serde::Deserialize;

/// A pointer to a blob: what kind it is, its digest, and its size.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Descriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

impl Descriptor {
    pub fn ref_name(&self) -> Option<&str> {
        self.annotations
            .get("org.opencontainers.image.ref.name")
            .map(|s| s.as_str())
    }
}

/// `index.json` at the root of the layout.
#[derive(Debug, Deserialize)]
pub struct Index {
    pub manifests: Vec<Descriptor>,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
}

#[derive(Debug, Deserialize)]
pub struct ImageConfig {
    pub rootfs: RootFs,
}

#[derive(Debug, Deserialize)]
pub struct RootFs {
    pub diff_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_manifest() {
        let json = r#"{
            "config": {"mediaType": "x", "digest": "sha256:aa", "size": 1},
            "layers": [{"mediaType": "y", "digest": "sha256:bb", "size": 2}]
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.layers.len(), 1);
        assert_eq!(m.layers[0].size, 2);
    }
}
