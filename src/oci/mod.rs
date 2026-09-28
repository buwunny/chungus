//! Reading OCI image layouts (the on-disk format `skopeo copy ... oci:<dir>` writes).
mod layout;
mod manifest;

pub use layout::Layout;
pub use manifest::{Descriptor, ImageConfig, Index, Manifest, RootFs};
