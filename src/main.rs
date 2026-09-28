//! Entry point: digests gzip layer blobs passed on the command line.
//!
//! Usage: chungus <layer.tar.gz>...

use chungus::oci::Descriptor;
use chungus::oci::Layout;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use chungus::hashing::{self, LayerDigests};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [cmd, dir] = args.as_slice()
        && cmd == "verify"
    {
        let ok = verify(Path::new(dir))?;
        std::process::exit(if ok { 0 } else { 1 });
    }

    let paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!("usage: chungus <layer.tar.gz>...");
        std::process::exit(2);
    }

    let tasks: Vec<_> = paths
        .into_iter()
        .map(|path| {
            tokio::task::spawn_blocking(move || {
                let file = BufReader::new(File::open(&path)?);
                hashing::digest_gzip_layer(file).map(|d| (path, d))
            })
        })
        .collect();

    for task in tasks {
        let (path, d): (PathBuf, LayerDigests) = task.await??;
        println!("{}", path.display());
        println!(
            "  compressed: {} ({} bytes)",
            d.compressed, d.compressed_size
        );
        println!(
            "  diff_id:    {} ({} bytes)",
            d.diff_id, d.uncompressed_size
        );
    }

    Ok(())
}

fn verify(dir: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let layout = Layout::new(dir);
    let index = layout.index()?;
    let mut all_ok = true;
    let mut seen = HashSet::new();
    for manifest_desc in &index.manifests {
        println!(
            "checking manifest {}",
            manifest_desc.ref_name().unwrap_or("<untagged>")
        );
        all_ok &= verify_image(&layout, manifest_desc, &mut seen)?;
    }
    Ok(all_ok)
}

fn verify_image(
    layout: &Layout,
    manifest_desc: &Descriptor,
    seen: &mut HashSet<String>,
) -> Result<bool, Box<dyn Error>> {
    let manifest = layout.manifest(manifest_desc)?;
    let config = layout.config(&manifest.config)?;

    if manifest.layers.len() != config.rootfs.diff_ids.len() {
        return Err("layer count doesn't match diff_id count".into());
    }

    let mut all_ok = true;
    for (layer, expected_diff_id) in manifest.layers.iter().zip(&config.rootfs.diff_ids) {
        if !seen.insert(layer.digest.clone()) {
            println!("already verified layer");
            continue;
        }
        let mut layer_ok = true;
        println!("checking layer {}", layer.digest);
        if layer.media_type != "application/vnd.oci.image.layer.v1.tar+gzip"
            && layer.media_type != "application/vnd.docker.image.rootfs.diff.tar.gzip"
        {
            println!("skipped (unsupported {})", layer.media_type);
            continue;
        }
        let file = BufReader::new(File::open(layout.blob_path(&layer.digest)?)?);
        let d = hashing::digest_gzip_layer(file)?;
        if d.compressed.to_string() != layer.digest {
            println!(
                "mismatch (compressed digest): expected {}, got {}",
                layer.digest, d.compressed
            );
            all_ok = false;
            layer_ok = false;
        }
        if d.compressed_size != layer.size {
            println!(
                "mismatch (compressed size): expected {}, got {}",
                layer.size, d.compressed_size
            );
            all_ok = false;
            layer_ok = false;
        }
        if d.diff_id.to_string() != *expected_diff_id {
            println!(
                "mismatch (diff id): expected {}, got {}",
                expected_diff_id, d.diff_id
            );
            all_ok = false;
            layer_ok = false;
        }
        if layer_ok {
            println!("ok");
        }
    }
    Ok(all_ok)
}
