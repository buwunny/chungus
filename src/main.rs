//! Entry point: digests gzip layer blobs passed on the command line.
//!
//! Usage: chungus <layer.tar.gz>...

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use chungus::hashing::{self, LayerDigests};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        println!("  compressed: {} ({} bytes)", d.compressed, d.compressed_size);
        println!("  diff_id:    {} ({} bytes)", d.diff_id, d.uncompressed_size);
    }

    Ok(())
}
