use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

use chungus::manifest::Manifest;
use chungus::store::Store;

#[derive(Parser)]
#[command(version, about = "Chunk, compress and deduplicate model files")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Pack a model file or directory into a chunk store and write its manifest.
    Pack {
        input: PathBuf,
        /// Chunk store directory (shared across models, so repeats dedup).
        #[arg(long, default_value = ".chungus/store")]
        store: PathBuf,
        /// Where to write the manifest JSON.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Rebuild the files in a manifest, verifying every chunk.
    Unpack {
        manifest: PathBuf,
        #[arg(long, default_value = ".chungus/store")]
        store: PathBuf,
        /// Output directory.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Report compression and dedup ratios for one or more models, without writing.
    Bench {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Pack {
            input,
            store,
            output,
        } => {
            let store = Store::open(&store)?;
            let (manifest, s) = chungus::pack(&input, &store)?;
            fs::write(&output, serde_json::to_vec_pretty(&manifest)?)
                .with_context(|| format!("write {}", output.display()))?;
            println!("packed {:.1} MB in {} chunks", mb(s.raw_bytes), s.chunks);
            println!(
                "new: {} chunks, {:.1} MB raw -> {:.1} MB stored ({:.1}%)",
                s.new_chunks,
                mb(s.new_raw_bytes),
                mb(s.new_stored_bytes),
                pct(s.new_stored_bytes, s.new_raw_bytes)
            );
            println!("root {}", manifest.root);
        }
        Cmd::Unpack {
            manifest,
            store,
            output,
        } => {
            let m: Manifest = serde_json::from_slice(
                &fs::read(&manifest).with_context(|| format!("read {}", manifest.display()))?,
            )?;
            chungus::unpack(&m, &Store::open(&store)?, &output)?;
            println!("unpacked {} files, all chunks verified", m.files.len());
        }
        Cmd::Bench { inputs } => {
            let r = chungus::bench(&inputs)?;
            let row = |name: &str, bytes: u64| {
                println!(
                    "{name:<28} {:>12.1} MB  {:>6.1}%",
                    mb(bytes),
                    pct(bytes, r.raw_bytes)
                );
            };
            row("raw", r.raw_bytes);
            row("zstd only", r.zstd_bytes);
            row("float transform + zstd", r.encoded_bytes);
            row("+ dedup (full pipeline)", r.dedup_bytes);
            println!(
                "chunks: {} total, {} unique ({:.1}% of raw bytes unique)",
                r.chunks,
                r.unique_chunks,
                pct(r.unique_raw_bytes, r.raw_bytes)
            );
            println!(
                "encode {:.0} MB/s, decode {:.0} MB/s (raw bytes, all cores)",
                mb(r.raw_bytes) / r.encode_secs.max(1e-9),
                mb(r.raw_bytes) / r.decode_secs.max(1e-9)
            );
        }
    }
    Ok(())
}
