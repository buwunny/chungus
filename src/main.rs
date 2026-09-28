use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chungus::manifest::Manifest;
use chungus::net;
use chungus::store::{self, Store};

#[derive(Parser)]
#[command(version, about = "Chunk, compress, deduplicate and share model files")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

const DEFAULT_STORE: &str = ".chungus/store";

#[derive(Subcommand)]
enum Cmd {
    /// Pack a model file or directory into a chunk store. Prints the manifest root.
    Pack {
        input: PathBuf,
        /// Chunk store directory (shared across models, so repeats dedup).
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Also write the manifest JSON here.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Rebuild the files of a manifest (a root hash or a manifest file), verifying every chunk.
    Unpack {
        manifest: String,
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Output directory.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// List the models in a store.
    List {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
    },
    /// Report compression and dedup ratios for one or more models, without writing.
    Bench {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Share this store with peers on the LAN (read-only HTTP, advertised over mDNS).
    Serve {
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        #[arg(long, default_value_t = net::DEFAULT_PORT)]
        port: u16,
        /// Don't advertise over mDNS; peers must name this node with --peer.
        #[arg(long)]
        no_mdns: bool,
    },
    /// Download a model by manifest root from LAN peers, falling back to an origin.
    Fetch {
        root: String,
        #[arg(long, default_value = DEFAULT_STORE)]
        store: PathBuf,
        /// Also unpack the files into this directory.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// A peer to use in addition to discovered ones, e.g. http://192.168.1.20:7447.
        #[arg(long)]
        peer: Vec<String>,
        /// A server to use only when no peer has a chunk.
        #[arg(long)]
        origin: Option<String>,
        /// Skip mDNS discovery and use only --peer and --origin.
        #[arg(long)]
        no_mdns: bool,
        /// How long to listen for peers on the LAN, in seconds.
        #[arg(long, default_value_t = 2.0)]
        discover_secs: f64,
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

/// A manifest argument is either a root hash in the store or a path to a manifest file.
fn load_manifest(arg: &str, store: &Store) -> Result<Manifest> {
    if store::is_hash(arg) && !Path::new(arg).exists() {
        return store.get_manifest(arg);
    }
    Ok(serde_json::from_slice(
        &fs::read(arg).with_context(|| format!("read {arg}"))?,
    )?)
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Pack {
            input,
            store,
            output,
        } => {
            let store = Store::open(&store)?;
            let (manifest, s) = chungus::pack(&input, &store)?;
            store.put_manifest(&manifest)?;
            if let Some(output) = output {
                fs::write(&output, serde_json::to_vec_pretty(&manifest)?)
                    .with_context(|| format!("write {}", output.display()))?;
            }
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
            let store = Store::open(&store)?;
            let m = load_manifest(&manifest, &store)?;
            chungus::unpack(&m, &store, &output)?;
            println!("unpacked {} files, all chunks verified", m.files.len());
        }
        Cmd::List { store } => {
            let store = Store::open(&store)?;
            for root in store.manifests()? {
                let m = store.get_manifest(&root)?;
                let size: u64 = m.files.iter().map(|f| f.size).sum();
                let names: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
                println!("{root}  {:>10.1} MB  {}", mb(size), names.join(", "));
            }
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
        Cmd::Serve {
            store,
            port,
            no_mdns,
        } => {
            let store = Arc::new(Store::open(&store)?);
            let addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("bind {addr}"))?;
            let _mdns = if no_mdns {
                None
            } else {
                Some(net::advertise(port)?)
            };
            println!(
                "serving {} models on port {port}{}",
                store.manifests()?.len(),
                if no_mdns {
                    ""
                } else {
                    ", advertised on the LAN"
                }
            );
            net::serve_on(listener, store).await?;
        }
        Cmd::Fetch {
            root,
            store,
            output,
            peer,
            origin,
            no_mdns,
            discover_secs,
        } => {
            let store = Arc::new(Store::open(&store)?);
            let mut peers = peer;
            if !no_mdns {
                let wait = Duration::from_secs_f64(discover_secs);
                let found = tokio::task::spawn_blocking(move || net::discover(wait)).await??;
                println!("found {} peer(s) on the LAN", found.len());
                peers.extend(found);
            }
            peers.sort();
            peers.dedup();
            let (manifest, s) = net::fetch(&root, store.clone(), &peers, origin.as_deref()).await?;
            let received: u64 = s.bytes_by_source.values().sum();
            println!(
                "{} chunks: {} already local, {} fetched ({:.1} MB) in {:.1}s",
                s.chunks,
                s.already_local,
                s.chunks - s.already_local,
                mb(received),
                s.secs
            );
            for (source, bytes) in &s.bytes_by_source {
                println!("  {source}: {:.1} MB", mb(*bytes));
            }
            if s.rejected > 0 {
                println!("  rejected {} bad chunk(s) and refetched them", s.rejected);
            }
            if let Some(output) = output {
                tokio::task::spawn_blocking(move || chungus::unpack(&manifest, &store, &output))
                    .await??;
                println!("unpacked, all chunks verified");
            }
        }
    }
    Ok(())
}
