//! chungus: content-defined chunking + lossless float transform + zstd, over a
//! content-addressed store.
//!
//! Pipeline for every file: split into segments (per tensor for safetensors and GGUF),
//! cut each segment with FastCDC, hash each raw chunk with BLAKE3, then store the chunk
//! once, encoded with the smallest of {stored, zstd, byte-plane + zstd}.

pub mod chunk;
pub mod downloads;
pub mod gguf;
pub mod hub;
pub mod lazy;
pub mod limits;
pub mod manifest;
#[cfg(target_os = "linux")]
pub mod mount;
pub mod net;
pub mod ollama;
pub mod p2p;
pub mod progress;
pub mod registry;
pub mod safetensors;
pub mod safety;
pub mod segment;
pub mod sign;
pub mod store;
pub mod transform;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use manifest::{ChunkRef, FileEntry, Manifest};
use segment::{Dtype, Segment};
use store::Store;

/// Segments for one file: per tensor for safetensors and GGUF, a single raw segment
/// otherwise, and for a GGUF file chungus can't parse.
pub fn file_segments(path: &Path, data: &[u8]) -> Result<Vec<Segment>> {
    let segs = match path.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => safetensors::segments(data)?,
        Some("gguf") => gguf::segments(data)?,
        _ => None,
    };
    if let Some(segs) = segs {
        return Ok(segs);
    }
    Ok(vec![Segment {
        start: 0,
        end: data.len() as u64,
        dtype: Dtype::Raw,
    }])
}

/// Every regular file under `input` (or `input` itself), with its path relative to `input`.
pub fn list_files(input: &Path) -> Result<Vec<(PathBuf, String)>> {
    if input.is_file() {
        let name = input.file_name().context("input has no file name")?;
        return Ok(vec![(
            input.to_path_buf(),
            name.to_string_lossy().into_owned(),
        )]);
    }
    let mut out = Vec::new();
    let mut stack = vec![input.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                let rel = path
                    .strip_prefix(input)?
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((path, rel));
            }
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(out)
}

fn read(path: &Path) -> Result<memmap2::Mmap> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    // Safety: we only read the mapping, and the tool documents that inputs must not be
    // modified while packing.
    unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))
}

#[derive(Default, Debug)]
pub struct PackStats {
    pub raw_bytes: u64,
    pub chunks: u64,
    pub new_chunks: u64,
    pub new_raw_bytes: u64,
    pub new_stored_bytes: u64,
    /// Files left out because they can run code when loaded, with the reason.
    pub skipped: Vec<String>,
}

/// Pack a file or directory into `store` and return its manifest.
pub fn pack(input: &Path, store: &Store) -> Result<(Manifest, PackStats)> {
    let mut stats = PackStats::default();
    let mut files = Vec::new();
    for (path, rel) in list_files(input)? {
        if let Some(why) = safety::refusal(&rel) {
            stats.skipped.push(why);
            continue;
        }
        let data = read(&path)?;
        let spans = chunk::chunk(&data, &file_segments(&path, &data)?);
        let new_chunks = AtomicU64::new(0);
        let new_raw = AtomicU64::new(0);
        let new_stored = AtomicU64::new(0);
        let chunks = spans
            .par_iter()
            .map(|s| -> Result<ChunkRef> {
                let raw = &data[s.start..s.end];
                let hash = blake3::hash(raw).to_hex().to_string();
                if !store.contains(&hash) {
                    let blob = store::encode(raw, s.dtype)?;
                    if store.put(&hash, &blob)? {
                        new_chunks.fetch_add(1, Ordering::Relaxed);
                        new_raw.fetch_add(raw.len() as u64, Ordering::Relaxed);
                        new_stored.fetch_add(blob.len() as u64, Ordering::Relaxed);
                    }
                }
                Ok(ChunkRef {
                    hash,
                    len: raw.len() as u32,
                    dtype: s.dtype,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        stats.raw_bytes += data.len() as u64;
        stats.chunks += chunks.len() as u64;
        stats.new_chunks += new_chunks.into_inner();
        stats.new_raw_bytes += new_raw.into_inner();
        stats.new_stored_bytes += new_stored.into_inner();
        files.push(FileEntry {
            path: rel,
            size: data.len() as u64,
            hash: blake3::hash(&data).to_hex().to_string(),
            chunks,
        });
    }
    if files.is_empty() && !stats.skipped.is_empty() {
        bail!(
            "nothing left to pack:\n  {}\nconvert the weights to safetensors first",
            stats.skipped.join("\n  ")
        );
    }
    Ok((Manifest::new(files), stats))
}

/// Rebuild every file in `manifest` under `out`, verifying each chunk and each file.
/// The header JSON of every safetensors file in `manifest`, read from the leading chunks
/// in `store`, keyed by path. The registry checks each against the manifest's chunk
/// hashes and counts the model's parameters from them.
pub fn safetensors_headers(
    manifest: &Manifest,
    store: &Store,
) -> Result<std::collections::BTreeMap<String, String>> {
    let mut out = std::collections::BTreeMap::new();
    for f in manifest
        .files
        .iter()
        .filter(|f| f.path.ends_with(".safetensors"))
    {
        let mut data = Vec::new();
        let mut json = None;
        for c in &f.chunks {
            data.extend(store::decode(&store.get(&c.hash)?, c.len as usize)?);
            if let Some(j) = safetensors::header_json(&data)? {
                json = Some(j.to_vec());
                break;
            }
        }
        // A file too short or too odd to be safetensors has no header to send.
        if let Some(json) = json.and_then(|j| String::from_utf8(j).ok()) {
            out.insert(f.path.clone(), json);
        }
    }
    Ok(out)
}

/// The header of every GGUF file in `manifest` through its tensor info and padding (see
/// [`gguf::header_len`]), read from the leading chunks in `store`, hex-encoded and keyed
/// by path, for the registry to check and count parameters from as it does
/// [`safetensors_headers`]. Files whose header isn't in chunks of its own (packed by a
/// chungus that didn't split GGUF) or that chungus can't parse are left out, as are all of
/// them if together they would be more than a publish carries.
pub fn gguf_headers(
    manifest: &Manifest,
    store: &Store,
) -> Result<std::collections::BTreeMap<String, String>> {
    let mut out = std::collections::BTreeMap::new();
    let mut total = 0;
    for f in manifest.files.iter().filter(|f| f.path.ends_with(".gguf")) {
        let mut data = Vec::new();
        for c in &f.chunks {
            data.extend(store::decode(&store.get(&c.hash)?, c.len as usize)?);
            match gguf::header_len(&data) {
                Ok(None) => continue,
                Ok(Some(n)) if n == data.len() => {
                    total += 2 * n;
                    out.insert(f.path.clone(), registry::to_hex(&data));
                }
                Ok(Some(_)) | Err(_) => {}
            }
            break;
        }
    }
    if total > registry::MAX_HEADERS / 2 {
        out.clear();
    }
    Ok(out)
}

pub fn unpack(manifest: &Manifest, store: &Store, out: &Path) -> Result<()> {
    if !manifest.verify_root() {
        bail!("manifest root does not match its contents");
    }
    safety::check_manifest(manifest)?;
    for f in &manifest.files {
        if f.path.split('/').any(|c| c == ".." || c.is_empty()) || f.path.starts_with('/') {
            bail!("unsafe path in manifest: {}", f.path);
        }
        let parts = f
            .chunks
            .par_iter()
            .map(|c| -> Result<Vec<u8>> {
                let raw = store::decode(&store.get(&c.hash)?, c.len as usize)?;
                if blake3::hash(&raw).to_hex().as_str() != c.hash {
                    bail!("chunk {} failed verification", c.hash);
                }
                Ok(raw)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut hasher = blake3::Hasher::new();
        parts.iter().for_each(|p| {
            hasher.update(p);
        });
        if hasher.finalize().to_hex().as_str() != f.hash {
            bail!("{} failed whole-file verification", f.path);
        }
        let dest = out.join(&f.path);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&dest, parts.concat()).with_context(|| format!("write {}", dest.display()))?;
    }
    Ok(())
}

#[derive(Default, Debug, serde::Serialize)]
pub struct BenchReport {
    pub raw_bytes: u64,
    /// zstd on every chunk, no transform, no dedup: the baseline.
    pub zstd_bytes: u64,
    /// Best encoding per chunk, no dedup: what the float transform alone buys.
    pub encoded_bytes: u64,
    /// Best encoding over unique chunks only: the full pipeline.
    pub dedup_bytes: u64,
    pub unique_raw_bytes: u64,
    pub chunks: u64,
    pub unique_chunks: u64,
    /// Wall time of FastCDC over every file.
    pub chunk_secs: f64,
    /// Wall time of BLAKE3 plus choosing and writing the best encoding, as `pack` does.
    pub encode_secs: f64,
    pub decode_secs: f64,
    /// One row per input, in order. Each counts only chunks no earlier input had, so the
    /// second of two model versions shows what downloading it after the first would cost.
    pub inputs: Vec<InputReport>,
    /// Totals per tensor dtype ("raw" is headers, non-safetensors files and other types).
    pub by_dtype: std::collections::BTreeMap<String, DtypeReport>,
}

#[derive(Default, Debug, serde::Serialize)]
pub struct InputReport {
    pub path: String,
    pub raw_bytes: u64,
    /// Raw bytes in chunks that no earlier input (or earlier part of this one) had.
    pub new_raw_bytes: u64,
    /// Encoded size of those new chunks: what fetching this input would transfer.
    pub new_stored_bytes: u64,
}

#[derive(Default, Debug, serde::Serialize)]
pub struct DtypeReport {
    pub raw_bytes: u64,
    pub zstd_bytes: u64,
    pub encoded_bytes: u64,
}

/// Measure the pipeline on inputs without writing a store. With `whole_files`, every
/// file is chunked as one raw segment, as if chungus didn't read weight formats, to show
/// what splitting per tensor adds.
pub fn bench(inputs: &[PathBuf], whole_files: bool) -> Result<BenchReport> {
    let mut r = BenchReport::default();
    let mut seen = HashSet::new();
    for input in inputs {
        let mut row = InputReport {
            path: input.display().to_string(),
            ..Default::default()
        };
        for (path, _) in list_files(input)? {
            let data = read(&path)?;
            let segments = if whole_files {
                vec![Segment {
                    start: 0,
                    end: data.len() as u64,
                    dtype: Dtype::Raw,
                }]
            } else {
                file_segments(&path, &data)?
            };
            let t = std::time::Instant::now();
            let spans = chunk::chunk(&data, &segments);
            r.chunk_secs += t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            let rows = spans
                .par_iter()
                .map(|s| -> Result<_> {
                    let raw = &data[s.start..s.end];
                    let hash = *blake3::hash(raw).as_bytes();
                    Ok((hash, s, store::encode(raw, s.dtype)?))
                })
                .collect::<Result<Vec<_>>>()?;
            r.encode_secs += t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            rows.par_iter()
                .map(|(_, s, blob)| store::decode(blob, s.end - s.start).map(|_| ()))
                .collect::<Result<()>>()?;
            r.decode_secs += t.elapsed().as_secs_f64();

            // The baseline, outside the timed passes.
            let zstd = spans
                .par_iter()
                .map(|s| {
                    Ok(
                        zstd::bulk::compress(&data[s.start..s.end], store::ZSTD_LEVEL)?.len()
                            as u64,
                    )
                })
                .collect::<Result<Vec<_>>>()?;

            r.raw_bytes += data.len() as u64;
            row.raw_bytes += data.len() as u64;
            for ((hash, s, blob), zstd) in rows.into_iter().zip(zstd) {
                let len = (s.end - s.start) as u64;
                let d = r
                    .by_dtype
                    .entry(format!("{:?}", s.dtype).to_lowercase())
                    .or_default();
                d.raw_bytes += len;
                d.zstd_bytes += zstd;
                d.encoded_bytes += blob.len() as u64;
                r.chunks += 1;
                r.zstd_bytes += zstd;
                r.encoded_bytes += blob.len() as u64;
                if seen.insert(hash) {
                    r.unique_chunks += 1;
                    r.unique_raw_bytes += len;
                    r.dedup_bytes += blob.len() as u64;
                    row.new_raw_bytes += len;
                    row.new_stored_bytes += blob.len() as u64;
                }
            }
        }
        r.inputs.push(row);
    }
    Ok(r)
}
