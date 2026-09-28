//! Lazy loading: read a model's files before they have finished downloading.
//!
//! A [`Lazy`] model answers reads at any offset of any file. Chunks the store lacks are
//! fetched on demand, each one verified against its hash as it arrives, and a prefetcher
//! fills in the rest in the order a loader is likely to want it: small files (configs,
//! tokenizers) first, then every safetensors header, then tensors layer by layer across
//! all shards. Each read also pulls the chunks just after it to the front of the queue,
//! so sequential readers find their next bytes already local.
//!
//! Everything fetched lands in the store, so once prefetching finishes the model is
//! complete, can be unpacked, and is served to peers like any other.
//!
//! Only manifests whose root commits to every chunk (format v2) can be read lazily: with
//! an older manifest, a chunk list could be swapped without changing the root, and that
//! would only show up once the whole file had been read.

use anyhow::{Context, Result, anyhow, bail};
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::manifest::{ChunkRef, Manifest};
use crate::store::{self, Store};

/// Chunks after a read that jump the prefetch queue.
const READAHEAD: usize = 16;
/// Decoded chunks kept in memory, since readers often make several small reads per chunk.
const CACHE: usize = 64;
/// Largest safetensors header the prefetcher will read to plan its order.
const MAX_HEADER: u64 = 100 << 20;

/// Somewhere to get chunks from: LAN peers ([`crate::net::HttpSource`]) or the swarm
/// ([`crate::p2p::ModelSources`]).
pub trait ChunkSource: Send + Sync {
    /// Fetch chunk `hash` (`len` raw bytes) into `store`, verified. False if no source had
    /// a good copy.
    fn fetch<'a>(&'a self, store: &'a Arc<Store>, hash: &'a str, len: usize)
    -> BoxFuture<'a, bool>;
}

/// Counters for how a lazy model has been filled.
#[derive(Default, Debug)]
pub struct Stats {
    /// Chunks fetched because a read needed them right then.
    pub on_demand: AtomicU64,
    /// Chunks fetched ahead of any read.
    pub prefetched: AtomicU64,
    /// Raw bytes of the model's unique chunks now in the store.
    pub local_bytes: AtomicU64,
}

pub struct Lazy {
    store: Arc<Store>,
    manifest: Manifest,
    source: Arc<dyn ChunkSource>,
    /// For each file, the offset where each of its chunks starts.
    starts: Vec<Vec<u64>>,
    inflight: Mutex<HashMap<String, Shared<BoxFuture<'static, bool>>>>,
    cache: Mutex<VecDeque<(String, Arc<Vec<u8>>)>>,
    urgent: Mutex<VecDeque<ChunkRef>>,
    wake: tokio::sync::Notify,
    pub stats: Stats,
    total_bytes: u64,
}

impl Lazy {
    pub fn new(
        store: Arc<Store>,
        manifest: Manifest,
        source: Arc<dyn ChunkSource>,
    ) -> Result<Arc<Lazy>> {
        if !manifest.verify_root() {
            bail!("manifest root does not match its contents");
        }
        crate::safety::check_manifest(&manifest)?;
        if !manifest.commits_to_chunks() {
            bail!(
                "{} was packed in an older format that can't be read before it is complete; \
                 fetch it in full, or re-pack it",
                manifest.root
            );
        }
        let starts = manifest
            .files
            .iter()
            .map(|f| {
                let mut at = 0;
                f.chunks
                    .iter()
                    .map(|c| {
                        let s = at;
                        at += c.len as u64;
                        s
                    })
                    .collect()
            })
            .collect();
        let mut seen = HashSet::new();
        let (mut total, mut local) = (0, 0);
        for c in manifest.files.iter().flat_map(|f| &f.chunks) {
            if seen.insert(&c.hash) {
                total += c.len as u64;
                if store.contains(&c.hash) {
                    local += c.len as u64;
                }
            }
        }
        let lazy = Lazy {
            store,
            manifest,
            source,
            starts,
            inflight: Default::default(),
            cache: Default::default(),
            urgent: Default::default(),
            wake: Default::default(),
            stats: Stats::default(),
            total_bytes: total,
        };
        lazy.stats.local_bytes.store(local, Ordering::Relaxed);
        Ok(Arc::new(lazy))
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Raw bytes of the model's unique chunks.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Up to `len` bytes of file `file` (an index into the manifest's files) from
    /// `offset`, fetching whatever the store lacks. Shorter only at the end of the file.
    pub async fn read(self: &Arc<Self>, file: usize, offset: u64, len: usize) -> Result<Vec<u8>> {
        let entry = self.manifest.files.get(file).context("no such file")?;
        let end = entry.size.min(offset.saturating_add(len as u64));
        if offset >= end {
            return Ok(Vec::new());
        }
        let starts = &self.starts[file];
        let first = starts.partition_point(|&s| s <= offset) - 1;
        let last = starts.partition_point(|&s| s < end) - 1;
        let chunks = &entry.chunks[first..=last];

        // Queue what comes next before waiting, so it is on its way meanwhile.
        {
            let mut urgent = self.urgent.lock().unwrap();
            let after = entry.chunks.iter().skip(last + 1).take(READAHEAD);
            for c in after.rev() {
                urgent.push_front(c.clone());
            }
            urgent.truncate(4 * READAHEAD);
        }
        self.wake.notify_one();

        let fetched = futures::future::join_all(chunks.iter().map(|c| self.ensure(c, true))).await;
        if let Some((c, _)) = chunks.iter().zip(&fetched).find(|(_, ok)| !**ok) {
            bail!("chunk {} is not available from any source", c.hash);
        }
        let mut out = Vec::with_capacity((end - offset) as usize);
        for (i, c) in chunks.iter().enumerate() {
            let raw = self.decoded(c).await?;
            let start = starts[first + i];
            let from = offset.saturating_sub(start) as usize;
            let to = ((end - start) as usize).min(raw.len());
            out.extend_from_slice(&raw[from..to]);
        }
        Ok(out)
    }

    /// Make sure chunk `c` is in the store, fetching it (once, however many callers ask)
    /// if not. False if no source had it.
    fn ensure(self: &Arc<Self>, c: &ChunkRef, demand: bool) -> BoxFuture<'static, bool> {
        if self.store.contains(&c.hash) {
            return futures::future::ready(true).boxed();
        }
        if self.store.is_blocked(&c.hash) {
            return futures::future::ready(false).boxed();
        }
        let fut = {
            let mut inflight = self.inflight.lock().unwrap();
            inflight
                .entry(c.hash.clone())
                .or_insert_with(|| {
                    let (this, c) = (self.clone(), c.clone());
                    async move {
                        let ok = this
                            .source
                            .fetch(&this.store, &c.hash, c.len as usize)
                            .await;
                        if ok {
                            this.stats
                                .local_bytes
                                .fetch_add(c.len as u64, Ordering::Relaxed);
                            let counter = if demand {
                                &this.stats.on_demand
                            } else {
                                &this.stats.prefetched
                            };
                            counter.fetch_add(1, Ordering::Relaxed);
                        }
                        this.inflight.lock().unwrap().remove(&c.hash);
                        ok
                    }
                    .boxed()
                    .shared()
                })
                .clone()
        };
        fut.boxed()
    }

    /// Chunk `c`'s raw bytes, from the cache or decoded from the store and re-verified.
    async fn decoded(&self, c: &ChunkRef) -> Result<Arc<Vec<u8>>> {
        if let Some((_, raw)) = self
            .cache
            .lock()
            .unwrap()
            .iter()
            .find(|(h, _)| *h == c.hash)
        {
            return Ok(raw.clone());
        }
        let (store, c2) = (self.store.clone(), c.clone());
        let raw = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
            let raw = store::decode(&store.get(&c2.hash)?, c2.len as usize)?;
            if blake3::hash(&raw).to_hex().as_str() != c2.hash {
                bail!("chunk {} failed verification", c2.hash);
            }
            Ok(raw)
        })
        .await??;
        let raw = Arc::new(raw);
        let mut cache = self.cache.lock().unwrap();
        cache.push_back((c.hash.clone(), raw.clone()));
        if cache.len() > CACHE {
            cache.pop_front();
        }
        Ok(raw)
    }

    /// Fetch every chunk the store lacks, `concurrency` at a time, reads' readahead first
    /// and then [`Lazy::prefetch_order`]. Returns once the whole model is local.
    pub async fn prefetch(self: &Arc<Self>, concurrency: usize) -> Result<()> {
        let order = self.prefetch_order().await?;
        let permits = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        let mut next = 0;
        let mut failed = Vec::new();
        loop {
            let permit = permits.clone().acquire_owned().await?;
            let chunk = loop {
                let urgent = self.urgent.lock().unwrap().pop_front();
                let c = match urgent {
                    Some(c) => c,
                    None if next < order.len() => {
                        next += 1;
                        order[next - 1].clone()
                    }
                    None => break None,
                };
                if !self.store.contains(&c.hash) {
                    break Some(c);
                }
            };
            let Some(c) = chunk else { break };
            let fut = self.ensure(&c, false);
            tasks.spawn(async move {
                let ok = fut.await;
                drop(permit);
                (c, ok)
            });
            while let Some(done) = tasks.try_join_next() {
                let (c, ok) = done?;
                if !ok {
                    failed.push(c);
                }
            }
        }
        while let Some(done) = tasks.join_next().await {
            let (c, ok) = done?;
            if !ok {
                failed.push(c);
            }
        }
        // One more try for anything that failed, in case a source was only briefly away.
        let retried = futures::future::join_all(failed.iter().map(|c| self.ensure(c, false))).await;
        let missing = retried.iter().filter(|ok| !**ok).count();
        if missing > 0 {
            return Err(anyhow!(
                "{missing} chunk(s) are not available from any source"
            ));
        }
        Ok(())
    }

    /// Every unique chunk, in the order a model loader is likely to need them.
    pub async fn prefetch_order(self: &Arc<Self>) -> Result<Vec<ChunkRef>> {
        let files = &self.manifest.files;
        let (tensors, other): (Vec<usize>, Vec<usize>) =
            (0..files.len()).partition(|&i| files[i].path.ends_with(".safetensors"));
        let mut ranges: Vec<(usize, u64, u64)> = Vec::new();
        // Small files first: configs and tokenizers are read before any weights.
        let mut other = other;
        other.sort_by_key(|&i| files[i].size);
        for &i in &other {
            ranges.push((i, 0, files[i].size));
        }
        // Then every header, so a loader can map all shards at once.
        let mut layers = Vec::new();
        for &i in &tensors {
            match self.tensor_ranges(i).await {
                Ok((header, t)) => {
                    ranges.push((i, 0, header));
                    layers.extend(
                        t.into_iter()
                            .map(|(name, s, e)| (layer_key(&name), i, s, e)),
                    );
                }
                // Not really safetensors: fetch it front to back.
                Err(_) => ranges.push((i, 0, files[i].size)),
            }
        }
        // Then tensors layer by layer, across all shards.
        layers.sort();
        ranges.extend(layers.into_iter().map(|(_, i, s, e)| (i, s, e)));
        // Then anything left, such as padding between tensors.
        ranges.extend((0..files.len()).map(|i| (i, 0, files[i].size)));

        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (file, start, end) in ranges {
            if start >= end {
                continue;
            }
            let starts = &self.starts[file];
            let first = starts.partition_point(|&s| s <= start) - 1;
            let last = starts.partition_point(|&s| s < end) - 1;
            for c in &files[file].chunks[first..=last] {
                if seen.insert(c.hash.clone()) {
                    out.push(c.clone());
                }
            }
        }
        Ok(out)
    }

    /// A safetensors file's header length (including its 8-byte size prefix) and each
    /// tensor's name and byte range, read through the model itself.
    async fn tensor_ranges(
        self: &Arc<Self>,
        file: usize,
    ) -> Result<(u64, Vec<(String, u64, u64)>)> {
        let prefix = self.read(file, 0, 8).await?;
        let len = u64::from_le_bytes(prefix.as_slice().try_into().context("short file")?);
        if len > MAX_HEADER {
            bail!("header too large");
        }
        let header = self.read(file, 8, len as usize).await?;
        let data = 8 + len;
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
        let mut out = Vec::new();
        for (name, v) in raw {
            if name == "__metadata__" {
                continue;
            }
            let offsets = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .context("offsets")?;
            let begin = offsets
                .first()
                .and_then(|o| o.as_u64())
                .context("offsets")?;
            let end = offsets.get(1).and_then(|o| o.as_u64()).context("offsets")?;
            out.push((name, data + begin, data + end));
        }
        Ok((data, out))
    }
}

/// Sort key putting tensors in the order a model runs: embeddings, then each numbered
/// layer (`model.layers.12.mlp...`), then the rest (final norm, output head).
fn layer_key(name: &str) -> (u8, u64, String) {
    let layer = name.split('.').find_map(|part| part.parse::<u64>().ok());
    let group = match layer {
        Some(_) => 1,
        None if name.contains("embed") || name.contains("wte") || name.contains("wpe") => 0,
        None => 2,
    };
    (group, layer.unwrap_or(0), name.to_string())
}

#[cfg(test)]
mod tests {
    use super::layer_key;

    #[test]
    fn layers_sort_numerically() {
        let mut names = vec![
            "lm_head.weight",
            "model.layers.10.mlp.up_proj.weight",
            "model.norm.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "model.layers.2.mlp.up_proj.weight",
        ];
        names.sort_by_key(|n| layer_key(n));
        assert_eq!(
            names,
            [
                "model.embed_tokens.weight",
                "model.layers.2.mlp.up_proj.weight",
                "model.layers.2.self_attn.q_proj.weight",
                "model.layers.10.mlp.up_proj.weight",
                "lm_head.weight",
                "model.norm.weight",
            ]
        );
    }
}
