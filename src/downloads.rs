//! Download counts for the registry's models.
//!
//! Peers move the bytes, so the registry never sees a download itself. What it does see
//! is `chungus fetch` and `chungus mount` looking a name up first, flagged with
//! `?download=1`. Each such lookup counts once per model, per client network (/24 or
//! /48), per UTC day, so that re-running a fetch, a CI loop or one server hammering the
//! endpoint adds at most one a day. Inflating a count takes many networks.
//!
//! No addresses are stored. A lookup is remembered as a hash of the network and name
//! under a random key that is thrown away at the end of the day, along with the hashes.
//! Counts survive restarts in a JSON file in the registry's data directory.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Days of per-model history kept, for "downloads in the last 30 days".
pub const RECENT_DAYS: u64 = 30;
/// Most distinct (network, model) pairs remembered in a day. Past it, new ones aren't
/// counted, so a flood of lookups can't exhaust memory.
const MAX_SEEN: usize = 2_000_000;

#[derive(Serialize, Deserialize, Default)]
struct State {
    /// Downloads of each name, all time.
    total: BTreeMap<String, u64>,
    /// Downloads of each name by UTC day (days since the epoch), for the last
    /// [`RECENT_DAYS`] days.
    recent: BTreeMap<u64, BTreeMap<String, u64>>,
    /// Downloads of every model by UTC day, all time.
    daily: BTreeMap<u64, u64>,
    /// The day `key` and `seen` belong to.
    day: u64,
    /// Today's hashing key, hex.
    key: String,
    /// Today's counted (network, name) pairs, as truncated keyed hashes in hex.
    seen: HashSet<String>,
}

/// Download counts of one model.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelDownloads {
    pub total: u64,
    pub last_30_days: u64,
}

/// Download counts across the registry.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Totals {
    pub total: u64,
    pub last_30_days: u64,
    /// Every day with downloads, oldest first: (days since the epoch, downloads).
    pub daily: Vec<(u64, u64)>,
    /// Bytes nodes served, as credited from downloaders' receipts (see
    /// [`crate::leaderboard`]). Filled in by the registry.
    #[serde(default)]
    pub bytes_served: u64,
}

pub struct Downloads {
    path: PathBuf,
    state: Mutex<State>,
    dirty: AtomicBool,
}

/// The network `ip` belongs to: its /24 for IPv4, its /48 for IPv6.
fn network(ip: IpAddr) -> Vec<u8> {
    match ip {
        IpAddr::V4(v4) => v4.octets()[..3].to_vec(),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => network(IpAddr::V4(v4)),
            None => v6.octets()[..6].to_vec(),
        },
    }
}

fn new_key() -> String {
    let mut k = [0u8; 32];
    // Without randomness a fixed key still dedupes; it only makes the hashes guessable.
    let _ = getrandom::fill(&mut k);
    blake3::Hash::from_bytes(k).to_hex().to_string()
}

impl Downloads {
    /// Load the counts at `path`, or start from none.
    pub fn open(path: &Path) -> Result<Downloads> {
        let state = match fs::read(path) {
            Ok(b) => serde_json::from_slice(&b).with_context(|| format!("{}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => State::default(),
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        Ok(Downloads {
            path: path.to_path_buf(),
            state: Mutex::new(state),
            dirty: AtomicBool::new(false),
        })
    }

    /// Count a download of `name` by a client at `ip`, at Unix time `now`. False if this
    /// network already counted for `name` today.
    pub fn count(&self, name: &str, ip: IpAddr, now: u64) -> bool {
        let today = now / 86400;
        let mut s = self.state.lock().unwrap();
        if s.day != today || s.key.is_empty() {
            s.day = today;
            s.key = new_key();
            s.seen.clear();
            s.recent.retain(|d, _| d + RECENT_DAYS > today);
        }
        let key = blake3::Hash::from_hex(&s.key).map(|h| *h.as_bytes());
        let key = key.unwrap_or_default();
        let mut h = blake3::Hasher::new_keyed(&key);
        h.update(&network(ip));
        h.update(b"\0");
        h.update(name.as_bytes());
        let id = h.finalize().to_hex()[..32].to_string();
        if s.seen.len() >= MAX_SEEN || !s.seen.insert(id) {
            return false;
        }
        *s.total.entry(name.to_string()).or_default() += 1;
        *s.recent
            .entry(today)
            .or_default()
            .entry(name.to_string())
            .or_default() += 1;
        *s.daily.entry(today).or_default() += 1;
        self.dirty.store(true, Ordering::Relaxed);
        true
    }

    /// Counts for `name` as of Unix time `now`.
    pub fn of(&self, name: &str, now: u64) -> ModelDownloads {
        let today = now / 86400;
        let s = self.state.lock().unwrap();
        ModelDownloads {
            total: s.total.get(name).copied().unwrap_or(0),
            last_30_days: s
                .recent
                .range(today.saturating_sub(RECENT_DAYS - 1)..)
                .filter_map(|(_, names)| names.get(name))
                .sum(),
        }
    }

    /// Counts across every model as of Unix time `now`.
    pub fn totals(&self, now: u64) -> Totals {
        let today = now / 86400;
        let s = self.state.lock().unwrap();
        Totals {
            total: s.daily.values().sum(),
            last_30_days: s
                .daily
                .range(today.saturating_sub(RECENT_DAYS - 1)..)
                .map(|(_, n)| n)
                .sum(),
            daily: s.daily.iter().map(|(d, n)| (*d, *n)).collect(),
            bytes_served: 0,
        }
    }

    /// Write the counts to disk if they changed since the last save.
    pub fn save(&self) -> Result<()> {
        if !self.dirty.swap(false, Ordering::Relaxed) {
            return Ok(());
        }
        let bytes = serde_json::to_vec(&*self.state.lock().unwrap())?;
        crate::store::write_atomic(&self.path, &bytes).inspect_err(|_| {
            self.dirty.store(true, Ordering::Relaxed);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_once_per_network_per_day_and_persist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("downloads.json");
        let d = Downloads::open(&path).unwrap();
        let day = 20_000 * 86400;
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();

        assert!(d.count("acme/tiny", ip("203.0.113.7"), day));
        // Same /24, same day: not again. Another model or network: yes.
        assert!(!d.count("acme/tiny", ip("203.0.113.200"), day + 5));
        assert!(!d.count("acme/tiny", ip("::ffff:203.0.113.9"), day + 5));
        assert!(d.count("acme/other", ip("203.0.113.7"), day));
        assert!(d.count("acme/tiny", ip("198.51.100.1"), day));
        // IPv6 counts per /48.
        assert!(d.count("acme/tiny", ip("2001:db8:1:1::1"), day));
        assert!(!d.count("acme/tiny", ip("2001:db8:1:ffff::2"), day));
        // Next day, the same network counts again.
        assert!(d.count("acme/tiny", ip("203.0.113.7"), day + 86400));

        let now = day + 86400;
        assert_eq!(
            d.of("acme/tiny", now),
            ModelDownloads {
                total: 4,
                last_30_days: 4
            }
        );
        assert_eq!(d.totals(now).total, 5);
        assert_eq!(d.totals(now).daily, vec![(20_000, 4), (20_001, 1)]);

        // Saved counts, and today's dedupe, survive a restart.
        d.save().unwrap();
        let d = Downloads::open(&path).unwrap();
        assert!(!d.count("acme/tiny", ip("203.0.113.7"), now));
        assert_eq!(d.of("acme/tiny", now).total, 4);
        // Old days drop out of the recent count but not the total.
        let later = now + 40 * 86400;
        assert_eq!(
            d.of("acme/tiny", later),
            ModelDownloads {
                total: 4,
                last_30_days: 0
            }
        );
        assert!(!fs::read_to_string(&path).unwrap().contains("203.0.113"));
    }
}
