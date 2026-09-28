//! Protections for a node's owner and for the network: how much a node uploads, to how
//! many peers, and which peers it lets into its DHT routing table.

use libp2p::multiaddr::Protocol;
use libp2p::{Multiaddr, PeerId};
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::time::{Duration, Instant};

/// Limits on what a node serves. The defaults suit a desktop; a dedicated node can raise
/// them.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Upload cap in bytes per second, across all peers. `None` is unlimited.
    pub upload_bytes_per_sec: Option<u64>,
    /// Open connections, in and out.
    pub max_connections: u32,
    /// Requests from one peer being served at once; more get a "busy" answer.
    pub max_requests_per_peer: usize,
    /// Requests being served at once, across all peers.
    pub max_uploads: usize,
    /// Routing-table entries from one IPv4 /24 or IPv6 /48 (see [`SubnetCaps`]).
    pub max_peers_per_subnet: usize,
    /// Download but never serve or announce anything.
    pub download_only: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            upload_bytes_per_sec: None,
            max_connections: 200,
            max_requests_per_peer: 16,
            max_uploads: 64,
            max_peers_per_subnet: 2,
            download_only: false,
        }
    }
}

/// A token bucket shared by everything a node uploads. Callers reserve bytes before
/// sending and wait for the bucket to refill; reservations queue in order, so one large
/// burst can't starve the rest.
pub struct RateLimiter {
    rate: f64,
    state: tokio::sync::Mutex<(f64, Instant)>,
}

impl RateLimiter {
    pub fn new(bytes_per_sec: u64) -> RateLimiter {
        let rate = bytes_per_sec.max(1) as f64;
        RateLimiter {
            rate,
            // Start full, allowing a one-second burst.
            state: tokio::sync::Mutex::new((rate, Instant::now())),
        }
    }

    /// Wait until `bytes` may be sent.
    pub async fn take(&self, bytes: u64) {
        let wait = {
            let mut st = self.state.lock().await;
            let now = Instant::now();
            let (tokens, last) = *st;
            let tokens = (tokens + now.duration_since(last).as_secs_f64() * self.rate)
                .min(self.rate)
                - bytes as f64;
            *st = (tokens, now);
            if tokens >= 0.0 {
                Duration::ZERO
            } else {
                Duration::from_secs_f64(-tokens / self.rate)
            }
        };
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }
}

/// The network a peer's address belongs to, for [`SubnetCaps`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Subnet {
    V4([u8; 3]),
    V6([u8; 6]),
}

/// The /24 or /48 of an address's first IP, or `None` for addresses that aren't capped:
/// loopback (local testing), and relayed addresses, whose first IP is the relay's.
pub fn subnet_of(addr: &Multiaddr) -> Option<Subnet> {
    if addr.iter().any(|p| matches!(p, Protocol::P2pCircuit)) {
        return None;
    }
    let ip = addr.iter().find_map(|p| match p {
        Protocol::Ip4(ip) => Some(IpAddr::V4(ip)),
        Protocol::Ip6(ip) => Some(IpAddr::V6(ip)),
        _ => None,
    })?;
    if ip.is_loopback() {
        return None;
    }
    Some(match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            Subnet::V4([o[0], o[1], o[2]])
        }
        IpAddr::V6(v6) => {
            let o = v6.octets();
            Subnet::V6([o[0], o[1], o[2], o[3], o[4], o[5]])
        }
    })
}

/// Keeps a DHT routing table diverse: at most `per_subnet` peers from any one /24 (IPv4)
/// or /48 (IPv6). An attacker running many node ids from a few servers then gets only a
/// few of them into anyone's table, which makes it much harder to surround a node or a key
/// (an eclipse attack) with Sybil identities.
pub struct SubnetCaps {
    per_subnet: usize,
    members: HashMap<Subnet, HashSet<PeerId>>,
    subnet: HashMap<PeerId, Subnet>,
}

impl SubnetCaps {
    /// `per_subnet` of 0 means no cap.
    pub fn new(per_subnet: usize) -> SubnetCaps {
        SubnetCaps {
            per_subnet,
            members: HashMap::new(),
            subnet: HashMap::new(),
        }
    }

    /// Whether `peer` at `addr` may enter the routing table. Admitting records it.
    pub fn admit(&mut self, peer: PeerId, addr: &Multiaddr) -> bool {
        if self.per_subnet == 0 {
            return true;
        }
        let Some(net) = subnet_of(addr) else {
            return true;
        };
        // A peer stays counted under the first subnet it was admitted from.
        if let Some(existing) = self.subnet.get(&peer) {
            return *existing == net || self.room_in(net);
        }
        if !self.room_in(net) {
            return false;
        }
        self.members.entry(net).or_default().insert(peer);
        self.subnet.insert(peer, net);
        true
    }

    fn room_in(&self, net: Subnet) -> bool {
        self.members.get(&net).map_or(0, HashSet::len) < self.per_subnet
    }

    /// `peer` left the routing table.
    pub fn remove(&mut self, peer: &PeerId) {
        if let Some(net) = self.subnet.remove(peer)
            && let Some(m) = self.members.get_mut(&net)
        {
            m.remove(peer);
            if m.is_empty() {
                self.members.remove(&net);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    #[test]
    fn subnets_are_capped() {
        let mut caps = SubnetCaps::new(2);
        let (a, b, c, d) = (
            PeerId::random(),
            PeerId::random(),
            PeerId::random(),
            PeerId::random(),
        );
        assert!(caps.admit(a, &addr("/ip4/203.0.113.1/tcp/4001")));
        assert!(caps.admit(b, &addr("/ip4/203.0.113.2/tcp/4001")));
        // A third node from the same /24 is refused; another /24 is fine.
        assert!(!caps.admit(c, &addr("/ip4/203.0.113.3/udp/4001/quic-v1")));
        assert!(caps.admit(c, &addr("/ip4/198.51.100.3/tcp/4001")));
        // Re-admitting a known peer doesn't use another slot.
        assert!(caps.admit(a, &addr("/ip4/203.0.113.1/tcp/4001")));
        // Once one leaves, there's room again.
        caps.remove(&b);
        assert!(caps.admit(d, &addr("/ip4/203.0.113.4/tcp/4001")));
        // IPv6 is grouped by /48.
        let mut caps = SubnetCaps::new(1);
        assert!(caps.admit(a, &addr("/ip6/2001:db8:1:1::1/tcp/4001")));
        assert!(!caps.admit(b, &addr("/ip6/2001:db8:1:2::1/tcp/4001")));
        assert!(caps.admit(b, &addr("/ip6/2001:db8:2::1/tcp/4001")));
        // Loopback and relayed addresses aren't capped.
        assert!(caps.admit(c, &addr("/ip4/127.0.0.1/tcp/1")));
        assert!(caps.admit(
            d,
            &addr("/ip6/2001:db8:1::9/tcp/4001/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN/p2p-circuit")
        ));
    }

    #[tokio::test]
    async fn rate_limiter_paces_uploads() {
        let lim = RateLimiter::new(100_000);
        let start = Instant::now();
        // The first second's worth goes at once; the next 50 KB waits about half a second.
        lim.take(100_000).await;
        assert!(start.elapsed() < Duration::from_millis(100));
        lim.take(50_000).await;
        let t = start.elapsed();
        assert!(
            t >= Duration::from_millis(400) && t < Duration::from_millis(900),
            "{t:?}"
        );
    }
}
