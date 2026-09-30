//! Terminal progress bar for long downloads: a bunny hops along the track like the
//! Chrome dinosaur game, jumping the cacti that scroll past, while it moves from left to
//! right with the download.
//!
//! ```text
//!           🐇                     fetching 3f9a01c2e7b4
//! ━━━━━━━━━━━━🌵━━━──────────🌵───  42%  75.3 / 180.6 MB  14.2 MB/s  ETA 0:07
//! ```
//!
//! It draws on stderr, and only when stderr is a terminal, so piped output and logs are
//! untouched. Set `CHUNGUS_NO_PROGRESS=1` to turn it off.

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const BUNNY: &str = "🐇";
const CACTUS: &str = "🌵";
const DONE: char = '━';
const AHEAD: char = '─';
/// Cactus spacing, repeated: uneven gaps so it doesn't look mechanical.
const CACTI: [usize; 4] = [0, 11, 28, 36];
const CACTI_PERIOD: usize = 50;
/// Downloads that finish sooner than this never show the bar.
const DELAY: Duration = Duration::from_millis(300);
const TICK: Duration = Duration::from_millis(90);

/// Bytes done out of a total, shared between a download and its display.
#[derive(Default, Debug)]
pub struct Bar {
    done: AtomicU64,
    total: AtomicU64,
}

impl Bar {
    pub fn new() -> Arc<Bar> {
        Arc::default()
    }

    pub fn get(&self) -> (u64, u64) {
        (self.done.load(Relaxed), self.total.load(Relaxed))
    }

    /// A reader for [`show`].
    pub fn progress(self: &Arc<Self>) -> impl Fn() -> (u64, u64) + Send + 'static {
        let bar = self.clone();
        move || bar.get()
    }
}

tokio::task_local! {
    static CURRENT: Arc<Bar>;
}

/// Run `fut` with every chunk download inside it counted on `bar`.
pub async fn track<F: Future>(bar: Arc<Bar>, fut: F) -> F::Output {
    CURRENT.scope(bar, fut).await
}

/// Chunk downloads are starting: `total` bytes are wanted, `done` of them already local.
pub(crate) fn begin(done: u64, total: u64) {
    let _ = CURRENT.try_with(|b| {
        b.total.fetch_add(total, Relaxed);
        b.done.fetch_add(done, Relaxed);
    });
}

/// `bytes` more have arrived.
pub(crate) fn advance(bytes: u64) {
    let _ = CURRENT.try_with(|b| b.done.fetch_add(bytes, Relaxed));
}

/// The bunny, drawn on stderr until [`Display::finish`].
pub struct Display {
    stop: oneshot::Sender<bool>,
    task: JoinHandle<()>,
}

/// Start drawing the progress that `progress()` reports as (done, total) bytes, or None
/// when stderr isn't a terminal.
pub fn show(
    label: impl Into<String>,
    progress: impl Fn() -> (u64, u64) + Send + 'static,
) -> Option<Display> {
    let off = std::env::var_os("CHUNGUS_NO_PROGRESS").is_some_and(|v| !v.is_empty() && v != "0");
    let dumb = std::env::var("TERM").is_ok_and(|t| t == "dumb");
    if off || dumb || !std::io::stderr().is_terminal() {
        return None;
    }
    let label = label.into();
    let (stop, mut stopped) = oneshot::channel::<bool>();
    let task = tokio::spawn(async move {
        let started = Instant::now();
        tokio::select! {
            _ = tokio::time::sleep(DELAY) => {}
            _ = &mut stopped => return,
        }
        let width = track_width();
        let mut speed = Speed::default();
        let mut drawn = false;
        let mut tick = 0;
        let ok = loop {
            let (done, total) = progress();
            let per_sec = speed.update(done, total);
            let lines = frame(tick, width, done, total, per_sec, Some(&label));
            draw(&lines, drawn);
            drawn = true;
            tick += 1;
            tokio::select! {
                _ = tokio::time::sleep(TICK) => {}
                r = &mut stopped => break r.unwrap_or(false),
            }
        };
        if ok {
            // Replace both lines with one: the bunny at the finish line.
            let (_, total) = progress();
            let last = finished(width, total, started.elapsed());
            eprint!("\r\x1b[1A\x1b[2K{last}\n\x1b[2K");
        } else {
            // Erase both lines so an error message starts clean.
            eprint!("\r\x1b[2K\x1b[1A\x1b[2K");
        }
        let _ = std::io::stderr().flush();
    });
    Some(Display { stop, task })
}

impl Display {
    /// Stop drawing: on success leave the bunny at the finish line, otherwise erase it.
    pub async fn finish(self, ok: bool) {
        let _ = self.stop.send(ok);
        let _ = self.task.await;
    }
}

/// Finish `display`, if there is one, according to how `result` went.
pub async fn done<T, E>(display: Option<Display>, result: &Result<T, E>) {
    if let Some(d) = display {
        d.finish(result.is_ok()).await;
    }
}

fn draw(lines: &[String; 2], redraw: bool) {
    let mut err = std::io::stderr().lock();
    if redraw {
        let _ = write!(err, "\r\x1b[1A");
    }
    let _ = write!(err, "\x1b[2K{}\n\x1b[2K{}", lines[0], lines[1]);
    let _ = err.flush();
}

/// Track cells: whatever the terminal leaves after the ~45 columns of numbers.
fn track_width() -> usize {
    let cols = std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.parse::<usize>().ok())
        .unwrap_or(80);
    cols.saturating_sub(48).clamp(12, 40)
}

/// Bytes per second since the download started moving.
#[derive(Default)]
struct Speed {
    since: Option<(Instant, u64)>,
}

impl Speed {
    fn update(&mut self, done: u64, total: u64) -> Option<f64> {
        if total == 0 {
            return None;
        }
        let (t0, d0) = *self.since.get_or_insert((Instant::now(), done));
        let secs = t0.elapsed().as_secs_f64();
        (secs >= 0.5 && done > d0).then(|| (done - d0) as f64 / secs)
    }
}

/// The bunny's cell: it runs from 0 to `width - 2` (it's two columns wide).
fn bunny_at(width: usize, done: u64, total: u64) -> usize {
    if total == 0 {
        return 0;
    }
    let last = width - 2;
    (done.min(total) as u128 * last as u128 / total as u128) as usize
}

/// Is there a cactus at cell `i` on frame `tick`? The cacti scroll one cell left per tick.
fn cactus(tick: usize, i: usize) -> bool {
    CACTI.contains(&((i + tick) % CACTI_PERIOD))
}

/// One animation frame: the air above the track, and the track with the numbers.
fn frame(
    tick: usize,
    width: usize,
    done: u64,
    total: u64,
    per_sec: Option<f64>,
    label: Option<&str>,
) -> [String; 2] {
    let at = bunny_at(width, done, total);
    // Jump just before a cactus reaches the bunny and land once it has passed.
    let airborne = (at.saturating_sub(2)..=at + 2).any(|i| i < width - 1 && cactus(tick, i));

    let mut air = " ".repeat(at);
    if airborne {
        air.push_str(BUNNY);
        air.push_str(&" ".repeat(width - at - 2));
    } else {
        air.push_str(&" ".repeat(width - at));
    }
    if let Some(label) = label {
        air.push_str("  ");
        air.push_str(label);
    }

    let mut ground = String::new();
    let mut i = 0;
    while i < width {
        if i == at && !airborne {
            ground.push_str(BUNNY);
            i += 2;
        } else if i + 1 < width && cactus(tick, i) {
            ground.push_str(CACTUS);
            i += 2;
        } else {
            ground.push(if i < at { DONE } else { AHEAD });
            i += 1;
        }
    }
    ground.push_str("  ");
    ground.push_str(&numbers(done, total, per_sec));
    [air, ground]
}

/// The last frame: the bunny sits at the finish line.
fn finished(width: usize, total: u64, took: Duration) -> String {
    let mut line: String = std::iter::repeat_n(DONE, width - 2).collect();
    line.push_str(BUNNY);
    line.push_str(&format!(
        "  100%  {} in {:.1}s",
        size(total),
        took.as_secs_f64()
    ));
    line
}

fn numbers(done: u64, total: u64, per_sec: Option<f64>) -> String {
    if total == 0 {
        return "looking for peers".into();
    }
    let done = done.min(total);
    let mut s = format!(
        "{:>3}%  {} / {}",
        done * 100 / total,
        size_num(done, total),
        size(total)
    );
    if let Some(r) = per_sec {
        s.push_str(&format!("  {}/s", size(r as u64)));
        let left = ((total - done) as f64 / r).ceil() as u64;
        s.push_str(&format!("  ETA {}:{:02}", left / 60, left % 60));
    }
    s
}

/// `bytes` in the unit that suits it.
fn size(bytes: u64) -> String {
    let (unit, div) = unit(bytes);
    format!("{:.1} {unit}", bytes as f64 / div)
}

/// `bytes` as a bare number in the unit of `of`, for "12.0 / 180.6 MB".
fn size_num(bytes: u64, of: u64) -> String {
    format!("{:.1}", bytes as f64 / unit(of).1)
}

fn unit(bytes: u64) -> (&'static str, f64) {
    match bytes {
        b if b >= 1_000_000_000 => ("GB", 1e9),
        b if b >= 1_000_000 => ("MB", 1e6),
        _ => ("KB", 1e3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Terminal columns taken by a line, counting each emoji as two.
    fn cols(s: &str) -> usize {
        s.chars()
            .map(|c| if c.len_utf8() == 4 { 2 } else { 1 })
            .sum()
    }

    fn track_part(line: &str, width: usize) -> String {
        let mut out = String::new();
        let mut n = 0;
        for c in line.chars() {
            if n >= width {
                break;
            }
            n += if c.len_utf8() == 4 { 2 } else { 1 };
            out.push(c);
        }
        out
    }

    #[test]
    fn track_keeps_its_width_on_every_frame() {
        for width in [12, 23, 40] {
            for tick in 0..CACTI_PERIOD {
                for pct in [0, 1, 33, 50, 99, 100] {
                    let [air, ground] = frame(tick, width, pct, 100, None, None);
                    assert_eq!(cols(&air), width, "air, tick {tick}, {pct}%");
                    assert_eq!(cols(&track_part(&ground, width)), width);
                    assert_eq!(
                        ground.matches(BUNNY).count() + air.matches(BUNNY).count(),
                        1
                    );
                }
            }
        }
    }

    #[test]
    fn bunny_runs_left_to_right() {
        assert_eq!(bunny_at(30, 0, 0), 0);
        assert_eq!(bunny_at(30, 0, 100), 0);
        assert_eq!(bunny_at(30, 50, 100), 14);
        assert_eq!(bunny_at(30, 100, 100), 28);
        assert_eq!(bunny_at(30, 500, 100), 28);
    }

    #[test]
    fn bunny_jumps_every_cactus() {
        // Over a full cactus cycle the bunny is in the air whenever a cactus passes under
        // it, and back on the ground in between.
        let width = 30;
        let (mut hops, mut landings) = (0, 0);
        for tick in 0..CACTI_PERIOD {
            let [air, ground] = frame(tick, width, 50, 100, None, None);
            if air.contains(BUNNY) {
                hops += 1;
                let under = track_part(&ground, width);
                assert!(!under.contains(BUNNY));
            } else {
                landings += 1;
            }
        }
        assert!(hops > 0 && landings > 0);
    }

    #[test]
    fn numbers_read_well() {
        assert_eq!(numbers(0, 0, None), "looking for peers");
        assert_eq!(
            numbers(75_300_000, 180_600_000, Some(14_200_000.0)),
            " 41%  75.3 / 180.6 MB  14.2 MB/s  ETA 0:08"
        );
        assert_eq!(
            numbers(2_000_000_000, 4_000_000_000, None),
            " 50%  2.0 / 4.0 GB"
        );
    }

    #[tokio::test]
    async fn chunk_progress_reaches_the_tracked_bar() {
        let bar = Bar::new();
        track(bar.clone(), async {
            begin(10, 100);
            advance(40);
        })
        .await;
        assert_eq!(bar.get(), (50, 100));
        // Outside a tracked future nothing is counted, and nothing breaks.
        advance(5);
        assert_eq!(bar.get(), (50, 100));
    }
}
