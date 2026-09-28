//! Content-defined chunking with FastCDC, run separately inside each segment.

use fastcdc::v2020::FastCDC;

use crate::segment::{Dtype, Segment};

pub const MIN_SIZE: usize = 16 * 1024;
pub const AVG_SIZE: usize = 64 * 1024;
pub const MAX_SIZE: usize = 256 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct ChunkSpan {
    pub start: usize,
    pub end: usize,
    pub dtype: Dtype,
}

/// Cut every segment into chunks. Cut points inside a float segment are rounded down to
/// the element width so the byte-plane transform always sees whole elements; the result
/// is still content-defined because the rounding depends only on the content-chosen cut.
pub fn chunk(data: &[u8], segments: &[Segment]) -> Vec<ChunkSpan> {
    let mut out = Vec::new();
    for seg in segments {
        let (s, e) = (seg.start as usize, seg.end as usize);
        let bytes = &data[s..e];
        if bytes.is_empty() {
            continue;
        }
        let width = seg.dtype.width();
        if bytes.len() <= MIN_SIZE {
            out.push(ChunkSpan {
                start: s,
                end: e,
                dtype: seg.dtype,
            });
            continue;
        }
        let mut prev = 0;
        for c in FastCDC::new(bytes, MIN_SIZE, AVG_SIZE, MAX_SIZE) {
            let cut = c.offset + c.length;
            let cut = if cut == bytes.len() {
                cut
            } else {
                cut - cut % width
            };
            if cut > prev {
                out.push(ChunkSpan {
                    start: s + prev,
                    end: s + cut,
                    dtype: seg.dtype,
                });
                prev = cut;
            }
        }
        if prev < bytes.len() {
            out.push(ChunkSpan {
                start: s + prev,
                end: e,
                dtype: seg.dtype,
            });
        }
    }
    out
}
