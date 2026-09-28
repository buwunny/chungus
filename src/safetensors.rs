//! Minimal safetensors header parsing: just enough to find tensor byte ranges and dtypes.
//!
//! Layout: 8-byte little-endian header length N, N bytes of JSON, then the tensor data.
//! Each JSON entry (except `__metadata__`) has `dtype` and `data_offsets: [begin, end]`
//! relative to the start of the data section.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::BTreeMap;

use crate::segment::{Dtype, Segment};

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    data_offsets: [u64; 2],
}

/// Split a safetensors file into segments: one for the header, one per tensor, and
/// raw segments for any gaps. Returns `None` if the bytes don't look like safetensors.
pub fn segments(data: &[u8]) -> Result<Option<Vec<Segment>>> {
    if data.len() < 8 {
        return Ok(None);
    }
    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap());
    let data_start = match header_len.checked_add(8) {
        Some(s) if s <= data.len() as u64 => s,
        _ => return Ok(None),
    };
    let header = &data[8..data_start as usize];
    if header.first() != Some(&b'{') {
        return Ok(None);
    }
    let raw: BTreeMap<String, serde_json::Value> = match serde_json::from_slice(header) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let mut tensors = Vec::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo =
            serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
        let [begin, end] = info.data_offsets;
        if begin > end || data_start + end > data.len() as u64 {
            bail!("tensor {name} has out-of-range offsets {begin}..{end}");
        }
        tensors.push((
            data_start + begin,
            data_start + end,
            Dtype::from_safetensors(&info.dtype),
        ));
    }
    tensors.sort_by_key(|t| t.0);

    let mut out = vec![Segment {
        start: 0,
        end: data_start,
        dtype: Dtype::Raw,
    }];
    let mut cursor = data_start;
    for (start, end, dtype) in tensors {
        if start < cursor {
            bail!("overlapping tensors at byte {start}");
        }
        if start > cursor {
            out.push(Segment {
                start: cursor,
                end: start,
                dtype: Dtype::Raw,
            });
        }
        if end > start {
            out.push(Segment { start, end, dtype });
        }
        cursor = end;
    }
    if cursor < data.len() as u64 {
        out.push(Segment {
            start: cursor,
            end: data.len() as u64,
            dtype: Dtype::Raw,
        });
    }
    Ok(Some(out))
}
