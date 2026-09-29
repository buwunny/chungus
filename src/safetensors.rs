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

/// The header of a safetensors file's leading bytes (`8`-byte length, then JSON): its
/// JSON, once `data` holds all of it. `None` while more bytes are needed.
pub fn header_json(data: &[u8]) -> Result<Option<&[u8]>> {
    let Some(len) = data.get(..8) else {
        return Ok(None);
    };
    let len = u64::from_le_bytes(len.try_into().unwrap());
    if len > MAX_HEADER {
        bail!("safetensors header of {len} bytes is too large");
    }
    Ok(data.get(8..8 + len as usize))
}

/// Largest header accepted, as in the safetensors reference implementation.
pub const MAX_HEADER: u64 = 100 << 20;

#[derive(Deserialize)]
struct Shape {
    dtype: String,
    shape: Vec<u64>,
}

/// Parameters per dtype (`"BF16"`, ...) of the tensors a header lists.
pub fn params(json: &[u8]) -> Result<BTreeMap<String, u64>> {
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(json).context("safetensors header isn't JSON")?;
    let mut out = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let t: Shape = serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
        let n = t
            .shape
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d))
            .with_context(|| format!("tensor {name} is too large"))?;
        *out.entry(t.dtype).or_default() += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_parameters_per_dtype() {
        let json = br#"{"__metadata__":{"format":"pt"},
            "a":{"dtype":"BF16","shape":[4,3],"data_offsets":[0,24]},
            "b":{"dtype":"BF16","shape":[5],"data_offsets":[24,34]},
            "c":{"dtype":"F32","shape":[],"data_offsets":[34,38]}}"#;
        let p = params(json).unwrap();
        assert_eq!(p, BTreeMap::from([("BF16".into(), 17), ("F32".into(), 1)]));

        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(json);
        assert_eq!(header_json(&file[..20]).unwrap(), None);
        assert_eq!(header_json(&file).unwrap(), Some(&json[..]));
        let huge = (MAX_HEADER + 1).to_le_bytes();
        assert!(header_json(&huge).is_err());
    }
}
