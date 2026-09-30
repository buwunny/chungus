//! Minimal GGUF parsing: just enough to find tensor byte ranges and types.
//!
//! Layout (little-endian): magic `GGUF`, version (u32, 2 or 3), tensor count (u64),
//! key/value count (u64), the key/value pairs, then per tensor its name, dimension count
//! (u32), dimensions (u64 each), ggml type (u32) and offset (u64, relative to the data
//! section). The data section starts after the tensor info, padded to
//! `general.alignment` (default 32).
//!
//! Unlike [`crate::safetensors`], anything malformed gives `Ok(None)` rather than an
//! error, so the file is packed as one raw segment: `chungus hub` packs files on the fly,
//! and a parser bug shouldn't stop it serving a file llama.cpp can load.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

use crate::segment::{Dtype, Segment};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// Deepest nesting of arrays inside a key/value pair.
const MAX_DEPTH: u32 = 8;
/// Most dimensions a tensor has (`GGML_MAX_DIMS`).
const MAX_DIMS: u32 = 4;
/// Largest header (key/values and tensor info) [`header_len`] accepts. Most of a GGUF
/// header is the tokenizer: 5.9 MB for Qwen2.5 0.5B.
pub const MAX_HEADER: usize = 64 << 20;

/// One ggml type: its name, and its block size in elements and bytes, where known.
struct GgmlType {
    name: &'static str,
    block: Option<(u64, u64)>,
    dtype: Dtype,
}

/// A ggml type by id. Sizes are known only for the types tested against real files; a
/// tensor of any other type runs to the next tensor's offset.
fn ggml_type(id: u32) -> Option<GgmlType> {
    let (name, block, dtype) = match id {
        0 => ("F32", Some((1, 4)), Dtype::F32),
        1 => ("F16", Some((1, 2)), Dtype::F16),
        2 => ("Q4_0", Some((32, 18)), Dtype::Raw),
        3 => ("Q4_1", Some((32, 20)), Dtype::Raw),
        6 => ("Q5_0", Some((32, 22)), Dtype::Raw),
        7 => ("Q5_1", Some((32, 24)), Dtype::Raw),
        8 => ("Q8_0", Some((32, 34)), Dtype::Raw),
        9 => ("Q8_1", None, Dtype::Raw),
        10 => ("Q2_K", Some((256, 84)), Dtype::Raw),
        11 => ("Q3_K", Some((256, 110)), Dtype::Raw),
        12 => ("Q4_K", Some((256, 144)), Dtype::Raw),
        13 => ("Q5_K", Some((256, 176)), Dtype::Raw),
        14 => ("Q6_K", Some((256, 210)), Dtype::Raw),
        15 => ("Q8_K", None, Dtype::Raw),
        16 => ("IQ2_XXS", None, Dtype::Raw),
        17 => ("IQ2_XS", None, Dtype::Raw),
        18 => ("IQ3_XXS", None, Dtype::Raw),
        19 => ("IQ1_S", None, Dtype::Raw),
        20 => ("IQ4_NL", None, Dtype::Raw),
        21 => ("IQ3_S", None, Dtype::Raw),
        22 => ("IQ2_S", None, Dtype::Raw),
        23 => ("IQ4_XS", None, Dtype::Raw),
        24 => ("I8", Some((1, 1)), Dtype::Raw),
        25 => ("I16", Some((1, 2)), Dtype::Raw),
        26 => ("I32", Some((1, 4)), Dtype::Raw),
        27 => ("I64", Some((1, 8)), Dtype::Raw),
        28 => ("F64", Some((1, 8)), Dtype::F64),
        29 => ("IQ1_M", None, Dtype::Raw),
        30 => ("BF16", Some((1, 2)), Dtype::Bf16),
        34 => ("TQ1_0", None, Dtype::Raw),
        35 => ("TQ2_0", None, Dtype::Raw),
        39 => ("MXFP4", None, Dtype::Raw),
        _ => return None,
    };
    Some(GgmlType { name, block, dtype })
}

/// Why parsing stopped.
#[derive(Debug, PartialEq)]
enum Error {
    /// The bytes end before the header does.
    Truncated,
    /// The bytes aren't a GGUF header chungus understands.
    Invalid,
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: u64) -> Result<&'a [u8], Error> {
        let end = usize::try_from(n)
            .ok()
            .and_then(|n| self.at.checked_add(n))
            .ok_or(Error::Invalid)?;
        let b = self.data.get(self.at..end).ok_or(Error::Truncated)?;
        self.at = end;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<&'a [u8], Error> {
        let len = self.u64()?;
        self.take(len)
    }

    /// Skip one value of type `ty`.
    fn skip_value(&mut self, ty: u32, depth: u32) -> Result<(), Error> {
        match value_size(ty) {
            Some(n) => {
                self.take(n)?;
            }
            None if ty == 8 => {
                self.string()?;
            }
            None if ty == 9 => {
                if depth >= MAX_DEPTH {
                    return Err(Error::Invalid);
                }
                let elem = self.u32()?;
                let count = self.u64()?;
                match value_size(elem) {
                    Some(n) => {
                        self.take(count.checked_mul(n).ok_or(Error::Invalid)?)?;
                    }
                    // Strings and arrays each take at least 8 bytes, so a huge count
                    // runs out of bytes rather than looping for long.
                    None => {
                        for _ in 0..count {
                            self.skip_value(elem, depth + 1)?;
                        }
                    }
                }
            }
            None => return Err(Error::Invalid),
        }
        Ok(())
    }
}

/// Size of a fixed-size value type; `None` for strings, arrays and unknown types.
fn value_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

struct Tensor {
    ty: u32,
    elements: u64,
    /// Relative to the data section.
    offset: u64,
}

struct Header {
    /// End of the tensor info, padded to the alignment: where the data section starts.
    data_start: u64,
    tensors: Vec<Tensor>,
}

/// Parse the header in `data`'s leading bytes.
fn parse(data: &[u8]) -> Result<Header, Error> {
    let mut r = Reader { data, at: 0 };
    if r.take(4)? != MAGIC {
        return Err(Error::Invalid);
    }
    // Version 1 used 32-bit counts.
    if !matches!(r.u32()?, 2 | 3) {
        return Err(Error::Invalid);
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;

    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        if key == b"general.alignment" {
            if ty != 4 {
                return Err(Error::Invalid);
            }
            alignment = u64::from(r.u32()?);
            if !alignment.is_power_of_two() {
                return Err(Error::Invalid);
            }
        } else {
            r.skip_value(ty, 0)?;
        }
    }

    // Each tensor's info takes at least 24 bytes (an empty name, no dimensions), so
    // anything claiming more than the bytes left can hold is truncated or forged.
    let left = (data.len() - r.at) as u64;
    if n_tensors > left / 24 {
        return Err(if data.len() >= MAX_HEADER {
            Error::Invalid
        } else {
            Error::Truncated
        });
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        r.string()?;
        let dims = r.u32()?;
        if dims > MAX_DIMS {
            return Err(Error::Invalid);
        }
        let mut elements = 1u64;
        for _ in 0..dims {
            elements = elements.checked_mul(r.u64()?).ok_or(Error::Invalid)?;
        }
        let ty = r.u32()?;
        let offset = r.u64()?;
        if offset % alignment != 0 {
            return Err(Error::Invalid);
        }
        tensors.push(Tensor {
            ty,
            elements,
            offset,
        });
    }
    let data_start = (r.at as u64)
        .checked_next_multiple_of(alignment)
        .ok_or(Error::Invalid)?;
    Ok(Header {
        data_start,
        tensors,
    })
}

/// Split a GGUF file into segments: the header (key/values, tensor info and padding) as
/// one raw segment, one per tensor, and raw segments for the gaps between tensors.
/// `None` if the bytes aren't GGUF chungus understands; the file is then one raw segment.
pub fn segments(data: &[u8]) -> Result<Option<Vec<Segment>>> {
    Ok(segments_inner(data).ok())
}

fn segments_inner(data: &[u8]) -> Result<Vec<Segment>, Error> {
    let len = data.len() as u64;
    let h = parse(data)?;
    if h.data_start > len {
        // A file of metadata only (a vocabulary) may stop before the padding.
        if h.tensors.is_empty() {
            return Ok(vec![raw(0, len)]);
        }
        return Err(Error::Invalid);
    }
    let mut tensors = h
        .tensors
        .iter()
        .map(|t| {
            let start = h.data_start.checked_add(t.offset).ok_or(Error::Invalid)?;
            let ty = ggml_type(t.ty);
            let size = match ty.as_ref().and_then(|t| t.block) {
                Some((elems, bytes)) if t.elements % elems == 0 => Some(
                    (t.elements / elems)
                        .checked_mul(bytes)
                        .ok_or(Error::Invalid)?,
                ),
                Some(_) => return Err(Error::Invalid),
                None => None,
            };
            let dtype = ty.map_or(Dtype::Raw, |t| t.dtype);
            Ok((start, size, dtype))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tensors.sort_by_key(|t| t.0);

    let mut out = vec![raw(0, h.data_start)];
    let mut cursor = h.data_start;
    for (i, &(start, size, dtype)) in tensors.iter().enumerate() {
        let next = tensors.get(i + 1).map_or(len, |t| t.0);
        let end = match size {
            Some(n) => start.checked_add(n).ok_or(Error::Invalid)?,
            None => next,
        };
        if start < cursor || start > end || end > next || end > len {
            return Err(Error::Invalid);
        }
        if start > cursor {
            out.push(raw(cursor, start));
        }
        if end > start {
            out.push(Segment { start, end, dtype });
        }
        cursor = end;
    }
    if cursor < len {
        out.push(raw(cursor, len));
    }
    Ok(out)
}

fn raw(start: u64, end: u64) -> Segment {
    Segment {
        start,
        end,
        dtype: Dtype::Raw,
    }
}

/// The length of the header in a GGUF file's leading bytes (through the tensor info and
/// its padding, which [`segments`] makes one segment), once `data` holds all of it.
/// `None` while more bytes are needed.
pub fn header_len(data: &[u8]) -> Result<Option<usize>> {
    let data = &data[..data.len().min(MAX_HEADER)];
    match parse(data) {
        Ok(h) if h.data_start as usize <= data.len() => Ok(Some(h.data_start as usize)),
        Ok(_) | Err(Error::Truncated) if data.len() < MAX_HEADER => Ok(None),
        Ok(_) | Err(Error::Truncated) => bail!("GGUF header is over {} MB", MAX_HEADER >> 20),
        Err(Error::Invalid) => bail!("not a GGUF header chungus understands"),
    }
}

/// Elements per ggml type (`"Q8_0"`, `"F32"`, ...) of the tensors a header (as
/// [`header_len`] measures it) lists. A type chungus doesn't know is named by its id.
pub fn params(header: &[u8]) -> Result<BTreeMap<String, u64>> {
    let h = match parse(header) {
        Ok(h) => h,
        Err(Error::Truncated) => bail!("GGUF header is truncated"),
        Err(Error::Invalid) => bail!("not a GGUF header chungus understands"),
    };
    let mut out = BTreeMap::new();
    for t in h.tensors {
        let name = ggml_type(t.ty).map_or_else(|| format!("type {}", t.ty), |g| g.name.into());
        let n: &mut u64 = out.entry(name).or_default();
        *n = n
            .checked_add(t.elements)
            .ok_or_else(|| anyhow::anyhow!("GGUF parameter count overflows"))?;
    }
    Ok(out)
}

/// A tiny GGUF writer, for tests here and elsewhere (the registry tests, the fuzz seed).
#[doc(hidden)]
pub mod testing {
    /// A value in a key/value pair.
    pub enum Value {
        U8(u8),
        I8(i8),
        U16(u16),
        I16(i16),
        U32(u32),
        I32(i32),
        F32(f32),
        Bool(bool),
        Str(String),
        Array(u32, Vec<Value>),
        U64(u64),
        I64(i64),
        F64(f64),
    }

    impl Value {
        fn ty(&self) -> u32 {
            match self {
                Value::U8(_) => 0,
                Value::I8(_) => 1,
                Value::U16(_) => 2,
                Value::I16(_) => 3,
                Value::U32(_) => 4,
                Value::I32(_) => 5,
                Value::F32(_) => 6,
                Value::Bool(_) => 7,
                Value::Str(_) => 8,
                Value::Array(..) => 9,
                Value::U64(_) => 10,
                Value::I64(_) => 11,
                Value::F64(_) => 12,
            }
        }

        fn write(&self, out: &mut Vec<u8>) {
            match self {
                Value::U8(v) => out.push(*v),
                Value::I8(v) => out.extend(v.to_le_bytes()),
                Value::U16(v) => out.extend(v.to_le_bytes()),
                Value::I16(v) => out.extend(v.to_le_bytes()),
                Value::U32(v) => out.extend(v.to_le_bytes()),
                Value::I32(v) => out.extend(v.to_le_bytes()),
                Value::F32(v) => out.extend(v.to_le_bytes()),
                Value::Bool(v) => out.push(u8::from(*v)),
                Value::Str(s) => string(out, s),
                Value::Array(ty, items) => {
                    out.extend(ty.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    items.iter().for_each(|v| v.write(out));
                }
                Value::U64(v) => out.extend(v.to_le_bytes()),
                Value::I64(v) => out.extend(v.to_le_bytes()),
                Value::F64(v) => out.extend(v.to_le_bytes()),
            }
        }
    }

    /// A tensor: name, dimensions, ggml type id and its data.
    pub struct Tensor {
        pub name: String,
        pub dims: Vec<u64>,
        pub ty: u32,
        pub data: Vec<u8>,
    }

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A version 3 GGUF file of `kv` and `tensors`, each tensor aligned to `alignment`
    /// (written as `general.alignment` unless it's the default 32).
    pub fn write(kv: Vec<(&str, Value)>, tensors: &[Tensor], alignment: u64) -> Vec<u8> {
        let mut kv = kv;
        if alignment != 32 {
            kv.push(("general.alignment", Value::U32(alignment as u32)));
        }
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((kv.len() as u64).to_le_bytes());
        for (k, v) in &kv {
            string(&mut out, k);
            out.extend(v.ty().to_le_bytes());
            v.write(&mut out);
        }
        let mut offset = 0u64;
        for t in tensors {
            string(&mut out, &t.name);
            out.extend((t.dims.len() as u32).to_le_bytes());
            t.dims.iter().for_each(|d| out.extend(d.to_le_bytes()));
            out.extend(t.ty.to_le_bytes());
            out.extend(offset.to_le_bytes());
            offset = (offset + t.data.len() as u64).next_multiple_of(alignment);
        }
        let pad = |out: &mut Vec<u8>| {
            out.resize((out.len() as u64).next_multiple_of(alignment) as usize, 0)
        };
        pad(&mut out);
        for t in tensors {
            out.extend(&t.data);
            pad(&mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{Tensor, Value, write};
    use super::*;

    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    fn every_value() -> Vec<(&'static str, Value)> {
        vec![
            ("general.architecture", Value::Str("llama".into())),
            ("a.u8", Value::U8(1)),
            ("a.i8", Value::I8(-1)),
            ("a.u16", Value::U16(2)),
            ("a.i16", Value::I16(-2)),
            ("a.u32", Value::U32(3)),
            ("a.i32", Value::I32(-3)),
            ("a.f32", Value::F32(0.5)),
            ("a.bool", Value::Bool(true)),
            ("a.u64", Value::U64(4)),
            ("a.i64", Value::I64(-4)),
            ("a.f64", Value::F64(0.25)),
            (
                "tokenizer.ggml.tokens",
                Value::Array(8, vec![Value::Str("a".into()), Value::Str("bc".into())]),
            ),
            (
                "a.nested",
                Value::Array(
                    9,
                    vec![
                        Value::Array(4, vec![Value::U32(1), Value::U32(2)]),
                        Value::Array(8, vec![Value::Str("x".into())]),
                    ],
                ),
            ),
        ]
    }

    fn tensors() -> Vec<Tensor> {
        vec![
            Tensor {
                name: "token_embd.weight".into(),
                dims: vec![64, 1024],
                ty: 8, // Q8_0: 2048 blocks of 34 bytes
                data: bytes(2048 * 34, 1),
            },
            Tensor {
                name: "norm.weight".into(),
                dims: vec![64],
                ty: 0,
                data: bytes(256, 2),
            },
            Tensor {
                name: "odd.weight".into(),
                dims: vec![256],
                ty: 16, // IQ2_XXS: size unknown, runs to the next offset
                data: bytes(66, 3),
            },
            Tensor {
                name: "output.weight".into(),
                dims: vec![128, 512],
                ty: 30,
                data: bytes(128 * 512 * 2, 4),
            },
        ]
    }

    fn check_cover(data: &[u8], segs: &[Segment]) {
        let mut at = 0;
        for s in segs {
            assert_eq!(s.start, at, "segments are contiguous");
            assert!(s.end > s.start, "no empty segments");
            at = s.end;
        }
        assert_eq!(at, data.len() as u64, "segments cover the file");
    }

    #[test]
    fn segments_cover_the_file() {
        for alignment in [32, 64] {
            let ts = tensors();
            let file = write(every_value(), &ts, alignment);
            let segs = segments(&file).unwrap().unwrap();
            check_cover(&file, &segs);
            let header = header_len(&file).unwrap().unwrap();
            assert_eq!(segs[0].end, header as u64);
            assert_eq!(header as u64 % alignment, 0);

            // Each tensor is a segment of its own, with its own bytes and dtype.
            let exact = |t: &Tensor| {
                segs.iter()
                    .find(|s| file[s.start as usize..s.end as usize] == t.data[..])
                    .unwrap_or_else(|| panic!("no segment for {}", t.name))
            };
            assert_eq!(exact(&ts[0]).dtype, Dtype::Raw);
            assert_eq!(exact(&ts[1]).dtype, Dtype::F32);
            let bf16 = exact(&ts[3]);
            assert_eq!(bf16.dtype, Dtype::Bf16);
            // The unknown type runs up to the next tensor, padding included.
            let odd = segs
                .iter()
                .find(|s| file[s.start as usize..].starts_with(&ts[2].data))
                .unwrap();
            assert_eq!(odd.end, bf16.start);
        }
    }

    #[test]
    fn counts_parameters_per_type() {
        let file = write(every_value(), &tensors(), 32);
        let n = header_len(&file).unwrap().unwrap();
        assert_eq!(header_len(&file[..n - 1]).unwrap(), None);
        assert_eq!(header_len(&file[..10]).unwrap(), None);
        assert!(header_len(b"GGML-not-gguf").is_err());
        let p = params(&file[..n]).unwrap();
        assert_eq!(
            p,
            BTreeMap::from([
                ("BF16".into(), 128 * 512),
                ("F32".into(), 64),
                ("IQ2_XXS".into(), 256),
                ("Q8_0".into(), 64 * 1024),
            ])
        );
    }

    /// The point of segmenting: a tensor two files share chunks the same in both, even
    /// when their headers differ in length.
    #[test]
    fn shared_tensor_gives_identical_chunks() {
        let shared = Tensor {
            name: "output.weight".into(),
            dims: vec![32, 32768],
            ty: 8,
            data: bytes(32768 * 34, 9),
        };
        let a = write(
            vec![("general.name", Value::Str("a".into()))],
            &[
                Tensor {
                    name: "blk.0".into(),
                    dims: vec![32, 1000],
                    ty: 8,
                    data: bytes(1000 * 34, 10),
                },
                shared,
            ],
            32,
        );
        let b = write(
            vec![(
                "general.name",
                Value::Str("a longer name, and so a shifted header".into()),
            )],
            &[
                Tensor {
                    name: "blk.0".into(),
                    dims: vec![32, 4000],
                    ty: 2,
                    data: bytes(4000 * 18, 11),
                },
                Tensor {
                    name: "output.weight".into(),
                    dims: vec![32, 32768],
                    ty: 8,
                    data: bytes(32768 * 34, 9),
                },
            ],
            32,
        );
        let hashes = |f: &[u8]| {
            let segs = segments(f).unwrap().unwrap();
            let seg = segs.iter().max_by_key(|s| s.end - s.start).unwrap();
            crate::chunk::chunk(f, &[*seg])
                .iter()
                .map(|c| blake3::hash(&f[c.start..c.end]))
                .collect::<Vec<_>>()
        };
        let (ha, hb) = (hashes(&a), hashes(&b));
        assert!(ha.len() > 4);
        assert_eq!(ha, hb);
    }

    #[test]
    fn bad_input_is_none_not_a_panic() {
        let file = write(every_value(), &tensors(), 32);
        for n in 0..file.len().min(4096) {
            let _ = segments(&file[..n]).unwrap();
        }
        // Truncated data: the last tensor now runs past the end.
        assert!(segments(&file[..file.len() - 100]).unwrap().is_none());
        assert!(segments(b"GGUF").unwrap().is_none());

        let mut v1 = file.clone();
        v1[4] = 1;
        assert!(segments(&v1).unwrap().is_none());

        // Every single-byte corruption of the header parses or is refused cleanly.
        let n = header_len(&file).unwrap().unwrap();
        for i in 0..n {
            let mut f = file.clone();
            f[i] ^= 0xff;
            if let Some(segs) = segments(&f).unwrap() {
                check_cover(&f, &segs);
            }
            let _ = params(&f[..n]);
        }

        // A count of tensors no file could hold.
        let mut f = file.clone();
        f[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(segments(&f).unwrap().is_none());
        // Overlapping tensors: the second starts inside the first.
        let ts = [
            Tensor {
                name: "a".into(),
                dims: vec![64],
                ty: 0,
                data: bytes(256, 1),
            },
            Tensor {
                name: "b".into(),
                dims: vec![64],
                ty: 0,
                data: bytes(256, 2),
            },
        ];
        let mut f = write(vec![], &ts, 32);
        assert!(segments(&f).unwrap().is_some());
        // Tensor "b": its name, then dimensions (u32 1, u64 64), type (u32 0), offset.
        let name = [1, 0, 0, 0, 0, 0, 0, 0, b'b'];
        let at = f.windows(9).position(|w| w == name).unwrap() + 9 + 4 + 8 + 4;
        f[at..at + 8].copy_from_slice(&128u64.to_le_bytes());
        assert!(segments(&f).unwrap().is_none());
    }

    #[test]
    fn metadata_only_file() {
        let f = write(every_value(), &[], 32);
        let trimmed = &f[..f.len() - 1];
        for data in [&f[..], trimmed] {
            let segs = segments(data).unwrap().unwrap();
            check_cover(data, &segs);
        }
    }

    #[test]
    fn fuzz_seed_is_gguf() {
        let seed = include_bytes!("../fuzz/seeds/gguf/tiny.gguf");
        let segs = segments(seed).unwrap().unwrap();
        check_cover(seed, &segs);
        assert!(segs.len() > 3);
    }

    /// Against the real Q4_K_M file the benchmark downloads (`uv run bench/run.py --set
    /// full`), or the file `CHUNGUS_GGUF` names: `cargo test -- --ignored real_q4_k_m`.
    #[test]
    #[ignore]
    fn real_q4_k_m_file() {
        let path = std::env::var_os("CHUNGUS_GGUF")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let snapshots = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("bench/.cache/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots");
                std::fs::read_dir(snapshots)
                    .ok()?
                    .filter_map(|e| Some(e.ok()?.path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf")))
                    .find(|p| p.exists())
            })
            .expect("no Q4_K_M file: run the benchmark's full set or set CHUNGUS_GGUF");
        let data = std::fs::read(&path).unwrap();
        let segs = segments(&data).unwrap().expect("parses");
        check_cover(&data, &segs);
        let h = parse(&data).ok().unwrap();
        assert_eq!(h.tensors.len(), 291);
        let tensor_segs = segs[1..].iter().filter(|s| s.end - s.start > 32).count();
        assert_eq!(tensor_segs, 291);
    }
}
