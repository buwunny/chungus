//! Lossless byte-plane transform for float data.
//!
//! For width-w elements, byte k of every element is gathered into plane k. For BF16 this
//! puts the sign+exponent bytes together (low entropy, compress well) and the mantissa
//! bytes together (close to random). Trailing bytes that don't form a whole element are
//! copied through unchanged.

/// Split `src` into `width` byte planes.
pub fn split(src: &[u8], width: usize) -> Vec<u8> {
    if width <= 1 {
        return src.to_vec();
    }
    let n = src.len() / width;
    let mut out = vec![0u8; src.len()];
    for (k, plane) in out[..n * width].chunks_exact_mut(n).enumerate() {
        for (i, b) in plane.iter_mut().enumerate() {
            *b = src[i * width + k];
        }
    }
    out[n * width..].copy_from_slice(&src[n * width..]);
    out
}

/// Inverse of [`split`].
pub fn join(src: &[u8], width: usize) -> Vec<u8> {
    if width <= 1 {
        return src.to_vec();
    }
    let n = src.len() / width;
    let mut out = vec![0u8; src.len()];
    for (k, plane) in src[..n * width].chunks_exact(n).enumerate() {
        for (i, &b) in plane.iter().enumerate() {
            out[i * width + k] = b;
        }
    }
    out[n * width..].copy_from_slice(&src[n * width..]);
    out
}

/// Float layouts where the exponent can be pulled out into its own byte plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FloatKind {
    /// 1 sign, 8 exponent, 7 mantissa bits.
    Bf16,
    /// 1 sign, 8 exponent, 23 mantissa bits.
    F32,
}

impl FloatKind {
    pub fn width(self) -> usize {
        match self {
            FloatKind::Bf16 => 2,
            FloatKind::F32 => 4,
        }
    }
}

/// Bit-level float transform. Plain byte planes cut the exponent in two (its lowest bit
/// shares a byte with the mantissa), so this rearranges each element as
/// `[exponent byte][sign + mantissa bytes]` before splitting into planes. The exponent
/// plane then holds only the low-entropy exponents, which zstd compresses well.
pub fn split_exponent(src: &[u8], kind: FloatKind) -> Vec<u8> {
    let w = kind.width();
    let n = src.len() / w;
    let mut out = vec![0u8; src.len()];
    match kind {
        FloatKind::Bf16 => {
            for (i, e) in src[..n * 2].as_chunks::<2>().0.iter().enumerate() {
                let v = u16::from_le_bytes(*e);
                out[i] = (v >> 7) as u8;
                out[n + i] = ((v >> 15) << 7) as u8 | (v & 0x7f) as u8;
            }
        }
        FloatKind::F32 => {
            for (i, e) in src[..n * 4].as_chunks::<4>().0.iter().enumerate() {
                let v = u32::from_le_bytes(*e);
                let sm = ((v >> 31) << 23) | (v & 0x7f_ffff);
                out[i] = (v >> 23) as u8;
                out[n + i] = sm as u8;
                out[2 * n + i] = (sm >> 8) as u8;
                out[3 * n + i] = (sm >> 16) as u8;
            }
        }
    }
    out[n * w..].copy_from_slice(&src[n * w..]);
    out
}

/// Inverse of [`split_exponent`].
pub fn join_exponent(src: &[u8], kind: FloatKind) -> Vec<u8> {
    let w = kind.width();
    let n = src.len() / w;
    let mut out = vec![0u8; src.len()];
    match kind {
        FloatKind::Bf16 => {
            for (i, e) in out[..n * 2].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let (exp, sm) = (src[i] as u16, src[n + i] as u16);
                let v = ((sm >> 7) << 15) | ((exp & 0xff) << 7) | (sm & 0x7f);
                e.copy_from_slice(&v.to_le_bytes());
            }
        }
        FloatKind::F32 => {
            for (i, e) in out[..n * 4].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let sm = src[n + i] as u32
                    | (src[2 * n + i] as u32) << 8
                    | (src[3 * n + i] as u32) << 16;
                let v = ((sm >> 23) << 31) | ((src[i] as u32) << 23) | (sm & 0x7f_ffff);
                e.copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    out[n * w..].copy_from_slice(&src[n * w..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_width_and_tail() {
        let data: Vec<u8> = (0..1003u32).map(|i| (i * 31 % 251) as u8).collect();
        for w in [1, 2, 4, 8] {
            assert_eq!(join(&split(&data, w), w), data, "width {w}");
        }
    }

    #[test]
    fn exponent_split_round_trips_all_bit_patterns() {
        let bf16: Vec<u8> = (0..=u16::MAX)
            .flat_map(|v| v.to_le_bytes())
            .chain([7])
            .collect();
        assert_eq!(
            join_exponent(&split_exponent(&bf16, FloatKind::Bf16), FloatKind::Bf16),
            bf16
        );
        let f32: Vec<u8> = (0..200_000u32)
            .map(|i| i.wrapping_mul(0x9E37_79B9))
            .flat_map(|v| v.to_le_bytes())
            .chain([1, 2, 3])
            .collect();
        assert_eq!(
            join_exponent(&split_exponent(&f32, FloatKind::F32), FloatKind::F32),
            f32
        );
    }

    #[test]
    fn exponent_split_isolates_exponent() {
        // 1.0 in BF16 is 0x3F80: sign 0, exponent 127, mantissa 0.
        assert_eq!(
            split_exponent(&0x3f80u16.to_le_bytes(), FloatKind::Bf16),
            vec![127, 0]
        );
    }

    #[test]
    fn groups_bytes_by_position() {
        assert_eq!(split(&[1, 2, 3, 4, 5, 6], 2), vec![1, 3, 5, 2, 4, 6]);
    }
}
