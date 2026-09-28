//! Lossless byte-plane transform for float data.
//!
//! For width-w elements, byte k of every element is gathered into plane k. For BF16 this
//! puts the sign+exponent bytes together (low entropy, compress well) and the mantissa
//! bytes together (close to random). Trailing bytes that don't form a whole element are
//! copied through unchanged.
//!
//! Widths 2 and 4 (and the exponent transforms built on them) run SIMD kernels: SSSE3 on
//! x86-64 when the CPU has it, NEON on AArch64. Each kernel handles whole groups of 16
//! elements and the scalar code finishes the rest, so every path produces the same bytes.

/// Split `src` into `width` byte planes.
pub fn split(src: &[u8], width: usize) -> Vec<u8> {
    if width <= 1 {
        return src.to_vec();
    }
    let n = src.len() / width;
    let mut out = vec![0u8; src.len()];
    let done = match width {
        2 => simd::split2(src, &mut out, n, false),
        4 => simd::split4(src, &mut out, n, false),
        _ => 0,
    };
    for k in 0..width {
        for i in done..n {
            out[k * n + i] = src[i * width + k];
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
    let done = match width {
        2 => simd::join2(src, &mut out, n, false),
        4 => simd::join4(src, &mut out, n, false),
        _ => 0,
    };
    for k in 0..width {
        for i in done..n {
            out[i * width + k] = src[k * n + i];
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
            let done = simd::split2(src, &mut out, n, true);
            for i in done..n {
                let v = u16::from_le_bytes([src[2 * i], src[2 * i + 1]]);
                out[i] = (v >> 7) as u8;
                out[n + i] = ((v >> 15) << 7) as u8 | (v & 0x7f) as u8;
            }
        }
        FloatKind::F32 => {
            let done = simd::split4(src, &mut out, n, true);
            for i in done..n {
                let v = u32::from_le_bytes(src[4 * i..4 * i + 4].try_into().unwrap());
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
            let done = simd::join2(src, &mut out, n, true);
            for i in done..n {
                let (exp, sm) = (src[i] as u16, src[n + i] as u16);
                let v = ((sm >> 7) << 15) | (exp << 7) | (sm & 0x7f);
                out[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        FloatKind::F32 => {
            let done = simd::join4(src, &mut out, n, true);
            for i in done..n {
                let sm = src[n + i] as u32
                    | (src[2 * n + i] as u32) << 8
                    | (src[3 * n + i] as u32) << 16;
                let v = ((sm >> 23) << 31) | ((src[i] as u32) << 23) | (sm & 0x7f_ffff);
                out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    out[n * w..].copy_from_slice(&src[n * w..]);
    out
}

/// SIMD kernels. Each takes the source, the output and the element count `n`, handles
/// the first elements in groups of 16, and returns how many it handled. With `exponent`
/// set, it also applies the float rearrangement of [`split_exponent`] (for width 2, BF16;
/// for width 4, F32).
///
/// The rearrangement works byte by byte. With a float's little-endian bytes `b0..`:
/// - BF16: exponent = `b1 << 1 | b0 >> 7`, sign+mantissa = `b1 & 0x80 | b0 & 0x7f`.
/// - F32: exponent = `b3 << 1 | b2 >> 7`, then `b0`, `b1`, and `b3 & 0x80 | b2 & 0x7f`.
///
/// So after splitting into byte planes, the exponent transform only mixes the last two
/// planes, and undoing it is the same kind of mix before joining.
mod simd {
    #[cfg(target_arch = "x86_64")]
    pub use x86::*;

    #[cfg(target_arch = "aarch64")]
    pub use neon::*;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    mod fallback {
        pub fn split2(_: &[u8], _: &mut [u8], _: usize, _: bool) -> usize {
            0
        }
        pub fn join2(_: &[u8], _: &mut [u8], _: usize, _: bool) -> usize {
            0
        }
        pub fn split4(_: &[u8], _: &mut [u8], _: usize, _: bool) -> usize {
            0
        }
        pub fn join4(_: &[u8], _: &mut [u8], _: usize, _: bool) -> usize {
            0
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub use fallback::*;

    #[cfg(target_arch = "x86_64")]
    mod x86 {
        use std::arch::x86_64::*;

        fn enabled() -> bool {
            is_x86_feature_detected!("ssse3")
        }

        pub fn split2(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            if !enabled() {
                return 0;
            }
            assert!(src.len() >= 2 * n && out.len() >= 2 * n);
            // SAFETY: ssse3 is present, and the kernel stays within 2n bytes of each.
            unsafe { split2_ssse3(src.as_ptr(), out.as_mut_ptr(), n, exponent) }
        }

        pub fn join2(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            if !enabled() {
                return 0;
            }
            assert!(src.len() >= 2 * n && out.len() >= 2 * n);
            // SAFETY: as above.
            unsafe { join2_ssse3(src.as_ptr(), out.as_mut_ptr(), n, exponent) }
        }

        pub fn split4(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            if !enabled() {
                return 0;
            }
            assert!(src.len() >= 4 * n && out.len() >= 4 * n);
            // SAFETY: as above, within 4n bytes.
            unsafe { split4_ssse3(src.as_ptr(), out.as_mut_ptr(), n, exponent) }
        }

        pub fn join4(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            if !enabled() {
                return 0;
            }
            assert!(src.len() >= 4 * n && out.len() >= 4 * n);
            // SAFETY: as above.
            unsafe { join4_ssse3(src.as_ptr(), out.as_mut_ptr(), n, exponent) }
        }

        /// The exponent mix on two byte vectors `(lo, hi)`: returns
        /// `(hi << 1 | lo >> 7, hi & 0x80 | lo & 0x7f)`. x86 has no byte shifts, so shift
        /// 16-bit lanes and mask off what crossed between bytes.
        #[target_feature(enable = "ssse3")]
        fn mix(lo: __m128i, hi: __m128i) -> (__m128i, __m128i) {
            let exp = _mm_or_si128(
                _mm_and_si128(_mm_slli_epi16(hi, 1), _mm_set1_epi8(0xfeu8 as i8)),
                _mm_and_si128(_mm_srli_epi16(lo, 7), _mm_set1_epi8(0x01)),
            );
            let sm = _mm_or_si128(
                _mm_and_si128(hi, _mm_set1_epi8(0x80u8 as i8)),
                _mm_and_si128(lo, _mm_set1_epi8(0x7f)),
            );
            (exp, sm)
        }

        /// Inverse of [`mix`]: from `(exp, sm)` returns
        /// `(lo, hi) = (exp << 7 | sm & 0x7f, sm & 0x80 | exp >> 1)`.
        #[target_feature(enable = "ssse3")]
        fn unmix(exp: __m128i, sm: __m128i) -> (__m128i, __m128i) {
            let lo = _mm_or_si128(
                _mm_and_si128(_mm_slli_epi16(exp, 7), _mm_set1_epi8(0x80u8 as i8)),
                _mm_and_si128(sm, _mm_set1_epi8(0x7f)),
            );
            let hi = _mm_or_si128(
                _mm_and_si128(sm, _mm_set1_epi8(0x80u8 as i8)),
                _mm_and_si128(_mm_srli_epi16(exp, 1), _mm_set1_epi8(0x7f)),
            );
            (lo, hi)
        }

        #[target_feature(enable = "ssse3")]
        unsafe fn split2_ssse3(src: *const u8, out: *mut u8, n: usize, exponent: bool) -> usize {
            // Gather the even bytes into the low half and the odd bytes into the high half.
            let deinterleave = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
            let end = n / 16 * 16;
            let mut i = 0;
            while i < end {
                // SAFETY: i + 16 <= n, so bytes 2i..2i+32 of src and i..i+16, n+i..n+i+16
                // of out are in bounds.
                unsafe {
                    let a = _mm_shuffle_epi8(_mm_loadu_si128(src.add(2 * i).cast()), deinterleave);
                    let b =
                        _mm_shuffle_epi8(_mm_loadu_si128(src.add(2 * i + 16).cast()), deinterleave);
                    let (mut p0, mut p1) = (_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b));
                    if exponent {
                        (p0, p1) = mix(p0, p1);
                    }
                    _mm_storeu_si128(out.add(i).cast(), p0);
                    _mm_storeu_si128(out.add(n + i).cast(), p1);
                }
                i += 16;
            }
            end
        }

        #[target_feature(enable = "ssse3")]
        unsafe fn join2_ssse3(src: *const u8, out: *mut u8, n: usize, exponent: bool) -> usize {
            let end = n / 16 * 16;
            let mut i = 0;
            while i < end {
                // SAFETY: as in split2_ssse3, with src and out swapped.
                unsafe {
                    let mut p0 = _mm_loadu_si128(src.add(i).cast());
                    let mut p1 = _mm_loadu_si128(src.add(n + i).cast());
                    if exponent {
                        (p0, p1) = unmix(p0, p1);
                    }
                    _mm_storeu_si128(out.add(2 * i).cast(), _mm_unpacklo_epi8(p0, p1));
                    _mm_storeu_si128(out.add(2 * i + 16).cast(), _mm_unpackhi_epi8(p0, p1));
                }
                i += 16;
            }
            end
        }

        #[target_feature(enable = "ssse3")]
        unsafe fn split4_ssse3(src: *const u8, out: *mut u8, n: usize, exponent: bool) -> usize {
            // Within each 4-element vector, gather byte k of every element into dword k.
            let transpose = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
            let end = n / 16 * 16;
            let mut i = 0;
            while i < end {
                // SAFETY: i + 16 <= n, so bytes 4i..4i+64 of src and k*n+i..k*n+i+16 of
                // out are in bounds.
                unsafe {
                    let load = |j: usize| {
                        _mm_shuffle_epi8(_mm_loadu_si128(src.add(4 * i + 16 * j).cast()), transpose)
                    };
                    let (x0, x1, x2, x3) = (load(0), load(1), load(2), load(3));
                    // A 4x4 transpose of dwords across the four vectors.
                    let t0 = _mm_unpacklo_epi32(x0, x1);
                    let t1 = _mm_unpacklo_epi32(x2, x3);
                    let t2 = _mm_unpackhi_epi32(x0, x1);
                    let t3 = _mm_unpackhi_epi32(x2, x3);
                    let p0 = _mm_unpacklo_epi64(t0, t1);
                    let p1 = _mm_unpackhi_epi64(t0, t1);
                    let (mut p2, mut p3) = (_mm_unpacklo_epi64(t2, t3), _mm_unpackhi_epi64(t2, t3));
                    let planes = if exponent {
                        (p2, p3) = mix(p2, p3);
                        [p2, p0, p1, p3]
                    } else {
                        [p0, p1, p2, p3]
                    };
                    for (k, p) in planes.into_iter().enumerate() {
                        _mm_storeu_si128(out.add(k * n + i).cast(), p);
                    }
                }
                i += 16;
            }
            end
        }

        #[target_feature(enable = "ssse3")]
        unsafe fn join4_ssse3(src: *const u8, out: *mut u8, n: usize, exponent: bool) -> usize {
            let end = n / 16 * 16;
            let mut i = 0;
            while i < end {
                // SAFETY: as in split4_ssse3, with src and out swapped.
                unsafe {
                    let load = |k: usize| _mm_loadu_si128(src.add(k * n + i).cast());
                    let [p0, p1, p2, p3] = if exponent {
                        let (b2, b3) = unmix(load(0), load(3));
                        [load(1), load(2), b2, b3]
                    } else {
                        [load(0), load(1), load(2), load(3)]
                    };
                    let t0 = _mm_unpacklo_epi8(p0, p1);
                    let t1 = _mm_unpackhi_epi8(p0, p1);
                    let t2 = _mm_unpacklo_epi8(p2, p3);
                    let t3 = _mm_unpackhi_epi8(p2, p3);
                    let o = out.add(4 * i);
                    _mm_storeu_si128(o.cast(), _mm_unpacklo_epi16(t0, t2));
                    _mm_storeu_si128(o.add(16).cast(), _mm_unpackhi_epi16(t0, t2));
                    _mm_storeu_si128(o.add(32).cast(), _mm_unpacklo_epi16(t1, t3));
                    _mm_storeu_si128(o.add(48).cast(), _mm_unpackhi_epi16(t1, t3));
                }
                i += 16;
            }
            end
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use std::arch::aarch64::*;

        /// See the x86 `mix`: `(hi << 1 | lo >> 7, hi & 0x80 | lo & 0x7f)`.
        #[inline(always)]
        unsafe fn mix(lo: uint8x16_t, hi: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
            // SAFETY: NEON is always present on AArch64.
            unsafe {
                let exp = vsriq_n_u8::<7>(vshlq_n_u8::<1>(hi), lo);
                let sm = vbslq_u8(vdupq_n_u8(0x80), hi, lo);
                (exp, sm)
            }
        }

        /// Inverse of [`mix`]: `(exp << 7 | sm & 0x7f, sm & 0x80 | exp >> 1)`.
        #[inline(always)]
        unsafe fn unmix(exp: uint8x16_t, sm: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
            // SAFETY: as above.
            unsafe {
                let lo = vbslq_u8(vdupq_n_u8(0x80), vshlq_n_u8::<7>(exp), sm);
                let hi = vbslq_u8(vdupq_n_u8(0x80), sm, vshrq_n_u8::<1>(exp));
                (lo, hi)
            }
        }

        pub fn split2(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            assert!(src.len() >= 2 * n && out.len() >= 2 * n);
            let end = n / 16 * 16;
            for i in (0..end).step_by(16) {
                // SAFETY: i + 16 <= n, so every access is in bounds.
                unsafe {
                    let v = vld2q_u8(src.as_ptr().add(2 * i));
                    let (p0, p1) = if exponent { mix(v.0, v.1) } else { (v.0, v.1) };
                    vst1q_u8(out.as_mut_ptr().add(i), p0);
                    vst1q_u8(out.as_mut_ptr().add(n + i), p1);
                }
            }
            end
        }

        pub fn join2(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            assert!(src.len() >= 2 * n && out.len() >= 2 * n);
            let end = n / 16 * 16;
            for i in (0..end).step_by(16) {
                // SAFETY: as above.
                unsafe {
                    let p0 = vld1q_u8(src.as_ptr().add(i));
                    let p1 = vld1q_u8(src.as_ptr().add(n + i));
                    let (b0, b1) = if exponent { unmix(p0, p1) } else { (p0, p1) };
                    vst2q_u8(out.as_mut_ptr().add(2 * i), uint8x16x2_t(b0, b1));
                }
            }
            end
        }

        pub fn split4(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            assert!(src.len() >= 4 * n && out.len() >= 4 * n);
            let end = n / 16 * 16;
            for i in (0..end).step_by(16) {
                // SAFETY: as above, within 4n bytes.
                unsafe {
                    let v = vld4q_u8(src.as_ptr().add(4 * i));
                    let planes = if exponent {
                        let (exp, sm2) = mix(v.2, v.3);
                        [exp, v.0, v.1, sm2]
                    } else {
                        [v.0, v.1, v.2, v.3]
                    };
                    for (k, p) in planes.into_iter().enumerate() {
                        vst1q_u8(out.as_mut_ptr().add(k * n + i), p);
                    }
                }
            }
            end
        }

        pub fn join4(src: &[u8], out: &mut [u8], n: usize, exponent: bool) -> usize {
            assert!(src.len() >= 4 * n && out.len() >= 4 * n);
            let end = n / 16 * 16;
            for i in (0..end).step_by(16) {
                // SAFETY: as above.
                unsafe {
                    let load = |k: usize| vld1q_u8(src.as_ptr().add(k * n + i));
                    let v = if exponent {
                        let (b2, b3) = unmix(load(0), load(3));
                        uint8x16x4_t(load(1), load(2), b2, b3)
                    } else {
                        uint8x16x4_t(load(0), load(1), load(2), load(3))
                    };
                    vst4q_u8(out.as_mut_ptr().add(4 * i), v);
                }
            }
            end
        }
    }
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

    /// The original scalar transforms. The SIMD paths must produce exactly these bytes,
    /// or chunks already stored would decode differently.
    mod reference {
        use super::FloatKind;

        pub fn split(src: &[u8], w: usize) -> Vec<u8> {
            let n = src.len() / w;
            let mut out = src.to_vec();
            for k in 0..w {
                for i in 0..n {
                    out[k * n + i] = src[i * w + k];
                }
            }
            out
        }

        pub fn split_exponent(src: &[u8], kind: FloatKind) -> Vec<u8> {
            let w = kind.width();
            let n = src.len() / w;
            let mut out = src.to_vec();
            for i in 0..n {
                let e = &src[i * w..(i + 1) * w];
                if kind == FloatKind::Bf16 {
                    let v = u16::from_le_bytes([e[0], e[1]]);
                    out[i] = (v >> 7) as u8;
                    out[n + i] = ((v >> 15) << 7) as u8 | (v & 0x7f) as u8;
                } else {
                    let v = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                    let sm = ((v >> 31) << 23) | (v & 0x7f_ffff);
                    out[i] = (v >> 23) as u8;
                    out[n + i] = sm as u8;
                    out[2 * n + i] = (sm >> 8) as u8;
                    out[3 * n + i] = (sm >> 16) as u8;
                }
            }
            out
        }
    }

    #[test]
    fn simd_matches_the_scalar_format() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let data: Vec<u8> = (0..70_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 56) as u8
            })
            .collect();
        // Every length up to a few SIMD groups, so each tail size is covered, plus a
        // chunk-sized one.
        for len in (0..300).chain([65_536, 70_000]) {
            let d = &data[..len];
            for w in [2, 4, 8] {
                let planes = split(d, w);
                assert_eq!(planes, reference::split(d, w), "split {w} of {len}");
                assert_eq!(join(&planes, w), d, "join {w} of {len}");
            }
            for kind in [FloatKind::Bf16, FloatKind::F32] {
                let planes = split_exponent(d, kind);
                assert_eq!(
                    planes,
                    reference::split_exponent(d, kind),
                    "{kind:?} of {len}"
                );
                assert_eq!(join_exponent(&planes, kind), d, "{kind:?} join of {len}");
            }
        }
    }

    #[test]
    fn groups_bytes_by_position() {
        assert_eq!(split(&[1, 2, 3, 4, 5, 6], 2), vec![1, 3, 5, 2, 4, 6]);
    }
}
