//! Segments are byte ranges of a file with a known element type. Chunking happens
//! inside segments so chunk boundaries never straddle two tensors.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// Unknown or non-float bytes: no byte-plane transform.
    Raw,
    Bf16,
    F16,
    F32,
    F64,
}

impl Dtype {
    pub fn from_safetensors(s: &str) -> Self {
        match s {
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            _ => Dtype::Raw,
        }
    }

    /// Layout for the exponent-split transform, where one applies.
    pub fn float_kind(self) -> Option<crate::transform::FloatKind> {
        match self {
            Dtype::Bf16 => Some(crate::transform::FloatKind::Bf16),
            Dtype::F32 => Some(crate::transform::FloatKind::F32),
            _ => None,
        }
    }

    /// Element width in bytes, used both for the byte-plane transform and to keep
    /// chunk boundaries on element boundaries. `Raw` is 1.
    pub fn width(self) -> usize {
        match self {
            Dtype::Raw => 1,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub start: u64,
    pub end: u64,
    pub dtype: Dtype,
}
