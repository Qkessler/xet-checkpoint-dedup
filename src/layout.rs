//! Safetensors byte layout: where the header ends and where each tensor's bytes live in the file.
//!
//! A safetensors file is an 8-byte little-endian header length `N`, then `N` bytes of JSON, then the data section.
//! Each tensor's `data_offsets` are relative to the start of the data section, so its absolute byte range is
//! `8 + N + data_offsets`. [`CheckpointLayout::parse`] rejects files whose tensors overlap or leave holes in the data
//! section, so the header plus every tensor range covers the file exactly once.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Bytes taken by the little-endian `u64` header length at the start of every safetensors file.
pub const HEADER_LENGTH_PREFIX_BYTES: u64 = 8;

/// Reasons a byte buffer is not a safetensors file this tool can attribute.
#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("file is {file_bytes} bytes, too short for a safetensors header")]
    TooShort { file_bytes: u64 },
    #[error("header claims {header_json_bytes} bytes but the file is only {file_bytes} bytes")]
    HeaderPastEnd { header_json_bytes: u64, file_bytes: u64 },
    #[error("header is not valid JSON: {0}")]
    HeaderJson(#[from] serde_json::Error),
    #[error(
        "tensor {name} has data_offsets [{start_offset_bytes}, {end_offset_bytes}) that are reversed or past the end \
         of the data section"
    )]
    BadOffsets {
        name: String,
        start_offset_bytes: u64,
        end_offset_bytes: u64,
    },
    #[error(
        "tensor {name} starts at data offset {start_offset_bytes} but the previous tensor ends at {expected_offset_bytes}"
    )]
    NotContiguous {
        name: String,
        start_offset_bytes: u64,
        expected_offset_bytes: u64,
    },
    #[error("tensors end at data offset {covered_bytes} but the data section is {data_bytes} bytes")]
    TrailingBytes { covered_bytes: u64, data_bytes: u64 },
}

#[derive(Debug, Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// One tensor's identity and absolute byte range in its file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorLayout {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute offset of the tensor's first byte in the file.
    pub start_byte: u64,
    /// Absolute offset one past the tensor's last byte in the file.
    pub end_byte: u64,
}

impl TensorLayout {
    /// Size of the tensor's byte range.
    pub fn bytes(&self) -> u64 {
        self.end_byte - self.start_byte
    }

    /// Number of elements: the product of the shape, or 1 for a 0-d tensor.
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Bytes per element for the safetensors dtype, or `None` for dtypes this tool does not compare element-wise.
    pub fn element_bytes(&self) -> Option<u64> {
        dtype_element_bytes(&self.dtype)
    }
}

/// Bytes per element of a safetensors dtype string, or `None` when the dtype has no whole-byte element size.
pub fn dtype_element_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "U16" | "I16" | "F16" | "BF16" => Some(2),
        "U32" | "I32" | "F32" => Some(4),
        "U64" | "I64" | "F64" => Some(8),
        _ => None,
    }
}

/// The header and tensor byte ranges of one safetensors file, with tensors sorted by position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckpointLayout {
    pub file_bytes: u64,
    /// Bytes before the data section: the 8-byte length prefix plus the JSON header, including its padding.
    pub header_bytes: u64,
    pub tensors: Vec<TensorLayout>,
}

impl CheckpointLayout {
    /// Parses the header of a whole safetensors file held in memory.
    ///
    /// Fails when the file is shorter than its header claims, the header is not the expected JSON, or the tensors
    /// overlap, leave a hole, or stop short of the end of the file.
    pub fn parse(file: &[u8]) -> Result<Self, LayoutError> {
        let file_bytes = file.len() as u64;
        if file_bytes < HEADER_LENGTH_PREFIX_BYTES {
            return Err(LayoutError::TooShort { file_bytes });
        }
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&file[..8]);
        let header_json_bytes = u64::from_le_bytes(prefix);
        let header_bytes = HEADER_LENGTH_PREFIX_BYTES
            .checked_add(header_json_bytes)
            .filter(|&end| end <= file_bytes)
            .ok_or(LayoutError::HeaderPastEnd {
                header_json_bytes,
                file_bytes,
            })?;
        let header_json = &file[8..header_bytes as usize];
        let mut entries: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header_json)?;
        entries.remove("__metadata__");

        let data_bytes = file_bytes - header_bytes;
        let mut tensors = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            let raw: RawTensorEntry = serde_json::from_value(value)?;
            let [start, end] = raw.data_offsets;
            if start > end || end > data_bytes {
                return Err(LayoutError::BadOffsets {
                    name,
                    start_offset_bytes: start,
                    end_offset_bytes: end,
                });
            }
            tensors.push(TensorLayout {
                name,
                dtype: raw.dtype,
                shape: raw.shape,
                start_byte: header_bytes + start,
                end_byte: header_bytes + end,
            });
        }
        tensors.sort_by_key(|tensor| (tensor.start_byte, tensor.end_byte));

        let mut covered = header_bytes;
        for tensor in &tensors {
            if tensor.start_byte != covered {
                return Err(LayoutError::NotContiguous {
                    name: tensor.name.clone(),
                    start_offset_bytes: tensor.start_byte - header_bytes,
                    expected_offset_bytes: covered - header_bytes,
                });
            }
            covered = tensor.end_byte;
        }
        if covered != file_bytes {
            return Err(LayoutError::TrailingBytes {
                covered_bytes: covered - header_bytes,
                data_bytes,
            });
        }

        Ok(Self {
            file_bytes,
            header_bytes,
            tensors,
        })
    }

    /// The tensor with this name, if the file has one.
    pub fn tensor(&self, name: &str) -> Option<&TensorLayout> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// The role a row plays regardless of layer: one of the tensor kinds in CONTEXT.md, or the file header, which gets
/// its own row so that attributed bytes add up to the file size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorKind {
    Header,
    Embedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    MlpGate,
    MlpUp,
    MlpDown,
    Norm,
    OutputHead,
    Other,
}

impl TensorKind {
    /// Classifies a tensor by its Hugging Face tensor name (Llama, Qwen2 and Mistral naming).
    pub fn of(name: &str) -> Self {
        let rules: [(&str, TensorKind); 10] = [
            ("embed_tokens", TensorKind::Embedding),
            ("q_proj", TensorKind::AttentionQ),
            ("k_proj", TensorKind::AttentionK),
            ("v_proj", TensorKind::AttentionV),
            ("o_proj", TensorKind::AttentionO),
            ("gate_proj", TensorKind::MlpGate),
            ("up_proj", TensorKind::MlpUp),
            ("down_proj", TensorKind::MlpDown),
            ("norm", TensorKind::Norm),
            ("lm_head", TensorKind::OutputHead),
        ];
        rules
            .iter()
            .find(|(fragment, _)| name.contains(fragment))
            .map(|&(_, kind)| kind)
            .unwrap_or(TensorKind::Other)
    }

    /// Short name used in printed tables and roll-up groups. Matches the serialized form.
    pub fn label(self) -> &'static str {
        match self {
            TensorKind::Header => "header",
            TensorKind::Embedding => "embedding",
            TensorKind::AttentionQ => "attention_q",
            TensorKind::AttentionK => "attention_k",
            TensorKind::AttentionV => "attention_v",
            TensorKind::AttentionO => "attention_o",
            TensorKind::MlpGate => "mlp_gate",
            TensorKind::MlpUp => "mlp_up",
            TensorKind::MlpDown => "mlp_down",
            TensorKind::Norm => "norm",
            TensorKind::OutputHead => "output_head",
            TensorKind::Other => "other",
        }
    }
}

/// The layer index of a tensor named `...layers.N....`, or `None` for tensors outside the repeated blocks.
pub fn layer_of(name: &str) -> Option<u32> {
    let after = name.split("layers.").nth(1)?;
    after.split('.').next()?.parse().ok()
}

#[cfg(test)]
pub mod test_support {
    //! Builders for tiny synthetic safetensors files used by the unit and property tests.

    use serde_json::json;

    /// A tensor to place in a synthetic file: name, dtype, shape, and raw little-endian bytes.
    #[derive(Debug, Clone)]
    pub struct SyntheticTensor {
        pub name: String,
        pub dtype: &'static str,
        pub shape: Vec<u64>,
        pub data: Vec<u8>,
    }

    /// Serializes tensors into a safetensors file, writing the header keys and the data in the order given.
    pub fn build_safetensors(tensors: &[SyntheticTensor]) -> Vec<u8> {
        build_safetensors_with_key_order(tensors, &(0..tensors.len()).collect::<Vec<_>>())
    }

    /// Like [`build_safetensors`], but writes the JSON header keys in `key_order` while keeping the data order.
    pub fn build_safetensors_with_key_order(tensors: &[SyntheticTensor], key_order: &[usize]) -> Vec<u8> {
        build_safetensors_with_header(tensors, key_order, None)
    }

    /// Writes the header keys in `key_order`, with an optional `__metadata__` entry first, then the data in the
    /// order given.
    pub fn build_safetensors_with_header(
        tensors: &[SyntheticTensor],
        key_order: &[usize],
        metadata: Option<&str>,
    ) -> Vec<u8> {
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut cursor = 0u64;
        for tensor in tensors {
            offsets.push([cursor, cursor + tensor.data.len() as u64]);
            cursor += tensor.data.len() as u64;
        }
        let mut entries: Vec<String> = metadata
            .map(|value| format!("\"__metadata__\":{}", json!({ "note": value })))
            .into_iter()
            .collect();
        for &index in key_order {
            let tensor = &tensors[index];
            let entry = json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": offsets[index]});
            entries.push(format!("{}:{}", json!(tensor.name), entry));
        }
        let mut header = format!("{{{}", entries.join(","));
        header.push('}');
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        for tensor in tensors {
            file.extend_from_slice(&tensor.data);
        }
        file
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    fn tensor(name: &str, rows: u64, columns: u64) -> SyntheticTensor {
        SyntheticTensor {
            name: name.to_string(),
            dtype: "BF16",
            shape: vec![rows, columns],
            data: vec![7u8; (rows * columns * 2) as usize],
        }
    }

    #[test]
    fn parses_absolute_ranges_sorted_by_position() {
        let file = build_safetensors_with_key_order(
            &[
                tensor("model.layers.0.self_attn.q_proj.weight", 2, 3),
                tensor("model.norm.weight", 1, 4),
            ],
            &[1, 0],
        );
        let layout = CheckpointLayout::parse(&file).unwrap();
        assert_eq!(layout.tensors[0].name, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(layout.tensors[0].start_byte, layout.header_bytes);
        assert_eq!(layout.tensors[0].bytes(), 12);
        assert_eq!(layout.tensors[1].start_byte, layout.header_bytes + 12);
        assert_eq!(layout.tensors[1].end_byte, layout.file_bytes);
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut file = build_safetensors(&[tensor("a", 1, 2)]);
        file.push(0);
        assert!(matches!(CheckpointLayout::parse(&file), Err(LayoutError::TrailingBytes { .. })));
    }

    #[test]
    fn rejects_header_longer_than_file() {
        let mut file = 1000u64.to_le_bytes().to_vec();
        file.extend_from_slice(b"{}");
        assert!(matches!(CheckpointLayout::parse(&file), Err(LayoutError::HeaderPastEnd { .. })));
    }

    #[test]
    fn classifies_qwen_tensor_names() {
        assert_eq!(TensorKind::of("model.embed_tokens.weight"), TensorKind::Embedding);
        assert_eq!(TensorKind::of("model.layers.3.self_attn.k_proj.bias"), TensorKind::AttentionK);
        assert_eq!(TensorKind::of("model.layers.3.mlp.down_proj.weight"), TensorKind::MlpDown);
        assert_eq!(TensorKind::of("model.layers.3.post_attention_layernorm.weight"), TensorKind::Norm);
        assert_eq!(TensorKind::of("model.norm.weight"), TensorKind::Norm);
        assert_eq!(TensorKind::of("lm_head.weight"), TensorKind::OutputHead);
        assert_eq!(layer_of("model.layers.17.mlp.up_proj.weight"), Some(17));
        assert_eq!(layer_of("model.embed_tokens.weight"), None);
    }
}
