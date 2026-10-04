//! Chunk overlap of a next checkpoint against its base, attributed to tensors, next to the unchanged element
//! fraction of each tensor.
//!
//! A chunk of the next file is *reused* when its hash appears anywhere in the base file's chunks. Each chunk's bytes
//! are credited to the rows (the header, then each tensor) whose byte range they fall in, so a chunk that spans
//! several tensors credits each one with only its own bytes. The header row plus all tensor rows sum to the file size.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::chunking::{ChunkRecord, MAXIMUM_CHUNK_BYTES, chunk_file, hash_set};
use crate::layout::{CheckpointLayout, LayoutError, TensorKind, TensorLayout, layer_of};

/// Bytes a sparse delta spends on each changed element's flat index (an `int32`, as in the delta-weight-sync blog).
pub const SPARSE_INDEX_BYTES: u64 = 4;

/// One row of the per-tensor table: the header or a single tensor of the next checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorRow {
    pub name: String,
    pub layer: Option<u32>,
    pub kind: TensorKind,
    pub dtype: String,
    pub bytes: u64,
    pub element_count: u64,
    /// Elements compared against the base. Zero when the base lacks the tensor or its dtype or shape differ.
    pub compared_elements: u64,
    pub unchanged_elements: u64,
    /// `unchanged_elements / compared_elements`, or `None` when nothing could be compared.
    pub unchanged_element_fraction: Option<f64>,
    pub chunk_overlap_bytes: u64,
    pub chunk_overlap_percent: f64,
    /// Bytes this tensor adds to a sparse delta: changed elements times (index bytes + element bytes). A tensor
    /// that cannot be compared ships in full.
    pub sparse_delta_bytes: u64,
}

/// A sum over a group of rows (one layer, one tensor kind, or the whole file).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RollupRow {
    pub group: String,
    pub bytes: u64,
    pub element_count: u64,
    pub compared_elements: u64,
    pub unchanged_elements: u64,
    pub unchanged_element_fraction: Option<f64>,
    pub chunk_overlap_bytes: u64,
    pub chunk_overlap_percent: f64,
    pub sparse_delta_bytes: u64,
}

/// Chunk counts behind the overlap numbers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChunkStats {
    pub base_chunks: u64,
    pub next_chunks: u64,
    pub next_reused_chunks: u64,
    /// Chunks of the next file cut at the maximum chunk size because no content-defined boundary appeared.
    pub next_max_size_chunks: u64,
    pub next_mean_chunk_bytes: f64,
}

/// Everything `overlap` measures for one checkpoint pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OverlapReport {
    pub tensors: Vec<TensorRow>,
    pub by_layer: Vec<RollupRow>,
    pub by_kind: Vec<RollupRow>,
    pub whole_file: RollupRow,
    pub chunk_stats: ChunkStats,
}

/// Measures the next checkpoint against the base. Both buffers must be whole safetensors files.
pub fn measure_overlap(base_file: &[u8], next_file: &[u8]) -> Result<OverlapReport, LayoutError> {
    let base_layout = CheckpointLayout::parse(base_file)?;
    let next_layout = CheckpointLayout::parse(next_file)?;
    let base_chunks = chunk_file(base_file);
    let next_chunks = chunk_file(next_file);
    let base_hashes = hash_set(&base_chunks);
    let reused: Vec<bool> = next_chunks.iter().map(|chunk| base_hashes.contains(&chunk.hash)).collect();

    let overlap_by_row = attribute_reused_bytes(&next_layout, &next_chunks, &reused);
    let mut tensors = Vec::with_capacity(next_layout.tensors.len() + 1);
    tensors.push(header_row(&next_layout, overlap_by_row[0]));
    for (tensor, &chunk_overlap_bytes) in next_layout.tensors.iter().zip(&overlap_by_row[1..]) {
        tensors.push(tensor_row(tensor, base_file, &base_layout, next_file, chunk_overlap_bytes));
    }

    let by_layer = rollup(&tensors, |row| match row.layer {
        Some(layer) => format!("layer {layer:03}"),
        None => "no layer".to_string(),
    });
    let by_kind = rollup(&tensors, |row| row.kind.label().to_string());
    let whole_file = rollup(&tensors, |_| "whole file".to_string()).remove(0);
    let next_reused_chunks = reused.iter().filter(|&&is_reused| is_reused).count() as u64;
    let chunk_stats = ChunkStats {
        base_chunks: base_chunks.len() as u64,
        next_chunks: next_chunks.len() as u64,
        next_reused_chunks,
        next_max_size_chunks: next_chunks.iter().filter(|chunk| chunk.bytes == MAXIMUM_CHUNK_BYTES).count() as u64,
        next_mean_chunk_bytes: mean_chunk_bytes(&next_chunks),
    };
    Ok(OverlapReport {
        tensors,
        by_layer,
        by_kind,
        whole_file,
        chunk_stats,
    })
}

/// Mean chunk size in bytes, or 0 for an empty file.
pub fn mean_chunk_bytes(chunks: &[ChunkRecord]) -> f64 {
    if chunks.is_empty() {
        return 0.0;
    }
    chunks.iter().map(|chunk| chunk.bytes).sum::<u64>() as f64 / chunks.len() as f64
}

/// Credits each reused chunk's bytes to the rows its range covers. Row 0 is the header and row `i + 1` is tensor `i`
/// of `layout`. `chunks` must tile the file in order, as [`chunk_file`] returns them.
pub fn attribute_reused_bytes(layout: &CheckpointLayout, chunks: &[ChunkRecord], reused: &[bool]) -> Vec<u64> {
    let row_ends: Vec<u64> = std::iter::once(layout.header_bytes)
        .chain(layout.tensors.iter().map(|tensor| tensor.end_byte))
        .collect();
    let mut overlap_by_row = vec![0u64; row_ends.len()];
    let mut row_index = 0;
    for (chunk, &is_reused) in chunks.iter().zip(reused) {
        let mut cursor = chunk.start_byte;
        while cursor < chunk.end_byte() {
            while row_ends[row_index] <= cursor {
                row_index += 1;
            }
            let covered_end = chunk.end_byte().min(row_ends[row_index]);
            if is_reused {
                overlap_by_row[row_index] += covered_end - cursor;
            }
            cursor = covered_end;
        }
    }
    overlap_by_row
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        100.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn fraction(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

fn header_row(layout: &CheckpointLayout, chunk_overlap_bytes: u64) -> TensorRow {
    TensorRow {
        name: "(header)".to_string(),
        layer: None,
        kind: TensorKind::Header,
        dtype: "-".to_string(),
        bytes: layout.header_bytes,
        element_count: 0,
        compared_elements: 0,
        unchanged_elements: 0,
        unchanged_element_fraction: None,
        chunk_overlap_bytes,
        chunk_overlap_percent: percent(chunk_overlap_bytes, layout.header_bytes),
        sparse_delta_bytes: 0,
    }
}

fn tensor_row(
    tensor: &TensorLayout,
    base_file: &[u8],
    base_layout: &CheckpointLayout,
    next_file: &[u8],
    chunk_overlap_bytes: u64,
) -> TensorRow {
    let element_count = tensor.element_count();
    let comparison = base_layout
        .tensor(&tensor.name)
        .filter(|base_tensor| base_tensor.dtype == tensor.dtype && base_tensor.shape == tensor.shape)
        .zip(tensor.element_bytes())
        .map(|(base_tensor, element_bytes)| {
            let base_bytes = &base_file[base_tensor.start_byte as usize..base_tensor.end_byte as usize];
            let next_bytes = &next_file[tensor.start_byte as usize..tensor.end_byte as usize];
            count_unchanged_elements(base_bytes, next_bytes, element_bytes as usize)
        });
    let (compared_elements, unchanged_elements) = match comparison {
        Some(unchanged) => (element_count, unchanged),
        None => (0, 0),
    };
    let sparse_delta_bytes = match (comparison, tensor.element_bytes()) {
        (Some(unchanged), Some(element_bytes)) => (element_count - unchanged) * (SPARSE_INDEX_BYTES + element_bytes),
        (None, Some(element_bytes)) => element_count * (SPARSE_INDEX_BYTES + element_bytes),
        (_, None) => tensor.bytes(),
    };
    TensorRow {
        name: tensor.name.clone(),
        layer: layer_of(&tensor.name),
        kind: TensorKind::of(&tensor.name),
        dtype: tensor.dtype.clone(),
        bytes: tensor.bytes(),
        element_count,
        compared_elements,
        unchanged_elements,
        unchanged_element_fraction: fraction(unchanged_elements, compared_elements),
        chunk_overlap_bytes,
        chunk_overlap_percent: percent(chunk_overlap_bytes, tensor.bytes()),
        sparse_delta_bytes,
    }
}

/// Counts elements whose bytes are identical in both buffers. Both buffers hold the same number of elements.
fn count_unchanged_elements(base_bytes: &[u8], next_bytes: &[u8], element_bytes: usize) -> u64 {
    base_bytes
        .chunks_exact(element_bytes)
        .zip(next_bytes.chunks_exact(element_bytes))
        .filter(|(base_element, next_element)| base_element == next_element)
        .count() as u64
}

fn rollup(rows: &[TensorRow], group_of: impl Fn(&TensorRow) -> String) -> Vec<RollupRow> {
    let mut groups: BTreeMap<String, RollupRow> = BTreeMap::new();
    for row in rows {
        let group = group_of(row);
        let sum = groups.entry(group.clone()).or_insert_with(|| RollupRow {
            group,
            bytes: 0,
            element_count: 0,
            compared_elements: 0,
            unchanged_elements: 0,
            unchanged_element_fraction: None,
            chunk_overlap_bytes: 0,
            chunk_overlap_percent: 0.0,
            sparse_delta_bytes: 0,
        });
        sum.bytes += row.bytes;
        sum.element_count += row.element_count;
        sum.compared_elements += row.compared_elements;
        sum.unchanged_elements += row.unchanged_elements;
        sum.chunk_overlap_bytes += row.chunk_overlap_bytes;
        sum.sparse_delta_bytes += row.sparse_delta_bytes;
    }
    groups
        .into_values()
        .map(|mut sum| {
            sum.unchanged_element_fraction = fraction(sum.unchanged_elements, sum.compared_elements);
            sum.chunk_overlap_percent = percent(sum.chunk_overlap_bytes, sum.bytes);
            sum
        })
        .collect()
}
