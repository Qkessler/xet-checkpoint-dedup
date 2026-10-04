//! Perturbation sweep: chunk overlap against change density, for scattered and clustered perturbations.
//!
//! Each point perturbs an in-memory copy of the base checkpoint with [`perturb`] and chunks it, so a sweep of
//! hundreds of points never writes a perturbed file. `perturb` on the command line produces the same bytes for the
//! same density, mode and seed, which the outer loop checks.
//!
//! The theory curve for scattered changes is `exp(-values_per_chunk * density)`, with `values_per_chunk` the base's
//! mean chunk size over 2 bytes: the chance that none of a chunk's elements changed. The size-aware curve weights the
//! same chance by each actual base chunk's bytes, `sum(bytes * (1 - density)^(bytes / 2)) / sum(bytes)`, which is
//! what a byte-weighted overlap converges to when chunk sizes vary.

use std::collections::HashSet;
use std::sync::Mutex;

use serde::Serialize;
use xet_core_structures::merklehash::MerkleHash;

use crate::chunking::{ChunkRecord, MAXIMUM_CHUNK_BYTES, chunk_file, hash_set};
use crate::layout::{CheckpointLayout, LayoutError};
use crate::overlap::mean_chunk_bytes;
use crate::perturb::{PerturbError, PerturbMode, perturb};

/// One finished job: its position in the job list and its point, or the error that stopped it.
type IndexedOutcome = (usize, Result<(SweepPoint, u64), PerturbError>);

/// Bytes per bf16 element, used to turn chunk bytes into values per chunk.
const BF16_BYTES: f64 = 2.0;

/// Reasons a sweep cannot run.
#[derive(Debug, thiserror::Error)]
pub enum SweepError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Perturb(#[from] PerturbError),
}

/// What the sweep varies.
#[derive(Debug, Clone, Serialize)]
pub struct SweepPlan {
    pub densities: Vec<f64>,
    pub modes: Vec<PerturbMode>,
    pub seeds: Vec<u64>,
}

/// One perturbed copy of the base.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepPoint {
    pub mode: PerturbMode,
    pub density: f64,
    pub seed: u64,
    pub altered_elements: u64,
    pub file_bytes: u64,
    pub chunk_overlap_bytes: u64,
    pub chunk_overlap_fraction: f64,
    pub next_chunks: u64,
    pub next_reused_chunks: u64,
}

/// The seeds of one (mode, density) cell, summarised.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepCell {
    pub mode: PerturbMode,
    pub density: f64,
    pub seed_count: u64,
    pub mean_chunk_overlap_fraction: f64,
    /// Standard error of the mean over seeds: sample standard deviation over the square root of the seed count.
    pub standard_error: f64,
    pub theory_chunk_overlap_fraction: f64,
    pub size_aware_theory_chunk_overlap_fraction: f64,
    /// Mean over seeds of reused chunks over next-file chunks (a chunk count, not bytes).
    pub mean_reused_chunk_fraction: f64,
    /// Expected reused-chunk share if every base chunk survives with probability `(1 - density)^(bytes / 2)`.
    pub size_aware_theory_reused_chunk_fraction: f64,
    /// Mean over seeds of next-file chunks over base chunks. Near 1 means changes rarely move chunk boundaries.
    pub mean_next_to_base_chunk_ratio: f64,
}

/// A whole sweep: the plan, the base's chunk statistics, every point, and the per-cell summary.
#[derive(Debug, Clone, Serialize)]
pub struct SweepReport {
    pub plan: SweepPlan,
    pub eligible_elements: u64,
    pub base_chunks: u64,
    pub base_mean_chunk_bytes: f64,
    /// Standard deviation of base chunk sizes over their mean.
    pub base_chunk_bytes_coefficient_of_variation: f64,
    /// Share of base chunks cut at the maximum chunk size.
    pub base_max_size_chunk_share: f64,
    pub values_per_chunk: f64,
    pub points: Vec<SweepPoint>,
    pub cells: Vec<SweepCell>,
}

/// Densities from 1e-6 to 1e-1 at 1, 2 and 5 per decade.
pub fn default_densities() -> Vec<f64> {
    let mut densities = Vec::new();
    for exponent in -6..=-1 {
        for mantissa in [1.0, 2.0, 5.0] {
            let density = mantissa * 10f64.powi(exponent);
            if density <= 0.1 + f64::EPSILON {
                densities.push(density);
            }
        }
    }
    densities
}

/// Chance that a chunk of `values_per_chunk` elements has no altered element, at a scattered `density`.
pub fn theory_overlap(values_per_chunk: f64, density: f64) -> f64 {
    (-values_per_chunk * density).exp()
}

fn size_aware_theory_overlap(base_chunks: &[ChunkRecord], density: f64) -> f64 {
    let total_bytes: u64 = base_chunks.iter().map(|chunk| chunk.bytes).sum();
    let unchanged_log = (1.0 - density).ln();
    let expected_reused_bytes: f64 = base_chunks
        .iter()
        .map(|chunk| chunk.bytes as f64 * (unchanged_log * chunk.bytes as f64 / BF16_BYTES).exp())
        .sum();
    expected_reused_bytes / total_bytes as f64
}

/// Runs every (mode, density, seed) point of `plan` on `worker_count` threads. Each worker holds one copy of the
/// base, so peak memory is about `worker_count + 1` times the file size.
///
/// The queue and results locks are only held to pop a job or push a result, and no code that can panic runs while
/// they are held, so they cannot be poisoned; the `expect`s on them state that invariant.
pub fn run_sweep(base_file: &[u8], plan: &SweepPlan, worker_count: usize) -> Result<SweepReport, SweepError> {
    let layout = CheckpointLayout::parse(base_file)?;
    let base_chunks = chunk_file(base_file);
    let base_hashes = hash_set(&base_chunks);
    let base_mean_chunk_bytes = mean_chunk_bytes(&base_chunks);
    let values_per_chunk = base_mean_chunk_bytes / BF16_BYTES;

    let mut jobs: Vec<(PerturbMode, f64, u64)> = Vec::new();
    for &mode in &plan.modes {
        for &density in &plan.densities {
            for &seed in &plan.seeds {
                jobs.push((mode, density, seed));
            }
        }
    }
    let queue = Mutex::new(jobs.into_iter().enumerate());
    let results: Mutex<Vec<IndexedOutcome>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..worker_count.max(1) {
            scope.spawn(|| {
                let mut buffer = base_file.to_vec();
                loop {
                    let Some((index, (mode, density, seed))) = queue.lock().expect("queue lock").next() else {
                        break;
                    };
                    buffer.copy_from_slice(base_file);
                    let outcome = measure_point(&mut buffer, &layout, &base_hashes, mode, density, seed);
                    results.lock().expect("results lock").push((index, outcome));
                }
            });
        }
    });
    let mut indexed = results.into_inner().expect("results lock");
    indexed.sort_by_key(|(index, _)| *index);
    let mut points = Vec::with_capacity(indexed.len());
    let mut eligible_elements = 0;
    for (_, outcome) in indexed {
        let (point, point_eligible_elements) = outcome?;
        eligible_elements = point_eligible_elements;
        points.push(point);
    }

    let cells = summarise(&points, plan, values_per_chunk, &base_chunks);
    Ok(SweepReport {
        plan: plan.clone(),
        eligible_elements,
        base_chunks: base_chunks.len() as u64,
        base_mean_chunk_bytes,
        base_chunk_bytes_coefficient_of_variation: coefficient_of_variation(&base_chunks, base_mean_chunk_bytes),
        base_max_size_chunk_share: base_chunks.iter().filter(|chunk| chunk.bytes == MAXIMUM_CHUNK_BYTES).count() as f64
            / base_chunks.len() as f64,
        values_per_chunk,
        points,
        cells,
    })
}

/// Perturbs `buffer` (a fresh copy of the base) and measures its whole-file chunk overlap against the base.
fn measure_point(
    buffer: &mut [u8],
    layout: &CheckpointLayout,
    base_hashes: &HashSet<MerkleHash>,
    mode: PerturbMode,
    density: f64,
    seed: u64,
) -> Result<(SweepPoint, u64), PerturbError> {
    let summary = perturb(buffer, layout, density, mode, seed)?;
    let next_chunks = chunk_file(buffer);
    let reused: Vec<&ChunkRecord> = next_chunks.iter().filter(|chunk| base_hashes.contains(&chunk.hash)).collect();
    let chunk_overlap_bytes: u64 = reused.iter().map(|chunk| chunk.bytes).sum();
    let point = SweepPoint {
        mode,
        density,
        seed,
        altered_elements: summary.altered_elements,
        file_bytes: buffer.len() as u64,
        chunk_overlap_bytes,
        chunk_overlap_fraction: chunk_overlap_bytes as f64 / buffer.len() as f64,
        next_chunks: next_chunks.len() as u64,
        next_reused_chunks: reused.len() as u64,
    };
    Ok((point, summary.eligible_elements))
}

fn coefficient_of_variation(chunks: &[ChunkRecord], mean_bytes: f64) -> f64 {
    let variance = chunks
        .iter()
        .map(|chunk| (chunk.bytes as f64 - mean_bytes).powi(2))
        .sum::<f64>()
        / chunks.len() as f64;
    variance.sqrt() / mean_bytes
}

fn size_aware_theory_reused_chunks(base_chunks: &[ChunkRecord], density: f64) -> f64 {
    let unchanged_log = (1.0 - density).ln();
    base_chunks
        .iter()
        .map(|chunk| (unchanged_log * chunk.bytes as f64 / BF16_BYTES).exp())
        .sum::<f64>()
        / base_chunks.len() as f64
}

fn summarise(
    points: &[SweepPoint],
    plan: &SweepPlan,
    values_per_chunk: f64,
    base_chunks: &[ChunkRecord],
) -> Vec<SweepCell> {
    let mut cells = Vec::new();
    for &mode in &plan.modes {
        for &density in &plan.densities {
            let cell_points: Vec<&SweepPoint> = points
                .iter()
                .filter(|point| point.mode == mode && point.density == density)
                .collect();
            let fractions: Vec<f64> = cell_points.iter().map(|point| point.chunk_overlap_fraction).collect();
            let mean_over_points = |value: &dyn Fn(&SweepPoint) -> f64| {
                cell_points.iter().map(|point| value(point)).sum::<f64>() / cell_points.len() as f64
            };
            let seed_count = fractions.len() as f64;
            let mean = fractions.iter().sum::<f64>() / seed_count;
            let variance = if fractions.len() > 1 {
                fractions.iter().map(|fraction| (fraction - mean).powi(2)).sum::<f64>() / (seed_count - 1.0)
            } else {
                0.0
            };
            cells.push(SweepCell {
                mode,
                density,
                seed_count: fractions.len() as u64,
                mean_chunk_overlap_fraction: mean,
                standard_error: (variance / seed_count).sqrt(),
                theory_chunk_overlap_fraction: theory_overlap(values_per_chunk, density),
                size_aware_theory_chunk_overlap_fraction: size_aware_theory_overlap(base_chunks, density),
                mean_reused_chunk_fraction: mean_over_points(&|point| {
                    point.next_reused_chunks as f64 / point.next_chunks as f64
                }),
                size_aware_theory_reused_chunk_fraction: size_aware_theory_reused_chunks(base_chunks, density),
                mean_next_to_base_chunk_ratio: mean_over_points(&|point| {
                    point.next_chunks as f64 / base_chunks.len() as f64
                }),
            });
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_densities_span_five_decades_at_three_per_decade() {
        let densities = default_densities();
        assert_eq!(densities.len(), 16);
        assert_eq!(densities[0], 1e-6);
        assert!((densities[15] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn size_aware_theory_equals_plain_theory_for_equal_chunks() {
        let chunks: Vec<ChunkRecord> = (0..4)
            .map(|index| ChunkRecord {
                start_byte: index * 65536,
                bytes: 65536,
                hash: MerkleHash::default(),
            })
            .collect();
        let density = 1e-5;
        let plain = theory_overlap(32768.0, density);
        assert!((size_aware_theory_overlap(&chunks, density) - plain).abs() < 1e-4);
    }
}
