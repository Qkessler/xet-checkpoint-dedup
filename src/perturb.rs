//! Synthetic perturbations of a bf16 checkpoint at a chosen change density.
//!
//! Altering an element flips the least significant mantissa bit of its bf16 value (bit 0 of the element's first,
//! little-endian byte). The exponent is untouched, so a finite value moves one ulp and stays finite. Only BF16
//! tensors are eligible, and the header is never touched. The number of altered elements is exactly
//! `round(density * eligible elements)` in every mode; the modes differ only in where those elements sit.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use crate::layout::{CheckpointLayout, TensorLayout};

const BF16_BYTES: u64 = 2;
const BF16_DTYPE: &str = "BF16";
const BF16_EXPONENT_MASK: u16 = 0x7f80;

/// Where the altered elements are placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PerturbMode {
    /// Elements chosen uniformly at random across every BF16 tensor.
    Scattered,
    /// Whole rows (the contiguous last-dimension runs of a tensor), chosen at random. The last row picked is altered
    /// only as far as needed to reach the exact count.
    Rows,
    /// Whole tensors, chosen at random. The last tensor picked is altered from its start only as far as needed.
    Tensors,
}

/// Reasons a perturbation cannot be applied.
#[derive(Debug, thiserror::Error)]
pub enum PerturbError {
    #[error("density {0} is outside [0, 1]")]
    DensityOutOfRange(f64),
    #[error("checkpoint has no BF16 tensors to perturb")]
    NoBf16Tensors,
    #[error("element at byte {start_byte} is Inf or NaN, and flipping its last bit would not keep it finite")]
    NonFiniteElement { start_byte: u64 },
}

/// What a perturbation did.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerturbSummary {
    pub mode: PerturbMode,
    pub density: f64,
    pub seed: u64,
    pub eligible_elements: u64,
    pub altered_elements: u64,
}

/// A contiguous run of BF16 elements: the absolute byte offset of its first element and its length in elements.
#[derive(Debug, Clone, Copy)]
struct ElementRun {
    start_byte: u64,
    element_count: u64,
}

/// Alters `round(density * eligible)` BF16 elements of `file` in place. Same inputs and seed give the same bytes.
///
/// Fails before touching `file` on a bad density or a file with no BF16 tensors. Fails part way, leaving `file`
/// partly altered, if a chosen element is Inf or NaN.
pub fn perturb(
    file: &mut [u8],
    layout: &CheckpointLayout,
    density: f64,
    mode: PerturbMode,
    seed: u64,
) -> Result<PerturbSummary, PerturbError> {
    if !(0.0..=1.0).contains(&density) {
        return Err(PerturbError::DensityOutOfRange(density));
    }
    let bf16_tensors: Vec<&TensorLayout> = layout.tensors.iter().filter(|tensor| tensor.dtype == BF16_DTYPE).collect();
    if bf16_tensors.is_empty() {
        return Err(PerturbError::NoBf16Tensors);
    }
    let eligible_elements: u64 = bf16_tensors.iter().map(|tensor| tensor.element_count()).sum();
    let target_elements = (density * eligible_elements as f64).round() as u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut tensor_runs: Vec<ElementRun> = bf16_tensors.iter().map(|tensor| whole_tensor_run(tensor)).collect();

    match mode {
        PerturbMode::Scattered => {
            let flat_indices = sample_distinct(eligible_elements, target_elements, &mut rng);
            alter_flat_indices(file, &tensor_runs, &flat_indices)?;
        },
        PerturbMode::Rows => {
            let mut rows: Vec<ElementRun> = bf16_tensors.iter().flat_map(|tensor| rows_of(tensor)).collect();
            rows.shuffle(&mut rng);
            alter_runs_until(file, &rows, target_elements)?;
        },
        PerturbMode::Tensors => {
            tensor_runs.shuffle(&mut rng);
            alter_runs_until(file, &tensor_runs, target_elements)?;
        },
    }

    Ok(PerturbSummary {
        mode,
        density,
        seed,
        eligible_elements,
        altered_elements: target_elements,
    })
}

fn whole_tensor_run(tensor: &TensorLayout) -> ElementRun {
    ElementRun {
        start_byte: tensor.start_byte,
        element_count: tensor.element_count(),
    }
}

/// Rows of a tensor: runs of its last dimension. A 0-d or 1-d tensor is one row.
fn rows_of(tensor: &TensorLayout) -> impl Iterator<Item = ElementRun> + '_ {
    let row_elements = match tensor.shape.as_slice() {
        [] | [_] => tensor.element_count(),
        [.., last] => *last,
    };
    let row_count = tensor.element_count().checked_div(row_elements).unwrap_or(0);
    (0..row_count).map(move |row| ElementRun {
        start_byte: tensor.start_byte + row * row_elements * BF16_BYTES,
        element_count: row_elements,
    })
}

fn alter_runs_until(file: &mut [u8], runs: &[ElementRun], target_elements: u64) -> Result<(), PerturbError> {
    let mut remaining_elements = target_elements;
    for run in runs {
        if remaining_elements == 0 {
            break;
        }
        let altered_here = run.element_count.min(remaining_elements);
        for element in 0..altered_here {
            flip_bf16_lsb(file, run.start_byte + element * BF16_BYTES)?;
        }
        remaining_elements -= altered_here;
    }
    Ok(())
}

/// Alters the elements at ascending `flat_indices`, counting elements across `runs` in order.
fn alter_flat_indices(file: &mut [u8], runs: &[ElementRun], flat_indices: &[u64]) -> Result<(), PerturbError> {
    let mut run_iter = runs.iter();
    let mut run = run_iter.next();
    let mut run_first_index = 0u64;
    for &flat_index in flat_indices {
        while let Some(current) = run {
            if flat_index < run_first_index + current.element_count {
                break;
            }
            run_first_index += current.element_count;
            run = run_iter.next();
        }
        let current = run.expect("flat indices stay below the eligible element count");
        flip_bf16_lsb(file, current.start_byte + (flat_index - run_first_index) * BF16_BYTES)?;
    }
    Ok(())
}

fn flip_bf16_lsb(file: &mut [u8], element_start_byte: u64) -> Result<(), PerturbError> {
    let low_byte = element_start_byte as usize;
    let bits = u16::from_le_bytes([file[low_byte], file[low_byte + 1]]);
    if bits & BF16_EXPONENT_MASK == BF16_EXPONENT_MASK {
        return Err(PerturbError::NonFiniteElement {
            start_byte: element_start_byte,
        });
    }
    file[low_byte] ^= 1;
    Ok(())
}

/// `count` distinct indices in `0..population`, uniformly at random, in ascending order.
///
/// Uses a bitmap with rejection when `count` is at most half the population, and picks the complement otherwise, so
/// each draw succeeds with probability at least one half.
fn sample_distinct(population: u64, count: u64, rng: &mut ChaCha8Rng) -> Vec<u64> {
    let pick_complement = count > population / 2;
    let draw_count = if pick_complement { population - count } else { count };
    let mut bitmap = vec![0u64; population.div_ceil(64) as usize];
    let mut drawn = 0;
    while drawn < draw_count {
        let index = rng.random_range(0..population);
        let (word, bit) = ((index / 64) as usize, index % 64);
        if bitmap[word] & (1 << bit) == 0 {
            bitmap[word] |= 1 << bit;
            drawn += 1;
        }
    }
    (0..population)
        .filter(|&index| (bitmap[(index / 64) as usize] >> (index % 64) & 1 == 1) != pick_complement)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::test_support::{SyntheticTensor, build_safetensors};

    fn file_with(shapes: &[(u64, u64)]) -> Vec<u8> {
        let tensors: Vec<SyntheticTensor> = shapes
            .iter()
            .enumerate()
            .map(|(index, &(rows, columns))| SyntheticTensor {
                name: format!("t{index}"),
                dtype: "BF16",
                shape: vec![rows, columns],
                data: (0..rows * columns)
                    .flat_map(|element| ((element % 500) as u16 + 0x3f00).to_le_bytes())
                    .collect(),
            })
            .collect();
        build_safetensors(&tensors)
    }

    fn changed_elements(before: &[u8], after: &[u8]) -> Vec<usize> {
        before
            .chunks_exact(2)
            .zip(after.chunks_exact(2))
            .enumerate()
            .filter(|(_, (left, right))| left != right)
            .map(|(index, _)| index)
            .collect()
    }

    #[test]
    fn every_mode_alters_the_exact_count_and_never_the_header() {
        let original = file_with(&[(16, 32), (8, 64), (4, 4)]);
        let layout = CheckpointLayout::parse(&original).unwrap();
        for mode in [PerturbMode::Scattered, PerturbMode::Rows, PerturbMode::Tensors] {
            let mut perturbed = original.clone();
            let summary = perturb(&mut perturbed, &layout, 0.1, mode, 7).unwrap();
            assert_eq!(summary.eligible_elements, 16 * 32 + 8 * 64 + 16);
            assert_eq!(summary.altered_elements, 104);
            let header = layout.header_bytes as usize;
            assert_eq!(perturbed[..header], original[..header]);
            assert_eq!(changed_elements(&original[header..], &perturbed[header..]).len(), 104, "{mode:?}");
        }
    }

    #[test]
    fn rows_mode_alters_contiguous_whole_rows() {
        let original = file_with(&[(64, 32)]);
        let layout = CheckpointLayout::parse(&original).unwrap();
        let mut perturbed = original.clone();
        perturb(&mut perturbed, &layout, 0.125, PerturbMode::Rows, 3).unwrap();
        let header = layout.header_bytes as usize;
        let changed = changed_elements(&original[header..], &perturbed[header..]);
        let mut rows: Vec<usize> = changed.iter().map(|element| element / 32).collect();
        rows.dedup();
        assert_eq!(rows.len(), 8);
    }

    #[test]
    fn same_seed_gives_same_bytes_and_flip_moves_one_ulp() {
        let original = file_with(&[(32, 32)]);
        let layout = CheckpointLayout::parse(&original).unwrap();
        let mut first = original.clone();
        let mut second = original.clone();
        perturb(&mut first, &layout, 0.3, PerturbMode::Scattered, 11).unwrap();
        perturb(&mut second, &layout, 0.3, PerturbMode::Scattered, 11).unwrap();
        assert_eq!(first, second);
        let header = layout.header_bytes as usize;
        for (before, after) in original[header..].chunks_exact(2).zip(first[header..].chunks_exact(2)) {
            let delta = u16::from_le_bytes([before[0], before[1]]) ^ u16::from_le_bytes([after[0], after[1]]);
            assert!(delta == 0 || delta == 1);
        }
    }

    #[test]
    fn refuses_to_turn_infinity_into_nan() {
        let mut file = build_safetensors(&[SyntheticTensor {
            name: "t".to_string(),
            dtype: "BF16",
            shape: vec![1],
            data: 0x7f80u16.to_le_bytes().to_vec(),
        }]);
        let layout = CheckpointLayout::parse(&file).unwrap();
        let outcome = perturb(&mut file, &layout, 1.0, PerturbMode::Scattered, 0);
        assert!(matches!(outcome, Err(PerturbError::NonFiniteElement { .. })));
    }

    #[test]
    fn rejects_density_above_one() {
        let mut file = file_with(&[(2, 2)]);
        let layout = CheckpointLayout::parse(&file).unwrap();
        assert!(perturb(&mut file, &layout, 1.5, PerturbMode::Scattered, 0).is_err());
    }

    #[test]
    fn sampling_more_than_half_uses_the_complement() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let picked = sample_distinct(100, 90, &mut rng);
        assert_eq!(picked.len(), 90);
        assert!(picked.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
