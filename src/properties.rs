//! Property tests 1 to 5 from `docs/plan.md`, on small synthetic safetensors files of a few MiB.

use proptest::prelude::*;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::chunking::{MAXIMUM_CHUNK_BYTES as MAX_CHUNK_BYTES, chunk_file, hash_set};
use crate::layout::test_support::{SyntheticTensor, build_safetensors, build_safetensors_with_header};
use crate::layout::{CheckpointLayout, TensorKind};
use crate::overlap::measure_overlap;
use crate::perturb::{PerturbMode, perturb};

/// Clears the top exponent bit of a bf16 high byte, so random elements are always finite.
const FINITE_HIGH_BYTE_MASK: u8 = 0xbf;

fn synthetic_tensors(seed: u64, shapes: &[(u64, u64)]) -> Vec<SyntheticTensor> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    shapes
        .iter()
        .enumerate()
        .map(|(index, &(rows, columns))| {
            let mut data = vec![0u8; (rows * columns * 2) as usize];
            rng.fill_bytes(&mut data);
            for element in data.chunks_exact_mut(2) {
                element[1] &= FINITE_HIGH_BYTE_MASK;
            }
            SyntheticTensor {
                name: format!("model.layers.{index}.mlp.up_proj.weight"),
                dtype: "BF16",
                shape: vec![rows, columns],
                data,
            }
        })
        .collect()
}

/// Two to six tensors of 256 KiB to 2 MiB each, so files span roughly 8 to 190 chunks and some chunks cross tensors.
fn shapes_strategy() -> impl Strategy<Value = Vec<(u64, u64)>> {
    prop::collection::vec((128u64..256, 1024u64..4096), 2..6)
}

fn mode_strategy() -> impl Strategy<Value = PerturbMode> {
    prop_oneof![
        Just(PerturbMode::Scattered),
        Just(PerturbMode::Rows),
        Just(PerturbMode::Tensors)
    ]
}

/// Reused bytes per row, by intersecting every reused chunk with every row range. Quadratic, and independent of the
/// sweep in `overlap::attribute_reused_bytes`.
fn reused_bytes_per_row_by_brute_force(base: &[u8], next: &[u8]) -> Vec<u64> {
    let layout = CheckpointLayout::parse(next).unwrap();
    let mut row_ranges = vec![(0, layout.header_bytes)];
    row_ranges.extend(layout.tensors.iter().map(|tensor| (tensor.start_byte, tensor.end_byte)));
    let base_hashes = hash_set(&chunk_file(base));
    let reused_chunks: Vec<_> = chunk_file(next)
        .into_iter()
        .filter(|chunk| base_hashes.contains(&chunk.hash))
        .collect();
    row_ranges
        .iter()
        .map(|&(row_start, row_end)| {
            reused_chunks
                .iter()
                .map(|chunk| chunk.end_byte().min(row_end).saturating_sub(chunk.start_byte.max(row_start)))
                .sum()
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Property 1: attributed bytes summed over every row equal the file size, and each row's reused bytes match an
    /// independent chunk-by-row intersection.
    #[test]
    fn attributed_bytes_reconcile_with_file_size(
        seed in any::<u64>(),
        shapes in shapes_strategy(),
        density in 0.0f64..0.00003,
        mode in mode_strategy(),
    ) {
        let base = build_safetensors(&synthetic_tensors(seed, &shapes));
        let mut next = base.clone();
        let layout = CheckpointLayout::parse(&base).unwrap();
        perturb(&mut next, &layout, density, mode, seed).unwrap();
        let report = measure_overlap(&base, &next).unwrap();

        prop_assert_eq!(report.tensors.iter().map(|row| row.bytes).sum::<u64>(), next.len() as u64);
        prop_assert_eq!(report.whole_file.bytes, next.len() as u64);
        let attributed: Vec<u64> = report.tensors.iter().map(|row| row.chunk_overlap_bytes).collect();
        prop_assert_eq!(attributed, reused_bytes_per_row_by_brute_force(&base, &next));
    }

    /// Property 2: a file against itself is 100% overlap and every element is unchanged.
    #[test]
    fn self_overlap_is_complete(seed in any::<u64>(), shapes in shapes_strategy()) {
        let file = build_safetensors(&synthetic_tensors(seed, &shapes));
        let report = measure_overlap(&file, &file).unwrap();
        prop_assert_eq!(report.whole_file.chunk_overlap_bytes, file.len() as u64);
        prop_assert_eq!(report.whole_file.unchanged_element_fraction, Some(1.0));
        prop_assert_eq!(report.whole_file.sparse_delta_bytes, 0);
    }

    /// Property 3: a perturbation at density 0 leaves the file byte-identical, so overlap is 100%.
    #[test]
    fn zero_density_perturbation_keeps_full_overlap(
        seed in any::<u64>(),
        shapes in shapes_strategy(),
        mode in mode_strategy(),
    ) {
        let base = build_safetensors(&synthetic_tensors(seed, &shapes));
        let mut next = base.clone();
        let layout = CheckpointLayout::parse(&base).unwrap();
        let summary = perturb(&mut next, &layout, 0.0, mode, seed).unwrap();
        prop_assert_eq!(summary.altered_elements, 0);
        let report = measure_overlap(&base, &next).unwrap();
        prop_assert_eq!(report.whole_file.chunk_overlap_bytes, next.len() as u64);
    }

    /// Property 4: rewriting the JSON header with the keys in another order loses at most two chunks' worth of
    /// data-section overlap. When the rewrite also changes the header length (a metadata entry), the data section
    /// shifts, and the shift survives every forced cut at the maximum chunk size, so the bound adds the base's leading
    /// run of maximum-size chunks.
    #[test]
    fn rewritten_header_costs_a_few_chunks_of_data(
        seed in any::<u64>(),
        shapes in shapes_strategy(),
        rotation in 1usize..6,
        metadata_bytes in prop::option::of(0usize..300),
    ) {
        let tensors = synthetic_tensors(seed, &shapes);
        let base = build_safetensors(&tensors);
        let mut key_order: Vec<usize> = (0..tensors.len()).collect();
        key_order.rotate_left(rotation % tensors.len());
        key_order.reverse();
        let note = metadata_bytes.map(|bytes| "x".repeat(bytes));
        let next = build_safetensors_with_header(&tensors, &key_order, note.as_deref());
        let report = measure_overlap(&base, &next).unwrap();
        let data_rows = report.tensors.iter().filter(|row| row.kind != TensorKind::Header);
        let data_bytes: u64 = data_rows.clone().map(|row| row.bytes).sum();
        let lost_data_bytes = data_bytes - data_rows.map(|row| row.chunk_overlap_bytes).sum::<u64>();
        let shift_carry_bytes: u64 = if note.is_some() {
            chunk_file(&base).iter().take_while(|chunk| chunk.bytes == MAX_CHUNK_BYTES).map(|chunk| chunk.bytes).sum()
        } else {
            0
        };
        prop_assert!(data_bytes >= 4 * MAX_CHUNK_BYTES);
        prop_assert!(
            lost_data_bytes <= 2 * MAX_CHUNK_BYTES + shift_carry_bytes,
            "lost {} of {} data bytes", lost_data_bytes, data_bytes
        );
    }

    /// Property 5: changing one element costs at most two chunks of overlap.
    #[test]
    fn one_changed_element_costs_at_most_two_chunks(
        seed in any::<u64>(),
        shapes in shapes_strategy(),
        element_pick in any::<u64>(),
    ) {
        let base = build_safetensors(&synthetic_tensors(seed, &shapes));
        let layout = CheckpointLayout::parse(&base).unwrap();
        let data_elements = (layout.file_bytes - layout.header_bytes) / 2;
        let mut next = base.clone();
        next[(layout.header_bytes + (element_pick % data_elements) * 2) as usize] ^= 1;
        let report = measure_overlap(&base, &next).unwrap();
        let new_chunks = report.chunk_stats.next_chunks - report.chunk_stats.next_reused_chunks;
        prop_assert!(new_chunks <= 2, "{} new chunks", new_chunks);
        prop_assert!(report.whole_file.bytes - report.whole_file.chunk_overlap_bytes <= 2 * MAX_CHUNK_BYTES);
        prop_assert_eq!(report.whole_file.unchanged_elements, report.whole_file.compared_elements - 1);
    }
}
