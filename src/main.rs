//! `ckpt-dedup`: measures Xet chunk reuse between consecutive model checkpoints.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ckpt_dedup::layout::CheckpointLayout;
use ckpt_dedup::overlap::{OverlapReport, RollupRow, measure_overlap};
use ckpt_dedup::perturb::{PerturbMode, perturb};
use ckpt_dedup::results::{InputStamp, repo_root, write_result};
use ckpt_dedup::sweep::{SweepPlan, SweepReport, default_densities, run_sweep};
use ckpt_dedup::upload::{UploadReport, measure_upload};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Directory that receives `<name>.json`.
    #[arg(long, global = true, default_value_os_t = repo_root().join("results"))]
    results_dir: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Chunk both checkpoints, mark the next file's chunks as reused or new, and attribute bytes to tensors.
    Overlap {
        base: PathBuf,
        next: PathBuf,
        /// Result name; the run writes `<results-dir>/<name>.json`.
        #[arg(long)]
        name: String,
        /// Also print one row per tensor.
        #[arg(long)]
        per_tensor: bool,
    },
    /// Write a copy of a bf16 checkpoint with a fraction of its elements moved by one ulp.
    Perturb {
        input: PathBuf,
        /// Fraction of BF16 elements to alter, in [0, 1].
        #[arg(long)]
        density: f64,
        #[arg(long, value_enum)]
        mode: PerturbMode,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Commit the base then the next checkpoint to a fresh `local://` store and report the next file's dedup metrics.
    Upload {
        base: PathBuf,
        next: PathBuf,
        #[arg(long)]
        name: String,
        /// Directory that holds the temporary store (removed afterwards). Keep it on disk, not on a RAM-backed /tmp.
        #[arg(long, default_value_os_t = default_scratch_dir())]
        scratch_dir: PathBuf,
    },
    /// Chunk overlap of perturbed copies against the original, over densities 1e-6 to 1e-1, every mode, and seeds.
    Sweep {
        input: PathBuf,
        #[arg(long)]
        name: String,
        /// Seeds per (mode, density) cell: 0..seed_count.
        #[arg(long, default_value_t = 5)]
        seed_count: u64,
        /// Worker threads. Each holds one copy of the input in memory.
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Overlap {
            base,
            next,
            name,
            per_tensor,
        } => run_overlap(&cli.results_dir, &base, &next, &name, per_tensor),
        Command::Perturb {
            input,
            density,
            mode,
            seed,
            output,
        } => run_perturb(&input, density, mode, seed, &output),
        Command::Upload {
            base,
            next,
            name,
            scratch_dir,
        } => run_upload(&cli.results_dir, &base, &next, &name, &scratch_dir),
        Command::Sweep {
            input,
            name,
            seed_count,
            workers,
        } => run_sweep_command(&cli.results_dir, &input, &name, seed_count, workers),
    }
}

fn default_scratch_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/xet-checkpoint-dedup/scratch")
}

fn run_upload(results_dir: &Path, base: &Path, next: &Path, name: &str, scratch_dir: &Path) -> Result<()> {
    let base_file = read_input(base)?;
    let next_file = read_input(next)?;
    let report =
        measure_upload(base, next, &base_file, &next_file, scratch_dir).context("uploading to a local store")?;
    let inputs = [
        InputStamp::of("base", base, &base_file),
        InputStamp::of("next", next, &next_file),
    ];
    let path = write_result(results_dir, name, "upload", &inputs, None, &report)?;
    print_upload(&report);
    println!("wrote {}", path.display());
    Ok(())
}

fn print_upload(report: &UploadReport) {
    let metrics = &report.next.dedup_metrics;
    println!("next file bytes          {:>14}", report.next.file_bytes);
    println!(
        "reported file bytes      {:>14}",
        report
            .next
            .reported_file_bytes
            .map_or_else(|| "-".to_string(), |bytes| bytes.to_string())
    );
    println!("new_bytes                {:>14}", metrics.new_bytes);
    println!("deduped_bytes (raw)      {:>14}", metrics.deduped_bytes);
    println!("defrag_prevented_bytes   {:>14}", metrics.defrag_prevented_dedup_bytes);
    println!("xorb_bytes_uploaded      {:>14}", metrics.xorb_bytes_uploaded);
    println!("shard_bytes_uploaded     {:>14}", metrics.shard_bytes_uploaded);
    println!("chunk overlap bytes      {:>14}", report.chunk_overlap_bytes);
    println!("internal repeat bytes    {:>14}", report.next_internal_repeat_bytes);
    println!("accepted dedup bytes     {:>14}", report.accepted_dedup_bytes);
    println!("transferred bytes        {:>14}", report.transferred_bytes);
    println!("defrag gap bytes         {:>14}", report.defrag_gap_bytes);
}

fn read_input(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

fn run_overlap(results_dir: &Path, base: &Path, next: &Path, name: &str, per_tensor: bool) -> Result<()> {
    let base_file = read_input(base)?;
    let next_file = read_input(next)?;
    let report = measure_overlap(&base_file, &next_file).context("measuring chunk overlap")?;
    let inputs = [
        InputStamp::of("base", base, &base_file),
        InputStamp::of("next", next, &next_file),
    ];
    let path = write_result(results_dir, name, "overlap", &inputs, None, &report)?;
    print_overlap(&report, per_tensor);
    println!("wrote {}", path.display());
    Ok(())
}

fn run_perturb(input: &Path, density: f64, mode: PerturbMode, seed: u64, output: &Path) -> Result<()> {
    let mut file = read_input(input)?;
    let layout = CheckpointLayout::parse(&file).context("parsing the safetensors header")?;
    let summary = perturb(&mut file, &layout, density, mode, seed)?;
    std::fs::write(output, &file).with_context(|| format!("writing {}", output.display()))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn run_sweep_command(results_dir: &Path, input: &Path, name: &str, seed_count: u64, workers: usize) -> Result<()> {
    let base_file = read_input(input)?;
    let plan = SweepPlan {
        densities: default_densities(),
        modes: vec![PerturbMode::Scattered, PerturbMode::Rows, PerturbMode::Tensors],
        seeds: (0..seed_count).collect(),
    };
    let report = run_sweep(&base_file, &plan, workers)?;
    let inputs = [InputStamp::of("base", input, &base_file)];
    let path = write_result(results_dir, name, "sweep", &inputs, None, &report)?;
    print_sweep(&report);
    println!("wrote {}", path.display());
    Ok(())
}

fn print_sweep(report: &SweepReport) {
    println!(
        "values per chunk {:.1} (mean base chunk {:.0} bytes)",
        report.values_per_chunk, report.base_mean_chunk_bytes
    );
    println!(
        "{:<10} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "mode", "density", "mean", "stderr", "theory", "size-aware"
    );
    for cell in &report.cells {
        println!(
            "{:<10} {:>9.0e} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            format!("{:?}", cell.mode).to_lowercase(),
            cell.density,
            cell.mean_chunk_overlap_fraction,
            cell.standard_error,
            cell.theory_chunk_overlap_fraction,
            cell.size_aware_theory_chunk_overlap_fraction
        );
    }
}

fn format_fraction(fraction: Option<f64>) -> String {
    fraction.map_or_else(|| "-".to_string(), |value| format!("{value:.6}"))
}

fn print_rollup_table(title: &str, rows: &[RollupRow]) {
    println!("\n{title}");
    println!(
        "{:<20} {:>14} {:>12} {:>16} {:>9} {:>14}",
        "group", "bytes", "unchanged", "overlap_bytes", "overlap%", "sparse_delta"
    );
    for row in rows {
        println!(
            "{:<20} {:>14} {:>12} {:>16} {:>9.3} {:>14}",
            row.group,
            row.bytes,
            format_fraction(row.unchanged_element_fraction),
            row.chunk_overlap_bytes,
            row.chunk_overlap_percent,
            row.sparse_delta_bytes
        );
    }
}

fn print_overlap(report: &OverlapReport, per_tensor: bool) {
    if per_tensor {
        println!(
            "{:<56} {:>5} {:<12} {:<5} {:>12} {:>10} {:>12} {:>9}",
            "tensor", "layer", "kind", "dtype", "bytes", "unchanged", "overlap", "overlap%"
        );
        for row in &report.tensors {
            println!(
                "{:<56} {:>5} {:<12} {:<5} {:>12} {:>10} {:>12} {:>9.3}",
                row.name,
                row.layer.map_or_else(|| "-".to_string(), |layer| layer.to_string()),
                row.kind.label(),
                row.dtype,
                row.bytes,
                format_fraction(row.unchanged_element_fraction),
                row.chunk_overlap_bytes,
                row.chunk_overlap_percent
            );
        }
    }
    print_rollup_table("by tensor kind", &report.by_kind);
    print_rollup_table("by layer", &report.by_layer);
    print_rollup_table("whole file", std::slice::from_ref(&report.whole_file));
    let stats = &report.chunk_stats;
    println!(
        "\nchunks: base {} next {} reused {} max-size {} (mean next chunk {:.0} bytes)",
        stats.base_chunks,
        stats.next_chunks,
        stats.next_reused_chunks,
        stats.next_max_size_chunks,
        stats.next_mean_chunk_bytes
    );
    println!(
        "sparse delta: {} bytes ({:.4}% of the next file)",
        report.whole_file.sparse_delta_bytes,
        100.0 * report.whole_file.sparse_delta_bytes as f64 / report.whole_file.bytes.max(1) as f64
    );
}
