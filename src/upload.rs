//! Uploads a checkpoint pair to a fresh `local://` Xet store through the session API and reports what the upload path
//! actually transferred for the next checkpoint.
//!
//! The base is committed first, then the next checkpoint is committed to the same store from the same session, so
//! the next file dedups against the base's shard. `DeduplicationMetrics` come straight from the commit reports.
//!
//! The crates.io 1.6.0 release added a dedup match to `deduped_bytes` before defrag prevention decided whether to take
//! it, and then counted the refused chunks again as new. xet-core #931 fixed that, and this tool builds on xet-core
//! at `d2f667c`, which includes the fix (ADR 0006). Every chunk is either new or deduped, so the dedup actually taken
//! is `file bytes - new_bytes` in any version; the derived fields use that (ADR 0005), and the outer loop checks that
//! the fixed `deduped_bytes` and reported file size now agree with it.
//!
//! The session disables client telemetry, so a measurement never sends anything over the network, even though
//! xet-core already emits no telemetry for `local://` stores.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;
use xet::XetError;
use xet::xet_session::{DeduplicationMetrics, Sha256Policy, XetConfig, XetSessionBuilder};

use crate::chunking::{chunk_file, hash_set};

/// Reasons an upload measurement cannot run.
#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error("creating the local store under {path}: {source}")]
    Store { path: PathBuf, source: std::io::Error },
    #[error("reading {path}: {source}")]
    Read { path: PathBuf, source: std::io::Error },
    #[error("Xet session: {0}")]
    Xet(#[from] XetError),
    #[error("Xet config: {0}")]
    Config(#[from] xet_runtime::config::ConfigError),
    #[error("upload reported {new_bytes} new bytes for a {file_bytes}-byte file")]
    NewBytesExceedFile { new_bytes: u64, file_bytes: u64 },
}

/// The raw metrics of one commit, which holds exactly one file.
#[derive(Debug, Clone, Serialize)]
pub struct CommitMetrics {
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Size the upload path reports for the file. Equal to `file_bytes` once the accounting is fixed (xet-core #931);
    /// in crates.io 1.6.0 it exceeded `file_bytes` by `defrag_prevented_dedup_bytes`.
    pub reported_file_bytes: Option<u64>,
    pub dedup_metrics: DeduplicationMetrics,
}

/// What `upload` measures for one checkpoint pair.
#[derive(Debug, Clone, Serialize)]
pub struct UploadReport {
    pub base: CommitMetrics,
    pub next: CommitMetrics,
    /// Bytes of the next file in chunks whose hash appears in the base (as `overlap` computes it).
    pub chunk_overlap_bytes: u64,
    /// Bytes of the next file in chunks not in the base but repeated earlier in the next file itself. The upload
    /// path can dedup these too, so they bound the dedup taken from above together with `chunk_overlap_bytes`.
    pub next_internal_repeat_bytes: u64,
    /// `next file bytes - new_bytes`: the dedup the upload path actually took.
    pub accepted_dedup_bytes: u64,
    /// `new_bytes + shard_bytes_uploaded` for the next commit: the plan's transferred bytes.
    pub transferred_bytes: u64,
    /// `chunk_overlap_bytes - accepted_dedup_bytes`: overlap the upload path re-sent anyway. Signed, because
    /// internal repeats can in principle push accepted dedup above the overlap.
    pub defrag_gap_bytes: i64,
    /// `chunk_overlap_bytes - deduped_bytes` with the raw metric. Equal to `defrag_gap_bytes` once the accounting is
    /// fixed; kept so a regression in the accounting shows up.
    pub raw_defrag_gap_bytes: i64,
}

/// Commits `base` then `next` to a fresh `local://` store created under `scratch_dir`, which is removed afterwards.
/// `base_file` and `next_file` are the same files already read into memory, used for the chunk overlap.
pub fn measure_upload(
    base: &Path,
    next: &Path,
    base_file: &[u8],
    next_file: &[u8],
    scratch_dir: &Path,
) -> Result<UploadReport, UploadError> {
    std::fs::create_dir_all(scratch_dir).map_err(|source| UploadError::Store {
        path: scratch_dir.to_path_buf(),
        source,
    })?;
    let store = tempfile::Builder::new()
        .prefix("local-store-")
        .tempdir_in(scratch_dir)
        .map_err(|source| UploadError::Store {
            path: scratch_dir.to_path_buf(),
            source,
        })?;
    let endpoint = format!("local://{}", store.path().join("cas").display());
    let config = XetConfig::new().with_config("telemetry.enabled", false)?;
    let session = XetSessionBuilder::new_with_config(config).build()?;
    let commit_one = |path: &Path| -> Result<CommitMetrics, UploadError> {
        let commit = session.new_upload_commit()?.with_endpoint(&endpoint).build_blocking()?;
        let handle = commit.upload_from_path_blocking(path.to_path_buf(), Sha256Policy::Skip)?;
        let file = handle.finalize_ingestion_blocking()?;
        let report = commit.commit_blocking()?;
        let file_bytes = std::fs::metadata(path)
            .map_err(|source| UploadError::Read {
                path: path.to_path_buf(),
                source,
            })?
            .len();
        Ok(CommitMetrics {
            file_bytes,
            reported_file_bytes: file.xet_info.file_size(),
            dedup_metrics: report.dedup_metrics,
        })
    };
    let base_metrics = commit_one(base)?;
    let next_metrics = commit_one(next)?;

    let (chunk_overlap_bytes, next_internal_repeat_bytes) = chunk_reuse(base_file, next_file);
    let new_bytes = next_metrics.dedup_metrics.new_bytes;
    let accepted_dedup_bytes =
        next_metrics
            .file_bytes
            .checked_sub(new_bytes)
            .ok_or(UploadError::NewBytesExceedFile {
                new_bytes,
                file_bytes: next_metrics.file_bytes,
            })?;
    Ok(UploadReport {
        transferred_bytes: next_metrics.dedup_metrics.new_bytes + next_metrics.dedup_metrics.shard_bytes_uploaded,
        defrag_gap_bytes: chunk_overlap_bytes as i64 - accepted_dedup_bytes as i64,
        raw_defrag_gap_bytes: chunk_overlap_bytes as i64 - next_metrics.dedup_metrics.deduped_bytes as i64,
        accepted_dedup_bytes,
        chunk_overlap_bytes,
        next_internal_repeat_bytes,
        base: base_metrics,
        next: next_metrics,
    })
}

/// Bytes of `next` in chunks found in `base`, and bytes in chunks not in `base` but seen earlier in `next`.
fn chunk_reuse(base_file: &[u8], next_file: &[u8]) -> (u64, u64) {
    let base_hashes = hash_set(&chunk_file(base_file));
    let mut seen_in_next = HashSet::new();
    let mut overlap_bytes = 0;
    let mut internal_repeat_bytes = 0;
    for chunk in chunk_file(next_file) {
        if base_hashes.contains(&chunk.hash) {
            overlap_bytes += chunk.bytes;
        } else if !seen_in_next.insert(chunk.hash) {
            internal_repeat_bytes += chunk.bytes;
        }
    }
    (overlap_bytes, internal_repeat_bytes)
}
