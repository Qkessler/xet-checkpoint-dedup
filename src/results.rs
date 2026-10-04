//! Result files: every run writes `results/<name>.json`, stamped with the tool's git commit, the Xet crate
//! versions, each input's sha256 and Hub revision, and the seed when there is one.
//!
//! Hub provenance is looked up by sha256 in `data/manifest.toml`, so an input that is not listed there is still
//! stamped with its path and hash, just without a repo and revision.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;
use sha2::{Digest, Sha256};

/// The repository root, where `data/manifest.toml` and `results/` live.
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The tool build that produced a result.
#[derive(Debug, Clone, Serialize)]
pub struct ToolStamp {
    pub git_commit: String,
    /// True when the working tree had uncommitted changes, so `git_commit` alone does not reproduce the tool.
    pub git_dirty: bool,
    pub crate_versions: BTreeMap<String, String>,
}

impl ToolStamp {
    /// Reads the commit and dirty state from git at run time. Changes under `results/` do not count as dirty,
    /// because runs write there. Without git, the commit is `unknown` and the tree counts as dirty.
    pub fn current() -> Self {
        let git = |args: &[&str]| {
            Command::new("git")
                .arg("-C")
                .arg(repo_root())
                .args(args)
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        };
        let crate_versions = env!("CKPT_DEDUP_CRATE_VERSIONS")
            .split(',')
            .filter_map(|pair| pair.split_once('='))
            .map(|(name, version)| (name.to_string(), version.to_string()))
            .collect();
        Self {
            git_commit: git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string()),
            git_dirty: git(&["status", "--porcelain", "--", ".", ":!results"]).is_none_or(|status| !status.is_empty()),
            crate_versions,
        }
    }
}

/// One input file of a run.
#[derive(Debug, Clone, Serialize)]
pub struct InputStamp {
    pub role: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub manifest_key: Option<String>,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub file: Option<String>,
}

impl InputStamp {
    /// Stamps an input already read into memory. `role` names its part in the run, such as `base` or `next`.
    pub fn of(role: &str, path: &Path, contents: &[u8]) -> Self {
        let sha256 = sha256_hex(contents);
        let manifest_entry = manifest_entry_with_sha256(&sha256);
        let field = |key: &str| {
            manifest_entry
                .as_ref()
                .and_then(|(_, entry)| entry.get(key))
                .and_then(|value| value.as_str())
                .map(str::to_string)
        };
        Self {
            role: role.to_string(),
            path: path.display().to_string(),
            bytes: contents.len() as u64,
            sha256: sha256.clone(),
            repo_id: field("repo_id"),
            revision: field("revision"),
            file: field("file"),
            manifest_key: manifest_entry.map(|(key, _)| key),
        }
    }
}

/// Lowercase hex sha256 of `contents`, as Hub LFS pointers and `sha256sum` print it.
pub fn sha256_hex(contents: &[u8]) -> String {
    Sha256::digest(contents).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn manifest_entry_with_sha256(sha256: &str) -> Option<(String, toml::Table)> {
    let manifest: toml::Table = fs::read_to_string(repo_root().join("data/manifest.toml")).ok()?.parse().ok()?;
    let inputs = manifest.get("inputs")?.as_table()?;
    inputs.iter().find_map(|(key, entry)| {
        let entry = entry.as_table()?;
        (entry.get("sha256")?.as_str()? == sha256).then(|| (key.clone(), entry.clone()))
    })
}

/// The envelope written to `results/<name>.json`.
#[derive(Debug, Serialize)]
pub struct ResultFile<'a, T: Serialize> {
    pub name: &'a str,
    pub kind: &'a str,
    pub tool: ToolStamp,
    pub inputs: &'a [InputStamp],
    pub seed: Option<u64>,
    pub result: &'a T,
}

/// Writes `<results_dir>/<name>.json` and returns its path.
pub fn write_result<T: Serialize>(
    results_dir: &Path,
    name: &str,
    kind: &str,
    inputs: &[InputStamp],
    seed: Option<u64>,
    result: &T,
) -> std::io::Result<PathBuf> {
    fs::create_dir_all(results_dir)?;
    let path = results_dir.join(format!("{name}.json"));
    let envelope = ResultFile {
        name,
        kind,
        tool: ToolStamp::current(),
        inputs,
        seed,
        result,
    };
    let mut json = serde_json::to_string_pretty(&envelope).map_err(std::io::Error::other)?;
    json.push('\n');
    fs::write(&path, json)?;
    Ok(path)
}
