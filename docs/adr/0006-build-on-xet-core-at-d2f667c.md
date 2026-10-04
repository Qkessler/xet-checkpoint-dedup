# Build on xet-core at d2f667c, not crates.io 1.6.0

ADR 0002 pinned the crates.io `hf-xet` and `xet-data` 1.6.0 release. That release miscounts dedup when defrag prevention refuses a match: the match lands in `deduped_bytes` and is then counted again as new data, so `deduped_bytes` overstates the dedup taken, and the reported file size exceeds the real one by `defrag_prevented_dedup_bytes` (ADR 0005). xet-core #931 fixed the accounting, and no release after 1.6.0 exists on crates.io yet. The tool therefore depends on xet-core by git at `d2f667c`, the commit the background research read and that the chunker oracle already builds from. Every number comes from the engine with the fix.

The fix changes only accounting. `defrag_prevention.rs` and the chunker are identical between 1.6.0 and `d2f667c`, so which chunks are sent does not change. Re-measuring confirmed it: transferred bytes are byte-identical on every pair, and only `deduped_bytes` and the reported file size moved.

## Considered Options

- **Stay on crates.io 1.6.0 and derive accepted dedup** (ADR 0005): correct numbers, but the raw metrics stay wrong, and results come from an engine with a known bug.
- **Patch 1.6.0 locally with `[patch.crates-io]`**: reproduces the fix, but the engine then matches neither a release nor an upstream commit.
- **Git dependency on xet-core at `d2f667c`** (chosen): an exact upstream commit with the fix. Anyone can rebuild it from the public repo.

## Consequences

- Result stamps record the Xet crates as `1.6.0+git.d2f667c` (`build.rs` reads the git source from `Cargo.lock`), because the version number alone no longer says which code ran.
- The outer loop asserts the fix on every pair: `deduped_bytes == file bytes - new_bytes`, and the reported file size equals the size on disk. A regression in the accounting breaks `just e2e`.
- ADR 0005's derived `accepted_dedup_bytes` stays as the definition of the defrag gap. It now equals the raw metric, and the check above keeps it that way.
- `d2f667c` adds client telemetry. xet-core emits none for `local://` stores, and `upload` also turns it off in the session config, so a measurement never writes to the network.
- When xet-core publishes a release with #931, switch back to crates.io at that version.
