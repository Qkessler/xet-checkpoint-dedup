# Draft: docs drift in the Xet environment variable docs

Prepared, not opened. Opening a PR writes to GitHub, which is a hard stop in AGENTS.md, so the owner decides whether and where to send these. Every value was checked against xet-core at `d2f667c` (`xet_runtime/src/config/groups/*.rs`); see `docs/background/xet-stack.md` section 3.4.

## hub-docs: `docs/hub/xet/using-xet-storage.md` (`hub-docs-xet-env-table.patch`)

Title: "Update Xet environment variable defaults to match hf-xet"

The environment variable table on the Xet storage page has drifted from the defaults in hf-xet. This updates the rows that changed and adds the telemetry variables.

- `HF_XET_CLIENT_READ_TIMEOUT`: 120s to 300s. The default was raised in xet-core #808 (`client.rs`, `read_timeout`).
- `HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY`: 1 to 2, and `HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY`: 1 to 4 (`client.rs`).
- The three download buffer defaults are now derived from usable memory in both standard and HP mode (xet-core #943, `reconstruction.rs`, `utils/system_memory.rs`): usable / 16, / 64, / 4 in standard mode and / 8, / 32, / 2 in HP mode, clamped between shared floors and the old HP constants. The table lists the derivation, the clamps, and the old constants as the fallback, and names `HF_XET_DISABLE_MEMORY_DERIVED_DOWNLOAD_BUFFERS`.
- A new Telemetry table with all six `HF_XET_TELEMETRY_*` variables and the HF-wide opt-outs that also disable telemetry (`telemetry.rs`).

## xet-core: `HF_XET_LOG_FILE` is `HF_XET_LOG_DEST` (`xet-core-log-dest.patch`)

Title: "Docs: HF_XET_LOG_FILE is HF_XET_LOG_DEST"

The debugging snippets in `README.md`, `scripts/diag/README.md` and `AGENTS.md`, and the doc comment on `init_logging` in `xet_pkg/src/lib.rs`, name `HF_XET_LOG_FILE`, which no code reads. The variable is `HF_XET_LOG_DEST` (`xet_runtime/src/config/groups/log.rs`). Its default is also not stdout: when unset, logs go to the `logs/` directory in the Xet cache.

## Left out on purpose

- `ac_min_upload_concurrency`: the doc comment says 2, but the value is 1. The hub-docs table already says 1, so the fix belongs in a code comment, as a separate xet-core PR.
- The "4MB" global dedup spacing comment in `data.rs` (256 chunks is about 16 MiB), and the xorb spec page's "no explicit limit on number of chunks", which conflicts with the 8,192 constant. Both are spec or code-comment fixes, not docs-table fixes.

## Check before sending

`git apply --check` passes for both patches, against hub-docs `main` as fetched on 2026-10-04 and the xet-core clone at `d2f667c`. The PR numbers (#808, #943) come from `docs/background/xet-stack.md`, not from the shallow clone's history. Rebase if either file has moved since then.
