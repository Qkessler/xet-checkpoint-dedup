# Rust on the Xet crates for every measurement

All measurements come from one Rust binary built on the crates.io `xet-data` and `hf-xet` crates, pinned to 1.6.0 to match xet-core commit `d2f667c`. `overlap` uses `xet_data::deduplication::Chunker` directly and attributes bytes to tensors. `upload` uses the session API, the same code that the Python `hf_xet` package wraps, against a `local://` store, and reports `DeduplicationMetrics` for the next checkpoint. Python is used only to download checkpoints, generate training runs, and draw charts from files the Rust tool writes.

The obvious path is Python on `hf_xet` or `huggingface_hub`. We avoid it because the point is to work directly on the engine the target team maintains, and because per-tensor attribution needs chunk boundaries that the Python API does not expose.

## Considered Options

- **Python `XetSession` for the ground truth**: the same engine underneath, but it adds a second language to the measurement path and hides chunk boundaries.
- **`xtool dedup` dry run**: needs the `internal-tools` feature build and would only repeat what `upload` reports.
- **Rust binary on the published crates** (chosen): one tool, exact upstream line references, and findings that map straight onto xet-core code.

## Consequences

- Roll-ups by layer and by tensor kind are done in Rust with plain aggregation, not the `polars` crate, so the tool's own output carries the answer.
- Code follows xet-core's `AGENTS.md` conventions: its `rustfmt.toml`, `clippy -D warnings`, `//!` and `///` docs, `thiserror` in library code, and `anyhow` in the binary.
