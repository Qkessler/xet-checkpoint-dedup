# Agent guide

This repo measures how much of a model checkpoint Xet storage reuses when the next step's checkpoint is uploaded in full. Start with `docs/plan.md`. It holds the goal, the steps, and when each step is done. `CONTEXT.md` is the glossary, and `docs/adr/` records the decisions behind the method.

## Work on your own

This is self-driven research. Decide, record why, and keep going. Ask the owner only when a hard stop below applies, or when you are still stuck after you have fought the problem.

**Hard stops. Always ask first:**

- Anything that needs a credential, writes to the network, or publishes. That includes creating a GitHub repo, pushing, uploading to the Hub, and posting comments.
- Anything that would go past the one-day budget in `docs/plan.md`.

**Fight first, then ask.** This applies to a broken invariant, a training step slower than 2 minutes, a crate API that differs from what `docs/background/` says, and any result that looks surprising.

1. Write at least 3 hypotheses in `docs/run-log.md`.
2. Test each one and log what happened.
3. Run `/code-review` on the current state and act on what it finds.
4. Only if you are still stuck after that, or after about 2 hours on one problem, stop and ask.

**Never "fix" a surprising result until it looks expected.** Investigate it, log it, and flag it at the top of the final report.

## Loops

**Inner loop (`just check`, under about 30 s, offline).** It runs nightly `rustfmt --check`, `clippy -D warnings`, `cargo test` (unit tests plus `proptest` properties on tiny synthetic safetensors files), the chunker golden test (once the reference file is cached), and the README staleness check. Do not start the next step of the plan until it is green.

**Outer loop (`just e2e`, minutes).** It runs the tool on real checkpoints and asserts the invariants listed in `docs/plan.md`. If an invariant breaks, investigate. Never relax a check to make it pass.

**Review.** Run `/code-review` at the end of every step of the plan, before committing that step. Answer every finding in the run log, either as fixed (name the commit) or as rebutted (give the evidence).

## Commits

Commit at meaningful checkpoints: units an on-call engineer would want to roll back on their own. Examples: "Add overlap subcommand with tensor attribution", or "Add perturbation sweep". Do not commit every green test run. Each commit must leave `just check` green.

Use an imperative subject of at most 72 characters, with no body, and no `Ticket:` or `Session:` lines.

## Run log

`docs/run-log.md` is append-only, with dated entries covering what you ran, what you saw, and what you decided. Put the session link at the top. Use it as the lab notebook. Results and decisions that outlive the session move to the README or an ADR.

## Numbers

Never type a result number by hand. Every run writes `results/<name>.json`, stamped with the tool's git commit, the crate versions, the input sha256s and HF revisions, and the seed. The README tables and charts are generated from those files between marker comments.

## Code style

Follow xet-core's conventions so the code reads like upstream:

- Format with xet-core's `rustfmt.toml` (copied here): nightly `rustfmt`, 120 columns.
- `cargo clippy -- -D warnings` must be clean.
- Use `//!` for module docs and `///` for public items. Explain purpose, invariants, units, and failure modes, and don't restate the code. Avoid other comments unless they are needed.
- Use `thiserror` in library code and `anyhow` in the binary. Use `unwrap` and `expect` only in tests or for invariants you have justified.
- Use full descriptive names, and include units in numeric names (`chunk_bytes`, `step_gap`).
- Pin the toolchain to the stable version xet-core CI uses (`1.95.0`, via `rust-toolchain.toml`), and use nightly only for `fmt`. Commit `Cargo.lock`.

Python is for glue only: downloads, generating the training run, charts, and the numpy oracle. Run it with `uv`, and use polars for any tabular work.

## Writing style

This applies to every doc: README, ADRs, run log, plan, and commit subjects.

- Use the inverted pyramid. Put the answer first, then the evidence, then the detail.
- Write in the active, direct voice. "The chunker splits the file", not "the file is split".
- No weasel words ("arguably", "somewhat", "it seems", "fairly") and no pedantic words. Clear beats precise and obscure.
- Use plain ASCII in anything that may be pasted elsewhere. No arrows, star glyphs, or emoji.
- Use the terms in `CONTEXT.md`, and none of the words it lists under `_Avoid_`.

## Environment

- `/usr/bin/env bash` shebangs fail in the agent harness. Call `bash script.sh` explicitly, or keep logic in the justfile and Rust.
- Capture long build and test output to `~/.tmp/` (never `/tmp`, which is RAM-backed here), read what you need, and then delete the file.
- Checkpoints and caches live in `~/.cache/xet-checkpoint-dedup/`, outside git. `data/manifest.toml` lists every input, and `just fetch` restores them.
- `local/` is gitignored. It holds the owner's private notes and must never be committed or quoted in public files.
