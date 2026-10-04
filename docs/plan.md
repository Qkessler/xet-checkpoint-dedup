# Plan: does Xet dedup consecutive RL checkpoints?

The delta-weight-sync blog (https://huggingface.co/blog/delta-weight-sync, 2026-05-27) says that if you uploaded full checkpoints every RL step instead of sparse deltas, "Xet would _still_ only transfer the changed chunks". This repo tests the claim's implied meaning: that full uploads cost about as little as sparse deltas. Rough arithmetic says they do not. About 1% of bf16 elements change per step, a chunk holds about 32,768 of them, so almost every chunk contains a change. We measure it with Xet's own Rust crates, explain it tensor by tensor, and find the change density at which chunk dedup starts to work.

The verdict rule is fixed in ADR 0001: transfer within 3x of the sparse delta means "holds", and at least 50% of the anchor means "does not hold".

Budget: one working day. Run the steps in order. If time runs short, cut from the bottom of "Cut order".

## Expected direction (not a result)

Prior evidence, unverified by us: a background research pass compared reconstruction terms for public consecutive checkpoints on the Hub and found 0 to 0.48% shared chunks (`docs/background/checkpoint-data-sources.md`, section 4). If our numbers come out far above that on scattered real data (for example, chunk overlap above 50%), treat it as a surprising result. AGENTS.md says how to handle one.

## The tool

There is one Rust binary, working name `ckpt-dedup`, built on crates.io `xet-data` and `hf-xet` 1.6.0 (ADR 0002). The background research for these crates was done against xet-core at commit `d2f667c`, cloned at `/workplace/enrikes/huggingface-xet/xet-core`. Use that clone to read the real API, then depend on the published crates. Amended by ADR 0006: the published 1.6.0 crates miscount dedup when defrag prevention refuses a match, so the tool depends on xet-core at `d2f667c` by git, which has the fix.

**`overlap <base> <next>`.** This subcommand:

- chunks both files with `xet_data::deduplication::Chunker` (`Chunker::new(target)`, `next_block(&[u8], is_final)`, `finish()`);
- collects the base's chunk hashes;
- marks each of the next file's chunks as reused or new;
- attributes bytes to tensors (see "Attribution");
- compares the two files element by element for the unchanged element fraction.

Each tensor row has: name, layer, tensor kind, dtype, bytes, unchanged element fraction, chunk overlap bytes, and chunk overlap percent. The subcommand also prints roll-ups by layer and by tensor kind (plain Rust aggregation, no polars crate), a whole-file line, and the sparse-delta size (changed elements x 6 bytes: a 4-byte index plus a 2-byte bf16 value). It writes `results/<name>.json`.

**`upload <base> <next>`.** This subcommand uses the session API (`XetSession`, `new_upload_commit`, endpoint `local://<fresh temp dir>`). It commits the base, then commits the next file to the same store, and reports the next file's `DeduplicationMetrics`: `new_bytes`, `deduped_bytes`, `defrag_prevented_dedup_bytes`, and the uploaded xorb and shard bytes. Transferred bytes are `new_bytes` plus the uploaded shard bytes. The defrag gap is `chunk overlap bytes - deduped bytes`. Check how xet-core's own tests use `local://` (`xet_pkg/tests/test_xet_session.rs`).

**`perturb <input> --density P --mode scattered|rows|tensors --seed S`.** This writes a copy of a bf16 checkpoint with a fraction P of its elements altered. Alter an element by flipping the least significant mantissa bit, so the value moves one ulp and stays finite. Leave the header untouched.

- **`scattered`**: picks elements uniformly at random.
- **`rows`**: picks whole contiguous runs (rows of 2-D tensors) until it reaches the same count.
- **`tensors`**: picks whole tensors.

**Attribution.** Each tensor's byte range is `8 + header_len + data_offsets` from the safetensors header. If a chunk covers several tensors, each tensor is credited with the bytes of the chunk inside its range, and inherits the chunk's reused or new status. The header bytes get their own row. The sum over all rows must equal the file size exactly.

## Steps

Every step ends with `just check` green, a `/code-review` pass that is answered in the run log, and one commit.

1. **Scaffold and `overlap`.**
   - Set up the crate, `rust-toolchain.toml`, xet-core's `rustfmt.toml`, and a `justfile` with `check`, `e2e`, `fetch`, `sweep`, `train` and `report`. Run `cargo install just` first, because it isn't installed.
   - Build `overlap` with attribution and the element comparison.
   - Add `proptest` properties 1 to 5 (below) and the chunker golden test.
   - Done when the inner loop is green and `overlap A A` reports 100% on a real checkpoint.
2. **Perturbation sweep.**
   - Build `perturb` and a `just sweep` recipe. Run densities from 1e-6 to 1e-1 (about 3 points per decade), in all three modes, with 5 seeds each, on one real Qwen2.5-0.5B-Instruct bf16 checkpoint.
   - Draw chart 1: chunk overlap against change density, on a log x axis, with the theory curve `exp(-values_per_chunk * P)` and the measured points.
   - Done when the scattered points stay inside the fixed tolerance (below) and the clustered points stay clearly above the curve.
   - This step alone gives a defensible result. Protect it.
3. **Generate the training run.**
   - The setup is in `docs/background/checkpoint-data-sources.md` section 5: Qwen2.5-0.5B-Instruct, TRL GRPO on CPU, AdamW, lr 1e-6, constant schedule, weight decay 0, beta 0, fp32 master weights, and a bf16 snapshot after every optimizer step.
   - Time one step first. Over 2 minutes per step counts as a "fight first" problem. The fix is fewer steps, not a smaller model.
   - Run 11 or more steps, so the step gaps 1, 2, 5 and 10 all exist. Then run a second, short run with half the layers frozen (variant b).
   - Known risk: Qwen2.5-0.5B ties `embed_tokens` and `lm_head`, and `safetensors.torch.save_file` rejects shared tensors. Deduplicate them before saving.
   - Record the config, seed, and library versions in `data/manifest.toml`.
4. **`upload` and the defrag gap.**
   - Build `upload` and run it on every pair from step 3.
   - Done when the outer-loop invariants hold.
5. **Third-party pairs.**
   - `just fetch` downloads `ChenJakcy6/Qwen2-0.5B-GRPO-test` revisions `5012bf807a40` and `1cd8897a89b0` (3 steps apart), and `ajjyy/Qwen2-0.5B-GRPO-attempt1` steps 3150 and 3160 (revisions in the background report, 10 steps apart). These are public reads.
   - Run `overlap` and `upload` on both pairs.
   - Reproduce one row of the research pass's reconstruction-API table by hand, and cite it (ADR 0003).
6. **README and charts.**
   - The README covers, in inverted-pyramid order: the question and answer, the claim quoted verbatim, the verdict rule, results, method, limitations, and future work.
   - Chart 1 is the sweep. Chart 2 is per tensor kind: the unchanged element fraction against chunk overlap. Chart 3 is a table per pair: anchor size, sparse delta, chunk overlap, transferred bytes, defrag gap, and verdict.
   - Generate all of it from `results/`. Python with polars and matplotlib draws the charts.
   - Credit the blog's sparse encoding. The tone is "here is where chunk dedup holds and where it does not", not a takedown.
7. **Docs-drift PR prep (only if time is left).**
   - Prepare, but do not open, a docs fix for the hub-docs Xet env table and the xet-core README (`docs/background/xet-stack.md` section 3.4).
   - Write it as a patch file plus a draft description. Opening it is a hard stop.

## Cut order

If time runs short, cut from the top of this list first:

1. Step 7.
2. The ajjyy pair.
3. Variant b (frozen layers).
4. The hand-reproduced reconstruction-API row.

Never cut steps 1, 2 or 6.

## Tests

**`proptest` properties (inner loop):**

1. Attributed bytes summed over every row equal the file size exactly.
2. `overlap(A, A)` is 100%, and the unchanged element fraction is 1.0.
3. A perturbation with density 0 gives 100% chunk overlap.
4. Rewriting the JSON header with the tensors in a different key order keeps chunk overlap of the data section within a few chunks of 100%.
5. Changing one element costs at most 2 chunks of overlap.

**Oracles (they don't trust our code):**

1. **Golden test, inner loop once cached.** Our chunker wrapper reproduces the `.chunks` file of the spec reference file `Electric_Vehicle_Population_Data_20250917.csv` byte for byte. The file comes from the `xet-team/xet-spec-reference-files` dataset. Cache it under `~/.cache/xet-checkpoint-dedup/`, and pin its sha256 in the test.
2. **Outer loop.** Whole-file chunk overlap computed a second way: run xet-core's `chunk` example (`cargo build --release -p xet-data --example chunk` in the clone) on both files, then intersect the hashes with `sort` and `join`. The result must match `overlap` exactly.
3. **Outer loop.** The unchanged element fraction per tensor, recomputed with numpy for one real pair, must match exactly.

**Outer-loop invariants:**

- `overlap(A, A)` is 100%, and `upload` of A twice gives `new_bytes` of about 0 (shard overhead only).
- Transferred data bytes are at least `next file size - chunk overlap bytes`. The defrag gap is never negative.
- Attributed bytes reconcile with the file size for every pair.
- For the scattered sweep, the mean of 5 seeds is within 3 percentage points of `exp(-values_per_chunk * P)`, or within 3 standard errors, whichever is wider. `values_per_chunk` is the measured mean chunk size divided by 2 bytes. This tolerance is fixed now, so never tune it. Amended by ADR 0007: the tolerance is unchanged, but the curve is now the same survival chance weighted by the base's actual chunk sizes, because chunk sizes vary and overlap is measured in bytes.
- At the same density, clustered modes give more chunk overlap than scattered.
- Oracles 2 and 3 match.

## Data

- **Our runs:** step 3, saved under `~/.cache/xet-checkpoint-dedup/runs/`.
- **Third party:** the pairs in step 5.
- **Optional:** `allenai/OLMo-2-0425-1B-RLVR1` branches `step_200` and `step_400`, only after everything else is done.

`data/manifest.toml` lists each input's source, revision, file and sha256, and `just fetch` restores everything.

## Future work (goes in the README)

- **A real Hub upload of one pair to a private bucket.** It would exercise the server, global dedup and shard registration. We skipped it for offline reproducibility and the time budget (ADR 0003).
- **A Hub reconstruction-API cross-check built into the tool.** It shows what the Hub actually shares between public checkpoints, without downloading them (ADR 0003).
- **fp32 master-weight checkpoints**, and a higher learning rate (1e-5).
- **A one-sentence correction to the blog.** For example: "Xet dedups unchanged chunks, but at RL learning rates bf16 changes land in almost every chunk, so the sparse encoding is what makes this cheap." Discuss it with the blog authors first, rather than sending it cold.
- **The docs-drift fixes**, if step 7 was cut.

## Done means

- `just check` and `just e2e` are green on a clean clone, after `just fetch`.
- The README is generated from `results/`, with no hand-typed numbers.
- Every `/code-review` finding has been answered in the run log.
- The run log opens with a summary: the verdict, any surprising results, and anything that still needs the owner's judgment.
- Nothing has been pushed or published.
