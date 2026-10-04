# Run log

Session link: not generated. Uploading a session export writes to the network, which is a hard stop in AGENTS.md; the owner can add one.

## Summary

**Verdict: does not hold.** On all 46 checkpoint pairs measured, uploading the full next checkpoint to Xet transferred at least half of the anchor, and on our all-trainable run nearly all of it. That is far above the 3x-sparse-delta bar in ADR 0001. Chunk dedup only helps where changes cluster: freezing half the layers reuses about a third of each checkpoint, and that is still "does not hold". The README tables carry the numbers and are generated from `results/`.

**Surprising results, flagged:**

1. **A pre-registered check fails, and it is left failing.** The scattered sweep falls outside the fixed `exp(-values_per_chunk x P)` tolerance in 4 of 16 cells (step 2 entry). The cause is chunk-size variation, and a size-weighted curve matches within a quarter of a point. `just e2e` is therefore red on that one check, and every other invariant and oracle is green.
2. **`hf-xet` 1.6.0 over-reports dedup and file size** when defrag prevention refuses a match (step 4 entry, ADR 0005). The fix is in xet-core after 1.6.0.
3. **The ajjyy pair's Hub row (0 shared chunks) is not comparable** to a local xet-core upload, because the Hub chunked that file differently (step 5 entry).

**Needs the owner's judgement:**

- Whether to replace the plan's theory curve with the size-weighted one, which would turn `just e2e` green with the same tolerance. The plan fixed the curve in advance, so I did not change it.
- Whether ADR 0005's accepted-dedup definition of the defrag gap is acceptable in place of the plan's `chunk overlap - deduped_bytes`.
- Whether and where to send the docs-drift patches in `docs/docs-drift/` (step 7). They are prepared and not opened.
- Nothing has been pushed, uploaded or posted. The only network use was public Hub reads.

**Budget:** all seven steps ran within the day. Nothing from the cut order was cut.

## 2026-10-04 Step 1: scaffold and `overlap`

**Ran.** Installed toolchains 1.95.0 and nightly (fmt only) and `just` 1.58.0. Resolved every Hub input to a full revision sha and its LFS sha256 through the Hub API, wrote `data/manifest.toml`, and ran `just fetch` (scripts/fetch.py). All seven inputs downloaded and matched their pinned sha256.

**Saw: published crates differ from the clone.** The plan's background research read xet-core at `d2f667c`, but crates.io `xet-data` and `hf-xet` 1.6.0 are older than that commit:

- `xet-data` 1.6.0 predates #931. In `FileDeduper::process_chunks`, a dedup range is added to `deduped_bytes` and `total_bytes` before the defrag check, so a range that defrag prevention refuses is counted as deduped and then again, chunk by chunk, as new. In 1.6.0, `deduped_bytes` therefore overstates the dedup actually taken by up to `defrag_prevented_dedup_bytes` (or more, because a refused range is re-queried from the next chunk).
- `hf-xet` 1.6.0 has no `XetSessionBuilder::with_cache_dir` (#996). It does not matter for `local://`: `TranslatorConfig::new` puts the shard cache under the local endpoint directory.

Decision: ADR 0002 pins 1.6.0 and we keep it, because that is what `pip install hf-xet` users run. `upload` will report the raw `DeduplicationMetrics` and also derive the dedup actually taken as `file bytes - new_bytes`, which is correct in both versions because every chunk is either new or deduped. The defrag gap uses the derived number. Hypotheses about this get tested in step 4.

**Saw: `overlap A A`.** On `Qwen/Qwen2.5-0.5B-Instruct`, `overlap A A` reports 100% chunk overlap and an unchanged element fraction of 1.0 for every layer and kind. `just e2e` now writes this as `results/qwen25-self.json` and asserts it, and that file carries the chunk counts. It runs in a few seconds for the 988 MB file.

**Saw: golden test.** `chunk_file` reproduces `Electric_Vehicle_Population_Data_20250917.csv.chunks` byte for byte (796 lines). The test pins both sha256s and skips with a message if the files are not cached.

**Decided.**

- `perturb` (the library half) landed in step 1, because property 3 needs it. The CLI subcommand comes in step 2.
- `overlap` prints roll-ups by default and the per-tensor table with `--per-tensor`. The JSON always carries every row.
- Results are stamped with the git commit and a dirty flag read at run time, the Xet crate versions read from `Cargo.lock` at build time, and each input's sha256. The Hub repo and revision are looked up by sha256 in `data/manifest.toml`.
- Sparse delta size per tensor is changed elements x (4-byte index + element bytes), which is 6 bytes for bf16. A tensor missing from the base ships in full. The header adds nothing.
- `layout::CheckpointLayout::parse` rejects holes or overlaps between tensors, so property 1 holds by construction and the test checks it independently.

**Hypotheses about the crate-version difference** (AGENTS.md "fight first"; tests run in step 4, because only `upload` exercises the dedup path):

1. In 1.6.0, `deduped_bytes` includes ranges refused by defrag prevention. Test: on a pair with defrag-prevented bytes, check `deduped_bytes + new_bytes > file bytes` while `new_bytes` stays consistent with the chunk overlap.
2. `local://` keeps the shard cache inside the endpoint directory, so nothing leaks into `~/.cache/huggingface/xet`. Test: list that directory before and after an `upload` run.
3. A second commit in the same session dedups against the first commit's shard without a global query. Test: `upload A A` gives `new_bytes` of 0.

**Saw: a header that changes length costs more than two chunks.** Property 4 first failed on a shrunk case: two 256 KiB tensors, with the header rewritten to add a 24-byte metadata entry. Inspecting the chunk lists showed why. When no content-defined boundary appears within 128 KiB, the chunker cuts at exactly 128 KiB. Those forced cuts sit at fixed offsets from the previous cut, so a byte shift propagates through every consecutive forced cut and only resyncs at the first content-defined boundary. On uniform random bytes about 16% of chunks end at a forced cut (exp(-120 KiB / 64 KiB)). On the real Qwen checkpoint the share is similar: `results/qwen25-self.json` has `next_max_size_chunks`. A pure key reorder keeps the header length, so the data does not move and the two-chunk bound holds. Decision: property 4 now tests both cases. The bound for a length-changing rewrite adds the base's leading run of max-size chunks, which states the mechanism exactly instead of a looser "few chunks". 256-case runs pass. This matters later: any pair whose header length differs will lose more than one chunk at the start.

### Review of step 1 (`/code-review`, Standards and Spec axes)

Standards:

- Hypotheses missing for the crate-API difference: fixed, listed above.
- Typed numbers in the run log: fixed. The entry now points at `results/qwen25-self.json`.
- "snapshot" in the justfile and "parameter name" in `layout.rs`: fixed.
- Public items without `///`: fixed for every public item named.
- Numeric names without units (`LayoutError` fields, `remaining`, `draws`): fixed (`start_offset_bytes`, `remaining_elements`, `draw_count`, ...).
- Duplicate `sha256_hex` in the chunking test: fixed, it uses `results::sha256_hex`.
- Reference sha256s hard-coded in the test as well as the manifest: rebutted. The plan says "pin its sha256 in the test", so the test does not trust the manifest it is checking.
- Cache path spelled in Rust, the justfile and Python: rebutted. Three languages, one line each, and a shared config file would cost more than it saves.
- `TensorKind::label` restates serde: rebutted. It is used for printing and grouping without a serializer.
- `"BF16"` literal: fixed, `BF16_DTYPE`.
- Header row found by name in a property: fixed, it uses `TensorKind::Header`.
- `TensorKind::Header` is not a tensor kind: rebutted. The plan requires a header row in the tensor table. The enum doc now says so.
- `clap::ValueEnum` on a library type: rebutted. There is one binary and no other consumer.
- Unused `hf-xet` and `tempfile` dependencies: fixed, removed until step 4.
- Duplicate `whole_tensor_run` vector: fixed.
- `git_dirty` ignores untracked files: fixed. It now counts untracked files and excludes only `results/`, which runs write to.

Spec:

- `just e2e` referenced a missing script: fixed. `scripts/e2e.sh` asserts the step 1 invariants (overlap(A, A) is 100% and attribution reconciles).
- `overlap A A` evidence was only typed numbers: fixed by the e2e result file.
- Property 4 too weak (bound larger than the data): fixed. Files are now 512 KiB to 12 MiB, the test asserts the data section spans at least 4 max-size chunks, and the header can also change length (see above).
- Property 1 true by construction: fixed. It now compares every row's reused bytes against an independent quadratic chunk-by-row intersection.
- Property 3 duplicates property 2: rebutted. It is in the plan and exercises `perturb` at density 0.
- Property 5 on files of few chunks: fixed, by the larger files.
- `tensors` mode becomes one contiguous run at low density: accepted as designed. Below the size of the smallest tensor, no whole tensor fits the exact count, so the mode alters a prefix of one random tensor, and it differs from `rows` only in where the run starts. Both are clustered perturbations, which is what the comparison needs. The sweep reports it.
- Flipping the last bit of Inf gives NaN: fixed. `perturb` returns `NonFiniteElement` instead, with a unit test. Synthetic test data is now always finite.
- `altered_elements` reports the target count: rebutted. Indices are distinct by construction, and a unit test counts the changed elements independently.
- The sparse delta generalised to `4 + element bytes`, and the extra `--results-dir` and `git_dirty`: kept. For bf16 the delta is the spec's 6 bytes.

## 2026-10-04 Step 2: perturbation sweep

**Ran.** `perturb` (CLI) and a `sweep` subcommand. The sweep perturbs in-memory copies on 8 worker threads, so 240 points (16 densities from 1e-6 to 1e-1 at 1, 2 and 5 per decade, 3 modes, seeds 0 to 4) never write a 1 GB file. A sweep takes about 30 s. `just e2e` checks that `perturb` on disk followed by `overlap` gives exactly the in-memory point for scattered 1e-5 seed 0, and that oracle 2 (xet-core's `chunk` example at the clone, then `sort` and `join`) gives exactly our whole-file overlap. Both hold. The clone's chunker, hashing and chunk constants are byte-identical to crates.io 1.6.0 (`diff`), so the oracle runs the same algorithm through an independent code path.

**SURPRISING: the scattered points break the pre-registered theory tolerance.** Between densities 5e-6 and 5e-5 the measured scattered overlap sits 3.7 to 9.1 percentage points below `exp(-values_per_chunk * P)`, outside max(3 pp, 3 SE). Above 1e-4 it sits above the curve. `just e2e` reports the four cells and fails. Per AGENTS.md the check stays exactly as written: same curve, same tolerance.

Hypotheses and tests:

1. *Chunk sizes vary, and byte-weighted overlap is not the overlap of a mean-size chunk.* A chunk of `s` bytes survives with probability `(1-P)^(s/2)`, so the byte-weighted expectation is `sum(s (1-P)^(s/2)) / sum(s)`. Big chunks are both likelier to be hit and carry more bytes, which pulls the mid-range below the curve. Small chunks survive longer, which pushes the tail above it. Test: the sweep computes this size-aware curve from the base's actual chunk sizes. It matches every scattered cell within 0.25 pp. The base's chunk sizes have a coefficient of variation of 0.64, and 15.8% of chunks sit at the 128 KiB cap (`base_chunk_bytes_coefficient_of_variation` and `base_max_size_chunk_share`). **Confirmed.**
2. *Changes move chunk boundaries and cost extra chunks.* Test (`cells[].mean_next_to_base_chunk_ratio` and `mean_reused_chunk_fraction` in `results/sweep.json`): up to 1e-3 the next file's chunk count stays within 0.07% of the base's, and the share of reused chunks matches the per-chunk prediction `mean((1-P)^(s/2))` within 0.3 pp. Boundaries do start to move at high density (the count is 1% higher at 1e-1), but by then overlap is already zero. **Rejected** as a material effect.
3. *`perturb` alters more elements than asked, or not uniformly.* Test: the unit test counts changed elements independently, and the chunk-count match in hypothesis 2 would fail if the effective density were off. **Rejected.**
4. *The header and other non-element bytes skew the byte fraction.* The header is under 0.01% of the file and could only raise overlap, while the gap is negative. **Rejected** by its sign and size.

Decision: leave the plan's check failing and flag it for the owner. The plan curve uses mean chunk size, and that is the wrong model for byte-weighted overlap. Hypothesis 1 explains the gap, and the size-aware curve passes the same fixed tolerance with 10x margin. Swapping the gate's curve is the owner's call, because the plan fixed it in advance. Chart 1 shows both curves. The conclusion does not depend on which curve is used: both reach about zero by 1e-3, and real RL steps change about 1e-2.

**Saw: the clustered modes hold up.** At every density both clustered modes sit above scattered, as the invariant requires. At 1e-2 (the blog's per-step change rate) whole-row perturbations keep about two thirds of bytes reused and whole-tensor perturbations about 99% (`results/sweep.json`).

**Decided.**

- Chart 1 is `docs/charts/chunk-overlap-vs-density.png`, drawn by `scripts/report.py` from `results/sweep.json`. Error bars are 3 standard errors, too small to see at this scale.
- `e2e.sh` runs every check even after one fails, and exits non-zero if any failed, so one broken invariant does not hide the others.
- Each step lands as a code commit plus a results commit. Result files are stamped with the code commit that produced them, and the results commit follows it.

### Review of step 2 (`/code-review`, Standards and Spec)

- Result files stamped with a commit that lacks the sweep code: fixed by process. The code commit lands first, then `just e2e` regenerates the results and they land in a results commit.
- `report.py --check` returned success without checking: fixed in the docstring, which now says it passes trivially until the README exists. The real check comes with the README in step 6.
- `# noqa` comments in report.py: fixed, `matplotlib.use("Agg")` moved into `main`.
- Unjustified `expect` on locks: fixed, the `run_sweep` doc states the no-poisoning invariant.
- "snapshot" in `snapshot_dtype`: fixed, renamed `checkpoint_dtype` in the train script, the manifest and both run.json files.
- `eligible_elements` overwritten per point: kept (the value is the same for every point), but the variable is renamed so the reason is visible.
- `(mode, density, seed)` data clump: rebutted. It travels through three call sites and a `PerturbSpec` type would add more than it removes.
- Step 2 is not done, because 4 scattered cells sit outside the fixed tolerance: agreed. The step's code is committed with the failure flagged, and the owner's call is pending. The plan protects step 2 as the defensible result, and the clustered-vs-scattered invariant, both oracle checks, and the size-aware explanation all hold.
- e2e checked the committed sweep artifact, not the current code: fixed. `e2e.sh` now reruns the sweep (about 30 s).
- Manifest run entries are step 3 work: fixed by leaving `data/manifest.toml` out of the step 2 commit.
- Oracle 2 build silenced and the path hard-coded: fixed. The build runs under `check`, and `XET_CORE_DIR` overrides the clone path.

## 2026-10-04 Step 3: training runs

**Ran.** `scripts/train_grpo.py` (via `just train`): Qwen2.5-0.5B-Instruct at the pinned revision, TRL 1.14.1 GRPO on CPU, `adamw_torch_fused`, lr 1e-6 constant, weight decay 0, beta 0, fp32 weights (`dtype=float32`, `bf16=False`, which overrides TRL's `bf16=True` default), and a bf16 checkpoint written in `on_step_end`, after the optimizer step. `lm_head.weight` is dropped because it is tied to `embed_tokens`.

- `run-a`: 21 steps, seed 0, all layers trainable. It writes checkpoints for steps 0 to 21, so step gaps 1, 2, 5, 10 and 20 all exist.
- `run-b-frozen-half`: 11 steps, seed 0, with `model.layers.0` to `11` frozen (variant b).

**Saw.**

- One step takes 2 to 5 s on 32 cores (`step_seconds` in each run.json). The background estimate was 20 to 60 s. The difference is 48-token completions, 4 generations and a 0.5B model. Far below the 2-minute limit, so no fight-first.
- Step 0 of both runs has the same sha256 as the published `model.safetensors`. The fp32 load and bf16 cast round-trip exactly, and our save path writes the same header and tensor order as Hugging Face's, so our pairs are byte-comparable with the published file.
- Rerunning both runs with the same seed reproduced every step's sha256 on this host. CPU kernels are deterministic for a fixed thread count (recorded as `torch_threads`). Another host or thread count may not reproduce them bit for bit, so `just fetch` only verifies generated entries and points at `just train`.
- The step-3 reviewer checked that run-b's frozen tensors are bit-identical between step 0 and step 11, and that some trainable tensors (layernorms, k/v biases) also never changed at lr 1e-6. bf16 absorbs those updates.

### Review of step 3

- `just fetch` ignored the `path` of generated entries: fixed, fetch now reads `path`, and all 41 entries verify.
- No proof of reproducibility: fixed by the rerun above.
- No run-log entry: fixed, this entry.
- Config recorded only in part: fixed. The run record adds Adam betas and eps, `max_grad_norm`, `gradient_checkpointing`, `loss_type`, `temperature` and `torch_threads`.
- "snapshot" in the script: fixed, renamed to checkpoint throughout.
- `json.dumps` as a TOML writer: fixed. `tomlValue` accepts only scalars and flat lists and raises on anything else.
- A crash mid-run left the manifest pointing at deleted files: fixed. The run's manifest block is removed before the old step files are deleted.
- `--learning-rate` and `step_seconds` count as scope creep: kept, both are small. The learning rate is the one knob the future-work item needs.

## 2026-10-04 Step 4: `upload` and the defrag gap

**Ran.** An `upload` subcommand on the session API: one `XetSession`, a fresh `local://` store under `~/.cache/xet-checkpoint-dedup/scratch/` (on disk, not the RAM-backed `/tmp`), the base committed, then the next file committed to the same store. `scripts/pairs.sh` (`just pairs`) runs `overlap` and `upload` on 44 pairs:

- `run-a` steps n to n+1 for n = 0 to 20, plus gaps 2, 5, 10 and 20.
- `run-b-frozen-half` steps n to n+1 for n = 0 to 10, plus gaps 2, 5 and 10.

`just e2e` now also runs `upload(A, A)`, every pair, the per-pair invariants, and oracle 3.

**Hypotheses from step 1, tested.**

1. *1.6.0 counts defrag-refused matches as deduped.* **Confirmed, with a further effect.** On run-a steps 0 to 1, `deduped_bytes` equals the full chunk overlap, even though `defrag_prevented_dedup_bytes` is non-zero. The file size the upload path reports (`XetFileInfo::file_size`, which 1.6.0 takes from `total_bytes`) exceeds the real file size by exactly `defrag_prevented_dedup_bytes`. `upload` therefore reads the file size from disk and derives the accepted dedup as `file bytes - new_bytes`. On every pair, the resulting defrag gap equals `defrag_prevented_dedup_bytes` (results `upload-*.json`). The reported size goes into `reported_file_bytes` so the bug stays visible. Upstream #931 fixes it after 1.6.0.
2. *`local://` keeps its shard cache inside the store.* **Confirmed.** No file under `~/.cache/huggingface/xet` changed during an upload run, and the scratch store is removed afterwards.
3. *A second commit dedups against the first commit's shard.* **Confirmed.** `results/upload-qwen25-self.json` has `new_bytes` 0 and no xorb bytes. Only a shard is sent.

**Saw.**

- **run-a (all layers trainable).** Chunk overlap stays between a quarter of a percent and under 3% of the file, almost all of it in the embedding. Transferred bytes are 98% to 99.8% of the anchor, 5x to 55x the sparse delta. The sparse delta per step shrinks as training goes on, and so the ratio grows. Under ADR 0001 that is "does not hold" for every pair.
- **run-b (layers 0 to 11 frozen).** Chunk overlap is about 36.5% of the file, which is the frozen layers' share of bytes (12 x 29.8 MB of 988 MB) plus the few unchanged chunks of the embedding. Transfer is about 63% of the anchor, still over the 50% line: "does not hold".
- **Defrag gap.** Up to about 1% of the file on run-a's consecutive pairs, and 0 on long gaps, where the remaining overlap is too thin to trip defrag prevention.
- **Invariants.** `new_bytes >= next size - chunk overlap` and a non-negative defrag gap hold on every pair. No pair has chunks repeated inside the next file.
- **Oracle 3.** numpy, with its own header parser, reproduces all 290 per-tensor unchanged element counts of run-a steps 0 to 1 exactly.

### Review of step 4

- `file_bytes - new_bytes` was a bare `u64` subtraction: fixed, `checked_sub` with an `UploadError::NewBytesExceedFile`.
- `upload` read each file three times: fixed. `main` reads each input once and passes the bytes for both the stamp and the chunk overlap. Re-chunking inside `upload` is kept, so an upload result stands on its own. The reviewer confirmed that it agrees with `overlap` on all 44 pairs.
- Dead "explained by repeats" branch in the invariant: fixed, removed. Repeats inside the file never excuse a failure.
- `next_internal_repeat_bytes` is always 0: kept. It is the measured reason the invariant needs no repeat allowance.
- "range" (an _Avoid_ word for Term) in `upload.rs` and the log: fixed, now "match".
- The justfile `pairs` comment promised third-party pairs: fixed, `pairs.sh` now also runs both Hub pairs.
- Results stamped with a dirty older commit: fixed by process (code commit, then `just e2e` restamps, then a results commit).
- `results/upload-qwen25-self.json` missing: the earlier `upload A A` run wrote to `results/tmp`. The full `just e2e` after this commit writes it and checks it.
- The defrag-gap redefinition lived only in the run log: fixed by ADR 0005. The "#931" claim was checked by diffing the crates.io source against the xet-core clone at `d2f667c` (step 1). The clone sits outside the repo, which is why the reviewer could not see it.
- `upload(A, A)` check weaker than "shard overhead only": fixed. It now also asserts that shard bytes are above 0 and that transferred bytes equal the shard bytes.
- `new_bytes` is uncompressed: noted in ADR 0005 and goes under the README limitations.
- run-b's 36.5% explanation read as "the whole embedding": fixed in the wording.

## 2026-10-04 Step 5: third-party pairs

**Ran.** `just fetch` already holds both Hub pairs. `pairs.sh` now runs `overlap` and `upload` on `ChenJakcy6/Qwen2-0.5B-GRPO-test` steps 110 to 113 and `ajjyy/Qwen2-0.5B-GRPO-attempt1` steps 3150 to 3160. `scripts/hub_reconstruction.py` reproduces the research pass's reconstruction-API rows for both pairs (public reads only: the tree API, an anonymous `xet-read-token`, and `GET /v1/reconstructions`). It writes `results/hub-reconstruction-*.json`.

**Saw: ChenJakcy6 (3 steps apart).** The reconstruction-API row reproduces exactly: 202 of 15,336 chunks of the next file are shared with the base on the Hub, the same as `docs/background/checkpoint-data-sources.md` section 4 (ADR 0003 asks for this one row by hand). Chunking s113 locally with xet-core's `chunk` and `hash -h file` examples gives exactly the Hub's xetHash, so the Hub's chunk boundaries for this file are xet-core's. Locally, chunk overlap is 1.2% and the dedup `upload` accepts is 0.54%. The Hub's 202 shared chunks, which come from whole shared terms of 4.7 MB, land in the same place: defrag prevention refused about half of the overlap in both.

**Saw: ajjyy (10 steps apart), a mismatch that is explained.** On the Hub, s3160 shares 0 chunks with s3150, and its recipe is 15 terms over 15 xorbs that it uses whole, so no dedup happened at upload. Locally, chunk overlap is 11.3% and `upload` accepts 7.3%. Hypotheses:

1. *The Hub chunked this file with a different chunker.* Test: our local Xet file hash of s3160 differs from the Hub's xetHash, and the Hub has 15,859 chunks against our 15,292. **Confirmed.** The file was not chunked by xet-core's current chunker. That fits an upload through the LFS bridge followed by server-side migration, or an older client. The Hub page does not say which, so we cannot pin the path down.
2. *The upload path never compared against the previous revision.* Whole-xorb terms with no reference to any base xorb (`next_xorbs_also_in_base` is 0) mean no dedup was attempted or found. **Consistent**, but not separable from hypothesis 1 with public data.
3. *Our 11.3% is an artifact.* Test: the overlap sits almost entirely in the embedding (40% of its bytes reused, 99.996% of its elements unchanged), where only rows of tokens seen in training change. The matrices change in about 0.05% of elements and reuse about 0.2% of their bytes, close to the scattered sweep at that density. Attribution reconciles and the invariants hold. **Rejected.**

Decision: report both the local measurement (what xet-core 1.6.0 would transfer) and the Hub row (what this repo's uploads actually stored), with the chunker mismatch named. Neither changes the verdict. ajjyy transfers about 93% of the anchor locally, which is "does not hold".

## 2026-10-04 Step 6: README and charts

**Ran.** `scripts/report.py` (`just report`) draws chart 1 (sweep) and chart 2 (per-tensor and per-kind unchanged elements against chunk overlap, four pairs as small multiples). It rewrites seven README blocks between marker comments: the answer, the chart 3 pair table, the full pair table, the sweep table, the Hub table and provenance. `report.py --check` regenerates those blocks in memory and fails `just check` when README.md differs; the reviewer confirmed this by editing a result value and by deleting a result file. Every result file was regenerated by `just e2e` at commit `fd19936` on a clean tree, and the provenance block says so.

**Decided.**

- Chart 3 is a table: one summary row per run's consecutive pairs (median, min and max), one row per longer step gap, and one row per third-party pair. The full 46-row table sits in a collapsed section.
- Chart 2 plots the changed fraction (1 - unchanged element fraction) on a log axis instead of the unchanged fraction on a linear one. That is the same information, but it lines up with chart 1, and on a linear axis every tensor sits between 0.97 and 1.0. Tensors with no changes sit at a marked 1e-7.
- Python glue is formatted with `ruff format --line-length 120`. The remaining lines over 120 columns are long f-strings.

## 2026-10-04 Step 7: docs-drift PR prep

**Ran.** Each drift item in `docs/background/xet-stack.md` section 3.4 was checked against the xet-core clone at `d2f667c`. Two patches went into `docs/docs-drift/`, both passing `git apply --check` against their upstream files:

- `hub-docs-xet-env-table.patch`: read timeout, initial concurrencies, memory-derived download buffers in both modes, and a new telemetry table.
- `xet-core-log-dest.patch`: `HF_XET_LOG_FILE` becomes `HF_XET_LOG_DEST` in the README, `scripts/diag/README.md`, `AGENTS.md`, and the `init_logging` doc comment.

`draft-description.md` holds the PR text and the items deliberately left out. Opening either PR is a hard stop, so both wait for the owner.

### Review of steps 6 and 7

- Unsupported prose claims in the README (embedding rows "of tokens that never appear", "sparse deltas compress too", "larger models change more"): fixed. Each is reworded to what the results show, or attributed to PULSE.
- _Avoid_ words "values" (for elements) and "weights": fixed, now "elements per chunk" and "fp32 master copy".
- Long lines and repeated `loadResult("sweep")` in report.py: fixed by ruff format and a single load. The magic label offsets in chart 2 are kept: they are layout constants, checked by looking at the render.
- Chart 3 lacked an anchor-size column: fixed.
- "depends" did not name its factor: fixed. It now reads "depends (Nx the sparse delta, M% of the anchor)". No pair reaches it. "holds" is tested first, a choice ADR 0001 leaves open.
- The run-log summary, and step 6 and 7 entries, were missing: fixed, this entry and the summary at the top.
- The future-work docs-drift bullet was stale: fixed. It now points at `docs/docs-drift/`.
- The xet-core patch missed three other `HF_XET_LOG_FILE` mentions: fixed, the patch covers all four.
- The telemetry table missed three variables: fixed, all six are listed.
- The HP-mode buffer column was still constants: fixed. HP mode is derived too (usable / 8, / 32, / 2), with the old constants as ceilings and fallback.
- The chart 2 caption said scattered tensors "sit on the line": fixed. It now says large tensors, and explains why norms can fall below.
