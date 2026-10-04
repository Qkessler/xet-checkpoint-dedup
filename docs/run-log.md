# Run log

Session link: not generated. Uploading a session export writes to the network, which is a hard stop in AGENTS.md; the owner can add one.

## Summary

(Written at the end of the run.)

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
