# Measure accepted dedup as file bytes minus new bytes

The plan defines the defrag gap as `chunk overlap bytes - deduped bytes`. On the crates.io `hf-xet` 1.6.0 release, which ADR 0002 first pinned, that formula reads a wrong number. ADR 0006 later moved the tool to xet-core `d2f667c`, where the raw metric is fixed and both formulas agree. `FileDeduper::process_chunks` adds a dedup match to `deduped_bytes` and `total_bytes` before defrag prevention decides whether to take it. When the match is refused, its chunks are counted again as new. So `deduped_bytes` includes refused matches, and `total_bytes` (which the upload path also reports as the file's size) exceeds the real size by `defrag_prevented_dedup_bytes`. xet-core fixed the accounting after 1.6.0 (#931, present in the clone at `d2f667c`).

Every chunk of an uploaded file is either new or deduped, so the dedup the upload path actually took is `file bytes - new_bytes`, with the file size read from disk. That holds in every version. `upload` reports it as `accepted_dedup_bytes`, and the defrag gap is `chunk overlap bytes - accepted_dedup_bytes`. The plan's formula, applied to the raw metric, is kept as `raw_defrag_gap_bytes`. On 1.6.0 it showed the accounting difference. On `d2f667c` it equals the defrag gap, and `just e2e` checks that.

On every pair measured, the defrag gap equals `defrag_prevented_dedup_bytes` and the raw `deduped_bytes` equals the chunk overlap. That confirms the reading of the code. CONTEXT.md's definition of the defrag gap ("bytes counted in chunk overlap that the upload path still re-sends because defrag prevention refused the match") is unchanged. Only the arithmetic that measures it changes.

## Consequences

- Transferred bytes stay as the plan defines them (`new_bytes + shard_bytes_uploaded`). `new_bytes` was never affected.
- `new_bytes` counts uncompressed chunk bytes. The bytes actually written to xorbs (`xorb_bytes_uploaded`) are lower, because of LZ4 and byte-grouping compression. Both are reported, and the verdict uses `new_bytes`, as the plan says.
