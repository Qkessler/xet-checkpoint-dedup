# Does Xet dedup consecutive RL checkpoints?

<!-- BEGIN answer -->
<!-- END answer -->

The sparse encoding in the delta-weight-sync blog is what makes per-step weight sync cheap. Xet's chunk dedup helps when changes cluster (frozen layers, untouched embedding rows), and this repo measures how far each case gets.

## The claim

From [Shipping a Trillion Parameters With a Hub Bucket: Delta Weight Sync in TRL](https://huggingface.co/blog/delta-weight-sync) (2026-05-27), section 3.1:

> The practical upshot, which is delightful in this context, is that even if we were too lazy to write the sparse encoding and just uploaded full anchors every step, Xet would _still_ only transfer the changed chunks.

Read literally, this is true by construction: Xet sends the chunks that differ. This repo tests the reading a practitioner would act on, that uploading full checkpoints every step costs about as little as the blog's sparse deltas.

## Verdict rule

Fixed before measuring ([ADR 0001](docs/adr/0001-test-the-implied-claim-with-preset-thresholds.md)). For each checkpoint pair, compare the bytes Xet's upload path transfers for the next checkpoint with the sparse delta for the same pair (changed elements x 6 bytes: a 4-byte index plus a 2-byte bf16 value):

- transferred bytes within 3x of the sparse delta: **holds**;
- transferred bytes at least 50% of the anchor (the full checkpoint): **does not hold**;
- anything in between: **depends**, with the factor named.

## Results

### Per checkpoint pair (chart 3)

<!-- BEGIN pairs -->
<!-- END pairs -->

"Chunk overlap" is the share of the next file's bytes in chunks whose hash also appears in the base. "Transferred" is `new_bytes + shard_bytes_uploaded` from Xet's own `DeduplicationMetrics` after uploading the base and then the next checkpoint to the same local store. The "defrag gap" is chunk overlap that the upload path re-sent anyway, because defrag prevention refused short matches.

<details>
<summary>Every pair measured</summary>

<!-- BEGIN all-pairs -->
<!-- END all-pairs -->

</details>

### Where chunk dedup works: change density (chart 1)

![Chunk overlap against change density for scattered, row-clustered and tensor-clustered perturbations](docs/charts/chunk-overlap-vs-density.png)

Synthetic perturbations of `Qwen/Qwen2.5-0.5B-Instruct`: a fraction P of bf16 elements moved by one ulp, either scattered uniformly, or clustered into whole rows or whole tensors. Error bars are 3 standard errors over seeds.

<!-- BEGIN sweep -->
<!-- END sweep -->

### Which tensors keep their chunks (chart 2)

![Per-tensor and per-kind unchanged elements against chunk overlap, for four pairs](docs/charts/unchanged-elements-vs-chunk-overlap.png)

Each grey dot is one tensor and each blue dot one tensor kind. Tensors whose changes are scattered sit on the dotted theory line. Points far above it have clustered changes: frozen layers at 100%, and embedding rows of tokens that never appear in training.

### What the Hub stored for the third-party pairs

<!-- BEGIN hub -->
<!-- END hub -->

The Hub columns come from the CAS reconstruction API (`scripts/hub_reconstruction.py`): a chunk is shared when the next file's recipe points at the same xorb chunk as the base's. The ChenJakcy6 row reproduces the earlier research pass exactly, and the Hub's chunking of that file matches xet-core's (same Xet file hash). The ajjyy file was chunked by something else on the Hub (its Xet file hash differs from ours), and its recipe shows no dedup at all, so its Hub row is not comparable to the local measurement.

## Surprises

1. **The pre-registered theory curve misses the scattered sweep.** The plan fixed `exp(-values_per_chunk x P)`, with `values_per_chunk` the mean chunk size over 2 bytes, and a tolerance of 3 percentage points or 3 standard errors. At mid-range densities the scattered points fall outside it (see the sweep table). The cause is chunk-size variation: Xet chunks range from 8 KiB to 128 KiB, a chunk of `s` bytes survives with probability `(1-P)^(s/2)`, and the byte-weighted average of that is not its value at the mean size. Large chunks are both likelier to be hit and carry more bytes. The size-weighted curve matches every cell. The check is left failing as pre-registered, and `just e2e` reports it. Whether to replace the curve is the owner's call. The conclusion is the same under either curve, because both reach zero long before the per-step change rates measured here.
2. **`hf-xet` 1.6.0 over-reports dedup.** In the published 1.6.0 crates, a dedup match that defrag prevention refuses is counted in `deduped_bytes` and then again as new data. So `deduped_bytes` overstates the dedup taken, and the reported file size exceeds the real one by exactly `defrag_prevented_dedup_bytes`. xet-core fixed this after 1.6.0 (#931). This repo measures accepted dedup as `file bytes - new_bytes` ([ADR 0005](docs/adr/0005-measure-accepted-dedup-as-file-bytes-minus-new-bytes.md)).
3. **A header that changes length costs more than one chunk.** When no content-defined boundary appears within 128 KiB, the chunker cuts at exactly 128 KiB. A byte shift propagates through runs of such forced cuts, and bf16 checkpoints have many of them (`next_max_size_chunks` in `results/overlap-*.json`). Our pairs keep the same header length, so this does not affect them.

## Method

- **One Rust binary** (`ckpt-dedup`) on the crates.io `xet-data`, `xet-core-structures` and `hf-xet` 1.6.0 crates ([ADR 0002](docs/adr/0002-rust-on-the-xet-crates-for-every-measurement.md)).
  - `overlap` chunks both files with `xet_data::deduplication::Chunker`, the chunker uploads use, and marks each chunk of the next file as reused or new. It credits each chunk's bytes to the header or the tensors its byte range covers, and compares the files element by element.
  - `upload` commits the base and then the next file to a fresh `local://` store through `XetSession` and reads `DeduplicationMetrics` ([ADR 0003](docs/adr/0003-local-store-only.md)).
  - `perturb` and `sweep` write and measure synthetic perturbations.
- **Checkpoints.**
  - Our own: Qwen2.5-0.5B-Instruct trained with TRL GRPO on CPU, AdamW, lr 1e-6 constant, weight decay 0, beta 0, fp32 weights, and a bf16 checkpoint after every optimizer step ([ADR 0004](docs/adr/0004-self-generated-best-case-checkpoints.md)). This is the blog's setup at its most favourable: the smallest learning rate and a step gap of 1. A second run freezes layers 0 to 11. Step 0 is byte-identical to the published checkpoint. The full config and library versions are in `data/manifest.toml`.
  - Third party: two public GRPO repos on the Hub, at the revisions in `data/manifest.toml`.
- **Checks that do not trust the tool.**
  - Our chunker reproduces the Xet spec's reference `.chunks` file byte for byte.
  - xet-core's own `chunk` example, followed by `sort` and `join`, gives exactly our whole-file chunk overlap.
  - numpy, with its own header parser, gives exactly our per-tensor unchanged element counts.
  - Property tests cover attribution, self-overlap, header rewrites and single-element changes.

## Limitations

- **A local store, not the Hub.** Within one pair, global dedup adds nothing, because the base's shard is already local. A real Hub upload also adds network and server behaviour that we do not measure.
- **Uncompressed transfer bytes.** `new_bytes` counts chunk bytes before compression. Xet compresses chunks (LZ4, byte grouping), so the bytes written to xorbs are lower (`xorb_bytes_uploaded` in `results/upload-*.json`). Sparse deltas compress too, so the comparison stays like for like.
- **Two small models and one learning rate.** Larger models, higher learning rates and longer step gaps change more elements per step, which makes chunk dedup worse, not better.
- **Our own runs are tiny.** One prompt and four completions per step. That changes which elements move, but not the density regime the blog and PULSE report.
- **Generated checkpoints reproduce on this host.** CPU kernels are deterministic for a fixed thread count. Another host may produce different bytes, so `just fetch` verifies them but cannot download them.

## Future work

- **A real Hub upload of one pair to a private bucket**, to exercise the server, global dedup and shard registration. Skipped for offline reproducibility and the time budget ([ADR 0003](docs/adr/0003-local-store-only.md)).
- **A Hub reconstruction-API cross-check built into the tool**, to show what the Hub actually shares between public checkpoints without downloading them. For now `scripts/hub_reconstruction.py` does it for two pairs.
- **fp32 master-weight checkpoints**, and a higher learning rate (1e-5).
- **A one-sentence correction to the blog**, for example: "Xet dedups unchanged chunks, but at RL learning rates bf16 changes land in almost every chunk, so the sparse encoding is what makes this cheap." Discuss it with the blog authors first, rather than sending it cold.
- **The docs-drift fixes** for the hub-docs Xet environment table and the xet-core README (`docs/background/xet-stack.md` section 3.4).

## Reproduce

```bash
just fetch   # download and verify every Hub input in data/manifest.toml
just train --run run-a --steps 21 --seed 0
just train --run run-b-frozen-half --steps 11 --seed 0 --freeze-first-layers 12
just check   # inner loop: fmt, clippy, unit and property tests, golden chunker test, README staleness
just e2e     # outer loop: sweep, every pair, oracles and invariants (needs a xet-core clone, XET_CORE_DIR)
just report  # redraw charts and regenerate the tables in this README from results/
```

<!-- BEGIN provenance -->
<!-- END provenance -->

Plan: [docs/plan.md](docs/plan.md). Lab notebook: [docs/run-log.md](docs/run-log.md). Glossary: [CONTEXT.md](CONTEXT.md).
