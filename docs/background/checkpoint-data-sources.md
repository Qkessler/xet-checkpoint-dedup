# Checkpoint data sources for testing "Xet would still only transfer the changed chunks"

Researched 2026-10-04. Sources: blog markdown at https://raw.githubusercontent.com/huggingface/blog/main/delta-weight-sync.md, TRL PRs via GitHub API, PULSE paper (arXiv 2602.03839), HF Hub public API, Xet CAS reconstruction API (anonymous read token).

## TL;DR

- The blog's Xet claim is an aside, not measured. All its numbers come from the sparse-safetensors path (and from the PULSE paper).
- Back-of-envelope: a ~64 KiB Xet chunk holds ~32k bf16 values. If ~1% of elements change and the changes are spread evenly, the chance that a chunk has no change is `0.99^32768 ≈ e^-329 ≈ 0`. So nearly every chunk changes. For half the chunks to stay identical, the changed fraction per step would need to be at most ~2e-5 (99.998% sparsity). That only fails to hold if the changes are clustered (e.g. frozen layers, or embedding rows that get no gradient).
- I measured this on real public data, without downloading any weights, using the Xet reconstruction API. Consecutive bf16 GRPO checkpoints on the Hub share **0.00–0.48%** of their Xet chunks (details in section 4). This is HF's own storage reporting how much was actually deduplicated at upload time.
- This host has no GPU (`nvidia-smi: command not found`), 32 vCPU (AMD EPYC 9R14, avx512_bf16), 61 GiB RAM and 1.8 TB free disk. A CPU GRPO run on Qwen2.5-0.5B-Instruct that saves every step is feasible (section 5).

## 1. Blog setup (https://huggingface.co/blog/delta-weight-sync)

Title: "Shipping a Trillion Parameters With a Hub Bucket: Delta Weight Sync in TRL". Authors: aminediroHF, qgallouedec, kashif, lewtun, edbeeching, albertvillanova, lvwerra, sergiopaniego.

### The exact Xet claim (section 3.1, "What is a Bucket?")

> Under the hood, buckets are backed by **Xet**, the Hub's content-defined chunking storage layer. Xet looks at every file you upload, slices it into chunks based on its actual content (not fixed offsets), and deduplicates against everything already in the bucket. The practical upshot, which is delightful in this context, is that even if we were too lazy to write the sparse encoding and just uploaded full anchors every step, Xet would _still_ only transfer the changed chunks. Sparse encoding + Xet stack: we pay for what moved, and we pay for it once.

A related claim in section 6:

> They all pull from the same bucket. Xet content-addresses storage so consecutive anchors share chunks at rest (which keeps your bucket from blowing up)

Neither claim comes with a measurement. Anchors are written every N=10 syncs, so consecutive anchors are 10 steps apart, which makes sharing even less likely than between single steps.

### Sparsity figure and where it comes from

The TL;DR says:

> Between two consecutive RL optimizer steps, **roughly 99% of bf16 weights are bit-identical** (and never less than 98% in the worst case).

> On Qwen3-0.6B, the per-step payload drops from 1.2 GB to **20 to 35 MB**.

Section 2 credits the PULSE paper for this number:

> They measure this empirically across Qwen2.5 (0.5B/1.5B/7B), Llama-3.2-3B, and Gemma-3-4B, and consistently find a mean per-step sparsity of **~99%, with a standard deviation of 0.2 to 0.4% over 400 training steps**. The worst-case step stays above 98%.

- **Per-element.** The figure counts individual bf16 elements whose bit pattern is unchanged. The sample log line is `Delta: 1234567/200000000 elements changed (sparsity=99.38%)`, and delta metadata carries `sparsity=0.9938`. The blog has no per-layer breakdown.
- **How it is measured.** `BF16ChangeDetector` uses optimizer pre-step and post-step hooks. It takes a CPU snapshot of `p.to(bfloat16)` before the step and compares with `!=` after it, giving a boolean mask. Elements are matched via `data_ptr()`. They tried predicting the mask from Adam's m and v; recall was about 30%, so they dropped it.
- **Argument.** The blog quotes the PULSE argument: with Adam, |Δw| ≈ η ≈ 3e-6, while the bf16 visibility threshold is |w|/256 ≈ 4e-5 to 4e-4 for |w| between 1e-2 and 1e-1. Updates below the threshold are absorbed. The trainer keeps fp32 master weights and the bf16 cast is what changes.
- **Optimizer, learning rate, dtype.** AdamW. The example script `examples/scripts/openenv/async_wordle.py` (PR head `d6504b7e`) defaults to `learning_rate=1e-6`, `bf16=True`, `num_generations=16`, `per_device_train_batch_size=32`, `max_steps=100`, `max_completion_length=1024` and model `Qwen/Qwen3-1.7B`. `AsyncGRPOConfig` also defaults lr to 1e-6. The payload numbers quoted are for Qwen3-0.6B.
- **Algorithm.** Async GRPO (`trl.experimental.async_grpo`) on Wordle (OpenEnv / TextArena).
- **Sync protocol.** Hub Bucket with `anchors/step_XXXXXX.safetensors` (full bf16, every N=10 syncs, metadata `sparse=False`) and `deltas/step_XXXXXX.safetensors`. A delta holds, per changed parameter, `<name>.indices` (int32 flat) and `<name>.values` (bf16), with metadata `sparse=True, model_version, sparsity, changed_params`. Uploads use `batch_bucket_files`, downloads use `download_bucket_files`. Control is a POST of `{repo_id, filename}` to the vLLM `DeltaWorkerExtension` (`--worker-extension-cls`). The sample log reports a 27.4 MB delta, 9.4 s total and 1.1 s inference pause.

### Code and data releases

- The blog says "We landed [a TRL PR](https://github.com/huggingface/trl/pull/5417)". **That is inaccurate.** #5417 ("Delta weight sync using Xet buckets", branch `delta-weight-sync`) was **closed unmerged** on 2026-09-14 as superseded by #5937.
- https://github.com/huggingface/trl/pull/5937 ("Delta weight sync for AsyncGRPO (sparse patches over HF Bucket)", branch `delta-weight-sync-v3`) was **also closed unmerged** on 2026-09-14 ("Closing this one for now to keep the queue tidy. The branch is conflicting"). It uses `LowByteChangeDetector` and later `AdamWInversionChangeDetector`, plus vLLM-native `SparseWeightPatch` from https://github.com/vllm-project/vllm/pull/40096. Its description states: "At steady state, a delta is ~1–10% of the model, and sparsity rises as the LR decays". That is weaker than the blog's 99%.
- #5932 (`delta-weight-sync-v2`, a WIP) was also closed.
- The code is still readable on those branches. Files: `trl/experimental/async_grpo/{delta_engine.py,weight_diff.py}` and `examples/scripts/openenv/async_wordle.py`.
- I found no public buckets or checkpoints from their runs. The blog says "The full run logs are linked in the companion PR".

### PULSE paper (https://huggingface.co/papers/2602.03839, Miahi & Belilovsky)

- GRPO with DAPO-style config on MATH (plus MBPP for code).
- Controlled-sparsity runs: lr ~3e-6, Adam betas (0.9, 0.999), **weight decay = 0 and KL β = 0** during the primary measurements, 4 seeds.
- Deployment and PULSELoCo runs: lr 1e-6, betas (0.9, 0.95).
- Gradients are about 99% dense. Sparsity is a property of the bf16 cast of the fp32 master weights.
- With staleness S=32 and k=1, sparsity stays above 98.5%. Over multi-step windows (k>1) it stays above 97.5%.
- Learning rates above ~5e-6 destabilize training.

## 2. Public consecutive checkpoint series

I listed revisions via `GET https://huggingface.co/api/models/<id>/refs` (branches) or `.../commits/main?limit=N` (each `Training in progress, step N` commit from `push_to_hub` with `hub_strategy="every_save"` is a downloadable git revision). Files are listed via `.../tree/<rev>`. To download one revision: `hf download <id> model.safetensors --revision <sha-or-branch>`, or `huggingface_hub.hf_hub_download(repo_id, "model.safetensors", revision=...)`.

| Rank | Repo | Params | dtype on disk | Spacing | Size / file | Type | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `ajjyy/Qwen2-0.5B-GRPO-attempt1` | 0.49B (Qwen2-0.5B-Instruct base) | bf16 (`model.safetensors`) | **10 steps**, 200+ commits (…3150, 3160, 3170) | 988,097,824 B | RL (TRL GRPO) | Pushed 2025-08-11, about 1 min per commit. lr unknown (likely the TRL default 1e-6). Revs: `868f0e70…` = s3150, `8ac61c6b…` = s3160, `310aea15…` = s3170 |
| 2 | `ChenJakcy6/Qwen2-0.5B-GRPO-test` | 0.49B | bf16 | 10 steps, plus a **3-step** pair (110 → 113) | 988 MB (+2 MB adapter) | RL (GRPO) | Revs `9604b769e2db` s90, `c0a783d19046` s100, `5012bf807a40` s110, `1cd8897a89b0` s113 |
| 3 | `shawntzx/Qwen2.5-0.5B-GRPO-2_26_17k` | 0.49B | bf16 | 50 steps (plus a 3-step pair 950 → 953) | 988 MB | RL (GRPO) | Revs `cff55f692b02` s900, `95faef4acce9` s950, `e8b8e57406cf` s953 |
| 4 | `allenai/OLMo-2-0425-1B-RLVR1` | 1.48B | bf16 | branches `step_200`…`step_2600` (**200 steps** apart; the README says weights were saved every 20 steps but only every 200th is published) | 2,969,854,224 B | RL (RLVR, official AI2) | Most credible provenance. Branches are named, so it is easy to script |
| 5 | `4everStudent/Qwen3-0.6B-GRPO-prop-prediction-linear-reward` | 0.6B | bf16 | 50 steps (≈39 commits) | 1.19 GB | RL (GRPO) | Same family as the blog's Qwen3-0.6B |
| 6 | `4everStudent/Qwen2-0.5B-GRPO-test-5epochs` | 0.49B | bf16 | 250 steps, 200+ commits | 988 MB | RL | |
| ctrl | `EleutherAI/pythia-70m` / `pythia-160m` | 70M / 160M | `model.safetensors` is fp32 (281 MB for 70m); `pytorch_model.bin` fp16 | `step0,1,2,4,…,512` log-spaced, then every 1000 | 281 MB (70m) | Pretraining (high LR) | Negative control only. Revisions `stepN` |
| ctrl | `allenai/OLMo-2-0425-1B` | 1.48B | bf16 | `stage1-step*` every 10k; `stage2-ingredient*-step*` every 1000 | ~3 GB | Pretraining / anneal | |
| ctrl | `HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints` | 135M | — | 240k steps | — | Pretraining | Too sparse to be useful |
| ctrl | `HuggingFaceTB/SmolLM3-3B-checkpoints` | 3B | — | 40k (stages); `lc-*` every 4k; post-train `it-*` | — | Pretraining / mid-training | |

I scanned about 1,200 small RL-tagged repos (GRPO, PPO, RLOO, DAPO, RLVR in the search). I found **no public series with true 1-step spacing**: the 1-step and 2-step gaps I saw were restarts. The smallest real spacings are 3, 5 and 10 steps.

## 3. Why spacing hardly matters for the Xet question

Assume p is the fraction of elements that change and C ≈ 32,768 bf16 values per chunk. Then P(chunk unchanged) = (1-p)^C.

| p | P(unchanged chunk) |
|---|---|
| 1e-2 | ~0 |
| 1e-3 | ~6e-15 |
| 1e-4 | 0.038 |
| 2e-5 | 0.52 |

PULSE and the blog report p ≈ 0.6–2% per step. Even a single step is therefore four orders of magnitude beyond the point where full-snapshot dedup could help, unless changes cluster. Possible exceptions are untied embedding rows for tokens that never appear (zero gradient means Adam leaves them alone, provided weight decay is 0), frozen modules, and LoRA.

Qwen2-0.5B, Qwen2.5-0.5B and Qwen3-0.6B all have **tied** embeddings, so the softmax gradient reaches every vocabulary row.

## 4. Direct measurement: Xet chunk sharing between consecutive public checkpoints

Method (read-only, no weight download):

1. `GET /api/models/<repo>/tree/<rev>` returns `model.safetensors.xetHash`.
2. `GET /api/models/<repo>/xet-read-token/<rev>` returns an anonymous token (headers `x-xet-access-token` and `x-xet-cas-url`).
3. `GET https://cas-server.xethub.hf.co/v1/reconstructions/<xetHash>` with `Authorization: Bearer <token>` returns the file's `terms`, each a (xorb hash, chunk range).
4. Count the (xorb, chunk-index) pairs the newer file shares with the previous one. Any chunk dedupe at upload time shows up as a reference to an older xorb.

| Pair | Gap | Chunks in newer file | Shared chunks | Shared bytes |
|---|---|---|---|---|
| ajjyy s3150 → s3160 | 10 | 15,859 | 0 | 0.00% |
| ajjyy s3160 → s3170 | 10 | 15,884 | 0 | 0.00% |
| ChenJakcy6 s90 → s100 | 10 | 15,345 | 11 | 0.05% |
| ChenJakcy6 s100 → s110 | 10 | 15,320 | 36 | 0.09% |
| ChenJakcy6 s110 → s113 | **3** | 15,336 | 202 | **0.48%** |
| shawntzx s900 → s950 | 50 | 15,963 | 0 | 0.00% |
| shawntzx s950 → s953 | 3 | 15,960 | 0 | 0.00% |
| OLMo-2-1B-RLVR1 step_200 → step_400 | 200 | 48,307 | 0 | 0.00% |

The average chunk is ~62 KiB (988 MB / ~15.9k chunks), which matches the 64 KiB target. The non-zero rows show that dedup does fire when chunks match, but on these real GRPO runs at most ~0.5% of chunks repeat. The helper script was `~/.tmp/xetcmp.py` (it lives in the scratch area and is not kept).

Caveats:

- Hyperparameters are unknown for the community repos. The learning rate may be higher than the blog's 1e-6, and steps may be 10 apart rather than 1.
- Dedup only happens if the uploader's client found the earlier chunks (hf_xet global dedup and shard lookup). The ChenJakcy6 rows show it works in at least that pipeline.
- To settle the 1-step, lr=1e-6 case, generate your own data (section 5) and run the same comparison either locally with the xet-core chunker or by uploading to a private bucket and querying reconstructions.

## 5. Generating our own consecutive bf16 checkpoints

Host facts:

- `nvidia-smi`: not found, so no GPU.
- `nproc`: 32 (AMD EPYC 9R14, avx512_bf16).
- `free -g`: 61 GiB total, 55 GiB available, no swap.
- Disk: 1.8 TB free.
- System Python is 3.9 with no torch, transformers or trl installed. Use `uv` (at `~/.local/bin/uv`) to create a Python 3.12 venv.

Setup:

```bash
uv venv -p 3.12 ~/venvs/grpo && source ~/venvs/grpo/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install "trl>=0.20" transformers datasets accelerate safetensors hf_xet
```

Minimal CPU GRPO script. It keeps fp32 master weights (as PULSE and the blog assume) and dumps a bf16 cast after every optimizer step, which is exactly what the inference engine would receive:

```python
import os
import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

modelId = "Qwen/Qwen2.5-0.5B-Instruct"
outputDir = os.path.expanduser("~/grpo-ckpts")


def lengthReward(completions, **kwargs):
    return [-abs(len(completion[0]["content"]) - 40) / 40 for completion in completions]


class Bf16SnapshotCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        stateDict = {name: tensor.detach().to(torch.bfloat16).contiguous() for name, tensor in model.state_dict().items()}
        save_file(stateDict, f"{outputDir}/step_{state.global_step:05d}.safetensors")


prompts = Dataset.from_dict({"prompt": [[{"role": "user", "content": f"Describe the number {index} briefly."}] for index in range(512)]})
config = GRPOConfig(
    output_dir=outputDir,
    use_cpu=True,
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    weight_decay=0.0,
    beta=0.0,
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_completion_length=48,
    max_steps=30,
    save_strategy="no",
    logging_steps=1,
    report_to="none",
)
os.makedirs(outputDir, exist_ok=True)
trainer = GRPOTrainer(model=modelId, reward_funcs=lengthReward, args=config, train_dataset=prompts, callbacks=[Bf16SnapshotCallback()])
Bf16SnapshotCallback().on_step_end(config, trainer.state, None, model=trainer.model)
trainer.train()
```

Notes:

- With `num_generations=4` and `per_device_train_batch_size=4`, each step uses one prompt and 4 completions. Check that `generation_batch_size` divides correctly for your TRL version; recent TRL needs `per_device_train_batch_size * grad_accum` to be a multiple of `num_generations`.
- Expect very roughly 20–60 s per step on 32 cores for 0.5B with 48-token completions. AdamW fp32 state for 0.5B is about 8 GB RAM. Each snapshot is about 0.99 GB, so 30 steps need ~30 GB.
- Each `step_*.safetensors` uses the same tensor order and header, so byte offsets line up. Measure:
  - element sparsity: `(a != b).float().mean()` per tensor;
  - per-64 KiB-block "any change" fraction, as a fixed-size proxy;
  - true Xet chunk reuse, either by running the xet-core chunker (`deduplication` crate / `hf_xet`) on both files and intersecting chunk hashes, or by uploading two snapshots to a private bucket and using the reconstruction-API diff from section 4.
- Sweep lr (1e-6, 3e-6, 1e-5) and include a no-op control step at lr=0. Optionally untie embeddings or use Qwen3-0.6B to match the blog.
- A cheaper fallback is a plain PyTorch loop: AdamW at lr 1e-6 on any next-token loss, casting to bf16 after each step. It reproduces the bf16-absorption statistics without needing RL generation, though RL gradients differ in structure.
