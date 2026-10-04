# Generate our own best-case checkpoints, plus third-party pairs

No public small-model RL run publishes checkpoints one step apart; the closest found is three steps apart. So we generate our own: Qwen2.5-0.5B-Instruct, GRPO on CPU, AdamW, lr 1e-6, weight decay 0, with fp32 master weights cast to bf16 and saved every step. The setup mirrors the blog's and is deliberately chosen to favour its claim (smallest step gap, smallest learning rate). The README states the exact config and that choice. Third-party evidence comes from `ChenJakcy6/Qwen2-0.5B-GRPO-test` (a pair three steps apart) and `ajjyy/Qwen2-0.5B-GRPO-attempt1` (a pair ten steps apart). `allenai/OLMo-2-0425-1B-RLVR1` is added only if time is left.

## Consequences

- Before committing to a full run, one training step is timed on this host (32 vCPU, no GPU). If a step takes more than about 2 minutes, we save fewer steps rather than switch to a smaller model.
- The synthetic perturbation sweep (scattered against clustered, at a range of change densities) runs on a real checkpoint and does not depend on the training run succeeding.
