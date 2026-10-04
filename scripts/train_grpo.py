"""Generate a GRPO training run on CPU with a bf16 checkpoint after every optimizer step.

Setup from docs/background/checkpoint-data-sources.md section 5: Qwen2.5-0.5B-Instruct, TRL GRPO, AdamW, lr 1e-6,
constant schedule, weight decay 0, beta 0, fp32 master weights. Each snapshot is the bf16 cast of the fp32 weights,
which is what an inference engine would receive. Qwen2.5-0.5B ties `lm_head` to `embed_tokens`, so the snapshot
drops `lm_head.weight`, the same way the published checkpoint does.

Writes ~/.cache/xet-checkpoint-dedup/runs/<run>/step_NNNNN.safetensors and run.json, and records the run's config,
seed, library versions and per-step sha256s in data/manifest.toml between BEGIN/END markers.
"""

import argparse
import hashlib
import json
import os
import re
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

modelId = "Qwen/Qwen2.5-0.5B-Instruct"
modelRevision = "7ae557604adf67be50417f59c2c2f167def9a775"
cacheRoot = Path.home() / ".cache" / "xet-checkpoint-dedup"
manifestPath = Path(__file__).resolve().parent.parent / "data" / "manifest.toml"
libraries = ["torch", "transformers", "trl", "datasets", "accelerate", "safetensors"]


def lengthReward(completions, **kwargs):
    return [-abs(len(completion[0]["content"]) - 40) / 40 for completion in completions]


def fileSha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Bf16SnapshotCallback(TrainerCallback):
    def __init__(self, runDir: Path):
        self.runDir = runDir
        self.stepSeconds = []
        self.lastStepEnd = None

    def snapshot(self, model, step: int):
        stateDict = {name: tensor.detach().to(torch.bfloat16).contiguous() for name, tensor in model.state_dict().items()}
        if model.config.tie_word_embeddings:
            stateDict.pop("lm_head.weight", None)
        save_file(stateDict, str(self.runDir / f"step_{step:05d}.safetensors"), metadata={"format": "pt"})

    def on_step_begin(self, args, state, control, **kwargs):
        if self.lastStepEnd is None:
            self.lastStepEnd = time.monotonic()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        now = time.monotonic()
        self.stepSeconds.append(now - self.lastStepEnd)
        print(f"step {state.global_step} took {self.stepSeconds[-1]:.1f} s", flush=True)
        self.snapshot(model, state.global_step)
        self.lastStepEnd = time.monotonic()


def recordInManifest(runName: str, runRecord: dict, stepFiles: list[Path]):
    lines = [f"# BEGIN run {runName} (written by scripts/train_grpo.py)", f"[runs.{runName}]"]
    tables = {key: value for key, value in runRecord.items() if isinstance(value, dict)}
    for key, value in runRecord.items():
        if key not in tables:
            lines.append(f"{key} = {json.dumps(value)}")
    for tableName, table in tables.items():
        lines.append(f"\n[runs.{runName}.{tableName}]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in table.items()]
    for stepFile in stepFiles:
        lines += [
            "",
            f"[inputs.{runName}-{stepFile.stem.replace('_', '-')}]",
            'kind = "generated"',
            f'run = "{runName}"',
            f'file = "{stepFile.name}"',
            f'path = "runs/{runName}/{stepFile.name}"',
            f'sha256 = "{fileSha256(stepFile)}"',
        ]
    lines.append(f"# END run {runName}")
    block = "\n".join(lines) + "\n"
    manifest = manifestPath.read_text()
    pattern = re.compile(rf"\n# BEGIN run {re.escape(runName)} .*?# END run {re.escape(runName)}\n", re.S)
    manifest = pattern.sub("", manifest)
    manifestPath.write_text(manifest.rstrip("\n") + "\n\n" + block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name, e.g. run-a")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-first-layers", type=int, default=0, help="freeze model.layers.0..N-1")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count())
    set_seed(args.seed)
    runDir = cacheRoot / "runs" / args.run
    runDir.mkdir(parents=True, exist_ok=True)
    for stale in runDir.glob("step_*.safetensors"):
        stale.unlink()

    tokenizer = AutoTokenizer.from_pretrained(modelId, revision=modelRevision)
    model = AutoModelForCausalLM.from_pretrained(modelId, revision=modelRevision, dtype=torch.float32)
    for name, parameter in model.named_parameters():
        match = re.search(r"layers\.(\d+)\.", name)
        if match and int(match.group(1)) < args.freeze_first_layers:
            parameter.requires_grad_(False)

    prompts = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": f"Describe the number {index} briefly."}] for index in range(512)]}
    )
    config = GRPOConfig(
        output_dir=str(runDir / "trainer"),
        use_cpu=True,
        bf16=False,
        seed=args.seed,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        beta=0.0,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_completion_length=48,
        max_steps=args.steps,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
    )
    snapshots = Bf16SnapshotCallback(runDir)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=lengthReward,
        args=config,
        train_dataset=prompts,
        callbacks=[snapshots],
    )
    snapshots.snapshot(trainer.model, 0)
    trainer.train()

    runRecord = {
        "model_id": modelId,
        "model_revision": modelRevision,
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "freeze_first_layers": args.freeze_first_layers,
        "optimizer": config.optim if isinstance(config.optim, str) else config.optim.value,
        "lr_scheduler_type": "constant",
        "weight_decay": 0.0,
        "beta": 0.0,
        "num_generations": 4,
        "per_device_train_batch_size": 4,
        "max_completion_length": 48,
        "master_weights": "fp32",
        "checkpoint_dtype": "bf16",
        "library_versions": {library: version(library) for library in libraries},
        "step_seconds": [round(seconds, 1) for seconds in snapshots.stepSeconds],
    }
    (runDir / "run.json").write_text(json.dumps(runRecord, indent=2) + "\n")
    recordInManifest(args.run, runRecord, sorted(runDir.glob("step_*.safetensors")))


if __name__ == "__main__":
    main()
