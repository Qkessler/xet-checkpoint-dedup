set shell := ["bash", "-euo", "pipefail", "-c"]

cache := env_var('HOME') / ".cache/xet-checkpoint-dedup"
inputs := cache / "inputs"
bin := "target/release/ckpt-dedup"

# Inner loop: format, lint, unit and property tests, README staleness. Offline, about 30 s.
check:
    cargo +nightly fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test --release
    bash scripts/check-readme.sh

# Outer loop: run the tool on real checkpoints and assert the invariants in docs/plan.md.
e2e: build
    bash scripts/e2e.sh

build:
    cargo build --release

# Download every Hub input in data/manifest.toml and verify sha256s.
fetch:
    uv run python scripts/fetch.py

# Perturbation sweep on the Qwen2.5-0.5B-Instruct checkpoint.
sweep: build
    {{bin}} sweep {{inputs}}/qwen25-05b-instruct/model.safetensors --name sweep

# Generate our own GRPO training run (CPU) saving a bf16 checkpoint after every optimizer step.
train *args:
    uv run --group train python scripts/train_grpo.py {{args}}

# Regenerate README tables and charts from results/.
report:
    uv run python scripts/report.py

# Run overlap and upload on every checkpoint pair: our runs and the third-party pairs.
pairs: build
    bash scripts/pairs.sh
