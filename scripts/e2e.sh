# Outer loop: runs ckpt-dedup on real checkpoints and asserts the invariants in docs/plan.md.
# Run through `just e2e`, which builds the release binary first. Needs `just fetch`.
set -euo pipefail
cd "$(dirname "$0")/.."
bin=target/release/ckpt-dedup
inputs="$HOME/.cache/xet-checkpoint-dedup/inputs"
qwen="$inputs/qwen25-05b-instruct/model.safetensors"

echo "== overlap(A, A) on Qwen2.5-0.5B-Instruct"
"$bin" overlap "$qwen" "$qwen" --name qwen25-self > /dev/null
uv run --offline python scripts/e2e_checks.py self-overlap results/qwen25-self.json

echo "e2e: all invariants hold"
