# Outer loop: runs ckpt-dedup on real checkpoints and asserts the invariants in docs/plan.md.
# Run through `just e2e`, which builds the release binary first. Needs `just fetch` and a xet-core clone
# (XET_CORE_DIR, at commit d2f667c) for oracle 2.
# Every check runs even after one fails; the script exits non-zero if any invariant broke.
set -uo pipefail
cd "$(dirname "$0")/.."
bin=target/release/ckpt-dedup
cache="$HOME/.cache/xet-checkpoint-dedup"
inputs="$cache/inputs"
scratch="$cache/scratch"
qwen="$inputs/qwen25-05b-instruct/model.safetensors"
xet_core_dir="${XET_CORE_DIR:-/workplace/enrikes/huggingface-xet/xet-core}"
xet_chunk_example="$xet_core_dir/target/release/examples/chunk"
mkdir -p "$scratch"
failures=0
check() {
    if ! "$@"; then
        failures=$((failures + 1))
    fi
}
checks() {
    check uv run --offline python scripts/e2e_checks.py "$@"
}

echo "== overlap(A, A) on Qwen2.5-0.5B-Instruct"
check "$bin" overlap "$qwen" "$qwen" --name qwen25-self > /dev/null
checks self-overlap results/qwen25-self.json

echo "== perturbation sweep"
check "$bin" sweep "$qwen" --name sweep > /dev/null
checks sweep results/sweep.json

echo "== perturb on disk matches the in-memory sweep point, and oracle 2 matches overlap"
perturbed="$scratch/qwen25-scattered-1e-5-seed0.safetensors"
check "$bin" perturb "$qwen" --density 1e-5 --mode scattered --seed 0 --output "$perturbed" > /dev/null
check "$bin" overlap "$qwen" "$perturbed" --name qwen25-scattered-1e-5-seed0 > /dev/null
checks sweep-point results/qwen25-scattered-1e-5-seed0.json results/sweep.json scattered 1e-5 0
if [ ! -x "$xet_chunk_example" ]; then
    echo "building xet-core's chunk example in $xet_core_dir (set XET_CORE_DIR to override)"
    check bash -c "cd '$xet_core_dir' && cargo build --release -p xet-data --example chunk"
fi
oracle_overlap_bytes() {
    "$xet_chunk_example" -i "$1" | cut -d' ' -f1 | sort -u > "$scratch/base.hashes"
    "$xet_chunk_example" -i "$2" | sort -k1,1 | join - "$scratch/base.hashes" | awk '{total += $2} END {print total + 0}'
}
checks oracle-chunk-overlap results/qwen25-scattered-1e-5-seed0.json "$(oracle_overlap_bytes "$qwen" "$perturbed")"
rm -f "$perturbed" "$scratch/base.hashes"

echo "== upload(A, A) sends only a shard"
check "$bin" upload "$qwen" "$qwen" --name upload-qwen25-self > /dev/null
checks upload-self results/upload-qwen25-self.json

echo "== every checkpoint pair: overlap and upload invariants"
check bash scripts/pairs.sh > /dev/null
checks overlap results/overlap-*.json
checks upload results/upload-run-*.json results/upload-chenjakcy6-*.json results/upload-ajjyy-*.json

echo "== oracle 3: numpy unchanged element counts for run-a steps 0 and 1"
check uv run --offline python scripts/oracle_unchanged_elements.py results/overlap-run-a-s00-s01.json \
    "$cache/runs/run-a/step_00000.safetensors" "$cache/runs/run-a/step_00001.safetensors"

if [ "$failures" -gt 0 ]; then
    echo "e2e: $failures check(s) broke an invariant"
    exit 1
fi
echo "e2e: all invariants hold"
