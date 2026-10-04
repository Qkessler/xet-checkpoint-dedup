# Runs `overlap` and `upload` on every checkpoint pair and writes results/{overlap,upload}-<pair>.json.
# Pairs: consecutive steps of each of our runs, longer step gaps (2, 5, 10, 20) from fixed bases, and the
# third-party Hub pairs from data/manifest.toml.
set -euo pipefail
cd "$(dirname "$0")/.."
bin=target/release/ckpt-dedup
runs="$HOME/.cache/xet-checkpoint-dedup/runs"

measure_pair() {
    local run=$1 base_step=$2 next_step=$3
    local pair
    pair=$(printf '%s-s%02d-s%02d' "$run" "$base_step" "$next_step")
    local base next
    base=$(printf '%s/%s/step_%05d.safetensors' "$runs" "$run" "$base_step")
    next=$(printf '%s/%s/step_%05d.safetensors' "$runs" "$run" "$next_step")
    "$bin" overlap "$base" "$next" --name "overlap-$pair" > /dev/null
    "$bin" upload "$base" "$next" --name "upload-$pair" > /dev/null
    echo "measured $pair"
}

for step in $(seq 0 20); do
    measure_pair run-a "$step" $((step + 1))
done
for pair in "0 2" "0 5" "0 10" "0 20" "10 12" "10 15" "10 20" "1 21"; do
    measure_pair run-a $pair
done
for step in $(seq 0 10); do
    measure_pair run-b-frozen-half "$step" $((step + 1))
done
for pair in "0 2" "0 5" "0 10" "1 11"; do
    measure_pair run-b-frozen-half $pair
done

inputs="$HOME/.cache/xet-checkpoint-dedup/inputs"
measure_hub_pair() {
    local base_key=$1 next_key=$2 pair=$3
    "$bin" overlap "$inputs/$base_key/model.safetensors" "$inputs/$next_key/model.safetensors" --name "overlap-$pair" > /dev/null
    "$bin" upload "$inputs/$base_key/model.safetensors" "$inputs/$next_key/model.safetensors" --name "upload-$pair" > /dev/null
    echo "measured $pair"
}
measure_hub_pair chenjakcy6-s110 chenjakcy6-s113 chenjakcy6-s110-s113
measure_hub_pair ajjyy-s3150 ajjyy-s3160 ajjyy-s3150-s3160
