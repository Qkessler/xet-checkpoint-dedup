# Fails when README.md differs from what scripts/report.py would generate from results/.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f README.md ] || [ ! -f scripts/report.py ]; then
    echo "readme check: no README.md or report.py yet, skipping"
    exit 0
fi
uv run --offline python scripts/report.py --check
