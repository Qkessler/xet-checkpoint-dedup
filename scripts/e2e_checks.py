"""Invariant checks for the outer loop. Each subcommand reads result files and exits non-zero on a violation."""

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fail(message: str):
    print(f"INVARIANT BROKEN: {message}", file=sys.stderr)
    sys.exit(1)


def checkAttribution(result: dict, label: str):
    report = result["result"]
    nextBytes = next(item["bytes"] for item in result["inputs"] if item["role"] == "next")
    attributedBytes = sum(row["bytes"] for row in report["tensors"])
    if attributedBytes != nextBytes:
        fail(f"{label}: attributed bytes {attributedBytes} != file size {nextBytes}")
    attributedOverlap = sum(row["chunk_overlap_bytes"] for row in report["tensors"])
    if attributedOverlap != report["whole_file"]["chunk_overlap_bytes"]:
        fail(f"{label}: per-row overlap {attributedOverlap} != whole-file overlap")


def selfOverlap(path: str):
    result = load(path)
    checkAttribution(result, path)
    wholeFile = result["result"]["whole_file"]
    if wholeFile["chunk_overlap_bytes"] != wholeFile["bytes"]:
        fail(f"{path}: overlap(A, A) is {wholeFile['chunk_overlap_percent']}%, not 100%")
    if wholeFile["unchanged_element_fraction"] != 1.0:
        fail(f"{path}: overlap(A, A) unchanged element fraction is {wholeFile['unchanged_element_fraction']}")
    print(f"ok self-overlap {path}")


def main():
    command, *paths = sys.argv[1:]
    checks = {"self-overlap": selfOverlap}
    for path in paths:
        checks[command](path)


if __name__ == "__main__":
    main()
