"""Oracle 3: recompute each tensor's unchanged element count with numpy and compare with an overlap result exactly.

Parses the safetensors header itself and compares raw element bytes, so it shares no code with ckpt-dedup.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

elementDtypes = {
    "BF16": np.uint16,
    "F16": np.uint16,
    "F32": np.uint32,
    "I64": np.uint64,
    "I32": np.uint32,
    "U8": np.uint8,
}


def tensorViews(path: Path) -> dict:
    data = np.memmap(path, dtype=np.uint8, mode="r")
    (headerBytes,) = struct.unpack("<Q", bytes(data[:8]))
    header = json.loads(bytes(data[8 : 8 + headerBytes]))
    header.pop("__metadata__", None)
    dataStart = 8 + headerBytes
    views = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        views[name] = data[dataStart + start : dataStart + end].view(elementDtypes[entry["dtype"]])
    return views


def main() -> int:
    overlapPath, basePath, nextPath = sys.argv[1:]
    rows = {row["name"]: row for row in json.loads(Path(overlapPath).read_text())["result"]["tensors"]}
    baseViews, nextViews = tensorViews(Path(basePath)), tensorViews(Path(nextPath))
    mismatches = 0
    for name, nextView in nextViews.items():
        expected = int(np.count_nonzero(baseViews[name] == nextView))
        if rows[name]["unchanged_elements"] != expected:
            mismatches += 1
            print(f"MISMATCH {name}: ckpt-dedup {rows[name]['unchanged_elements']} numpy {expected}", file=sys.stderr)
    print(f"oracle 3: {len(nextViews)} tensors compared, {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
