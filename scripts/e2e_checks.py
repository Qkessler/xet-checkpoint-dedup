"""Invariant checks for the outer loop. Each subcommand reads result files and exits non-zero on a violation."""

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


failures = []


def fail(message: str):
    print(f"INVARIANT BROKEN: {message}", file=sys.stderr)
    failures.append(message)


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


def sweep(path: str):
    """Scattered cells within max(3 pp, 3 standard errors) of exp(-values_per_chunk * P); clustered above scattered.

    The tolerance and the curve are fixed in docs/plan.md and are never tuned. The size-aware curve is reported
    alongside for diagnosis only.
    """
    report = load(path)["result"]
    cells = {(cell["mode"], cell["density"]): cell for cell in report["cells"]}
    for (mode, density), cell in sorted(cells.items()):
        if mode != "scattered":
            continue
        tolerance = max(0.03, 3 * cell["standard_error"])
        mean = cell["mean_chunk_overlap_fraction"]
        plainGap = mean - cell["theory_chunk_overlap_fraction"]
        sizeAwareGap = mean - cell["size_aware_theory_chunk_overlap_fraction"]
        verdict = "ok" if abs(plainGap) <= tolerance else "OUTSIDE"
        print(
            f"  scattered {density:7.0e} mean {mean:.4f} theory gap {plainGap:+.4f} "
            f"size-aware gap {sizeAwareGap:+.4f} tolerance {tolerance:.4f} {verdict}"
        )
        if abs(plainGap) > tolerance:
            fail(
                f"{path}: scattered density {density:g} is {plainGap:+.4f} from exp(-values_per_chunk * P), "
                f"tolerance {tolerance:.4f}"
            )
        for clusteredMode in ("rows", "tensors"):
            clustered = cells[(clusteredMode, density)]["mean_chunk_overlap_fraction"]
            if not clustered > mean:
                fail(
                    f"{path}: {clusteredMode} overlap {clustered:.4f} is not above scattered {mean:.4f} "
                    f"at density {density:g}"
                )
    print(f"done sweep {path}")


def sweepPoint(overlapPath: str, sweepPath: str, mode: str, density: str, seed: str):
    """The on-disk perturb + overlap path gives exactly the in-memory sweep's chunk overlap for the same point."""
    overlap = load(overlapPath)
    checkAttribution(overlap, overlapPath)
    onDisk = overlap["result"]["whole_file"]["chunk_overlap_bytes"]
    matching = [
        point
        for point in load(sweepPath)["result"]["points"]
        if point["mode"] == mode and abs(point["density"] - float(density)) < 1e-15 and point["seed"] == int(seed)
    ]
    if len(matching) != 1:
        fail(f"{sweepPath}: expected one point for {mode} {density} seed {seed}, found {len(matching)}")
        return
    if matching[0]["chunk_overlap_bytes"] != onDisk:
        fail(f"{overlapPath}: on-disk overlap {onDisk} != sweep point {matching[0]['chunk_overlap_bytes']}")
    print(f"ok sweep-point {overlapPath} matches {sweepPath}")


def oracleChunkOverlap(overlapPath: str, oracleBytes: str):
    """Oracle 2: whole-file chunk overlap from xet-core's chunk example plus sort and join equals ours exactly."""
    ours = load(overlapPath)["result"]["whole_file"]["chunk_overlap_bytes"]
    if ours != int(oracleBytes):
        fail(f"{overlapPath}: overlap {ours} != xet-core chunk example + join {oracleBytes}")
    print(f"ok oracle-chunk-overlap {overlapPath} ({ours} bytes)")


def uploadPair(path: str):
    """Transferred data bytes are at least next size minus chunk overlap, the defrag gap is never negative, and the
    dedup accounting is the fixed one (xet-core #931, ADR 0006): deduped_bytes and the reported file size agree with
    what was actually sent."""
    report = load(path)["result"]
    nextBytes = report["next"]["file_bytes"]
    newBytes = report["next"]["dedup_metrics"]["new_bytes"]
    if newBytes < nextBytes - report["chunk_overlap_bytes"]:
        fail(f"{path}: new_bytes {newBytes} < next size - chunk overlap {nextBytes - report['chunk_overlap_bytes']}")
    if report["defrag_gap_bytes"] < 0:
        fail(f"{path}: defrag gap {report['defrag_gap_bytes']} is negative")
    dedupedBytes = report["next"]["dedup_metrics"]["deduped_bytes"]
    if dedupedBytes != report["accepted_dedup_bytes"]:
        fail(f"{path}: deduped_bytes {dedupedBytes} != file bytes - new_bytes {report['accepted_dedup_bytes']}")
    if report["next"]["reported_file_bytes"] != nextBytes:
        fail(f"{path}: reported file size {report['next']['reported_file_bytes']} != {nextBytes} on disk")
    print(f"ok upload {path}")


def uploadSelf(path: str):
    """upload(A, A) sends no new data: new_bytes is 0, no xorb goes out, and transferred bytes are the shard alone."""
    report = load(path)["result"]
    metrics = report["next"]["dedup_metrics"]
    if metrics["new_bytes"] != 0 or metrics["xorb_bytes_uploaded"] != 0:
        fail(f"{path}: upload(A, A) new_bytes {metrics['new_bytes']}, xorb bytes {metrics['xorb_bytes_uploaded']}")
    if not (metrics["shard_bytes_uploaded"] > 0 and report["transferred_bytes"] == metrics["shard_bytes_uploaded"]):
        fail(f"{path}: upload(A, A) transferred {report['transferred_bytes']} is not shard overhead only")
    print(f"ok upload-self {path} (shard {metrics['shard_bytes_uploaded']} bytes)")


def overlapPair(path: str):
    checkAttribution(load(path), path)
    print(f"ok overlap {path}")


def main():
    command, *arguments = sys.argv[1:]
    perPath = {
        "self-overlap": selfOverlap,
        "sweep": sweep,
        "upload": uploadPair,
        "upload-self": uploadSelf,
        "overlap": overlapPair,
    }
    if command in perPath:
        for path in arguments:
            perPath[command](path)
    elif command == "sweep-point":
        sweepPoint(*arguments)
    elif command == "oracle-chunk-overlap":
        oracleChunkOverlap(*arguments)
    else:
        raise SystemExit(f"unknown check {command}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
