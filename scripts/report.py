"""Generate the README charts and tables from results/*.json.

`report.py` redraws docs/charts/*.png and rewrites every README block between `<!-- BEGIN name -->` and
`<!-- END name -->` markers. `report.py --check` regenerates the README blocks in memory and fails if README.md
differs or a block is missing, so a stale README breaks `just check`. Charts are not compared.

Verdicts follow ADR 0001: transferred bytes within 3x of the sparse delta "holds", at least 50% of the anchor "does
not hold", anything between "depends".
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib
import polars as pl
from matplotlib import pyplot as plt

repoRoot = Path(__file__).resolve().parent.parent
resultsDir = repoRoot / "results"
chartsDir = repoRoot / "docs" / "charts"

surface = "#fcfcfb"
textPrimary = "#0b0b0b"
textSecondary = "#52514e"
gridColor = "#e4e3df"
modeColors = {"scattered": "#2a78d6", "rows": "#eb6834", "tensors": "#1baf7a"}
modeLabels = {"scattered": "scattered", "rows": "clustered: rows", "tensors": "clustered: tensors"}


def loadResult(name: str) -> dict:
    return json.loads((resultsDir / f"{name}.json").read_text())


def styleAxes(axes):
    axes.set_facecolor(surface)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(textSecondary)
    axes.tick_params(colors=textSecondary, labelsize=9)
    axes.grid(True, which="major", color=gridColor, linewidth=0.8)
    axes.set_axisbelow(True)


def drawSweepChart(sweep: dict, path: Path):
    report = sweep["result"]
    cells = pl.DataFrame(report["cells"])
    figure, axes = plt.subplots(figsize=(8, 5.6), dpi=150, facecolor=surface)
    styleAxes(axes)

    theory = cells.filter(pl.col("mode") == "scattered").sort("density")
    axes.plot(
        theory["density"],
        theory["theory_chunk_overlap_fraction"] * 100,
        color=textSecondary,
        linewidth=1.5,
        linestyle="--",
        label="theory: exp(-values_per_chunk x P)",
    )
    axes.plot(
        theory["density"],
        theory["size_aware_theory_chunk_overlap_fraction"] * 100,
        color=textPrimary,
        linewidth=1,
        linestyle=":",
        zorder=4,
        label="theory, weighted by actual chunk sizes",
    )
    for mode, color in modeColors.items():
        modeCells = cells.filter(pl.col("mode") == mode).sort("density")
        axes.errorbar(
            modeCells["density"],
            modeCells["mean_chunk_overlap_fraction"] * 100,
            yerr=modeCells["standard_error"] * 100 * 3,
            color=color,
            marker="o",
            markersize=5,
            markeredgecolor=surface,
            markeredgewidth=1.5,
            linewidth=2,
            capsize=0,
            label=f"measured, {modeLabels[mode]} (mean of {modeCells['seed_count'][0]} seeds)",
        )
    axes.set_xscale("log")
    axes.set_ylim(-3, 103)
    axes.set_xlabel("change density P (fraction of bf16 elements altered)", color=textPrimary, fontsize=10)
    axes.set_ylabel("chunk overlap (% of next file bytes)", color=textPrimary, fontsize=10)
    axes.set_title(
        "Chunk overlap vs change density, Qwen2.5-0.5B-Instruct (bf16)",
        color=textPrimary,
        fontsize=11,
        loc="left",
    )
    axes.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        labelcolor=textSecondary,
    )
    figure.tight_layout()
    figure.savefig(path, facecolor=surface, metadata={"Software": None})
    plt.close(figure)


HOLDS_WITHIN_SPARSE_MULTIPLE = 3.0
DOES_NOT_HOLD_ANCHOR_SHARE = 0.5
thirdPartyPairs = {
    "chenjakcy6-s110-s113": ("ChenJakcy6/Qwen2-0.5B-GRPO-test", 110, 113),
    "ajjyy-s3150-s3160": ("ajjyy/Qwen2-0.5B-GRPO-attempt1", 3150, 3160),
}
runLabels = {"run-a": "ours, all layers trainable", "run-b-frozen-half": "ours, layers 0-11 frozen"}
kindOrder = [
    "embedding",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "norm",
]


def megabytes(byteCount: float) -> str:
    return f"{byteCount / 1e6:,.1f} MB"


def percent(fraction: float, digits: int = 1) -> str:
    return f"{100 * fraction:.{digits}f}%"


def verdict(transferredBytes: int, sparseDeltaBytes: int, anchorBytes: int) -> str:
    """ADR 0001. "holds" is tested first; "depends" names where the pair sits between the two thresholds."""
    if transferredBytes <= HOLDS_WITHIN_SPARSE_MULTIPLE * sparseDeltaBytes:
        return "holds"
    if transferredBytes >= DOES_NOT_HOLD_ANCHOR_SHARE * anchorBytes:
        return "does not hold"
    return (
        f"depends ({transferredBytes / sparseDeltaBytes:.1f}x the sparse delta, "
        f"{percent(transferredBytes / anchorBytes)} of the anchor)"
    )


def loadPairs() -> list[dict]:
    pairs = []
    for uploadPath in sorted(resultsDir.glob("upload-*.json")):
        pairName = uploadPath.stem.removeprefix("upload-")
        overlapPath = resultsDir / f"overlap-{pairName}.json"
        if not overlapPath.exists():
            continue
        upload = json.loads(uploadPath.read_text())["result"]
        wholeFile = json.loads(overlapPath.read_text())["result"]["whole_file"]
        if pairName in thirdPartyPairs:
            repoId, baseStep, nextStep = thirdPartyPairs[pairName]
            source = repoId
        else:
            match = re.fullmatch(r"(.+)-s(\d+)-s(\d+)", pairName)
            if not match:
                continue
            run, baseStep, nextStep = match.group(1), int(match.group(2)), int(match.group(3))
            source = run
        anchorBytes = upload["next"]["file_bytes"]
        pairs.append(
            {
                "pair": pairName,
                "source": source,
                "base_step": baseStep,
                "next_step": nextStep,
                "step_gap": nextStep - baseStep,
                "anchor_bytes": anchorBytes,
                "unchanged_element_fraction": wholeFile["unchanged_element_fraction"],
                "sparse_delta_bytes": wholeFile["sparse_delta_bytes"],
                "chunk_overlap_bytes": wholeFile["chunk_overlap_bytes"],
                "accepted_dedup_bytes": upload["accepted_dedup_bytes"],
                "transferred_bytes": upload["transferred_bytes"],
                "defrag_gap_bytes": upload["defrag_gap_bytes"],
                "xorb_bytes_uploaded": upload["next"]["dedup_metrics"]["xorb_bytes_uploaded"],
                "verdict": verdict(upload["transferred_bytes"], wholeFile["sparse_delta_bytes"], anchorBytes),
            }
        )
    return pairs


def pairRow(pair: dict, label: str) -> str:
    anchor = pair["anchor_bytes"]
    return (
        f"| {label} | {pair['step_gap']} | {megabytes(anchor)} | {percent(1 - pair['unchanged_element_fraction'], 2)} "
        f"| {megabytes(pair['sparse_delta_bytes'])} | {percent(pair['chunk_overlap_bytes'] / anchor)} "
        f"| {megabytes(pair['transferred_bytes'])} ({percent(pair['transferred_bytes'] / anchor)}) "
        f"| {pair['transferred_bytes'] / pair['sparse_delta_bytes']:.1f}x "
        f"| {percent(pair['defrag_gap_bytes'] / anchor, 2)} | {pair['verdict']} |"
    )


tableHeader = (
    "| pair | step gap | anchor | elements changed | sparse delta | chunk overlap | transferred (share of anchor) "
    "| transferred / sparse delta | defrag gap | verdict |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
)


def consecutiveSummaryRow(pairs: list[dict], label: str) -> str:
    def spread(values: list[float], formatter) -> str:
        return f"{formatter(statistics.median(values))} ({formatter(min(values))} to {formatter(max(values))})"

    anchor = pairs[0]["anchor_bytes"]
    verdicts = sorted({pair["verdict"] for pair in pairs})
    return (
        f"| {label}, {len(pairs)} consecutive pairs: median (min to max) | 1 | {megabytes(anchor)} "
        f"| {spread([1 - pair['unchanged_element_fraction'] for pair in pairs], lambda value: percent(value, 2))} "
        f"| {spread([pair['sparse_delta_bytes'] for pair in pairs], megabytes)} "
        f"| {spread([pair['chunk_overlap_bytes'] / anchor for pair in pairs], percent)} "
        f"| {spread([pair['transferred_bytes'] / anchor for pair in pairs], percent)} "
        f"| {spread([pair['transferred_bytes'] / pair['sparse_delta_bytes'] for pair in pairs], lambda value: f'{value:.1f}x')} "
        f"| {spread([pair['defrag_gap_bytes'] / anchor for pair in pairs], lambda value: percent(value, 2))} "
        f"| {', '.join(verdicts)} |"
    )


def pairLabel(pair: dict) -> str:
    if pair["source"] in runLabels:
        return f"{pair['source']} ({runLabels[pair['source']]}) steps {pair['base_step']} to {pair['next_step']}"
    return f"{pair['source']} steps {pair['base_step']} to {pair['next_step']}"


def answerBlock(pairs: list[dict], sweep: dict) -> str:
    anchorShares = [pair["transferred_bytes"] / pair["anchor_bytes"] for pair in pairs]
    sparseMultiples = [pair["transferred_bytes"] / pair["sparse_delta_bytes"] for pair in pairs]
    verdictCounts = {
        name: sum(1 for pair in pairs if pair["verdict"].startswith(name))
        for name in ("holds", "depends", "does not hold")
    }
    trainable = [pair for pair in pairs if pair["source"] == "run-a" and pair["step_gap"] == 1]
    frozen = [pair for pair in pairs if pair["source"] == "run-b-frozen-half" and pair["step_gap"] == 1]
    if verdictCounts["does not hold"] == len(pairs):
        headline = "**No.**"
    elif verdictCounts["holds"] == len(pairs):
        headline = "**Yes.**"
    else:
        headline = "**Only in part.**"
    lines = [
        f"{headline} Across {len(pairs)} checkpoint pairs, uploading the full next checkpoint transferred "
        f"{percent(min(anchorShares))} to {percent(max(anchorShares))} of its size, which is "
        f"{min(sparseMultiples):.1f}x to {max(sparseMultiples):.1f}x the blog's sparse delta for the same pair. "
        f"Under the rule fixed before measuring (ADR 0001), the claim holds on {verdictCounts['holds']} pairs, "
        f"depends on {verdictCounts['depends']}, and does not hold on {verdictCounts['does not hold']}.",
        "",
    ]
    if trainable:
        lines.append(
            f"- On our own one-step-apart GRPO checkpoints (lr 1e-6, the blog's setup and its best case), a median "
            f"{percent(statistics.median(1 - pair['unchanged_element_fraction'] for pair in trainable), 2)} of elements "
            f"change per step, yet only {percent(statistics.median(pair['chunk_overlap_bytes'] / pair['anchor_bytes'] for pair in trainable))} "
            f"of the bytes sit in chunks Xet can reuse."
        )
    if frozen:
        lines.append(
            f"- Freezing half the layers is the case where chunk dedup works: "
            f"{percent(statistics.median(pair['chunk_overlap_bytes'] / pair['anchor_bytes'] for pair in frozen))} "
            f"of each checkpoint is reused, and the rest is still sent in full."
        )
    scatteredAtOnePercent = next(
        cell["mean_chunk_overlap_fraction"]
        for cell in sweep["result"]["cells"]
        if cell["mode"] == "scattered" and abs(cell["density"] - 0.01) < 1e-12
    )
    lines.append(
        f"- The reason is arithmetic, not a defect: a chunk holds about {sweep['result']['values_per_chunk']:,.0f} bf16 "
        f"elements, so when 1% of elements change at random, {percent(scatteredAtOnePercent, 2)} of bytes stay in "
        f"reusable chunks (chart 1). Chunk dedup needs changes to cluster."
    )
    return "\n".join(lines)


def pairsBlock(pairs: list[dict]) -> str:
    rows = [tableHeader]
    for source in ("run-a", "run-b-frozen-half"):
        sourcePairs = [pair for pair in pairs if pair["source"] == source]
        consecutive = [pair for pair in sourcePairs if pair["step_gap"] == 1]
        if consecutive:
            rows.append(consecutiveSummaryRow(consecutive, f"{source} ({runLabels[source]})"))
        for pair in sorted(
            (pair for pair in sourcePairs if pair["step_gap"] > 1),
            key=lambda pair: (pair["step_gap"], pair["base_step"]),
        ):
            rows.append(pairRow(pair, pairLabel(pair)))
    for pair in pairs:
        if pair["pair"] in thirdPartyPairs:
            rows.append(pairRow(pair, pairLabel(pair)))
    return "\n".join(rows)


def allPairsBlock(pairs: list[dict]) -> str:
    rows = [tableHeader]
    rows += [
        pairRow(pair, pairLabel(pair))
        for pair in sorted(pairs, key=lambda pair: (pair["source"], pair["base_step"], pair["next_step"]))
    ]
    return "\n".join(rows)


def sweepBlock(sweep: dict) -> str:
    report = sweep["result"]
    cells = report["cells"]

    def firstDensityBelow(mode: str, threshold: float) -> str:
        modeCells = sorted((cell for cell in cells if cell["mode"] == mode), key=lambda cell: cell["density"])
        below = next((cell for cell in modeCells if cell["mean_chunk_overlap_fraction"] < threshold), None)
        return f"{below['density']:g}" if below else "not reached by 0.1"

    def overlapAt(mode: str, density: float) -> str:
        cell = next(cell for cell in cells if cell["mode"] == mode and abs(cell["density"] - density) < density * 1e-9)
        return percent(cell["mean_chunk_overlap_fraction"])

    scattered = [cell for cell in cells if cell["mode"] == "scattered"]
    outside = [
        cell
        for cell in scattered
        if abs(cell["mean_chunk_overlap_fraction"] - cell["theory_chunk_overlap_fraction"])
        > max(0.03, 3 * cell["standard_error"])
    ]
    sizeAwareWorst = max(
        abs(cell["mean_chunk_overlap_fraction"] - cell["size_aware_theory_chunk_overlap_fraction"])
        for cell in scattered
    )
    plainWorst = max(
        abs(cell["mean_chunk_overlap_fraction"] - cell["theory_chunk_overlap_fraction"]) for cell in scattered
    )
    seeds = len(report["plan"]["seeds"])
    return "\n".join(
        [
            f"| | scattered | clustered: rows | clustered: tensors |",
            "| --- | ---: | ---: | ---: |",
            f"| chunk overlap at density 1e-2 (about the blog's per-step rate) | {overlapAt('scattered', 0.01)} | {overlapAt('rows', 0.01)} | {overlapAt('tensors', 0.01)} |",
            f"| chunk overlap at density 1e-4 | {overlapAt('scattered', 0.0001)} | {overlapAt('rows', 0.0001)} | {overlapAt('tensors', 0.0001)} |",
            f"| first density with overlap under 50% | {firstDensityBelow('scattered', 0.5)} | {firstDensityBelow('rows', 0.5)} | {firstDensityBelow('tensors', 0.5)} |",
            f"| first density with overlap under 1% | {firstDensityBelow('scattered', 0.01)} | {firstDensityBelow('rows', 0.01)} | {firstDensityBelow('tensors', 0.01)} |",
            "",
            f"Mean chunk {report['base_mean_chunk_bytes']:,.0f} bytes, so {report['values_per_chunk']:,.0f} bf16 elements per chunk; "
            f"{seeds} seeds per cell. The pre-registered curve `exp(-values_per_chunk x P)` misses the scattered points by up to "
            f"{100 * plainWorst:.1f} percentage points, and {len(outside)} of {len(scattered)} scattered cells fall outside its fixed "
            f"tolerance. The same chance weighted by the actual chunk sizes misses by at most {100 * sizeAwareWorst:.2f} points (see Surprises).",
        ]
    )


def hubBlock(pairs: list[dict]) -> str:
    rows = [
        "| pair | Hub: chunks of next file | Hub: chunks shared with base | Hub: bytes in shared terms | local: chunk overlap "
        "| local: dedup accepted by upload |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pairName, (repoId, baseStep, nextStep) in thirdPartyPairs.items():
        hubPath = resultsDir / f"hub-reconstruction-{pairName}.json"
        pair = next((pair for pair in pairs if pair["pair"] == pairName), None)
        if not hubPath.exists() or pair is None:
            continue
        hub = json.loads(hubPath.read_text())["result"]
        rows.append(
            f"| {repoId} steps {baseStep} to {nextStep} | {hub['next_chunks']:,} | {hub['shared_chunks']:,} "
            f"({percent(hub['shared_chunks'] / hub['next_chunks'], 2)}) "
            f"| {percent(hub['shared_bytes_in_whole_shared_terms'] / hub['next_file_bytes'], 2)} "
            f"| {percent(pair['chunk_overlap_bytes'] / pair['anchor_bytes'], 2)} "
            f"| {percent(pair['accepted_dedup_bytes'] / pair['anchor_bytes'], 2)} |"
        )
    return "\n".join(rows)


def provenanceBlock() -> str:
    commits = set()
    crateVersions = set()
    dirty = 0
    files = 0
    for path in sorted(resultsDir.glob("*.json")):
        tool = json.loads(path.read_text()).get("tool", {})
        files += 1
        commits.add(tool.get("git_commit", "unknown")[:7])
        dirty += 1 if tool.get("git_dirty") else 0
        crateVersions.update(f"{name} {version}" for name, version in tool.get("crate_versions", {}).items())
    return (
        f"Generated from {files} files in `results/`, produced by tool commit(s) {', '.join(sorted(commits))} "
        f"({dirty} from a dirty tree), on {', '.join(sorted(crateVersions))}."
    )


def drawKindChart(sweep: dict, path: Path):
    panels = [
        ("run-a-s10-s11", "ours, all layers trainable, steps 10 to 11"),
        ("run-b-frozen-half-s05-s06", "ours, layers 0-11 frozen, steps 5 to 6"),
        ("chenjakcy6-s110-s113", "ChenJakcy6, steps 110 to 113"),
        ("ajjyy-s3150-s3160", "ajjyy, steps 3150 to 3160"),
    ]
    zeroChange = 1e-7
    theory = sorted(
        (cell for cell in sweep["result"]["cells"] if cell["mode"] == "scattered"), key=lambda cell: cell["density"]
    )
    figure, axesGrid = plt.subplots(2, 2, figsize=(10, 7.5), dpi=150, facecolor=surface, sharex=True, sharey=True)
    for axes, (pairName, title) in zip(axesGrid.flat, panels):
        styleAxes(axes)
        axes.plot(
            [cell["density"] for cell in theory],
            [100 * cell["size_aware_theory_chunk_overlap_fraction"] for cell in theory],
            color=textSecondary,
            linestyle=":",
            linewidth=1,
            label="scattered changes, chunk-size-weighted theory",
        )
        overlapPath = resultsDir / f"overlap-{pairName}.json"
        if not overlapPath.exists():
            continue
        report = json.loads(overlapPath.read_text())["result"]
        tensors = [row for row in report["tensors"] if row["unchanged_element_fraction"] is not None]
        axes.scatter(
            [max(1 - row["unchanged_element_fraction"], zeroChange) for row in tensors],
            [row["chunk_overlap_percent"] for row in tensors],
            s=10,
            color="#b9b8b2",
            label="one tensor",
            zorder=2,
        )
        kinds = [row for row in report["by_kind"] if row["group"] in kindOrder]
        axes.scatter(
            [max(1 - row["unchanged_element_fraction"], zeroChange) for row in kinds],
            [row["chunk_overlap_percent"] for row in kinds],
            s=40,
            color=modeColors["scattered"],
            edgecolor=surface,
            linewidth=1.5,
            label="tensor kind (roll-up)",
            zorder=3,
        )
        labelGroups = {"embedding": ["embedding"], "norm": ["norm"]}
        labelGroups["attention and MLP kinds"] = [kind for kind in kindOrder if kind not in ("embedding", "norm")]
        labelOffsets = {
            "embedding": (-6, 14, "right"),
            "norm": (0, -14, "center"),
            "attention and MLP kinds": (6, 8, "left"),
        }
        for label, members in labelGroups.items():
            memberRows = [row for row in kinds if row["group"] in members]
            if not memberRows:
                continue
            anchorRow = max(memberRows, key=lambda row: 1 - row["unchanged_element_fraction"])
            axes.annotate(
                label,
                (
                    max(1 - anchorRow["unchanged_element_fraction"], zeroChange),
                    max(row["chunk_overlap_percent"] for row in memberRows),
                ),
                xytext=labelOffsets[label][:2],
                textcoords="offset points",
                ha=labelOffsets[label][2],
                fontsize=7,
                color=textSecondary,
            )
        axes.set_title(title, color=textPrimary, fontsize=9, loc="left")
        axes.set_xscale("log")
        axes.set_xlim(zeroChange / 2, 0.2)
        axes.set_ylim(-16, 103)
    for axes in axesGrid[1]:
        axes.set_xlabel(
            "elements changed (1 - unchanged element fraction; 1e-7 means none)", color=textPrimary, fontsize=8
        )
    for axes in axesGrid[:, 0]:
        axes.set_ylabel("chunk overlap (% of bytes)", color=textPrimary, fontsize=9)
    handles, labels = axesGrid[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=textSecondary)
    figure.suptitle(
        "Unchanged elements vs chunk overlap, per tensor and per tensor kind",
        color=textPrimary,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(path, facecolor=surface, metadata={"Software": None})
    plt.close(figure)


def buildBlocks(sweep: dict) -> dict:
    pairs = loadPairs()
    return {
        "answer": answerBlock(pairs, sweep),
        "pairs": pairsBlock(pairs),
        "all-pairs": allPairsBlock(pairs),
        "sweep": sweepBlock(sweep),
        "hub": hubBlock(pairs),
        "provenance": provenanceBlock(),
    }


def renderReadme(readme: str, blocks: dict) -> str:
    for name, content in blocks.items():
        begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        pattern = re.compile(rf"{re.escape(begin)}.*?{re.escape(end)}", re.S)
        if not pattern.search(readme):
            raise SystemExit(f"README.md has no {begin} ... {end} block")
        readme = pattern.sub(lambda match: f"{begin}\n{content}\n{end}", readme)
    return readme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README.md is stale; draw nothing")
    args = parser.parse_args()
    readmePath = repoRoot / "README.md"
    readme = readmePath.read_text()
    sweep = loadResult("sweep")
    rendered = renderReadme(readme, buildBlocks(sweep))
    if args.check:
        if rendered != readme:
            print("README.md is stale: run `just report`", file=sys.stderr)
            return 1
        print("README.md is up to date with results/")
        return 0
    matplotlib.use("Agg")
    chartsDir.mkdir(parents=True, exist_ok=True)
    drawSweepChart(sweep, chartsDir / "chunk-overlap-vs-density.png")
    drawKindChart(sweep, chartsDir / "unchanged-elements-vs-chunk-overlap.png")
    readmePath.write_text(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
