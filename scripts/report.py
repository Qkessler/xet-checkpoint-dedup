"""Generate the README charts and tables from results/*.json.

`report.py` redraws docs/charts/*.png, rewrites every README block between `<!-- BEGIN name -->` and
`<!-- END name -->` markers, and rewrites docs/results.md, which holds every table. `report.py --check` regenerates
both in memory and fails if either differs or a README block is missing, so a stale README breaks `just check`.
Charts are not compared.

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
        label="plan's pre-registered curve: exp(-values_per_chunk x P)",
    )
    axes.plot(
        theory["density"],
        theory["size_aware_theory_chunk_overlap_fraction"] * 100,
        color=textPrimary,
        linewidth=1,
        linestyle=":",
        zorder=4,
        label="theory weighted by actual chunk sizes (checked, ADR 0007)",
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
        headline = "**Xet dedup helps, and the sparse encoding does the heavy lifting.**"
    elif verdictCounts["holds"] == len(pairs):
        headline = "**Xet dedup alone makes full uploads as cheap as sparse deltas.**"
    else:
        headline = "**Xet dedup alone is enough for some checkpoints and not for others.**"
    lines = [
        f"{headline} Across {len(pairs)} checkpoint pairs, a full upload transferred "
        f"{percent(min(anchorShares))} to {percent(max(anchorShares))} of the checkpoint, which is "
        f"{min(sparseMultiples):.1f}x to {max(sparseMultiples):.1f}x the sparse delta for the same pair. The bar we set "
        f"before measuring (ADR 0001) was a full upload within 3x of the sparse delta: {verdictCounts['holds']} pairs "
        f"reached it, {verdictCounts['depends']} landed in between, and {verdictCounts['does not hold']} sent at least "
        "half the checkpoint.",
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
            f"- Freezing half the layers shows Xet dedup at work: "
            f"{percent(statistics.median(pair['chunk_overlap_bytes'] / pair['anchor_bytes'] for pair in frozen))} "
            f"of each checkpoint is reused for free, and the rest (the trained layers and the embedding) is sent again."
        )
    scatteredAtOnePercent = next(
        cell["mean_chunk_overlap_fraction"]
        for cell in sweep["result"]["cells"]
        if cell["mode"] == "scattered" and abs(cell["density"] - 0.01) < 1e-12
    )
    lines.append(
        f"- The reason is arithmetic, not a flaw in Xet: a chunk holds about {sweep['result']['values_per_chunk']:,.0f} bf16 "
        f"elements, so when 1% of elements change at random, {percent(scatteredAtOnePercent, 2)} of bytes stay in "
        f'reusable chunks (see "Where chunk dedup works"). Chunk dedup needs changes to cluster.'
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

    def outsideTolerance(curveKey: str) -> int:
        return sum(
            1
            for cell in scattered
            if abs(cell["mean_chunk_overlap_fraction"] - cell[curveKey]) > max(0.03, 3 * cell["standard_error"])
        )

    def worstGap(curveKey: str) -> float:
        return max(abs(cell["mean_chunk_overlap_fraction"] - cell[curveKey]) for cell in scattered)

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
            f"{seeds} seeds per cell. Against the checked curve, the chance a chunk has no change weighted by the actual chunk "
            f"sizes (ADR 0007), the scattered points miss by at most {100 * worstGap('size_aware_theory_chunk_overlap_fraction'):.2f} "
            f"percentage points, and {outsideTolerance('size_aware_theory_chunk_overlap_fraction')} of {len(scattered)} cells fall "
            f"outside the fixed tolerance. The plan's pre-registered curve `exp(-values_per_chunk x P)` misses by up to "
            f"{100 * worstGap('theory_chunk_overlap_fraction'):.1f} points, with "
            f"{outsideTolerance('theory_chunk_overlap_fraction')} of {len(scattered)} cells outside (see Surprises).",
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


def consecutivePairs(pairs: list[dict], source: str) -> list[dict]:
    return [pair for pair in pairs if pair["source"] == source and pair["step_gap"] == 1]


def valueBlock(pairs: list[dict]) -> str:
    trainable = consecutivePairs(pairs, "run-a")
    frozen = consecutivePairs(pairs, "run-b-frozen-half")
    sentPerStep = statistics.median(pair["transferred_bytes"] for pair in trainable)
    medianRatio = statistics.median(pair["transferred_bytes"] / pair["sparse_delta_bytes"] for pair in trainable)
    sparsePerStep = statistics.median(pair["sparse_delta_bytes"] for pair in trainable)
    storedShare = statistics.median(1 - pair["accepted_dedup_bytes"] / pair["anchor_bytes"] for pair in trainable)
    frozenReuse = statistics.median(pair["accepted_dedup_bytes"] / pair["anchor_bytes"] for pair in frozen)
    projectedSteps = 1000
    return "\n".join(
        [
            "If storage dedup alone made full uploads cheap, RL teams could skip the sparse encoding and let Xet do the "
            "work. That would be a big simplification, so it was worth measuring with Xet's own code. The answer: the "
            "blog's sparse encoding earns its place. On our own run, which uses the blog's setup at its most "
            f"favourable, a full upload sends a median **{megabytes(sentPerStep)} per step** and the sparse delta "
            f"**{megabytes(sparsePerStep)}**, a median of **{medianRatio:.1f}x less** per weight sync. Over a "
            f"{projectedSteps:,}-step run that projects to about {sparsePerStep * projectedSteps / 1e9:,.0f} GB instead "
            f"of {sentPerStep * projectedSteps / 1e9:,.0f} GB (median step x step count).",
            "",
            "What this gives each audience:",
            "",
            "- **RL teams syncing weights through a Hub bucket: confidence that the sparse path is worth its code.** "
            "Xet dedup does not stand in for it. TRL's pull requests for sparse sync "
            "([#5417](https://github.com/huggingface/trl/pull/5417), "
            "[#5937](https://github.com/huggingface/trl/pull/5937)) were closed without merging on 2026-09-14, and "
            "these numbers make the case for landing that work.",
            "- **Anyone keeping checkpoint history in a bucket: a realistic storage budget.** Enterprise buckets bill on "
            f"deduplicated bytes, and each new checkpoint of our run adds a median {percent(storedShare)} of its size as "
            "new chunks (counted before Xet's compression).",
            "- **Xet and TRL maintainers: a map of where content-defined dedup shines.** It pays when whole regions stay "
            f"bit-identical: with half the layers frozen, a median {percent(frozenReuse)} of each checkpoint was reused. "
            "The work also turned up a dedup accounting bug in the released `hf-xet` 1.6.0. It is fixed in xet-core and "
            "waiting for a release.",
        ]
    )


def drawValueChart(pairs: list[dict], path: Path):
    groups = [
        ("ours, all layers trainable", consecutivePairs(pairs, "run-a")),
        ("ours, half the layers frozen", consecutivePairs(pairs, "run-b-frozen-half")),
    ]
    bars = [
        ("full checkpoint", lambda pair: pair["anchor_bytes"], "#b9b8b2"),
        ("sent by a full upload, after Xet dedup", lambda pair: pair["transferred_bytes"], modeColors["rows"]),
        ("sent as a sparse delta", lambda pair: pair["sparse_delta_bytes"], modeColors["scattered"]),
    ]
    figure, axes = plt.subplots(figsize=(8, 3.8), dpi=150, facecolor=surface)
    styleAxes(axes)
    axes.grid(True, axis="x", color=gridColor, linewidth=0.8)
    axes.grid(False, axis="y")
    barHeight = 0.26
    for groupIndex, (groupLabel, groupPairs) in enumerate(groups):
        for barIndex, (barLabel, value, color) in enumerate(bars):
            megabytesPerStep = statistics.median(value(pair) for pair in groupPairs) / 1e6
            position = groupIndex - (barIndex - 1) * barHeight
            axes.barh(
                position,
                megabytesPerStep,
                height=barHeight - 0.04,
                color=color,
                label=barLabel if groupIndex == 0 else None,
            )
            axes.text(
                megabytesPerStep + 8,
                position,
                f"{megabytesPerStep:,.0f} MB",
                va="center",
                fontsize=8,
                color=textPrimary,
            )
    axes.set_yticks(range(len(groups)), [label for label, _ in groups], color=textPrimary, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlim(0, 1150)
    axes.set_xlabel("bytes per weight sync, median over one-step pairs (MB)", color=textPrimary, fontsize=9)
    axes.set_title("What one RL step costs to sync, Qwen2.5-0.5B", color=textPrimary, fontsize=11, loc="left")
    axes.legend(
        fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, labelcolor=textSecondary
    )
    figure.tight_layout()
    figure.savefig(path, facecolor=surface, metadata={"Software": None})
    plt.close(figure)


def drawVerdictChart(pairs: list[dict], path: Path):
    sourceStyles = {
        "run-a": ("ours, all layers trainable", modeColors["scattered"]),
        "run-b-frozen-half": ("ours, half the layers frozen", modeColors["rows"]),
    }
    figure, axes = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=surface)
    styleAxes(axes)
    xMaximum = 2000
    axes.axvspan(1, HOLDS_WITHIN_SPARSE_MULTIPLE, color="#e3f4ec", zorder=0)
    axes.fill_between([HOLDS_WITHIN_SPARSE_MULTIPLE, xMaximum], 50, 100, color="#fbe6e0", zorder=0)
    axes.fill_between([HOLDS_WITHIN_SPARSE_MULTIPLE, xMaximum], 0, 50, color="#fdf3dc", zorder=0)
    axes.axvline(HOLDS_WITHIN_SPARSE_MULTIPLE, color=textPrimary, linestyle="--", linewidth=1)
    axes.plot([HOLDS_WITHIN_SPARSE_MULTIPLE, xMaximum], [50, 50], color=textPrimary, linestyle="--", linewidth=1)
    axes.text(1.7, 4, "holds", ha="center", fontsize=9, color=textPrimary, fontweight="bold")
    axes.text(
        3.4, 53, "does not hold: at least half the checkpoint is sent", fontsize=9, color=textPrimary, fontweight="bold"
    )
    axes.text(3.4, 4, "depends", fontsize=9, color=textPrimary, fontweight="bold")
    for source, (label, color) in sourceStyles.items():
        sourcePairs = [pair for pair in pairs if pair["source"] == source]
        axes.scatter(
            [pair["transferred_bytes"] / pair["sparse_delta_bytes"] for pair in sourcePairs],
            [100 * pair["transferred_bytes"] / pair["anchor_bytes"] for pair in sourcePairs],
            s=36,
            color=color,
            edgecolor=surface,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
    thirdParty = [pair for pair in pairs if pair["pair"] in thirdPartyPairs]
    axes.scatter(
        [pair["transferred_bytes"] / pair["sparse_delta_bytes"] for pair in thirdParty],
        [100 * pair["transferred_bytes"] / pair["anchor_bytes"] for pair in thirdParty],
        s=36,
        color=modeColors["tensors"],
        edgecolor=surface,
        linewidth=1.5,
        label="third-party Hub checkpoints",
        zorder=3,
    )
    axes.set_xscale("log")
    axes.set_xlim(1, xMaximum)
    axes.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1x", "3x", "10x", "30x", "100x", "300x", "1000x"])
    axes.minorticks_off()
    axes.set_ylim(0, 103)
    axes.set_xlabel(
        "transferred bytes of a full upload, as a multiple of the sparse delta", color=textPrimary, fontsize=9
    )
    axes.set_ylabel("transferred bytes, % of the full checkpoint", color=textPrimary, fontsize=9)
    axes.set_title(
        f"Every checkpoint pair against the verdict rule ({len(pairs)} pairs)",
        color=textPrimary,
        fontsize=11,
        loc="left",
    )
    axes.legend(
        fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, labelcolor=textSecondary
    )
    figure.tight_layout()
    figure.savefig(path, facecolor=surface, metadata={"Software": None})
    plt.close(figure)


def resultsDocument(pairs: list[dict], sweep: dict) -> str:
    return "\n".join(
        [
            "# Results tables",
            "",
            "Generated by `scripts/report.py` from `results/*.json`. Do not edit by hand: `just report` rewrites this file, "
            "and `just check` fails if it is stale. The README carries the charts; this file carries the numbers behind them.",
            "",
            "## Verdict rule",
            "",
            "Fixed before measuring ([ADR 0001](adr/0001-test-the-implied-claim-with-preset-thresholds.md)): transferred bytes "
            "within 3x of the sparse delta **holds**, at least 50% of the anchor **does not hold**, anything between "
            "**depends**. The sparse delta is changed elements x 6 bytes (a 4-byte index plus a 2-byte bf16 value).",
            "",
            "## Per checkpoint pair",
            "",
            pairsBlock(pairs),
            "",
            '"Chunk overlap" is the share of the next file\'s bytes in chunks whose hash also appears in the base. '
            '"Transferred" is `new_bytes + shard_bytes_uploaded` from Xet\'s own `DeduplicationMetrics` after uploading the '
            'base and then the next checkpoint to the same local store. The "defrag gap" is chunk overlap that the upload '
            "path re-sent anyway, because defrag prevention refused short matches.",
            "",
            "## Every pair measured",
            "",
            allPairsBlock(pairs),
            "",
            "## Perturbation sweep",
            "",
            sweepBlock(sweep),
            "",
            "## What the Hub stored for the third-party pairs",
            "",
            hubBlock(pairs),
            "",
            "The Hub columns come from the CAS reconstruction API (`scripts/hub_reconstruction.py`). The ChenJakcy6 file's "
            "chunking on the Hub matches xet-core's (same Xet file hash); the ajjyy file was chunked differently on the Hub and "
            "shows no dedup at all, so its Hub row is not comparable to the local measurement.",
            "",
            provenanceBlock(),
            "",
        ]
    )


def buildBlocks(pairs: list[dict], sweep: dict) -> dict:
    return {
        "value": valueBlock(pairs),
        "answer": answerBlock(pairs, sweep),
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
    parser.add_argument(
        "--check", action="store_true", help="fail if README.md or docs/results.md is stale; draw nothing"
    )
    args = parser.parse_args()
    readmePath = repoRoot / "README.md"
    resultsDocumentPath = repoRoot / "docs" / "results.md"
    readme = readmePath.read_text()
    sweep = loadResult("sweep")
    pairs = loadPairs()
    rendered = renderReadme(readme, buildBlocks(pairs, sweep))
    renderedResults = resultsDocument(pairs, sweep)
    if args.check:
        stale = [
            name
            for name, current, expected in (
                ("README.md", readme, rendered),
                (
                    "docs/results.md",
                    resultsDocumentPath.read_text() if resultsDocumentPath.exists() else "",
                    renderedResults,
                ),
            )
            if current != expected
        ]
        if stale:
            print(f"{', '.join(stale)} stale: run `just report`", file=sys.stderr)
            return 1
        print("README.md and docs/results.md are up to date with results/")
        return 0
    matplotlib.use("Agg")
    chartsDir.mkdir(parents=True, exist_ok=True)
    drawValueChart(pairs, chartsDir / "cost-per-weight-sync.png")
    drawVerdictChart(pairs, chartsDir / "pairs-against-verdict-rule.png")
    drawSweepChart(sweep, chartsDir / "chunk-overlap-vs-density.png")
    drawKindChart(sweep, chartsDir / "unchanged-elements-vs-chunk-overlap.png")
    readmePath.write_text(rendered)
    resultsDocumentPath.write_text(renderedResults)
    return 0


if __name__ == "__main__":
    sys.exit(main())
