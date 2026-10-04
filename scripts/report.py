"""Generate the README charts and tables from results/*.json.

`report.py` redraws docs/charts/*.png and rewrites every README block between `<!-- BEGIN name -->` and
`<!-- END name -->` markers. `report.py --check` will regenerate the README blocks in memory and fail if README.md
differs, so a stale README breaks `just check`. Until the README exists (plan step 6), `--check` passes trivially.
"""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README.md is stale; draw nothing")
    args = parser.parse_args()
    if args.check:
        return 0
    matplotlib.use("Agg")
    chartsDir.mkdir(parents=True, exist_ok=True)
    drawSweepChart(loadResult("sweep"), chartsDir / "chunk-overlap-vs-density.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
