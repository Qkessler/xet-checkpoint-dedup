# Check the sweep against the chunk-size-weighted curve

The plan fixed the scattered-sweep check before measuring: the mean of 5 seeds must be within max(3 percentage points, 3 standard errors) of `exp(-values_per_chunk x P)`, where `values_per_chunk` is the mean chunk size over 2 bytes. The measured points fell outside it in 4 of 16 cells, by up to 9.1 points (run log, step 2). The cause is in the model, not the tool. Overlap is measured in bytes, and Xet chunks vary from 8 KiB to 128 KiB (coefficient of variation 0.64, 16% at the cap). A chunk of `s` bytes survives scattered changes with probability `(1-P)^(s/2)`, and the byte-weighted average of that is not its value at the mean size. Large chunks are both likelier to be hit and carry more bytes.

The check now uses the same survival probability weighted by the actual chunk sizes of the base checkpoint:

```
expected overlap(P) = sum over base chunks of s * (1-P)^(s/2), divided by the sum of s
```

The tolerance is unchanged. The owner approved the swap on 2026-10-04.

This curve was adopted **after** seeing the data, so it is not pre-registered. Three things limit how much that matters:

- It has no free parameters. It reads only the base file's chunk sizes and never the measured overlap.
- It is the plan's model without the "every chunk has the mean size" shortcut. At a single chunk size the two curves coincide (unit test `size_aware_theory_equals_plain_theory_for_equal_chunks`).
- It makes a second prediction that was checked separately: the share of reused chunks (a count, not bytes) matches `mean((1-P)^(s/2))` within 0.3 points, and the chunk count stays within 0.07% of the base's up to density 1e-3.

## Considered Options

- **Keep the plan's curve and leave the check failing**: honest, but it leaves the outer loop permanently red for a known, explained reason, which hides new failures.
- **Widen the tolerance**: tuning a check to pass. Ruled out by AGENTS.md.
- **Swap in the chunk-size-weighted curve, with the same tolerance** (chosen).

## Consequences

- `just e2e` checks scattered cells against `size_aware_theory_chunk_overlap_fraction` and prints the gap to the plan's curve alongside it. The plan's curve stays in `results/sweep.json` and in chart 1, labelled as the pre-registered one.
- No verdict changes. Verdicts compare transferred bytes with the sparse delta (ADR 0001) and never use the curve. The curve only checks that the tool behaves like the model on synthetic data.
- The README reports the swap as a surprise, and says plainly that the curve in the gate was chosen after measuring.
