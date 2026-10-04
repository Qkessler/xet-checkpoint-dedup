# Test the implied claim, with thresholds fixed before measuring

The delta-weight-sync blog says that uploading full anchors every step, Xet "would _still_ only transfer the changed chunks". Read literally, this is true by construction, so we test the implied reading instead: that full-anchor uploads cost about as little as sparse deltas. The verdict compares transferred bytes per step against the blog's own sparse-delta size for the same checkpoint pair. Within 3x of the sparse delta counts as "holds". At least 50% of the anchor counts as "does not hold". Anything in between is reported as "depends on X", with X named. The thresholds are written here before any measurement, so the result cannot move the goalposts.

## Consequences

- The README title asks a question, and the write-up credits the blog's sparse encoding, which our result is expected to support rather than undermine.
- Every reported pair needs a sparse-delta size next to it. We compute it ourselves from the unchanged element fraction (changed elements x (4-byte index + 2-byte bf16 value)), because the blog only gives 20 to 35 MB for Qwen3-0.6B.
