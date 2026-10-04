# Local store only; Hub measurements are future work

`upload` targets a `local://` store, never a Hub bucket or repo. Within one checkpoint pair, global dedup adds almost nothing, because the base checkpoint was just uploaded to the same store and its shard is already in the local cache. A local store is also reproducible by anyone without credentials, and leaves nothing behind to clean up.

## Consequences

The README lists these as future work, together with the trade-off:

- **Real Hub upload of one pair to a private bucket.** It would exercise the server, global dedup queries, and the real shard registration path. We skipped it to keep the result reproducible offline and to fit the one-day budget.
- **Hub reconstruction-API cross-check.** Fetching the reconstruction terms of public consecutive checkpoints shows which chunks the Hub actually shares between them, without downloading any weights. It is an independent check on our numbers. For now one row is reproduced by hand and cited, rather than built into the tool.
