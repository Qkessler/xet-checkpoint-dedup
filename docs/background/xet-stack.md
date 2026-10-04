# Hugging Face Xet storage stack: primary-source research

Researched 2026-10-03. Source of truth is xet-core at commit `d2f667c` (2026-10-02, workspace version 1.6.0), cloned to `/workplace/enrikes/huggingface-xet/xet-core`, plus huggingface_hub `main` at `d711944` (2026-10-02), the Xet protocol spec, hub-docs, and the HF blog. Every claim carries a reference key; the keys resolve in the Sources table at the end. Anything I did not verify directly is marked UNVERIFIED. Anything I ran myself is marked VERIFIED-LOCAL.

Shorthand: `XC/<path>` means `https://github.com/huggingface/xet-core/blob/d2f667cc247a7ae54a2cf93e413f8fee96c3c927/<path>`. `HH/<path>` means `https://github.com/huggingface/huggingface_hub/blob/d711944/<path>`.

---

## 1. What xet-core and hf-xet are

### 1.1 Identity, language, license

- xet-core is the Rust client for Xet storage. The README says it is "not meant to be used directly" and is meant to be called from huggingface_hub [XC/README.md].
- Language: Rust, edition 2024. Python bindings use PyO3 0.29 (abi3-py38) and are built with maturin [XC/Cargo.toml], [XC/hf_xet/Cargo.toml], [XC/hf_xet/pyproject.toml].
- License: Apache-2.0 for the workspace and the PyPI package [XC/Cargo.toml], [XC/hf_xet/pyproject.toml]. GitHub reports the same, plus about 606 stars and 43 open issues and PRs as of 2026-10-03 [GH-API-repo].
- History: the code was adapted from the earlier XetHub `xetdata/xet-core`, which had deep git integration and a very different backend [XC/README.md]. XetHub joined HF in August 2024 [BLOG-xethub-joins-hf].
- Versions: hf-xet 1.6.0 shipped on PyPI on 2026-08-03 [PYPI-hf-xet]. All five crates are at 1.6.0 on crates.io and were first published on 2026-03-31 [CRATES]. A PR to bump to `1.7.0-dev1` is open (#988) [GH-PR-988].

### 1.2 Repo layout (after the March 2026 restructure)

Crates were merged into five published packages in March 2026 [XC/api_changes/update_260309_package_restructure.md]. The root workspace members are `xet_runtime`, `xet_core_structures`, `xet_client`, `xet_data`, `xet_pkg`, and `git_xet`. `hf_xet`, `simulation`, `wasm/*`, and `examples/xet_pkg_napi` are excluded and have their own manifests [XC/Cargo.toml], [XC/AGENTS.md].

| Dir | Cargo package (crates.io) | Responsibility |
| --- | --- | --- |
| `xet_pkg/` | `hf-xet` (library name `xet`) | Public Rust session API (`XetSession`, upload commits, download groups). Also contains the `xtool` binary [XC/AGENTS.md], [XC/xet_pkg/Cargo.toml] |
| `xet_data/` | `xet-data` | Chunking (`deduplication/chunking.rs`), dedup (`file_deduplication.rs`, `defrag_prevention.rs`), the upload session (`processing/`), and reconstruction (`file_reconstruction/`) [XC/AGENTS.md] |
| `xet_client/` | `xet-client` | CAS and Hub HTTP clients, retries, adaptive concurrency, chunk cache, telemetry sink, and a simulation local server [XC/AGENTS.md] |
| `xet_core_structures/` | `xet-core-structures` | MerkleHash and the aggregated hashes, the MDB shard format, the xorb format, and compression (LZ4, BG4) [XC/AGENTS.md] |
| `xet_runtime/` | `xet-runtime` | Async runtime, the `config_group!` config system (all `HF_XET_*` vars), logging, and the cache dir [XC/AGENTS.md] |
| `hf_xet/` | PyPI `hf-xet` | PyO3 bindings. Default features are `no-default-cache` and `elevated_information_level` [XC/hf_xet/Cargo.toml] |
| `git_xet/` | `git_xet` (not published) | `git-xet`, a Git LFS custom transfer agent [XC/README.md], [XC/git_xet/src/constants.rs] |
| `wasm/hf_xet_thin_wasm` | published wasm crate | Chunker, xorb/file/verification hash, HMAC, and MerkleHashSubtree for huggingface.js [XC/wasm/hf_xet_thin_wasm/src/lib.rs], [XC/wasm/hf_xet_thin_wasm/README.md] |
| `wasm/hf_xet_wasm` | example only | Browser upload/download smoke test, not an SDK [XC/README.md] |
| `simulation/` | not published | Upload concurrency simulations and chunk-cache and sha256 benches. Bins: `run_upload_scenario`, `run_upload_simulations` [XC/simulation/Cargo.toml] |
| `openapi/cas.openapi.yaml` | | OpenAPI for the CAS endpoints [XC/openapi/cas.openapi.yaml] |
| `api_changes/` | | Dated change records written for AI agents and downstream maintainers. PRs with API changes are expected to add one [XC/api_changes/README.md] |
| `scripts/diag/` | | Per-OS diagnostic wrappers (stack traces, core dumps) [XC/README.md] |

### 1.3 How hf-xet plugs into huggingface_hub

- huggingface_hub 2.1.1 (on PyPI 2026-10-01) depends on `hf-xet>=1.6.0,<2.0.0`, but only on `x86_64/amd64/AMD64/arm64/aarch64`. It is also available as the `hf_xet` extra [PYPI-hfhub-2.1.1], [HH/setup.py].
- Historical requirements: hf_xet has been installed by default since huggingface_hub 0.32.0. For 0.30.0 to <0.32.0 you had to `pip install hf-xet` yourself. Versions below 0.30.0 go through the LFS bridge [DOC-using-xet].
- Activation: `is_xet_available()` returns True when the `hf_xet` package is importable, unless `HF_HUB_DISABLE_XET` is truthy [HH/src/huggingface_hub/utils/_runtime.py], [HH/src/huggingface_hub/constants.py].
- Current hub code goes through the new `hf_xet.XetSession` API. One process-global `XetSessionHolder` is created lazily and rebuilt after sigint-abort or fork [HH/src/huggingface_hub/utils/_xet.py]. Uploads register files into a `XetSession` upload commit, which chunks, dedups, and uploads them [HH/src/huggingface_hub/_upload_pipeline.py].
- Fallback when Xet is unavailable or disabled: legacy clients keep working through the LFS bridge. When a non-Xet client uploads, the file lands in LFS storage first and is then migrated to Xet in the background [DOC-using-xet], [BLOG-migrating].
- Other hub env vars: `HF_XET_HIGH_PERFORMANCE`, and `HF_XET_CACHE`, which defaults to `$HF_HOME/xet`. `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in favour of `HF_XET_HIGH_PERFORMANCE` [HH/src/huggingface_hub/constants.py].
- Progress UI: the hub shows network bytes with no denominator, because "byte count is hard to predict (dedup/compression)" [HH/src/huggingface_hub/utils/_xet_progress_reporting.py]. I found no code in huggingface_hub that reads `XetCommitReport.dedup_metrics` or surfaces a dedup ratio to the user (grep of `_upload_pipeline.py`, `_commit_api.py`, `utils/_xet*.py`) [HH/src/huggingface_hub/_upload_pipeline.py]. UNVERIFIED that no other path does. Older hub versions showed a "New Data Upload" bar [BLOG-parquet-cdc].

---

## 2. How Xet storage works

The protocol spec is published as "Xet Protocol Specification, Version 1.1.0" [SPEC-index]. Its stated goal is that independent implementations "MUST produce the same hashes, objects, and API behavior" [SPEC-index]. The reference implementations are xet-core (Rust) and huggingface.js (XetBlob download, `uploadShards` upload, with chunking through hf_xet_thin_wasm or `@huggingface/xetchunk-wasm`) [SPEC-index], [XC/wasm/hf_xet_thin_wasm/CHUNKER_BENCHMARK.md].

### 2.1 Content-defined chunking

- Algorithm: GearHash CDC, using the `gearhash` crate's `Hasher::default()` table. The per-byte update is `h = (h << 1) + TABLE[b]` with wrapping arithmetic [SPEC-chunking], [XC/xet_data/src/deduplication/chunking.rs].
- Constants: `TARGET_CHUNK_SIZE = 64 KiB`, `MINIMUM_CHUNK_DIVISOR = 8` (min 8 KiB), `MAXIMUM_CHUNK_MULTIPLIER = 2` (max 128 KiB) [XC/xet_core_structures/src/xorb_object/constants.rs].
- Mask: `target - 1`, shifted all the way left, which gives `0xFFFF000000000000`, so 16 high bits and a 1/2^16 match probability per byte [XC/xet_data/src/deduplication/chunking.rs], [SPEC-chunking].
- Skip-ahead: the chunker skips `min - 64 - 1` bytes before testing (the hash window is 64 bytes). There is also an explicit guard that rejects a match that fires before the minimum chunk size, then keeps scanning [XC/xet_data/src/deduplication/chunking.rs].
- Reset: `set_hash(0)` after every boundary, which keeps streamed chunking stable [XC/xet_data/src/deduplication/chunking.rs], [SPEC-chunking].
- Missing guard in huggingface.js: until 2026-08-27, huggingface.js lacked the guard. It accepted 8,128-byte chunks below the minimum and diverged from xet-core at chunk 44 of a random corpus [XC/wasm/hf_xet_thin_wasm/CHUNKER_BENCHMARK.md]. This was fixed in huggingface.js #2422, and #2423 added a 60-case conformance matrix that runs against a vendored xet-core wasm oracle [HFJS-PR-2422], [HFJS-PR-2423].
- Stable boundaries (2026-04/05): `next_stable_chunk_boundary` returns a boundary followed by two consecutive chunks with sizes in `[2*min, max - min)`. It is used to make rechunking around dirty ranges converge for range uploads [XC/xet_core_structures/src/xorb_object/constants.rs], [XC/api_changes/update_260424_next_stable_chunk_boundary.md].
- Parallel chunking: `find_partitions` exists, and the design is written up in `xet_data/src/deduplication/parallel chunking.pdf` [XC/xet_data/src/deduplication/chunking.rs]. A parallel-chunk example PR (#525) has been open since 2025-10 [GH-PR-525].
- Reference vectors live in the `xet-team/xet-spec-reference-files` dataset: an EV CSV, its `.chunks`, `.xet-file-hash`, a `.xorb`, and shards [SPEC-chunking], [HF-API-specref].
- VERIFIED-LOCAL: `cargo build --release -p xet-data --example chunk --example hash` built in about 1 minute. Running `chunk -i Electric_Vehicle_Population_Data_20250917.csv` produced 796 lines that are byte-identical to the reference `.chunks`. Running `hash -h file` over them gave `118a5332...6aaf97`, which matches both the reference file and the Hub tree API's `xetHash` for that file [HF-API-specref].

### 2.2 Hashing (BLAKE3, keyed)

- Chunk hash: `blake3::keyed_hash(DATA_KEY, bytes)` [XC/xet_core_structures/src/merklehash/data_hash.rs], [SPEC-hashing].
- Xorb hash: a Merkle root over `(chunk_hash, size)` pairs [XC/xet_core_structures/src/merklehash/aggregated_hashes.rs], [SPEC-hashing].
  - Internal nodes are `blake3::keyed_hash(INTERNAL_NODE_KEY, "{hash:x} : {size}\n"...)`.
  - The tree is content-defined. A group is cut when `hash % 4 == 0`, with a minimum of 2 and a maximum of 9 children (`2*BF+1`, BF=4).
- File hash: the same aggregated root, then HMAC'd (BLAKE3 keyed) with a 32-byte zero key. `file_hash_with_salt` is the variant with a non-zero salt [XC/xet_core_structures/src/merklehash/aggregated_hashes.rs], [SPEC-hashing].
- Term verification hash: `blake3::keyed_hash(VERIFICATION_KEY, concat(raw chunk hashes in range))`. Every term in an uploaded shard needs one [SPEC-hashing], [XC/xet_core_structures/src/metadata_shard/chunk_verification.rs].
- Hex encoding: the string form reverses the byte order within each 8-byte (u64) word. This is a classic interop trap [XC/xet_core_structures/src/merklehash/data_hash.rs], [SPEC-hashing].

### 2.3 Xorbs

- Limits: `MAX_XORB_BYTES = 64 MiB` and `MAX_XORB_CHUNKS = 8 * 1024` [XC/xet_core_structures/src/xorb_object/constants.rs].
  - The spec says the CAS rejects xorbs over 64 MiB serialized. It also says there is "no explicit limit on the number of chunks", which conflicts with the 8,192 constant and with the dedup spec page, which states 8,192 [SPEC-xorb], [SPEC-dedup].
  - Expect roughly 1,024 chunks per xorb [SPEC-xorb].
- Chunk header (8 bytes, packed): `version u8`, `compressed_len u24 LE`, `compression_scheme u8`, `uncompressed_len u24 LE` [XC/xet_core_structures/src/xorb_object/xorb_chunk_format.rs], [SPEC-xorb].
- Footer/info (`XorbObjectInfoV1`): ident `XETBLOB` version 1, then a hashes section (`XBLBHSH`) and a boundaries section (`XBLBBND`, v1 with unpacked info). The default info length is 92 bytes [XC/xet_core_structures/src/xorb_object/xorb_object_format.rs].
  - Uploads POST the chunks without the footer [GH-PR-974].
- Uniqueness nonce (June 2026): a 4-byte per-upload nonce sits in the 16-byte footer buffer. It is excluded from the xorb hash and lets otherwise byte-identical serializations differ, for S3 ETag differentiation [XC/api_changes/update_260604_xorb_uniqueness_nonce.md], [GH-PR-865].
- Compression schemes: `None=0`, `LZ4=1` (lz4_flex frame), `ByteGrouping4LZ4=2`, and `Auto=99`, which is never written to data [XC/xet_core_structures/src/xorb_object/compression_scheme.rs].
  - `Auto` resolves per chunk through `BG4Predictor`, a KL-divergence heuristic over per-byte popcount distributions [XC/xet_core_structures/src/xorb_object/compression_scheme.rs], [SPEC-xorb].
  - Config: `HF_XET_XORB_COMPRESSION_POLICY` (`auto|none|lz4|bg4-lz4`) and `HF_XET_XORB_COMPRESSION_SCHEME_RETEST_INTERVAL=32` [XC/xet_runtime/src/config/groups/xorb.rs].
  - An open PR adds AVX-512 support to the BG4 predictor (#965) [GH-PR-965].

### 2.4 Shards (MDB shard format)

- Header: a 32-byte tag, version 2. Footer version 1 [XC/xet_core_structures/src/metadata_shard/shard_format.rs].
- Sections [XC/xet_core_structures/src/metadata_shard/shard_format.rs], [XC/xet_core_structures/src/metadata_shard/file_structs.rs], [XC/xet_core_structures/src/metadata_shard/xorb_structs.rs]:
  - File info: a `FileDataSequenceHeader` (file_hash, flags, num_entries) followed by `FileDataSequenceEntry` terms (xorb_hash, unpacked_segment_bytes, chunk_index_start/end). Optional parts are `FileVerificationEntry` (range_hash) and `FileMetadataExt` (sha256).
  - Xorb ("CAS") info: a `XorbChunkSequenceHeader` (xorb_hash, num_entries, bytes) followed by `XorbChunkSequenceEntry` items (chunk_hash, byte_range_start, unpacked_bytes).
  - Lookup tables are sorted on truncated u64 keys: a file lookup, a xorb lookup, and a chunk lookup that maps to (xorb idx, chunk offset).
- Footer fields [XC/xet_core_structures/src/metadata_shard/shard_format.rs]:
  - Section offsets and counts.
  - `chunk_hash_hmac_key`: when non-zero, chunk hashes in the shard are HMAC'd. This is used for global dedup responses.
  - `shard_creation_timestamp` and `shard_key_expiry`.
  - `stored_bytes_on_disk`, `materialized_bytes`, and `stored_bytes`.
- Target shard size is 64 MiB. Local shard cache expiry is 3 weeks [XC/xet_core_structures/src/metadata_shard/constants.rs]. The cache limit is 16 GB (`HF_XET_SHARD_CACHE_SIZE_LIMIT`), and the in-memory chunk index is capped at 64 MiB [XC/xet_runtime/src/config/groups/shard.rs].
- Open PR #993 makes an uploaded shard's bytes depend only on its content [GH-PR-993].

### 2.5 CAS server API

From `openapi/cas.openapi.yaml` and `remote_client.rs`:

- `GET /v1/reconstructions/{file_id}` and `GET /v2/reconstructions/{file_id}`. v2 is multi-range optimized and returns 404/501 when unsupported, in which case the client falls back to v1. Range is passed in a header [XC/openapi/cas.openapi.yaml], [XC/api_changes/update_260316_v2_reconstruction_multirange.md].
  - v1 response: `offset_into_first_range`, `terms[{hash, range{start,end}, unpacked_length}]`, and `fetch_info{xorb -> [{url, url_range, range}]}`.
  - v2 response: replaces `fetch_info` with `xorbs` multi-range fetches.
  - Selected with `HF_XET_CLIENT_RECONSTRUCTION_API_VERSION` (auto by default) [XC/xet_runtime/src/config/groups/client.rs].
- `GET /v1/chunks/{prefix}/{hash}`: the global dedup query. It returns shard bytes and needs only `read` scope [XC/openapi/cas.openapi.yaml]. The client builds it as `{endpoint}/v1/chunks/{key}` [XC/xet_client/src/cas_client/remote_client.rs].
- `POST /v1/xorbs/{prefix}/{hash}`: returns `was_inserted` [XC/openapi/cas.openapi.yaml].
- `POST /v1/shards`: registers files. There is also `POST /v2/shards`, which streams finalization progress (July 2026) [XC/xet_client/src/cas_client/remote_client.rs], [GH-PR-884].
- `POST /v1/telemetry` (July/Aug 2026) [XC/api_changes/update_260728_client_transfer_telemetry.md].
- Proposed, not merged: `POST /v1/xorbs/grants` for direct-to-S3 presigned xorb PUTs behind `HF_XET_XORB_DIRECT_UPLOAD` (#974) [GH-PR-974].
- Auth: Xet tokens come from the Hub. The headers `X-Xet-Cas-Url`, `X-Xet-Access-Token`, `X-Xet-Token-Expiration`, and `X-Xet-Session-Id` appear in git-xet [XC/git_xet/src/constants.rs]. See also the spec auth page [SPEC-index].

### 2.6 File reconstruction

- A file is a list of terms. Each term is a xorb hash plus a chunk range `[start, end)`. The client fetches xorb byte ranges through presigned URLs, decompresses the chunks, and concatenates them [SPEC-dedup], [XC/openapi/cas.openapi.yaml].
- Download tuning:
  - Fetch sizes: `HF_XET_RECONSTRUCTION_MIN/MAX_RECONSTRUCTION_FETCH_SIZE` (256 MB / 8 GB) [XC/xet_runtime/src/config/groups/reconstruction.rs].
  - Buffer sizes: since #943 (2026-08-28) these default from usable, cgroup-aware system memory instead of constants [XC/xet_runtime/src/config/groups/reconstruction.rs], [GH-PR-943].
- Chunk cache: 10 GB in the Rust crates, but 0 (disabled) in the hf_xet wheel because of the `no-default-cache` feature. Enable it with `HF_XET_CHUNK_CACHE_SIZE_BYTES` [XC/xet_runtime/src/config/groups/chunk_cache.rs], [XC/hf_xet/Cargo.toml], [DOC-using-xet].
- Resumable downloads do not exist today. PoC #989 (Wauplin) proposes an opt-in `reuse_existing` mode: it re-chunks the existing local segments and verifies each one against the shard's verification hash [GH-PR-989].

### 2.7 Global dedup

- Three tiers [SPEC-dedup], [XC/xet_data/src/deduplication/file_deduplication.rs]:
  1. An in-session hash table.
  2. The local shard cache.
  3. A global query.
- Eligibility for a global query: the file's first chunk, or `hash % 1024 == 0` (`MDB_SHARD_GLOBAL_DEDUP_CHUNK_MODULUS`, about 0.1% "key chunks"). The spec states this as the last 8 bytes read as a little-endian u64, mod 1024 [XC/xet_core_structures/src/metadata_shard/constants.rs], [SPEC-dedup], [BLOG-chunks-to-blocks].
- Spacing: at least `HF_XET_DATA_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES=256` chunks (about 16 MB at 64 KiB average) between queries [XC/xet_runtime/src/config/groups/data.rs].
  - The config doc comment says "every 256 chunks or 4MB of data", but 256 x 64 KiB is 16 MiB. That is a doc-comment inconsistency (my arithmetic).
  - The spec recommends about one query per 4 MB [SPEC-dedup].
- Global queries can be turned off with `HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED` [XC/xet_runtime/src/config/groups/deduplication.rs].
- Pass structure: queries run in the background. If a query returns a new shard, the file is deduped a second time against it [XC/xet_data/src/deduplication/file_deduplication.rs].
- The response is a shard with HMAC'd chunk hashes. The client HMACs its own hashes with the key from the footer to find matches, so raw hashes of other people's chunks are never revealed [SPEC-dedup], [XC/xet_core_structures/src/metadata_shard/shard_format.rs].
- Defrag prevention: the dedup code tracks a rolling chunks-per-range (CPR) average over the last `NRANGES=128` ranges. It refuses dedup ranges shorter than the target when CPR is below `MIN_N_CHUNKS_PER_RANGE=8`, with hysteresis factor 0.5 [XC/xet_runtime/src/config/groups/deduplication.rs], [XC/xet_data/src/deduplication/defrag_prevention.rs], [SPEC-dedup].
  - RFC #887 (XciD, open) documents a failure mode on mostly-matchable re-uploads: "~2.1x stored bytes" and "CPR pinned at exactly the min_n_chunks_per_range hysteresis equilibrium" [GH-PR-887].
  - #931 (2026-08) fixed double counting in fragmentation reduction [GH-PR-931].

### 2.8 Server-side copy and buckets (context)

- Storage Buckets (2026-03) are mutable, S3-like storage backed by Xet. Enterprise billing is on deduplicated storage [BLOG-storage-buckets].
- `hf buckets cp` between repos and buckets moves only hashes [DOC-hub-dedup].
- Range uploads, `XetSession.new_range_upload()`, support partial file rewrites. Python bindings are proposed in open PR #951 [GH-PR-951].

---

## 3. Public surface

### 3.1 Python API (`hf_xet`)

Registered in `hf_xet/src/lib.rs` [XC/hf_xet/src/lib.rs]:
- New session API (exposed to Python 2026-05, #792 [GH-PR-792]):
  - Session and config: `XetSession`, `XetConfig` (get/set).
  - Uploads: `XetUploadCommit`, `XetFileUpload`, `XetStreamUpload`.
  - Downloads: `XetFileDownloadGroup`, `XetFileDownload`, `XetDownloadStreamGroup`, `XetDownloadStream`, `XetUnorderedDownloadStream`.
  - Results and progress: `XetFileInfo`, `XetFileMetadata`, `XetCommitReport`, `XetDownloadReport`, `GroupProgressReport`, `ItemProgressReport`, `DeduplicationMetrics`.
  - SHA-256 markers: `COMPUTE_SHA256` and `SKIP_SHA256`.
- Legacy functions: `upload_bytes`, `upload_files`, `download_files`, `hash_files`, `force_sigint_shutdown` [XC/hf_xet/src/lib.rs], [XC/hf_xet/src/legacy/functions.rs].
  - `hash_files(file_paths)` computes the Xet file hash, size, and sha256 locally without uploading [XC/hf_xet/src/legacy/functions.rs], [XC/xet_data/src/processing/data_client.rs].
- Dedup metrics are exposed: `XetCommitReport.dedup_metrics` and `XetFileMetadata.dedup_metrics` (per file) [XC/xet_pkg/src/xet_session/upload_commit.rs]. The `DeduplicationMetrics` fields are [XC/xet_data/src/deduplication/dedup_metrics.rs]:
  - `total_bytes`, `deduped_bytes`, `new_bytes`
  - `deduped_bytes_by_global_dedup`, `defrag_prevented_dedup_bytes`
  - the same set for chunks
  - `xorb_bytes_uploaded`, `shard_bytes_uploaded`, `total_bytes_uploaded`
- Offline local CAS: `XetSession().new_upload_commit(endpoint="local://<dir>")` works. The hf_xet test suite uses it [XC/hf_xet/tests/conftest.py], [XC/hf_xet/tests/test_upload_commit.py]. This gives fully offline dedup measurement from Python (UNVERIFIED by me at runtime).

### 3.2 CLI tools and binaries

- `git-xet`: the only distributed binary (GitHub releases, Homebrew, winget, MSI) [XC/README.md], [DOC-using-xet].
- `xtool`, in `xet_pkg/src/bin/xtool/`, needs `--features internal-tools` [XC/xet_pkg/Cargo.toml], [XC/xet_pkg/src/bin/xtool/main.rs].
  - Subcommands: `file upload` (local endpoints only, since remote uploads would be garbage collected), `file download`, `dedup`, and `query`.
  - Global flags: `--endpoint`, `--token`, `--repo-type`/`--repo-id` (Hub mode), `-c KEY=VALUE` config overrides.
  - #759 added GC simulation [GH-PR-759].
- `xtool dedup FILES [-r] [-s] [-o out.json] [--compression N] [--migrate]` [XC/xet_pkg/src/bin/xtool/dedup.rs]:
  - It is a dry run by default. It prints `hash: size -> new_bytes` per file and "Transmitted N bytes", and writes MDBFileInfo (reconstruction terms) as JSON.
  - In Hub mode the dry-run client still issues global dedup queries. Only xorb upload is short-circuited by `dry_run` [XC/xet_client/src/cas_client/remote_client.rs]. My reading is that global dedup is not gated on dry_run (UNVERIFIED at runtime).
- `xet-data` examples, which need no Hub or network [XC/xet_data/Cargo.toml]:
  - `chunk`: file or stdin to `<hash> <len>` lines.
  - `hash`: `-h chunk|xorb|file|range` over those lines.
  - `xorb-check`: validates a xorb against its hash.
  - VERIFIED-LOCAL, see 2.1.
- The `x` bin (`xet_data/src/processing/bin/example.rs`, internal-tools) and a chunk-cache bin exist [XC/xet_data/Cargo.toml]. UNVERIFIED what they do.

### 3.3 Standalone Rust use (can I chunk locally without the Hub?)

Yes:
- `xet_data::deduplication::Chunker::new(target)` with `next_block(&[u8], is_final) -> Vec<Chunk>` and `finish()`. Each `Chunk` carries `{hash, data}` [XC/xet_data/src/deduplication/chunking.rs], [XC/xet_core_structures/src/xorb_object/chunk.rs].
- `xet_core_structures::merklehash::{compute_data_hash, xorb_hash, file_hash}` [XC/xet_core_structures/src/merklehash/aggregated_hashes.rs].
- A whole upload session can run against `local://` or memory CAS clients [XC/xet_data/src/processing/remote_client_interface.rs].
- All of these crates are on crates.io at 1.6.0 [CRATES].
- For JS: `hf_xet_thin_wasm` (Chunker, compute_xorb_hash, compute_file_hash, compute_verification_hash, compute_hmac) [XC/wasm/hf_xet_thin_wasm/src/lib.rs].

### 3.4 Environment variables (from `xet_runtime/src/config/groups/*`)

- Naming: `HF_XET_<GROUP>_<FIELD>`, defined with `config_group!` [XC/AGENTS.md]. Legacy aliases are mapped in `aliases.rs`, for example `HF_XET_FIXED_UPLOAD_CONCURRENCY`, which sets initial, min, and max together [XC/xet_runtime/src/config/aliases.rs].
- Cache dir: `HF_XET_CACHE`, then `$HF_HOME/xet`, then `$XDG_CACHE_HOME/huggingface/xet`, then `~/.cache/huggingface/xet` [XC/xet_runtime/src/core/cache_dir.rs].
- High-performance mode: `HF_XET_HIGH_PERFORMANCE` or `HF_XET_HP` [XC/xet_runtime/src/utils/configuration_utils.rs]. It sets [XC/xet_runtime/src/config/xet_config.rs]:
  - file ingestion 100
  - AC max 124, min 4, initial 16 (upload and download)
  - fetch 1 GB / 16 GB
  - memory-derived HP buffers
- Docs recommend HP mode only with at least 64 GB RAM [DOC-using-xet].
- Groups and representative vars [XC/xet_runtime/src/config/groups/]:
  - `CLIENT_*`: retries 5 / 3 s / 6 min; `READ_TIMEOUT=300s`; `CONNECT_TIMEOUT=60s`; adaptive concurrency `AC_*` with initial upload 2, initial download 4, max 64; `ENABLE_MULTIRANGE_FETCHING`; `UNIX_SOCKET_PATH`; `RECONSTRUCTION_API_VERSION`.
  - `DATA_*`: ingestion 8, block 8 MB, global dedup spacing, progress intervals, `DEFAULT_CAS_ENDPOINT`.
  - `DEDUPLICATION_*`: the defrag knobs and global query on/off.
  - `XORB_*`, `SHARD_*`, `CHUNK_CACHE_SIZE_BYTES`, `RECONSTRUCTION_*`, `SESSION_DIR_NAME`.
  - `LOG_*`: `DEST` (file or dir/), `FORMAT=json`, `PREFIX`, dir cleanup 250 MB / 1 d / 14 d.
  - `SYSTEM_MONITOR_*`: `ENABLED`, `LOG_PATH`, `SAMPLE_INTERVAL`.
  - `TELEMETRY_*`.
- Log level comes from `RUST_LOG`. Defaults are `info` for file and `warn` for console [XC/xet_runtime/src/logging/constants.rs], [XC/xet_runtime/src/logging/init.rs].
  - The README suggests `HF_XET_LOG_FILE`, but in current code that name appears only in a doc comment. The real variable is `HF_XET_LOG_DEST` [XC/README.md], [XC/xet_pkg/src/lib.rs], [XC/xet_runtime/src/config/groups/log.rs]. UNVERIFIED that there is no other runtime read, since the clone is shallow and I could not check history.
- Docs drift (the hub-docs env table vs code, my comparison) [DOC-using-xet-src], [XC/xet_runtime/src/config/groups/client.rs]:

  | Variable | Docs | Code |
  | --- | --- | --- |
  | `HF_XET_CLIENT_READ_TIMEOUT` | 120s | 300s (raised in #808) [GH-PR-808] |
  | `AC_INITIAL_UPLOAD_CONCURRENCY` | 1 | 2 |
  | `AC_INITIAL_DOWNLOAD_CONCURRENCY` | 1 | 4 |

  - The doc comment for `ac_min_upload_concurrency` says 2, but the value is 1.
  - The reconstruction buffer rows list constants that are now memory-derived (#943).
  - The `HF_XET_TELEMETRY_*` vars are not in the docs table.

### 3.5 Logging, telemetry, observing dedup

- Telemetry (Aug 2026, #932 to #937): at the end of each transfer, one best-effort summary goes to `POST /v1/telemetry`. It carries byte counts, dedup effectiveness, throughput, and wall time, and contains no names, paths, hashes, or repo/user ids [XC/README.md], [XC/api_changes/update_260728_client_transfer_telemetry.md].
  - Upload documents carry 40 metric keys and download documents carry 23.
  - Dry runs send nothing.
  - Opt out with `HF_XET_TELEMETRY_ENABLED=0` or any of the HF opt-outs (`HF_HUB_DISABLE_TELEMETRY`, `DO_NOT_TRACK`, `HF_HUB_OFFLINE`, ...) (#987).
- System monitor: `HF_XET_SYSTEM_MONITOR_ENABLED` and `_LOG_PATH` turn on periodic resource sampling [XC/xet_runtime/src/config/groups/system_monitor.rs]. UNVERIFIED what exactly is sampled.
- Ways a user can measure dedup:
  1. Rust or Python session API: `XetCommitReport.dedup_metrics` [XC/xet_pkg/src/xet_session/upload_commit.rs].
  2. `xtool dedup` dry run against the Hub, or against a `local://` dir [XC/xet_pkg/src/bin/xtool/dedup.rs].
  3. Local `chunk` example output plus set arithmetic. This gives local-only dedup with no global dedup and no defrag effects.
  4. The Hub UI and progress bar show network bytes only [HH/src/huggingface_hub/utils/_xet_progress_reporting.py].
  5. The Hub API tree exposes `xetHash` per file, which can be compared with a local `hash_files` result [HF-API-specref].
  6. `RUST_LOG=info` logs. UNVERIFIED which dedup lines they print.

---

## 4. Existing dedup estimation tools (do not duplicate)

- `huggingface/dataset-dedupe-estimator` (`de` CLI, by kszucs): a chunk-level dedup estimator for Parquet collections and git revisions. It compares JSONL, Parquet, and CDC-Parquet (Snappy/ZSTD) and plots results. It was used in the Parquet CDC blog [GH-dedupe-estimator], [BLOG-parquet-cdc].
- `xtool dedup` in xet-core, with real dedup including global dedup in Hub mode [XC/xet_pkg/src/bin/xtool/dedup.rs].
- `xet-data` `chunk`/`hash` examples [XC/xet_data/Cargo.toml].
- Spaces:
  - `xet-team/quantization-dedup`: a block-level dedup visualization of bartowski/gemma-2-9b-it-GGUF [SPACE-quantization-dedup].
  - `znation/xet-repo-dedupe`: repo-level dedup visualization [SPACE-znation].
  - `xet-team/lfs-analysis`: Hub LFS storage analysis [BLOG-files-to-chunks].
- huggingface.js conformance: a 60-case chunk/hash matrix against a xet-core wasm oracle [HFJS-PR-2423]. Do not build another cross-impl test-vector suite for chunking.
- The #887 harness: two ignored exploration tests that reproduce defrag-oscillation on a local server [GH-PR-887].

---

## 5. Activity in xet-core (2026-04 to 2026-10)

### 5.1 Open issues worth knowing (all from [GH-API-issues])

- #942 `XetFileDownloadGroup.abort()` keeps downloading. This is the only `good first issue` (2026-08-16), and it already has two open PRs, #949 and #945, so it is effectively taken [GH-ISSUE-942].
- #992 `XetUploadCommit::commit()` fails after one stream upload is aborted or dropped (2026-09-25).
- #991 Avoid invalidating builds with `git_version!` (enhancement).
- #964 perf: download stream groups rebuild HTTP clients and TLS roots repeatedly.
- #956 every 403 on xorb fetch is treated as URL expiry, so 0 bytes are written for minutes.
- #930 and #875: adaptive concurrency (fair sharing, and a degenerate RTT predictor). #875 has PR #960 from an external contributor.
- #853 `new_upload_commit().build()` rescans the whole shard cache (about 4 s, 708 `.mdb` files) on every call. Hoytak's open PR #862 (session builder context, config, and runtime sharing) appears to target this; inferred from the PR title, not checked [GH-ISSUE-853].
- #920, #909 (proxy), #904 (signed models), #892 (ReFS/Dev Drive), #889 (ppc64le CI), #846 (winget PATH), #763 (errno lost before Python), #592, #527.

### 5.2 Active PR themes (merged since 2026-04, from [GH-API-pulls])

- Client telemetry stack: #932 to #937, #940, #922 (generated metrics schema plus a CI compatibility gate), #987.
- WASM target: #841, #873, #952 (BLAKE3 SIMD plus chunker benchmark).
- Session API to Python: #792, #825, #856, #990 (GIL release), #996 (cache dir on builder).
- Range or partial uploads and stable chunk boundaries: #717, #815, #845, #863. #948 and #951 are open.
- Memory and robustness: #943 (cgroup-aware buffers), #941 (cap untrusted shard parser allocations), #774 (FD leak).
- Shards and xorbs: #865 (nonce), #884 (v2 shards streaming), #879, #834. #993 is open.
- GC simulation: #759, #762, #786, #878, #882, #958, #962.
- Dedup correctness: #931. RFC #887 is open.
- Dependency and CVE hygiene, crate releases.
- Open RFCs and PoCs:
  - #974: direct-to-S3 xorb upload.
  - #989: reuse local data on download.
  - #881: C ABI crate `xet_capi`.
  - #820: configurable chunk-cache eviction policies (external).
  - #965: AVX-512 BG4 predictor (external).
- External contributors merged in this window include `Expyron` (#900 LazyLock), `tisonkun` (#830), `valkum` (#957 rustls-no-provider), and `arpitjain099` (#843) [GH-API-pulls]. Small, focused PRs from outsiders do get merged.

### 5.3 Contribution guidelines

- CONTRIBUTING.md is a generic scikit-learn-derived guide: search for existing issues first, include OS/versions/repro/full traceback, describe motivation for features [XC/CONTRIBUTING.md].
- AGENTS.md is the real bar [XC/AGENTS.md]:
  - Simplicity first: "No premature abstractions", "Don't implement features until needed".
  - Run `cargo +nightly fmt --check` (root and hf_xet), `cargo clippy -r -- -D warnings` (root and hf_xet), and `cargo test --package <changed>`.
  - The full CI feature set is `strict simulation internal-tools git-xet-for-integration-test`.
  - Dependency changes need `cargo machete` and `cargo audit`, and must leave no lockfile drift.
  - wasm32 must stay green: use `web_time::Instant`, `?Send` async-trait, and fs gated behind cfg.
  - Use `tracing` for logging, `thiserror` errors, and `config_group!` for any new knob.
  - Add an `api_changes/update_<yymmdd>_*.md` file for API changes.
  - Report which checks ran.
- rustfmt: 120-column code and comments [XC/AGENTS.md], [XC/rustfmt.toml].

---

## 6. HF blog posts on Xet

All from [BLOG-index], with the individual post files under `https://github.com/huggingface/blog/blob/main/<slug>.md`.

| Date | Post | One-liner and key numbers |
| --- | --- | --- |
| 2024-08-08 | xethub-joins-hf | XetHub is acquired. Hub stats at the time: 12 PB in LFS (280M files), 6 PB/day CloudFront [BLOG-xethub-joins-hf] |
| 2024-10-05 | improve_parquet_dedupe | Parquet dedup study. Appends are 99.1% deduped (20 MB new), row modifications 89% (230 MB). Argues for CDC-friendly Parquet writing [BLOG-improve-parquet] |
| 2024-11-20 | from-files-to-chunks | CDC rationale. More than 30 PB in LFS. About 50% storage and transfer gains in XetHub benchmarks, e.g. CORD-19 at 53% less storage [BLOG-files-to-chunks] |
| 2024-11-26 | rearchitecting-uploads-and-downloads | CAS plus a regional caching design. 8.2M upload requests and 130.8 TB in the snapshot; the top 7 countries make up 80% of upload bytes [BLOG-rearchitecting] |
| 2025-02-12 | from-chunks-to-blocks | Chunks are aggregated into 64 MB blocks (1,000x fewer CAS entries). Key chunks are a 0.1% modulo subset for global dedup. 45 PB and 690B chunks if handled naively; the gemma-2-9b-it-GGUF repo goes from 191 GB to 97 GB (about 2x faster upload) [BLOG-chunks-to-blocks] |
| 2025-03-18 | xet-on-the-hub | First production migration. About 6% of downloads on Xet, 4.5 TB of repos migrated. Read amplification of 3.5:1 from hf_transfer 10 MB ranges was fixed, cutting GET latency by about 35% [BLOG-xet-on-hub] |
| 2025-07-15 | migrating-the-hub-to-xet | LFS-to-Xet background migration. More than 1M users on Xet; Xet became the default for new users in May 2025. bartowski's 500 TB / 2,000 repos ran at CAS about 35 Gb/s, later peaks reached about 300 Gb/s [BLOG-migrating] |
| 2025-07-25 | parquet-cdc | Parquet CDC in PyArrow. 21 PB of datasets, 4 PB of them Parquet. Shows the "New Data Upload" bar and dedup tables from the dedupe estimator [BLOG-parquet-cdc] |
| 2025-10-27 | streaming-datasets | Streaming-datasets efficiency (100x). Mentions Xet only tangentially [BLOG-index] |
| 2026-03-10 | storage-buckets | Mutable, S3-like buckets backed by Xet. Enterprise billing on deduplicated bytes [BLOG-storage-buckets] |
| 2026-05-27 | delta-weight-sync | TRL RL weight sync through a bucket. About 99% of bf16 weights are unchanged per step; sparse safetensors cut 1.2 GB to 20-35 MB. It claims that "even if we ... just uploaded full anchors every step, Xet would still only transfer the changed chunks" [BLOG-delta-weight-sync] |
| 2026-07-07 | skypilot-hf-storage | Zero-egress storage with SkyPilot on HF storage. Xet appears only in passing [BLOG-index] |

---

## 7. Gaps and opportunities for a small genuine build

Ranked by fit to a low-level Xet storage role. Effort assumes evenings.

1. **Dedup-boundary probe for RL and fine-tune checkpoints: test the delta-weight-sync claim.**
   - The claim: the blog says full-anchor uploads would "still only transfer the changed chunks" [BLOG-delta-weight-sync].
   - My back-of-envelope (UNVERIFIED): a 64 KiB chunk holds 32,768 bf16 values. If 1% of elements change independently, P(chunk unchanged) is 0.99^32768, roughly e^-327, which is about 0. Chunk dedup would then be near 0% unless changes cluster (frozen layers, untouched embedding rows, LoRA).
   - Build: chunk consecutive checkpoints with the `chunk` example or `Chunker`, and report dedup per tensor and per layer. Then confirm with `xtool dedup` against `local://` and read `DeduplicationMetrics` (new_bytes, defrag_prevented).
   - Why on-point: it measures where CDC dedup holds and where it breaks on a workload HF is actively promoting, and the result is crisp and falsifiable.
   - Effort: 2-4 days.
   - Do not duplicate: dataset-dedupe-estimator (Parquet only) [GH-dedupe-estimator], or the quantization-dedup Space (GGUF visualization) [SPACE-quantization-dedup].
2. **Fragmentation and defrag-prevention analyzer for the reconstruction JSON from `xtool dedup -o`.**
   - Build: compute terms per file, a chunks-per-range histogram, xorbs touched, and `defrag_prevented_dedup_bytes` across synthetic edit patterns (append, insert, interleaved re-import). Publish a write-up of where the CPR=8 and hysteresis-0.5 thresholds bite.
   - Why on-point: it feeds directly into open RFC #887 [GH-PR-887] and fix #931 [GH-PR-931].
   - Effort: 3-5 days.
   - Do not duplicate: the two ignored harness tests in #887. Build an external analyzer and a parameter sweep instead, and coordinate on the RFC thread.
3. **Small upstream code PR from open issues that have no open PR whose title matches (I did not check for linked PRs).**
   - Candidates: #964 (reuse the HTTP client and TLS roots across download stream groups) [GH-API-issues]; #956 (distinguish a 403 caused by expiry from other 403s and keep the body for diagnostics) [GH-API-issues]; #763 (preserve errno through to Python) [GH-API-issues].
   - Why on-point: client-side networking and error plumbing in the actual product, which reviewers can merge.
   - Effort: 1-3 days each.
   - Do not duplicate: #942, which already has #949 and #945; #853, which #862 targets; #875, which #960 targets.
4. **Docs-drift fix in hub-docs and the xet-core README.**
   - Fixes:
     - `HF_XET_CLIENT_READ_TIMEOUT` should be 300s.
     - AC initial upload and download should be 2 and 4.
     - The reconstruction buffers are memory-derived since #943.
     - Add the telemetry vars.
     - In the README, `HF_XET_LOG_FILE` should become `HF_XET_LOG_DEST`.
     - The `ac_min_upload_concurrency` doc comment says 2 but the value is 1.
     - The global dedup spacing comment says "4MB" but 256 chunks is about 16 MiB.
     - The xorb spec page says "no explicit limit on number of chunks" while the code uses 8,192.
   - Evidence: [DOC-using-xet-src], [XC/xet_runtime/src/config/groups/client.rs], [XC/xet_runtime/src/config/groups/data.rs], [SPEC-xorb].
   - Why on-point: it shows careful source reading, and it is low risk.
   - Effort: half a day. It is a good companion PR, not a headline.
5. **A `xet-dedup-report` mini tool around the Python session API.**
   - Build: upload a directory to a `local://` CAS (or a dry run against the Hub), then print `XetCommitReport.dedup_metrics` per file, including global vs local dedup and defrag-prevented bytes. Optionally propose surfacing a one-line dedup summary in huggingface_hub after uploads, since the hub currently shows only network bytes [HH/src/huggingface_hub/utils/_xet_progress_reporting.py].
   - Why on-point: it uses the newly exposed Python session API (#792) [GH-PR-792].
   - Effort: 1-2 days.
   - Do not duplicate: `xtool dedup` (Rust, internal-tools feature) [XC/xet_pkg/src/bin/xtool/dedup.rs]. Its value is Python ergonomics and per-file breakdown, so keep it thin. Open an issue on huggingface_hub before writing a PR there.
6. **Revive or benchmark parallel chunking.**
   - PR #525 has been open since 2025-10 [GH-PR-525], and `find_partitions` plus `parallel chunking.pdf` already exist [XC/xet_data/src/deduplication/chunking.rs].
   - Build: measure single-thread chunk+hash throughput against parallel partitions on a large safetensors file, check boundary equivalence on the spec reference file, and report.
   - Why on-point: it sits on the CPU-bound hot path of uploads.
   - Effort: 2-4 days.
   - Do not duplicate: the CHUNKER_BENCHMARK (native vs wasm) [XC/wasm/hf_xet_thin_wasm/CHUNKER_BENCHMARK.md]. Talk to the #525 author or maintainers before building.

Do not build: another chunk-conformance suite (huggingface.js #2423 already has one) [HFJS-PR-2423], a Parquet dedup estimator [GH-dedupe-estimator], or a GGUF block visualizer [SPACE-quantization-dedup].

---

## Sources

| Key | URL |
| --- | --- |
| XC/... | `https://github.com/huggingface/xet-core/blob/d2f667cc247a7ae54a2cf93e413f8fee96c3c927/...` (path as given) |
| HH/... | `https://github.com/huggingface/huggingface_hub/blob/d711944/...` (path as given) |
| GH-API-repo | https://api.github.com/repos/huggingface/xet-core |
| GH-API-issues | https://api.github.com/repos/huggingface/xet-core/issues?state=open&per_page=100 (fetched 2026-10-03) and https://github.com/huggingface/xet-core/issues |
| GH-API-pulls | https://api.github.com/repos/huggingface/xet-core/pulls?state=closed&per_page=100 (fetched 2026-10-03) and https://github.com/huggingface/xet-core/pulls |
| GH-PR-NNN / GH-ISSUE-NNN | https://github.com/huggingface/xet-core/pull/NNN or https://github.com/huggingface/xet-core/issues/NNN |
| HFJS-PR-2422 | https://github.com/huggingface/huggingface.js/pull/2422 |
| HFJS-PR-2423 | https://github.com/huggingface/huggingface.js/pull/2423 |
| PYPI-hf-xet | https://pypi.org/project/hf-xet/ |
| PYPI-hfhub-2.1.1 | https://pypi.org/pypi/huggingface-hub/2.1.1/json |
| CRATES | https://crates.io/crates/hf-xet , https://crates.io/crates/xet-data (and xet-client, xet-core-structures, xet-runtime) |
| SPEC-index | https://huggingface.co/docs/xet/index |
| SPEC-chunking | https://huggingface.co/docs/xet/chunking |
| SPEC-hashing | https://huggingface.co/docs/xet/hashing |
| SPEC-xorb | https://huggingface.co/docs/xet/xorb |
| SPEC-dedup | https://huggingface.co/docs/xet/deduplication |
| DOC-storage-backends | https://huggingface.co/docs/hub/storage-backends |
| DOC-using-xet | https://huggingface.co/docs/hub/xet/using-xet-storage |
| DOC-using-xet-src | https://github.com/huggingface/hub-docs/blob/main/docs/hub/xet/using-xet-storage.md |
| DOC-hub-dedup | https://huggingface.co/docs/hub/xet/deduplication |
| HF-API-specref | https://huggingface.co/api/datasets/xet-team/xet-spec-reference-files/tree/main and https://huggingface.co/datasets/xet-team/xet-spec-reference-files |
| GH-dedupe-estimator | https://github.com/huggingface/dataset-dedupe-estimator |
| SPACE-quantization-dedup | https://huggingface.co/spaces/xet-team/quantization-dedup |
| SPACE-znation | https://huggingface.co/spaces/znation/xet-repo-dedupe |
| BLOG-index | https://github.com/huggingface/blog/blob/main/_blog.yml |
| BLOG-xethub-joins-hf | https://huggingface.co/blog/xethub-joins-hf |
| BLOG-improve-parquet | https://huggingface.co/blog/improve_parquet_dedupe |
| BLOG-files-to-chunks | https://huggingface.co/blog/from-files-to-chunks |
| BLOG-rearchitecting | https://huggingface.co/blog/rearchitecting-uploads-and-downloads |
| BLOG-chunks-to-blocks | https://huggingface.co/blog/from-chunks-to-blocks |
| BLOG-xet-on-hub | https://huggingface.co/blog/xet-on-the-hub |
| BLOG-migrating | https://huggingface.co/blog/migrating-the-hub-to-xet |
| BLOG-parquet-cdc | https://huggingface.co/blog/parquet-cdc |
| BLOG-storage-buckets | https://huggingface.co/blog/storage-buckets |
| BLOG-delta-weight-sync | https://huggingface.co/blog/delta-weight-sync |
