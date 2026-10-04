"""Download every `hub` input in data/manifest.toml into the cache and verify sha256s.

Generated inputs (our own training runs) are only verified, because `just train` writes them.
"""

import hashlib
import shutil
import sys
import tomllib
from pathlib import Path

from huggingface_hub import hf_hub_download

cacheRoot = Path.home() / ".cache" / "xet-checkpoint-dedup"
manifestPath = Path(__file__).resolve().parent.parent / "data" / "manifest.toml"


def fileSha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = tomllib.loads(manifestPath.read_text())
    failures = []
    for key, entry in manifest["inputs"].items():
        target = cacheRoot / "inputs" / key / entry["file"]
        if entry["kind"] == "hub" and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=entry["repo_id"],
                repo_type=entry["repo_type"],
                revision=entry["revision"],
                filename=entry["file"],
                cache_dir=cacheRoot / "hf",
            )
            shutil.copyfile(downloaded, target)
        if not target.exists():
            failures.append(f"{key}: missing {target} (run `just train`)")
            continue
        actualSha256 = fileSha256(target)
        if entry["sha256"] and actualSha256 != entry["sha256"]:
            failures.append(f"{key}: sha256 {actualSha256} != manifest {entry['sha256']}")
            continue
        print(f"ok {key} {actualSha256} {target}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
