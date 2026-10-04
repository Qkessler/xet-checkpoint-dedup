"""Count the Xet chunks two public Hub revisions of a file share, from the CAS reconstruction API.

For each revision: the Hub tree API gives the file's xetHash, the xet-read-token endpoint gives an anonymous CAS
token, and `GET /v1/reconstructions/<xetHash>` gives the file's terms (xorb hash plus chunk index range). A chunk of
the next file is shared when the same (xorb, chunk index) appears in the base file's terms, which is what dedup at
upload time leaves behind. Public reads only; writes results/<name>.json.
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

hubUrl = "https://huggingface.co"
resultsDir = Path(__file__).resolve().parent.parent / "results"


def getJson(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read()), response.headers


def fileTerms(repoId: str, revision: str, fileName: str) -> dict:
    tree, _ = getJson(f"{hubUrl}/api/models/{repoId}/tree/{revision}")
    entry = next(item for item in tree if item["path"] == fileName)
    _, tokenHeaders = getJson(f"{hubUrl}/api/models/{repoId}/xet-read-token/{revision}")
    token = tokenHeaders["x-xet-access-token"]
    casUrl = tokenHeaders["x-xet-cas-url"]
    reconstruction, _ = getJson(f"{casUrl}/v1/reconstructions/{entry['xetHash']}", {"Authorization": f"Bearer {token}"})
    return {"xet_hash": entry["xetHash"], "file_bytes": entry["size"], "terms": reconstruction["terms"]}


def chunkKeys(terms: list) -> list:
    keys = []
    for term in terms:
        for chunkIndex in range(term["range"]["start"], term["range"]["end"]):
            keys.append((term["hash"], chunkIndex))
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--next-revision", required=True)
    parser.add_argument("--file", default="model.safetensors")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    base = fileTerms(args.repo_id, args.base_revision, args.file)
    following = fileTerms(args.repo_id, args.next_revision, args.file)
    baseKeys = set(chunkKeys(base["terms"]))
    nextKeys = chunkKeys(following["terms"])
    sharedChunks = sum(1 for key in nextKeys if key in baseKeys)
    sharedBytes = sum(
        term["unpacked_length"]
        for term in following["terms"]
        if all((term["hash"], index) in baseKeys for index in range(term["range"]["start"], term["range"]["end"]))
    )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    result = {
        "name": args.name,
        "kind": "hub-reconstruction",
        "tool": {"git_commit": commit, "script": "scripts/hub_reconstruction.py"},
        "inputs": [
            {
                "role": "base",
                "repo_id": args.repo_id,
                "revision": args.base_revision,
                "file": args.file,
                "xet_hash": base["xet_hash"],
            },
            {
                "role": "next",
                "repo_id": args.repo_id,
                "revision": args.next_revision,
                "file": args.file,
                "xet_hash": following["xet_hash"],
            },
        ],
        "result": {
            "next_file_bytes": following["file_bytes"],
            "next_terms": len(following["terms"]),
            "next_chunks": len(nextKeys),
            "shared_chunks": sharedChunks,
            "shared_bytes_in_whole_shared_terms": sharedBytes,
            "base_xorbs": len({term["hash"] for term in base["terms"]}),
            "next_xorbs": len({term["hash"] for term in following["terms"]}),
            "next_xorbs_also_in_base": len(
                {term["hash"] for term in following["terms"]} & {term["hash"] for term in base["terms"]}
            ),
        },
    }
    resultsDir.mkdir(exist_ok=True)
    (resultsDir / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["result"], indent=2))


if __name__ == "__main__":
    main()
